# Forge-4 Handoff -- Engine Cache + Background Compiler + Telemetry

**Owner:** Forge-4
**Reads this:** Forge-3 (orchestrator wiring), Forge-2 (specialists, telemetry write-side), Crucible-1 (verification)
**Brief satisfied:** `briefs/forge-4_cache_compiler.md` D1-D3
**Status:** Phase A delivered. 38 / 38 pytest tests pass on this host (system Python 3.14, `tests/forge4/`).

---

## Files delivered

| Path | Status | Public symbols |
|---|---|---|
| `core/__init__.py` | NEW | (package marker) |
| `core/engine_cache.py` | NEW | `EngineCacheManager`, `BuildArtifact` |
| `core/background_compiler.py` | NEW | `BackgroundCompiler`, `BackendRegistry`, `CompileJob` |
| `core/comfyui_telemetry.py` | NEW | `record_generation`, `record_compile`, `ensure_schema`, `fetch_recent_generations`, `fetch_recent_compiles` |
| `tests/forge4/conftest.py` | NEW | (sys.path injector for repo root) |
| `tests/forge4/test_engine_cache.py` | NEW | 19 tests, including the 100 MiB roundtrip + 60s in-use eviction guard |
| `tests/forge4/test_background_compiler.py` | NEW | 15 tests, including idle-gating, cancel-mid-compile orphan check, restart restoration |
| `tests/forge4/test_telemetry.py` | NEW | 4 tests, schema + roundtrip + filter |

No `acceleration/` files modified. No `core/gpu_optimizer_v6.py` modified. No `core/experts/` files touched. All Forge-4 modules import only from `acceleration.backend` (Forge-1's ABC + dataclasses). Workload-neutrality and role isolation preserved.

---

## D1. EngineCacheManager

**Import:** `from core.engine_cache import EngineCacheManager, BuildArtifact`

**Constructor:**

```python
EngineCacheManager(
    cache_dir: Path,
    db_path: Path = Path("data/engine_cache.db"),
    max_disk_bytes: int = 200 * 1024**3,   # 200 GiB default
    in_use_window_s: int = 60,             # eviction lockout window
)
```

On construction:
- creates `<cache_dir>/comfyui/`, `<cache_dir>/houdini/`, `<cache_dir>/ue5/`,
- creates `<db_path>.parent/`,
- runs `CREATE TABLE IF NOT EXISTS artifacts ...` (idempotent).

**Public API:**

```python
def compute_key(self, app, model_sha256, shape_w, shape_h, batch,
                precision, backend_name, backend_ver,
                driver_ver="596.36", sm_arch=89) -> str
    # 32-char blake2b. `app` is FIRST in the digest input -> no cross-app
    # collisions. Raises ValueError on app not in {"comfyui","houdini","ue5"}.

def lookup(self, hash_key: str) -> Optional[BuildArtifact]
    # Miss / unknown / unparseable -> None (never raises).
    # Self-heals: if the row exists but the file vanished, drops the row.
    # Hit -> updates last_used_utc, increments use_count.

def lookup_by_params(self, app, model_sha256, shape_w, shape_h, batch,
                     precision, backend_name, backend_ver,
                     driver_ver="596.36", sm_arch=89) -> Optional[BuildArtifact]
    # Convenience: compute_key + lookup.

def store(self, hash_key: str, plan_bytes: bytes,
          descriptor: BuildArtifactDescriptor,
          app: str = None,                # required: pass explicitly OR via descriptor.notes["app"]
          model_sha256: str = "",
          shape_w: int = 0, shape_h: int = 0, batch: int = 1,
          driver_ver: str = "596.36", sm_arch: int = 89,
          ) -> BuildArtifact
    # Atomic file write (.plan.tmp -> os.replace -> .plan).
    # ON CONFLICT DO UPDATE on hash_key (idempotent re-store works).
    # SQLite write retries 3x with backoff; raises after that with the
    # blocker hint from the brief ("data/ permissions; need user resolution").

def evict_lru(self, target_bytes: int = None,
              target_app: str = None) -> int
    # Returns bytes freed.
    # Default target_bytes = 90% of max_disk_bytes.
    # Refuses to evict an entry whose last_used_utc is within in_use_window_s
    # (default 60s).
    # Greedy oldest-first; logs and returns partial when target unreachable.

def disk_used_bytes(app: str = None) -> int
def disk_used_gb(app: str = None) -> float
def stats() -> dict
    # {"cache_dir","db_path","max_disk_bytes","total_bytes","total_gb",
    #  "per_app": {app: {"count","bytes","gb"}}}.

def invalidate_app(app: str) -> int
def invalidate_on_driver_change(current_driver: str) -> int
    # Drops every row whose driver_ver != current_driver. Call on startup.
def invalidate_stale_models(model_dir: Path) -> int
    # Hashes every file under model_dir; for each cached entry whose
    # model_sha256 has a newer mtime on disk than created_utc, evicts.
```

**`BuildArtifact` dataclass** (returned by `lookup` / `store`):

```python
@dataclass
class BuildArtifact:
    hash_key: str
    app: str
    artifact_path: Path
    descriptor: BuildArtifactDescriptor   # rehydrated; artifact_bytes == b"" (lazy)
    disk_bytes: int
    compile_seconds: Optional[float]
    runtime_seconds_p50: Optional[float]
    created_utc: str
    last_used_utc: str
    use_count: int

    def read_bytes(self) -> bytes        # lazy load from disk
```

**SQLite schema** (`data/engine_cache.db`, table `artifacts`):
- `hash_key TEXT PRIMARY KEY`
- `app TEXT NOT NULL`, `model_sha256 TEXT NOT NULL`
- `shape_w INTEGER, shape_h INTEGER, batch INTEGER, precision TEXT`
- `backend_name TEXT, backend_ver TEXT, driver_ver TEXT, sm_arch INTEGER`
- `artifact_path TEXT NOT NULL, artifact_format TEXT, disk_bytes INTEGER NOT NULL`
- `compile_seconds REAL, runtime_seconds_p50 REAL`
- `notes_json TEXT, shape_min_json TEXT, shape_opt_json TEXT, shape_max_json TEXT`
- `created_utc TEXT NOT NULL, last_used_utc TEXT NOT NULL, use_count INTEGER NOT NULL DEFAULT 0`
- Indexes: `idx_app_lru(app, last_used_utc)`, `idx_app_model(app, model_sha256)`.

WAL mode + `synchronous=NORMAL`. `check_same_thread=False` so the cache is callable from the BackgroundCompiler worker thread.

**Storage layout:**
```
<cache_dir>/
  comfyui/<hash_key>.plan
  houdini/<hash_key>.plan
  ue5/<hash_key>.plan
```

---

## D2. BackgroundCompiler

**Import:** `from core.background_compiler import BackgroundCompiler, BackendRegistry, CompileJob`

**Constructor:**

```python
BackgroundCompiler(
    engine_cache: EngineCacheManager,
    idle_threshold_pct: float = 5.0,
    idle_dwell_seconds: int = 30,
    max_concurrent: int = 1,
    poll_interval_s: float = 5.0,
    registry: BackendRegistry = None,        # Forge-3 should pass its own
    idle_probe = None,                        # _GPUIdleProbe by default
    queue_path: Path = Path("data/compile_queue.json"),
    telemetry_path: Path = Path("data/comfyui_telemetry.db"),
)
```

On construction: ensures `queue_path.parent` exists, then **restores any prior queue from disk**. Running jobs from the previous process are re-classified as pending with `last_error="interrupted_by_restart"`.

**Public API:**

```python
def start() -> None
    # Idempotent. Spawns the daemon worker thread.

def stop(timeout: float = 30.0) -> None
    # Signals all in-flight CancelTokens with reason="shutdown".
    # In-flight jobs end up requeued (their .compile() raises
    # CompileCancelled, _handle_cancel branches to requeue).

def enqueue(compile_params: dict) -> str
    # Returns job_id. Persists the queue.
    # compile_params required keys: "app", "backend_name".
    # See "compile_params shape" below.

def cancel(job_id: str) -> bool
    # Pending: removed.
    # Running: cancel_token fired; backend.compile raises CompileCancelled;
    #          the job goes to "cancelled" (NOT requeued -- explicit user
    #          cancel signal).

def pause_app(app: str, reason: str = "foreground_active") -> int
    # Sets pause_reason on all pending jobs for that app and fires cancel
    # tokens on running jobs. Cancelled-via-pause jobs are REQUEUED (not
    # marked failed). Returns count affected.

def resume_app(app: str) -> int

def status() -> dict
    # {
    #   "queue_depth_per_app": {app: count},
    #   "pending_total": int, "completed_total": int, "failed_total": int,
    #   "current_jobs": [{job_id, app, backend, state, attempts}, ...],
    #   "paused_apps": [...],
    #   "idle_state": {running, dwell_seconds_observed, dwell_required, threshold_pct},
    #   "estimated_time_remaining": float (seconds, summed via backend priors),
    #   "registered_backends": [name, ...],
    # }
```

**`compile_params` shape** (the orchestrator builds this dict; backend-agnostic):

```python
{
    "app":             "comfyui" | "houdini" | "ue5",      # required
    "backend_name":    "tensorrt" | "sage_attention_v2" | ...,  # required
    "backend_ver":     "1.0.0",                            # used for cache key

    "model_sha256":    "<64 hex chars>",
    "shape_w":          int,
    "shape_h":          int,
    "batch":            int,
    "precision":       "fp16" | "bf16" | "fp8" | ...,
    "model_kind":      "unet" | "gemm" | "attention" | ..., # for backend.applies_to
    "driver_ver":      "596.36",                            # default if absent
    "sm_arch":          89,                                  # default if absent
    "priority":         int,                                 # higher = pops first

    "source": {                  # required
      # one of:
      "kind": "onnx",           "path": "C:\\...\\model.onnx", "opset": 17,
      # or:
      "kind": "torch_module",   "pickle_bytes": <bytes-or-base64>, "module_class_qualname": "...",
      # or:
      "kind": "trt_plan",       "plan_bytes": <bytes-or-base64>,
    },

    "shape_range": {             # optional; falls back to static((batch,4,h,w))
      "min": [1, 4, 64, 64],
      "opt": [1, 4, 64, 64],
      "max": [1, 4, 64, 64],
    },
}
```

**BackendRegistry:**

```python
reg = BackendRegistry()
reg.register(TensorRTBackend())
reg.register(SageAttentionV2Backend())
reg.register(CutlassFP8GemmBackend())  # enabled=False in Phase A; that's OK
reg.get(name) -> Optional[AccelerationBackend]
reg.names() -> List[str]
```

Forge-3 owns instantiation and registration of Forge-1's backends. `BackgroundCompiler` simply consumes the registry.

**Compile loop (one tick):**
1. Read `idle_probe.utilization_pct()`.
2. If util > `idle_threshold_pct` -> reset `_idle_dwell_counter` and continue.
3. Else accumulate elapsed time into `_idle_dwell_counter`.
4. If counter >= `idle_dwell_seconds` AND a free slot exists AND a dispatchable job exists (not in `_paused_apps`, no `pause_reason`) -> pop + run.
5. Sleep `poll_interval_s` (interruptible via stop_event).

**Compile execution:**
1. Resolve backend from registry; hard-fail if missing or `.enabled=False`.
2. Materialize `BackendSource` + `ShapeRange` from the params dict.
3. Call `backend.compile(source, shape_range, precision, cancel_token, progress_cb)`.
4. Compute the cache key (matches `EngineCacheManager.compute_key` exactly).
5. `engine_cache.store(hash_key, descriptor.artifact_bytes, descriptor, app=...)`.
6. Write a row to `compiles` table in `comfyui_telemetry.db` (compile-side telemetry).
7. Mark job completed; persist queue.

**Failure modes:**
- `CompileCancelled` from `pause:*` or `foreground_active` -> requeue (state=pending), keep `attempts`.
- `CompileCancelled` from explicit `cancel(job_id)` -> mark cancelled (no requeue).
- `CompileCancelled` from `shutdown` -> requeue so next start picks up.
- Other exception -> increment `attempts`; up to 3 attempts; after that mark failed.
- Backend not registered or `.enabled=False` -> hard fail (no retries).

**Cancel mid-compile cleanup:** after every job, regardless of outcome, the worker scans `<cache_dir>/<app>/*.plan.tmp` and unlinks any orphans. The brief's invariant "Cancel mid-compile leaves no `.plan.tmp` orphans" is enforced by both `EngineCacheManager.store`'s try/finally AND this post-job sweep. Verified by `test_cancel_mid_compile_leaves_no_tmp_orphans`.

**Queue persistence:** every `enqueue` / `dequeue` / `pause` / `resume` / state transition writes `data/compile_queue.json` atomically (`.tmp` -> `os.replace`). On restart the constructor re-hydrates pending jobs and re-classifies running jobs as pending. Verified by `test_queue_persists_across_restart` and `test_running_jobs_are_requeued_after_restart`.

**GPU-idle probe:** `_GPUIdleProbe` lazy-imports `pynvml`. Three init attempts; after that the probe permanently reports 0% util and logs the brief's blocker hint ("GPU monitoring unavailable; cannot idle-gate"). Tests inject a `FakeIdleProbe` so they don't need a live GPU.

---

## D3. Telemetry schema + client

**Import:** `from core.comfyui_telemetry import record_generation, record_compile, ensure_schema, fetch_recent_generations, fetch_recent_compiles`

**Database file:** `data/comfyui_telemetry.db` (default; injectable via `BackgroundCompiler.telemetry_path`).

**Tables:**

```sql
-- Foreground generation rows. Forge-2's specialists / collectors call
-- record_generation(...) after every ComfyUI generation. Schema is locked
-- to the brief.
CREATE TABLE generations (
    id              INTEGER PRIMARY KEY,
    ts_utc          TEXT NOT NULL,
    workflow_hash   TEXT,
    model_sha256    TEXT,
    shape_w         INTEGER,
    shape_h         INTEGER,
    batch           INTEGER,
    sampler         TEXT,
    steps           INTEGER,
    precision       TEXT,
    backend_set     TEXT,        -- "sage_attn,trt_unet"
    wall_seconds    REAL,
    samples_per_sec REAL,
    vram_peak_gb    REAL
);
CREATE INDEX idx_workload ON generations(model_sha256, shape_w, shape_h);

-- Compile-side rows. BackgroundCompiler writes one per successful build.
CREATE TABLE compiles (
    id               INTEGER PRIMARY KEY,
    ts_utc           TEXT NOT NULL,
    app              TEXT NOT NULL,    -- "comfyui" | "houdini" | "ue5"
    backend_name     TEXT,
    backend_version  TEXT,
    model_sha256     TEXT,
    shape_w          INTEGER,
    shape_h          INTEGER,
    batch            INTEGER,
    precision        TEXT,
    compile_seconds  REAL,
    artifact_bytes   INTEGER,
    notes_json       TEXT
);
CREATE INDEX idx_compile_workload ON compiles(model_sha256, shape_w, shape_h);
```

`ensure_schema(db_path)` is idempotent (safe to call from every collector); module-level cache avoids the round-trip on every call.

**Client signatures:**

```python
def record_generation(
    db_path: Path, ts_utc: str,
    workflow_hash: str | None, model_sha256: str | None,
    shape_w: int, shape_h: int, batch: int,
    sampler: str | None, steps: int | None,
    precision: str | None,
    backend_set: list[str],     # joined as comma-separated
    wall_seconds: float, samples_per_sec: float, vram_peak_gb: float,
) -> int    # row id

def record_compile(
    db_path: Path, ts_utc: str, app: str,
    backend_name: str, backend_version: str,
    model_sha256: str,
    shape_w: int, shape_h: int, batch: int, precision: str,
    compile_seconds: float, artifact_bytes: int,
    notes: dict | None = None,
) -> int    # row id
```

Read helpers (used by the dashboard):

```python
fetch_recent_generations(db_path, limit=50, model_sha256=None) -> list[dict]
fetch_recent_compiles(db_path, limit=50, app=None) -> list[dict]
```

---

## Expected startup behavior (Forge-3, wire this in `__init__`)

```python
from pathlib import Path
from core.engine_cache import EngineCacheManager
from core.background_compiler import BackgroundCompiler, BackendRegistry
from acceleration.tensorrt_engine import TensorRTBackend
from acceleration.sage_attention import SageAttentionV2Backend
from acceleration.cutlass_integration import CutlassFP8GemmBackend

# 1. Cache (creates D:\OptimizerV8_EngineCache\{comfyui,houdini,ue5}\
#    + data/engine_cache.db on first run).
cache = EngineCacheManager(
    cache_dir=Path(config.tensorrt.cache_dir),  # default D:\OptimizerV8_EngineCache
    db_path=Path("data/engine_cache.db"),
    max_disk_bytes=int(config.tensorrt.max_cache_gb * 1024**3),
)

# Optional: drop entries whose driver_ver != current.
cache.invalidate_on_driver_change(current_driver=detect_driver())

# 2. Backend registry + Forge-1 backends.
reg = BackendRegistry()
reg.register(TensorRTBackend())
reg.register(SageAttentionV2Backend())
reg.register(CutlassFP8GemmBackend())  # Phase A: enabled=False; that's fine

# 3. Background compiler (restores queue from disk on construction).
bc = BackgroundCompiler(
    engine_cache=cache,
    idle_threshold_pct=config.tensorrt.idle_threshold_pct,    # default 5.0
    idle_dwell_seconds=config.tensorrt.idle_dwell_seconds,    # default 30
    poll_interval_s=5.0,
    max_concurrent=1,
    registry=reg,
    queue_path=Path("data/compile_queue.json"),
    telemetry_path=Path("data/comfyui_telemetry.db"),
)
bc.start()

# 4. On orchestrator shutdown:
# bc.stop(timeout=30.0)
```

**Idle polling** runs continuously in the daemon thread; the main orchestrator does NOT need to drive it. Status snapshots come from `bc.status()`.

**CrossAppCoordinator (Forge-2)** signals foreground activity via `bc.pause_app("comfyui")` and `bc.resume_app("comfyui")`. Pausing an app fires cancel tokens on its running jobs and refuses to dispatch its pending jobs. Cancelled-via-pause jobs are requeued.

---

## Paths + env vars

**Default cache dir:** `D:\OptimizerV8_EngineCache\{comfyui,houdini,ue5}\<hash_key>.plan`. On test environments where `D:\` is not writable, pass `cache_dir=tmp_path / "EngineCache"` (the orchestrator does this via `config.tensorrt.cache_dir`).

**Default DB paths (relative to repo root, since the launcher cd's into the repo):**
- `data/engine_cache.db` (Forge-4 owns)
- `data/compile_queue.json` (Forge-4 owns)
- `data/comfyui_telemetry.db` (Forge-4 owns; Forge-2 writes generation rows)

**Env vars consumed (none required; all paths injectable):** none. Forge-4's modules don't read env vars directly. The orchestrator surfaces config via constructor args.

**Env vars Forge-4 MAY consume in a later phase (informational, NOT used today):**
- `OPTIMIZER_V8_CACHE_DIR` -- override `cache_dir`.
- `OPTIMIZER_V8_TELEMETRY_DB` -- override `telemetry_path`.

These are NOT wired today; if Forge-3 wants env-var override, the orchestrator should resolve env -> config and pass via constructor.

---

## Cache-key invariants (Crucible-1)

Verified in tests:
- `test_compute_key_app_first_no_cross_app_collision` -- `app|` is FIRST in the digest input; identical other-fields across `comfyui`/`houdini`/`ue5` produce different keys.
- `test_compute_key_changes_on_any_field` -- perturbing any of `{shape_w, shape_h, batch, precision, backend_name, backend_ver, driver_ver, sm_arch, model_sha256}` changes the digest.
- `test_compute_key_is_32_hex` -- always 32 lowercase hex chars (parses via `int(key, 16)`).
- `test_compute_key_rejects_bad_app` -- only the three named apps pass.

---

## Done criteria (verified)

| Criterion | Test |
|---|---|
| `lookup(unknown_key)` returns None | `test_lookup_unknown_key_returns_none` |
| Store -> lookup roundtrip works for >= 100MB plan bytes | `test_store_then_lookup_100mb` |
| Eviction frees >= requested bytes when over budget | `test_evict_lru_frees_at_least_requested` |
| Engine in use within last 60s is NOT evicted | `test_evict_lru_refuses_in_use_entries` |
| BackgroundCompiler does not start compile while GPU > 5% utilized | `test_does_not_start_compile_while_gpu_busy` |
| Cancel mid-compile leaves no `.plan.tmp` files behind | `test_cancel_mid_compile_leaves_no_tmp_orphans` |
| Queue persists across orchestrator restart | `test_queue_persists_across_restart` |
| Cache key starts with `app\|` | `test_compute_key_app_first_no_cross_app_collision` |
| 38 / 38 forge-4 pytest tests pass | full suite |

---

## Run the suite

```
cd C:\Users\User\OneDrive\Desktop\Optimizer_V7_G3
C:\Python314\python.exe -m pytest tests/forge4/ -v
```

Set `FORGE4_SKIP_LARGE=1` in env to skip the 100 MiB roundtrip test on slow filesystems.

---

## Open questions / non-blocker observations

- **`invalidate_stale_models` rehashes every file under `model_dir`.** For a 200 GB models tree this is slow. Recommended call site: orchestrator startup only, behind an opt-in config flag. Phase B can switch to mtime-only heuristic with periodic deep-rehash sampling.
- **`pynvml` is deprecated** (warning observed in test output). Forge-1 also depends on this; recommend a coordinated migration to `nvidia-ml-py` in Phase B.
- **Compile retry backoff is "scheduled by next idle window"** rather than time-based. If the orchestrator wants a hard 30s cooldown after a failure, Forge-3 should add `retry_after_utc` to `CompileJob` and we'll honour it; not added today since the brief didn't specify timing.
- **`generations` table** is shared by all three apps despite the historical "comfyui_telemetry.db" filename. The DB filename matches the brief; the schema accepts cross-app rows naturally via `backend_set`. If Forge-3 prefers per-app DBs, it's a one-line config change in `BackgroundCompiler.__init__`.
- **Single-input ONNX assumption** is inherited from Forge-1's TRT backend; we don't add it.

If anything in this handoff is wrong or missing, file a blocker rather than freelancing across the role boundary (Commandment 5).
