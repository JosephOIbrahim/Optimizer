# Forge-4 — EngineCacheManager + BackgroundCompiler + Telemetry persistence

**Read first:** `briefs/SHARED_CONTEXT.md`, `briefs/_handoff_forge1.md`.

## Authority boundary (Commandment 5)
- **You implement:** `core/engine_cache.py`, `core/background_compiler.py`, the SQLite schemas for `data/engine_cache.db` and `data/comfyui_telemetry.db`.
- **You do NOT touch:** `core/gpu_optimizer_v6.py`, `core/gpu_optimizer_v8.py` (Forge-3 imports your modules), `core/experts/*` (Forge-2), `acceleration/*` (Forge-1).
- You **may** depend on Forge-1's `AccelerationBackend` ABC (read-only).

## Deliverables

### D1. `core/engine_cache.py` — `EngineCacheManager`

```python
class EngineCacheManager:
    def __init__(self, cache_dir: Path, db_path: Path = Path("data/engine_cache.db"),
                 max_disk_bytes: int = 200 * 1024**3): ...

    def compute_key(self, app: str, model_sha256: str, shape_w: int, shape_h: int,
                    batch: int, precision: str, backend_name: str, backend_ver: str,
                    driver_ver: str = "596.36", sm_arch: int = 89) -> str:
        """blake2b → 32-char hex; matches SHARED_CONTEXT format."""

    def lookup(self, hash_key: str) -> Optional[BuildArtifact]: ...
    def store(self, hash_key: str, plan_bytes: bytes, descriptor: BuildArtifactDescriptor) -> BuildArtifact: ...
    def evict_lru(self, target_bytes: int = None, target_app: Optional[str] = None) -> int:
        """Evict LRU until disk usage < target_bytes (or 90% of max). Returns bytes freed."""

    def disk_used_gb(self, app: Optional[str] = None) -> float: ...
    def invalidate_app(self, app: str) -> int: ...
    def invalidate_on_driver_change(self, current_driver: str) -> int: ...
    def invalidate_stale_models(self, model_dir: Path) -> int:
        """Compare model file mtimes vs cache entries; evict mismatches."""
```

Storage layout:
- Cache root: `D:\OptimizerV8_EngineCache\<app>\<hash_key>.plan`
- SQLite at `data/engine_cache.db` schema:
  ```sql
  CREATE TABLE artifacts (
    hash_key TEXT PRIMARY KEY,
    app TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    shape_w INTEGER, shape_h INTEGER, batch INTEGER,
    precision TEXT,
    backend_name TEXT, backend_ver TEXT,
    driver_ver TEXT, sm_arch INTEGER,
    artifact_path TEXT NOT NULL,
    disk_bytes INTEGER NOT NULL,
    compile_seconds REAL,
    runtime_seconds_p50 REAL,
    created_utc TEXT NOT NULL,
    last_used_utc TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0
  );
  CREATE INDEX idx_app_lru ON artifacts(app, last_used_utc);
  ```

Eviction policy:
- LRU within `app` if `target_app` specified, else global LRU.
- Refuse to evict an artifact whose `last_used_utc` is within the last 60 seconds (prevents evicting an in-use engine).
- Free space until either target met or all eligible candidates exhausted; if can't reach target, log + return what was freed.

### D2. `core/background_compiler.py` — `BackgroundCompiler`

```python
class BackgroundCompiler:
    def __init__(self, engine_cache: EngineCacheManager,
                 idle_threshold_pct: float = 5.0,
                 idle_dwell_seconds: int = 30,
                 max_concurrent: int = 1): ...

    def enqueue(self, compile_params: dict) -> str:
        """Returns job_id. Persists queue to data/compile_queue.json so it survives restart."""

    def pause_app(self, app: str) -> int:
        """Pauses pending+running compiles for that app. Used by CrossAppCoordinator."""

    def resume_app(self, app: str) -> int: ...

    def cancel(self, job_id: str) -> bool: ...

    def status(self) -> dict:
        """{queue_depth_per_app, current_job, idle_state, estimated_time_remaining}"""
```

Compile loop (runs in background thread, started when orchestrator boots):
1. Poll system idle: read GPU utilization via `pynvml`. If > `idle_threshold_pct` → reset dwell counter, sleep 5s, repeat.
2. Once dwell counter reaches `idle_dwell_seconds` → pop next job from queue.
3. Resolve `Backend` from `compile_params["backend_name"]` (Forge-1's registry).
4. Call `backend.compile(source, shape_range, precision, cancel_token, progress_cb)`.
5. On completion: `engine_cache.store(...)`. Telemetry log: write `(app, model, shape, compile_seconds)` row.
6. On `cancel_token` triggered (foreground app started): cooperatively abort, write `.plan.tmp` cleanup, requeue job.
7. On exception: 3 retries with backoff; after 3 failures, mark job failed and surface to next orchestrator cycle.

Cancel mechanism:
- `CancelToken` is a simple `threading.Event` wrapper. Backend's progress_cb checks `is_set()` periodically and returns early.
- TRT supports interruption via builder progress monitor — wire this.
- CUTLASS NVCC compiles get `subprocess.terminate()` on cancel.

### D3. `data/comfyui_telemetry.db` schema (Forge-2 writes rows; you own schema migration)

```sql
CREATE TABLE generations (
  id INTEGER PRIMARY KEY,
  ts_utc TEXT NOT NULL,
  workflow_hash TEXT,
  model_sha256 TEXT,
  shape_w INTEGER, shape_h INTEGER, batch INTEGER,
  sampler TEXT, steps INTEGER,
  precision TEXT,
  backend_set TEXT,    -- comma-separated: "sage_attn,trt_unet"
  wall_seconds REAL,
  samples_per_sec REAL,
  vram_peak_gb REAL
);
CREATE INDEX idx_workload ON generations(model_sha256, shape_w, shape_h);
```

Provide a thin client `core/comfyui_telemetry.py` with `record_generation(...)` that Forge-2's collectors call.

## Sequence
1. **Scout.** Read `acceleration/cutlass_integration.py` style (ABC + dataclass + manager) — match it.
2. Implement `EngineCacheManager` minus eviction. Pytest: `lookup` miss returns None; `store` writes file + DB row; `lookup` after store hits.
3. Add eviction + tests (Crucible-1 will hammer this).
4. Implement `BackgroundCompiler` skeleton (queue, persistence, idle polling). Pytest: enqueue → status reflects queue.
5. Wire actual compile call (calls Forge-1 backend). Pytest with a stub backend that sleeps 2s.
6. Add cancel + preemption. Pytest: enqueue + cancel before idle dwell elapses → never compiles.
7. Add telemetry schema + client.

## Bounded failure
- SQLite write 3× → blocker: "data/ permissions; need user resolution."
- pynvml init 3× → blocker: "GPU monitoring unavailable; cannot idle-gate."
- TRT plan deserialization fails on lookup hit → invalidate that entry, return None, do not raise.

## Done criteria (Crucible-1 verifies)
- `EngineCacheManager.lookup(unknown_key)` returns None.
- Store→lookup roundtrip works for ≥ 100MB plan bytes.
- Eviction frees ≥ requested bytes when over budget.
- Engine in use within last 60s is NOT evicted.
- BackgroundCompiler does not start compile while GPU > 5% utilized.
- Cancel mid-compile leaves no `.plan.tmp` files behind.
- Queue persists across orchestrator restart.

## Handoff
Write `briefs/_handoff_forge4.md` listing: public class signatures, paths, env vars, expected behavior on startup (cache dir creation, queue restoration). Forge-3 reads this.
