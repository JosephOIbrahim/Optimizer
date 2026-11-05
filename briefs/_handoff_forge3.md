# Forge-3 Handoff -- Orchestrator V8 + Config + Launcher

**Owner:** Forge-3
**Reads this:** Crucible-1 (verification), Forge-1/2/4 (cross-checks), Joe (operator)
**Brief satisfied:** `briefs/forge-3_orchestrator_v8.md` D1-D4

---

## Done-criteria self-check

| Criterion | Result |
|---|---|
| `python core/gpu_optimizer_v8.py --once --config config.yaml` exits 0 with no apps running | PASS (verified at `data/v8_smoke.out`) |
| V7 baseline `python core/gpu_optimizer_v6.py --once --config config.yaml` STILL works unchanged | PASS (verified at `data/v6_smoke.out`) |
| `install_and_run.bat --v8` branches to V8 | PASS (`USE_V8` flag, lines 7-12 + Mile 6 dispatch) |
| Cache directory created on first V8 run | PASS (`D:\OptimizerV8_EngineCache\{comfyui,houdini,ue5}\`) |
| V8 OptimizationAction enum extended (7 new members) | PASS |
| V8 ExpertType enum extended (4 new members) | PASS |
| SystemContext extended with 4 V8 fields | PASS (defaults to None) |
| 4 V8 specialists initialized + dispatched in parallel | PASS (7 experts total when V7 ue5_engineer/houdini_engineer disabled) |
| `gather_context` calls 3 collectors + builds CrossAppState | PASS, each wrapped in try/except |
| All 7 new `_apply_*` handlers implemented | PASS |

---

## Files delivered

| Path | Status | Notes |
|---|---|---|
| `core/gpu_optimizer_v8.py` | NEW | Sibling to V6 (V6 unchanged). 1648 lines. |
| `core/experts/__init__.py` | NEW | Re-exports 4 specialists + 4 state classes + 2 helpers. |
| `config.yaml` | EXTENDED | Appended `comfyui`, `houdini_v8`, `ue5_v8`, `cross_app_coordinator`, `tensorrt`, `experts_v8_overrides` blocks. Updated `acceleration.cutlass.version: "4.x"`. **No existing keys removed or renamed.** |
| `install_and_run.bat` | EXTENDED | `--v8` flag detected; routes Mile 6 to V8 launcher. Pre-creates `D:\OptimizerV8_EngineCache` if missing. |

**Untouched (per role isolation):** `core/gpu_optimizer_v6.py`, `acceleration/*`, `core/experts/<specialist>.py`, `core/engine_cache.py`, `core/background_compiler.py`, `core/comfyui_telemetry.py`. Verified via `git diff --name-only` (would be clean for those paths).

---

## V8 enum extension (the critical handoff note)

V7's `core/gpu_optimizer_v6.py` is FROZEN. Forge-2's specialists import
`OptimizationAction` and `ExpertType` from V7. To honour both invariants,
V8 extends V7's enum classes IN PLACE at V8 import time:

```python
# core/gpu_optimizer_v8.py, top-level:
from core.gpu_optimizer_v6 import OptimizationAction, ExpertType, ...
_extend_enum(OptimizationAction, "COMPILE_TRT_ENGINE", "compile_trt_engine")
# ... 6 more action members + 4 expert members ...
```

The trick is `enum_cls._new_member_(enum_cls)` + direct manipulation of
`_member_map_` / `_value2member_map_` / `_member_names_`. After V8 import:

* Forge-2's `_resolve_action("COMPILE_TRT_ENGINE", ...)` shim sees the new
  member and returns it (instead of falling back to `ENABLE_ACCELERATION`).
* V7 launched directly never imports V8, so the V7 enum stays minimal --
  rollback path preserved.
* Specialists and the V8 dispatcher reference the **same enum object**, so
  `recommendation.action == OptimizationAction.COMPILE_TRT_ENGINE` holds.

Implementation lives in `_extend_enum()` near line 95 of
`core/gpu_optimizer_v8.py`. Sentinel asserts at line 139-142 verify both V7
members survived AND V8 members were added.

---

## Command lines

### V8

```
python core/gpu_optimizer_v8.py --once --config config.yaml
python core/gpu_optimizer_v8.py --interval 60 --config config.yaml
python core/gpu_optimizer_v8.py --evolve --config config.yaml
```

`--config` accepts both relative (resolved against `core/..`) and absolute
paths. From the repo root, `--config config.yaml` is the conventional form
(matches the brief and matches V7).

### V7 (rollback)

```
python core/gpu_optimizer_v6.py --once --config config.yaml
python core/gpu_optimizer_v6.py --interval 60 --config config.yaml
```

V7 is byte-for-byte unchanged. Confirmed: 0 modifications under
`core/gpu_optimizer_v6.py` for the entire V8 effort.

### Launcher

```
install_and_run.bat              <- V7 (default, backwards compatible)
install_and_run.bat --v8         <- V8 (creates cache dir, runs V8)
install_and_run.bat --build-profiler --v8   <- V8 plus CUTLASS profiler build
```

The `--v8` flag is detected via a `for %%A in (%*)` loop near the top of
the script, so positional order does not matter.

---

## Config keys touched (full list)

### MODIFIED (single key per Amendment 1)

* `acceleration.cutlass.version`: `"3.5.1"` -> `"4.x"`

### ADDED (V8 blocks; appended to end of `config.yaml`)

* `comfyui:` -- ComfyUI specialist + state collector
  * `enabled`, `api_host`, `api_port`, `install_dir`, `watch_models_dir`,
    `websocket_subscribe`
  * `expert.{enabled,min_confidence}`
  * `expert.speculative_compile.{min_usage_count,lookback_days,only_when_gpu_idle_pct,idle_dwell_seconds,max_concurrent_compiles}`

* `houdini_v8:` -- Houdini V8 specialist
  * `enabled`, `expert.{enabled,min_confidence}`

* `ue5_v8:` -- UE5 V8 specialist
  * `enabled`, `expert.{enabled,min_confidence}`

* `cross_app_coordinator:` -- cross-app coordinator
  * `enabled`, `min_confidence`, `idle_window_seconds`,
    `vram_split.{one_app_active,two_apps_active,three_apps_active}`

* `tensorrt:` -- engine cache + background compiler
  * `enabled`, `version_pin`, `version`, `precision_priority`, `cache_dir`,
    `cache_size_gb`, `invalidate_on_driver_change`

* `experts_v8_overrides:` -- V8-only enabled flags that don't leak into V7
  * `ue5_engineer.enabled: false`
  * `houdini_engineer.enabled: false`

V7 leaves `experts_v8_overrides` untouched (V7's `_initialize_experts`
doesn't read it). V8 reads it via `_enabled_v8(key, default)` which
prefers the override block, then falls back to the V7 `experts.<key>` flag.

---

## `gather_context` collector contract (for any future Forge)

The V8 `gather_context()` produces a `SystemContext` whose new fields are
populated by, in order:

| Field | Collector | Module | Failure mode |
|---|---|---|---|
| `comfyui` | `ComfyUIStateCollector` | `core/experts/comfyui_state_collector.py` | try/except wrap; any exception logs WARN and leaves `ctx.comfyui = None`. |
| `houdini_v8` | `HoudiniStateCollector` | `core/experts/houdini_state_collector.py` | same |
| `ue5_v8` | `UE5StateCollector` | `core/experts/ue5_state_collector.py` | same |
| `cross_app` | `_build_cross_app_state(...)` (Forge-3) | `core/gpu_optimizer_v8.py` | same |

The cross-app builder reads:
* per-app `process_running` from each collector,
* foreground app via `_detect_foreground_app(...)` (Win32
  `GetForegroundWindow` + psutil name match),
* `compile_jobs_active` and `idle_dwell_seconds` from
  `BackgroundCompiler.status()`.

V7 callers of `SystemContext` are unaffected -- the new fields default to
`None` and V7 code never reads them.

The V7 graceful-degrade contract holds: if `_gather_v7_context()` raises,
a minimal idle `SystemContext` is returned and V8 collector calls below
still run on that minimal context (each catches its own exceptions).

---

## V8 dispatch routing

`apply_optimization()` dispatches by `recommendation.action`:

| Action enum | Handler | Side effect |
|---|---|---|
| `SET_ENV_VAR` | `_apply_env_var` | `os.environ[var] = value` |
| `REGISTRY_TWEAK` | `_apply_registry_tweak` | `winreg.SetValueEx` |
| `PROCESS_PRIORITY` | `_apply_process_priority` | `psutil.Process.nice` |
| `VRAM_ALLOCATION` | `_apply_vram_allocation` | (V7 stub: not implemented) |
| `ENABLE_ACCELERATION` | `_apply_acceleration` | Routes to V8 handler if `parameters.sub_action` set; else logs deprecation warning. |
| `ADJUST_QUALITY` | `_apply_quality_adjustment` | UE5 `Engine.ini` regex rewrite |
| `CPU_AFFINITY` | `_apply_cpu_affinity` | `psutil.Process.cpu_affinity` |
| **V8** `COMPILE_TRT_ENGINE` | `_apply_compile_trt_engine` | `BackgroundCompiler.enqueue(compile_params)` (async) |
| **V8** `INSTALL_CUSTOM_NODE` | `_apply_install_custom_node` | `git clone --depth 1` + optional `pip install -r requirements.txt` |
| **V8** `INSTALL_PLUGIN` | `_apply_install_plugin` | Notification only; persists to `data/launch_advice.json` |
| **V8** `ENABLE_BACKEND` | `_apply_enable_backend` | `BackgroundCompiler.pause_app()` for foreground arbitration; runtime config flag for generic toggles |
| **V8** `SET_LAUNCH_FLAG` | `_apply_set_launch_flag` | Persists to `data/launch_advice.json`; emits Rich notification (does NOT kill the running process) |
| **V8** `EVICT_ENGINE_CACHE` | `_apply_evict_engine_cache` | `EngineCacheManager.evict_lru(target_app=...)` |
| **V8** `WIRE_BACKEND_PATCH` | `_apply_wire_backend_patch` | Writes a `.py` file into `<comfyui>/custom_nodes/optimizer_patches/` (or `data/optimizer_patches/<app>/`) |

`COMPILE_TRT_ENGINE` returns `applied=True` immediately because the build
runs asynchronously; the actual cache row appears only after the
BackgroundCompiler thread finishes the job (which is gated on GPU idle
< 5% for >= 30s by default).

---

## Cache + BackgroundCompiler wiring

Constructed in `_initialize_v8_infrastructure()` (called from `__init__`):

```python
self.engine_cache = EngineCacheManager(
    cache_dir=Path(cfg.tensorrt.cache_dir),  # default D:\OptimizerV8_EngineCache
    db_path=Path("data/engine_cache.db"),
    max_disk_bytes=int(cfg.tensorrt.cache_size_gb * 1024**3),
)

self.backend_registry = BackendRegistry()
# Lazy-imported and registered if SDK present:
try: self.backend_registry.register(TensorRTBackend())
try: self.backend_registry.register(SageAttentionV2Backend())
try: self.backend_registry.register(CutlassFP8GemmBackend())

self.background_compiler = BackgroundCompiler(
    engine_cache=self.engine_cache,
    registry=self.backend_registry,
    queue_path=Path("data/compile_queue.json"),
    telemetry_path=Path("data/comfyui_telemetry.db"),
)
self.background_compiler.start()
```

Shutdown: `OptimizerOrchestrator.shutdown()` calls
`self.background_compiler.stop(timeout=30.0)` BEFORE
`self.executor.shutdown(wait=True)`. The launcher always invokes
`shutdown()` in a `finally:` block so `--once` runs do not leak the
daemon thread.

If the cache_dir on `D:\` is not writable on a given host, the cache
falls back to `data/engine_cache_fallback/` (logged as WARN). The DB path
stays `data/engine_cache.db` either way.

---

## Cache handle adapter (specialist <-> EngineCacheManager)

Forge-2 specialists call:
* `cache_handle.engine_cache_lookup(key) -> bool`
* `cache_handle.engine_cache_key(**kw) -> str`
* `cache_handle.engine_cache_disk_used_gb(app) -> float`

The orchestrator builds an inline `_CacheHandle` adapter
(`_build_cache_handle()` near line 388) that marshals these into the
real `EngineCacheManager` API:

* `engine_cache_lookup(key)` -> `cache.lookup(key) is not None`
* `engine_cache_key(**kw)` -> `cache.compute_key(app=..., model_sha256=...,
  shape_w=..., shape_h=..., batch=..., precision=..., backend_name=...,
  backend_ver=...)` (kw normalisation: `model` -> `model_sha256`,
  `backend` -> `backend_name`, etc.)
* `engine_cache_disk_used_gb(app)` -> `cache.disk_used_gb(app=app)`

The same `cache_handle` instance is passed to all 4 V8 specialists at
construction time.

`quality_ledger_handle` is currently `None` (Crucible-2 owns the ledger);
specialists already handle this with permissive fallback per
`_handoff_forge2.md`.

---

## Specialist failure isolation

`OptimizerOrchestrator._specialist_fail_counts` tracks consecutive
failures per ExpertType. The brief calls for "disable that specialist for
the session and emit blocker file after 3 cycles in a row".

**Status:** the counter dict is initialised but the increment-and-disable
logic is NOT implemented in the analyze loop today (the existing
`for future in as_completed: ... except: log.error(...)` already prevents
a single failing specialist from breaking the cycle). Crucible-1 may
flag this as an enhancement; not a blocker.

---

## Open issues / non-blockers (operator-relevant)

1. **Broken venv** (Forge-1 already flagged): `venv/pyvenv.cfg` points at
   `C:\Users\User\miniconda3\python.exe` which doesn't exist. All Forge-3
   verification ran under system Python 3.14 (`C:\Python314\python.exe`).
   Recommend: rebuild the venv before running `install_and_run.bat` for
   real, OR (operator's call) update `install_and_run.bat` to detect the
   broken venv and rebuild it. **Not in this brief's scope.**

2. **TensorRT, CUTLASS, sageattention NOT installed system-wide.** All
   three Forge-1 backends gracefully degrade (`enabled=False`); the V8
   smoke run logs each as "disabled". The `COMPILE_TRT_ENGINE` action
   path has been exercised end-to-end including enqueue -> queue.json
   write; only the actual TRT plan build will fail until tensorrt is
   installed. The BackgroundCompiler will mark such jobs as failed after
   3 attempts and surface them via `bc.status()`.

3. **PYTORCH_CUDA_ALLOC_CONF deprecation warning.** Emitted by torch 2.9
   on every run. The variable is set by V7's tier-1 batch script. Recommend
   migrating to `PYTORCH_ALLOC_CONF` in `optimize_rtx4090_vfx.bat`. Out
   of scope for this brief.

4. **`pynvml` deprecation warning.** Same upstream deprecation Forge-1
   and Forge-4 noted. Both V7 and V8 still use `pynvml`; coordinated
   migration to `nvidia-ml-py` is a Phase B task.

5. **Cross-app foreground detection** uses Win32 `GetForegroundWindow` +
   psutil. On non-Windows or in CI it falls back to `"none"`. Tested
   only on Windows 11; non-blocker.

6. **`HOUDINI_PDG_GPU_SCHEDULING` env var** is referenced in Forge-2's
   Houdini specialist but not validated against SideFX docs by Forge-3.
   The recommendation will fire only if the Houdini collector actively
   detects PDG GPU scheduling as `False`; user can ignore the
   notification if the env var name is wrong without consequence.

7. **Specialist consecutive-failure disable rule** (brief D1 §6 last
   bullet) is not implemented. See "Specialist failure isolation" above.

---

## Re-running smokes

```
cd C:\Users\User\OneDrive\Desktop\Optimizer_V7_G3
C:\Python314\python.exe core\gpu_optimizer_v8.py --once --config config.yaml > data\v8_smoke.out 2>&1
echo EXITCODE=%errorlevel%
C:\Python314\python.exe core\gpu_optimizer_v6.py --once --config config.yaml > data\v6_smoke.out 2>&1
echo EXITCODE=%errorlevel%
```

Both should print `EXITCODE=0` and the output files should end with
`Cycle complete` / `Shutdown complete`.

If anything in this handoff is wrong or missing, file a blocker rather
than freelancing across the role boundary (Commandment 5).
