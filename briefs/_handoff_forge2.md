# Forge-2 Handoff -- V8 Specialists

**Owner:** Forge-2
**Reads this:** Forge-3 (orchestrator + enum + config + SystemContext wiring), Crucible-1 (verification)
**Brief satisfied:** `briefs/forge-2_specialists.md` Sections A / B / C / D
**Status:** All 4 specialists implemented under `core/experts/`. Each imports cleanly under system Python 3.14 (`C:\Python314\python.exe`). Each returns non-empty `List[Recommendation]` when its app is detected with the conditions specified in the brief.

---

## Files delivered

| Path | Status | Public symbols |
|---|---|---|
| `core/experts/comfyui_specialist.py` | NEW | `ComfyUISpecialist` |
| `core/experts/comfyui_state_collector.py` | NEW | `ComfyUIState`, `ComfyUIStateCollector`, `WorkloadEvent` |
| `core/experts/houdini_specialist.py` | NEW | `HoudiniSpecialist` |
| `core/experts/houdini_state_collector.py` | NEW | `HoudiniState`, `HoudiniStateCollector` |
| `core/experts/ue5_specialist.py` | NEW | `UE5Specialist` |
| `core/experts/ue5_state_collector.py` | NEW | `UE5State`, `UE5StateCollector` |
| `core/experts/cross_app_coordinator.py` | NEW | `CrossAppCoordinator`, `CrossAppState`, `UsageHistoryEntry`, `PrefetchCandidate`, `vram_budget`, `predict_next_engine` |

**Not created (Forge-3 owns):** `core/experts/__init__.py`. Python 3.3+ namespace-package semantics make this optional for import — verified via `python -c "from core.experts.comfyui_specialist import ComfyUISpecialist"`. If Forge-3 prefers an explicit init for re-exports, that's their call.

**Untouched (per role isolation):** `core/gpu_optimizer_v6.py`, `core/gpu_optimizer_v8.py`, `acceleration/*`, `config.yaml`, `core/engine_cache.py`, `core/background_compiler.py`. Verified via grep that no specialist imports another specialist or imports from `core/engine_cache.py` / `core/background_compiler.py`.

---

## Per-specialist contract (the Forge-3 input)

### 1. `core.experts.comfyui_specialist.ComfyUISpecialist`

| Field | Value |
|---|---|
| `ExpertType` claimed | `ExpertType.COMFYUI_SPECIALIST` (Forge-3 must add) |
| Fallback when enum missing | `ExpertType.GPU_SPECIALIST` (so Phase A imports don't crash) |
| `SystemContext.*` field required | `comfyui: Optional[ComfyUIState] = None` |
| State collector | `core.experts.comfyui_state_collector.ComfyUIStateCollector().collect()` |
| Constructor signature | `ComfyUISpecialist(config, cache_handle=None, quality_ledger_handle=None)` |
| `OptimizationAction` values produced | `INSTALL_CUSTOM_NODE`, `SET_LAUNCH_FLAG`, `COMPILE_TRT_ENGINE`, `EVICT_ENGINE_CACHE` |
| Phase B additional actions | `COMPILE_TRT_ENGINE` (with `precision="fp8"`) gated by `quality_ledger_handle.fp8_admitted(...)` and `config["phase_b_enabled"]` |
| Cache handle calls used | `engine_cache_lookup(key)`, `engine_cache_key(**kw)`, `engine_cache_disk_used_gb(app)` |
| Quality ledger handle calls used | `fp8_admitted(app, model_sha)` |
| Config keys consumed | `tensorrt.cache_size_gb` (default 50.0), `tensorrt.version` (default `"11.x"`), `phase_b_enabled` (default False), `min_confidence` |

### 2. `core.experts.houdini_specialist.HoudiniSpecialist`

| Field | Value |
|---|---|
| `ExpertType` claimed | `ExpertType.HOUDINI_SPECIALIST_V8` (Forge-3 must add — V7 `HOUDINI_ENGINEER` stays separate) |
| Fallback when enum missing | `ExpertType.HOUDINI_ENGINEER` (V7 value — non-conflicting because the V7 expert is unchanged) |
| `SystemContext.*` field required | `houdini_v8: Optional[HoudiniState] = None` |
| State collector | `core.experts.houdini_state_collector.HoudiniStateCollector(mcp_client=...).collect()` |
| Constructor signature | `HoudiniSpecialist(config, cache_handle=None, quality_ledger_handle=None)` |
| `OptimizationAction` values produced | `COMPILE_TRT_ENGINE`, `SET_ENV_VAR` (existing V7 action), and Phase B `ENABLE_BACKEND` |
| Phase B additional actions | `ENABLE_BACKEND` for `houdini:vellum:cutensor` gated on `config["phase_b_enabled"]` |
| Cache handle calls used | `engine_cache_lookup(key)`, `engine_cache_key(**kw)` |
| Quality ledger handle calls used | (reserved for Phase B; not consulted in Phase A) |
| Config keys consumed | `tensorrt.version`, `phase_b_enabled`, `min_confidence` |

**Note:** R2 (`houdini:karma:denoiser`) and R3 (`houdini:pdg`) emit using `OptimizationAction.SET_ENV_VAR` — already a V7 enum value. Brief Section B references these as `ENV_VAR`; I read this as a brief-level shorthand for the existing `SET_ENV_VAR` value, not a request for a new enum member. If Forge-3 instead wants a distinct `ENV_VAR` value, file a config-side mapping; the parameters dict already carries `name`, `value`, `variable`, `scope`, `restart_required` so dispatch is unambiguous.

### 3. `core.experts.ue5_specialist.UE5Specialist`

| Field | Value |
|---|---|
| `ExpertType` claimed | `ExpertType.UE5_SPECIALIST_V8` (Forge-3 must add) |
| Fallback when enum missing | `ExpertType.UE5_ENGINEER` |
| `SystemContext.*` field required | `ue5_v8: Optional[UE5State] = None` |
| State collector | `core.experts.ue5_state_collector.UE5StateCollector(project_search_roots=[...]).collect()` |
| Constructor signature | `UE5Specialist(config, cache_handle=None, quality_ledger_handle=None)` |
| `OptimizationAction` values produced | `INSTALL_PLUGIN`, `ADJUST_QUALITY` (existing V7 action) |
| Phase B additional actions | none implemented; brief R4 deferred |
| Cache handle calls used | (none in Phase A) |
| Quality ledger handle calls used | (none in Phase A) |
| Config keys consumed | `ue5.nim_marketplace_url` (optional, has default), `min_confidence` |

### 4. `core.experts.cross_app_coordinator.CrossAppCoordinator`

| Field | Value |
|---|---|
| `ExpertType` claimed | `ExpertType.CROSS_APP_COORDINATOR` (Forge-3 must add) |
| Fallback when enum missing | `ExpertType.SYSTEM_INTEGRATOR` |
| `SystemContext.*` field required | `cross_app: Optional[CrossAppState] = None` |
| State collector | none in this module — Forge-3 builds `CrossAppState` from per-app collector outputs + foreground-window detection. Helpers `vram_budget(active)` and `predict_next_engine(history)` are exported here for reuse. |
| Constructor signature | `CrossAppCoordinator(config, cache_handle=None, quality_ledger_handle=None)` |
| `OptimizationAction` values produced | `ENABLE_BACKEND`, `VRAM_ALLOCATION` (existing V7), `COMPILE_TRT_ENGINE` |
| Cache handle calls used | `engine_cache_lookup(key)` only |
| Quality ledger handle calls used | (none) |
| Config keys consumed | `cross_app.idle_window_seconds` (default 60.0), `min_confidence` |

---

## Aggregate enum delta for Forge-3

### `OptimizationAction` additions (any not-already-present V8 values referenced by Forge-2)
- `COMPILE_TRT_ENGINE` (used by ComfyUI R3+R4, Houdini R1, CrossApp R3)
- `INSTALL_CUSTOM_NODE` (ComfyUI R1, R2)
- `INSTALL_PLUGIN` (UE5 R1, R2)
- `ENABLE_BACKEND` (CrossApp R1; Houdini Phase B R4)
- `SET_LAUNCH_FLAG` (ComfyUI R1)
- `EVICT_ENGINE_CACHE` (ComfyUI R5)

These are exactly the V8 additions enumerated in `SHARED_CONTEXT.md` (minus `WIRE_BACKEND_PATCH`, which Forge-2 does not currently emit — Forge-3/4 may use it elsewhere).

### `ExpertType` additions (must be added by Forge-3)
- `COMFYUI_SPECIALIST`
- `HOUDINI_SPECIALIST_V8`
- `UE5_SPECIALIST_V8`
- `CROSS_APP_COORDINATOR`

### `SystemContext` field additions (must be added by Forge-3, all `Optional[...] = None`)
- `comfyui: Optional[ComfyUIState] = None`  (import: `from core.experts.comfyui_state_collector import ComfyUIState`)
- `houdini_v8: Optional[HoudiniState] = None`  (`from core.experts.houdini_state_collector import HoudiniState`)
- `ue5_v8: Optional[UE5State] = None`  (`from core.experts.ue5_state_collector import UE5State`)
- `cross_app: Optional[CrossAppState] = None`  (`from core.experts.cross_app_coordinator import CrossAppState`)

---

## Permissive-fallback behavior (operational note)

System brief said: "If not injected (None), fall back to permissive behavior (treat lookups as 'not cached' so recommendations still emit)." Implemented as follows:

- `cache_handle=None` -> `_cache_lookup(key)` returns `False`, so R3/R4 always emit a `COMPILE_TRT_ENGINE` recommendation, and `_cache_disk_used_gb("comfyui")` returns `0.0`, so R5 (cache eviction) does not fire spuriously.
- `quality_ledger_handle=None` -> `_fp8_admitted(...)` returns `False`, so Phase B FP8 promotion stays gated (correct conservative default).
- Any exception inside an injected handle is caught, logged at DEBUG, and treated as the permissive default. Specialists never raise out of `analyze()`.

This means Forge-3 can wire specialists into the orchestrator BEFORE Forge-4's cache/ledger backends land, and Phase A behavior degrades safely.

---

## State-collector entry points (Forge-3 wires these into `gather_context()`)

```python
# comfyui
from core.experts.comfyui_state_collector import ComfyUIStateCollector
ctx.comfyui = ComfyUIStateCollector().collect()

# houdini
from core.experts.houdini_state_collector import HoudiniStateCollector
ctx.houdini_v8 = HoudiniStateCollector(mcp_client=mcp).collect()

# ue5
from core.experts.ue5_state_collector import UE5StateCollector
ctx.ue5_v8 = UE5StateCollector(project_search_roots=[Path(...)]).collect()

# cross_app -- Forge-3 builds this directly from per-app state + a foreground
# detector (Forge-3 owns process-priority logic per V7 SystemIntegrator)
from core.experts.cross_app_coordinator import CrossAppState, UsageHistoryEntry
```

Each collector's `.collect()` is exception-safe (catches and logs at DEBUG).

---

## Workload-histogram method placement (deviation from brief literal)

Brief Section A R3 reads `hot = ctx.comfyui_workload_histogram_hot(min_count=3, lookback_days=14)` -- as a method on `ctx` directly. Modifying `SystemContext` is Forge-3's territory, so I attached the method to `ComfyUIState` instead and call it as `state.workload_histogram_hot(min_count=3, lookback_days=14)` inside the specialist. This stays inside Forge-2's authority and produces the same outcome. No action required from Forge-3 unless they want a top-level proxy on `SystemContext`.

---

## Done-criteria self-check

- [x] All 4 specialists produce non-empty `Recommendation` lists when their app is detected and conditions are met (smoke-tested via `python -c` driver).
- [x] No imports between specialists (`grep -r "from core.experts.[a-z]" core/experts/` returns no matches).
- [x] No imports from `core/engine_cache.py` or `core/background_compiler.py` (grep clean).
- [x] All four `python -c "from core.experts.<X> import <Class>"` succeed under `C:\Python314\python.exe`.
- [x] Decision rules in brief §A/B/C/D implemented exactly (R1..R5 ComfyUI, R1..R4 Houdini incl. Phase B guard, R1..R3 UE5, R1..R3 CrossApp).
- [x] No new `OptimizationAction` enum values added (delta listed above for Forge-3).
- [x] `config.yaml` not modified.
- [x] No file outside `core/experts/` modified.

---

## Open questions for Forge-3

- **`SystemContext.timestamp`/`workflow_mode`/etc. retention:** Forge-3 may want `comfyui_v8` instead of `comfyui` for symmetry with `houdini_v8`/`ue5_v8`. I named the field `comfyui` (matching SHARED_CONTEXT.md verbatim) but the semantic is V8-only -- happy to rename if it's cleaner; the rename is a single edit per file.
- **`ENV_VAR` vs `SET_ENV_VAR` shorthand in brief §B:** As noted above, Houdini R2/R3 currently emit `SET_ENV_VAR`. If a distinct V8 `ENV_VAR` member is wanted, file a brief amendment.

If anything in this handoff is wrong or missing, file a blocker rather than freelancing across the role boundary.
