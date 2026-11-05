# Forge-3 — Orchestrator V8 wiring + config + SystemContext extensions

**Read first:** `briefs/SHARED_CONTEXT.md`, `briefs/_handoff_forge1.md`, `briefs/_handoff_forge2.md`, `briefs/_handoff_forge4.md`.

## Authority boundary (Commandment 5)
- **You implement:** `core/gpu_optimizer_v8.py` (sibling to V6, V7-style class layout), `config.yaml` extensions, `core/experts/__init__.py`, the new `_apply_*` handlers for the new actions, ComfyUI/Houdini/UE5 detection inside `gather_context`.
- **You do NOT touch:** `core/gpu_optimizer_v6.py` (frozen rollback path), `acceleration/*` (Forge-1), `core/experts/<specialist>.py` body code (Forge-2), `core/engine_cache.py` / `core/background_compiler.py` body code (Forge-4).
- You **may** import from any of those modules — that is the point of this role.

## Deliverables

### D1. `core/gpu_optimizer_v8.py`
- Start by **copying** `core/gpu_optimizer_v6.py` to `core/gpu_optimizer_v8.py`. Do not modify v6 after the copy.
- Replace `OptimizerOrchestrator` class implementation per the diffs below. Keep the class name.

#### Required changes
1. Imports: add `from core.experts.comfyui_specialist import ComfyUISpecialist` (and analogous for Houdini, UE5, CrossApp), `from core.engine_cache import EngineCacheManager`, `from core.background_compiler import BackgroundCompiler`.
2. Add new `OptimizationAction` enum members from SHARED_CONTEXT.md (do not remove existing).
3. Add new `ExpertType` enum members from SHARED_CONTEXT.md (do not remove existing).
4. `_initialize_experts`: after the V7 5-expert init, add the 4 V8 specialists. The V7 `HoudiniEngineer` and `UE5Engineer` are **disabled** in V8 (set `enabled: false` in V8's config block) — they're replaced by the V8 specialists. NVIDIA Researcher, GPU Specialist, and System Integrator stay active.
5. Add to `__init__`: `self.engine_cache = EngineCacheManager(config.tensorrt.cache_dir)`, `self.background_compiler = BackgroundCompiler(self.engine_cache)`.
6. `gather_context`:
   - Keep V7 logic (so existing tests still pass).
   - Add ComfyUI detection: scan `psutil` processes for `python.exe` whose cmdline contains `main.py` AND a path matching ComfyUI install. Populate `ctx.comfyui` via Forge-2's `ComfyUIStateCollector`.
   - Add Houdini V8 state via `HoudiniStateCollector`.
   - Add UE5 V8 state via `UE5StateCollector`.
   - Add cross-app state via `CrossAppStateCollector`.
   - Each collector wrapped in try/except — failure must not break the context gather.

#### New `_apply_*` handlers (you write these)
- `_apply_compile_trt_engine(rec)` → calls `self.background_compiler.enqueue(rec.parameters)`. Returns `OptimizationResult(applied=True)` immediately (compile is async). Logs queue position.
- `_apply_install_custom_node(rec)` → ComfyUI custom_nodes/ git clone + pip install in venv. Auto-install confirmed by user (decision #4). On completion: emit user-visible Rich notification "Installed {node}; restart ComfyUI". Idempotent.
- `_apply_install_plugin(rec)` → UE5: write a notification only (`auto_install=False` in Forge-2 brief R1). Surface plugin URL + minimum version. Does NOT modify the UE5 install.
- `_apply_enable_backend(rec)` → toggle a config flag (writes to `config.yaml` if appropriate, or sets a runtime flag). Specific to the backend named in `rec.parameters`.
- `_apply_set_launch_flag(rec)` → writes a desired-flag annotation to `data/launch_advice.json`; emits Rich notification "Restart {app} with: {flag}". Does NOT kill the running process.
- `_apply_evict_engine_cache(rec)` → calls `self.engine_cache.evict_lru(target_app=rec.parameters['app'])`.
- `_apply_wire_backend_patch(rec)` → writes a small monkeypatch .py file into the target app's auto-import directory (e.g. ComfyUI `extras/optimizer_patches/`).
- **Replace** the existing stub `_apply_acceleration` (V6 line 1378) — it should now dispatch to the appropriate handler above based on `rec.parameters["sub_action"]`, OR mark deprecated and route via the explicit new actions.

### D2. `config.yaml` extensions
Append (do not modify existing keys other than `acceleration.cutlass.version`):

```yaml
comfyui:
  enabled: true
  api_host: "127.0.0.1"
  api_port: 8188
  install_dir: "C:/Users/User/ComfyUI"
  watch_models_dir: "C:/Users/User/ComfyUI/models"
  websocket_subscribe: true
  expert:
    enabled: true
    min_confidence: 0.80
    speculative_compile:
      min_usage_count: 3
      lookback_days: 14
      only_when_gpu_idle_pct: 5
      idle_dwell_seconds: 30
      max_concurrent_compiles: 1

houdini_v8:
  enabled: true
  expert:
    enabled: true
    min_confidence: 0.80

ue5_v8:
  enabled: true
  expert:
    enabled: true
    min_confidence: 0.80

cross_app_coordinator:
  enabled: true
  min_confidence: 0.85
  vram_split:
    one_app_active: { primary: 18 }
    two_apps_active: { comfyui: 11, houdini: 11, ue5: 11 }   # take any 2
    three_apps_active: { comfyui: 3, houdini: 11, ue5: 8 }   # 22GB + 2GB system

tensorrt:
  enabled: true
  version_pin: "11.x"
  precision_priority: ["fp16"]   # fp8 added in Phase B
  cache_dir: "D:/OptimizerV8_EngineCache"
  cache_size_gb: 200
  invalidate_on_driver_change: true

# Disable V7 specialists in V8 mode (V7 still runs them when called directly)
experts:
  ue5_engineer:
    enabled: false        # superseded by ue5_v8 in V8 orchestrator
  houdini_engineer:
    enabled: false        # superseded by houdini_v8 in V8 orchestrator
```

Update `acceleration.cutlass.version: "4.x"` (was 3.5.1) per Amendment 1.

### D3. `core/experts/__init__.py`
- Standard `__init__.py` with `__all__` listing the 4 specialists.

### D4. `install_and_run.bat` updates (small)
- Branch on optional argument `--v8` to launch `core/gpu_optimizer_v8.py` instead of v6.
- Set `D:\OptimizerV8_EngineCache` env var; create directory if missing.

## Sequence
1. **Scout.** Read `core/gpu_optimizer_v6.py` end-to-end (it's the parent). Read both Forge-1 and Forge-4 handoffs. Read each Forge-2 specialist file.
2. Copy v6 → v8.
3. Add enums (verify import by running `python -c "from core.gpu_optimizer_v8 import OptimizationAction; assert OptimizationAction.COMPILE_TRT_ENGINE"`).
4. Wire `gather_context` collectors. Verify with `python core/gpu_optimizer_v8.py --once --config config.yaml` (single-cycle smoke).
5. Implement `_apply_*` handlers one at a time. After each: re-run single-cycle smoke.
6. Wire `_initialize_experts` to register the 4 new specialists.
7. Update `config.yaml`. Re-run smoke.
8. `install_and_run.bat` --v8 branch last.

## Bounded failure
- v6 imports failing in v8 after copy → blocker: "Forge-3 cannot reproduce v6 baseline; rollback path compromised."
- A specialist crashes during gather → catch and log; do not let it break gather. If it crashes 3 cycles in a row → disable that specialist for the session and emit blocker file.

## Done criteria (Crucible-1 verifies)
- `python core/gpu_optimizer_v8.py --once --config config.yaml` runs to completion with exit 0 even when no apps are running.
- With ComfyUI running locally, the smoke run produces ≥ 1 ComfyUI recommendation in logs.
- V7 path (`python core/gpu_optimizer_v6.py --once`) still works unchanged. Verified by running both.
- Cache directory created on first V8 run.

## Handoff
Write `briefs/_handoff_forge3.md` documenting: command lines (`--v8`, `--config`, `--once`, `--interval`), config keys touched, and the `gather_context` collector contract for any future Forge.
