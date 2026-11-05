# Forge-2 — Specialists (ComfyUI / Houdini / UE5 / CrossApp)

**Read first:** `briefs/SHARED_CONTEXT.md` and `briefs/_handoff_forge1.md` (the latter is Forge-1's deliverable; if absent, raise blocker).

## Authority boundary (Commandment 5)
- **You implement:** the 4 specialist classes under `core/experts/` plus their `SystemContext` state collectors.
- **You do NOT touch:** `acceleration/*` (Forge-1), `core/engine_cache.py` (Forge-4), `core/background_compiler.py` (Forge-4), `core/gpu_optimizer_v8.py` (Forge-3), `config.yaml` (Forge-3 lands the schema; you may *propose* keys via blocker).
- **Forge-2 produces `Recommendation`s only.** If you need a new `OptimizationAction` value that isn't in the SHARED_CONTEXT additions list, raise a blocker — do NOT add to the enum yourself.
- **No freelancing.** The decision rules in §A/B/C/D below are normative. If you think a rule should change, document it in `briefs/_blockers_forge2.md` and proceed without the change.

## Implementation order (mandatory — 1 then 2 then 3 then 4)
1. ComfyUISpecialist (most prior art — easiest baseline).
2. HoudiniSpecialist.
3. UE5Specialist.
4. CrossAppCoordinator (needs the others to exist).

After each one: import-check (`python -c "from core.experts.X import X"`), then move to the next.

---

## A. `core/experts/comfyui_specialist.py`

### State collector (`comfyui_state_collector.py`, same module dir)
- Polls `http://127.0.0.1:8188/system_stats`, `/queue`, `/history?max_items=20`.
- Subscribes to `/ws` for completion events; on completion, appends `(workflow_hash, model_sha, shape, batch, sampler, steps, wall_seconds)` to an in-memory ring buffer (Forge-4's telemetry persists it).
- Reads ComfyUI's `extra_model_paths.yaml` if present to find model dir.
- Detects custom nodes by listing `<install_dir>/custom_nodes/`.
- Parses launch flags from the ComfyUI process cmdline via `psutil`.
- All HTTP calls have 1-second timeouts; failures degrade `ComfyUIState.api_reachable=False` rather than raise.

### Decision rules (normative — implement exactly these, no additions)

```
state = ctx.comfyui
if not state or not state.process_running:
    return []

recs = []

# R1: SageAttention 2 install or enable
if "ComfyUI-SageAttention" not in state.custom_nodes_installed:
    recs.append(Rec(INSTALL_CUSTOM_NODE, target="comfyui:sage_attention",
                    parameters={"node_repo": "thu-ml/SageAttention", "ref": "v2.0"},
                    confidence=0.92, expected_impact=50.0))
elif "--use-sage-attention" not in state.launch_flags:
    recs.append(Rec(SET_LAUNCH_FLAG, target="comfyui:launch",
                    parameters={"add_flag": "--use-sage-attention", "restart_required": True},
                    confidence=0.92, expected_impact=50.0))

# R2: ComfyUI-TensorRT custom node install
if "ComfyUI_TensorRT" not in state.custom_nodes_installed:
    recs.append(Rec(INSTALL_CUSTOM_NODE, target="comfyui:tensorrt",
                    parameters={"node_repo": "comfyanonymous/ComfyUI_TensorRT", "ref": "main"},
                    confidence=0.88, expected_impact=85.0))

# R3: Compile TRT engine for hot (model, shape) — speculative, idle-gated
hot = ctx.comfyui_workload_histogram_hot(min_count=3, lookback_days=14)
for (model_sha, shape) in hot:
    key = engine_cache_key(app="comfyui", model=model_sha, shape=shape, precision="fp16",
                           backend="tensorrt", backend_ver=trt_version)
    if not engine_cache_lookup(key):  # exposed via orchestrator hook
        recs.append(Rec(COMPILE_TRT_ENGINE, target=f"comfyui:{model_sha}:{shape}",
                        parameters={"precision": "fp16", "shape_range": derive_range(shape),
                                    "model_kind": "unet"},
                        confidence=0.85, expected_impact=70.0))

# R4 (Phase B): FP8 promotion gated on quality ledger — STUB GUARD in Phase A
if PHASE_B and quality_ledger.fp8_admitted("comfyui", model_sha):
    ...

# R5: cache eviction
if engine_cache_disk_used_gb("comfyui") > config.tensorrt.cache_size_gb * 0.90:
    recs.append(Rec(EVICT_ENGINE_CACHE, target="comfyui:cache",
                    parameters={"app": "comfyui"}, confidence=0.95, expected_impact=0))

return self.filter_by_confidence(recs)
```

The `engine_cache_lookup` / `engine_cache_disk_used_gb` are Forge-4 deliverables; you call them through the orchestrator-injected handle (Forge-3 wires it).

---

## B. `core/experts/houdini_specialist.py`

### State collector (`houdini_state_collector.py`)
- Detects `houdini.exe` / `houdinifx.exe` / `houdinicore.exe` running.
- Reads `$HOUDINI_USER_PREF_DIR/houdini21.0/PYTHONPATH` to know what's loaded.
- Detects active ML SOPs via Houdini's HTTP-based houdini-context MCP if present (V7 already wires `mcp_integration: true`).
- Captures Karma XPU active flag from environment + scene if obtainable.

### Decision rules

```
state = ctx.houdini_v8
if not state or not state.process_running:
    return []
recs = []

# R1: ML SOP ONNX detection → recommend TRT conversion
for onnx_ref in state.ml_sop_onnx_paths:
    onnx_sha = sha256_file(onnx_ref)
    key = engine_cache_key(app="houdini", model=onnx_sha, shape=onnx_input_shape(onnx_ref),
                           precision="fp16", backend="tensorrt", backend_ver=trt_version)
    if not engine_cache_lookup(key):
        recs.append(Rec(COMPILE_TRT_ENGINE, target=f"houdini:ml_sop:{onnx_sha[:8]}",
                        parameters={"precision": "fp16",
                                    "shape_range": fixed_shape_range(onnx_input_shape(onnx_ref)),
                                    "model_kind": "ml_sop", "source_onnx": str(onnx_ref)},
                        confidence=0.82, expected_impact=55.0))

# R2: OptiX denoiser config (Karma XPU)
if state.karma_xpu_active and not state.optix_denoiser_enabled:
    recs.append(Rec(ENV_VAR, target="houdini:karma:denoiser",
                    parameters={"name": "HOUDINI_OPTIX_DENOISER", "value": "1"},
                    confidence=0.88, expected_impact=18.0))

# R3: TOPs/PDG GPU scheduler
if state.tops_active and state.pdg_gpu_scheduling_enabled is False:
    recs.append(Rec(ENV_VAR, target="houdini:pdg",
                    parameters={"name": "HOUDINI_PDG_GPU_SCHEDULING", "value": "1"},
                    confidence=0.85, expected_impact=22.0))

# R4 (Phase B): cuTensor Vellum solver — defer to Phase B; emit only when PHASE_B flag set

return self.filter_by_confidence(recs)
```

---

## C. `core/experts/ue5_specialist.py`

### State collector (`ue5_state_collector.py`)
- Detects `UnrealEditor.exe` / shipped game exes.
- Reads project's `Plugins/` directory to detect NVIDIA NIM Plugin, DLSS plugin versions.
- Reads `UserEngine.ini` (path already in V7 config) to know active CVars.
- Checks DLSS DLL versions in `Engine/Plugins/Runtime/Nvidia/DLSS/Binaries/`.

### Decision rules

```
state = ctx.ue5_v8
if not state or not state.process_running:
    return []
recs = []

# R1: NVIDIA NIM Plugin install advisory
if not state.nim_plugin_installed:
    recs.append(Rec(INSTALL_PLUGIN, target="ue5:nim",
                    parameters={"plugin": "NVIDIA_NIM",
                                "marketplace_url": "<NIM Marketplace URL — read from registry>",
                                "auto_install": False,  # advisory only; UE5 plugin installs are user-driven
                                "notification": "Install NVIDIA NIM Plugin to enable TRT-LLM in UE5."},
                    confidence=0.78, expected_impact=30.0))

# R2: DLSS 4 / DLSS-RR version verification
if state.dlss_dll_version and state.dlss_dll_version < "4.0":
    recs.append(Rec(INSTALL_PLUGIN, target="ue5:dlss",
                    parameters={"plugin": "DLSS_4", "current": state.dlss_dll_version,
                                "minimum": "4.0", "notification": True},
                    confidence=0.85, expected_impact=15.0))

# R3: Path Tracer GPU denoiser
if state.path_tracer_in_use and state.cvars.get("r.PathTracing.Denoiser") != "1":
    recs.append(Rec(ADJUST_QUALITY, target="ue5:path_tracer_denoiser",
                    parameters={"engine_ini_path": state.engine_ini,
                                "engine_ini_setting": "r.PathTracing.Denoiser=1"},
                    confidence=0.90, expected_impact=12.0))

# R4 (Phase B): NIM TRT-LLM model integration patterns

return self.filter_by_confidence(recs)
```

---

## D. `core/experts/cross_app_coordinator.py`

### Authority
Owns inter-app arbitration. Does NOT itself read TRT or backend libraries; reads only `SystemContext`.

### Decision rules

```
state = ctx.cross_app
recs = []

# R1: foreground process priority
fg = state.foreground_pid_app  # one of {comfyui, houdini, ue5, none}
for bg_app in ("comfyui", "houdini", "ue5"):
    if bg_app != fg and state.compile_jobs_active.get(bg_app, 0) > 0:
        recs.append(Rec(ENABLE_BACKEND, target=f"compiler:pause:{bg_app}",
                        parameters={"action": "pause_background_compiles", "app": bg_app},
                        confidence=0.95, expected_impact=0))

# R2: VRAM budget per active-app set (mirrors V7 WorkflowMode logic but app-keyed)
active = [a for a in ("comfyui", "houdini", "ue5") if state.app_running.get(a)]
budget = vram_budget(active)   # helper: 1 app→18GB; 2 apps→11/11/2; 3 apps→11/8/3 + 2GB sys
for app, gb in budget.items():
    recs.append(Rec(VRAM_ALLOCATION, target=f"vram:{app}",
                    parameters={"limit_gb": gb}, confidence=0.90, expected_impact=0))

# R3: idle-window prefetch — if all apps idle for 60s, recommend prefetching next-most-likely engine
if state.idle_dwell_seconds >= 60:
    candidate = predict_next_engine(state.usage_history)  # simple LRU/freq blend
    if candidate and not engine_cache_lookup(candidate.key):
        recs.append(Rec(COMPILE_TRT_ENGINE, target=f"prefetch:{candidate.key[:8]}",
                        parameters=candidate.compile_params,
                        confidence=0.65, expected_impact=20.0))

return self.filter_by_confidence(recs)
```

## Bounded failure
- 3 consecutive HTTP polling failures → mark `api_reachable=False`, do not retry inside this cycle.
- ML SOP ONNX path doesn't exist on disk after detection → log + skip; no blocker.
- If a specialist's import fails 3× → blocker: "{specialist} cannot import; missing handoff or schema mismatch."

## Done criteria
- All 4 specialists produce non-empty `Recommendation` lists when their app is detected and conditions are met (Crucible-1 verifies via fixture).
- No imports from each other (`grep`'d).
- No imports from `core/engine_cache.py` or `core/background_compiler.py` directly — only via injected handles received in `__init__`.

## Handoff
Write `briefs/_handoff_forge2.md` listing: each specialist's class path, the `ExpertType` value it claims, the new `OptimizationAction` values it produces, and the `SystemContext.*` field it requires. This is Forge-3's input.
