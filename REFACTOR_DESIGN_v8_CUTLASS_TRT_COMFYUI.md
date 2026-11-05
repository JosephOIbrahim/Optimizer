# Optimizer V8 — Architect's Design Document
## CUTLASS 4.x + TensorRT 11.x + ComfyUI-First Refactor

| Field | Value |
|---|---|
| **Phase** | ARCHITECT — design only. Per Commandment 5/8 no code mutations until human gate clears. |
| **Successor roles** | Forge-1, Forge-2, Forge-3, Forge-4 (implementers, role-isolated). Crucible-1, Crucible-2 (adversarial verifiers). |
| **Target HW** | RTX 4090 sm_89 (24GB GDDR6X, 1.01 TB/s, 4th-gen tensor cores), Threadripper PRO 7965WX (24c/48t), 128GB DDR5, PCIe Gen4 x16, ReBAR on, Studio 596.36. |
| **Primary workload** | ComfyUI (SDXL / Flux / Cosmos / Wan / video-DiT). |
| **Preserved workloads** | Houdini 21, UE5.6+ — must not regress. |
| **Versioning** | New `core/gpu_optimizer_v8.py` sibling. V7 left intact for rollback. |

---

## 1. Scout Findings — Boundaries (Commandment 1)

### 1.1 Frozen surface (do NOT change)
| Symbol | Location | Reason frozen |
|---|---|---|
| `Expert(ABC).analyze(ctx) -> List[Recommendation]` | `core/gpu_optimizer_v6.py:170` | Used by `OptimizerOrchestrator.analyze_workload()` parallel dispatch. |
| `Recommendation` dataclass field shape | `core/gpu_optimizer_v6.py:127` | Touched in `aggregate_recommendations`, `apply_optimization`, telemetry. |
| `OptimizationAction` enum values | `core/gpu_optimizer_v6.py:78` | Existing values referenced by 5 experts; only **add**, never rename or remove. |
| `gather_context()` graceful-degrade contract | `core/gpu_optimizer_v6.py:1105-1119` | Exception path must still return a usable `SystemContext`. |

### 1.2 Touchable surface (may extend)
- `SystemContext` — additive `Optional` fields with defaults are safe.
- `ExpertType` enum — add new members.
- `OptimizationAction` enum — add new members.
- `OptimizerOrchestrator._initialize_experts` — register additional experts.
- `OptimizerOrchestrator._apply_*` dispatch — add new handlers; **replace** the current stub at `_apply_acceleration` (line 1378) which only logs a warning.
- `config.yaml` — add new top-level keys (`comfyui:`, `tensorrt:`); existing `acceleration.cutlass` / `acceleration.cutensor` may be extended.
- `acceleration/cutlass_integration.py` — fully built but **never imported by the orchestrator today**. We import it.

### 1.3 Empty / placeholder (greenfield, no convention to preserve)
- `integrations/` — empty.
- `monitoring/` — empty.
- `benchmarks/` — empty.
- `acceleration/cutensor_vellum_solver.py` — placeholder.
- `dashboard/app.py` — placeholder.

---

## 2. First-Principles Restatement of the Problem

The current architecture is a **stateless config tuner**: read system state → emit env-var / registry / process-priority recommendations → apply immediately, with no build-time, no artifacts, no async lifecycle. This shape is correct for "set `CUDA_CACHE_MAXSIZE=4GB`" but **structurally cannot deliver CUTLASS or TensorRT for diffusion workloads** because those require:

| Property | Current arch | What CUTLASS/TRT need |
|---|---|---|
| Output type | Side-effect (env mutation) | Build artifact (compiled engine, .plan or .so) |
| Cost | Microseconds | 5–45 minutes per artifact |
| Cache key | None | (model_hash, shape_range, precision, driver_ver, backend_ver, hw_arch) |
| Lifecycle | Apply once | Compile → cache → invalidate on model/driver swap |
| Failure mode | Skip the env var | Fall back to PyTorch path; never block the user |
| Concurrency | Synchronous | Background, gated by GPU-idle signal |

**Conclusion:** The refactor is not "add CUTLASS/TRT recommendations to existing experts." It is **introducing a build-and-cache pipeline as a first-class orchestrator capability**, alongside the existing config-tuner pipeline. Both pipelines are kept; the workload detector routes between them.

---

## 3. Architecture — V8

### 3.1 New top-level capabilities
```
OptimizerOrchestrator (V8)
├── ConfigPipeline (existing, V7 logic preserved)
│   ├── 5 existing experts → Recommendations(ENV_VAR, REGISTRY, …)
│   └── _apply_* handlers (existing)
└── BuildPipeline (NEW)
    ├── ComfyUISpecialist expert → Recommendations(COMPILE_TRT, INSTALL_NODE, …)
    ├── EngineCacheManager (lookup/store/evict, keyed hash)
    ├── BackgroundCompiler (queued, GPU-idle gated)
    └── _apply_build_* handlers (NEW)
```

The orchestrator's main loop is unchanged in shape: gather → dispatch → aggregate → apply. The change is that "apply" now routes to either config or build handlers based on `OptimizationAction`.

### 3.2 New abstractions

#### 3.2.1 `AccelerationBackend` (new ABC, in `acceleration/backend.py`)
The plug point for every speedup mechanism. Each backend declares:
```
applies_to(model_kind, shape) -> bool
estimated_compile_seconds() -> float
estimated_speedup() -> float          # vs torch baseline, measured-or-prior
estimated_quality_delta() -> float    # 0.0=lossless, 1.0=visibly degraded
compile(model, shape, precision) -> BuildArtifact
load(artifact) -> Callable
```
Concrete subclasses: `TensorRTBackend`, `SageAttentionV2Backend`, `NunchakuSVDQuantBackend`, `CutlassFP8GemmBackend`, `TorchCompileFP8Backend`, `XformersBackend` (baseline-fallback).

#### 3.2.2 `BuildArtifact` (dataclass)
```
hash_key: str               # blake2b of (model_sha, shape, precision, backend_ver, driver_ver, sm_arch)
artifact_path: Path         # e.g. D:\ComfyUI_TRT_Cache\<key>.plan
backend: str
model_ref: str              # source model hash + filename
shape_range: tuple          # (min, opt, max) for TRT dynamic shapes
precision: str
compile_seconds_actual: float
runtime_seconds_p50: float | None    # filled by telemetry after first uses
disk_bytes: int
created_utc: datetime
last_used_utc: datetime
```

#### 3.2.3 `EngineCacheManager` (in `core/engine_cache.py`)
- Backed by SQLite at `data/engine_cache.db` + artifact directory on the 3.6TB SSD.
- `lookup(hash_key)` / `store(artifact)` / `evict_lru(target_bytes)` / `invalidate_on_driver_change()`.
- Eviction policy: LRU constrained by configurable disk budget (default 200 GB, < 6% of 3.6TB SSD).
- Invalidation triggers: (a) driver version change, (b) ComfyUI version change, (c) model file mtime change, (d) backend version change.

#### 3.2.4 `BackgroundCompiler` (in `core/background_compiler.py`)
- Owns a single async queue and a single dedicated CUDA context.
- Polls `gather_context()` for `gpu_utilization_percent`; blocks compile until `< idle_threshold` (default 5%) for `idle_dwell_seconds` (default 30).
- Hard preemption: if interactive workload starts (ComfyUI queue depth > 0 or Houdini/UE5 process appears), pause current compile (TensorRT supports `IBuilderConfig.set_progress_monitor` for interruption hooks; CUTLASS NVCC compiles get SIGTERM).
- Concurrency: 1 compile at a time (TRT builder is VRAM-heavy).
- Persists queue across restarts.

### 3.3 New Expert: `ComfyUISpecialist` (in `core/experts/comfyui_specialist.py`)
| Attribute | Value |
|---|---|
| Confidence threshold | 0.80 |
| ExpertType | `COMFYUI_SPECIALIST` (new enum value) |
| Domain knowledge | ComfyUI process detection, graph parsing, model registry, sampler/scheduler vocabulary, VRAM tier flags (`--lowvram` / `--highvram` / `--gpu-only`), custom node ecosystem (TensorRT, Nunchaku, SageAttn). |
| Inputs | Extended `SystemContext.comfyui` (new field, see §3.4). |
| Outputs | `Recommendation(action ∈ {COMPILE_TRT_ENGINE, INSTALL_CUSTOM_NODE, ENABLE_BACKEND, SET_LAUNCH_FLAG, EVICT_ENGINE_CACHE, ENV_VAR})` |
| Decision rules | See §4. |

### 3.4 `SystemContext` extensions (additive, all `Optional` with `None` default)
```python
comfyui: Optional[ComfyUIState] = None

@dataclass
class ComfyUIState:
    process_running: bool
    pid: Optional[int]
    api_reachable: bool                  # http://127.0.0.1:8188/system_stats
    version: Optional[str]               # parsed from /system_stats
    queue_depth: int                     # from /queue
    queue_running: List[GraphSummary]    # from /queue
    history_recent: List[GraphSummary]   # from /history?max_items=20
    model_dir: Optional[Path]
    custom_nodes_installed: Set[str]     # detected from custom_nodes/
    launch_flags: List[str]              # parsed from cmdline
    last_completion_seconds: Optional[float]   # from websocket /ws

@dataclass
class GraphSummary:
    workflow_hash: str
    primary_model: Optional[str]         # SDXL / Flux / Cosmos / Wan / etc.
    primary_model_sha256: Optional[str]
    resolution_w: int
    resolution_h: int
    batch_size: int
    sampler: str                         # e.g. "dpmpp_2m_karras"
    steps: int
    backends_in_use: Set[str]            # {"sage_attn", "trt_unet", ...}
```

### 3.5 New `OptimizationAction` values
```
COMPILE_TRT_ENGINE       # async build, gated on idle
INSTALL_CUSTOM_NODE      # subprocess: git clone + pip install into ComfyUI/custom_nodes
ENABLE_BACKEND           # toggle a backend in ComfyUI runtime config
SET_LAUNCH_FLAG          # advise user to restart ComfyUI with new flags (config write only)
EVICT_ENGINE_CACHE       # disk reclaim
WIRE_BACKEND_PATCH       # write a small monkeypatch into ComfyUI/extras/optimizer_patches/ that ComfyUI auto-imports
```

### 3.6 Backend matrix — what each lib actually does

| Backend | Covers | Speedup* | Compile cost | Quality risk | Phase |
|---|---|---|---|---|---|
| **SageAttention 2** (drop-in for SDPA) | All attention layers | 1.4–1.8× e2e | None (just a launch flag + custom node) | Low | A |
| **ComfyUI-TensorRT (UNet/DiT)** | UNet/DiT forward | 1.7–2.5× e2e | 8–25 min/(model, shape) | Low (FP16) / Medium (FP8) | A |
| **ComfyUI-TensorRT (VAE)** | VAE decode | 1.3–1.6× | 2–5 min | Low | A |
| **TensorRT-LLM diffusion plugins** | Flux, MMDiT-class | 1.8–2.3× | 15–40 min | Low | B |
| **Nunchaku SVDQuant W4A4** | Whole UNet/DiT | 3–5× | 20–60 min (model conversion offline) | Medium-high; per-model quality eval required | B |
| **CUTLASS 4.x FP8 GroupedGemm** | Custom MLP fusions, multi-LoRA batched | 1.5–2.2× on covered ops | NVCC compile per-kernel, ≤ 60s | Low | C |
| **torch.compile + torchao FP8** | PyTorch fallback path | 1.2–1.6× | 30–90s warm-up | Low | A (baseline) |
| **xformers** | Attention fallback when SageAttn unavailable | 1.1–1.3× | None | Low | A (fallback) |

*Speedup numbers are priors from public benchmarks (NVIDIA/comfyui-tensorrt repo readme, Sage v2 paper, Nunchaku readme); real values come from Crucible-2's measured ledger and overwrite the priors per (model, shape).

### 3.7 Decision logic (ComfyUISpecialist)

Pseudocode for the `analyze` body. Forge-2 must implement this logic faithfully and **may not** add additional rules without architect re-approval (Commandment 5: no freelancing).

```
recs = []
state = ctx.comfyui
if not state or not state.process_running:
    return recs

# Rule 1: SageAttention 2 — install + enable if absent
if "ComfyUI-SageAttention" not in state.custom_nodes_installed:
    recs.append(INSTALL_CUSTOM_NODE @ confidence=0.92, impact=+50%)
elif "--use-sage-attention" not in state.launch_flags:
    recs.append(SET_LAUNCH_FLAG @ confidence=0.92, impact=+50%)

# Rule 2: ComfyUI-TensorRT — install if absent
if "ComfyUI_TensorRT" not in state.custom_nodes_installed:
    recs.append(INSTALL_CUSTOM_NODE @ confidence=0.88, impact=+85%)

# Rule 3: TRT engine compile — only if (model, shape) seen ≥ N times AND no cache hit
hot_shapes = WorkloadHistogram.hot(min_count=3, lookback_days=14)
for (model, shape) in hot_shapes:
    key = engine_cache.compute_key(model, shape, precision="fp16", backend="trt_v11")
    if not engine_cache.lookup(key):
        recs.append(COMPILE_TRT_ENGINE @ confidence=0.85, impact=+70%, gated_on_idle=True)

# Rule 4: Quality-gated FP8 promotion — only if FP16 engine's quality measured & FP8 quality measured ≥ 0.97 LPIPS
for (model, shape) in fp16_validated:
    if quality_ledger.fp8_score(model) >= 0.97:
        recs.append(COMPILE_TRT_ENGINE precision="fp8" @ confidence=0.78, impact=+30%)

# Rule 5: Cache eviction — if disk_used > budget * 0.90
if engine_cache.disk_used_gb > config.tensorrt.cache_size_gb * 0.90:
    recs.append(EVICT_ENGINE_CACHE @ confidence=0.95)

# Rule 6: Phase B+ — Nunchaku, CUTLASS custom kernels — out of scope for Phase A

return self.filter_by_confidence(recs)
```

### 3.8 NVIDIA Researcher expert — narrow extension
The existing researcher already scrapes `nvidia/cutlass` and `nvidia/TensorRT-LLM`. Forge-3 extends it to also subscribe to:
- `comfyanonymous/ComfyUI` releases (for breaking-change detection that invalidates engine cache).
- `comfyanonymous/ComfyUI_TensorRT` releases.
- `mit-han-lab/nunchaku` releases.
- `thu-ml/SageAttention` releases.

Output: bumps `backend_ver` constants which feed the engine cache key (forces clean recompile when needed — Commandment 2: regression-as-invariant).

### 3.9 Telemetry feedback loop
- A small ComfyUI-side patch (deployed via `WIRE_BACKEND_PATCH`) writes per-generation timing + VRAM peak + sampler config to `data/comfyui_telemetry.db`.
- Schema: `(timestamp, workflow_hash, model_sha, shape_w, shape_h, batch, precision, backend_set, wall_seconds, samples_per_sec, vram_peak_gb)`.
- Genetic tuner reads this ledger and evolves a `backend_choice` gene per (model, shape) bucket — this replaces the current FPS-based fitness with measured `samples_per_sec`.

### 3.10 Quality gates (Crucible-2 territory)
A backend that produces wrong outputs is worse than a slow one. Before any backend's recommendation can graduate from "experimental" to "default-on", Crucible-2 must measure:
- LPIPS distance vs FP32 reference on a fixed 8-prompt, 4-seed eval set per model class.
- CLIP score delta.
- Pareto rule: a backend is admitted only if `speedup ≥ 1.3 AND lpips_delta ≤ 0.04 AND clip_delta ≤ -0.02`.
- Numbers stored in `data/quality_ledger.db`. ComfyUISpecialist reads this ledger before recommending FP8 / Nunchaku.

---

## 4. Configuration shape (preview — Forge-3 lands the YAML diff)

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

tensorrt:
  enabled: true
  version_pin: "11.x"
  precision_priority: ["fp8", "fp16"]
  fp8_requires_quality_gate: true
  cache_dir: "D:/ComfyUI_TRT_Cache"        # 3.6TB SSD, off OneDrive
  cache_size_gb: 200
  invalidate_on_driver_change: true

acceleration:
  cutlass:
    version: "4.x"                          # was 3.5.1; bump pinned to commit, not branch
    install_path: "${CUTLASS_PATH}"
    fp8_groupedgemm_kernels:
      enabled: false                        # Phase C
  sage_attention:
    enabled: true
    custom_node_ref: "thu-ml/SageAttention@v2.0"
  nunchaku:
    enabled: false                          # Phase B; opt-in
```

---

## 5. Role decomposition (Commandments 5 & 6)

Each Forge agent receives **only its own section** of this document as its brief. Forge agents do not see one another's scopes — this enforces role isolation.

### Forge-1 — Acceleration backends (no orchestrator changes)
- Files: `acceleration/backend.py` (new ABC), `acceleration/tensorrt_engine.py` (new), `acceleration/sage_attention.py` (new), extend `acceleration/cutlass_integration.py` with `CutlassFP8GemmBackend` class only (do not modify existing `CUTLASSManager` API).
- Out of scope: orchestrator wiring, expert classes, `SystemContext` changes, config schema changes.
- Handoff to Forge-3: list of new public symbols + their signatures.

### Forge-2 — ComfyUI specialist expert
- Files: `core/experts/__init__.py` (new), `core/experts/comfyui_specialist.py` (new), `core/experts/comfyui_state_collector.py` (new — owns the HTTP/WebSocket polling).
- Inputs: `acceleration.backend.AccelerationBackend` ABC (read-only — Forge-1's deliverable).
- Outputs: `Recommendation` lists only. **Forge-2 does NOT add new `_apply_*` handlers** — the orchestrator wiring is Forge-3's job. If Forge-2 needs an action that doesn't exist, they raise it as an explicit blocker (Commandment 4).
- Out of scope: `OptimizationAction` enum changes, orchestrator dispatch, config YAML.

### Forge-3 — Orchestrator wiring (V8 sibling)
- Files: `core/gpu_optimizer_v8.py` (new — copy of v6 + diffs), `config.yaml` (extend), `core/__init__.py` (route).
- Adds: new `OptimizationAction` members, new `_apply_*` handlers, `SystemContext.comfyui` field, ComfyUI process detection in `gather_context`, registers `ComfyUISpecialist` in `_initialize_experts`.
- Does NOT modify: `core/gpu_optimizer_v6.py` (rollback path), expert internal logic, backend internals.

### Forge-4 — Engine cache + background compiler
- Files: `core/engine_cache.py` (new), `core/background_compiler.py` (new), SQLite migration in `data/engine_cache.db`.
- Owns: cache key computation, eviction, idle-gated compile loop.
- Used by Forge-3 from `_apply_compile_trt_engine` handler.

### Crucible-1 — Adversarial verifier (correctness)
- Edge cases (mandatory, not bonus — Commandment 7):
  - Stale engine after model file mtime change → must invalidate.
  - Driver version change → must invalidate all engines.
  - ComfyUI process disappears mid-monitor → expert must not crash, must downgrade context gracefully.
  - Cache directory full → eviction triggered, no failed compile.
  - Background compile during user request → preemption works, no VRAM leak.
  - TRT compile timeout → falls back to PyTorch baseline; no orphaned `.plan.tmp` file.
  - Concurrent ComfyUI WebSocket connections → polling client tolerates.
  - Houdini and ComfyUI both running → existing 5 experts still get their recommendations applied.
- **Cannot weaken tests to make them pass** (Commandment 7). Cannot edit Forge code; raises bugs as blockers.

### Crucible-2 — Performance verifier
- For each backend admitted, runs the 8-prompt × 4-seed eval per model class (SDXL, Flux, Cosmos, Wan).
- Writes LPIPS/CLIP/wall-clock to `data/quality_ledger.db`.
- Pareto admission rule (§3.10) is enforced here.
- Authority: can **block** a backend from being recommended-default; **cannot** modify the backend itself (returns blocker to Forge-1).

---

## 6. Phasing — minimum-viable-first (Commandment 8)

The gate Joe is being asked to clear is **scope of Phase A only.** Phases B and C re-gate at their own boundaries.

### Phase A — "ComfyUI gets fast tomorrow" (~5 work-days)
- Forge-1: `tensorrt_engine.py`, `sage_attention.py`, `backend.py` ABC.
- Forge-2: `ComfyUISpecialist` with rules 1, 2, 3, 5 (skip rule 4 FP8-promotion until Phase B).
- Forge-3: orchestrator V8, ENV_VAR + INSTALL_CUSTOM_NODE + COMPILE_TRT_ENGINE handlers, `SystemContext.comfyui` field, ComfyUI HTTP polling client.
- Forge-4: engine cache + background compiler.
- Crucible-1: edge-case suite for the above.
- Crucible-2: SDXL@1024² and Flux@1024² baselines + post-TRT measurements.
- Deliverable: `comfyui --use-sage-attention` running + ≥ 1 cached TRT engine for the user's most-used (model, resolution).

### Phase B — "more speed, more risk" (~5 work-days)
- Nunchaku SVDQuant.
- TRT-LLM diffusion plugins for Flux/MMDiT.
- FP8 quality gate + automatic FP8 promotion.
- Genetic tuner gene set extended.

### Phase C — "research" (open-ended)
- Custom CUTLASS FP8 GroupedGemm kernels for multi-LoRA / batched workflows.
- Custom OptiX hooks (Houdini Karma XPU side).

---

## 7. Risks / known unknowns / explicit blockers (Commandment 4)

| ID | Risk | Mitigation | Status |
|---|---|---|---|
| R1 | CUTLASS 4.x Python DSL still moving fast as of 2026-01 | Pin to commit hash, not branch; cache `pip freeze` output | Open — Forge-1 to pin |
| R2 | TRT 11 driver requirement (≥ 575) | Joe's driver is 596.36 → ✓ | Closed |
| R3 | ComfyUI WebSocket schema drift across versions | NVIDIA Researcher expert subscribes to ComfyUI releases; client is version-aware | Open — Forge-2 |
| R4 | TRT compile time blocks the user | BackgroundCompiler is idle-gated; preemption on workload start | Designed |
| R5 | FP8 / Nunchaku quality regressions | Quality gate (§3.10); admission Pareto rule | Designed |
| R6 | Engine cache disk consumption (200 GB default) | Configurable; LRU eviction; lives on D: SSD off OneDrive | Designed |
| R7 | OneDrive sync churn on `data/*.db` | All telemetry/cache writes redirect to a path **outside OneDrive** (D: or `%LOCALAPPDATA%`) | Open — Forge-3 must add this to config defaults |
| R8 | ComfyUI not always at 8188 (user may have changed port) | Read from ComfyUI's `extra_model_paths.yaml` if present; fall back to scanning `python.exe` cmdline `--port` | Designed |
| R9 | Custom node install needs ComfyUI restart | `INSTALL_CUSTOM_NODE` action's apply handler emits a user-visible "restart ComfyUI" notification; does not auto-kill the process | Designed |

### Explicit blockers (Architect cannot resolve without user input)
- **B1.** Joe must confirm the SSD path for engine cache. Default proposed: `D:\ComfyUI_TRT_Cache` (3.6TB SSD). If Joe wants it on a different drive, set in config.
- **B2.** Joe must confirm whether to install ComfyUI custom nodes automatically (default: yes, with notification) or only suggest (default: no, surface the install command).
- **B3.** Phase B's Nunchaku requires offline model conversion (~60 min/model). Confirm whether that's acceptable to gate on Phase B start.

---

## 8. Human Gate (Commandment 8)

This document is the contract. Before any code is written, Joe approves:

1. **Scope**: Phase A only? Or Phase A+B?
2. **Naming**: Optimizer V8 as a sibling (V7 preserved) — yes/no?
3. **Engine cache location**: `D:\ComfyUI_TRT_Cache` — yes/different path?
4. **Custom node auto-install**: auto with notification — yes/manual-only?
5. **Frozen surface (§1.1)**: any objection to those being treated as immutable in this refactor?

On approval, the next message from the Architect role is a **signed-off, role-isolated brief per Forge agent** (the §5 sections become standalone briefs that Forge-N receives without §1-4 context — preventing freelancing).

---

*End of original Architect's Design Document. Gate cleared 2026-05-06.*

---

## AMENDMENT 1 — System-Wide Scope (post-gate, 2026-05-06)

**Gate decisions (recorded):**
1. Phase A **and** Phase B in scope. Phase C deferred.
2. V8 as sibling of V7 confirmed.
3. Engine cache on D: SSD confirmed. **Path renamed** `D:\OptimizerV8_EngineCache` (was `D:\ComfyUI_TRT_Cache`) because the cache now serves Houdini + UE5 + ComfyUI; subdirs `comfyui/`, `houdini/`, `ue5/`.
4. Custom node / plugin auto-install **with user-visible notification** confirmed.
5. Frozen surface §1.1 stands.

**Architectural delta — system-wide expansion:**

The original Phase A was ComfyUI-led. User's "system wide" instruction reshapes scope: Houdini and UE5 must receive CUTLASS/TRT acceleration through the same shared infrastructure (EngineCache + BackgroundCompiler + Backend ABC), not as a follow-up. Backends remain **workload-agnostic** — a `TensorRTBackend` knows nothing about ComfyUI vs Houdini vs UE5; specialists feed it model+shape pairs, get back a `BuildArtifact`.

**Specialist roster (V8 final):**

| Specialist | Source | Role |
|---|---|---|
| `NVIDIAResearcher` | V7 (extended in §3.8) | Tracks CUTLASS, TRT-LLM, ComfyUI, Houdini ML, NIM releases. |
| `GPUSpecialist` | V7 (unchanged) | CUDA env vars, driver settings. |
| `ComfyUISpecialist` | NEW | Diffusion workload optimization. |
| `HoudiniSpecialist` | NEW (replaces V7 `HoudiniEngineer` in V8 only — V7 file untouched) | Karma XPU, ML SOPs, Vellum, Pyro, OptiX denoiser, TOPs scheduling. |
| `UE5Specialist` | NEW (replaces V7 `UE5Engineer` in V8 only) | Lumen, Nanite, VSM, Path Tracer, NVIDIA NIM Plugin, DLSS verification. |
| `CrossAppCoordinator` | NEW | Arbitrates cross-app compile queue, VRAM allocation, idle-window scheduling. |
| `SystemIntegrator` | V7 (unchanged) | PCIe, ReBAR, GPU scheduler, thermal. |

**System-wide acceleration map (canonical):**

| App | Phase A targets (scope of Forge work) | Phase B targets |
|---|---|---|
| ComfyUI | SageAttn 2, ComfyUI-TensorRT (UNet+VAE, FP16 only in A) | Nunchaku SVDQuant W4A4, TRT-LLM diffusion plugins for Flux/MMDiT, FP8 promotion gated on quality ledger |
| Houdini 21 | ML SOP ONNX→TRT auto-conversion, OptiX denoiser registry-pin, TOPs/PDG GPU scheduler config | cuTensor Vellum solver (CUTLASS 4.x FP8 GroupedGemm backend; fills the existing `acceleration/cutensor_vellum_solver.py` placeholder), Pyro advection einsum, Karma XPU ML denoiser TRT |
| UE5.6 | NVIDIA NIM Plugin detection + config, DLSS 4 / DLSS-RR verification, Path Tracer GPU denoiser tuning | NIM TRT-LLM integration patterns, Lumen ML denoiser config when shipped |
| System | EngineCacheManager (app-keyed), BackgroundCompiler (idle-gated, app-aware preemption), CrossAppCoordinator | Cross-app quality ledger, genetic tuner gene set per (app, workflow) |

**Updated risk register additions:**

| ID | Risk | Mitigation | Owner |
|---|---|---|---|
| R10 | NIM Plugin for UE5.6 release/version drift | NVIDIA Researcher subscribes to NIM releases; UE5Specialist falls back to "advise install" if absent | Forge-2 |
| R11 | Houdini 21 ML SOP API changed from H20 | Specialist queries Houdini's running version via `hou.applicationVersion()` IPC; per-version code paths | Forge-2 |
| R12 | TRT engines created in one CUDA context fail in another | EngineCache stores serialized .plan files only; deserialization is per-process | Forge-4 |
| R13 | Engine cache hash collision between apps | Cache key includes `app` field as first component | Forge-4 |
| R14 | Houdini and ComfyUI both running with both wanting compile time | CrossAppCoordinator: foreground PID wins; background app's compiles paused | Forge-2 (CrossApp) |

**Phasing — system-wide A+B (committed):**

- **Phase A** (~7 work-days, was 5): all 4 new specialists at minimum-rule depth + shared infra.
- **Phase B** (~7 work-days): Nunchaku, cuTensor Vellum (Houdini), NIM TRT-LLM (UE5), FP8 quality gate, gene set extension.
- **Phase C**: deferred.

**Per-Forge briefs:** emitted as separate role-isolated artifacts in `briefs/`. Each Forge agent receives only its own brief plus a small read-only "shared context" excerpt (data-structure shapes, frozen surface). They do **not** see one another's briefs (Commandment 5 enforcement).

*End of Amendment 1. Briefs follow.*
