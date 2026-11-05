# Shared Context — read-only, included by every Forge/Crucible brief

This file is the **only** cross-agent context. If a Forge needs information that isn't here and isn't in their own brief, they raise a blocker (Commandment 4).

## Hardware
- GPU: NVIDIA GeForce RTX 4090 (sm_89 / Ada Lovelace), 24 GB GDDR6X, 1.01 TB/s, 4th-gen tensor cores (FP8/BF16/TF32/FP16/INT8).
- CPU: AMD Threadripper PRO 7965WX, 24 cores / 48 threads.
- RAM: 128 GB DDR5.
- Driver: Studio 596.36 (TensorRT 11.x compatible).
- OS: Windows 11 Pro for Workstations.
- Storage: 3.6 TB SSD on D: (engine cache target), 3.6 TB HDD.

## Project layout (relative to repo root `Optimizer_V7_G3/`)
```
acceleration/
  backend.py                    [Forge-1 owns]
  tensorrt_engine.py            [Forge-1 owns]
  sage_attention.py             [Forge-1 owns]
  cutlass_integration.py        [Forge-1 extends — does NOT modify CUTLASSManager API]
  cutensor_vellum_solver.py     [Phase B, Forge-1 fills the placeholder]
  high_impact_optimizer.py      [V7, frozen]
core/
  gpu_optimizer_v6.py           [V7, FROZEN — do not touch]
  gpu_optimizer_v8.py           [Forge-3 owns — sibling, V7 preserved]
  genetic_tuner.py              [V7; Phase B Forge-3 extends gene set]
  experts/
    __init__.py                 [Forge-3 creates]
    comfyui_specialist.py       [Forge-2 owns]
    houdini_specialist.py       [Forge-2 owns]
    ue5_specialist.py           [Forge-2 owns]
    cross_app_coordinator.py    [Forge-2 owns]
  engine_cache.py               [Forge-4 owns]
  background_compiler.py        [Forge-4 owns]
config.yaml                     [Forge-3 extends]
data/
  engine_cache.db               [Forge-4 owns the schema]
  comfyui_telemetry.db          [Forge-4 owns the schema]
  quality_ledger.db             [Crucible-2 owns the schema]
briefs/                         [this directory; do not modify]
```

## Frozen surface (V7 invariants — do NOT modify)
- `Expert(ABC).analyze(ctx) -> List[Recommendation]` — core/gpu_optimizer_v6.py:170
- `Recommendation` dataclass field shape — core/gpu_optimizer_v6.py:127
- `OptimizationAction` enum values: only **add** new members; never rename/remove. — :78
- `gather_context()` graceful-degrade contract (must return safe `SystemContext` on exception) — :1105
- `core/gpu_optimizer_v6.py` — file unchanged for the entire V8 effort. Rollback path.

## Frozen data structures (copy-paste; do not redefine)

```python
@dataclass
class Recommendation:
    expert: ExpertType
    action: OptimizationAction
    target: str
    parameters: Dict[str, Any]
    confidence: float           # 0.0 - 1.0
    expected_impact: float      # estimated % gain
    reasoning: str
    research_citation: Optional[str] = None
    requires: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)

@dataclass
class OptimizationResult:
    recommendation: Recommendation
    applied: bool
    timestamp: datetime
    actual_impact: Optional[float] = None
    error: Optional[str] = None
    previous_value: Optional[Any] = None
    can_rollback: bool = True
```

## V8 additions (Forge-3 lands these; everyone else can rely on the shapes)

```python
# OptimizationAction additions (Forge-3 adds; Forge-2 produces; Forge-3 dispatches)
COMPILE_TRT_ENGINE       # async build; gated on idle
INSTALL_CUSTOM_NODE      # ComfyUI custom_nodes/ git clone + pip install
INSTALL_PLUGIN           # UE5 Marketplace plugin verification + install advisory
ENABLE_BACKEND           # toggle a backend on a target app
SET_LAUNCH_FLAG          # advise restart with new flags (config write only)
EVICT_ENGINE_CACHE       # reclaim disk
WIRE_BACKEND_PATCH       # write a small monkeypatch into target app's extras dir

# ExpertType additions
COMFYUI_SPECIALIST
HOUDINI_SPECIALIST_V8       # NB: V7 HOUDINI_ENGINEER stays, V8 uses this
UE5_SPECIALIST_V8           # NB: V7 UE5_ENGINEER stays, V8 uses this
CROSS_APP_COORDINATOR

# SystemContext additive optional fields (default None)
comfyui: Optional[ComfyUIState] = None
houdini_v8: Optional[HoudiniState] = None
ue5_v8: Optional[UE5State] = None
cross_app: Optional[CrossAppState] = None
```

## Cache key format (Forge-4 owns; everyone else consumes)

```
hash_key = blake2b(
    f"{app}|{model_sha256}|{shape_w}x{shape_h}|{batch}|{precision}|"
    f"{backend_name}@{backend_ver}|drv={driver_ver}|sm={89}"
).hexdigest()[:32]
```

`app` ∈ {"comfyui", "houdini", "ue5"}. Cache root: `D:\OptimizerV8_EngineCache\<app>\<hash_key>.plan`.

## Operational commandments (every agent obeys)

1. **Scout before mutate** — read 2-3 existing examples before creating anything.
2. **Verify after every mutation** — run pytest target after each file change.
3. **Bounded failure → escalate** — 3 retries, then stop and surface what failed.
4. **Complete output or explicit blocker** — no stubs, no TODOs, no truncation.
5. **Role isolation** — work strictly within the brief. Never freelance across boundaries.
6. **Explicit handoffs** — produce the named deliverable; that's the contract.
7. **Adversarial verification** — Crucible cannot weaken tests to make them pass.
8. **No human gates** between brief acceptance and brief completion. The post-design gate has cleared. Re-gate only on Phase C entry.
