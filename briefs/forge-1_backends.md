# Forge-1 — Acceleration Backends (workload-agnostic)

**Read first:** `briefs/SHARED_CONTEXT.md`. That is the only context you have outside this brief.

## Authority boundary (Commandment 5)
- **You implement:** the backend ABC and 4 concrete backend classes.
- **You do NOT touch:** any file under `core/`, `config.yaml`, any expert class, the V7 orchestrator, the cache or compiler. If your design needs an interface that doesn't exist on the orchestrator side, raise a blocker (Commandment 4) — do not invent it.
- **Workload neutrality is non-negotiable:** a backend must not import anything from `core/experts/` or know whether it's serving ComfyUI, Houdini, or UE5. Backends speak in terms of (model bytes / ONNX path / nn.Module, shape, precision) — never in terms of apps.

## Deliverables (4 files)

### D1. `acceleration/backend.py` — the ABC every backend implements
```python
class AccelerationBackend(ABC):
    name: str
    version: str

    @abstractmethod
    def applies_to(self, model_kind: str, shape: Tuple[int, ...], precision: str) -> bool: ...

    @abstractmethod
    def estimated_compile_seconds(self, model_kind: str, shape) -> float: ...

    @abstractmethod
    def estimated_speedup(self, model_kind: str, shape) -> float:
        """Prior estimate vs torch baseline. Crucible-2 overwrites with measured values per (model, shape)."""

    @abstractmethod
    def estimated_quality_delta(self) -> float:
        """0.0 = lossless, 1.0 = visibly degraded. Used by quality gate."""

    @abstractmethod
    def compile(
        self,
        source: BackendSource,        # union: OnnxPath | TorchModulePickle | TrtPlanBytes
        shape_range: ShapeRange,
        precision: str,
        cancel_token: CancelToken,    # Forge-4's cooperative cancel
        progress_cb: Callable[[float], None] | None = None,
    ) -> BuildArtifactDescriptor:
        """Returns descriptor only. Forge-4's EngineCacheManager actually persists the bytes."""

    @abstractmethod
    def load(self, artifact_path: Path) -> LoadedEngine: ...
```

`BackendSource`, `ShapeRange`, `CancelToken`, `BuildArtifactDescriptor`, `LoadedEngine` are dataclasses you define in `backend.py`. Make them simple and explicit. No inheritance trees beyond what's shown.

### D2. `acceleration/tensorrt_engine.py` — `TensorRTBackend`
- TRT 11.x via `tensorrt` Python package.
- Supports FP16 (default in Phase A), FP8 explicit (Phase B).
- Dynamic shapes via min/opt/max profile.
- Compile uses `IBuilderConfig.set_flag(BuilderFlag.FP16/FP8)`, `INetworkDefinition` from ONNX parser.
- Honor `cancel_token` by checking inside a progress hook.
- Output: serialized `.plan` bytes returned via `BuildArtifactDescriptor`.

### D3. `acceleration/sage_attention.py` — `SageAttentionV2Backend`
- Wraps the `sageattention` PyPI package (pin to v2.x).
- `compile` is a no-op (returns a stub `BuildArtifactDescriptor` with `compile_seconds=0`); `applies_to` returns True only for attention-shape inputs.
- `load` returns a callable that monkeypatches `torch.nn.functional.scaled_dot_product_attention` *only when invoked through this backend's context manager* (no global mutation).

### D4. `acceleration/cutlass_integration.py` — **extend, do not modify** existing code
- Add a new class `CutlassFP8GemmBackend(AccelerationBackend)` at the bottom of the file.
- Reuses existing `CUTLASSManager` for installation discovery and NVCC flags.
- `compile` writes a tiny CUDA .cu shim per (M, N, K, precision), compiles via NVCC, returns `.so` path.
- Phase A: register but `enabled=False` in config (Phase C will turn it on; we ship the class now so Forge-2 can reference it). **Document this** in the class docstring.

## Sequence
1. **Scout.** Read `acceleration/cutlass_integration.py` (existing, 453 lines). Match its style/imports.
2. Create `backend.py` first; verify imports cleanly.
3. Implement `TensorRTBackend`. Smoke-test against a 64×64×64 ONNX gemm (you may include a test file under `tests/` if pytest layout allows; otherwise emit a verification script under `scripts/forge1_smoke.py`).
4. Implement `SageAttentionV2Backend`. Smoke-test the monkeypatch context manager (enter → SDPA replaced; exit → restored).
5. Extend `cutlass_integration.py`. Smoke-test that `CutlassFP8GemmBackend` instantiates and reports `applies_to` correctly.
6. After each file: run `python -m pytest tests/forge1/ -x` (or smoke script).

## Bounded failure
- TensorRT install fails 3× → blocker: "TRT 11.x wheel not available for Python {ver}; need driver/CUDA pinning info from user."
- CUTLASS already-installed path mismatch 3× → blocker: "CUTLASS_PATH env var not aligned with `CUTLASSManager.DEFAULT_PATH`; need user decision."

## Done criteria (Crucible-1 will verify)
- All 4 backends import without errors.
- `TensorRTBackend.compile()` produces a valid `.plan` for a trivial ONNX.
- `SageAttentionV2Backend.load()` returns a context manager that toggles SDPA cleanly.
- `CutlassFP8GemmBackend` instantiates and reports correct `applies_to` for FP8 GEMM shapes.
- No imports from `core/experts/`, `core/engine_cache.py`, `core/background_compiler.py`. Verified via grep.

## Handoff
Write `briefs/_handoff_forge1.md` with: list of public symbols + signatures + import paths + smoke-test command. This is what Forge-2 reads to know how to call your backends.
