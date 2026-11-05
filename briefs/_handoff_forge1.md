# Forge-1 Handoff -- Acceleration Backends

**Owner:** Forge-1
**Reads this:** Forge-2 (specialists), Forge-4 (cache + background compiler), Crucible-1 (verification)
**Brief satisfied:** `briefs/forge-1_backends.md` D1-D4
**Status:** Phase A delivered. All 4 backends importable, 20/22 pytest tests pass on this host (2 skipped pending TensorRT install -- see "Environment notes" below).

---

## Files delivered

| Path | Status | Public symbols |
|---|---|---|
| `acceleration/backend.py` | NEW | `AccelerationBackend`, `BackendSource`, `OnnxPath`, `TorchModulePickle`, `TrtPlanBytes`, `ShapeRange`, `CancelToken`, `CompileCancelled`, `BuildArtifactDescriptor`, `LoadedEngine` |
| `acceleration/tensorrt_engine.py` | NEW | `TensorRTBackend` |
| `acceleration/sage_attention.py` | NEW | `SageAttentionV2Backend`, `SageSDPAPatch` |
| `acceleration/cutlass_integration.py` | EXTENDED (existing 453-line module untouched; new class appended) | `CutlassFP8GemmBackend` |
| `tests/forge1/conftest.py` | NEW | (sys.path injector for repo root) |
| `tests/forge1/test_backends.py` | NEW | 22 tests covering all 4 backends + workload-neutrality grep |

No `core/` files modified. No `config.yaml` modified. No imports from `core/experts/`, `core/engine_cache.py`, or `core/background_compiler.py` (verified via grep + parametrized pytest).

---

## D1. AccelerationBackend ABC

**Import:** `from acceleration.backend import AccelerationBackend, ...`

**ABC contract** (every concrete backend implements all six):

```python
class AccelerationBackend(ABC):
    name: str           # short tag used in cache key (e.g. "tensorrt")
    version: str        # backend impl version + underlying SDK version
    enabled: bool       # set True in __init__ iff SDK importable

    def applies_to(self, model_kind: str, shape: Tuple[int, ...], precision: str) -> bool: ...
    def estimated_compile_seconds(self, model_kind: str, shape) -> float: ...
    def estimated_speedup(self, model_kind: str, shape) -> float: ...      # vs torch baseline
    def estimated_quality_delta(self) -> float: ...                         # 0=lossless, 1=visibly bad
    def compile(self, source: BackendSource, shape_range: ShapeRange,
                precision: str, cancel_token: CancelToken,
                progress_cb: Callable[[float], None] | None = None,
                ) -> BuildArtifactDescriptor: ...
    def load(self, artifact_path: Path) -> LoadedEngine: ...
```

**Source dataclasses:**

```python
@dataclass
class OnnxPath:           path: Path; opset: Optional[int] = None
@dataclass
class TorchModulePickle:  pickle_bytes: bytes; module_class_qualname: str = ""
@dataclass
class TrtPlanBytes:       plan_bytes: bytes; plan_format_version: str = "trt_plan_v1"

BackendSource = Union[OnnxPath, TorchModulePickle, TrtPlanBytes]
```

**ShapeRange** (dynamic-shape profile):

```python
@dataclass
class ShapeRange:
    min_shape: Tuple[int, ...]
    opt_shape: Tuple[int, ...]
    max_shape: Tuple[int, ...]

    @classmethod
    def static(cls, shape: Tuple[int, ...]) -> "ShapeRange": ...
```

Construction validates rank match + monotonicity (raises `ValueError` otherwise).

**CancelToken** (cooperative cancel for Forge-4's background compiler):

```python
tok = CancelToken()
tok.cancel("gpu_hot")          # set the flag
tok.is_cancelled()             # check
tok.raise_if_cancelled()       # raises CompileCancelled if set
```

Backends call `cancel_token.raise_if_cancelled()` inside any progress hook.

**BuildArtifactDescriptor** (what `compile()` returns; Forge-4 persists `.artifact_bytes`):

```python
@dataclass
class BuildArtifactDescriptor:
    backend_name: str
    backend_version: str
    artifact_bytes: bytes
    artifact_format: str        # "trt_plan", "cutlass_so", "noop"
    precision: str
    shape_range: ShapeRange
    compile_seconds: float
    notes: Dict[str, Any]
    # property: size_bytes
```

**LoadedEngine** (what `load()` returns; `runner` shape varies by backend):

```python
@dataclass
class LoadedEngine:
    backend_name: str
    backend_version: str
    runner: Any              # see per-backend section below
    metadata: Dict[str, Any]
```

**Smoke-test command:**
```
python -m pytest tests/forge1/test_backends.py::test_backend_abc_imports tests/forge1/test_backends.py::test_shape_range_static tests/forge1/test_backends.py::test_cancel_token_lifecycle -v
```

---

## D2. TensorRTBackend

**Import:** `from acceleration.tensorrt_engine import TensorRTBackend`

**Constructor:**
```python
be = TensorRTBackend()
# be.enabled is True iff `tensorrt` Python package imports successfully.
# When disabled, applies_to() returns False; compile() raises with the install hint.
```

**applies_to:** True for `model_kind in {"gemm","conv","unet","transformer_block","attention","controlnet","vae"}` and `precision in {"fp32","fp16","bf16","tf32","fp8","int8"}`. Also rejects trivially small shapes (all dims <= 8).

**compile:** Accepts `OnnxPath | TrtPlanBytes`. `TorchModulePickle` raises `NotImplementedError` -- caller exports to ONNX first (the orchestrator already plans to do this).
- FP16 default; BF16, FP8, INT8, TF32 supported via `BuilderFlag`.
- Builds a single optimization profile from `ShapeRange.min/opt/max`.
- Returns `BuildArtifactDescriptor(artifact_format="trt_plan", artifact_bytes=<serialized plan>)`.
- Honors `cancel_token` before/after the blocking `build_serialized_network` call and inside the parse step.

**load:** Returns `LoadedEngine(runner=_TrtRunner(...))` where `_TrtRunner` exposes:
- `num_io_tensors() -> int`
- `io_tensor_name(idx) -> str`
- `set_input_shape(name, shape) -> None`
- `execute_async_v3(stream_handle: int) -> bool`

The orchestrator (Forge-3) is responsible for marshalling torch tensors via DLPack/`data_ptr()` and calling `execute_async_v3` on the active CUDA stream.

**Smoke-test commands:**
```
# disabled-path (always runnable):
python -m pytest tests/forge1/test_backends.py::test_tensorrt_backend_degrades_when_missing -v

# enabled-path (runs end-to-end build of a 64x64x64 ONNX gemm; gated on tensorrt + onnx):
python -m pytest tests/forge1/test_backends.py::test_tensorrt_backend_compile_64x64x64_onnx_gemm -v
```

**How to call from Forge-2 / Forge-4:**
```python
from acceleration.tensorrt_engine import TensorRTBackend
from acceleration.backend import OnnxPath, ShapeRange, CancelToken

be = TensorRTBackend()
if be.enabled and be.applies_to("unet", (1, 4, 64, 64), "fp16"):
    desc = be.compile(
        OnnxPath(Path(r"C:\path\to\model.onnx")),
        ShapeRange.static((1, 4, 64, 64)),
        precision="fp16",
        cancel_token=CancelToken(),
        progress_cb=lambda p: print(f"trt build {p*100:.0f}%"),
    )
    # Forge-4 writes desc.artifact_bytes to D:\OptimizerV8_EngineCache\<app>\<key>.plan
    # Later:
    engine = be.load(plan_path_from_cache)
    runner = engine.runner   # _TrtRunner -- use execute_async_v3
```

---

## D3. SageAttentionV2Backend

**Import:** `from acceleration.sage_attention import SageAttentionV2Backend`

**Constructor:**
```python
be = SageAttentionV2Backend()
# be.enabled is True iff `sageattention` AND `torch` import.
# Pinned to v2.x; warns if a non-2 major is detected.
```

**applies_to:** True for attention-shape inputs only (`model_kind in {"attention","sdpa","transformer_block","self_attention","cross_attention","mha"}` AND rank >= 3).

**compile:** No-op. Returns `BuildArtifactDescriptor(artifact_bytes=b"", artifact_format="noop", compile_seconds=0.0)`. Cache layer can record the choice without writing bytes.

**load:** Returns `LoadedEngine(runner=SageSDPAPatch(...))`. **`runner` is a context manager.** Use it like this:

```python
engine = be.load(Path("ignored.bin"))   # path is unused for this backend
with engine.runner:
    # Inside this block, torch.nn.functional.scaled_dot_product_attention
    # is replaced by a sageattention-backed implementation.
    output = my_diffusion_model(latents, prompt_emb)
# On exit, SDPA is restored. No global mutation persists.
```

**Re-entrancy:** A single `SageSDPAPatch` instance is **not** re-enterable (raises if you try). Call `be.load(...)` again for a fresh patch.

**Smoke-test command:**
```
python -m pytest tests/forge1/test_backends.py::test_sage_backend_monkeypatch_toggles_sdpa -v
```

---

## D4. CutlassFP8GemmBackend (Phase A: registered, disabled)

**Import:** `from acceleration.cutlass_integration import CutlassFP8GemmBackend`

**Constructor:**
```python
be = CutlassFP8GemmBackend()                # enabled=False (Phase A policy)
be = CutlassFP8GemmBackend(force_enable=True)  # lab-only, requires CUTLASS + nvcc
```

**Phase A status (per brief):** The class is shipped so Forge-2 can reference it, but the orchestrator MUST NOT dispatch `compile()` yet. Phase C will flip via `config.yaml` (Forge-3 owns the toggle). `compile()` and `load()` raise `RuntimeError` when `enabled=False`.

**applies_to:** True iff:
- `model_kind == "gemm"`
- `precision in {"fp8","fp8_e4m3","fp8_e5m2"}`
- shape rank == 3 (must be `(M, N, K)`)
- each dim in [16, 16384] AND multiple of 8

`applies_to` may return True even when `enabled=False` so the orchestrator can preview reachability. Always gate `compile()` calls on `be.enabled` in the dispatcher.

**compile (when enabled):** Renders a CUTLASS GEMM shim `.cu`, runs NVCC with flags from `CUTLASSManager.get_nvcc_flags(prec_mode)`, returns the compiled `.dll`/`.so` bytes in `BuildArtifactDescriptor(artifact_format="cutlass_so")`. (M, N, K) baked into the kernel; cache key already keys on shape.

**load (when enabled):** ctypes-loads the shim and returns `LoadedEngine(runner=_CutlassFP8GemmRunner(...))` exposing:
- `gemm(a_ptr: int, b_ptr: int, c_ptr: int, m, n, k, alpha=1.0, beta=0.0) -> int`
  Pointers are raw GPU device pointers (`tensor.data_ptr()`); returns 0 on success.

**CUTLASS install path:** Reuses existing `CUTLASSManager` (no API changes). Defaults to `CUTLASSManager.DEFAULT_PATH` or `CUTLASS_PATH` env var. The existing `install_and_run.bat` clones CUTLASS 3.5.1, which is sufficient (sm_89 FP8 kernels exist in 3.5.x).

**Smoke-test command:**
```
python -m pytest tests/forge1/test_backends.py::test_cutlass_fp8_gemm_backend_phase_a_disabled tests/forge1/test_backends.py::test_cutlass_fp8_gemm_applies_to_correct_shapes tests/forge1/test_backends.py::test_cutlass_fp8_gemm_compile_raises_when_disabled -v
```

---

## Workload-neutrality verification

Brief requirement: backends MUST NOT import from `core/experts/`, `core/engine_cache.py`, or `core/background_compiler.py`.

Grep verification (run from repo root):
```
# Should produce no output:
grep -rE "from core\.(experts|engine_cache|background_compiler)|import core\.(experts|engine_cache|background_compiler)" acceleration/
```

Pytest-parametrized verification (already in suite):
```
python -m pytest tests/forge1/test_backends.py::test_no_forbidden_imports -v
```

All four backend modules pass.

---

## Run the full suite

```
cd C:\Users\User\OneDrive\Desktop\Optimizer_V7_G3
python -m pytest tests/forge1/ -x -v
```

**Current state on the dev host:** 20 passed, 2 skipped. The 2 skipped tests both gate on `tensorrt` being importable; they will activate automatically once TRT 11.x is installed.

---

## Environment notes (read me, Forge-2 / Forge-4)

1. **The committed `venv/` at the repo root is broken.** Its `pyvenv.cfg` points to `C:\Users\User\miniconda3\python.exe`, which does not exist on the current dev host. Activating it falls through to system Python 3.14 (`C:\Python314\python.exe`). All Forge-1 work was therefore verified against system Python 3.14, which has `torch 2.9.1+cu130`, `sageattention`, `onnx 1.21.0`, `numpy 2.2.6`, and `pytest 8.4.2` available. **Recommend rebuilding the venv from scratch** (`py -3.13 -m venv venv` after restoring miniconda 3.13 OR `py -3.12 -m venv venv` to match the project's Python 3.12 path on `C:\Users\User\AppData\Local\Programs\Python\Python312`).

2. **TensorRT 11.x is not installed** on the dev host. Forge-1 code degrades cleanly (`be.enabled=False`); no install was attempted because (a) the venv is broken and (b) installing TRT into system Python without permission is out of scope. **Recommended next step (Forge-4 or operator):** rebuild venv, then
   ```
   pip install tensorrt --index-url https://pypi.nvidia.com
   ```
   Per the brief's "LIKELY BLOCKER" instruction. Once installed, the 2 skipped pytest tests will exercise the full ONNX -> plan path.

3. **CUTLASS 4.x DSL not used.** Per the brief, Phase A keeps the class disabled and uses the existing 3.5.1 install path via `CUTLASSManager`. Phase C may migrate to the 4.x Python DSL.

---

## Open questions for Forge-3 / Forge-4

- **Multi-input ONNX models in `TensorRTBackend.compile()`:** Today the same `ShapeRange` is applied to every input tensor. For multi-input models with different ranks, we'll need a `Dict[str, ShapeRange]` overload. **Filing this as a non-blocker observation, not a blocker.**
- **`CancelToken` mid-build cooperative cancellation:** TRT's `build_serialized_network` is a single blocking call with no public progress hook in TRT 11.x. We check the token before/after and inside the ONNX parse step; mid-build cancellation will require either NVTX-style hooks (TRT 11.x feature -- not yet validated) or process-level kill from Forge-4.

If anything in this handoff is wrong or missing, file a blocker rather than freelancing across the role boundary (Commandment 5).
