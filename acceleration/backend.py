"""
Acceleration Backend ABC and shared dataclasses for Optimizer_V8.

This module defines the **workload-agnostic** contract that every concrete
acceleration backend implements. Backends speak in terms of
(model bytes / ONNX path / nn.Module pickle, shape range, precision) and
**MUST NOT** know whether they are serving ComfyUI, Houdini, or UE5.

Owners (per briefs/SHARED_CONTEXT.md and briefs/forge-1_backends.md):
- This file is owned by Forge-1.
- It must not import from `core/experts/`, `core/engine_cache.py`, or
  `core/background_compiler.py`. Those modules consume this ABC; this ABC
  consumes nothing from them.

Concrete backends (siblings of this module):
- TensorRTBackend           -- acceleration/tensorrt_engine.py
- SageAttentionV2Backend    -- acceleration/sage_attention.py
- CutlassFP8GemmBackend     -- acceleration/cutlass_integration.py (extension)

The persistence layer (Forge-4) is responsible for actually writing the
serialized engine bytes to disk. Backends only return descriptors.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Source descriptors (input to compile())
# ============================================================================

@dataclass
class OnnxPath:
    """A model expressed as a path to an ONNX file on disk."""
    path: Path
    opset: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)


@dataclass
class TorchModulePickle:
    """A model expressed as pickled torch.nn.Module bytes.

    The backend is responsible for unpickling in the appropriate process.
    Pickle bytes (not the live module) keep this dataclass picklable for
    cross-process compile dispatch (Forge-4's background compiler).
    """
    pickle_bytes: bytes
    module_class_qualname: str = ""  # for diagnostics only


@dataclass
class TrtPlanBytes:
    """An already-serialized TensorRT plan (used for re-load / re-export)."""
    plan_bytes: bytes
    plan_format_version: str = "trt_plan_v1"


# Union alias used in the ABC signature.
BackendSource = Union[OnnxPath, TorchModulePickle, TrtPlanBytes]


# ============================================================================
# Shape range (dynamic shape profile: min / opt / max)
# ============================================================================

@dataclass
class ShapeRange:
    """
    Dynamic shape profile for a backend compile call.

    Each field is a tuple sized to the model's input rank. For static-shape
    compiles, set min == opt == max.
    """
    min_shape: Tuple[int, ...]
    opt_shape: Tuple[int, ...]
    max_shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.min_shape) == len(self.opt_shape) == len(self.max_shape)):
            raise ValueError(
                f"ShapeRange ranks must match: "
                f"min={self.min_shape} opt={self.opt_shape} max={self.max_shape}"
            )
        for axis, (lo, mid, hi) in enumerate(
            zip(self.min_shape, self.opt_shape, self.max_shape)
        ):
            if not (lo <= mid <= hi):
                raise ValueError(
                    f"ShapeRange axis {axis} not monotone: "
                    f"min={lo}, opt={mid}, max={hi}"
                )

    @classmethod
    def static(cls, shape: Tuple[int, ...]) -> "ShapeRange":
        return cls(min_shape=shape, opt_shape=shape, max_shape=shape)


# ============================================================================
# Cooperative cancellation
# ============================================================================

class CancelToken:
    """
    Cooperative cancel signal threaded through compile() so Forge-4's
    background compiler can stop a long build when GPU goes hot.

    Backends should call .raise_if_cancelled() inside any progress hook
    they pass to the underlying compiler (TRT builder, NVCC subprocess
    poll, etc.).
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: Optional[str] = None

    def cancel(self, reason: str = "user_cancel") -> None:
        self._reason = reason
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CompileCancelled(self._reason or "cancelled")


class CompileCancelled(RuntimeError):
    """Raised inside compile() when the cancel token fires."""


# ============================================================================
# Compile output / load output
# ============================================================================

@dataclass
class BuildArtifactDescriptor:
    """
    What compile() returns. Forge-4's EngineCacheManager decides where the
    bytes get persisted; this descriptor only carries the bytes plus the
    metadata needed to key the cache and re-instantiate the engine later.

    Fields:
        backend_name:    e.g. "tensorrt", "sage_attention_v2",
                         "cutlass_fp8_gemm".
        backend_version: backend implementation version string.
        artifact_bytes:  the actual serialized engine (plan, .so bytes,
                         or sentinel for no-op backends).
        artifact_format: short tag describing artifact_bytes encoding.
                         "trt_plan", "cutlass_so", "noop".
        precision:       precision the engine was compiled for.
        shape_range:     shape range it was compiled for.
        compile_seconds: wall time spent in compile().
        notes:           free-form diagnostics, surfaced to the dashboard.
    """
    backend_name: str
    backend_version: str
    artifact_bytes: bytes
    artifact_format: str
    precision: str
    shape_range: ShapeRange
    compile_seconds: float
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.artifact_bytes)


@dataclass
class LoadedEngine:
    """
    What load() returns. Backends differ in what `runner` is:

      - TensorRTBackend.load        -> a callable wrapping an
                                       IExecutionContext that consumes
                                       torch tensors.
      - SageAttentionV2Backend.load -> a context manager that monkeypatches
                                       torch.nn.functional.SDPA on enter
                                       and restores it on exit.
      - CutlassFP8GemmBackend.load  -> a callable that ctypes-binds the
                                       compiled .so and exposes gemm(A,B,C).

    `runner` is intentionally typed as Any. The orchestrator dispatches by
    inspecting `backend_name` rather than runner type.
    """
    backend_name: str
    backend_version: str
    runner: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# The ABC
# ============================================================================

class AccelerationBackend(ABC):
    """
    Workload-agnostic acceleration backend.

    A backend MUST implement every abstract method below. A backend MUST NOT
    import any module from `core/experts/`, `core/engine_cache.py`, or
    `core/background_compiler.py`. It learns about the model only via the
    BackendSource / ShapeRange / precision arguments.

    The two class attributes (name, version) are inspected by the cache key
    formula in SHARED_CONTEXT.md (cf. `backend_name`@`backend_ver` token).
    """

    name: str = "abstract_backend"
    version: str = "0.0.0"

    # Subclasses set this in their __init__ when their underlying SDK is
    # actually importable / installed. The orchestrator checks this before
    # dispatching compile().
    enabled: bool = False

    # ------------------------------------------------------------------ ABC

    @abstractmethod
    def applies_to(
        self,
        model_kind: str,
        shape: Tuple[int, ...],
        precision: str,
    ) -> bool:
        """
        Cheap, fast filter. Returns True iff this backend is a candidate
        for the given (model_kind, shape, precision) triple.

        `model_kind` is a free-form lowercase string the experts agree on:
        "gemm", "attention", "conv", "unet", "transformer_block", etc.
        Backends ignore unknown kinds (return False).
        """

    @abstractmethod
    def estimated_compile_seconds(
        self,
        model_kind: str,
        shape: Tuple[int, ...],
    ) -> float:
        """Prior estimate of how long compile() will block. Used by the
        background compiler to budget idle-window builds."""

    @abstractmethod
    def estimated_speedup(
        self,
        model_kind: str,
        shape: Tuple[int, ...],
    ) -> float:
        """Prior estimate vs torch baseline (e.g. 1.8 = 1.8x).

        Crucible-2 overwrites this estimate with measured values per
        (model, shape) once benchmarks land.
        """

    @abstractmethod
    def estimated_quality_delta(self) -> float:
        """0.0 = bit-exact / lossless, 1.0 = visibly degraded.

        Used by the quality gate before the orchestrator advises swapping
        to the backend. FP16/BF16 backends typically report ~0.05; FP8
        backends ~0.15-0.25 depending on calibration.
        """

    @abstractmethod
    def compile(
        self,
        source: BackendSource,
        shape_range: ShapeRange,
        precision: str,
        cancel_token: CancelToken,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> BuildArtifactDescriptor:
        """
        Compile a model into a backend-specific artifact.

        Returns the descriptor only. Forge-4's EngineCacheManager is
        responsible for persisting `descriptor.artifact_bytes` to
        `D:\\OptimizerV8_EngineCache\\<app>\\<hash_key>.plan` (or whatever
        path the cache key resolves to).

        The implementation MUST honour `cancel_token` -- check
        `cancel_token.raise_if_cancelled()` inside any progress hook it
        passes to the underlying compiler.

        `progress_cb`, if supplied, is called with a float in [0.0, 1.0].
        """

    @abstractmethod
    def load(self, artifact_path: Path) -> LoadedEngine:
        """Load a previously-compiled artifact from disk and return a
        LoadedEngine ready for the orchestrator to dispatch against."""

    # -------------------------------------------------------------- helpers

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} name={self.name!r} version={self.version!r} "
            f"enabled={self.enabled}>"
        )


# ============================================================================
# Module CLI (sanity print only)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AccelerationBackend ABC -- Optimizer_V8 (Forge-1)")
    print("=" * 60)
    print(f"Source dataclasses : OnnxPath, TorchModulePickle, TrtPlanBytes")
    print(f"Shape descriptor   : ShapeRange (min/opt/max)")
    print(f"Cancel token       : CancelToken / CompileCancelled")
    print(f"Compile output     : BuildArtifactDescriptor")
    print(f"Load output        : LoadedEngine")
    print(f"ABC                : AccelerationBackend (6 abstract methods)")
    print("=" * 60)
