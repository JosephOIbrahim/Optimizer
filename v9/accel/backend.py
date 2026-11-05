"""V9 — AccelerationBackend ABC + artifact dataclasses.

Ported from v8's acceleration/backend.py (the design was sound) and cleaned:
  * workload-agnostic — a backend knows nothing about ComfyUI/Houdini/UE5;
    advisors hand it (model_kind, shape, precision) and get a BuildArtifact.
  * degrade-safe — concrete backends probe their SDK in __init__ and set
    `enabled=False` on ImportError; `compile()` raises a clear install hint.
  * no hardware literals — the cache key's driver/arch come from HardwareProfile
    via the build layer, not from inside the backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union


# --- compile sources --------------------------------------------------------
@dataclass
class OnnxPath:
    path: Path
    opset: Optional[int] = None


@dataclass
class TorchModulePickle:
    pickle_bytes: bytes
    module_class_qualname: str = ""


@dataclass
class TrtPlanBytes:
    plan_bytes: bytes
    plan_format_version: str = "trt_plan_v1"


BackendSource = Union[OnnxPath, TorchModulePickle, TrtPlanBytes]


# --- dynamic-shape profile --------------------------------------------------
@dataclass
class ShapeRange:
    min_shape: Tuple[int, ...]
    opt_shape: Tuple[int, ...]
    max_shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        ranks = {len(self.min_shape), len(self.opt_shape), len(self.max_shape)}
        if len(ranks) != 1:
            raise ValueError("min/opt/max shapes must share rank")
        for lo, mid, hi in zip(self.min_shape, self.opt_shape, self.max_shape):
            if not (lo <= mid <= hi):
                raise ValueError(
                    f"shape not monotonic: min={self.min_shape} "
                    f"opt={self.opt_shape} max={self.max_shape}")

    @classmethod
    def static(cls, shape: Tuple[int, ...]) -> "ShapeRange":
        return cls(shape, shape, shape)


# --- cooperative cancellation (for the background compiler) -----------------
class CompileCancelled(Exception):
    """Raised inside compile() when the cancel token is set."""


class CancelToken:
    def __init__(self) -> None:
        self._cancelled = False
        self._reason = ""

    def cancel(self, reason: str = "") -> None:
        self._cancelled = True
        self._reason = reason

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CompileCancelled(self._reason or "compile cancelled")


# --- compile output / load result ------------------------------------------
@dataclass
class BuildArtifactDescriptor:
    backend_name: str
    backend_version: str
    artifact_bytes: bytes
    artifact_format: str                       # "trt_plan" | "cutlass_so" | "noop"
    precision: str
    shape_range: ShapeRange
    compile_seconds: float
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.artifact_bytes)


@dataclass
class LoadedEngine:
    backend_name: str
    backend_version: str
    runner: Any                                # backend-specific (see subclass)
    metadata: dict[str, Any] = field(default_factory=dict)


# --- the plug point ---------------------------------------------------------
class AccelerationBackend(ABC):
    name: str = "backend"
    version: str = "0.0"

    def __init__(self) -> None:
        #: set True in subclass __init__ iff the underlying SDK imported
        self.enabled: bool = False

    @abstractmethod
    def applies_to(self, model_kind: str, shape: Tuple[int, ...],
                   precision: str) -> bool: ...

    @abstractmethod
    def estimated_compile_seconds(self, model_kind: str,
                                  shape: Tuple[int, ...]) -> float: ...

    @abstractmethod
    def estimated_speedup(self, model_kind: str,
                          shape: Tuple[int, ...]) -> float: ...

    @abstractmethod
    def estimated_quality_delta(self) -> float:
        """0.0 = lossless, 1.0 = visibly degraded."""
        ...

    @abstractmethod
    def compile(self, source: BackendSource, shape_range: ShapeRange,
                precision: str, cancel_token: CancelToken,
                progress_cb: Optional[Callable[[float], None]] = None,
                ) -> BuildArtifactDescriptor: ...

    @abstractmethod
    def load(self, artifact_path: Path) -> LoadedEngine: ...
