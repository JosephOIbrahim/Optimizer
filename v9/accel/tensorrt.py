"""V9 — TensorRT acceleration backend.

Workload-agnostic: this module knows nothing about ComfyUI / Houdini / UE5.
It is fed an ONNX path or a serialized plan and produces a serialized TRT
engine plan. The build/cache layer persists the bytes; this module just
returns the descriptor.

Degrade-safe: the `tensorrt` package is imported lazily in __init__ so this
module imports cleanly on a host where TRT is not installed. When TRT is
missing, `enabled` is False, `applies_to()` returns False, and `compile()`
raises with a clear install hint.

No hardware literals: the workspace budget and shape buckets are size
heuristics, not driver/arch facts. Driver/arch keying of the cache lives in
the build layer (via HardwareProfile), never here.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from v9.accel.backend import (
    AccelerationBackend,
    BackendSource,
    BuildArtifactDescriptor,
    CancelToken,
    LoadedEngine,
    OnnxPath,
    ShapeRange,
    TorchModulePickle,
    TrtPlanBytes,
)

logger = logging.getLogger("v9.accel.tensorrt")

_INSTALL_HINT = (
    "Install tensorrt via "
    "`pip install tensorrt --index-url https://pypi.nvidia.com`."
)


def _try_import_tensorrt():
    """Import tensorrt lazily. Returns (module, version) or (None, reason)."""
    try:
        import tensorrt as trt  # type: ignore
    except ImportError as exc:
        return None, f"tensorrt import failed: {exc}"
    except Exception as exc:  # pragma: no cover - broken install
        return None, f"tensorrt import error: {exc}"
    version = getattr(trt, "__version__", "unknown")
    return trt, version


def _make_trt_logger(trt_module: Any) -> Any:
    """A tiny TRT ILogger adapter that forwards to our stdlib logger."""
    class _Adapter(trt_module.ILogger):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()

        def log(self, severity, msg) -> None:  # type: ignore[override]
            sev = str(severity).rsplit(".", 1)[-1]
            if sev in {"INTERNAL_ERROR", "ERROR"}:
                logger.error("[TRT] %s", msg)
            elif sev == "WARNING":
                logger.warning("[TRT] %s", msg)
            else:
                logger.debug("[TRT] %s", msg)
    return _Adapter()


class TensorRTBackend(AccelerationBackend):
    """TensorRT acceleration backend. FP16 default, FP8/BF16/INT8 flag-gated."""

    name: str = "tensorrt"
    version: str = "1.0.0"

    # Compile-time priors used to budget background builds. Conservative;
    # the benchmark harness overwrites measured values per shape.
    _PRIOR_BUILD_SECONDS = {
        "tiny": 8.0,
        "small": 30.0,
        "medium": 90.0,
        "large": 300.0,
        "xlarge": 900.0,
    }

    # Speedup priors vs torch baseline, keyed (model_kind, precision).
    _PRIOR_SPEEDUP = {
        ("gemm", "fp16"): 2.4,
        ("gemm", "fp8"): 3.6,
        ("conv", "fp16"): 2.0,
        ("conv", "fp8"): 2.9,
        ("unet", "fp16"): 1.9,
        ("unet", "fp8"): 2.7,
        ("transformer_block", "fp16"): 1.8,
        ("transformer_block", "fp8"): 2.6,
    }

    def __init__(self) -> None:
        super().__init__()
        trt, version_or_reason = _try_import_tensorrt()
        if trt is None:
            self._trt = None
            self._trt_version = "unavailable"
            self.enabled = False
            self._unavailable_reason = version_or_reason
            logger.info(
                "TensorRTBackend disabled (%s). %s",
                version_or_reason, _INSTALL_HINT,
            )
        else:
            self._trt = trt
            self._trt_version = version_or_reason
            self.enabled = True
            self._unavailable_reason = None
            self.version = f"1.0.0+trt{version_or_reason}"
            logger.info("TensorRTBackend ready (tensorrt=%s).", version_or_reason)

    # ----------------------------------------------------------- ABC impl

    def applies_to(self, model_kind: str, shape: Tuple[int, ...],
                   precision: str) -> bool:
        if not self.enabled:
            return False
        kind = (model_kind or "").lower()
        prec = (precision or "").lower()
        if kind not in {"gemm", "conv", "unet", "transformer_block",
                        "attention", "controlnet", "vae"}:
            return False
        if prec not in {"fp32", "fp16", "bf16", "tf32", "fp8", "int8"}:
            return False
        # Skip trivially small ops where kernel launch dominates.
        if shape and all(s <= 8 for s in shape):
            return False
        return True

    def estimated_compile_seconds(self, model_kind: str,
                                  shape: Tuple[int, ...]) -> float:
        return self._PRIOR_BUILD_SECONDS[self._classify_shape(shape)]

    def estimated_speedup(self, model_kind: str,
                          shape: Tuple[int, ...]) -> float:
        return self._PRIOR_SPEEDUP.get((model_kind.lower(), "fp16"), 1.5)

    def estimated_quality_delta(self) -> float:
        # FP16 in TRT is empirically near-lossless. FP8 quality is the
        # quality gate's responsibility, not a backend prior.
        return 0.05

    def compile(self, source: BackendSource, shape_range: ShapeRange,
                precision: str, cancel_token: CancelToken,
                progress_cb: Optional[Callable[[float], None]] = None,
                ) -> BuildArtifactDescriptor:
        if not self.enabled:
            raise RuntimeError(
                f"TensorRTBackend not available: {self._unavailable_reason}. "
                f"{_INSTALL_HINT}"
            )

        trt = self._trt
        prec = (precision or "fp16").lower()
        t0 = time.perf_counter()

        # ---- 1. Source already a serialized plan -> passthrough ----------
        if isinstance(source, TrtPlanBytes):
            if progress_cb:
                progress_cb(1.0)
            return BuildArtifactDescriptor(
                backend_name=self.name,
                backend_version=self.version,
                artifact_bytes=source.plan_bytes,
                artifact_format="trt_plan",
                precision=prec,
                shape_range=shape_range,
                compile_seconds=time.perf_counter() - t0,
                notes={"path": "passthrough"},
            )

        if isinstance(source, TorchModulePickle):
            raise NotImplementedError(
                "TensorRTBackend.compile(TorchModulePickle) requires the "
                "caller to export to ONNX first; pass an OnnxPath."
            )

        if not isinstance(source, OnnxPath):
            raise TypeError(f"Unsupported BackendSource: {type(source)!r}")

        onnx_path: Path = source.path
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX source not found: {onnx_path}")

        # ---- 2. Build TRT network from ONNX ------------------------------
        trt_logger = _make_trt_logger(trt)
        builder = trt.Builder(trt_logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, trt_logger)

        cancel_token.raise_if_cancelled()
        if progress_cb:
            progress_cb(0.05)

        with open(onnx_path, "rb") as fh:
            onnx_bytes = fh.read()
        if not parser.parse(onnx_bytes):
            errs = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")

        if progress_cb:
            progress_cb(0.20)
        cancel_token.raise_if_cancelled()

        # ---- 3. Builder config + dynamic shape profile -------------------
        config = builder.create_builder_config()
        # 4 GiB workspace pool (a size heuristic, not a hardware fact).
        try:
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30),
            )
        except AttributeError:
            config.max_workspace_size = 4 * (1 << 30)  # type: ignore[attr-defined]

        self._apply_precision_flag(trt, config, prec)

        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            profile.set_shape(
                inp.name,
                shape_range.min_shape,
                shape_range.opt_shape,
                shape_range.max_shape,
            )
        config.add_optimization_profile(profile)

        if progress_cb:
            progress_cb(0.35)
        cancel_token.raise_if_cancelled()

        # ---- 4. Build serialized plan ------------------------------------
        try:
            host_mem = builder.build_serialized_network(network, config)
        except Exception as exc:  # pragma: no cover - TRT-internal
            raise RuntimeError(f"TRT build failed: {exc}") from exc

        cancel_token.raise_if_cancelled()
        if host_mem is None:
            raise RuntimeError(
                "TRT build_serialized_network returned None "
                "(check TRT logger output for errors)."
            )

        plan_bytes = bytes(host_mem)
        if progress_cb:
            progress_cb(1.0)

        elapsed = time.perf_counter() - t0
        logger.info(
            "TensorRTBackend.compile OK: %d bytes in %.2fs (%s)",
            len(plan_bytes), elapsed, prec,
        )

        return BuildArtifactDescriptor(
            backend_name=self.name,
            backend_version=self.version,
            artifact_bytes=plan_bytes,
            artifact_format="trt_plan",
            precision=prec,
            shape_range=shape_range,
            compile_seconds=elapsed,
            notes={
                "trt_version": self._trt_version,
                "onnx_path": str(onnx_path),
                "num_network_inputs": int(network.num_inputs),
                "num_network_outputs": int(network.num_outputs),
            },
        )

    def load(self, artifact_path: Path) -> LoadedEngine:
        if not self.enabled:
            raise RuntimeError(
                f"TensorRTBackend not available: {self._unavailable_reason}. "
                f"{_INSTALL_HINT}"
            )
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"TRT plan not found: {artifact_path}")

        trt = self._trt
        trt_logger = _make_trt_logger(trt)
        runtime = trt.Runtime(trt_logger)
        with open(artifact_path, "rb") as fh:
            plan_bytes = fh.read()
        engine = runtime.deserialize_cuda_engine(plan_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {artifact_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(
                f"Failed to create TRT execution context: {artifact_path}"
            )

        runner = _TrtRunner(engine=engine, context=context, trt_module=trt)
        return LoadedEngine(
            backend_name=self.name,
            backend_version=self.version,
            runner=runner,
            metadata={
                "plan_path": str(artifact_path),
                "plan_size_bytes": len(plan_bytes),
                "trt_version": self._trt_version,
            },
        )

    # ------------------------------------------------------------- helpers

    def _apply_precision_flag(self, trt: Any, config: Any, prec: str) -> None:
        if prec == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif prec == "bf16":
            if hasattr(trt.BuilderFlag, "BF16"):
                config.set_flag(trt.BuilderFlag.BF16)
            else:
                logger.warning("BF16 flag unavailable; falling back to FP16.")
                config.set_flag(trt.BuilderFlag.FP16)
        elif prec == "fp8":
            if not hasattr(trt.BuilderFlag, "FP8"):
                raise RuntimeError("TRT FP8 flag unavailable in this build.")
            config.set_flag(trt.BuilderFlag.FP8)
        elif prec == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
        elif prec in {"tf32", "fp32", ""}:
            # TF32 is TRT's default for FP32 paths on Ampere+; nothing to toggle.
            pass
        else:
            raise ValueError(f"Unsupported precision for TRT: {prec!r}")

    @staticmethod
    def _classify_shape(shape: Tuple[int, ...]) -> str:
        vol = 1
        for s in shape:
            vol *= max(int(s), 1)
        if vol <= 64 * 64 * 64:
            return "tiny"
        if vol <= 256 * 256 * 256:
            return "small"
        if vol <= 1024 * 1024 * 64:
            return "medium"
        if vol <= 4096 * 4096 * 32:
            return "large"
        return "xlarge"


class _TrtRunner:
    """Minimal callable wrapping an IExecutionContext. The orchestrator
    marshals torch tensors to/from the engine; this exposes engine + context
    plus helpers to adapt to whatever shape arrives at inference time."""

    def __init__(self, engine: Any, context: Any, trt_module: Any) -> None:
        self.engine = engine
        self.context = context
        self._trt = trt_module

    def num_io_tensors(self) -> int:
        return int(self.engine.num_io_tensors)

    def io_tensor_name(self, idx: int) -> str:
        return self.engine.get_tensor_name(idx)

    def set_input_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        self.context.set_input_shape(name, shape)

    def execute_async_v3(self, stream_handle: int) -> bool:
        return bool(self.context.execute_async_v3(stream_handle))


if __name__ == "__main__":
    be = TensorRTBackend()
    print(f"name={be.name} version={be.version} enabled={be.enabled}")
    if not be.enabled:
        print(f"unavailable_reason: {be._unavailable_reason}")
    print("applies_to(gemm,(64,64),fp16) ->",
          be.applies_to("gemm", (64, 64), "fp16"))
