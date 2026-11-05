"""Forge-1 backend smoke tests.

These tests verify:
  1. The backend ABC and concrete classes import cleanly.
  2. ShapeRange / CancelToken / dataclasses behave correctly.
  3. TensorRTBackend degrades gracefully when tensorrt is absent and
     accepts a TrtPlanBytes passthrough when present.
  4. SageAttentionV2Backend's monkeypatch context manager swaps and
     restores torch.nn.functional.scaled_dot_product_attention.
  5. CutlassFP8GemmBackend instantiates, stays disabled in Phase A,
     and reports applies_to correctly.
  6. No backend module imports anything from core/experts/,
     core/engine_cache.py, or core/background_compiler.py.

Tests skip cleanly when an optional package is missing, so the suite
is green on hosts without TRT or sageattention installed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

def test_backend_abc_imports():
    from acceleration import backend
    assert backend.AccelerationBackend is not None
    assert backend.ShapeRange is not None
    assert backend.CancelToken is not None
    assert backend.BuildArtifactDescriptor is not None
    assert backend.LoadedEngine is not None
    assert backend.OnnxPath is not None
    assert backend.TorchModulePickle is not None
    assert backend.TrtPlanBytes is not None


def test_concrete_backends_import():
    from acceleration.tensorrt_engine import TensorRTBackend
    from acceleration.sage_attention import SageAttentionV2Backend
    from acceleration.cutlass_integration import CutlassFP8GemmBackend
    assert TensorRTBackend is not None
    assert SageAttentionV2Backend is not None
    assert CutlassFP8GemmBackend is not None


# ---------------------------------------------------------------------------
# 2. ABC dataclasses
# ---------------------------------------------------------------------------

def test_shape_range_static():
    from acceleration.backend import ShapeRange
    sr = ShapeRange.static((1, 3, 64, 64))
    assert sr.min_shape == sr.opt_shape == sr.max_shape == (1, 3, 64, 64)


def test_shape_range_min_opt_max_validates_monotone():
    from acceleration.backend import ShapeRange
    with pytest.raises(ValueError):
        ShapeRange(min_shape=(2,), opt_shape=(1,), max_shape=(4,))


def test_shape_range_validates_rank_match():
    from acceleration.backend import ShapeRange
    with pytest.raises(ValueError):
        ShapeRange(min_shape=(1, 3), opt_shape=(1, 3, 4), max_shape=(1, 3))


def test_cancel_token_lifecycle():
    from acceleration.backend import CancelToken, CompileCancelled
    tok = CancelToken()
    assert not tok.is_cancelled()
    tok.raise_if_cancelled()  # no-op
    tok.cancel("test")
    assert tok.is_cancelled()
    assert tok.reason == "test"
    with pytest.raises(CompileCancelled):
        tok.raise_if_cancelled()


# ---------------------------------------------------------------------------
# 3. TensorRTBackend
# ---------------------------------------------------------------------------

def test_tensorrt_backend_degrades_when_missing():
    """Without tensorrt installed, the backend must still construct."""
    from acceleration.tensorrt_engine import TensorRTBackend
    be = TensorRTBackend()
    if be.enabled:
        pytest.skip("tensorrt is installed; degradation path not exercised")
    assert be.enabled is False
    assert be.applies_to("gemm", (64, 64), "fp16") is False
    # compile() must raise with a clear message.
    from acceleration.backend import CancelToken, OnnxPath, ShapeRange
    with pytest.raises(RuntimeError, match="tensorrt"):
        be.compile(
            OnnxPath(Path("nonexistent.onnx")),
            ShapeRange.static((1, 64, 64)),
            "fp16",
            CancelToken(),
        )


def test_tensorrt_backend_priors_are_finite():
    from acceleration.tensorrt_engine import TensorRTBackend
    be = TensorRTBackend()
    s = be.estimated_compile_seconds("gemm", (1024, 1024, 1024))
    assert s > 0 and s < 100000
    sp = be.estimated_speedup("gemm", (1024, 1024))
    assert 1.0 <= sp <= 10.0
    qd = be.estimated_quality_delta()
    assert 0.0 <= qd <= 1.0


def test_tensorrt_backend_plan_passthrough():
    """When given pre-serialized TrtPlanBytes, compile() short-circuits.

    This path requires tensorrt to be importable because the backend's
    enabled-flag gates the compile() entrypoint. Skip otherwise.
    """
    from acceleration.tensorrt_engine import TensorRTBackend
    from acceleration.backend import CancelToken, ShapeRange, TrtPlanBytes

    be = TensorRTBackend()
    if not be.enabled:
        pytest.skip("tensorrt not installed; plan passthrough requires enabled backend")

    fake_plan = b"FAKE_PLAN_BYTES_FOR_PASSTHROUGH"
    desc = be.compile(
        TrtPlanBytes(plan_bytes=fake_plan),
        ShapeRange.static((1, 64, 64)),
        "fp16",
        CancelToken(),
    )
    assert desc.artifact_bytes == fake_plan
    assert desc.artifact_format == "trt_plan"
    assert desc.precision == "fp16"


def test_tensorrt_backend_compile_64x64x64_onnx_gemm(tmp_path: Path):
    """Compile a trivial 64x64x64 ONNX GEMM end-to-end.

    Done criterion (briefs/forge-1_backends.md): "TensorRTBackend.compile()
    produces a valid `.plan` for a trivial ONNX." Requires both `onnx` and
    `tensorrt` to be importable; skips cleanly otherwise.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("tensorrt")

    from acceleration.tensorrt_engine import TensorRTBackend
    from acceleration.backend import CancelToken, OnnxPath, ShapeRange

    import numpy as np
    import onnx
    from onnx import helper, TensorProto

    # ---- Build a 64x64x64 GEMM ONNX in tmp_path -----------------------
    M, N, K = 64, 64, 64
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    B_init = helper.make_tensor(
        "B", TensorProto.FLOAT, [K, N],
        np.random.RandomState(0).randn(K, N).astype(np.float32).flatten().tolist(),
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "trivial_gemm", [A], [Y], [B_init])
    model = helper.make_model(
        graph, producer_name="forge1_smoke",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx_file = tmp_path / "trivial_gemm.onnx"
    onnx.save(model, str(onnx_file))

    # ---- Compile via TensorRTBackend ----------------------------------
    be = TensorRTBackend()
    assert be.enabled is True, "tensorrt importable but backend not enabled"

    desc = be.compile(
        OnnxPath(onnx_file),
        ShapeRange.static((M, K)),
        "fp16",
        CancelToken(),
    )

    assert desc.backend_name == "tensorrt"
    assert desc.artifact_format == "trt_plan"
    assert desc.precision == "fp16"
    assert desc.size_bytes > 0
    # A valid TRT plan starts with the "tffl" / version magic; we just
    # assert the plan is non-trivial and round-trips through load().
    plan_path = tmp_path / "trivial_gemm.plan"
    plan_path.write_bytes(desc.artifact_bytes)
    loaded = be.load(plan_path)
    assert loaded.backend_name == "tensorrt"
    assert loaded.runner is not None


# ---------------------------------------------------------------------------
# 4. SageAttentionV2Backend
# ---------------------------------------------------------------------------

def test_sage_backend_constructs():
    from acceleration.sage_attention import SageAttentionV2Backend
    be = SageAttentionV2Backend()
    # Constructor must always succeed; enabled flag reflects availability.
    assert isinstance(be.enabled, bool)
    qd = be.estimated_quality_delta()
    assert 0.0 <= qd <= 1.0


def test_sage_backend_applies_to_attention_only():
    from acceleration.sage_attention import SageAttentionV2Backend
    be = SageAttentionV2Backend()
    if not be.enabled:
        pytest.skip("sageattention not installed")
    assert be.applies_to("attention", (1, 16, 1024, 64), "fp16") is True
    assert be.applies_to("self_attention", (1, 16, 1024, 64), "fp16") is True
    assert be.applies_to("gemm", (64, 64), "fp16") is False
    assert be.applies_to("attention", (16,), "fp16") is False  # rank too low


def test_sage_backend_compile_is_noop():
    from acceleration.sage_attention import SageAttentionV2Backend
    from acceleration.backend import CancelToken, OnnxPath, ShapeRange

    be = SageAttentionV2Backend()
    if not be.enabled:
        pytest.skip("sageattention not installed")

    desc = be.compile(
        OnnxPath(Path("ignored.onnx")),
        ShapeRange.static((1, 16, 1024, 64)),
        "fp16",
        CancelToken(),
    )
    assert desc.artifact_bytes == b""
    assert desc.artifact_format == "noop"
    assert desc.compile_seconds == 0.0


def test_sage_backend_monkeypatch_toggles_sdpa():
    """Enter context: SDPA replaced. Exit: original restored."""
    pytest.importorskip("torch")
    from acceleration.sage_attention import SageAttentionV2Backend
    be = SageAttentionV2Backend()
    if not be.enabled:
        pytest.skip("sageattention not installed")

    import torch.nn.functional as F  # type: ignore
    original = F.scaled_dot_product_attention
    engine = be.load(Path("ignored.bin"))

    # Before enter: original.
    assert F.scaled_dot_product_attention is original

    with engine.runner as ctx:
        assert ctx.is_active is True
        # Inside: patched.
        assert F.scaled_dot_product_attention is not original

    # After exit: restored.
    assert F.scaled_dot_product_attention is original


# ---------------------------------------------------------------------------
# 5. CutlassFP8GemmBackend
# ---------------------------------------------------------------------------

def test_cutlass_fp8_gemm_backend_phase_a_disabled():
    """Phase A must register the class with enabled=False by default."""
    from acceleration.cutlass_integration import CutlassFP8GemmBackend
    be = CutlassFP8GemmBackend()
    assert be.enabled is False
    # Class is still importable / introspectable.
    assert be.name == "cutlass_fp8_gemm"
    assert be.version.startswith("0.1.0-phaseA")


def test_cutlass_fp8_gemm_applies_to_correct_shapes():
    from acceleration.cutlass_integration import CutlassFP8GemmBackend
    be = CutlassFP8GemmBackend()
    # FP8 + GEMM + aligned dims = True.
    assert be.applies_to("gemm", (1024, 1024, 1024), "fp8") is True
    assert be.applies_to("gemm", (1024, 1024, 1024), "fp8_e4m3") is True
    assert be.applies_to("gemm", (1024, 1024, 1024), "fp8_e5m2") is True
    # Wrong precision.
    assert be.applies_to("gemm", (1024, 1024, 1024), "fp16") is False
    # Wrong kind.
    assert be.applies_to("conv", (1024, 1024, 1024), "fp8") is False
    # Wrong rank (must be 3 for M,N,K).
    assert be.applies_to("gemm", (1024, 1024), "fp8") is False
    # Misaligned dim (1023 not multiple of 8).
    assert be.applies_to("gemm", (1023, 1024, 1024), "fp8") is False
    # Below min dim.
    assert be.applies_to("gemm", (8, 1024, 1024), "fp8") is False


def test_cutlass_fp8_gemm_compile_raises_when_disabled():
    from acceleration.cutlass_integration import CutlassFP8GemmBackend
    from acceleration.backend import (
        CancelToken,
        OnnxPath,
        ShapeRange,
    )
    be = CutlassFP8GemmBackend()
    assert be.enabled is False
    with pytest.raises(RuntimeError, match="disabled in Phase A"):
        be.compile(
            OnnxPath(Path("ignored.onnx")),
            ShapeRange.static((1024, 1024, 1024)),
            "fp8",
            CancelToken(),
        )


def test_cutlass_fp8_gemm_priors():
    from acceleration.cutlass_integration import CutlassFP8GemmBackend
    be = CutlassFP8GemmBackend()
    assert be.estimated_compile_seconds("gemm", (1024, 1024, 1024)) > 0
    assert be.estimated_speedup("gemm", (1024, 1024, 1024)) >= 1.0
    qd = be.estimated_quality_delta()
    assert 0.0 <= qd <= 1.0


# ---------------------------------------------------------------------------
# 6. Workload-neutrality: no forbidden imports
# ---------------------------------------------------------------------------

FORBIDDEN_PREFIXES = (
    "core.experts",
    "core.engine_cache",
    "core.background_compiler",
)

@pytest.mark.parametrize(
    "module_name",
    [
        "acceleration.backend",
        "acceleration.tensorrt_engine",
        "acceleration.sage_attention",
        "acceleration.cutlass_integration",
    ],
)
def test_no_forbidden_imports(module_name: str):
    mod = importlib.import_module(module_name)
    # Walk the source for raw `from core.X` / `import core.X` strings.
    src = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_PREFIXES:
        assert f"from {forbidden}" not in src, (
            f"{module_name} imports from forbidden {forbidden}"
        )
        assert f"import {forbidden}" not in src, (
            f"{module_name} imports forbidden {forbidden}"
        )
