"""V11 — tests for the profiler classifier (M2) and the decide step (M3).

Pure logic on synthetic profiles; no GPU required.
"""
from __future__ import annotations

from v9.hwmodel import Bound, PrecisionPeak, Roofline
from v9.profiler import (
    KernelProfile,
    WorkloadProfile,
    _op_bytes,
    classify_dominant,
    classify_workload,
)
from v9.control import recommend_from_bound
from v9.recommend import Action


def _rl() -> Roofline:
    rl = Roofline("t", 89, 128, 3105, 2745, 10501, 384, 1008.0, 900.0)
    rl.peaks["fp16"] = PrecisionPeak("fp16", 202.0, 160.0, 200.0)
    rl.peaks["fp8"] = PrecisionPeak("fp8", 404.0, 330.0, 401.0)
    return rl


def _wp(name, cuda_s, ach) -> WorkloadProfile:
    wp = WorkloadProfile(precision="fp16", measured=True)
    wp.kernels = [KernelProfile(name, cuda_time_s=cuda_s, flops=2e12,
                                achieved_tflops=ach)]
    return wp


def test_classify_dominant_compute():
    _, b, _ = classify_dominant(_rl(), _wp("aten::mm", 0.007, 150.0))  # 94% of 160
    assert b is Bound.COMPUTE


def test_classify_dominant_memory():
    _, b, _ = classify_dominant(_rl(), _wp("aten::mm", 0.007, 80.0))   # 50% of 160
    assert b is Bound.MEMORY


def test_classify_dominant_latency_short():
    _, b, _ = classify_dominant(_rl(), _wp("tiny", 1e-6, 1.0))
    assert b is Bound.LATENCY


def test_classify_dominant_clamped_no_faster_than_light():
    _, b, why = classify_dominant(_rl(), _wp("aten::mm", 0.007, 175.0))  # > 160
    assert b is Bound.COMPUTE
    assert "100%" in why and "175" not in why.split("=")[-1]  # clamped to <=100%


def test_knife_edge_is_balanced_not_flipping():
    """REGRESSION (crit 1): a near-peak GEMM at ~70% must land in the BALANCED
    hysteresis band (0.60-0.80), never flip MEMORY<->COMPUTE on run-to-run noise."""
    _, b, _ = classify_dominant(_rl(), _wp("aten::mm", 0.007, 112.0))  # 112/160 = 70%
    assert b is Bound.BALANCED


def test_wrong_kernel_minority_matmul_not_trusted():
    """REGRESSION (crit 2): a 2%-runtime matmul must NOT classify a 98%-elementwise
    workload. The time-dominant (no-flops) kernel wins, honestly hedged."""
    wp = WorkloadProfile(precision="fp16", measured=True)
    wp.kernels = [
        KernelProfile("aten::add", cuda_time_s=0.050, flops=0.0, achieved_tflops=None),
        KernelProfile("aten::mm", cuda_time_s=0.001, flops=2e12, achieved_tflops=165.0),
    ]
    wp.total_cuda_s = 0.051
    kp, b, why = classify_dominant(_rl(), wp)
    assert kp.name == "aten::add"        # NOT the minority matmul
    assert b is not Bound.COMPUTE        # key guard: never borrow the matmul's verdict
    assert "FLOP estimate" in why        # honest: inconclusive without flops/bytes


def test_implausible_fraction_is_balanced_suspect():
    """REGRESSION (clamp major): achieved >>ceiling is a data fault, not COMPUTE."""
    _, b, why = classify_dominant(_rl(), _wp("aten::conv", 0.007, 200.0))  # 125% of 160
    assert b is Bound.BALANCED
    assert "suspect" in why


def test_op_bytes_matmul_and_elementwise():
    # mm: read both operands + write output
    assert _op_bytes("aten::mm", [[8192, 8192], [8192, 8192]], 2) == 3 * 8192 * 8192 * 2
    # addmm: bias + mat1 + mat2 -> read operands + write output
    assert _op_bytes("aten::addmm", [[8192], [8192, 8192], [8192, 8192]], 2) \
        == 3 * 8192 * 8192 * 2
    # elementwise: read 2 inputs + write 1 output
    assert _op_bytes("aten::mul", [[64], [64]], 2) == (64 + 64 + 64) * 2
    # unparseable -> None
    assert _op_bytes("aten::layer_norm", [], 2) is None


def test_classify_workload_compute_via_oi():
    """Mile-2b: a square GEMM has OI >> ridge -> COMPUTE by true intensity."""
    rl = _rl()
    wp = WorkloadProfile(precision="fp16", measured=True)
    wp.kernels = [KernelProfile("aten::mm", cuda_time_s=0.007, flops=2 * 8192**3,
                                achieved_tflops=157.0,
                                shapes=[[8192, 8192], [8192, 8192]])]
    wp.total_cuda_s = 0.007
    _, res = classify_workload(rl, wp)
    assert res.bound is Bound.COMPUTE
    assert res.oi and res.oi > 1000          # well above the ridge
    assert res.bytes_source == "estimated"


def test_classify_workload_memory_via_oi_with_bandwidth_confirm():
    """A streaming elementwise has OI << ridge AND saturates bandwidth -> MEMORY."""
    rl = _rl()
    n = 64 * 1024 * 1024
    wp = WorkloadProfile(precision="fp16", measured=True)
    wp.kernels = [KernelProfile("aten::mul", cuda_time_s=0.0005, flops=n,
                                achieved_tflops=0.7, shapes=[[n], [n]])]
    wp.total_cuda_s = 0.0005
    _, res = classify_workload(rl, wp)
    assert res.bound is Bound.MEMORY
    assert res.oi is not None and res.oi < 1
    assert res.achieved_bw_frac and res.achieved_bw_frac > 0.5  # bandwidth confirms


def test_recommend_from_bound_covers_all():
    rl = _rl()
    expect = {Bound.MEMORY: Action.ENABLE_BACKEND,
              Bound.COMPUTE: Action.ENABLE_BACKEND,
              Bound.LATENCY: Action.ENABLE_BACKEND,
              Bound.BALANCED: Action.NOTIFY}
    for bound, act in expect.items():
        recs = recommend_from_bound(bound, "kernel", "fp16", rl)
        assert recs and recs[0].action is act, bound


def test_memory_bound_recommends_quantization_when_fp8_available():
    recs = recommend_from_bound(Bound.MEMORY, "k", "fp16", _rl())
    # memory-bound at fp16 with fp8 available -> a precision/quantize lever appears
    assert any("quantize" in r.target or "precision" in str(r.parameters)
               for r in recs)
