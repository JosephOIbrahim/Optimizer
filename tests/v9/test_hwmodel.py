"""V11-Mile-1 — tests for the roofline hardware model.

classify() is pure logic (no GPU); derive_roofline is exercised degrade-safe
(measure=False). Covers the adversarial-verify regressions: ridge-absent must
NOT default to COMPUTE, and realized fractions must stay <= ~100%.
"""
from __future__ import annotations

from v9.hwmodel import (
    Bound,
    ClassifyResult,
    PrecisionPeak,
    Roofline,
    classify,
    derive_roofline,
)


def _synthetic_roofline() -> Roofline:
    """A roofline with known ceilings (4090-ish, theoretical @ max clock)."""
    rl = Roofline(
        gpu_name="test-4090", sm_arch=89, sm_count=128,
        max_sm_clock_mhz=3105, achieved_sm_clock_mhz=2745,
        mem_clock_mhz=10501, mem_bus_bits=384,
        dram_bw_theoretical_gbps=1008.0, dram_bw_measured_gbps=900.0,
    )
    rl.peaks["fp32"] = PrecisionPeak("fp32", 101.0, 90.0, 100.0)
    rl.peaks["fp16"] = PrecisionPeak("fp16", 202.0, 166.0, 200.0)
    rl.peaks["fp8"] = PrecisionPeak("fp8", 404.0, 335.0, 401.0)
    # int8: TOPS, deliberately no ridge
    rl.peaks["int8"] = PrecisionPeak("int8", 404.0, 166.0, None)
    return rl


def test_memory_bound_low_intensity():
    rl = _synthetic_roofline()
    r = classify(rl, flops=1e10, dram_bytes=1e9, precision="fp16")  # OI=10 << ridge 200
    assert isinstance(r, ClassifyResult)
    assert r.bound is Bound.MEMORY, r.reason


def test_compute_bound_high_intensity():
    rl = _synthetic_roofline()
    r = classify(rl, flops=1e12, dram_bytes=1e9, precision="fp16")  # OI=1000 >> ridge
    assert r.bound is Bound.COMPUTE, r.reason


def test_latency_bound_short_kernel():
    rl = _synthetic_roofline()
    r = classify(rl, flops=1e12, dram_bytes=1e6, precision="fp16", duration_s=1e-6)
    assert r.bound is Bound.LATENCY, r.reason


def test_latency_bound_low_occupancy():
    rl = _synthetic_roofline()
    r = classify(rl, flops=1e12, dram_bytes=1e9, precision="fp16", occupancy=0.10)
    assert r.bound is Bound.LATENCY, r.reason


def test_latency_demote_near_ridge_when_neither_ceiling_reached():
    rl = _synthetic_roofline()
    # OI 300 is NEAR the ridge (200); low achieved on both -> demote to LATENCY
    r = classify(rl, flops=3e11, dram_bytes=1e9, precision="fp16",
                 achieved_tflops=20.0, achieved_bw_gbps=50.0)
    assert r.bound is Bound.LATENCY, r.reason


def test_deep_oi_not_demoted_by_noisy_low_fraction():
    """REGRESSION (verify major 1/3): a kernel at OI >> ridge stays COMPUTE even
    if a single achieved-fraction sample is low (profiler self-time noise)."""
    rl = _synthetic_roofline()
    r = classify(rl, flops=1e12, dram_bytes=1e9, precision="fp16",   # OI 1000 >> 2*ridge
                 achieved_tflops=20.0, achieved_bw_gbps=50.0)
    assert r.bound is Bound.COMPUTE, r.reason


def test_data_fault_guard_on_oi_path():
    """REGRESSION (verify major 4): >1.1x ceiling on the OI path -> BALANCED."""
    rl = _synthetic_roofline()
    r = classify(rl, flops=1e12, dram_bytes=1e9, precision="fp16",
                 achieved_tflops=250.0)        # 250 / 166 measured = 1.5x > 1.10
    assert r.bound is Bound.BALANCED
    assert "suspect" in r.reason
    # headroom must be clamped at >= 0 (no negative/faster-than-light)
    if r.headroom_frac is not None:
        assert r.headroom_frac >= 0.0


def test_balanced_near_ridge():
    rl = _synthetic_roofline()
    # OI = 200, exactly the fp16 ridge -> BALANCED
    r = classify(rl, flops=2.0e11, dram_bytes=1e9, precision="fp16")
    assert r.bound is Bound.BALANCED, r.reason


def test_unknown_precision_is_safe():
    rl = _synthetic_roofline()
    r = classify(rl, flops=1e10, dram_bytes=1e9, precision="fp4")
    assert r.bound is Bound.BALANCED  # no ceiling -> safe, never crashes


def test_ridge_absent_does_not_default_to_compute():
    """REGRESSION (verify major): a precision with no ridge (DRAM BW undetected,
    or int8/TOPS) must return BALANCED, never a confident COMPUTE."""
    rl = _synthetic_roofline()
    # int8 has no ridge; a low-OI kernel must NOT be called COMPUTE
    r = classify(rl, flops=1e6, dram_bytes=1e9, precision="int8")
    assert r.bound is Bound.BALANCED, r.reason
    # and the same when the whole BW is undetected
    rl.dram_bw_theoretical_gbps = None
    rl.dram_bw_measured_gbps = None
    rl.peaks["fp16"].ridge_flop_per_byte = None
    r2 = classify(rl, flops=1e10, dram_bytes=1e9, precision="fp16")
    assert r2.bound is Bound.BALANCED, r2.reason


def test_structured_result_carries_numbers():
    rl = _synthetic_roofline()
    r = classify(rl, flops=1e12, dram_bytes=1e9, precision="fp16")
    assert r.oi == 1000.0
    assert r.ridge == 200.0
    assert r.ceiling_tflops == 202.0
    assert r.bw_basis == "measured"


def test_realized_fraction_never_exceeds_one():
    """REGRESSION (verify major): theoretical @ max clock >= measured @ achieved,
    so realized fraction must be <= 1.0 (no faster-than-light)."""
    rl = _synthetic_roofline()
    for p in rl.peaks.values():
        if p.realized_fraction is not None:
            assert p.realized_fraction <= 1.0, p.precision


def test_derive_roofline_degrade_safe():
    from v9.config import load_config
    rl = derive_roofline(load_config().profile, measure=False)
    assert isinstance(rl, Roofline)
    assert rl.sm_arch in (0, 80, 86, 89, 90)
    if "fp32" in rl.peaks and rl.dram_bw_theoretical_gbps:
        p = rl.peaks["fp32"]
        assert p.theoretical > 0
        assert p.ridge_flop_per_byte and p.ridge_flop_per_byte > 0


def test_ridge_is_peak_over_bandwidth():
    rl = _synthetic_roofline()
    p = rl.peaks["fp16"]
    expected = p.theoretical * 1e12 / (rl.dram_bw_theoretical_gbps * 1e9)
    assert abs(p.ridge_flop_per_byte - expected) / expected < 0.05
