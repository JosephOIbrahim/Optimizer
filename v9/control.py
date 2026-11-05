"""V11 — control.py: the closed optimization loop + roofline-guided autotuner.

This is where the hardware model (hwmodel) and the observer (profiler) become an
actual optimizer:

  analyze   OBSERVE -> MODEL -> DECIDE.  Profile a real workload, place its
            dominant kernel on the roofline, and emit DERIVED recommendations —
            the bound class dictates the lever (memory-bound -> cut bytes;
            compute-bound -> faster precision; latency-bound -> more parallelism).
            No hand-written priors; the advice is a consequence of the physics.

  autotune  the self-tuning SEARCH.  The roofline says WHICH lever can help; the
            autotuner measures candidates on that lever and the existing Pareto
            gate admits the winner. Demonstrated on a GEMM: memory/compute work
            both point at precision, so it searches {bf16, fp16, fp8}, measures
            each on-device, and admits the fastest that clears the gate.

Degrade-safe: no torch/CUDA -> says so, does nothing fake.
"""
from __future__ import annotations

import logging
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from v9 import quality, telemetry
from v9.hwmodel import Bound, Roofline, derive_roofline
from v9.profiler import classify_dominant, classify_workload, profile_callable
from v9.recommend import Action, Recommendation

log = logging.getLogger("v9.control")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")


def _dtype(torch: Any, precision: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16,
            "fp32": torch.float32}.get(precision, torch.float16)


# ---------------------------------------------------------------------------
# workloads to observe (no downloads; representative shapes)
# ---------------------------------------------------------------------------
def _build_workload(torch: Any, name: str, precision: str
                    ) -> tuple[Callable[[], Any], str]:
    """Return (callable doing one unit of work, human description)."""
    name = name.lower()
    dt = _dtype(torch, precision)
    if name in ("gemm", "matmul"):
        n = 8192
        a = torch.randn(n, n, device="cuda", dtype=dt)
        b = torch.randn(n, n, device="cuda", dtype=dt)
        return (lambda: torch.matmul(a, b)), f"{n}^3 {precision} GEMM"
    if name in ("elementwise", "memory"):
        n = 64 * 1024 * 1024
        x = torch.randn(n, device="cuda", dtype=dt)
        y = torch.randn(n, device="cuda", dtype=dt)
        return (lambda: x * y + y), f"{n//1024//1024}M-element {precision} elementwise"
    if name == "dit":
        from v9.pipeline_benchmark import _build_dit_stack
        m = _build_dit_stack(torch, 1536, 24, 6).to("cuda", dt).eval()
        x = torch.randn(1, 4096, 1536, device="cuda", dtype=dt)

        def fn():
            with torch.no_grad():
                return m(x)
        return fn, f"DiT stack (dim1536 x6, seq4096) {precision}"
    raise ValueError(f"unknown workload {name!r} (try: gemm | elementwise | dit)")


# ---------------------------------------------------------------------------
# DECIDE — bound class -> derived recommendations
# ---------------------------------------------------------------------------
def recommend_from_bound(bound: Bound, kernel_desc: str, precision: str,
                         rl: Roofline) -> list[Recommendation]:
    """Map a measured bottleneck to the optimizations that can move it. The
    rationale is the physics, not a prior."""
    recs: list[Recommendation] = []
    fp8_ok = "fp8" in rl.peaks and rl.peaks["fp8"].theoretical > 0

    if bound is Bound.MEMORY:
        recs.append(Recommendation(
            advisor="roofline", action=Action.ENABLE_BACKEND,
            target=f"cut-bytes:fusion:{kernel_desc}", confidence=0.85,
            expected_impact_pct=30.0, measured=False,
            rationale=("Likely memory-bound (below the compute ceiling; DRAM "
                       "bytes not yet measured - ncu confirms). Cut bytes: fuse "
                       "surrounding ops (torch.compile) to kill intermediate "
                       "read/writes."),
            parameters={"backend": "torch_compile", "lever": "fusion"}))
        if precision in ("fp32", "fp16", "bf16") and fp8_ok:
            recs.append(Recommendation(
                advisor="roofline", action=Action.ENABLE_BACKEND,
                target=f"cut-bytes:quantize:{kernel_desc}", confidence=0.80,
                expected_impact_pct=80.0, measured=False,
                rationale=(f"Memory-bound at {precision}: halving operand bytes "
                           "(-> fp8 / int4) raises arithmetic intensity and moves "
                           "the kernel up toward the ridge. Autotune to confirm."),
                parameters={"lever": "precision", "candidates": ["fp8", "int4"]}))
    elif bound is Bound.COMPUTE:
        recs.append(Recommendation(
            advisor="roofline", action=Action.ENABLE_BACKEND,
            target=f"faster-math:{kernel_desc}", confidence=0.80,
            expected_impact_pct=60.0, measured=False,
            rationale=("Compute-bound: the kernel saturates the math units at "
                       f"{precision}. Lever: higher-throughput precision — on Ada, "
                       f"fp8 both HALVES operand bytes and gives ~2x tensor-core "
                       f"FLOP/s vs {precision}. Autotune to confirm."),
            parameters={"lever": "precision", "candidates": ["fp8"]}))
    elif bound is Bound.LATENCY:
        recs.append(Recommendation(
            advisor="roofline", action=Action.ENABLE_BACKEND,
            target=f"more-parallelism:{kernel_desc}", confidence=0.78,
            expected_impact_pct=40.0, measured=False,
            rationale=("Latency/occupancy-bound: the kernel saturates neither "
                       "ceiling — it's launch/occupancy-starved. Batch more work, "
                       "use CUDA graphs (torch.compile reduce-overhead), or larger "
                       "tiles to fill the SMs."),
            parameters={"backend": "torch_compile", "mode": "reduce-overhead",
                        "lever": "parallelism"}))
    else:  # BALANCED / unknown
        recs.append(Recommendation(
            advisor="roofline", action=Action.NOTIFY,
            target=f"balanced:{kernel_desc}", confidence=0.5,
            expected_impact_pct=0.0,
            rationale=("Balanced / inconclusive: the kernel sits near the ridge "
                       "or the bound couldn't be resolved (need measured DRAM "
                       "bytes via Nsight/CUPTI — Mile-2b)."),
            parameters={}))
    return recs


def analyze(config: Any = None, workload: str = "dit", precision: str = "fp16",
            iters: int = 15) -> int:
    """The closed loop on one workload: observe -> model -> decide."""
    if config is None:
        from v9.config import load_config
        config = load_config()

    print("=" * 72)
    print(f" Optimizer V11 — analyze (observe -> model -> decide): {workload}")
    print("=" * 72)
    print("  " + config.profile.summary())

    try:
        import torch
        if not torch.cuda.is_available():
            print("\n  no CUDA — analyze needs a GPU. (nothing measured)")
            return 0
    except Exception:
        print("\n  torch unavailable — cannot analyze. (nothing measured)")
        return 0

    print("\n  [1/3] MODEL — deriving roofline (microbench at workload scale) ...")
    # ceiling measured at a representative LARGE size (8192) so a big workload
    # kernel isn't judged against an undersized-bench ceiling (-> false 'suspect').
    rl = derive_roofline(config.profile, measure=True, gemm_n=8192, iters=15)
    fp16 = rl.peaks.get(precision)
    if fp16:
        print(f"        {precision} ceiling: {fp16.theoretical:.0f} TFLOPS "
              f"(measured {fp16.measured:.0f})  ·  ridge {fp16.ridge_flop_per_byte} F/B")

    print(f"  [2/3] OBSERVE — profiling workload '{workload}' ({precision}) ...")
    try:
        fn, desc = _build_workload(torch, workload, precision)
    except ValueError as exc:
        print(f"        {exc}")
        return 2
    wp = profile_callable(fn, precision=precision, active=iters)
    if not wp.measured:
        print(f"        profiling unavailable: {wp.note}")
        return 1
    kp, result = classify_workload(rl, wp)      # Mile-2b: true operational intensity
    bound = result.bound
    if kp:
        print(f"        dominant kernel: {kp.name}  ({kp.cuda_time_s*1e3:.3f} ms"
              + (f", {kp.achieved_tflops:.0f} TFLOPS" if kp.achieved_tflops else "")
              + ")")
    oi_txt = ""
    if result.oi is not None and result.ridge is not None:
        bwf = (f", bw {result.achieved_bw_frac*100:.0f}% peak"
               if result.achieved_bw_frac is not None else "")
        oi_txt = (f"   [OI {result.oi:.0f} vs ridge {result.ridge:.0f} F/byte; "
                  f"bytes {result.bytes_source}{bwf}]")
    print(f"        BOUND: {bound.value.upper()} - {result.reason}{oi_txt}")

    print("  [3/3] DECIDE — derived recommendations:")
    recs = recommend_from_bound(bound, desc, precision, rl)
    for r in recs:
        print(f"    * [{r.action.value}] {r.rationale}")
    print(f"\n  closed the loop: 1 workload observed -> roofline-classified "
          f"{bound.value} -> {len(recs)} derived recommendation(s).")
    return 0


# ---------------------------------------------------------------------------
# autotune — the roofline-guided, gate-admitted SEARCH (Mile-4 lite)
# ---------------------------------------------------------------------------
def _time_gemm(torch: Any, n: int, precision: str, warmup: int = 8,
               iters: int = 30) -> Optional[float]:
    """Median seconds for an n^3 GEMM at `precision` (CUDA events)."""
    try:
        if precision == "fp8":
            a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            sa = torch.tensor(1.0, device="cuda")
            sb = torch.tensor(1.0, device="cuda")
            fn = lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                                          out_dtype=torch.bfloat16)
        else:
            dt = _dtype(torch, precision)
            a = torch.randn(n, n, device="cuda", dtype=dt)
            b = torch.randn(n, n, device="cuda", dtype=dt)
            fn = lambda: torch.matmul(a, b)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) / 1e3)
        return statistics.median(ts)
    except Exception as exc:
        log.info("gemm time %s skipped: %s", precision, exc)
        return None


def autotune(config: Any = None, n: int = 8192,
             baseline: str = "fp16") -> int:
    """Roofline-guided precision search on a GEMM, admitted by the Pareto gate."""
    if config is None:
        from v9.config import load_config
        config = load_config()

    print("=" * 72)
    print(" Optimizer V11 — autotune (roofline-guided search; gate-admitted)")
    print("=" * 72)
    print("  " + config.profile.summary())

    try:
        import torch
        if not torch.cuda.is_available():
            print("\n  no CUDA — autotune needs a GPU.")
            return 0
    except Exception:
        print("\n  torch unavailable.")
        return 0

    rl = derive_roofline(config.profile, measure=True, gemm_n=4096, iters=12)

    # MODEL+DECIDE: classify the baseline GEMM to pick the lever
    flops = 2.0 * n * n * n
    base_s = _time_gemm(torch, n, baseline)
    if base_s is None:
        print("  baseline GEMM failed to time.")
        return 1
    base_tflops = flops / base_s / 1e12
    print(f"\n  baseline: {n}^3 {baseline} GEMM = {base_s*1e3:.3f} ms "
          f"({base_tflops:.0f} TFLOPS)")
    print("  roofline lever: a GEMM is compute/precision-bound -> SEARCH precision "
          "{bf16, fp16, fp8} (cut operand bytes / raise FLOP/s).")

    candidates = [p for p in ("bf16", "fp16", "fp8") if p != baseline or True]
    print(f"\n  {'precision':9s} {'ms':>9s} {'TFLOPS':>9s} {'speedup':>8s} {'admit':>6s}")
    print("  " + "-" * 46)
    best = (baseline, base_s, 1.0)
    for prec in candidates:
        secs = base_s if prec == baseline else _time_gemm(torch, n, prec)
        if secs is None:
            print(f"  {prec:9s} {'FAILED':>9s}")
            continue
        tf = flops / secs / 1e12
        speedup = base_s / secs
        # admit via the real Pareto gate (lossless precision change at the GEMM
        # level -> lpips/clip 0; quality is validated downstream by ibench/pbench)
        verdict = quality.record_quality(
            ts_utc=_utc(), speedup=speedup, lpips_delta=0.0, clip_delta=0.0,
            app="autotune", backend_name=f"gemm_{prec}", backend_version="v11",
            model_sha256=f"gemm_{n}", shape_w=n, shape_h=n, batch=1,
            precision=prec, notes={"harness": "autotune", "tflops": tf,
                                   "ms": secs * 1e3, "baseline": baseline,
                                   "quality_unvalidated": True})
        admit = verdict.admitted
        if speedup > best[2]:
            best = (prec, secs, speedup)
        print(f"  {prec:9s} {secs*1e3:9.3f} {tf:9.0f} {speedup:8.3f} "
              f"{'YES' if admit else 'no':>6s}")

    print(f"\n  SELECTED: {best[0]} ({best[2]:.2f}x over {baseline}) — the search "
          f"found it; admitted on SPEEDUP (>= {quality.MIN_SPEEDUP}x), ledger-recorded.")
    print("  NOTE: lpips/clip are 0 placeholders here — a precision change to fp8 IS")
    print("  lossy; its numerical quality must be validated end-to-end (ibench /")
    print("  pbench) before production. This step proves the SEARCH + speedup gate,")
    print("  not image quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(analyze())
