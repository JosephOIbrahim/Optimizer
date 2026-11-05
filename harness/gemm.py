"""
gemm — 8192^3 matrix multiply at the precisions the GeForce rate-lock touches.

Same test V12's truth module ran on 2026-09-16 (fp16 165.26 TFLOPS = 50.0% of
the 330.3 ceiling). Run it before and after anything that claims to change
tensor throughput and the number answers for itself.

    E:\\OptimizerV12\\venv\\Scripts\\python.exe gemm.py --reps 7

Modes (all fp32 accumulate unless noted — that is the locked path):
    fp16      torch.float16 @ fp16, reduced-precision reduction OFF
    bf16      torch.bfloat16, reduced-precision reduction OFF
    tf32      fp32 tensors with TF32 matmul allowed
    fp32      plain CUDA-core fp32 (control: unaffected by the tensor lock)

Reports median TFLOPS per mode with the % of the run spent at the power cap.
A 4090 running fp16 GEMM near 450 W is power-capped, so a tensor unlock shows
up as less than 2x here; that is the lever interaction, not a measurement bug.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import common, throttle  # noqa: E402


def run_mode(torch, mode: str, n: int, iters: int, warm: int):
    dev = "cuda"
    if mode == "fp16":
        dt = torch.float16
    elif mode == "bf16":
        dt = torch.bfloat16
    else:
        dt = torch.float32
    torch.backends.cuda.matmul.allow_tf32 = (mode == "tf32")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    a = torch.randn(n, n, device=dev, dtype=dt)
    b = torch.randn(n, n, device=dev, dtype=dt)
    c = torch.empty(n, n, device=dev, dtype=dt)
    for _ in range(warm):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    snap_a = throttle.Snapshot.take()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    snap_b = throttle.Snapshot.take()
    tflops = 2.0 * n ** 3 * iters / t / 1e12
    del a, b, c
    torch.cuda.empty_cache()
    return tflops, throttle.delta(snap_a, snap_b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=150, help="matmuls per timed sample (~1 s at 8192^3; the throttle counters need >~0.5 s to register)")
    ap.add_argument("--reps", type=int, default=7, help="timed samples per mode (first discarded)")
    ap.add_argument("--modes", default="fp16,bf16,tf32,fp32")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not importable in this interpreter. Use E:\\OptimizerV12\\venv\\Scripts\\python.exe")
        return 2
    if not torch.cuda.is_available():
        print("torch sees no CUDA device")
        return 2
    if not args.skip_preflight and not common.preflight():
        return 1

    name = torch.cuda.get_device_name(0)
    print(f"\n{name}  torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"N={args.n}  {args.iters} matmuls/sample  {args.reps} samples/mode\n")
    results = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        samples, thr = [], []
        for i in range(args.reps):
            tf, d = run_mode(torch, mode, args.n, args.iters, warm=2 if i == 0 else 0)
            if i == 0:
                continue  # warm-up sample
            samples.append(tf)
            thr.append(d)
        s = common.summarize(samples)
        cap = sum(d.get("power_frac", 0.0) for d in thr) / max(len(thr), 1)
        print(f"{mode:5s}  median {s['median']:7.1f} TFLOPS  cv {s.get('cv', 0):.2%}  "
              f"at power cap {cap:.0%}  [{throttle.describe(thr[-1])}]")
        results[mode] = {"tflops": samples, "summary": s, "throttle": thr}

    if "fp16" in results:
        med = results["fp16"]["summary"]["median"]
        print(f"\nfp16 vs the 330.3 TFLOPS full-rate ceiling: {med / 330.3:.1%}"
              f"  ({'locked half-rate territory' if med < 200 else 'above the locked rate'})")

    rec = common.write_measurement("gemm", {
        "harness": "gemm", "label": args.label, "n": args.n, "iters": args.iters,
        "device": name, "torch": torch.__version__, "cuda": torch.version.cuda,
        "results": results,
    })
    print(f"\nsaved {rec}")
    print("# V12 has no gemm objective yet (vellum is fp8 solve_s). These samples are the\n"
          "# before/after record for tensor-rate claims; add a 'gemm' objective to\n"
          "# OptimizerV12/v12/objective.py (tflops, HIGHER_IS_BETTER) to gate on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
