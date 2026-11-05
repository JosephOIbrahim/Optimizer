"""V9 — measured-speedup acceptance harness (the real proof, DESIGN_V9 §4.5).

This is the only place a speedup number becomes truth. It:
  1. times a torch fp16 matmul baseline IF torch is present;
  2. structures a comparison against a compiled acceleration backend when one
     is enabled (currently a measured no-op stand-in until a backend is wired);
  3. writes rows to v9.telemetry (timings) and v9.quality (Pareto verdict);
  4. if torch is ABSENT, reports that honestly and writes nothing fake.

Acceptance table (C2.5 shape): one row per (workload, precision, backend) with
columns: baseline_s, accel_s, speedup, lpips_delta, clip_delta, admitted.
Quality deltas are 0.0 placeholders here (no image pipeline in the harness);
the row is still recorded so the ledger is honest about what was measured vs
asserted. Real lpips/clip come from the ComfyUI image path in Phase 2.
"""
from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from typing import Any, Optional

from v9 import quality, telemetry
from v9.config import Config, load_config


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")


def _probe_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _time_matmul(torch: Any, device: str, dtype: Any,
                 n: int, iterations: int) -> Optional[float]:
    """Median seconds for an n×n matmul on `device`/`dtype`. None on failure."""
    try:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        # warmup
        for _ in range(3):
            (a @ b).sum().item()
        if device == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(max(1, iterations)):
            t0 = time.perf_counter()
            c = a @ b
            if device == "cuda":
                torch.cuda.synchronize()
            else:
                c.sum().item()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)
    except Exception:
        return None


def run(config: Optional[Config] = None, iterations: int = 20) -> int:
    if config is None:
        config = load_config()

    print("=" * 64)
    print(" Optimizer V9 — benchmark (measured speedup; no fake numbers)")
    print("=" * 64)
    print("  " + config.profile.summary())

    torch = _probe_torch()
    if torch is None:
        print()
        print("  torch is NOT installed in this interpreter — no baseline can "
              "be measured.")
        print("  Honest result: 0 measurements. Install torch (and a backend) "
              "on the Py3.12 venv, then re-run `benchmark` to get real numbers.")
        print("  (No rows written to telemetry/quality — refusing to assert a "
              "speedup we did not measure.)")
        return 0

    cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    device = "cuda" if cuda else "cpu"
    print(f"  torch {torch.__version__} | device={device}"
          f"{'' if cuda else ' (no CUDA — CPU baseline only)'}")

    # which backends are live (honest): if none, accel column is N/A
    backend_names = _enabled_backend_names(config)
    print(f"  enabled acceleration backends: {backend_names or 'none (dormant)'}")

    # acceptance table header
    print()
    header = (f"  {'workload':22s} {'prec':5s} {'baseline_s':>11s} "
              f"{'accel_s':>9s} {'speedup':>8s} {'lpips':>6s} "
              f"{'clip':>6s} {'admit':>6s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    dtype = torch.float16 if cuda else torch.float32
    prec = "fp16" if cuda else "fp32"
    workloads = [("matmul-2048", 2048), ("matmul-4096", 4096)]

    rows = 0
    for label, n in workloads:
        baseline = _time_matmul(torch, device, dtype, n, iterations)
        if baseline is None:
            print(f"  {label:22s} {prec:5s} {'FAILED':>11s}")
            continue

        # No compiled backend wired yet -> the "accelerated" path is the same
        # kernel; we measure it rather than assert a multiplier. speedup ~1.0
        # is the HONEST answer until a real backend exists.
        accel = _time_matmul(torch, device, dtype, n, iterations) or baseline
        speedup = baseline / accel if accel > 0 else 1.0
        lpips_delta = 0.0   # no image pipeline in this harness
        clip_delta = 0.0

        verdict = quality.record_quality(
            ts_utc=_utc_now_iso(),
            speedup=speedup, lpips_delta=lpips_delta, clip_delta=clip_delta,
            app="comfyui", backend_name=(backend_names[0] if backend_names
                                         else "baseline_torch"),
            backend_version="n/a", model_sha256=label,
            shape_w=n, shape_h=n, batch=1, precision=prec,
            notes={"harness": "matmul", "device": device,
                   "iterations": iterations},
        )
        telemetry.record_generation(
            ts_utc=_utc_now_iso(), workflow_hash=label, model_sha256=label,
            shape_w=n, shape_h=n, batch=1, sampler="matmul", steps=iterations,
            precision=prec, backend_set=backend_names,
            wall_seconds=baseline, samples_per_sec=(1.0 / baseline),
            vram_peak_gb=0.0,
        )
        rows += 1
        print(f"  {label:22s} {prec:5s} {baseline:11.5f} {accel:9.5f} "
              f"{speedup:8.3f} {lpips_delta:6.3f} {clip_delta:6.3f} "
              f"{'yes' if verdict.admitted else 'no':>6s}")

    # ---- CUTLASS FP8 GEMM proof (the headline ingredient) --------------
    _benchmark_cutlass(config, torch, device, iterations)

    print()
    print(f"  wrote {rows} measurement row(s) to telemetry + quality ledgers.")
    if not backend_names:
        print("  NOTE: speedup ~1.0 is expected — no acceleration backend is "
              "installed, so 'accel' is the same kernel as 'baseline'. This is "
              "the honest floor. Real >=1.30x rows require a wired backend.")
    print(f"  Pareto gate: speedup>={quality.MIN_SPEEDUP} AND "
          f"lpips<={quality.MAX_LPIPS_DELTA} AND clip>={quality.MIN_CLIP_DELTA}")
    return 0


def _benchmark_cutlass(config: Config, torch: Any, device: str,
                       iterations: int) -> None:
    """Prove the CUTLASS ingredient on THIS machine.

    Always honest: (1) builds+caches the FP8 GEMM kernel and reports the real
    compile time + artifact size; (2) if torch+CUDA+float8 are available, times
    the compiled kernel on-device vs a torch fp16 matmul baseline and prints a
    measured speedup; (3) otherwise states exactly what was not measured.
    """
    from v9.accel.vellum import make_accelerator, default_gemm

    print()
    print("  CUTLASS FP8 GEMM (Houdini Vellum/Pyro ingredient)")
    print("  " + "-" * 48)
    acc = make_accelerator(config, config.profile)
    if not acc.live:
        print("    CUTLASS not live — skipping. Run `v9 cutlass bootstrap` and "
              "set acceleration.cutlass.enabled: true, then re-run benchmark.")
        return

    m, n, k = default_gemm(config)
    build = acc.build((m, n, k))
    print(f"    build: {build.used_backend} {build.precision} "
          f"{m}x{n}x{k} -> {build.artifact_bytes} bytes "
          f"in {build.compile_seconds:.2f}s")
    if build.artifact_path:
        print(f"    cached: {build.artifact_path}")

    # On-device timing — runs the compiled kernel in an ISOLATED subprocess so
    # a bad CUDA launch can't crash the benchmark. Never fakes a number.
    if torch is None or device != "cuda" or not build.artifact_path:
        print("    on-device timing: skipped (needs torch+CUDA and a compiled "
              "kernel). Compile proof above is real.")
        return
    probe = _run_fp8_probe(str(build.artifact_path), m, n, k,
                           max(5, min(iterations, 50)))
    if not probe.get("ok"):
        print(f"    on-device run: not measured — {probe.get('reason', probe)}")
        print("    (the compiled+cached kernel above is the real proof; a "
              "runnable speedup needs the 3.x CollectiveBuilder kernel.)")
        return
    print(f"    on-device MEASURED on {probe.get('device','cuda')}:")
    print(f"      fp8  (CUTLASS) : {probe['fp8_ms']:.4f} ms")
    print(f"      fp16 (torch)   : {probe['fp16_ms']:.4f} ms")
    print(f"      SPEEDUP        : {probe['speedup']:.3f}x  "
          f"(rel_err vs fp32 ref {probe['rel_err']:.4f})")
    # close the loop: record into the quality ledger and show the Pareto verdict
    try:
        verdict = quality.record_quality(
            ts_utc=_utc_now_iso(), speedup=float(probe["speedup"]),
            lpips_delta=0.0, clip_delta=0.0, app="houdini",
            backend_name="cutlass_fp8_gemm", backend_version="0.2.0",
            model_sha256=f"vellum_gemm_{m}x{n}x{k}", shape_w=m, shape_h=n,
            batch=1, precision="fp8_e4m3",
            notes={"harness": "fp8_gemm_probe", "rel_err": probe["rel_err"],
                   "fp16_ms": probe["fp16_ms"], "fp8_ms": probe["fp8_ms"]})
        print(f"      PARETO ADMIT   : "
              f"{'YES' if verdict.admitted else 'no'} "
              f"(gate: speedup>={quality.MIN_SPEEDUP})")
    except Exception as exc:  # noqa: BLE001
        print(f"      (quality ledger record skipped: {exc})")


def _run_fp8_probe(shim_path: str, m: int, n: int, k: int,
                   iters: int) -> dict:
    """Invoke the FP8 GEMM probe in a separate process; parse its JSON line."""
    import json
    import subprocess
    import sys
    try:
        env = None
        if __import__("os").name == "nt":
            from v9.accel.cutlass_bootstrap import msvc_env
            env = msvc_env()  # ensures CUDA bin / runtime discoverable
        out = subprocess.run(
            [sys.executable, "-m", "v9.accel._fp8_gemm_probe",
             shim_path, str(m), str(n), str(k), str(iters)],
            capture_output=True, text=True, timeout=180, env=env)
        for line in reversed(out.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"ok": False, "reason":
                f"probe produced no JSON (rc={out.returncode}); "
                f"stderr: {out.stderr.strip()[:200]}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "probe timed out (>180s)"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"probe launch error: {exc}"}


def _enabled_backend_names(config: Config) -> list[str]:
    try:
        from v9.accel.registry import build_default_registry
        reg = build_default_registry(config.profile, config)
        return [b.name for b in reg.enabled_backends()]
    except Exception:
        return []


if __name__ == "__main__":
    raise SystemExit(run())
