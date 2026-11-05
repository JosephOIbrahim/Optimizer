"""V9 — on-device FP8 GEMM probe (run as a subprocess for crash isolation).

Loads a compiled CUTLASS FP8 GEMM shim (.dll/.so), runs it against torch
float8 device tensors, checks correctness vs a torch reference, and times it
against a torch fp16 matmul baseline at the same shape. Prints a single JSON
line to stdout.

Run in a SEPARATE process (see benchmark._benchmark_cutlass) so that a bad
kernel launch — which can hard-crash the CUDA context — cannot take down the
benchmark itself. Any failure is reported honestly as JSON; we never fake a
speedup.

Usage:
    python -m v9.accel._fp8_gemm_probe <shim_path> <M> <N> <K> <iters>
"""
from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys
import time


def _emit(obj: dict) -> None:
    print(json.dumps(obj))


def main() -> int:
    if len(sys.argv) < 6:
        _emit({"ok": False, "reason": "usage: shim M N K iters"})
        return 2
    shim_path = sys.argv[1]
    m, n, k, iters = (int(sys.argv[2]), int(sys.argv[3]),
                      int(sys.argv[4]), int(sys.argv[5]))

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _emit({"ok": False, "reason": f"torch import failed: {exc}"})
        return 0
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        _emit({"ok": False, "reason": "no CUDA device"})
        return 0
    f8 = getattr(torch, "float8_e4m3fn", None)
    if f8 is None:
        _emit({"ok": False, "reason": "torch lacks float8_e4m3fn"})
        return 0

    # Make the CUDA runtime DLLs the shim links against discoverable (Windows).
    if os.name == "nt":
        for var in ("CUDA_PATH", "CUDA_PATH_V13_2"):
            base = os.environ.get(var)
            if base and os.path.isdir(os.path.join(base, "bin")):
                try:
                    os.add_dll_directory(os.path.join(base, "bin"))
                except OSError:
                    pass

    try:
        dll = ctypes.CDLL(shim_path)
        fn = dll.cutlass_fp8_gemm
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int,
                       ctypes.c_float, ctypes.c_float]
    except Exception as exc:  # noqa: BLE001
        _emit({"ok": False, "reason": f"shim load failed: {exc}"})
        return 0

    dev = "cuda"
    # Shim contract: A row-major (M,K) ld=K; B col-major (K,N) ld=K stored as
    # (N,K) row-major; C row-major (M,N). So ref = A @ B = a_f @ b_f.T .
    a = (torch.randn(m, k, device=dev) * 0.25).to(f8)
    b = (torch.randn(n, k, device=dev) * 0.25).to(f8)
    c = torch.zeros(m, n, device=dev, dtype=torch.float32)
    torch.cuda.synchronize()

    rc = int(fn(a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k, 1.0, 0.0))
    torch.cuda.synchronize()
    if rc != 0:
        _emit({"ok": False, "rc": rc,
               "reason": ("kernel returned non-zero (classic device::Gemm FP8 "
                          "config may not launch on this arch; needs the 3.x "
                          "CollectiveBuilder path)")})
        return 0

    # correctness vs torch reference (FP8 is low-precision; loose tolerance)
    ref = a.float() @ b.float().t()
    denom = ref.abs().max().item() + 1e-6
    rel_err = (c - ref).abs().max().item() / denom

    # time the FP8 kernel
    fp8_samples = []
    for _ in range(max(1, iters)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k, 1.0, 0.0)
        torch.cuda.synchronize()
        fp8_samples.append(time.perf_counter() - t0)
    fp8_ms = statistics.median(fp8_samples) * 1000.0

    # torch fp16 baseline at the same shape
    af = torch.randn(m, k, device=dev, dtype=torch.float16)
    bf = torch.randn(k, n, device=dev, dtype=torch.float16)
    for _ in range(3):
        (af @ bf).sum()
    torch.cuda.synchronize()
    fp16_samples = []
    for _ in range(max(1, iters)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = af @ bf
        torch.cuda.synchronize()
        fp16_samples.append(time.perf_counter() - t0)
    fp16_ms = statistics.median(fp16_samples) * 1000.0

    _emit({"ok": True, "rc": 0, "M": m, "N": n, "K": k,
           "rel_err": rel_err, "fp8_ms": fp8_ms, "fp16_ms": fp16_ms,
           "speedup": (fp16_ms / fp8_ms) if fp8_ms > 0 else 0.0,
           "device": torch.cuda.get_device_name(0)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
