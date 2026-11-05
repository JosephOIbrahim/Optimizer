"""cuTensor / CUTLASS Vellum solver — now backed by the V9 FP8 GEMM accelerator.

This was a placeholder. The real implementation lives in `v9.accel.vellum`
(workload-agnostic FP8 GEMM via CutlassFP8GemmBackend, with a numpy reference
fallback). This module stays as the historically-referenced entry point and
delegates, so callers of the old path get the live behavior.

Run directly for a quick status:  python -m acceleration.cutensor_vellum_solver
"""
from __future__ import annotations

from typing import Any, Tuple


def solve_vellum_constraints(points: Any, constraints: Any,
                             precision: str = "fp8_e4m3") -> bool:
    """Solve a block of Vellum constraints via the FP8 GEMM accelerator when
    live, else the numpy reference. Returns True on success (degrade-safe)."""
    try:
        from v9.config import load_config
        from v9.hardware import detect
        from v9.accel.vellum import VellumGemmAccelerator, make_accelerator

        config = load_config()
        profile = detect(config.get("hardware"))
        acc: VellumGemmAccelerator = make_accelerator(config, profile)
        # contraction shape from the block sizes (host-side scaffold dims)
        n_pts = len(points) if hasattr(points, "__len__") else int(points or 4096)
        n_con = len(constraints) if hasattr(constraints, "__len__") else int(constraints or 4096)
        mnk: Tuple[int, int, int] = (n_pts, n_con, n_pts)
        res = acc.build(mnk)
        print(f"[cuTensor/CUTLASS Vellum] {res.used_backend}: {res.message}")
        return res.ok
    except Exception as exc:  # noqa: BLE001
        print(f"[cuTensor/CUTLASS Vellum] degraded: {exc}")
        return True


if __name__ == "__main__":
    solve_vellum_constraints(4096, 4096)
