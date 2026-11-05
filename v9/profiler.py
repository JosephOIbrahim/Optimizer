"""V11 — profiler.py: workload observability (the OBSERVE step of the control loop).

The roofline (hwmodel.py) is the ceiling; this measures where a real workload
actually sits against it. Wraps the PyTorch profiler (Kineto -> CUPTI, shipped in
torch) to produce, per hot GPU kernel, the achieved compute rate — which, placed
against the precision-matched roofline peak, names the bottleneck.

Honest scope (and its one real limitation): the PyTorch profiler reliably gives
per-op CUDA time and a FLOP estimate for the heavy ops (matmul/conv) via
`with_flops`. It does NOT give measured DRAM traffic — that needs Nsight
Compute / CUPTI hardware counters (Mile-2b). So classification here uses the
*achieved-fraction* path (how close the op runs to the compute ceiling), which is
robust with profiler data and sufficient for the dominant signal: a big GEMM/conv
that reaches a high fraction of the compute peak is compute-bound; one that does
not (the common diffusion/VFX case) is memory- or latency-bound. Operational-
intensity (OI) classification is offered when algorithmic bytes are estimable
from op shapes, clearly labelled as an upper-bound estimate.

Degrade-safe: no torch/profiler -> returns an empty WorkloadProfile and says so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from v9.hwmodel import Bound, Roofline

log = logging.getLogger("v9.profiler")


@dataclass
class KernelProfile:
    name: str
    cuda_time_s: float
    count: int = 0
    flops: float = 0.0                       # profiler estimate (per call); 0 if unknown
    achieved_tflops: Optional[float] = None  # flops / cuda_time
    input_shapes: str = ""                    # truncated, for display
    shapes: Any = None                        # raw input_shapes list, for OI (Mile-2b)

    @property
    def has_flops(self) -> bool:
        return self.flops > 0 and self.achieved_tflops is not None


# operand-byte size per precision (for analytical DRAM-byte / OI estimates)
_DSIZE = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}


@dataclass
class WorkloadProfile:
    kernels: list[KernelProfile] = field(default_factory=list)
    total_cuda_s: float = 0.0
    precision: str = "fp16"
    measured: bool = False
    note: str = ""

    @property
    def dominant(self) -> Optional[KernelProfile]:
        """The kernel that owns the most GPU time (the thing to optimize first)."""
        return max(self.kernels, key=lambda k: k.cuda_time_s, default=None)

    @property
    def dominant_with_flops(self) -> Optional[KernelProfile]:
        """The most time-consuming kernel that has a FLOP estimate (so it can be
        placed on the compute roofline)."""
        flopped = [k for k in self.kernels if k.has_flops]
        return max(flopped, key=lambda k: k.cuda_time_s, default=None)


def _probe_torch():
    try:
        import torch  # type: ignore
        from torch.profiler import profile, ProfilerActivity  # type: ignore
        return torch, profile, ProfilerActivity
    except Exception:
        return None, None, None


def profile_callable(fn: Callable[[], Any], *, precision: str = "fp16",
                     warmup: int = 5, active: int = 10,
                     top_k: int = 12) -> WorkloadProfile:
    """Profile `fn` (one forward / one step) and return the hot GPU kernels with
    achieved compute rates. `fn` is called `warmup`+`active` times."""
    torch, profile, ProfilerActivity = _probe_torch()
    if torch is None:
        return WorkloadProfile(note="torch.profiler unavailable", measured=False)
    if not torch.cuda.is_available():
        return WorkloadProfile(note="no CUDA device", measured=False)

    try:
        for _ in range(max(0, warmup)):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True, record_shapes=True) as prof:
            for _ in range(max(1, active)):
                fn()
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - profiling must never crash the loop
        return WorkloadProfile(note=f"profiling failed: {exc}", measured=False)

    wp = WorkloadProfile(precision=precision, measured=True)
    try:
        # group_by_input_shape=True is REQUIRED to populate input_shapes (needed
        # for the analytical DRAM-byte / operational-intensity estimate, Mile-2b).
        events = prof.key_averages(group_by_input_shape=True)
        # CUDA self-time per op key; FunctionEventAvg.self_device_time_total is us
        rows = []
        for e in events:
            cuda_us = _cuda_self_us(e)
            if cuda_us <= 0:
                continue
            key = str(getattr(e, "key", "?"))
            # keep only aten:: ops — they carry the device time AND the flops/
            # shapes; the raw device-kernel rows (e.g. ampere_*gemm) duplicate the
            # same time and would double-count the total.
            if not key.startswith("aten::"):
                continue
            flops = float(getattr(e, "flops", 0) or 0)
            cuda_s = cuda_us / 1e6 / max(1, active)   # per-invocation
            flops_per = flops / max(1, active)
            ach = (flops_per / cuda_s / 1e12) if (flops_per > 0 and cuda_s > 0) else None
            raw_shapes = getattr(e, "input_shapes", None)
            rows.append(KernelProfile(
                name=key[:48], cuda_time_s=cuda_s,
                count=int(getattr(e, "count", 0) or 0),
                flops=flops_per, achieved_tflops=ach, shapes=raw_shapes,
                input_shapes=str(raw_shapes)[:40]))
        rows.sort(key=lambda k: k.cuda_time_s, reverse=True)
        wp.kernels = rows[:top_k]
        wp.total_cuda_s = sum(k.cuda_time_s for k in rows)
    except Exception as exc:  # noqa: BLE001
        wp.note = f"profile parse degraded: {exc}"
    return wp


def _cuda_self_us(e: Any) -> float:
    """Self CUDA/device time in microseconds across torch versions."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v:
            return float(v)
    return 0.0


def classify_dominant(roofline: Roofline, wp: WorkloadProfile
                      ) -> tuple[Optional[KernelProfile], Bound, str]:
    """Name the bottleneck of the workload's dominant kernel.

    Honest by construction (hardened by the M2-M4 adversarial pass):
      * the FLOP-bearing kernel is trusted ONLY if it owns a real share (>=40%)
        of GPU time; otherwise the time-dominant kernel is classified — so a
        2%-of-runtime matmul never labels a 98%-elementwise workload COMPUTE;
      * a hysteresis BALANCED band (0.60-0.80 of the compute ceiling) keeps a
        near-peak GEMM from flipping MEMORY<->COMPUTE on run-to-run noise;
      * with no measured DRAM bytes (Mile-2b), MEMORY/LATENCY are reported as the
        LIKELY bound with an explicit "needs ncu to confirm", never asserted;
      * an achieved fraction far above the ceiling is a data-quality fault (bad
        FLOP estimate / wrong precision key), not a confident COMPUTE.
    The compute ceiling is the SHARED hwmodel.saturation_ceiling policy, so this
    layer and hwmodel.classify can never disagree on "% of peak".
    """
    from v9.hwmodel import saturation_ceiling

    if not wp.kernels:
        return None, Bound.BALANCED, "no kernels profiled"
    total = wp.total_cuda_s or sum(k.cuda_time_s for k in wp.kernels) or 1.0

    kp_flops = wp.dominant_with_flops
    if kp_flops is not None and kp_flops.cuda_time_s >= 0.40 * total:
        kp = kp_flops                     # trust it: it dominates GPU time
    else:
        kp = wp.dominant
        if kp is None:
            return None, Bound.BALANCED, "no kernels profiled"
        if not kp.has_flops:
            if kp.cuda_time_s < 5e-6:
                return kp, Bound.LATENCY, f"{kp.name}: {kp.cuda_time_s*1e6:.1f}us (launch-bound)"
            share = kp.cuda_time_s / total
            # no FLOP estimate AND no bytes -> we genuinely cannot place it on the
            # roofline: it could be a memory-bound elementwise/norm OR a
            # compute-bound fused attention. Honest = inconclusive; don't assert.
            return kp, Bound.BALANCED, (
                f"{kp.name}: dominant ({share*100:.0f}% of GPU time) but no FLOP "
                f"estimate -> can't place on roofline; inconclusive (needs ncu "
                f"dram__bytes + flops to resolve)")

    peak_obj = roofline.peaks.get(wp.precision)
    if kp.achieved_tflops is None or peak_obj is None or not peak_obj.theoretical:
        if kp.cuda_time_s < 5e-6:
            return kp, Bound.LATENCY, f"{kp.name}: {kp.cuda_time_s*1e6:.1f}us (launch-bound)"
        return kp, Bound.BALANCED, f"{kp.name}: no compute-rate signal; need ncu (Mile-2b)"

    peak = saturation_ceiling(peak_obj)
    raw = (kp.achieved_tflops / peak) if peak else 0.0
    if raw > 1.10:                        # impossible -> data fault, not COMPUTE
        return kp, Bound.BALANCED, (
            f"{kp.name}: {kp.achieved_tflops:.0f} TF = {raw*100:.0f}% > ceiling "
            f"-> FLOP estimate or precision key suspect; not classifying")
    frac = min(1.0, raw)

    if kp.cuda_time_s < 5e-6:
        return kp, Bound.LATENCY, (
            f"{kp.name}: {kp.cuda_time_s*1e6:.1f}us, {frac*100:.0f}% peak -> launch-bound")
    if frac >= 0.80:
        return kp, Bound.COMPUTE, (
            f"{kp.name}: {kp.achieved_tflops:.0f} TF = {frac*100:.0f}% of "
            f"{wp.precision} ceiling -> compute-bound")
    if frac >= 0.60:                      # noise band: don't flip the lever
        return kp, Bound.BALANCED, (
            f"{kp.name}: {frac*100:.0f}% of {wp.precision} ceiling -> near-peak but "
            f"ambiguous (no DRAM-byte measurement to separate compute vs memory)")
    # Below 60% of the compute ceiling with real (>=5us) duration: the kernel is
    # NOT compute-limited, so without byte evidence the likely bound is MEMORY
    # (bandwidth) — this is the common diffusion/streaming case. (A low-occupancy
    # kernel can also live here; ncu occupancy separates them — Mile-2b.) LATENCY
    # is reserved for genuinely tiny launch-bound kernels (the <5us gate above).
    return kp, Bound.MEMORY, (
        f"{kp.name}: {kp.achieved_tflops:.0f} TF = {frac*100:.0f}% of {wp.precision} "
        f"ceiling (below) -> likely memory/bandwidth-bound; could be occupancy-bound "
        f"(needs ncu dram__bytes + occupancy to confirm)")


def _op_bytes(name: str, shapes: Any, dsize: int) -> Optional[float]:
    """Analytical MINIMUM DRAM bytes for a known op from its input shapes:
    read both matmul operands + write the output, or read inputs + write output
    for elementwise. An ESTIMATE — ignores cache reuse (overstates OI for GEMMs,
    which is safe: a GEMM is so far above the ridge it stays compute-bound) and
    tiling spill. Exact for no-reuse streaming ops. None if shapes unusable.
    Labelled bytes_source='estimated' to the classifier (which widens its band).
    """
    try:
        n = (name or "").lower()
        sh = [list(s) for s in (shapes or [])
              if isinstance(s, (list, tuple)) and len(s) > 0]
        if not sh:
            return None

        def numel(s: list) -> int:
            r = 1
            for d in s:
                r *= int(d)
            return r

        # matmul family: parse STRICTLY; if shapes don't fit the contract, return
        # None (do NOT fall through to the generic formula, which would silently
        # mis-model a matmul). baddbmm/addmm operands are mat1=sh[1], mat2=sh[2]
        # (sh[0] is the bias) — check them BEFORE bmm/mm (whose operands are
        # sh[0], sh[1]); note "baddbmm" contains "bmm" so order matters.
        if "baddbmm" in n or "addmm" in n:
            if len(sh) >= 3 and len(sh[1]) >= 2 and len(sh[2]) >= 2:
                m, k = sh[1][-2], sh[1][-1]
                k2, nn = sh[2][-2], sh[2][-1]
                batch = 1
                for d in sh[1][:-2]:
                    batch *= int(d)
                return batch * (m * k + k2 * nn + m * nn) * dsize
            return None
        if "bmm" in n:
            if len(sh) >= 2 and len(sh[0]) == 3 and len(sh[1]) == 3:
                bb, m, k = sh[0]; _, k2, nn = sh[1]
                return bb * (m * k + k2 * nn + m * nn) * dsize
            return None
        if "mm" in n or "matmul" in n:
            if len(sh) >= 2 and len(sh[0]) == 2 and len(sh[1]) == 2:
                m, k = sh[0]; k2, nn = sh[1]
                return (m * k + k2 * nn + m * nn) * dsize
            return None
        # generic pointwise/elementwise: read every input + write one output
        out = max(numel(s) for s in sh)
        total_in = sum(numel(s) for s in sh)
        return (total_in + out) * dsize
    except Exception:  # noqa: BLE001 - estimate is best-effort
        return None


def classify_workload(roofline: Roofline, wp: WorkloadProfile):
    """Mile-2b: TRUE operational-intensity classification. Computes the dominant
    op's analytical DRAM bytes (estimated) and places it on the FULL roofline via
    hwmodel.classify: OI = flops/bytes vs the ridge is the BOUND axis (compute vs
    memory). The achieved compute/bandwidth fractions add the SATURATION/LATENCY
    axis (is the op actually busy or idle/launch-bound) — they are NOT an
    independent check on the bound: achieved_bw = bytes/time shares the same byte
    estimate as OI, so a wrong byte estimate moves both together. Real independent
    byte validation needs measured dram__bytes (ncu) — the bw axis only confirms
    'is this op streaming near peak GIVEN my byte estimate'. Falls back to the
    compute-axis classify_dominant when no op has usable flops+shapes. Returns
    (KernelProfile|None, ClassifyResult).
    """
    from v9.hwmodel import ClassifyResult
    from v9.hwmodel import classify as roofline_classify

    if not wp.kernels:
        return None, ClassifyResult(Bound.BALANCED, "no kernels profiled")
    total = wp.total_cuda_s or sum(k.cuda_time_s for k in wp.kernels) or 1.0
    dsize = _DSIZE.get(wp.precision, 2)

    cands = []
    for k in wp.kernels:
        if k.flops and k.shapes is not None:
            by = _op_bytes(k.name, k.shapes, dsize)
            if by:
                cands.append((k, by))
    if cands:
        kp, by = max(cands, key=lambda kb: kb[0].cuda_time_s)
        if kp.cuda_time_s >= 0.40 * total:          # time-share gate (no wrong-kernel)
            t = kp.cuda_time_s
            ach_tf = kp.achieved_tflops or (kp.flops / t / 1e12 if t else None)
            ach_bw = (by / t / 1e9) if t else None  # GB/s
            res = roofline_classify(
                roofline, flops=kp.flops, dram_bytes=by, precision=wp.precision,
                achieved_tflops=ach_tf, achieved_bw_gbps=ach_bw,
                duration_s=t, bytes_source="estimated")
            return kp, res

    # no usable flops+shapes op dominates -> compute-axis heuristic (hedged)
    kp, bound, reason = classify_dominant(roofline, wp)
    return kp, ClassifyResult(bound, reason, bytes_source="n/a")


if __name__ == "__main__":
    # tiny self-check: profile a matmul and classify it
    import torch
    from v9.config import load_config
    from v9.hwmodel import derive_roofline
    rl = derive_roofline(load_config().profile, measure=True, gemm_n=4096, iters=15)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    wp = profile_callable(lambda: torch.matmul(a, b), precision="fp16")
    kp, bound, why = classify_dominant(rl, wp)
    print(f"dominant bound: {bound.value} — {why}")
