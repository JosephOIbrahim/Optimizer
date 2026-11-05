"""V11 — hwmodel.py: the GPU roofline. The hardware model the optimizer was missing.

First principles: you cannot optimize what you do not model. This module derives
the machine's **speed-of-light** ceilings — peak compute (per precision) and peak
DRAM bandwidth — places a workload's arithmetic intensity against them, and
classifies the bottleneck (memory- / compute- / latency-bound). That bound class
is what tells the rest of the optimizer *which* optimization can possibly help:

  * memory-bound  -> cut bytes  (fusion, quantization)        <- most VFX/diffusion
  * compute-bound -> faster math (higher-throughput precision, tensor cores)
  * latency-bound -> more parallelism (CUDA graphs, bigger tiles, fewer launches)

DETECT, DON'T HARDCODE (the project's cardinal rule): the ceilings are DERIVED
from detected hardware (SM count, live clocks, bus width) and VALIDATED by
microbenchmark — never copied from a marketing spec sheet. The only constants
here are *architectural invariants* keyed by compute-capability (FP32 lanes/SM,
the dense tensor-core multiple), exactly like hardware.py's `_SM_BY_NAME` table.

Two clocks, kept distinct (this is the load-bearing correctness point):
  * THEORETICAL peak is pinned to the **max** SM clock — a stable, reproducible
    absolute speed-of-light that never moves run-to-run and that a measured
    kernel can never exceed (so realized fractions are always <= 100%).
  * The **achieved** SM clock (sampled live under load, ~2745 MHz on this 4090 —
    between the 870 idle and 3105 max) is reported as a separate datum, never as
    the ceiling. cuBLAS hitting ~88% of the max-clock peak reflects both the
    clock droop and ~95% kernel efficiency; that is honest, not a defect.

Hard-won correctness (validated on this 4090 by the V11-Mile-1 gather microbench):
  * Tensor peaks use the DENSE, FP32-ACCUMULATE path (cuBLAS default): FP16/BF16
    ~= 2x FP32, FP8 ~= 4x, INT8 ~= 4x (TOPS). The "330/660" box-art numbers are
    FP16-accumulate and/or 2:4-sparsity-doubled.
  * DRAM theoretical = bus/8 * mem_clock*2 (GDDR6X DDR) ~= 1008 GB/s; achievable
    streaming ceiling ~= 90% (measured 918 GB/s). Theoretical drives ridge math;
    achievable drives achieved-fraction comparisons.
  * `torch._scaled_mm` (FP8) needs B column-major (pass `b.t()`); CUDA-event
    timing only; >=512 MB buffers to clear the 72 MB L2 for the BW probe.
  * INT8 is reported in TOPS, not TFLOPS, and is excluded from the FLOP/byte
    ridge (dimensionally inconsistent); `torch._int_mm` under-reports (~46%) so
    its measured value is a conservative floor, not the realizable ceiling.

Degrade-safe: no torch/CUDA -> derive theoretical from clocks only, no microbench.
No pynvml -> fall back to nvidia-smi for clocks; if all fail, say so honestly. When
the DRAM-BW (hence ridge) is undetected, classify() returns BALANCED rather than
guessing — it never emits a confident wrong bound in a degraded environment.
"""
from __future__ import annotations

import logging
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

log = logging.getLogger("v9.hwmodel")

# --- architectural invariants per compute capability (NOT per-machine values;
# fixed by the chip design, like hardware.py:_SM_BY_NAME). tensor_mult is the
# DENSE, FP32-accumulate tensor-core peak as a multiple of the FP32-lane peak. --
_ARCH: dict[int, dict[str, Any]] = {
    89: {"fp32_lanes_per_sm": 128,                       # Ada Lovelace (RTX 40)
         "tensor_mult": {"fp16": 2.0, "bf16": 2.0, "fp8": 4.0, "int8": 4.0}},
    86: {"fp32_lanes_per_sm": 128,                       # Ampere consumer (RTX 30)
         "tensor_mult": {"fp16": 2.0, "bf16": 2.0, "int8": 4.0}},   # no fp8 HW
    80: {"fp32_lanes_per_sm": 64,                        # A100 (Ampere DC)
         "tensor_mult": {"fp16": 4.0, "bf16": 4.0, "int8": 8.0}},
    90: {"fp32_lanes_per_sm": 128,                       # Hopper (approx)
         "tensor_mult": {"fp16": 4.0, "bf16": 4.0, "fp8": 8.0, "int8": 8.0}},
}
# precisions reported in FLOPS (and thus eligible for a FLOP/byte ridge); int8 is
# reported in TOPS and is deliberately excluded from the ridge / OI comparison.
_FLOP_PRECISIONS = {"fp32", "fp16", "bf16", "fp8"}
# achievable streaming BW as a fraction of theoretical, when not measured
# (this 4090 measured ~0.91; use the mid-high of the 0.82-0.91 band).
_BW_ACHIEVABLE_FRACTION = 0.90


class Bound(Enum):
    MEMORY = "memory-bound"
    COMPUTE = "compute-bound"
    LATENCY = "latency-bound"
    BALANCED = "balanced"


@dataclass
class PrecisionPeak:
    precision: str
    theoretical: float                       # TFLOPS (TOPS for int8) @ MAX clock
    measured: Optional[float] = None         # microbench achievable (@ achieved clk)
    ridge_flop_per_byte: Optional[float] = None    # vs theoretical DRAM BW; None for int8

    @property
    def realized_fraction(self) -> Optional[float]:
        """measured / theoretical@max — always <= ~1.0 (max clock >= achieved)."""
        if self.measured and self.theoretical:
            return self.measured / self.theoretical
        return None


@dataclass
class Roofline:
    gpu_name: str
    sm_arch: int
    sm_count: int
    max_sm_clock_mhz: Optional[int]
    achieved_sm_clock_mhz: Optional[int]     # sampled under compute load (datum, not ceiling)
    mem_clock_mhz: Optional[int]
    mem_bus_bits: Optional[int]
    dram_bw_theoretical_gbps: Optional[float]
    dram_bw_measured_gbps: Optional[float]
    peaks: dict[str, PrecisionPeak] = field(default_factory=dict)
    measured: bool = False
    clock_for_peak_mhz: Optional[int] = None  # the MAX clock the theoretical peak uses
    notes: list[str] = field(default_factory=list)

    @property
    def dram_bw_achievable_gbps(self) -> Optional[float]:
        """The realistic streaming ceiling for achieved-fraction comparisons:
        the measured copy if we have it, else ~90% of theoretical."""
        if self.dram_bw_measured_gbps:
            return self.dram_bw_measured_gbps
        if self.dram_bw_theoretical_gbps:
            return self.dram_bw_theoretical_gbps * _BW_ACHIEVABLE_FRACTION
        return None

    @property
    def bw_basis(self) -> str:
        return "measured" if self.dram_bw_measured_gbps else (
            f"estimated-{_BW_ACHIEVABLE_FRACTION:.2f}*theoretical")


@dataclass
class ClassifyResult:
    """The structured contract the roofline-advisor (Mile 3) and autotuner
    (Mile 4) consume — numbers, not just a label."""
    bound: Bound
    reason: str
    oi: Optional[float] = None                    # operational intensity, FLOP/byte
    ridge: Optional[float] = None
    ceiling_tflops: Optional[float] = None        # precision compute ceiling used
    achieved_compute_frac: Optional[float] = None # of the compute ceiling
    achieved_bw_frac: Optional[float] = None      # of the achievable BW ceiling
    headroom_frac: Optional[float] = None         # 1 - dominant achieved fraction
    bw_basis: str = "n/a"                          # measured | estimated*theoretical
    bytes_source: str = "estimated"               # measured | estimated (caller-supplied)

    def __bool__(self) -> bool:                    # always a usable result
        return True


# ---------------------------------------------------------------------------
# clock / topology detection (detect, don't hardcode)
# ---------------------------------------------------------------------------
def _detect_clocks() -> dict[str, Optional[int]]:
    """max SM clock, max mem clock, current SM clock, bus width — via pynvml,
    falling back to nvidia-smi. Every field degrades to None."""
    out: dict[str, Optional[int]] = {"max_sm": None, "max_mem": None,
                                     "cur_sm": None, "bus_bits": None}
    try:
        import pynvml  # nvidia-ml-py exposes the same module
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            out["max_sm"] = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
            out["max_mem"] = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
            out["cur_sm"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            try:
                out["bus_bits"] = pynvml.nvmlDeviceGetMemoryBusWidth(h)
            except Exception:
                pass
        finally:
            pynvml.nvmlShutdown()
        if out["max_sm"]:
            return out
    except Exception as exc:
        log.debug("pynvml clock probe failed: %s", exc)
    if shutil.which("nvidia-smi"):
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.max.memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=6)
            parts = [p.strip() for p in q.stdout.strip().splitlines()[0].split(",")]
            if parts and parts[0].isdigit():
                out["max_sm"] = int(parts[0])
            if len(parts) > 1 and parts[1].isdigit():
                out["max_mem"] = int(parts[1])
        except Exception as exc:
            log.debug("nvidia-smi clock probe failed: %s", exc)
    return out


def _detect_sm_count(profile: Any) -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            cores = pynvml.nvmlDeviceGetNumGpuCores(h)
            arch = _ARCH.get(getattr(getattr(profile, "gpu", None), "sm_arch", 0))
            if cores and arch:
                return cores // arch["fp32_lanes_per_sm"]
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# microbench (validate the derivation) — CUDA events, warmup, median
# ---------------------------------------------------------------------------
class _ClockSampler:
    """Polls the live SM clock in a daemon thread for the duration of a `with`
    block; `.median_mhz` is the achieved clock under load. Reported as a datum
    only — the theoretical ceiling does NOT depend on it."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = False
        self._t: Optional[threading.Thread] = None
        self._nv = None
        self._h = None

    def __enter__(self) -> "_ClockSampler":
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nv = None
            return self

        def loop() -> None:
            while not self._stop:
                try:
                    self.samples.append(
                        self._nv.nvmlDeviceGetClockInfo(self._h,
                                                        self._nv.NVML_CLOCK_SM))
                except Exception:
                    pass
                time.sleep(0.02)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self._stop = True
        if self._t:
            self._t.join(timeout=0.5)
        try:
            if self._nv:
                self._nv.nvmlShutdown()
        except Exception:
            pass

    @property
    def median_mhz(self) -> Optional[int]:
        # upper-half median: the clock the GEMMs actually ran at (ignore dips)
        if not self.samples:
            return None
        s = sorted(self.samples)
        return int(statistics.median(s[len(s) // 2:]))


def _time_cuda(torch: Any, fn: Callable[[], Any],
               warmup: int, iters: int) -> float:
    """Median seconds per call via CUDA events (the only correct GPU timer)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start.record()
        out = fn()                       # noqa: F841 - keep ref (no DCE)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


def _measure_compute(torch: Any, n: int, warmup: int,
                     iters: int) -> dict[str, float]:
    """Achieved peak per precision (TFLOPS; int8 in TOPS), n×n×n GEMMs."""
    res: dict[str, float] = {}
    flop = 2.0 * n * n * n

    def tflops(secs: float) -> float:
        return flop / secs / 1e12

    fp32a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    fp32b = torch.randn(n, n, device="cuda", dtype=torch.float32)
    res["fp32"] = tflops(_time_cuda(torch, lambda: torch.matmul(fp32a, fp32b),
                                    warmup, iters))
    del fp32a, fp32b

    for prec, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        a = torch.randn(n, n, device="cuda", dtype=dt)
        b = torch.randn(n, n, device="cuda", dtype=dt)
        res[prec] = tflops(_time_cuda(torch, lambda: torch.matmul(a, b),
                                      warmup, iters))
        del a, b

    try:                                           # FP8 e4m3 — B column-major
        a16 = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b16 = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        a8 = a16.to(torch.float8_e4m3fn)
        b8 = b16.to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device="cuda")
        sb = torch.tensor(1.0, device="cuda")
        res["fp8"] = tflops(_time_cuda(
            torch, lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb,
                                            out_dtype=torch.bfloat16),
            warmup, iters))
        del a16, b16, a8, b8
    except Exception as exc:
        log.info("fp8 peak microbench skipped: %s", exc)

    try:                                           # INT8 (TOPS) — conservative floor
        ai = torch.randint(-127, 128, (n, n), dtype=torch.int8, device="cuda")
        bi = torch.randint(-127, 128, (n, n), dtype=torch.int8, device="cuda")
        res["int8"] = tflops(_time_cuda(torch, lambda: torch._int_mm(ai, bi),
                                        warmup, iters))
        del ai, bi
    except Exception as exc:
        log.info("int8 peak microbench skipped: %s", exc)

    torch.cuda.empty_cache()
    return res


def _measure_dram_bw(torch: Any, mb: int, warmup: int, iters: int) -> float:
    """GB/s of a large device-to-device copy (buffer >> 72 MB L2)."""
    n = (mb * 1024 * 1024) // 4
    x = torch.empty(n, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    y = torch.empty_like(x)
    secs = _time_cuda(torch, lambda: y.copy_(x), warmup, iters)
    bytes_moved = n * 4 * 2          # read x + write y
    del x, y
    torch.cuda.empty_cache()
    return bytes_moved / secs / 1e9


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def derive_roofline(profile: Any, *, measure: bool = True,
                    gemm_n: int = 8192, copy_mb: int = 512,
                    warmup: int = 10, iters: int = 30) -> Roofline:
    """Derive the roofline from detected hardware, validate by microbench.

    THEORETICAL peaks come from (detected sm_count) × (arch lanes) × **max
    clock** and (detected bus) × (detected mem clock) × 2 — a stable, reproducible
    absolute speed-of-light. If `measure` and torch+CUDA are present,
    microbenchmarks fill the measured (achievable) peaks and sample the achieved
    clock as a separate datum.
    """
    gpu = getattr(profile, "gpu", None)
    sm_arch = getattr(gpu, "sm_arch", None) or 0
    name = getattr(gpu, "name", None) or "GPU?"
    arch = _ARCH.get(sm_arch)
    notes: list[str] = []

    clocks = _detect_clocks()
    sm_count = _detect_sm_count(profile) or 0
    bus_bits = clocks["bus_bits"] or (
        384 if (gpu and (gpu.vram_total_mb or 0) >= 20000) else None)

    rl = Roofline(
        gpu_name=name, sm_arch=sm_arch, sm_count=sm_count,
        max_sm_clock_mhz=clocks["max_sm"], achieved_sm_clock_mhz=None,
        mem_clock_mhz=clocks["max_mem"], mem_bus_bits=bus_bits,
        dram_bw_theoretical_gbps=None, dram_bw_measured_gbps=None, notes=notes)

    if arch is None:
        notes.append(f"unknown architecture sm_{sm_arch}; cannot derive peaks.")
        return rl
    if not sm_count or not clocks["max_sm"]:
        notes.append("sm_count or max clock undetected; peaks unavailable.")
        return rl

    if bus_bits and clocks["max_mem"]:
        rl.dram_bw_theoretical_gbps = round(
            (bus_bits / 8) * (clocks["max_mem"] * 2 * 1e6) / 1e9, 1)  # GDDR6X x2

    # microbench (optional): measured peaks + achieved clock datum
    measured: dict[str, float] = {}
    if measure:
        try:
            import torch
            if torch.cuda.is_available():
                with _ClockSampler() as samp:
                    measured = _measure_compute(torch, gemm_n, warmup, iters)
                rl.achieved_sm_clock_mhz = samp.median_mhz   # compute-phase only
                rl.dram_bw_measured_gbps = round(
                    _measure_dram_bw(torch, copy_mb, warmup, iters), 1)
                rl.measured = True
            else:
                notes.append("torch present but CUDA unavailable; theoretical only.")
        except Exception as exc:
            notes.append(f"microbench failed ({exc}); theoretical only.")

    # THEORETICAL peak pins to the MAX clock — stable absolute speed-of-light.
    clk_for_peak = clocks["max_sm"]
    rl.clock_for_peak_mhz = clk_for_peak
    lanes = arch["fp32_lanes_per_sm"]
    fp32_peak = sm_count * lanes * 2 * (clk_for_peak * 1e6) / 1e12   # TFLOPS
    dram_th = rl.dram_bw_theoretical_gbps or 0.0

    def ridge(peak_tflops: float) -> Optional[float]:
        return round(peak_tflops * 1e12 / (dram_th * 1e9), 1) if dram_th else None

    rl.peaks["fp32"] = PrecisionPeak("fp32", round(fp32_peak, 1),
                                     measured.get("fp32"), ridge(fp32_peak))
    for prec, mult in arch["tensor_mult"].items():
        peak = fp32_peak * mult
        # int8 is TOPS -> no FLOP/byte ridge (dimensionally inconsistent)
        r = ridge(peak) if prec in _FLOP_PRECISIONS else None
        rl.peaks[prec] = PrecisionPeak(prec, round(peak, 1), measured.get(prec), r)
    return rl


def saturation_ceiling(peak_obj: PrecisionPeak) -> float:
    """The realistically-achievable compute ceiling for achieved-fraction
    comparisons: the MEASURED (microbench) peak when present, else theoretical.
    SHARED by hwmodel.classify and profiler.classify_dominant so the OBSERVE and
    MODEL layers can never return different bounds for the same numbers."""
    return peak_obj.measured or peak_obj.theoretical


def classify(rl: Roofline, *, flops: float, dram_bytes: float,
             precision: str = "fp16", achieved_tflops: Optional[float] = None,
             achieved_bw_gbps: Optional[float] = None,
             occupancy: Optional[float] = None,
             duration_s: Optional[float] = None,
             bytes_source: str = "estimated") -> ClassifyResult:
    """Place a kernel on the roofline and name its bottleneck (structured result).

    Rule: latency gate first (tiny / under-occupied / saturates-neither), then
    OI=flops/dram_bytes vs the precision ridge (< -> MEMORY, >= -> COMPUTE), with
    a +/-15% BALANCED band, then a STEP-2 demotion to LATENCY if neither ceiling
    is >70% reached.

    `dram_bytes` MUST be MEASURED DRAM traffic (Nsight/CUPTI `dram__bytes`), not
    tensor array sizes — caches/fusion make array-size byte counts optimistic and
    flip MEMORY<->COMPUTE near the ridge. Pass bytes_source='measured' when it is;
    'estimated' widens the BALANCED band to stay honest about the uncertainty.
    """
    peak_obj = rl.peaks.get(precision)
    if peak_obj is None or not peak_obj.theoretical:
        return ClassifyResult(Bound.BALANCED, f"no ceiling for precision {precision!r}",
                              bytes_source=bytes_source)
    peak = peak_obj.theoretical
    ridge = peak_obj.ridge_flop_per_byte
    ach_bw_ceiling = rl.dram_bw_achievable_gbps or 0.0
    # the saturation ceiling for the >70% / <20% tests — the SHARED policy
    # (measured achievable peak when present), identical to profiler's, so the
    # two layers agree on what "% of peak" means.
    sat_ceiling = saturation_ceiling(peak_obj)
    comp_frac = (achieved_tflops / sat_ceiling) if achieved_tflops else None
    bw_frac = (achieved_bw_gbps / ach_bw_ceiling) if (achieved_bw_gbps and ach_bw_ceiling) else None

    def result(bound: Bound, reason: str, oi: Optional[float] = None) -> ClassifyResult:
        fr = [f for f in (comp_frac, bw_frac) if f is not None]
        dm = min(1.0, max(fr)) if fr else None      # clamp: no faster-than-light headroom
        return ClassifyResult(
            bound=bound, reason=reason, oi=oi, ridge=ridge, ceiling_tflops=peak,
            achieved_compute_frac=comp_frac, achieved_bw_frac=bw_frac,
            headroom_frac=(1 - dm) if dm is not None else None,
            bw_basis=rl.bw_basis, bytes_source=bytes_source)

    # data-fault guard (shared policy with profiler.classify_dominant): an
    # impossible compute fraction means the FLOP estimate / precision key is
    # wrong -> refuse a confident bound instead of asserting COMPUTE.
    if comp_frac is not None and comp_frac > 1.10:
        return result(Bound.BALANCED,
                      f"achieved {comp_frac*100:.0f}% > ceiling -> FLOP estimate "
                      f"or precision key suspect; not classifying")

    # OI is the bound axis; compute it first so the (noisy, single-sample)
    # achieved-fraction latency checks can be confined to the AMBIGUOUS near-ridge
    # zone. NOTE: comp_frac/bw_frac are derived from the SAME (flops, bytes, time)
    # triple as OI, so they are NOT an independent confirmation of the bound; they
    # only add the absolute-saturation/latency axis (is the kernel busy or idle).
    # True independent byte validation needs measured dram__bytes (ncu).
    oi = flops / max(dram_bytes, 1.0)
    near_ridge = bool(ridge) and (0.5 * ridge) <= oi <= (2.0 * ridge)

    # OI-independent latency signals (a tiny / under-occupied kernel is
    # launch-bound regardless of its intensity).
    if duration_s is not None and duration_s < 5e-6:
        return result(Bound.LATENCY, f"duration {duration_s*1e6:.1f}us < 5us (launch-bound)", oi)
    if occupancy is not None and occupancy < 0.30:
        return result(Bound.LATENCY, f"occupancy {occupancy:.2f} < 0.30", oi)
    # 'saturates neither engine' is only meaningful near the ridge — a kernel deep
    # on the OI axis is doing real work even if one sample reads low.
    if (near_ridge and comp_frac is not None and bw_frac is not None
            and comp_frac < 0.20 and bw_frac < 0.20):
        return result(Bound.LATENCY, "saturates neither engine (<20% of both, near ridge)", oi)

    if not ridge:        # DRAM BW undetected, or int8/TOPS -> do NOT guess
        return result(Bound.BALANCED,
                      f"no ridge for {precision!r} (DRAM BW undetected or TOPS); "
                      f"bound unknown")

    # STEP 1 — operational intensity vs ridge (the decisive bound axis)
    band = 0.25 if bytes_source != "measured" else 0.15   # estimated bytes -> wider band
    if abs(oi - ridge) / ridge <= band:
        if comp_frac is not None and bw_frac is not None:
            dm2 = "compute" if comp_frac >= bw_frac else "memory"
            return result(Bound.BALANCED, f"OI {oi:.0f} ~ ridge {ridge:.0f} (dom {dm2})", oi)
        return result(Bound.BALANCED, f"OI {oi:.0f} ~ ridge {ridge:.0f}", oi)
    bound = Bound.MEMORY if oi < ridge else Bound.COMPUTE

    # STEP 2 — only in the ambiguous near-ridge zone may a saturates-neither
    # sample demote to LATENCY. Far from the ridge, OI is decisive (fixes the
    # cold-start COMPUTE<->LATENCY flip from one noisy profiler-timed sample).
    if (near_ridge and comp_frac is not None and bw_frac is not None
            and comp_frac < 0.70 and bw_frac < 0.70):
        return result(Bound.LATENCY,
                      f"OI {oi:.0f} ~ ridge but neither ceiling >70% reached", oi)
    return result(bound, f"OI {oi:.1f} {'<' if bound is Bound.MEMORY else '>='} ridge {ridge:.0f}", oi)


# ---------------------------------------------------------------------------
# CLI harness
# ---------------------------------------------------------------------------
def _hr(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _fp8_speed_of_light() -> Optional[tuple[float, float]]:
    """Most recent CUTLASS-Vellum FP8 GEMM from the quality ledger:
    (achieved_tflops, K). Reuses real measured data; None if no row."""
    try:
        from v9 import quality
        import json
        for row in quality.fetch_recent_quality(limit=25,
                                                 backend_name="cutlass_fp8_gemm"):
            notes = row.get("notes_json")
            notes = json.loads(notes) if isinstance(notes, str) else (notes or {})
            fp8_ms = notes.get("fp8_ms")
            k = row.get("shape_w") or 4096
            if fp8_ms:
                return 2.0 * k * k * k / (fp8_ms / 1e3) / 1e12, float(k)
    except Exception as exc:
        log.debug("fp8 SoL lookup skipped: %s", exc)
    return None


def run(config: Any = None, measure: bool = True, gemm_n: int = 8192,
        iters: int = 30) -> int:
    if config is None:
        from v9.config import load_config
        config = load_config()
    profile = config.profile

    print("=" * 72)
    print(" Optimizer V11 — roofline (the hardware model; derived + measured)")
    print("=" * 72)
    print("  " + profile.summary())

    rl = derive_roofline(profile, measure=measure, gemm_n=gemm_n, iters=iters)

    _hr("Topology + clocks (DETECTED)")
    print(f"  {rl.gpu_name}  sm_{rl.sm_arch}  ·  {rl.sm_count} SMs  ·  "
          f"bus {rl.mem_bus_bits}-bit")
    print(f"  SM clock: max {rl.max_sm_clock_mhz} MHz (peak pinned here)"
          + (f"  ·  achieved-under-load {rl.achieved_sm_clock_mhz} MHz"
             if rl.achieved_sm_clock_mhz else "  (achieved: not measured)"))

    _hr("Memory roofline")
    th = rl.dram_bw_theoretical_gbps
    me = rl.dram_bw_measured_gbps
    print(f"  DRAM bandwidth: theoretical {th} GB/s"
          + (f"  ·  measured {me} GB/s ({me/th*100:.0f}%)" if me and th else "")
          + f"  ·  classify uses {rl.bw_basis}")

    _hr("Compute roofline  (theoretical @ MAX clock; dense FP32-accumulate)")
    print(f"  {'precision':9s} {'theoretical':>13s} {'measured':>11s} "
          f"{'realized':>9s} {'ridge(F/B)':>11s}")
    print("  " + "-" * 56)
    for prec in ("fp32", "fp16", "bf16", "fp8", "int8"):
        p = rl.peaks.get(prec)
        if not p:
            continue
        unit = "TOPS" if prec == "int8" else "TFLOPS"
        meas = f"{p.measured:.1f}" if p.measured else "—"
        frac = f"{p.realized_fraction*100:.0f}%" if p.realized_fraction else "—"
        ridge = f"{p.ridge_flop_per_byte:.0f}" if p.ridge_flop_per_byte else "n/a"
        print(f"  {prec:9s} {p.theoretical:>8.1f} {unit:4s} {meas:>11s} "
              f"{frac:>9s} {ridge:>11s}")

    # --- speed-of-light check vs the MEASURED (achievable) FP8 ceiling ---
    _hr("Speed-of-light: CUTLASS-Vellum FP8 GEMM vs the FP8 ceiling")
    fp8 = rl.peaks.get("fp8")
    ceiling = fp8.measured if fp8 else None        # the achievable ceiling, not theoretical
    sol = _fp8_speed_of_light()
    if ceiling and sol:
        achieved, k = sol
        frac = achieved / ceiling
        print(f"  cached kernel: {achieved:.1f} TFLOPS @ {int(k)}^3   "
              f"measured ceiling: {ceiling:.1f} TFLOPS")
        print(f"  -> {frac*100:.0f}% of speed-of-light"
              + (f"  ·  ~{(1-frac)*100:.0f}% headroom for the Mile-4 autotuner"
                 if frac < 0.95 else "  ·  essentially at the ceiling"))
    elif sol and not ceiling:
        print("  FP8 ceiling NOT measured (run without --no-measure); refusing to "
              "compute headroom against the inflated theoretical peak.")
    elif ceiling:
        print(f"  FP8 measured ceiling {ceiling:.1f} TFLOPS; no cached Vellum FP8 "
              f"row in the quality ledger — run `v9 benchmark` to populate it.")
    else:
        print("  FP8 speed-of-light check unavailable (no measured fp8 peak / ledger row).")

    if rl.notes:
        _hr("Notes")
        for n in rl.notes:
            print(f"  · {n}")

    print()
    print("roofline: complete (the model the V11 control loop reasons against).")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
