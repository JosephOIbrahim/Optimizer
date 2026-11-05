"""
V12 — one live source of hardware truth, and the checks that keep it honest.

MachineOptimizer2026 hardcodes `studio_driver: "610.62"` in hardware_profile.py.
The box is on 616.92. That is not a cosmetic staleness bug: the driver number is
the thing that decides WHICH optimizations are even legal, so a hardcoded value
means the catalog is being filtered against a machine that does not exist.

The first version of this module fixed the staleness by reading everything live,
and then shipped three numbers that were wrong in exactly the same way the
hardcoded driver was wrong — plausible, unexamined, and load-bearing. All three
were found on 2026-09-16 by comparing what the code claimed against what the
machine did. They are recorded here because the *shape* of the mistake is the
thing worth defending against:

1. **SM count was 128x too high.** `nvmlDeviceGetNumGpuCores` returns CUDA cores
   (16384), not streaming multiprocessors (128). It is a well-named function that
   answers a different question than the one it was asked, which is why nothing
   complained. The CUDA driver API answers the actual question
   (`cuDeviceGetAttribute(16)`), and torch's `multi_processor_count` — a
   separate code path — agrees.

2. **Every FLOP rate was 2x too low**, so every derived ceiling was halved. The
   fp32 rate is refuted by measurement alone: a 128 FLOP/SM/clk rate would need a
   5385 MHz clock to reach the 88.20 TFLOPS a real 8192^3 FP32 GEMM achieved on
   this box, and the card tops out at 3105. The fp16 rate is *not* refutable that
   way — 512/SM/clk needs only 2521 MHz, which the card can hold — so it is pinned
   by Ada's fixed precision ratios (fp16 = 2 x tf32 = 4 x fp32) once fp32 is
   settled, which puts it at 1024. Stating which of the two is proven by what is
   the point: a check that gets credited for more than it tests is the same
   failure as a number that looks more authoritative than it is.

3. **A live clock was used where a maximum was required.** `mem_clock_mhz` was
   overwritten with `nvmlDeviceGetClockInfo` (the instantaneous clock), then fed
   to `mem_bandwidth_gbs`. An idle 4090 sits at ~405 MHz on the memory clock, so
   the module would have reported 39 GB/s of bandwidth and a ridge point 26x too
   high — while looking entirely reasonable. Roofline math needs maximums; live
   clocks are a different quantity and now live in different fields.

The defence is structural, not a fix to three lines. Every field is now read from
up to three independent readers (NVML, the CUDA driver API, nvidia-smi) and the
merge refuses to publish a value that a second reader contradicts, that falls
outside a physically possible band, or that fails the internal identity
`cuda_cores / sm_count == lanes_per_SM`. The FLOP table is checked by asking
whether reaching a *measured* throughput would require a clock the silicon does
not have — not by comparing it against a spec sheet, because the spec sheet is
what was wrong.

Where V9 already does this well, V12 delegates rather than reimplements:
`v9/hwmodel.py` has a real, microbench-validated roofline. If it imports, we use
it. If it does not, we compute the same quantities here and say so.
"""
from __future__ import annotations

import ctypes
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import paths

log = logging.getLogger("v12.truth")

# ---------------------------------------------------------------------------
# FLOP rates for Ada (sm_89), per SM per clock, DENSE (sparsity is never assumed)
#
# These are the only performance constants in this file and they are structural —
# tensor-core datapath width, not SKU. Everything SKU-specific is measured.
#
# These were all 2x lower until 2026-09-16, when a measured GEMM exceeded the
# ceiling they implied. The numbers below are the smallest values consistent with
# every measurement taken on this machine and every published Ada figure:
#   fp16 165.26 TFLOPS measured  @ ~2690 MHz sustained  -> >= 480/SM/clk
#   fp32  88.20 TFLOPS measured  @ ~2690 MHz sustained  -> >= 256/SM/clk
#   fp32 82.6 TFLOPS datasheet   @ 2520 MHz boost       ->    256/SM/clk
#   fp16 330.3 TFLOPS datasheet  @ 2520 MHz boost       ->   1024/SM/clk
# The datasheet and the measurement agree once the rate is doubled, and the
# measurement alone is already enough to refute the old table.
# ---------------------------------------------------------------------------
FLOP_PER_SM_CLK: Dict[str, float] = {
    "fp32_cuda":    256.0,   # 128 FP32 lanes x 2 (FMA)
    "fp16_tensor": 1024.0,   # dense, FP32 accumulate
    "bf16_tensor": 1024.0,
    "tf32_tensor":  512.0,
    "fp8_tensor":  2048.0,
    "int8_tensor": 2048.0,
}

#: A real GEMM run on THIS machine, used as the floor a FLOP rate must be able to
#: account for. A rate is refuted when reaching this throughput would need a clock
#: the card does not have. Checked at read time rather than trusted. Re-measure:
#:   D:\OptimizerV9\venv\Scripts\python.exe -c "import torch;a=torch.randn(8192,8192,dtype=torch.float32,device='cuda');print((2*8192**3)/(a@a).numel())"
#: (any large-square matmul; see the 2026-09-16 session for the timing harness)
MEASURED_FLOOR_TFLOPS: Dict[str, Tuple[float, str]] = {
    "fp32_cuda":   (88.20, "torch 2.8.0+cu126, 8192^3 fp32 matmul, 2026-09-16"),
    "fp16_tensor": (165.26, "torch 2.8.0+cu126, 8192^3 fp16 matmul, 2026-09-16"),
}

#: Consumer Ada cards boost ABOVE their rated clock: the rated figure is a
#: guaranteed minimum, not a cap. The fp32 measurement above needs roughly
#: 2690 MHz across 128 SMs — 6.8% over the 2520 MHz rated clock — so a check that
#: treated the rated clock as a hard ceiling would fire on a perfectly healthy
#: card, which is a false alarm and therefore a bug of its own.
#:
#: Used ONLY to decide which clocks make a measurement *impossible* for a given
#: FLOP rate. Never used to publish a number.
BOOST_HEADROOM = 1.25

#: CUDA cores per SM, by architecture generation. Used only as the set of
#: *permitted ratios* in the cross-check below — never to derive one from the
#: other, because deriving is how the original bug would have been hidden.
LANES_PER_SM_PERMITTED = (64.0, 128.0, 256.0)

#: Physically possible bands. A value outside its band is not published, however
#: confident its reader was. These exist to stop a plausible-but-wrong number
#: from reaching the roofline math, which is what the SM-count bug did.
BANDS: Dict[str, Tuple[float, float]] = {
    "sm_count":            (1, 1024),
    "cuda_cores":          (64, 262144),
    "mem_bus_width_bits":  (32, 8192),
    "clock_sm_max_mhz":    (100, 5000),
    "clock_sm_rated_mhz":  (100, 5000),
    "clock_mem_max_mhz":   (100, 20000),
    "vram_total_bytes":    (1 << 30, 1 << 42),
    "power_limit_w":       (1.0, 2000.0),
}

#: On this hardware, only these precisions have a real kernel path (see the
#: cuTile FP8 finding: FP8 raises TileUnsupportedFeatureError on the sm80 family,
#: but that is cuTile's limit, not the silicon's — cuBLAS/CUTLASS FP8 GEMM is
#: proven at 296.7 TFLOPS on this exact box, which is 45% of the 660.6 dense
#: ceiling and implausibly close to peak if the ceiling were half that).
BACKEND_REACHABLE = {
    "cutlass_cpp": True,
    "cublas": True,
    "triton": True,
    "cuda_tile": True,      # no FP8 under cuTile
    "cute_dsl": False,      # POSIX-only, unreachable on Windows
}


# ---------------------------------------------------------------------------
# reader 1 — NVML, via nvidia-ml-py
# ---------------------------------------------------------------------------
def _nvml():
    """The NVML binding, or None.

    `pynvml` and `nvidia-ml-py` both install under the module name `pynvml`; the
    standalone `pynvml` package is deprecated and emits a FutureWarning on import.
    That warning is real advice for anyone installing it deliberately, and pure
    noise printed into an operator's report by a tool that knowingly handles both.
    Suppressed narrowly, at the import, rather than globally or by filter string.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import pynvml  # nvidia-ml-py; the standalone `pynvml` package is deprecated
        return pynvml
    except ImportError:
        return None


def _read_nvml() -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """Read what NVML can answer. Missing answers are omitted, not zeroed.

    Note what is deliberately NOT here: `sm_count` from `nvmlDeviceGetNumGpuCores`.
    That function returns CUDA cores. It is recorded under the name that describes
    what it actually returns.
    """
    vals: Dict[str, Any] = {}
    src: Dict[str, str] = {}
    errs: List[str] = []
    p = _nvml()
    if p is None:
        return vals, src, ["nvml: nvidia-ml-py not importable"]
    try:
        p.nvmlInit()
    except Exception as e:
        return vals, src, [f"nvml: nvmlInit failed: {e}"]

    #: NVML enum values, as integers so a binding that does not export the name
    #: still works. Verified on this box: clock id 1 -> 3105 MHz (SM max),
    #: id 2 -> 10501 MHz (memory max).
    clock_ids = {
        "sm": int(getattr(p, "NVML_CLOCK_SM", 1)),
        "mem": int(getattr(p, "NVML_CLOCK_MEM", 2)),
    }

    try:
        h = p.nvmlDeviceGetHandleByIndex(0)

        def take(key: str, fn_name: str, *a, note: str = "") -> None:
            """Resolve the symbol INSIDE the guard, by name.

            Fetching `p.nvmlDeviceGetAttribute` at the call site would raise
            AttributeError in _read_nvml's own frame — outside this try — and take
            the entire NVML read down with it. The deprecated `pynvml` package
            genuinely lacks that symbol, so on this machine that is not
            hypothetical: it is the default path. One missing function must cost
            one field.
            """
            try:
                fn = getattr(p, fn_name, None)
                if fn is None:
                    raise AttributeError(f"{fn_name} absent from this binding")
                v = fn(*a)
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "replace")
                if isinstance(v, (tuple, list)):
                    v = v[0]
                vals[key] = v
                src[key] = "nvml" + (":" + note if note else "")
            except Exception as e:
                errs.append(f"nvml:{key}: {e}")

        take("name", "nvmlDeviceGetName", h)
        take("driver_version", "nvmlSystemGetDriverVersion")

        def _cuda_api():
            fn = getattr(p, "nvmlSystemGetCudaDriverVersion_v2", None)
            if fn is None:
                raise AttributeError("nvmlSystemGetCudaDriverVersion_v2 absent")
            v = fn()
            return f"{v // 1000}.{(v % 1000) // 10}" if isinstance(v, int) else v
        try:
            vals["cuda_driver_api"] = _cuda_api()
            src["cuda_driver_api"] = "nvml"
        except Exception as e:
            errs.append(f"nvml:cuda_driver_api: {e}")

        # SM count: ONLY from the attribute API, which answers the real question.
        # On a binding that lacks it (the deprecated `pynvml` package has no
        # nvmlDeviceGetAttribute, and nvml.dll on this box does not export it),
        # this records an error and the field stays absent — which is correct.
        # The CUDA reader fills it instead.
        take("sm_count", "nvmlDeviceGetAttribute", h, 16, note="attr16")

        # CUDA cores: the honest name for what NumGpuCores returns. This is NOT
        # the SM count and is never used as one.
        take("cuda_cores", "nvmlDeviceGetNumGpuCores", h, note="NumGpuCores")

        def _cc():
            major, minor = p.nvmlDeviceGetCudaComputeCapability(h)
            return f"{major}.{minor}"
        try:
            vals["compute_capability"] = _cc()
            src["compute_capability"] = "nvml"
        except Exception as e:
            errs.append(f"nvml:compute_capability: {e}")

        def _vram():
            return int(p.nvmlDeviceGetMemoryInfo(h).total)
        try:
            vals["vram_total_bytes"] = _vram()
            src["vram_total_bytes"] = "nvml"
        except Exception as e:
            errs.append(f"nvml:vram_total_bytes: {e}")

        take("mem_bus_width_bits", "nvmlDeviceGetMemoryBusWidth", h)

        # MAX clocks. This is the only clock a roofline may use. `GetMaxClockInfo`
        # is a ceiling; `GetClockInfo` is an instantaneous reading that drops to
        # ~405 MHz on an idle 4090 and would silently deflate bandwidth 26x.
        take("clock_sm_max_mhz", "nvmlDeviceGetMaxClockInfo", h, clock_ids["sm"])
        take("clock_mem_max_mhz", "nvmlDeviceGetMaxClockInfo", h, clock_ids["mem"])

        # Live clocks, kept as their own fields. Useful for a watchdog, never for
        # a peak. They are here so the two can never be confused again.
        take("clock_sm_live_mhz", "nvmlDeviceGetClockInfo", h, clock_ids["sm"], note="live")
        take("clock_mem_live_mhz", "nvmlDeviceGetClockInfo", h, clock_ids["mem"], note="live")

        def _limits():
            lo, hi = p.nvmlDeviceGetPowerManagementLimitConstraints(h)
            return (lo / 1000.0, hi / 1000.0, p.nvmlDeviceGetPowerManagementLimit(h) / 1000.0)
        try:
            lo, hi, cur = _limits()
            vals["power_limit_min_w"], vals["power_limit_max_w"], vals["power_limit_w"] = lo, hi, cur
            for k in ("power_limit_min_w", "power_limit_max_w", "power_limit_w"):
                src[k] = "nvml"
        except Exception as e:
            errs.append(f"nvml:power_limit: {e}")
    finally:
        try:
            p.nvmlShutdown()
        except Exception:
            pass
    return vals, src, errs


# ---------------------------------------------------------------------------
# reader 2 — the CUDA driver API, via ctypes on nvcuda.dll
#
# This reader exists because it needs no pip install, is present wherever a
# driver is, and answers the two questions NVML on this box gets wrong or cannot
# answer at all: multiprocessor count, and the compute capability.
#
# ATTRIBUTE IDS ARE VERSION-SENSITIVE and were confirmed by reading them, not by
# looking them up. On this driver (616.92, which exposes CUDA 13.4; 12.6 is only
# torch's build toolkit, not the driver's), attribute 37 returned
# 384 — the correct memory bus width for a 4090 — while 38 returned 75497472,
# which is not a bus width. Older headers put MEMORY_BUS_WIDTH at 38. Anything
# reading an ID from a table without checking the returned value is one driver
# update away from a silently wrong number, which is this module's founding bug.
# ---------------------------------------------------------------------------
class _CudaDriver:
    ATTR_MULTIPROCESSOR_COUNT = 16
    ATTR_CLOCK_RATE_KHZ = 13
    ATTR_MEMORY_BUS_WIDTH = 37
    ATTR_CC_MAJOR = 75
    ATTR_CC_MINOR = 76

    def __init__(self) -> None:
        self.lib = None
        self.dev = None
        self.error = ""
        try:
            self.lib = ctypes.CDLL("nvcuda.dll")
        except OSError as e:
            self.error = f"nvcuda.dll not loadable: {e}"
            return
        try:
            if self.lib.cuInit(0) != 0:
                self.error = "cuInit(0) failed"
                return
            dev = ctypes.c_int(0)
            if self.lib.cuDeviceGet(ctypes.byref(dev), ctypes.c_int(0)) != 0:
                self.error = "cuDeviceGet(0) failed"
                return
            self.dev = dev
        except Exception as e:
            self.error = f"cuda init: {e}"

    @property
    def ok(self) -> bool:
        return self.dev is not None

    def attr(self, attr_id: int) -> Optional[int]:
        if not self.ok:
            return None
        try:
            out = ctypes.c_int(0)
            rc = self.lib.cuDeviceGetAttribute(ctypes.byref(out), ctypes.c_int(attr_id),
                                              self.dev)
            return int(out.value) if rc == 0 else None
        except Exception:
            return None

    def name(self) -> Optional[str]:
        if not self.ok:
            return None
        try:
            buf = ctypes.create_string_buffer(128)
            if self.lib.cuDeviceGetName(buf, ctypes.c_int(128), self.dev) == 0:
                return buf.value.decode("utf-8", "replace")
        except Exception:
            pass
        return None


def _read_cuda() -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    d = _CudaDriver()
    if not d.ok:
        return {}, {}, [f"cuda: {d.error}"]

    vals: Dict[str, Any] = {}
    src: Dict[str, str] = {}
    errs: List[str] = []

    n = d.name()
    if n:
        vals["name"] = n
        src["name"] = "cuda:cuDeviceGetName"

    sm = d.attr(_CudaDriver.ATTR_MULTIPROCESSOR_COUNT)
    if sm is not None:
        vals["sm_count"] = sm
        src["sm_count"] = "cuda:attr16"

    major = d.attr(_CudaDriver.ATTR_CC_MAJOR)
    minor = d.attr(_CudaDriver.ATTR_CC_MINOR)
    if major is not None and minor is not None:
        vals["compute_capability"] = f"{major}.{minor}"
        src["compute_capability"] = "cuda:attr75/76"

    bus = d.attr(_CudaDriver.ATTR_MEMORY_BUS_WIDTH)
    if bus is not None:
        vals["mem_bus_width_bits"] = bus
        src["mem_bus_width_bits"] = "cuda:attr37"

    # CLOCK_RATE is the graphics-clock ceiling — the RATED boost clock, 2520 MHz
    # on this box, which is the figure a peak should be computed at. It is a
    # different quantity from NVML's max-boost SM clock (3105 MHz), which is a
    # clock the driver may request but no sustained kernel holds. Both are real
    # and they legitimately differ, so they live in separate fields rather than
    # competing in the merge.
    khz = d.attr(_CudaDriver.ATTR_CLOCK_RATE_KHZ)
    if khz is not None:
        vals["clock_sm_rated_mhz"] = round(khz / 1000.0)
        src["clock_sm_rated_mhz"] = "cuda:attr13"

    # A CUDA reader that returned nothing usable is worth saying out loud.
    if not vals:
        errs.append("cuda: nvcuda.dll loaded but returned no usable attribute")
    return vals, src, errs


# ---------------------------------------------------------------------------
# reader 3 — nvidia-smi
#
# The last resort, and the one that keeps the tool useful on an interpreter with
# no pynvml at all — which is exactly the interpreter the launcher used to pick.
# It is also the only reader that works when NVML refuses to initialise.
# ---------------------------------------------------------------------------
_SMI_FIELDS = (
    "name,driver_version,compute_cap,memory.total,clocks.max.sm,clocks.max.mem,"
    "power.min_limit,power.max_limit,power.default_limit"
)

#: index in the CSV above -> (field, transform)
_SMI_MAP: Tuple[Tuple[int, str, str], ...] = (
    (0, "name", "str"),
    (1, "driver_version", "str"),
    (2, "compute_capability", "str"),
    (3, "vram_total_bytes", "mib"),
    (4, "clock_sm_max_mhz", "int"),
    (5, "clock_mem_max_mhz", "int"),
    (6, "power_limit_min_w", "float"),
    (7, "power_limit_max_w", "float"),
    (8, "power_limit_w", "float"),
)


def _read_smi() -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_SMI_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except Exception as e:
        return {}, {}, [f"smi: {e}"]
    if not out.strip():
        return {}, {}, ["smi: nvidia-smi produced no output"]

    parts = [x.strip() for x in out.splitlines()[0].split(",")]
    vals: Dict[str, Any] = {}
    src: Dict[str, str] = {}
    for idx, key, kind in _SMI_MAP:
        if idx >= len(parts):
            continue
        raw = parts[idx]
        # nvidia-smi writes N/A (or [N/A]) for anything the driver branch does not
        # expose. Absent is the right answer; zero would be a lie.
        if not raw or "N/A" in raw:
            continue
        try:
            if kind == "str":
                vals[key] = raw
            elif kind == "mib":
                vals[key] = int(float(raw)) * 1024 * 1024
            elif kind == "int":
                vals[key] = int(float(raw))
            else:
                vals[key] = float(raw)
            src[key] = "nvidia-smi"
        except ValueError:
            continue
    return vals, src, []


# ---------------------------------------------------------------------------
# merge — precedence, contradiction, and the physical bands
# ---------------------------------------------------------------------------
#: Which reader wins for each field, best first. The order is a claim about
#: authority, not about convenience:
#:   sm_count prefers CUDA because it is the only reader that asks for SM count.
#:   Clocks and power prefer NVML because it separates max from live, which
#:   nvidia-smi's query output does not.
PRECEDENCE: Dict[str, Tuple[str, ...]] = {
    "name":               ("nvml", "cuda", "smi"),
    "driver_version":     ("nvml", "smi"),
    "cuda_driver_api":    ("nvml",),
    "sm_count":           ("cuda", "nvml"),
    "cuda_cores":         ("nvml",),
    "compute_capability": ("nvml", "cuda", "smi"),
    "vram_total_bytes":   ("nvml", "smi"),
    "mem_bus_width_bits": ("nvml", "cuda"),
    "clock_sm_max_mhz":   ("nvml", "smi"),
    "clock_mem_max_mhz":  ("nvml", "smi"),
    "clock_sm_live_mhz":  ("nvml",),
    "clock_mem_live_mhz": ("nvml",),
    "clock_sm_rated_mhz": ("cuda",),   # nvidia-smi has no rated-clock query; its
                                       # clocks.max.sm is the same 3105 MHz max NVML reports
    "power_limit_w":      ("nvml", "smi"),
    "power_limit_min_w":  ("nvml", "smi"),
    "power_limit_max_w":  ("nvml", "smi"),
}


@dataclass
class MergeResult:
    values: Dict[str, Any] = field(default_factory=dict)
    source: Dict[str, str] = field(default_factory=dict)
    disagreements: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _merge(readers: List[Tuple[str, Dict[str, Any], Dict[str, str]]]) -> MergeResult:
    """Combine readers, refusing to publish anything two readers contradict.

    Two readers disagreeing about one field is the single most valuable signal
    available from a machine: it means at least one of them is being read wrong,
    and no amount of confidence from either resolves which. So the field is
    dropped and named, rather than resolved by precedence — precedence would have
    silently picked a winner and hidden the disagreement.
    """
    m = MergeResult()
    for field_name, order in PRECEDENCE.items():
        candidates: List[Tuple[str, Any]] = []
        for reader_name in order:
            for rname, vals, _src in readers:
                if rname == reader_name and field_name in vals:
                    candidates.append((reader_name, vals[field_name]))
                    break
        if not candidates:
            continue

        winner_name, winner = candidates[0]
        for other_name, other in candidates[1:]:
            if not _agrees(field_name, winner, other):
                m.disagreements.append(
                    f"{field_name}: {other_name} says {other!r} but {winner_name} says "
                    f"{winner!r} — refusing to publish either"
                )
                # Drop it. A field that two readers contradict is not a value with
                # an uncertainty; it is an unresolved question.
                winner = None
                break
        if winner is None:
            continue

        # Physical band. This is the check that would have caught a 16384 SM count
        # on a GPU that has 128 of them, without needing a second reader.
        band = BANDS.get(field_name)
        if band and isinstance(winner, (int, float)) and not isinstance(winner, bool):
            lo, hi = band
            if not (lo <= winner <= hi):
                m.errors.append(
                    f"{field_name}: {winner_name} returned {winner!r}, outside the "
                    f"possible range {lo:g}..{hi:g} — discarded"
                )
                continue
        m.values[field_name] = winner
        m.source[field_name] = _src_for(readers, winner_name, field_name)

    # Bus width has a shape as well as a range: it is always a multiple of 32 bits.
    bw = m.values.get("mem_bus_width_bits")
    if isinstance(bw, int) and bw % 32:
        m.errors.append(
            f"mem_bus_width_bits: {m.source.get('mem_bus_width_bits')} returned {bw}, "
            f"which is not a multiple of 32 — discarded"
        )
        m.values.pop("mem_bus_width_bits", None)
        m.source.pop("mem_bus_width_bits", None)

    m.errors += _cross_checks(m)
    return m


def _src_for(readers, reader_name: str, field_name: str) -> str:
    for rname, _vals, src in readers:
        if rname == reader_name:
            return src.get(field_name) or rname
    return reader_name


#: Fields whose values are identities, not measurements. Two readers must agree
#: exactly on these — a core count that differs by 2% is not the same count.
#: Everything else numeric is a reading, and different readers round readings
#: differently: 3105 vs 3050 MHz is the same clock, reported twice.
EXACT_FIELDS = frozenset({
    "sm_count", "cuda_cores", "vram_total_bytes", "mem_bus_width_bits",
})


def _agrees(field_name: str, a: Any, b: Any) -> bool:
    """Whether two readers are describing the same value.

    Exact for identities, strings, and the fields in EXACT_FIELDS; 2% for measured
    quantities — clocks and power limits come back rounded by different readers,
    and a 55 MHz disagreement about the same clock is not a contradiction.
    """
    if isinstance(a, str) or isinstance(b, str):
        return str(a).strip().lower() == str(b).strip().lower()
    if field_name in EXACT_FIELDS:
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        hi = max(abs(float(a)), abs(float(b)))
        return hi == 0.0 or abs(float(a) - float(b)) / hi <= 0.02
    return a == b


def _cross_checks(m: MergeResult) -> List[str]:
    """Relations that must hold between fields, independent of any reader.

    These are the checks with no source to blame: they compare two measurements
    of the same silicon against each other.
    """
    out: List[str] = []

    # 1. cuda_cores / sm_count must be a real lanes-per-SM figure. Two numbers
    #    that describe the same silicon have to agree about its architecture.
    #
    #    The band above already catches the historical 16384-vs-128 mistake,
    #    because 16384 SMs is impossible. This identity catches the harder case
    #    the band cannot: a plausible-but-wrong SM count. 512 SMs would pass every
    #    range check and still be four times this card's real 128 — but 16384
    #    cores over 512 SMs is 32 lanes per SM, which no NVIDIA SM has ever had.
    cores = m.values.get("cuda_cores")
    sms = m.values.get("sm_count")
    if cores and sms:
        lanes = cores / sms
        if lanes not in LANES_PER_SM_PERMITTED:
            out.append(
                f"sm_count and cuda_cores disagree about the architecture: {cores} "
                f"cores over {sms} SMs is {lanes:g} lanes per SM, but every NVIDIA SM "
                f"has {LANES_PER_SM_PERMITTED}. One of these is being read from the "
                f"wrong function and there is no way to tell which, so neither is "
                f"published — an unpublished FLOP ceiling is recoverable, a wrong one "
                f"is not."
            )
            # Both, not one. sm_count feeds every ceiling in this module, and a
            # wrong-but-plausible SM count is more dangerous than a missing one.
            m.values.pop("cuda_cores", None)
            m.source.pop("cuda_cores", None)
            m.values.pop("sm_count", None)
            m.source.pop("sm_count", None)
    elif cores and not sms:
        out.append(
            f"cuda_cores is known ({cores}) but sm_count is not — every FLOP "
            f"ceiling stays unpublished rather than guessed; a real SM count is "
            f"the one number this module will not estimate"
        )

    # 2. Every FLOP rate must imply a clock the silicon can actually reach.
    #
    #    Expressed as a required clock rather than a TFLOPS comparison because
    #    that is the thing that is either possible or not: a rate is refuted when
    #    reaching the measured throughput would need a clock this card does not
    #    have. The clock used is the highest the card could POSSIBLY have held —
    #    the driver's maximum if any reader supplied one, otherwise the rated
    #    clock plus documented boost headroom. Judging at the rated clock alone
    #    would fire on a healthy card that boosted, and a false alarm is its own
    #    bug.
    #
    #    WHAT THIS CAN AND CANNOT PROVE. It refutes the fp32 rate decisively: the
    #    128/SM/clk the table used to carry would need 5385 MHz to reach the
    #    88.20 TFLOPS measured here, and this card tops out at 3105. It cannot
    #    refute the old fp16 rate on its own — 512/SM/clk needs only 2521 MHz,
    #    which the card can hold — and it must not be described as if it did. Once
    #    fp32 is pinned at 256, Ada's fixed precision ratios (fp16 = 2 x tf32 =
    #    4 x fp32, asserted in tests/test_truth.py) put fp16 at 1024, and the
    #    published Ada figures agree. Two independent routes, one of them a
    #    measurement, neither of them a datasheet taken on faith.
    rated = m.values.get("clock_sm_rated_mhz")
    mx = m.values.get("clock_sm_max_mhz")
    impossibly_high_clk = mx or (rated * BOOST_HEADROOM if rated else 0)
    if sms and impossibly_high_clk:
        for kind, (floor, note) in MEASURED_FLOOR_TFLOPS.items():
            rate = FLOP_PER_SM_CLK.get(kind)
            if not rate:
                continue
            required_mhz = (floor * 1e12 / (rate * sms)) / 1e6
            if required_mhz > impossibly_high_clk:
                out.append(
                    f"FLOP_PER_SM_CLK[{kind}]={rate:g} cannot be right: the "
                    f"{floor:.2f} TFLOPS measured on this box ({note}) would require "
                    f"{required_mhz:.0f} MHz across {sms} SMs, and the highest clock "
                    f"this card can reach is {impossibly_high_clk:.0f} MHz. A "
                    f"measurement cannot require a clock the silicon does not have."
                )

    # 3. A rated boost clock below the driver's max-boost clock is expected and
    #    fine (2520 vs 3105 here). It is only worth a note when the rated clock is
    #    *higher*, which would mean one of the two is not a clock at all.
    if rated and mx and rated > mx * 1.05:
        out.append(
            f"the rated boost clock ({rated} MHz) exceeds the driver's maximum "
            f"({mx} MHz) — one of these is not a clock rate"
        )
    return out


@dataclass
class GpuTruth:
    name: str = ""
    driver_version: str = ""
    cuda_driver_api: str = ""
    sm_count: int = 0
    cuda_cores: int = 0
    compute_capability: str = ""
    mem_bus_width_bits: int = 0
    vram_total_bytes: int = 0

    #: MAXIMA. The only clocks a roofline may use.
    clock_sm_max_mhz: int = 0
    clock_mem_max_mhz: int = 0
    #: INSTANTANEOUS. For a watchdog, never for a peak. An idle 4090 reads ~405 MHz
    #: on the memory clock; feeding that to bandwidth would understate it 26x.
    clock_sm_live_mhz: int = 0
    clock_mem_live_mhz: int = 0
    #: The RATED boost clock, read from the device as CUDA CLOCK_RATE (2520 MHz on
    #: this box, matching the spec sheet's 2.52 GHz boost). This is the clock a peak
    #: is computed at. NVML's max-boost figure (3105 MHz) is HIGHER but is a clock
    #: the driver may request, not one a kernel sustains — a 4090 under a sustained
    #: GEMM sits near 2690 MHz. Computing a roofline against 3105 gives a 407 TFLOPS
    #: FP16 ceiling that nothing can reach and that makes every kernel look worse
    #: than it is. A ceiling is only useful if the silicon can touch it.
    clock_sm_rated_mhz: int = 0

    power_limit_w: float = 0.0
    power_limit_min_w: float = 0.0
    power_limit_max_w: float = 0.0
    hags_enabled: Optional[bool] = None      # HwSchMode==2; load-bearing on WDDM 3.2
    resizable_bar: Optional[bool] = None

    #: The memory clock the bandwidth figure actually used, so a reader can see
    #: which quantity produced the number rather than trusting the field name.
    mem_clock_used_mhz: int = 0

    # provenance — one entry per field that could plausibly be wrong
    source: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    #: Fields two independent readers contradicted about. Never silently resolved.
    disagreements: List[str] = field(default_factory=list)
    #: Which readers answered at all, so "everything reads zero" is diagnosable.
    readers: List[str] = field(default_factory=list)
    #: A reader that could not answer a field ANOTHER reader did publish. Not a
    #: failure about a value — the field was read — but kept, because it means the
    #: published value had one fewer reader able to contradict it.
    notes: List[str] = field(default_factory=list)

    # ---- derived ---------------------------------------------------------
    @property
    def sm_clock_for_peak_mhz(self) -> int:
        """The clock a peak is computed at: the RATED boost clock.

        Prefers CUDA's CLOCK_RATE (2520 MHz here — the spec-sheet boost figure,
        read from the device rather than cited). Falls back to NVML's max-boost
        (3105 MHz) only when the rated clock is unknown, because an over-stated
        ceiling is less wrong than no ceiling at all — but it is recorded in
        `.source`, so a reader can see which was used.

        Live clocks are never candidates for this, at any precedence.
        """
        return self.clock_sm_rated_mhz or self.clock_sm_max_mhz

    @property
    def mem_bandwidth_gbs(self) -> float:
        """GDDR6X: bus_width_bytes * effective_clock. `mem_clock_max_mhz` is the
        effective data-rate clock (10501 MHz on this box), so x2 for DDR gives
        21 Gbps per pin and 1008 GB/s across a 384-bit bus — which is the
        published figure, derived rather than copied.

        Uses MAX only. Live memory clocks fall to ~405 MHz at idle; a bandwidth
        derived from one would be 39 GB/s and would look perfectly reasonable.
        """
        if not (self.mem_bus_width_bits and self.clock_mem_max_mhz):
            return 0.0
        return (self.mem_bus_width_bits / 8.0) * self.clock_mem_max_mhz * 2.0 * 1e6 / 1e9

    def peak_tflops(self, kind: str = "fp16_tensor") -> float:
        rate = FLOP_PER_SM_CLK.get(kind, 0.0)
        if not (rate and self.sm_count and self.sm_clock_for_peak_mhz):
            return 0.0
        return rate * self.sm_count * (self.sm_clock_for_peak_mhz * 1e6) / 1e12

    def ridge_point(self, kind: str = "fp16_tensor") -> float:
        """Arithmetic intensity at which a kernel stops being memory-bound."""
        bw = self.mem_bandwidth_gbs
        return (self.peak_tflops(kind) * 1e12 / (bw * 1e9)) if bw else 0.0

    @property
    def complete(self) -> bool:
        """Whether enough was read for the roofline to mean anything."""
        return bool(self.sm_count and self.sm_clock_for_peak_mhz
                    and self.mem_bus_width_bits and self.clock_mem_max_mhz)

    def values_read(self) -> List[str]:
        """Field names a reader actually supplied — the complement of `.errors`.

        Exists so a report can be honest about which numbers are readings and
        which are still sitting at their default. The old status line printed
        "0 SM @ 0 MHz" without distinguishing the two, which is how a failed read
        passed for a measurement.
        """
        return [k for k, src in self.source.items()
                if getattr(self, k, None) not in (None, "", 0)]

    @property
    def summary(self) -> str:
        if not self.complete:
            missing = [n for n, ok in (
                ("SM count", self.sm_count),
                ("SM clock", self.sm_clock_for_peak_mhz),
                ("bus width", self.mem_bus_width_bits),
                ("memory clock", self.clock_mem_max_mhz),
            ) if not ok]
            return (f"{self.name or 'GPU'} | driver {self.driver_version or '?'} | "
                    f"incomplete: no {', '.join(missing)} — via "
                    f"{', '.join(self.readers) or 'no reader'}")
        return (
            f"{self.name} | driver {self.driver_version} | CC {self.compute_capability} | "
            f"{self.sm_count} SM @ {self.sm_clock_for_peak_mhz} MHz | "
            f"{self.vram_total_bytes / 2**30:.1f} GiB | "
            f"{self.mem_bandwidth_gbs:.0f} GB/s"
        )


@dataclass
class RooflineTruth:
    """V9's roofline, when it is importable. `delegated` says which engine
    produced these numbers, so a reader never has to guess."""
    delegated: bool = False
    bound: str = ""                # "compute" | "memory" | "balanced" | "unknown"
    saturation_ceiling: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        who = "v9.hwmodel" if self.delegated else "v12.truth (local derivation)"
        return f"roofline via {who}: bound={self.bound or 'unknown'}"


def _reg_read(root, path: str) -> Optional[Any]:
    try:
        import winreg
        with winreg.OpenKey(root, path) as k:
            return winreg.QueryValueEx(k, "")[0]
    except Exception:
        return None


def _hags_enabled() -> Optional[bool]:
    """HwSchMode==2 at HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers.
    Reads from the 64-bit view explicitly — a 32-bit Python would otherwise see
    a redirected key and report the wrong answer."""
    try:
        import winreg
        # HwSchMode is a named value, not the default
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers",
            0,
            winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0),
        ) as k:
            raw = winreg.QueryValueEx(k, "HwSchMode")[0]
        return int(raw) == 2
    except Exception:
        return None


def _resizable_bar() -> Optional[bool]:
    """Best-effort: nvidia-smi reports it in some driver branches only."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-q"], capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        for line in out.splitlines():
            if "Resizable BAR" in line:
                return "Yes" in line
    except Exception:
        pass
    return None


def gpu_truth() -> GpuTruth:
    """Read the GPU from every reader that will answer, then merge.

    Never raises. Fields that no reader could establish stay at their zero value
    AND appear in `.errors`, because a zero that is indistinguishable from a
    failed read is worse than a missing key — that is precisely how the
    "0 SM @ 0 MHz" status line went unnoticed.
    """
    readers: List[Tuple[str, Dict[str, Any], Dict[str, str]]] = []
    raw_errors: List[str] = []
    for name, fn in (("nvml", _read_nvml), ("cuda", _read_cuda), ("smi", _read_smi)):
        try:
            vals, src, errs = fn()
        except Exception as e:                        # a reader must not kill the read
            vals, src, errs = {}, {}, [f"{name}: reader crashed: {e}"]
        readers.append((name, vals, src))
        raw_errors += errs

    m = _merge(readers)
    t = GpuTruth()
    for k, v in m.values.items():
        if hasattr(t, k):
            setattr(t, k, v)
    t.source = dict(m.source)
    t.disagreements = m.disagreements
    t.readers = [n for n, vals, _ in readers if vals]

    # The clock the bandwidth actually used, recorded rather than inferred.
    if "clock_mem_max_mhz" in m.values:
        t.mem_clock_used_mhz = int(m.values["clock_mem_max_mhz"])

    # A reader's "reader:field: ..." miss is a field failure only if the field
    # stayed unpublished. Keyed on what the MERGE published, not on what some
    # reader supplied: a value dropped by a band, the identity, or a disagreement
    # leaves the field unread, and then this miss is part of why — it stays an
    # error. Reader-level failures ("nvml: nvmlInit failed") always stay errors.
    t.errors = []
    for e in raw_errors:
        reader, _, rest = e.partition(":")
        field_name, sep, _msg = rest.partition(":")
        if sep and reader in PRECEDENCE.get(field_name, ()) and field_name in m.values:
            t.notes.append(e)
        else:
            t.errors.append(e)
    t.errors += m.errors

    # Every field the summary promises but nobody supplied, named individually.
    for field_name, label in (("sm_count", "SM count"), ("name", "device name"),
                              ("driver_version", "driver version"),
                              ("clock_sm_rated_mhz", "rated boost clock"),
                              ("clock_sm_max_mhz", "max SM clock"),
                              ("mem_bus_width_bits", "memory bus width"),
                              ("clock_mem_max_mhz", "max memory clock"),
                              ("vram_total_bytes", "VRAM size")):
        if not getattr(t, field_name, None):
            t.errors.append(f"{label} unread: no reader answered "
                            f"({', '.join(t.readers) or 'none'})")

    t.hags_enabled = _hags_enabled()
    if t.hags_enabled is not None:
        t.source["hags_enabled"] = "registry:HKLM\\...\\GraphicsDrivers\\HwSchMode"
    t.resizable_bar = _resizable_bar()
    if t.resizable_bar is not None:
        t.source["resizable_bar"] = "nvidia-smi -q"
    return t


def roofline() -> RooflineTruth:
    """Prefer V9's validated model. Say which one answered."""
    try:
        import sys
        from pathlib import Path
        v9root = Path(paths.REPO_ROOT)
        if str(v9root) not in sys.path:
            sys.path.insert(0, str(v9root))
        from v9 import hwmodel  # type: ignore
        r = hwmodel.derive_roofline()
        return RooflineTruth(
            delegated=True,
            bound=getattr(r, "bound", "") or "",
            saturation_ceiling={"value": float(getattr(r, "saturation_ceiling", 0.0) or 0.0)},
            notes=["numbers produced by v9.hwmodel.derive_roofline (microbench-validated)"],
        )
    except Exception as e:
        log.debug("v9.hwmodel unavailable (%s) — deriving locally", e)

    t = gpu_truth()
    notes = [f"v9.hwmodel not importable; local derivation from {', '.join(t.readers) or 'no reader'}"]
    if not t.complete:
        notes.append(
            f"insufficient live data ({t.sm_count} SM, {t.sm_clock_for_peak_mhz} MHz, "
            f"{t.mem_bus_width_bits}-bit bus, {t.clock_mem_max_mhz} MHz mem) — "
            f"bound=unknown rather than a guess"
        )
        return RooflineTruth(delegated=False, bound="unknown", notes=notes)
    return RooflineTruth(
        delegated=False,
        bound="unknown",
        saturation_ceiling={
            "fp16_tflops": t.peak_tflops("fp16_tensor"),
            "fp8_tflops": t.peak_tflops("fp8_tensor"),
            "fp32_tflops": t.peak_tflops("fp32_cuda"),
            "bandwidth_gbs": t.mem_bandwidth_gbs,
            "ridge_fp16": t.ridge_point("fp16_tensor"),
            "ridge_fp8": t.ridge_point("fp8_tensor"),
        },
        notes=notes,
    )


def describe(as_json: bool = False) -> str:
    t = gpu_truth()
    r = roofline()
    if as_json:
        return json.dumps(
            {"gpu": asdict(t), "roofline": asdict(r), "backends": BACKEND_REACHABLE},
            indent=2, default=str,
        )
    lines = [t.summary, r.summary]
    if t.complete:
        lines.append(
            f"peak @ {t.sm_clock_for_peak_mhz} MHz rated boost "
            f"({t.source.get('clock_sm_rated_mhz', 'source unknown')}; driver max "
            f"{t.clock_sm_max_mhz} MHz, which no sustained kernel holds):"
        )
        for kind, (floor, note) in MEASURED_FLOOR_TFLOPS.items():
            peak = t.peak_tflops(kind)
            pct = (floor / peak * 100.0) if peak else 0.0
            over = ("  (over 100%: the card boosts above its rated clock — the rated "
                    "figure is a guaranteed minimum, not a cap)") if pct > 100 else ""
            lines.append(
                f"  {kind:<14} {peak:7.1f} TFLOPS ceiling  — measured {floor:6.2f} "
                f"= {pct:4.1f}% of ceiling{over}"
            )
            lines.append(f"  {'':<14} floor source: {note}")
        for kind in ("fp8_tensor", "fp32_cuda"):
            if kind in MEASURED_FLOOR_TFLOPS:
                continue
            lines.append(f"  {kind:<14} {t.peak_tflops(kind):7.1f} TFLOPS ceiling  "
                         f"— no measurement on record; treat as a bound, not a target")
        lines.append(f"  {'bandwidth':<14} {t.mem_bandwidth_gbs:7.1f} GB/s    "
                     f"({t.mem_bus_width_bits}-bit x {t.mem_clock_used_mhz} MHz x2)")
        lines.append(f"  {'ridge fp16':<14} {t.ridge_point('fp16_tensor'):7.1f} FLOP/byte "
                     f"— below this a kernel cannot be compute-bound")
    if t.hags_enabled is not None:
        lines.append(f"HAGS: {'ON' if t.hags_enabled else 'OFF'}"
                     + ("  (load-bearing on WDDM 3.2 — DLSS FG / flip queue)" if t.hags_enabled else
                        "  (WARNING: DLSS Frame Gen, Smooth Motion, flip queue fail silently when off)"))
    # The power ceiling and the range it may be moved within. Read but never
    # published before this: the operator could not see that the card was
    # sitting 150 W below what it would accept. Whether raising it pays depends
    # entirely on the workload — this box's Karma render draws 104 W of 450 W
    # (0% at the cap), while fp16 GEMM sits AT the cap — so the headroom is
    # only meaningful next to a measurement, but it has to be visible to be
    # worth measuring.
    if t.power_limit_w:
        lo = t.power_limit_min_w or 0.0
        hi = t.power_limit_max_w or 0.0
        head = ""
        if hi and hi > t.power_limit_w + 1.0:
            head = (f"  — {hi - t.power_limit_w:.0f} W of headroom is available "
                    f"(this card accepts up to {hi:.0f} W)")
        lines.append(f"power ceiling: {t.power_limit_w:.0f} W"
                     + (f" (allowed {lo:.0f}-{hi:.0f} W)" if hi else "") + head)
    if t.disagreements:
        lines.append("")
        lines.append("READERS DISAGREE — nothing was published for these fields:")
        for d in t.disagreements:
            lines.append(f"  ! {d}")
    if t.errors:
        lines.append("")
        lines.append("unreadable / discarded:")
        for e in t.errors:
            lines.append(f"  - {e}")
    if t.notes:
        lines.append("")
        lines.append("read by one reader, missed by another (published; one fewer cross-check):")
        for n in t.notes:
            lines.append(f"  . {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
