"""
throttle — what fraction of a run sat at a clock-event limit.

The 4090 logged 6.27 h of SW power capping in 5.7 days (read 2026-11-03). That
number is useless per workload until it is read as a DELTA around one run.
`Snapshot.before()` ... run ... `Snapshot.after()` gives:

    power_cap_frac   fraction of wall time the SW power cap held clocks down
    thermal_frac     fraction under HW/SW thermal slowdown
    avg_power_w      from the energy counter, not a polled average
    sm_clock_mhz     live at the after-snapshot (informational)

Reads NVML through pynvml when present, else parses `nvidia-smi -q -d PERFORMANCE`.
Neither path writes anything to the GPU.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    import pynvml  # type: ignore
    _HAVE_NVML = True
except Exception:  # pragma: no cover
    pynvml = None
    _HAVE_NVML = False

# NVML perf-policy ids (nvml.h): POWER=0, THERMAL=1, SYNC_BOOST=2, BOARD_LIMIT=3,
# LOW_UTILIZATION=4, RELIABILITY=5, TOTAL_APP_CLOCKS=10, TOTAL_BASE_CLOCKS=11
_POLICIES = {"power": 0, "thermal": 1, "sync_boost": 2, "board_limit": 3,
             "reliability": 5}


@dataclass
class Snapshot:
    t_wall: float
    violation_us: Dict[str, int] = field(default_factory=dict)
    energy_mj: Optional[int] = None
    sm_clock_mhz: Optional[int] = None
    mem_clock_mhz: Optional[int] = None
    power_limit_w: Optional[float] = None
    source: str = ""

    @classmethod
    def take(cls, index: int = 0) -> "Snapshot":
        if _HAVE_NVML:
            try:
                return cls._nvml(index)
            except Exception:
                pass
        return cls._smi()

    @classmethod
    def _nvml(cls, index: int) -> "Snapshot":
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(index)
            v: Dict[str, int] = {}
            for name, pid in _POLICIES.items():
                try:
                    st = pynvml.nvmlDeviceGetViolationStatus(h, pid)
                    v[name] = int(st.violationTime)  # ns on recent drivers? see note below
                except Exception:
                    pass
            energy = None
            try:
                energy = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
            except Exception:
                pass
            sm = mem = None
            try:
                sm = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                mem = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
            except Exception:
                pass
            pl = None
            try:
                pl = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            except Exception:
                pass
            return cls(time.perf_counter(), v, energy, sm, mem, pl, "nvml")
        finally:
            pynvml.nvmlShutdown()

    @classmethod
    def _smi(cls) -> "Snapshot":
        out = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE,CLOCK,POWER"],
                             capture_output=True, text=True, timeout=30).stdout
        v: Dict[str, int] = {}
        for label, key in (("SW Power Capping", "power"), ("SW Thermal Slowdown", "thermal"),
                           ("HW Thermal Slowdown", "hw_thermal"), ("Sync Boost", "sync_boost"),
                           ("HW Power Braking", "hw_power_brake")):
            m = re.search(rf"{label}\s*:\s*(\d+) us", out)
            if m:
                v[key] = int(m.group(1)) * 1000  # store as ns to match NVML path
        sm = _grab(out, r"\n\s+SM\s*:\s*(\d+) MHz")
        mem = _grab(out, r"\n\s+Memory\s*:\s*(\d+) MHz")
        pl = _grab(out, r"Current Power Limit\s*:\s*([\d.]+) W", float)
        return cls(time.perf_counter(), v, None, sm, mem, pl, "nvidia-smi")


def _grab(text: str, pat: str, cast=int):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None


def delta(a: Snapshot, b: Snapshot) -> Dict[str, float]:
    """Fractions of wall time between two snapshots spent under each limit.

    NOTE on units: nvidia-smi prints the counters in microseconds; NVML's
    violationTime is documented in nanoseconds on current drivers. The smi path
    converts to ns so both agree. If a fraction ever exceeds 1.0 the unit
    assumption is wrong for this driver — the value is reported, not clamped,
    so that shows up instead of hiding."""
    wall_ns = max((b.t_wall - a.t_wall) * 1e9, 1.0)
    out: Dict[str, float] = {"wall_s": wall_ns / 1e9, "source": a.source}  # type: ignore
    for k in set(a.violation_us) | set(b.violation_us):
        d = b.violation_us.get(k, 0) - a.violation_us.get(k, 0)
        out[f"{k}_frac"] = d / wall_ns
    if a.energy_mj is not None and b.energy_mj is not None and b.energy_mj >= a.energy_mj:
        out["avg_power_w"] = (b.energy_mj - a.energy_mj) / 1000.0 / (wall_ns / 1e9)
    if b.sm_clock_mhz is not None:
        out["sm_clock_mhz_after"] = float(b.sm_clock_mhz)
    if b.power_limit_w is not None:
        out["power_limit_w"] = b.power_limit_w
    return out


def describe(d: Dict[str, float]) -> str:
    bits = []
    if "power_frac" in d:
        bits.append(f"at power cap {d['power_frac']:.0%}")
    for k in ("thermal_frac", "hw_thermal_frac"):
        if k in d and d[k] > 0:
            bits.append(f"thermal {d[k]:.0%}")
    if "avg_power_w" in d:
        bits.append(f"avg {d['avg_power_w']:.0f} W")
    if "power_limit_w" in d:
        bits.append(f"limit {d['power_limit_w']:.0f} W")
    return ", ".join(bits) or "no throttle counters read"


if __name__ == "__main__":
    import sys
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    a = Snapshot.take()
    print(f"watching {secs:.0f}s via {a.source} ... run something on the GPU now")
    time.sleep(secs)
    b = Snapshot.take()
    print(describe(delta(a, b)))
