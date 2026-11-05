"""
V12 — power measurement that does not lie about its own resolution.

This module exists because of a mistake made during V12's own research, and the
mistake is worth preserving as documentation.

While validating the `sw_power_cap` throttle finding on this box, power was
sampled at 0.4 s intervals and reported as "421 W average (94% of limit)". That
number was then described as "solid" while only the clock figure was flagged as
possibly aliased. Both were aliased. `nvmlDeviceGetPowerUsage()` on Ampere and
later (GA100 excepted) returns a **trailing 1-second boxcar average** — it is not
an instantaneous reading. Sampling it every 400 ms samples the same one-second
window two and a half times, so the series looks smooth and high-resolution and
is neither.

The fix is `NVML_FI_DEV_POWER_INSTANT` (field id 186) read through
`nvmlDeviceGetFieldValues()`, which reports the actual sensor value with a
documented ~100 ms floor on Ada. This module always prefers it, always records
which sensor answered, and never lets the two be confused:

    PowerReading.instant  -> True  means a real point sample
                          -> False means a trailing 1 s mean, and `window_ms` says so

A caller that needs a mean over T seconds must integrate the INSTANT series, not
sample the AVERAGE sensor — those are different measurements and only one of them
is honest about what it covers.
"""
from __future__ import annotations

import ctypes
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

log = logging.getLogger("v12.power")

NVML_SUCCESS = 0
NVML_FI_DEV_POWER_INSTANT = 186
NVML_FI_DEV_POWER_AVERAGE = 187
NVML_VALUE_TYPE_DOUBLE = 2

#: Ada's documented floor for the instant sensor. Sampling faster than this
#: returns the same value repeatedly — which is not an error, but a caller that
#: does not know it will compute confidence intervals on duplicated samples.
INSTANT_RESOLUTION_MS = 100.0
#: The trailing-mean sensor's window. Not configurable; it is what the hardware does.
AVERAGE_WINDOW_MS = 1000.0


@dataclass
class PowerReading:
    watts: float
    instant: bool
    source: str                     # "nvml:field186" | "nvml:GetPowerUsage"
    window_ms: float = 0.0          # 0 for a true point sample
    t: float = field(default_factory=time.monotonic)

    @property
    def resolution_ms(self) -> float:
        return INSTANT_RESOLUTION_MS if self.instant else AVERAGE_WINDOW_MS

    def __str__(self) -> str:
        if self.instant:
            return f"{self.watts:.1f} W (instant, ~{INSTANT_RESOLUTION_MS:.0f} ms sensor)"
        return f"{self.watts:.1f} W ({AVERAGE_WINDOW_MS/1000:.0f} s trailing mean — NOT a point sample)"


class PowerSampler:
    """Reads GPU power honestly.

    Use `read()` for a single honest value. Use `series()` when you want a mean
    over a window — it samples at the sensor's own rate and integrates, rather
    than sampling a boxcar faster than the boxcar updates.
    """

    def __init__(self, index: int = 0, limit_w: Optional[float] = None):
        self.index = index
        self.limit_w = limit_w
        self._p = None
        self._h = None
        self._instant_ok: Optional[bool] = None   # None = not yet probed
        self._last_error = ""

    # ---- lifecycle -------------------------------------------------------
    def open(self) -> bool:
        try:
            import pynvml
        except ImportError:
            self._last_error = "nvidia-ml-py not importable"
            return False
        try:
            pynvml.nvmlInit()
            self._p = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        except Exception as e:
            self._last_error = f"nvmlInit/handle: {e}"
            return False
        if self.limit_w is None:
            try:
                self.limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(self._h) / 1000.0
            except Exception:
                self.limit_w = None
        return True

    def close(self) -> None:
        try:
            if self._p is not None:
                self._p.nvmlShutdown()
        except Exception:
            pass
        self._p = self._h = None

    def __enter__(self) -> "PowerSampler":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- the two sensors -------------------------------------------------
    def _read_instant(self) -> Optional[float]:
        """NVML_FI_DEV_POWER_INSTANT via nvmlDeviceGetFieldValues. Returns None
        when the driver does not support it (then we fall back, and SAY SO)."""
        p, h = self._p, self._h
        if p is None or h is None:
            return None
        try:
            fv = p.c_nvmlFieldValue_t()
            fv.fieldId = NVML_FI_DEV_POWER_INSTANT
            fv.scopeId = 0
            arr = (p.c_nvmlFieldValue_t * 1)(fv)
            p.nvmlDeviceGetFieldValues(h, arr)
            got = arr[0]
            if int(getattr(got, "nvmlReturn", 1)) != NVML_SUCCESS:
                self._last_error = f"field186 nvmlReturn={getattr(got, 'nvmlReturn', '?')}"
                return None
            v = got.value
            # The union member depends on valueType; 186 is DOUBLE.
            for attr in ("dVal", "flVal", "dbVal"):
                if hasattr(v, attr):
                    try:
                        return float(getattr(v, attr))
                    except Exception:
                        continue
            return None
        except Exception as e:
            self._last_error = f"field186: {e}"
            return None

    def _read_average(self) -> Optional[float]:
        try:
            return float(self._p.nvmlDeviceGetPowerUsage(self._h)) / 1000.0
        except Exception as e:
            self._last_error = f"GetPowerUsage: {e}"
            return None

    def read(self) -> Optional[PowerReading]:
        if self._p is None and not self.open():
            return None
        if self._instant_ok is not False:          # None (unprobed) or True
            w = self._read_instant()
            if w is not None:
                if self._instant_ok is None:
                    log.info("power: using NVML_FI_DEV_POWER_INSTANT (field 186), ~%.0f ms",
                             INSTANT_RESOLUTION_MS)
                    self._instant_ok = True
                return PowerReading(watts=w, instant=True, source="nvml:field186", window_ms=0.0)
            if self._instant_ok is None:
                log.warning(
                    "power: NVML_FI_DEV_POWER_INSTANT unavailable (%s); falling back to "
                    "GetPowerUsage, which is a %.0f s TRAILING MEAN. Values will look "
                    "smoother than reality and short spikes will be invisible.",
                    self._last_error, AVERAGE_WINDOW_MS / 1000,
                )
                self._instant_ok = False
        w = self._read_average()
        if w is None:
            return None
        return PowerReading(watts=w, instant=False, source="nvml:GetPowerUsage",
                            window_ms=AVERAGE_WINDOW_MS)

    # ---- aggregations ----------------------------------------------------
    def series(self, seconds: float, on_sample: Optional[Callable[[PowerReading], None]] = None
               ) -> List[PowerReading]:
        """Sample for `seconds` at the sensor's own rate.

        Deliberately does NOT let the caller choose a faster interval: asking for
        more samples than the sensor produces is exactly how the original
        aliasing error happened. On the instant sensor we poll at the 100 ms
        floor; on the fallback we poll at 1 s, because polling a 1 s boxcar
        faster than 1 s yields duplicates that inflate n and shrink any stderr
        computed from them.
        """
        out: List[PowerReading] = []
        first = self.read()
        if first is None:
            return out
        interval = first.resolution_ms / 1000.0
        deadline = time.monotonic() + seconds
        out.append(first)
        if on_sample:
            on_sample(first)
        while time.monotonic() < deadline:
            time.sleep(interval)
            r = self.read()
            if r is None:
                continue
            out.append(r)
            if on_sample:
                on_sample(r)
        return out

    @staticmethod
    def summarise(readings: List[PowerReading], limit_w: Optional[float] = None) -> dict:
        if not readings:
            return {"n": 0, "note": "no readings"}
        vals = [r.watts for r in readings]
        instant = readings[0].instant
        d = {
            "n": len(vals),
            "mean_w": statistics.fmean(vals),
            "max_w": max(vals),
            "min_w": min(vals),
            "stdev_w": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "sensor": readings[0].source,
            "instant": instant,
            "window_ms": readings[0].window_ms,
        }
        if limit_w:
            d["pct_of_limit"] = 100.0 * d["mean_w"] / limit_w
        if not instant:
            # Say the quiet part: a stdev from duplicated boxcar samples is
            # meaningless, and a reader must not treat it as noise.
            d["caveat"] = (
                f"sampled from a {AVERAGE_WINDOW_MS/1000:.0f} s trailing mean; stdev is "
                f"not sensor noise and short transients are invisible"
            )
        return d

    # ---- throttle reasons ------------------------------------------------
    def throttle_reasons(self) -> dict:
        """`nvmlDeviceGetCurrentClocksThrottleReasons` is a bitmask, so this is
        NOT subject to the aliasing problem above — it reports state, not an
        average. The `sw_power_cap` finding (0x4) stands regardless."""
        bits = {
            0x0000000000000001: "gpu_idle",
            0x0000000000000002: "applications_clocks_setting",
            0x0000000000000004: "sw_power_cap",
            0x0000000000000008: "hw_slowdown",
            0x0000000000000010: "sync_boost",
            0x0000000000000020: "sw_thermal_slowdown",
            0x0000000000000040: "hw_thermal_slowdown",
            0x0000000000000080: "hw_power_brake_slowdown",
            0x0000000000000100: "display_clock_setting",
        }
        if self._p is None:
            return {}
        try:
            v = int(self._p.nvmlDeviceGetCurrentClocksThrottleReasons(self._h))
        except Exception as e:
            self._last_error = str(e)
            return {}
        return {name: True for bit, name in bits.items() if v & bit}


def read_once(index: int = 0) -> Optional[PowerReading]:
    with PowerSampler(index=index) as s:
        return s.read()


if __name__ == "__main__":
    with PowerSampler() as s:
        r = s.read()
        print("power:", r if r else "unavailable")
        if s.limit_w:
            print(f"limit: {s.limit_w:.0f} W")
        print("throttle:", s.throttle_reasons() or "none")
        print("2 s series ->", PowerSampler.summarise(s.series(2.0), s.limit_w))
