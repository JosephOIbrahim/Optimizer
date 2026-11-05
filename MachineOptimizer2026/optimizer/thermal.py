"""Runtime thermal governor for the watchdog.

The only power/clock write that actually works on this consumer GeForce under WDDM is
`nvidia-smi -lgc / -rgc` (lock/reset GPU core clock) — verified, reversible, clears on
reboot. -pl / -lmc / -pm are unavailable here. The governor steps the core-clock cap
down under thermal load and releases it (with hysteresis) on cooldown. It never touches
memory clocks, power limit, or the color-critical display.
"""
from __future__ import annotations

from typing import Optional

from .executor import run_ps


class ThermalGovernor:
    def __init__(self, cfg: dict, safety, dry_run: bool, is_admin: bool):
        self.reduce = int(cfg["thermal_reduce"])
        self.critical = int(cfg["thermal_critical"])
        self.safety = safety
        self.dry_run = dry_run
        self.is_admin = is_admin
        self.cap: Optional[int] = None   # None = unlocked, else capped MHz
        self._cool = 0                    # consecutive cool samples (hysteresis)

    def _lock(self, mhz: int) -> None:
        if not self.dry_run and self.is_admin:
            run_ps(f"nvidia-smi -lgc 0,{mhz}")

    def _reset(self) -> None:
        if not self.dry_run and self.is_admin:
            run_ps("nvidia-smi -rgc")

    def step(self, ctx) -> Optional[str]:
        """Evaluate one telemetry sample; apply/adjust the cap. Returns a status string
        for anything the user should see, else None."""
        t = ctx.temp_c
        thr = ctx.throttle_reasons
        thermal_active = any("thermal" in r for r in thr)

        if t >= self.critical or "hw_slowdown" in thr:
            self.safety.event("thermal_critical", {"temp": t, "throttle": thr})
            return f"CRITICAL {t}C"
        if t >= 89:
            if self.cap != 2100:
                self._lock(2100); self.cap = 2100
                self.safety.event("clock_cap", {"mhz": 2100, "temp": t})
                self._cool = 0
                return f"clock cap 2100MHz ({t}C)"
            self._cool = 0
            return None
        if t >= self.reduce or thermal_active:
            if self.cap not in (2100, 2400):
                self._lock(2400); self.cap = 2400
                self.safety.event("clock_cap", {"mhz": 2400, "temp": t})
                self._cool = 0
                return f"clock cap 2400MHz ({t}C)"
            self._cool = 0
            return None
        if t <= 80 and self.cap is not None:
            self._cool += 1
            if self._cool >= 1:   # watchdog interval (>=15s) provides the hysteresis dwell
                self._reset(); self.cap = None; self._cool = 0
                self.safety.event("clock_release", {"temp": t})
                return f"clock cap released ({t}C)"
        return None

    def release_on_exit(self) -> None:
        if self.cap is not None:
            self._reset()
            self.cap = None
