"""Thermal / power agent. Honest about consumer-GeForce + WDDM limits: the watchdog
does the real runtime work (telemetry + dynamic -lgc clock cap + back-off, wired in
the orchestrator). This agent surfaces the two high-value MANUAL levers (MSI
Afterburner undervolt + fan curve, with exact starting values) and the availability
facts, since none of that can be safely scripted from a .bat."""
from __future__ import annotations

from typing import List

from ..base import Agent, Context, Proposal, Risk
from ..builders import guided, prop


class ThermalPowerAgent(Agent):
    name = "thermal_power"
    domain = "thermal_power"

    def propose(self, ctx: Context) -> List[Proposal]:
        out: List[Proposal] = []

        out.append(prop(
            "gpu_undervolt", "RTX 4090 undervolt (MSI Afterburner, manual)", self.domain,
            why="A flat V/F cap (~0.925V @ ~2650MHz) holds near-stock performance at markedly "
                "lower power/heat/noise — ideal next to a quiet color-review station. Can't be "
                "authored by a script; the watchdog can only invoke a saved profile.",
            impact="cooler, quieter sustained Karma/UE renders at ~stock speed.", confidence=0.85,
            changes=[guided(
                "gpu_undervolt", "Afterburner Curve Editor: ~0.925V @ ~2650MHz flat cap",
                "MSI Afterburner > Curve Editor (Ctrl+F): drag the ~900-950mV point to "
                "~2600-2700MHz, flatten everything to the right, Apply. Validate with a 20-30min "
                "Karma XPU / UE render. Save to Profile 1 and enable 'Apply at startup'. Tune per "
                "silicon. Revert: 'Reset' + Apply, disable 'Apply at startup'.")]))

        out.append(prop(
            "gpu_fan_temp_limit", "Custom fan curve + temp limit (Afterburner, manual)", self.domain,
            why="A more aggressive fan curve (100% by ~85C) with a ~80-83C temp-limit priority "
                "keeps the 4090 off its thermal ceiling during long renders.",
            impact="lower sustained temps; protects boost residency.", confidence=0.85,
            changes=[guided(
                "gpu_fan_temp_limit", "Afterburner fan curve to 100% by 85C; Temp Limit ~80-83",
                "Afterburner > Settings > Fan: user-defined curve (60% by 70C, 80% by 80C, 100% by "
                "85C); keep zero-RPM idle for acoustics. Optionally lower Temp Limit to ~80. Save "
                "to a profile. Revert: Fan = Auto, Temp Limit = default 83.")]))

        out.append(prop(
            "power_lever_facts", "What is/ isn't scriptable on this GeForce+WDDM box", self.domain,
            why="Set expectations honestly so no one wastes time on locked knobs.",
            impact="clarity — prevents chasing unavailable controls.", confidence=0.9,
            changes=[guided(
                "power_lever_facts", "GeForce + WDDM power/clock reality",
                "AVAILABLE (the watchdog uses these): nvidia-smi -lgc / -rgc (lock/reset GPU core "
                "clock) — the dynamic thermal cap. NOT available on this box: -pl (power limit, "
                "board-locked), -lmc (memory clock lock), -pm (persistence, WDDM N/A), -ac "
                "(app clocks, Tesla-only). Undervolt/fan = Afterburner only (manual). The watchdog "
                "monitors temp/power/throttle at 1 Hz and steps the clock cap under thermal load.")]))
        return out
