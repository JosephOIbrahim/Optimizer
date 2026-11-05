"""V9 — System-integration config-tuning advisor.

Ported from v6's SystemIntegrator. Covers the Windows/host-level side-effects:
PCIe-gen + Resizable-BAR validation, the Windows hardware-GPU-scheduler
(HAGS) registry value, thermal throttling guidance, per-process priority, and
Threadripper CPU affinity pinning.

Every hardware fact (PCIe gen, ReBAR state, GPU temp, core counts) is read
from ctx / ctx.profile — never hardcoded. Tuning policy (which priority, which
cores, warning thresholds) comes from the config `system` section.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from v9.advisors.base import Advisor
from v9.recommend import Action, Recommendation

if TYPE_CHECKING:
    from v9.context import SystemContext

# Process-name substrings that map to the apps we care about pinning/priority.
_PRIORITY_PROC_MATCH = {
    "houdini": ("houdini", "hindie", "husk"),
    "ue5": ("unrealeditor", "ue5editor", "ue4editor", "unreal"),
}
# Apps that get pinned to performance cores (heavy, latency-sensitive VFX).
_PERF_CORE_APPS = ("houdini", "ue5")

# Windows hardware-accelerated GPU scheduling (HAGS).
_HAGS_KEY = (r"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control"
             r"\GraphicsDrivers")
_HAGS_VALUE = "HwSchMode"
_HAGS_ENABLED = 2


class SystemAdvisor(Advisor):
    name = "system"
    min_confidence = 0.85

    def __init__(self, config=None, cache_handle=None, quality_ledger=None):
        super().__init__(config, cache_handle, quality_ledger)
        self.process_priority: dict[str, str] = (
            self.config.get("process_priority") or {
                "houdini": "ABOVE_NORMAL",
                "ue5": "ABOVE_NORMAL",
                "comfyui": "NORMAL",
            })
        thermal = self.config.get("thermal") or {}
        self.temp_warning_c = float(thermal.get("gpu_temp_warning_c", 83))
        self.temp_critical_c = float(thermal.get("gpu_temp_critical_c", 90))
        affinity = self.config.get("affinity") or {}
        self.enable_pinning = bool(affinity.get("enable_pinning", True))
        self.performance_cores: list[int] = list(
            affinity.get("performance_cores") or [])
        self.background_cores: list[int] = list(
            affinity.get("background_cores") or [])
        self.want_hags = bool(self.config.get("windows_gpu_scheduler", True))

    # ------------------------------------------------------------------
    def _analyze(self, ctx: "SystemContext") -> list[Recommendation]:
        recs: list[Recommendation] = []
        gpu = ctx.profile.gpu if ctx.profile else None

        # ---- PCIe generation (detected, tri-state) ------------------------
        if gpu and gpu.pcie_gen is not None and gpu.pcie_gen < 4:
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.NOTIFY,
                target="pcie_link_generation",
                confidence=1.0,
                expected_impact_pct=15.0,
                rationale=(
                    f"PCIe link is Gen{gpu.pcie_gen} (Gen4 expected). "
                    "Check BIOS PCIe settings for full host-GPU bandwidth."
                ),
                parameters={"detected_gen": gpu.pcie_gen, "expected_gen": 4},
            ))

        # ---- Resizable BAR (only flag when *known* disabled) --------------
        if gpu and gpu.resizable_bar is False:
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.NOTIFY,
                target="resizable_bar",
                confidence=0.95,
                expected_impact_pct=10.0,
                rationale=(
                    "Resizable BAR is disabled. Enabling it in BIOS improves "
                    "VRAM access patterns for renderers and large-buffer loads."
                ),
                parameters={"resizable_bar": False},
            ))

        # ---- Windows hardware GPU scheduler (HAGS) ------------------------
        if self.want_hags:
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.REGISTRY_TWEAK,
                target="windows_gpu_scheduler",
                confidence=0.90,
                expected_impact_pct=7.0,
                rationale=(
                    "Ensure Hardware-Accelerated GPU Scheduling (HAGS) is on "
                    f"({_HAGS_VALUE}={_HAGS_ENABLED}); reduces scheduling "
                    "latency. Requires reboot to take effect."
                ),
                parameters={
                    "key": _HAGS_KEY,
                    "value_name": _HAGS_VALUE,
                    "value_data": _HAGS_ENABLED,
                    "value_type": "REG_DWORD",
                    "requires_reboot": True,
                },
            ))

        # ---- Thermal guidance ---------------------------------------------
        temp = ctx.gpu_temp_c
        if temp is not None and temp > self.temp_warning_c:
            critical = temp >= self.temp_critical_c
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.ADJUST_QUALITY,
                target="thermal_throttle_guard",
                confidence=0.95 if critical else 0.92,
                expected_impact_pct=-5.0,   # trades quality to avoid throttle
                rationale=(
                    f"GPU at {temp:.0f}C "
                    f"({'critical' if critical else 'warning'} "
                    f">{self.temp_critical_c if critical else self.temp_warning_c:.0f}C); "
                    "reduce sample/quality settings or raise fan curve to "
                    "prevent clock throttling."
                ),
                parameters={
                    "reduce_quality": True,
                    "increase_fan_speed": True,
                    "critical": critical,
                },
            ))

        # ---- Process priority ---------------------------------------------
        for app, priority in self.process_priority.items():
            if not ctx.running.get(app):
                continue
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.PROCESS_PRIORITY,
                target=f"priority:{app}",
                confidence=0.85,
                expected_impact_pct=5.0,
                rationale=(
                    f"Set {app} to {priority} priority for better CPU "
                    "scheduling during renders/sims."
                ),
                parameters={"app": app, "priority": priority,
                            "match": _PRIORITY_PROC_MATCH.get(app, (app,))},
            ))

        # ---- CPU affinity pinning (Threadripper) --------------------------
        if self.enable_pinning and self.performance_cores:
            for app in _PERF_CORE_APPS:
                if not ctx.running.get(app):
                    continue
                ncores = (ctx.profile.cpu.logical_cores
                          if ctx.profile and ctx.profile.cpu else None)
                cores = [c for c in self.performance_cores
                         if ncores is None or c < ncores]
                if not cores:
                    continue
                recs.append(Recommendation(
                    advisor=self.name,
                    action=Action.CPU_AFFINITY,
                    target=f"affinity:{app}",
                    confidence=0.95,
                    expected_impact_pct=10.0,
                    rationale=(
                        f"Pin {app} to performance cores {cores[0]}-"
                        f"{cores[-1]} to cut cross-CCD latency and isolate "
                        "background interference."
                    ),
                    parameters={"app": app, "cores": cores,
                                "kind": "performance"},
                ))

        return recs
