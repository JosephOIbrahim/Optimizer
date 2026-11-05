"""
Dynamic workflows: map the live workload to a mode + a VRAM budget for the 24GB card.

The watchdog calls detect() every cycle. When the mode changes it re-plans with the
subset of proposals whose `applies_when` includes the new mode, and re-fences VRAM.
Budgets account for the 6K XDR desktop compositor, which itself holds ~1-2GB.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .base import Context, WorkloadMode


@dataclass
class VramBudget:
    """GB allocation for the 24GB card. reserve = compositor(6K XDR) + headroom."""
    houdini: float = 0.0
    unreal: float = 0.0
    comfyui: float = 0.0
    reserve: float = 2.0


# Defaults; the synthesis pass from the research workflow may refine these into config.
BUDGETS: Dict[WorkloadMode, VramBudget] = {
    WorkloadMode.IDLE:          VramBudget(0, 0, 0, 2.0),
    WorkloadMode.HOUDINI_HEAVY: VramBudget(houdini=20.0, reserve=4.0),
    WorkloadMode.UE5_HEAVY:     VramBudget(unreal=20.0, reserve=4.0),
    WorkloadMode.COMFYUI_HEAVY: VramBudget(comfyui=20.0, reserve=4.0),
    WorkloadMode.PARALLEL:      VramBudget(houdini=10.0, unreal=9.0, comfyui=2.0, reserve=3.0),
}


def detect(ctx: Context) -> WorkloadMode:
    r = ctx.running
    active = [k for k, v in r.items() if v]
    if len(active) >= 2:
        return WorkloadMode.PARALLEL
    if r.get("houdini"):
        return WorkloadMode.HOUDINI_HEAVY
    if r.get("unreal"):
        return WorkloadMode.UE5_HEAVY
    if r.get("comfyui"):
        return WorkloadMode.COMFYUI_HEAVY
    return WorkloadMode.IDLE


def budget_for(mode: WorkloadMode) -> VramBudget:
    return BUDGETS.get(mode, BUDGETS[WorkloadMode.IDLE])
