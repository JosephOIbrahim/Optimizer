"""
core.experts -- V8 specialist package.

Forge-3 owns this __init__.py. Forge-2 owns the specialist modules; Forge-3
re-exports them so the V8 orchestrator (and any external import) can do:

    from core.experts import (
        ComfyUISpecialist,
        HoudiniSpecialist,
        UE5Specialist,
        CrossAppCoordinator,
    )

State collectors are also re-exported for the same reason. Imports are
defensive: if an optional dep inside a specialist module is missing, the
re-export falls back to None rather than aborting package import.
"""

from __future__ import annotations

__all__ = [
    "ComfyUISpecialist",
    "ComfyUIState",
    "ComfyUIStateCollector",
    "HoudiniSpecialist",
    "HoudiniState",
    "HoudiniStateCollector",
    "UE5Specialist",
    "UE5State",
    "UE5StateCollector",
    "CrossAppCoordinator",
    "CrossAppState",
    "UsageHistoryEntry",
    "PrefetchCandidate",
    "vram_budget",
    "predict_next_engine",
]


def _safe_import(modname: str, attrnames: list):
    """Import attrs from `modname`; on failure bind each attr to None."""
    out = {}
    try:
        mod = __import__(modname, fromlist=attrnames)
        for a in attrnames:
            out[a] = getattr(mod, a, None)
    except Exception:
        for a in attrnames:
            out[a] = None
    return out


_comfy = _safe_import(
    "core.experts.comfyui_specialist", ["ComfyUISpecialist"],
)
_comfy_state = _safe_import(
    "core.experts.comfyui_state_collector",
    ["ComfyUIState", "ComfyUIStateCollector"],
)
_houdini = _safe_import(
    "core.experts.houdini_specialist", ["HoudiniSpecialist"],
)
_houdini_state = _safe_import(
    "core.experts.houdini_state_collector",
    ["HoudiniState", "HoudiniStateCollector"],
)
_ue5 = _safe_import(
    "core.experts.ue5_specialist", ["UE5Specialist"],
)
_ue5_state = _safe_import(
    "core.experts.ue5_state_collector",
    ["UE5State", "UE5StateCollector"],
)
_cross = _safe_import(
    "core.experts.cross_app_coordinator",
    [
        "CrossAppCoordinator",
        "CrossAppState",
        "UsageHistoryEntry",
        "PrefetchCandidate",
        "vram_budget",
        "predict_next_engine",
    ],
)

ComfyUISpecialist = _comfy["ComfyUISpecialist"]
ComfyUIState = _comfy_state["ComfyUIState"]
ComfyUIStateCollector = _comfy_state["ComfyUIStateCollector"]

HoudiniSpecialist = _houdini["HoudiniSpecialist"]
HoudiniState = _houdini_state["HoudiniState"]
HoudiniStateCollector = _houdini_state["HoudiniStateCollector"]

UE5Specialist = _ue5["UE5Specialist"]
UE5State = _ue5_state["UE5State"]
UE5StateCollector = _ue5_state["UE5StateCollector"]

CrossAppCoordinator = _cross["CrossAppCoordinator"]
CrossAppState = _cross["CrossAppState"]
UsageHistoryEntry = _cross["UsageHistoryEntry"]
PrefetchCandidate = _cross["PrefetchCandidate"]
vram_budget = _cross["vram_budget"]
predict_next_engine = _cross["predict_next_engine"]
