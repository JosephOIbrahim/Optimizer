"""V9 — the recommendation contract shared by advisors and appliers.

One flat `Action` enum. No frozen sibling, no `_extend_enum` poking
`_member_map_`. Every member that any advisor can emit is declared here, once.

Two families of action, matching the two optimization mechanisms:
  * CONFIG_* : instant, reversible side-effects (env / registry / priority).
  * BUILD_*  : expensive, cached, background-compiled artifacts.
The applier dispatches on this distinction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Action(Enum):
    # --- config-tuning family (instant, reversible) ---------------------
    SET_ENV_VAR = "set_env_var"
    REGISTRY_TWEAK = "registry_tweak"
    PROCESS_PRIORITY = "process_priority"
    CPU_AFFINITY = "cpu_affinity"
    VRAM_ALLOCATION = "vram_allocation"
    ADJUST_QUALITY = "adjust_quality"          # UE5 Engine.ini cvars, etc.
    SET_LAUNCH_FLAG = "set_launch_flag"        # advise relaunch w/ flag (no kill)
    # --- build-and-cache family (async, gated on idle) ------------------
    COMPILE_TRT_ENGINE = "compile_trt_engine"
    ENABLE_BACKEND = "enable_backend"
    INSTALL_CUSTOM_NODE = "install_custom_node"
    INSTALL_PLUGIN = "install_plugin"
    WIRE_BACKEND_PATCH = "wire_backend_patch"
    EVICT_ENGINE_CACHE = "evict_engine_cache"
    # --- meta -----------------------------------------------------------
    NOTIFY = "notify"                          # surface info; no mutation

    @property
    def is_build(self) -> bool:
        return self in _BUILD_ACTIONS


_BUILD_ACTIONS = {
    Action.COMPILE_TRT_ENGINE, Action.ENABLE_BACKEND, Action.INSTALL_CUSTOM_NODE,
    Action.INSTALL_PLUGIN, Action.WIRE_BACKEND_PATCH, Action.EVICT_ENGINE_CACHE,
}


@dataclass
class Recommendation:
    """A single proposed change. `advisor` is the honest source name
    (e.g. "gpu", "comfyui") — no "expert/MoE" framing."""
    advisor: str
    action: Action
    target: str                                # human-readable subject
    confidence: float                          # 0.0–1.0
    expected_impact_pct: float = 0.0           # +% estimate (prior or measured)
    rationale: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    gated_on_idle: bool = False                # build actions: wait for GPU idle
    measured: bool = False                     # True iff impact is benchmarked

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if self.action.is_build and not self.gated_on_idle:
            # build actions default to idle-gated unless explicitly immediate
            self.gated_on_idle = self.parameters.get("immediate", False) is False


def aggregate(
    recs: list[Recommendation],
    min_confidence: float = 0.0,
) -> list[Recommendation]:
    """Dedup by (action, target) keeping the highest-confidence rec, drop
    anything under threshold, and sort by confidence×impact descending.

    This replaces v8's "weighted MoE voting" with what it actually was: a
    confidence-ranked, de-duplicated list. No magic, no quorum.
    """
    best: dict[tuple[Action, str], Recommendation] = {}
    for r in recs:
        if r.confidence < min_confidence:
            continue
        key = (r.action, r.target)
        cur = best.get(key)
        if cur is None or r.confidence > cur.confidence:
            best[key] = r
    out = list(best.values())
    out.sort(key=lambda r: (r.confidence * max(r.expected_impact_pct, 1.0)),
             reverse=True)
    return out
