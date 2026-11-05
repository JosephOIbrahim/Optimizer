"""
CrossAppCoordinator (Forge-2)

Implements the decision rules in briefs/forge-2_specialists.md Section D
exactly. Owns inter-app arbitration. Reads only SystemContext - never
touches TRT or backend libraries directly.

The coordinator is the only specialist whose state object exposes whole-
system telemetry: foreground app, per-app run state, per-app compile-job
counts, idle dwell, and a usage history for prefetch prediction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

from core.gpu_optimizer_v6 import (
    Expert,
    ExpertType,
    OptimizationAction,
    Recommendation,
    SystemContext,
)


# ---------------------------------------------------------------------------
# State exposed to the specialist
# ---------------------------------------------------------------------------

@dataclass
class UsageHistoryEntry:
    """One past compile we observed; feeds the LRU/freq prefetch heuristic."""

    cache_key: str
    compile_params: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class PrefetchCandidate:
    """Output of predict_next_engine; consumed by R3."""

    key: str
    compile_params: Dict[str, Any]


@dataclass
class CrossAppState:
    """Snapshot of multi-app coordination state. Wired into SystemContext.cross_app."""

    foreground_pid_app: str = "none"          # one of {comfyui, houdini, ue5, none}
    app_running: Dict[str, bool] = field(default_factory=lambda: {
        "comfyui": False, "houdini": False, "ue5": False
    })
    compile_jobs_active: Dict[str, int] = field(default_factory=lambda: {
        "comfyui": 0, "houdini": 0, "ue5": 0
    })
    idle_dwell_seconds: float = 0.0
    usage_history: Deque[UsageHistoryEntry] = field(
        default_factory=lambda: deque(maxlen=256)
    )


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _resolve_expert_type() -> ExpertType:
    for name in ("CROSS_APP_COORDINATOR",):
        if hasattr(ExpertType, name):
            return getattr(ExpertType, name)
    return ExpertType.SYSTEM_INTEGRATOR  # V7 fallback


def _resolve_action(*candidates: str) -> OptimizationAction:
    for name in candidates:
        if hasattr(OptimizationAction, name):
            return getattr(OptimizationAction, name)
    return OptimizationAction.ENABLE_ACCELERATION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vram_budget(active: List[str]) -> Dict[str, float]:
    """
    Mirror V7 WorkflowMode logic but app-keyed (per brief).
    1 app  -> 18 GB to that app, 2 GB system reserve elsewhere.
    2 apps -> 11 / 11 GB.
    3 apps -> 11 / 8 / 3 GB plus 2 GB system reserve.
    """
    n = len(active)
    if n == 0:
        return {}
    if n == 1:
        return {active[0]: 18.0}
    if n == 2:
        return {active[0]: 11.0, active[1]: 11.0}
    # 3+: assign 11 / 8 / 3 in input order. Anything beyond gets a sliver.
    budget: Dict[str, float] = {}
    sizes = [11.0, 8.0, 3.0]
    for idx, app in enumerate(active):
        budget[app] = sizes[idx] if idx < len(sizes) else 1.0
    return budget


def predict_next_engine(
    history: "Deque[UsageHistoryEntry] | List[UsageHistoryEntry]",
) -> Optional[PrefetchCandidate]:
    """
    Simple LRU + frequency blend. Score each cache_key by:
        score = freq + 0.5 * (1 / age_rank)
    Pick the top score; return None if history is empty.
    """
    if not history:
        return None

    items = list(history)
    freq: Dict[str, int] = {}
    last_seen: Dict[str, int] = {}
    params: Dict[str, Dict[str, Any]] = {}
    for idx, ent in enumerate(items):
        key = ent.cache_key
        freq[key] = freq.get(key, 0) + 1
        last_seen[key] = idx
        params[key] = ent.compile_params

    # Age rank: most recent has rank 1.
    n = len(items)
    best_key: Optional[str] = None
    best_score = -1.0
    for key, f in freq.items():
        age_rank = n - last_seen[key]
        score = float(f) + 0.5 * (1.0 / max(age_rank, 1))
        if score > best_score:
            best_score = score
            best_key = key

    if best_key is None:
        return None
    return PrefetchCandidate(key=best_key, compile_params=params[best_key])


# ---------------------------------------------------------------------------
# Specialist
# ---------------------------------------------------------------------------

class CrossAppCoordinator(Expert):
    """V8 cross-app coordinator. See Section D of forge-2_specialists.md."""

    def __init__(
        self,
        config: Dict[str, Any],
        cache_handle: Optional[Any] = None,
        quality_ledger_handle: Optional[Any] = None,
    ) -> None:
        self.cache_handle = cache_handle
        self.quality_ledger_handle = quality_ledger_handle
        self.idle_window_seconds = float(
            config.get("cross_app", {}).get("idle_window_seconds", 60.0)
        )
        super().__init__(config, _resolve_expert_type())
        self.logger = logging.getLogger("Expert.cross_app_coordinator")

    def _load_knowledge_base(self) -> None:
        self.knowledge_base = {
            "apps": ("comfyui", "houdini", "ue5"),
            "idle_window_seconds": self.idle_window_seconds,
        }

    # ------------------------------------------------------------------
    # Handle wrappers
    # ------------------------------------------------------------------
    def _cache_lookup(self, key: str) -> bool:
        if self.cache_handle is None:
            return False
        try:
            return bool(self.cache_handle.engine_cache_lookup(key))
        except Exception as exc:
            self.logger.debug("cache_handle.engine_cache_lookup failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # analyze (normative decision rules - Section D)
    # ------------------------------------------------------------------
    def analyze(self, context: SystemContext) -> List[Recommendation]:
        state = getattr(context, "cross_app", None)
        if state is None:
            return []

        recs: List[Recommendation] = []
        apps = self.knowledge_base["apps"]

        fg = getattr(state, "foreground_pid_app", "none") or "none"
        compile_jobs = getattr(state, "compile_jobs_active", {}) or {}
        app_running = getattr(state, "app_running", {}) or {}

        # ---- R1: foreground process priority ------------------------------
        for bg_app in apps:
            if bg_app == fg:
                continue
            if compile_jobs.get(bg_app, 0) > 0:
                recs.append(Recommendation(
                    expert=self.expert_type,
                    action=_resolve_action("ENABLE_BACKEND", "PROCESS_PRIORITY"),
                    target=f"compiler:pause:{bg_app}",
                    parameters={
                        "action": "pause_background_compiles",
                        "app": bg_app,
                        "foreground_app": fg,
                    },
                    confidence=0.95,
                    expected_impact=0.0,
                    reasoning=(
                        f"App '{bg_app}' has "
                        f"{compile_jobs.get(bg_app, 0)} background compile "
                        f"job(s) active while '{fg}' is foreground. Pause "
                        "background compiles to protect foreground latency."
                    ),
                ))

        # ---- R2: VRAM budget per active-app set ---------------------------
        active = [a for a in apps if app_running.get(a)]
        budget = vram_budget(active)
        for app, gb in budget.items():
            recs.append(Recommendation(
                expert=self.expert_type,
                action=_resolve_action("VRAM_ALLOCATION"),
                target=f"vram:{app}",
                parameters={"limit_gb": gb, "active_set": active, "app": app},
                confidence=0.90,
                expected_impact=0.0,
                reasoning=(
                    f"With {len(active)} VFX app(s) active "
                    f"({', '.join(active) or 'none'}), allocate "
                    f"{gb:.0f} GB VRAM to '{app}'."
                ),
            ))

        # ---- R3: idle-window prefetch -------------------------------------
        idle_dwell = float(getattr(state, "idle_dwell_seconds", 0.0) or 0.0)
        if idle_dwell >= self.idle_window_seconds:
            history = getattr(state, "usage_history", None) or []
            candidate = predict_next_engine(history)
            if candidate is not None and not self._cache_lookup(candidate.key):
                recs.append(Recommendation(
                    expert=self.expert_type,
                    action=_resolve_action("COMPILE_TRT_ENGINE", "ENABLE_ACCELERATION"),
                    target=f"prefetch:{candidate.key[:8]}",
                    parameters=dict(candidate.compile_params, prefetch=True,
                                    cache_key=candidate.key),
                    confidence=0.65,
                    expected_impact=20.0,
                    reasoning=(
                        f"All apps idle for {idle_dwell:.0f}s "
                        f"(>= {self.idle_window_seconds:.0f}s window). "
                        "Prefetching most-likely-next engine based on "
                        "LRU+freq history."
                    ),
                ))

        return self.filter_by_confidence(recs)
