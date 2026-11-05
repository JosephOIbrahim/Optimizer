"""V9 — Cross-app coordinator advisor.

Ports core/experts/cross_app_coordinator.py. Owns inter-app arbitration; reads
only the SystemContext (and the per-app states the other advisors publish onto
it) — never touches TRT or backend libraries directly.

Rules:
  R1  Foreground app wins: any background app with active compile jobs should
      pause those compiles to protect foreground latency.
  R2  VRAM budget per active-app set (foreground-weighted): 1->18, 2->11/11,
      3->11/8/3 GB, system reserve elsewhere.
  R3  Idle-window prefetch: when all apps idle past the window, predict the
      most-likely-next engine (LRU+freq) and speculatively compile it.

Compile-job counts and idle dwell come from the orchestrator via ctx (it owns
the background compiler + clock). When that telemetry is absent the coordinator
degrades to VRAM budgeting only — never raises.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Optional

from v9.advisors.base import Advisor
from v9.recommend import Action, Recommendation

if TYPE_CHECKING:
    from v9.context import SystemContext

log = logging.getLogger("v9.advisor.coordinator")

_APPS = ("comfyui", "houdini", "ue5")


@dataclass
class _UsageEntry:
    cache_key: str
    compile_params: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class _PrefetchCandidate:
    key: str
    compile_params: dict[str, Any]


def vram_budget(active: list[str]) -> dict[str, float]:
    """1 app -> 18 GB; 2 -> 11/11; 3+ -> 11/8/3 (in given order)."""
    n = len(active)
    if n == 0:
        return {}
    if n == 1:
        return {active[0]: 18.0}
    if n == 2:
        return {active[0]: 11.0, active[1]: 11.0}
    sizes = [11.0, 8.0, 3.0]
    return {app: (sizes[i] if i < len(sizes) else 1.0)
            for i, app in enumerate(active)}


def predict_next_engine(
        history: "Deque[_UsageEntry] | list[_UsageEntry]") -> Optional[_PrefetchCandidate]:
    """LRU+frequency blend: score = freq + 0.5/age_rank; pick the top."""
    items = list(history)
    if not items:
        return None
    freq: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    params: dict[str, dict[str, Any]] = {}
    for idx, ent in enumerate(items):
        freq[ent.cache_key] = freq.get(ent.cache_key, 0) + 1
        last_seen[ent.cache_key] = idx
        params[ent.cache_key] = ent.compile_params
    n = len(items)
    best_key, best_score = None, -1.0
    for key, f in freq.items():
        score = float(f) + 0.5 * (1.0 / max(n - last_seen[key], 1))
        if score > best_score:
            best_key, best_score = key, score
    if best_key is None:
        return None
    return _PrefetchCandidate(best_key, params[best_key])


class CoordinatorAdvisor(Advisor):
    name = "coordinator"
    min_confidence = 0.65

    def __init__(self, config=None, cache_handle=None, quality_ledger=None):
        super().__init__(config, cache_handle, quality_ledger)
        self.idle_window_seconds = float(
            self.config.get("idle_window_seconds", 60.0))

    def _cache_has(self, key: str) -> bool:
        if self.cache is None:
            return False
        try:
            return bool(self.cache.engine_cache_lookup(key))
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _analyze(self, ctx: "SystemContext") -> list[Recommendation]:
        recs: list[Recommendation] = []
        fg = ctx.foreground_app or "none"
        # Optional orchestrator-supplied telemetry, attached to ctx if present.
        compile_jobs: dict[str, int] = getattr(ctx, "compile_jobs_active", {}) or {}
        idle_dwell = float(getattr(ctx, "idle_dwell_seconds", 0.0) or 0.0)
        usage_history = getattr(ctx, "usage_history", None) or []

        # ---- R1: pause background compiles for non-foreground apps -------
        for bg in _APPS:
            if bg == fg:
                continue
            jobs = compile_jobs.get(bg, 0)
            if jobs > 0:
                recs.append(Recommendation(
                    advisor=self.name,
                    action=Action.ENABLE_BACKEND,
                    target=f"compiler:pause:{bg}",
                    confidence=0.95,
                    expected_impact_pct=0.0,
                    rationale=(
                        f"'{bg}' has {jobs} background compile job(s) while "
                        f"'{fg}' is foreground; pause them to protect "
                        "foreground latency."
                    ),
                    parameters={"action": "pause_background_compiles",
                                "app": bg, "foreground_app": fg,
                                "immediate": True},
                ))

        # ---- R2: VRAM budget per active set ------------------------------
        active = [a for a in _APPS if ctx.running.get(a)]
        for app, gb in vram_budget(active).items():
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.VRAM_ALLOCATION,
                target=f"vram:{app}",
                confidence=0.90,
                expected_impact_pct=0.0,
                rationale=(
                    f"With {len(active)} VFX app(s) active "
                    f"({', '.join(active) or 'none'}), allocate {gb:.0f} GB "
                    f"VRAM to '{app}'."
                ),
                parameters={"limit_gb": gb, "active_set": active, "app": app},
            ))

        # ---- R3: idle-window prefetch ------------------------------------
        if idle_dwell >= self.idle_window_seconds:
            candidate = predict_next_engine(usage_history)
            if candidate is not None and not self._cache_has(candidate.key):
                recs.append(Recommendation(
                    advisor=self.name,
                    action=Action.COMPILE_TRT_ENGINE,
                    target=f"prefetch:{candidate.key[:8]}",
                    confidence=0.65,
                    expected_impact_pct=20.0,
                    rationale=(
                        f"All apps idle {idle_dwell:.0f}s "
                        f"(>= {self.idle_window_seconds:.0f}s window); prefetch "
                        "most-likely-next engine via LRU+freq history."
                    ),
                    gated_on_idle=True,
                    parameters=dict(candidate.compile_params, prefetch=True,
                                    cache_key=candidate.key),
                ))

        return recs
