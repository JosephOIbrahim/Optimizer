"""V9 — Advisor ABC.

An Advisor is a domain rule-set: given a SystemContext, it returns scored
Recommendations. That's it. No voting, no "expert" mystique. The orchestrator
runs all enabled advisors (in parallel), aggregates their output via
`recommend.aggregate`, and applies the top ones.

Hard invariant (Design §4.2): `analyze()` MUST NOT raise. The base class wraps
the subclass hook in a guard so one misbehaving advisor can never break the
cycle — it just contributes zero recommendations and logs.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from v9.recommend import Recommendation

if TYPE_CHECKING:
    from v9.context import SystemContext

log = logging.getLogger("v9.advisor")


class Advisor(ABC):
    #: short, honest source tag used in Recommendation.advisor and config keys
    name: str = "advisor"
    #: recommendations below this are dropped by the advisor itself
    min_confidence: float = 0.70

    def __init__(self, config: dict[str, Any] | None = None,
                 cache_handle: Any = None,
                 quality_ledger: Any = None) -> None:
        self.config = config or {}
        self.cache = cache_handle              # may be None -> permissive
        self.quality_ledger = quality_ledger   # may be None -> conservative
        self.min_confidence = float(
            self.config.get("min_confidence", self.min_confidence))

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    @abstractmethod
    def _analyze(self, ctx: "SystemContext") -> list[Recommendation]:
        """Subclass hook. May assume ctx is well-formed; may NOT raise in the
        common path, but if it does the base guard catches it."""
        ...

    def analyze(self, ctx: "SystemContext") -> list[Recommendation]:
        """Public, guaranteed-non-raising entry point used by the orchestrator."""
        if not self.enabled:
            return []
        try:
            recs = self._analyze(ctx) or []
        except Exception:
            log.warning("advisor %s raised in analyze(); contributing 0 recs",
                        self.name, exc_info=True)
            return []
        # advisor-local confidence floor
        return [r for r in recs if r.confidence >= self.min_confidence]
