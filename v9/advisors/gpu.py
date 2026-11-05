"""V9 — GPU config-tuning advisor.

Ported from v6's GPUSpecialist (the CUDA-env-var half — driver/registry power
mode stays dormant, exactly as v6 left it, because PowerMizerMode is an NVAPI
setting the registry applier can't satisfy).

Emits CONFIG-family recommendations only: CUDA environment variables read from
the config `gpu` section, plus a VRAM-pressure GC nudge. Zero hardware literals
— the VRAM total comes from ctx (which sources it from HardwareProfile).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from v9.advisors.base import Advisor
from v9.recommend import Action, Recommendation

if TYPE_CHECKING:
    from v9.context import SystemContext

# Sensible CUDA env defaults if config omits them. These are tuning values,
# not hardware facts — safe to keep as constants.
_DEFAULT_CUDA_ENV = {
    "CUDA_CACHE_MAXSIZE": "4294967296",   # 4 GB shader/kernel cache
    "CUDA_CACHE_DISABLE": "0",
    "CUDA_MODULE_LOADING": "LAZY",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


class GpuAdvisor(Advisor):
    name = "gpu"
    min_confidence = 0.85

    def __init__(self, config=None, cache_handle=None, quality_ledger=None):
        super().__init__(config, cache_handle, quality_ledger)
        cuda = self.config.get("cuda_env_vars") or {}
        # config wins where it specifies; defaults fill the rest
        self.cuda_env_vars = {**_DEFAULT_CUDA_ENV, **cuda}
        self.vram_warning_frac = float(self.config.get("vram_warning_frac", 0.90))

    def _analyze(self, ctx: "SystemContext") -> list[Recommendation]:
        recs: list[Recommendation] = []

        # ---- CUDA env vars: recommend any that differ from current env -----
        for var, want in self.cuda_env_vars.items():
            want = str(want)
            current = os.environ.get(var)
            if current == want:
                continue
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.SET_ENV_VAR,
                target=var,
                confidence=0.95,
                expected_impact_pct=5.0,
                rationale=(
                    f"{var} currently {current!r}; set to {want!r} to improve "
                    "kernel-cache reuse and CUDA module load behavior."
                ),
                parameters={"variable": var, "value": want, "scope": "user"},
            ))

        # ---- VRAM pressure: GC / shrink caches (crash prevention) ----------
        total = ctx.vram_total_gb or 0.0
        if total > 0 and (ctx.vram_used_gb / total) > self.vram_warning_frac:
            pct = 100.0 * ctx.vram_used_gb / total
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.VRAM_ALLOCATION,
                target="reduce_allocations",
                confidence=0.88,
                expected_impact_pct=0.0,   # prevents OOM, not a speedup
                rationale=(
                    f"VRAM at {pct:.1f}% of {total:.0f} GB; trigger GC and "
                    "shrink caches to avoid an out-of-memory crash."
                ),
                parameters={"trigger_gc": True, "reduce_cache_sizes": True},
            ))

        return recs
