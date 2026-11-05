"""
UE5Specialist (Forge-2)

Implements the decision rules in briefs/forge-2_specialists.md Section C
exactly. Replaces the V7 UE5Engineer for V8 workflows; V7 UE5Engineer
remains untouched in core/gpu_optimizer_v6.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.gpu_optimizer_v6 import (
    Expert,
    ExpertType,
    OptimizationAction,
    Recommendation,
    SystemContext,
)


# ---------------------------------------------------------------------------
# Resolution helpers (mirror sibling specialists)
# ---------------------------------------------------------------------------

def _resolve_expert_type() -> ExpertType:
    for name in ("UE5_SPECIALIST_V8", "UE5_SPECIALIST"):
        if hasattr(ExpertType, name):
            return getattr(ExpertType, name)
    return ExpertType.UE5_ENGINEER  # V7 fallback


def _resolve_action(*candidates: str) -> OptimizationAction:
    for name in candidates:
        if hasattr(OptimizationAction, name):
            return getattr(OptimizationAction, name)
    return OptimizationAction.ENABLE_ACCELERATION


def _version_tuple(s: Optional[str]) -> Optional[tuple]:
    if not s:
        return None
    parts = []
    for tok in s.split("."):
        try:
            parts.append(int(tok))
        except ValueError:
            parts.append(0)
    return tuple(parts) if parts else None


def _version_lt(a: Optional[str], b: str) -> bool:
    """Semantic 'a < b'. None => False (don't recommend without evidence)."""
    ta = _version_tuple(a)
    tb = _version_tuple(b)
    if ta is None or tb is None:
        return False
    return ta < tb


# ---------------------------------------------------------------------------
# Specialist
# ---------------------------------------------------------------------------

class UE5Specialist(Expert):
    """V8 specialist for UE5.6+. See Section C of forge-2_specialists.md."""

    DLSS_MIN_VERSION = "4.0"

    def __init__(
        self,
        config: Dict[str, Any],
        cache_handle: Optional[Any] = None,
        quality_ledger_handle: Optional[Any] = None,
    ) -> None:
        # cache + ledger handles accepted for symmetry with siblings; UE5
        # rules don't currently consult them but Phase B will.
        self.cache_handle = cache_handle
        self.quality_ledger_handle = quality_ledger_handle
        self.nim_marketplace_url = (
            config.get("ue5", {}).get("nim_marketplace_url")
            or "https://www.unrealengine.com/marketplace/en-US/product/nvidia-nim"
        )
        super().__init__(config, _resolve_expert_type())
        self.logger = logging.getLogger("Expert.ue5_specialist")

    def _load_knowledge_base(self) -> None:
        self.knowledge_base = {
            "dlss_min_version": self.DLSS_MIN_VERSION,
            "denoiser_cvar": "r.PathTracing.Denoiser",
        }

    # ------------------------------------------------------------------
    # analyze (normative decision rules - Section C)
    # ------------------------------------------------------------------
    def analyze(self, context: SystemContext) -> List[Recommendation]:
        state = getattr(context, "ue5_v8", None)
        if state is None or not getattr(state, "process_running", False):
            return []

        recs: List[Recommendation] = []

        # ---- R1: NVIDIA NIM Plugin install advisory -----------------------
        if not getattr(state, "nim_plugin_installed", False):
            recs.append(Recommendation(
                expert=self.expert_type,
                action=_resolve_action("INSTALL_PLUGIN", "ENABLE_ACCELERATION"),
                target="ue5:nim",
                parameters={
                    "plugin": "NVIDIA_NIM",
                    "marketplace_url": self.nim_marketplace_url,
                    "auto_install": False,
                    "notification": (
                        "Install NVIDIA NIM Plugin to enable TRT-LLM in UE5."
                    ),
                },
                confidence=0.78,
                expected_impact=30.0,
                reasoning=(
                    "NVIDIA NIM Plugin is not present in this project. "
                    "NIM exposes TRT-LLM serving inside UE5 for runtime "
                    "AI behavior; install is user-driven (Marketplace)."
                ),
                research_citation="NVIDIA NIM for Unreal Engine",
            ))

        # ---- R2: DLSS 4 / DLSS-RR version verification --------------------
        dlss_ver = getattr(state, "dlss_dll_version", None)
        if dlss_ver and _version_lt(dlss_ver, self.DLSS_MIN_VERSION):
            recs.append(Recommendation(
                expert=self.expert_type,
                action=_resolve_action("INSTALL_PLUGIN", "ENABLE_ACCELERATION"),
                target="ue5:dlss",
                parameters={
                    "plugin": "DLSS_4",
                    "current": dlss_ver,
                    "minimum": self.DLSS_MIN_VERSION,
                    "notification": True,
                },
                confidence=0.85,
                expected_impact=15.0,
                reasoning=(
                    f"DLSS DLL version {dlss_ver} is older than required "
                    f"{self.DLSS_MIN_VERSION}; upgrade to DLSS 4 / DLSS-RR "
                    "to benefit from Ray Reconstruction on Ada."
                ),
                research_citation="NVIDIA DLSS 4 release notes",
            ))

        # ---- R3: Path Tracer GPU denoiser ---------------------------------
        cvars = getattr(state, "cvars", {}) or {}
        if (
            getattr(state, "path_tracer_in_use", False)
            and cvars.get(self.knowledge_base["denoiser_cvar"]) != "1"
        ):
            engine_ini = getattr(state, "engine_ini", None)
            recs.append(Recommendation(
                expert=self.expert_type,
                action=_resolve_action("ADJUST_QUALITY"),
                target="ue5:path_tracer_denoiser",
                parameters={
                    "engine_ini_path": str(engine_ini) if engine_ini else None,
                    "engine_ini_setting": "r.PathTracing.Denoiser=1",
                    "section": "/Script/Engine.RendererSettings",
                },
                confidence=0.90,
                expected_impact=12.0,
                reasoning=(
                    "Path Tracer is in use but r.PathTracing.Denoiser is "
                    "not set to 1. Enabling the GPU denoiser converges "
                    "stills ~12% faster on Ada."
                ),
                research_citation="UE5 Path Tracer documentation",
            ))

        # R4 (Phase B) NIM TRT-LLM model integration patterns are deferred.

        return self.filter_by_confidence(recs)
