"""
ComfyUISpecialist (Forge-2)

Implements the decision rules in briefs/forge-2_specialists.md Section A
exactly. Produces Recommendation lists only.

The specialist consumes:
    ctx.comfyui : ComfyUIState   (wired by Forge-3)

External handles injected by the orchestrator (Forge-3) at construction time:
    cache_handle           - exposes engine_cache_lookup(key) -> bool,
                                       engine_cache_key(**kw) -> str,
                                       engine_cache_disk_used_gb(app) -> float
    quality_ledger_handle  - exposes fp8_admitted(app, model_sha) -> bool

If a handle is None, this specialist falls back to permissive behavior
(treat lookups as "not cached" so recommendations still emit). Forge-4 owns
the real implementations; this module never imports from
core/engine_cache.py or core/background_compiler.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.gpu_optimizer_v6 import (
    Expert,
    ExpertType,
    OptimizationAction,
    Recommendation,
    SystemContext,
)


# ---------------------------------------------------------------------------
# ExpertType resolution
# ---------------------------------------------------------------------------

def _resolve_expert_type() -> ExpertType:
    """
    Prefer V8 enum value if Forge-3 has landed it; otherwise fall back to
    a V7 value so import + analyze() still works during phased rollout.
    """
    for name in ("COMFYUI_SPECIALIST", "COMFYUI"):
        if hasattr(ExpertType, name):
            return getattr(ExpertType, name)
    # V7 fallback - reuse the closest existing value rather than blocking.
    return ExpertType.GPU_SPECIALIST


def _resolve_action(*candidates: str) -> OptimizationAction:
    """
    Return the first OptimizationAction member in `candidates` that exists.
    Lets Forge-2 reference V8 actions before Forge-3 has landed the enum
    additions; falls back to a V7 generic so the module still imports.
    """
    for name in candidates:
        if hasattr(OptimizationAction, name):
            return getattr(OptimizationAction, name)
    # Last-resort fallback so analyze() still emits *something* recognizable.
    return OptimizationAction.ENABLE_ACCELERATION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_shape_range(shape: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Derive a (min, opt, max) shape range for TRT compilation given the hot
    shape detected for ComfyUI workloads. ComfyUI shapes are (W, H); we
    package them as 4D (B, C, H, W) so the orchestrator can hand them
    straight to acceleration.backend.ShapeRange.
    """
    w, h = shape
    if w <= 0 or h <= 0:
        # Permissive default for malformed shapes.
        w, h = 512, 512

    min_shape = (1, 4, max(h - 64, 64), max(w - 64, 64))
    opt_shape = (1, 4, h, w)
    max_shape = (2, 4, h + 64, w + 64)
    return {"min": min_shape, "opt": opt_shape, "max": max_shape}


# ---------------------------------------------------------------------------
# Specialist
# ---------------------------------------------------------------------------

class ComfyUISpecialist(Expert):
    """
    V8 specialist for ComfyUI. See Section A of forge-2_specialists.md
    for the exact (normative) decision rules.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        cache_handle: Optional[Any] = None,
        quality_ledger_handle: Optional[Any] = None,
    ) -> None:
        # V8 features carried as instance state. Set BEFORE super().__init__
        # because the parent calls _load_knowledge_base().
        self.cache_handle = cache_handle
        self.quality_ledger_handle = quality_ledger_handle
        self.trt_version = (config.get("tensorrt") or {}).get("version", "11.x")
        self.cache_size_gb = (config.get("tensorrt") or {}).get("cache_size_gb", 50.0)
        self.phase_b = bool(config.get("phase_b_enabled", False))

        super().__init__(config, _resolve_expert_type())
        self.logger = logging.getLogger("Expert.comfyui_specialist")

    def _load_knowledge_base(self) -> None:
        self.knowledge_base = {
            "required_custom_nodes": [
                "ComfyUI-SageAttention",
                "ComfyUI_TensorRT",
            ],
            "required_launch_flags": ["--use-sage-attention"],
            "node_repos": {
                "ComfyUI-SageAttention": ("thu-ml/SageAttention", "v2.0"),
                "ComfyUI_TensorRT": ("comfyanonymous/ComfyUI_TensorRT", "main"),
            },
        }

    # ------------------------------------------------------------------
    # External-handle wrappers (permissive fallback when None)
    # ------------------------------------------------------------------
    def _cache_lookup(self, key: str) -> bool:
        if self.cache_handle is None:
            return False
        try:
            return bool(self.cache_handle.engine_cache_lookup(key))
        except Exception as exc:
            self.logger.debug("cache_handle.engine_cache_lookup failed: %s", exc)
            return False

    def _cache_key(self, **kwargs: Any) -> str:
        if self.cache_handle is None:
            # Synthetic deterministic key so logs can be correlated.
            return "uncached:" + "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        try:
            return self.cache_handle.engine_cache_key(**kwargs)
        except Exception as exc:
            self.logger.debug("cache_handle.engine_cache_key failed: %s", exc)
            return "uncached:error"

    def _cache_disk_used_gb(self, app: str) -> float:
        if self.cache_handle is None:
            return 0.0
        try:
            return float(self.cache_handle.engine_cache_disk_used_gb(app))
        except Exception as exc:
            self.logger.debug("cache_handle.engine_cache_disk_used_gb failed: %s", exc)
            return 0.0

    def _fp8_admitted(self, app: str, model_sha: str) -> bool:
        if self.quality_ledger_handle is None:
            return False
        try:
            return bool(self.quality_ledger_handle.fp8_admitted(app, model_sha))
        except Exception as exc:
            self.logger.debug("quality_ledger_handle.fp8_admitted failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # analyze (normative decision rules - Section A)
    # ------------------------------------------------------------------
    def analyze(self, context: SystemContext) -> List[Recommendation]:
        state = getattr(context, "comfyui", None)
        if state is None or not getattr(state, "process_running", False):
            return []

        recs: List[Recommendation] = []

        installed = set(getattr(state, "custom_nodes_installed", []) or [])
        flags = list(getattr(state, "launch_flags", []) or [])

        # ---- R1: SageAttention 2 install or enable ------------------------
        if "ComfyUI-SageAttention" not in installed:
            repo, ref = self.knowledge_base["node_repos"]["ComfyUI-SageAttention"]
            recs.append(Recommendation(
                expert=self.expert_type,
                action=_resolve_action("INSTALL_CUSTOM_NODE", "ENABLE_ACCELERATION"),
                target="comfyui:sage_attention",
                parameters={"node_repo": repo, "ref": ref},
                confidence=0.92,
                expected_impact=50.0,
                reasoning=(
                    "ComfyUI-SageAttention not installed. Sage Attention 2 "
                    "delivers ~1.5x SDPA speedup on Ada (sm_89) FP16/BF16 "
                    "kernels; install custom_node and restart."
                ),
                research_citation="thu-ml/SageAttention v2.0",
            ))
        elif "--use-sage-attention" not in flags:
            recs.append(Recommendation(
                expert=self.expert_type,
                action=_resolve_action("SET_LAUNCH_FLAG", "SET_ENV_VAR"),
                target="comfyui:launch",
                parameters={"add_flag": "--use-sage-attention", "restart_required": True},
                confidence=0.92,
                expected_impact=50.0,
                reasoning=(
                    "ComfyUI-SageAttention is installed but the runtime was "
                    "started without --use-sage-attention; SDPA monkeypatch "
                    "only activates when the flag is present."
                ),
            ))

        # ---- R2: ComfyUI-TensorRT custom node install ---------------------
        if "ComfyUI_TensorRT" not in installed:
            repo, ref = self.knowledge_base["node_repos"]["ComfyUI_TensorRT"]
            recs.append(Recommendation(
                expert=self.expert_type,
                action=_resolve_action("INSTALL_CUSTOM_NODE", "ENABLE_ACCELERATION"),
                target="comfyui:tensorrt",
                parameters={"node_repo": repo, "ref": ref},
                confidence=0.88,
                expected_impact=85.0,
                reasoning=(
                    "ComfyUI_TensorRT custom node not installed. Required for "
                    "TRT engine compile + load on diffusion UNets and "
                    "delivers up to ~1.85x throughput vs torch baseline."
                ),
                research_citation="comfyanonymous/ComfyUI_TensorRT",
            ))

        # ---- R3: Compile TRT engine for hot (model, shape) ----------------
        try:
            hot = state.workload_histogram_hot(min_count=3, lookback_days=14)
        except Exception as exc:
            self.logger.debug("workload_histogram_hot failed: %s", exc)
            hot = []

        for model_sha, shape in hot:
            shape_range = _derive_shape_range(shape)
            key = self._cache_key(
                app="comfyui",
                model=model_sha,
                shape=shape,
                precision="fp16",
                backend="tensorrt",
                backend_ver=self.trt_version,
            )
            if self._cache_lookup(key):
                continue

            recs.append(Recommendation(
                expert=self.expert_type,
                action=_resolve_action("COMPILE_TRT_ENGINE", "ENABLE_ACCELERATION"),
                target=f"comfyui:{model_sha}:{shape[0]}x{shape[1]}",
                parameters={
                    "precision": "fp16",
                    "shape_range": shape_range,
                    "model_kind": "unet",
                    "cache_key": key,
                    "app": "comfyui",
                    "model_sha": model_sha,
                },
                confidence=0.85,
                expected_impact=70.0,
                reasoning=(
                    f"Hot workload detected: model={model_sha} shape={shape} "
                    f"with >= 3 runs in last 14 days, no cached TRT plan. "
                    "Speculative idle-gated compile."
                ),
                requires=["tensorrt", "ComfyUI_TensorRT"],
            ))

            # ---- R4: Phase B FP8 promotion (gated) -----------------------
            if self.phase_b and self._fp8_admitted("comfyui", model_sha):
                fp8_key = self._cache_key(
                    app="comfyui",
                    model=model_sha,
                    shape=shape,
                    precision="fp8",
                    backend="tensorrt",
                    backend_ver=self.trt_version,
                )
                if not self._cache_lookup(fp8_key):
                    recs.append(Recommendation(
                        expert=self.expert_type,
                        action=_resolve_action("COMPILE_TRT_ENGINE", "ENABLE_ACCELERATION"),
                        target=f"comfyui:{model_sha}:{shape[0]}x{shape[1]}:fp8",
                        parameters={
                            "precision": "fp8",
                            "shape_range": shape_range,
                            "model_kind": "unet",
                            "cache_key": fp8_key,
                            "app": "comfyui",
                            "model_sha": model_sha,
                        },
                        confidence=0.80,
                        expected_impact=120.0,
                        reasoning=(
                            "Phase B: quality ledger has admitted FP8 for "
                            "this model; ~2x throughput on Ada FP8 tensor "
                            "cores vs FP16."
                        ),
                        requires=["tensorrt", "ComfyUI_TensorRT", "fp8_calibration"],
                    ))

        # ---- R5: Cache eviction -------------------------------------------
        used_gb = self._cache_disk_used_gb("comfyui")
        if used_gb > self.cache_size_gb * 0.90:
            recs.append(Recommendation(
                expert=self.expert_type,
                action=_resolve_action("EVICT_ENGINE_CACHE", "VRAM_ALLOCATION"),
                target="comfyui:cache",
                parameters={
                    "app": "comfyui",
                    "used_gb": used_gb,
                    "limit_gb": self.cache_size_gb,
                },
                confidence=0.95,
                expected_impact=0.0,
                reasoning=(
                    f"Engine cache at {used_gb:.1f} GB > 90% of "
                    f"{self.cache_size_gb:.1f} GB budget; evict LRU plans "
                    "to reclaim disk."
                ),
            ))

        return self.filter_by_confidence(recs)
