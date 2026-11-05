"""V9 — Houdini 21 advisor.

Ports core/experts/houdini_specialist.py + houdini_state_collector.py, inlined
and cleaned. Collects its own Houdini state inside _analyze (gated on
ctx.running["houdini"]) then applies:

  R1  ML SOP ONNX -> recommend fp16 TRT engine compile (idle-gated, cache-keyed)
  R2  Karma XPU active + OptiX denoiser off -> enable HOUDINI_OPTIX_DENOISER
  R3  TOPs/PDG active + GPU scheduling explicitly off -> re-enable
  R4  Phase B: Vellum solver active -> enable cuTensor backend

TRT compiles are looked up via self.cache (permissive when None). No hardware
literals, no build-layer imports.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from v9.advisors.base import Advisor
from v9.recommend import Action, Recommendation

if TYPE_CHECKING:
    from v9.context import SystemContext

log = logging.getLogger("v9.advisor.houdini")

_PROC_NAMES = {"houdini.exe", "houdinifx.exe", "houdinicore.exe",
               "houdini", "houdinifx", "houdinicore"}
_DENOISER_VAR = "HOUDINI_OPTIX_DENOISER"
_PDG_VAR = "HOUDINI_PDG_GPU_SCHEDULING"
_TRUE = {"1", "true", "True"}


@dataclass
class _HoudiniState:
    process_running: bool = False
    user_pref_dir: Optional[Path] = None
    ml_sop_onnx_paths: list[Path] = field(default_factory=list)
    karma_xpu_active: bool = False
    optix_denoiser_enabled: bool = False
    tops_active: bool = False
    pdg_gpu_scheduling_enabled: Optional[bool] = None    # tri-state
    vellum_solver_active: bool = False


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fp:
            while True:
                buf = fp.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except OSError:
        return ""


def _onnx_input_shape(path: Path) -> Optional[tuple[int, ...]]:
    """Best-effort ONNX first-input shape; falls back to a baseline."""
    try:
        import onnx  # type: ignore
        model = onnx.load(str(path))
        dims = [int(d.dim_value) if d.dim_value > 0 else 1
                for d in model.graph.input[0].type.tensor_type.shape.dim]
        return tuple(dims) if dims else None
    except Exception:
        try:
            path.stat()
            return (1, 4, 64, 64)
        except OSError:
            return None


def _fixed_shape_range(shape: Optional[tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    if not shape:
        shape = (1, 4, 64, 64)
    return {"min": shape, "opt": shape, "max": shape}


class HoudiniAdvisor(Advisor):
    name = "houdini"
    min_confidence = 0.70

    def __init__(self, config=None, cache_handle=None, quality_ledger=None):
        super().__init__(config, cache_handle, quality_ledger)
        trt = self.config.get("tensorrt") or {}
        self.trt_version = str(trt.get("version", "11.x"))
        self.phase_b = bool(self.config.get("phase_b_enabled", False))
        # CUTLASS opt-in (acceleration.cutlass.enabled) reaches us via the
        # 'acceleration' block the orchestrator forwards.
        accel = self.config.get("acceleration") or {}
        cutlass = (accel.get("cutlass", {}) or {}) if isinstance(accel, dict) else {}
        vellum = (cutlass.get("vellum", {}) or {}) if isinstance(cutlass, dict) else {}
        self.cutlass_enabled = bool(cutlass.get("enabled", False))
        self.cutlass_vellum_enabled = bool(vellum.get("enabled", True))
        self.cutlass_precision = str(vellum.get("precision", "fp8_e4m3"))
        dg = vellum.get("default_gemm", [4096, 4096, 4096])
        try:
            self.cutlass_gemm = (int(dg[0]), int(dg[1]), int(dg[2]))
        except Exception:  # noqa: BLE001
            self.cutlass_gemm = (4096, 4096, 4096)

    # ------------------------------------------------------------------
    def _cache_key(self, **kw: Any) -> str:
        if self.cache is None:
            return "uncached:" + "|".join(f"{k}={v}" for k, v in sorted(kw.items()))
        try:
            return self.cache.engine_cache_key(**kw)
        except Exception:
            return "uncached:error"

    def _cache_has(self, key: str) -> bool:
        if self.cache is None:
            return False
        try:
            return bool(self.cache.engine_cache_lookup(key))
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _analyze(self, ctx: "SystemContext") -> list[Recommendation]:
        if not ctx.running.get("houdini"):
            return []
        state = self._collect()
        ctx.houdini = state
        if not state.process_running:
            return []

        recs: list[Recommendation] = []

        # ---- R1: ML SOP ONNX -> TRT --------------------------------------
        for onnx_ref in state.ml_sop_onnx_paths:
            try:
                onnx_path = Path(onnx_ref)
            except Exception:
                continue
            if not onnx_path.exists():
                continue
            onnx_sha = _sha256_file(onnx_path)
            if not onnx_sha:
                continue
            shape = _onnx_input_shape(onnx_path)
            key = self._cache_key(app="houdini", model=onnx_sha, shape=shape,
                                  precision="fp16", backend="tensorrt",
                                  backend_ver=self.trt_version)
            if self._cache_has(key):
                continue
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.COMPILE_TRT_ENGINE,
                target=f"houdini:ml_sop:{onnx_sha[:8]}",
                confidence=0.82,
                expected_impact_pct=55.0,
                rationale=(
                    f"ML SOP ONNX {onnx_path.name} ({onnx_sha[:8]}) has no "
                    "cached TRT plan; convert to fp16 TRT engine for in-SOP "
                    "inference."
                ),
                gated_on_idle=True,
                parameters={"precision": "fp16",
                            "shape_range": _fixed_shape_range(shape),
                            "model_kind": "ml_sop", "source_onnx": str(onnx_path),
                            "cache_key": key, "app": "houdini"},
            ))

        # ---- R2: Karma XPU OptiX denoiser --------------------------------
        if state.karma_xpu_active and not state.optix_denoiser_enabled:
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.SET_ENV_VAR,
                target="houdini:karma:denoiser",
                confidence=0.88,
                expected_impact_pct=18.0,
                rationale=(
                    f"Karma XPU active but {_DENOISER_VAR} unset; enabling the "
                    "OptiX denoiser converges noisy renders ~18% faster with "
                    "negligible quality loss."
                ),
                parameters={"variable": _DENOISER_VAR, "value": "1",
                            "scope": "user", "restart_required": True},
            ))

        # ---- R3: TOPs/PDG GPU scheduler ----------------------------------
        if state.tops_active and state.pdg_gpu_scheduling_enabled is False:
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.SET_ENV_VAR,
                target="houdini:pdg:gpu_scheduling",
                confidence=0.85,
                expected_impact_pct=22.0,
                rationale=(
                    f"TOPs/PDG graph active but {_PDG_VAR} is explicitly "
                    "disabled; re-enable to let PDG schedule GPU work groups."
                ),
                parameters={"variable": _PDG_VAR, "value": "1",
                            "scope": "user", "restart_required": True},
            ))

        # ---- R4: CUTLASS FP8 GEMM for Vellum/Pyro solve ------------------
        # Opt-in via acceleration.cutlass.enabled (no longer a Phase-B stub).
        # When the solver is active and CUTLASS is in the recipe, recommend
        # building+caching the FP8 GEMM kernel for the solve contraction.
        if (self.cutlass_enabled and self.cutlass_vellum_enabled
                and (state.vellum_solver_active or state.tops_active)):
            m, n, k = self.cutlass_gemm
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.ENABLE_BACKEND,
                target="houdini:vellum:cutlass_fp8_gemm",
                confidence=0.74,
                expected_impact_pct=40.0,
                rationale=(
                    f"Vellum/Pyro solve active and CUTLASS is in the recipe; "
                    f"build the FP8 ({self.cutlass_precision}) GEMM kernel for "
                    f"the {m}x{n}x{k} contraction (~1.7x vs fp16 on Ada FP8 "
                    "tensor cores). Idle-gated build, cached on the SSD."
                ),
                gated_on_idle=True,
                parameters={"backend": "cutlass_fp8_gemm", "app": "houdini",
                            "model_kind": "gemm", "precision": self.cutlass_precision,
                            "gemm_mnk": [m, n, k], "vellum": True},
            ))

        return recs

    # ------------------------------------------------------------------
    # state collection (never raises)
    # ------------------------------------------------------------------
    def _collect(self) -> _HoudiniState:
        state = _HoudiniState(process_running=True)
        try:
            procs = self._find_procs()
            if procs:
                env = self._merge_env(procs)
                state.user_pref_dir = self._pref_dir(env)
                state.karma_xpu_active = self._karma_active(env)
                state.optix_denoiser_enabled = env.get(_DENOISER_VAR, "0") in _TRUE
                state.tops_active = self._tops_active(procs, env)
                pdg = env.get(_PDG_VAR)
                state.pdg_gpu_scheduling_enabled = \
                    None if pdg is None else (pdg in _TRUE)
                state.vellum_solver_active = \
                    env.get("HOUDINI_VELLUM_ACTIVE", "0") in _TRUE
            else:
                state.process_running = False
            state.ml_sop_onnx_paths = self._discover_onnx(state)
        except Exception:
            log.debug("houdini state collect degraded", exc_info=True)
        return state

    @staticmethod
    def _find_procs():
        try:
            import psutil  # type: ignore
        except Exception:
            return []
        out = []
        try:
            for p in psutil.process_iter(attrs=["name", "environ", "cmdline"]):
                try:
                    if (p.info.get("name") or "").lower() in _PROC_NAMES:
                        out.append(p)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return out

    @staticmethod
    def _merge_env(procs) -> dict[str, str]:
        merged: dict[str, str] = {}
        for p in procs:
            try:
                for k, v in (p.environ() or {}).items():
                    merged.setdefault(k, str(v))
            except Exception:
                continue
        for k in ("HOUDINI_USER_PREF_DIR", _DENOISER_VAR, _PDG_VAR, "KARMA_XPU"):
            merged.setdefault(k, os.environ.get(k, ""))
        return merged

    @staticmethod
    def _pref_dir(env: dict[str, str]) -> Optional[Path]:
        cand = env.get("HOUDINI_USER_PREF_DIR") or os.environ.get("HOUDINI_USER_PREF_DIR")
        if cand:
            p = Path(cand)
            if p.exists():
                return p
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            default = Path(userprofile) / "Documents" / "houdini21.0"
            if default.exists():
                return default
        return None

    @staticmethod
    def _karma_active(env: dict[str, str]) -> bool:
        if env.get("KARMA_XPU", "0") in _TRUE:
            return True
        delegate = (env.get("HOUDINI_RENDER_DELEGATE", "")
                    or env.get("HD_DEFAULT_RENDERER", ""))
        return "xpu" in delegate.lower()

    @staticmethod
    def _tops_active(procs, env: dict[str, str]) -> bool:
        for p in procs:
            try:
                cmd = " ".join(p.info.get("cmdline") or []).lower()
            except Exception:
                cmd = ""
            if "topnet" in cmd or "pdg" in cmd or "hython" in cmd:
                return True
        return env.get("HOUDINI_PDG_ACTIVE", "0") in _TRUE

    def _discover_onnx(self, state: _HoudiniState) -> list[Path]:
        found: list[Path] = []
        if state.user_pref_dir is not None:
            try:
                for onnx in state.user_pref_dir.rglob("*.onnx"):
                    if onnx.is_file():
                        found.append(onnx)
                        if len(found) >= 32:
                            break
            except OSError:
                pass
        return found
