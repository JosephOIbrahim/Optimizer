"""
Houdini State Collector (Forge-2)

Collects runtime state from a local SideFX Houdini 21 instance for the V8
specialist.

Responsibilities (per brief Section B):
    - Detects houdini.exe / houdinifx.exe / houdinicore.exe running.
    - Reads $HOUDINI_USER_PREF_DIR/houdini21.0/PYTHONPATH if present.
    - Detects active ML SOPs via the Houdini-context MCP if reachable
      (V7 already wires `mcp_integration: true`); permissive fallback when
      unreachable.
    - Captures Karma XPU active flag from environment + scene.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

logger = logging.getLogger("collector.houdini")

HOUDINI_PROCESS_NAMES = {"houdini.exe", "houdinifx.exe", "houdinicore.exe",
                         "houdini", "houdinifx", "houdinicore"}


# ---------------------------------------------------------------------------
# State exposed to the specialist
# ---------------------------------------------------------------------------

@dataclass
class HoudiniState:
    """
    Snapshot of a local Houdini 21 instance.
    Wired into SystemContext.houdini_v8 by Forge-3.
    """

    process_running: bool = False
    process_names_detected: List[str] = field(default_factory=list)
    user_pref_dir: Optional[Path] = None
    pythonpath: List[Path] = field(default_factory=list)

    # ML SOP / ONNX inventory (R1).
    ml_sop_onnx_paths: List[Path] = field(default_factory=list)

    # Karma XPU + denoiser (R2).
    karma_xpu_active: bool = False
    optix_denoiser_enabled: bool = False

    # TOPs / PDG (R3).
    tops_active: bool = False
    pdg_gpu_scheduling_enabled: Optional[bool] = None  # tri-state

    # Vellum (Phase B - R4).
    vellum_solver_active: bool = False


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class HoudiniStateCollector:
    """Builds a HoudiniState snapshot. Never raises."""

    def __init__(self, mcp_client: Optional[Any] = None) -> None:
        self.mcp_client = mcp_client

    def collect(self) -> HoudiniState:
        state = HoudiniState()
        try:
            procs = self._find_houdini_processes()
            if procs:
                state.process_running = True
                state.process_names_detected = sorted({p[0] for p in procs})
                env = self._collect_envs(procs)
                state.user_pref_dir = self._infer_pref_dir(env)
                state.pythonpath = self._read_pythonpath(state.user_pref_dir)
                state.karma_xpu_active = self._detect_karma_xpu(env)
                state.optix_denoiser_enabled = (
                    env.get("HOUDINI_OPTIX_DENOISER", "0") in {"1", "true", "True"}
                )
                state.tops_active = self._detect_tops_active(procs, env)
                pdg_flag = env.get("HOUDINI_PDG_GPU_SCHEDULING")
                if pdg_flag is None:
                    state.pdg_gpu_scheduling_enabled = None
                else:
                    state.pdg_gpu_scheduling_enabled = pdg_flag in {"1", "true", "True"}
                state.vellum_solver_active = self._detect_vellum_active(env)

            state.ml_sop_onnx_paths = self._discover_ml_sop_onnx(state)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HoudiniStateCollector.collect failed: %s", exc)
        return state

    # ------------------------------------------------------------------
    # Process discovery
    # ------------------------------------------------------------------
    def _find_houdini_processes(self):
        if psutil is None:
            return []
        out = []
        try:
            for p in psutil.process_iter(attrs=["name", "exe", "environ", "cmdline"]):
                try:
                    name = (p.info.get("name") or "").lower()
                    if name in HOUDINI_PROCESS_NAMES:
                        out.append((name, p))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as exc:  # pragma: no cover
            logger.debug("psutil.process_iter failed: %s", exc)
        return out

    def _collect_envs(self, procs) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for _, p in procs:
            try:
                env = p.environ() or {}
                for k, v in env.items():
                    merged.setdefault(k, str(v))
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
            except Exception:
                continue
        # Fall back to current process env for keys we still care about.
        for k in ("HOUDINI_USER_PREF_DIR", "HOUDINI_OPTIX_DENOISER",
                  "HOUDINI_PDG_GPU_SCHEDULING", "KARMA_XPU"):
            merged.setdefault(k, os.environ.get(k, ""))
        return merged

    # ------------------------------------------------------------------
    # Pref dir & PYTHONPATH
    # ------------------------------------------------------------------
    def _infer_pref_dir(self, env: Dict[str, str]) -> Optional[Path]:
        candidate = env.get("HOUDINI_USER_PREF_DIR") or os.environ.get("HOUDINI_USER_PREF_DIR")
        if candidate:
            p = Path(candidate)
            if p.exists():
                return p
        # Default Windows location.
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            default = Path(userprofile) / "Documents" / "houdini21.0"
            if default.exists():
                return default
        return None

    def _read_pythonpath(self, pref_dir: Optional[Path]) -> List[Path]:
        if pref_dir is None:
            return []
        candidates = [
            pref_dir / "PYTHONPATH",
            pref_dir / "houdini21.0" / "PYTHONPATH",
        ]
        for f in candidates:
            if f.exists() and f.is_file():
                try:
                    return [Path(line.strip()) for line in f.read_text(encoding="utf-8").splitlines()
                            if line.strip()]
                except OSError:
                    continue
        return []

    # ------------------------------------------------------------------
    # ML SOP ONNX discovery
    # ------------------------------------------------------------------
    def _discover_ml_sop_onnx(self, state: HoudiniState) -> List[Path]:
        """
        Best-effort ONNX discovery. First tries the houdini-context MCP
        (V7 wires it); falls back to scanning the user pref dir for .onnx.
        Non-existent paths are dropped (per brief: log + skip; no blocker).
        """
        found: List[Path] = []
        if self.mcp_client is not None:
            try:
                resp = self.mcp_client.call_tool(
                    "list_active_ml_sops", {"include_onnx_paths": True}
                ) or {}
                paths = resp.get("onnx_paths") or []
                for p in paths:
                    try:
                        candidate = Path(p)
                        if candidate.exists():
                            found.append(candidate)
                        else:
                            logger.debug("MCP-reported ONNX missing on disk: %s", p)
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug("MCP list_active_ml_sops failed: %s", exc)

        if not found and state.user_pref_dir is not None:
            try:
                for onnx in state.user_pref_dir.rglob("*.onnx"):
                    if onnx.is_file():
                        found.append(onnx)
                        if len(found) >= 32:
                            break
            except OSError:
                pass
        return found

    # ------------------------------------------------------------------
    # Feature detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_karma_xpu(env: Dict[str, str]) -> bool:
        if env.get("KARMA_XPU", "0") in {"1", "true", "True"}:
            return True
        # Heuristic: presence of HOUDINI_OCL_VENDOR=NVIDIA + RENDERDELEGATE=BRAY_HdKarmaXPU
        delegate = env.get("HOUDINI_RENDER_DELEGATE", "") or env.get("HD_DEFAULT_RENDERER", "")
        return "xpu" in delegate.lower()

    @staticmethod
    def _detect_tops_active(procs, env: Dict[str, str]) -> bool:
        for name, p in procs:
            try:
                cmd = " ".join(p.info.get("cmdline") or []).lower()
            except Exception:
                cmd = ""
            if "topnet" in cmd or "pdg" in cmd or "hython" in name:
                return True
        return env.get("HOUDINI_PDG_ACTIVE", "0") in {"1", "true", "True"}

    @staticmethod
    def _detect_vellum_active(env: Dict[str, str]) -> bool:
        return env.get("HOUDINI_VELLUM_ACTIVE", "0") in {"1", "true", "True"}
