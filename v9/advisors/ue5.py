"""V9 — Unreal Engine 5.6+ advisor.

Merges core/experts/ue5_specialist.py (NIM / DLSS / Path-Tracer denoiser, with
its state collector) and the v6 UE5Engineer cvar tuning (Nanite streaming pool,
Lumen hardware ray tracing, VSM quality). Collects its own UE5 state inside
_analyze, gated on ctx.running["ue5"].

Decision rules:
  R1  NVIDIA NIM plugin not present -> advise marketplace install (notify)
  R2  DLSS DLL older than min version -> advise DLSS 4 upgrade (notify)
  R3  Path Tracer in use + denoiser cvar off -> set r.PathTracing.Denoiser=1
  R4  Free VRAM headroom -> raise Nanite streaming pool
  R5  Lumen hardware ray tracing not enabled -> enable it
  R6  VSM cache/page-pool quality cvars

All ADJUST_QUALITY recs carry Engine.ini cvar payloads; the applier writes them.
"""
from __future__ import annotations

import configparser
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from v9.advisors.base import Advisor
from v9.recommend import Action, Recommendation

if TYPE_CHECKING:
    from v9.context import SystemContext

log = logging.getLogger("v9.advisor.ue5")

_PROC_NAMES = {"unrealeditor.exe", "unrealeditor", "ue5editor.exe", "ue4editor.exe"}
_DLSS_VER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_DENOISER_CVAR = "r.PathTracing.Denoiser"
_TRUE = {"1", "true", "True"}
_DLSS_MIN_VERSION = "4.0"
_NIM_URL = "https://www.unrealengine.com/marketplace/en-US/product/nvidia-nim"


@dataclass
class _UE5State:
    process_running: bool = False
    engine_dir: Optional[Path] = None
    project_dir: Optional[Path] = None
    engine_ini: Optional[Path] = None
    nim_plugin_installed: bool = False
    dlss_plugin_installed: bool = False
    dlss_dll_version: Optional[str] = None
    cvars: dict[str, str] = field(default_factory=dict)
    path_tracer_in_use: bool = False


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
    ta, tb = _version_tuple(a), _version_tuple(b)
    if ta is None or tb is None:
        return False
    return ta < tb


class UE5Advisor(Advisor):
    name = "ue5"
    min_confidence = 0.78

    def __init__(self, config=None, cache_handle=None, quality_ledger=None):
        super().__init__(config, cache_handle, quality_ledger)
        self.nim_url = str(self.config.get("nim_marketplace_url") or _NIM_URL)
        self.dlss_min = str(self.config.get("dlss_min_version", _DLSS_MIN_VERSION))
        self.project_roots = [Path(p) for p in
                              (self.config.get("project_search_roots") or [])]
        self.nanite_pool_mb = int(self.config.get("nanite_pool_mb", 12288))
        self.vram_headroom_gb = float(self.config.get("vram_headroom_gb", 10.0))

    # ------------------------------------------------------------------
    def _analyze(self, ctx: "SystemContext") -> list[Recommendation]:
        if not ctx.running.get("ue5"):
            return []
        state = self._collect()
        ctx.ue5 = state
        if not state.process_running:
            return []

        recs: list[Recommendation] = []
        cvars = state.cvars
        ini = str(state.engine_ini) if state.engine_ini else None

        # ---- R1: NIM plugin ----------------------------------------------
        if not state.nim_plugin_installed:
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.NOTIFY,
                target="ue5:nim_plugin",
                confidence=0.78,
                expected_impact_pct=30.0,
                rationale=(
                    "NVIDIA NIM plugin not present; it exposes TRT-LLM serving "
                    "inside UE5. Install is user-driven via the Marketplace."
                ),
                parameters={"plugin": "NVIDIA_NIM", "marketplace_url": self.nim_url},
            ))

        # ---- R2: DLSS version --------------------------------------------
        if state.dlss_dll_version and _version_lt(state.dlss_dll_version, self.dlss_min):
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.NOTIFY,
                target="ue5:dlss_version",
                confidence=0.85,
                expected_impact_pct=15.0,
                rationale=(
                    f"DLSS DLL {state.dlss_dll_version} < required "
                    f"{self.dlss_min}; upgrade to DLSS 4 / DLSS-RR for Ray "
                    "Reconstruction."
                ),
                parameters={"plugin": "DLSS_4", "current": state.dlss_dll_version,
                            "minimum": self.dlss_min},
            ))

        # ---- R3: Path Tracer denoiser ------------------------------------
        if state.path_tracer_in_use and cvars.get(_DENOISER_CVAR) != "1":
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.ADJUST_QUALITY,
                target="ue5:path_tracer_denoiser",
                confidence=0.90,
                expected_impact_pct=12.0,
                rationale=(
                    "Path Tracer in use but r.PathTracing.Denoiser != 1; "
                    "enabling the GPU denoiser converges stills ~12% faster."
                ),
                parameters={"engine_ini_path": ini,
                            "section": "/Script/Engine.RendererSettings",
                            "cvars": {_DENOISER_CVAR: "1"}},
            ))

        # ---- R4: Nanite streaming pool (VRAM headroom) -------------------
        if ctx.vram_free_gb > self.vram_headroom_gb:
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.ADJUST_QUALITY,
                target="ue5:nanite_streaming_pool",
                confidence=0.90,
                expected_impact_pct=30.0,
                rationale=(
                    f"{ctx.vram_free_gb:.1f} GB VRAM free; raise the Nanite "
                    f"streaming pool to {self.nanite_pool_mb} MB to eliminate "
                    "streaming hitches on large scenes."
                ),
                parameters={"engine_ini_path": ini,
                            "section": "/Script/Engine.RendererSettings",
                            "cvars": {
                                "r.Streaming.PoolSize": str(self.nanite_pool_mb),
                                "r.Nanite.Streaming.NumInitialRootPages": "512",
                            }},
            ))

        # ---- R5: Lumen hardware ray tracing ------------------------------
        if cvars.get("r.Lumen.HardwareRayTracing") != "1":
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.ADJUST_QUALITY,
                target="ue5:lumen_hardware_rt",
                confidence=0.93,
                expected_impact_pct=100.0,
                rationale=(
                    "Lumen hardware ray tracing not enabled; dedicated RT cores "
                    "give 2x+ over software Lumen."
                ),
                parameters={"engine_ini_path": ini,
                            "section": "/Script/Engine.RendererSettings",
                            "cvars": {
                                "r.Lumen.HardwareRayTracing": "1",
                                "r.Lumen.HardwareRayTracing.LightingMode": "2",
                            }},
            ))

        # ---- R6: VSM quality ---------------------------------------------
        recs.append(Recommendation(
            advisor=self.name,
            action=Action.ADJUST_QUALITY,
            target="ue5:vsm_quality",
            confidence=0.85,
            expected_impact_pct=20.0,
            rationale=(
                "Enable static shadow caching and raise the VSM physical page "
                "pool; ample VRAM supports higher-quality virtual shadow maps."
            ),
            parameters={"engine_ini_path": ini,
                        "section": "/Script/Engine.RendererSettings",
                        "cvars": {
                            "r.Shadow.Virtual.Cache.StaticSeparate": "1",
                            "r.Shadow.Virtual.MaxPhysicalPages": "4096",
                        }},
        ))

        return recs

    # ------------------------------------------------------------------
    # state collection (never raises)
    # ------------------------------------------------------------------
    def _collect(self) -> _UE5State:
        state = _UE5State(process_running=True)
        try:
            proc = self._find_proc()
            if proc is not None:
                exe = self._exe(proc)
                state.engine_dir = self._engine_dir(exe)
                state.project_dir = self._project_dir(proc)
            else:
                state.process_running = False

            if state.project_dir is None and self.project_roots:
                state.project_dir = self.project_roots[0]

            if state.project_dir is not None:
                state.engine_ini = self._find_engine_ini(state.project_dir)
                state.cvars = self._parse_cvars(state.engine_ini)
                state.nim_plugin_installed = self._has_plugin(state.project_dir, "NVIDIA_NIM")
                state.dlss_plugin_installed = self._has_plugin(state.project_dir, "DLSS")
            if state.engine_dir is not None:
                state.dlss_dll_version = self._dlss_version(state.engine_dir)

            state.path_tracer_in_use = (
                state.cvars.get("r.PathTracing", "0") in _TRUE
                or state.cvars.get("r.PathTracing.Enable", "0") in _TRUE)
        except Exception:
            log.debug("ue5 state collect degraded", exc_info=True)
        return state

    @staticmethod
    def _find_proc():
        try:
            import psutil  # type: ignore
        except Exception:
            return None
        try:
            for p in psutil.process_iter(attrs=["name", "exe", "cmdline"]):
                try:
                    if (p.info.get("name") or "").lower() in _PROC_NAMES:
                        return p
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            return None
        return None

    @staticmethod
    def _exe(proc) -> Optional[Path]:
        try:
            exe = proc.info.get("exe")
            return Path(exe) if exe else None
        except Exception:
            return None

    @staticmethod
    def _engine_dir(exe: Optional[Path]) -> Optional[Path]:
        if exe is None:
            return None
        try:
            return exe.parent.parent.parent
        except Exception:
            return None

    @staticmethod
    def _project_dir(proc) -> Optional[Path]:
        try:
            for tok in proc.info.get("cmdline") or []:
                if tok.lower().endswith(".uproject"):
                    p = Path(tok).parent
                    if p.exists():
                        return p
        except Exception:
            pass
        return None

    @staticmethod
    def _find_engine_ini(project_dir: Path) -> Optional[Path]:
        for c in (project_dir / "Saved/Config/WindowsEditor/UserEngine.ini",
                  project_dir / "Saved/Config/Windows/UserEngine.ini",
                  project_dir / "Config/DefaultEngine.ini"):
            if c.exists():
                return c
        return None

    @staticmethod
    def _parse_cvars(engine_ini: Optional[Path]) -> dict[str, str]:
        if engine_ini is None or not engine_ini.exists():
            return {}
        parser = configparser.ConfigParser(strict=False, interpolation=None)
        parser.optionxform = str  # CVars are case-sensitive
        try:
            parser.read(engine_ini, encoding="utf-8")
        except (configparser.Error, OSError, UnicodeDecodeError):
            return {}
        cvars: dict[str, str] = {}
        for section in parser.sections():
            if "ConsoleVariables" in section or section.endswith("SystemSettings"):
                for k, v in parser.items(section):
                    cvars[k] = v
        return cvars

    @staticmethod
    def _has_plugin(project_dir: Path, substr: str) -> bool:
        plugins = project_dir / "Plugins"
        if not plugins.exists():
            return False
        try:
            for child in plugins.iterdir():
                if not child.is_dir():
                    continue
                if substr.lower() in child.name.lower():
                    return True
                try:
                    for up in child.glob("*.uplugin"):
                        if substr.lower() in up.stem.lower():
                            return True
                except OSError:
                    continue
        except OSError:
            return False
        return False

    def _dlss_version(self, engine_dir: Path) -> Optional[str]:
        bin_dir = engine_dir / "Plugins/Runtime/Nvidia/DLSS/Binaries"
        if not bin_dir.exists():
            return None
        try:
            for dll in bin_dir.rglob("nvngx_dlss*.dll"):
                ver = self._dll_version(dll)
                if ver:
                    return ver
        except OSError:
            return None
        return None

    @staticmethod
    def _dll_version(dll_path: Path) -> Optional[str]:
        try:
            import ctypes
            ver_dll = ctypes.windll.version  # type: ignore[attr-defined]
            size = ver_dll.GetFileVersionInfoSizeW(str(dll_path), None)
            if not size:
                raise OSError
            buf = ctypes.create_string_buffer(size)
            if not ver_dll.GetFileVersionInfoW(str(dll_path), 0, size, buf):
                raise OSError
            value = ctypes.c_void_p()
            length = ctypes.c_uint()
            if not ver_dll.VerQueryValueW(buf, "\\", ctypes.byref(value),
                                          ctypes.byref(length)):
                raise OSError
            ffi = ctypes.cast(value, ctypes.POINTER(ctypes.c_uint * 13)).contents
            ms, ls = ffi[2], ffi[3]
            return f"{(ms >> 16) & 0xFFFF}.{ms & 0xFFFF}.{(ls >> 16) & 0xFFFF}"
        except Exception:
            m = _DLSS_VER_RE.search(dll_path.name)
            if m:
                return ".".join(g for g in m.groups() if g)
        return None
