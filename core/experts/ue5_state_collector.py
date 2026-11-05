"""
UE5 State Collector (Forge-2)

Collects runtime state from a local Unreal Engine 5 instance for the V8
specialist.

Responsibilities (per brief Section C):
    - Detects UnrealEditor.exe / shipped game exes.
    - Reads project's Plugins/ directory for NIM Plugin, DLSS plugin.
    - Reads UserEngine.ini (path already in V7 config) for active CVars.
    - Checks DLSS DLL versions in
      Engine/Plugins/Runtime/Nvidia/DLSS/Binaries/.
"""

from __future__ import annotations

import configparser
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

logger = logging.getLogger("collector.ue5")

UE5_PROCESS_NAMES = {
    "unrealeditor.exe",
    "unrealeditor",
    "ue5editor.exe",
    "ue4editor.exe",
}

DLSS_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


# ---------------------------------------------------------------------------
# State exposed to the specialist
# ---------------------------------------------------------------------------

@dataclass
class UE5State:
    """Snapshot of a local UE5 instance. Wired into SystemContext.ue5_v8."""

    process_running: bool = False
    editor_exe: Optional[Path] = None
    project_dir: Optional[Path] = None
    engine_dir: Optional[Path] = None
    engine_ini: Optional[Path] = None

    # Plugins.
    nim_plugin_installed: bool = False
    dlss_plugin_installed: bool = False
    dlss_dll_version: Optional[str] = None  # "4.0.10" style or None

    # CVars parsed from UserEngine.ini / DefaultEngine.ini.
    cvars: Dict[str, str] = field(default_factory=dict)

    # Path tracer in use (heuristic from cvars).
    path_tracer_in_use: bool = False


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class UE5StateCollector:
    """Builds a UE5State snapshot. Never raises."""

    def __init__(self, project_search_roots: Optional[List[Path]] = None) -> None:
        self.project_search_roots = list(project_search_roots) if project_search_roots else []

    def collect(self) -> UE5State:
        state = UE5State()
        try:
            proc = self._find_ue5_process()
            if proc is not None:
                state.process_running = True
                state.editor_exe = self._extract_exe(proc)
                state.engine_dir = self._infer_engine_dir(state.editor_exe)
                state.project_dir = self._infer_project_dir(proc)

            # Even if no process, allow inference from configured roots.
            if state.project_dir is None and self.project_search_roots:
                state.project_dir = self.project_search_roots[0]

            if state.project_dir is not None:
                state.engine_ini = self._find_engine_ini(state.project_dir)
                state.cvars = self._parse_cvars(state.engine_ini)
                state.nim_plugin_installed = self._has_plugin(state.project_dir, "NVIDIA_NIM")
                state.dlss_plugin_installed = self._has_plugin(state.project_dir, "DLSS")

            if state.engine_dir is not None:
                state.dlss_dll_version = self._read_dlss_version(state.engine_dir)

            state.path_tracer_in_use = (
                state.cvars.get("r.PathTracing", "0") in {"1", "true", "True"}
                or state.cvars.get("r.PathTracing.Enable", "0") in {"1", "true", "True"}
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("UE5StateCollector.collect failed: %s", exc)
        return state

    # ------------------------------------------------------------------
    # Process discovery
    # ------------------------------------------------------------------
    def _find_ue5_process(self):
        if psutil is None:
            return None
        try:
            for p in psutil.process_iter(attrs=["name", "exe", "cmdline"]):
                try:
                    name = (p.info.get("name") or "").lower()
                    if name in UE5_PROCESS_NAMES:
                        return p
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as exc:  # pragma: no cover
            logger.debug("psutil.process_iter failed: %s", exc)
        return None

    def _extract_exe(self, proc) -> Optional[Path]:
        try:
            exe = proc.info.get("exe")
            if exe:
                return Path(exe)
        except Exception:
            pass
        return None

    def _infer_engine_dir(self, exe: Optional[Path]) -> Optional[Path]:
        if exe is None:
            return None
        # UE5 layout: <Engine>/Binaries/Win64/UnrealEditor.exe
        try:
            return exe.parent.parent.parent
        except Exception:
            return None

    def _infer_project_dir(self, proc) -> Optional[Path]:
        try:
            cmdline = proc.info.get("cmdline") or []
        except Exception:
            cmdline = []
        for tok in cmdline:
            tok_l = tok.lower()
            if tok_l.endswith(".uproject"):
                p = Path(tok).parent
                if p.exists():
                    return p
        return None

    # ------------------------------------------------------------------
    # Project layout
    # ------------------------------------------------------------------
    def _find_engine_ini(self, project_dir: Path) -> Optional[Path]:
        candidates = [
            project_dir / "Saved" / "Config" / "WindowsEditor" / "UserEngine.ini",
            project_dir / "Saved" / "Config" / "Windows" / "UserEngine.ini",
            project_dir / "Config" / "DefaultEngine.ini",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _parse_cvars(self, engine_ini: Optional[Path]) -> Dict[str, str]:
        if engine_ini is None or not engine_ini.exists():
            return {}
        cvars: Dict[str, str] = {}
        parser = configparser.ConfigParser(strict=False, interpolation=None)
        parser.optionxform = str  # preserve case (CVars are case-sensitive enough to matter)
        try:
            parser.read(engine_ini, encoding="utf-8")
        except (configparser.Error, OSError, UnicodeDecodeError) as exc:
            logger.debug("Failed to parse %s: %s", engine_ini, exc)
            return {}
        for section in parser.sections():
            if "ConsoleVariables" in section or section.endswith("SystemSettings"):
                for k, v in parser.items(section):
                    cvars[k] = v
        return cvars

    def _has_plugin(self, project_dir: Path, plugin_name_substr: str) -> bool:
        plugins_dir = project_dir / "Plugins"
        if not plugins_dir.exists():
            return False
        try:
            for child in plugins_dir.iterdir():
                if not child.is_dir():
                    continue
                if plugin_name_substr.lower() in child.name.lower():
                    return True
                # Also peek at .uplugin files inside.
                try:
                    for uplugin in child.glob("*.uplugin"):
                        if plugin_name_substr.lower() in uplugin.stem.lower():
                            return True
                except OSError:
                    continue
        except OSError:
            return False
        return False

    # ------------------------------------------------------------------
    # DLSS
    # ------------------------------------------------------------------
    def _read_dlss_version(self, engine_dir: Path) -> Optional[str]:
        bin_dir = engine_dir / "Plugins" / "Runtime" / "Nvidia" / "DLSS" / "Binaries"
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
        # Try Windows file-version metadata first via ctypes (no extra deps).
        try:
            import ctypes
            from ctypes import wintypes  # type: ignore
            size = ctypes.windll.version.GetFileVersionInfoSizeW(str(dll_path), None)
            if not size:
                return None
            buf = ctypes.create_string_buffer(size)
            if not ctypes.windll.version.GetFileVersionInfoW(str(dll_path), 0, size, buf):
                return None
            value = ctypes.c_void_p()
            length = ctypes.c_uint()
            if not ctypes.windll.version.VerQueryValueW(
                buf, "\\", ctypes.byref(value), ctypes.byref(length)
            ):
                return None
            ffi = ctypes.cast(value, ctypes.POINTER(ctypes.c_uint * 13)).contents
            ms, ls = ffi[2], ffi[3]
            major = (ms >> 16) & 0xFFFF
            minor = ms & 0xFFFF
            build = (ls >> 16) & 0xFFFF
            return f"{major}.{minor}.{build}"
        except Exception:
            # Fallback: parse version-looking substring out of filename.
            m = DLSS_VERSION_RE.search(dll_path.name)
            if m:
                return ".".join(g for g in m.groups() if g)
        return None
