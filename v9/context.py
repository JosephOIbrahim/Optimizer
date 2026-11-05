"""V9 — SystemContext + gather(). FROZEN CONTRACT: advisors import this
dataclass and read these fields; Foundation may extend gather()'s body but
must not change the field names/types advisors depend on.

Invariant (Design §4.2): gather() NEVER raises. On total probe failure it
returns a minimal idle context so the cycle continues.

Per-app deep state (ComfyUI graph, Houdini ML SOPs, UE5 plugins) is NOT
collected here — each advisor collects its own app specifics, gated on the
`running` flags below. This keeps context.py free of app-specific coupling.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from v9.hardware import HardwareProfile, detect

log = logging.getLogger("v9.context")

# process-name substrings -> logical app
_APP_PROCS = {
    "comfyui": ("comfyui", "main.py"),     # python + ComfyUI/main.py in cmdline
    "houdini": ("houdini", "hindie", "husk"),
    "ue5": ("unrealeditor", "ue4editor", "ue5editor", "unreal"),
}


@dataclass
class SystemContext:
    timestamp: float = 0.0
    profile: Optional[HardwareProfile] = None

    # live GPU state
    gpu_utilization_pct: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 24.0
    vram_total_gb: float = 24.0
    gpu_temp_c: Optional[float] = None
    gpu_power_w: Optional[float] = None

    # process / workload state
    workflow_mode: str = "idle"               # idle | sequential | parallel
    running: dict[str, bool] = field(
        default_factory=lambda: {"comfyui": False, "houdini": False, "ue5": False})
    processes: list[tuple[str, int]] = field(default_factory=list)
    foreground_app: str = "none"              # comfyui|houdini|ue5|other|none

    # advisor-owned per-app deep state (filled by advisors, not by gather)
    comfyui: Any = None
    houdini: Any = None
    ue5: Any = None

    @property
    def active_app_count(self) -> int:
        return sum(1 for v in self.running.values() if v)


def _gpu_live(ctx: SystemContext) -> None:
    """Fill live GPU util/vram/temp/power. Tries pynvml then nvidia-smi."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            ctx.gpu_utilization_pct = float(util.gpu)
            ctx.vram_total_gb = round(mem.total / 1024**3, 2)
            ctx.vram_used_gb = round(mem.used / 1024**3, 2)
            ctx.vram_free_gb = round(mem.free / 1024**3, 2)
            try:
                ctx.gpu_temp_c = float(pynvml.nvmlDeviceGetTemperature(
                    h, pynvml.NVML_TEMPERATURE_GPU))
                ctx.gpu_power_w = round(
                    pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                pass
            return
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    # nvidia-smi fallback
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4)
            if out.returncode == 0:
                f = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
                ctx.gpu_utilization_pct = float(f[0])
                ctx.vram_used_gb = round(float(f[1]) / 1024, 2)
                ctx.vram_total_gb = round(float(f[2]) / 1024, 2)
                ctx.vram_free_gb = round(ctx.vram_total_gb - ctx.vram_used_gb, 2)
                ctx.gpu_temp_c = float(f[3])
                ctx.gpu_power_w = float(f[4])
        except Exception:
            pass


def _processes(ctx: SystemContext) -> None:
    try:
        import psutil  # type: ignore
    except Exception:
        return
    for p in psutil.process_iter(["name", "pid", "cmdline"]):
        try:
            name = (p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            ctx.processes.append((name, p.info["pid"]))
            for app, needles in _APP_PROCS.items():
                if app == "comfyui":
                    if "python" in name and "main.py" in cmd and "comfyui" in cmd:
                        ctx.running["comfyui"] = True
                elif any(n in name for n in needles):
                    ctx.running[app] = True
        except Exception:
            continue


def _foreground(ctx: SystemContext) -> None:
    """Best-effort Win32 foreground-app detection; 'none' off-Windows."""
    try:
        import psutil  # type: ignore
        import ctypes
        user32 = ctypes.windll.user32          # type: ignore[attr-defined]
        hwnd = user32.GetForegroundWindow()
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = psutil.Process(pid.value).name().lower()
        for app, needles in _APP_PROCS.items():
            if app == "comfyui" and "python" in name:
                ctx.foreground_app = "comfyui"
                return
            if any(n in name for n in needles):
                ctx.foreground_app = app
                return
        ctx.foreground_app = "other"
    except Exception:
        ctx.foreground_app = "none"


def gather(profile: Optional[HardwareProfile] = None,
           config: Optional[dict] = None,
           timestamp: float = 0.0) -> SystemContext:
    """Build the base system context. Never raises."""
    ctx = SystemContext(timestamp=timestamp)
    try:
        ctx.profile = profile or detect(
            (config or {}).get("hardware") if config else None)
        if ctx.profile.gpu.vram_total_gb:
            ctx.vram_total_gb = ctx.profile.gpu.vram_total_gb
            ctx.vram_free_gb = ctx.profile.gpu.vram_total_gb
        _gpu_live(ctx)
        _processes(ctx)
        _foreground(ctx)
        n = ctx.active_app_count
        ctx.workflow_mode = "idle" if n == 0 else ("sequential" if n == 1
                                                    else "parallel")
    except Exception:
        log.warning("gather() degraded to minimal idle context", exc_info=True)
    return ctx


if __name__ == "__main__":
    c = gather()
    print(f"mode={c.workflow_mode} gpu={c.gpu_utilization_pct}% "
          f"vram={c.vram_used_gb}/{c.vram_total_gb}GB temp={c.gpu_temp_c}C "
          f"running={c.running} fg={c.foreground_app}")
