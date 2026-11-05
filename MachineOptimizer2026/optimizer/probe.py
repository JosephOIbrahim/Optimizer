"""
Probe live machine state into a Context. Everything here is best-effort: a failed
probe degrades a field to a safe default, it never raises.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from typing import Dict, List, Optional

from .base import Context
from .hardware_profile import PROFILE

# nvidia-smi clock-throttle bitmask -> human reason (the ones that matter for a watchdog)
_THROTTLE_BITS = {
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000020: "sw_thermal",
    0x0000000000000040: "hw_thermal",
    0x0000000000000080: "hw_power_brake",
}


def _run(cmd: list, timeout: int = 15) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def _run_ps(script: str, timeout: int = 20) -> str:
    return _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout)


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _gpu(ctx: Context) -> None:
    fields = ("utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,"
              "pstate,pcie.link.gen.current,clocks_throttle_reasons.active")
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not out:
        return
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    try:
        ctx.gpu_util = int(float(parts[0]))
        ctx.vram_used_gb = round(float(parts[1]) / 1024, 2)
        ctx.vram_total_gb = round(float(parts[2]) / 1024, 2)
        ctx.temp_c = int(float(parts[3]))
        ctx.power_w = round(float(parts[4]), 1)
        ctx.raw["pstate"] = parts[5]
        ctx.pcie_gen = int(float(parts[6]))
        mask = int(parts[7], 16) if parts[7].lower().startswith("0x") else int(parts[7] or 0)
        ctx.throttle_reasons = [name for bit, name in _THROTTLE_BITS.items() if mask & bit]
    except (ValueError, IndexError):
        pass


def _running(ctx: Context) -> None:
    running: Dict[str, bool] = {"houdini": False, "unreal": False, "comfyui": False}
    ps = ("Get-CimInstance Win32_Process | "
          "Select-Object Name,CommandLine | ConvertTo-Csv -NoTypeInformation")
    out = _run_ps(ps)
    hou = set(x.lower() for x in PROFILE.app_executables["houdini"])
    ue = set(x.lower() for x in PROFILE.app_executables["unreal"])
    for line in out.splitlines():
        low = line.lower()
        name = low.split(",", 1)[0].strip('"')
        if name in hou:
            running["houdini"] = True
        if name in ue:
            running["unreal"] = True
        if name in ("python.exe", "pythonw.exe") and ("comfyui" in low or "main.py" in low):
            running["comfyui"] = True
    ctx.running = running


def _cpu(ctx: Context) -> None:
    ctx.logical_cores = os.cpu_count() or PROFILE.cpu.threads
    out = _run_ps("(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
    try:
        ctx.physical_cores = int(out) or PROFILE.cpu.cores
    except ValueError:
        ctx.physical_cores = PROFILE.cpu.cores
    ctx.ccds = PROFILE.cpu.likely_ccds


def _fastest_nvme(ctx: Context) -> None:
    # Pick the fixed volume with the most free space for scratch/cache/DDC — but NEVER the
    # OS drive (so scratch/pagefile/Defender exclusions can't land on the system volume).
    ps = ("$sys=$env:SystemDrive.TrimEnd(':'); Get-Volume | "
          "Where-Object { $_.DriveLetter -and $_.DriveType -eq 'Fixed' -and $_.DriveLetter -ne $sys } | "
          "Sort-Object SizeRemaining -Descending | Select-Object -First 1 -ExpandProperty DriveLetter")
    letter = _run_ps(ps).strip()
    if letter:
        ctx.fastest_nvme = f"{letter}:"


def _python(ctx: Context) -> None:
    # Prefer 3.13 then 3.12 for VFX/ML wheel compatibility (3.14 is default but riskier).
    out = _run(["py", "-0p"])
    chosen = None
    for pref in ("3.13", "3.12"):
        for line in out.splitlines():
            if pref in line:
                parts = line.split()
                chosen = parts[-1] if parts else None
                break
        if chosen:
            break
    ctx.python_exe = chosen or shutil.which("python")


def _driver(ctx: Context) -> None:
    """Live driver version.

    PROFILE.gpu.studio_driver is a dated snapshot: it sat at 610.62 while the
    machine ran 616.92, and nothing caught it because nothing consumed the field.
    Anything that actually needs the number reads it here instead.
    """
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if out:
        ctx.raw["driver_version"] = out.splitlines()[0].strip()


def probe() -> Context:
    ctx = Context()
    ctx.is_admin = is_admin()
    _gpu(ctx)
    _driver(ctx)
    _running(ctx)
    _cpu(ctx)
    _fastest_nvme(ctx)
    _python(ctx)
    return ctx
