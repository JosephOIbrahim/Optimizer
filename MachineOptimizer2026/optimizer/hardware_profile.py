"""
Static, verified hardware profile for THIS machine (DESKTOP-B2MBQRI).

Everything here was confirmed from the NVIDIA system report + nvidia-smi on
2026-07-12. Agents read this instead of re-probing constants at runtime.
Runtime-variable facts (temps, VRAM in use, fastest NVMe) live in probe.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class GPUProfile:
    name: str = "NVIDIA GeForce RTX 4090"
    brand: str = "GeForce"            # consumer, NOT Quadro/RTX-pro -> many nvidia-smi knobs locked
    arch: str = "Ada Lovelace"
    sm: str = "sm_89"
    cuda_cores: int = 16384
    vram_gb: int = 24                 # dedicated GDDR6X
    shared_sysmem_gb: int = 65        # WDDM shared surface -> sysmem fallback risk
    mem_bandwidth_tbps: float = 1.01
    driver_model: str = "WDDM"        # not TCC
    #: DATED SNAPSHOT — verified 616.92 on 2026-09-16 (NVIDIA system report).
    #: It said 610.62 until then, and nothing noticed, because nothing reads it.
    #: A driver updates itself, so a frozen version string here goes stale without
    #: anyone editing it. Do not branch on this: read it live from
    #: probe._driver(), which lands in Context.raw["driver_version"].
    studio_driver: str = "616.92"
    #: DATED SNAPSHOT — the CUDA version the 616.92 driver exposes (driver API /
    #: UMD), verified 13.4 on 2026-10-01 (nvidia-smi "CUDA UMD Version"). It said 13.3
    #: until then. Not a toolkit version (torch here is built against 12.6).
    #: Nothing reads it; same rule as studio_driver — read it live, never branch on it.
    cuda_version: str = "13.4"
    vbios: str = "95.02.47.00.01"
    pcie_gen_max: int = 4
    pcie_width: str = "16x"
    resizable_bar: bool = True
    persistence_mode_available: bool = False   # confirmed N/A on this box


@dataclass(frozen=True)
class CPUProfile:
    name: str = "AMD Ryzen Threadripper PRO 7965WX"
    cores: int = 24
    threads: int = 48
    base_ghz: float = 4.20
    arch: str = "Zen4"
    # 7965WX is a 4-CCD part; NPS/UMA confirmed at runtime by probe.
    likely_ccds: int = 4


@dataclass(frozen=True)
class DisplayProfile:
    name: str = "Apple Pro Display XDR"
    width: int = 6016
    height: int = 3384
    refresh_hz: int = 60
    bit_depth: int = 10               # 10/12-bit capable; reference/color-critical
    hdr: bool = True
    peak_nits: int = 1600
    vrr: bool = True
    color_critical: bool = True       # NEVER downgrade color/bit-depth/HDR


@dataclass(frozen=True)
class HardwareProfile:
    device_name: str = "DESKTOP-B2MBQRI"
    os_name: str = "Windows 11 Pro for Workstations"
    os_build: str = "26200"
    ram_gb: int = 128
    gpu: GPUProfile = field(default_factory=GPUProfile)
    cpu: CPUProfile = field(default_factory=CPUProfile)
    display: DisplayProfile = field(default_factory=DisplayProfile)

    # Executable names the driver profiles / affinity target.
    app_executables: Dict[str, List[str]] = field(default_factory=lambda: {
        "houdini": ["houdini.exe", "houdinifx.exe", "hython.exe", "husk.exe"],
        "unreal":  ["UnrealEditor.exe", "UnrealEditor-Cmd.exe"],
        "comfyui": ["python.exe", "pythonw.exe"],   # disambiguated by ComfyUI\main.py in cmdline
    })

    # Existing live acceleration caches to COMPLEMENT, never duplicate (see memory: V9/V11).
    external_caches: Dict[str, str] = field(default_factory=lambda: {
        "torchinductor": r"D:\OptimizerV9\torchinductor_cache",
        "tensorrt_engine": r"D:\OptimizerV9\engine_cache",
        "cutlass": r"D:\OptimizerV9\cutlass",
        "comfyui_engine": r"D:\OptimizerV8_EngineCache\comfyui",
        "houdini_engine": r"D:\OptimizerV8_EngineCache\houdini",
        "ue5_engine": r"D:\OptimizerV8_EngineCache\ue5",
    })

    # Thermal guardrails for the continuous watchdog (RTX 4090 GDDR6X).
    thermal_c_warn: int = 83
    thermal_c_reduce: int = 87
    thermal_c_critical: int = 90


PROFILE = HardwareProfile()
