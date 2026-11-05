"""V9 — HardwareProfile: the single source of truth for the machine.

First-principles rule: no driver version, no sm_arch, no VRAM size is ever
hardcoded anywhere else in the codebase. Everything reads from here, and here
*detects* the machine at runtime (nvidia-smi -> pynvml -> psutil -> config
override). This kills the class of bug where config.yaml said one driver
version, the cache code defaulted to another, and the real driver was a third.

Dependency-light: stdlib only for the happy path; pynvml / psutil are used if
present but never required. Every probe degrades to None rather than raising.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional

# RTX-family compute-capability fallback when nvidia-smi can't report it
# (older driver/query). Keyed by a substring of the marketing name.
_SM_BY_NAME = {
    "RTX 4090": "8.9", "RTX 4080": "8.9", "RTX 4070": "8.9", "RTX 4060": "8.9",
    "RTX 5090": "12.0", "RTX 5080": "12.0",
    "RTX 3090": "8.6", "RTX 3080": "8.6",
    "A100": "8.0", "H100": "9.0",
}


@dataclass
class GPUInfo:
    name: Optional[str] = None
    driver_version: Optional[str] = None          # detected at runtime (NNN.NN)
    driver_branch: Optional[str] = None           # "Studio" | "Game Ready" | None
    compute_capability: Optional[str] = None      # e.g. "8.9"
    sm_arch: Optional[int] = None                 # e.g. 89
    vram_total_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None
    cuda_version: Optional[str] = None            # driver-reported CUDA, e.g. "13.0"
    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    resizable_bar: Optional[bool] = None
    source: str = "none"                          # which probe succeeded

    @property
    def vram_total_gb(self) -> Optional[float]:
        return round(self.vram_total_mb / 1024, 1) if self.vram_total_mb else None


@dataclass
class CPUInfo:
    name: Optional[str] = None
    physical_cores: Optional[int] = None
    logical_cores: Optional[int] = None


@dataclass
class HardwareProfile:
    gpu: GPUInfo = field(default_factory=GPUInfo)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    ram_total_gb: Optional[float] = None
    os_name: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")

    # ---- the questions the rest of the system actually asks -------------
    def cache_key_fields(self) -> dict:
        """The hardware facts that must invalidate a build cache when they
        change. Driver + arch are the load-bearing ones."""
        return {
            "driver_ver": self.gpu.driver_version or "unknown",
            "sm_arch": self.gpu.sm_arch or 0,
            "gpu_name": self.gpu.name or "unknown",
        }

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        g = self.gpu
        return (
            f"{g.name or 'GPU?'} · sm_{g.sm_arch or '?'} · "
            f"{g.vram_total_gb or '?'} GB · driver {g.driver_version or '?'}"
            f"{(' ' + g.driver_branch) if g.driver_branch else ''} · "
            f"CUDA {g.cuda_version or '?'} · "
            f"{self.cpu.name or 'CPU?'} ({self.cpu.physical_cores or '?'}c/"
            f"{self.cpu.logical_cores or '?'}t) · {self.ram_total_gb or '?'} GB RAM"
        )


# ---------------------------------------------------------------------------
# Probes — each returns partial info or raises; callers swallow exceptions.
# ---------------------------------------------------------------------------
def _run(cmd: list[str], timeout: float = 4.0) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return out.stdout if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _probe_nvidia_smi(gpu: GPUInfo) -> bool:
    """Primary GPU probe. nvidia-smi ships with every driver."""
    if not shutil.which("nvidia-smi"):
        return False
    # Query the fields that exist across driver generations; fall back per-field.
    fields = "name,driver_version,memory.total,memory.free,compute_cap"
    out = _run(["nvidia-smi", f"--query-gpu={fields}",
                "--format=csv,noheader,nounits"])
    if not out:
        # compute_cap unsupported on this driver -> retry without it.
        out = _run(["nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.free",
                    "--format=csv,noheader,nounits"])
        if not out:
            return False
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    gpu.name = parts[0] or None
    gpu.driver_version = parts[1] or None
    if len(parts) > 2 and parts[2].replace(".", "").isdigit():
        gpu.vram_total_mb = int(float(parts[2]))
    if len(parts) > 3 and parts[3].replace(".", "").isdigit():
        gpu.vram_free_mb = int(float(parts[3]))
    if len(parts) > 4 and parts[4]:
        gpu.compute_capability = parts[4]
    # CUDA driver version (separate query, not in --query-gpu on all builds)
    smi = _run(["nvidia-smi"])
    if smi:
        m = re.search(r"CUDA Version:\s*([\d.]+)", smi)
        if m:
            gpu.cuda_version = m.group(1)
    gpu.source = "nvidia-smi"
    return True


def _probe_pynvml(gpu: GPUInfo) -> bool:
    """Secondary probe — richer (PCIe gen/width) when pynvml is installed."""
    try:
        import pynvml  # type: ignore
    except ImportError:
        return False
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        if not gpu.name:
            gpu.name = pynvml.nvmlDeviceGetName(h)
            if isinstance(gpu.name, bytes):
                gpu.name = gpu.name.decode()
        if not gpu.driver_version:
            dv = pynvml.nvmlSystemGetDriverVersion()
            gpu.driver_version = dv.decode() if isinstance(dv, bytes) else dv
        if not gpu.vram_total_mb:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpu.vram_total_mb = int(mem.total / 1024 / 1024)
            gpu.vram_free_mb = int(mem.free / 1024 / 1024)
        try:
            cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            gpu.compute_capability = f"{cc[0]}.{cc[1]}"
        except Exception:
            pass
        try:
            gpu.pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h)
            gpu.pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(h)
        except Exception:
            pass
        if gpu.source == "none":
            gpu.source = "pynvml"
        return True
    except Exception:
        return False
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _derive_sm_arch(gpu: GPUInfo) -> None:
    """compute_capability '8.9' -> sm_arch 89. Fall back to name table."""
    if not gpu.compute_capability and gpu.name:
        for key, cc in _SM_BY_NAME.items():
            if key in gpu.name:
                gpu.compute_capability = cc
                break
    if gpu.compute_capability:
        try:
            major, minor = gpu.compute_capability.split(".")
            gpu.sm_arch = int(major) * 10 + int(minor)
        except (ValueError, AttributeError):
            pass


def _probe_cpu(cpu: CPUInfo) -> None:
    cpu.logical_cores = os.cpu_count()
    try:
        import psutil  # type: ignore
        cpu.physical_cores = psutil.cpu_count(logical=False)
    except Exception:
        pass
    # CPU marketing name
    name = platform.processor() or ""
    if not name or name.lower() in ("", "x86_64", "amd64", "intel64"):
        if platform.system() == "Windows":
            out = _run(["wmic", "cpu", "get", "name"]) or ""
            lines = [l.strip() for l in out.splitlines() if l.strip()
                     and l.strip().lower() != "name"]
            name = lines[0] if lines else name
    cpu.name = name or None


def _probe_ram(profile: HardwareProfile) -> None:
    try:
        import psutil  # type: ignore
        profile.ram_total_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect(config_overrides: Optional[dict] = None) -> HardwareProfile:
    """Detect the machine. config_overrides (from config.yaml hardware block)
    fill ONLY gaps the probes couldn't resolve — detection always wins over
    config so a stale config can never lie about the live driver."""
    profile = HardwareProfile()
    _probe_nvidia_smi(profile.gpu)
    _probe_pynvml(profile.gpu)        # fills PCIe + any gaps
    _derive_sm_arch(profile.gpu)
    _probe_cpu(profile.cpu)
    _probe_ram(profile)

    if config_overrides:
        _apply_overrides(profile, config_overrides)
    return profile


def _apply_overrides(profile: HardwareProfile, cfg: dict) -> None:
    """Fill only-still-missing fields from config. Never overwrites a detected
    value (detection is authoritative for live state like the driver)."""
    g = cfg.get("gpu", {}) if isinstance(cfg, dict) else {}
    if profile.gpu.name is None:
        profile.gpu.name = g.get("model")
    if profile.gpu.resizable_bar is None:
        profile.gpu.resizable_bar = g.get("resizable_bar")
    if profile.gpu.pcie_gen is None:
        profile.gpu.pcie_gen = g.get("pcie_gen")
    if profile.gpu.vram_total_mb is None and g.get("vram_total_gb"):
        profile.gpu.vram_total_mb = int(g["vram_total_gb"]) * 1024
    c = cfg.get("cpu", {}) if isinstance(cfg, dict) else {}
    if profile.cpu.name is None:
        profile.cpu.name = c.get("model")
    s = cfg.get("system", {}) if isinstance(cfg, dict) else {}
    if profile.ram_total_gb is None and s.get("ram_gb"):
        profile.ram_total_gb = float(s["ram_gb"])


if __name__ == "__main__":
    p = detect()
    print(p.summary())
    print()
    print("cache-key hardware fields:", p.cache_key_fields())
    print("gpu probe source:", p.gpu.source)
