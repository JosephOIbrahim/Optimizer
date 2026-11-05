"""V9 — typed config loader.

One config, no V6/V8 duality. `load_config()` returns a `Config` that is:
  * dict-like     — `cfg.get("gpu", {})`, `cfg["gpu"]`, `"gpu" in cfg`
  * section-attr  — `cfg.gpu` returns the section dict
  * default-first — every section has sane defaults baked in, so the system
                    runs with NO config file at all.

Layering (lowest → highest precedence):
  1. DEFAULTS                  (this module — runs headless)
  2. config.yaml on disk       (deep-merged over defaults)
  3. live HardwareProfile      (detection ALWAYS wins for live HW state)

Principle (DESIGN_V9 §4.1): detect, don't hardcode. The `hardware` section of
config is only OVERRIDE/GAP hints — it never carries a driver version, and
`hardware.detect()` is authoritative for everything it can probe. We expose the
detected profile on `cfg.profile` so advisors don't each re-detect.

Section names (must match advisors): hardware, gpu, system, comfyui, houdini,
ue5, coordinator, tensorrt, monitoring.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

from v9.hardware import HardwareProfile, detect
from v9.paths import cache_dir, db_path, log_path

log = logging.getLogger("v9.config")

# Default config filename sits beside this module.
_DEFAULT_CONFIG_NAME = "config.yaml"


# ---------------------------------------------------------------------------
# DEFAULTS — the system must run fully headless from these alone.
# No driver version, no arch literal: detection owns live hardware truth.
# These carry forward the genuinely useful tuning values ported from v8.
# ---------------------------------------------------------------------------
def _defaults() -> dict[str, Any]:
    return {
        "identity": {
            "name": "Optimizer V9",
            "version": "9.0",
        },

        # GAP/OVERRIDE hints only. detect() fills these; config never lies
        # about the live driver. vram/pcie/rebar are last-resort fallbacks
        # used ONLY when a probe couldn't resolve them.
        "hardware": {
            "gpu": {
                # model is a fallback name if nvidia-smi/pynvml are absent.
                "model": None,
                "resizable_bar": None,
                "pcie_gen": None,
                "vram_total_gb": None,
            },
            "cpu": {
                "model": None,
            },
            "system": {
                "ram_gb": None,
            },
        },

        # --- GPU config-tuning advisor (CUDA env, driver knobs) ----------
        "gpu": {
            "enabled": True,
            "min_confidence": 0.85,
            "cuda_env_vars": {
                "CUDA_CACHE_MAXSIZE": "4294967296",      # 4 GB shader cache
                "CUDA_CACHE_DISABLE": "0",
                "CUDA_FORCE_PTX_JIT": "0",
                "CUDA_MODULE_LOADING": "LAZY",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            "cudnn_env_vars": {
                "CUDNN_BENCHMARK": "1",
                "TORCH_CUDNN_V8_API_ENABLED": "1",
            },
            "power_mode": "prefer_maximum_performance",
        },

        # --- System integration advisor (registry, priority, thermal) ---
        "system": {
            "enabled": True,
            "min_confidence": 0.90,
            "process_priority": {
                "houdini": "ABOVE_NORMAL",
                "unrealeditor": "ABOVE_NORMAL",
                "comfyui": "NORMAL",
            },
            "cpu_affinity": {
                "enable_pinning": True,
                "performance_cores": list(range(0, 21)),   # 0-20
                "background_cores": [21, 22, 23],
            },
            "thermal": {
                "gpu_temp_warning_c": 83,
                "gpu_temp_critical_c": 90,
                "power_limit_w": 450,
            },
            "pcie": {
                "check_gen4": True,
                "check_x16_lanes": True,
                "check_resizable_bar": True,
            },
            "windows_gpu_scheduler": {
                "enabled": True,
                "registry_key":
                    r"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet"
                    r"\Control\GraphicsDrivers",
                "value_name": "HwSchMode",
                "value_data": 2,                            # HAGS on
            },
        },

        # --- ComfyUI build-pipeline advisor (primary workload) -----------
        "comfyui": {
            "enabled": True,
            "min_confidence": 0.80,
            "api_host": "127.0.0.1",
            "api_port": 8188,
            # install_dir is a fallback hint only; the advisor auto-detects the
            # real path from a running process, the comfy-cli workspace, and
            # common locations (this box: C:/ComfyUI).
            "install_dir": "C:/ComfyUI",
            "watch_models_dir": "C:/ComfyUI/models",
            "websocket_subscribe": True,
            # CLI renders: scan the output dir for recently written images whose
            # embedded workflow JSON reveals (model, shape) even with no live
            # server. Empty -> derived from install_dir/output.
            "output_dir": "",
            "cli_scan_enabled": True,
            "env_vars": {
                "PYTORCH_CUDA_ALLOC_CONF":
                    "max_split_size_mb:512,garbage_collection_threshold:0.9",
                "FORCE_MEM_EFFICIENT_ATTN": "1",
                "CUDA_MODULE_LOADING": "LAZY",
            },
            "speculative_compile": {
                "min_usage_count": 3,
                "lookback_days": 14,
                "only_when_gpu_idle_pct": 5,
                "idle_dwell_seconds": 30,
                "max_concurrent_compiles": 1,
            },
        },

        # --- Houdini build-pipeline advisor ------------------------------
        "houdini": {
            "enabled": True,
            "min_confidence": 0.80,
            "env_vars": {
                "HOUDINI_OCL_DEVICENUMBER": "0",
                "HOUDINI_OCL_DEVICETYPE": "GPU",
                "HOUDINI_OCL_VENDOR": "NVIDIA",
                "HOUDINI_VDB_FAST_TEMP_FILE": "1",
                "HOUDINI_DSO_ERROR": "2",
            },
            "karma_xpu": {
                "samples_per_pixel_min": 256,
                "samples_per_pixel_max": 2048,
                "enable_denoiser": True,
                "max_ray_depth": 8,
                "russian_roulette_depth": 3,
            },
            "pdg_tops": {
                # match detected logical-core count at runtime if available;
                # this is the fallback used only when detection is unavailable.
                "max_concurrent_cooks": 24,
                "gpu_scheduling": True,
            },
        },

        # --- UE5 build-pipeline advisor (Engine.ini cvars) ---------------
        "ue5": {
            "enabled": True,
            "min_confidence": 0.80,
            "engine_ini_path":
                "C:/Users/User/AppData/Local/UnrealEngine/Engine/Config/"
                "UserEngine.ini",
            "console_variables": {
                # Lumen
                "r.Lumen.HardwareRayTracing": "1",
                "r.Lumen.HardwareRayTracing.LightingMode": "2",
                "r.Lumen.Reflections.ScreenTraces": "4",
                "r.Lumen.RadiosityProbeSpacing": "4.0",
                "r.Lumen.TranslucencyReflections.FrontLayer."
                "EnableForProject": "1",
                # Nanite
                "r.Nanite.MaxPixelsPerEdge": "1.0",
                "r.Nanite.Streaming.NumInitialRootPages": "512",
                # Virtual Shadow Maps
                "r.Shadow.Virtual.Cache.StaticSeparate": "1",
                "r.Shadow.Virtual.ResolutionLodBiasDirectional": "0.0",
                "r.Shadow.Virtual.MaxPhysicalPages": "4096",
                # Path Tracer
                "r.PathTracing.SamplesPerPixel": "256",
                "r.PathTracing.MaxBounces": "32",
                "r.PathTracing.Denoiser": "1",
            },
            "ddc_path": "D:/UE_DDC",
            "ddc_fallback_path": "C:/UE_DDC",
        },

        # --- Cross-app coordinator (foreground wins; bg compiles pause) --
        "coordinator": {
            "enabled": True,
            "min_confidence": 0.85,
            "idle_window_seconds": 60.0,
            # VRAM split strategies (GB). Totals stay under detected VRAM.
            "vram_split": {
                "one_app_active": {"primary": 18},
                "two_apps_active":
                    {"comfyui": 11, "houdini": 11, "ue5": 11},
                "three_apps_active":
                    {"comfyui": 3, "houdini": 11, "ue5": 8},
            },
            "fencing": {
                "soft_limit_percent": 90,
                "hard_limit_percent": 95,
                "monitoring_interval_ms": 100,
            },
        },

        # --- TensorRT engine cache + background compiler -----------------
        # NOTE: cache_dir is resolved at load time from paths.py (off-OneDrive),
        # never a hardcoded D:\ literal. invalidate_on_driver_change is keyed
        # off HardwareProfile.cache_key_fields() in the build layer.
        "tensorrt": {
            "enabled": True,
            "precision_priority": ["fp16"],          # fp8 added later
            "cache_size_gb": 200,
            "invalidate_on_driver_change": True,
            # quality admission gate (Pareto): keep only if all hold.
            "quality_gate": {
                "min_speedup": 1.30,
                "max_lpips": 0.04,
                "min_clip_delta": -0.02,
            },
        },

        # --- CUTLASS acceleration (headline ingredient) ------------------
        # The CUTLASS FP8 GEMM backend is opt-in: it only joins the recipe when
        # `enabled` is true AND the toolchain is present (run `v9 cutlass
        # bootstrap`). Its first wired home is the Houdini Vellum/Pyro solve
        # GEMM (see v9/accel/vellum.py). install_dir is a hint; the bootstrapper
        # defaults to an off-OneDrive path under the V9 data root.
        "acceleration": {
            "cutlass": {
                "enabled": False,
                "auto_bootstrap": False,    # never clone inside an opt cycle
                "install_dir": "",          # "" -> paths.DATA_ROOT/cutlass
                "version_tag": "v4.5.1",
                # Houdini Vellum/Pyro: the constraint/advection solve is cast as
                # batched FP8 GEMM. These are the default contraction dims used
                # when the advisor can't infer them from the live solver.
                "vellum": {
                    "enabled": True,
                    "precision": "fp8_e4m3",
                    "default_gemm": [4096, 4096, 4096],   # (M, N, K), 8-aligned
                },
            },
        },

        # --- monitoring / telemetry --------------------------------------
        "monitoring": {
            "enabled": True,
            "sampling_interval_ms": 100,
            "ring_buffer_duration_seconds": 3600,
            "alerts": {
                "vram_warning_threshold_percent": 90,
                "vram_critical_threshold_percent": 95,
                "temp_warning_c": 83,
                "temp_critical_c": 90,
                "power_limit_w": 450,
            },
        },

        # --- logging ------------------------------------------------------
        "logging": {
            "level": "INFO",
            "max_size_mb": 100,
            "backup_count": 5,
            "verbose_advisors": False,
        },
    }


# ---------------------------------------------------------------------------
# deep merge: override values win, dicts merge recursively
# ---------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file → dict. Degrades to {} on any failure (missing file,
    no PyYAML, parse error) so the system always runs on defaults."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        log.warning("PyYAML not installed; ignoring %s, using defaults", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        log.warning("failed to parse %s; using defaults", path, exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Config — dict-like wrapper with attribute + section access
# ---------------------------------------------------------------------------
class Config:
    """Read-only-ish view over the merged config dict, plus the detected
    HardwareProfile. Sections are plain dicts.

    Access styles (all equivalent for reading a section):
        cfg.get("gpu", {})        # dict-like, with default
        cfg["gpu"]                # subscript (KeyError if missing)
        cfg.gpu                   # attribute (AttributeError if missing)
        "gpu" in cfg              # membership
    """

    def __init__(self, data: dict[str, Any], profile: HardwareProfile,
                 source_path: Optional[Path] = None) -> None:
        self._data = data
        self.profile = profile
        self.source_path = source_path

    # --- dict-like ------------------------------------------------------
    def get(self, section: str, default: Any = None) -> Any:
        return self._data.get(section, default)

    def __getitem__(self, section: str) -> Any:
        return self._data[section]

    def __contains__(self, section: str) -> bool:
        return section in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def as_dict(self) -> dict[str, Any]:
        """A deep copy of the merged config (safe to mutate by callers)."""
        return copy.deepcopy(self._data)

    # --- attribute access for sections ----------------------------------
    def __getattr__(self, name: str) -> Any:
        # only reached if normal attribute lookup fails (so self._data etc.
        # never route here, avoiding recursion)
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError:
            raise AttributeError(name)
        if name in data:
            return data[name]
        raise AttributeError(
            f"Config has no section {name!r}; "
            f"known sections: {sorted(data)}")

    def __repr__(self) -> str:
        return (f"<Config sections={sorted(self._data)} "
                f"gpu={self.profile.gpu.name!r} "
                f"driver={self.profile.gpu.driver_version!r}>")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> Config:
    """Load merged config. With no path and no file on disk, returns a fully
    usable Config from defaults alone.

    Detection always wins for live hardware state: the `hardware` section is
    fed to `hardware.detect()` purely as GAP hints, and the resulting
    HardwareProfile (authoritative) is attached as `cfg.profile`.
    """
    # 1. resolve the file path
    if path is not None:
        cfg_path = Path(path)
    else:
        cfg_path = Path(__file__).resolve().parent / _DEFAULT_CONFIG_NAME

    # 2. defaults <- file
    merged = _deep_merge(_defaults(), _load_yaml(cfg_path))

    # 3. detect hardware; config hardware block is only GAP/override hints
    profile = detect(merged.get("hardware"))

    # 4. resolve mutable paths off-OneDrive into the tensorrt/monitoring
    #    sections so downstream consumers never invent a D:\ literal.
    _resolve_paths(merged, profile)

    return Config(merged, profile, source_path=cfg_path if cfg_path.exists()
                  else None)


def _resolve_paths(merged: dict[str, Any], profile: HardwareProfile) -> None:
    """Inject off-OneDrive paths (from paths.py) and detection-derived
    runtime values into the merged config. Detection wins over any stale
    literal that slipped into the file."""
    trt = merged.setdefault("tensorrt", {})
    trt["cache_dir"] = str(cache_dir("tensorrt"))
    trt["cache_db_path"] = str(db_path("engine_cache"))

    mon = merged.setdefault("monitoring", {})
    mon["telemetry_db_path"] = str(db_path("telemetry"))
    mon["quality_db_path"] = str(db_path("quality"))

    cu = merged.setdefault("comfyui", {})
    cu["telemetry_db_path"] = str(db_path("comfyui_telemetry"))

    lg = merged.setdefault("logging", {})
    lg["file"] = str(log_path("optimizer"))

    # detection-derived: PDG cook concurrency tracks the real thread count
    # when we could probe it (overrides the conservative default).
    if profile.cpu.logical_cores:
        hou = merged.setdefault("houdini", {})
        pdg = hou.setdefault("pdg_tops", {})
        pdg["max_concurrent_cooks"] = profile.cpu.logical_cores


if __name__ == "__main__":
    c = load_config()
    print(c)
    print("data_root paths:")
    print("  tensorrt.cache_dir =", c.get("tensorrt", {}).get("cache_dir"))
    print("  monitoring.telemetry_db_path =",
          c.get("monitoring", {}).get("telemetry_db_path"))
    print("  comfyui.telemetry_db_path =",
          c.get("comfyui", {}).get("telemetry_db_path"))
    print("sections:", sorted(c.keys()))
    print("hw summary:", c.profile.summary())
