"""
Optimizer V8 - GPU Optimization Orchestrator with 9-Expert MoE Architecture

V8 = V7 baseline (5 experts: NVIDIAResearcher, GPUSpecialist, UE5Engineer,
HoudiniEngineer, SystemIntegrator) + 4 V8 specialists (ComfyUISpecialist,
HoudiniSpecialistV8, UE5SpecialistV8, CrossAppCoordinator).

V7 specialists UE5Engineer + HoudiniEngineer are disabled in V8 (the V8
specialists supersede them) but remain importable so the V7 file
core/gpu_optimizer_v6.py keeps functioning unchanged (rollback path).

V8 also wires:
  - acceleration backends (Forge-1): TensorRT, SageAttentionV2, CUTLASS FP8.
  - persistent engine cache + idle-gated background compiler (Forge-4).
  - per-app state collectors (Forge-2): ComfyUI / Houdini / UE5.

Hardware Target:
    - GPU: NVIDIA GeForce RTX 4090 (24GB, sm_89)
    - CPU: AMD Threadripper PRO 7965WX (24C/48T)
    - RAM: 128GB
    - OS: Windows 11 Pro

Applications:
    - SideFX Houdini 21 (Karma XPU, PDG/TOPs, Vellum, Pyro, ML SOPs)
    - Unreal Engine 5.6+ (Lumen, Nanite, VSM, Path Tracer, NIM, DLSS 4)
    - ComfyUI (Diffusion models, SageAttention, TensorRT engines)

Author: Generated for Joseph Ibrahim's VFX workstation
Version: 8.0 (V8 / Forge-3)
"""

import os
import sys
import time
import json
import asyncio
import logging
import sqlite3
import threading
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import re
import yaml
import psutil
import pynvml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
import winreg
import configparser
# Allow `python core/gpu_optimizer_v8.py ...` from the repo root AND
# `python gpu_optimizer_v8.py ...` from inside core/. The repo root must be
# on sys.path for `from core.gpu_optimizer_v6 import ...` to resolve.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from core.genetic_tuner import GeneticTuner
except ImportError:
    # Fallback if running from inside core/
    from genetic_tuner import GeneticTuner  # type: ignore[no-redef]

# Initialize rich console for beautiful terminal output
console = Console()

# ============================================================================
# TYPE DEFINITIONS & ENUMS
# ============================================================================
#
# V8 reuses V7's data classes and enums verbatim (V7 is FROZEN; we extend
# in place rather than redefine). This means:
#   - the same Recommendation/OptimizationResult dataclasses,
#   - the same Expert ABC,
#   - the same WorkflowMode enum,
#   - the same OptimizationAction enum (with V8 members added below),
#   - the same ExpertType enum (with V8 members added below),
#   - the same SystemContext dataclass (with V8 fields attached via setattr
#     in gather_context(); the V8 fields are documented Optional[Any] so any
#     V7-only consumer gracefully ignores them).
# Forge-2 specialists already `from core.gpu_optimizer_v6 import (...)`,
# so they share the *same enum objects* with V8 dispatch -- the V8 members
# we add here are visible to them via getattr(OptimizationAction, "X").

from core.gpu_optimizer_v6 import (  # noqa: E402  (intentional re-export)
    WorkflowMode,
    OptimizationAction,
    ExpertType,
    SystemContext,
    Recommendation,
    OptimizationResult,
    Expert,
    NVIDIAResearcher,
    GPUSpecialist,
    UE5Engineer,
    HoudiniEngineer,
    SystemIntegrator,
)


def _extend_enum(enum_cls: type, name: str, value: Any) -> None:
    """Add a new member to an existing Enum class. Idempotent."""
    if hasattr(enum_cls, name):
        return
    new_member = enum_cls._new_member_(enum_cls)  # type: ignore[attr-defined]
    new_member._name_ = name
    new_member._value_ = value
    type(enum_cls).__setattr__(enum_cls, name, new_member)
    enum_cls._member_map_[name] = new_member          # type: ignore[attr-defined]
    enum_cls._value2member_map_[value] = new_member   # type: ignore[attr-defined]
    enum_cls._member_names_.append(name)              # type: ignore[attr-defined]


# Add V8 OptimizationAction members (per SHARED_CONTEXT.md).
for _name, _value in [
    ("COMPILE_TRT_ENGINE", "compile_trt_engine"),
    ("INSTALL_CUSTOM_NODE", "install_custom_node"),
    ("INSTALL_PLUGIN", "install_plugin"),
    ("ENABLE_BACKEND", "enable_backend"),
    ("SET_LAUNCH_FLAG", "set_launch_flag"),
    ("EVICT_ENGINE_CACHE", "evict_engine_cache"),
    ("WIRE_BACKEND_PATCH", "wire_backend_patch"),
]:
    _extend_enum(OptimizationAction, _name, _value)

# Add V8 ExpertType members.
for _name, _value in [
    ("COMFYUI_SPECIALIST", "comfyui_specialist"),
    ("HOUDINI_SPECIALIST_V8", "houdini_specialist_v8"),
    ("UE5_SPECIALIST_V8", "ue5_specialist_v8"),
    ("CROSS_APP_COORDINATOR", "cross_app_coordinator"),
]:
    _extend_enum(ExpertType, _name, _value)

# Sentinel: the V7 enum members were never removed; assert post-conditions
# so a future maintainer sees this contract spelled out.
assert OptimizationAction.SET_ENV_VAR.value == "set_env_var"
assert ExpertType.HOUDINI_ENGINEER.value == "houdini_engineer"
assert hasattr(OptimizationAction, "COMPILE_TRT_ENGINE")
assert hasattr(ExpertType, "COMFYUI_SPECIALIST")



# ============================================================================
# V8 BACKENDS / CACHE / COLLECTORS / SPECIALISTS
# ============================================================================
#
# Forge-1 backends: imported lazily inside _initialize_v8_infrastructure()
# so a missing optional SDK does not break orchestrator import.
#
# Forge-4 cache + background compiler: imported eagerly (pure Python).
#
# Forge-2 specialists + state collectors: imported lazily for the same
# reason backends are -- a specialist file might in turn import an
# optional dep at module scope.

from core.engine_cache import EngineCacheManager
from core.background_compiler import BackgroundCompiler, BackendRegistry



# ============================================================================
# ORCHESTRATOR - JET-NEMOTRON MOE COORDINATOR
# ============================================================================

class OptimizerOrchestrator:
    """
    Central coordinator for 5-expert MoE system

    Uses Jet-Nemotron multi-agent pattern:
    1. Detect workload change
    2. Gather context from telemetry
    3. Dispatch to all experts in parallel
    4. Aggregate recommendations via weighted voting
    5. Apply optimizations
    6. Monitor for impact
    7. Learn from outcomes
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger("Orchestrator")

        # ----- V8 infrastructure (cache + background compiler + backends) ----
        self._initialize_v8_infrastructure()

        # ----- V8 collectors (instantiated once; reused per cycle) -----------
        self._initialize_v8_collectors()

        # Initialize experts (V7 + V8)
        self.experts: Dict[ExpertType, Expert] = {}
        self._initialize_experts()

        # Thread pool for parallel expert execution (5 V7 + up to 4 V8 = 9)
        self.executor = ThreadPoolExecutor(max_workers=9)

        # Decision history (for TensorRT-LLM training)
        self.decision_history: List[Dict] = []

        # TensorRT-LLM client (vfx-deterministic-llm API for Phase 2)
        self.llm_client = None

        # Current system context
        self.current_context: Optional[SystemContext] = None

        # Applied optimizations
        self.applied_optimizations: List[OptimizationResult] = []

        # Specialist consecutive-failure counters (for the Forge-3 "disable
        # after 3 cycles" rule documented in forge-3_orchestrator_v8.md).
        self._specialist_fail_counts: Dict[ExpertType, int] = {}

        # Initialize Genetic Tuner (Alpha-Evolve)
        self.genetic_tuner = GeneticTuner()
        self.genetic_tuner.initialize_population(self.config)

        self.logger.info(
            "OptimizerV8 orchestrator initialized: %d experts, cache=%s, bc_started=%s",
            len(self.experts),
            getattr(self.engine_cache, "cache_dir", "?"),
            self.background_compiler is not None,
        )

    # ------------------------------------------------------------------
    # V8 infrastructure
    # ------------------------------------------------------------------
    def _initialize_v8_infrastructure(self) -> None:
        """
        Wire up Forge-4 EngineCacheManager + BackgroundCompiler + Forge-1
        backends. Backends import lazily so a missing optional SDK does not
        abort orchestrator startup.
        """
        trt_cfg = (self.config.get("tensorrt") or {}) if isinstance(self.config, dict) else {}
        cache_dir = Path(trt_cfg.get("cache_dir", "D:/OptimizerV8_EngineCache"))
        cache_size_gb = float(trt_cfg.get("cache_size_gb", 200.0))

        # Single source of truth for the durable telemetry DB. Both the
        # BackgroundCompiler (compile-cost rows) and the ComfyUIStateCollector
        # (Path B persist + restore) read this attribute.
        comfy_cfg = (self.config.get("comfyui") or {}) if isinstance(self.config, dict) else {}
        self._telemetry_db_path = Path(
            comfy_cfg.get("telemetry_db_path", "data/comfyui_telemetry.db")
        )

        # EngineCacheManager creates <cache_dir>/{comfyui,houdini,ue5}/ on
        # construction; if D:\ is not writable we fall back to a local dir.
        try:
            self.engine_cache = EngineCacheManager(
                cache_dir=cache_dir,
                db_path=Path("data/engine_cache.db"),
                max_disk_bytes=int(cache_size_gb * (1024 ** 3)),
            )
        except Exception as exc:
            fallback = Path("data/engine_cache_fallback")
            self.logger.warning(
                "EngineCacheManager could not use %s (%s); falling back to %s",
                cache_dir, exc, fallback,
            )
            self.engine_cache = EngineCacheManager(
                cache_dir=fallback,
                db_path=Path("data/engine_cache.db"),
                max_disk_bytes=int(cache_size_gb * (1024 ** 3)),
            )

        # Backend registry. Lazy backend imports so missing TRT/Sage/CUTLASS
        # python pkgs don't break orchestrator import.
        self.backend_registry = BackendRegistry()
        try:
            from acceleration.tensorrt_engine import TensorRTBackend
            self.backend_registry.register(TensorRTBackend())
        except Exception as exc:
            self.logger.debug("TensorRTBackend unavailable: %s", exc)
        try:
            from acceleration.sage_attention import SageAttentionV2Backend
            self.backend_registry.register(SageAttentionV2Backend())
        except Exception as exc:
            self.logger.debug("SageAttentionV2Backend unavailable: %s", exc)
        try:
            from acceleration.cutlass_integration import CutlassFP8GemmBackend
            self.backend_registry.register(CutlassFP8GemmBackend())
        except Exception as exc:
            self.logger.debug("CutlassFP8GemmBackend unavailable: %s", exc)

        # Background compiler (idle-gated). Restored queue is automatic.
        self.background_compiler = BackgroundCompiler(
            engine_cache=self.engine_cache,
            registry=self.backend_registry,
            queue_path=Path("data/compile_queue.json"),
            telemetry_path=self._telemetry_db_path,
        )
        try:
            self.background_compiler.start()
        except Exception as exc:
            self.logger.warning("BackgroundCompiler.start() failed: %s", exc)

    # ------------------------------------------------------------------
    # V8 collectors
    # ------------------------------------------------------------------
    def _initialize_v8_collectors(self) -> None:
        """
        Instantiate per-app state collectors. Imports are lazy so a missing
        optional dep inside any one collector module does not break startup
        (each is wrapped + logged).
        """
        self._collector_comfyui = None
        self._collector_houdini = None
        self._collector_ue5 = None

        try:
            from core.experts.comfyui_state_collector import ComfyUIStateCollector
            comfy_cfg = (self.config.get("comfyui") or {})
            self._collector_comfyui = ComfyUIStateCollector(
                host=comfy_cfg.get("api_host", "127.0.0.1"),
                port=int(comfy_cfg.get("api_port", 8188)),
                telemetry_db_path=self._telemetry_db_path,
            )
        except Exception as exc:
            self.logger.debug("ComfyUIStateCollector unavailable: %s", exc)

        try:
            from core.experts.houdini_state_collector import HoudiniStateCollector
            self._collector_houdini = HoudiniStateCollector(mcp_client=None)
        except Exception as exc:
            self.logger.debug("HoudiniStateCollector unavailable: %s", exc)

        try:
            from core.experts.ue5_state_collector import UE5StateCollector
            project_roots: List[Path] = []
            ue5_cfg = (self.config.get("experts") or {}).get("ue5_engineer") or {}
            ini_path = ue5_cfg.get("engine_ini_path")
            if ini_path:
                # The collector is project-rooted; back out two levels from
                # Saved/Config/.../UserEngine.ini if applicable.
                p = Path(ini_path)
                parents = list(p.parents)
                if len(parents) >= 4:
                    project_roots.append(parents[3])
            self._collector_ue5 = UE5StateCollector(project_search_roots=project_roots)
        except Exception as exc:
            self.logger.debug("UE5StateCollector unavailable: %s", exc)

    # ------------------------------------------------------------------
    # cache / quality-ledger handles for V8 specialists
    # ------------------------------------------------------------------
    def _build_cache_handle(self) -> Any:
        """
        Adapter exposing the small surface specialists call:
            engine_cache_lookup(key) -> bool
            engine_cache_key(**kw) -> str
            engine_cache_disk_used_gb(app) -> float
        """
        cache = self.engine_cache

        class _CacheHandle:
            def engine_cache_lookup(self, key: str) -> bool:
                try:
                    return cache.lookup(key) is not None
                except Exception:
                    return False

            def engine_cache_key(self, **kw: Any) -> str:
                # Specialists pass a flexible kwargs dict (model, shape,
                # precision, backend, backend_ver, app); marshall to the
                # EngineCacheManager.compute_key signature.
                try:
                    shape = kw.get("shape") or (0, 0)
                    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                        shape_w, shape_h = int(shape[0]), int(shape[1])
                    else:
                        shape_w = shape_h = 0
                    return cache.compute_key(
                        app=str(kw.get("app", "comfyui")),
                        model_sha256=str(kw.get("model", kw.get("model_sha256", ""))),
                        shape_w=shape_w,
                        shape_h=shape_h,
                        batch=int(kw.get("batch", 1)),
                        precision=str(kw.get("precision", "fp16")),
                        backend_name=str(kw.get("backend", kw.get("backend_name", "tensorrt"))),
                        backend_ver=str(kw.get("backend_ver", "11.x")),
                    )
                except Exception:
                    return "uncached:error"

            def engine_cache_disk_used_gb(self, app: str) -> float:
                try:
                    return float(cache.disk_used_gb(app=app))
                except Exception:
                    return 0.0

        return _CacheHandle()

    def _load_config(self) -> Dict:
        """Load YAML configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure rich logging"""
        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )

    def _initialize_experts(self):
        """
        Initialize V7 + V8 experts.

        V7 experts (5):
          - NVIDIAResearcher, GPUSpecialist, SystemIntegrator     -> stay enabled in V8
          - HoudiniEngineer, UE5Engineer                          -> superseded by
            V8 specialists; enabled defaults to FALSE per Forge-3 brief D2.
        V8 specialists (4): ComfyUI, HoudiniV8, UE5V8, CrossApp.

        Each specialist gets cache_handle + (optional) quality_ledger_handle.
        Specialist construction failures are caught + logged; the
        orchestrator still starts (degrades to remaining experts).
        """
        expert_configs = self.config.get("experts", {})
        # V8-only override map (so V7's same `experts:` block keeps its keys
        # intact when V7 is launched directly per the rollback contract).
        v8_overrides = self.config.get("experts_v8_overrides", {}) or {}

        def _enabled_v8(key: str, default: bool) -> bool:
            ov = v8_overrides.get(key)
            if isinstance(ov, dict) and "enabled" in ov:
                return bool(ov["enabled"])
            return bool(expert_configs.get(key, {}).get("enabled", default))

        # ---- V7 experts -----------------------------------------------------

        if _enabled_v8("nvidia_researcher", True):
            self.experts[ExpertType.NVIDIA_RESEARCHER] = NVIDIAResearcher(
                expert_configs.get("nvidia_researcher", {})
            )

        if _enabled_v8("gpu_specialist", True):
            self.experts[ExpertType.GPU_SPECIALIST] = GPUSpecialist(
                expert_configs.get("gpu_specialist", {})
            )

        # V8 disables ue5_engineer + houdini_engineer by default (the V8
        # specialists supersede them); the override block in config.yaml
        # makes this explicit. V7 launched directly STILL enables them
        # because V7 doesn't read `experts_v8_overrides`.
        if _enabled_v8("ue5_engineer", False):
            self.experts[ExpertType.UE5_ENGINEER] = UE5Engineer(
                expert_configs.get("ue5_engineer", {})
            )

        if _enabled_v8("houdini_engineer", False):
            self.experts[ExpertType.HOUDINI_ENGINEER] = HoudiniEngineer(
                expert_configs.get("houdini_engineer", {}),
                mcp_client=None,
            )

        if _enabled_v8("system_integrator", True):
            self.experts[ExpertType.SYSTEM_INTEGRATOR] = SystemIntegrator(
                expert_configs.get("system_integrator", {})
            )

        # ---- V8 specialists -------------------------------------------------

        cache_handle = self._build_cache_handle()
        quality_ledger_handle = None  # Crucible-2 lands this; permissive None.

        # ComfyUI specialist (config from top-level `comfyui:` block + tensorrt)
        comfy_cfg = self.config.get("comfyui") or {}
        if (comfy_cfg.get("expert") or {}).get("enabled", True) and comfy_cfg.get("enabled", True):
            try:
                from core.experts.comfyui_specialist import ComfyUISpecialist
                cfg = dict(comfy_cfg.get("expert") or {})
                cfg.setdefault("min_confidence", 0.80)
                cfg["tensorrt"] = self.config.get("tensorrt") or {}
                cfg["phase_b_enabled"] = bool(self.config.get("phase_b_enabled", False))
                self.experts[ExpertType.COMFYUI_SPECIALIST] = ComfyUISpecialist(
                    cfg, cache_handle=cache_handle,
                    quality_ledger_handle=quality_ledger_handle,
                )
            except Exception as exc:
                self.logger.warning("ComfyUISpecialist init failed: %s", exc)

        # Houdini V8 specialist
        houd_cfg = self.config.get("houdini_v8") or {}
        if (houd_cfg.get("expert") or {}).get("enabled", True) and houd_cfg.get("enabled", True):
            try:
                from core.experts.houdini_specialist import HoudiniSpecialist
                cfg = dict(houd_cfg.get("expert") or {})
                cfg.setdefault("min_confidence", 0.80)
                cfg["tensorrt"] = self.config.get("tensorrt") or {}
                cfg["phase_b_enabled"] = bool(self.config.get("phase_b_enabled", False))
                self.experts[ExpertType.HOUDINI_SPECIALIST_V8] = HoudiniSpecialist(
                    cfg, cache_handle=cache_handle,
                    quality_ledger_handle=quality_ledger_handle,
                )
            except Exception as exc:
                self.logger.warning("HoudiniSpecialist init failed: %s", exc)

        # UE5 V8 specialist
        ue5_cfg = self.config.get("ue5_v8") or {}
        if (ue5_cfg.get("expert") or {}).get("enabled", True) and ue5_cfg.get("enabled", True):
            try:
                from core.experts.ue5_specialist import UE5Specialist
                cfg = dict(ue5_cfg.get("expert") or {})
                cfg.setdefault("min_confidence", 0.80)
                cfg["ue5"] = {
                    "nim_marketplace_url": ue5_cfg.get("nim_marketplace_url"),
                }
                self.experts[ExpertType.UE5_SPECIALIST_V8] = UE5Specialist(
                    cfg, cache_handle=cache_handle,
                    quality_ledger_handle=quality_ledger_handle,
                )
            except Exception as exc:
                self.logger.warning("UE5Specialist init failed: %s", exc)

        # Cross-app coordinator
        cross_cfg = self.config.get("cross_app_coordinator") or {}
        if cross_cfg.get("enabled", True):
            try:
                from core.experts.cross_app_coordinator import CrossAppCoordinator
                cfg = dict(cross_cfg)
                cfg.setdefault("min_confidence", 0.85)
                cfg["cross_app"] = {
                    "idle_window_seconds": cfg.get("idle_window_seconds", 60.0),
                }
                self.experts[ExpertType.CROSS_APP_COORDINATOR] = CrossAppCoordinator(
                    cfg, cache_handle=cache_handle,
                    quality_ledger_handle=quality_ledger_handle,
                )
            except Exception as exc:
                self.logger.warning("CrossAppCoordinator init failed: %s", exc)

        self.logger.info(
            "Initialized %d experts: %s",
            len(self.experts),
            ", ".join(e.value for e in self.experts.keys()),
        )

    def gather_context(self) -> SystemContext:
        """
        Gather current system state for expert analysis.

        V7 fields collected via pynvml + psutil. V8 fields populated by
        per-app collectors (ComfyUI / Houdini / UE5) plus a cross-app state
        derived from the per-app outputs. Each collector wrapped in
        try/except per Forge-3 brief D1 §6.

        Graceful-degrade contract (SHARED_CONTEXT.md): on ANY failure return
        a safe SystemContext rather than raising.
        """
        ctx = self._gather_v7_context()

        # ---- V8 ComfyUI state ----------------------------------------------
        comfyui_state = None
        if self._collector_comfyui is not None:
            try:
                comfyui_state = self._collector_comfyui.collect()
                # If ComfyUI process detected, ensure it's reflected in
                # active_processes so legacy V7 experts still see it.
                if getattr(comfyui_state, "process_running", False) \
                        and "python.exe" not in ctx.active_processes:
                    ctx.active_processes.append("python.exe")
            except Exception as exc:
                self.logger.warning("ComfyUI collector failed: %s", exc)
        ctx.comfyui = comfyui_state

        # ---- V8 Houdini state ----------------------------------------------
        houdini_state = None
        if self._collector_houdini is not None:
            try:
                houdini_state = self._collector_houdini.collect()
            except Exception as exc:
                self.logger.warning("Houdini collector failed: %s", exc)
        ctx.houdini_v8 = houdini_state

        # ---- V8 UE5 state --------------------------------------------------
        ue5_state = None
        if self._collector_ue5 is not None:
            try:
                ue5_state = self._collector_ue5.collect()
            except Exception as exc:
                self.logger.warning("UE5 collector failed: %s", exc)
        ctx.ue5_v8 = ue5_state

        # ---- V8 cross-app state (derived) ----------------------------------
        try:
            ctx.cross_app = self._build_cross_app_state(
                ctx, comfyui_state, houdini_state, ue5_state
            )
        except Exception as exc:
            self.logger.warning("CrossApp state assembly failed: %s", exc)
            ctx.cross_app = None

        return ctx

    def _gather_v7_context(self) -> SystemContext:
        """V7-equivalent context (kept identical so existing tests pass)."""
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)

            active_processes: List[str] = []
            process_vram: Dict[str, float] = {}

            for proc in psutil.process_iter(['name', 'pid']):
                try:
                    name = proc.info['name']
                    if name in ['houdini.exe', 'UnrealEditor.exe', 'python.exe']:
                        active_processes.append(name)
                        process_vram[name] = 0.0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            num_vfx_apps = len(active_processes)
            if num_vfx_apps == 0:
                mode = WorkflowMode.IDLE
            elif num_vfx_apps == 1:
                mode = WorkflowMode.SEQUENTIAL
            else:
                mode = WorkflowMode.PARALLEL

            ctx = SystemContext(
                timestamp=datetime.now(),
                workflow_mode=mode,
                gpu_utilization_percent=utilization.gpu,
                vram_used_gb=memory.used / (1024 ** 3),
                vram_free_gb=memory.free / (1024 ** 3),
                gpu_temp_c=temp,
                gpu_power_w=power,
                gpu_clock_mhz=clock,
                active_processes=active_processes,
                process_vram=process_vram,
                cpu_utilization_percent=psutil.cpu_percent(),
                ram_used_gb=psutil.virtual_memory().used / (1024 ** 3),
            )

            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return ctx

        except Exception as exc:
            self.logger.error(f"Failed to gather V7 context: {exc}")
            return SystemContext(
                timestamp=datetime.now(),
                workflow_mode=WorkflowMode.IDLE,
                gpu_utilization_percent=0.0,
                vram_used_gb=0.0,
                vram_free_gb=24.0,
                gpu_temp_c=0.0,
                gpu_power_w=0.0,
                gpu_clock_mhz=0,
                active_processes=[],
                process_vram={},
            )

    def _build_cross_app_state(
        self,
        ctx: SystemContext,
        comfyui_state: Any,
        houdini_state: Any,
        ue5_state: Any,
    ) -> Any:
        """
        Build CrossAppState from per-app collector outputs + a coarse
        foreground-window probe. Returns the dataclass instance Forge-2's
        coordinator expects, or None if the import fails.
        """
        try:
            from core.experts.cross_app_coordinator import CrossAppState
        except Exception as exc:
            self.logger.debug("CrossAppState import failed: %s", exc)
            return None

        app_running = {
            "comfyui": bool(getattr(comfyui_state, "process_running", False)),
            "houdini": bool(getattr(houdini_state, "process_running", False)),
            "ue5": bool(getattr(ue5_state, "process_running", False)),
        }

        # Foreground app via Win32 GetForegroundWindow + process name match.
        # Wrapped: any failure -> "none".
        foreground_app = "none"
        try:
            foreground_app = self._detect_foreground_app(app_running)
        except Exception as exc:
            self.logger.debug("Foreground detection failed: %s", exc)

        # Compile-jobs-active per app from BackgroundCompiler status.
        compile_jobs_active = {"comfyui": 0, "houdini": 0, "ue5": 0}
        try:
            status = self.background_compiler.status() if self.background_compiler else {}
            depths = status.get("queue_depth_per_app") or {}
            for app in compile_jobs_active:
                compile_jobs_active[app] = int(depths.get(app, 0))
        except Exception as exc:
            self.logger.debug("BC status query failed: %s", exc)

        # Idle dwell (seconds the GPU has been < idle_threshold).
        idle_dwell = 0.0
        try:
            status = self.background_compiler.status() if self.background_compiler else {}
            idle = status.get("idle_state") or {}
            idle_dwell = float(idle.get("dwell_seconds_observed", 0.0) or 0.0)
        except Exception:
            pass

        return CrossAppState(
            foreground_pid_app=foreground_app,
            app_running=app_running,
            compile_jobs_active=compile_jobs_active,
            idle_dwell_seconds=idle_dwell,
        )

    @staticmethod
    def _detect_foreground_app(app_running: Dict[str, bool]) -> str:
        """
        Best-effort: returns one of {comfyui, houdini, ue5, none}. Uses
        Win32 GetForegroundWindow + process-name heuristic; falls back to
        "none" if Win32 isn't available (CI / headless).
        """
        try:
            import ctypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return "none"
            pid = ctypes.c_uint32(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if not pid.value:
                return "none"
            try:
                proc = psutil.Process(pid.value)
                name = (proc.name() or "").lower()
            except Exception:
                return "none"
            if "houdini" in name:
                return "houdini" if app_running.get("houdini") else "none"
            if "unrealeditor" in name or "ue5editor" in name:
                return "ue5" if app_running.get("ue5") else "none"
            if "python" in name:
                # Best-effort: assume the python foreground is the running ComfyUI.
                return "comfyui" if app_running.get("comfyui") else "none"
            return "none"
        except Exception:
            return "none"

    def analyze_workload(self) -> List[Recommendation]:
        """
        Dispatch analysis to all experts in parallel

        Returns aggregated recommendations from all experts
        """
        context = self.gather_context()
        self.current_context = context

        self.logger.info(
            f"Analyzing workload: {context.workflow_mode.value}, "
            f"GPU: {context.gpu_utilization_percent:.0f}%, "
            f"VRAM: {context.vram_used_gb:.1f}GB/{context.vram_used_gb + context.vram_free_gb:.1f}GB"
        )

        # Parallel expert execution
        futures = {
            self.executor.submit(expert.analyze, context): expert_type
            for expert_type, expert in self.experts.items()
        }

        all_recommendations = []

        for future in as_completed(futures):
            expert_type = futures[future]
            try:
                recommendations = future.result(timeout=10.0)
                all_recommendations.extend(recommendations)
                self.logger.debug(
                    f"{expert_type.value}: {len(recommendations)} recommendations"
                )
            except Exception as e:
                self.logger.error(f"{expert_type.value} failed: {e}")

        return all_recommendations

    def aggregate_recommendations(
        self,
        recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """
        Aggregate recommendations using weighted voting

        Resolves conflicts, filters duplicates, ranks by impact
        """
        # Remove conflicts
        filtered = self._resolve_conflicts(recommendations)

        # Sort by expected impact * confidence
        ranked = sorted(
            filtered,
            key=lambda r: r.expected_impact * r.confidence,
            reverse=True
        )

        return ranked

    def _resolve_conflicts(
        self,
        recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """
        Remove conflicting recommendations

        If multiple recommendations target the same setting,
        keep the one with highest confidence
        """
        target_map: Dict[str, Recommendation] = {}

        for rec in recommendations:
            target = rec.target

            if target not in target_map:
                target_map[target] = rec
            else:
                # Keep recommendation with higher confidence
                if rec.confidence > target_map[target].confidence:
                    target_map[target] = rec

        return list(target_map.values())

    def apply_optimization(self, recommendation: Recommendation) -> OptimizationResult:
        """
        Apply a single optimization recommendation

        Returns result with success/failure status
        """
        self.logger.info(
            f"Applying: {recommendation.action.value} on {recommendation.target} "
            f"(confidence: {recommendation.confidence:.2f})"
        )

        try:
            action = recommendation.action
            # V7 actions
            if action == OptimizationAction.SET_ENV_VAR:
                return self._apply_env_var(recommendation)
            elif action == OptimizationAction.REGISTRY_TWEAK:
                return self._apply_registry_tweak(recommendation)
            elif action == OptimizationAction.PROCESS_PRIORITY:
                return self._apply_process_priority(recommendation)
            elif action == OptimizationAction.VRAM_ALLOCATION:
                return self._apply_vram_allocation(recommendation)
            elif action == OptimizationAction.ENABLE_ACCELERATION:
                return self._apply_acceleration(recommendation)
            elif action == OptimizationAction.ADJUST_QUALITY:
                return self._apply_quality_adjustment(recommendation)
            elif action == OptimizationAction.CPU_AFFINITY:
                return self._apply_cpu_affinity(recommendation)
            # V8 actions
            elif action == OptimizationAction.COMPILE_TRT_ENGINE:
                return self._apply_compile_trt_engine(recommendation)
            elif action == OptimizationAction.INSTALL_CUSTOM_NODE:
                return self._apply_install_custom_node(recommendation)
            elif action == OptimizationAction.INSTALL_PLUGIN:
                return self._apply_install_plugin(recommendation)
            elif action == OptimizationAction.ENABLE_BACKEND:
                return self._apply_enable_backend(recommendation)
            elif action == OptimizationAction.SET_LAUNCH_FLAG:
                return self._apply_set_launch_flag(recommendation)
            elif action == OptimizationAction.EVICT_ENGINE_CACHE:
                return self._apply_evict_engine_cache(recommendation)
            elif action == OptimizationAction.WIRE_BACKEND_PATCH:
                return self._apply_wire_backend_patch(recommendation)
            else:
                raise ValueError(f"Unknown action: {recommendation.action}")

        except Exception as e:
            self.logger.error(f"Failed to apply optimization: {e}")
            return OptimizationResult(
                recommendation=recommendation,
                applied=False,
                timestamp=datetime.now(),
                error=str(e),
                can_rollback=False,
            )

    def _apply_env_var(self, rec: Recommendation) -> OptimizationResult:
        """Set environment variable"""
        var = rec.parameters.get("variable")
        value = rec.parameters.get("value")

        previous = os.environ.get(var)
        os.environ[var] = str(value)

        # TODO: Persist to system environment if scope == "system"

        return OptimizationResult(
            recommendation=rec,
            applied=True,
            timestamp=datetime.now(),
            previous_value=previous,
        )

    def _apply_registry_tweak(self, rec: Recommendation) -> OptimizationResult:
        """Apply Windows registry modification"""
        key_path = rec.parameters.get("key_path", r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers")
        value_name = rec.parameters.get("value_name")
        value_data = rec.parameters.get("value_data")
        value_type = rec.parameters.get("value_type", winreg.REG_DWORD)

        if not value_name or value_data is None:
             return OptimizationResult(
                recommendation=rec,
                applied=False,
                timestamp=datetime.now(),
                error="Missing registry parameters",
            )

        try:
            # Open key with write permissions
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_WRITE) as key:
                winreg.SetValueEx(key, value_name, 0, value_type, value_data)
            
            return OptimizationResult(
                recommendation=rec,
                applied=True,
                timestamp=datetime.now(),
            )
        except Exception as e:
            self.logger.error(f"Registry tweak failed: {e}")
            return OptimizationResult(
                recommendation=rec,
                applied=False,
                timestamp=datetime.now(),
                error=str(e),
            )

    def _apply_process_priority(self, rec: Recommendation) -> OptimizationResult:
        """Adjust process priority"""
        target_process = rec.target
        priority_str = rec.parameters.get("priority", "NORMAL")
        
        # Map string priority to psutil constants
        priority_map = {
            "IDLE": psutil.IDLE_PRIORITY_CLASS,
            "BELOW_NORMAL": psutil.BELOW_NORMAL_PRIORITY_CLASS,
            "NORMAL": psutil.NORMAL_PRIORITY_CLASS,
            "ABOVE_NORMAL": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
            "HIGH": psutil.HIGH_PRIORITY_CLASS,
            "REALTIME": psutil.REALTIME_PRIORITY_CLASS
        }
        priority_val = priority_map.get(priority_str.upper(), psutil.NORMAL_PRIORITY_CLASS)
        
        applied_count = 0
        errors = []

        for proc in psutil.process_iter(['name', 'pid']):
            if proc.info['name'] == target_process:
                try:
                    p = psutil.Process(proc.info['pid'])
                    p.nice(priority_val)
                    applied_count += 1
                except Exception as e:
                    errors.append(str(e))

        if applied_count > 0:
            return OptimizationResult(
                recommendation=rec,
                applied=True,
                timestamp=datetime.now(),
                actual_impact=None 
            )
        else:
             return OptimizationResult(
                recommendation=rec,
                applied=False,
                timestamp=datetime.now(),
                error=f"Process not found or access denied. Errors: {errors}" if errors else "Process not found"
            )



    def _apply_cpu_affinity(self, rec: Recommendation) -> OptimizationResult:
        """Apply CPU core affinity (pinning)"""
        target_process = rec.target
        cores = rec.parameters.get("cores", [])
        
        if not cores:
            return OptimizationResult(rec, False, datetime.now(), error="No cores specified")
            
        applied_count = 0
        errors = []
        
        for proc in psutil.process_iter(['name', 'pid']):
            if proc.info['name'] == target_process:
                try:
                    p = psutil.Process(proc.info['pid'])
                    p.cpu_affinity(cores)
                    applied_count += 1
                except Exception as e:
                    errors.append(str(e))
                    
        if applied_count > 0:
            return OptimizationResult(rec, True, datetime.now())
        else:
            return OptimizationResult(
                rec, 
                False, 
                datetime.now(), 
                error=f"Process not found or access denied. Errors: {errors}" if errors else "Process not found"
            )

    def _apply_vram_allocation(self, rec: Recommendation) -> OptimizationResult:
        """Adjust VRAM allocation limits"""
        # TODO: Implement VRAM fencing
        self.logger.warning("VRAM allocation not yet implemented")
        return OptimizationResult(
            recommendation=rec,
            applied=False,
            timestamp=datetime.now(),
            error="Not implemented",
        )

    def _apply_acceleration(self, rec: Recommendation) -> OptimizationResult:
        """
        Legacy V7 ENABLE_ACCELERATION dispatch. V8 specialists prefer the
        explicit new actions; if a recommendation arrives here with a
        `sub_action` parameter, route to the matching V8 handler. Otherwise
        log the legacy fallback and return applied=False (parity with V7).
        """
        sub = (rec.parameters or {}).get("sub_action")
        if sub == "compile_trt_engine":
            return self._apply_compile_trt_engine(rec)
        if sub == "install_custom_node":
            return self._apply_install_custom_node(rec)
        if sub == "install_plugin":
            return self._apply_install_plugin(rec)
        if sub == "enable_backend":
            return self._apply_enable_backend(rec)
        if sub == "evict_engine_cache":
            return self._apply_evict_engine_cache(rec)
        # Otherwise: V7 stub behaviour preserved.
        self.logger.warning(
            "Legacy ENABLE_ACCELERATION on %s: V8 prefers explicit actions; "
            "no sub_action provided.", rec.target,
        )
        return OptimizationResult(
            recommendation=rec,
            applied=False,
            timestamp=datetime.now(),
            error="ENABLE_ACCELERATION deprecated in V8; use COMPILE_TRT_ENGINE / "
                  "INSTALL_CUSTOM_NODE / INSTALL_PLUGIN / ENABLE_BACKEND.",
        )

    # ------------------------------------------------------------------
    # V8 _apply_* handlers
    # ------------------------------------------------------------------
    def _apply_compile_trt_engine(self, rec: Recommendation) -> OptimizationResult:
        """
        Enqueue a TRT compile job on the BackgroundCompiler (idle-gated).
        Returns applied=True immediately because the compile is async.
        """
        params = dict(rec.parameters or {})
        # Marshal to BackgroundCompiler.enqueue(compile_params) shape.
        compile_params: Dict[str, Any] = {
            "app": params.get("app", "comfyui"),
            "backend_name": params.get("backend_name", "tensorrt"),
            "backend_ver": params.get("backend_ver", "11.x"),
            "model_sha256": params.get("model_sha", params.get("model_sha256", "")),
            "shape_w": 0,
            "shape_h": 0,
            "batch": int(params.get("batch", 1)),
            "precision": params.get("precision", "fp16"),
            "model_kind": params.get("model_kind", "unet"),
            "priority": int(params.get("priority", 0)),
        }
        # Lift opt-shape into the flat fields if the recommendation supplies
        # the {min,opt,max} dict.
        sr = params.get("shape_range") or {}
        opt = sr.get("opt") if isinstance(sr, dict) else None
        if isinstance(opt, (list, tuple)) and len(opt) >= 4:
            compile_params["shape_h"] = int(opt[2])
            compile_params["shape_w"] = int(opt[3])
            compile_params["batch"] = int(opt[0])
        if isinstance(sr, dict) and {"min", "opt", "max"} <= set(sr.keys()):
            compile_params["shape_range"] = {
                "min": list(sr["min"]),
                "opt": list(sr["opt"]),
                "max": list(sr["max"]),
            }

        # Source: specialists may pass `source_onnx` (Houdini) or leave it
        # absent (ComfyUI runs the export at compile time). If source is
        # missing, fall back to a synthesized placeholder so the BG worker
        # records the intent without crashing -- the actual model export
        # is owned by the per-app extension layer in Phase B.
        src_path = params.get("source_onnx")
        if src_path:
            compile_params["source"] = {"kind": "onnx", "path": str(src_path)}
        else:
            compile_params["source"] = {
                "kind": "onnx",
                "path": str(params.get("path", "")),
            }

        try:
            job_id = self.background_compiler.enqueue(compile_params)
            status = self.background_compiler.status() or {}
            depth = status.get("pending_total", "?")
            self.logger.info(
                "Enqueued TRT compile job %s (target=%s, queue depth=%s)",
                job_id, rec.target, depth,
            )
            return OptimizationResult(
                recommendation=rec,
                applied=True,
                timestamp=datetime.now(),
                previous_value=None,
            )
        except Exception as exc:
            self.logger.error("BackgroundCompiler.enqueue failed: %s", exc)
            return OptimizationResult(
                recommendation=rec,
                applied=False,
                timestamp=datetime.now(),
                error=f"enqueue failed: {exc}",
            )

    def _apply_install_custom_node(self, rec: Recommendation) -> OptimizationResult:
        """
        ComfyUI custom_nodes/ git clone + pip install. Auto-install per
        Forge-3 brief D1 (decision #4). Idempotent: skip if already
        installed (matched by repo basename).
        """
        params = rec.parameters or {}
        repo = params.get("node_repo") or params.get("repo")
        ref = params.get("ref", "main")
        if not repo:
            return OptimizationResult(
                recommendation=rec, applied=False, timestamp=datetime.now(),
                error="node_repo missing",
            )

        comfy_cfg = self.config.get("comfyui") or {}
        install_dir = Path(comfy_cfg.get("install_dir", "C:/Users/User/ComfyUI"))
        custom_nodes_dir = install_dir / "custom_nodes"
        try:
            custom_nodes_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return OptimizationResult(
                recommendation=rec, applied=False, timestamp=datetime.now(),
                error=f"cannot create custom_nodes/: {exc}",
            )

        # repo may be "owner/name" form (per Forge-2 specialist) or full URL.
        if "://" in repo:
            url = repo
            name = repo.rstrip("/").split("/")[-1].replace(".git", "")
        else:
            url = f"https://github.com/{repo}.git"
            name = repo.split("/")[-1]
        target = custom_nodes_dir / name

        if target.exists():
            self.logger.info("Custom node %s already installed; skipping.", name)
            console.print(f"[INFO] Custom node already installed: {name}")
            return OptimizationResult(
                recommendation=rec, applied=True, timestamp=datetime.now(),
                previous_value="already_installed",
            )

        try:
            cmd = ["git", "clone", "--depth", "1", "--branch", ref, url, str(target)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if res.returncode != 0:
                # retry without --branch for repos where ref is a SHA / default branch
                res = subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(target)],
                    capture_output=True, text=True, timeout=180,
                )
            if res.returncode != 0:
                return OptimizationResult(
                    recommendation=rec, applied=False, timestamp=datetime.now(),
                    error=f"git clone failed: {res.stderr.strip()[:300]}",
                )
        except Exception as exc:
            return OptimizationResult(
                recommendation=rec, applied=False, timestamp=datetime.now(),
                error=f"git clone exception: {exc}",
            )

        # Optional: pip install -r requirements.txt inside the new node dir.
        req = target / "requirements.txt"
        if req.exists():
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req)],
                    capture_output=True, text=True, timeout=300,
                )
            except Exception as exc:
                self.logger.warning("pip install for %s failed: %s", name, exc)

        console.print(f"[+] Installed ComfyUI custom node: {name}; restart ComfyUI to load.")
        return OptimizationResult(
            recommendation=rec, applied=True, timestamp=datetime.now(),
        )

    def _apply_install_plugin(self, rec: Recommendation) -> OptimizationResult:
        """
        UE5 plugin install advisory. Per Forge-2 brief R1 the install is
        notification-only (auto_install=False). Surfaces marketplace URL +
        minimum version to the user; does NOT modify the UE5 install.
        """
        params = rec.parameters or {}
        plugin = params.get("plugin", "<unknown>")
        url = params.get("marketplace_url") or params.get("url") or ""
        minimum = params.get("minimum") or params.get("min_version") or ""
        msg = (
            f"[UE5 Plugin Advisory] {plugin}"
            + (f" (>= {minimum})" if minimum else "")
            + (f"  Marketplace: {url}" if url else "")
        )
        console.print(f"[!] {msg}")
        self.logger.info(msg)

        # Persist alongside launch advice for transparency.
        try:
            advice_path = Path("data/launch_advice.json")
            advice_path.parent.mkdir(parents=True, exist_ok=True)
            existing: Dict[str, Any] = {}
            if advice_path.exists():
                try:
                    existing = json.loads(advice_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
            ue5 = existing.setdefault("ue5", {})
            plugins = ue5.setdefault("plugin_advisories", [])
            plugins.append({
                "plugin": plugin, "minimum": minimum, "url": url,
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
            advice_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger.debug("launch_advice.json write failed: %s", exc)

        return OptimizationResult(
            recommendation=rec, applied=True, timestamp=datetime.now(),
        )

    def _apply_enable_backend(self, rec: Recommendation) -> OptimizationResult:
        """
        Toggle a backend at runtime. CrossAppCoordinator R1 emits this
        with `parameters.action == "pause_background_compiles"` to pause a
        background app's compiles while another app is foreground; that's
        wired directly into BackgroundCompiler.pause_app(...).
        """
        params = rec.parameters or {}
        action = params.get("action")
        app = params.get("app")
        if action == "pause_background_compiles" and app:
            try:
                affected = self.background_compiler.pause_app(
                    app, reason=f"foreground_active:{params.get('foreground_app','?')}",
                )
                self.logger.info("Paused %d compile job(s) for app=%s", affected, app)
                return OptimizationResult(
                    recommendation=rec, applied=True, timestamp=datetime.now(),
                    previous_value=affected,
                )
            except Exception as exc:
                return OptimizationResult(
                    recommendation=rec, applied=False, timestamp=datetime.now(),
                    error=str(exc),
                )

        # Generic backend-flag toggle (writes to runtime-only config map; the
        # YAML is not modified to avoid cross-process race with the user's
        # editor). The runtime flag is consulted by experts via cache_handle.
        backend = params.get("backend") or params.get("name")
        if backend:
            self.config.setdefault("acceleration_runtime", {})[backend] = bool(
                params.get("enable", True)
            )
            return OptimizationResult(
                recommendation=rec, applied=True, timestamp=datetime.now(),
            )
        return OptimizationResult(
            recommendation=rec, applied=False, timestamp=datetime.now(),
            error="ENABLE_BACKEND requires either action=pause_background_compiles+app or backend=<name>",
        )

    def _apply_set_launch_flag(self, rec: Recommendation) -> OptimizationResult:
        """
        Persist a desired launch-flag annotation to data/launch_advice.json
        and emit a Rich notification. Does NOT kill the running process
        (per Forge-3 brief D1).
        """
        params = rec.parameters or {}
        app = params.get("app")
        if not app:
            # Infer from target string e.g. "comfyui:launch".
            tgt = (rec.target or "").lower()
            if tgt.startswith("comfyui"):
                app = "comfyui"
            elif tgt.startswith("houdini"):
                app = "houdini"
            elif tgt.startswith("ue5"):
                app = "ue5"
            else:
                app = "comfyui"

        flag = params.get("add_flag") or params.get("flag")
        if not flag:
            return OptimizationResult(
                recommendation=rec, applied=False, timestamp=datetime.now(),
                error="add_flag missing",
            )

        try:
            advice_path = Path("data/launch_advice.json")
            advice_path.parent.mkdir(parents=True, exist_ok=True)
            existing: Dict[str, Any] = {}
            if advice_path.exists():
                try:
                    existing = json.loads(advice_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
            app_advice = existing.setdefault(app, {})
            flags = app_advice.setdefault("desired_flags", [])
            if flag not in flags:
                flags.append(flag)
            app_advice["last_updated"] = datetime.now().isoformat(timespec="seconds")
            advice_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception as exc:
            return OptimizationResult(
                recommendation=rec, applied=False, timestamp=datetime.now(),
                error=f"launch_advice write failed: {exc}",
            )

        console.print(f"[!] Restart {app} with: {flag}")
        return OptimizationResult(
            recommendation=rec, applied=True, timestamp=datetime.now(),
        )

    def _apply_evict_engine_cache(self, rec: Recommendation) -> OptimizationResult:
        """Reclaim disk by LRU-evicting cached engine artifacts."""
        params = rec.parameters or {}
        app = params.get("app")
        try:
            freed = self.engine_cache.evict_lru(target_app=app)
            self.logger.info("Engine cache eviction freed %.2f MB (app=%s)",
                             freed / (1024 ** 2), app)
            return OptimizationResult(
                recommendation=rec, applied=True, timestamp=datetime.now(),
                previous_value=freed,
            )
        except Exception as exc:
            return OptimizationResult(
                recommendation=rec, applied=False, timestamp=datetime.now(),
                error=str(exc),
            )

    def _apply_wire_backend_patch(self, rec: Recommendation) -> OptimizationResult:
        """
        Write a small monkeypatch .py file into the target app's
        auto-import directory (e.g. ComfyUI custom_nodes/optimizer_patches/).
        The patch contents are taken verbatim from rec.parameters['code']
        if provided; otherwise a noop comment is written.
        """
        params = rec.parameters or {}
        app = params.get("app", "comfyui")
        patch_name = params.get("name") or f"optimizer_patch_{rec.target.replace(':','_')}"
        code = params.get("code", f"# {patch_name} -- noop placeholder\n")

        if app == "comfyui":
            comfy_cfg = self.config.get("comfyui") or {}
            install_dir = Path(comfy_cfg.get("install_dir", "C:/Users/User/ComfyUI"))
            target_dir = install_dir / "custom_nodes" / "optimizer_patches"
        else:
            target_dir = Path("data") / "optimizer_patches" / app

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "__init__.py").touch(exist_ok=True)
            (target_dir / f"{patch_name}.py").write_text(code, encoding="utf-8")
        except Exception as exc:
            return OptimizationResult(
                recommendation=rec, applied=False, timestamp=datetime.now(),
                error=f"patch write failed: {exc}",
            )

        return OptimizationResult(
            recommendation=rec, applied=True, timestamp=datetime.now(),
        )

    def _apply_quality_adjustment(self, rec: Recommendation) -> OptimizationResult:
        """Adjust rendering/simulation quality settings"""
        # Handle UE5 INI modification (including evolved settings)
        if "ue5" in rec.target or "nanite" in rec.target or "lumen" in rec.target or "evolved" in rec.target:
             engine_ini = rec.parameters.get("engine_ini_path")
             setting = rec.parameters.get("engine_ini_setting") # e.g. "r.Streaming.PoolSize=12288"
             
             if engine_ini and setting:
                 # Ensure directory exists
                 os.makedirs(os.path.dirname(engine_ini), exist_ok=True)
                 
                 # Create file if it doesn't exist
                 if not os.path.exists(engine_ini):
                     with open(engine_ini, 'w') as f:
                         f.write("[ConsoleVariables]\n")
                 
                 try:
                     with open(engine_ini, 'r') as f:
                         content = f.read()
                     
                     if "[ConsoleVariables]" not in content:
                         content += "\n[ConsoleVariables]\n"
                     
                     key, val = setting.split('=')
                     
                     # Robust Regex Replacement
                     # Looks for "key=..." or "key = ..." inside the file
                     pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
                     
                     if pattern.search(content):
                         # Replace existing value
                         new_content = pattern.sub(f"{key}={val}", content)
                         action_type = "Updated"
                     else:
                         # Append to [ConsoleVariables] section
                         # Find the section and add the line after it
                         new_content = content.replace("[ConsoleVariables]", f"[ConsoleVariables]\n{key}={val}")
                         action_type = "Added"
                         
                     with open(engine_ini, 'w') as f:
                         f.write(new_content)
                         
                     self.logger.info(f"{action_type} {key} in {engine_ini}")
                     return OptimizationResult(rec, True, datetime.now())
                     
                 except Exception as e:
                     self.logger.error(f"Failed to write INI: {e}")
                     return OptimizationResult(rec, False, datetime.now(), error=str(e))
        
        self.logger.warning(f"Quality adjustment for {rec.target} not fully implemented")
        return OptimizationResult(
            recommendation=rec,
            applied=False,
            timestamp=datetime.now(),
            error="Not fully implemented",
        )

    def run_optimization_cycle(self):
        """
        Single optimization cycle:
        1. Analyze workload
        2. Aggregate recommendations
        3. Apply optimizations
        4. Log results
        """
        console.print("\n[bold cyan]=== OptimizerV8 Cycle ===[/bold cyan]\n")

        # Phase 1: Analysis
        with console.status("[bold green]Analyzing workload...") as status:
            recommendations = self.analyze_workload()
            console.print(f"[OK] Collected {len(recommendations)} recommendations")

        if not recommendations:
            console.print("[yellow]No optimizations needed[/yellow]")
            return

        # Phase 2: Aggregation
        ranked_recommendations = self.aggregate_recommendations(recommendations)

        # Display recommendations
        table = Table(title="Top Optimization Recommendations")
        table.add_column("Expert", style="cyan")
        table.add_column("Action", style="green")
        table.add_column("Target", style="yellow")
        table.add_column("Impact", justify="right")
        table.add_column("Confidence", justify="right")

        for rec in ranked_recommendations[:10]:  # Top 10
            table.add_row(
                rec.expert.value,
                rec.action.value,
                rec.target,
                f"+{rec.expected_impact:.1f}%",
                f"{rec.confidence:.0%}",
            )

        console.print(table)

        # Phase 3: Application (top 5 optimizations)
        console.print("\n[bold]Applying top optimizations...[/bold]")

        for rec in ranked_recommendations[:5]:
            result = self.apply_optimization(rec)
            self.applied_optimizations.append(result)

            if result.applied:
                console.print(f"[+] Applied: {rec.target}")
            else:
                console.print(f"[X] Failed: {rec.target} ({result.error})")

        console.print(f"\n[bold green]Cycle complete[/bold green]\n")

    def start_monitoring_loop(self, interval_seconds: int = 60):
        """
        Start continuous monitoring and optimization loop

        Args:
            interval_seconds: Time between optimization cycles
        """
        console.print("[bold cyan]Starting OptimizerV8 monitoring loop...[/bold cyan]")

        try:
            while True:
                self.run_optimization_cycle()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped by user[/yellow]")
            self.shutdown()

    def run_evolution_cycle(self):
        """Run a single genetic evolution cycle (Nightly Optimization)"""
        console.print("\n[bold magenta]=== Starting Genetic Evolution Cycle ===[/bold magenta]")
        console.print("Optimizing configuration parameters based on fitness...")
        
        best_genome = self.genetic_tuner.run_evolution_cycle()
        best_config = self.genetic_tuner.get_best_config()
        
        console.print(f"\n[green]Evolution Complete![/green]")
        console.print(f"Best Fitness Score: {best_genome.fitness_score:.2f}")
        console.print("Top Configuration Genes:")
        for key, val in best_config.items():
            console.print(f"  - {key}: {val}")
            
        # Apply the best configuration to the live system
        # (In a real scenario, we might want user approval first)
        self._apply_evolved_config(best_config)

    def _apply_evolved_config(self, config: Dict[str, Any]):
        """Apply the evolved configuration to the running system"""
        # Map genes back to expert recommendations
        # This is a simplified application logic
        
        # UE5 Genes
        if "r.Streaming.PoolSize" in config:
            self.apply_optimization(Recommendation(
                expert=ExpertType.UE5_ENGINEER,
                action=OptimizationAction.ADJUST_QUALITY,
                target="evolved_pool_size",
                parameters={
                    "engine_ini_setting": f"r.Streaming.PoolSize={config['r.Streaming.PoolSize']}",
                    "engine_ini_path": self.experts[ExpertType.UE5_ENGINEER].engine_ini_path
                },
                confidence=1.0,
                expected_impact=5.0,
                reasoning="Genetic algorithm determined optimal pool size"
            ))

    def shutdown(self):
        """Cleanup and shutdown orchestrator"""
        self.logger.info("Shutting down OptimizerV8...")
        try:
            if getattr(self, "background_compiler", None) is not None:
                self.background_compiler.stop(timeout=30.0)
        except Exception as exc:
            self.logger.warning("BackgroundCompiler.stop failed: %s", exc)
        try:
            self.executor.shutdown(wait=True)
        except Exception as exc:
            self.logger.warning("Executor shutdown failed: %s", exc)
        console.print("[bold green]Shutdown complete[/bold green]")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for OptimizerV8"""
    import argparse

    parser = argparse.ArgumentParser(
        description="OptimizerV8 - RTX 4090 GPU Optimization Orchestrator (V8)"
    )
    parser.add_argument(
        "--config",
        default="../config.yaml",
        help="Path to configuration file (relative to this file's parent dir, "
             "or an absolute path).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds (continuous mode only).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single optimization cycle and exit (smoke / CI mode).",
    )
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Run genetic evolution cycle (Nightly Mode).",
    )

    args = parser.parse_args()

    # Allow absolute or relative config paths.
    cfg_arg = Path(args.config)
    if cfg_arg.is_absolute():
        config_path = cfg_arg
    else:
        config_path = Path(__file__).parent.parent / cfg_arg

    orchestrator = OptimizerOrchestrator(str(config_path))

    try:
        if args.evolve:
            orchestrator.run_evolution_cycle()
        elif args.once:
            orchestrator.run_optimization_cycle()
        else:
            orchestrator.start_monitoring_loop(args.interval)
    finally:
        # Always stop the background compiler thread; otherwise the daemon
        # will hold the process open in --once smoke mode.
        try:
            orchestrator.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
