"""
CUTLASS 3.x Integration Module for Optimizer_V7_G3
Target: RTX 4090 (SM89 - Ada Lovelace)
CUDA: 13.0 | 24GB VRAM | 4th Gen Tensor Cores

This module provides CUTLASS configuration, validation, and acceleration
utilities that integrate with the existing cuTensor backend.

CUTLASS 3.x Features for SM89:
- Thread Block Clusters (Ada Lovelace)
- Async Copy (cp.async)
- FP8/BF16/TF32 Tensor Core operations
- Optimized GEMM kernels

Correction (2026-09-16): thread-block clusters are NOT available on SM89. They
require compute capability 9.0 (Hopper). Ada is sm_89 and does not support them,
so SM89Config.features['thread_block_clusters'] is False and any caller that
branches on it will take the non-cluster path.
"""

import os
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Supported precision modes for CUTLASS operations"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    TF32 = "tf32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"


@dataclass
class SM89Config:
    """
    RTX 4090 (SM89 Ada Lovelace) Hardware Configuration

    Based on official NVIDIA specs for GeForce RTX 4090:
    - 16384 CUDA cores (128 SMs * 128 cores/SM)
    - 512 4th-gen Tensor Cores
    - 72MB L2 Cache
    - 100KB Shared Memory per SM
    - 24GB GDDR6X @ 1008 GB/s
    """
    compute_capability: int = 89
    sm_count: int = 128
    cuda_cores_per_sm: int = 128
    tensor_cores_per_sm: int = 4
    shared_memory_kb: int = 100
    l2_cache_mb: int = 72
    register_file_kb: int = 256
    max_threads_per_sm: int = 1536
    max_warps_per_sm: int = 48
    warp_size: int = 32

    # Memory specs
    vram_gb: int = 24
    memory_bandwidth_gbps: int = 1008

    # Tensor Core supported precisions (4th gen)
    supported_precisions: List[PrecisionMode] = field(default_factory=lambda: [
        PrecisionMode.FP32,
        PrecisionMode.FP16,
        PrecisionMode.BF16,
        PrecisionMode.TF32,
        PrecisionMode.FP8_E4M3,
        PrecisionMode.FP8_E5M2,
        PrecisionMode.INT8,
    ])

    # Ada Lovelace specific features
    features: Dict[str, bool] = field(default_factory=lambda: {
        # NOT Ada. Thread-block clusters arrive with compute capability 9.0
        # (Hopper, sm_90); Ada is sm_89 and has no cluster support at all. This
        # read True for as long as the file has existed, which is worse than a
        # missing entry: a caller branching on it would take the cluster path and
        # either fail to compile or silently land on a non-cluster fallback while
        # believing otherwise.
        'thread_block_clusters': False,
        'async_copy': True,
        'tensor_memory_accelerator': False,  # Hopper only (SM90)
        'wgmma': False,  # Hopper only
        'distributed_shared_memory': False,  # Hopper only
        'fp8_tensor_cores': True,
        'shader_execution_reordering': True,  # Ada feature
    })


@dataclass
class CUTLASSInstallation:
    """CUTLASS installation status and paths"""
    installed: bool = False
    version: str = "unknown"
    path: Optional[Path] = None
    include_path: Optional[Path] = None
    build_path: Optional[Path] = None
    profiler_path: Optional[Path] = None


class CUTLASSManager:
    """
    Manages CUTLASS installation, configuration, and integration
    with Optimizer_V7_G3 acceleration pipeline.
    """

    # Default installation path (set by install_cutlass.bat)
    DEFAULT_PATH = r"C:\Users\User\OneDrive\Desktop\Dads_Toolbox_For_Repairs\CUTLASS"

    def __init__(self, config_path: Optional[Path] = None):
        self.sm89 = SM89Config()
        self.installation = self._detect_installation()
        self.config_path = config_path

    def _detect_installation(self) -> CUTLASSInstallation:
        """Detect CUTLASS installation from environment or default path"""
        install = CUTLASSInstallation()

        # Check environment variables (set by install_cutlass.bat)
        cutlass_path = os.environ.get('CUTLASS_PATH')
        cutlass_include = os.environ.get('CUTLASS_INCLUDE_DIR')

        # Fallback to default path if env vars not set
        if not cutlass_path and Path(self.DEFAULT_PATH).exists():
            cutlass_path = self.DEFAULT_PATH
            cutlass_include = os.path.join(self.DEFAULT_PATH, 'include')

        if cutlass_path:
            path = Path(cutlass_path)
            if path.exists():
                install.path = path
                install.include_path = Path(cutlass_include) if cutlass_include else path / 'include'
                install.build_path = path / 'build'

                # Check for cutlass.h header
                cutlass_header = install.include_path / 'cutlass' / 'cutlass.h'
                if cutlass_header.exists():
                    install.installed = True
                    install.version = self._detect_version(path)

                # Check for profiler
                profiler = install.build_path / 'tools' / 'profiler' / 'cutlass_profiler.exe'
                if profiler.exists():
                    install.profiler_path = profiler

        return install

    def _detect_version(self, path: Path) -> str:
        """Detect CUTLASS version from source"""
        version_file = path / 'VERSION'
        if version_file.exists():
            return version_file.read_text().strip()

        # Try CMakeLists.txt
        cmake_file = path / 'CMakeLists.txt'
        if cmake_file.exists():
            content = cmake_file.read_text()
            for line in content.split('\n'):
                if 'CUTLASS_VERSION' in line and 'set' in line.lower():
                    # Extract version from set(CUTLASS_VERSION "x.y.z")
                    parts = line.split('"')
                    if len(parts) >= 2:
                        return parts[1]
        return "3.5.1"  # Default version from installer

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive CUTLASS status"""
        return {
            'installed': self.installation.installed,
            'version': self.installation.version,
            'path': str(self.installation.path) if self.installation.path else None,
            'include_path': str(self.installation.include_path) if self.installation.include_path else None,
            'has_profiler': self.installation.profiler_path is not None,
            'target_arch': f'sm_{self.sm89.compute_capability}',
            'features': self.sm89.features,
            'supported_precisions': [p.value for p in self.sm89.supported_precisions],
        }

    def get_nvcc_flags(self, precision: PrecisionMode = PrecisionMode.FP16) -> List[str]:
        """
        Get optimized NVCC compilation flags for SM89

        Args:
            precision: Target precision mode

        Returns:
            List of NVCC flags optimized for RTX 4090
        """
        flags = [
            f'-arch=sm_{self.sm89.compute_capability}',
            '-O3',
            '-DNDEBUG',
            '--expt-relaxed-constexpr',
            '-Xcompiler', '/MP24',  # Threadripper 24 cores
        ]

        # Precision-specific flags
        if precision in [PrecisionMode.FP16, PrecisionMode.BF16, PrecisionMode.TF32]:
            flags.append('--use_fast_math')

        # Add include path if available
        if self.installation.include_path:
            flags.extend(['-I', str(self.installation.include_path)])

        return flags

    def get_optimal_tile_config(self, m: int, n: int, k: int) -> Dict[str, int]:
        """
        Calculate optimal tile configuration for GEMM operation

        Based on SM89 register file and shared memory constraints.

        Args:
            m, n, k: GEMM dimensions (C = A @ B, A is MxK, B is KxN)

        Returns:
            Optimal tile sizes for the given dimensions
        """
        # Default optimal tiles for SM89 (empirically derived)
        default_tiles = {
            'tile_m': 128,
            'tile_n': 128,
            'tile_k': 32,
            'warp_m': 64,
            'warp_n': 64,
            'warp_k': 32,
            'stages': 4,  # Pipeline stages
        }

        # Adjust for small matrices
        if m < 256 or n < 256:
            default_tiles['tile_m'] = 64
            default_tiles['tile_n'] = 64
            default_tiles['warp_m'] = 32
            default_tiles['warp_n'] = 32
            default_tiles['stages'] = 2

        # Adjust for very large K dimension (memory bound)
        if k > 4096:
            default_tiles['tile_k'] = 64
            default_tiles['warp_k'] = 64

        return default_tiles

    def estimate_performance(
        self,
        m: int,
        n: int,
        k: int,
        precision: PrecisionMode = PrecisionMode.FP16
    ) -> Dict[str, float]:
        """
        Estimate GEMM performance for given dimensions and precision

        Based on theoretical peak performance of RTX 4090:
        - FP16 Tensor: 660 TFLOPS
        - FP32: 83 TFLOPS
        - TF32: 165 TFLOPS
        - FP8: 1320 TFLOPS

        Args:
            m, n, k: GEMM dimensions
            precision: Computation precision

        Returns:
            Performance estimates including TFLOPS and time
        """
        # Theoretical peak TFLOPS for RTX 4090
        peak_tflops = {
            PrecisionMode.FP32: 83.0,
            PrecisionMode.FP16: 660.0,
            PrecisionMode.BF16: 660.0,
            PrecisionMode.TF32: 165.0,
            PrecisionMode.FP8_E4M3: 1320.0,
            PrecisionMode.FP8_E5M2: 1320.0,
            PrecisionMode.INT8: 1320.0,
        }

        # FLOPS for GEMM: 2 * M * N * K
        total_flops = 2.0 * m * n * k

        # Achievable efficiency (typically 60-80% for well-optimized kernels)
        efficiency = 0.70

        peak = peak_tflops.get(precision, 660.0)
        achievable_tflops = peak * efficiency
        estimated_time_ms = (total_flops / (achievable_tflops * 1e12)) * 1000

        return {
            'total_flops': total_flops,
            'peak_tflops': peak,
            'achievable_tflops': achievable_tflops,
            'efficiency': efficiency,
            'estimated_time_ms': estimated_time_ms,
            'precision': precision.value,
        }

    def run_profiler(
        self,
        operation: str = "gemm",
        m: int = 1024,
        n: int = 1024,
        k: int = 1024,
        precision: str = "f16",
    ) -> Optional[Dict[str, Any]]:
        """
        Run CUTLASS profiler for performance benchmarking

        Args:
            operation: Operation type (gemm, conv2d, etc.)
            m, n, k: Dimensions
            precision: Precision (f16, f32, tf32, bf16)

        Returns:
            Profiler results or None if profiler unavailable
        """
        if not self.installation.profiler_path:
            logger.warning("CUTLASS profiler not available")
            return None

        cmd = [
            str(self.installation.profiler_path),
            f'--operation={operation}',
            f'--m={m}',
            f'--n={n}',
            f'--k={k}',
            f'--A={precision}',
            f'--B={precision}',
            f'--C={precision}',
            f'--accum=f32',
            '--warmup_iterations=10',
            '--profiling_iterations=100',
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return self._parse_profiler_output(result.stdout)
            else:
                logger.error(f"Profiler error: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.error("Profiler timed out")
            return None
        except Exception as e:
            logger.error(f"Profiler failed: {e}")
            return None

    def _parse_profiler_output(self, output: str) -> Dict[str, Any]:
        """Parse CUTLASS profiler output"""
        results = {
            'raw_output': output,
            'best_kernel': None,
            'gflops': 0.0,
            'runtime_ms': 0.0,
        }

        # Parse CSV-style output from profiler
        lines = output.strip().split('\n')
        for line in lines:
            if 'GFLOPs' in line and ',' in line:
                parts = line.split(',')
                try:
                    results['gflops'] = float(parts[-2])
                    results['runtime_ms'] = float(parts[-1])
                except (ValueError, IndexError):
                    pass

        return results


def create_cutlass_manager(config_path: Optional[str] = None) -> CUTLASSManager:
    """Factory function to create CUTLASSManager instance"""
    return CUTLASSManager(Path(config_path) if config_path else None)


def get_cutlass_status() -> Dict[str, Any]:
    """Quick status check for CUTLASS installation"""
    manager = CUTLASSManager()
    return manager.get_status()


def validate_installation() -> bool:
    """Validate CUTLASS is properly installed and configured"""
    manager = CUTLASSManager()

    if not manager.installation.installed:
        logger.error("CUTLASS not installed. Run install_cutlass.bat first.")
        return False

    # Check include path has required headers
    required_headers = [
        'cutlass/cutlass.h',
        'cutlass/gemm/gemm.h',
        'cutlass/arch/arch.h',
    ]

    if manager.installation.include_path:
        for header in required_headers:
            header_path = manager.installation.include_path / header
            if not header_path.exists():
                logger.error(f"Missing header: {header}")
                return False

    logger.info(f"CUTLASS {manager.installation.version} validated successfully")
    return True


# CLI interface
if __name__ == '__main__':
    import json

    print("=" * 60)
    print("CUTLASS 3.x Integration - RTX 4090 (SM89)")
    print("=" * 60)
    print()

    manager = CUTLASSManager()
    status = manager.get_status()

    print(f"Installed: {status['installed']}")
    print(f"Version: {status['version']}")
    print(f"Path: {status['path']}")
    print(f"Target: {status['target_arch']}")
    print()

    if status['installed']:
        print("Features:")
        for feat, enabled in status['features'].items():
            mark = "[X]" if enabled else "[ ]"
            print(f"  {mark} {feat}")
        print()

        print("Supported Precisions:")
        for prec in status['supported_precisions']:
            print(f"  - {prec}")
        print()

        # Performance estimate example
        print("Performance Estimate (1024x1024x1024 GEMM, FP16):")
        perf = manager.estimate_performance(1024, 1024, 1024, PrecisionMode.FP16)
        print(f"  Peak: {perf['peak_tflops']:.1f} TFLOPS")
        print(f"  Achievable: {perf['achievable_tflops']:.1f} TFLOPS")
        print(f"  Est. Time: {perf['estimated_time_ms']:.3f} ms")
    else:
        print("CUTLASS not installed. Run install_cutlass.bat")

    print()
    print("=" * 60)


# ============================================================================
# Forge-1 extension: CutlassFP8GemmBackend (AccelerationBackend)
# ============================================================================
#
# This class is appended by Forge-1 (briefs/forge-1_backends.md, D4). It does
# NOT modify any of the CUTLASSManager API above; it consumes that manager
# read-only for installation discovery and NVCC flag generation.
#
# Phase A status: registered with `enabled=False`. The brief explicitly asks
# us NOT to gate Phase A on CUTLASS 4.x being installed. The existing
# install_and_run.bat clones CUTLASS v3.5.1, which is fine for the NVCC-based
# .cu shim path used here (CUTLASS 3.x kernels work on sm_89). Phase C will
# either flip `enabled=True` once shim compile is wired into a packaged build
# environment, or migrate to the CUTLASS 4.x Python DSL when available.
#
# Workload neutrality: this backend imports nothing from `core/experts/` and
# speaks only in (M, N, K, precision) terms.

import hashlib
import shutil
import tempfile
import time
from typing import Callable, Tuple

# Local-relative imports of the ABC and dataclasses defined by Forge-1.
from .backend import (
    AccelerationBackend,
    BackendSource,
    BuildArtifactDescriptor,
    CancelToken,
    LoadedEngine,
    OnnxPath,
    ShapeRange,
    TorchModulePickle,
    TrtPlanBytes,
)


class CutlassFP8GemmBackend(AccelerationBackend):
    """
    CUTLASS FP8 GEMM acceleration backend (Phase A: registered, disabled).

    Generates a tiny CUDA shim per (M, N, K, precision) that instantiates a
    CUTLASS GEMM template, compiles it via NVCC, and returns the resulting
    DLL/.so bytes through BuildArtifactDescriptor. Forge-4's cache layer
    persists those bytes; load() re-attaches via ctypes.

    Phase status:
        Phase A : `enabled = False` -- class is shipped so Forge-2 can
                  reference the symbol but the orchestrator must NOT
                  dispatch compile() yet.
        Phase B : enabled where calibration data exists.
        Phase C : flipped on by Forge-3 via config.yaml (Forge-3 owns the
                  toggle).

    Why disabled-by-default in Phase A:
        - The existing install_and_run.bat clones CUTLASS 3.5.1, which has
          enough header coverage for sm_89 FP8 GEMM, but the Phase-A
          smoke test runs on hosts without NVCC on PATH. Failing to flip
          on is safer than failing a build mid-orchestration.
        - CUTLASS 4.x's Python DSL is the long-term path; we explicitly
          decided not to block this brief on it.
    """

    name: str = "cutlass_fp8_gemm"
    version: str = "0.1.0-phaseA"

    # FP8 GEMM ranges this backend will accept once enabled.
    _MIN_DIM = 16
    _MAX_DIM = 16384
    _DIM_ALIGNMENT = 8  # FP8 tensor cores want multiples of 8.

    def __init__(
        self,
        manager: Optional[CUTLASSManager] = None,
        force_enable: bool = False,
    ) -> None:
        self._manager = manager or CUTLASSManager()
        self._installed = self._manager.installation.installed
        self._cutlass_version = self._manager.installation.version

        # Phase A: stay disabled regardless of NVCC discovery, per brief.
        # `force_enable` is for Crucible-2 / Phase C lab use only.
        if force_enable and self._installed and self._nvcc_on_path():
            self.enabled = True
            logger.info(
                "CutlassFP8GemmBackend force-enabled (CUTLASS=%s, NVCC=found).",
                self._cutlass_version,
            )
        else:
            self.enabled = False
            if not self._installed:
                logger.info(
                    "CutlassFP8GemmBackend disabled: CUTLASS not installed "
                    "(run install_cutlass.bat). Class is still importable.",
                )
            elif not self._nvcc_on_path():
                logger.info(
                    "CutlassFP8GemmBackend disabled: nvcc not on PATH. "
                    "Class is still importable.",
                )
            else:
                logger.info(
                    "CutlassFP8GemmBackend disabled by Phase-A policy. "
                    "Set force_enable=True to opt in for lab use.",
                )

        self.version = f"0.1.0-phaseA+cutlass{self._cutlass_version}"

    # ----------------------------------------------------------- ABC impl

    def applies_to(
        self,
        model_kind: str,
        shape: Tuple[int, ...],
        precision: str,
    ) -> bool:
        kind = (model_kind or "").lower()
        prec = (precision or "").lower()
        if kind != "gemm":
            return False
        if prec not in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            return False
        if len(shape) != 3:
            # Expect (M, N, K) for GEMM.
            return False
        m, n, k = shape
        for dim in (m, n, k):
            if dim < self._MIN_DIM or dim > self._MAX_DIM:
                return False
            if dim % self._DIM_ALIGNMENT != 0:
                return False
        # Note: applies_to may return True even when self.enabled is False,
        # so the orchestrator can preview reachability under Phase A. The
        # orchestrator MUST also gate on `self.enabled` before calling
        # compile(). compile() itself will raise if disabled.
        return True

    def estimated_compile_seconds(
        self,
        model_kind: str,
        shape: Tuple[int, ...],
    ) -> float:
        # NVCC compile of a single GEMM shim. Empirically ~30-60s cold,
        # ~10-15s with ccache. Use 45 as a conservative prior.
        return 45.0

    def estimated_speedup(
        self,
        model_kind: str,
        shape: Tuple[int, ...],
    ) -> float:
        # FP8 vs FP16 baseline on sm_89 -- theoretical 2x; practical ~1.7x
        # accounting for memory-bound regimes.
        return 1.7

    def estimated_quality_delta(self) -> float:
        # FP8 introduces visible noise on undertuned calibration. Crucible-2
        # owns the per-(model, shape) measurement; this prior is conservative.
        return 0.18

    def compile(
        self,
        source: BackendSource,
        shape_range: ShapeRange,
        precision: str,
        cancel_token: CancelToken,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> BuildArtifactDescriptor:
        if not self.enabled:
            raise RuntimeError(
                "CutlassFP8GemmBackend is disabled in Phase A. "
                "Re-instantiate with force_enable=True for lab use, "
                "or wait for Phase C config.yaml flip.",
            )
        if not self._installed:
            raise RuntimeError(
                "CUTLASS not installed. Run install_cutlass.bat or set "
                "CUTLASS_PATH env var."
            )
        if not self._nvcc_on_path():
            raise RuntimeError("nvcc not on PATH; cannot compile CUTLASS shim.")

        # We expect the source to encode (M, N, K) via shape_range.opt_shape.
        if not isinstance(source, (OnnxPath, TorchModulePickle, TrtPlanBytes)):
            raise TypeError(f"Unsupported BackendSource: {type(source)!r}")
        if len(shape_range.opt_shape) != 3:
            raise ValueError(
                "CutlassFP8GemmBackend expects shape_range.opt_shape == (M, N, K). "
                f"Got {shape_range.opt_shape}."
            )

        m, n, k = shape_range.opt_shape
        prec_mode = self._precision_to_mode(precision)
        t0 = time.perf_counter()

        # ---- 1. Generate shim source ----------------------------------
        shim_src = self._render_shim(m, n, k, prec_mode)

        # ---- 2. NVCC compile to .so / .dll ---------------------------
        cancel_token.raise_if_cancelled()
        if progress_cb:
            progress_cb(0.10)

        with tempfile.TemporaryDirectory(prefix="cutlass_fp8_") as td:
            td_path = Path(td)
            cu_file = td_path / "shim.cu"
            so_file = td_path / ("shim.dll" if os.name == "nt" else "shim.so")
            cu_file.write_text(shim_src, encoding="utf-8")

            nvcc_cmd: List[str] = ["nvcc"]
            nvcc_cmd.extend(self._manager.get_nvcc_flags(prec_mode))
            nvcc_cmd.extend(["-shared", "-Xcompiler", "/LD" if os.name == "nt" else "-fPIC"])
            nvcc_cmd.extend(["-o", str(so_file), str(cu_file)])

            cancel_token.raise_if_cancelled()
            if progress_cb:
                progress_cb(0.25)

            try:
                proc = subprocess.run(
                    nvcc_cmd,
                    capture_output=True, text=True, timeout=600,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"NVCC compile timed out after 600s for shim "
                    f"(M={m},N={n},K={k}): {exc}"
                ) from exc

            cancel_token.raise_if_cancelled()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"NVCC failed (code {proc.returncode}):\n"
                    f"--- stdout ---\n{proc.stdout}\n"
                    f"--- stderr ---\n{proc.stderr}"
                )

            so_bytes = so_file.read_bytes()

        if progress_cb:
            progress_cb(1.0)

        elapsed = time.perf_counter() - t0
        digest = hashlib.blake2b(so_bytes, digest_size=8).hexdigest()
        logger.info(
            "CutlassFP8GemmBackend.compile OK: M=%d N=%d K=%d %s "
            "%d bytes in %.2fs (digest %s)",
            m, n, k, prec_mode.value, len(so_bytes), elapsed, digest,
        )

        return BuildArtifactDescriptor(
            backend_name=self.name,
            backend_version=self.version,
            artifact_bytes=so_bytes,
            artifact_format="cutlass_so",
            precision=prec_mode.value,
            shape_range=shape_range,
            compile_seconds=elapsed,
            notes={
                "M": int(m), "N": int(n), "K": int(k),
                "cutlass_version": self._cutlass_version,
                "shim_digest": digest,
                "platform": "windows" if os.name == "nt" else "posix",
            },
        )

    def load(self, artifact_path: Path) -> LoadedEngine:
        if not self.enabled:
            raise RuntimeError(
                "CutlassFP8GemmBackend is disabled in Phase A; "
                "load() requires force_enable=True.",
            )
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"CUTLASS shim not found: {artifact_path}")

        import ctypes
        try:
            dll = ctypes.CDLL(str(artifact_path))
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load CUTLASS shim {artifact_path}: {exc}"
            ) from exc

        runner = _CutlassFP8GemmRunner(dll=dll, path=artifact_path)
        return LoadedEngine(
            backend_name=self.name,
            backend_version=self.version,
            runner=runner,
            metadata={
                "shim_path": str(artifact_path),
                "shim_size_bytes": artifact_path.stat().st_size,
                "cutlass_version": self._cutlass_version,
            },
        )

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _nvcc_on_path() -> bool:
        return shutil.which("nvcc") is not None

    @staticmethod
    def _precision_to_mode(precision: str) -> PrecisionMode:
        prec = (precision or "").lower()
        if prec in {"fp8", "fp8_e4m3"}:
            return PrecisionMode.FP8_E4M3
        if prec == "fp8_e5m2":
            return PrecisionMode.FP8_E5M2
        raise ValueError(
            f"CutlassFP8GemmBackend only accepts fp8 precision; got {precision!r}"
        )

    @staticmethod
    def _render_shim(m: int, n: int, k: int, prec: PrecisionMode) -> str:
        """
        Render a minimal CUTLASS FP8 GEMM shim source.

        This shim exposes a C ABI:
            extern "C" int cutlass_fp8_gemm(
                const void* A, const void* B, void* C,
                int M, int N, int K, float alpha, float beta);

        Phase-A note: this is a buildable scaffold that the Phase-C
        kernel-tuning pass will replace with measured-best tile configs.
        We compile-time bake (M, N, K) into the kernel for now (no
        dynamic shapes within a single shim); the cache key already
        keys on (M, N, K), so a cache miss simply rebuilds.
        """
        e4m3 = prec is PrecisionMode.FP8_E4M3
        elem_a = "cutlass::float_e4m3_t" if e4m3 else "cutlass::float_e5m2_t"
        elem_b = elem_a
        return f"""\
// Auto-generated by acceleration/cutlass_integration.py::CutlassFP8GemmBackend
// shape: M={m} N={n} K={k}  precision: {prec.value}
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

using ElementA = {elem_a};
using ElementB = {elem_b};
using ElementC = float;
using ElementAccumulator = float;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89
>;

extern "C" int cutlass_fp8_gemm(
    const void* A, const void* B, void* C,
    int M, int N, int K, float alpha, float beta)
{{
    if (M != {m} || N != {n} || K != {k}) return -2;
    Gemm gemm_op;
    typename Gemm::Arguments args(
        {{M, N, K}},
        {{reinterpret_cast<ElementA const*>(A), K}},
        {{reinterpret_cast<ElementB const*>(B), K}},
        {{reinterpret_cast<ElementC*>(C), N}},
        {{reinterpret_cast<ElementC*>(C), N}},
        {{alpha, beta}}
    );
    cutlass::Status st = gemm_op(args);
    return st == cutlass::Status::kSuccess ? 0 : -1;
}}
"""


class _CutlassFP8GemmRunner:
    """Thin ctypes wrapper around the compiled CUTLASS shim DLL/.so."""

    def __init__(self, dll: Any, path: Path) -> None:
        self.dll = dll
        self.path = path
        # The shim's C entrypoint signature.
        try:
            import ctypes
            fn = self.dll.cutlass_fp8_gemm
            fn.restype = ctypes.c_int
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float,
            ]
            self._fn = fn
        except AttributeError as exc:
            raise RuntimeError(
                f"Loaded shim missing 'cutlass_fp8_gemm' symbol: {exc}"
            ) from exc

    def gemm(
        self,
        a_ptr: int, b_ptr: int, c_ptr: int,
        m: int, n: int, k: int,
        alpha: float = 1.0, beta: float = 0.0,
    ) -> int:
        """Invoke the FP8 GEMM. Returns 0 on success, negative on error."""
        return int(self._fn(a_ptr, b_ptr, c_ptr, m, n, k, alpha, beta))

