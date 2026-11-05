"""V9 — CUTLASS FP8 GEMM acceleration backend.

Generates a tiny CUDA shim per (M, N, K, precision) that instantiates a
CUTLASS GEMM template, compiles it via NVCC, and returns the resulting
DLL/.so bytes through BuildArtifactDescriptor. The build/cache layer persists
those bytes; load() re-attaches via ctypes.

Workload-agnostic: imports nothing from v9.advisors / v9.build / v9.context.
It speaks only in (M, N, K, precision) terms.

POLICY: this backend is **disabled by default** (Phase C). The class ships so
the registry can report it honestly, but the orchestrator must NOT dispatch
compile() until Phase C activation. `applies_to()` may return True (so the
registry can show reachability), but compile()/load() are gated on `enabled`
and raise a clear message if called while disabled.

No hardware literals: the target SM arch comes from a HardwareProfile.gpu.
sm_arch passed in at construction (never hardcoded). CUTLASS install location
comes from CUTLASS_PATH / CUTLASS_INCLUDE_DIR env vars (set by the installer),
never a baked path.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from v9.accel.backend import (
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

logger = logging.getLogger("v9.accel.cutlass")

_INSTALL_HINT = (
    "Install CUTLASS and set $CUTLASS_PATH (and ensure nvcc is on PATH)."
)


class PrecisionMode(Enum):
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


@dataclass
class CutlassInstallation:
    """Discovered CUTLASS install. All paths come from the environment, never
    a hardcoded literal."""
    installed: bool = False
    version: str = "unknown"
    path: Optional[Path] = None
    include_path: Optional[Path] = None


def _detect_cutlass() -> CutlassInstallation:
    """Discover CUTLASS from CUTLASS_PATH / CUTLASS_INCLUDE_DIR. Degrade-safe:
    any probe failure yields an `installed=False` record, never an exception."""
    install = CutlassInstallation()
    try:
        cutlass_path = os.environ.get("CUTLASS_PATH")
        cutlass_include = os.environ.get("CUTLASS_INCLUDE_DIR")
        if not cutlass_path:
            return install
        path = Path(cutlass_path)
        if not path.exists():
            return install
        install.path = path
        install.include_path = (
            Path(cutlass_include) if cutlass_include else path / "include"
        )
        header = install.include_path / "cutlass" / "cutlass.h"
        if header.exists():
            install.installed = True
            install.version = _detect_version(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("CUTLASS detection failed: %s", exc)
    return install


def _detect_version(path: Path) -> str:
    """Best-effort CUTLASS version. Prefer the version.h macros (stable across
    4.x), then a VERSION file, then a sanitized CMakeLists scrape. Reject any
    value containing unexpanded CMake vars / whitespace ($ { } space)."""
    import re as _re

    def _clean(v: str) -> Optional[str]:
        v = v.strip()
        if v and not _re.search(r"[${}\s]", v):
            return v
        return None

    try:
        vh = path / "include" / "cutlass" / "version.h"
        if vh.exists():
            text = vh.read_text(encoding="utf-8", errors="ignore")
            parts = {}
            for comp in ("MAJOR", "MINOR", "PATCH"):
                m = _re.search(rf"CUTLASS_{comp}\s+(\d+)", text)
                if m:
                    parts[comp] = m.group(1)
            if {"MAJOR", "MINOR", "PATCH"} <= parts.keys():
                return f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"
        version_file = path / "VERSION"
        if version_file.exists():
            cleaned = _clean(version_file.read_text(encoding="utf-8"))
            if cleaned:
                return cleaned
        cmake = path / "CMakeLists.txt"
        if cmake.exists():
            for line in cmake.read_text(encoding="utf-8").splitlines():
                if "CUTLASS_VERSION" in line and "set" in line.lower():
                    bits = line.split('"')
                    if len(bits) >= 2:
                        cleaned = _clean(bits[1])
                        if cleaned:
                            return cleaned
    except Exception:  # pragma: no cover - defensive
        pass
    return "unknown"


class CutlassFP8GemmBackend(AccelerationBackend):
    """CUTLASS FP8 GEMM backend (Phase-C-disabled by policy).

    Disabled by default even when CUTLASS + nvcc are present. `force_enable`
    opts in for lab / benchmark use only.
    """

    name: str = "cutlass_fp8_gemm"
    version: str = "0.2.0"          # 0.2.0 = tuned Ada FP8 config

    # FP8 GEMM ranges this backend accepts once enabled.
    _MIN_DIM = 16
    _MAX_DIM = 16384
    _DIM_ALIGNMENT = 16  # tuned kernel uses 16-elem vectorized FP8 loads.

    def __init__(self, sm_arch: Optional[int] = None,
                 enabled: bool = False,
                 force_enable: bool = False) -> None:
        """sm_arch comes from HardwareProfile.gpu.sm_arch (e.g. 89). It is
        only used to target nvcc; never hardcoded. If unknown, compile() will
        raise rather than guess an arch.

        `enabled` is the config switch (acceleration.cutlass.enabled). When
        true AND the toolchain (CUTLASS headers + nvcc + a known sm_arch) is
        actually present, the backend goes live and participates in the recipe.
        `force_enable` is the legacy lab override and is treated identically.
        The backend can NEVER go live when the toolchain is missing — honesty
        wins over the config flag (DESIGN_V9 §4.4)."""
        super().__init__()
        self._sm_arch = sm_arch
        self._install = _detect_cutlass()
        self._installed = self._install.installed
        self._cutlass_version = self._install.version

        nvcc = self._nvcc_on_path()
        want_on = bool(enabled or force_enable)
        # Config opts the backend in; toolchain presence decides if it can be.
        if want_on and self._installed and nvcc and sm_arch:
            self.enabled = True
            self._unavailable_reason = None
            logger.info(
                "CutlassFP8GemmBackend ENABLED (CUTLASS=%s, nvcc=found, sm_%s) "
                "— part of the recipe.", self._cutlass_version, sm_arch,
            )
        else:
            self.enabled = False
            if not want_on:
                self._unavailable_reason = (
                    "disabled by config (set acceleration.cutlass.enabled: true "
                    "and run `v9 cutlass bootstrap`)")
            elif not self._installed:
                self._unavailable_reason = (
                    "CUTLASS not installed (run `v9 cutlass bootstrap`)")
            elif not nvcc:
                self._unavailable_reason = "nvcc not on PATH"
            elif not sm_arch:
                self._unavailable_reason = "GPU sm_arch unknown (no HardwareProfile)"
            else:
                self._unavailable_reason = "disabled"
            logger.info(
                "CutlassFP8GemmBackend disabled: %s. Class still importable.",
                self._unavailable_reason,
            )

        self.version = f"0.1.0+cutlass{self._cutlass_version}"

    # ----------------------------------------------------------- ABC impl

    def applies_to(self, model_kind: str, shape: Tuple[int, ...],
                   precision: str) -> bool:
        # NOTE: returns reachability irrespective of `enabled` so the registry
        # can preview it. The orchestrator MUST gate compile() on `enabled`.
        kind = (model_kind or "").lower()
        prec = (precision or "").lower()
        if kind != "gemm":
            return False
        if prec not in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            return False
        if len(shape) != 3:  # expect (M, N, K)
            return False
        for dim in shape:
            if dim < self._MIN_DIM or dim > self._MAX_DIM:
                return False
            if dim % self._DIM_ALIGNMENT != 0:
                return False
        return True

    def estimated_compile_seconds(self, model_kind: str,
                                  shape: Tuple[int, ...]) -> float:
        # NVCC compile of a single GEMM shim: ~30-60s cold. Conservative prior.
        return 45.0

    def estimated_speedup(self, model_kind: str,
                          shape: Tuple[int, ...]) -> float:
        # FP8 vs FP16 baseline: theoretical 2x; practical ~1.7x in
        # memory-bound regimes.
        return 1.7

    def estimated_quality_delta(self) -> float:
        # FP8 introduces visible noise without good calibration; conservative.
        return 0.18

    def compile(self, source: BackendSource, shape_range: ShapeRange,
                precision: str, cancel_token: CancelToken,
                progress_cb: Optional[Callable[[float], None]] = None,
                ) -> BuildArtifactDescriptor:
        if not self.enabled:
            raise RuntimeError(
                f"CutlassFP8GemmBackend disabled: {self._unavailable_reason}. "
                f"Re-instantiate with force_enable=True for lab use. "
                f"{_INSTALL_HINT}"
            )
        if not self._sm_arch:
            raise RuntimeError(
                "CutlassFP8GemmBackend: GPU sm_arch unknown; cannot target "
                "nvcc. Pass sm_arch from HardwareProfile."
            )
        if not isinstance(source, (OnnxPath, TorchModulePickle, TrtPlanBytes)):
            raise TypeError(f"Unsupported BackendSource: {type(source)!r}")
        if len(shape_range.opt_shape) != 3:
            raise ValueError(
                "CutlassFP8GemmBackend expects shape_range.opt_shape == "
                f"(M, N, K); got {shape_range.opt_shape}."
            )

        m, n, k = shape_range.opt_shape
        prec_mode = self._precision_to_mode(precision)
        t0 = time.perf_counter()

        shim_src = self._render_shim(m, n, k, prec_mode, self._sm_arch)

        cancel_token.raise_if_cancelled()
        if progress_cb:
            progress_cb(0.10)

        with tempfile.TemporaryDirectory(prefix="cutlass_fp8_") as td:
            td_path = Path(td)
            cu_file = td_path / "shim.cu"
            so_file = td_path / ("shim.dll" if os.name == "nt" else "shim.so")
            cu_file.write_text(shim_src, encoding="utf-8")

            nvcc_cmd = ["nvcc"] + self._nvcc_flags(self._sm_arch)
            if os.name == "nt":
                # /LD = build a DLL; /Zc:preprocessor required by CUDA 13.x CCCL.
                nvcc_cmd += ["-shared", "-Xcompiler", "/LD",
                             "-Xcompiler", "/Zc:preprocessor"]
            else:
                nvcc_cmd += ["-shared", "-Xcompiler", "-fPIC"]
            nvcc_cmd += ["-o", str(so_file), str(cu_file)]

            cancel_token.raise_if_cancelled()
            if progress_cb:
                progress_cb(0.25)

            try:
                # On Windows nvcc needs the MSVC host compiler; import the
                # vcvars64 env so background compiles work outside a dev prompt.
                compile_env = None
                if os.name == "nt":
                    try:
                        from v9.accel.cutlass_bootstrap import msvc_env
                        compile_env = msvc_env()
                    except Exception:  # noqa: BLE001
                        compile_env = None
                proc = subprocess.run(
                    nvcc_cmd, capture_output=True, text=True, timeout=600,
                    env=compile_env,
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
            "CutlassFP8GemmBackend.compile OK: M=%d N=%d K=%d %s %d bytes "
            "in %.2fs (digest %s)",
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
                "sm_arch": int(self._sm_arch),
                "cutlass_version": self._cutlass_version,
                "shim_digest": digest,
                "platform": "windows" if os.name == "nt" else "posix",
            },
        )

    def load(self, artifact_path: Path) -> LoadedEngine:
        if not self.enabled:
            raise RuntimeError(
                f"CutlassFP8GemmBackend disabled: {self._unavailable_reason}. "
                f"load() requires force_enable=True. {_INSTALL_HINT}"
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

    def _nvcc_flags(self, sm_arch: int) -> List[str]:
        """NVCC flags for the detected arch. The arch comes from
        HardwareProfile; the include path from the discovered install."""
        flags = [
            f"-arch=sm_{sm_arch}",
            "-std=c++17",            # CUDA 13.x libcu++ requires >= C++17
            "-O3",
            "-DNDEBUG",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
        ]
        if self._install.include_path:
            flags += ["-I", str(self._install.include_path)]
        return flags

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
    def _render_shim(m: int, n: int, k: int, prec: PrecisionMode,
                     sm_arch: int) -> str:
        """Render a minimal CUTLASS FP8 GEMM shim. Exposes a C ABI:
            extern "C" int cutlass_fp8_gemm(
                const void* A, const void* B, void* C,
                int M, int N, int K, float alpha, float beta);
        (M, N, K) are baked in; the cache key already keys on them, so a miss
        simply rebuilds. The CUTLASS arch tag is derived from sm_arch."""
        elem = ("cutlass::float_e4m3_t" if prec is PrecisionMode.FP8_E4M3
                else "cutlass::float_e5m2_t")
        arch_tag = f"cutlass::arch::Sm{sm_arch}"
        # Tuned Ada (sm_89) FP8 tensor-core config — CUTLASS example-58 lineage.
        # The DEFAULT device::Gemm config lost to cuBLAS (0.53x); these explicit
        # tile/MMA shapes target the FP8 tensor cores properly. AlignmentA/B=16
        # (16 e4m3 elems = 16 B vectorized loads) -> M,N,K must be multiples of
        # 16 (enforced by _DIM_ALIGNMENT). Output kept fp32 for a lossless
        # comparison against the reference.
        # Tile config is env-overridable so the kernel can be AUTOTUNED without
        # editing source (V9_TBM/TBN/TBK/WPM/WPN/WPK/STAGES). The defaults below
        # are the AUTOTUNED WINNER on RTX 4090 / sm_89 (2026-06-02): a small
        # 64x128 threadblock with 4 pipeline stages maximizes SM occupancy and
        # latency-hiding — 1.90x over cuBLAS fp16 at 4096^3 (the 128x128 default
        # only hit 0.53x because too few CTAs starved the SMs).
        tb_m = int(os.environ.get("V9_TBM", 64))
        tb_n = int(os.environ.get("V9_TBN", 128))
        tb_k = int(os.environ.get("V9_TBK", 64))
        wp_m = int(os.environ.get("V9_WPM", 32))
        wp_n = int(os.environ.get("V9_WPN", 64))
        wp_k = int(os.environ.get("V9_WPK", 64))
        in_m, in_n, in_k = 16, 8, 32
        stages = int(os.environ.get("V9_STAGES", 4))
        align = 16
        return f"""\
// Auto-generated by v9/accel/cutlass.py::CutlassFP8GemmBackend (tuned Ada FP8)
// shape: M={m} N={n} K={k}  precision: {prec.value}  arch: sm_{sm_arch}
// config: TB {tb_m}x{tb_n}x{tb_k} · warp {wp_m}x{wp_n}x{wp_k} · "
//         MMA {in_m}x{in_n}x{in_k} · {stages} stages · align {align}
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

// Windows: a DLL exports nothing unless declared; ctypes/GetProcAddress needs
// this or the symbol is invisible (extern "C" only fixes name mangling).
#ifdef _WIN32
#define V9_EXPORT extern "C" __declspec(dllexport)
#else
#define V9_EXPORT extern "C"
#endif

using ElementA = {elem};
using ElementB = {elem};
using ElementC = float;
using ElementAccumulator = float;
using ElementCompute = float;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    {arch_tag},
    cutlass::gemm::GemmShape<{tb_m}, {tb_n}, {tb_k}>,
    cutlass::gemm::GemmShape<{wp_m}, {wp_n}, {wp_k}>,
    cutlass::gemm::GemmShape<{in_m}, {in_n}, {in_k}>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    {stages},
    {align},
    {align}
>;

V9_EXPORT int cutlass_fp8_gemm(
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
    if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return -3;
    cutlass::Status st = gemm_op(args);
    return st == cutlass::Status::kSuccess ? 0 : -1;
}}
"""


class _CutlassFP8GemmRunner:
    """Thin ctypes wrapper around the compiled CUTLASS shim DLL/.so."""

    def __init__(self, dll: Any, path: Path) -> None:
        self.dll = dll
        self.path = path
        import ctypes
        try:
            fn = self.dll.cutlass_fp8_gemm
        except AttributeError as exc:
            raise RuntimeError(
                f"Loaded shim missing 'cutlass_fp8_gemm' symbol: {exc}"
            ) from exc
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
        ]
        self._fn = fn

    def gemm(self, a_ptr: int, b_ptr: int, c_ptr: int,
             m: int, n: int, k: int,
             alpha: float = 1.0, beta: float = 0.0) -> int:
        """Invoke the FP8 GEMM. Returns 0 on success, negative on error."""
        return int(self._fn(a_ptr, b_ptr, c_ptr, m, n, k, alpha, beta))


if __name__ == "__main__":
    be = CutlassFP8GemmBackend(sm_arch=89)
    print(f"name={be.name} version={be.version} enabled={be.enabled}")
    print(f"unavailable_reason: {be._unavailable_reason}")
    print("applies_to(gemm,(1024,1024,1024),fp8) ->",
          be.applies_to("gemm", (1024, 1024, 1024), "fp8"))
