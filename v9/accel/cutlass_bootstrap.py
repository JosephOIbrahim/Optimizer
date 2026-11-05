"""V9 — CUTLASS bootstrap: turn the dormant CUTLASS backend into an installed,
usable ingredient of the recipe.

CUTLASS is header-only (templates), so "installing" it is: clone the repo to an
off-OneDrive location, point $CUTLASS_PATH / $CUTLASS_INCLUDE_DIR at it, verify
nvcc can see <cutlass/cutlass.h>, and compile a one-line smoke kernel to prove
the toolchain end-to-end. No pip wheel, no system package.

Design rules honored:
  * Off-OneDrive — clone lands under paths.DATA_ROOT (e.g. D:/OptimizerV9/cutlass
    by default), never inside the OneDrive-synced repo tree.
  * Detect, don't hardcode — the sm_arch handed to the smoke compile comes from
    a HardwareProfile, never a literal.
  * Honest + degrade-safe — every step returns a structured result; nothing here
    raises into the caller. A missing git/nvcc is reported, not fatal.
  * Persist for new processes — $CUTLASS_PATH is set in this process (so the
    very next backend probe sees it) AND via setx on Windows (so a fresh
    ComfyUI / Houdini / husk launch inherits it).

This module performs a NETWORK CLONE and writes a user env var, so it is only
invoked by the explicit `v9 cutlass bootstrap` subcommand — never automatically
inside an optimization cycle.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from v9 import paths

log = logging.getLogger("v9.accel.cutlass_bootstrap")

# Pinned to a tag, not a moving branch (DESIGN risk R1: CUTLASS DSL moves fast).
# A header-only checkout at a known-good tag keeps the smoke kernel reproducible.
_CUTLASS_REPO = "https://github.com/NVIDIA/cutlass.git"
_CUTLASS_DEFAULT_TAG = "v4.5.1"


@dataclass
class BootstrapResult:
    ok: bool
    cutlass_path: Optional[Path] = None
    include_path: Optional[Path] = None
    version_tag: str = ""
    persisted_env: bool = False
    smoke_compiled: bool = False
    steps: list[str] = field(default_factory=list)
    error: str = ""

    def summary(self) -> str:
        head = "OK" if self.ok else "FAILED"
        bits = [f"CUTLASS bootstrap: {head}"]
        if self.cutlass_path:
            bits.append(f"path={self.cutlass_path}")
        if self.version_tag:
            bits.append(f"tag={self.version_tag}")
        bits.append(f"env_persisted={self.persisted_env}")
        bits.append(f"smoke={'pass' if self.smoke_compiled else 'skip/fail'}")
        if self.error:
            bits.append(f"error={self.error}")
        return " | ".join(bits)


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


# ---------------------------------------------------------------------------
# MSVC host-compiler environment (nvcc on Windows needs cl.exe + INCLUDE/LIB).
# We import the vcvars64 environment once so CUTLASS compiles work from ANY
# shell — not only a "Developer Command Prompt". This is what lets the
# background compiler build kernels unattended.
# ---------------------------------------------------------------------------
_MSVC_ENV_CACHE: Optional[dict] = None


def _vswhere_install_path() -> Optional[Path]:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio"
                   r"\Installer\vswhere.exe")
    if not vswhere.exists():
        return None
    try:
        out = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True, timeout=30)
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        return Path(lines[0]) if lines else None
    except (OSError, subprocess.SubprocessError):
        return None


def msvc_env() -> Optional[dict]:
    """Return an environment dict with MSVC (cl.exe, INCLUDE, LIB) + the current
    PATH (so nvcc/CUDA stay reachable), by importing vcvars64.bat. None when no
    MSVC is found or this isn't Windows. Result is cached for the process."""
    global _MSVC_ENV_CACHE
    if _MSVC_ENV_CACHE is not None:
        return _MSVC_ENV_CACHE or None
    if os.name != "nt":
        _MSVC_ENV_CACHE = {}
        return None
    install = _vswhere_install_path()
    if install is None:
        _MSVC_ENV_CACHE = {}
        return None
    vcvars = install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        _MSVC_ENV_CACHE = {}
        return None
    try:
        # cmd inherits this process's env (CUDA on PATH), vcvars adds MSVC,
        # then `set` dumps the full merged environment for us to capture.
        # NB: pass a single STRING (not a list). A list makes subprocess
        # C-runtime-escape the quotes, which cmd.exe mis-parses; the string
        # form lets cmd's quote-preservation rule keep the spaced .bat path
        # intact (the distutils._msvccompiler pattern).
        out = subprocess.run(
            f'cmd /c "{vcvars}" >nul 2>&1 && set',
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            _MSVC_ENV_CACHE = {}
            return None
        env = dict(os.environ)
        for line in out.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                env[k] = v
        _MSVC_ENV_CACHE = env
        log.info("MSVC host environment imported from %s", vcvars)
        return env
    except (OSError, subprocess.SubprocessError):
        _MSVC_ENV_CACHE = {}
        return None


def default_install_dir() -> Path:
    """Off-OneDrive CUTLASS home (sibling of the engine cache)."""
    return paths.DATA_ROOT / "cutlass"


def already_installed(install_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the include path if a usable CUTLASS checkout exists, else None.
    Checks the explicit dir, then $CUTLASS_PATH, then the default location."""
    candidates: list[Path] = []
    if install_dir:
        candidates.append(Path(install_dir))
    env_path = os.environ.get("CUTLASS_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(default_install_dir())
    for root in candidates:
        try:
            inc = root / "include"
            if (inc / "cutlass" / "cutlass.h").exists():
                return inc
        except OSError:
            continue
    return None


def bootstrap(sm_arch: Optional[int] = None,
              install_dir: Optional[Path] = None,
              tag: str = _CUTLASS_DEFAULT_TAG,
              persist_env: bool = True,
              run_smoke: bool = True) -> BootstrapResult:
    """Clone CUTLASS (if absent), wire $CUTLASS_PATH, verify the header, and
    optionally compile a smoke kernel targeting `sm_arch`. Never raises."""
    res = BootstrapResult(ok=False)
    install_dir = Path(install_dir) if install_dir else default_install_dir()

    try:
        # ---- 0. nvcc gate (CUTLASS without a compiler is useless) ----------
        nvcc = _which("nvcc")
        if not nvcc:
            res.error = ("nvcc not on PATH — install the CUDA Toolkit. CUTLASS "
                         "is header-only but needs nvcc to compile kernels.")
            res.steps.append("nvcc: NOT FOUND")
            return res
        res.steps.append(f"nvcc: {nvcc}")

        # ---- 1. already present? -------------------------------------------
        existing_inc = already_installed(install_dir)
        if existing_inc is not None:
            res.steps.append(f"CUTLASS already present at {existing_inc.parent}")
            res.cutlass_path = existing_inc.parent
            res.include_path = existing_inc
        else:
            # ---- 2. clone --------------------------------------------------
            git = _which("git")
            if not git:
                res.error = "git not on PATH — cannot clone CUTLASS."
                res.steps.append("git: NOT FOUND")
                return res
            res.steps.append(f"git: {git}")
            install_dir.parent.mkdir(parents=True, exist_ok=True)
            res.steps.append(f"cloning {_CUTLASS_REPO}@{tag} -> {install_dir}")
            clone = subprocess.run(
                [git, "clone", "--depth", "1", "--branch", tag,
                 _CUTLASS_REPO, str(install_dir)],
                capture_output=True, text=True, timeout=600,
            )
            if clone.returncode != 0:
                res.error = f"git clone failed: {clone.stderr.strip()[:400]}"
                return res
            inc = install_dir / "include"
            if not (inc / "cutlass" / "cutlass.h").exists():
                res.error = (f"clone completed but {inc}/cutlass/cutlass.h "
                             "missing — unexpected layout.")
                return res
            res.cutlass_path = install_dir
            res.include_path = inc
            res.steps.append("clone OK; cutlass.h present")

        res.version_tag = tag

        # ---- 3. wire env (this process + persisted) ------------------------
        os.environ["CUTLASS_PATH"] = str(res.cutlass_path)
        os.environ["CUTLASS_INCLUDE_DIR"] = str(res.include_path)
        res.steps.append("set $CUTLASS_PATH / $CUTLASS_INCLUDE_DIR for session")
        if persist_env:
            res.persisted_env = _persist_env(
                {"CUTLASS_PATH": str(res.cutlass_path),
                 "CUTLASS_INCLUDE_DIR": str(res.include_path)})
            res.steps.append(
                "persisted env via setx (new processes only)"
                if res.persisted_env else "env NOT persisted (setx failed)")

        # ---- 4. smoke compile ---------------------------------------------
        if run_smoke:
            if not sm_arch:
                res.steps.append("smoke skipped: sm_arch unknown "
                                 "(pass a HardwareProfile)")
            else:
                ok, msg = _smoke_compile(res.include_path, sm_arch)
                res.smoke_compiled = ok
                res.steps.append(f"smoke compile: {msg}")
                if not ok:
                    res.error = msg
                    return res

        res.ok = True
        return res
    except subprocess.TimeoutExpired as exc:
        res.error = f"timed out: {exc}"
        return res
    except Exception as exc:  # noqa: BLE001 — bootstrap must never raise
        res.error = f"unexpected: {exc}"
        log.debug("bootstrap failed", exc_info=True)
        return res


def _persist_env(vars_: dict[str, str]) -> bool:
    """setx each var for the current user. True only if all succeed."""
    if os.name != "nt":
        # POSIX: we can't persist for arbitrary future shells safely; report False.
        return False
    ok = True
    for k, v in vars_.items():
        try:
            out = subprocess.run(["setx", k, v], capture_output=True,
                                 text=True, timeout=8)
            ok = ok and out.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
    return ok


def _smoke_compile(include_path: Path, sm_arch: int) -> tuple[bool, str]:
    """Compile a trivial translation unit that includes CUTLASS headers and
    references an FP8 GEMM type. Proves nvcc + headers + arch agree. Compiles to
    an object only (no link) to stay fast and dependency-free."""
    src = f"""\
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm{sm_arch}>;
extern "C" int v9_cutlass_smoke() {{ Gemm op; (void)op; return 0; }}
"""
    try:
        with tempfile.TemporaryDirectory(prefix="v9_cutlass_smoke_") as td:
            cu = Path(td) / "smoke.cu"
            obj = Path(td) / ("smoke.obj" if os.name == "nt" else "smoke.o")
            cu.write_text(src, encoding="utf-8")
            cmd = ["nvcc", f"-arch=sm_{sm_arch}", "-std=c++17",
                   "--expt-relaxed-constexpr", "-c",
                   "-I", str(include_path), "-o", str(obj), str(cu)]
            if os.name == "nt":
                # CUDA 13.x CCCL headers require MSVC's conforming preprocessor.
                cmd += ["-Xcompiler", "/Zc:preprocessor"]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=300, env=msvc_env())
            if proc.returncode == 0 and obj.exists():
                return True, f"sm_{sm_arch} object built OK"
            return False, (f"nvcc rc={proc.returncode}: "
                           f"{(proc.stderr or proc.stdout).strip()[:400]}")
    except subprocess.TimeoutExpired:
        return False, "smoke compile timed out (>300s)"
    except Exception as exc:  # noqa: BLE001
        return False, f"smoke compile error: {exc}"


if __name__ == "__main__":
    from v9.hardware import detect
    prof = detect()
    print(bootstrap(sm_arch=prof.gpu.sm_arch).summary())
