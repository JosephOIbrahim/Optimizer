"""
V12 — the KERNEL-tier actuator.

What is actually true on this box, established by research and measurement:

  * **CUTLASS consumed as headers, compiled by a single nvcc invocation, works** —
    proven here at 296.7 TFLOPS on the RTX 4090. This is the path V12 drives.
  * **CUTLASS's own CMake build does not work on Windows** (upstream issue #2911,
    still open). That is a packaging problem, not a capability problem, and it is
    why this module never invokes CUTLASS's CMake.
  * **cuTile is available** (`cuda-tile` 1.6.0, win_amd64, sm80-family since 1.2.0)
    but **cannot emit FP8** — `TileUnsupportedFeatureError` on the sm80 family. FP8
    is the primary diffusion lever here, so cuTile is a non-FP8 candidate only.
  * **CuTe DSL is unreachable** — POSIX-only wheels, no Windows build. Any design
    that depends on it is a design that does not run on this machine.
  * **FP4/NVFP4 and thread-block clusters are hard-blocked on sm_89.** Not slow —
    absent. Checkpoints that use them fall back to emulation, up to 2x *slower*
    than FP8.

The failure modes this module exists to prevent are all flag-level, and all silent
until link time: a missing `-std=c++17`, a missing `--expt-relaxed-constexpr`, a
missing `/Zc:__cplusplus` or `/bigobj`, and CUTLASS headers that are not
self-contained unless `cute/tensor.hpp` is included first (upstream issue #1827).
Each is applied here rather than left to the caller to remember.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import paths
from ..change import Change, ChangeKind, Snapshot
from ..admission import ApplyResult

log = logging.getLogger("v12.actuator.kernel")

SM_ARCH = "sm_89"

#: The flags that make a single-invocation CUTLASS compile succeed on Windows.
#: Each one corresponds to a documented failure when absent.
NVCC_FLAGS = [
    f"-arch={SM_ARCH}",
    "-std=c++17",                    # CUTLASS 3.x requires it; absent -> cryptic template errors
    "--expt-relaxed-constexpr",      # CUTLASS uses relaxed constexpr heavily
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--shared",
    "-Xcompiler", "/Zc:__cplusplus", # MSVC reports 199711L otherwise; CUTLASS gates on __cplusplus
    "-Xcompiler", "/bigobj",         # template-heavy TUs exceed the default object section limit
    "-Xcompiler", "/MD",
    "-DNDEBUG",
]

#: CUTLASS headers are not self-contained; this must come first (issue #1827).
REQUIRED_FIRST_INCLUDE = "cute/tensor.hpp"


@dataclass
class Toolchain:
    nvcc: str = ""
    nvcc_version: str = ""
    cuda_include: str = ""
    cutlass_include: str = ""
    cl: str = ""
    long_paths: Optional[bool] = None
    ok: bool = False
    problems: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return (f"nvcc {self.nvcc_version} @ {self.nvcc}; CUTLASS @ {self.cutlass_include}; "
                    f"long paths {'ON' if self.long_paths else 'OFF (MAX_PATH applies)'}")
        return "toolchain incomplete: " + "; ".join(self.problems)


def _long_paths_enabled() -> Optional[bool]:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem", 0,
                            winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0)) as k:
            return int(winreg.QueryValueEx(k, "LongPathsEnabled")[0]) == 1
    except Exception:
        return None


def probe_toolchain() -> Toolchain:
    """Find the compiler and the headers, and report precisely what is missing.

    MAX_PATH is checked deliberately: CUTLASS template instantiation produces
    object paths that blow past 260 characters, and the failure surfaces as an
    unrelated-looking 'cannot open file' deep in the build.
    """
    t = Toolchain()
    t.long_paths = _long_paths_enabled()
    if t.long_paths is False:
        t.problems.append(
            "LongPathsEnabled=0 — CUTLASS template object paths exceed MAX_PATH (260) and "
            "will fail to open; set HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem"
            "\\LongPathsEnabled=1 and reboot")

    nvcc = shutil.which("nvcc")
    if not nvcc:
        for cand in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe",
                     r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"):
            if Path(cand).exists():
                nvcc = cand
                break
    if nvcc:
        t.nvcc = nvcc
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True, text=True,
                                 timeout=20,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            for line in out.splitlines():
                if "release" in line.lower():
                    t.nvcc_version = line.strip()
                    break
        except Exception as e:
            t.problems.append(f"nvcc --version failed: {e}")
    else:
        t.problems.append("nvcc not found on PATH or in the CUDA toolkit directories")

    if t.nvcc:
        t.cuda_include = str(Path(t.nvcc).parent.parent / "include")

    # CUTLASS may arrive several ways; check the ones this project has used.
    cutlass_candidates = [
        Path(r"D:\OptimizerV9\cutlass\include"),
        Path(r"D:\OptimizerV9\third_party\cutlass\include"),
        Path(r"C:\Users\User\OneDrive\Desktop\Optimizer\third_party\cutlass\include"),
    ]
    for c in cutlass_candidates:
        if (c / "cute" / "tensor.hpp").exists():
            t.cutlass_include = str(c)
            break
    if not t.cutlass_include:
        t.problems.append("CUTLASS headers not found (need a directory containing cute/tensor.hpp)")

    t.cl = shutil.which("cl") or ""
    if not t.cl:
        t.problems.append("cl.exe (MSVC) not on PATH — nvcc needs it for host compilation")

    t.ok = bool(t.nvcc and t.cutlass_include and t.cl) and t.long_paths is not False
    return t


@dataclass
class KernelBuild:
    name: str
    source: str
    artifact: str = ""
    arch: str = SM_ARCH
    flags: List[str] = field(default_factory=list)
    ok: bool = False
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    sha256: str = ""
    built: float = field(default_factory=time.time)

    def manifest(self) -> Dict[str, object]:
        d = asdict(self)
        d.pop("stdout", None)
        d.pop("stderr", None)
        return d


#: Where compiled artifacts land. This is the cache MOS2026 and V9 both reach for;
#: keeping it in one place is what makes the coupling checkable.
KERNEL_CACHE = paths.DATA_ROOT / "kernels"


def relevance(objective, roofline_bound: str = "") -> Tuple[bool, str]:
    """Is a kernel change even the right lever for this objective?

    A memory-bound workload gains nothing from a better GEMM. V9's roofline can
    say which regime a workload is in, and this is the DERIVED evidence class doing
    real work: the model predicts the ceiling, and the prediction decides whether
    spending a compile is worth it at all.
    """
    if roofline_bound == "memory":
        return False, ("roofline says this workload is memory-bound — a faster kernel "
                       "cannot help; reduce bytes moved instead")
    if roofline_bound == "compute":
        return True, "roofline says compute-bound — kernel work is the right lever"
    return True, ("roofline bound unknown — proceeding, but the predicted gain is not "
                  "derivable, so this cannot be admitted as DERIVED")


def compile_source(name: str, source_text: str, extra_flags: Optional[List[str]] = None,
                   timeout: int = 900) -> KernelBuild:
    """Compile one CUDA translation unit into a .dll.

    Exactly one nvcc invocation, with the full flag set. CUTLASS's CMake is never
    used — it does not build on Windows (upstream #2911), and its failure would be
    indistinguishable here from a real compile error.
    """
    t = probe_toolchain()
    b = KernelBuild(name=name, source=source_text, flags=list(NVCC_FLAGS) + list(extra_flags or []))
    if not t.ok:
        b.stderr = "toolchain not ready: " + "; ".join(t.problems)
        return b

    KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
    src = KERNEL_CACHE / f"{name}.cu"
    src.write_text(source_text, encoding="utf-8")
    out = KERNEL_CACHE / f"{name}.dll"

    cmd = [
        t.nvcc,
        *b.flags,
        "-I", t.cuda_include,
        "-I", t.cutlass_include,
        str(src),
        "-o", str(out),
    ]
    start = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        b.stdout, b.stderr = p.stdout or "", p.stderr or ""
        b.ok = (p.returncode == 0 and out.exists())
    except subprocess.TimeoutExpired:
        b.stderr = f"compile timed out after {timeout}s"
    except Exception as e:
        b.stderr = f"{type(e).__name__}: {e}"
    b.duration_s = time.time() - start
    b.artifact = str(out) if b.ok else ""
    if b.ok:
        b.sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
    (KERNEL_CACHE / f"{name}.manifest.json").write_text(
        json.dumps(b.manifest(), indent=2), encoding="utf-8")
    return b


class KernelActuator:
    """Applies KERNEL-tier changes by producing a compiled artifact."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def apply(self, c: Change) -> ApplyResult:
        if c.kind is not ChangeKind.KERNEL_ARTIFACT:
            return ApplyResult(change_id=c.id, applied=False,
                               error=f"KernelActuator cannot carry {c.kind.value}")
        source = c.extra.get("source")
        if not source:
            return ApplyResult(change_id=c.id, applied=False,
                               error="no CUDA source supplied in change.extra['source']")
        if self.dry_run:
            return ApplyResult(change_id=c.id, applied=False, detail="dry-run: would compile")
        b = compile_source(c.extra.get("name", c.id), source,
                           c.extra.get("flags"))
        if not b.ok:
            tail = "\n".join((b.stderr or "").strip().splitlines()[-12:])
            return ApplyResult(change_id=c.id, applied=False,
                               error=f"compile failed after {b.duration_s:.1f}s:\n{tail}")
        return ApplyResult(
            change_id=c.id, applied=True, verified=True,
            detail=f"{b.artifact} ({b.duration_s:.1f}s, sha256 {b.sha256[:12]})",
            verification_source="artifact exists and hashed — this is a real compile, "
                                "not a claimed one; whether it is FASTER is a separate "
                                "measurement and is not established here")

    def apply_undo(self, c: Change) -> ApplyResult:
        """Remove the artifact. Compilation is additive, so this is a true restore:
        the prior state was 'no artifact', which is exactly what we return to."""
        name = c.extra.get("name", c.id)
        removed = []
        for p in (KERNEL_CACHE / f"{name}.dll", KERNEL_CACHE / f"{name}.cu",
                  KERNEL_CACHE / f"{name}.manifest.json"):
            try:
                if p.exists():
                    p.unlink()
                    removed.append(p.name)
            except Exception as e:
                return ApplyResult(change_id=c.id, applied=False, error=str(e))
        return ApplyResult(change_id=c.id, applied=True,
                           detail="removed " + (", ".join(removed) or "nothing (already absent)"))

    def verify(self, c: Change) -> ApplyResult:
        name = c.extra.get("name", c.id)
        p = KERNEL_CACHE / f"{name}.dll"
        return ApplyResult(change_id=c.id, applied=c.applied, verified=p.exists(),
                           verification_source=f"artifact presence at {p}" if p.exists()
                           else "artifact missing")


def toolchain_report() -> str:
    t = probe_toolchain()
    L = ["V12 kernel toolchain", "-" * 68, "  " + t.summary(), ""]
    L.append(f"  kernel cache : {KERNEL_CACHE}")
    try:
        arts = sorted(KERNEL_CACHE.glob("*.dll"))
        L.append(f"  artifacts    : {len(arts)}")
        for a in arts[:10]:
            L.append(f"    {a.name}  {a.stat().st_size / 1024:.0f} KiB")
    except Exception:
        pass
    L.append("")
    L.append("  unreachable on this hardware (not a config problem):")
    L.append("    CuTe DSL        POSIX-only wheels")
    L.append("    FP4 / NVFP4     Blackwell (sm_100+); emulated on Ada at up to 2x slower than FP8")
    L.append("    thread clusters CC 9.0+ (Hopper); hard-blocked on sm_89")
    L.append("    cuTile + FP8    TileUnsupportedFeatureError on the sm80 family")
    return "\n".join(L)


if __name__ == "__main__":
    print(toolchain_report())
