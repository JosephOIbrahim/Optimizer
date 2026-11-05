"""V9 — the render-driver: apply the recipe, launch a render, track progress.

`v9 render houdini scene.usd`  or  `v9 render comfyui workflow.json`

This is the half of the goal the monitor loop never covered: actually *driving*
a render with the optimization recipe applied and reporting progress in Joe's
marathon-marker style.

Flow:
  1. Resolve the recipe env for the chosen app (global CUDA + per-app env vars
     from config) and inject it into THIS process so the spawned render
     inherits it. Honest: env vars only — no registry/admin side-effects here.
  2. Locate the renderer binary:
       houdini -> husk.exe from the newest installed Houdini 21 build
       comfyui -> the `comfy` CLI (comfy-cli), else python <install>/main.py
  3. Launch it, stream stdout, parse progress markers, and print mile markers
     (=== Mile X of ~Y ===) plus a final wall-clock.

Degrade-safe: a missing binary or bad path returns a clear non-zero, never a
traceback. `--dry-run` prints the resolved recipe + command and launches
nothing.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from v9.config import Config

# Houdini install root on Windows; newest build wins.
_SIDEFX_ROOT = Path(r"C:\Program Files\Side Effects Software")

# progress patterns
_HUSK_FRAME = re.compile(r"(?:Rendering|render(?:ed)?)\s+frame\s+(\d+)", re.I)
_HUSK_PERCENT = re.compile(r"(\d{1,3})\s*%")
_COMFY_STEP = re.compile(r"(\d+)\s*/\s*(\d+)")


# ---------------------------------------------------------------------------
# recipe env
# ---------------------------------------------------------------------------
def _recipe_env(app: str, config: Config) -> dict[str, str]:
    """The env vars to apply before this app's render. Global CUDA/cuDNN first,
    then the app's own block (app wins on conflict)."""
    env: dict[str, str] = {}
    gpu = config.get("gpu", {}) or {}
    for block in ("cuda_env_vars", "cudnn_env_vars"):
        for k, v in (gpu.get(block, {}) or {}).items():
            env[str(k)] = str(v)
    app_cfg = config.get(app, {}) or {}
    for k, v in (app_cfg.get("env_vars", {}) or {}).items():
        env[str(k)] = str(v)
    # Houdini render-time niceties not in the env_vars block
    if app == "houdini":
        karma = app_cfg.get("karma_xpu", {}) or {}
        if karma.get("enable_denoiser", True):
            env.setdefault("HOUDINI_OPTIX_DENOISER", "1")
    # CUTLASS, if it's in the recipe, must be visible to the child too
    cu = os.environ.get("CUTLASS_PATH")
    if cu:
        env.setdefault("CUTLASS_PATH", cu)
        if os.environ.get("CUTLASS_INCLUDE_DIR"):
            env.setdefault("CUTLASS_INCLUDE_DIR", os.environ["CUTLASS_INCLUDE_DIR"])
    return env


def _apply_env(env: dict[str, str]) -> None:
    for k, v in env.items():
        os.environ[k] = v


# ---------------------------------------------------------------------------
# binary resolution
# ---------------------------------------------------------------------------
def _newest_houdini() -> Optional[Path]:
    try:
        builds = sorted(
            (d for d in _SIDEFX_ROOT.glob("Houdini 21.*") if d.is_dir()),
            key=lambda d: _version_key(d.name), reverse=True)
        for b in builds:
            husk = b / "bin" / "husk.exe"
            if husk.exists():
                return husk
    except OSError:
        pass
    return None


def _version_key(name: str) -> tuple:
    nums = re.findall(r"\d+", name)
    return tuple(int(x) for x in nums) if nums else (0,)


def _comfy_cli() -> Optional[str]:
    import shutil
    return shutil.which("comfy")


# ---------------------------------------------------------------------------
# command construction
# ---------------------------------------------------------------------------
def _houdini_cmd(husk: Path, scene: str, out: Optional[str],
                 frames: Optional[str], extra: list[str]) -> list[str]:
    cmd = [str(husk), "--verbose", "3"]
    if out:
        cmd += ["--output", out]
    if frames:
        # frames "1-100" / "1-100x2" -> husk --frame/--frame-count or range
        m = re.match(r"(\d+)-(\d+)(?:x(\d+))?$", frames)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            inc = int(m.group(3) or 1)
            cmd += ["--frame", str(start),
                    "--frame-count", str(max(1, (end - start) // inc + 1)),
                    "--frame-inc", str(inc)]
        else:
            cmd += ["--frame", frames]
    cmd += extra
    cmd += [scene]
    return cmd


def _comfy_cmd(comfy: str, workflow: str, out: Optional[str],
               extra: list[str]) -> list[str]:
    # comfy-cli: queue a workflow and wait for it (CLI render).
    cmd = [comfy, "run", "--workflow", workflow, "--wait", "--timeout", "3600"]
    cmd += extra
    return cmd


# ---------------------------------------------------------------------------
# launch + progress
# ---------------------------------------------------------------------------
def _stream(cmd: list[str], total_frames: Optional[int],
            label: str) -> int:
    print(f"  launching: {' '.join(cmd)}")
    print()
    t0 = time.perf_counter()
    last_mile = -1
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
    except FileNotFoundError:
        print(f"  ERROR: cannot launch {cmd[0]!r} (not found).")
        return 127
    except OSError as exc:
        print(f"  ERROR: launch failed: {exc}")
        return 1

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        print(f"    {line}")
        mile = _progress_mile(line, total_frames)
        if mile is not None and mile != last_mile:
            last_mile = mile
            tag = (f"~{total_frames}" if total_frames else "~?")
            elapsed = time.perf_counter() - t0
            print(f"=== {label}: Mile {mile} of {tag} "
                  f"({elapsed:.0f}s elapsed) ===")
    proc.wait()
    elapsed = time.perf_counter() - t0
    status = "OK" if proc.returncode == 0 else f"exit {proc.returncode}"
    print()
    print(f"=== {label}: render finished — {status} in {elapsed:.1f}s ===")
    return proc.returncode


def _progress_mile(line: str, total: Optional[int]) -> Optional[int]:
    m = _HUSK_FRAME.search(line)
    if m:
        return int(m.group(1))
    m = _COMFY_STEP.search(line)
    if m:
        return int(m.group(1))
    m = _HUSK_PERCENT.search(line)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------
def run(app: str, file: str, config: Config,
        extra_args: Optional[list[str]] = None, dry_run: bool = False,
        out: Optional[str] = None, frames: Optional[str] = None) -> int:
    extra = [a for a in (extra_args or []) if a != "--"]
    scene = str(Path(file))
    if not dry_run and not Path(scene).exists():
        print(f"  ERROR: file not found: {scene}")
        return 2

    print("=" * 64)
    print(f" Optimizer V9 — render driver ({app})")
    print("=" * 64)
    print("  " + config.profile.summary())

    env = _recipe_env(app, config)
    print(f"\n  recipe env ({len(env)} vars applied to the render process):")
    for k, v in env.items():
        shown = v if len(v) <= 48 else v[:45] + "..."
        print(f"    {k}={shown}")

    if app == "houdini":
        husk = _newest_houdini()
        if husk is None:
            print("\n  ERROR: husk.exe not found under "
                  f"{_SIDEFX_ROOT}. Is Houdini 21 installed?")
            return 3
        print(f"\n  renderer: {husk}")
        cmd = _houdini_cmd(husk, scene, out, frames, extra)
        total = _frame_count(frames)
        label = "Houdini/Karma"
    elif app == "comfyui":
        comfy = _comfy_cli()
        if comfy is None:
            print("\n  ERROR: `comfy` CLI not found on PATH. Install comfy-cli "
                  "(`pip install comfy-cli`) or start ComfyUI and use its API.")
            return 3
        print(f"\n  renderer: {comfy} (comfy-cli)")
        cmd = _comfy_cmd(comfy, scene, out, extra)
        total = None
        label = "ComfyUI"
    else:
        print(f"  unknown app: {app}")
        return 2

    if dry_run:
        print(f"\n  [dry-run] would run: {' '.join(cmd)}")
        print("  [dry-run] no render launched.")
        return 0

    _apply_env(env)
    return _stream(cmd, total, label)


def _frame_count(frames: Optional[str]) -> Optional[int]:
    if not frames:
        return None
    m = re.match(r"(\d+)-(\d+)(?:x(\d+))?$", frames)
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2))
    inc = int(m.group(3) or 1)
    return max(1, (end - start) // inc + 1)
