"""V9 — doctor: the HONEST health + capability report.

Prints, and never lies (DESIGN_V9 §4.4):
  * HardwareProfile.summary()      — what we actually detected
  * paths.info()                   — data root + off-OneDrive check
  * venv health                    — the committed venv is broken (its python
                                     points at a nonexistent miniconda); we say
                                     so and recommend a py -3.12 rebuild
  * backend capability_map()       — which accel backends are installed vs
                                     dormant (TRT / Sage / CUTLASS) and WHY

`run()` always returns 0 — doctor is a report, not a gate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

from v9 import paths
from v9.config import Config, load_config


def _hr(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _check_venv() -> tuple[bool, str]:
    """Inspect the committed venv. Returns (healthy, message). The known-broken
    state is a pyvenv.cfg / python.exe pointing at a vanished miniconda."""
    repo = paths.REPO_ROOT
    venv = repo / "venv"
    if not venv.exists():
        return False, "no venv/ present"
    py = venv / "Scripts" / "python.exe"
    cfg = venv / "pyvenv.cfg"
    home = None
    if cfg.exists():
        try:
            for line in cfg.read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("home"):
                    home = line.split("=", 1)[1].strip()
                    break
        except OSError:
            pass
    if home and not Path(home).exists():
        return False, (f"venv home points at missing interpreter: {home} "
                       "(broken)")
    if not py.exists():
        return False, "venv/Scripts/python.exe missing (broken)"
    return True, f"venv python present: {py}"


def run(config: Optional[Config] = None) -> int:
    if config is None:
        config = load_config()

    print("=" * 64)
    print(" Optimizer V9 — doctor (honest capability report)")
    print("=" * 64)

    # ---- hardware ----------------------------------------------------
    _hr("Hardware (DETECTED — single source of truth)")
    profile = config.profile
    print("  " + profile.summary())
    print(f"  probe source: {profile.gpu.source}")
    print(f"  cache-key fields: {profile.cache_key_fields()}")
    if profile.gpu.driver_version is None:
        print("  WARNING: no GPU driver detected (nvidia-smi / pynvml absent?).")

    # ---- paths -------------------------------------------------------
    _hr("Paths (mutable state must live off-OneDrive)")
    info = paths.info()
    print(f"  data_root:  {info['data_root']}")
    print(f"  repo_root:  {info['repo_root']}")
    if info["under_onedrive"]:
        print("  WARNING: data root is UNDER OneDrive — set $OPTIMIZER_V9_DATA "
              "to an off-OneDrive path to avoid SQLite/WAL sync churn.")
    else:
        print("  OK: data root is off-OneDrive.")

    # ---- python / venv ----------------------------------------------
    _hr("Python / virtual environment")
    print(f"  running interpreter: {sys.executable}")
    print(f"  python version:      {sys.version.split()[0]}")
    healthy, msg = _check_venv()
    print(f"  committed venv:      {'OK' if healthy else 'BROKEN'} — {msg}")
    if not healthy:
        print("  RECOMMEND: rebuild the venv on Python 3.12 (TRT/torch wheels "
              "target 3.10–3.12; 3.14 has no TRT wheels yet):")
        print("    py -3.12 -m venv venv")
        print("    venv\\Scripts\\activate")
        print("    pip install -r requirements.txt")

    # ---- core dependency probes -------------------------------------
    _hr("Core dependencies")
    for mod, why in (("psutil", "process / affinity / VRAM context"),
                     ("pynvml", "rich GPU telemetry (PCIe, util, temp)"),
                     ("yaml", "config.yaml parsing"),
                     ("torch", "benchmark baseline + diffusion runtime")):
        present, ver = _probe_module(mod)
        state = f"present (v{ver})" if present else "ABSENT"
        print(f"  {mod:10s} {state:18s} — {why}")

    # ---- backend capability map -------------------------------------
    _hr("Acceleration backends (installed vs dormant)")
    cap = _capability_map(profile, config)
    if not cap:
        print("  (registry unavailable)")
    for name, meta in cap.items():
        state = "ENABLED " if meta["enabled"] else "dormant "
        line = f"  {name:20s} {state} v={meta['version']}"
        if meta.get("reason"):
            line += f"   reason: {meta['reason']}"
        print(line)
    enabled = [n for n, m in cap.items() if m["enabled"]]
    if not enabled:
        print()
        print("  NOTE: no acceleration backend is active. The build-and-cache "
              "half is dormant — V9 will still apply config tuning, but the "
              "+40-150% compile wins require installing a backend (see "
              "Mile 5 activation). doctor is NOT claiming a speedup that "
              "isn't there.")

    # ---- CUTLASS toolchain (the recipe's headline ingredient) -------
    _hr("CUTLASS toolchain (Houdini Vellum/Pyro FP8 GEMM)")
    _report_cutlass(profile, config)

    # ---- engine cache ------------------------------------------------
    _hr("Engine cache")
    _report_cache(profile, config)

    print()
    print("doctor: report complete (exit 0 — this is a report, not a gate).")
    return 0


def _probe_module(name: str) -> tuple[bool, str]:
    try:
        mod = __import__(name)
        return True, str(getattr(mod, "__version__", "?"))
    except Exception:
        return False, ""


def _capability_map(profile: Any,
                    config: Optional[Config] = None) -> dict[str, dict[str, Any]]:
    try:
        from v9.accel.registry import build_default_registry
        reg = build_default_registry(profile, config)
        return reg.capability_map()
    except Exception as exc:  # noqa: BLE001
        return {"<registry error>": {"enabled": False, "version": "?",
                                     "reason": str(exc)}}


def _report_cutlass(profile: Any, config: Config) -> None:
    """Honest CUTLASS readiness: nvcc, $CUTLASS_PATH, header, config opt-in.
    Tells the user the exact next step rather than a vague 'dormant'."""
    import shutil as _sh

    nvcc = _sh.which("nvcc")
    print(f"  nvcc:          {nvcc or 'NOT FOUND (install CUDA Toolkit)'}")

    accel = (config.get("acceleration", {}) or {}) if config else {}
    cutlass_cfg = (accel.get("cutlass", {}) or {}) if isinstance(accel, dict) else {}
    opted_in = bool(cutlass_cfg.get("enabled", False))
    print(f"  config opt-in: acceleration.cutlass.enabled = {opted_in}")

    inc = None
    try:
        from v9.accel import cutlass_bootstrap as cb
        inc = cb.already_installed()
        if inc is not None:
            print(f"  CUTLASS source: {inc.parent}  (header found)")
        else:
            print(f"  CUTLASS source: NOT FOUND "
                  f"(default target {cb.default_install_dir()})")
    except Exception as exc:  # noqa: BLE001
        print(f"  CUTLASS source: probe error: {exc}")

    sm = getattr(getattr(profile, "gpu", None), "sm_arch", None)
    if not nvcc:
        print("  -> blocked: install the CUDA Toolkit so nvcc is on PATH.")
    elif inc is None:
        print(f"  -> next step: `v9 cutlass bootstrap` clones CUTLASS to an "
              f"off-OneDrive dir and targets sm_{sm or '?'}.")
    elif not opted_in:
        print("  -> next step: set acceleration.cutlass.enabled: true in "
              "config.yaml to add it to the recipe (toolchain is ready).")
    else:
        print("  -> READY: CUTLASS is installed, opted-in, and part of the "
              "recipe. Prove it with `v9 benchmark`.")


def _report_cache(profile: Any, config: Config) -> None:
    try:
        from v9.build.cache import EngineCacheManager
        trt = config.get("tensorrt", {}) or {}
        max_gb = float(trt.get("cache_size_gb", 200))
        cache = EngineCacheManager(hardware=profile,
                                   max_disk_bytes=int(max_gb * 1024 ** 3))
        stats = cache.stats()
        print(f"  cache_dir: {stats['cache_dir']}")
        print(f"  db_path:   {stats['db_path']}")
        print(f"  used:      {stats['total_gb']:.2f} / "
              f"{stats['max_disk_gb']:.0f} GB  "
              f"(driver={stats['driver_ver']}, sm={stats['sm_arch']})")
        per = stats.get("per_app", {})
        if per:
            for app, d in per.items():
                print(f"    {app:10s} {d['count']} engines, {d['gb']:.2f} GB")
        else:
            print("    (empty — no engines compiled yet)")
    except Exception as exc:  # noqa: BLE001
        print(f"  cache unavailable: {exc}")


if __name__ == "__main__":
    raise SystemExit(run())
