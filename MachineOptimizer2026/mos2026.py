"""
Frozen entry point for MachineOptimizer 2026.

Why this file exists rather than pointing PyInstaller at `optimizer/orchestrator.py`
directly: freezing this tool has two failure modes that are both silent, and a
silent failure in a tool that rewrites the registry is the failure this project
exists to prevent.

  1. **Zero agents.** `orchestrator.discover_agents()` finds agents with
     `pkgutil.iter_modules` + `importlib.import_module` — fully dynamic. PyInstaller's
     static analysis cannot see through that. A frozen build missing
     `--collect-submodules optimizer` loads *no agents at all*, prints an empty
     plan, exits 0, and looks exactly like a machine that needed no changes. This
     entry point counts them and refuses to run if the count is zero.

  2. **Config not found.** `load_config()` reads `ROOT/config.toml`, where ROOT is
     derived from `__file__`. Inside a one-file bundle `__file__` lives in a temp
     extraction directory, so config.toml would never be found and the tool would
     quietly run on built-in DEFAULTS instead of the machine's tuned settings.
     If a config.toml sits beside the .exe, it is used.

Both checks are cheap and both turn a silent wrong answer into a loud refusal.

Build with:
    Build-MachineOptimizer-EXE.bat
or directly:
    pyinstaller --onefile --uac-admin --name MachineOptimizer2026 ^
        --collect-submodules optimizer --paths . mos2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _redirect_config_for_frozen() -> None:
    """Point the orchestrator's ROOT at the .exe's own folder when a config.toml
    is there. Done before main() runs, because load_config() resolves ROOT at call
    time from the module global."""
    if not getattr(sys, "frozen", False):
        return
    try:
        from optimizer import orchestrator  # type: ignore
    except Exception:
        return
    beside_exe = Path(sys.executable).resolve().parent
    if (beside_exe / "config.toml").exists():
        orchestrator.ROOT = beside_exe


def _assert_agents_present() -> int:
    """A frozen build with no agents is broken, not clean. Say so and stop."""
    try:
        from optimizer.orchestrator import discover_agents  # type: ignore
    except Exception as e:
        print(f"[FATAL] cannot import the optimizer package: {type(e).__name__}: {e}")
        return 3
    n = len(discover_agents())
    if n == 0:
        print("[FATAL] 0 agents loaded.")
        print("        A frozen build that finds no agents would print an empty plan")
        print("        and exit 0 -- indistinguishable from a machine needing nothing.")
        print("        Rebuild with:  pyinstaller ... --collect-submodules optimizer")
        return 4
    return 0


def run() -> int:
    if getattr(sys, "frozen", False):
        _redirect_config_for_frozen()
        rc = _assert_agents_present()
        if rc:
            return rc

    from optimizer.orchestrator import main  # type: ignore
    return main()


if __name__ == "__main__":
    sys.exit(run())
