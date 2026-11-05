"""Crucible C1.4 (essentials) — V7/V8 orchestrator integration smokes.

Verifies:
  * V7 baseline is still runnable (rollback path is real).
  * V8 --once smoke run exits 0 (Forge-3 done-criteria).
  * V8 specialist registration: 4 V8 specialists are present and the V7
    ue5_engineer/houdini_engineer experts are disabled per
    `experts_v8_overrides`.
  * V7 enum doesn't leak: importing V7 alone does NOT see the V8-only
    OptimizationAction members (Forge-3's _extend_enum trick is V8-side
    only, so a fresh V7 launch keeps a clean enum).

The smoke runs use `C:\\Python314\\python.exe` because the local venv is
known-broken (pyvenv.cfg points to a missing miniconda interpreter --
documented in `_handoff_forge3.md` open issue #1).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PY = r"C:\Python314\python.exe"
SMOKE_TIMEOUT_S = 180  # --once should be well under this


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=SMOKE_TIMEOUT_S,
    )


@pytest.mark.skipif(not Path(PY).exists(), reason="C:\\Python314\\python.exe not present")
def test_v7_baseline_smoke_exit_zero() -> None:
    """V7 regression check: gpu_optimizer_v6.py --once must still run clean."""
    proc = _run(
        [PY, str(REPO_ROOT / "core" / "gpu_optimizer_v6.py"),
         "--once", "--config", "config.yaml"],
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"V7 baseline failed: rc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


@pytest.mark.skipif(not Path(PY).exists(), reason="C:\\Python314\\python.exe not present")
def test_v8_smoke_exit_zero() -> None:
    """V8 done-criteria smoke: gpu_optimizer_v8.py --once must exit 0."""
    proc = _run(
        [PY, str(REPO_ROOT / "core" / "gpu_optimizer_v8.py"),
         "--once", "--config", "config.yaml"],
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"V8 smoke failed: rc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


@pytest.mark.skipif(not Path(PY).exists(), reason="C:\\Python314\\python.exe not present")
def test_v8_registers_four_specialists_and_disables_v7_engineers() -> None:
    """Spawn a subprocess that imports V8, instantiates the orchestrator
    against config.yaml, and prints the registered ExpertType names. We then
    assert the 4 V8 specialists are present and the two superseded V7
    engineers are absent.

    Subprocess isolation matters: V8's _extend_enum mutates the V7 enum in
    place, and we want to make sure we are testing what `--v8` actually does,
    not whatever this pytest process happens to have already imported.
    """
    script = textwrap.dedent(
        """
        import sys, json
        from pathlib import Path
        sys.path.insert(0, r"%(repo)s")
        from core.gpu_optimizer_v8 import OptimizerOrchestrator, ExpertType
        orch = OptimizerOrchestrator(r"%(cfg)s")
        names = sorted(et.name for et in orch.experts.keys())
        try:
            orch.shutdown()
        except Exception:
            pass
        print("EXPERTS=" + json.dumps(names))
        """
    ) % {"repo": str(REPO_ROOT), "cfg": str(REPO_ROOT / "config.yaml")}

    proc = subprocess.run(
        [PY, "-c", script],
        capture_output=True, text=True, timeout=SMOKE_TIMEOUT_S,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"Orchestrator init failed: rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    out = proc.stdout
    marker = "EXPERTS="
    idx = out.rfind(marker)
    assert idx >= 0, f"Did not find EXPERTS=... marker in stdout:\n{out}"
    import json
    names = set(json.loads(out[idx + len(marker):].splitlines()[0]))

    expected_v8 = {
        "COMFYUI_SPECIALIST",
        "HOUDINI_SPECIALIST_V8",
        "UE5_SPECIALIST_V8",
        "CROSS_APP_COORDINATOR",
    }
    missing = expected_v8 - names
    assert not missing, f"Missing V8 specialists: {missing}; saw {names}"

    # The two engineers superseded by V8 specialists must not be registered.
    assert "UE5_ENGINEER" not in names, (
        f"V7 UE5_ENGINEER must be disabled in V8; saw {names}"
    )
    assert "HOUDINI_ENGINEER" not in names, (
        f"V7 HOUDINI_ENGINEER must be disabled in V8; saw {names}"
    )


@pytest.mark.skipif(not Path(PY).exists(), reason="C:\\Python314\\python.exe not present")
def test_v7_enum_is_clean_when_v8_not_imported() -> None:
    """Forge-3's `_extend_enum` mutates V7's `OptimizationAction` enum *only
    when V8 is imported*. A fresh V7-only Python process must therefore have
    a minimal enum (no COMPILE_TRT_ENGINE, no INSTALL_CUSTOM_NODE, etc.).

    This guards the rollback path: if a future change moved `_extend_enum`
    into `gpu_optimizer_v6.py`, V7 launched standalone would suddenly carry
    V8-only members and break the frozen-surface contract.
    """
    script = textwrap.dedent(
        """
        import sys, json
        sys.path.insert(0, r"%(repo)s")
        # Import V7 only -- explicitly NOT the V8 module.
        from core.gpu_optimizer_v6 import OptimizationAction, ExpertType
        out = {
            "actions": sorted(m.name for m in OptimizationAction),
            "experts": sorted(m.name for m in ExpertType),
        }
        print("ENUMS=" + json.dumps(out))
        """
    ) % {"repo": str(REPO_ROOT)}

    proc = subprocess.run(
        [PY, "-c", script],
        capture_output=True, text=True, timeout=SMOKE_TIMEOUT_S,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"V7-only import subprocess failed: rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    import json
    marker = "ENUMS="
    idx = proc.stdout.rfind(marker)
    assert idx >= 0
    payload = json.loads(proc.stdout[idx + len(marker):].splitlines()[0])

    # V8-only OptimizationAction members must NOT be in the V7-only view.
    v8_only_actions = {
        "COMPILE_TRT_ENGINE",
        "INSTALL_CUSTOM_NODE",
        "INSTALL_PLUGIN",
        "ENABLE_BACKEND",
        "SET_LAUNCH_FLAG",
        "EVICT_ENGINE_CACHE",
        "WIRE_BACKEND_PATCH",
    }
    leaked_actions = v8_only_actions & set(payload["actions"])
    assert not leaked_actions, (
        f"V8-only OptimizationAction members leaked into V7-only enum view: "
        f"{leaked_actions}. Forge-3's _extend_enum must run on V8 import only."
    )

    v8_only_experts = {
        "COMFYUI_SPECIALIST",
        "HOUDINI_SPECIALIST_V8",
        "UE5_SPECIALIST_V8",
        "CROSS_APP_COORDINATOR",
    }
    leaked_experts = v8_only_experts & set(payload["experts"])
    assert not leaked_experts, (
        f"V8-only ExpertType members leaked into V7-only enum view: "
        f"{leaked_experts}"
    )

    # Sanity: V7 base enum members are still there.
    assert "SET_ENV_VAR" in payload["actions"]
    assert "GPU_SPECIALIST" in payload["experts"]
