"""CRUCIBLE — Integration smoke tests for Optimizer V9 CLI.

Drives the real CLI as a subprocess under THIS interpreter (the same
C:/Python314/python.exe the suite runs on), asserting the DESIGN_V9 §6 Mile-3
acceptance criteria:

  * `python -m v9.cli once`    exits 0 with a clean shutdown.
  * `python -m v9.cli doctor`  exits 0 and prints the DETECTED driver value
    (whatever hardware.detect() reports on this machine) — never a stale
    hardcoded literal.

`once` is run with --dry-run so the suite never mutates the registry or the
user environment (the live `once` persists env vars and writes HwSchMode).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(*cli_args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Invoke `python -m v9.cli <args>` from the repo root, warnings silenced."""
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "ignore"
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "v9.cli", *cli_args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_cli_help_exits_zero() -> None:
    proc = _run_cli("--help")
    assert proc.returncode == 0, proc.stderr


def test_cli_once_dry_run_exits_zero() -> None:
    """One cycle, non-mutating, clean exit. (--dry-run so the test rig is not
    altered; the §6 criterion is the exit code + clean shutdown.)"""
    proc = _run_cli("once", "--dry-run")
    assert proc.returncode == 0, (
        f"`v9.cli once --dry-run` exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    assert "clean shutdown" in proc.stdout.lower(), proc.stdout


def test_cli_doctor_exits_zero() -> None:
    proc = _run_cli("doctor")
    assert proc.returncode == 0, (
        f"`v9.cli doctor` exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def test_cli_doctor_prints_detected_driver_not_stale_literal() -> None:
    """doctor must print the DETECTED driver, and must NOT print either of the
    v8-era stale literals (596.36 / 581.80)."""
    from v9.hardware import detect

    detected = detect().gpu.driver_version
    proc = _run_cli("doctor")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout

    # never the stale literals
    assert "596.36" not in out, "doctor printed stale literal 596.36"
    assert "581.80" not in out, "doctor printed stale literal 581.80"

    if detected:
        assert detected in out, (
            f"doctor did not print the DETECTED driver {detected!r}.\n"
            f"STDOUT:\n{out}")
    else:
        # No GPU/driver on this box — doctor must say so honestly, still exit 0.
        assert "no GPU driver detected" in out or "driver ?" in out, out


def test_cli_doctor_reports_data_root_off_onedrive() -> None:
    proc = _run_cli("doctor")
    assert proc.returncode == 0
    out = proc.stdout
    assert "data_root" in out
    # The honest report must not warn about OneDrive coupling on this rig
    # (D:/ or %LOCALAPPDATA% should win); if it does, that's a §4.3 defect.
    assert "off-OneDrive" in out, out


def test_cli_no_subcommand_prints_help_exits_zero() -> None:
    proc = _run_cli()
    assert proc.returncode == 0, proc.stderr
    assert "once" in proc.stdout and "doctor" in proc.stdout
