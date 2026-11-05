"""
lever_run — RETIRED. It refuses, and the refusal cannot be bypassed.

    python lever_run.py --objective render --levers husk_threads

This module used to apply ONE change, measure it, revert it, and score it
against the latest recorded baseline measurement. The design was "read the
baseline from disk, measure the candidate now, compare" — and the middle
step is the problem: the two numbers can be from different sessions, and
the difference between them is then indistinguishable from the effect being
measured.

On 2026-11-03 that produced the ledger's most misleading verdict: a
thread-count sweep admitted at +12.34% (p=1.9e-05) against a baseline
measured 48 minutes earlier, under different resident processes. Re-timed
the same way, the change was worth nothing.

So the module refuses. The refusal is not only at the door: the baseline-
reading helper itself refuses, and the verdict helper has been removed, so
importing them cannot be used to get around it. The only way to compare two
arms is harness/levers.py, which measures them alternately in one session
and refuses a verdict when the control drifts more than 5%.

Levers (all of them now live in harness/levers.py)
  husk_threads   husk -j N for N in --threads (default 8,16,24,48).
  lock_clocks    nvidia-smi -lgc MIN,MAX per entry in --clocks (default 2520,2745).
  usd_variants   render alternate USDs (from variant_usd.py).
  p2_off         CUDA Force P2 State = Off via NVIDIA Profile Inspector.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import common  # noqa: E402

sys.path.insert(0, str(common.V12_DIR))
from v12.objective import render_objective  # noqa: E402

HERE = Path(__file__).resolve().parent
NIP_P2_OFF = HERE / "p2_off.nip"
NIP_P2_ON = HERE / "p2_on.nip"
NIP_TEMPLATE = """<?xml version="1.0" encoding="utf-16"?>
<ArrayOfProfile xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <Profile>
    <ProfileName>Base Profile</ProfileName>
    <Executeables />
    <Settings>
      <ProfileSetting>
        <SettingNameInfo>CUDA - Force P2 State</SettingNameInfo>
        <SettingID>1343646814</SettingID>
        <SettingValue>{value}</SettingValue>
        <ValueType>Dword</ValueType>
      </ProfileSetting>
    </Settings>
  </Profile>
</ArrayOfProfile>
"""


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def latest_baseline_measurement(objective: str) -> dict:
    """REFUSES to hand over a baseline from another session.

    This used to return the most recent baseline on disk. main() called
    refuse_stale_comparison() before calling it, which made the refusal a
    DISCIPLINE rather than a guarantee: the helper was still importable, so
    any caller — a script, a notebook, another agent — could read a stale
    baseline and gate a candidate against it without ever meeting the
    refusal.

    The refusal has to live inside the function that does the reading, or it
    is only a suggestion. So the reading itself refuses.

    On 2026-11-03 this module produced the ledger's most misleading verdict:
    a thread-count sweep admitted at +12.34% (p=1.9e-05) against a baseline
    measured 48 minutes earlier, under different resident processes. Re-timed
    the same way, the change was worth nothing.
    """
    refuse_stale_comparison(
        "REFUSING TO HAND OVER A BASELINE MEASUREMENT FROM ANOTHER SESSION.")


def refuse_stale_comparison(headline: Optional[str] = None) -> None:
    """Refuse to gate a candidate against a baseline from another session.

    This module's whole design was "read the baseline from disk, measure the
    candidate now, compare". The middle step is the problem: the two numbers
    can be from different sessions, and the difference between them is then
    indistinguishable from the effect being measured.

    No statistical gate can catch it — Welch's t-test on two INDEPENDENT
    series attributes the drift to the candidate, which is exactly what
    "independent" means. The fix is to measure both arms alternately in one
    session, which is what harness/levers.py does.

    `headline` lets a caller that is merely READING a baseline refuse in the
    same words while saying what it was doing. It exists because the refusal
    now lives inside the reading helpers, not only in front of them.
    """
    head = (headline or
            "REFUSING TO GATE A CANDIDATE AGAINST A BASELINE FROM ANOTHER SESSION.")
    raise SystemExit(
        f"\n  {head}\n"
        "\n"
        "  This module compared a candidate measured now against a baseline read\n"
        "  from disk, which was usually measured hours earlier under different\n"
        "  resident apps. Drift between the two sessions is indistinguishable\n"
        "  from the effect under test.\n"
        "\n"
        "  On 2026-11-03 this produced ADMIT +12.34% (p=1.9e-05) for a change\n"
        "  worth nothing. That verdict is still in the ledger.\n"
        "\n"
        "  Measure both arms together instead:\n"
        "\n"
        "    Optimize-The-4090.bat --request render|shot|compute\n"
        "    python harness\\optimize.py --request render --levers manual\n"
        "\n"
        "  Both arms are measured alternately in one session, and a baseline that\n"
        "  drifts more than 5% during the run refuses the verdict outright.")


def run_harness(base: dict, label: str, extra: list) -> dict:
    cmd = [sys.executable, str(HERE / "husk_frame.py"), "--usd", base["usd"], "--frame", str(base["frame"]),
           "--settings", base.get("settings", ""), "--reps", "6", "--label", label,
           "--skip-preflight"] + extra
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise RuntimeError(f"harness failed for {label}")
    files = sorted(glob.glob(str(common.MEASURE_ROOT / "render_*.json")))
    d = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    d["_path"] = files[-1]
    return d


def v12_dry_run(key: str, cand: dict) -> str:
    cmd = [common.V12_PYTHON, "-m", "v12", "run", "--objective", "render",
           "--candidate-for", f"{key}={common.series(cand['render_s'])}",
           "--guardrail-for", f"{key}:ssim={common.series(cand.get('ssim') or [])}"]
    r = subprocess.run(cmd, cwd=str(common.V12_DIR), capture_output=True, text=True)
    return r.stdout[-2500:]


def smi(*args) -> str:
    r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", default="render", choices=["render"])
    ap.add_argument("--levers", default="husk_threads")
    ap.add_argument("--threads", default="8,16,24,48")
    ap.add_argument("--clocks", default="2520,2745")
    ap.add_argument("--npi", default="", help="path to nvidiaProfileInspector.exe (p2_off)")
    ap.add_argument("--usd-variants", default="", help="usd_variants lever: label=path;label=path (variant USDs from variant_usd.py)")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    if not args.skip_preflight and not common.preflight():
        return 1

    # REFUSED. This module's design was "read the baseline from disk, measure
    # the candidate now, compare" — and the middle step is the problem. The
    # two numbers can be from different sessions, and the difference between
    # them is then indistinguishable from the effect under test.
    #
    # On 2026-11-03 this produced the ledger's most misleading verdict: a
    # thread-count sweep admitted at +12.34% (p=1.9e-05) against a baseline
    # measured 48 minutes earlier under different resident processes. Re-timed
    # the same way, the change was worth nothing.
    #
    # The levers themselves are not lost — they are implemented in
    # harness/levers.py, which measures both arms alternately in one session
    # and refuses a verdict when the control drifts more than 5%.
    refuse_stale_comparison()

    # Everything below this line used to exist and was unreachable. It is
    # gone rather than commented out, because a reader (or a script) could
    # have copied it back into service without noticing that the baseline it
    # gated against came from another session — which is the exact failure
    # this module was retired for. The verdict helper it called has been
    # removed for the same reason: it computed an ADMIT/DENY from any two
    # dicts, so it was a bypass around the refusal rather than a gate.
    #
    # The levers themselves are not lost — they are implemented in
    # harness/levers.py, which measures both arms alternately in one session
    # and refuses a verdict when the control drifts more than 5%.
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
