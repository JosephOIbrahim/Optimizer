"""
method — the five moves, driven end to end. Not a new optimizer: it calls the
husk harness for numbers and V12 for the verdict. Baseline and gate both go
through V12's own Objective, so nothing here can call a change a win on its own.

    baseline    preflight (must be CLEAR) -> 6-rep render -> record in V12
    candidate   REFUSES - a candidate cannot be gated against a baseline from
                another session. The refusal lives inside latest_baseline()
                too, so it cannot be bypassed by importing the helper.
    ledger      what V12 has kept, and how much was measured
    preflight   is the GPU clear

Render objective only (husk). Defaults read E:\\OptimizerV12\\harness_config.json
for the scene; override with --usd / --frame / --settings.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harness import common  # noqa: E402

sys.path.insert(0, str(common.V12_DIR))
from v12.objective import render_objective  # noqa: E402

RENDERPY = r"E:\OptimizerV12\venv\Scripts\python.exe"
if not Path(RENDERPY).exists():
    RENDERPY = common.V12_PYTHON  # still renders; ssim guardrail needs PIL/numpy though
DEFAULT_USD = r"E:\OptimizerV12\scenes\ovoid_hero_v002.usd"
DEFAULT_SETTINGS = "/Render/OVOID_CPU_RENDER_V013"
DEFAULT_FRAME = 503


def scene_defaults(args):
    cfg = common.load_config()
    usd = args.usd or cfg.get("usd") or DEFAULT_USD
    return usd, args.frame, args.settings


def run_render(label: str, usd: str, frame: int, settings: str) -> dict:
    cmd = [RENDERPY, str(HERE / "husk_frame.py"), "--usd", usd, "--frame", str(frame),
           "--settings", settings, "--reps", "6", "--label", label, "--skip-preflight"]
    print(f"\n--- rendering 6 reps, label '{label}' (first is warm-up) ---\n", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"render failed (exit {r.returncode})")
    files = sorted(glob.glob(str(common.MEASURE_ROOT / "render_*.json")))
    if not files:
        raise SystemExit("no render measurement was written")
    d = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    d["_path"] = files[-1]
    return d


def latest_baseline() -> dict:
    """REFUSES to hand over a baseline from another session.

    This used to return the most recent baseline on disk. Two CLI entry
    points called refuse_stale_comparison() before calling it, which made the
    refusal a DISCIPLINE rather than a guarantee: the helper was still
    importable, so any caller — a script, a notebook, another agent — could
    read a stale baseline and gate a candidate against it without ever
    meeting the refusal.

    The refusal has to live inside the function that does the reading, or it
    is only a suggestion. So the reading itself refuses: there is no way to
    get a baseline out of this module any more, which is the point. The only
    way to compare two arms is harness/levers.py, which measures them
    alternately in one session.

    Why a baseline from another session cannot be gated against: a file on
    disk may have been written in an earlier session, under a different set
    of resident apps, at a different clock state. Comparing a candidate
    measured NOW against a baseline measured THEN charges that drift to the
    candidate, and no statistical test can tell the difference — that is what
    "independent series" means.

    On 2026-11-03 this produced a verdict of ADMIT +12.34% (p=1.9e-05) for a
    thread-count change that was worth nothing.
    """
    refuse_stale_comparison(
        "BASELINE",
        "REFUSING TO HAND OVER A BASELINE FROM ANOTHER SESSION.")


def refuse_stale_comparison(what: str, headline: Optional[str] = None) -> None:
    """Refuse a comparison against a baseline from another session.

    The refusal is the fix. Silently comparing a fresh candidate against a
    stale baseline is how a +12% ghost gets into the ledger, and the ghost is
    indistinguishable from a real win once it is written down.

    `headline` exists so that a caller which is not comparing two things —
    one that is merely READING a baseline — can say so in its own words
    while refusing in exactly the same way.
    """
    head = (headline or
            f"REFUSING TO COMPARE A {what} AGAINST A BASELINE FROM ANOTHER SESSION.")
    raise SystemExit(
        f"\n  {head}\n"
        f"\n"
        f"  The baseline on disk may have been measured hours ago, under a different\n"
        f"  set of resident apps and a different machine state. Any drift between\n"
        f"  the two sessions is indistinguishable from the effect you are trying to\n"
        f"  measure, and no statistical gate can separate them.\n"
        f"\n"
        f"  This is not hypothetical. On 2026-11-03 a thread-count sweep was\n"
        f"  admitted at +12.34% (p=1.9e-05) against a baseline from an earlier\n"
        f"  session. Re-timed the same way, the same change was worth nothing.\n"
        f"\n"
        f"  Measure both arms together instead:\n"
        f"\n"
        f"    Optimize-The-4090.bat --request render|shot|compute\n"
        f"\n"
        f"  or, for a change you apply by hand, the manual lever:\n"
        f"\n"
        f"    python harness\\optimize.py --request render --levers manual\n"
        f"\n"
        f"  Both arms are then measured alternately in one session, and a baseline\n"
        f"  that drifts more than 5% during the run refuses the verdict outright.")


def preflight_clear() -> bool:
    r = subprocess.run([common.V12_PYTHON, "-m", "v12", "preflight"], cwd=str(common.V12_DIR))
    return r.returncode == 0


def cmd_preflight(args) -> int:
    return 0 if preflight_clear() else 1


def cmd_baseline(args) -> int:
    usd, frame, settings = scene_defaults(args)
    print("MOVE 01 — BASELINE.  Measure where you stand, then record it.")
    if not args.force and not preflight_clear():
        print("\nGPU is not clear. Close the heavy apps and run this again "
              "(or pass --force to measure anyway, knowing the number is suspect).")
        return 1
    d = run_render("baseline", usd, frame, settings)
    rs = d["render_s"]
    ssim = d.get("ssim") or []
    cmd = [common.V12_PYTHON, "-m", "v12", "baseline", "--objective", "render",
           "--record", common.series(rs), "--harness", "husk_frame"]
    if ssim:
        cmd += ["--guardrail", f"ssim={common.series(ssim)}"]
    print("\n--- recording baseline in V12 ---")
    r = subprocess.run(cmd, cwd=str(common.V12_DIR))
    if r.returncode == 0:
        print(f"\nBaseline is {d['summary']['mean']:.2f} s (render only), cv "
              f"{d['summary'].get('cv', 0):.2%}. V12 now has something to gate against.")
    return r.returncode


def cmd_candidate(args) -> int:
    """MOVE 03/04 — measure a change and gate it. REFUSED as written.

    The original implementation compared a candidate measured NOW against a
    baseline read from DISK, which was usually measured in an earlier session.
    Any drift between the two sessions was charged to the candidate.

    On 2026-11-03 that produced ADMIT +12.34% (p=1.9e-05) for a
    thread-count change worth nothing. The Field Notes call it "the move that
    evaporated"; the verdict sat in the ledger.

    So this path no longer produces a verdict. It points at the one that
    does: harness/levers.py, where both arms are measured alternately in one
    session and a drifting control refuses the result.
    """
    refuse_stale_comparison("CANDIDATE")


def cmd_ledger(args) -> int:
    return subprocess.run([common.V12_PYTHON, "-m", "v12", "ledger"], cwd=str(common.V12_DIR)).returncode


def build_parser():
    p = argparse.ArgumentParser(prog="method", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usd", default="")
    p.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    p.add_argument("--settings", default=DEFAULT_SETTINGS)
    p.add_argument("--force", action="store_true", help="measure even if preflight is not CLEAR")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline").set_defaults(func=cmd_baseline)
    c = sub.add_parser("candidate")
    c.add_argument("--label", required=True)
    c.set_defaults(func=cmd_candidate)
    sub.add_parser("ledger").set_defaults(func=cmd_ledger)
    sub.add_parser("preflight").set_defaults(func=cmd_preflight)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    raise SystemExit(a.func(a))
