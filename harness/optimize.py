"""
optimize — the next-generation optimizer: pick a LEVER, measure it honestly.

What this is
------------
The previous generation measured a lever against a baseline taken at some
earlier time, on a machine in some other state, and called the difference a
result. On this box that produced a +12.34% "ADMIT" (p=1.9e-05) for a
thread-count change that was worth nothing when the baseline was re-timed the
same way as the candidate. The Field Notes (No. 01) call this "the move that
evaporated", and they are describing this repo's own ledger.

This module fixes the METHOD, not just the numbers:

  * baseline and candidate are measured ALTERNATELY, in one session
    (harness/paired.py), so drift cancels instead of being charged to the
    candidate;
  * the comparison is a PAIRED t-test, which is the right test for that data
    (v12/objective.py: paired_t_test);
  * a baseline that drifts more than 5% across the run REFUSES the verdict —
    a verdict computed from an unstable control is not a verdict;
  * the workload fingerprint is real — scene, frame count, samples, settings
    prim, torch build — so a 3-frame shot is never compared against a
    1-frame baseline;
  * the ssim guardrail compares the CANDIDATE's frames against the BASELINE's,
    so a change that alters the picture is caught;
  * before any lever is proposed, the tool asks what is actually binding
    (harness/constraint.py) — on this box the render path is not power-bound
    at all, so a power lever would be a waste of an afternoon;
  * the record keeps the RAW paired series, not just the summary statistics,
    so the effect, the p-value and the drift can be recomputed by a reader
    instead of taken on trust. A record that cannot be checked is a claim;
    this repo already keeps retracted verdicts and superseded baselines on
    disk for exactly that reason.

The lever implementations live in harness/levers.py — one implementation,
used by every front door, so there is no second way to measure a lever badly.

What it does NOT do
-------------------
It does not apply anything to the machine without a verdict, and it does not
keep anything applied. The power ceiling can be applied and restored exactly
(the getter is trustworthy and the setter round-trips), but the operator
decides whether to keep it — and when the restore FAILS, the run says so
instead of reporting that it left nothing applied. The `manual` lever is the
one exception by design: it asks the operator to toggle their own change and
never reverts it.

The levers
----------
    frames_per_process  render a shot in one process, not N  (scene/launch)
    aov_variant         render with fewer AOVs                (scene)
    power_ceiling       raise/lower the card's power ceiling  (SYSTEM tier)
    husk_threads        render with a fixed thread count      (scene/launch)
    manual              a change the operator applies by hand

Usage
-----
    python harness\\optimize.py --request render --usd <scene> --frames 3
    python harness\\optimize.py --request compute
    python harness\\optimize.py --request render --levers power_ceiling
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harness import common, constraint, levers  # noqa: E402

sys.path.insert(0, str(common.V12_DIR))
from v12.objective import (Objective, gemm_objective,  # noqa: E402
                           render_objective, shot_objective)


def machine_spec() -> Dict[str, Any]:
    """What this machine is, read live. Never hardcoded."""
    return levers.machine_spec()


def spec_lines(spec: Dict[str, Any]) -> List[str]:
    out = []
    if spec.get("name"):
        lo = spec.get("power_limit_min_w") or 0
        hi = spec.get("power_limit_max_w") or 0
        cur = spec.get("power_limit_w") or 0
        room = f"  (allowed {lo:.0f}-{hi:.0f} W)" if hi else ""
        head = f"  {hi - cur:.0f} W of headroom above the current ceiling" if hi > cur else ""
        out.append(f"  {spec['name']}  {spec.get('vram_gib', 0):.0f} GiB  "
                   f"{spec.get('sm_count', 0)} SM  driver {spec.get('driver', '?')}")
        out.append(f"  power ceiling {cur:.0f} W{room}{head}")
    else:
        out.append(f"  (could not read the GPU: {spec.get('error')})")
    return out


def objective_for(request: str, resolution: str = "1024x429", samples: int = 128,
                  frames: int = 3) -> Objective:
    """The objective that matches what the operator asked to optimize.

    A request is a workload. Each one has a different binding constraint, so
    each gets its own objective and its own levers:
      render   one hero frame          -> render_s
      shot     a sequence of frames     -> shot_s
      compute  tensor throughput        -> tflops
    """
    if request == "shot":
        return shot_objective(resolution=resolution, samples=samples, frames=frames)
    if request == "compute":
        return gemm_objective(n=8192, precision="fp16")
    return render_objective(resolution=resolution, samples=samples)


def plan_levers(request: str, obj: Objective, spec: Dict[str, Any],
                verdict: Optional[constraint.ConstraintVerdict] = None) -> List[str]:
    """Which levers are worth trying for THIS request on THIS machine.

    The point of the Field Notes is that "optimize the 4090" is not one
    question. A workload that never reaches the card's ceiling cannot be
    helped by a power lever, and the tool should say so rather than spend an
    afternoon proving it.

    Each request gets the levers whose METRIC is the one its objective
    scores, because a verdict computed from the wrong quantity is not a
    verdict at all:
      render   scores render_s (one frame's shading) -> AOV variants
      shot     scores shot_s  (the whole shot)       -> frames per process
      compute  scores tflops                         -> power ceiling

    The metric is not a preference here, it is a REQUIREMENT, and the plan
    is filtered by it: a lever that measures a different quantity than the
    objective scores would produce a number about the wrong thing, with the
    sign inverted by the two metrics' opposite directions. So the plan never
    proposes a lever the runner would refuse.
    """
    planned: List[str] = []
    if request == "compute":
        # Tensor work is the workload that reaches the ceiling on this card.
        planned.append("power_ceiling")
    elif request == "render":
        # One frame: the setup tax cannot be amortized across frames that
        # do not exist, so the honest lever is scene-side — shade fewer AOVs.
        planned.append("aov_variant")
    elif request == "shot":
        planned.append("frames_per_process")
    # A power lever only makes sense if the measurement says the card is
    # pinned. Without that evidence it is a guess, and the notes are against
    # guesses.
    if verdict and verdict.constraint is constraint.Constraint.POWER \
            and request in ("render", "shot"):
        # The power lever measures TFLOPS, so under a render or shot
        # objective it cannot answer the question "would raising the ceiling
        # make THIS workload faster" — it would answer it about a GEMM.
        # The filter below drops it, and the operator is told why by the
        # runner's refusal if they ask for it explicitly.
        planned.append("power_ceiling")
    # Never propose a lever the runner would refuse: the metric has to be
    # the one the objective scores.
    return [n for n in planned if levers.metric_mismatch(n, obj) is None]


def _profile(args, spec: Dict[str, Any]) -> int:
    """Report what is limiting the machine right now, without running a lever.

    This is the honest first move. The Field Notes' first finding was that the
    4090 was innocent for the render workload — 100 W of a 450 W budget, 0% at
    the cap — and every lever aimed at the card was aimed at a constraint that
    was not binding. Profiling first is how that finding gets applied instead
    of just remembered.
    """
    print("\n  PROFILING")
    print("  " + "-" * 68)
    try:
        r = levers._run_gemm()
        verdict = constraint.classify([r["throttle"]], util_pct=0.0)
        print("  compute (fp16 GEMM):")
        print("    " + verdict.summary().replace("\n", "\n    "))
    except Exception as e:
        print(f"    (could not run the compute probe: {type(e).__name__}: {e})")
    print()
    print("  the card's own headroom:")
    cur = spec.get("power_limit_w") or 0.0
    hi = spec.get("power_limit_max_w") or 0.0
    if cur and hi:
        print(f"    ceiling {cur:.0f} W, this card accepts up to {hi:.0f} W "
              f"({hi - cur:.0f} W more)")
        print("    whether that is worth taking depends entirely on the workload: "
              "measured on this box,")
        print("    fp16 GEMM gained +8.50% (433 W -> 548 W drawn) while a Karma")
        print("    render gained nothing (104 W drawn, never near the cap).")
    return 0


def result_block(r: levers.Result) -> Dict[str, Any]:
    """One lever's block in the record.

    The summary statistics and the RAW SERIES both belong here. The summary
    says what the harness concluded; the series are what a reader needs to
    check that conclusion — the effect, the p-value and the drift are all
    recomputable from the three lists, and a record that carries only the
    summary is a claim rather than a measurement.

    A nested `detail` block is dropped: it would put a copy of the record
    inside the record, which is how records grow into something nobody can
    read.

    `notes` and `applied` are here because they are EVIDENCE, not decoration.
    The notes carry the harness's explanation of what it measured and why
    (the setup tax on the shot path, the variant name on the AOV path), and
    `applied` is the standing assertion that the machine was left as it was
    found. Both used to reach the file — the 11-04 records carry
    "applied": false and a setup_tax_s — and both were dropped when this
    block was rewritten around the series, which left a record that said
    WHAT happened and not WHY, or not even that nothing was left changed.
    A field that is dropped here is a field nobody can check.
    """
    block: Dict[str, Any] = {k: v for k, v in r.detail.items() if k != "detail"}
    block.update({
        "ok": r.ok, "why": r.why, "effect": r.effect,
        "p_value": r.p_value, "drift": r.drift,
        "contention_events": r.contention_events,
        "applied": r.applied,
        "restore_error": r.restore_error,
        "notes": list(r.notes),
    })
    return block


def build_record(request: str, obj: Objective, spec: Dict[str, Any],
                 results: List[Tuple[str, levers.Result]]) -> Dict[str, Any]:
    """The record of one optimize run.

    Kept separate from `main` so the shape of the record can be tested
    without running a lever, and so the record is built in exactly one place.
    """
    return {
        "request": request, "objective": obj.workload.name,
        "spec": spec,
        "results": {n: result_block(r) for n, r in results},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", default="render",
                    choices=["render", "shot", "compute", "profile"],
                    help="what you want to optimize")
    ap.add_argument("--usd", default=r"E:\OptimizerV12\scenes\ovoid_hero_v002.usd")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--start-frame", type=int, default=503)
    ap.add_argument("--settings", default="/Render/OVOID_CPU_RENDER_V013")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--res-scale", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3,
                    help="repetitions per arm; the shot objective needs 3 to reach "
                         "MEASURED, the GEMM objective 5")
    ap.add_argument("--warmup", type=int, default=1,
                    help="warm-up repetitions of each arm, run and DISCARD before "
                         "measuring. The first repetition pays for a cold page "
                         "cache, a cold shader cache and a cold BVH — the machine "
                         "being cold, not the workload — and on the shot path that "
                         "read as drift and made the gate refuse the campaign")
    ap.add_argument("--levers", default="", help="comma-separated lever names to run")
    ap.add_argument("--variant", default="beauty_only",
                    choices=["beauty_only", "no_crypto"],
                    help="which AOV-reduced scene the render request tests against")
    ap.add_argument("--threads", type=int, default=0,
                    help="thread count for the husk_threads lever")
    ap.add_argument("--quiet-timeout", type=float, default=900.0,
                    help="seconds to wait for the GPU to go idle before measuring "
                         "(0 = do not wait). A contended measurement is worse than "
                         "a late one: the drift gate refuses it and the wait was "
                         "for nothing.")
    ap.add_argument("--probe", action="store_true",
                    help="only report what is binding; measure nothing")
    args = ap.parse_args()

    print("=" * 72)
    print("  OPTIMIZE THE 4090")
    print("=" * 72)
    spec = machine_spec()
    for line in spec_lines(spec):
        print(line)
    obj = objective_for(args.request, samples=args.samples, frames=args.frames)
    print(f"  request: {args.request}  ->  objective '{obj.workload.name}' "
          f"({obj.primary.name}, {obj.primary.direction.value})")

    if args.request in ("render", "shot") and not Path(args.usd).exists():
        print(f"\n  no such scene: {args.usd}")
        return 2

    if not common.preflight():
        print("\n  preflight BLOCKED — something that would change the machine "
              "while we measure it is running. Close it and try again. "
              "(preflight runs WITHOUT --quiet, so this is about a competing "
              "OPTIMIZER or an unreadable process table, not about ordinary "
              "apps holding GPU memory — a number taken while another process "
              "holds the card is caught by the contention check further down, "
              "which waits for the card to clear.)")
        return 1

    if args.request == "profile":
        return _profile(args, spec)

    # What is binding? For the compute request the GEMM telemetry answers it.
    verdict = None
    if args.request == "compute":
        r = levers._run_gemm()
        verdict = constraint.classify([r["throttle"]], util_pct=0.0)
        print("\n" + verdict.summary())
    else:
        print("\n  (the render path's constraint is measured as part of the lever "
              "run, where the per-frame times are available)")

    chosen = ([l.strip() for l in args.levers.split(",") if l.strip()]
              if args.levers else plan_levers(args.request, obj, spec, verdict))
    if not chosen:
        print("\n  no lever is worth trying for this request on this machine.")
        return 0

    print(f"\n  levers worth trying: {', '.join(chosen)}")
    if args.probe:
        return 0

    ctx = levers.Context.from_args(args)
    results = []
    for name in chosen:
        print("\n" + "-" * 72)
        print(f"  LEVER: {name}")
        print("-" * 72)
        results.append((name, levers.run(name, ctx, obj)))

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for name, r in results:
        print(r.summary())
    rec = common.write_measurement("optimize", build_record(
        args.request, obj, spec, results))
    print(f"\n  saved {rec}")
    print(closure_report(results, chosen))
    return 0


def closure_report(results: List[Tuple[str, levers.Result]],
                   chosen: List[str]) -> str:
    """The last thing the operator reads, and it has to be TRUE.

    The machine was left alone — UNLESS a restore failed, and until now the
    run said the same thing either way. A failed power-ceiling restore leaves
    the card at a limit the operator never chose, and telling them "nothing
    was left applied" at that moment is the worst thing the tool could say.

    The manual lever is the other case the old text got wrong: it asks the
    operator to toggle their own change and never reverts it, so "nothing was
    left applied" is false about the one thing the operator can act on.

    It is a function, not a print, so a test can hold the exact words the
    operator will read.
    """
    unrestored = [(n, r) for n, r in results if r.restore_error]
    if unrestored:
        lines = [""]
        for name, r in unrestored:
            lines.append(f"  !! {name}: {r.restore_error}")
            lines.append("  !! THE MACHINE WAS NOT LEFT AS YOU FOUND IT.")
            lines.append("  !! Set it back by hand, from an elevated shell:")
            lines.append(f"  !!     nvidia-smi -pl {r.detail.get('power_prior_w', 0):.0f}")
        lines.append("")
        return "\n".join(lines)
    if "manual" in chosen:
        return ("  nothing the tool applied was left applied — but the manual lever\n"
                "  does not revert YOUR change: it asks you to toggle it, and leaves\n"
                "  it wherever you left it. Put it back yourself if that is what you\n"
                "  want.")
    return "  nothing was left applied. What to keep is your call."


if __name__ == "__main__":
    raise SystemExit(main())
