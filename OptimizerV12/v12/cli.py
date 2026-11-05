"""
V12 — operator entry point.

Every subcommand answers a question an operator actually asks, and none of them
mutate anything without being told to twice: `run` is a dry run unless `--apply`
is passed, and `--apply` without an objective is refused by the Gate, not by a
prompt.

    status      what machine am I on, and what can it do
    preflight   is anything else running that would corrupt a measurement
    kernels     is the compile toolchain ready, and what is unreachable
    baseline    record / list measurement baselines (with guardrail series)
    plan        which candidate to measure next for an objective (read-only)
    run         the loop: observe -> model -> decide -> admit -> act -> remember
    ledger      what has been done, and how much of it was actually measured
    rollback    undo. Deliberately requires no evidence.
    migrate     Phase 0a — move MachineOptimizer2026's state off OneDrive

Run `python -m v12.cli <subcommand> --help` for any of them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import baseline as baseline_mod
from . import paths
from .ledger import open_ledger
from .objective import Objective


def _console_codepage() -> str:
    """The console's own active output codepage, or '' when there is no console.

    Used to decide whether the UTF-8 report text can actually be rendered. The
    launcher sets `chcp 65001`, but a direct `python -m v12` may not, and writing
    UTF-8 into a cp437 console produces mojibake rather than an error — so this
    asks first instead of assuming."""
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        return str(ctypes.windll.kernel32.GetConsoleOutputCP())
    except Exception:
        return ""


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _setup_console() -> None:
    """Make the report text render on whatever console this is.

    The launcher runs `chcp 65001`, but a direct `python -m v12` does not — and a
    cp437 console handed UTF-8 bytes does not fail. It prints mojibake, which is
    worse than either an error or a fallback, because it still looks like output.
    The reports are full of em dashes and arrows, so this is the common case, not
    an edge case.

    So: UTF-8 when the console is UTF-8, and otherwise the console's own codepage
    with replacement — where the worst case is a `?` where an em dash was.
    """
    cp = _console_codepage()
    if not cp or cp == "0":
        return
    target = "utf-8" if cp == "65001" else f"cp{cp}"
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding=target, errors="replace")
        except (LookupError, ValueError, OSError, AttributeError):
            # Unknown codepage name, or a stream that cannot be reconfigured
            # (a pipe wrapped by a test harness, a redirected file). Neither is
            # worth failing a run over.
            pass


def _objectives() -> dict:
    from .objective import (diffusion_objective, gemm_objective,
                            render_objective, shot_objective, vellum_objective)
    return {
        "diffusion": diffusion_objective,
        "vellum": vellum_objective,
        "render": render_objective,
        "shot": shot_objective,
        "gemm": gemm_objective,
    }


def _load_objective(name: Optional[str]) -> Optional[Objective]:
    if not name:
        return None
    makers = _objectives()
    if name not in makers:
        raise SystemExit(f"unknown objective '{name}'; choose from: "
                         f"{', '.join(sorted(makers))}")
    return makers[name]()


def _parse_samples(text: Optional[str]) -> Optional[List[float]]:
    if not text:
        return None
    try:
        return [float(x) for x in text.replace(",", " ").split()]
    except ValueError:
        raise SystemExit(f"could not read samples from {text!r} — expected numbers, "
                         f"e.g. --samples '41.2,40.9,41.5,40.7'")


def _parse_keyed_samples(items: Optional[List[str]]) -> Optional[Dict[str, List[float]]]:
    """`KEY=SAMPLES` pairs -> {change_key: samples}. Split on the LAST '=': change
    keys contain '|' and ':' but sample lists never contain '='."""
    if not items:
        return None
    out: Dict[str, List[float]] = {}
    for item in items:
        key, sep, text = item.rpartition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"could not read {item!r} — expected CHANGE_KEY=SAMPLES, "
                             f"e.g. --candidate-for 'system|env|user|VAR=38.1,38.4'")
        if key in out:
            raise SystemExit(f"samples for {key!r} were given twice")
        samples = _parse_samples(text)
        if not samples:
            raise SystemExit(f"no samples after '=' in {item!r}")
        out[key] = samples
    return out


def _guardrail_metric(name: str, obj: Objective, item: str) -> str:
    """A guardrail metric name the objective declares, or SystemExit. An
    undeclared one is refused rather than carried: nothing would ever check it."""
    from .objective import KNOWN_METRICS
    if name not in KNOWN_METRICS:
        raise SystemExit(f"unknown guardrail metric {name!r} in {item!r}; known: "
                         f"{', '.join(sorted(KNOWN_METRICS))}")
    if name not in obj.guardrails:
        raise SystemExit(f"guardrail {name!r} in {item!r} is not declared by this "
                         f"objective (declared: "
                         f"{', '.join(sorted(obj.guardrails)) or 'none'})")
    return name


def _parse_guardrail_baseline(items: Optional[List[str]], obj: Objective
                              ) -> Optional[Dict[str, List[float]]]:
    """`METRIC=SAMPLES` pairs -> {metric: samples}, the baseline side of each
    guardrail, recorded with the baseline."""
    if not items:
        return None
    out: Dict[str, List[float]] = {}
    for item in items:
        name, sep, text = item.partition("=")
        name = name.strip()
        if not sep or not name:
            raise SystemExit(f"could not read {item!r} — expected METRIC=SAMPLES, "
                             f"e.g. --guardrail 'lpips=0.21,0.20,0.22'")
        _guardrail_metric(name, obj, item)
        if name in out:
            raise SystemExit(f"guardrail {name!r} was given twice")
        samples = _parse_samples(text)
        if not samples or len(samples) < 2:
            raise SystemExit(f"guardrail {name!r} needs at least 2 samples in {item!r}")
        out[name] = samples
    return out


def _parse_keyed_guardrails(items: Optional[List[str]], obj: Optional[Objective]
                            ) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """`KEY:METRIC=SAMPLES` -> {change_key: {metric: candidate samples}}.

    Split on the LAST '=' (as --candidate-for), then on the LAST ':' of the left
    side: change keys contain ':' (C:\\..., HKLM:...) but metric names never do."""
    if not items:
        return None
    if obj is None:
        raise SystemExit("--guardrail-for needs --objective: guardrails are declared "
                         "by the objective")
    out: Dict[str, Dict[str, List[float]]] = {}
    for item in items:
        left, sep, text = item.rpartition("=")
        key, csep, name = left.rpartition(":")
        key, name = key.strip(), name.strip()
        if not sep or not csep or not key or not name:
            raise SystemExit(f"could not read {item!r} — expected CHANGE_KEY:METRIC=SAMPLES, "
                             f"e.g. --guardrail-for 'system|env|user|VAR:lpips=0.21,0.20'")
        _guardrail_metric(name, obj, item)
        if name in out.get(key, {}):
            raise SystemExit(f"guardrail {name!r} for {key!r} was given twice")
        samples = _parse_samples(text)
        if not samples or len(samples) < 2:
            raise SystemExit(f"guardrail {name!r} for {key!r} needs at least 2 samples")
        out.setdefault(key, {})[name] = samples
    return out


def _pair_guardrails(candidates: Optional[Dict[str, Dict[str, List[float]]]],
                     bl: Optional[baseline_mod.Baseline]
                     ) -> Optional[Dict[str, Dict[str, tuple]]]:
    """Pair each candidate guardrail series with the baseline's own series for
    that metric. A baseline without one pairs with () — and the objective denies
    a declared guardrail with no baseline side, which is the honest outcome."""
    if not candidates:
        return None
    stored = bl.guardrail_samples if bl is not None else {}
    return {key: {m: (list(stored.get(m, [])), cand) for m, cand in by_metric.items()}
            for key, by_metric in candidates.items()}


def _parse_keyed_measurements(items: Optional[List[str]]
                              ) -> Optional[Dict[str, baseline_mod.Measurement]]:
    """`KEY=JSON` -> {change_key: Measurement}.

    The candidate side of a comparison carries its provenance: when it was
    measured, in which session, under what machine conditions, and the per-pair
    differences against its control. That provenance is what lets the gate
    refuse a comparison it cannot honestly make — a bare series of floats says
    nothing about when it was taken, so it cannot be shown to describe the same
    machine state as the baseline.

    The JSON is read as a whole, because every field is needed to make the
    decision: a measurement with a session but no timestamp cannot be placed in
    time, and one with neither is not a measurement.
    """
    if not items:
        return None
    out: Dict[str, baseline_mod.Measurement] = {}
    for item in items:
        key, sep, text = item.rpartition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"could not read {item!r} — expected CHANGE_KEY=JSON, "
                             f"e.g. --candidate-measurement 'system|env|user|VAR="
                             f"{{\"session_id\": \"...\", \"measured_at\": 0}}'")
        if key in out:
            raise SystemExit(f"measurement for {key!r} was given twice")
        try:
            raw = json.loads(text)
        except ValueError as e:
            raise SystemExit(f"could not read the measurement for {key!r} as JSON: {e}")
        if not isinstance(raw, dict):
            raise SystemExit(f"the measurement for {key!r} must be a JSON object")
        m = baseline_mod.Measurement.from_extra({"measurement": dict(raw)})
        if m is None:
            raise SystemExit(f"could not read the measurement for {key!r}")
        # The session id is NOT defaulted. A measurement that does not name the
        # session it belongs to cannot be shown to have been measured alongside
        # the baseline, and a gate that guesses one is a gate that has stopped
        # asking. The refusal is the gate's job, not the parser's.
        out[key] = m
    return out


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def _jev_line() -> str:
    """The one 'jev:' line. Imported lazily: status must work, and say so, on an
    interpreter where jev or its SDK cannot load."""
    try:
        from . import jev
        return jev.status_line()
    except Exception:  # noqa: BLE001 — an annotation layer must not break status
        return "jev: unavailable (v12.jev failed to import)"


def cmd_status(args) -> int:
    from .plane import status
    print(status())
    print(_jev_line())
    return 0


def _process_annotations(table) -> List[str]:
    """Jev labels for unresolved python.exe rows (plane.annotate_processes).
    Display only, and never allowed to break preflight."""
    try:
        from .plane import annotate_processes
        return annotate_processes(table)
    except Exception:  # noqa: BLE001 — an annotation layer must not break preflight
        return []


def cmd_preflight(args) -> int:
    from . import guard
    # One read: the verdict and the annotations describe the same instant.
    table = guard._process_table()
    ok, collisions, notes = guard.preflight(require_quiet=args.quiet, table=table)
    # The headline must not claim the GPU is free when the notes below it say
    # a process holds it. On 2026-11-04 this line read "CLEAR — nothing is
    # competing for the GPU" on a card at 98% and 288 W, because the work was
    # in WSL and the Windows process table could not see it.
    if not ok:
        headline = "BLOCKED"
    elif any(c.kind in ("gpu-app", "wsl-gpu-app") for c in collisions):
        headline = "CLEAR of optimizers — but GPU work IS resident (see below)"
    elif any("GPU-resident" in n for n in notes):
        headline = "CLEAR of optimizers — but GPU work IS resident (see below)"
    else:
        headline = "CLEAR — nothing is competing for the GPU"
    print("preflight: " + headline)
    print("-" * 68)
    for n in notes:
        print("  " + n)
    # Annotation status only. Jev plays no part in the CLEAR/BLOCKED decision above.
    print("  " + _jev_line())
    # Asked only after the decision, and not at all when it blocked.
    if ok:
        for a in _process_annotations(table):
            print("  " + a)
    if collisions:
        print()
        print("  processes that would confound a measurement:")
        for c in collisions:
            print(f"    [{c.kind}] {c.name} (pid {c.pid})")
    print()
    print("  why this matters: on this machine there is no cross-process GPU")
    print("  partitioning. MIG is datacenter-only, MPS is Linux/QNX-only, and TCC")
    print("  is unreachable on GeForce. Two GPU workloads share the device, so any")
    print("  measurement taken while another one runs describes neither.")
    return 0 if ok else 1


def cmd_kernels(args) -> int:
    from .actuators.kernel import toolchain_report
    print(toolchain_report())
    return 0


def cmd_baseline(args) -> int:
    obj = _load_objective(args.objective)
    if args.supersede:
        # Withdrawing a baseline is a supported operation, not a hand-edit of
        # the JSON, so the reason is mandatory and the file keeps its shape.
        if not args.reason:
            raise SystemExit("--supersede needs --reason: say WHY this baseline "
                             "cannot support the comparisons it was used for")
        try:
            b = baseline_mod.supersede(args.supersede, args.reason)
        except ValueError as e:
            raise SystemExit(str(e))
        # Re-index so the ledger's copy agrees with the file. A withdrawn
        # baseline that still looks usable in the index is a quiet
        # disagreement between two records of the same thing.
        with open_ledger() as lg:
            lg.index_baseline(b)
        print(f"superseded baseline {b.id} ({b.metric}, mean {b.mean:.4f})")
        print(f"  {b.path()}")
        print(f"  reason: {b.superseded_reason}")
        print("  it is skipped by 'v12 baseline' lookups and refused by the gate, so")
        print("  it can no longer gate anything. The numbers stay on disk.")
        return 0
    if args.record:
        if not obj:
            raise SystemExit("--record needs --objective")
        samples = _parse_samples(args.record)
        if not samples:
            raise SystemExit("--record needs numbers, e.g. --record '41.2,40.9,41.5'")
        guards = _parse_guardrail_baseline(args.guardrail, obj)
        missing = sorted(set(obj.guardrails) - set(guards or {}))
        if missing:
            print(f"NOTE: no baseline series for guardrail(s) {', '.join(missing)} — "
                  f"MEASURED changes compared against this baseline will be denied "
                  f"until one is recorded with them (--guardrail METRIC=SAMPLES)")
        try:
            b = baseline_mod.record(obj, samples, harness=args.harness or "", notes=[],
                                    guardrail_samples=guards)
        except ValueError as e:
            raise SystemExit(str(e))
        print(f"recorded baseline {b.id} ({b.n} samples, mean {b.mean:.4f} "
              f"{obj.primary.unit if obj.primary else ''}, cv {b.cv:.2%})")
        print(f"  {b.path()}")
        with open_ledger() as lg:
            lg.index_baseline(b)
        return 0
    print(baseline_mod.describe(obj))
    return 0


def cmd_run(args) -> int:
    from .plane import Plane
    obj = _load_objective(args.objective)

    bl = None
    if obj:
        if args.baseline:
            bl = baseline_mod.Baseline.load(Path(args.baseline))
        else:
            bl = baseline_mod.load_latest(obj)
        if bl and not bl.usable:
            print(f"NOTE: baseline {bl.id} has cv {bl.cv:.2%} — too noisy to conclude "
                  f"from. Pin the clocks to reduce variance before trusting a verdict.")
        if bl:
            ok, why = bl.valid_for(obj)
            if not ok:
                print(f"NOTE: baseline {bl.id} does not apply here — {why}")
                bl = None

    # Samples are evidence about ONE change. A bare --candidate series is handed to
    # the plane as-is, which attributes it only when exactly one MEASURED candidate
    # is eligible and otherwise refuses it with a note in the report. Per-change
    # series are keyed by change_key.
    if args.candidate and args.candidate_for:
        raise SystemExit("use either --candidate (one series, only when exactly one "
                         "MEASURED change is eligible) or --candidate-for KEY=SAMPLES "
                         "(per change), not both")
    samples = (_parse_keyed_samples(args.candidate_for) if args.candidate_for
               else _parse_samples(args.candidate))
    # Guardrails are keyed only, like --candidate-for: each is evidence about one
    # change. The candidate side comes from here, the baseline side from the
    # baseline itself, so both halves were measured under the conditions they
    # claim.
    cand_guards = _parse_keyed_guardrails(args.guardrail_for, obj)
    guardrails = _pair_guardrails(cand_guards, bl)
    # The candidate's provenance. Keyed like the samples, because it describes
    # the same measurement: without it, a MEASURED change is refused rather
    # than compared against a baseline it may never have been measured with.
    measurements = _parse_keyed_measurements(args.candidate_measurement)
    if obj and obj.guardrails and samples:
        declared = ", ".join(sorted(obj.guardrails))
        if not cand_guards:
            print(f"NOTE: objective '{args.objective}' declares guardrail(s) {declared}; "
                  f"without --guardrail-for KEY:METRIC=SAMPLES every MEASURED change "
                  f"is denied")
        if bl is not None and set(obj.guardrails) - set(bl.guardrail_samples):
            print(f"NOTE: baseline {bl.id} holds no series for guardrail(s) "
                  + ", ".join(sorted(set(obj.guardrails) - set(bl.guardrail_samples)))
                  + " — MEASURED changes will be denied until one is recorded")

    plane = Plane(
        objective=obj,
        dry_run=not args.apply,
        allow_destructive_reset=args.allow_destructive_reset,
        require_quiet=args.quiet,
    )
    result = plane.run_cycle(baseline_for_objective=bl, candidate_samples=samples,
                             guardrail_samples=guardrails,
                             candidate_measurements=measurements)
    print(result.report())

    if not args.apply:
        print()
        print("  This was a DRY RUN. Nothing on this machine was changed.")
        if result.admitted:
            print("  To apply what was admitted:  --apply")
        return 0
    return 0


#: Printed under every plan. Fixed text, tested: it names no evidence class.
PLAN_FOOTER = ("  an order-hint only decides what to time first. It admits nothing: every "
               "change\n  still needs its own samples against a valid baseline "
               "(run --candidate-for KEY=SAMPLES).")


def cmd_plan(args) -> int:
    """The measure-next list for one objective. Read-only by construction: it
    gathers the catalog through plane.gather_catalog, which builds no Plane, no
    Gate and no ledger, takes no snapshot and records nothing. The order comes
    from v12/data/catalog_relevance.json (relevance.py); a bad or missing file
    falls back to catalog order and says so."""
    from . import relevance
    from .change import ChangeKind, change_key
    from .plane import gather_catalog

    path = Path(args.relevance) if args.relevance else relevance.DEFAULT_PATH
    rf = None
    try:
        rf = relevance.load(path)
        source = (f"{path} (model {rf.model or '?'}, drafted {rf.drafted_ts or '?'}, "
                  f"{'reviewed' if rf.reviewed else 'not reviewed'})")
    except relevance.RelevanceError as e:
        source = f"none usable — {e}; listed in catalog order"

    notes: List[str] = []
    cands, meta = gather_catalog(notes)
    listed = [c for c in cands if c.kind is not ChangeKind.GUIDED]
    advisory = len(cands) - len(listed)
    ordered = relevance.order([c.id for c in listed], rf, args.objective)
    untagged = sum(1 for _, lvl in ordered if lvl is None)

    print(f"V12 plan — measure-next order for objective '{args.objective}' "
          f"(read-only: no gate, no decisions, nothing changed)")
    print("-" * 68)
    if relevance.is_draft(rf):
        print("  " + relevance.DRAFT_LINE)
    print(f"  order-hint source: {source}")
    print()
    print(f"  {'#':>3}  {'order-hint':<10}  change_key / title")
    for n, (i, lvl) in enumerate(ordered, 1):
        c = listed[i]
        k = change_key(c)
        m = meta.get((k, c.id))
        title = (m.title if m is not None and m.title else c.description)
        hint = f"{lvl}/{relevance.MAX_LEVEL}" if lvl is not None else "-"
        print(f"  {n:>3}  {hint:<10}  {k}")
        print(f"  {'':>3}  {'':<10}  {title}  [{c.id}]")
    print()
    print(f"  untagged: {untagged} of {len(listed)} (no order-hint for "
          f"'{args.objective}'; listed last, in catalog order)")
    if advisory:
        print(f"  advisory-only (never applied): {advisory} not listed")
    for n in notes:
        print(f"  note: {n}")
    print()
    print(PLAN_FOOTER)
    return 0


def cmd_ledger(args) -> int:
    with open_ledger() as lg:
        print(lg.honesty_report(session_only=args.session))
        rows = lg.recent_applied(limit=args.limit)
        if rows:
            print()
            print(f"  last {len(rows)} applied:")
            for r in rows:
                eff = f"effect {r['effect']:+.3f}" if r["effect"] is not None else "effect NOT measured"
                rev = " [reverted]" if r["reverted"] else ""
                print(f"    {r['change_id']:<28} {r['evidence']:<9} {eff}{rev}")
        _print_judgments(lg, session_only=args.session)
    return 0


def _print_judgments(lg, session_only: bool) -> None:
    """Jev's rows, in their own section, after — never inside — the honesty
    metric. Every field that came from the model is sanitized before printing:
    it is network text, and a terminal executes escape codes."""
    summarize = getattr(lg, "judgments_summary", None)
    if summarize is None:
        return
    js = summarize(session_only=session_only)
    if not js["total"]:
        return
    from .jev import sanitize_for_terminal as clean
    print()
    print("  Jev annotations - not evidence (never counted above, never admitted):")
    print("    " + clean(", ".join(f"{k} {v}" for k, v in sorted(js["by_outcome"].items()))))
    for r in js["recent"]:
        what = (f"{r['label']}" if r["outcome"] == "ok"
                else f"unlabelled ({r['outcome']})")
        hit = " [cached]" if r["cache_hit"] else ""
        print("    " + clean(f"{r['stage']:<8} {r['question_id']:<20} "
                             f"{(r['subject_key'] or '')[:40]:<40} {what} "
                             f"[{r['consumed_as']}]{hit}"))


def cmd_rollback(args) -> int:
    from .plane import Plane
    with open_ledger() as lg:
        rows = [r for r in lg.recent_applied(limit=500) if not r["reverted"]]
    if not rows:
        print("nothing is currently applied — nothing to roll back.")
        return 0
    if args.list or not args.change:
        print("currently applied changes (rollback takes no verdict — undo must not")
        print("depend on the evidence that may be exactly what failed):")
        print("-" * 68)
        for r in rows:
            print(f"  {r['change_id']:<30} {r['tier']:<9} {r['evidence']:<9} "
                  f"key={r['change_key']}")
        print()
        print("  roll back one:   --change <change_id>")
        print("  roll back all:   --all")
        return 0

    print("rollback by change_id needs the live Change objects from the session that")
    print("applied them; this build does not reconstruct a Change from the ledger yet.")
    print("Use MachineOptimizer2026's own Restore-This-Machine.bat for the system tier,")
    print("which reads its restore master — 37 recoverable changes, sha256-verified.")
    return 2


def cmd_migrate(args) -> int:
    from . import migrate
    paths.ensure_dirs()
    r = migrate.run(force=args.force)
    print(r.summary())
    ok, msg = migrate.verify()
    print()
    print(("VERIFY OK: " if ok else "VERIFY FAILED: ") + msg)
    return 0 if (r.ok and ok) else 1


def cmd_truth(args) -> int:
    from . import truth
    print(truth.describe(as_json=args.json))
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="v12", description="V12 — gated optimizer control plane")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="machine, roofline, and baseline state")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("preflight", help="is anything competing for the GPU")
    s.add_argument("--quiet", action="store_true",
                   help="treat a running GPU app as blocking, not just noisy")
    s.set_defaults(func=cmd_preflight)

    s = sub.add_parser("kernels", help="compile toolchain readiness")
    s.set_defaults(func=cmd_kernels)

    s = sub.add_parser("truth", help="live hardware truth, with provenance")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_truth)

    s = sub.add_parser("baseline", help="record or list measurement baselines")
    s.add_argument("--objective", choices=sorted(_objectives()))
    s.add_argument("--record", metavar="SAMPLES",
                   help="comma-separated numbers, e.g. '41.2,40.9,41.5'")
    s.add_argument("--harness", default="")
    s.add_argument("--guardrail", metavar="METRIC=SAMPLES", action="append",
                   help="baseline-side series for a guardrail the objective declares, "
                        "measured with --record, e.g. 'lpips=0.21,0.20,0.22'; repeatable")
    s.add_argument("--supersede", metavar="ID",
                   help="withdraw a baseline as a reference; it is skipped by "
                        "lookups and refused by the gate, so it can no longer gate "
                        "anything. The numbers stay on disk.")
    s.add_argument("--reason", default="",
                   help="why the baseline is being superseded (required with "
                        "--supersede)")
    s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("run", help="one cycle; dry run unless --apply")
    s.add_argument("--objective", choices=sorted(_objectives()),
                   help="needed for any change to be admitted")
    s.add_argument("--apply", action="store_true",
                   help="actually mutate the machine (default is a dry run)")
    s.add_argument("--baseline", metavar="PATH", help="explicit baseline file")
    s.add_argument("--candidate", metavar="SAMPLES",
                   help="candidate measurement, e.g. '38.1,38.4,37.9'; used only when "
                        "exactly one MEASURED change is eligible")
    s.add_argument("--candidate-for", metavar="KEY=SAMPLES", action="append",
                   help="candidate measurement for one change, keyed by its change_key "
                        "(shown in the decisions list); repeatable")
    s.add_argument("--guardrail-for", metavar="KEY:METRIC=SAMPLES", action="append",
                   help="candidate-side series for one guardrail of one change, e.g. "
                        "'system|env|user|VAR:lpips=0.21,0.20'; paired with the "
                        "baseline's own series for that metric (baseline --guardrail). "
                        "An objective that declares a guardrail denies a change whose "
                        "guardrail is not supplied; repeatable")
    s.add_argument("--candidate-measurement", metavar="KEY=JSON", action="append",
                   help="the provenance of one candidate measurement: when it was "
                        "measured, in which session, under what machine conditions, "
                        "and the per-pair differences against its control. Without "
                        "it a MEASURED change is refused, because a bare series of "
                        "floats cannot be shown to describe the same machine state "
                        "as the baseline. repeatable")
    s.add_argument("--allow-destructive-reset", action="store_true",
                   help="permit changes whose rollback resets to vendor default, "
                        "losing the operator's prior value (there is no getter, so "
                        "the prior cannot be restored)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("plan", help="measure-next order for an objective (read-only)")
    s.add_argument("--objective", choices=sorted(_objectives()), required=True)
    s.add_argument("--relevance", metavar="PATH",
                   help="relevance tags file (default v12/data/catalog_relevance.json)")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("ledger", help="what was done, and how much was measured")
    s.add_argument("--limit", type=int, default=25)
    s.add_argument("--session", action="store_true", help="this session only")
    s.set_defaults(func=cmd_ledger)

    s = sub.add_parser("rollback", help="undo (requires no verdict)")
    s.add_argument("--list", action="store_true")
    s.add_argument("--change", metavar="CHANGE_ID")
    s.add_argument("--all", action="store_true")
    s.set_defaults(func=cmd_rollback)

    s = sub.add_parser("migrate", help="Phase 0a — move MOS2026 state off OneDrive")
    s.add_argument("--force", action="store_true",
                   help="re-run even if a destination already exists")
    s.set_defaults(func=cmd_migrate)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    _setup_console()
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted — nothing was left half-applied")
        return 130


if __name__ == "__main__":
    sys.exit(main())
