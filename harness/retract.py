"""
retract — withdraw a verdict, with the reason attached, in the place it lives.

Why this exists
---------------
The harness writes its measurement records as JSON under
E:\\OptimizerV12\\measurements. One of them, levers_20261103_192945.json, records
this verdict:

    husk_j48   ADMIT  (+12.34%)  p=1.882e-05  n=5/5

It is wrong. The candidate was measured at 19:29 and gated against a baseline
measured at 18:41 — 48 minutes earlier, under a different set of resident
processes. The two numbers described two different machine states and the
difference between them was charged to the candidate.

Re-measured on 2026-11-04 with both arms measured ALTERNATELY in one session
(harness/levers.py, husk_threads lever): DENY, -0.89%. The lever was never a
win.

So the record has to change. But HOW it changes matters, because the failure
mode here is a record that quietly stops telling the truth:

  * DELETING the verdict erases the evidence that it was ever made, and the
    ledger's honesty report counts denials as carefully as admissions — a
    vanished denial is a lie in the other direction;
  * EDITING the verdict string to say "DENY" fabricates a verdict that was
    never reached;
  * LEAVING it alone leaves a false ADMIT in the store, where the next person
    to read the ledger inherits it.

So the verdict is kept, and marked. The original text stays readable, and the
record gains a retraction that says what the verdict was, why it does not
hold, and what was measured instead.

    python harness\\retract.py --file <path> --label husk_j48 \\
        --reason "..." --replacement <path>

Two record shapes
-----------------
A measurement record is written in one of two shapes, and a retraction has to
reach into BOTH of them:

  * `levers_*.json` — a LIST of result blocks, each carrying its own "label"
    and a "verdict" string;
  * `optimize_*.json` — a DICT of result blocks keyed by the lever name,
    where the block carries no label and no verdict string: the claim lives
    in the boolean "ok" and the human-readable "why".

The second shape is the one this harness writes most often, and it is the
shape the false ADMITs of 2026-11-04 were written in — so a retraction that
only reached the first shape would have left them standing, which is the one
outcome this module exists to prevent.

In the second shape the "ok" boolean is NOT flipped. `ok: false` in these
records means the gate REFUSED the run (the control drifted, say), so
flipping it would fabricate a refusal that was never reached — the mirror
image of the failure this module exists to prevent. The original value is
kept beside the retraction instead, and the "why" string carries the mark.

Usage
-----
    python harness\\retract.py --file E:\\...\\levers_20261103_192945.json \\
        --label husk_j48 --reason "measured against a stale baseline" \\
        --replacement E:\\...\\optimize_20261104_111511.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, doc: Dict[str, Any]) -> None:
    """Write the record back, atomically.

    A half-written measurement record is worse than a wrong one, because it
    cannot be read at all and nobody can tell that it was ever complete.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)


def _find_result(doc: Dict[str, Any], label: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Find one result in a record, in either of the two shapes on disk.

    Returns (result, is_list). The two shapes carry the claim differently:

      * a LIST of blocks, each with its own "label" and a "verdict" string
        (levers_*.json);
      * a DICT of blocks keyed by the lever name, with no label and no
        verdict string — the claim is the boolean "ok" and the text "why"
        (optimize_*.json).

    A retraction that only understood one shape would leave the other's
    verdicts unmarked, which is the failure this module exists to prevent.
    """
    results = doc.get("results")
    if isinstance(results, list):
        for r in results:
            if r.get("label") == label:
                return r, True
        return None, True
    if isinstance(results, dict):
        r = results.get(label)
        if isinstance(r, dict):
            return r, False
        return None, False
    return None, isinstance(results, list)


def _labels(doc: Dict[str, Any]) -> List[str]:
    """Every label in the record, so a refusal can say what IS there."""
    results = doc.get("results")
    if isinstance(results, list):
        return [r.get("label") for r in results if isinstance(r, dict)]
    if isinstance(results, dict):
        return list(results.keys())
    return []


def _mark_withdrawn(r: Dict[str, Any], is_list: bool) -> None:
    """Mark the live claim as withdrawn, keeping its original text visible.

    The string a reader sees is prefixed with "RETRACTED — was: " and the
    original still sits inside it, so the claim is marked AND readable. The
    "ok" boolean is deliberately NOT flipped: in these records false means
    the gate REFUSED the run, so flipping true to false would fabricate a
    denial that was never reached — the mirror image of the failure this
    mechanism exists to prevent.
    """
    if is_list:
        r["verdict"] = f"RETRACTED — was: {r.get('verdict', '')}"
    else:
        r["why"] = f"RETRACTED — was: {r.get('why', '')}"


def retract_result(path: Path, label: str, reason: str,
                   replacement: Optional[str] = None,
                   replacement_verdict: str = "",
                   when: Optional[float] = None) -> Dict[str, Any]:
    """Mark one result in a measurement record as RETRACTED.

    The original verdict text is kept. A retraction that overwrites the
    verdict has destroyed the record of what was claimed, and the honesty
    report counts verdicts by what they said.
    """
    if not reason or not reason.strip():
        raise ValueError("a retraction needs a reason — say WHY the verdict does "
                         "not hold")
    doc = _load(path)
    if not doc.get("results"):
        raise ValueError(f"{path} has no 'results' to retract from")
    r, is_list = _find_result(doc, label)
    if r is None:
        raise ValueError(f"no result labelled {label!r} in {path} "
                         f"(have: {_labels(doc)})")
    if r.get("retracted"):
        prior = r["retracted"]
        if prior.get("reason") == reason.strip():
            return r                     # already retracted, identically: nothing to do
        raise ValueError(f"{label} is already retracted with a different reason; "
                         f"refusing to overwrite it")

    retraction: Dict[str, Any] = {
        "date": time.strftime("%Y-%m-%d", time.gmtime(when if when is not None else time.time())),
        "reason": reason.strip(),
        "original_verdict": r.get("verdict", "") if is_list else r.get("why", ""),
        "replacement_measurement": replacement or "",
        "replacement_verdict": replacement_verdict or "",
    }
    if is_list:
        retraction["original_mean_s"] = r.get("mean_s")
    else:
        # The boolean the gate actually returned, kept beside the retraction
        # so the record still says what was decided — it just says the
        # decision does not hold.
        retraction["original_ok"] = r.get("ok")
        # The two means, under names that do not name a unit: what they
        # measure depends on the objective (seconds for a render, TFLOPS for
        # a GEMM), and a field called mean_s holding TFLOPS would be a lie
        # about the unit.
        if "baseline_mean" in r:
            retraction["original_baseline_mean"] = r.get("baseline_mean")
        if "candidate_mean" in r:
            retraction["original_candidate_mean"] = r.get("candidate_mean")
    r["retracted"] = retraction
    # The verdict string stays, but it is marked as withdrawn so a reader who
    # greps for "ADMIT" does not find a live-looking admission.
    _mark_withdrawn(r, is_list)
    _write(path, doc)
    return r


def update_replacement(path: Path, label: str, replacement: str,
                       replacement_verdict: str = "",
                       corrected_reason: str = "",
                       when: Optional[float] = None) -> Dict[str, Any]:
    """Point an EXISTING retraction at the measurement that replaced it.

    A retraction is often made before the replacement has been measured — you
    know the verdict does not hold, and the honest re-measurement is still to
    come. That is a legitimate order of operations, and this is the operation
    for it.

    Two rules, both about not losing the audit trail:

      * the replacement pointer is filled in, and the original retraction date
        and reason are left exactly as they were;
      * if the ORIGINAL REASON contains a claim that has since become false
        ("no replacement figure exists yet"), the correction is RECORDED
        alongside it under `reason_corrected` rather than overwriting it.

    Overwriting the reason would be the exact failure this whole mechanism
    exists to prevent: a record that changes to suit the present. The
    correction keeps both, so a reader can see what was claimed and when it
    stopped being true.
    """
    doc = _load(path)
    r, _is_list = _find_result(doc, label)
    if r is None:
        raise ValueError(f"no result labelled {label!r} in {path} "
                         f"(have: {_labels(doc)})")
    ret = r.get("retracted")
    if not ret:
        raise ValueError(f"{label} is not retracted — retract it first, then point "
                         f"it at its replacement")
    # Keep the previous replacement visible. A measurement can be superseded
    # more than once — the first replacement may itself turn out to have been
    # taken on a busy machine — and a chain of them is the honest record.
    # Overwriting would leave no trace that there ever was one.
    prior = ret.get("replacement_measurement")
    if prior and prior != replacement:
        history = ret.setdefault("replacement_history", [])
        history.append({
            "date": ret.get("replacement_updated", ret.get("date", "")),
            "measurement": prior,
            "verdict": ret.get("replacement_verdict", ""),
            "superseded_because": corrected_reason.strip() if corrected_reason.strip()
                                  else "superseded by a later measurement",
        })
    ret["replacement_measurement"] = replacement
    ret["replacement_verdict"] = replacement_verdict
    ret["replacement_updated"] = time.strftime(
        "%Y-%m-%d", time.gmtime(when if when is not None else time.time()))
    if corrected_reason and corrected_reason.strip():
        # A second correction must not ERASE the first one. The correction is
        # the record of what the retraction's own reason got wrong; losing an
        # earlier correction is the same failure as losing an earlier
        # measurement, and in the same place. Corrections chain, the way
        # replacements chain.
        prior_correction = ret.get("reason_corrected")
        if prior_correction and prior_correction.get("reason") != corrected_reason.strip():
            chain = ret.setdefault("reason_correction_history", [])
            chain.append({
                "date": prior_correction.get("date", ""),
                "reason": prior_correction.get("reason", ""),
                "superseded_because": corrected_reason.strip(),
            })
        ret["reason_corrected"] = {
            "date": time.strftime("%Y-%m-%d",
                                  time.gmtime(when if when is not None else time.time())),
            "reason": corrected_reason.strip(),
        }
    _write(path, doc)
    return r


def mark_superseded(path: Path, reason: str, when: Optional[float] = None
                    ) -> Dict[str, Any]:
    """Mark a whole measurement record as superseded.

    Used for the baselines the retracted verdict was gated against: they were
    real measurements of a real machine, but they cannot serve as the control
    for a comparison, so they must not be used as one.
    """
    if not reason or not reason.strip():
        raise ValueError("superseding needs a reason")
    doc = _load(path)
    if doc.get("superseded"):
        prior = doc["superseded"]
        if prior.get("reason") == reason.strip():
            return doc
        raise ValueError("already superseded with a different reason; refusing to "
                         "overwrite it")
    doc["superseded"] = {
        "date": time.strftime("%Y-%m-%d", time.gmtime(when if when is not None else time.time())),
        "reason": reason.strip(),
    }
    _write(path, doc)
    return doc


def correct_supersede_reason(path: Path, corrected_reason: str,
                             when: Optional[float] = None) -> Dict[str, Any]:
    """Correct the reason a record was superseded — keeping the earlier text.

    A supersede reason is a factual claim about the record, exactly like a
    retraction's reason, and it can be wrong in the same way: another
    record's numbers attributed to this one, or a defect this record does not
    have. Overwriting it would destroy the record of what was claimed, which
    is the failure this module exists to prevent — and it would make the
    correction invisible, so the next reader inherits a confident false
    statement.

    So the correction is recorded ALONGSIDE, and the earlier text stays
    readable. Corrections chain, the way replacements chain: a second
    correction does not erase the first.

    This mirrors `update_replacement`'s handling of a retraction's reason,
    because it is the same situation and the same rule.
    """
    if not corrected_reason or not corrected_reason.strip():
        raise ValueError("a correction needs the corrected text")
    doc = _load(path)
    block = doc.get("superseded")
    if not block:
        raise ValueError(f"{path} is not marked superseded — mark it first, then "
                         f"correct the reason")
    prior = block.get("reason_corrected")
    if prior and prior.get("reason") == corrected_reason.strip():
        return doc                 # same correction again: nothing to do
    if prior:
        # A second correction must not ERASE the first. The correction is the
        # record of what the supersede's own reason got wrong; losing an
        # earlier correction is the same failure as losing an earlier
        # measurement, in the same place.
        chain = block.setdefault("reason_correction_history", [])
        chain.append({
            "date": prior.get("date", ""),
            "reason": prior.get("reason", ""),
            "superseded_because": corrected_reason.strip(),
        })
    block["reason_corrected"] = {
        "date": time.strftime("%Y-%m-%d", time.gmtime(when if when is not None else time.time())),
        "reason": corrected_reason.strip(),
    }
    _write(path, doc)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="the measurement record to annotate")
    ap.add_argument("--label", default="",
                    help="which result to retract (omit to supersede the whole record)")
    ap.add_argument("--reason", default="",
                    help="why the verdict does not hold (required unless --update)")
    ap.add_argument("--replacement", default="",
                    help="the measurement that supersedes the retracted verdict")
    ap.add_argument("--replacement-verdict", default="",
                    help="what that measurement actually found")
    ap.add_argument("--corrected-reason", default="",
                    help="if the ORIGINAL reason contains a claim the new measurement "
                         "makes false, the correction is recorded alongside it — "
                         "the original is never overwritten")
    ap.add_argument("--update", action="store_true",
                    help="point an EXISTING retraction at its replacement, instead of "
                         "retracting anew")
    ap.add_argument("--correct-supersede-reason", action="store_true",
                    help="correct the reason a WHOLE RECORD was superseded, without "
                         "erasing the earlier text (the correction is recorded "
                         "alongside it, and corrections chain)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"no such file: {path}")
        return 2
    if args.dry_run:
        print(f"would annotate {path}")
        return 0
    try:
        if args.correct_supersede_reason:
            if not args.corrected_reason:
                raise ValueError("--correct-supersede-reason needs "
                                 "--corrected-reason: the corrected text")
            if args.label:
                raise ValueError("--correct-supersede-reason is for a whole record; "
                                 "to correct a RESULT's retraction reason use "
                                 "--update --corrected-reason")
            doc = correct_supersede_reason(path, args.corrected_reason)
            block = doc["superseded"]
            print(f"corrected the supersede reason of {path.name}")
            print("  the original reason stands as recorded:")
            print(f"    {block['reason']}")
            if block.get("reason_corrected"):
                print(f"  corrected ({block['reason_corrected']['date']}):")
                print(f"    {block['reason_corrected']['reason']}")
        elif args.update:
            if not args.label:
                raise ValueError("--update needs --label: which retraction to update")
            r = update_replacement(path, args.label, args.replacement,
                                   args.replacement_verdict, args.corrected_reason)
            ret = r["retracted"]
            print(f"updated the retraction of {args.label} in {path.name}")
            print(f"  replaced by: {ret['replacement_measurement']}")
            print(f"  which found: {ret['replacement_verdict']}")
            if ret.get("reason_corrected"):
                print(f"  reason corrected ({ret['reason_corrected']['date']}):")
                print(f"    {ret['reason_corrected']['reason']}")
            else:
                print("  the original reason stands as recorded")
        elif args.label:
            if not args.reason:
                raise ValueError("a retraction needs --reason: say WHY the verdict "
                                 "does not hold")
            r = retract_result(path, args.label, args.reason, args.replacement,
                               args.replacement_verdict)
            print(f"retracted {args.label} in {path.name}")
            print(f"  was: {r['retracted']['original_verdict']}")
            print(f"  reason: {r['retracted']['reason']}")
            if r["retracted"]["replacement_measurement"]:
                print(f"  replaced by: {r['retracted']['replacement_measurement']}")
                if r["retracted"]["replacement_verdict"]:
                    print(f"  which found: {r['retracted']['replacement_verdict']}")
        else:
            if not args.reason:
                raise ValueError("superseding needs --reason")
            mark_superseded(path, args.reason)
            print(f"marked {path.name} as superseded")
            print(f"  reason: {args.reason}")
    except ValueError as e:
        print(f"refused: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
