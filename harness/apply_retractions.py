"""
apply_retractions — point the retracted verdicts at the fresh measurements.

This is the housekeeping half of the retraction campaign. Each retracted
verdict keeps its ORIGINAL claim visible ("RETRACTED — was: ADMIT (+12.34%)")
and points at the measurement that replaced it. When a lever is re-measured,
the pointer moves and the measurement it displaced goes into the history
chain with the reason it stopped being current.

The reason a corrected reason is recorded ALONGSIDE the original rather than
over it: the original reason was true when it was written, and a reader who
sees only the newest text cannot tell that.

Usage
-----
    python harness\\apply_retractions.py --mapping MAPPING.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from harness import retract  # noqa: E402

#: The live store. Overridable so the script can be exercised against a copy
#: without touching the real records.
LIVE_MEASUREMENTS = Path(r"E:\OptimizerV12\measurements")
MEASUREMENTS = Path(os.environ.get("OPTIMIZER_MEASURE_ROOT",
                                   str(LIVE_MEASUREMENTS)))


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_result(results: dict, label: str) -> dict:
    """The measurement block for a retraction's label.

    A retraction names the VARIANT ("aov_beauty_only"); a measurement keys
    its results by the LEVER ("aov_variant") and carries the variant inside.
    So the lookup has to bridge the two.
    """
    if label in results:
        return results[label]
    for v in results.values():
        if isinstance(v, dict) and v.get("variant") == label:
            return v
    # One lever in the record: it is the one.
    if len(results) == 1:
        return next(iter(results.values()))
    return {}


def describe(path: Path, label: str) -> str:
    """One line about a measurement, read back out of the record itself."""
    doc = read(path)
    results = doc.get("results") or {}
    v = _find_result(results, label)
    if not v:
        return f"{label} — measured with the current harness"
    bits = [v.get("variant") or label]
    if v.get("effect") is not None:
        bits.append(f"{v['effect'] * 100:+.2f}%")
    if v.get("p_value") is not None:
        bits.append(f"p={v['p_value']:.3g}")
    if v.get("n_baseline") is not None:
        bits.append(f"n={v['n_baseline']:.0f}")
    if v.get("drift") is not None:
        bits.append(f"drift {v['drift'] * 100:+.2f}%")
    if v.get("control_spread") is not None:
        bits.append(f"spread {v['control_spread'] * 100:.2f}%")
    if v.get("contention_events") is not None:
        bits.append(f"contention {v['contention_events']}")
    if v.get("variant"):
        bits.append(f"({v['variant']})")
    if v.get("ok") is not None:
        bits.append("ADMIT" if v["ok"] else "DENY")
    return "  ".join(bits)


def sync_repo_copies(entries) -> bool:
    """Move the repo's committed copies of the records this run edited.

    A retraction edits a record IN PLACE. The repo keeps a committed copy of
    every record — that copy is the evidence the operator cards cite — and a
    copy goes stale the moment the original is edited, which is precisely
    what a retraction does. So the tool that writes the retraction has to
    move the copy in the same breath, or the repo quietly disagrees with
    the live store about what was measured.

    Skipped when the store is a COPY (a test's temporary directory), because
    syncing a copy into the real repo would overwrite the real evidence with
    a fixture. That is the failure mode this guard exists to prevent.

    Returns True when the repo's copies are current (including when the sync
    was correctly skipped), False when they are known to be stale.
    """
    try:
        live = MEASUREMENTS.resolve() == LIVE_MEASUREMENTS.resolve()
    except OSError:
        live = False
    if not live:
        print("  sync           : skipped — this is a copy of the store, "
              "not the live one")
        return True

    try:
        from harness import sync_measurements
    except ImportError as e:  # pragma: no cover - import guard
        print(f"  sync           : SKIPPED ({e})")
        print("                   the repo's copies of these records are now "
              "STALE — run Sync-Measurements.bat")
        return False

    lines = ["Point the retracted verdicts at the current measurements", ""]
    lines.append("A retraction edits a record in place. The repo keeps a "
                 "committed copy of")
    lines.append("every record, and that copy is what the operator cards cite "
                 "as evidence,")
    lines.append("so it has to move with the original or the two disagree "
                 "about what was")
    lines.append("measured.")
    lines.append("")
    for entry in entries:
        lines.append(f"  {entry['record']}  {entry['label']}  ->  "
                     f"{Path(entry['measurement']).name}")
    lines.append("")
    lines.append("Co-Authored-By: Claude Code <noreply@anthropic.com>")
    message = "\n".join(lines)

    rc = sync_measurements.main(["--source", str(MEASUREMENTS),
                                  "--message", message])
    if rc != 0:
        print("  sync           : FAILED")
        print("                   the repo's copies of these records are now "
              "STALE — run Sync-Measurements.bat")
        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", required=True,
                    help="JSON: [{record, label, measurement, reason}, ...]")
    ap.add_argument("--no-sync", action="store_true",
                    help="do not copy the edited records into the repo")
    args = ap.parse_args(argv)

    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    applied = []
    for entry in mapping:
        record = MEASUREMENTS / entry["record"]
        new = MEASUREMENTS / entry["measurement"]
        if not record.exists():
            print(f"[ERROR] no such record: {record}")
            return 1
        if not new.exists():
            print(f"[ERROR] no such measurement: {new}")
            return 1
        doc = read(record)
        label = entry["label"]
        result = [r for r in doc["results"] if r.get("label") == label]
        if not result:
            print(f"[ERROR] {record.name} has no result labelled {label!r}")
            return 1
        r = result[0]
        if not r.get("retracted"):
            print(f"[ERROR] {label} in {record.name} is not retracted")
            return 1

        prior = r["retracted"].get("replacement_measurement", "")
        summary = describe(new, label)
        verdict = f"{summary} — measured with the current harness"
        print(f"\n{label}")
        print(f"  was pointing at : {Path(prior).name if prior else '(none)'}")
        print(f"  now points at  : {new.name}")
        print(f"  says           : {verdict}")
        try:
            retract.update_replacement(record, label, str(new), verdict,
                                       entry["reason"])
        except ValueError as e:
            print(f"[ERROR] {e}")
            return 1
        print("  updated")
        applied.append(entry)

    if applied and not args.no_sync:
        print()
        if not sync_repo_copies(applied):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
