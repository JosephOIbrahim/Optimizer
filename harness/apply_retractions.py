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
MEASUREMENTS = Path(os.environ.get("OPTIMIZER_MEASURE_ROOT",
                                   r"E:\OptimizerV12\measurements"))


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def describe(path: Path, label: str) -> str:
    """One line about a measurement, read back out of the record itself."""
    doc = read(path)
    results = doc.get("results") or {}
    v = results.get(label) or {}
    bits = [label]
    if v.get("effect") is not None:
        bits.append(f"{v['effect']:+.2f}%")
    if v.get("p_value") is not None:
        bits.append(f"p={v['p_value']:.3g}")
    if v.get("n_baseline") is not None:
        bits.append(f"n={v['n_baseline']:.0f}")
    if v.get("drift") is not None:
        bits.append(f"drift {v['drift']:+.2f}%")
    if v.get("control_spread") is not None:
        bits.append(f"spread {v['control_spread']:.2f}%")
    if v.get("contention_events") is not None:
        bits.append(f"contention {v['contention_events']}")
    if v.get("variant"):
        bits.append(f"({v['variant']})")
    if v.get("ok") is not None:
        bits.append("ADMIT" if v["ok"] else "DENY")
    return "  ".join(bits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", required=True,
                    help="JSON: [{record, label, measurement, reason}, ...]")
    args = ap.parse_args(argv)

    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
