"""
Tests for moving a retraction's pointer to a fresh measurement.

The retraction campaign is a chain: each retracted verdict keeps its ORIGINAL
claim visible and points at the measurement that replaced it. When the lever
is re-measured, the pointer moves, and the measurement it displaced goes into
the history chain with the reason it stopped being current.

Two rules make that chain honest, and they are what these tests pin:

  * the original claim is NEVER deleted — the live verdict still reads
    "RETRACTED — was: ADMIT (+12.34%)", so a reader who greps for an
    admission still finds the original text, marked;
  * a corrected reason is recorded ALONGSIDE the original rather than over
    it, because the original reason was true when it was written and a reader
    who sees only the newest text cannot tell that.

Nothing here touches the real records: every test works on a copy.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _gitless_copy(tmp: Path) -> Path:
    """A copy of the three retraction records plus one measurement."""
    src = Path(r"E:\OptimizerV12\measurements")
    dst = tmp / "measurements"
    dst.mkdir()
    for name in ("levers_20261103_221724.json",
                 "levers_20261103_222710.json",
                 "levers_20261103_192945.json",
                 "optimize_20261105_065602.json"):
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    return dst


class TestMovingAPointer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="apply_retr_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = _gitless_copy(self.tmp)
        self.env = dict(os.environ, OPTIMIZER_MEASURE_ROOT=str(self.store))

    def _mapping(self, measurement="optimize_20261105_065602.json"):
        m = self.tmp / "mapping.json"
        m.write_text(json.dumps([{
            "record": "levers_20261103_221724.json",
            "label": "aov_beauty_only",
            "measurement": measurement,
            "reason": "re-measured with the current harness",
        }]), encoding="utf-8")
        return m

    def _run(self, mapping=None):
        r = subprocess.run(
            [sys.executable, str(REPO / "harness" / "apply_retractions.py"),
             "--mapping", str(mapping or self._mapping())],
            capture_output=True, text=True, timeout=120, env=self.env)
        return r

    def test_the_pointer_moves(self):
        self._run()
        doc = json.loads((self.store / "levers_20261103_221724.json")
                         .read_text(encoding="utf-8"))
        r = [x for x in doc["results"] if x["label"] == "aov_beauty_only"][0]
        self.assertTrue(r["retracted"]["replacement_measurement"]
                        .endswith("optimize_20261105_065602.json"))

    def test_the_original_claim_survives(self):
        """The point of the whole chain: the original text is still there,
        marked as withdrawn."""
        self._run()
        doc = json.loads((self.store / "levers_20261103_221724.json")
                         .read_text(encoding="utf-8"))
        r = [x for x in doc["results"] if x["label"] == "aov_beauty_only"][0]
        self.assertTrue(r["verdict"].startswith("RETRACTED"))
        self.assertIn("+16.91", r["verdict"])
        self.assertIn("+16.91", r["retracted"]["original_verdict"])

    def test_the_displaced_measurement_goes_into_the_history(self):
        self._run()
        doc = json.loads((self.store / "levers_20261103_221724.json")
                         .read_text(encoding="utf-8"))
        r = [x for x in doc["results"] if x["label"] == "aov_beauty_only"][0]
        hist = r["retracted"].get("replacement_history", [])
        self.assertTrue(hist, "the measurement it displaced must stay visible")
        self.assertTrue(hist[-1]["measurement"].endswith("optimize_20261104_231946.json"))
        self.assertIn("re-measured with the current harness",
                      hist[-1]["superseded_because"])

    def test_the_original_reason_is_not_overwritten(self):
        """The original reason was true when it was written. A reader who
        sees only the newest text cannot tell that."""
        before = json.loads((self.store / "levers_20261103_221724.json")
                            .read_text(encoding="utf-8"))
        r = [x for x in before["results"] if x["label"] == "aov_beauty_only"][0]
        original_reason = r["retracted"]["reason"]
        self._run()
        after = json.loads((self.store / "levers_20261103_221724.json")
                           .read_text(encoding="utf-8"))
        r2 = [x for x in after["results"] if x["label"] == "aov_beauty_only"][0]
        self.assertEqual(r2["retracted"]["reason"], original_reason)
        self.assertTrue(r2["retracted"].get("reason_corrected"))

    def test_a_missing_measurement_is_refused_cleanly(self):
        r = self._run(self._mapping(measurement="optimize_99999999_999999.json"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no such measurement", r.stdout)

    def test_a_label_that_is_not_retracted_is_refused(self):
        m = self._mapping()
        m.write_text(json.dumps([{
            "record": "levers_20261103_221724.json",
            "label": "not_a_label",
            "measurement": "optimize_20261105_065602.json",
            "reason": "x",
        }]), encoding="utf-8")
        r = self._run(m)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no result labelled", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
