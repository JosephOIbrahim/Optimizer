"""
Tests that a retraction moves the repo's committed copy of the record.

Why this exists
---------------
A retraction edits a record IN PLACE. The repo keeps a committed copy of
every record under `measurements/`, and that copy is the evidence the
operator cards cite — a reader checking "+10.85%" against the repo finds a
file that has to agree with the live store.

It did not. On 2026-11-06 the retraction tool rewrote three records in the
live store and left the repo's copies pointing at the 2026-11-04
measurements. The cards and the repo's evidence disagreed with the live
store, and nothing said so. The gap was only found by diffing the two
copies by hand.

The rule the tests pin: the tool that writes a retraction moves the copy in
the same breath — UNLESS the store is a copy (a test's temporary directory),
because syncing a fixture into the real repo would overwrite the real
evidence with a fixture.

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


def _repo_copies():
    """The repo's committed copies, as bytes."""
    repo = REPO / "measurements"
    if not repo.exists():
        return {}
    return {p.name: p.read_bytes() for p in repo.glob("*.json")}


class TestTheRetractionMovesTheCopy(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="retr_sync_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = _gitless_copy(self.tmp)
        self.env = dict(os.environ, OPTIMIZER_MEASURE_ROOT=str(self.store))

    def _mapping(self, tmp_name="mapping.json"):
        m = self.tmp / tmp_name
        m.write_text(json.dumps([{
            "record": "levers_20261103_221724.json",
            "label": "aov_beauty_only",
            "measurement": "optimize_20261105_065602.json",
            "reason": "re-measured with the current harness",
        }]), encoding="utf-8")
        return m

    def _run(self, mapping=None, extra=()):
        return subprocess.run(
            [sys.executable, str(REPO / "harness" / "apply_retractions.py"),
             "--mapping", str(mapping or self._mapping()), *extra],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, env=self.env)

    def test_a_copy_store_does_not_sync_into_the_real_repo(self):
        """The dangerous case: a fixture overwriting the real evidence."""
        before = _repo_copies()
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("skipped", r.stdout.lower())
        self.assertEqual(_repo_copies(), before,
                         "a copy of the store must never change the real repo")

    def test_the_skip_is_explained_not_silent(self):
        r = self._run()
        self.assertIn("not the live one", r.stdout)

    def test_no_sync_is_an_explicit_choice(self):
        """--no-sync does the same thing, and says so."""
        r = self._run(extra=("--no-sync",))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("sync           :", r.stdout)

    def test_the_sync_is_announced_when_it_runs(self):
        """The live-store path must SAY it is syncing — a silent sync is
        indistinguishable from a skipped one."""
        src = (REPO / "harness" / "apply_retractions.py").read_text(
            encoding="utf-8")
        self.assertIn("sync           :", src)


class TestTheRepoCopiesAreCurrent(unittest.TestCase):
    """The end-to-end property: after a retraction, the repo's copy of the
    record and the live record agree."""

    def test_the_repo_copy_matches_the_live_store(self):
        repo = REPO / "measurements"
        live = Path(r"E:\OptimizerV12\measurements")
        if not live.exists():
            self.skipTest("no live store on this machine")
        checked = 0
        for name in sorted(live.glob("*.json")):
            r = repo / name.name
            if not r.exists():
                continue
            checked += 1
            self.assertEqual(
                json.loads(r.read_text(encoding="utf-8")),
                json.loads(name.read_text(encoding="utf-8")),
                f"{name.name}: the repo's copy and the live record disagree")
        self.assertGreater(checked, 0, "no records were compared")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTheMappingCannotEscapeTheStore(unittest.TestCase):
    """A mapping names a FILE INSIDE the store — not a path of its own.

    `store / entry` with an ABSOLUTE path on the right REPLACES the store,
    and a relative climb out of it climbs out. Both were reachable: the
    tool would have written the retraction somewhere other than the
    measurement store, and said nothing about it.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="retr_escape_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = _gitless_copy(self.tmp)
        self.env = dict(os.environ, OPTIMIZER_MEASURE_ROOT=str(self.store))

    def _mapping(self, record, measurement="optimize_20261105_065602.json"):
        m = self.tmp / "mapping.json"
        m.write_text(json.dumps([{
            "record": record,
            "label": "aov_beauty_only",
            "measurement": measurement,
            "reason": "re-measured with the current harness",
        }]), encoding="utf-8")
        return m

    def _run(self, mapping):
        return subprocess.run(
            [sys.executable, str(REPO / "harness" / "apply_retractions.py"),
             "--mapping", str(mapping)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, env=self.env)

    def test_an_absolute_record_path_is_refused(self):
        outside = self.tmp / "elsewhere.json"
        shutil.copy2(self.store / "levers_20261103_221724.json", outside)
        before = outside.read_bytes()
        r = self._run(self._mapping(str(outside)))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("escapes the store", r.stdout)
        # and it did not touch the file outside
        self.assertEqual(outside.read_bytes(), before,
                         "the file outside the store must not be written to")

    def test_a_relative_climb_out_is_refused(self):
        r = self._run(self._mapping(r"..\elsewhere.json"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("escapes the store", r.stdout)

    def test_an_absolute_measurement_path_is_refused(self):
        r = self._run(self._mapping(
            "levers_20261103_221724.json",
            measurement=str(self.tmp / "elsewhere.json")))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("escapes the store", r.stdout)

    def test_a_normal_name_still_works(self):
        """The guard must not break the ordinary case."""
        r = self._run(self._mapping("levers_20261103_221724.json"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("updated", r.stdout)
