r"""
Tests for the sync that keeps the committed measurement records current.

What it has to get right
------------------------
The committed copies under `measurements/` are the evidence the operator
cards cite. The live store is `E:\OptimizerV12\measurements`, deliberately
off OneDrive, so the repo copy is a copy — and it goes stale the moment a
record is written, or EDITED IN PLACE.

The edit-in-place case is the one that matters. A retraction rewrites a
record; that is the entire point of a retraction. A sync that copied "the
new files" would never notice, because the file name has not changed. So
this compares CONTENT.

Three things it must not do:
  * commit anything outside `measurements/` — the operator may have source
    edits in flight, and those are not this script's business;
  * delete a committed record that has gone from the live store — deleting
    evidence is a decision, not a side effect. It is reported instead;
  * commit a record that does not parse — a truncated write would otherwise
    land in the history.

Nothing here touches the real live store or the real git history: each test
builds its own source directory and its own repository.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import sync_measurements as sync  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                       text=True, timeout=60)
    return p.stdout


class _Sandbox:
    """A throwaway git repository with its own measurements directory."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sync_test_"))
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.source = self.tmp / "store"
        self.source.mkdir()
        self.dest = self.repo / "measurements"
        self.dest.mkdir()
        _git(self.repo, "init", "-b", "main")
        _git(self.repo, "config", "user.name", "test")
        _git(self.repo, "config", "user.email", "test@example.invalid")

    def write_record(self, source: Path, name: str, payload: dict) -> Path:
        p = source / name
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return p

    def commit_all(self, message="add"):
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-m", message)

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestThePlanFindsWhatChanged(unittest.TestCase):
    """The plan is the whole correctness of the script."""

    def setUp(self):
        self.sb = _Sandbox()
        self.addCleanup(self.sb.cleanup)

    def _plan(self):
        return sync.plan(self.sb.source, self.sb.dest, self.sb.repo)

    def test_a_new_record_is_new(self):
        self.sb.write_record(self.sb.source, "optimize_1.json", {"a": 1})
        p = self._plan()
        self.assertEqual(p["new"], ["optimize_1.json"])
        self.assertEqual(p["changed"], [])

    def test_a_record_edited_in_place_is_changed(self):
        """The case a copy-by-name script would miss entirely."""
        name = "optimize_1.json"
        self.sb.write_record(self.sb.source, name, {"ok": True, "why": "ADMIT"})
        shutil.copy2(self.sb.source / name, self.sb.dest / name)
        self.sb.commit_all()
        # Now the retraction lands: the file is rewritten, the name is the same.
        self.sb.write_record(self.sb.source, name, {"ok": False, "why": "RETRACTED"})
        p = self._plan()
        self.assertEqual(p["changed"], [name])
        self.assertEqual(p["new"], [])
        self.assertEqual(p["unchanged"], [])

    def test_an_unchanged_record_is_unchanged(self):
        name = "optimize_1.json"
        self.sb.write_record(self.sb.source, name, {"ok": True})
        shutil.copy2(self.sb.source / name, self.sb.dest / name)
        self.sb.commit_all()
        p = self._plan()
        self.assertEqual(p["unchanged"], [name])
        self.assertEqual(p["new"], [])
        self.assertEqual(p["changed"], [])

    def test_a_corrupt_record_is_invalid_and_not_copied(self):
        """A truncated write must not land in the history."""
        (self.sb.source / "optimize_bad.json").write_text('{"truncated": ', encoding="utf-8")
        p = self._plan()
        self.assertEqual(len(p["invalid"]), 1)
        self.assertIn("optimize_bad.json", p["invalid"][0])
        self.assertNotIn("optimize_bad.json", p["new"])
        self.assertNotIn("optimize_bad.json", p["changed"])

    def test_a_record_missing_from_the_store_is_reported_not_deleted(self):
        """Deleting evidence is a decision, not a side effect."""
        name = "optimize_1.json"
        self.sb.write_record(self.sb.source, name, {"ok": True})
        shutil.copy2(self.sb.source / name, self.sb.dest / name)
        self.sb.commit_all()
        (self.sb.source / name).unlink()
        p = self._plan()
        self.assertEqual(p["missing"], [name])
        self.assertTrue((self.sb.dest / name).exists(),
                        "the committed copy must not be deleted")

    def test_a_non_record_file_in_the_repo_is_not_a_missing_record(self):
        """The directory's README is a note about the records, not one."""
        (self.sb.dest / "README.md").write_text("notes", encoding="utf-8")
        self.sb.commit_all()
        p = self._plan()
        self.assertEqual(p["missing"], [])

    def test_a_missing_store_is_an_error_not_an_empty_plan(self):
        with self.assertRaises(sync.SyncError):
            sync.plan(self.sb.tmp / "no_such_store", self.sb.dest, self.sb.repo)

    def test_an_empty_store_reports_rather_than_refuses(self):
        """An empty store is not a broken path — it is either a fresh repo
        with nothing to sync, or records that were moved out of the store.
        The second is worth REPORTING, and refusing over it would hide the
        one case the operator needs to see."""
        p = self._plan()
        self.assertEqual(p["new"], [])
        self.assertEqual(p["changed"], [])
        self.assertEqual(p["missing"], [])

    def test_an_empty_store_with_tracked_records_reports_them_missing(self):
        name = "optimize_1.json"
        self.sb.write_record(self.sb.source, name, {"ok": True})
        shutil.copy2(self.sb.source / name, self.sb.dest / name)
        self.sb.commit_all()
        (self.sb.source / name).unlink()
        p = self._plan()
        self.assertEqual(p["missing"], [name])


class TestLineEndingsDoNotFakeAChange(unittest.TestCase):
    """The live store's records are CRLF; the committed blobs are LF.

    Comparing raw bytes would report every record as changed on every run,
    which is worse than useless — it would bury a real change in forty false
    ones. The comparison therefore asks GIT what blob it would store, so the
    repo's own normalisation rules decide.
    """

    def setUp(self):
        self.sb = _Sandbox()
        self.addCleanup(self.sb.cleanup)

    def test_crlf_in_the_store_is_not_a_change(self):
        name = "optimize_1.json"
        p = self.sb.source / name
        p.write_text(json.dumps({"ok": True}, indent=2), encoding="utf-8")
        # The same content with Windows line endings.
        p.write_bytes(p.read_text(encoding="utf-8").replace("\n", "\r\n")
                      .encode("utf-8"))
        shutil.copy2(p, self.sb.dest / name)
        self.sb.commit_all()
        plan = sync.plan(self.sb.source, self.sb.dest, self.sb.repo)
        self.assertEqual(plan["changed"], [],
                         "a line-ending difference is not a content change")

    def test_a_real_content_change_is_still_seen(self):
        name = "optimize_1.json"
        p = self.sb.source / name
        p.write_text(json.dumps({"ok": True}, indent=2), encoding="utf-8")
        shutil.copy2(p, self.sb.dest / name)
        self.sb.commit_all()
        p.write_text(json.dumps({"ok": False}, indent=2), encoding="utf-8")
        plan = sync.plan(self.sb.source, self.sb.dest, self.sb.repo)
        self.assertEqual(plan["changed"], [name])


class TestTheCommitTouchesOnlyMeasurements(unittest.TestCase):
    """The operator may have source edits in flight."""

    def setUp(self):
        self.sb = _Sandbox()
        self.addCleanup(self.sb.cleanup)

    def test_only_the_records_are_staged(self):
        """A source file the operator has edited must not be swept in."""
        (self.sb.repo / "harness").mkdir()
        (self.sb.repo / "harness" / "levers.py").write_text("x = 1\n", encoding="utf-8")
        (self.sb.repo / "README.md").write_text("hello\n", encoding="utf-8")
        self.sb.commit_all("initial")
        # The operator edits a source file and does not commit it.
        (self.sb.repo / "harness" / "levers.py").write_text("x = 2\n", encoding="utf-8")
        # A new record arrives.
        self.sb.write_record(self.sb.source, "optimize_new.json", {"ok": True})
        shutil.copy2(self.sb.source / "optimize_new.json",
                     self.sb.dest / "optimize_new.json")
        ok, out = sync.commit(self.sb.repo, ["optimize_new.json"], "sync")
        self.assertTrue(ok, out)
        committed = _git(self.sb.repo, "show", "--name-only", "--format=", "HEAD")
        self.assertIn("measurements/optimize_new.json", committed)
        self.assertNotIn("harness/levers.py", committed)
        # And the operator's edit is still uncommitted, as it was.
        status = _git(self.sb.repo, "status", "--short")
        self.assertIn("harness/levers.py", status)


    def test_nothing_to_commit_is_success_not_failure(self):
        """The copies were current after all.

        This is the happy case, not an error: the plan saw a change, the
        copy happened, and git found nothing to record because the
        committed state already matched. Reporting that as a failure would
        leave the operator thinking something went wrong.
        """
        name = "optimize_1.json"
        self.sb.write_record(self.sb.source, name, {"ok": True})
        shutil.copy2(self.sb.source / name, self.sb.dest / name)
        self.sb.commit_all()
        ok, out = sync.commit(self.sb.repo, [name], "sync")
        self.assertTrue(ok, out)
        self.assertIn("already current", out)


class TestTheCommitMessageSaysWhatChanged(unittest.TestCase):
    def test_it_names_the_new_and_the_changed(self):
        msg = sync.build_message(["optimize_new.json"],
                                 ["optimize_edited.json"])
        self.assertIn("optimize_new.json", msg)
        self.assertIn("optimize_edited.json", msg)
        self.assertIn("New records", msg)
        self.assertIn("Changed records", msg)

    def test_it_explains_why_the_copies_exist(self):
        msg = sync.build_message([], [])
        self.assertIn("E:\\OptimizerV12\\measurements", msg)

    def test_it_carries_the_attribution(self):
        msg = sync.build_message([], [])
        self.assertIn("Co-Authored-By:", msg)


class TestTheCopyIsFaithful(unittest.TestCase):
    def setUp(self):
        self.sb = _Sandbox()
        self.addCleanup(self.sb.cleanup)

    def test_the_copy_is_byte_identical(self):
        self.sb.write_record(self.sb.source, "optimize_1.json",
                             {"ok": True, "n": [1, 2, 3]})
        sync.copy_files(self.sb.source, self.sb.dest, ["optimize_1.json"])
        self.assertEqual((self.sb.dest / "optimize_1.json").read_bytes(),
                         (self.sb.source / "optimize_1.json").read_bytes())

    def test_the_copy_lands_where_the_reader_will_look(self):
        """measurements/, next to the code, not back on E:."""
        self.assertEqual(sync.DEST.name, "measurements")
        self.assertEqual(sync.DEST, REPO / "measurements")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTheCampaignCommit(unittest.TestCase):
    """One commit, saying what the campaign found.

    A record that is committed on its own, in a commit that only says
    "sync the measurement records", tells a reader of the history nothing
    about what was measured. The campaign's own record should land in a
    commit that says what the campaign found.
    """

    def setUp(self):
        self.sb = _Sandbox()
        self.addCleanup(self.sb.cleanup)

    def _campaign_record(self) -> dict:
        return {
            "request": "render",
            "objective": "karma-xpu",
            "results": {
                "aov_variant": {
                    "ok": True,
                    "effect": 0.0979,
                    "p_value": 3.72e-05,
                    "n_baseline": 5.0,
                    "drift": -0.0182,
                    "control_spread": 0.027,
                    "contention_events": 0,
                    "variant": "beauty_only",
                }
            },
        }

    def test_the_message_names_what_the_campaign_found(self):
        msg = sync.build_campaign_message(self._campaign_record(), [], [])
        self.assertIn("karma-xpu", msg)
        self.assertIn("render", msg)
        self.assertIn("aov_variant", msg)
        self.assertIn("+9.79%", msg)
        self.assertIn("beauty_only", msg)

    def test_the_message_reads_the_effect_out_of_the_record(self):
        """Not a paraphrase the caller supplies — the record itself, so the
        message cannot claim something the record does not say."""
        msg = sync.build_campaign_message(self._campaign_record(), [], [])
        self.assertIn("9.79", msg)
        # A record with a different effect must produce a different message.
        other = self._campaign_record()
        other["results"]["aov_variant"]["effect"] = 0.5
        self.assertIn("50.00%", sync.build_campaign_message(other, [], []))

    def test_the_message_lists_the_records(self):
        msg = sync.build_campaign_message(
            self._campaign_record(),
            ["optimize_20261105_120000.json"],
            ["levers_20261103_221724.json"])
        self.assertIn("optimize_20261105_120000.json", msg)
        self.assertIn("levers_20261103_221724.json", msg)

    def test_staging_and_committing_lands_one_commit(self):
        """The campaign's record and an edited retraction, one commit."""
        name = "levers_20261103_221724.json"
        # A retraction in place: the record is committed, then EDITED.
        self.sb.write_record(self.sb.source, name, {"verdict": "ADMIT +16.91%"})
        shutil.copy2(self.sb.source / name, self.sb.dest / name)
        self.sb.commit_all()
        self.sb.write_record(self.sb.source, name, {"verdict": "RETRACTED - was: ADMIT +16.91%"})

        # And the campaign's own new record.
        self.sb.write_record(self.sb.source, "optimize_20261105_120000.json",
                             self._campaign_record())

        p = sync.sync_and_stage(self.sb.source, self.sb.dest, self.sb.repo)
        self.assertEqual(p["new"], ["optimize_20261105_120000.json"])
        self.assertEqual(p["changed"], [name])
        # Staged, not committed.
        self.assertEqual(len(_git(self.sb.repo, "log", "--oneline").strip().splitlines()), 1)

        record = json.loads((self.sb.dest / "optimize_20261105_120000.json")
                            .read_text(encoding="utf-8"))
        msg = sync.build_campaign_message(record, p["new"], p["changed"])
        ok, out = sync.commit_staged(self.sb.repo, msg)
        self.assertTrue(ok, out)

        # ONE new commit, containing both records.
        log = _git(self.sb.repo, "log", "--oneline")
        self.assertEqual(len(log.strip().splitlines()), 2, "expected one new commit")
        committed = _git(self.sb.repo, "show", "--name-only", "--format=", "HEAD")
        self.assertIn("measurements/optimize_20261105_120000.json", committed)
        self.assertIn("measurements/levers_20261103_221724.json", committed)

    def test_only_measurements_is_committed_even_with_other_staged_work(self):
        """The operator may have something else staged. A measurement run
        must not sweep it into the commit."""
        (self.sb.repo / "harness").mkdir()
        (self.sb.repo / "harness" / "levers.py").write_text("x = 1\n", encoding="utf-8")
        self.sb.commit_all("initial")
        # The operator stages a source edit.
        (self.sb.repo / "harness" / "levers.py").write_text("x = 2\n", encoding="utf-8")
        _git(self.sb.repo, "add", "--", "harness/levers.py")

        self.sb.write_record(self.sb.source, "optimize_20261105_120000.json",
                             self._campaign_record())
        p = sync.sync_and_stage(self.sb.source, self.sb.dest, self.sb.repo)
        ok, out = sync.commit_staged(self.sb.repo, "campaign")
        self.assertTrue(ok, out)

        committed = _git(self.sb.repo, "show", "--name-only", "--format=", "HEAD")
        self.assertIn("measurements/optimize_20261105_120000.json", committed)
        self.assertNotIn("harness/levers.py", committed,
                         "the operator's staged edit must not be swept in")
        # And it is still staged, exactly as it was.
        staged = _git(self.sb.repo, "diff", "--cached", "--name-only")
        self.assertIn("harness/levers.py", staged)

    def test_committing_nothing_staged_is_not_a_failure(self):
        ok, out = sync.commit_staged(self.sb.repo, "campaign")
        self.assertTrue(ok, out)
        self.assertIn("nothing staged", out)
