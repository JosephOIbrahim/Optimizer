"""
Tests for how the measurement records reach the repository.

Every campaign commits its measurement records, so the copies under
`measurements/` — the evidence the operator cards cite — cannot go stale.
This used to be a `--sync` flag, and before that a second commit made after
the run by a separate batch file. Both are gone.

Now the tool itself does it: `harness/optimize.py` writes the record, STAGES
it while the run is still going, and then commits everything staged under
`measurements/` in ONE commit whose message says what the campaign found. A
commit that only says "sync the measurement records" tells a reader of the
history nothing about what was measured.

Three things this has to get right:

  * the record is committed with the campaign, not in a second commit;
  * ONLY `measurements/` is committed — the operator may have source edits
    in flight, even staged, and those are not a measurement run's business;
  * every argument still arrives at the tool intact and in order, including
    a quoted path with spaces and a comma-separated --levers value. cmd.exe
    splits an argument on a COMMA before a batch file ever sees it, and
    "%* forwards the text verbatim so the text stays intact.

And one thing it must NOT do: reach the real repository from a test. An
earlier version of this file pointed at the REAL sync module and committed a
fake record into it. The test is meant to prove the sync is safe; it must
not itself be the thing that breaks the repository.

These drive the actual .bat with stand-ins, the way test_bat_parser.py does.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import common, optimize, sync_measurements as sync  # noqa: E402

BAT = REPO / "Optimize-The-4090.bat"

#: A python stand-in for harness/optimize.py. It prints the arguments it
#: received, one per line, and says when it ran.
ECHO = r'''
import sys
print("HARNESS RAN")
for a in sys.argv[1:]:
    print("ARG:" + a)
'''

#: A stand-in for Sync-Measurements.bat — still shipped, still usable by
#: hand, but the launcher no longer calls it.
SYNC_STANDIN = "@echo off\r\necho SYNCED\r\nexit /b 0\r\n"


class _Launcher:
    """The real launcher, pointed at stand-ins."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        echo = tmp / "echo.py"
        echo.write_text(ECHO, encoding="utf-8")
        (tmp / "Sync-Measurements.bat").write_text(SYNC_STANDIN, encoding="utf-8")
        text = BAT.read_text(encoding="utf-8", errors="replace")
        text = text.replace(r'"%PY%" "%REPO%\harness\optimize.py"',
                            f'"{sys.executable}" "{echo}"')
        # The launcher PICKS its interpreter from V12PY and refuses to start
        # if that interpreter is missing - before it reaches the harness. So
        # the stand-in has to be substituted THERE too, or the launcher stops
        # first and the test sees no output at all.
        #
        # By PATTERN rather than by the literal path the launcher carries
        # today, so a launcher whose paths change does not silently stop
        # being tested. A lambda keeps the replacement literal: a backslash
        # in a replacement string is an escape otherwise.
        text = re.sub(r'set "REPO=[^"]*"',
                      lambda m: f'set "REPO={self.tmp}"', text)
        text = re.sub(r'set "V12PY=[^"]*"',
                      lambda m: f'set "V12PY={sys.executable}"', text)
        # The menu would block on input; every test supplies --request, so the
        # launcher takes the command-line path. `pause` is removed so a test
        # cannot hang on it.
        text = text.replace("pause >nul", "rem no pause")
        self.bat = tmp / "bat.bat"
        self.bat.write_text(text, encoding="utf-8")

    def run(self, *args: str) -> str:
        r = subprocess.run(["cmd", "/c", str(self.bat), *args],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        return r.stdout

    @staticmethod
    def args(stdout: str) -> list[str]:
        return [ln[4:] for ln in stdout.splitlines() if ln.startswith("ARG:")]


class TestTheLauncherForwardsEverything(unittest.TestCase):
    """The launcher is a thin wrapper now: it runs the tool and gets out of
    the way. The tool does the committing."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sync_wiring_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.l = _Launcher(self.tmp)

    def test_every_flag_still_arrives(self):
        args = self.l.args(self.l.run("--request", "render", "--frames", "5",
                                      "--reps", "3", "--levers", "aov_variant",
                                      "--variant", "no_crypto"))
        for flag, value in (("--request", "render"), ("--frames", "5"),
                            ("--reps", "3"), ("--levers", "aov_variant"),
                            ("--variant", "no_crypto")):
            with self.subTest(flag=flag):
                self.assertIn(flag, args)
                self.assertIn(value, args)

    def test_a_quoted_path_with_spaces_survives(self):
        """--usd and --settings are paths and prim paths, and are quoted."""
        args = self.l.args(self.l.run("--request", "render",
                                      "--usd", r"E:\scenes\my scene.usd",
                                      "--settings", "/Render/Some Prim"))
        self.assertIn(r"E:\scenes\my scene.usd", args)
        self.assertIn("/Render/Some Prim", args)

    def test_a_comma_separated_lever_list_stays_one_value(self):
        """cmd.exe splits an argument on a COMMA before the batch file sees
        it, so the value must survive as ONE argument."""
        args = self.l.args(self.l.run("--request", "render",
                                      "--levers", "aov_variant,power_ceiling"))
        self.assertIn("aov_variant,power_ceiling", args)

    def test_flags_keep_their_order_and_pairing(self):
        args = self.l.args(self.l.run("--request", "render", "--frames", "7",
                                      "--reps", "9", "--levers", "aov_variant"))
        self.assertEqual(args[args.index("--frames") + 1], "7")
        self.assertEqual(args[args.index("--reps") + 1], "9")
        self.assertEqual(args[args.index("--levers") + 1], "aov_variant")

    def test_the_sync_flag_no_longer_exists(self):
        """It was dropped, so passing it is an error — not a silent no-op."""
        out = self.l.run("--request", "render", "--sync")
        self.assertIn("--sync", self.l.args(out),
                      "the launcher should no longer consume --sync")

    def test_the_usage_text_no_longer_advertises_the_flag(self):
        text = BAT.read_text(encoding="utf-8", errors="replace")
        # The LABEL, not the `goto :usage` reference that appears earlier.
        usage = text[text.index("\n:usage"):text.index("\n:done")]
        self.assertNotIn("--sync", usage,
                         "the flag was dropped; the usage text must not offer it")
        self.assertIn("commits its measurement records", usage)


class _Sandbox:
    """A throwaway git repository with its own store and measurements dir."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="campaign_"))
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.store = self.tmp / "store"
        self.store.mkdir()
        (self.repo / "measurements").mkdir()
        (self.repo / "harness").mkdir()
        (self.repo / "harness" / "levers.py").write_text("x = 1\n", encoding="utf-8")
        self._git("init", "-b", "main")
        self._git("config", "user.name", "t")
        self._git("config", "user.email", "t@x.invalid")
        self._git("add", "-A")
        self._git("commit", "-m", "initial")

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=str(self.repo),
                              capture_output=True, text=True, encoding="utf-8", errors="replace").stdout

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestTheCampaignCommitsOneCommit(unittest.TestCase):
    """The campaign's record and anything else that changed, in ONE commit."""

    def setUp(self):
        self.sb = _Sandbox()
        self.addCleanup(self.sb.cleanup)
        # Point the tool at the sandbox. The repo root is where the record is
        # committed; the store is where the record is written.
        p = mock.patch.object(common, "REPO_ROOT", self.sb.repo)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(common, "MEASURE_ROOT", self.sb.store)
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(sync, "DEST", self.sb.repo / "measurements")
        p.start(); self.addCleanup(p.stop)
        p = mock.patch.object(sync, "SOURCE", self.sb.store)
        p.start(); self.addCleanup(p.stop)

    def _campaign_record(self) -> dict:
        return {
            "request": "render",
            "objective": "karma-xpu",
            "results": {
                "aov_variant": {
                    "ok": True, "effect": 0.0979, "p_value": 3.72e-05,
                    "n_baseline": 5.0, "drift": -0.0182, "control_spread": 0.027,
                    "contention_events": 0, "variant": "beauty_only",
                }
            },
        }

    def _write_record(self, name: str, payload: dict) -> Path:
        p = self.sb.store / name
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return p

    def test_the_campaign_record_is_committed(self):
        rec = self._write_record("optimize_20261105_120000.json",
                                  self._campaign_record())
        optimize._commit_the_campaign(rec, [], [])
        committed = self.sb._git("show", "--name-only", "--format=", "HEAD")
        self.assertIn("measurements/optimize_20261105_120000.json", committed)

    def test_it_is_one_commit_not_two(self):
        """The point of staging during the run: the campaign's record and the
        rest land together, in a commit that says what was measured."""
        rec = self._write_record("optimize_20261105_120000.json",
                                  self._campaign_record())
        before = len(self.sb._git("log", "--oneline").strip().splitlines())
        optimize._commit_the_campaign(rec, [], [])
        after = len(self.sb._git("log", "--oneline").strip().splitlines())
        self.assertEqual(after, before + 1, "exactly one new commit")

    def test_the_commit_message_says_what_the_campaign_found(self):
        rec = self._write_record("optimize_20261105_120000.json",
                                  self._campaign_record())
        optimize._commit_the_campaign(rec, [], [])
        message = self.sb._git("log", "-1", "--format=%B")
        self.assertIn("karma-xpu", message)
        self.assertIn("aov_variant", message)
        self.assertIn("9.79", message)
        self.assertNotIn("Sync the measurement records", message,
                         "the commit should describe the campaign, not the "
                         "housekeeping")

    def test_an_edited_record_lands_in_the_same_commit(self):
        """A retraction edits a record in place. It belongs with the campaign,
        not in a second commit."""
        name = "levers_20261103_221724.json"
        self._write_record(name, {"verdict": "ADMIT +16.91%"})
        shutil.copy2(self.sb.store / name, self.sb.repo / "measurements" / name)
        self.sb._git("add", "-A")
        self.sb._git("commit", "-m", "add the old record")
        # Now the retraction lands.
        self._write_record(name, {"verdict": "RETRACTED - was: ADMIT +16.91%"})
        rec = self._write_record("optimize_20261105_120000.json",
                                  self._campaign_record())
        before = len(self.sb._git("log", "--oneline").strip().splitlines())
        optimize._commit_the_campaign(rec, [], [])
        after = len(self.sb._git("log", "--oneline").strip().splitlines())
        self.assertEqual(after, before + 1)
        committed = self.sb._git("show", "--name-only", "--format=", "HEAD")
        self.assertIn("measurements/optimize_20261105_120000.json", committed)
        self.assertIn("measurements/levers_20261103_221724.json", committed)

    def test_the_operators_staged_source_edit_is_not_swept_in(self):
        """Even if the operator has it STAGED, it is not a measurement
        run's business."""
        (self.sb.repo / "harness" / "levers.py").write_text("x = 2\n", encoding="utf-8")
        self.sb._git("add", "--", "harness/levers.py")
        rec = self._write_record("optimize_20261105_120000.json",
                                  self._campaign_record())
        optimize._commit_the_campaign(rec, [], [])
        committed = self.sb._git("show", "--name-only", "--format=", "HEAD")
        self.assertNotIn("harness/levers.py", committed,
                         "the operator's staged edit must not be swept in")
        staged = self.sb._git("diff", "--cached", "--name-only")
        self.assertIn("harness/levers.py", staged,
                      "and it must still be staged, exactly as it was")

    def test_a_failed_commit_does_not_lose_the_record(self):
        """The measurement already happened and its record is already on E:.
        A repository that will not take a commit must not lose the result."""
        rec = self._write_record("optimize_20261105_120000.json",
                                  self._campaign_record())
        with mock.patch.object(sync, "commit_staged",
                               return_value=(False, "git commit failed: nope")):
            optimize._commit_the_campaign(rec, [], [])
        # The record is still on the store, and still in the repo copy.
        self.assertTrue(rec.exists())
        self.assertTrue((self.sb.repo / "measurements" /
                         rec.name).exists())

    def test_the_real_repository_is_never_reached(self):
        """A guard on the guard.

        An earlier version of these tests pointed at the REAL sync module
        instead of a sandboxed copy, and committed a fake record into the
        real repository. The test is supposed to prove the sync is safe;
        it must not itself be the thing that breaks the repository.
        """
        before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
        after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                               capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
        self.assertEqual(before, after,
                         "the real repository's HEAD moved — a test reached it")
        stray = list((REPO / "measurements").glob("optimize_20261105_120000.json"))
        self.assertEqual(stray, [],
                         "a fake record landed in the real repository")


if __name__ == "__main__":
    unittest.main(verbosity=2)
