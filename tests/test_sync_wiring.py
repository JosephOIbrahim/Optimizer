"""
Tests for the --sync wiring in Optimize-The-4090.bat.

`--sync` refreshes the committed measurement records after a campaign, so the
copies under `measurements/` cannot go stale. It is a FLAG, not the default:
a measurement run must not commit to the operator's repository without them
asking for it.

Three things this has to get right:

  * `--sync` is CONSUMED by the launcher. The python tool has never heard of
    it, and an unknown flag on that command line is an error.
  * every OTHER argument still arrives, intact and in order — including a
    quoted path with spaces. The flag is removed from the forwarded text as
    a substring rather than by rebuilding a token list, because rebuilding
    means re-quoting every argument by hand, and a path with spaces is
    exactly what that would break.
  * the sync runs AFTER the campaign, not before, so it picks up the record
    the campaign just wrote.

And one thing it must NOT do: sweep in a source edit the operator has in
flight. The sync commits only `measurements/`, so an edited `levers.py` is
left alone. That property is tested in test_sync_measurements.py, against
the sync itself; this file checks the wiring does not defeat it.

These drive the actual .bat with stand-ins, the way test_bat_parser.py does.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BAT = REPO / "Optimize-The-4090.bat"

#: A python stand-in for harness/optimize.py. It prints the arguments it
#: received, one per line, and says when it ran.
ECHO = r'''
import sys, time
print("HARNESS RAN")
for a in sys.argv[1:]:
    print("ARG:" + a)
'''

#: A stand-in for Sync-Measurements.bat.
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
        text = text.replace(r'set "REPO=C:\Users\User\OneDrive\Desktop\Optimizer"',
                            f'set "REPO={tmp}"')
        # The menu would block on input; every test supplies --request, so the
        # launcher takes the command-line path. `pause` is removed so a test
        # cannot hang on it.
        text = text.replace("pause >nul", "rem no pause")
        self.bat = tmp / "bat.bat"
        self.bat.write_text(text, encoding="utf-8")

    def run(self, *args: str) -> str:
        r = subprocess.run(["cmd", "/c", str(self.bat), *args],
                           capture_output=True, text=True, timeout=120)
        return r.stdout

    @staticmethod
    def args(stdout: str) -> list[str]:
        return [ln[4:] for ln in stdout.splitlines() if ln.startswith("ARG:")]


class TestTheSyncFlag(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sync_wiring_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.l = _Launcher(self.tmp)

    def test_sync_is_consumed_not_forwarded(self):
        """The python tool has never heard of --sync; forwarding it would be
        an error on the command line."""
        args = self.l.args(self.l.run("--request", "render", "--sync"))
        self.assertNotIn("--sync", args)
        self.assertIn("--request", args)
        self.assertIn("render", args)

    def test_the_sync_runs_after_the_campaign(self):
        """It has to pick up the record the campaign just wrote."""
        out = self.l.run("--request", "render", "--sync")
        self.assertIn("SYNCED", out)
        self.assertLess(out.index("HARNESS RAN"), out.index("SYNCED"),
                        "the sync must run AFTER the campaign, not before")

    def test_without_the_flag_the_sync_does_not_run(self):
        """A measurement run must not commit to the operator's repository
        without them asking for it."""
        out = self.l.run("--request", "render")
        self.assertNotIn("SYNCED", out)

    def test_every_other_flag_still_arrives(self):
        args = self.l.args(self.l.run("--request", "render", "--frames", "5",
                                      "--reps", "3", "--levers", "aov_variant",
                                      "--variant", "no_crypto", "--sync"))
        for flag, value in (("--request", "render"), ("--frames", "5"),
                            ("--reps", "3"), ("--levers", "aov_variant"),
                            ("--variant", "no_crypto")):
            with self.subTest(flag=flag):
                self.assertIn(flag, args)
                self.assertIn(value, args)

    def test_a_quoted_path_with_spaces_survives_the_strip(self):
        """The flag is removed as a substring, not by rebuilding a token list.

        Rebuilding means re-quoting every argument by hand, and a path with
        spaces is exactly what that would break. This is the reason the
        launcher forwards %* verbatim.
        """
        args = self.l.args(self.l.run("--request", "render",
                                      "--usd", r"E:\scenes\my scene.usd",
                                      "--sync"))
        self.assertIn(r"E:\scenes\my scene.usd", args)
        self.assertNotIn("--sync", args)

    def test_a_comma_separated_lever_list_stays_one_value(self):
        """cmd.exe splits an argument on a COMMA before the batch file sees
        it, so the value must survive as ONE argument."""
        args = self.l.args(self.l.run("--request", "render",
                                      "--levers", "aov_variant,power_ceiling",
                                      "--sync"))
        self.assertIn("aov_variant,power_ceiling", args)

    def test_the_sync_flag_may_appear_in_any_order(self):
        """The detection is over the whole tail, not just the first
        argument."""
        args = self.l.args(self.l.run("--sync", "--request", "render"))
        self.assertNotIn("--sync", args)
        self.assertIn("render", args)

    def test_the_sync_does_not_swallow_a_value(self):
        """Stripping "--sync" must not eat the argument after it."""
        args = self.l.args(self.l.run("--request", "render", "--sync", "--reps", "4"))
        self.assertEqual(args[args.index("--reps") + 1], "4")

    def test_the_usage_text_documents_the_flag(self):
        text = BAT.read_text(encoding="utf-8", errors="replace")
        self.assertIn("--sync", text)


class TestTheSyncRunsAfterTheRealCampaign(unittest.TestCase):
    """The end-to-end shape: a campaign writes a record, then the sync picks
    it up and commits it — without touching anything else."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sync_e2e_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_the_real_repository_is_never_reached(self):
        """A guard on the guard.

        An earlier version of this test pointed at the REAL sync module
        instead of a sandboxed copy, and committed a fake record into the
        real repository. The test is supposed to prove the sync is safe;
        it must not itself be the thing that breaks the repository.
        """
        before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                capture_output=True, text=True).stdout.strip()
        after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                               capture_output=True, text=True).stdout.strip()
        self.assertEqual(before, after,
                         "the real repository's HEAD moved — a test reached it")
        stray = list((REPO / "measurements").glob("optimize_20261105_120000.json"))
        self.assertEqual(stray, [],
                         "a fake record landed in the real repository")

    def test_a_record_written_by_the_campaign_ends_up_committed(self):
        """The whole point: no staleness.

        A stand-in harness writes a record into a fake live store, the sync
        runs, and the record must be committed — and the operator's uncommitted
        source edit must still be uncommitted.
        """
        store = self.tmp / "store"
        store.mkdir()
        repo = self.tmp / "repo"
        repo.mkdir()
        (repo / "measurements").mkdir()
        (repo / "harness").mkdir()
        (repo / "harness" / "levers.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, capture_output=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@x.invalid"], cwd=repo,
                       capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, capture_output=True)
        # The operator has a source edit in flight.
        (repo / "harness" / "levers.py").write_text("x = 2\n", encoding="utf-8")

        # A stand-in harness that writes a record into the fake store.
        echo = self.tmp / "echo.py"
        echo.write_text(
            "import json, pathlib, sys\n"
            f"p = pathlib.Path(r'{store}') / 'optimize_20261105_120000.json'\n"
            "p.write_text(json.dumps({'request':'render','results':{}}))\n"
            "for a in sys.argv[1:]: print('ARG:'+a)\n",
            encoding="utf-8")

        text = BAT.read_text(encoding="utf-8", errors="replace")
        text = text.replace(r'"%PY%" "%REPO%\harness\optimize.py"',
                            f'"{sys.executable}" "{echo}"')
        text = text.replace(r'set "REPO=C:\Users\User\OneDrive\Desktop\Optimizer"',
                            f'set "REPO={repo}"')
        text = text.replace("pause >nul", "rem no pause")
        # The launcher calls Sync-Measurements.bat. Point that at the SANDBOX
        # copy of the sync, with the fake store — never the real one. An
        # earlier version of this test reached the real repository and
        # committed a fake record into it, which is exactly the kind of
        # mistake a test must not be able to make.
        sync_src = (REPO / "harness" / "sync_measurements.py").read_text(encoding="utf-8")
        sync_src = sync_src.replace('REPO_ROOT = HERE.parent',
                                    f'REPO_ROOT = pathlib.Path(r"{repo}")')
        sync_src = sync_src.replace('DEST = REPO_ROOT / "measurements"',
                                    f'DEST = pathlib.Path(r"{repo}") / "measurements"')
        sync_src = sync_src.replace("from pathlib import Path",
                                    "import pathlib\nfrom pathlib import Path")
        (repo / "harness" / "sync_measurements.py").write_text(sync_src, encoding="utf-8")
        text = text.replace(r'call "%REPO%\Sync-Measurements.bat" /nopause',
                            f'"{sys.executable}" "{repo / "harness" / "sync_measurements.py"}" --source "{store}"')
        out = repo / "bat.bat"
        out.write_text(text, encoding="utf-8")

        r = subprocess.run(["cmd", "/c", str(out), "--request", "render", "--sync"],
                           capture_output=True, text=True, timeout=120)
        self.assertIn("copied 1 record(s)", r.stdout,
                      f"the sync should have copied the new record: {r.stdout}")

        # The record is committed.
        committed = subprocess.run(["git", "show", "--name-only", "--format=", "HEAD"],
                                  cwd=repo, capture_output=True, text=True).stdout
        self.assertIn("measurements/optimize_20261105_120000.json", committed)
        # And the operator's source edit is NOT in that commit.
        self.assertNotIn("harness/levers.py", committed)
        # And it is still uncommitted, exactly as it was.
        status = subprocess.run(["git", "status", "--short"], cwd=repo,
                                capture_output=True, text=True).stdout
        self.assertIn("harness/levers.py", status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
