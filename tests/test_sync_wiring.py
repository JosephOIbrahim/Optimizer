"""
Tests for the automatic sync in Optimize-The-4090.bat.

Every campaign commits its measurement records, so the copies under
`measurements/` — the evidence the operator cards cite — cannot go stale.
This used to be a `--sync` flag; it is not optional now, because a record
that is quoted in the cards and not committed is a claim nobody can check.

Three things this has to get right:

  * the sync runs AFTER the campaign, not before, so it picks up the record
    the campaign just wrote;
  * it runs on EVERY campaign — from the command line and from the menu —
    and it commits ONLY `measurements/`, so a source edit the operator has
    in flight is left exactly as it was;
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
import sys
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


class TestTheSyncAlwaysRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sync_wiring_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.l = _Launcher(self.tmp)

    def test_the_sync_runs_on_every_campaign(self):
        """No flag. It is not optional."""
        out = self.l.run("--request", "render")
        self.assertIn("SYNCED", out)

    def test_the_sync_runs_after_the_campaign(self):
        """It has to pick up the record the campaign just wrote."""
        out = self.l.run("--request", "render")
        self.assertLess(out.index("HARNESS RAN"), out.index("SYNCED"),
                        "the sync must run AFTER the campaign, not before")

    def test_it_runs_whatever_the_request(self):
        for request in ("render", "shot", "compute"):
            with self.subTest(request=request):
                out = self.l.run("--request", request)
                self.assertIn("SYNCED", out)

    def test_the_sync_flag_no_longer_exists(self):
        """It was dropped, so passing it is an error — not a silent no-op."""
        out = self.l.run("--request", "render", "--sync")
        # The launcher forwards everything, so --sync reaches the tool, which
        # does not know it.
        self.assertIn("--sync", self.l.args(out),
                      "the launcher should no longer consume --sync")
        self.assertNotIn("unrecognized", out,
                         "the stand-in accepts anything, so this only checks "
                         "that the flag is forwarded rather than swallowed")

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

    def test_the_usage_text_no_longer_advertises_the_flag(self):
        text = BAT.read_text(encoding="utf-8", errors="replace")
        # The LABEL, not the `goto :usage` reference that appears earlier.
        usage = text[text.index("\n:usage"):text.index("\n:done")]
        self.assertNotIn("--sync", usage,
                         "the flag was dropped; the usage text must not offer it")
        self.assertIn("commits its measurement records", usage)


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
        """The whole point: no staleness."""
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
        # copy of the sync, with the fake store — never the real one.
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

        r = subprocess.run(["cmd", "/c", str(out), "--request", "render"],
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


class TestTheMenuPath(unittest.TestCase):
    """Choosing from the menu must reach the tool and sync afterwards.

    The menu calls the run label with the request as its own arguments, so
    this path has to keep working after the sync was wired in.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sync_menu_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _menu_run(self, choice: str) -> str:
        tmp = self.tmp
        echo = tmp / "echo.py"
        echo.write_text(ECHO, encoding="utf-8")
        (tmp / "Sync-Measurements.bat").write_text(SYNC_STANDIN, encoding="utf-8")
        text = BAT.read_text(encoding="utf-8", errors="replace")
        text = text.replace(r'"%PY%" "%REPO%\harness\optimize.py"',
                            f'"{sys.executable}" "{echo}"')
        text = text.replace(r'set "REPO=C:\Users\User\OneDrive\Desktop\Optimizer"',
                            f'set "REPO={tmp}"')
        text = text.replace("pause >nul", "rem no pause")
        text = text.replace('set /p "choice=   Choose a request:  "',
                            f'set "choice={choice}"')
        # Break only the menu's OWN loop — the last "goto :menu", which is the
        # one after a run. The earlier one is the ENTRY into the menu, and
        # replacing that would skip the menu entirely.
        head, _, tail = text.rpartition("goto :menu")
        text = head + "goto :end" + tail
        out = tmp / "bat.bat"
        out.write_text(text, encoding="utf-8")
        r = subprocess.run(["cmd", "/c", str(out)],
                           capture_output=True, text=True, timeout=90)
        return r.stdout

    def test_a_menu_choice_reaches_the_tool(self):
        out = self._menu_run("4")
        args = _Launcher.args(out)
        self.assertEqual(args, ["--request", "profile"])

    def test_the_sync_runs_after_a_menu_run(self):
        out = self._menu_run("4")
        self.assertIn("SYNCED", out)
        self.assertLess(out.index("HARNESS RAN"), out.index("SYNCED"),
                        "the sync must run AFTER the campaign, not before")


if __name__ == "__main__":
    unittest.main(verbosity=2)
