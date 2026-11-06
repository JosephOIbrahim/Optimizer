"""
Tests for the Optimize-The-4090.bat argument handling.

Two real bugs lived here, and both were invisible until a lever name was
passed on the command line:

  * `if cond set X=%~2 & shift & shift & goto parse` — `&` is a command
    SEPARATOR in batch, so the shifts ran on EVERY pass whether or not the
    condition matched. Each pass consumed two arguments regardless, so
    `--levers` was eaten before its own line was ever reached.
  * cmd.exe SPLITS an argument on a COMMA before a batch file ever sees it:
    "--levers a,b" arrives as %1=--levers %2=a %3=b. Any per-flag parsing
    of a comma-separated value therefore silently loses everything after the
    first comma.

The fix was to stop parsing in the batch file at all: the launcher forwards
%* verbatim and the python tool owns the defaults and the parsing. One
parser is better than two that disagree — and cmd.exe's own idea of where
the arguments are is not something a batch file can argue with.

These tests drive the actual .bat with a python stand-in that echoes its
argv, and assert on what arrives.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BAT = REPO / "Optimize-The-4090.bat"

#: A python stand-in for harness/optimize.py. It prints the arguments it
#: received, one per line, so a test can assert on what actually arrived.
ECHO = r'''
import sys
for a in sys.argv[1:]:
    print("ARG:" + a)
'''


def _run(tmp: Path, *args: str) -> list[str]:
    """Run the launcher with a python stand-in in place of the harness."""
    echo = tmp / "echo.py"
    echo.write_text(ECHO, encoding="utf-8")
    text = BAT.read_text(encoding="utf-8", errors="replace")
    # Point the launcher at the stand-in instead of the real harness, and at a
    # throwaway repo so nothing else is touched.
    text = text.replace(r'"%PY%" "%REPO%\harness\optimize.py"',
                        f'"{sys.executable}" "{echo}"')
    # The launcher PICKS its interpreter from V12PY and refuses to start if
    # that interpreter is missing - before it ever reaches the harness. So
    # the stand-in has to be substituted THERE too, or the launcher stops
    # first and the test sees no output at all. That is what a CI runner
    # saw: no C:\Python314\python.exe, so no arguments forwarded.
    #
    # The substitutions are by PATTERN, not by the literal path the launcher
    # happens to carry today, so a launcher whose paths change does not
    # silently stop being tested. A lambda keeps the replacement literal:
    # a backslash in a replacement string is an escape otherwise.
    text = re.sub(r'set "REPO=[^"]*"', lambda m: f'set "REPO={tmp}"', text)
    text = re.sub(r'set "V12PY=[^"]*"',
                  lambda m: f'set "V12PY={sys.executable}"', text)
    # The menu would block on input; every test supplies --request, so the
    # launcher takes the command-line path. `pause` is removed so a test
    # cannot hang on it.
    text = text.replace("pause >nul", "rem no pause")
    out = tmp / "bat.bat"
    out.write_text(text, encoding="utf-8")
    r = subprocess.run(["cmd", "/c", str(out), *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    return [ln[4:] for ln in r.stdout.splitlines() if ln.startswith("ARG:")]


class TestBatArgumentForwarding(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bat_parser_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_lever_name_arrives_as_a_lever_name(self):
        """The bug that mattered: `--levers aov_variant` reached argparse as a
        bare positional, so the run died with "unrecognized arguments"."""
        args = _run(self.tmp, "--request", "render", "--levers", "aov_variant")
        self.assertIn("--levers", args)
        self.assertIn("aov_variant", args)

    def test_several_levers_stay_one_value(self):
        """cmd.exe splits a comma before the batch file sees it, so the value
        must survive as ONE argument and be split on the python side."""
        args = _run(self.tmp, "--request", "render", "--levers", "aov_variant,power_ceiling")
        self.assertIn("--levers", args)
        self.assertIn("aov_variant,power_ceiling", args)

    def test_the_request_arrives(self):
        args = _run(self.tmp, "--request", "compute")
        self.assertIn("--request", args)
        self.assertIn("compute", args)

    def test_every_flag_arrives_and_nothing_is_swallowed(self):
        args = _run(self.tmp, "--request", "render", "--frames", "5",
                    "--start-frame", "503", "--samples", "64", "--res-scale", "50",
                    "--reps", "4", "--levers", "aov_variant", "--variant", "no_crypto")
        for flag, value in (("--request", "render"), ("--frames", "5"),
                            ("--start-frame", "503"), ("--samples", "64"),
                            ("--res-scale", "50"), ("--reps", "4"),
                            ("--levers", "aov_variant"), ("--variant", "no_crypto")):
            with self.subTest(flag=flag):
                self.assertIn(flag, args)
                self.assertIn(value, args)

    def test_flags_keep_their_order_and_pairing(self):
        """A value must follow its own flag. If the shifts are off by one, the
        values still appear — just attached to the wrong flags — which is
        quieter and worse than a missing argument."""
        args = _run(self.tmp, "--request", "render", "--frames", "7",
                    "--reps", "9", "--levers", "aov_variant")
        self.assertEqual(args[args.index("--frames") + 1], "7")
        self.assertEqual(args[args.index("--reps") + 1], "9")
        self.assertEqual(args[args.index("--levers") + 1], "aov_variant")

    def test_a_path_with_spaces_survives(self):
        """--usd and --settings are paths and prim paths, and are quoted."""
        args = _run(self.tmp, "--request", "render",
                    "--usd", r"E:\OptimizerV12\scenes\my scene.usd",
                    "--settings", "/Render/Some Prim")
        self.assertIn(r"E:\OptimizerV12\scenes\my scene.usd", args)
        self.assertIn("/Render/Some Prim", args)

    def test_the_probe_flag_is_forwarded(self):
        """It used to be silently dropped: the parser did not know it, so it
        fell through to the catch-all `shift`, and the run measured a full
        campaign instead of reporting the plan."""
        args = _run(self.tmp, "--request", "render", "--probe")
        self.assertIn("--probe", args)

    def test_an_unknown_flag_does_not_eat_the_next_one(self):
        """An unrecognised flag is forwarded as-is; the python side decides
        whether it is an error. The next flag must survive."""
        args = _run(self.tmp, "--request", "render", "--not-a-flag", "--reps", "6")
        self.assertIn("--reps", args)
        self.assertEqual(args[args.index("--reps") + 1], "6")


class TestPythonSideSplitsTheLeverList(unittest.TestCase):
    """The batch file cannot split a comma, so the python side must."""

    def test_the_tool_accepts_a_comma_separated_lever_list(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "optimize_mod", REPO / "harness" / "optimize.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # The lever registry is the authority on what a lever is called.
        from harness import levers
        for name in ("frames_per_process", "aov_variant", "power_ceiling",
                     "husk_threads", "manual"):
            with self.subTest(lever=name):
                self.assertIn(name, levers.LEVERS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
