"""
Tests for the paired engine's quiet gate.

The gate exists because of what was observed on this box on 2026-11-04: the
card was quiet when the measurement started, a 10.6 GB process appeared on the
third repetition, and the control drifted -70.9% over the run. The drift gate
refused the result — correctly — but the ten minutes were gone, and the
operator had waited out a one-hour timeout first for a machine that was busy
the whole time.

Checking once at the start is not enough. The gate is called before EVERY
repetition, so a job that appears mid-run pauses the measurement instead of
spoiling it. And the pauses are counted, because a measurement taken across a
contended window is a measurement of something more than the workload, and the
record should say so.

The signal is PROCESS IDENTITY, not utilisation. That is not a preference.
During a Karma render this card is busy BY DESIGN — measured at 94%
utilisation by our own workload — so a utilisation threshold cannot tell our
render from somebody else's. An earlier version of this detector sampled
utilisation and reported a repetition that took 148 s (where a clean frame
takes 48 s) as CLEAN, because our own render had made the card busy.

No GPU, no measurement: the arms are stubs, and the contention probe is
stubbed too.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import common, paired  # noqa: E402


def _quiet(*a, **kw):
    """A contention probe that sees nobody on the card."""
    return []


def _busy(*a, **kw):
    """A contention probe that always sees somebody on the card."""
    return ["nuke.exe:10240MiB"]


class _Scripted:
    """Reports contention for the first N calls, then clean.

    Lets a test say "the first attempt saw a busy card, the retry did not" —
    which is the case the retry logic exists for.
    """

    def __init__(self, busy_count: int):
        self.busy_count = busy_count
        self.made = 0

    def __call__(self, *a, **kw):
        self.made += 1
        return ["nuke.exe:10240MiB"] if self.made <= self.busy_count else []


#: The real probe, captured at IMPORT time — before any test can patch it.
#:
#: It used to be captured per test, into an attribute that a second patch
#: could clobber, and a subclass's setUp that did not call super()'s could
#: skip the cleanup entirely. Either way the module was left patched for the
#: rest of the session, and the leak only showed up once another test file
#: started to depend on the real probe. Capturing it once, here, removes the
#: ordering hazard altogether.
_REAL_PROBE = common.other_gpu_processes


class _ProbeTest(unittest.TestCase):
    """Base: stub the contention probe so the live card's load cannot affect
    the test.

    Without this the tests read whatever else is running on this box, which is
    exactly the variable the production code exists to control.
    """

    def setUp(self):
        self.addCleanup(
            lambda: setattr(common, "other_gpu_processes", _REAL_PROBE))

    def _use(self, probe):
        common.other_gpu_processes = probe


class TestQuietGate(_ProbeTest):
    def setUp(self):
        self._use(_quiet)

    def _arm(self, values, name="arm"):
        it = iter(values)

        def arm(index):
            return {"value": next(it)}
        arm.__name__ = name
        return arm

    def test_the_gate_is_called_before_every_measured_arm(self):
        """Each ARM's measurement should start on a quiet card — not each
        rep, because the two arms of a rep are minutes apart and the second
        one is exactly the sample a wave-rendering job would spoil."""
        calls = []
        p = paired.Pair("t", self._arm([100.0] * 5), self._arm([90.0] * 5),
                        reps=5, warmup=0,
                        quiet_check=lambda: (calls.append(1), True)[1])
        p.run()
        self.assertEqual(len(calls), 10, "one call per arm per rep (2 arms x 5 reps)")

    def test_the_gate_is_not_called_during_the_warmup(self):
        """The warm-up repetition is not a measurement, so a busy machine
        during it does not need to pause the run."""
        calls = []
        p = paired.Pair("t", self._arm([100.0, 100.0, 100.0, 100.0]),
                        self._arm([90.0, 90.0, 90.0, 90.0]), reps=2, warmup=1,
                        quiet_check=lambda: (calls.append(1), True)[1])
        p.run()
        self.assertEqual(len(calls), 4, "one call per arm per measured rep, not the warm-up")

    def test_the_measurement_still_runs_when_the_gate_gives_up(self):
        """The gate is a courtesy to the operator, not a veto. If the machine
        stays busy, the measurement proceeds and the DRIFT gate decides
        whether the result is usable — that is the authority."""
        p = paired.Pair("t", self._arm([100.0] * 3), self._arm([90.0] * 3),
                        reps=3, warmup=0, quiet_check=lambda: False)
        res = p.run()
        self.assertEqual(len(res.baseline.values), 3)
        self.assertEqual(len(res.candidate.values), 3)
        self.assertEqual(len(res.diffs), 3)

    def test_no_gate_means_no_calls_and_no_events(self):
        """The engine must still work for callers that do not want a gate."""
        p = paired.Pair("t", self._arm([100.0] * 2), self._arm([90.0] * 2),
                        reps=2, warmup=0)
        res = p.run()
        self.assertEqual(res.contended_samples, 0)
        self.assertEqual(len(res.diffs), 2)


class TestPerRepContentionDetection(_ProbeTest):
    """A repetition taken on a contended card must not enter the series silently.

    Waiting for quiet before a repetition is not enough: the repetition takes
    ~48 s, and a competing job that renders in waves can start in the middle
    of one. The drift gate catches that afterwards and refuses the whole run —
    correct, but the run is lost.

    So each repetition checks who else is on the card while it runs, and a
    contaminated one is discarded and retried.
    """

    def setUp(self):
        self._use(_quiet)

    def _arm(self, values):
        it = iter(values)

        def arm(index):
            return {"value": next(it)}
        return arm

    def test_a_contaminated_repetition_is_discarded_and_retried(self):
        """The arm is asked twice for that repetition; only the second
        value is kept."""
        seen = []
        values = [1.0, 2.0]

        def arm(index):
            seen.append(index)
            return {"value": values.pop(0)}

        # The FIRST probe call sees contention, the second is clean.
        self._use(_Scripted(busy_count=1))
        p = paired.Pair("t", arm, self._arm([90.0] * 5), reps=1, warmup=0)
        p.quiet_check = None
        res = p.run()
        self.assertGreaterEqual(len(seen), 2,
                                "the contaminated attempt must be retried")
        self.assertEqual(res.contended_samples, 0,
                         "a retried repetition that then came back clean is not contended")

    def test_a_clean_repetition_is_kept(self):
        p = paired.Pair("t", self._arm([100.0] * 3), self._arm([90.0] * 3),
                        reps=3, warmup=0)
        res = p.run()
        self.assertEqual(res.contended_samples, 0)
        self.assertEqual(len(res.diffs), 3)

    def test_a_repetition_that_stays_contaminated_is_kept_but_counted(self):
        """If every attempt sees contention, the value is kept and the
        contention is counted — the drift gate then decides whether the
        series is usable. Hiding the sample would be worse.

        Both arms are measured, so a busy card marks one sample per arm: two
        for one repetition."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        def arm(index):
            return {"value": values.pop(0)}

        self._use(_busy)
        p = paired.Pair("t", arm, self._arm([90.0] * 5), reps=1, warmup=0)
        p.quiet_check = None
        p.max_attempts = 2
        res = p.run()
        self.assertEqual(res.contended_samples, 2, "one per arm")
        self.assertEqual(len(res.baseline.values), 1)
        self.assertEqual(len(res.candidate.values), 1)

    def test_the_contenders_are_named_in_the_record(self):
        """A reader has to be able to tell WHOSE job spoiled the measurement,
        not just that one did."""
        values = [1.0, 2.0, 3.0, 4.0]

        def arm(index):
            return {"value": values.pop(0)}

        self._use(_busy)
        p = paired.Pair("t", arm, self._arm([90.0] * 5), reps=1, warmup=0)
        p.quiet_check = None
        p.max_attempts = 1
        res = p.run()
        self.assertTrue(res.contenders)
        self.assertIn("nuke.exe", res.contenders[0])


class TestTheWatcher(unittest.TestCase):
    """The watcher is what catches a rival that arrives AND LEAVES mid-render.

    Checking only before and after a measurement misses it: the rival is
    gone by the time the render ends, and the repetition that was spoiled
    looks clean. That is exactly what happened on this box — a Houdini watch
    job woke up, took the card for a few frames and dropped it, leaving a
    repetition that took 148 s where a clean frame takes 48 s.
    """

    def test_a_quiet_workload_reports_no_contenders(self):
        common.other_gpu_processes = _quiet
        self.addCleanup(
            lambda: setattr(common, "other_gpu_processes", _REAL_PROBE))
        with common.ContenderWatcher(interval=0.05) as w:
            time.sleep(0.2)
        self.assertTrue(w.clean)
        self.assertEqual(w.contenders, [])

    def test_a_rival_that_arrives_and_leaves_is_still_caught(self):
        """The whole point: the rival is GONE by the end, and must still be
        recorded."""
        state = {"busy": False, "calls": 0}

        def probe(*a, **kw):
            state["calls"] += 1
            # busy for a window in the middle, then gone again
            busy = 2 <= state["calls"] <= 4
            return ["houdinifx.exe:4962MiB"] if busy else []

        common.other_gpu_processes = probe
        self.addCleanup(
            lambda: setattr(common, "other_gpu_processes", _REAL_PROBE))
        with common.ContenderWatcher(interval=0.02) as w:
            time.sleep(0.5)
        self.assertFalse(w.clean, "a rival that came and went must still be seen")
        self.assertTrue(any("houdini" in c for c in w.contenders))


class TestWslIsProbedSeparately(unittest.TestCase):
    """WSL GPU work is invisible to the Windows per-process counter.

    Found the hard way on 2026-11-04, and it matters: the card was at 98%
    utilisation and 288 W with NO Windows process accounting for it. The work
    was a python3.12 process inside a WSL Ubuntu, whose GPU memory is
    allocated through the vGPU path and never appears in the Windows
    commit counter. The Windows-side detector reported an IDLE card on a card
    that was pinned.

    A detector that says "idle" on a pinned card is worse than no detector:
    it lets a contaminated measurement through with a clean-looking series.
    """

    def test_the_utf16_distribution_list_is_decoded(self):
        """`wsl --list` returns UTF-16 on this box, so the name arrives as
        bytes separated by NULs: 'U\\x00b\\x00u\\x00n\\x00t\\x00u\\x00'. Read
        naively it looks like three short names, or like 'Ub'. Neither is
        the distribution."""
        raw = "U\x00b\x00u\x00n\x00t\x00u\x00\n\x00\n\x00"

        class _R:
            stdout = raw

        with mock.patch.object(common.subprocess, "run", return_value=_R()):
            self.assertEqual(common._wsl_running_distros(), ["Ubuntu"])

    def test_the_wsl_probe_does_not_start_a_stopped_distribution(self):
        """A detector must never change the state it is measuring. Invoking
        `wsl -d <name>` on a STOPPED distribution STARTS it, so the probe
        enumerates the RUNNING ones instead."""
        src = (Path(__file__).resolve().parent.parent / "harness" / "common.py")
        text = src.read_text(encoding="utf-8")
        self.assertIn('"--list"', text)
        self.assertIn('"--running"', text)
        # The probe must not name a distribution directly.
        self.assertNotIn('"-d", "Ubuntu"', text)

    def test_we_do_not_probe_wsl_from_inside_wsl(self):
        """If the harness itself runs in WSL, the WSL probe would find its
        own work. The Windows side owns that question."""
        os.environ["WSL_DISTRO_NAME"] = "Ubuntu"
        try:
            self.assertEqual(common._wsl_gpu_processes(), [])
        finally:
            del os.environ["WSL_DISTRO_NAME"]

    def test_a_wsl_gpu_process_would_be_reported(self):
        """The probe's own output format, checked against a stubbed query —
        so the test does not depend on whether a WSL job happens to be
        running right now."""
        entries = common._wsl_gpu_processes()
        for e in entries:
            self.assertTrue(e.startswith("wsl:"), e)


class TestTheDetectorItself(unittest.TestCase):
    """The contention probe has to be right about who counts.

    Two failure modes, both observed on this box:

      * dwm.exe (the desktop compositor) commits ~2.2 GB permanently and
        csrss.exe ~385 MB. Counting them makes every check a false positive,
        and a detector that always says "busy" is worse than none — it would
        have refused every measurement on this machine.
      * our own render is the card being busy. A utilisation threshold
        cannot tell our render from somebody else's; process identity can.
    """

    def test_the_always_resident_graphics_processes_are_exempt(self):
        for name in ("dwm.exe", "csrss.exe", "System"):
            self.assertIn(name.lower(), common.OS_GRAPHICS_EXEMPT)

    def test_a_render_tool_counts_regardless_of_size(self):
        """Houdini with a scene loaded committed 4962 MiB, but a hython
        batch job can be small and still be rendering."""
        self.assertTrue(common._render_app("houdini.exe"))
        self.assertTrue(common._render_app("hython.exe"))
        self.assertTrue(common._render_app("husk.exe"))
        self.assertTrue(common._render_app("Nuke14.0.exe"))

    def test_an_ordinary_desktop_app_is_not_a_render_tool(self):
        self.assertFalse(common._render_app("chrome.exe"))
        self.assertFalse(common._render_app("explorer.exe"))
        self.assertFalse(common._render_app("EpicGamesLauncher.exe"))

    def test_stray_python_processes_are_not_render_tools(self):
        """This box runs ~23 of them. Naming them would be noise."""
        self.assertFalse(common._render_app("python.exe"))
        self.assertFalse(common._render_app("python3.13.exe"))

    def test_our_own_process_tree_is_found(self):
        """The probe excludes our tree, so our own renderer is not a
        competitor with ourselves."""
        mine = common.my_process_tree()
        self.assertIn(os.getpid(), mine)

    def test_a_child_process_is_in_the_tree(self):
        """The renderer is a CHILD of this process — it is the one holding
        the card, and it must not be mistaken for a competitor."""
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
        try:
            time.sleep(0.3)
            self.assertIn(proc.pid, common.my_process_tree())
        finally:
            proc.kill()
            proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
