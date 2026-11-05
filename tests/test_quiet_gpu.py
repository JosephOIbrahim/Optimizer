"""
Tests for the "is the GPU quiet enough to measure on" check.

The check exists because a contaminated measurement is worse than a late one:
the drift gate refuses it, and the operator has waited for nothing. It was
refused on this box at +16.3% drift, with another project's pytest runs and a
Nuke frameserver on the card.

Two things are tested here:

  * the DECISION — a card that is doing work, or that somebody else is holding
    a large share of, counts as busy; a quiet card does not;
  * the HONESTY — the check is a heuristic, and the drift gate on the control
    remains the authority on whether a measurement is usable. Utilisation
    alone cannot tell a contended run from a clean one, which is why the
    check asks about VRAM occupancy as well: a frameserver can hold 16 GB
    while the GPU reads idle.

The decision logic is factored out so it can be tested without a busy card.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import common  # noqa: E402
from harness.common import busy_from_readings  # noqa: E402


class TestBusyDecision(unittest.TestCase):
    """The decision, tested without needing a busy card."""

    def test_a_quiet_card_is_not_busy(self):
        busy, peak, used, total = busy_from_readings([2.0, 3.0, 1.0], 2442, 24564)
        self.assertFalse(busy)
        self.assertAlmostEqual(peak, 3.0)

    def test_a_card_doing_work_is_busy(self):
        busy, peak, _, _ = busy_from_readings([97.0, 98.0, 97.0], 2442, 24564)
        self.assertTrue(busy)
        self.assertAlmostEqual(peak, 98.0)

    def test_vram_occupancy_alone_does_not_decide(self):
        """The false positive that cost an hour.

        On WDDM, nvidia-smi's memory.used counts shared-system-memory
        allocations: it read 17.8 GB "used" while the sum of per-process GPU
        commit was 5.5 GB. A 12.4 GB phantom, from a card that was idle.

        So high occupancy is REPORTED but does not block. The honest measure of
        occupancy is the per-process commit counter (v12.baseline), which is
        what V12's own preflight already reasons about.
        """
        busy, peak, used, total = busy_from_readings([1.0, 2.0, 1.0], 17838, 24564)
        self.assertFalse(busy, "a WDDM shared-memory phantom must not block a measurement")
        self.assertAlmostEqual(peak, 2.0)
        # The numbers are still reported, so the operator sees the context.
        self.assertEqual(used, 17838)
        self.assertEqual(total, 24564)

    def test_the_peak_is_reported_not_the_last_sample(self):
        """A single busy sample in the middle of a quiet run is a burst, and
        the peak is what matters."""
        busy, peak, _, _ = busy_from_readings([1.0, 95.0, 1.0], 2442, 24564)
        self.assertTrue(busy)
        self.assertAlmostEqual(peak, 95.0)

    def test_unreadable_readings_are_not_busy(self):
        """A failed read must not block a measurement forever. The drift gate
        is the backstop."""
        busy, peak, used, total = busy_from_readings([], 0, 0)
        self.assertFalse(busy)
        self.assertEqual((peak, used, total), (0.0, 0, 0))


class TestTheThresholdIsCalibratedToThisDesktop(unittest.TestCase):
    """The threshold must clear the DESKTOP's own burst.

    Measured on this box on 2026-11-04, 113 samples over two minutes with
    nothing running but the desktop:

        utilisation  min 4%   median 24%   p95 41%   max 72%
        power        min 32 W median 33 W  max 70 W

    The threshold used to be 10%, which is BELOW this desktop's own idle
    burst. So the gate declared an idle card busy and waited for a quiet
    that never came — the measurement stalled on a card that was doing
    nothing but compositing a desktop.

    What separates the states is power, not utilisation alone:

        desktop compositing    4-72% util     32-70 W
        a Karma render         94%+  util    104 W+
        a WSL vLLM benchmark   98%  util   288-300 W
    """

    def test_the_desktop_burst_is_not_busy(self):
        """The worst desktop burst measured must not block a measurement."""
        busy, peak, _, _ = busy_from_readings([4.0, 24.0, 41.0, 72.0], 2339, 24564)
        self.assertFalse(busy,
                         "the desktop's own burst must not read as a busy card")

    def test_a_real_render_is_busy(self):
        """A Karma render sits above 90% while it works."""
        busy, peak, _, _ = busy_from_readings([94.0, 96.0], 4000, 24564)
        self.assertTrue(busy)

    def test_a_compute_job_is_busy(self):
        """The WSL benchmark that was actually detected held the card at 98%."""
        busy, peak, _, _ = busy_from_readings([98.0], 16000, 24564)
        self.assertTrue(busy)

    def test_the_threshold_clears_the_desktop_and_catches_a_render(self):
        """The number itself, asserted against both ends of the measurement."""
        self.assertGreater(common.DESKTOP_UTIL_CEILING_PCT, 72.0,
                           "must clear the desktop's worst observed burst")
        self.assertLess(common.DESKTOP_UTIL_CEILING_PCT, 94.0,
                        "must still catch a render, which sits above 90%")

    def test_the_old_ten_percent_threshold_would_have_failed(self):
        """Pins the defect: at 10% the desktop's own median burst reads as
        busy, so the gate would never open on an idle card."""
        busy, _, _, _ = busy_from_readings([24.0], 2339, 24564,
                                           util_limit_pct=10.0)
        self.assertTrue(busy, "the old threshold misread the desktop as busy")
        busy_now, _, _, _ = busy_from_readings([24.0], 2339, 24564)
        self.assertFalse(busy_now, "the current threshold must not")


class TestQuietWaitContract(unittest.TestCase):
    def test_the_wait_reports_what_it_found(self):
        """It has to say what it is waiting for, or the operator cannot tell
        a slow start from a hung one."""
        src = (REPO / "harness" / "common.py").read_text(encoding="utf-8")
        self.assertIn("waiting for it to settle", src)
        self.assertIn("still busy", src)

    def test_the_wait_gives_up_rather_than_waiting_forever(self):
        """A machine that stays busy is a fact about the machine, and the
        caller should be told rather than left waiting."""
        src = (REPO / "harness" / "common.py").read_text(encoding="utf-8")
        self.assertIn("still busy", src)
        self.assertIn("timeout_s", src)
        # And it must say it is proceeding anyway, so a refusal later is not
        # a mystery.
        self.assertIn("measuring", src)


class TestRealReading(unittest.TestCase):
    """One live reading, to prove the query still works on this box."""

    def test_it_reads_the_real_card(self):
        busy, peak, used, total = common.gpu_busy(samples=1)
        self.assertGreater(total, 0, "nvidia-smi did not report VRAM")
        self.assertGreaterEqual(used, 0)
        self.assertGreaterEqual(peak, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
