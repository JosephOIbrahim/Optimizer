"""
Tests for the next-generation harness: paired measurement, the constraint
classifier, and the shot parser.

The paired measurement is the important one. It exists because of a specific
failure on this box, and the failure is a methodological one, not a bug:

    A thread-count sweep was gated against a baseline measured 48 minutes
    earlier, under different resident processes. It came back ADMIT +12.34%
    (p=1.9e-05). Re-timed the same way as the candidate, the same comparison
    was worth nothing — measured with both arms alternately in one session it
    is DENY -0.60%. (The "+0.7%" that first appeared against it was the
    phantom's own cross-session arithmetic, not a measurement of the lever.)
    The Field Notes call it "the move that evaporated".

These tests pin the properties that make that impossible:

  * the two arms are measured ALTERNATELY, so drift hits both;
  * the arm order alternates too, so whichever arm runs second cannot be
    biased by always paying the first arm's warm-up;
  * the drift across the run is reported, because that is the number that
    would have flagged the stale comparison.

Nothing here renders, touches the GPU, or writes outside a temp dir.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import common, constraint, paired  # noqa: E402


class _QuietSampler:
    """A contention probe that sees nobody on the card.

    The engine checks who else holds the card while each repetition runs, so
    a test that used the real probe would measure this box's other tenants —
    the exact variable the production code exists to control.
    """

    def __call__(self, *a, **kw):
        return []


from harness.constraint import Constraint  # noqa: E402
from harness.shot import parse_frame_times  # noqa: E402


class TestPairedMeasurement(unittest.TestCase):
    """The properties that make a paired A/B trustworthy."""

    def setUp(self):
        # The engine checks who else is on the card while each repetition
        # runs, and a repetition taken on a contended card is discarded and
        # retried. On a shared machine that would make these tests read
        # whatever else is running — so the probe is swapped for a quiet one.
        self._orig = common.other_gpu_processes
        common.other_gpu_processes = _QuietSampler()
        self.addCleanup(lambda: setattr(common, "other_gpu_processes", self._orig))

    def _arm(self, values, name="arm"):
        it = iter(values)
        def arm(index):
            return {"value": next(it)}
        arm.__name__ = name
        return arm

    def test_the_arms_are_measured_alternately(self):
        """Both arms run in the same session, one after the other, repeatedly."""
        order = []
        def mk(tag, v):
            def arm(index):
                order.append(tag)
                return {"value": v}
            return arm
        p = paired.Pair("t", mk("b", 100.0), mk("c", 90.0), reps=4, warmup=0)
        p.run()
        self.assertEqual(len(order), 8)
        # No two consecutive runs of the same arm once past the first pair.
        self.assertNotEqual(order[2], order[3])

    def test_the_order_of_the_arms_alternates_between_repetitions(self):
        """If one arm always ran second it would always pay the first arm's
        warm-up, and the effect would be biased by construction."""
        order = []
        def mk(tag, v):
            def arm(index):
                order.append(tag)
                return {"value": v}
            return arm
        paired.Pair("t", mk("b", 100.0), mk("c", 90.0), reps=4, warmup=0).run()
        # rep 1: b then c; rep 2: c then b; rep 3: b then c; rep 4: c then b
        self.assertEqual(order[:4], ["b", "c", "c", "b"])
        self.assertEqual(order[4:8], ["b", "c", "c", "b"])

    def test_the_warmup_repetition_is_excluded_from_the_samples(self):
        """The first repetition of either arm pays for a cold page cache, a
        cold shader cache and a cold GPU clock. Those are the machine being
        cold, not the workload."""
        p = paired.Pair("t", self._arm([999.0, 100.0, 100.0]),
                        self._arm([888.0, 90.0, 90.0]), reps=2, warmup=1)
        res = p.run()
        self.assertNotIn(999.0, res.baseline.values)
        self.assertNotIn(888.0, res.candidate.values)
        self.assertEqual(len(res.baseline.values), 2)

    def test_the_differences_are_paired_position_by_position(self):
        res = paired.Pair("t", self._arm([100.0, 100.0]),
                           self._arm([80.0, 80.0]), reps=2, warmup=0).run()
        self.assertEqual(res.diffs, [-20.0, -20.0])

    def test_drift_across_the_run_is_reported(self):
        """The number that would have caught the stale comparison: the baseline
        moved while we were measuring."""
        res = paired.Pair("t", self._arm([100.0, 100.0, 120.0]),
                           self._arm([90.0, 90.0, 90.0]), reps=3, warmup=0).run()
        self.assertAlmostEqual(res.baseline_drift, 0.20, places=6)

    def test_a_stable_baseline_reports_no_drift(self):
        res = paired.Pair("t", self._arm([100.0, 100.0, 100.0]),
                           self._arm([90.0, 90.0, 90.0]), reps=3, warmup=0).run()
        self.assertAlmostEqual(res.baseline_drift, 0.0, places=9)

    def test_the_effect_is_a_fractional_improvement(self):
        res = paired.Pair("t", self._arm([100.0] * 3),
                           self._arm([80.0] * 3), reps=3, warmup=0).run()
        self.assertAlmostEqual(res.effect_pct(lower_is_better=True), 0.20, places=6)
        self.assertAlmostEqual(res.effect_pct(lower_is_better=False), -0.20, places=6)

    def test_an_arm_that_returns_nonsense_is_refused(self):
        with self.assertRaises(TypeError):
            paired.Pair("t", lambda i: 1.0, lambda i: {"value": 1.0},
                        reps=1, warmup=0).run()
        with self.assertRaises(KeyError):
            paired.Pair("t", lambda i: {}, lambda i: {"value": 1.0},
                        reps=1, warmup=0).run()


class TestDriftGate(unittest.TestCase):
    """A verdict computed from an unstable CONTROL is not a verdict.

    During development a second render was running concurrently. The control
    drifted -33.8% across five repetitions (152.6, 152.0, 152.3, 126.1,
    101.0 TFLOPS) and the paired test still returned p=0.0146 — a
    "significant" win bought with a starved control. The drift check has to
    come before the statistics.
    """

    def setUp(self):
        # Same reason as TestPairedMeasurement: the engine checks who else is
        # on the card while each repetition runs, so the real probe would
        # measure this box's other tenants.
        self._orig = common.other_gpu_processes
        common.other_gpu_processes = _QuietSampler()
        self.addCleanup(lambda: setattr(common, "other_gpu_processes", self._orig))

    def _arm(self, values, name="arm"):
        it = iter(values)
        def arm(index):
            return {"value": next(it)}
        arm.__name__ = name
        return arm

    def test_a_moved_baseline_is_flagged(self):
        res = paired.Pair("t", self._arm([152.6, 152.0, 152.3, 126.1, 101.0]),
                           self._arm([165.2, 165.2, 165.6, 165.2, 111.8]),
                           reps=5, warmup=0).run()
        self.assertTrue(res.drift_exceeded)
        self.assertLess(res.baseline_drift, -0.3)

    def test_a_stable_baseline_is_not_flagged(self):
        """The measured drift of a quiet machine: -0.20% on the compute path
        and -1.29% on the render path, over three to five repetitions."""
        res = paired.Pair("t", self._arm([159.78, 159.5, 159.6, 159.4, 159.8]),
                           self._arm([175.36, 175.3, 175.7, 174.8, 175.4]),
                           reps=5, warmup=0).run()
        self.assertFalse(res.drift_exceeded)

    def test_the_drift_limit_sits_above_the_noise_of_a_quiet_machine(self):
        """5% must clear the drift a quiet machine actually shows (-1.29% on
        the render path) and still catch the starved-control case (-33.8%)."""
        self.assertGreater(paired.PairResult.DRIFT_LIMIT, 0.0129)
        self.assertLess(paired.PairResult.DRIFT_LIMIT, 0.338)

    def _arm(self, values):
        it = iter(values)
        def arm(index):
            return {"value": next(it)}
        return arm


class TestConstraintClassifier(unittest.TestCase):
    """Which bound is doing the work — asked before any lever is proposed."""

    def test_a_card_pinned_at_its_ceiling_is_power_bound(self):
        v = constraint.classify([{"power_frac": 0.06, "avg_power_w": 439,
                                  "power_limit_w": 450}])
        self.assertIs(v.constraint, Constraint.POWER)

    def test_a_card_at_96_percent_of_its_ceiling_is_power_bound_even_if_the_counter_read_zero(self):
        """The duty-cycle counter is read over one short window, so it can
        miss a workload that is running right at the wall. Measured on this
        box: fp16 GEMM drew 433 W of 450 W and the counter read 0% on some
        samples and 5.8% on others."""
        v = constraint.classify([{"power_frac": 0.0, "avg_power_w": 433,
                                  "power_limit_w": 450}])
        self.assertIs(v.constraint, Constraint.POWER)

    def test_an_idle_card_is_not_power_bound(self):
        """The Field Notes' first finding: a Karma render drew 100 W of a
        450 W budget and spent 0% of its life at the cap."""
        v = constraint.classify([{"power_frac": 0.0, "avg_power_w": 104,
                                  "power_limit_w": 450}],
                                frame_times=[54.7, 16.03, 15.66])
        self.assertIsNot(v.constraint, Constraint.POWER)

    def test_a_heavy_first_frame_is_setup_bound(self):
        """The setup tax is the first frame's cost minus what the later frames
        pay, and it is amortizable by rendering more frames per process."""
        v = constraint.classify([{"power_frac": 0.0, "avg_power_w": 104,
                                  "power_limit_w": 450}],
                                frame_times=[54.7, 16.03, 15.66])
        self.assertIs(v.constraint, Constraint.SETUP)
        self.assertAlmostEqual(v.setup_tax_s, 38.9, places=1)

    def test_equal_frames_are_not_setup_bound(self):
        v = constraint.classify([{"power_frac": 0.0, "avg_power_w": 104,
                                  "power_limit_w": 450}],
                                frame_times=[48.1, 48.0, 48.2])
        self.assertIsNot(v.constraint, Constraint.SETUP)

    def test_the_setup_tax_needs_more_than_one_frame(self):
        self.assertEqual(constraint.setup_tax([50.0]), 0.0)
        self.assertEqual(constraint.setup_tax([]), 0.0)

    def test_the_setup_tax_is_never_negative(self):
        self.assertEqual(constraint.setup_tax([10.0, 30.0, 30.0]), 0.0)

    def test_the_headroom_is_reported(self):
        v = constraint.classify([{"power_frac": 0.0, "avg_power_w": 104,
                                  "power_limit_w": 450}])
        self.assertAlmostEqual(v.power_headroom_w, 346.0, places=1)


class TestShotParser(unittest.TestCase):
    """husk's timers are cumulative; the per-frame times are the differences."""

    LOG = ("[07:08:52] >>> Render seq_0503.png (1 of 3) - 1024x429 frame: 503\n"
           "[07:08:54]   Load Wall Clock Time: 0:00:03.25\n"
           "[07:08:54] Update Wall Clock Time: 0:00:01.68\n"
           "[07:09:45] Total Wall Clock Time: 0:00:54.70\n"
           "[07:09:46] >>> Render seq_0504.png (2 of 3) - 1024x429 frame: 504\n"
           "[07:09:46] Update Wall Clock Time: 0:00:53.56\n"
           "[07:10:01] Total Wall Clock Time: 0:01:10.73\n"
           "[07:10:02] >>> Render seq_0505.png (3 of 3) - 1024x429 frame: 505\n"
           "[07:10:02] Update Wall Clock Time: 0:01:09.58\n"
           "[07:10:17] Total Wall Clock Time: 0:01:26.39\n")

    def test_the_per_frame_wall_times_are_the_differences(self):
        """Taken from a real husk log on this box. Reading the cumulative
        values directly would make every frame look like the whole run."""
        wall, _, _ = parse_frame_times(self.LOG)
        self.assertEqual(len(wall), 3)
        self.assertAlmostEqual(wall[0], 54.70, places=2)
        self.assertAlmostEqual(wall[1], 16.03, places=2)
        self.assertAlmostEqual(wall[2], 15.66, places=2)

    def test_the_resolutions_are_read(self):
        _, _, res = parse_frame_times(self.LOG)
        self.assertEqual(res, ["1024x429", "1024x429", "1024x429"])

    def test_an_empty_log_yields_nothing_rather_than_raising(self):
        wall, render, res = parse_frame_times("")
        self.assertEqual((wall, render, res), ([], [], []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
