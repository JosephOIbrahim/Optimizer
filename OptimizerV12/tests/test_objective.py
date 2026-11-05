"""
Tests for the Objective contract and the statistics underneath it.

Two things are being tested, and they are different in kind.

**The contract** — that an Objective names what is measured, which direction is
better, how much is enough, and what must not regress. V9 measured a matmul and
filed it against a UNet because nothing in the record said which workload the
number described; `Workload.fingerprint` is the fix and these tests pin it.

**The arithmetic** — `_student_t_cdf` is a dependency-free regularized incomplete
beta, because this code has to run on a bare interpreter with no SciPy. A
hand-rolled CDF that is quietly wrong turns every verdict into a coin flip that
looks authoritative, so it is checked against a published t-table rather than
against itself.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12.objective import (KNOWN_METRICS, Direction, Metric, Objective,  # noqa: E402
                           Workload, WorkloadKind, _mean, _student_t_cdf, _var,
                           diffusion_objective, render_objective,
                           vellum_objective, welch_t_test)


def _obj(metric="fps", min_effect=0.05, min_samples=5, guardrails=None) -> Objective:
    return Objective(
        workload=Workload(WorkloadKind.MICROBENCH, "test", "1x1", None, ""),
        primary=KNOWN_METRICS[metric],
        min_effect=min_effect,
        min_samples=min_samples,
        guardrails=guardrails or {},
    )


class TestStudentTTable(unittest.TestCase):
    """The published table, not a re-run of our own function. If the CDF drifts,
    every p-value drifts with it and nothing else in the suite would notice."""

    #: (t, df, CDF(t))
    TABLE = [
        (0.0000, 10, 0.500),
        (1.0000, 1, 0.750),        # Cauchy: 0.5 + atan(t)/pi, exact
        (6.3138, 1, 0.950),
        (12.706, 1, 0.975),
        (2.9200, 2, 0.950),
        (4.3027, 2, 0.975),
        (2.0150, 5, 0.950),
        (1.8125, 10, 0.950),
        (2.2281, 10, 0.975),
        (1.7247, 20, 0.950),
        (1.6973, 30, 0.950),
        (1.6602, 100, 0.950),
    ]

    def test_matches_the_table(self):
        for t, df, want in self.TABLE:
            with self.subTest(t=t, df=df):
                self.assertAlmostEqual(_student_t_cdf(t, df), want, delta=2e-3)

    def test_is_symmetric_about_the_origin(self):
        for t in (0.4, 1.0, 2.5, 7.0):
            for df in (1, 3, 10, 60):
                with self.subTest(t=t, df=df):
                    self.assertAlmostEqual(_student_t_cdf(-t, df) + _student_t_cdf(t, df),
                                           1.0, places=6)

    def test_is_monotone(self):
        prev = -1.0
        for t in [x * 0.25 for x in range(-20, 21)]:
            cur = _student_t_cdf(t, 7)
            self.assertGreaterEqual(cur, prev)
            prev = cur

    def test_stays_in_range_at_the_extremes(self):
        self.assertAlmostEqual(_student_t_cdf(0.0, 3), 0.5, places=9)
        self.assertGreater(_student_t_cdf(500.0, 1), 0.999)
        self.assertLess(_student_t_cdf(-500.0, 1), 0.001)

    def test_a_nonpositive_df_does_not_divide_by_zero(self):
        self.assertEqual(_student_t_cdf(1.0, 0), 0.5)


class TestWelch(unittest.TestCase):
    def test_too_few_samples_returns_a_non_significant_p(self):
        self.assertEqual(welch_t_test([1.0], [2.0], Direction.HIGHER_IS_BETTER), 1.0)
        self.assertEqual(welch_t_test([], [], Direction.HIGHER_IS_BETTER), 1.0)

    def test_identical_deterministic_runs_are_not_significant(self):
        self.assertEqual(welch_t_test([5.0] * 4, [5.0] * 4, Direction.HIGHER_IS_BETTER), 1.0)

    def test_zero_variance_with_a_real_difference_is_taken_at_face_value(self):
        """Zero spread on both sides means a deterministic harness. A difference
        there is real, and refusing it would make the test unusable on exactly the
        workloads where measurement is cleanest."""
        p = welch_t_test([1.0] * 3, [9.0] * 3, Direction.HIGHER_IS_BETTER)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1e-100)
        self.assertEqual(welch_t_test([9.0] * 3, [1.0] * 3, Direction.HIGHER_IS_BETTER), 1.0)

    def test_direction_flips_which_side_wins(self):
        slow, fast = [10.0] * 3, [5.0] * 3
        p = welch_t_test(slow, fast, Direction.LOWER_IS_BETTER)
        self.assertGreater(p, 0.0, "5 < 10 is better when lower is better")
        self.assertLess(p, 1e-100)
        self.assertEqual(welch_t_test(fast, slow, Direction.LOWER_IS_BETTER), 1.0)

    def test_equal_variance_is_antisymmetric(self):
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        b = [20.0, 21.0, 22.0, 23.0, 24.0]
        p_ab = welch_t_test(a, b, Direction.HIGHER_IS_BETTER)
        p_ba = welch_t_test(b, a, Direction.HIGHER_IS_BETTER)
        self.assertAlmostEqual(p_ab + p_ba, 1.0, places=6)
        self.assertLess(p_ab, 0.01)

    def test_a_stuttering_candidate_is_not_called_a_win(self):
        """The reason for Welch over Student: a candidate that stutters has a wider
        spread, and pooling variance would hide exactly that."""
        steady = [100.0, 100.1, 99.9, 100.0, 100.05]
        stutter = [200.0, 50.0, 200.0, 50.0, 200.0]
        self.assertGreater(_mean(stutter), _mean(steady))          # sanity: better mean
        self.assertGreater(welch_t_test(steady, stutter, Direction.HIGHER_IS_BETTER), 0.05)


class TestMetric(unittest.TestCase):
    def test_improvement_respects_direction(self):
        lower = KNOWN_METRICS["frame_ms"]
        higher = KNOWN_METRICS["fps"]
        self.assertAlmostEqual(lower.improvement(100.0, 80.0), 0.20, places=9)
        self.assertAlmostEqual(higher.improvement(100.0, 120.0), 0.20, places=9)
        self.assertAlmostEqual(lower.improvement(80.0, 100.0), -0.25, places=9)

    def test_improvement_returns_zero_rather_than_dividing_by_zero(self):
        m = KNOWN_METRICS["fps"]
        self.assertEqual(m.improvement(0.0, 100.0), 0.0)
        self.assertEqual(m.improvement(float("nan"), 100.0), 0.0)
        self.assertEqual(m.improvement(100.0, float("inf")), 0.0)

    def test_better_respects_direction(self):
        self.assertTrue(KNOWN_METRICS["fps"].better(120.0, 100.0))
        self.assertTrue(KNOWN_METRICS["frame_ms"].better(80.0, 100.0))
        self.assertFalse(KNOWN_METRICS["frame_ms"].better(120.0, 100.0))

    def test_the_guardrail_directions_are_the_pareto_gate_directions(self):
        self.assertIs(KNOWN_METRICS["lpips"].direction, Direction.LOWER_IS_BETTER)
        self.assertIs(KNOWN_METRICS["clip"].direction, Direction.HIGHER_IS_BETTER)
        self.assertIs(KNOWN_METRICS["ssim"].direction, Direction.HIGHER_IS_BETTER)


class TestWorkloadFingerprint(unittest.TestCase):
    """The fix for the matmul-as-UNet failure: identity that cannot drift."""

    def _wl(self, **kw) -> Workload:
        base = dict(kind=WorkloadKind.DIFFUSION, name="comfyui-sdxl",
                    resolution="1024x1024", steps=20, precision="bf16")
        base.update(kw)
        return Workload(**base)

    def test_is_stable_for_the_same_workload(self):
        self.assertEqual(self._wl().fingerprint, self._wl().fingerprint)

    def test_changes_when_anything_that_matters_changes(self):
        base = self._wl().fingerprint
        for kw in ({"resolution": "512x512"}, {"steps": 30}, {"precision": "fp8_e4m3"},
                   {"name": "other-model"}, {"kind": WorkloadKind.MICROBENCH}):
            with self.subTest(kw=kw):
                self.assertNotEqual(self._wl(**kw).fingerprint, base)

    def test_extra_metadata_is_order_independent(self):
        a = self._wl(extra={"vae": "sdxl", "scheduler": "dpm"})
        b = self._wl(extra={"scheduler": "dpm", "vae": "sdxl"})
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_a_matmul_is_not_compatible_with_a_unet(self):
        matmul = Workload(WorkloadKind.MICROBENCH, "matmul", "4096x4096", None, "fp16")
        unet = Workload(WorkloadKind.DIFFUSION, "comfyui-sdxl", "1024x1024", 20, "bf16")
        ok, why = matmul.compatible_with(unet)
        self.assertFalse(ok)
        self.assertIn("kind differs", why)

    def test_same_kind_but_different_fingerprint_is_incompatible(self):
        ok, why = self._wl().compatible_with(self._wl(resolution="2048x2048"))
        self.assertFalse(ok)
        self.assertIn("fingerprint", why)

    def test_identical_workloads_are_compatible(self):
        ok, why = self._wl().compatible_with(self._wl())
        self.assertTrue(ok)
        self.assertEqual(why, "")


class TestAdmitsContract(unittest.TestCase):
    def test_returns_three_values(self):
        r = _obj().admits([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        self.assertEqual(len(r), 3)
        self.assertIsInstance(r[0], bool)
        self.assertIsInstance(r[1], str)
        self.assertIsInstance(r[2], dict)

    def test_too_few_samples_on_either_side_is_refused(self):
        for b, c in (([1.0], [1.0, 2.0, 3.0]), ([1.0, 2.0, 3.0], [1.0]), ([], [])):
            with self.subTest(b=b, c=c):
                ok, why, _ = _obj().admits(b, c)
                self.assertFalse(ok)
                self.assertIn("not enough samples", why)

    def test_an_effect_below_the_threshold_is_refused(self):
        ok, why, d = _obj(min_effect=0.50).admits(
            [100.0, 100.0, 100.0, 100.0, 100.0], [101.0] * 5)
        self.assertFalse(ok)
        self.assertIn("below", why)
        self.assertAlmostEqual(d["effect"], 0.01, places=6)

    def test_a_big_but_noisy_difference_is_refused_as_luck(self):
        """Practical significance without statistical significance is exactly the
        failure mode this pairing exists to catch."""
        ok, why, d = _obj("fps").admits([1.0, 2.0, 3.0, 4.0, 5.0],
                                        [1.0, 2.0, 3.0, 4.0, 20.0])
        self.assertFalse(ok)
        self.assertIn("not significant", why)
        self.assertGreater(d["p_value"], 0.05)

    def test_a_statistically_real_but_tiny_win_is_refused_as_not_worth_it(self):
        ok, why, _ = _obj("fps", min_effect=0.10).admits(
            [100.0, 100.1, 99.9, 100.0, 100.05], [101.0, 101.1, 100.9, 101.0, 101.05])
        self.assertFalse(ok)
        self.assertIn("below", why)

    def test_an_effect_that_clears_the_threshold_but_lacks_samples_is_refused(self):
        ok, why, _ = _obj("fps", min_samples=8).admits(
            [100.0, 100.1], [120.0, 120.1])
        self.assertFalse(ok)
        self.assertIn("n<8", why)

    def test_a_clean_win_is_admitted(self):
        ok, why, d = _obj("fps").admits(
            [100.0, 100.1, 99.9, 100.0, 100.05], [120.0, 120.1, 119.9, 120.0, 120.05])
        self.assertTrue(ok)
        self.assertIn("clears both", why)
        self.assertLess(d["p_value"], 0.05)
        self.assertAlmostEqual(d["effect"], 0.20, places=2)

    def test_guardrails(self):
        base = [100.0, 100.1, 99.9, 100.0, 100.05]
        cand = [80.0, 80.1, 79.9, 80.0, 80.05]
        obj = _obj("frame_ms", guardrails={"lpips": 0.04})
        ok, why, d = obj.admits(base, cand, {"lpips": ([0.2] * 5, [0.3] * 5)})
        self.assertFalse(ok)
        self.assertIn("guardrail 'lpips'", why)
        self.assertAlmostEqual(d["guardrail_lpips"], -0.5, places=6)

        ok, why, _ = obj.admits(base, cand, {"lpips": ([0.200] * 5, [0.204] * 5)})
        self.assertTrue(ok, why)

    def test_an_unknown_guardrail_is_skipped_rather_than_crashing(self):
        ok, _, _ = _obj("frame_ms", guardrails={}).admits(
            [100.0, 100.1, 99.9, 100.0, 100.05], [80.0, 80.1, 79.9, 80.0, 80.05],
            {"not_a_metric": ([1.0], [2.0])})
        self.assertTrue(ok)


class TestProvidedObjectives(unittest.TestCase):
    def test_the_three_real_objectives_are_well_formed(self):
        for obj in (diffusion_objective(), vellum_objective(), render_objective()):
            with self.subTest(obj=obj.workload.name):
                self.assertGreater(obj.min_effect, 0.0)
                self.assertGreaterEqual(obj.min_samples, 2)
                self.assertEqual(obj.harness, obj.harness.strip())
                self.assertTrue(obj.harness,
                                "an objective with no named harness is how V9 filed a "
                                "matmul against a UNet")

    def test_objectives_are_not_interchangeable(self):
        a, b = diffusion_objective(), render_objective()
        ok, why = a.workload.compatible_with(b.workload)
        self.assertFalse(ok)
        self.assertIn("kind differs", why)

    def test_helpers(self):
        self.assertEqual(_mean([1.0, 2.0, 3.0]), 2.0)
        self.assertEqual(_mean([]), 0.0)
        self.assertAlmostEqual(_var([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 4.571428, places=5)
        self.assertEqual(_var([1.0]), 0.0)

    def test_render_objective_carries_its_guardrail(self):
        self.assertIn("ssim", render_objective().guardrails)
        self.assertEqual([m.name for m in render_objective().guardrail_metrics], ["ssim"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
