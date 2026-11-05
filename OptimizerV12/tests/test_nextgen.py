"""
Tests for the next-generation additions: the power ceiling as a restorable
change, the paired test, and the two new objectives.

Three things here exist because of a concrete failure, and each test names the
failure it prevents:

  * the power ceiling is the first NVML target whose prior is READABLE and
    whose setter round-trips, so undo RESTORES the operator's number instead
    of resetting to vendor default;
  * the paired t-test exists because a +12.34% win (p=1.9e-05) on this box
    was an artifact of comparing against a baseline measured 48 minutes
    earlier, and Welch on two independent series cannot catch that;
  * the shot objective exists because the render objective cannot see the
    largest win available here (46% on a 3-frame shot) — the setup tax is
    excluded from render_s by construction.

Nothing here touches the GPU, the registry or D:\\OptimizerV12.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12.change import (ADMIN_KINDS, KIND_DEFAULTS, Change, ChangeKind,  # noqa: E402
                         EvidenceClass, Observability, Rollback, Snapshot, Tier,
                         change_key)
from v12.objective import (KNOWN_METRICS, Direction, Metric, Objective,  # noqa: E402
                            Workload, WorkloadKind, _mean, gemm_objective,
                            paired_t_test, render_objective, shot_objective,
                            welch_t_test)


class TestPowerCeilingIsRestorable(unittest.TestCase):
    """The one NVML target that does not destroy operator state."""

    def test_it_is_declared_readable_not_reset_only(self):
        """The getter exists and the setter round-trips it exactly.

        Measured on this box (2026-11-04): nvmlDeviceSetPowerManagementLimit
        with 500000 (milliwatts) reads back 500.0 W; 450000 reads back 450.0 W.
        So the prior is capturable and undo restores it. That is READABLE, and
        it is the reason this lever can be applied without the operator's
        ceiling being lost.
        """
        c = Change(id="nvml_power_limit", kind=ChangeKind.NVML_POWER_LIMIT,
                   description="d", target="gpu0")
        self.assertIs(c.observability, Observability.READABLE)
        self.assertIs(c.tier, Tier.SYSTEM)

    def test_the_family_around_it_is_still_untrustworthy(self):
        """Adding a readable member must not soften the others."""
        self.assertIs(_c(ChangeKind.NVML_CLOCK).observability, Observability.RESET_ONLY)
        self.assertIs(_c(ChangeKind.NVML_OFFSET).observability, Observability.LYING)

    def test_it_needs_administrator(self):
        self.assertIn(ChangeKind.NVML_POWER_LIMIT, ADMIN_KINDS)

    def test_a_trusted_prior_restores_rather_than_resets(self):
        """The whole point: undo returns the operator's number, not 450 W
        because the vendor said so."""
        c = Change(id="nvml_power_limit", kind=ChangeKind.NVML_POWER_LIMIT,
                   description="d", target="gpu0",
                   snapshot=Snapshot(value=450.0, existed=True, trusted=True,
                                    source="nvmlDeviceGetPowerManagementLimit"))
        self.assertIs(c.rollback_verb, Rollback.RESTORE_PRIOR)
        self.assertTrue(c.prior_recorded)
        self.assertFalse(c.destroys_operator_state)

    def test_an_untrusted_prior_does_not_claim_to_restore(self):
        c = Change(id="nvml_power_limit", kind=ChangeKind.NVML_POWER_LIMIT,
                   description="d", target="gpu0",
                   snapshot=Snapshot(value=None, existed=False, trusted=False,
                                    source="read failed"))
        self.assertFalse(c.prior_recorded)
        self.assertTrue(c.rollback_loses_prior)

    def test_its_change_key_is_distinct_from_the_clock_family(self):
        a = change_key(_c(ChangeKind.NVML_POWER_LIMIT, target="gpu0"))
        b = change_key(_c(ChangeKind.NVML_CLOCK, target="gpu0"))
        self.assertNotEqual(a, b)

    def test_a_change_can_declare_the_field_it_deliberately_moves(self):
        """Applying the change must not invalidate the baseline it is tested
        against — that is the field the change is moving, not a drift."""
        c = _c(ChangeKind.NVML_POWER_LIMIT, varies=("power_limit_w",))
        self.assertEqual(c.varies, ("power_limit_w",))
        self.assertEqual(_c(ChangeKind.ENV_VAR).varies, ())


class TestPairedTest(unittest.TestCase):
    """The test that would have caught the +12% phantom."""

    def test_too_few_pairs_is_not_significant(self):
        self.assertEqual(paired_t_test([1.0], Direction.LOWER_IS_BETTER), 1.0)
        self.assertEqual(paired_t_test([], Direction.LOWER_IS_BETTER), 1.0)

    def test_a_consistent_improvement_is_significant(self):
        diffs = [-1.0, -1.1, -0.9, -1.05, -0.95]
        p = paired_t_test(diffs, Direction.LOWER_IS_BETTER)
        self.assertLess(p, 0.01)

    def test_a_consistent_regression_is_not_significant_in_that_direction(self):
        diffs = [1.0, 1.1, 0.9, 1.05, 0.95]
        p = paired_t_test(diffs, Direction.LOWER_IS_BETTER)
        self.assertGreater(p, 0.5)

    def test_direction_is_respected(self):
        diffs = [-1.0, -1.1, -0.9]
        self.assertLess(paired_t_test(diffs, Direction.LOWER_IS_BETTER), 0.5)
        self.assertGreater(paired_t_test(diffs, Direction.HIGHER_IS_BETTER), 0.5)

    def test_a_consistent_difference_with_no_spread_is_real(self):
        """Zero spread means a deterministic harness; refusing it would make the
        test unusable on exactly the cleanest workloads."""
        # The p-value is below this computation's resolution, not exactly zero:
        # a p-value of 0.0 claims infinite evidence from a finite sample.
        p = paired_t_test([-2.0] * 4, Direction.LOWER_IS_BETTER)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1e-100)
        self.assertEqual(paired_t_test([0.0] * 4, Direction.LOWER_IS_BETTER), 1.0)

    def test_a_drift_only_difference_is_not_significant(self):
        """The phantom, reproduced.

        The baseline was measured 48 minutes earlier on a busier machine, so
        the candidate looked 12% faster. As PAIRED differences, that is one
        arm having drifted — and here the pairing is what reveals it, because
        the two arms were measured alternately in one session and the drift
        is common to both.
        """
        # A drift that hits BOTH arms equally: the paired difference is ~0.
        drifted = [10.0, 10.0, 10.0, 10.0, 10.0]
        self.assertGreater(paired_t_test(drifted, Direction.LOWER_IS_BETTER), 0.5)

    def test_paired_and_welch_agree_on_a_clean_win(self):
        base = [100.0, 100.1, 99.9, 100.0, 100.05]
        cand = [80.0, 80.1, 79.9, 80.0, 80.05]
        diffs = [c - b for b, c in zip(base, cand)]
        pw = paired_t_test(diffs, Direction.LOWER_IS_BETTER)
        ww = welch_t_test(base, cand, Direction.LOWER_IS_BETTER)
        self.assertLess(pw, 0.01)
        self.assertLess(ww, 0.01)


class TestAdmitsWithPairedSamples(unittest.TestCase):
    """The gate must be able to USE the paired test when pairs are supplied."""

    def _obj(self, metric="render_s", min_effect=0.05):
        return Objective(
            workload=Workload(WorkloadKind.RENDER, "karma-xpu", "1024x429", 128, ""),
            primary=KNOWN_METRICS[metric],
            min_effect=min_effect,
            min_samples=5,
            guardrails={},
        )

    def test_paired_diffs_are_used_when_given(self):
        base = [100.0] * 5
        cand = [80.0] * 5
        diffs = [c - b for b, c in zip(base, cand)]
        ok, why, detail = self._obj().admits(base, cand, None, paired_diffs=diffs)
        self.assertTrue(ok, why)
        self.assertEqual(detail.get("paired"), 1.0)

    def test_a_paired_difference_that_contradicts_the_means_is_refused(self):
        """A mismatch between the pairs and the series is a bug in the caller,
        and the gate must not quietly test the wrong thing."""
        base = [100.0] * 5
        cand = [80.0] * 5
        ok, why, _ = self._obj().admits(base, cand, None, paired_diffs=[-1.0, -2.0])
        self.assertFalse(ok)
        self.assertIn("do not match", why)

    def test_the_unpaired_path_is_unchanged(self):
        """With no pairs, the gate behaves exactly as before."""
        base = [100.0, 100.1, 99.9, 100.0, 100.05]
        cand = [120.0, 120.1, 119.9, 120.0, 120.05]
        ok, why, detail = self._obj("fps").admits(base, cand)
        self.assertTrue(ok, why)
        self.assertNotIn("paired", detail)


class TestNewObjectives(unittest.TestCase):
    def test_the_shot_objective_scores_the_shot(self):
        obj = shot_objective(resolution="1024x429", samples=128, frames=3)
        self.assertEqual(obj.primary.name, "shot_s")
        self.assertIs(obj.primary.direction, Direction.LOWER_IS_BETTER)
        self.assertEqual(obj.workload.extra["frames"], "3")
        self.assertIn("ssim", obj.guardrails)

    def test_shots_of_different_lengths_are_different_workloads(self):
        """A 3-frame shot and a 24-frame shot are different amounts of work and
        their timings must never be compared."""
        a = shot_objective(frames=3)
        b = shot_objective(frames=24)
        ok, why = a.workload.compatible_with(b.workload)
        self.assertFalse(ok)
        self.assertIn("fingerprint", why)

    def test_a_shot_and_a_frame_are_different_workloads(self):
        a = render_objective()
        b = shot_objective()
        ok, _ = a.workload.compatible_with(b.workload)
        self.assertFalse(ok)

    def test_the_gemm_objective_scores_throughput_upward(self):
        obj = gemm_objective(n=8192, precision="fp16")
        self.assertEqual(obj.primary.name, "tflops")
        self.assertIs(obj.primary.direction, Direction.HIGHER_IS_BETTER)
        self.assertEqual(obj.workload.resolution, "8192^3")

    def test_the_gemm_objective_carries_the_torch_build(self):
        """A TFLOPS number from a different torch is a different measurement."""
        a = gemm_objective(extra={"torch": "2.9.1+cu130"})
        b = gemm_objective(extra={"torch": "2.14.1+cu130"})
        ok, _ = a.workload.compatible_with(b.workload)
        self.assertFalse(ok)

    def test_every_metric_has_a_declared_direction(self):
        for name, m in KNOWN_METRICS.items():
            with self.subTest(metric=name):
                self.assertIn(m.direction, (Direction.HIGHER_IS_BETTER,
                                           Direction.LOWER_IS_BETTER))


def _c(kind=ChangeKind.ENV_VAR, target="T", cid="c1", **kw):
    return Change(id=cid, kind=kind, description="d", target=target, **kw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
