"""
Tests for the metric a lever produces and the objective it is scored against.

The failure this prevents
--------------------------
A lever is not a free-standing measurement: it produces ONE quantity, and an
objective scores ONE quantity. Score one against the other and the verdict is
about the wrong thing — and because the two quantities have OPPOSITE
directions (`render_s` is lower-is-better, `tflops` is higher-is-better), the
effect's SIGN INVERTS.

That is not a hypothetical. `--request compute --levers aov_variant` scored a
render_s measurement (lower is better) against the compute objective's tflops
(higher is better): the candidate rendering 9% SLOWER was recorded as
ADMIT +9.14%, p=2.25e-08. And `--request render --levers power_ceiling`
recorded a card 13% slower as effect +13.33%.

Nothing downstream can catch that. The arithmetic is perfectly correct
arithmetic about the wrong quantity; the ssim guardrail has no images to
compare on a GEMM; and the drift gate is about the control, not the metric.

So each lever DECLARES the metric it produces, and the runner REFUSES a
mismatch before measuring anything. The refusal names both quantities, because
a refusal that names only one leaves the operator to work out which half they
got wrong.

Nothing here renders, touches the GPU, or writes outside a temp dir: the
refusal happens before any arm is built, so no measurement is even started.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "OptimizerV12"))

from harness import levers, optimize  # noqa: E402
from v12.objective import (gemm_objective, render_objective,  # noqa: E402
                           shot_objective)

#: The objective each request scores, by the request's own name.
OBJECTIVE_FOR = {
    "compute": gemm_objective,
    "render": render_objective,
    "shot": shot_objective,
}


class TestEveryLeverDeclaresItsMetric(unittest.TestCase):
    """A lever that does not say what it measures cannot be checked."""

    def test_every_lever_declares_a_metric(self):
        for name in levers.LEVERS:
            with self.subTest(lever=name):
                self.assertIn(name, levers.LEVER_METRIC,
                              f"{name} produces a quantity but does not say "
                              f"which one — so nothing can refuse to score it "
                              f"against the wrong objective")

    def test_no_metric_is_declared_for_a_lever_that_does_not_exist(self):
        """A stale entry would make the map lie about the registry."""
        self.assertEqual(set(levers.LEVER_METRIC), set(levers.LEVERS))

    def test_every_declared_metric_is_a_real_metric(self):
        """A typo here would refuse every lever that names it."""
        from v12.objective import KNOWN_METRICS
        for name, metric in levers.LEVER_METRIC.items():
            with self.subTest(lever=name):
                self.assertIn(metric, KNOWN_METRICS,
                              f"{name} declares the metric {metric!r}, which "
                              f"is not a metric V12 knows")


class TestTheDeclaredMetricsAreTheRightOnes(unittest.TestCase):
    """The map has to be RIGHT, not just present.

    Each lever's metric is read off what its arms actually measure: a lever's
    two arms are two measurements of the SAME quantity, because comparing a
    candidate against a control of a different kind is the failure this whole
    mechanism exists to prevent.
    """

    def test_frames_per_process_measures_the_whole_shot(self):
        """It compares one-frame-per-process against all-frames-in-one-process,
        so it produces the shot's wall time — not one frame's shading phase."""
        self.assertEqual(levers.LEVER_METRIC["frames_per_process"], "shot_s")

    def test_aov_variant_measures_one_frames_shading_phase(self):
        """It compares two scenes' render phases, so it produces render_s."""
        self.assertEqual(levers.LEVER_METRIC["aov_variant"], "render_s")

    def test_husk_threads_measures_one_frames_shading_phase(self):
        self.assertEqual(levers.LEVER_METRIC["husk_threads"], "render_s")

    def test_power_ceiling_measures_throughput(self):
        """It runs the GEMM at two ceilings, so it produces tflops."""
        self.assertEqual(levers.LEVER_METRIC["power_ceiling"], "tflops")

    def test_manual_measures_one_frames_shading_phase(self):
        """It renders a single frame and takes its render phase."""
        self.assertEqual(levers.LEVER_METRIC["manual"], "render_s")


class TestAMetricMismatchIsRefused(unittest.TestCase):
    """The refusal is the whole point. It has to happen, and be clear."""

    def _ctx(self):
        return levers.Context(usd=r"E:\OptimizerV12\scenes\ovoid_hero_v002.usd")

    def test_compute_with_aov_variant_is_refused(self):
        """The exact case from the audit: the render lever against the
        compute objective, whose directions are opposite."""
        obj = gemm_objective()
        result = levers.run("aov_variant", self._ctx(), obj)
        self.assertFalse(result.ok, "a metric mismatch must not be admitted")
        self.assertIn("render_s", result.why)
        self.assertIn("tflops", result.why)

    def test_render_with_power_ceiling_is_refused(self):
        """The other case from the audit: a card 13% slower recorded as
        +13.33% because TFLOPS went up while render time went down."""
        obj = render_objective()
        result = levers.run("power_ceiling", self._ctx(), obj)
        self.assertFalse(result.ok)
        self.assertIn("tflops", result.why)
        self.assertIn("render_s", result.why)

    def test_the_refusal_names_both_quantities(self):
        """A refusal that names only one leaves the operator to work out
        which half they got wrong.

        The objective is named by its own workload — the compute request
        scores the gemm objective — so the refusal says which objective
        would have done the scoring.
        """
        obj = gemm_objective()
        result = levers.run("aov_variant", self._ctx(), obj)
        self.assertIn("aov_variant", result.why)
        self.assertIn(obj.workload.name, result.why)

    def test_the_refusal_explains_the_sign_inversion(self):
        """Why it is refused, not merely that it was: the two metrics point
        opposite ways, so a slower render reads as a win."""
        obj = gemm_objective()
        result = levers.run("aov_variant", self._ctx(), obj)
        self.assertIn("inverted", result.why.lower())

    def test_the_refusal_happens_BEFORE_anything_is_measured(self):
        """No arm is built, no card is touched, no time is spent.

        The wait for a quiet card is also skipped: there is no measurement
        coming, so waiting for one would be waiting for nothing.
        """
        obj = gemm_objective()
        with mock.patch.object(levers.common, "wait_for_no_others") as w1, \
                mock.patch.object(levers.common, "wait_for_quiet_gpu") as w2, \
                mock.patch.object(levers, "gemm_arm") as g, \
                mock.patch.object(levers, "render_frame_arm") as r:
            result = levers.run("aov_variant", self._ctx(), obj)
        self.assertFalse(result.ok)
        w1.assert_not_called()
        w2.assert_not_called()
        g.assert_not_called()
        r.assert_not_called()

    def test_a_matching_lever_is_not_refused(self):
        """The refusal must not fire on a correct pairing."""
        self.assertIsNone(levers.metric_mismatch("aov_variant", render_objective()))
        self.assertIsNone(levers.metric_mismatch("power_ceiling", gemm_objective()))
        self.assertIsNone(levers.metric_mismatch("frames_per_process", shot_objective()))

    def test_every_planned_pairing_is_a_matching_one(self):
        """The requests the tool plans for, checked against the map."""
        for request, objective in OBJECTIVE_FOR.items():
            obj = objective()
            for lever in ("frames_per_process", "aov_variant", "husk_threads",
                          "power_ceiling", "manual"):
                produced = levers.LEVER_METRIC[lever]
                if produced == obj.primary.name:
                    self.assertIsNone(
                        levers.metric_mismatch(lever, obj),
                        f"{lever} matches {request} but was refused")


class TestThePlanNeverProposesAMismatch(unittest.TestCase):
    """The plan and the runner must agree.

    If the plan proposed a lever the runner refuses, the operator is told a
    lever is worth trying and then told it cannot be measured — which is the
    same confusion the refusal was meant to end.
    """

    def test_the_plan_is_metric_clean(self):
        spec = {"name": "RTX 4090", "power_limit_w": 450.0,
                "power_limit_max_w": 600.0}
        for request, objective in OBJECTIVE_FOR.items():
            obj = objective()
            with self.subTest(request=request):
                planned = optimize.plan_levers(request, obj, spec, None)
                for name in planned:
                    self.assertIsNone(
                        levers.metric_mismatch(name, obj),
                        f"the plan for {request} proposes {name}, which the "
                        f"runner would refuse")

    def test_the_plan_gives_each_request_its_own_lever(self):
        """The tool still proposes something for every request — the refusal
        must not empty the plan."""
        spec = {"name": "RTX 4090", "power_limit_w": 450.0,
                "power_limit_max_w": 600.0}
        self.assertIn("power_ceiling",
                      optimize.plan_levers("compute", gemm_objective(), spec, None))
        self.assertIn("aov_variant",
                      optimize.plan_levers("render", render_objective(), spec, None))
        self.assertIn("frames_per_process",
                      optimize.plan_levers("shot", shot_objective(), spec, None))

    def test_the_plan_drops_a_power_lever_it_cannot_score(self):
        """The power lever measures TFLOPS, so under a render objective it
        cannot answer "would raising the ceiling make THIS faster"."""
        spec = {"name": "RTX 4090", "power_limit_w": 450.0,
                "power_limit_max_w": 600.0}
        verdict = mock.Mock(constraint=__import__(
            "harness.constraint", fromlist=["Constraint"]).Constraint.POWER)
        for request in ("render", "shot"):
            obj = OBJECTIVE_FOR[request]()
            with self.subTest(request=request):
                planned = optimize.plan_levers(request, obj, spec, verdict)
                self.assertNotIn("power_ceiling", planned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
