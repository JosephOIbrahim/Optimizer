"""
Tests for the drift gate — the check that stands between a moved control and
a live false verdict.

The failure this exists to prevent
----------------------------------
On 2026-11-03 this repo gated a candidate against a baseline measured 48
minutes earlier and recorded ADMIT +12.34%, p=1.9e-05. The comparison
described two different machine states.

The harness was rebuilt so the control is measured in the SAME session as the
candidate, alternately, and gated with a paired t-test. Then a second defect
was found in that rebuild: the drift gate compared the FIRST control
repetition with the LAST. That sees a MONOTONE change and nothing else.

A busy window in the MIDDLE of a run leaves the first and last values equal,
so the drift reads +0.00% and the gate passes a control that moved
enormously. Reproduced through `levers.gate`:

    busy 52s for reps 2-4   drift +0.00%  PASS  ADMIT +6.75%  p=0.0128
    busy 56s for reps 2-4   drift +0.00%  PASS  ADMIT +10.98% p=0.0208
    busy 60s for reps 2-4   drift +0.00%  PASS  ADMIT +14.86% p=0.0247
    busy 70s for reps 2-4   drift +0.00%  PASS  ADMIT +23.20% p=0.0289

The control moved; the verdict blamed the candidate.

The fix is `control_spread` — the largest deviation of ANY repetition from
the control's median — which cannot miss a window anywhere in the run, plus
a refusal when the engine KNOWS a sample was contended.

Every one of those four rows is in the operator card, so they are written
down here as well: a card that quotes a figure the repo does not record is a
figure that can drift away from the code without anything failing.

The threshold still has to mean something, so these tests pin it against the
REAL recorded quiet-machine series on this box, not against invented ones.

Nothing here renders, touches the GPU, or writes outside a temp dir.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "OptimizerV12"))

from harness import levers, paired  # noqa: E402
from v12.objective import Direction, render_objective  # noqa: E402


def _result(base, cand):
    """A PairResult carrying two series, the way a run would."""
    res = paired.PairResult(baseline=paired.ArmResult("b"),
                             candidate=paired.ArmResult("c"))
    res.baseline.values = list(base)
    res.candidate.values = list(cand)
    res.diffs = [c - b for b, c in zip(base, cand)]
    res.baseline_drift = (base[-1] - base[0]) / abs(base[0])
    return res


#: The control series of real quiet-machine runs on this box, read from the
#: measurement records under E:\OptimizerV12\measurements. These are the
#: numbers the 5% limit has to clear.
QUIET_SERIES = {
    "render_193741": [48.06, 48.27, 47.48, 48.43, 48.06],
    "render_192419": [53.09, 53.37, 52.51, 52.46, 52.13],
    "render_192945": [47.92, 47.75, 47.53, 47.79, 47.56],
    "render_221724": [40.08, 40.33, 39.49, 39.61, 40.16],
    "render_224838": [29.49, 30.17, 30.25, 29.57, 29.49],
    "render_224502": [40.46, 40.2, 42.63, 42.82, 41.88],
    "gemm_072934": [160.45, 158.68, 159.47, 160.16],
    "gemm_072955": [174.98, 174.16, 174.83, 174.4],
    "shot_065602": [174.32, 176.35, 178.02],
}


class TestTheTransientWindowIsCaught(unittest.TestCase):
    """A busy window anywhere in the run must refuse the verdict."""

    def setUp(self):
        self.obj = render_objective(resolution="1024x429", samples=128)

    def _gate(self, base, cand=None):
        cand = cand if cand is not None else [47.0] * len(base)
        return levers.gate(_result(base, cand), self.obj,
                           {"ssim": ([0.996] * len(base), [0.995] * len(base))})

    def test_a_window_in_the_middle_is_refused(self):
        """The case the end-to-end statistic missed entirely."""
        for busy in (52, 56, 60, 70):
            with self.subTest(window=busy):
                base = [48.0, busy, busy, busy, 48.0]
                ok, why, _ = self._gate(base)
                self.assertFalse(ok, f"a {busy}s window must refuse the verdict")
                self.assertIn("middle", why)

    def test_the_end_to_end_drift_alone_would_have_passed(self):
        """Pins that the OLD statistic really did miss this shape."""
        base = [48.0, 60.0, 60.0, 60.0, 48.0]
        res = _result(base, [47.0] * 5)
        self.assertAlmostEqual(res.baseline_drift, 0.0, places=9)
        self.assertFalse(res.drift_exceeded,
                         "the old statistic reads this shape as steady")

    def test_the_figures_the_operator_card_quotes_are_exact(self):
        """The card publishes these four rows. Pin them here.

        The card's paragraph is the thing a reader actually has, so its
        numbers must be the ones the code produces — not a paraphrase and not
        a figure that drifted when the statistic changed.
        """
        # (busy seconds, spread, the effect the old gate would have admitted)
        rows = [
            (52, 0.0769, 0.0675),
            (56, 0.1429, 0.1098),
            (60, 0.2000, 0.1486),
            (70, 0.3143, 0.2320),
        ]
        for busy, want_spread, want_effect in rows:
            with self.subTest(window=busy):
                base = [48.0, float(busy), float(busy), float(busy), 48.0]
                res = _result(base, [47.0] * 5)
                self.assertAlmostEqual(res.control_spread, want_spread,
                                       places=3, msg=f"{busy}s spread")
                # And the effect the old statistic would have admitted.
                cand = [47.0] * 5
                eff = -(sum(cand) / 5 - sum(base) / 5) / (sum(base) / 5)
                self.assertAlmostEqual(eff, want_effect, places=3,
                                       msg=f"{busy}s effect")

    def test_a_window_at_the_start_is_refused(self):
        """The old statistic caught this one; it must still be caught."""
        ok, _, _ = self._gate([60.0, 60.0, 60.0, 48.0, 48.0])
        self.assertFalse(ok)

    def test_a_window_at_the_end_is_refused(self):
        ok, _, _ = self._gate([48.0, 48.0, 60.0, 60.0, 60.0])
        self.assertFalse(ok)

    def test_a_single_slow_repetition_is_refused(self):
        """One repetition 25% slower is not run-to-run noise."""
        ok, _, _ = self._gate([48.0, 48.0, 48.0, 48.0, 60.0])
        self.assertFalse(ok)


class TestTheQuietMachineStillPasses(unittest.TestCase):
    """The threshold must clear the machine's own noise.

    These are the REAL control series of quiet-machine runs on this box,
    read from the measurement records. A statistic that refuses these is
    refusing the machine, not a bad measurement.
    """

    def setUp(self):
        self.obj = render_objective(resolution="1024x429", samples=128)

    def test_every_recorded_quiet_series_clears_the_drift_check(self):
        for name, base in QUIET_SERIES.items():
            with self.subTest(series=name):
                res = _result(base, [x * 0.95 for x in base])
                self.assertFalse(res.control_exceeded,
                                 f"{name} is a quiet-machine series and must "
                                 f"clear the drift check (spread "
                                 f"{res.control_spread:.2%})")

    def test_the_worst_quiet_series_still_has_headroom(self):
        """The worst of them must not be sitting on the limit.

        The noisiest quiet series on this box is render_224502 at 4.01% —
        a multipass beauty scene whose own cv is 2.9%. The limit is 5%, so
        the worst case has about a fifth of the limit to spare, and a busy
        window of 52s on a 48s control is 7.7% and is caught.
        """
        worst = max(_result(b, b).control_spread for b in QUIET_SERIES.values())
        self.assertLess(worst, paired.PairResult.DRIFT_LIMIT,
                        f"the worst quiet series is at {worst:.2%}, at or over "
                        f"the {paired.PairResult.DRIFT_LIMIT:.0%} limit")
        # And the smallest busy window worth catching is still caught.
        busy = max(abs(52.0 - 48.0) / 52.0, 0.0)
        self.assertGreater(busy, worst,
                           "a busy window must exceed the quiet machine's "
                           "own spread, or the gate cannot tell them apart")

    def test_a_steady_control_admits_a_real_improvement(self):
        base = [48.0, 48.1, 47.9, 48.0, 48.05]
        cand = [43.5, 43.4, 43.6, 43.5, 43.55]
        ok, why, detail = levers.gate(_result(base, cand), self.obj,
                                      {"ssim": ([0.996] * 5, [0.995] * 5)})
        self.assertTrue(ok, why)
        self.assertAlmostEqual(detail["effect"], 0.094, places=2)


class TestContentionIsRecordedButTheControlDecides(unittest.TestCase):
    """A process holding a GPU CONTEXT is not a process RENDERING.

    The engine detects other GPU work on the card while a repetition runs.
    It used to count that and gate nothing; then, for a while, it counted that
    and REFUSED the verdict outright. Both were wrong.

    Refusing outright made the tool unable to measure on a normal desk. On
    2026-11-06 this box had Nuke and Houdini OPEN — idle, drawing 12%
    utilisation — so every repetition was flagged, every campaign was
    refused, and no measurement could be taken at all.

    Counting but not refusing is right, and the reason is the DIRECTION of
    the error:

      * a rival that hits the CONTROL moves the control, and the drift and
        spread checks catch it — they are checked FIRST, and they refuse;
      * a rival that hits the CANDIDATE arm makes the candidate read slow,
        so the effect is UNDERSTATED — a false DENY, not a false ADMIT.

    So contention that leaves the control steady can only move the verdict
    towards refusing it, which is the safe direction. A verdict that
    survives both checks was not made on a stolen card.
    """

    def setUp(self):
        self.obj = render_objective(resolution="1024x429", samples=128)

    def test_a_shared_card_with_a_steady_control_still_admits(self):
        base = [48.0, 48.1, 47.9, 48.0, 48.05]
        cand = [43.5, 43.4, 43.6, 43.5, 43.55]
        res = _result(base, cand)
        res.contended_samples = 1
        res.contenders = ["wsl:Ubuntu:1870: /python3.12"]
        ok, why, detail = levers.gate(res, self.obj,
                                     {"ssim": ([0.996] * 5, [0.995] * 5)})
        self.assertTrue(ok, why)

    def test_the_record_says_the_card_was_shared(self):
        """A reader who knows a rival was on the card can judge the number
        for themselves. A record that does not say so is asking them to
        trust it."""
        base = [48.0, 48.1, 47.9, 48.0, 48.05]
        res = _result(base, [43.5] * 5)
        res.contended_samples = 2
        res.contenders = ["houdinifx.exe:4962MiB"]
        ok, why, detail = levers.gate(res, self.obj,
                                     {"ssim": ([0.996] * 5, [0.995] * 5)})
        self.assertTrue(ok, why)
        self.assertEqual(detail["contended_samples"], 2)
        self.assertIn("houdinifx", detail["shared_card_note"])
        self.assertIn("houdinifx", why)

    def test_a_moved_control_is_refused_even_with_contention(self):
        """The control moving is the authority. The rivals are only the
        reason it might have moved."""
        base = [48.0, 60.0, 60.0, 60.0, 48.0]
        res = _result(base, [47.0] * 5)
        res.contended_samples = 3
        res.contenders = ["houdinifx.exe:4962MiB"]
        ok, why, _ = levers.gate(res, self.obj,
                                 {"ssim": ([0.996] * 5, [0.995] * 5)})
        self.assertFalse(ok)
        self.assertIn("middle", why)


class TestAShortPairedSeriesIsRefused(unittest.TestCase):
    """A paired request must not be silently downgraded to an independent one.

    Welch's t-test on two independent series cannot distinguish drift from
    effect — that is what "independent" means — and it is the exact
    arithmetic that produced the +12.34% phantom. So a paired comparison
    that cannot be tested as paired is refused, not downgraded.
    """

    def setUp(self):
        self.obj = render_objective(resolution="1024x429", samples=128)

    def _guards(self):
        """The objective DECLARES an ssim guardrail, so one must be supplied
        or the gate refuses for a different reason."""
        return {"ssim": ([0.996] * 5, [0.995] * 5)}

    def test_one_paired_difference_is_refused(self):
        ok, why, _ = self.obj.admits([48.0] * 5, [43.5] * 5, None,
                                      paired_diffs=[0.1])
        self.assertFalse(ok)
        self.assertIn("paired", why.lower())

    def test_an_empty_paired_series_is_refused(self):
        ok, why, _ = self.obj.admits([48.0] * 5, [43.5] * 5, None,
                                      paired_diffs=[])
        self.assertFalse(ok)
        self.assertIn("paired", why.lower())

    def test_the_refusal_names_the_why(self):
        _, why, _ = self.obj.admits([48.0] * 5, [43.5] * 5, None,
                                     paired_diffs=[0.1])
        self.assertIn("2", why, "the reason should say how many are needed")
        self.assertIn("drift", why.lower(),
                      "the reason should say why an independent test will not do")

    def test_a_proper_paired_series_still_admits(self):
        base = [48.0, 48.1, 47.9, 48.0, 48.05]
        cand = [43.5, 43.4, 43.6, 43.5, 43.55]
        ok, why, detail = self.obj.admits(base, cand, self._guards(),
                                          paired_diffs=[c - b for b, c in
                                                        zip(base, cand)])
        self.assertTrue(ok, why)
        self.assertEqual(detail.get("paired"), 1.0)

    def test_the_no_paired_diffs_path_still_works(self):
        """No request for a paired comparison means the independent test is
        the right one — that is the V12 admission path's own behaviour."""
        base = [48.0, 48.1, 47.9, 48.0, 48.05]
        cand = [43.5, 43.4, 43.6, 43.5, 43.55]
        ok, why, detail = self.obj.admits(base, cand, self._guards(),
                                          paired_diffs=None)
        self.assertTrue(ok, why)
        self.assertNotIn("paired", detail)


class TestTheOldNumberIsKept(unittest.TestCase):
    """Records written before this change stay comparable."""

    def test_the_end_to_end_drift_is_still_reported(self):
        base = [48.0, 48.5, 49.0, 49.5, 50.0]
        res = _result(base, [47.0] * 5)
        self.assertAlmostEqual(res.baseline_drift, (50.0 - 48.0) / 48.0, places=9)

    def test_the_spread_is_a_separate_number(self):
        base = [48.0, 60.0, 60.0, 60.0, 48.0]
        res = _result(base, [47.0] * 5)
        self.assertAlmostEqual(res.baseline_drift, 0.0, places=9)
        # The median is 60, so the endpoints sit 12/60 = 20% away from it.
        self.assertAlmostEqual(res.control_spread, 0.20, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
