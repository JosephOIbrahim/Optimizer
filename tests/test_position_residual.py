"""
How much does an ODD number of repetitions bias the paired difference?

The claim
---------
`harness/paired.py` alternates the arm order so that whichever arm runs
second does not always pay the first arm's warm-up. That cancels the
position effect ONLY at an EVEN count. At an odd count the candidate is
second in ceil(n/2) repetitions and the baseline in floor(n/2), so the mean
paired difference carries a residual of delta/n.

The operator card states this residual. This file pins the number, so the
card cannot drift away from the data.

The evidence is THIN and the file says so
-----------------------------------------
There is exactly ONE measurement record on this box with the raw series
needed to separate the position effect from each arm's own trend:
`optimize_20261105_065602.json` (the frames_per_process run, reps=3).

That gives one second-position sample per arm. It is not separately
identifiable from the linear trend across the run, and the two arms'
estimates DISAGREE by more than either estimate alone — which is the
signature of a sample too small to trust for a direction. So this file
reports an UPPER BOUND and refuses to assert a sign.

The residual is not a measurement of the position effect. It is the only
thing the single available record can say, and it is small: about 0.1 s on
the mean paired difference of a metric measured in seconds, which cannot
move a 5% effect threshold on its own.

Nothing here renders, touches the GPU, or writes outside a temp dir.
"""
from __future__ import annotations

import json
import statistics
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MEASURE = Path(r"E:\OptimizerV12\measurements\optimize_20261105_065602.json")


def _second_position_cost(series, second_in):
    """How much the SECOND position costs an arm, in seconds.

    Each arm's own linear trend is removed first, because a rising trend
    (thermal, cache warming) would otherwise be read as a position effect.
    What is left in each repetition is that repetition's residual; the mean
    residual over the repetitions where the arm ran second, minus the mean
    over the ones where it ran first, is the position's worth.
    """
    n = len(series)
    x = list(range(n))
    mx, my = statistics.fmean(x), statistics.fmean(series)
    num = sum((a - mx) * (b - my) for a, b in zip(x, series))
    den = sum((a - mx) ** 2 for a in x)
    slope = num / den if den else 0.0
    inter = my - slope * mx
    resid = [v - (inter + slope * i) for i, v in zip(x, series)]
    second = [resid[i] for i in range(n) if second_in[i]]
    first = [resid[i] for i in range(n) if not second_in[i]]
    return statistics.fmean(second) - statistics.fmean(first)


class TestThePositionResidual(unittest.TestCase):
    """The card's number, pinned to the record it came from."""

    @classmethod
    def setUpClass(cls):
        if not MEASURE.exists():
            raise unittest.SkipTest(f"{MEASURE} is not on this machine")
        cls.doc = json.loads(MEASURE.read_text(encoding="utf-8"))
        cls.v = cls.doc["results"]["frames_per_process"]
        cls.b = [float(x) for x in cls.v["baseline_series"]]
        cls.c = [float(x) for x in cls.v["candidate_series"]]

    def test_the_record_is_a_three_repetition_run(self):
        """The evidence is one odd-count run. Everything below is bounded
        by that."""
        self.assertEqual(len(self.b), 3)
        self.assertEqual(len(self.c), 3)

    def test_the_second_position_costs_the_candidate_about_a_third_of_a_second(self):
        # At reps=3 the order is: rep0 baseline-first, rep1 candidate-first,
        # rep2 baseline-first. So the BASELINE runs second in rep 1 only and
        # the CANDIDATE runs second in reps 0 and 2.
        base_cost = _second_position_cost(self.b, [False, True, False])
        cand_cost = _second_position_cost(self.c, [True, False, True])
        self.assertAlmostEqual(base_cost, 0.180, places=2)
        self.assertAlmostEqual(cand_cost, 0.341, places=2)

    def test_the_residual_on_the_mean_paired_difference(self):
        """The candidate is second twice and the baseline once, so the mean
        paired difference carries (2/3 - 1/3) of the second position's cost,
        divided by nothing — it is already a mean over n.

        Using the CANDIDATE's own second-position cost, because the candidate
        is the arm that runs second more often.
        """
        cand_cost = _second_position_cost(self.c, [True, False, True])
        residual = cand_cost / 3.0
        self.assertAlmostEqual(residual, 0.114, places=3)

    def test_the_residual_as_a_fraction_of_the_metric(self):
        """Against the baseline mean, which is what the effect is computed
        over: about 0.07%."""
        cand_cost = _second_position_cost(self.c, [True, False, True])
        residual = cand_cost / 3.0
        fraction = abs(residual) / statistics.fmean(self.b)
        self.assertAlmostEqual(fraction * 100, 0.065, places=2)

    def test_the_residual_cannot_move_a_five_percent_verdict(self):
        """The whole point of stating it: it eats into the margin, it does
        not decide the outcome."""
        cand_cost = _second_position_cost(self.c, [True, False, True])
        residual = cand_cost / 3.0
        fraction = abs(residual) / statistics.fmean(self.b)
        self.assertLess(fraction, 0.05,
                        "the residual must be far below the 5% threshold, or "
                        "the argument for leaving --reps alone does not hold")

    def test_the_sign_is_not_established(self):
        """Why no direction is claimed: the evidence is one sample per arm.

        The baseline runs second in exactly ONE repetition and the candidate
        in two, so each arm's second-position estimate rests on a single
        point. Two estimates from one point each cannot agree to the
        precision the claim needs, and here they differ by 0.161 s — most of
        the smaller estimate itself. That is what a sign NOT established
        looks like, and it is why this file reports an upper bound and the
        card states the residual without a direction.
        """
        base_cost = _second_position_cost(self.b, [False, True, False])
        cand_cost = _second_position_cost(self.c, [True, False, True])
        # One second-position sample for the baseline: one point per estimate.
        self.assertEqual(sum(1 for s in [False, True, False] if s), 1)
        # And the two one-point estimates disagree by most of the smaller one.
        self.assertGreater(abs(base_cost - cand_cost),
                           0.5 * min(abs(base_cost), abs(cand_cost)),
                           "if the two one-point estimates agreed closely, the "
                           "direction would be established and the card could "
                           "state one")


if __name__ == "__main__":
    unittest.main(verbosity=2)
