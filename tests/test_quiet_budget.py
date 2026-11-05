r"""
Tests for the quiet timeout as ONE budget for a whole campaign.

The bug
-------
The quiet timeout used to be a NUMBER handed to each thing that waits:
once before the lever, and again before every repetition. So a contended
machine waited the timeout, then waited it again, then waited it again per
repetition.

Observed on this box (2026-11-05): a campaign on a card that somebody else
was using waited 900 s, announced that it was measuring anyway, and then
waited ANOTHER 900 s before its first repetition. Two full timeouts for one
campaign that could not have measured anything — and the operator had no
idea which of those two timeouts they were in.

The fix is a DEADLINE rather than a number: everything that waits takes what
is left of the campaign's budget.

These tests pin the TOTAL. A test that only checks "the budget exists" would
pass while the double wait was still there.

Nothing here renders, touches the GPU, or writes outside a temp dir.
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import common, levers  # noqa: E402


class TestTheBudgetItself(unittest.TestCase):
    def test_a_fresh_budget_has_the_whole_timeout(self):
        b = common.QuietBudget(60.0)
        self.assertAlmostEqual(b.remaining, 60.0, delta=1.0)

    def test_the_remaining_time_falls(self):
        b = common.QuietBudget(60.0)
        time.sleep(0.05)
        self.assertLess(b.remaining, 60.0)

    def test_a_budget_is_never_negative(self):
        b = common.QuietBudget(0.0)
        self.assertEqual(b.remaining, 0.0)
        self.assertTrue(b.expired)

    def test_spent_time_is_recorded(self):
        b = common.QuietBudget(60.0)
        b.spend(12.0)
        self.assertAlmostEqual(b.waited_s, 12.0)

    def test_the_description_names_the_budget(self):
        b = common.QuietBudget(60.0)
        b.spend(12.0)
        text = b.describe()
        self.assertIn("12s", text)
        self.assertIn("60s", text)


class TestTheWaitHonoursTheBudget(unittest.TestCase):
    """A wait may take only what is LEFT of the campaign's budget."""

    def setUp(self):
        self.budget = common.QuietBudget(0.3)
        self.orig = common.other_gpu_processes
        common.other_gpu_processes = lambda *a, **kw: ["rival.exe:4962MiB"]
        self.addCleanup(lambda: setattr(common, "other_gpu_processes", self.orig))

    def test_the_wait_is_capped_by_the_budget(self):
        """It asks for 300 s; the budget says 0.3 s; it waits 0.3 s."""
        t0 = time.perf_counter()
        common.wait_for_no_others(timeout_s=300.0, poll_s=0.05, verbose=False,
                                  budget=self.budget)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 3.0,
                        f"the wait took {elapsed:.1f}s — it ignored the budget")

    def test_the_spent_time_is_charged_to_the_budget(self):
        common.wait_for_no_others(timeout_s=300.0, poll_s=0.05, verbose=False,
                                  budget=self.budget)
        self.assertGreater(self.budget.waited_s, 0.0)
        self.assertGreaterEqual(self.budget.exhausted, 1)

    def test_a_wait_without_a_budget_still_works(self):
        """The old signature — a timeout and no budget — must still work."""
        t0 = time.perf_counter()
        common.wait_for_no_others(timeout_s=0.2, poll_s=0.05, verbose=False)
        self.assertLess(time.perf_counter() - t0, 3.0)


class TestTheCampaignWaitsOnceNotPerPhase(unittest.TestCase):
    """THE TOTAL. This is the test that fails on the old code."""

    def setUp(self):
        self.orig = common.other_gpu_processes
        # Somebody else is on the card, and stays there.
        common.other_gpu_processes = lambda *a, **kw: ["rival.exe:4962MiB"]
        self.addCleanup(lambda: setattr(common, "other_gpu_processes", self.orig))

    def _campaign(self, timeout_s: float, reps: int) -> float:
        """Run the pre-lever wait and then every repetition's gate, and time
        the whole thing."""
        ctx = levers.Context(usd="x", quiet_timeout_s=timeout_s)
        budget = ctx.budget()
        t0 = time.perf_counter()
        # The wait before the lever.
        common.wait_for_no_others(timeout_s=timeout_s, poll_s=0.05, verbose=False,
                                  budget=budget)
        common.wait_for_quiet_gpu(timeout_s=timeout_s, poll_s=0.05, verbose=False,
                                  budget=budget)
        # And the gate before every repetition.
        gate = levers._quiet_gate(budget)
        for _ in range(reps):
            gate()
        return time.perf_counter() - t0

    def test_the_campaign_waits_at_most_the_timeout(self):
        """Four phases against a 1.5 s budget must not take 6 s.

        On the old code this was timeout x (1 + reps) — four times the
        budget, for one campaign.
        """
        elapsed = self._campaign(timeout_s=1.5, reps=3)
        self.assertLess(elapsed, 3.0,
                        f"the campaign waited {elapsed:.1f}s against a 1.5s "
                        f"budget — the timeout is being re-granted per phase")

    def test_more_repetitions_do_not_cost_more_waiting(self):
        """The budget is spent once. More repetitions must not buy more
        waiting — that was the whole point of the fix."""
        one = self._campaign(timeout_s=1.0, reps=1)
        three = self._campaign(timeout_s=1.0, reps=3)
        self.assertLess(abs(three - one), 1.0,
                        f"one repetition waited {one:.1f}s and three waited "
                        f"{three:.1f}s — the extra repetitions bought more "
                        f"waiting, so the budget is still per-phase")


class TestTheContextSharesOneBudget(unittest.TestCase):
    def test_the_budget_is_created_once(self):
        ctx = levers.Context(usd="x", quiet_timeout_s=30.0)
        first = ctx.budget()
        self.assertIs(first, ctx.budget(),
                      "the campaign must share ONE budget, not a fresh one "
                      "per wait")

    def test_the_budget_carries_the_context_timeout(self):
        ctx = levers.Context(usd="x", quiet_timeout_s=42.0)
        self.assertAlmostEqual(ctx.budget().timeout_s, 42.0)

    def test_a_pre_set_budget_is_kept(self):
        """A caller that made the budget — the campaign runner — keeps it."""
        ctx = levers.Context(usd="x", quiet_timeout_s=30.0)
        made = common.QuietBudget(30.0)
        ctx.quiet_budget = made
        self.assertIs(ctx.budget(), made)


if __name__ == "__main__":
    unittest.main(verbosity=2)
