"""
Tests for the retirement of The-Method.bat and the closure of the
stale-baseline comparison.

The thing being protected here is a specific false statement. On 2026-11-03
this repo gated a thread-count sweep against a baseline measured 48 minutes
earlier, under different resident processes, and recorded:

    ADMIT  +12.34%  p=1.882e-05

Re-timed the same way as the candidate, the same change was worth nothing —
measured with both arms alternately in one session it is DENY -0.60%. (The
"+0.7%" that first appeared against it was the phantom's own cross-session
arithmetic, not a measurement of the lever.) The Field Notes call it "the move
that evaporated". That verdict is still in the ledger.

A guardrail that can be removed by a later edit is not a guardrail, so these
tests check three things:

  * the retired launcher DELEGATES rather than measuring anything itself;
  * the two old modules REFUSE to produce a verdict from a stale baseline;
  * and the replacement measures both arms in one session, so there is
    exactly one way to measure a lever.

Nothing here renders, touches the GPU, or writes outside a temp dir.
"""
from __future__ import annotations

import io
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import levers, paired  # noqa: E402

THE_METHOD_BAT = REPO / "The-Method.bat"
OPTIMIZE_BAT = REPO / "Optimize-The-4090.bat"


class TestTheMethodIsRetired(unittest.TestCase):
    """The launcher must not measure anything itself any more."""

    def test_the_file_exists_and_is_a_batch_script(self):
        self.assertTrue(THE_METHOD_BAT.is_file())
        self.assertIn(".bat", THE_METHOD_BAT.name)

    def test_it_delegates_to_the_new_optimizer(self):
        text = THE_METHOD_BAT.read_text(encoding="utf-8", errors="replace")
        self.assertIn("Optimize-The-4090.bat", text,
                      "the retired launcher must hand over to the new one")

    def test_it_no_longer_drives_the_old_harness_modules(self):
        """The old modules are the ones that produced the ghost. A launcher
        that still called them would keep the ghost reachable."""
        text = THE_METHOD_BAT.read_text(encoding="utf-8", errors="replace")
        for old in ("harness\\method.py", "harness\\lever_run.py",
                     "harness/method.py", "harness/lever_run.py"):
            self.assertNotIn(old, text,
                             f"the retired launcher still references {old}")

    def test_it_says_why_it_was_retired(self):
        """A future reader has to be able to find the reason without
        archaeology, or the change looks like an unexplained deletion."""
        text = THE_METHOD_BAT.read_text(encoding="utf-8", errors="replace")
        self.assertRegex(text, r"12\.34|evaporat|another session",
                         "the retirement must state the failure that caused it")

    def test_it_passes_arguments_through(self):
        text = THE_METHOD_BAT.read_text(encoding="utf-8", errors="replace")
        self.assertIn("%*", text, "arguments must reach the new launcher")

    def test_the_new_optimizer_exists(self):
        self.assertTrue(OPTIMIZE_BAT.is_file())


class TestStaleComparisonsAreRefused(unittest.TestCase):
    """The two old modules must stop, not produce a wrong number."""

    def test_method_candidate_refuses(self):
        from harness import method
        with self.assertRaises(SystemExit) as ctx:
            method.cmd_candidate(_FakeArgs(label="anything"))
        msg = str(ctx.exception)
        self.assertIn("ANOTHER SESSION", msg)
        self.assertIn("12.34", msg, "the refusal must name the failure it prevents")

    def test_method_candidate_points_at_the_replacement(self):
        from harness import method
        with self.assertRaises(SystemExit) as ctx:
            method.cmd_candidate(_FakeArgs(label="anything"))
        self.assertIn("Optimize-The-4090.bat", str(ctx.exception))

    def test_lever_run_refuses(self):
        from harness import lever_run
        with self.assertRaises(SystemExit) as ctx:
            lever_run.refuse_stale_comparison()
        msg = str(ctx.exception)
        self.assertIn("ANOTHER SESSION", msg)
        self.assertIn("Optimize-The-4090.bat", msg)

    def test_the_refusal_names_both_arms(self):
        """A refusal that does not say WHY is indistinguishable from a crash."""
        from harness import method
        with self.assertRaises(SystemExit) as ctx:
            method.cmd_candidate(_FakeArgs(label="anything"))
        msg = str(ctx.exception)
        self.assertIn("drift", msg.lower())
        self.assertIn("5%", msg)


class TestOneWayToMeasureALever(unittest.TestCase):
    """Every front door must go through the same implementation."""

    def test_optimize_uses_the_shared_lever_module(self):
        from harness import optimize
        self.assertIs(optimize.levers, levers)

    def test_the_shared_module_owns_every_lever(self):
        self.assertEqual(sorted(levers.LEVERS),
                         ["aov_variant", "frames_per_process", "husk_threads",
                          "manual", "power_ceiling"])

    def test_the_ghost_lever_is_still_available_but_measured_honestly(self):
        """husk_threads produced the +12% phantom. It must not be deleted —
        a lever that cannot be measured is a lever that cannot be used — but
        it must be measured against a same-session control."""
        self.assertIn("husk_threads", levers.LEVERS)
        src = (REPO / "harness" / "levers.py").read_text(encoding="utf-8")
        self.assertIn("paired.Pair", src)
        self.assertIn("same session", src.lower())

    def test_every_lever_goes_through_the_paired_gate(self):
        src = (REPO / "harness" / "levers.py").read_text(encoding="utf-8")
        self.assertIn("def gate(", src)
        self.assertIn("res.drift_exceeded", src,
                      "the drift check must run before the statistics")
        self.assertIn("paired_diffs=res.diffs", src,
                      "the gate must use the paired test")

    def test_the_ssim_guardrail_is_scored_against_the_baseline(self):
        src = (REPO / "harness" / "levers.py").read_text(encoding="utf-8")
        self.assertIn("res.baseline.image_sets", src)
        self.assertIn("res.candidate.image_sets", src)


class _FakeArgs:
    def __init__(self, **kw):
        self.label = kw.get("label", "x")
        self.force = kw.get("force", False)
        self.usd = kw.get("usd", "")
        self.frame = kw.get("frame", 503)
        self.settings = kw.get("settings", "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
