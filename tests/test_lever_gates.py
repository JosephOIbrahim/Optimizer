"""
Tests for the gate on every lever.

Every lever measures two arms ALTERNATELY in one session, and every lever
must WAIT for the card before each repetition. The second half is the part
that was missing.

`lever_husk_threads` had no `quiet_check` — it went straight to measuring.
That is the lever that produced the +12.34% phantom on 2026-11-03, so the
lever that was measured worst was also the one with the least protection:
no wait before a repetition, and (at the time) no per-rep contention check.
It now waits, like every other lever.

`lever_manual` had the same gap: the operator is already present and already
being asked to change the machine, but the card can still be busy with
somebody else's work.

These tests read the source, because the property is about which levers WIRE
THE GATE, and the cheapest honest way to check that is to look at the code
that builds each one. The gate's own behaviour is covered in
test_paired_gate.py and test_quiet_gpu.py.

Nothing here renders, touches the GPU, or writes outside a temp dir.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import levers  # noqa: E402


LEVERS_SRC = (REPO / "harness" / "levers.py").read_text(encoding="utf-8")


def _lever_body(name: str) -> str:
    """The source of one lever function, up to the next top-level def."""
    start = LEVERS_SRC.index(f"def {name}(")
    rest = LEVERS_SRC[start + 1:]
    nxt = rest.find("\ndef ")
    return LEVERS_SRC[start:start + 1 + nxt] if nxt != -1 else LEVERS_SRC[start:]


class TestEveryLeverWaitsForTheCard(unittest.TestCase):
    """Each lever must gate every repetition on the card being free."""

    def _assert_gated(self, name):
        body = _lever_body(name)
        self.assertIn("quiet_check", body,
                      f"{name} has no quiet gate — it must wait for the card "
                      f"before each repetition")
        self.assertIn("_quiet_gate(ctx.quiet_timeout_s)", body,
                      f"{name} must use the shared gate, not its own")

    def test_frames_per_process_is_gated(self):
        self._assert_gated("lever_frames_per_process")

    def test_aov_variant_is_gated(self):
        self._assert_gated("lever_aov_variant")

    def test_power_ceiling_is_gated(self):
        self._assert_gated("lever_power_ceiling")

    def test_husk_threads_is_gated(self):
        """The one that was missing. The lever that produced the +12.34%
        phantom was the one lever with no wait before a repetition."""
        self._assert_gated("lever_husk_threads")

    def test_manual_is_gated(self):
        self._assert_gated("lever_manual")

    def test_no_lever_is_left_ungated(self):
        """Belt and braces: every registered lever gates its repetitions."""
        for name in levers.LEVERS:
            with self.subTest(lever=name):
                self._assert_gated("lever_" + name)


class TestTheLeverRunAlwaysWaitsFirst(unittest.TestCase):
    """The gate before the run, not only before each repetition."""

    def test_run_waits_for_the_card_before_the_lever(self):
        body = LEVERS_SRC[LEVERS_SRC.index("def run("):]
        self.assertIn("wait_for_no_others", body)
        self.assertIn("wait_for_quiet_gpu", body)

    def test_both_wait_helpers_are_used_not_just_one(self):
        """Process identity catches a rival holding memory; utilisation
        catches work holding none. One without the other leaves a hole."""
        body = LEVERS_SRC[LEVERS_SRC.index("def run("):]
        self.assertIn("common.wait_for_no_others", body)
        self.assertIn("common.wait_for_quiet_gpu", body)


class TestTheGateIsTheOneThatKnowsAboutWsl(unittest.TestCase):
    """The gate must be the process-identity gate, not the old one.

    The old check asked whether the card was BUSY, which during our own
    render is always true. The new one asks who holds it.
    """

    def test_the_gate_uses_process_identity(self):
        body = _lever_body("_quiet_gate")
        self.assertIn("wait_for_no_others", body)

    def test_the_gate_also_checks_utilisation(self):
        """Valid only BEFORE a repetition starts, when our own workload is
        not holding the card. It catches work that holds no memory."""
        body = _lever_body("_quiet_gate")
        self.assertIn("wait_for_quiet_gpu", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestThePowerLeverLetsTheCardSettle(unittest.TestCase):
    """Changing the power ceiling changes the card's POWER STATE, and the
    card needs a moment to re-tune its boost clock to the new limit.

    Measured on this box (2026-11-05): the first GEMM after the ceiling was
    raised from 450 W to 600 W read 157.8 TFLOPS where the card settled at
    177.5 — an 11% deficit with nothing to do with the lever. The three
    warm-up matmuls `_run_gemm` does are worth about 20 ms, which is not
    enough for a 150 W limit change.

    The symptom is a control that moves around for no reason, which the
    drift gate then refuses — correctly, and with a stated reason ("something
    else was using the GPU") that is wrong. A measurement that cannot tell a
    settling card from a competing job is not measuring what it claims.
    """

    def test_the_power_levers_arms_settle(self):
        src = (REPO / "harness" / "levers.py").read_text(encoding="utf-8")
        body = src[src.index("def lever_power_ceiling("):]
        body = body[:body.index("\ndef ")]
        self.assertIn("settle_s=POWER_SETTLE_S", body,
                      "the power lever's arms must let the card settle at the "
                      "new ceiling before measuring")

    def test_the_settle_is_long_enough_to_matter(self):
        """It has to be longer than the warm-up matmuls it replaces, or it
        changes nothing."""
        self.assertGreater(levers.POWER_SETTLE_S, 0.5,
                           "a settle under half a second will not cover a "
                           "150 W limit change")

    def test_only_the_power_lever_settles(self):
        """The settle is for a power-state change. A lever that does not
        change the power state does not need one, and paying for it on every
        lever would slow the whole tool down for nothing."""
        src = (REPO / "harness" / "levers.py").read_text(encoding="utf-8")
        self.assertEqual(src.count("settle_s=POWER_SETTLE_S"), 2,
                         "exactly the power lever's two arms")
