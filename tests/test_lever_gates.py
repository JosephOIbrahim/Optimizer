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
        # The gate draws on the CAMPAIGN's budget, not a fresh timeout per
        # phase. Handing it ctx.quiet_timeout_s again would let a contended
        # machine wait the timeout before the lever and then the timeout
        # again before every repetition.
        self.assertIn("_quiet_gate(ctx.budget())", body,
                      f"{name} must use the shared gate on the shared budget, "
                      f"not its own")

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
        self.assertIn("wait_for_quiet_gpu", body)

    def test_the_wait_asks_whether_anything_is_RENDERING(self):
        """Utilisation, not process identity.

        Waiting on process identity meant waiting for the operator to CLOSE
        their applications — Nuke and Houdini open and idle on this box —
        which is not something a measurement tool should ask of anyone. On
        2026-11-06 it made the tool wait the full timeout before measuring
        anything.

        The residents are still NAMED, so the operator can see who has the
        card. They are not waited on.
        """
        body = LEVERS_SRC[LEVERS_SRC.index("def run("):]
        self.assertIn("other_gpu_processes", body,
                      "the residents must still be named")
        self.assertNotIn("wait_for_no_others", body,
                         "the pre-lever wait must not block on apps merely "
                         "being open")


class TestTheGateWaitsOnActivityNotOnOpenApplications(unittest.TestCase):
    """The gate before each repetition.

    This is the one moment our own workload is not on the card, so the card's
    activity is a fair answer to "is anything rendering?".
    """

    def test_the_gate_uses_utilisation(self):
        body = _lever_body("_quiet_gate")
        self.assertIn("wait_for_quiet_gpu", body)

    def test_the_gate_does_not_wait_for_apps_to_close(self):
        body = _lever_body("_quiet_gate")
        self.assertNotIn("wait_for_no_others", body,
                         "waiting on process identity here means waiting for "
                         "the operator to close their applications")


class TestTheDriftGateStillKnowsAboutWsl(unittest.TestCase):
    """WSL GPU work is invisible to the Windows process table.

    The WATCHER still asks by process identity — that is how a WSL benchmark
    holding the card at 98% while no Windows process accounted for it was
    caught. It is the WAITS that changed, not the watcher.
    """

    def test_the_watcher_still_asks_who_holds_the_card(self):
        from harness import common
        src = (REPO / "harness" / "common.py").read_text(encoding="utf-8")
        self.assertIn("def other_gpu_processes", src)
        self.assertIn("class ContenderWatcher", src)
        # And the watcher is what marks a sample contended.
        self.assertIn("other_gpu_processes", src)
