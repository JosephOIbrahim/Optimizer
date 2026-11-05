"""
Tests for the warm-up repetition.

A warm-up repetition is run and DISCARDED before the measured series starts.
It exists because the first repetition of a workload pays for the machine
being COLD: a cold page cache, a cold shader cache, a cold BVH. Those are
costs of the machine warming up, not of the workload, and they are not
symmetrical between arms — the arm that launches more processes pays the cold
cost more times.

Found the hard way on 2026-11-05, on the shot path. The `frames_per_process`
lever rendered three frames one-per-process (the control arm) against three
frames in one process (the candidate). The control launches three husk
processes, so it paid the cold-cache cost three times and the first one paid
it hardest. With NO warm-up, that cold first repetition landed inside the
measured series:

    baseline  162.92  148.95  154.23      cv 4.54%
    drift     -5.34%   -> refused by the drift gate

The machine was idle. There were zero contention events. The gate was right
that the series was not steady, and wrong about WHY: it was a cold start, not
a changed machine. The fix is not to loosen the gate — it is to keep the cold
start out of the series, which is exactly what a warm-up is for.

Nothing here renders, touches the GPU, or writes outside a temp dir.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import levers, paired  # noqa: E402


def _arm(values):
    it = iter(values)

    def arm(index):
        return {"value": next(it)}
    return arm


class TestTheWarmUpIsExcluded(unittest.TestCase):
    def test_the_warm_up_repetition_is_not_in_the_series(self):
        res = paired.Pair("t", _arm([999.0, 100.0, 100.0, 100.0]),
                          _arm([888.0, 90.0, 90.0, 90.0]),
                          reps=3, warmup=1).run()
        self.assertNotIn(999.0, res.baseline.values)
        self.assertNotIn(888.0, res.candidate.values)
        self.assertEqual(len(res.baseline.values), 3)
        self.assertEqual(len(res.candidate.values), 3)

    def test_a_cold_first_repetition_does_not_read_as_drift(self):
        """The failure this exists to prevent.

        A cold first repetition of 162.92 followed by warm ones around 150
        reads as -5.3% drift and the gate refuses the campaign. With the
        cold repetition excluded, the series describes the machine.
        """
        res = paired.Pair("t", _arm([162.92, 148.95, 154.23, 150.1]),
                          _arm([84.20, 84.62, 87.74, 85.0]),
                          reps=3, warmup=1).run()
        self.assertFalse(res.drift_exceeded,
                         f"the warm series must not read as drift "
                         f"(it read {res.baseline_drift:+.2%})")

    def test_without_a_warm_up_the_same_data_does_read_as_drift(self):
        """Pins that the warm-up is what makes the difference, not luck."""
        res = paired.Pair("t", _arm([162.92, 148.95, 154.23]),
                          _arm([84.20, 84.62, 87.74]),
                          reps=3, warmup=0).run()
        self.assertTrue(res.drift_exceeded,
                        "without a warm-up the cold first repetition reads as drift")


class TestEveryLeverWarmsUp(unittest.TestCase):
    """A lever measured with no warm-up measures the machine's cold start."""

    def _body(self, name):
        src = (REPO / "harness" / "levers.py").read_text(encoding="utf-8")
        start = src.index(f"def {name}(")
        rest = src[start + 1:]
        nxt = rest.find("\ndef ")
        return src[start:start + 1 + nxt] if nxt != -1 else src[start:]

    def test_every_lever_uses_at_least_one_warm_up(self):
        for name in levers.LEVERS:
            with self.subTest(lever=name):
                body = self._body("lever_" + name)
                self.assertIn("warmup=max(ctx.warmup, 1)", body,
                              f"{name} must warm up at least once — the first "
                              f"repetition pays for a cold machine, not for the "
                              f"workload")

    def test_the_context_default_is_one(self):
        """`Context()` with no arguments must warm up.

        This was 0, which meant no lever ever warmed up through the command
        line — `from_args` did not set it either.
        """
        ctx = levers.Context(usd="x")
        self.assertEqual(ctx.warmup, 1)

    def test_the_command_line_can_ask_for_more(self):
        """An operator who wants a steadier series can pay for it."""
        class _Args:
            usd = "x"
            settings = ""
            samples = 128
            res_scale = 100
            start_frame = 503
            frames = 3
            reps = 3
            variant = "beauty_only"
            threads = 0
            warmup = 3
            quiet_timeout = 900
        self.assertEqual(levers.Context.from_args(_Args()).warmup, 3)

    def test_the_command_line_falls_back_to_the_default(self):
        class _Args:
            usd = "x"
            settings = ""
            samples = 128
            res_scale = 100
            start_frame = 503
            frames = 3
            reps = 3
            variant = "beauty_only"
            threads = 0
            quiet_timeout = 900
        self.assertEqual(levers.Context.from_args(_Args()).warmup, 1)


class TestTheWarmUpIsOnTheCommandLine(unittest.TestCase):
    def test_the_flag_exists(self):
        """An operator can see and change it."""
        src = (REPO / "harness" / "optimize.py").read_text(encoding="utf-8")
        self.assertIn('"--warmup"', src)

    def test_it_is_wired_into_the_context(self):
        src = (REPO / "harness" / "levers.py").read_text(encoding="utf-8")
        self.assertIn('getattr(args, "warmup"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
