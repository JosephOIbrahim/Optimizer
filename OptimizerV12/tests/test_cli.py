"""
Tests for the CLI's evidence plumbing — how guardrail samples get from the
command line to the Gate.

Was: `objective.admits` denies a declared guardrail with no samples, and the CLI
had no way to supply any, so `run --objective diffusion` (lpips, clip) and
`run --objective render` (ssim) could never admit a MEASURED change.

Now the two halves arrive separately, each from where it was measured:
  * baseline side — `baseline --record SAMPLES --guardrail METRIC=SAMPLES`,
    stored in the Baseline next to the primary series and its Conditions;
  * candidate side — `run --guardrail-for KEY:METRIC=SAMPLES`, keyed by
    change_key exactly like --candidate-for.

Nothing here touches the machine, the ledger, or D:\\OptimizerV12: the Plane,
the baseline store and the ledger are all replaced.
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12 import baseline as baseline_mod  # noqa: E402
from v12 import cli, plane  # noqa: E402
from v12.baseline import Baseline, Conditions  # noqa: E402
from v12.objective import diffusion_objective  # noqa: E402

#: A FILE_LINE-style key: a drive letter's ':' must not be taken for the metric.
KEY = r"system|file|C:\Users\x\Engine.ini|/Script/Engine|r.Foo"


def _bl(guardrails=None) -> Baseline:
    obj = diffusion_objective()
    return Baseline(
        objective_id="diffusion:comfyui-sdxl", workload=asdict(obj.workload),
        metric=obj.primary.name, direction=obj.primary.direction.value,
        samples=[100.0, 100.1, 99.9, 100.0, 100.05],
        conditions=asdict(Conditions()), guardrail_samples=dict(guardrails or {}),
    )


class _FakePlane:
    """Records what run_cycle was handed. Never observes anything."""
    seen = {}

    def __init__(self, **kw):
        pass

    def run_cycle(self, **kw):
        _FakePlane.seen = kw
        return plane.CycleResult()


def _run(argv):
    args = cli.build_parser().parse_args(argv)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = args.func(args)
    return rc, out.getvalue()


class TestParseKeyedGuardrails(unittest.TestCase):
    obj = diffusion_objective()

    def parse(self, *items):
        return cli._parse_keyed_guardrails(list(items), self.obj)

    def test_key_metric_and_samples_are_split_on_the_last_separators(self):
        self.assertEqual(self.parse(f"{KEY}:lpips=0.21,0.20,0.22"),
                         {KEY: {"lpips": [0.21, 0.20, 0.22]}})

    def test_several_metrics_for_one_key_are_grouped(self):
        out = self.parse(f"{KEY}:lpips=0.2,0.2", f"{KEY}:clip=0.3,0.3")
        self.assertEqual(out, {KEY: {"lpips": [0.2, 0.2], "clip": [0.3, 0.3]}})

    def test_malformed_items_are_refused(self):
        for bad in ("lpips=0.2,0.2",            # no key
                    f"{KEY}:lpips",             # no '='
                    f"{KEY}:=0.2,0.2",          # no metric
                    f"{KEY}:lpips=",            # no samples
                    f"{KEY}:lpips=0.2",         # one sample cannot be tested
                    f"{KEY}:lpips=0.2,abc",     # not numbers
                    f"{KEY}:nonsense=0.2,0.2",  # unknown metric
                    f"{KEY}:ssim=0.9,0.9"):     # known, but not declared by diffusion
            with self.subTest(item=bad), self.assertRaises(SystemExit):
                self.parse(bad)

    def test_a_repeated_key_metric_is_refused(self):
        with self.assertRaises(SystemExit):
            self.parse(f"{KEY}:lpips=0.2,0.2", f"{KEY}:lpips=0.3,0.3")

    def test_guardrails_without_an_objective_are_refused(self):
        with self.assertRaises(SystemExit):
            cli._parse_keyed_guardrails([f"{KEY}:lpips=0.2,0.2"], None)


class TestRunThreadsGuardrails(unittest.TestCase):
    def setUp(self):
        _FakePlane.seen = {}
        for p in (mock.patch.object(plane, "Plane", _FakePlane),
                  mock.patch.object(baseline_mod, "load_latest",
                                    return_value=_bl({"lpips": [0.2, 0.21],
                                                      "clip": [0.3, 0.31]}))):
            p.start()
            self.addCleanup(p.stop)

    def test_candidate_guardrails_are_paired_with_the_baselines_own_series(self):
        _run(["run", "--objective", "diffusion",
              "--candidate-for", f"{KEY}=80,80.1,79.9,80,80.05",
              "--guardrail-for", f"{KEY}:lpips=0.2,0.2",
              "--guardrail-for", f"{KEY}:clip=0.3,0.3"])
        self.assertEqual(_FakePlane.seen.get("guardrail_samples"),
                         {KEY: {"lpips": ([0.2, 0.21], [0.2, 0.2]),
                                "clip": ([0.3, 0.31], [0.3, 0.3])}})

    def test_a_baseline_without_the_guardrail_pairs_with_nothing(self):
        with mock.patch.object(baseline_mod, "load_latest", return_value=_bl()):
            _, out = _run(["run", "--objective", "diffusion",
                           "--candidate-for", f"{KEY}=80,80.1",
                           "--guardrail-for", f"{KEY}:lpips=0.2,0.2"])
        self.assertEqual(_FakePlane.seen["guardrail_samples"],
                         {KEY: {"lpips": ([], [0.2, 0.2])}})
        self.assertIn("holds no series for guardrail(s) clip, lpips", out)

    def test_no_guardrails_given_is_stated_not_silent(self):
        _, out = _run(["run", "--objective", "diffusion",
                       "--candidate-for", f"{KEY}=80,80.1"])
        self.assertIsNone(_FakePlane.seen["guardrail_samples"])
        self.assertIn("without --guardrail-for", out)


class TestBaselineRecordsGuardrails(unittest.TestCase):
    def test_guardrail_series_reach_record(self):
        seen = {}

        def record(obj, samples, **kw):
            seen.update(kw)
            return _bl(kw.get("guardrail_samples"))
        ledger = mock.MagicMock()
        with mock.patch.object(baseline_mod, "record", side_effect=record), \
             mock.patch.object(cli, "open_ledger", return_value=ledger):
            _, out = _run(["baseline", "--objective", "diffusion",
                           "--record", "100,100.1,99.9",
                           "--guardrail", "lpips=0.2,0.21"])
        self.assertEqual(seen["guardrail_samples"], {"lpips": [0.2, 0.21]})
        self.assertIn("no baseline series for guardrail(s) clip", out)

    def test_an_undeclared_guardrail_is_refused_before_anything_is_written(self):
        with mock.patch.object(baseline_mod, "record",
                               side_effect=AssertionError("recorded")), \
             self.assertRaises(SystemExit):
            _run(["baseline", "--objective", "diffusion", "--record", "100,100.1",
                  "--guardrail", "ssim=0.9,0.9"])


if __name__ == "__main__":
    unittest.main()
