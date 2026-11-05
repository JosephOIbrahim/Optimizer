"""
Tests for the measurement record: it must carry the RAW series, not only
the summary.

The failure this prevents
-------------------------
A measurement record used to hold summary statistics and nothing else: the
effect, the p-value, the drift, the verdict. The raw paired series were
computed, gated on, and then DISCARDED. A reader of the record could not
re-derive any of it — they had to take the harness's word for the number.

That is the exact position this repo was in before 2026-11-04, when a
+12.34% verdict (p=1.9e-05) sat in the ledger and turned out to be worth
nothing once the control was re-timed — the "+0.7%" that first appeared
against it was the phantom's own cross-session arithmetic, not a measurement
of the lever. It was retracted rather than deleted, and the superseded
baselines were left on disk, because a record nobody can check is a record
that can quietly change to suit the present.

So the record now carries what was actually measured:

    baseline_series           one value per repetition (the control)
    candidate_series          one value per repetition (the treatment)
    paired_diffs              candidate[i] - baseline[i]
    guardrail_ssim_baseline   the guardrail's own per-repetition scores
    guardrail_ssim_candidate

and these tests do the thing the record exists for: they recompute the
numbers FROM THE RECORD, as a reader who does not trust the harness would,
and check that the harness's own answer falls out.

Nothing here renders, touches the GPU, or writes outside a temp dir: the
arms are stubs, the contention probe is stubbed too, and the record is
written into a temporary directory rather than the real measurement root.
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import common, levers, optimize, paired  # noqa: E402

sys.path.insert(0, str(common.V12_DIR))
from v12.objective import paired_t_test, render_objective  # noqa: E402
from v12.objective import Direction  # noqa: E402


#: A control that stays put and a candidate that is genuinely faster, with
#: enough noise in the differences that the paired test has something to chew
#: on. The exact numbers do not matter; what matters is that they are a
#: real paired run and that the record can reproduce them.
BASELINE = [48.20, 48.10, 48.35, 48.05, 48.15]
CANDIDATE = [43.90, 43.55, 43.85, 43.60, 43.50]
SSIM_BASE = [0.9961, 0.9958, 0.9963, 0.9959, 0.9960]
SSIM_CAND = [0.9948, 0.9947, 0.9949, 0.9946, 0.9948]


def _arm(values):
    """A stub arm that hands back the next value in the list."""
    it = iter(values)

    def arm(index):
        return {"value": next(it)}
    return arm


def _no_gpu_probe(*a, **kw):
    """A contention probe that sees nobody on the card.

    The real probe asks the driver what else is on the GPU. These tests must
    not touch the card at all, so the probe is stubbed out — the same way
    test_paired_gate.py does it.
    """
    return []


class _RecordFixture(unittest.TestCase):
    """Builds a real record the way the harness builds one."""

    def setUp(self):
        p = mock.patch.object(common, "other_gpu_processes", _no_gpu_probe)
        p.start()
        self.addCleanup(p.stop)

    def _measure(self):
        """Run the paired engine on stub arms, gate it, and build a record.

        Everything here is the production path: `paired.Pair` for the series,
        `levers._finish` for the verdict and the detail, `optimize.build_record`
        for the shape of the file.
        """
        pair = paired.Pair("aov_variant", _arm(BASELINE), _arm(CANDIDATE),
                           reps=len(BASELINE), warmup=0, lower_is_better=True)
        res = pair.run()
        guards = {"ssim": (list(SSIM_BASE), list(SSIM_CAND))}
        obj = render_objective(resolution="1024x429", samples=128)
        with redirect_stdout(StringIO()):
            result = levers._finish("aov_variant", res, obj, guards)
        record = optimize.build_record("render", obj, {"name": "test card"},
                                       [("aov_variant", result)])
        return res, result, record

    def _written(self, record):
        """Write the record the way the harness writes it, into a temp dir."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with mock.patch.object(common, "MEASURE_ROOT", tmp), \
                    mock.patch.object(common, "CONFIG_PATH", tmp / "config.json"):
                path = common.write_measurement("optimize", record)
                text = path.read_text(encoding="utf-8")
        return text


class TestTheRecordCarriesTheRawSeries(_RecordFixture):
    """The series must be in the file, under names a reader can find."""

    def setUp(self):
        super().setUp()
        self.res, self.result, self.record = self._measure()
        self.block = self.record["results"]["aov_variant"]

    def test_the_three_series_are_present(self):
        for key in ("baseline_series", "candidate_series", "paired_diffs"):
            self.assertIn(key, self.block,
                          f"the record has no {key} — a reader cannot re-derive "
                          f"anything without it")

    def test_the_series_are_the_ones_that_were_measured(self):
        self.assertEqual(self.block["baseline_series"], BASELINE)
        self.assertEqual(self.block["candidate_series"], CANDIDATE)

    def test_the_summary_statistics_are_still_there(self):
        """The raw series ADD to the record; they do not replace it."""
        for key in ("effect", "p_value", "drift", "ok", "why"):
            self.assertIn(key, self.block)

    def test_the_guardrail_samples_are_persisted_too(self):
        """The guardrail's verdict is a number; its samples are the evidence."""
        self.assertEqual(self.block["guardrail_ssim_baseline"], SSIM_BASE)
        self.assertEqual(self.block["guardrail_ssim_candidate"], SSIM_CAND)

    def test_the_series_are_plain_lists_of_floats(self):
        """JSON has one number type and one array type. A numpy float or a
        tuple would not survive the write."""
        for key in ("baseline_series", "candidate_series", "paired_diffs"):
            series = self.block[key]
            self.assertIsInstance(series, list, f"{key} is not a list")
            for v in series:
                self.assertIsInstance(v, float, f"{key} holds {type(v).__name__}")
                self.assertNotIsInstance(v, bool)

    def test_the_series_survive_a_json_dump(self):
        json.dumps(self.block["baseline_series"])
        json.dumps(self.block["candidate_series"])
        json.dumps(self.block["paired_diffs"])


class TestTheRecordCanBeChecked(_RecordFixture):
    """The point of the raw series: the numbers can be recomputed from them.

    Every assertion here is what a reader of the record would do — take the
    series, apply the formula the harness says it applies, and see whether
    the same answer falls out.
    """

    def setUp(self):
        super().setUp()
        self.res, self.result, self.record = self._measure()
        self.block = self.record["results"]["aov_variant"]

    def test_the_effect_can_be_re_derived_from_the_recorded_series(self):
        """The same formula as paired.PairResult.effect_pct: the fractional
        improvement of the candidate over the baseline, signed so that
        positive means better. render_s is lower-is-better."""
        b = statistics.fmean(self.block["baseline_series"])
        c = statistics.fmean(self.block["candidate_series"])
        effect = (b - c) / abs(b)
        self.assertAlmostEqual(effect, self.block["effect"], places=9,
                               msg="the recorded effect is not the effect of the "
                                   "recorded series — the record contradicts itself")

    def test_the_paired_diffs_are_candidate_minus_baseline(self):
        base = self.block["baseline_series"]
        cand = self.block["candidate_series"]
        diffs = self.block["paired_diffs"]
        self.assertEqual(len(diffs), len(base))
        for i, d in enumerate(diffs):
            self.assertAlmostEqual(d, cand[i] - base[i], places=12,
                                   msg=f"pair {i} does not match its own series")

    def test_the_p_value_can_be_re_derived_from_the_recorded_diffs(self):
        p = paired_t_test(self.block["paired_diffs"], Direction.LOWER_IS_BETTER)
        self.assertAlmostEqual(p, self.block["p_value"], places=12)

    def test_the_drift_can_be_re_derived_from_the_recorded_baseline(self):
        base = self.block["baseline_series"]
        drift = (base[-1] - base[0]) / abs(base[0])
        self.assertAlmostEqual(drift, self.block["drift"], places=12)

    def test_the_recorded_means_match_the_recorded_series(self):
        self.assertAlmostEqual(statistics.fmean(self.block["baseline_series"]),
                               self.block["baseline_mean"], places=9)
        self.assertAlmostEqual(statistics.fmean(self.block["candidate_series"]),
                               self.block["candidate_mean"], places=9)


class TestTheRecordRoundTrips(_RecordFixture):
    """The file on disk must be the record, and must be readable JSON."""

    def setUp(self):
        super().setUp()
        self.res, self.result, self.record = self._measure()

    def test_the_record_survives_json_dumps_and_loads_unchanged(self):
        text = json.dumps(self.record)
        self.assertEqual(json.loads(text), self.record)

    def test_the_written_file_is_the_record(self):
        """What lands on disk is what was handed to the writer."""
        text = self._written(self.record)
        loaded = json.loads(text)
        for key, value in self.record.items():
            self.assertEqual(loaded[key], value,
                             f"the written record changed {key} on the way out")
        self.assertIn("recorded_utc", loaded)

    def test_the_written_file_is_strict_json(self):
        """Not `NaN`, not `Infinity` — a strict parser has to be able to read
        it, or the record cannot be checked by a tool that is not Python."""
        text = self._written(self.record)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        json.loads(text, parse_constant=_refuse_constant)

    def test_the_written_record_can_still_be_checked(self):
        """The check survives the write: the series are in the FILE, not just
        in the dict that was handed over."""
        loaded = json.loads(self._written(self.record))
        block = loaded["results"]["aov_variant"]
        b = statistics.fmean(block["baseline_series"])
        c = statistics.fmean(block["candidate_series"])
        self.assertAlmostEqual((b - c) / abs(b), block["effect"], places=9)


def _refuse_constant(name):
    raise AssertionError(f"the record contains {name}, which is not valid JSON")


class TestARefusedRunStillCarriesItsSeries(_RecordFixture):
    """A run the gate REFUSED is exactly the run a reader needs to see.

    When the control moved, the verdict is a refusal — and the effect is
    recorded as 0.0, because no effect was ever computed. The data that
    produced the refusal is still evidence, and a record that dropped it
    would make the refusal unauditable: the reader would have to trust the
    "why" text about a control that moved.
    """

    def test_a_drift_refused_record_keeps_the_series(self):
        drifting = [48.20, 48.10, 48.35, 48.05, 40.15]   # the control moved
        pair = paired.Pair("aov_variant", _arm(drifting), _arm(CANDIDATE),
                           reps=len(drifting), warmup=0, lower_is_better=True)
        res = pair.run()
        obj = render_objective(resolution="1024x429", samples=128)
        with redirect_stdout(StringIO()):
            result = levers._finish("aov_variant", res, obj, None)
        self.assertFalse(result.ok, "a control that moved must refuse the verdict")
        self.assertEqual(result.detail["baseline_series"], drifting)
        self.assertEqual(result.detail["candidate_series"], CANDIDATE)
        self.assertIn("BASELINE", result.why)
        self.assertIn("moved", result.why)


class TestNothingUnserialisableLeaksIn(_RecordFixture):
    """A record is written AFTER the measurement, so a TypeError at write
    time throws away an hour of GPU work and leaves nothing on disk."""

    def test_a_tuple_becomes_a_list(self):
        """Tuples serialise as JSON arrays, but a tuple in the record would
        not survive a round trip as the same type."""
        out = common.jsonable({"a": (1.0, 2.0)})
        self.assertEqual(out, {"a": [1.0, 2.0]})

    def test_a_numpy_scalar_becomes_a_python_number(self):
        import numpy as np
        self.assertEqual(common.jsonable(np.int64(7)), 7)
        self.assertEqual(common.jsonable(np.float32(1.5)), 1.5)
        self.assertIs(type(common.jsonable(np.int64(7))), int)
        self.assertIs(type(common.jsonable(np.bool_(True))), bool)

    def test_a_numpy_array_becomes_a_list(self):
        import numpy as np
        self.assertEqual(common.jsonable(np.array([1.0, 2.0])), [1.0, 2.0])

    def test_a_non_finite_float_becomes_null(self):
        """json writes NaN and Infinity by default, and neither is valid JSON.
        A record written that way cannot be re-read by a strict parser, and a
        record nobody can re-read is a record nobody can check."""
        self.assertIsNone(common.jsonable(float("nan")))
        self.assertIsNone(common.jsonable(float("inf")))
        self.assertIsNone(common.jsonable(-float("inf")))

    def test_an_unknown_object_does_not_take_the_record_down(self):
        class Weird:
            def __repr__(self):
                return "<weird>"
        out = common.jsonable({"x": Weird()})
        self.assertIsInstance(out["x"], str)
        json.dumps(out)

    def test_a_detail_with_a_numpy_value_still_writes(self):
        """The whole point: the write happens after the measurement, so it
        must not be the thing that fails."""
        import numpy as np
        record = {"results": {"aov_variant": {
            "effect": np.float64(0.09), "n": np.int64(5),
            "baseline_series": [np.float64(48.2), np.float64(48.1)],
        }}}
        text = self._written(record)
        loaded = json.loads(text)
        self.assertEqual(loaded["results"]["aov_variant"]["n"], 5)
        self.assertEqual(loaded["results"]["aov_variant"]["baseline_series"],
                         [48.2, 48.1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
