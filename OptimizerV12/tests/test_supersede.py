"""
Tests for retracting a verdict and superseding a baseline.

The thing being protected is a false statement. On 2026-11-03 this repo gated a
thread-count sweep against a baseline measured 48 minutes earlier, under
different resident processes, and recorded:

    ADMIT  +12.34%  p=1.882e-05

That was not a comparison of anything. Re-timed the same way as the candidate,
the same change was worth nothing (measured again with both arms alternately
in one session: DENY -0.89% on 2026-11-04, and DENY -0.60% on 2026-11-05).
(The "+0.7%" that first appeared against it was the phantom's own
cross-session arithmetic, not a measurement of the lever.)

Two failure modes had to be closed:

  * the FALSE VERDICT staying in the store, where the next reader inherits it;
  * and the opposite one — deleting or rewriting it — which erases the record
    that it was ever made. The ledger counts denials as carefully as
    admissions, so a vanished denial is a lie in the other direction.

So a retraction KEEPS the original text and marks it. A supersede keeps the
numbers and withdraws them as a reference. Neither deletes anything.

Nothing here touches the real ledger, the real baselines, or the GPU: the
tests build their own in a temp dir.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from v12 import baseline as bmod  # noqa: E402
from v12.objective import Objective, Workload, WorkloadKind  # noqa: E402
from v12.objective import KNOWN_METRICS  # noqa: E402

from harness import retract  # noqa: E402


def _obj(name="karma-xpu", metric="render_s"):
    return Objective(
        workload=Workload(WorkloadKind.RENDER, name, "1024x429", 128, ""),
        primary=KNOWN_METRICS[metric],
        min_effect=0.05,
        min_samples=5,
        guardrails={},
    )


class TestSupersedeRequiresAReason(unittest.TestCase):
    """A withdrawal without a reason is indistinguishable from a deletion."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v12_supersede_"))
        self._orig_dir = bmod.paths.BASELINE_DIR
        bmod.paths.BASELINE_DIR = self.tmp

    def tearDown(self):
        bmod.paths.BASELINE_DIR = self._orig_dir

    def _baseline(self, bid="b1"):
        b = bmod.Baseline(
            objective_id="render:karma-xpu",
            workload=_obj().workload.__dict__ if hasattr(_obj().workload, "__dict__")
            else {"kind": "render", "name": "karma-xpu"},
            metric="render_s", direction="lower_is_better",
            samples=[100.0, 100.1, 99.9, 100.0, 100.05],
            conditions=bmod.Conditions().__dict__,
        )
        b.id = bid
        b.save()
        return b

    def test_a_reason_is_mandatory(self):
        b = self._baseline()
        with self.assertRaises(ValueError) as ctx:
            bmod.supersede(b.id, "")
        self.assertIn("reason", str(ctx.exception).lower())

    def test_superseding_marks_the_baseline_and_keeps_the_numbers(self):
        b = self._baseline()
        out = bmod.supersede(b.id, "measured against a stale control")
        self.assertTrue(out.superseded)
        self.assertIn("stale control", out.superseded_reason)
        # The numbers survive: they were real measurements of a real machine.
        self.assertEqual(len(out.samples), 5)
        self.assertAlmostEqual(out.mean, 100.01, places=2)

    def test_a_superseded_baseline_is_refused_by_the_gate(self):
        b = self._baseline()
        bmod.supersede(b.id, "measured against a stale control")
        reloaded = bmod.Baseline.load(self.tmp / f"{b.id}.json")
        ok, why = reloaded.valid_for(_obj())
        self.assertFalse(ok)
        self.assertIn("SUPERSEDED", why)

    def test_a_superseded_baseline_is_skipped_by_load_latest(self):
        b = self._baseline()
        bmod.supersede(b.id, "measured against a stale control")
        self.assertIsNone(bmod.load_latest(_obj()),
                          "a withdrawn baseline must not be offered as a control")

    def test_superseding_an_unknown_baseline_refuses(self):
        with self.assertRaises(ValueError) as ctx:
            bmod.supersede("nope", "because")
        self.assertIn("no baseline", str(ctx.exception).lower())


class TestRetractionKeepsTheOriginalVerdict(unittest.TestCase):
    """The record must say what was claimed AND that it does not hold."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="retract_"))
        self.path = self.tmp / "levers.json"
        self.path.write_text(json.dumps({
            "objective": "render",
            "baseline": "somewhere",
            "baseline_mean_s": 54.426,
            "results": [
                {"lever": "husk_threads", "variant": 8, "label": "husk_j8",
                 "verdict": "DENY  (-38.38%)", "mean_s": 75.312},
                {"lever": "husk_threads", "variant": 48, "label": "husk_j48",
                 "verdict": "ADMIT  (+12.34%)  p=1.882e-05, n=5/5",
                 "mean_s": 47.71},
            ],
            "recorded_utc": "2026-11-04T00:29:45Z",
        }, indent=2), encoding="utf-8")

    def _doc(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_the_original_verdict_is_preserved(self):
        retract.retract_result(self.path, "husk_j48", "gated against a stale baseline")
        r = [x for x in self._doc()["results"] if x["label"] == "husk_j48"][0]
        self.assertIn("ADMIT", r["retracted"]["original_verdict"])
        self.assertIn("+12.34", r["retracted"]["original_verdict"])

    def test_the_live_verdict_is_marked_so_a_grep_cannot_miss_it(self):
        retract.retract_result(self.path, "husk_j48", "gated against a stale baseline")
        r = [x for x in self._doc()["results"] if x["label"] == "husk_j48"][0]
        self.assertTrue(r["verdict"].startswith("RETRACTED"))
        self.assertIn("+12.34", r["verdict"], "the original text stays visible")

    def test_the_reason_and_the_replacement_travel_with_it(self):
        retract.retract_result(self.path, "husk_j48", "gated against a stale baseline",
                               replacement="E:\\m.json",
                               replacement_verdict="DENY (-0.89%)")
        r = [x for x in self._doc()["results"] if x["label"] == "husk_j48"][0]
        self.assertIn("stale baseline", r["retracted"]["reason"])
        self.assertIn("DENY (-0.89%)", r["retracted"]["replacement_verdict"])
        self.assertIn("m.json", r["retracted"]["replacement_measurement"])

    def test_other_results_are_untouched(self):
        retract.retract_result(self.path, "husk_j48", "gated against a stale baseline")
        r = [x for x in self._doc()["results"] if x["label"] == "husk_j8"][0]
        self.assertNotIn("retracted", r)
        self.assertTrue(r["verdict"].startswith("DENY"))

    def test_a_retraction_needs_a_reason(self):
        with self.assertRaises(ValueError):
            retract.retract_result(self.path, "husk_j48", "")

    def test_retracting_an_unknown_label_refuses(self):
        with self.assertRaises(ValueError) as ctx:
            retract.retract_result(self.path, "nope", "because")
        self.assertIn("no result labelled", str(ctx.exception))

    def test_retracting_twice_identically_is_a_no_op(self):
        retract.retract_result(self.path, "husk_j48", "because")
        before = self.path.read_text(encoding="utf-8")
        retract.retract_result(self.path, "husk_j48", "because")
        self.assertEqual(before, self.path.read_text(encoding="utf-8"))

    def test_retracting_twice_differently_refuses(self):
        """A second, different reason is a decision someone must make
        deliberately — not something to overwrite silently."""
        retract.retract_result(self.path, "husk_j48", "because one")
        with self.assertRaises(ValueError):
            retract.retract_result(self.path, "husk_j48", "because two")

    def test_the_file_stays_valid_json(self):
        retract.retract_result(self.path, "husk_j48", "because")
        json.loads(self.path.read_text(encoding="utf-8"))


class TestSupersedeWholeRecord(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="supersede_rec_"))
        self.path = self.tmp / "render_x.json"
        self.path.write_text(json.dumps({
            "label": "baseline", "render_s": [48.0, 48.1], "summary": {"mean": 48.05},
        }, indent=2), encoding="utf-8")

    def test_the_numbers_survive(self):
        retract.mark_superseded(self.path, "cannot serve as a control")
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIn("superseded", d)
        self.assertEqual(d["render_s"], [48.0, 48.1])
        self.assertEqual(d["summary"]["mean"], 48.05)

    def test_it_needs_a_reason(self):
        with self.assertRaises(ValueError):
            retract.mark_superseded(self.path, "")


class TestCorrectingASupersedeReason(unittest.TestCase):
    """A supersede reason is a factual claim about the record, and it can be
    wrong in the same way a retraction's reason can be wrong.

    Found on 2026-11-05: the reason attached to one superseded record
    attributed ANOTHER record's numbers to it — a control that drifted -33.8%
    and a baseline mean of 136.81 TFLOPS, which belong to a sibling, while
    the record it was attached to had a baseline mean of 159.78 and could not
    show that starvation at all.

    A wrong reason is worse than no reason, because it is a confident false
    statement about a measurement, and a reader has no way to tell it from a
    true one. But overwriting it is the same failure the whole module exists
    to prevent: the record of what was claimed disappears.

    So the correction is recorded ALONGSIDE, the earlier text stays readable,
    and corrections chain.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="correct_supersede_"))
        self.path = self.tmp / "optimize_x.json"
        self.path.write_text(json.dumps({
            "request": "compute",
            "results": {"power_ceiling": {"ok": True, "why": "+9.75% improvement"}},
        }, indent=2), encoding="utf-8")
        retract.mark_superseded(
            self.path,
            "the control drifted -33.8%, so the baseline mean of 136.81 TFLOPS "
            "is a starved control")

    def _block(self):
        d = json.loads(self.path.read_text(encoding="utf-8"))
        return d["superseded"]

    def test_the_original_reason_stays_readable(self):
        retract.correct_supersede_reason(self.path, "the correct reason")
        self.assertIn("-33.8%", self._block()["reason"],
                      "the original text must survive the correction")

    def test_the_correction_is_recorded(self):
        retract.correct_supersede_reason(self.path, "the correct reason")
        block = self._block()
        self.assertIn("reason_corrected", block)
        self.assertIn("the correct reason", block["reason_corrected"]["reason"])
        self.assertTrue(block["reason_corrected"]["date"])

    def test_the_measurement_is_untouched(self):
        """The correction is about the REASON. The numbers stay as they were."""
        retract.correct_supersede_reason(self.path, "the correct reason")
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(d["results"]["power_ceiling"]["why"],
                         "+9.75% improvement")
        self.assertTrue(d["results"]["power_ceiling"]["ok"])

    def test_a_second_correction_does_not_erase_the_first(self):
        """Corrections chain, the way replacements chain."""
        retract.correct_supersede_reason(self.path, "the first correction")
        retract.correct_supersede_reason(self.path, "the second correction")
        block = self._block()
        self.assertIn("the second correction", block["reason_corrected"]["reason"])
        chain = block.get("reason_correction_history")
        self.assertTrue(chain, "the earlier correction must stay visible")
        self.assertIn("the first correction", chain[0]["reason"])

    def test_repeating_the_same_correction_adds_no_chain(self):
        retract.correct_supersede_reason(self.path, "the same correction")
        retract.correct_supersede_reason(self.path, "the same correction")
        self.assertIsNone(self._block().get("reason_correction_history"),
                          "repeating a correction is not a new correction")

    def test_correcting_a_record_that_is_not_superseded_refuses(self):
        fresh = self.tmp / "fresh.json"
        fresh.write_text(json.dumps({"results": {}}), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            retract.correct_supersede_reason(fresh, "a reason")
        self.assertIn("not marked superseded", str(ctx.exception))

    def test_a_correction_needs_text(self):
        with self.assertRaises(ValueError):
            retract.correct_supersede_reason(self.path, "")


class TestUpdateReplacement(unittest.TestCase):
    """A retraction is often made BEFORE its replacement has been measured.

    That is a legitimate order of operations: you know the verdict does not
    hold, and the honest re-measurement is still to come. This is the
    operation for it, and it must not lose the audit trail — the original
    reason may contain a claim ("no replacement figure exists yet") that the
    later measurement makes false, and overwriting the reason would be the
    exact failure this mechanism exists to prevent: a record that changes to
    suit the present.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="update_repl_"))
        self.path = self.tmp / "levers.json"
        self.path.write_text(json.dumps({
            "objective": "render",
            "results": [{
                "label": "aov_no_crypto",
                "verdict": "ADMIT  (+11.25%)  p=4.615e-07, n=5/5",
                "mean_s": 42.652,
            }],
        }, indent=2), encoding="utf-8")
        retract.retract_result(self.path, "aov_no_crypto",
                               "gated against a baseline measured 2h50m earlier. "
                               "The no_crypto variant was not re-measured yet, so it "
                               "has no replacement figure of its own.",
                               replacement="E:\\other.json",
                               replacement_verdict="ADMIT (+9.19%) — for the beauty_only variant")

    def _retracted(self):
        d = json.loads(self.path.read_text(encoding="utf-8"))
        return d["results"][0]["retracted"]

    def test_the_replacement_pointer_is_updated(self):
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\new.json", "DENY (+0.85%), paired, n=5")
        ret = self._retracted()
        self.assertEqual(ret["replacement_measurement"], "E:\\new.json")
        self.assertIn("DENY (+0.85%)", ret["replacement_verdict"])

    def test_the_original_reason_is_preserved(self):
        """The retraction's reason is a factual claim about what was wrong.
        Replacing it would erase the record of that claim."""
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\new.json", "DENY (+0.85%)")
        ret = self._retracted()
        self.assertIn("not re-measured yet", ret["reason"])

    def test_a_corrected_reason_is_recorded_alongside_not_over(self):
        """When the original reason contains a claim the new measurement makes
        false, the correction is ADDED. Both stay readable."""
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\new.json", "DENY (+0.85%)",
                                   corrected_reason="Re-measured 2026-11-04: the no_crypto "
                                                   "variant is +0.85%, DENY — not a win at all.")
        ret = self._retracted()
        self.assertIn("not re-measured yet", ret["reason"], "the original reason must survive")
        self.assertIn("reason_corrected", ret)
        self.assertIn("not a win at all", ret["reason_corrected"]["reason"])
        self.assertTrue(ret["reason_corrected"]["date"])

    def test_updating_a_verdict_that_is_not_retracted_refuses(self):
        fresh = self.tmp / "fresh.json"
        fresh.write_text(json.dumps({"results": [{"label": "x", "verdict": "ADMIT"}]}),
                         encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            retract.update_replacement(fresh, "x", "E:\\new.json")
        self.assertIn("not retracted", str(ctx.exception))

    def test_updating_an_unknown_label_refuses(self):
        with self.assertRaises(ValueError):
            retract.update_replacement(self.path, "nope", "E:\\new.json")

    def test_a_superseded_replacement_stays_visible(self):
        """A measurement can be superseded more than once — the first
        replacement may itself have been taken on a busy machine. A chain of
        them is the honest record; overwriting leaves no trace that there
        ever was one."""
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\second.json", "DENY (+0.85%)")
        ret = self._retracted()
        self.assertEqual(ret["replacement_measurement"], "E:\\second.json")
        hist = ret.get("replacement_history")
        self.assertTrue(hist, "the previous replacement must stay visible")
        self.assertIn("E:\\other.json", hist[0]["measurement"])
        self.assertIn("9.19%", hist[0]["verdict"])

    def test_updating_to_the_same_measurement_adds_no_history(self):
        """Not a supersede — the same file, still current."""
        before = self.path.read_text(encoding="utf-8")
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\other.json", "DENY (-0.89%)")
        ret = self._retracted()
        self.assertIsNone(ret.get("replacement_history"),
                          "re-pointing at the same file is not a supersede")

    def test_a_third_measurement_extends_the_chain_without_losing_it(self):
        """The chain is the record. Each superseded replacement stays
        visible, so a reader can see every measurement that was made and
        why each stopped being current."""
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\second.json", "DENY (+0.85%)")
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\third.json", "DENY (-0.12%)",
                                   corrected_reason="re-measured on an idle card")
        ret = self._retracted()
        self.assertEqual(ret["replacement_measurement"], "E:\\third.json")
        hist = ret["replacement_history"]
        self.assertEqual(len(hist), 2, "both superseded replacements stay visible")
        self.assertEqual(hist[0]["measurement"], "E:\\other.json")
        self.assertEqual(hist[1]["measurement"], "E:\\second.json")
        self.assertIn("9.19%", hist[0]["verdict"])
        self.assertIn("0.85%", hist[1]["verdict"])

    def test_the_original_verdict_survives_every_update(self):
        """The retraction's own account of what was claimed, and why it does
        not hold, is never touched by a later measurement."""
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\second.json", "DENY (+0.85%)")
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\third.json", "DENY (-0.12%)")
        ret = self._retracted()
        self.assertIn("ADMIT", ret["original_verdict"])
        self.assertIn("+11.25", ret["original_verdict"])
        self.assertIn("not re-measured yet", ret["reason"])

    def test_a_second_correction_does_not_erase_the_first(self):
        """Corrections chain, the way replacements chain.

        The correction is the record of what the retraction's own reason got
        wrong. If a later update overwrites it, the record of the earlier
        error is gone — and that is the same failure as losing an earlier
        measurement, in the same place: a record that changed to suit the
        present.

        Found the hard way on 2026-11-05, when a corrected reason containing
        a units error (seconds written as percentages) had to be replaced.
        The error must be visible in the chain, not silently erased.
        """
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\second.json", "DENY (+0.85%)",
                                   corrected_reason="the first correction, which had a units error")
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\third.json", "DENY (-0.12%)",
                                   corrected_reason="the second correction, which is right")
        ret = self._retracted()
        self.assertIn("the second correction", ret["reason_corrected"]["reason"])
        chain = ret.get("reason_correction_history")
        self.assertTrue(chain, "the earlier correction must stay visible")
        self.assertIn("units error", chain[0]["reason"])
        self.assertIn("the second correction", chain[0]["superseded_because"])

    def test_repeating_the_same_correction_adds_no_chain(self):
        """Not a correction of the correction — the same text, still current."""
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\second.json", "DENY (+0.85%)",
                                   corrected_reason="the same correction")
        retract.update_replacement(self.path, "aov_no_crypto",
                                   "E:\\second.json", "DENY (+0.85%)",
                                   corrected_reason="the same correction")
        ret = self._retracted()
        self.assertIsNone(ret.get("reason_correction_history"),
                          "repeating a correction is not a new correction")


class TestRetractionInTheOptimizeRecordShape(unittest.TestCase):
    """The retraction has to reach BOTH shapes of measurement record.

    Two shapes are on disk, and they carry the claim differently:

      * `levers_*.json` — a LIST of blocks, each with a "label" and a
        "verdict" string;
      * `optimize_*.json` — a DICT of blocks keyed by the lever name, with
        no label and no verdict string. The claim is the boolean "ok" and
        the human-readable "why".

    The optimize shape is the one this harness writes most often, and it is
    the shape the false ADMITs of 2026-11-04 were written in. A retraction
    that only understood the list shape would have left those standing —
    which is the one outcome the mechanism exists to prevent.

    Nothing here touches the real store or the GPU: the tests build their
    own records in a temp dir.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="retract_optimize_"))
        self.path = self.tmp / "optimize_x.json"
        self.path.write_text(json.dumps({
            "request": "compute",
            "objective": "gemm",
            "results": {
                "power_ceiling": {
                    "ok": True,
                    "why": "+12.99% improvement, p=0.0146, n=5/5 — clears both "
                           "significance and effect size",
                    "effect": 0.12994932240544288,
                    "p_value": 0.014598891648630596,
                },
            },
            "recorded_utc": "2026-11-04T13:23:20Z",
        }, indent=2), encoding="utf-8")

    def _doc(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_the_result_is_found_by_its_key(self):
        retract.retract_result(self.path, "power_ceiling", "starved control")
        block = self._doc()["results"]["power_ceiling"]
        self.assertIn("retracted", block)

    def test_the_original_claim_stays_readable(self):
        retract.retract_result(self.path, "power_ceiling", "starved control")
        block = self._doc()["results"]["power_ceiling"]
        self.assertIn("+12.99", block["retracted"]["original_verdict"])
        self.assertIn("+12.99", block["why"], "the original text stays visible")

    def test_the_live_claim_is_marked_so_a_grep_cannot_miss_it(self):
        retract.retract_result(self.path, "power_ceiling", "starved control")
        block = self._doc()["results"]["power_ceiling"]
        self.assertTrue(block["why"].startswith("RETRACTED"),
                        "a reader who greps for the claim must find it marked")

    def test_the_gate_boolean_is_not_flipped(self):
        """`ok: false` in these records means the gate REFUSED the run.

        Flipping it would fabricate a refusal that was never reached — the
        mirror image of the failure this mechanism exists to prevent. The
        original value is kept beside the retraction instead.
        """
        retract.retract_result(self.path, "power_ceiling", "starved control")
        block = self._doc()["results"]["power_ceiling"]
        self.assertTrue(block["ok"], "the gate did say OK; the record must say so")
        self.assertTrue(block["retracted"]["original_ok"])

    def test_the_reason_and_the_replacement_travel_with_it(self):
        retract.retract_result(self.path, "power_ceiling", "the control starved",
                               replacement="E:\\m.json",
                               replacement_verdict="ADMIT (+8.50%), paired")
        block = self._doc()["results"]["power_ceiling"]
        self.assertIn("starved", block["retracted"]["reason"])
        self.assertIn("+8.50%", block["retracted"]["replacement_verdict"])
        self.assertIn("m.json", block["retracted"]["replacement_measurement"])

    def test_other_results_are_untouched(self):
        d = self._doc()
        d["results"]["aov_variant"] = {"ok": True, "why": "+7.49% improvement"}
        self.path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        retract.retract_result(self.path, "power_ceiling", "starved control")
        other = self._doc()["results"]["aov_variant"]
        self.assertNotIn("retracted", other)
        self.assertEqual(other["why"], "+7.49% improvement")

    def test_retracting_an_unknown_label_refuses(self):
        with self.assertRaises(ValueError) as ctx:
            retract.retract_result(self.path, "nope", "because")
        self.assertIn("no result labelled", str(ctx.exception))

    def test_the_refusal_says_what_is_there(self):
        with self.assertRaises(ValueError) as ctx:
            retract.retract_result(self.path, "nope", "because")
        self.assertIn("power_ceiling", str(ctx.exception))

    def test_a_retraction_needs_a_reason(self):
        with self.assertRaises(ValueError):
            retract.retract_result(self.path, "power_ceiling", "")

    def test_retracting_twice_identically_is_a_no_op(self):
        retract.retract_result(self.path, "power_ceiling", "starved control")
        before = self.path.read_text(encoding="utf-8")
        retract.retract_result(self.path, "power_ceiling", "starved control")
        self.assertEqual(before, self.path.read_text(encoding="utf-8"))

    def test_retracting_twice_differently_refuses(self):
        retract.retract_result(self.path, "power_ceiling", "reason one")
        with self.assertRaises(ValueError):
            retract.retract_result(self.path, "power_ceiling", "reason two")

    def test_the_replacement_can_be_pointed_at_a_later_measurement(self):
        retract.retract_result(self.path, "power_ceiling", "starved control")
        retract.update_replacement(self.path, "power_ceiling",
                                   "E:\\later.json", "ADMIT (+8.50%), paired")
        ret = self._doc()["results"]["power_ceiling"]["retracted"]
        self.assertEqual(ret["replacement_measurement"], "E:\\later.json")
        self.assertIn("+8.50%", ret["replacement_verdict"])

    def test_a_corrected_reason_is_recorded_alongside_not_over(self):
        retract.retract_result(self.path, "power_ceiling", "the control starved")
        retract.update_replacement(self.path, "power_ceiling",
                                   "E:\\later.json", "ADMIT (+8.50%), paired",
                                   corrected_reason="Re-measured on a stable control.")
        ret = self._doc()["results"]["power_ceiling"]["retracted"]
        self.assertIn("starved", ret["reason"], "the original reason must survive")
        self.assertIn("stable control", ret["reason_corrected"]["reason"])

    def test_the_file_stays_valid_json(self):
        retract.retract_result(self.path, "power_ceiling", "starved control")
        json.loads(self.path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
