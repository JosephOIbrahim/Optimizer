"""
Tests for the stale-baseline refusal: the V12 admission path must refuse a
comparison it cannot honestly make.

The failure this exists for
---------------------------
On 2026-11-03 this repo produced ADMIT +12.34% (p=1.882e-05) for a
thread-count change that was worth nothing. The candidate was measured at 19:29
and gated against a baseline measured at 18:41 — 48 minutes earlier, under a
different set of resident processes. The two numbers described two different
machine states and the difference between them was charged to the candidate.
(The "+0.7%" that first appeared against it was the phantom's own cross-session
arithmetic, not a measurement of the lever — measured with both arms
alternately in one session the same change is DENY -0.60%.)

The arithmetic was not at fault. Welch's t-test on two INDEPENDENT series
cannot distinguish drift from effect; that is exactly what "independent" means.
What was missing was any way to tell the gate that the two series were not
independent — and there was none, because the candidate arrived at the gate as
a bare list of floats, carrying no timestamp, no conditions and no session
identity. The gate could not ask the question, so it never did.

So the fix has two halves, and these tests cover both:

  * the candidate must carry its PROVENANCE — when it was measured, in which
    session, under what conditions, and the per-pair differences against its
    control — and a comparison without it is REFUSED, not admitted;
  * the guidance an operator is handed must not be a command that gates a
    candidate against a baseline from another session.

The numbers in the first test are the REAL ones from that day, taken from the
records on disk (`E:\\OptimizerV12\\measurements\\render_20261103_184125.json`
and `..._192945.json`) and from the baseline files they became
(`D:\\OptimizerV12\\baselines\\f6d1f3438614.json` and `d87662056f00.json`).

Nothing here touches the GPU, writes outside a temp dir, or reads the real
measurement root.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "OptimizerV12"))

from v12.admission import Gate  # noqa: E402
from v12.baseline import (Baseline, Conditions, Measurement,  # noqa: E402
                          record)
from v12.change import (Change, ChangeKind, EvidenceClass, Observability,  # noqa: E402
                        Snapshot, change_key)
from v12.objective import Objective, render_objective  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _snap(value="0") -> Snapshot:
    return Snapshot(value=value, existed=True, trusted=True, source="test")


def _change(cid="c1", *, kind=ChangeKind.ENV_VAR, target="SOME_VAR",
            evidence=EvidenceClass.CITED, source="nvidia docs",
            snapshot=None, **kw) -> Change:
    return Change(id=cid, kind=kind, description="d", target=target,
                  evidence=evidence, evidence_source=source,
                  snapshot=snapshot, **kw)


#: The machine state both arms of a comparison were measured under. Readable
#: on every field, because an unread one is never a match.
def _conditions() -> Conditions:
    return Conditions(driver_version="617.42", sm_clock_max_mhz=3105,
                      mem_clock_max_mhz=10501, power_limit_w=450.0,
                      hags_enabled=True, competing_processes=[],
                      competing_method="gpu-commit>=1024MiB+significant-apps/heavy-count-v2")


#: The 2026-11-03 numbers, exactly as they were measured.
#:
#: The baseline is `render_20261103_184125.json` (18:41:25) — mean 54.426 s.
#: The candidate is `render_20261103_192945.json` (19:29:45) — mean 47.71 s.
#: The gap is 48 minutes and 20 seconds.
#:
#: Under the comparison that was in place on the day, these two series are
#: ADMITTED at +12.34%, p=1.882e-05 — which is the retracted ledger verdict
#: in `levers_20261103_192945.json`, digit for digit. That is the number this
#: whole change exists to stop producing.
GHOST_BASELINE = [53.78, 53.45, 55.43, 55.12, 54.35]
GHOST_CANDIDATE = [47.92, 47.75, 47.53, 47.79, 47.56]
GHOST_GAP_S = 48 * 60 + 20

#: The same two series in the form the incident was QUOTED in — the baseline
#: arm rounded to a mean of 54.43 s, the candidate reported as 47.71 s.
#:
#: Pinned separately because a replay is only evidence if it is the shape that
#: was actually reported, and because these two series are NOT
#: interchangeable: they give different figures (+12.34%, p=1.882e-05 for the
#: records; +12.35%, p=1.294e-10 for this rounded form, under the current
#: arithmetic). Both are refused by the gate, and both are pinned separately,
#: so neither can be quietly "corrected" into the other.
GHOST_BASELINE_QUOTED = [54.43, 54.5, 54.3, 54.5, 54.44]
GHOST_CANDIDATE_QUOTED = [47.71, 47.75, 47.7, 47.72, 47.68]

#: A fixed epoch so the tests do not depend on the clock. 2026-11-03 18:41:25Z.
T_BASE = 1793752885.0
T_CAND = T_BASE + GHOST_GAP_S


def _baseline(obj: Objective, samples, *, session_id="S1", measured_at=T_BASE,
              conditions=None) -> Baseline:
    """A Baseline as `record()` would have written it: a session id, the time
    the samples were measured, and the machine state at that moment."""
    return Baseline(
        objective_id=f"{obj.workload.kind.value}:{obj.workload.name}",
        workload=asdict(obj.workload),
        metric=obj.primary.name,
        direction=obj.primary.direction.value,
        samples=list(samples),
        conditions=asdict(conditions or _conditions()),
        session_id=session_id,
        measured_at=measured_at,
    )


def _measurement(obj: Objective, candidate, baseline_samples, *,
                 session_id="S1", measured_at=T_CAND, conditions=None,
                 paired_diffs=None) -> Measurement:
    """The candidate's provenance, as the paired harness would record it."""
    if paired_diffs is None:
        paired_diffs = [c - b for c, b in zip(candidate, baseline_samples)]
    return Measurement(
        measured_at=measured_at,
        session_id=session_id,
        conditions=asdict(conditions or _conditions()),
        paired_diffs=list(paired_diffs),
    )


def _render_obj() -> Objective:
    return render_objective()


def _measured_change(obj: Objective, **kw) -> Change:
    """A MEASURED change: the only class that may claim a speedup."""
    return _change(target="X", evidence=EvidenceClass.MEASURED,
                   snapshot=_snap("0"), **kw)


# ---------------------------------------------------------------------------
class TestTheGhostIsRefused(unittest.TestCase):
    """The real 2026-11-03 numbers, replayed through the real Gate.

    Before the fix this came back ADMIT at +12.34%, p=1.882e-05 — the verdict
    the ledger recorded, from the same arithmetic, on the same numbers.
    """

    def setUp(self):
        self.obj = _render_obj()
        self.change = _measured_change(self.obj)
        self.baseline = _baseline(self.obj, GHOST_BASELINE, session_id="S1",
                                  measured_at=T_BASE)
        # The candidate, 48 minutes later, in a DIFFERENT session — which is
        # what the ghost was: two series from two afternoons.
        self.candidate = list(GHOST_CANDIDATE)
        # The guardrails are SUPPLIED, unchanged, on both sides. This matters:
        # the render objective declares an ssim guardrail, and a missing
        # guardrail denies a comparison on its own. Supplying it means the
        # session check is the ONLY thing standing between this replay and an
        # admission — so a pass here is evidence about the session check and
        # not about anything else.
        self.guards = {"ssim": ([1.0, 0.996, 0.995], [1.0, 0.996, 0.995])}

    def _evaluate(self, measurement):
        return Gate(objective=self.obj).evaluate(
            self.change, baseline=self.baseline,
            candidate_samples=self.candidate,
            guardrail_samples=self.guards, measurement=measurement)

    def test_the_replay_is_refused_not_admitted(self):
        # A candidate with no provenance at all: the shape the CLI used to
        # hand the gate.
        v = self._evaluate(None)
        self.assertFalse(v.admitted,
                         f"the 2026-11-03 ghost was admitted: {v.reason}")

    def test_the_replay_is_refused_even_when_the_candidate_claims_a_session(self):
        # A candidate that claims the baseline's session but was measured 48
        # minutes away is still refused: the time gap is the evidence, and it
        # is checked.
        m = _measurement(self.obj, self.candidate, GHOST_BASELINE,
                         session_id="S1", measured_at=T_CAND)
        v = self._evaluate(m)
        self.assertFalse(v.admitted,
                         f"the 2026-11-03 ghost was admitted: {v.reason}")

    def test_the_quoted_replay_is_refused_too(self):
        """The same replay in the form the incident was QUOTED in — baseline
        [54.43, 54.5, 54.3, 54.5, 54.44] against candidate [47.71, ...].

        A replay is only evidence if it is the shape that was actually
        reported, so this pins the quoted form separately from the raw
        records. The two differ slightly in their arithmetic (the quoted
        series is +12.35%, p=1.294e-10, where the raw records give +12.34%,
        p=1.882e-05) but they are the same comparison, and both are refused.
        """
        b = _baseline(self.obj, GHOST_BASELINE_QUOTED, session_id="S1",
                      measured_at=T_BASE)
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=b,
            candidate_samples=list(GHOST_CANDIDATE_QUOTED),
            guardrail_samples=self.guards, measurement=None)
        self.assertFalse(v.admitted,
                         f"the quoted 2026-11-03 ghost was admitted: {v.reason}")

    def test_the_quoted_replay_would_have_been_admitted_by_the_old_arithmetic(self):
        """The quoted numbers, run through the comparison that was in place on
        the day: Welch on two independent series, no session check.

        This is what the fix has to beat, so it is pinned rather than
        asserted in prose. It is the same arithmetic the retracted ledger
        verdict was produced by, on the rounded series — which lands at +12.35%
        against the records' +12.34%, the difference being the rounding.

        As in the records test above, the p-value is pinned to what the code
        computes NOW. The pre-fix tail (`1 - (1 - 0.5*ib)`, subtracting two
        nearly-equal doubles) returned 1.4e-09 for this series; the
        difference between that and 1.294e-10 IS the cancellation error, so
        the old figure is the bug's value and pinning it would pin the defect.
        """
        ok, why, detail = self.obj.admits(
            GHOST_BASELINE_QUOTED, GHOST_CANDIDATE_QUOTED,
            {"ssim": ([1.0, 0.996, 0.995], [1.0, 0.996, 0.995])})
        self.assertTrue(ok,
                        "the quoted replay must still reproduce the phantom under "
                        "the old comparison, or the refusal above proves nothing")
        self.assertAlmostEqual(detail["effect"], 0.12348899584818314, places=9)
        self.assertAlmostEqual(detail["p_value"], 1.2938961816212717e-10, places=12)
        self.assertIn("+12.35% improvement", why)
        self.assertIn("p=1.294e-10", why)

    def test_the_replay_cannot_reach_the_independent_test(self):
        """The V12 path must not be able to fall through to Welch's t-test.

        `admits()` still supports an unpaired comparison — that is correct for
        the objective, which is a general tool. What must not happen is the
        GATE handing it two independent series and calling the result a
        verdict. So the refusal has to arrive BEFORE the statistics are
        computed, and the detail must never carry a p-value.
        """
        v = self._evaluate(None)
        self.assertFalse(v.admitted)
        self.assertIsNone(v.detail.get("p_value"),
                          "the ghost was gated on a p-value; a refusal must not "
                          "carry one")
        self.assertNotIn("paired", v.detail)

    def test_the_replay_would_have_been_admitted_by_the_old_arithmetic(self):
        """The test above is worth nothing if the numbers no longer produce the
        phantom. This pins that they still DO — under the old comparison.

        Welch on the two independent series is the arithmetic that admitted
        +12.34% on the day. If it no longer reproduces the phantom, then the
        refusal above is not evidence of anything.

        The p-value is pinned to what the code NOW computes, not to what it
        used to. The old tail was `1 - (1 - 0.5*ib)`, subtracting two
        nearly-equal doubles; below 2**-53 the inner subtraction rounded to
        exactly 1.0 and the tail came back as 0.0 for a t-statistic whose true
        p was 4.8e-23. Pinning the pre-fix figure would be pinning the DEFECT
        — the difference between the two numbers IS the cancellation error.
        So a test that pinned the old value would fail today, and the person
        who then "repaired" it would be restoring a bug they believe is a
        feature. This pins the current arithmetic, which fails only if that
        arithmetic drifts — which is the thing worth knowing.
        """
        ok, why, detail = self.obj.admits(
            GHOST_BASELINE, GHOST_CANDIDATE,
            {"ssim": ([1.0, 0.996, 0.995], [1.0, 0.996, 0.995])})
        self.assertTrue(ok,
                        "the replay must still reproduce the phantom under the old "
                        "comparison, or the refusal above proves nothing")
        # The effect: an apparent +12.34% win that was never a win.
        self.assertAlmostEqual(detail["effect"], 0.12339690589056702, places=9)
        # The p-value, as the code computes it today. See the docstring for why
        # this is the current figure and not the pre-fix one.
        self.assertAlmostEqual(detail["p_value"], 1.8822382680203898e-05, places=10)
        # And the number a reader would compare against the ledger.
        self.assertIn("p=1.882e-05", why)
        self.assertIn("+12.34% improvement", why)

    def test_the_refusal_names_the_reason_in_plain_words(self):
        v = self._evaluate(None)
        self.assertNotEqual(v.reason.strip(), "")
        self.assertNotEqual(v.reason.strip().lower(), "false")
        # An operator reading this must be able to act on it.
        self.assertRegex(v.reason.lower(), r"session|measurement|timestamp|record")

    def test_the_refusal_is_about_the_session_not_about_something_else(self):
        """The guardrails are supplied, so the only reason left for a refusal
        is the session check. This pins that — otherwise a refusal for an
        unrelated reason would satisfy the test above and prove nothing."""
        v = self._evaluate(None)
        self.assertNotIn("guardrail", v.reason.lower())
        self.assertNotIn("ssim", v.reason.lower())


# ---------------------------------------------------------------------------
class TestTheHonestCaseStillAdmits(unittest.TestCase):
    """The fix must not break the comparison that IS honest: both arms measured
    alternately in one session, with the paired differences."""

    def setUp(self):
        self.obj = _render_obj()
        # A real win: ~20% faster, which clears the objective's 5% threshold.
        self.baseline_samples = [48.20, 48.10, 48.35, 48.05, 48.15]
        self.candidate_samples = [38.20, 38.10, 38.35, 38.05, 38.15]
        self.baseline = _baseline(self.obj, self.baseline_samples, session_id="S7",
                                  measured_at=T_BASE)
        self.change = _measured_change(self.obj)
        # The render objective declares an ssim guardrail, so the comparison
        # carries both sides of it, unchanged.
        self.guards = {"ssim": ([1.0, 0.996, 0.995], [1.0, 0.994, 0.995])}

    def _evaluate(self, measurement):
        return Gate(objective=self.obj).evaluate(
            self.change, baseline=self.baseline,
            candidate_samples=self.candidate_samples,
            guardrail_samples=self.guards, measurement=measurement)

    def test_a_same_session_paired_comparison_admits(self):
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0)
        v = self._evaluate(m)
        self.assertTrue(v.admitted, v.reason)
        self.assertIs(v.evidence, EvidenceClass.MEASURED)
        self.assertEqual(v.baseline_id, self.baseline.id)

    def test_the_admitted_verdict_used_the_paired_test(self):
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0)
        v = self._evaluate(m)
        self.assertEqual(v.detail.get("paired"), 1.0,
                         "an admitted comparison must have been tested as paired")

    def test_the_same_session_comparison_is_not_refused_for_the_wrong_reason(self):
        """A refusal on the honest case would be as bad as the ghost. This
        checks the reason is not one of the provenance complaints."""
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0)
        v = self._evaluate(m)
        self.assertNotIn("session", v.reason.lower())
        self.assertNotIn("no paired differences", v.reason.lower())

    def test_a_candidate_from_a_different_session_is_refused(self):
        """Same numbers, same paired diffs — but a different session. The
        session id is what makes the difference, and it is checked."""
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S8", measured_at=T_BASE + 120.0)
        v = self._evaluate(m)
        self.assertFalse(v.admitted)
        self.assertIn("session", v.reason.lower())

    def test_a_candidate_measured_far_from_the_baseline_is_refused(self):
        """Same session, same numbers — but measured 48 minutes away, which is
        the ghost's shape with a matching label."""
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + GHOST_GAP_S)
        v = self._evaluate(m)
        self.assertFalse(v.admitted)
        self.assertIn("minute", v.reason.lower())

    def test_paired_differences_from_another_pairing_are_refused(self):
        """The differences must be the differences BETWEEN THESE TWO SERIES.
        A paired set carried over from another pairing would pass every other
        check and still describe a comparison nobody made."""
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0,
                         paired_diffs=[-1.0] * 5)
        v = self._evaluate(m)
        self.assertFalse(v.admitted)
        self.assertIn("pair", v.reason.lower())

    def test_paired_differences_that_survive_four_decimal_rounding_admit(self):
        """The series travel as decimal text — `common.series()` prints four
        decimal places — so the differences arrive with a little rounding in
        them. A reconciliation check that rejected those would refuse every
        honest comparison, which is as bad as the ghost."""
        rounded_base = [round(x, 4) for x in self.baseline_samples]
        rounded_cand = [round(x, 4) for x in self.candidate_samples]
        rounded_diffs = [round(c - b, 4) for c, b in zip(rounded_cand, rounded_base)]
        b = _baseline(self.obj, rounded_base, session_id="S7", measured_at=T_BASE)
        m = _measurement(self.obj, rounded_cand, rounded_base,
                         session_id="S7", measured_at=T_BASE + 120.0,
                         paired_diffs=rounded_diffs)
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=b, candidate_samples=rounded_cand,
            guardrail_samples=self.guards, measurement=m)
        self.assertTrue(v.admitted, v.reason)

    def test_a_comparison_without_paired_differences_is_refused(self):
        """The paired test is the correct test. Without the differences the
        comparison would fall through to Welch on two independent series, which
        is the exact arithmetic that produced the ghost."""
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0,
                         paired_diffs=[])
        v = self._evaluate(m)
        self.assertFalse(v.admitted)
        self.assertIn("paired", v.reason.lower())


# ---------------------------------------------------------------------------
class TestAComparisonThatCannotBeJudgedIsRefused(unittest.TestCase):
    """Fail closed. A comparison the gate cannot judge is refused, not
    admitted — because admitting what it cannot judge is how the ghost got in."""

    def setUp(self):
        self.obj = _render_obj()
        self.baseline_samples = [48.20, 48.10, 48.35, 48.05, 48.15]
        self.candidate_samples = [38.20, 38.10, 38.35, 38.05, 38.15]
        self.baseline = _baseline(self.obj, self.baseline_samples, session_id="S7",
                                  measured_at=T_BASE)
        self.change = _measured_change(self.obj)

    def _evaluate(self, measurement):
        return Gate(objective=self.obj).evaluate(
            self.change, baseline=self.baseline,
            candidate_samples=self.candidate_samples, measurement=measurement)

    def test_no_measurement_record_at_all_is_refused(self):
        """THE refusal this whole change exists for.

        A candidate with no measurement record has no timestamp, no session
        identity and no conditions, so there is nothing to place in time and
        nothing to show it describes the same machine state as the baseline.
        The gate must refuse it — and say which evidence is missing, so the
        operator knows what to re-measure.
        """
        v = self._evaluate(None)
        self.assertFalse(v.admitted)
        self.assertRegex(v.reason.lower(), r"measurement record|no timestamp|session")
        # The reason names the missing evidence rather than just failing.
        for missing in ("when it was measured", "which session"):
            self.assertIn(missing, v.reason.lower())
        # And it tells the operator what to do about it.
        self.assertIn("re-measure", v.reason.lower())
        # A refusal is not a verdict: no statistics were computed.
        self.assertIsNone(v.detail.get("p_value"))
        self.assertIsNone(v.detail.get("effect"))

    def test_a_candidate_with_no_measurement_record_never_reaches_the_statistics(self):
        """The refusal has to arrive BEFORE the arithmetic runs.

        This is the difference between refusing a comparison and merely
        disliking it: a verdict that carries a p-value has been computed, and
        a p-value is what ends up in the ledger.
        """
        v = self._evaluate(None)
        self.assertFalse(v.admitted)
        self.assertEqual(v.detail, {},
                         "a refusal must carry no computed statistics")

    def test_a_candidate_with_no_timestamp_is_refused(self):
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0)
        m.measured_at = 0.0
        v = self._evaluate(m)
        self.assertFalse(v.admitted)
        self.assertIn("timestamp", v.reason.lower())

    def test_a_candidate_with_no_session_is_refused(self):
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0)
        m.session_id = ""
        v = self._evaluate(m)
        self.assertFalse(v.admitted)
        self.assertIn("session", v.reason.lower())

    def test_a_baseline_with_no_session_is_refused(self):
        b = _baseline(self.obj, self.baseline_samples, session_id="",
                      measured_at=T_BASE)
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="", measured_at=T_BASE + 120.0)
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=b, candidate_samples=self.candidate_samples,
            measurement=m)
        self.assertFalse(v.admitted)
        self.assertIn("session", v.reason.lower())

    def test_a_candidate_whose_conditions_differ_is_refused(self):
        other = _conditions()
        other.power_limit_w = 600.0
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0,
                         conditions=other)
        v = self._evaluate(m)
        self.assertFalse(v.admitted)
        self.assertIn("same state", v.reason.lower())

    def test_the_refusal_is_always_a_reason_not_a_bare_false(self):
        """Every refusal on this path has to say WHY, in words an operator can
        act on."""
        for m in (None,
                  _measurement(self.obj, self.candidate_samples,
                               self.baseline_samples, session_id="",
                               measured_at=T_BASE + 120.0),
                  _measurement(self.obj, self.candidate_samples,
                               self.baseline_samples, session_id="S9",
                               measured_at=T_BASE + 120.0)):
            v = self._evaluate(m)
            self.assertFalse(v.admitted)
            self.assertGreater(len(v.reason.strip()), 40,
                               f"refusal reason is too terse to act on: {v.reason!r}")
            self.assertNotEqual(v.reason.strip().lower(), "false")


# ---------------------------------------------------------------------------
class TestTheMeasurementTravelsWithTheChange(unittest.TestCase):
    """The provenance rides on the change's `extra` dict, so it survives being
    handed around — the way `paired_diffs` was always meant to."""

    def setUp(self):
        self.obj = _render_obj()
        self.baseline_samples = [48.20, 48.10, 48.35, 48.05, 48.15]
        self.candidate_samples = [38.20, 38.10, 38.35, 38.05, 38.15]

    def test_a_measurement_on_the_change_is_honoured(self):
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0)
        change = _measured_change(self.obj, extra={"measurement": m.to_extra()})
        b = _baseline(self.obj, self.baseline_samples, session_id="S7",
                      measured_at=T_BASE)
        v = Gate(objective=self.obj).evaluate(
            change, baseline=b, candidate_samples=self.candidate_samples,
            guardrail_samples={"ssim": ([1.0, 0.996, 0.995],
                                        [1.0, 0.994, 0.995])})
        self.assertTrue(v.admitted, v.reason)

    def test_a_measurement_round_trips_through_its_json_form(self):
        m = _measurement(self.obj, self.candidate_samples, self.baseline_samples,
                         session_id="S7", measured_at=T_BASE + 120.0)
        back = Measurement.from_extra({"measurement": m.to_extra()})
        self.assertIsNotNone(back)
        self.assertEqual(back.session_id, "S7")
        self.assertEqual(back.measured_at, m.measured_at)
        self.assertEqual(back.paired_diffs, m.paired_diffs)
        self.assertEqual(back.conditions, m.conditions)

    def test_reading_a_measurement_that_is_not_there_gives_none(self):
        """Absent evidence is not evidence of sameness. It reads as None, and
        None refuses."""
        self.assertIsNone(Measurement.from_extra(None))
        self.assertIsNone(Measurement.from_extra({}))
        self.assertIsNone(Measurement.from_extra({"measurement": "not a dict"}))


# ---------------------------------------------------------------------------
class TestTheOperatorIsNotHandedTheStalePath(unittest.TestCase):
    """The guidance is part of the defect.

    `paste_lines` used to print `v12 run --candidate-for "KEY=..."` for every
    measurement — the exact command that compares a candidate against whatever
    baseline is on disk, and so the exact command that produced the ghost.
    """

    def _lines(self):
        from harness import common
        return common.paste_lines("render", [48.20, 48.10, 48.35, 48.05, 48.15],
                                  "husk_frame", {"ssim": [1.0, 0.994, 0.995]})

    def test_no_candidate_command_is_printed(self):
        text = "\n".join(self._lines())
        self.assertNotIn("--candidate-for", text,
                         "paste_lines still prints a command that gates a candidate "
                         "against a baseline from another session")

    def test_no_command_that_gates_against_a_stored_baseline_is_printed(self):
        text = "\n".join(self._lines())
        self.assertNotIn("v12 run", text,
                         "paste_lines still prints a V12 run command, which gates "
                         "the candidate against a baseline read from disk")

    def test_the_printed_guidance_names_the_honest_procedure(self):
        text = "\n".join(self._lines())
        self.assertIn("session", text.lower())
        self.assertIn("alternately", text.lower())

    def test_the_baseline_line_is_still_printed(self):
        """Recording where the machine stands is still a legitimate thing to do
        — it is only the CROSS-SESSION comparison that was the problem."""
        text = "\n".join(self._lines())
        self.assertIn("baseline --objective render", text)
        self.assertIn("--record", text)

    def test_no_printed_line_gates_a_candidate(self):
        """Whatever else it says, no printed line may contain a candidate
        submission."""
        for line in self._lines():
            self.assertNotIn("--candidate", line,
                             f"this line gates a candidate: {line!r}")


# ---------------------------------------------------------------------------
class TestRecordedBaselinesCarryASession(unittest.TestCase):
    """A baseline that does not know which session it belongs to cannot be
    compared with anything, so `record()` stamps it."""

    def test_record_stamps_a_session_and_a_measurement_time(self):
        import tempfile
        import v12.paths as paths

        with tempfile.TemporaryDirectory() as tmp:
            original = paths.BASELINE_DIR
            paths.BASELINE_DIR = Path(tmp)
            try:
                obj = _render_obj()
                b = record(obj, [48.20, 48.10, 48.35, 48.05, 48.15],
                           harness="test", conditions=_conditions())
                self.assertTrue(b.session_id, "a recorded baseline must name its session")
                self.assertGreater(b.measured_at, 0.0,
                                   "a recorded baseline must say when it was measured")
                # And it survives the round trip through disk.
                again = Baseline.load(b.path())
                self.assertEqual(again.session_id, b.session_id)
                self.assertEqual(again.measured_at, b.measured_at)
            finally:
                paths.BASELINE_DIR = original


class TestTheDocstringWarningIsTrue(unittest.TestCase):
    """`admits()` cannot check that the arms were measured together.

    Its docstring now says so, and says where the check that CAN is. This
    test keeps that warning from becoming a lie: if someone ever made
    `admits()` refuse an unpaired comparison, the warning would be wrong —
    but they would have to fix the ghost at the wrong layer, so it is far
    more likely the warning is right and should stay.
    """

    def test_the_statistics_function_alone_still_admits_the_ghost(self):
        """The raw arithmetic, on the ghost's own series."""
        obj = _render_obj()
        ok, why, detail = obj.admits(
            [54.43, 54.5, 54.3, 54.5, 54.44], [47.71] * 5,
            {"ssim": ([1.0] * 5, [1.0] * 5)})
        self.assertTrue(ok,
                        "if this ever becomes False the docstring warning is "
                        "stale and must be rewritten")
        self.assertGreater(detail["effect"], 0.12,
                           "the phantom's effect should still be visible here")

    def test_the_gate_refuses_before_the_statistics_run(self):
        """And the gate refuses it WITHOUT computing anything — the empty
        detail is the proof that the refusal precedes the arithmetic."""
        obj = _render_obj()
        gate = Gate(objective=obj)
        v = gate.evaluate(_measured_change(obj),
                          baseline=_baseline(obj, [54.43, 54.5, 54.3, 54.5, 54.44]),
                          candidate_samples=[47.71] * 5)
        self.assertFalse(v.admitted)
        self.assertEqual(v.detail, {},
                         "an empty detail means no statistics were computed — "
                         "the refusal came first")


if __name__ == "__main__":
    unittest.main()
