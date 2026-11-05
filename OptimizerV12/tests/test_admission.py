"""
Tests for admission — the verdict, the gate, and the actuator boundary.

The defect these tests exist for is not a bug in either predecessor. It is that
`v9/apply.py`'s `apply_all()` was `recs[:top_n]`, while a real Pareto gate sat
unused in `v9/quality.py`. The gate worked. Nothing called it.

A policy check at the top of `apply_all()` would have fixed that, and would have
rotted at the next fast path. So the check is a *parameter*:
`apply(self, change, *, verdict)`. The first test here asserts that property
structurally, via `inspect.signature`, because it is a property of the signature
and not of any particular call — a behavioural test would pass on a signature
that had quietly gained `verdict=None`.

The deliberate asymmetry, asserted just as structurally: `rollback()` takes no
verdict. Undoing a change that turned out badly must never be gated on having
proved it should be undone.
"""
from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12.admission import (ApplyResult, Gate, GatedActuator, UnadmittedChange,
                           Verdict, evidence_breakdown)  # noqa: E402
from v12.baseline import Baseline, Conditions, Measurement  # noqa: E402
from v12.change import (Change, ChangeKind, EvidenceClass, Observability,
                        Rollback, Snapshot, Tier, change_key)  # noqa: E402
from v12.objective import KNOWN_METRICS, Objective, diffusion_objective  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _snap(value=0, trusted=True) -> Snapshot:
    return Snapshot(value=value, existed=True, trusted=trusted, source="test")


def _change(cid="c1", *, kind=ChangeKind.ENV_VAR, target="SOME_VAR",
            evidence=EvidenceClass.CITED, source="nvidia docs",
            snapshot=None, **kw) -> Change:
    return Change(id=cid, kind=kind, description="d", target=target,
                  evidence=evidence, evidence_source=source,
                  snapshot=snapshot, **kw)


#: A tight, usable baseline series (CV well under any usability threshold).
_STEADY = [100.0, 100.1, 99.9, 100.0, 100.05]


#: The session both sides of a test comparison belong to. A candidate that
#: carries no session id cannot be shown to have been measured alongside the
#: baseline, so the gate refuses it — and the tests here are about the
#: STATISTICS, not about provenance, so both sides are given the same one.
_TEST_SESSION = "test-session"


def _readable_conditions() -> Conditions:
    """Conditions that were actually READ, so that "same machine state" is a
    fact the comparison can rely on rather than a pair of blanks.

    Conditions.differences() treats an unread field on either side as a
    difference: "we could not tell" is not "the same". A default
    Conditions() is unread on every field, so a fixture built from it would
    be refused for the wrong reason before it reached the statistics.
    """
    return Conditions(
        driver_version="999.99",
        sm_clock_max_mhz=3105,
        mem_clock_max_mhz=14001,
        power_limit_w=450.0,
        hags_enabled=True,
        workload_fingerprint="test-fingerprint",
        workload_kind="render",
        competing_processes=[],
        competing_method="test",
    )


def _baseline(obj: Objective, samples, session_id: str = _TEST_SESSION) -> Baseline:
    """A Baseline built in memory. `valid_for(obj)` with now=None skips the
    machine-conditions comparison, which is what a test of the STATISTICS wants."""
    return Baseline(
        objective_id=f"{obj.workload.kind.value}:{obj.workload.name}",
        workload=asdict(obj.workload),
        metric=obj.primary.name,
        direction=obj.primary.direction.value,
        samples=list(samples),
        conditions=asdict(_readable_conditions()),
        session_id=session_id,
        measured_at=1000.0,
    )


def _measurement(candidate, baseline=None, session_id: str = _TEST_SESSION) -> Measurement:
    """A candidate carrying the provenance the gate requires: when it was
    measured, in which session, under which conditions, and the per-pair
    differences that make the comparison a PAIRED one.

    `paired_diffs` is candidate[i] - baseline[i] for the two series actually
    being compared. The gate refuses a difference series that does not
    belong to the pairing, because it would be testing a different
    comparison than the one on offer.
    """
    diffs = None
    if baseline is not None:
        diffs = [float(c) - float(b) for c, b in zip(candidate, baseline)]
    return Measurement(measured_at=1000.0, session_id=session_id,
                       conditions=asdict(_readable_conditions()),
                       paired_diffs=diffs,
                       harness="test")


def _optin(**kw) -> Gate:
    """A Gate whose operator explicitly opted in to CITED changes. Tests that are
    about something OTHER than the cited-admission policy (verdict binding,
    rollback, verification) use this, so the default-deny is never what they hit."""
    return Gate(allow_unverified=True, **kw)


class Recorder:
    """A stand-in for whatever really mutates state. Records every call, so a
    test can assert not only that apply() raised but that nothing was reached."""

    def __init__(self):
        self.calls = []

    def apply(self, change):
        self.calls.append(("apply", change.id))
        return ApplyResult(change_id=change.id, applied=True, verified=True)

    def apply_undo(self, change):
        self.calls.append(("undo", change.id))
        return ApplyResult(change_id=change.id, applied=True, verified=True)

    def reset_default(self, change):
        self.calls.append(("reset", change.id))
        return ApplyResult(change_id=change.id, applied=True, verified=False)

    def verify(self, change):
        self.calls.append(("verify", change.id))
        return ApplyResult(change_id=change.id, applied=True, verified=True)


# ---------------------------------------------------------------------------
class TestTheGateIsUnbypassable(unittest.TestCase):
    """Structural, not behavioural. These are the tests that cannot rot."""

    def test_apply_cannot_be_called_without_a_verdict(self):
        sig = inspect.signature(GatedActuator.apply)
        self.assertIn("verdict", sig.parameters)
        v = sig.parameters["verdict"]
        self.assertIs(v.kind, inspect.Parameter.KEYWORD_ONLY,
                      "verdict must be keyword-only — positional would let a caller "
                      "pass something else in slot 2 and never notice")
        self.assertIs(v.default, inspect.Parameter.empty,
                      "verdict must have NO default. A default of None reopens the "
                      "exact hole: the actuator becomes callable without a decision.")

    def test_apply_has_no_kwargs_to_smuggle_a_verdict_through(self):
        sig = inspect.signature(GatedActuator.apply)
        self.assertFalse(any(p.kind is inspect.Parameter.VAR_KEYWORD
                             for p in sig.parameters.values()),
                         "**kwargs would re-admit the un-verdicted call path")

    def test_rollback_takes_no_verdict(self):
        """The asymmetry, asserted. Undo is a safety operation, not an optimization."""
        for name in ("rollback", "verify"):
            with self.subTest(method=name):
                sig = inspect.signature(getattr(GatedActuator, name))
                named = [p for p in sig.parameters if p != "self"]
                self.assertEqual(named, ["change"],
                                 f"{name}() must take only the change — gating the "
                                 f"safety path on evidence means a change that failed "
                                 f"to produce evidence cannot be undone")
                self.assertNotIn("verdict", named)

    def test_a_none_verdict_is_refused_not_tolerated(self):
        rec = Recorder()
        act = GatedActuator(rec, Tier.SYSTEM)
        with self.assertRaises(UnadmittedChange):
            act.apply(_change(), verdict=None)
        self.assertEqual(rec.calls, [])          # nothing was reached

    def test_a_non_verdict_is_refused(self):
        rec = Recorder()
        act = GatedActuator(rec, Tier.SYSTEM)
        with self.assertRaises(UnadmittedChange):
            act.apply(_change(), verdict=True)
        self.assertEqual(rec.calls, [])


class TestVerdictBinding(unittest.TestCase):
    """A verdict admits ONE change. Reusing an admitted verdict on a different
    change would let a single measurement authorize an unrelated mutation."""

    def test_a_denied_verdict_cannot_reach_the_actuator(self):
        rec = Recorder()
        act = GatedActuator(rec, Tier.SYSTEM)
        c = _change()
        denied = Gate().evaluate(_change(kind=ChangeKind.NVML_CLOCK, target="gpu"))
        self.assertFalse(denied.admitted)                 # sanity
        with self.assertRaises(UnadmittedChange):
            act.apply(c, verdict=denied)
        self.assertEqual(rec.calls, [], "a denial reached the actuator")

    def test_a_verdict_does_not_transfer_to_another_change(self):
        rec = Recorder()
        act = GatedActuator(rec, Tier.SYSTEM)
        a = _change("a", target="VAR_A", snapshot=_snap("0"))
        b = _change("b", target="VAR_B", snapshot=_snap("0"))
        self.assertNotEqual(change_key(a), change_key(b))  # sanity
        verdict = _optin().evaluate(a)
        self.assertTrue(verdict.admitted)
        with self.assertRaises(UnadmittedChange):
            act.apply(b, verdict=verdict)
        self.assertEqual(rec.calls, [])

    def test_two_changes_at_the_same_target_share_a_key(self):
        """The snapshot-once property inherited from MOS2026 must survive: the
        same target addressed twice is one address in config space."""
        a = _change("a", target="VAR_A")
        b = _change("b", target="VAR_A")
        self.assertEqual(change_key(a), change_key(b))

    def test_a_verdict_for_the_wrong_tier_is_refused(self):
        rec = Recorder()
        act = GatedActuator(rec, Tier.SYSTEM)
        c = _change(kind=ChangeKind.KERNEL_ARTIFACT, target="gemm.dll")
        self.assertIs(c.tier, Tier.KERNEL)                # sanity
        verdict = Verdict(admitted=True, change_key=change_key(c), change_id=c.id,
                          reason="x", evidence=EvidenceClass.CITED)
        with self.assertRaises(UnadmittedChange):
            act.apply(c, verdict=verdict)
        self.assertEqual(rec.calls, [])


class TestGatePolicy(unittest.TestCase):
    """Each branch of the Gate, and the reason it gives."""

    def test_guided_is_never_admitted_and_returns_instructions(self):
        c = _change(kind=ChangeKind.GUIDED, target="nvlink-topology",
                    extra={"instructions": ["enable P2P in BIOS", "reboot"]})
        v = Gate().evaluate(c)
        self.assertFalse(v.admitted)
        self.assertIs(v.evidence, EvidenceClass.GUIDED)
        self.assertEqual(v.guided_instructions, ["enable P2P in BIOS", "reboot"],
                         "a guided change that loses its instructions is just a refusal")

    def test_destroying_operator_state_needs_an_opt_in(self):
        c = _change(kind=ChangeKind.NVML_CLOCK, target="gpu-clocks")
        v = Gate().evaluate(c)
        self.assertFalse(v.admitted)
        self.assertIn("VENDOR DEFAULT", v.reason)
        self.assertIn("allow_destructive_reset", v.reason)

    def test_the_same_change_is_admitted_once_the_operator_opts_in(self):
        c = _change(kind=ChangeKind.NVML_CLOCK, target="gpu-clocks")
        v = _optin(allow_destructive_reset=True).evaluate(c)
        self.assertTrue(v.admitted)
        self.assertTrue(v.destructive_reset,
                        "the verdict must carry the destructive flag forward so the "
                        "ledger can show it later")

    def test_a_restorable_target_with_no_prior_is_denied(self):
        """This is a pipeline-order bug, not a hardware limit: OBSERVE must
        snapshot before ADMIT evaluates. Without this check the change would be
        admitted and its rollback would delete the operator's value forever."""
        c = _change(target="UNSNAPSHOTTED")               # READABLE, no snapshot
        v = Gate().evaluate(c)
        self.assertFalse(v.admitted)
        self.assertIn("OBSERVE", v.reason)

    def test_an_untrusted_snapshot_is_not_a_prior(self):
        c = _change(target="X", snapshot=_snap("0", trusted=False))
        self.assertFalse(Gate().evaluate(c).admitted)

    def test_cited_without_a_source_is_denied(self):
        v = Gate().evaluate(_change(target="X", source="", snapshot=_snap("0")))
        self.assertFalse(v.admitted)
        self.assertIn("no source", v.reason)

    def test_a_high_risk_change_may_not_rest_on_a_citation(self):
        v = _optin().evaluate(_change(target="X", snapshot=_snap("0"), risk="high"))
        self.assertFalse(v.admitted)
        self.assertIn("high-risk", v.reason)

    def test_derived_below_the_objective_threshold_is_denied(self):
        obj = diffusion_objective()
        c = _change(target="X", evidence=EvidenceClass.DERIVED,
                    snapshot=_snap("0"))
        v = Gate(objective=obj).evaluate(c, baseline=_baseline(obj, _STEADY),
                                         predicted_effect=0.01)
        self.assertFalse(v.admitted)
        self.assertIn("below", v.reason)

    def test_derived_above_the_threshold_is_admitted_but_recorded_as_derived(self):
        obj = diffusion_objective()
        c = _change(target="X", evidence=EvidenceClass.DERIVED, snapshot=_snap("0"))
        b = _baseline(obj, _STEADY)
        v = Gate(objective=obj).evaluate(c, baseline=b, predicted_effect=0.30)
        self.assertTrue(v.admitted)
        self.assertIs(v.evidence, EvidenceClass.DERIVED)
        self.assertEqual(v.baseline_id, b.id,
                         "a derived admission must name the measurement it rests on")
        self.assertIn("NOT measured", v.reason,
                      "a derived admission must never read as a measurement")

    def test_a_model_prediction_on_a_lying_target_is_denied(self):
        c = _change(kind=ChangeKind.NVML_OFFSET, target="offsets",
                    evidence=EvidenceClass.DERIVED)
        v = Gate(allow_destructive_reset=True).evaluate(c, predicted_effect=0.30)
        self.assertFalse(v.admitted)
        self.assertIn("lies", v.reason)


class TestCitedNeverAdmitsByDefault(unittest.TestCase):
    """DESIGN_V12 §3-4: 'CITED never admits' / 'CITED — recorded, never applied'.
    The Gate used to default allow_unverified=True, so every READABLE, sourced,
    low-risk citation was admitted by a Gate() nobody configured — and Plane
    builds its Gate without passing the flag at all."""

    def test_a_default_gate_denies_a_sourced_cited_change(self):
        c = _change(target="X", source="nvidia docs", snapshot=_snap("0"))
        v = Gate().evaluate(c)
        self.assertFalse(v.admitted, "a citation was admitted by a default Gate")
        self.assertIs(v.evidence, EvidenceClass.CITED)
        self.assertIn("CITED evidence is never admitted", v.reason)
        self.assertIn("allow_unverified", v.reason)
        # Was: "construct the Gate with allow_unverified=True to opt in" — an action
        # no operator entry point offers (Plane never passes it; the CLI has no flag).
        self.assertIn("No operator-facing opt-in exists", v.reason)
        self.assertNotIn("to opt in explicitly", v.reason)

    def test_a_default_gate_with_an_objective_still_denies_cited(self):
        v = Gate(objective=diffusion_objective()).evaluate(
            _change(target="X", snapshot=_snap("0")))
        self.assertFalse(v.admitted)

    def test_the_default_is_off_in_the_signature(self):
        """Asserted structurally, like the apply() signature: a behavioural test
        elsewhere would not notice the default flipping back."""
        p = inspect.signature(Gate.__init__).parameters["allow_unverified"]
        self.assertIs(p.default, False)

    def test_an_explicit_opt_in_still_admits_and_records_as_cited(self):
        v = _optin().evaluate(_change(target="X", snapshot=_snap("0")))
        self.assertTrue(v.admitted)
        self.assertIs(v.evidence, EvidenceClass.CITED)
        self.assertIn("UNVERIFIED", v.reason)

    def test_a_default_gate_counts_no_cited_admissions(self):
        gate = Gate()
        for i in range(3):
            gate.evaluate(_change(f"c{i}", target=f"T{i}", snapshot=_snap("0")))
        self.assertEqual(evidence_breakdown(gate.history)["cited"], 0)


class TestDerivedNeedsAMeasuredAnchor(unittest.TestCase):
    """DESIGN_V12 §3: DERIVED is 'admissible only with a measurement behind it'.
    The Gate used to admit DERIVED on predicted_effect alone — no objective and
    no baseline required, with a hard-coded 5% threshold standing in for one."""

    def setUp(self):
        self.obj = diffusion_objective()
        self.change = _change(target="X", evidence=EvidenceClass.DERIVED,
                              snapshot=_snap("0"))

    def test_a_prediction_with_no_objective_is_denied(self):
        v = Gate().evaluate(self.change, predicted_effect=0.50)
        self.assertFalse(v.admitted, "a bare prediction was admitted with no objective")
        self.assertIn("no Objective", v.reason)

    def test_a_prediction_with_no_baseline_is_denied(self):
        v = Gate(objective=self.obj).evaluate(self.change, predicted_effect=0.50)
        self.assertFalse(v.admitted, "a prediction was admitted with nothing measured")
        self.assertIn("no baseline", v.reason)

    def test_a_baseline_for_the_wrong_metric_is_denied(self):
        b = _baseline(self.obj, _STEADY)
        b.metric = "fps"
        v = Gate(objective=self.obj).evaluate(self.change, baseline=b,
                                              predicted_effect=0.50)
        self.assertFalse(v.admitted)
        self.assertIn("not valid", v.reason)

    def test_a_baseline_from_a_changed_machine_is_denied(self):
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=_baseline(self.obj, _STEADY), predicted_effect=0.50,
            now_conditions=Conditions(competing_processes=["houdini.exe"]))
        self.assertFalse(v.admitted)
        self.assertIn("machine changed", v.reason)

    def test_a_valid_baseline_without_a_prediction_is_denied(self):
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=_baseline(self.obj, _STEADY))
        self.assertFalse(v.admitted)
        self.assertIn("no predicted effect", v.reason)
        # Was: "the model gave no magnitude" — the plane never asks the model.
        self.assertIn("does not yet compute", v.reason)

    def test_a_nan_prediction_is_denied(self):
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=_baseline(self.obj, _STEADY),
            predicted_effect=float("nan"))
        self.assertFalse(v.admitted)

    def test_baseline_and_prediction_together_admit_as_derived(self):
        b = _baseline(self.obj, _STEADY)
        v = Gate(objective=self.obj).evaluate(self.change, baseline=b,
                                              predicted_effect=0.30)
        self.assertTrue(v.admitted)
        self.assertIs(v.evidence, EvidenceClass.DERIVED)
        self.assertEqual(v.baseline_id, b.id)


class TestMeasuredPath(unittest.TestCase):
    """MEASURED is the only class allowed to support the word 'faster', so it
    gets the strictest tests."""

    def setUp(self):
        self.obj = diffusion_objective()
        self.good_base = [100.0, 100.1, 99.9, 100.0, 100.05]
        self.good_cand = [80.0, 80.1, 79.9, 80.0, 80.05]
        self.change = _change(target="X", evidence=EvidenceClass.MEASURED,
                              snapshot=_snap("0"))
        # diffusion_objective declares lpips and clip; unchanged values for both.
        self.flat_guards = {"lpips": ([0.20] * 5, [0.20] * 5),
                            "clip": ([0.30] * 5, [0.30] * 5)}

    def test_measured_without_an_objective_is_denied(self):
        v = Gate(objective=None).evaluate(
            self.change, baseline=_baseline(self.obj, self.good_base),
            candidate_samples=self.good_cand,
            measurement=_measurement(self.good_cand, self.good_base))
        self.assertFalse(v.admitted)
        self.assertIn("no Objective", v.reason)

    def test_measured_without_a_baseline_is_denied(self):
        v = Gate(objective=self.obj).evaluate(self.change, candidate_samples=self.good_cand,
            measurement=_measurement(self.good_cand, self.good_base))
        self.assertFalse(v.admitted)
        self.assertIn("no baseline", v.reason)

    def test_measured_without_candidate_samples_is_denied(self):
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=_baseline(self.obj, self.good_base))
        self.assertFalse(v.admitted)
        self.assertIn("no candidate samples", v.reason)

    def test_a_clean_win_is_admitted_as_measured(self):
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=_baseline(self.obj, self.good_base),
            candidate_samples=self.good_cand, guardrail_samples=self.flat_guards,
            measurement=_measurement(self.good_cand, self.good_base))
        self.assertTrue(v.admitted, v.reason)
        self.assertIs(v.evidence, EvidenceClass.MEASURED)
        self.assertLess(v.detail["p_value"], self.obj.alpha)
        self.assertAlmostEqual(v.detail["effect"], 0.20, places=2)

    def test_a_guardrail_regression_vetoes_a_real_win(self):
        """The primary metric wins by 20% and the image gets measurably worse.
        V9's Pareto gate had this right and V12 keeps it."""
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=_baseline(self.obj, self.good_base),
            candidate_samples=self.good_cand,
            measurement=_measurement(self.good_cand, self.good_base),
            guardrail_samples={"lpips": ([0.20] * 5, [0.30] * 5)})
        self.assertFalse(v.admitted)
        self.assertIn("guardrail", v.reason)

    def test_a_guardrail_within_tolerance_does_not_veto(self):
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=_baseline(self.obj, self.good_base),
            candidate_samples=self.good_cand,
            measurement=_measurement(self.good_cand, self.good_base),
            guardrail_samples={**self.flat_guards,
                               "lpips": ([0.200] * 5, [0.201] * 5)})
        self.assertTrue(v.admitted, v.reason)
        self.assertIn("guardrail_lpips", v.detail)

    def test_a_declared_guardrail_with_no_samples_denies(self):
        """Was skipped: the objective declares lpips and clip, and a win with no
        guardrail samples at all was admitted with neither checked."""
        for guards in (None, {"lpips": self.flat_guards["lpips"]},
                       {"lpips": self.flat_guards["lpips"], "clip": ([], [])}):
            with self.subTest(guards=guards):
                v = Gate(objective=self.obj).evaluate(
                    self.change, baseline=_baseline(self.obj, self.good_base),
                    candidate_samples=self.good_cand, guardrail_samples=guards,
            measurement=_measurement(self.good_cand, self.good_base))
                self.assertFalse(v.admitted)
                self.assertIn("guardrail", v.reason)
                self.assertIn("not measured", v.reason)

    def test_a_baseline_for_the_wrong_metric_is_rejected_before_the_statistics(self):
        b = _baseline(self.obj, self.good_base)
        b.metric = "fps"
        v = Gate(objective=self.obj).evaluate(
            self.change, baseline=b, candidate_samples=self.good_cand,
            measurement=_measurement(self.good_cand, self.good_base))
        self.assertFalse(v.admitted)
        self.assertIn("fps", v.reason)


class TestVerificationIsNotEvidence(unittest.TestCase):
    """The MOS2026 failure restated. 'Verified' there only ever meant 'landed'.
    On a LYING target even that much is unavailable, and saying so is the point."""

    def test_a_lying_target_is_never_recorded_as_verified(self):
        c = _change(kind=ChangeKind.NVML_OFFSET, target="offsets")
        gate = _optin(allow_destructive_reset=True)
        v = gate.evaluate(c)
        self.assertTrue(v.admitted)
        rec = Recorder()
        res = GatedActuator(rec, Tier.SYSTEM).apply(c, verdict=v)
        self.assertTrue(res.applied)
        self.assertFalse(res.verified)
        self.assertIn("NOT VERIFIED", res.verification_source)
        self.assertIn("behavioural", res.verification_source)

    def test_verify_on_a_lying_target_does_not_consult_the_inner_actuator(self):
        c = _change(kind=ChangeKind.NVML_OFFSET, target="offsets")
        rec = Recorder()
        res = GatedActuator(rec, Tier.SYSTEM).verify(c)
        self.assertFalse(res.verified)
        self.assertEqual(rec.calls, [], "a read-back on a lying target is not evidence, "
                                        "so asking for one is worse than useless")
        self.assertIn("not evidence", res.verification_source)

    def test_a_readable_target_is_verified_normally(self):
        c = _change(target="X", snapshot=_snap("0"))
        rec = Recorder()
        res = GatedActuator(rec, Tier.SYSTEM).verify(c)
        self.assertTrue(res.verified)
        self.assertEqual(rec.calls, [("verify", c.id)])


class TestRollbackTakesNoEvidence(unittest.TestCase):
    def _admitted(self, change, gate=None):
        return (gate or _optin()).evaluate(change)

    def test_rollback_restores_the_prior_when_one_was_readable(self):
        c = _change(target="X", snapshot=_snap("0"))
        rec = Recorder()
        act = GatedActuator(rec, Tier.SYSTEM)
        act.apply(c, verdict=self._admitted(c))
        res = act.rollback(c)
        self.assertTrue(res.applied)
        self.assertEqual(rec.calls, [("apply", c.id), ("undo", c.id)])
        self.assertIs(c.rollback_verb, Rollback.RESTORE_PRIOR)

    def test_rollback_resets_when_the_prior_was_never_recoverable(self):
        c = _change(kind=ChangeKind.NVML_CLOCK, target="clocks")
        rec = Recorder()
        act = GatedActuator(rec, Tier.SYSTEM)
        act.apply(c, verdict=self._admitted(c, _optin(allow_destructive_reset=True)))
        act.rollback(c)
        self.assertEqual(rec.calls, [("apply", c.id), ("reset", c.id)],
                         "an unreadable prior must be RESET, never invented")

    def test_rollback_refuses_for_a_guided_change(self):
        c = _change(kind=ChangeKind.GUIDED, target="x",
                    extra={"instructions": ["do it by hand"]})
        rec = Recorder()
        res = GatedActuator(rec, Tier.SYSTEM).rollback(c)
        self.assertFalse(res.applied)
        self.assertIn("never applied", res.detail)
        self.assertEqual(rec.calls, [])

    def test_rollback_reports_a_missing_reset_rather_than_pretending(self):
        class NoReset(Recorder):
            reset_default = None
        c = _change(kind=ChangeKind.NVML_CLOCK, target="clocks")
        res = GatedActuator(NoReset(), Tier.SYSTEM).rollback(c)
        self.assertFalse(res.applied)
        self.assertIn("reset_default", res.error)


class TestHonestyMetric(unittest.TestCase):
    def test_the_breakdown_counts_only_what_was_admitted(self):
        obj = diffusion_objective()
        gate = Gate(objective=obj, allow_unverified=True)   # operator opted in to CITED
        changes = [
            _change("a", target="A", snapshot=_snap("0")),               # CITED, admitted
            _change("b", target="B", snapshot=_snap("0")),               # CITED, admitted
            _change("c", kind=ChangeKind.GUIDED, target="C"),            # denied
            _change("d", target="D"),                                    # denied: no prior
            _change("e", kind=ChangeKind.NVML_CLOCK, target="E"),        # denied: destructive
        ]
        gate.evaluate(_change("m", target="M", evidence=EvidenceClass.DERIVED,
                              snapshot=_snap("0")),
                      baseline=_baseline(obj, _STEADY), predicted_effect=0.40)
        for c in changes:
            gate.evaluate(c)

        b = evidence_breakdown(gate.history)
        self.assertEqual(b["cited"], 2)
        self.assertEqual(b["derived"], 1)
        self.assertEqual(b["measured"], 0)
        self.assertEqual(b["guided"], 0,
                         "a DENIED guided change must not inflate the guided count; "
                         "the breakdown reports the applied set, by evidence class")

    def test_a_denied_verdict_says_so(self):
        v = Gate().evaluate(_change(target="X"))
        self.assertFalse(v.admitted)
        self.assertTrue(v.describe().startswith("[DENY"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
