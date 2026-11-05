"""
V12 — admission. The verdict, the gate, and the actuator that cannot be called
without one.

The defect, restated at the point where it is fixed
---------------------------------------------------
`v9/apply.py` contains `def apply_all(...)` whose body is, in effect,
`recs[:top_n]`. A real Pareto gate exists in `v9/quality.py`
(`pareto_admit`: speedup >= 1.30 AND lpips <= 0.04 AND clip >= -0.02). The
actuator does not call it. The gate is not broken; it is simply not wired to the
thing that moves state.

The obvious fix is a policy check at the top of `apply_all`. That fix rots: the
next edit that adds a fast path, a `--force` flag, or an "urgent" branch drops the
check, and the failure is invisible because everything still works — it just
works on unvetted changes.

So V12 makes the verdict a *parameter*:

    def apply(self, change: Change, *, verdict: Verdict) -> ApplyResult

Keyword-only, no default, no sentinel, no optional. There is no way to call it
that does not name a verdict, and the verdict is checked for authorship against
the change it claims to admit. Dropping the gate now requires deleting an
argument, which is a visible act, not an omission.

The deliberate asymmetry
------------------------
Forward motion requires a verdict. **Rollback does not.** `rollback()` takes no
verdict and will refuse one if passed. Reverting a change that turned out to be
bad must never be gated on having proved that it should be reverted — that would
make the safety path depend on the very evidence that just failed to materialise.
Undoing is a safety operation; the gate guards optimization only.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .change import (Change, ChangeKind, EvidenceClass, Observability, Rollback,
                     Tier, change_key)
from .baseline import Baseline, Measurement
from .objective import Objective

log = logging.getLogger("v12.admission")


class UnadmittedChange(RuntimeError):
    """Raised when a change reaches an actuator without an admitting verdict.

    Deliberately an exception and not a warning: a warning here is exactly the
    silent-failure mode V12 exists to eliminate."""


#: How far a supplied paired difference may sit from the difference the two
#: series actually have.
#:
#: The differences are computed by the harness from the same two numbers, so
#: in principle they agree exactly. In practice the series travel as decimal
#: TEXT — `common.series()` prints four decimal places, and an operator can
#: type them by hand — so each of the three values carries up to 5e-5 of
#: rounding error. The tolerance covers that and nothing more: it is not there
#: to absorb a difference that genuinely belongs to another pairing.
_PAIR_RECONCILE_ABS_TOL = 1e-3
_PAIR_RECONCILE_REL_TOL = 1e-6


def _worst_pair_mismatch(baseline: Sequence[float], candidate: Sequence[float],
                          diffs: Sequence[float]):
    """Does `diffs` actually describe candidate-minus-baseline for these two
    series?

    Returns None when they reconcile, and (index, supplied, expected) for the
    first pair that does not. The check is what stops a paired set measured
    against a DIFFERENT baseline from being used to authorise this one: it
    would pass every other check in this file and still describe a comparison
    nobody made.
    """
    n = min(len(baseline), len(candidate), len(diffs))
    scale = max((abs(v) for v in baseline), default=0.0) or 1.0
    tol = max(_PAIR_RECONCILE_ABS_TOL, _PAIR_RECONCILE_REL_TOL * scale)
    for i in range(n):
        want = candidate[i] - baseline[i]
        got = diffs[i]
        if abs(got - want) > tol:
            return (i, got, want)
    return None


class GateRefusal(RuntimeError):
    """The Gate could not even evaluate — no objective, no baseline, no way to
    know. Distinct from a denial (which is a decision) so the operator can tell
    'we decided no' from 'we could not decide'."""


@dataclass
class Verdict:
    """The record of a decision. Carries WHO decided and on WHAT evidence, so the
    ledger can later report how much of the applied set was actually measured."""
    admitted: bool
    change_key: str
    change_id: str
    reason: str
    evidence: EvidenceClass
    detail: Dict[str, float] = field(default_factory=dict)
    objective_id: str = ""
    baseline_id: str = ""
    issued: float = field(default_factory=time.time)
    #: True when the Gate is permitting a change whose rollback cannot return the
    #: machine to its prior state. An operator has to have said yes to this.
    destructive_reset: bool = False
    guided_instructions: List[str] = field(default_factory=list)

    @property
    def evidence_class_name(self) -> str:
        return self.evidence.value

    def describe(self) -> str:
        tag = "ADMIT" if self.admitted else "DENY "
        ev = self.evidence.value.upper()
        extra = ""
        if self.detail.get("effect") is not None:
            extra = f" effect={self.detail['effect']:+.2%}"
            if self.detail.get("p_value") is not None:
                extra += f" p={self.detail['p_value']:.3g}"
        return f"[{tag}] {self.change_id} ({ev}){extra} — {self.reason}"


class Gate:
    """Produces Verdicts. The only component allowed to say yes.

    Policy, stated explicitly so it can be argued with rather than inferred:

      MEASURED  admitted when the objective's own test passes (significance AND
                effect size AND guardrails). This is the only class that can
                support the word "faster".
      DERIVED   admitted only with a measurement behind it: an Objective, a
                baseline that is valid for it, AND a predicted effect that clears
                the threshold. A prediction with nothing measured to anchor it is
                denied. Recorded as derived so it never counts as measured.
      CITED     never admitted by default (DESIGN_V12 §4: "recorded, never
                applied"). A source is not a verdict. An operator may opt in with
                `allow_unverified=True`, and even then only READABLE, non-high-risk
                changes pass, recorded as cited.
      GUIDED    never admitted. Returns the instructions instead.

    Anything whose rollback would reset operator state to vendor default is
    refused unless the operator opted in with `allow_destructive_reset=True`.
    """

    #: Evidence classes that may be applied at all, in descending trust. CITED is
    #: listed because an operator CAN opt in; by default it never admits.
    APPLICABLE = (EvidenceClass.MEASURED, EvidenceClass.DERIVED, EvidenceClass.CITED)

    def __init__(self, objective: Optional[Objective] = None,
                 allow_destructive_reset: bool = False,
                 allow_unverified: bool = False):
        self.objective = objective
        self.allow_destructive_reset = allow_destructive_reset
        self.allow_unverified = allow_unverified
        self.history: List[Verdict] = []

    # ---- helpers ---------------------------------------------------------
    def _record(self, v: Verdict) -> Verdict:
        self.history.append(v)
        log.info(v.describe())
        return v

    def _deny(self, change: Change, reason: str, ev: EvidenceClass,
              detail: Optional[Dict[str, float]] = None) -> Verdict:
        return self._record(Verdict(
            admitted=False, change_key=change_key(change), change_id=change.id,
            reason=reason, evidence=ev, detail=detail or {},
            objective_id=self._obj_id(),
        ))

    def _obj_id(self) -> str:
        if not self.objective:
            return ""
        return f"{self.objective.workload.kind.value}:{self.objective.workload.name}"

    # ---- the decision ----------------------------------------------------
    def evaluate(self, change: Change,
                 baseline: Optional[Baseline] = None,
                 candidate_samples: Optional[Sequence[float]] = None,
                 guardrail_samples: Optional[Dict[str, Sequence[Sequence[float]]]] = None,
                 predicted_effect: Optional[float] = None,
                 now_conditions: Any = None,
                 measurement: Optional[Measurement] = None) -> Verdict:
        """Decide. Never raises for a normal refusal — a refusal IS a verdict.

        `measurement` is the candidate's provenance: when it was measured, in
        which session, under what machine conditions, and the per-pair
        differences against its control. It is optional as an ARGUMENT and
        mandatory as a FACT — when it is absent, a MEASURED claim is refused,
        because a bare list of floats cannot be placed in time and therefore
        cannot be shown to describe the same machine state as the baseline.

        It is read from the change's `extra` dict when not passed explicitly,
        so a measurement can travel with the change it belongs to.
        """
        if measurement is None:
            measurement = Measurement.from_extra(getattr(change, "extra", None))
        # 1. GUIDED is never applied, by construction.
        if change.kind is ChangeKind.GUIDED or change.evidence is EvidenceClass.GUIDED:
            v = self._record(Verdict(
                admitted=False, change_key=change_key(change), change_id=change.id,
                reason="guided change: surfaced as instructions, never applied",
                evidence=EvidenceClass.GUIDED, objective_id=self._obj_id(),
                guided_instructions=list(change.extra.get("instructions", [])),
            ))
            return v

        # 2. Destroying operator state needs an explicit opt-in, before anything else.
        if change.destroys_operator_state and not self.allow_destructive_reset:
            return self._deny(
                change,
                f"rollback would reset to VENDOR DEFAULT, not to the operator's prior "
                f"state ({change.observability.value}: {self._why_unobservable(change)}) "
                f"— re-run with allow_destructive_reset to accept this",
                change.evidence,
            )

        # 2b. A restorable target with no captured prior is a pipeline-order bug,
        #     not a hardware limit. OBSERVE must snapshot before ADMIT evaluates.
        #     Without this check the change would be admitted and its rollback
        #     would silently delete the operator's value forever.
        if (change.observability is Observability.READABLE
                and not change.prior_recorded
                and not self.allow_destructive_reset):
            return self._deny(
                change,
                f"no prior value was captured for '{change.target}' before admission — "
                f"OBSERVE must snapshot this target first, otherwise rolling back would "
                f"delete the operator's value with no way to restore it",
                change.evidence,
            )

        # 3. MEASURED is the only path that may claim a speedup, and it must have
        #    the raw material: an objective, a valid baseline, fresh samples, and
        #    proof that the candidate was measured against THAT baseline.
        if change.evidence is EvidenceClass.MEASURED:
            return self._evaluate_measured(change, baseline, candidate_samples,
                                           guardrail_samples, now_conditions,
                                           measurement)

        # 4. DERIVED needs a model reference, a predicted magnitude, AND a measured
        #    anchor (DESIGN_V12 §3: "admissible only with a measurement behind
        #    it"). A prediction alone is a citation with arithmetic in it.
        if change.evidence is EvidenceClass.DERIVED:
            if not change.evidence_source:
                return self._deny(change, "DERIVED claim with no model reference "
                                          "(evidence_source is empty)", change.evidence)
            if not change.verify_is_trustworthy and change.observability is Observability.LYING:
                return self._deny(
                    change, "model-predicted change on a target whose getter lies — "
                            "there is no way to confirm it took effect", change.evidence)
            if self.objective is None:
                return self._deny(change, "DERIVED claim but no Objective is loaded — "
                                          "a prediction needs a measured anchor",
                                  change.evidence)
            if baseline is None:
                return self._deny(change, "DERIVED claim but no baseline exists for this "
                                          "workload — a prediction with nothing measured "
                                          "behind it is not admissible", change.evidence)
            ok, why = baseline.valid_for(self.objective, now_conditions)
            if not ok:
                return self._deny(change, f"baseline is not valid for this comparison — "
                                          f"{why}", change.evidence)
            if predicted_effect is None:
                return self._deny(change, "DERIVED claim with no predicted effect — none "
                                          "was supplied; the plane does not yet compute a "
                                          "per-change roofline prediction",
                                  change.evidence)
            eff = float(predicted_effect)
            need = self.objective.min_effect
            if not eff >= need:                              # NaN fails closed too
                return self._deny(
                    change, f"model predicts {eff:+.2%}, below the {need:.1%} this "
                            f"objective requires", change.evidence,
                    {"effect": eff, "min_effect": need})
            return self._record(Verdict(
                admitted=True, change_key=change_key(change), change_id=change.id,
                reason=f"model-predicted {eff:+.2%} >= {need:.1%} against baseline "
                       f"{baseline.id} (NOT measured on this box; recorded as derived)",
                evidence=EvidenceClass.DERIVED, detail={"effect": eff, "min_effect": need},
                objective_id=self._obj_id(), baseline_id=baseline.id,
                destructive_reset=change.destroys_operator_state,
            ))

        # 5. CITED — known-good vendor practice, unverified here.
        if change.evidence is EvidenceClass.CITED:
            if not change.evidence_source:
                return self._deny(change, "CITED claim with no source "
                                          "(evidence_source is empty)", change.evidence)
            if not self.allow_unverified:
                return self._deny(change, "CITED evidence is never admitted — a source is "
                                          "not a measurement (DESIGN_V12 §4: recorded, "
                                          "never applied). No operator-facing opt-in exists "
                                          "in this build; Gate(allow_unverified=True) is a "
                                          "programmatic switch only",
                                  change.evidence)
            if change.risk in ("high",):
                return self._deny(change, "high-risk change may not be applied on a "
                                          "cited-only basis", change.evidence)
            return self._record(Verdict(
                admitted=True, change_key=change_key(change), change_id=change.id,
                reason=f"cited practice ({change.evidence_source}) — applied UNVERIFIED "
                       f"on this machine",
                evidence=EvidenceClass.CITED, objective_id=self._obj_id(),
                destructive_reset=change.destroys_operator_state,
            ))

        return self._deny(change, f"unhandled evidence class {change.evidence}", change.evidence)

    def _evaluate_measured(self, change: Change, baseline: Optional[Baseline],
                           candidate: Optional[Sequence[float]],
                           guardrails: Optional[Dict[str, Sequence[Sequence[float]]]],
                           now_conditions: Any,
                           measurement: Optional[Measurement] = None) -> Verdict:
        if self.objective is None:
            return self._deny(change, "MEASURED claim but no Objective is loaded — "
                                      "there is nothing to measure against",
                              change.evidence)
        if baseline is None:
            return self._deny(change, "MEASURED claim but no baseline exists for this "
                                      "workload — record one first", change.evidence)
        if not candidate:
            return self._deny(change, "MEASURED claim but no candidate samples were "
                                      "supplied", change.evidence)

        ok, why = baseline.valid_for(self.objective, now_conditions,
                                    ignore=getattr(change, "varies", ()) or ())
        if not ok:
            return self._deny(change, f"baseline is not valid for this comparison — {why}",
                              change.evidence)

        # The same-session check. This is the check that closes the 2026-11-03
        # ghost: the baseline and the candidate have to be shown to come from
        # ONE measurement session, not merely to be about the same workload.
        # `valid_for` above cannot do this — on the day of the ghost every
        # condition it compares matched, and the difference between the two
        # series was still drift.
        same, why = baseline.same_session_as(
            measurement, ignore=getattr(change, "varies", ()) or ())
        if not same:
            return self._deny(change, f"baseline is not valid for this comparison — {why}",
                              change.evidence)

        gs = None
        if guardrails:
            gs = {k: (v[0], v[1]) for k, v in guardrails.items() if len(v) == 2}
        # The paired differences are the ONLY honest way to compare a candidate
        # with its control: they are measured against the same baseline arm,
        # in the same session, so the drift between the two cancels in the
        # difference instead of being charged to the candidate. Without them
        # this is a Welch comparison of two independent series, which is the
        # exact arithmetic that produced the ghost — so it is refused here
        # rather than being allowed to fall through to a test that cannot
        # distinguish drift from effect.
        paired_diffs = measurement.paired_diffs if measurement else None
        if not paired_diffs:
            return self._deny(change,
                              "the candidate was not measured against its control in "
                              "the same session — no paired differences were supplied, "
                              "and comparing two independent series cannot tell drift "
                              "from effect (this is how a +12% phantom was admitted "
                              "on 2026-11-03). Measure both arms alternately in one "
                              "session with the paired harness.",
                              change.evidence)
        # The differences have to be the differences BETWEEN THESE TWO SERIES.
        # A paired set carried over from another pairing would satisfy every
        # other check and still describe a comparison nobody made, so it is
        # reconciled here against the baseline and the candidate in hand.
        if len(paired_diffs) != len(candidate):
            return self._deny(change,
                              f"the paired differences ({len(paired_diffs)}) do not "
                              f"match the candidate series ({len(candidate)}) — they "
                              f"were measured against a different pairing, so they "
                              f"describe a comparison that was not made.",
                              change.evidence)
        mismatch = _worst_pair_mismatch(baseline.samples, list(candidate),
                                        list(paired_diffs))
        if mismatch is not None:
            i, got, want = mismatch
            return self._deny(change,
                              f"the paired differences do not match these two series "
                              f"(pair {i}: candidate - baseline = {want:.6g}, but the "
                              f"difference supplied is {got:.6g}) — they belong to "
                              f"another pairing, not to this comparison.",
                              change.evidence)
        admitted, reason, detail = self.objective.admits(
            baseline.samples, list(candidate), gs, paired_diffs=paired_diffs)
        if not admitted:
            return self._deny(change, reason, change.evidence, detail)
        return self._record(Verdict(
            admitted=True, change_key=change_key(change), change_id=change.id,
            reason=reason, evidence=EvidenceClass.MEASURED, detail=detail,
            objective_id=self._obj_id(), baseline_id=baseline.id,
            destructive_reset=change.destroys_operator_state,
        ))

    @staticmethod
    def _why_unobservable(change: Change) -> str:
        if change.observability is Observability.RESET_ONLY:
            return "no read-back API exists for this target"
        if change.observability is Observability.LYING:
            return "the getter for this target returns a plausible wrong value"
        return "unknown"


@dataclass
class ApplyResult:
    """The outcome of one actuator call.

    `verified` defaults to False on purpose. It is the honest answer for every
    path that did not actually establish anything — a refusal, a missing reset
    verb, an unreadable target — and those paths are exactly where a required
    field would turn a clean refusal into a TypeError. Verified is something a
    call EARNS, never something it starts with.
    """
    change_id: str
    applied: bool
    verified: bool = False
    detail: str = ""
    verification_source: str = ""
    error: str = ""


class GatedActuator:
    """Wraps a tier actuator and enforces the verdict at the call boundary.

    The wrapped object is whatever actually mutates state — MOS2026's executor for
    SYSTEM tier, an engine builder for APP_GRAPH, a compiler for KERNEL. This
    wrapper does not care what it is; it only ensures nobody reaches it without a
    decision.
    """

    def __init__(self, inner: Any, tier: Tier):
        self.inner = inner
        self.tier = tier
        self.applied_count = 0

    def apply(self, change: Change, *, verdict: Verdict) -> ApplyResult:
        """Apply `change`. `verdict` is required, keyword-only, and unmockable."""
        if verdict is None:                                  # defensive; unreachable via the language
            raise UnadmittedChange(
                f"apply() for '{change.id}' was called with verdict=None — this is the "
                f"exact failure mode V12 exists to prevent")
        if not isinstance(verdict, Verdict):
            raise UnadmittedChange(
                f"apply() for '{change.id}' received {type(verdict).__name__}, not a Verdict")
        if not verdict.admitted:
            raise UnadmittedChange(
                f"apply() refused for '{change.id}': {verdict.reason}")

        # A verdict is bound to the change it was issued for. Reusing an admitted
        # verdict on a different change is the loophole that would otherwise let
        # one measurement authorize an unrelated mutation.
        if verdict.change_key != change_key(change):
            raise UnadmittedChange(
                f"verdict was issued for '{verdict.change_key}' but presented for "
                f"'{change_key(change)}' ({change.id}) — a verdict admits one change, "
                f"not a family of them")
        if change.tier is not self.tier:
            raise UnadmittedChange(
                f"change '{change.id}' is {change.tier.value} tier but this actuator is "
                f"{self.tier.value}")

        if verdict.destructive_reset:
            log.warning(
                "applying '%s': rollback will RESET to vendor default, not restore the "
                "operator's prior state (%s)", change.id, change.observability.value)

        res = self.inner.apply(change)

        # Verification is where the LYING class bites. A read-back on such a target
        # is not evidence, so we do not record it as though it were.
        if isinstance(res, ApplyResult):
            if not change.verify_is_trustworthy:
                res.verified = False
                res.verification_source = (
                    f"NOT VERIFIED — {change.observability.value} target; read-back is "
                    f"not trustworthy, needs behavioural verification under load")
            if res.applied:
                self.applied_count += 1
            return res

        # Inner actuator returned something else; normalise.
        self.applied_count += 1
        return ApplyResult(change_id=change.id, applied=True, verified=False,
                           detail=str(res or ""))

    def rollback(self, change: Change) -> ApplyResult:
        """Undo. NO verdict required — deliberately.

        Gating the safety path on evidence would mean a change that failed to
        produce evidence could not be undone. Undoing is not an optimization and
        is therefore not subject to the admission policy."""
        verb = change.rollback_verb
        if verb is Rollback.REFUSE:
            return ApplyResult(change_id=change.id, applied=False,
                               detail="refused: guided change was never applied")
        if verb is Rollback.RESET_DEFAULT:
            log.warning(
                "rolling back '%s' by RESET-TO-DEFAULT; the operator's prior value was "
                "not recoverable (%s)", change.id, change.observability.value)
            fn = getattr(self.inner, "reset_default", None)
            if fn is None:
                return ApplyResult(change_id=change.id, applied=False,
                                   error="inner actuator has no reset_default()")
            res = fn(change)
        else:
            res = self.inner.apply_undo(change)
        return res if isinstance(res, ApplyResult) else ApplyResult(
            change_id=change.id, applied=True, verified=False, detail=str(res or ""))

    def verify(self, change: Change) -> ApplyResult:
        """Structural verification: did the value land?

        This is what MOS2026 does, and it is NOT nothing — a change that did not
        land cannot have helped. It is simply not sufficient, which is why it is
        named 'verify' and never 'confirmed'."""
        if not change.verify_is_trustworthy:
            return ApplyResult(
                change_id=change.id, applied=change.applied, verified=False,
                verification_source=f"{change.observability.value} target — read-back "
                                    f"is not evidence")
        fn = getattr(self.inner, "verify", None)
        if fn is None:
            return ApplyResult(change_id=change.id, applied=change.applied, verified=False,
                               verification_source="inner actuator cannot verify")
        res = fn(change)
        return res if isinstance(res, ApplyResult) else ApplyResult(
            change_id=change.id, applied=change.applied, verified=bool(res))


def evidence_breakdown(verdicts: Sequence[Verdict]) -> Dict[str, int]:
    """The honesty metric's raw material.

    A ledger that says "Applied 40, 40 verified" is the MOS2026 failure restated:
    every applied change is 'verified' because verification only ever meant
    'landed'. This breakdown is what V12 reports instead, so the ratio of measured
    to cited is visible rather than hidden behind a satisfying-looking number.
    """
    out = {e.value: 0 for e in EvidenceClass}
    for v in verdicts:
        if v.admitted:
            out[v.evidence.value] += 1
    return out
