"""
Tests for the Change ABI.

A `Change` carries three things neither predecessor carried together: its tier,
how its prior value can be recovered, and how its claimed benefit is known. Each
is tested here because each has a specific silent failure attached to it.

  * **Tier** — an actuator that takes a KERNEL change to a SYSTEM executor.
  * **Observability** — the one that costs real data. `nvmlDeviceResetGpuLockedClocks`
    returns to VENDOR DEFAULT, and `nvmlDeviceGetMinMaxClockOfPState` reports an
    offset as 0 even when one is set. A Change that models these as merely
    "readable" will record a fabricated prior and later write it back as truth.
  * **Evidence** — the field that makes `lpips=0.0/clip=0.0 by construction`
    visible instead of load-bearing.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12.change import (ADMIN_KINDS, KIND_DEFAULTS, Change, ChangeKind,  # noqa: E402
                        EvidenceClass, Observability, Rollback, Snapshot, Tier,
                        change_key, make)


def _c(kind=ChangeKind.ENV_VAR, target="T", cid="c1", **kw) -> Change:
    return Change(id=cid, kind=kind, description="d", target=target, **kw)


class TestKindDefaults(unittest.TestCase):
    def test_every_kind_has_a_declared_default(self):
        missing = [k.value for k in ChangeKind if k not in KIND_DEFAULTS]
        self.assertEqual(missing, [],
                         "an undeclared kind silently falls back to SYSTEM/READABLE, "
                         "which is the permissive answer — the wrong default")

    def test_the_unobservable_family_is_declared_as_such(self):
        """Researched on this box. Getting these wrong is how a prior gets invented."""
        self.assertIs(_c(ChangeKind.NVML_CLOCK).observability, Observability.RESET_ONLY,
                      "locked clocks: the setter works and NVML exposes NO getter")
        self.assertIs(_c(ChangeKind.NVML_OFFSET).observability, Observability.LYING,
                      "clock offsets: a getter exists and reports 0 even when an "
                      "offset is set — a plausible wrong answer")
        self.assertIs(_c(ChangeKind.NVAPI_PROFILE).observability, Observability.LYING)

    def test_lying_is_a_distinct_class_not_a_flavour_of_reset_only(self):
        """A getter that returns a plausible wrong value is worse than no getter,
        because a read-back branch would record it as ground truth."""
        self.assertIsNot(Observability.LYING, Observability.RESET_ONLY)

    def test_the_ordinary_family_is_readable(self):
        for kind in (ChangeKind.ENV_VAR, ChangeKind.REGISTRY, ChangeKind.POWER_PLAN,
                     ChangeKind.DEFENDER_EXCLUSION, ChangeKind.FILE_LINE,
                     ChangeKind.APP_CONFIG, ChangeKind.KERNEL_ARTIFACT):
            with self.subTest(kind=kind.value):
                self.assertIs(_c(kind).observability, Observability.READABLE)

    def test_tiers_are_assigned_by_kind(self):
        self.assertIs(_c(ChangeKind.ENV_VAR).tier, Tier.SYSTEM)
        self.assertIs(_c(ChangeKind.APP_CONFIG).tier, Tier.APP_GRAPH)
        self.assertIs(_c(ChangeKind.ENGINE).tier, Tier.APP_GRAPH)
        self.assertIs(_c(ChangeKind.KERNEL_ARTIFACT).tier, Tier.KERNEL)

    def test_an_explicit_override_survives_the_default(self):
        c = _c(ChangeKind.KERNEL_ARTIFACT, tier=Tier.APP_GRAPH)
        self.assertIs(c.tier, Tier.APP_GRAPH)

    def test_admin_kinds_are_flagged_without_being_asked(self):
        for kind in ADMIN_KINDS:
            with self.subTest(kind=kind.value):
                self.assertTrue(_c(kind).requires_admin)
        self.assertFalse(_c(ChangeKind.ENV_VAR).requires_admin)
        self.assertFalse(_c(ChangeKind.APP_CONFIG).requires_admin)

    def test_a_guided_kind_cannot_be_declared_as_anything_but_guided(self):
        c = _c(ChangeKind.GUIDED, evidence=EvidenceClass.MEASURED)
        self.assertIs(c.evidence, EvidenceClass.GUIDED)


class TestRollbackVerbs(unittest.TestCase):
    """Two verbs, not one. Collapsing them is exactly how operator state is lost."""

    def test_a_readable_target_with_a_trusted_snapshot_restores_the_prior(self):
        c = _c(snapshot=Snapshot("old", existed=True, trusted=True, source="winreg"))
        self.assertIs(c.rollback_verb, Rollback.RESTORE_PRIOR)
        self.assertTrue(c.prior_recorded)
        self.assertFalse(c.rollback_loses_prior)

    def test_a_readable_target_with_no_snapshot_resets_instead(self):
        c = _c()
        self.assertIs(c.rollback_verb, Rollback.RESET_DEFAULT)
        self.assertTrue(c.rollback_loses_prior,
                        "losing the prior here is a pipeline-order bug, not a hardware "
                        "limit — the Gate reports it as one")

    def test_an_untrusted_snapshot_is_not_a_prior(self):
        c = _c(snapshot=Snapshot("0", existed=True, trusted=False, source="LYING getter"))
        self.assertFalse(c.prior_recorded)
        self.assertIs(c.rollback_verb, Rollback.RESET_DEFAULT)

    def test_reset_only_and_lying_targets_always_reset(self):
        for kind in (ChangeKind.NVML_CLOCK, ChangeKind.NVML_OFFSET):
            with self.subTest(kind=kind.value):
                c = _c(kind)
                self.assertIs(c.rollback_verb, Rollback.RESET_DEFAULT)
                self.assertTrue(c.destroys_operator_state)
                self.assertTrue(c.rollback_loses_prior)

    def test_a_guided_change_refuses_rollback_entirely(self):
        self.assertIs(_c(ChangeKind.GUIDED).rollback_verb, Rollback.REFUSE)

    def test_a_lying_targets_read_back_is_not_trustworthy(self):
        self.assertFalse(_c(ChangeKind.NVML_OFFSET).verify_is_trustworthy)
        self.assertTrue(_c(ChangeKind.NVML_CLOCK).verify_is_trustworthy,
                        "a RESET_ONLY target has no getter to be misled by; it still "
                        "needs a reset verb, but that is a different problem")
        self.assertTrue(_c(ChangeKind.ENV_VAR).verify_is_trustworthy)


class TestChangeKey(unittest.TestCase):
    """The universal config address space. The property that matters is that the
    same target addressed twice is ONE address, so a prior is snapshotted once
    across watchdog cycles — the thing MOS2026 got right."""

    def test_the_same_target_is_one_address(self):
        self.assertEqual(change_key(_c(cid="a")), change_key(_c(cid="b")))

    def test_different_targets_are_different_addresses(self):
        self.assertNotEqual(change_key(_c(target="A")), change_key(_c(target="B")))

    def test_scope_is_part_of_an_env_var_address(self):
        self.assertNotEqual(change_key(_c(scope="user")),
                            change_key(_c(scope="machine")))

    def test_the_registry_value_name_is_part_of_the_address(self):
        a = _c(ChangeKind.REGISTRY, target=r"HKLM\X", reg_value_name="A")
        b = _c(ChangeKind.REGISTRY, target=r"HKLM\X", reg_value_name="B")
        self.assertNotEqual(change_key(a), change_key(b))

    def test_the_file_section_is_part_of_the_address(self):
        a = _c(ChangeKind.FILE_LINE, target="Engine.ini", section="A", file_key="k")
        b = _c(ChangeKind.FILE_LINE, target="Engine.ini", section="B", file_key="k")
        self.assertNotEqual(change_key(a), change_key(b))

    def test_the_address_is_tier_aware(self):
        a = _c(ChangeKind.ENV_VAR, tier=Tier.APP_GRAPH)
        self.assertTrue(change_key(a).startswith("app_graph|"))
        self.assertTrue(change_key(_c()).startswith("system|"))

    def test_there_is_one_clock_address_and_one_offset_address(self):
        """Locked clocks and clock offsets are each a single machine-wide setting,
        not one per call site. Two changes that both pin clocks address the same
        thing and must collide deliberately."""
        self.assertEqual(change_key(_c(ChangeKind.NVML_CLOCK, target="a")),
                         change_key(_c(ChangeKind.NVML_CLOCK, target="b")))
        self.assertEqual(change_key(_c(ChangeKind.NVML_OFFSET, target="a")),
                         change_key(_c(ChangeKind.NVML_OFFSET, target="b")))
        self.assertNotEqual(change_key(_c(ChangeKind.NVML_CLOCK, target="a")),
                            change_key(_c(ChangeKind.NVML_OFFSET, target="a")))

    def test_the_address_is_stable_across_calls(self):
        c = _c(ChangeKind.REGISTRY, target=r"HKLM\X", reg_value_name="V")
        self.assertEqual(change_key(c), change_key(c))


class TestAbi(unittest.TestCase):
    def test_a_change_survives_construction_with_only_the_required_fields(self):
        c = Change(id="x", kind=ChangeKind.ENV_VAR, description="d", target="T")
        self.assertEqual(c.scope, "user")
        self.assertEqual(c.risk, "low")
        self.assertFalse(c.applied)
        self.assertFalse(c.verified)
        self.assertIsNone(c.snapshot)
        self.assertEqual(c.extra, {})

    def test_make_applies_the_same_defaults_as_the_constructor(self):
        a = make(ChangeKind.NVML_CLOCK, "x", "d", "gpu")
        b = Change(id="x", kind=ChangeKind.NVML_CLOCK, description="d", target="gpu")
        self.assertEqual(change_key(a), change_key(b))
        self.assertIs(a.observability, b.observability)

    def test_the_evidence_classes_are_exactly_the_four_named(self):
        self.assertEqual({e.value for e in EvidenceClass},
                         {"measured", "derived", "cited", "guided"})

    def test_measured_is_distinguishable_from_the_rest(self):
        """The anti-fabrication point: a ledger row must say HOW its number was
        obtained, so a derived prediction can never be read as a measurement."""
        self.assertNotEqual(EvidenceClass.MEASURED, EvidenceClass.DERIVED)
        self.assertNotEqual(EvidenceClass.CITED, EvidenceClass.DERIVED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
