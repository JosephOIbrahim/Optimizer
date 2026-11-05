"""
V12 — the universal optimization ABI.

Every optimization on this machine is ONE `Change`. A Change is atomic,
reversible, tiered, and — the part neither predecessor had — **evidenced**.

Why this file exists
--------------------
MachineOptimizer2026 and V9 are near mirror images. MOS2026 has genuine
reversibility and a watchdog but no hardware model and no notion of "faster";
it verifies read-back == target and calls that success. V9 has a real roofline,
a real profiler and a real Pareto gate — and its actuator routes around the gate
(`apply.py: apply_all()` is `recs[:top_n]`).

So the shared defect is not in either engine. It is that at the moment state
changes hands, nothing checks whether the change was worth making. V12 fixes that
by making the claim itself typed (EvidenceClass) and by making the actuator
uncallable without a Verdict (see admission.py).

Two corrections that came out of hardware research, both encoded here
---------------------------------------------------------------------
1. **Not every target is readable.** MOS2026's executor assumes it can read a
   prior value and put it back. On this box that is false for a whole family of
   targets, and the failure is SILENT:

     * `nvmlDeviceSetGpuLockedClocks` works, but NVML exposes NO getter. There is
       no read-back API at all. And `nvmlDeviceResetGpuLockedClocks` returns you to
       VENDOR DEFAULT, not to whatever was there before — so if the operator had a
       lock or undervolt applied, running the optimizer destroys it unrecoverably.
     * `nvmlDeviceSetClockOffsets` is worse: the setter works AND a getter exists,
       but `nvmlDeviceGetMinMaxClockOfPState` reports the offset as 0 even when one
       is set. A read-back branch would record "0" as ground truth and be
       confidently wrong. Same class: NVIDIA App's power-management mode and CUDA
       Sysmem Fallback Policy, written over NVAPI with no supported read-back —
       and NVIDIA's own 616.56 release notes document that "Prefer Maximum
       Performance" may not be applied correctly.

   A getter that returns a plausible wrong answer is more dangerous than no
   getter, which is why Observability.LYING is a distinct class rather than a
   flavour of RESET_ONLY.

2. **Rollback is two verbs, not one.** `restore(prior)` where the prior is
   readable and trustworthy; `reset(default)` where it is not. Collapsing them is
   exactly how you silently destroy operator state.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


# --------------------------------------------------------------------------
# How the prior value can be recovered.
# --------------------------------------------------------------------------
class Observability(enum.Enum):
    READABLE = "readable"      # prior value readable + trustworthy -> restore it
    RESET_ONLY = "reset_only"  # NO getter exists -> prior is unrecoverable; reset to default only
    LYING = "lying"            # a getter exists but returns a wrong value -> NEVER trust read-back


# --------------------------------------------------------------------------
# Which actuator layer the change belongs to.
# --------------------------------------------------------------------------
class Tier(enum.Enum):
    SYSTEM = "system"          # env vars, registry, power plan, Defender, config files
    APP_GRAPH = "app_graph"    # torch.compile, TRT engines, Engine.ini cvars, houdini.env
    KERNEL = "kernel"          # compiled kernel artifacts (nvcc/CUTLASS), runtime wraps


# --------------------------------------------------------------------------
# How the claimed benefit is known. This is the anti-fabrication field.
# --------------------------------------------------------------------------
class EvidenceClass(enum.Enum):
    """V9's Pareto gate was real but ran on fabricated inputs — 153 of its 156
    ledger rows carry lpips=0.0/clip=0.0 by construction, and its TensorRT rows
    were validated on a matmul, never a real UNet. The gate could not tell, because
    nothing in the record said HOW the number was obtained. This field says."""
    MEASURED = "measured"      # baseline + candidate measurement exist on THIS box
    DERIVED = "derived"        # predicted from the roofline model (bound classification)
    CITED = "cited"            # vendor docs / known-good, with a source
    GUIDED = "guided"          # never auto-applied; surfaced as instructions only


class ChangeKind(enum.Enum):
    # --- SYSTEM tier (inherited from MachineOptimizer2026, which proved these work)
    ENV_VAR = "env_var"
    REGISTRY = "registry"
    POWER_PLAN = "power_plan"
    DEFENDER_EXCLUSION = "defender_exclusion"
    FILE_LINE = "file_line"
    COMMAND = "command"
    # --- SYSTEM tier, the non-observable family (new in V12)
    NVML_CLOCK = "nvml_clock"          # locked clocks: works, NO getter, reset-only
    NVML_OFFSET = "nvml_offset"        # clock offsets: works, getter LIES (reports 0)
    NVML_POWER_LIMIT = "nvml_power_limit"  # power ceiling: works, getter HONEST, restorable
    NVAPI_PROFILE = "nvapi_profile"    # NVIDIA App 3D profile: no supported read-back
    # --- APP_GRAPH tier
    APP_CONFIG = "app_config"          # structured app setting (Engine.ini, houdini.env)
    RUNTIME_WRAP = "runtime_wrap"      # SageAttention / torch.compile monkeypatch
    ENGINE = "engine"                  # TensorRT serialized plan
    # --- KERNEL tier
    KERNEL_ARTIFACT = "kernel_artifact"  # nvcc/CUTLASS compiled .dll/.cubin
    # --- never applied
    GUIDED = "guided"


#: Defaults so the common case is correct without the caller thinking about it,
#: while a mis-declaration stays visible in the record rather than silent.
KIND_DEFAULTS: Dict[ChangeKind, Dict[str, Any]] = {
    ChangeKind.ENV_VAR:            {"tier": Tier.SYSTEM,    "observability": Observability.READABLE},
    ChangeKind.REGISTRY:           {"tier": Tier.SYSTEM,    "observability": Observability.READABLE},
    ChangeKind.POWER_PLAN:         {"tier": Tier.SYSTEM,    "observability": Observability.READABLE},
    ChangeKind.DEFENDER_EXCLUSION: {"tier": Tier.SYSTEM,    "observability": Observability.READABLE},
    ChangeKind.FILE_LINE:          {"tier": Tier.SYSTEM,    "observability": Observability.READABLE},
    ChangeKind.COMMAND:            {"tier": Tier.SYSTEM,    "observability": Observability.READABLE},
    # The researched family — these are the whole point of the extra field.
    ChangeKind.NVML_CLOCK:         {"tier": Tier.SYSTEM,    "observability": Observability.RESET_ONLY},
    ChangeKind.NVML_OFFSET:        {"tier": Tier.SYSTEM,    "observability": Observability.LYING},
    # The ONLY NVML target on this card whose prior is readable AND whose
    # setter round-trips: nvmlDeviceGetPowerManagementLimit returns the
    # ceiling in watts, and nvmlDeviceSetPowerManagementLimit writes it back
    # exactly. So undo restores the operator's prior value instead of
    # resetting to vendor default — the first member of this family that
    # does not destroy operator state.
    ChangeKind.NVML_POWER_LIMIT:   {"tier": Tier.SYSTEM,    "observability": Observability.READABLE},
    ChangeKind.NVAPI_PROFILE:      {"tier": Tier.SYSTEM,    "observability": Observability.LYING},
    ChangeKind.APP_CONFIG:         {"tier": Tier.APP_GRAPH, "observability": Observability.READABLE},
    ChangeKind.RUNTIME_WRAP:       {"tier": Tier.APP_GRAPH, "observability": Observability.RESET_ONLY},
    ChangeKind.ENGINE:             {"tier": Tier.APP_GRAPH, "observability": Observability.READABLE},
    ChangeKind.KERNEL_ARTIFACT:    {"tier": Tier.KERNEL,    "observability": Observability.READABLE},
    ChangeKind.GUIDED:             {"tier": Tier.SYSTEM,    "observability": Observability.RESET_ONLY},
}

#: Kinds that need Administrator on this box (all verified in MOS2026's catalog).
ADMIN_KINDS = {
    ChangeKind.REGISTRY,
    ChangeKind.POWER_PLAN,
    ChangeKind.DEFENDER_EXCLUSION,
    ChangeKind.NVAPI_PROFILE,
    ChangeKind.NVML_CLOCK,
    ChangeKind.NVML_OFFSET,
    ChangeKind.NVML_POWER_LIMIT,
}


class Rollback(enum.Enum):
    """The two verbs. `prior` is only used for RESTORE_PRIOR."""
    RESTORE_PRIOR = "restore_prior"
    RESET_DEFAULT = "reset_default"
    REFUSE = "refuse"   # GUIDED, or a LYING target we will not touch automatically


@dataclass
class Snapshot:
    """A prior value, WITH its provenance.

    `trusted=False` means: do not write this back as if it were the machine's real
    prior state. Either there was no getter, or the getter is known to lie. The
    distinction is carried all the way into the ledger so a later reader can tell
    a genuine prior from a placeholder."""
    value: Any = None
    existed: bool = False
    trusted: bool = False
    source: str = ""           # how it was read, e.g. "winreg.QueryValueEx", "NONE (no getter)"
    error: str = ""


@dataclass
class Change:
    id: str
    kind: ChangeKind
    description: str
    target: str
    target_value: Any = None

    # tier + observability: taken from KIND_DEFAULTS at construction unless overridden
    tier: Tier = Tier.SYSTEM
    observability: Observability = Observability.READABLE

    # how the claimed benefit is known (anti-fabrication)
    evidence: EvidenceClass = EvidenceClass.CITED
    benefit_claim: str = ""
    evidence_source: str = ""       # URL / model reference / baseline id

    # --- kind-specific (mirrors MOS2026's proven executor contract) ---
    reg_value_name: Optional[str] = None
    reg_type: Optional[str] = None
    scope: str = "user"             # "user" | "machine"
    file_key: Optional[str] = None
    file_sep: str = "="
    section: Optional[str] = None
    snapshot_cmd: Optional[str] = None
    apply_cmd: Optional[str] = None
    undo_cmd: Optional[str] = None
    reset_cmd: Optional[str] = None   # RESET_ONLY targets: how to return to vendor default
    verify_cmd: Optional[str] = None

    # --- runtime bookkeeping ---
    snapshot: Optional[Snapshot] = None
    applied: bool = False
    verified: bool = False
    risk: str = "low"
    requires_admin: bool = False
    #: Condition fields this change DELIBERATELY alters, named as they appear in
    #: v12.baseline.Conditions (e.g. "power_limit_w").
    #:
    #: A baseline is normally invalidated when the machine changes underneath it,
    #: because the numbers then describe a machine that no longer exists. But a
    #: change that alters one of those fields IS the difference being tested, so
    #: excluding that field from the comparison is not a weakening of the check —
    #: it is what makes the check meaningful. Raising the power ceiling from 450 W
    #: to 600 W and then asking "is the old baseline still valid?" must not fail
    #: merely because the change under test was applied.
    #:
    #: Only a field this change actually moves may be named here. Every other
    #: field still invalidates the baseline exactly as before, so a driver
    #: update or a competing process appearing mid-run still refuses the
    #: comparison.
    varies: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        d = KIND_DEFAULTS.get(self.kind)
        if d is not None:
            # Only apply the default when the caller left the field at its own default,
            # so an explicit override survives.
            if self.tier is Tier.SYSTEM and d["tier"] is not Tier.SYSTEM:
                self.tier = d["tier"]
            if self.observability is Observability.READABLE and d["observability"] is not Observability.READABLE:
                self.observability = d["observability"]
        if self.kind in ADMIN_KINDS:
            self.requires_admin = True
        if self.kind is ChangeKind.GUIDED:
            self.evidence = EvidenceClass.GUIDED

    # ---- the rollback contract -------------------------------------------
    @property
    def rollback_verb(self) -> Rollback:
        if self.kind is ChangeKind.GUIDED:
            return Rollback.REFUSE
        if self.observability is Observability.READABLE:
            return Rollback.RESTORE_PRIOR if (self.snapshot and self.snapshot.trusted) else Rollback.RESET_DEFAULT
        # RESET_ONLY and LYING both reset; LYING additionally flags for the operator.
        return Rollback.RESET_DEFAULT

    @property
    def prior_recorded(self) -> bool:
        """Do we actually hold a trustworthy prior value for this target?

        A READABLE target is only restorable if its snapshot was taken AND the
        snapshot is trusted. This is why OBSERVE must run before ADMIT: the Gate
        asks this question, and a `False` here for a READABLE change means the
        pipeline ran out of order."""
        return bool(self.snapshot and self.snapshot.trusted)

    @property
    def destroys_operator_state(self) -> bool:
        """True when this target's CLASS can never be restored to what the
        operator had — the reset-only and lying families, by construction. The
        operator has to be told before any of these is touched."""
        return self.observability in (Observability.RESET_ONLY, Observability.LYING)

    @property
    def rollback_loses_prior(self) -> bool:
        """True when rolling back will lose the operator's actual prior value —
        either because the class makes it unrecoverable, or because we never
        captured it. The second case is a bug, not a property of the hardware,
        and the Gate reports it as one."""
        return self.destroys_operator_state or not self.prior_recorded

    @property
    def verify_is_trustworthy(self) -> bool:
        """A LYING target can be applied but its read-back must never be believed;
        it has to be verified behaviourally (under load) or not at all."""
        return self.observability is not Observability.LYING


def change_key(c: Change) -> str:
    """Stable, tier-aware identity for a Change target.

    Used to snapshot each original value exactly once across watchdog cycles —
    the property MOS2026 got right and which must survive the merge."""
    t = c.tier.value
    if c.kind is ChangeKind.ENV_VAR:
        return f"{t}|env|{c.scope}|{c.target}"
    if c.kind is ChangeKind.REGISTRY:
        return f"{t}|reg|{c.target}|{c.reg_value_name}"
    if c.kind is ChangeKind.DEFENDER_EXCLUSION:
        return f"{t}|def|{c.target}"
    if c.kind is ChangeKind.FILE_LINE:
        return f"{t}|file|{c.target}|{c.section or ''}|{c.file_key}"
    if c.kind is ChangeKind.NVML_CLOCK:
        return f"{t}|nvml|clocks"
    if c.kind is ChangeKind.NVML_OFFSET:
        return f"{t}|nvml|offsets"
    if c.kind is ChangeKind.NVML_POWER_LIMIT:
        return f"{t}|nvml|power_limit"
    if c.kind is ChangeKind.NVAPI_PROFILE:
        return f"{t}|nvapi|{c.target}"
    return f"{t}|{c.kind.value}|{c.target}"


def make(kind: ChangeKind, cid: str, description: str, target: str, **kw: Any) -> Change:
    """Construct a Change with kind defaults applied. Thin sugar so agents read
    cleanly; the constructor is still the source of truth."""
    return Change(id=cid, kind=kind, description=description, target=target, **kw)
