"""
V12 — the control plane. One loop, six stages, in this order:

    OBSERVE  read the machine and capture priors for every candidate target
    MODEL    derive the roofline, classify the workload, decide which levers apply
    DECIDE   assemble candidate Changes from the catalog
    ADMIT    the Gate issues a Verdict for each one — ADMIT BEFORE ACT, always
    ACT      the GatedActuator applies only what was admitted
    REMEMBER the ledger records the verdict, the evidence class, and the effect

The ordering of ADMIT and ACT is the whole design. V9 had a real Pareto gate at
stage 4 and an actuator that ran `recs[:top_n]` at stage 5. MachineOptimizer2026
had no stage 4 at all and treated stage 5's read-back as if it were stage 4's
verdict. Putting the gate in the signature of the actuator is what makes the order
unbreakable rather than conventional.

Two defaults that are deliberate:

  * `dry_run=True`. V12 does not mutate a machine until asked.
  * No objective means no optimization. If there is nothing to measure against,
    the plane reports what it would do and refuses to act — because "we could not
    decide" and "we decided no" are different, and only one of them is a reason to
    touch the registry.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import baseline as baseline_mod
from . import guard, paths, truth
from .actuators import SystemActuator, snapshot_change
from .admission import (ApplyResult, Gate, GatedActuator, Verdict,
                        evidence_breakdown)
from .change import Change, ChangeKind, EvidenceClass, Observability, Tier, change_key
from .ledger import Ledger, open_ledger
from .objective import KNOWN_METRICS, Objective, WorkloadKind

log = logging.getLogger("v12.plane")

#: Candidate samples as run_cycle accepts them: keyed by change_key (the only
#: shape that can support more than one MEASURED change), or one bare series,
#: which is attributed only when exactly one MEASURED candidate could use it.
CandidateSamples = Union[Mapping[str, Sequence[float]], Sequence[float]]

#: Guardrail samples per change: change_key -> {metric: (baseline, candidate)}.
#: Keyed only — a guardrail series is evidence about one change, exactly like a
#: candidate series, and an Objective that declares a guardrail denies a change
#: whose guardrail was never measured.
GuardrailSamples = Mapping[str, Mapping[str, Sequence[Sequence[float]]]]

#: MOS2026 ChangeType value -> V12 kind. A type not listed here is REFUSED at the
#: bridge, never defaulted: falling back to COMMAND would turn an unknown change
#: into an executable one.
MOS_KIND_MAP = {
    "env_var": ChangeKind.ENV_VAR, "registry": ChangeKind.REGISTRY,
    "power_plan": ChangeKind.POWER_PLAN, "defender_exclusion": ChangeKind.DEFENDER_EXCLUSION,
    "file_line": ChangeKind.FILE_LINE, "command": ChangeKind.COMMAND,
    "guided": ChangeKind.GUIDED,
}


@dataclass
class CycleResult:
    dry_run: bool = True
    objective_id: str = ""
    candidates: int = 0
    admitted: int = 0
    denied: int = 0
    applied: int = 0
    failed: int = 0
    verdicts: List[Verdict] = field(default_factory=list)
    results: List[ApplyResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    blocked: bool = False
    #: Jev process labels, display only (annotate_processes). Never in `notes`,
    #: which the cycle scans for blocking prefixes.
    annotations: List[str] = field(default_factory=list)

    def report(self) -> str:
        L = ["V12 cycle " + ("(DRY RUN — nothing was changed)" if self.dry_run else "(LIVE)"),
             "-" * 68,
             f"  objective : {self.objective_id or 'NONE — no objective loaded'}",
             f"  candidates: {self.candidates}",
             f"  admitted  : {self.admitted}   denied: {self.denied}",
             f"  applied   : {self.applied}   failed: {self.failed}"]
        if self.notes:
            L.append("")
            for n in self.notes:
                L.append(f"  note: {n}")
        if self.annotations:
            L.append("")
            for a in self.annotations:
                L.append(f"  {a}")
        if self.verdicts:
            L.append("")
            L.append("  decisions:")
            for v in self.verdicts:
                L.append("    " + v.describe())
                # MEASURED samples are keyed by change_key; show it so the
                # operator can supply them (--candidate-for KEY=SAMPLES), along
                # with the measurement that says when and where they were
                # taken (--candidate-measurement KEY=JSON). The samples alone
                # are not enough: a candidate with no session identity cannot
                # be shown to have been measured alongside its control.
                if v.evidence is EvidenceClass.MEASURED:
                    L.append(f"      key: {v.change_key}")
        return "\n".join(L)


# ---------------------------------------------------------------------------
# the built-in catalog
# ---------------------------------------------------------------------------
#: Levers V12 can carry itself, each with the evidence class and source it can
#: actually defend. Nothing here claims to be measured — because nothing here has
#: been measured on this machine yet, and saying otherwise is the failure mode the
#: whole system exists to prevent. They become MEASURED the day a baseline exists.
def builtin_catalog() -> List[Change]:
    cat: List[Change] = []

    cat.append(Change(
        id="nvml_pin_clocks",
        kind=ChangeKind.NVML_CLOCK,
        description="Pin GPU clocks (min==max) to remove boost-induced variance",
        target="gpu0",
        target_value=None,          # operator must choose; a range is ambiguous, a pin is not
        evidence=EvidenceClass.CITED,
        benefit_claim="Reduces run-to-run variance so measurements can conclude sooner",
        evidence_source="NVML nvmlDeviceSetGpuLockedClocks; requires Administrator; "
                        "does not persist across reboot",
        risk="low",
        extra={"needs_operator_value": "a (min,max) MHz pair, e.g. (1500,1500)"},
    ))

    cat.append(Change(
        id="nvml_power_limit",
        kind=ChangeKind.NVML_POWER_LIMIT,
        description="Raise the GPU power ceiling within the card's allowed range",
        target="gpu0",
        target_value=None,          # the operator (or the harness) chooses the watt value
        evidence=EvidenceClass.CITED,
        benefit_claim="More watts for a workload that is actually pinned at the cap. "
                      "Measured on this box 2026-11-04: fp16 GEMM at a 450 W ceiling "
                      "ran 159.8 TFLOPS with 439 W drawn and 2-6% of wall time at the "
                      "cap; at 600 W it ran 174.6 TFLOPS with 548 W drawn and 0% at "
                      "the cap (+9.3%). The same change does nothing for a Karma "
                      "render, which draws 104 W and is never near the cap — so this "
                      "lever is only worth applying to a workload the measurement "
                      "shows is power-bound.",
        evidence_source="NVML nvmlDeviceSetPowerManagementLimit; the getter is "
                        "trustworthy and the setter round-trips, so undo RESTORES the "
                        "operator's prior ceiling rather than resetting it. Requires "
                        "Administrator; does not persist across reboot.",
        risk="low",
        # The ceiling is the thing this change moves, so it must not invalidate
        # the baseline it is being tested against.
        varies=("power_limit_w",),
        extra={"needs_operator_value": "watts, e.g. 600 (this card allows 150-600 W)"},
    ))

    cat.append(Change(
        id="kernel_vellum_gemm",
        kind=ChangeKind.KERNEL_ARTIFACT,
        description="Compile a CUTLASS FP8 GEMM for the Vellum solver",
        target="vellum_gemm",
        evidence=EvidenceClass.DERIVED,
        benefit_claim="Roofline predicts a compute-bound GEMM benefits from a tuned kernel",
        evidence_source="v9.hwmodel roofline + measured 296.7 TFLOPS CUTLASS on this GPU",
        risk="low",
        extra={"name": "vellum_gemm_4096", "source": ""},
    ))

    return cat


def catalog_from_mos2026() -> List[Change]:
    """Pull MachineOptimizer2026's own verified catalog across the bridge.

    Its agents are the most-tested content in this project, and their claims have
    already survived contact with this machine. Rather than restate them here —
    which would let them drift — V12 imports them.
    """
    import sys
    out: List[Change] = []
    try:
        root = str(paths.REPO_ROOT / "MachineOptimizer2026")
        if root not in sys.path:
            sys.path.insert(0, root)
        from optimizer.base import ChangeType as MosType  # type: ignore
        from optimizer.hardware_profile import PROFILE  # type: ignore
        from optimizer import ui as mos_ui  # type: ignore
    except Exception as e:
        log.debug("MOS2026 catalog unavailable: %s", e)
        return out
    # The agents need a live Context, which the plane supplies at OBSERVE time;
    # enumeration happens in gather_candidates() where that context exists.
    return out


@dataclass(frozen=True)
class ProposalMeta:
    """What the MOS2026 Proposal that carried a change says about it: the text an
    operator reads in `v12 plan` and the draft tool shows Jev. Display metadata
    only. It lives in a side map keyed by (change_key, change.id) and is never
    copied into a Change field, so nothing the Gate or an actuator reads can
    depend on it."""
    title: str = ""
    domain: str = ""
    why: str = ""
    impact: str = ""
    applies_when: Tuple[str, ...] = ()
    note_2026: str = ""

    @classmethod
    def of(cls, prop) -> "ProposalMeta":
        return cls(
            title=str(getattr(prop, "title", "") or ""),
            domain=str(getattr(prop, "domain", "") or ""),
            why=str(getattr(prop, "why", "") or ""),
            impact=str(getattr(prop, "impact", "") or ""),
            applies_when=tuple(str(getattr(w, "value", w))
                               for w in (getattr(prop, "applies_when", None) or ())),
            note_2026=str(getattr(prop, "note_2026", "") or ""),
        )


#: Side map: (change_key, change.id) -> ProposalMeta.
CatalogMeta = Dict[Tuple[str, str], ProposalMeta]


def gather_catalog(notes: List[str]) -> Tuple[List[Change], CatalogMeta]:
    """The candidate catalog (builtin levers + MOS2026's agents) and the side map
    of proposal metadata, with no Observation, no snapshot, no preflight, no lock,
    no Gate and no ledger. Plane.gather_candidates adds the snapshots; `v12 plan`
    uses this directly so a read-only listing builds nothing that can admit.

    The metadata is read from the Proposal each surviving change sits in. MOS2026's
    Orchestrator._resolve (inside gather) only narrows `p.changes`; it never edits
    the title/domain/why/impact/applies_when/note_2026 read here, so this is the
    same text the proposal carried before _resolve ran. Collision rule: if two
    proposals deliver the same (change_key, id) with different metadata, the entry
    is dropped with a note - never a silent winner."""
    cands = list(builtin_catalog())
    n_builtin = len(cands)
    meta: CatalogMeta = {}
    collided: set = set()

    # MOS2026's agents, when reachable, are the larger verified catalog. Whether
    # it arrived is stated in the run report either way: a plane that silently
    # runs on its builtin levers looks exactly like one that ran on the catalog.
    imported: List[Change] = []
    refused: List[str] = []
    warned: List[str] = []          # MOS2026's own ui.warn lines, see below
    cfg_note = ""
    mos_ui = orig_warn = None
    try:
        import sys
        root = str(paths.REPO_ROOT / "MachineOptimizer2026")
        if root not in sys.path:
            sys.path.insert(0, root)
        from optimizer import orchestrator as mos_orch  # type: ignore
        from optimizer.probe import probe  # type: ignore
        from optimizer.base import WorkloadMode  # type: ignore
        # MOS2026 reports agent import and propose() failures only to its own
        # console via ui.warn. Tee them into the run report for the duration
        # of the import, or a half-loaded catalog reads as a full one.
        mos_ui = getattr(mos_orch, "ui", None)
        orig_warn = getattr(mos_ui, "warn", None)
        if orig_warn is not None:
            def _tee(msg, _orig=orig_warn):
                warned.append(str(msg))
                return _orig(msg)
            mos_ui.warn = _tee
        # MOS2026's own config (its DEFAULTS plus its own config.toml). The
        # Orchestrator reads cfg['min_confidence']; an empty dict KeyErrors.
        # That file decides which candidates V12 sees, so the report says so
        # (DESIGN_V12 §8: loading must be visible in the run report).
        cfg = mos_orch.load_config()
        orch = mos_orch.Orchestrator(cfg=cfg, dry_run=True)
        ctx = probe()                       # MOS2026's probe() takes no arguments
        ctx.workload = WorkloadMode.IDLE
        mos_root = Path(getattr(mos_orch, "ROOT", paths.MOS2026_DIR))
        toml = mos_root / "config.toml"
        agents = getattr(orch, "agents", None)
        cfg_note = (f"MOS2026 catalog filter: min_confidence={cfg.get('min_confidence')!r} "
                    f"(MOS2026 DEFAULTS, overlaid by {toml}: "
                    f"{'present' if toml.exists() else 'absent'}); workload=IDLE "
                    f"regardless of the objective"
                    + (f"; {len(agents)} agent(s) discovered" if agents is not None
                       else ""))
        for prop in orch.gather(ctx):
            prop_admin = bool(getattr(prop, "requires_admin", False))
            pm = ProposalMeta.of(prop)
            for mc in getattr(prop, "changes", []):
                try:
                    c = Plane._from_mos_change(mc, proposal_requires_admin=prop_admin)
                except ValueError as e:
                    refused.append(f"'{getattr(mc, 'id', '?')}': {e}")
                    continue
                imported.append(c)
                k = (change_key(c), c.id)
                if k in collided:
                    continue
                if k in meta and meta[k] != pm:
                    del meta[k]
                    collided.add(k)
                    continue
                meta[k] = pm
    except Exception as e:
        log.info("MOS2026 agent catalog not used: %s: %s", type(e).__name__, e)
        notes.append(f"MOS2026 catalog NOT loaded ({type(e).__name__}: {e}) — "
                     f"running on the {n_builtin} builtin changes only")
        meta = {}
    else:
        cands += imported
        notes.append(f"MOS2026 catalog loaded: {len(imported)} change(s)"
                     + (f", {len(refused)} refused" if refused else ""))
        notes.append(cfg_note)
        for why in refused:
            notes.append(f"refused MOS2026 change {why}")
        for key, cid in sorted(collided):
            notes.append(f"catalog metadata for '{cid}' ({key}) came from more than one "
                         f"proposal with different text — dropped, shown untitled")
    finally:
        if orig_warn is not None:
            mos_ui.warn = orig_warn
    for w in warned:
        notes.append(f"MOS2026 warned: {w}")
    return cands, meta


@dataclass
class Observation:
    gpu: truth.GpuTruth = field(default_factory=truth.gpu_truth)
    roofline: truth.RooflineTruth = field(default_factory=truth.roofline)
    conditions: Optional[baseline_mod.Conditions] = None
    gpu_apps: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: 'jev (not evidence): ...' lines. Display only: nothing reads them back,
    #: and gpu_apps / notes / conditions are fixed before they are asked for.
    annotations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# process annotations (Jev, display only)
# ---------------------------------------------------------------------------
#: Every line a Jev process label prints under. Says what it is on its face.
ANNOTATION_PREFIX = "jev (not evidence):"
#: Rows asked about per call, before the per-process question cap in jev.
MAX_ANNOTATED_PROCESSES = 16

#: What a python.exe the deterministic rules could not resolve might be running.
#: 'none' is jev's required no-match label; 'cannot_tell' is the honest answer to
#: a command line that says too little.
PROCESS_KINDS = {
    "comfyui_server": "a ComfyUI server (main.py of a ComfyUI install)",
    "diffusion_or_ml_job": "a diffusion, training or inference job (torch, diffusers, "
                           "a train.py or infer.py)",
    "houdini_python_tool": "a Houdini-side Python tool (PDG/TOPs worker, SideFX "
                           "script or utility)",
    "unreal_tool": "an Unreal Engine Python tool or build helper",
    "web_or_dev_server": "a web, API or development server (uvicorn, flask, "
                         "http.server, an MCP server)",
    "ide_or_language_server": "an IDE helper, debugger or language server",
    "general_utility": "a general utility or one-off script (downloader, file tool, "
                       "installer, test runner)",
    "cannot_tell": "the command line does not say enough to tell",
    "none": "None of these",
}
_PROCESS_INSTRUCTIONS = ("`image_name`, `executable_path` and `command_line` describe "
                         "one running Python process. Judging only from `command_line` "
                         "and `executable_path`, what kind of program is it?")


def annotate_processes(table, *, ledger: Optional[Ledger] = None) -> List[str]:
    """Jev labels for the python.exe rows the deterministic rules left
    unresolved (guard.unresolved_python), as printable lines. [] when Jev is off,
    when there is nothing unresolved, or on any failure.

    Callers run this AFTER the deterministic block decision and skip it when
    blocked. Its output goes to display only — a label never moves a python.exe
    in or out of gpu_apps, a collision, a note or `ok`. jev is imported here and
    nowhere else in this module (tests/test_invariants.py)."""
    try:
        rows = guard.unresolved_python(table)
        if not rows:
            return []
        from . import jev
        if not jev.switch_on():
            return []
        q = jev.Question.choice("process_kind", _PROCESS_INSTRUCTIONS, PROCESS_KINDS)
        out: List[str] = []
        for row in rows[:MAX_ANNOTATED_PROCESSES]:
            # Only the python row's own fields. pid and start time bind the label
            # (subject_key) but stay out of the state, so the same script under
            # two pids is one cached question.
            state = {"image_name": row.name, "executable_path": row.exe or "",
                     "command_line": row.cmdline or ""}
            subject = f"process:{row.pid}@{row.created or 'unknown'}"
            js = jev.judge("observe", subject, state, [q], consumed_as="display",
                           ledger=ledger)
            if not js:
                break
            j = js[0]
            out.append(jev.sanitize_for_terminal(
                f"{ANNOTATION_PREFIX} {row.name} pid {row.pid} "
                f"(started {row.created or 'unknown'}): {j.display()[len('jev: '):]}"))
            if j.outcome in ("offline", "error"):
                break               # the rest would fail the same way, 3 s each
        skipped = len(rows) - MAX_ANNOTATED_PROCESSES
        if skipped > 0:
            out.append(f"{ANNOTATION_PREFIX} {skipped} more python process(es) "
                       f"not asked (limit {MAX_ANNOTATED_PROCESSES})")
        return out
    except Exception:  # noqa: BLE001 — an annotation must never break a run
        log.debug("process annotation failed", exc_info=True)
        return []


class Plane:
    """The control plane. Owns the objective, the gate, the ledger, the guards."""

    def __init__(self, objective: Optional[Objective] = None,
                 dry_run: bool = True,
                 allow_destructive_reset: bool = False,
                 require_quiet: bool = False,
                 ledger: Optional[Ledger] = None):
        self.objective = objective
        self.dry_run = dry_run
        self.gate = Gate(objective=objective,
                         allow_destructive_reset=allow_destructive_reset)
        self.ledger = ledger or open_ledger()
        self.require_quiet = require_quiet
        self.sys_actuator = GatedActuator(SystemActuator(dry_run=dry_run), Tier.SYSTEM)
        self._lock = guard.SingleInstance()
        #: (change_key, change.id) -> ProposalMeta, filled by gather_candidates.
        #: Display metadata; never read by the Gate or an actuator.
        self.proposal_meta: CatalogMeta = {}

    # ---- stage 1: OBSERVE -------------------------------------------------
    def observe(self) -> Observation:
        o = Observation()
        # One process-table read for everything OBSERVE derives from it, so the
        # collision verdict, gpu_apps and the baseline conditions all describe the
        # same instant.
        table = guard._process_table()
        ok, collisions, notes = guard.preflight(require_quiet=self.require_quiet,
                                                table=table)
        o.notes += notes
        # Informational, in both modes. Was built from collisions, which carry
        # gpu-app entries only under require_quiet — so with Houdini resident a
        # normal run reported gpu_apps=[]. Blocking is still preflight's call; an
        # unreadable table leaves this [] and the unreadable note says why.
        o.gpu_apps = (sorted({c.name for c in guard.find_gpu_apps(table)})
                      if table is not None else [])
        if collisions:
            for c in collisions:
                if c.kind == "optimizer":
                    o.notes.append(f"COMPETING OPTIMIZER: {c.name} (pid {c.pid})")
        if self.objective:
            o.conditions = baseline_mod.Conditions.capture(
                self.objective, sampler_available=True, table=table)
        # Last, and only when preflight cleared: everything above is decided
        # before any label exists, so no label can reach it.
        if ok:
            o.annotations = annotate_processes(table, ledger=self.ledger)
        log.info("OBSERVE: %s", o.gpu.summary)
        return o

    # ---- stage 3: DECIDE --------------------------------------------------
    def gather_candidates(self, obs: Observation) -> List[Change]:
        """Assemble candidates and take their priors.

        The snapshot must happen HERE, before ADMIT, because the Gate checks
        whether a prior was recorded and refuses a restorable target without one.
        That check exists so a rollback can never silently delete a value we never
        captured. The proposal metadata side map (gather_catalog) is kept on
        self.proposal_meta for display; it never enters a Change.
        """
        cands, self.proposal_meta = gather_catalog(obs.notes)

        for c in cands:
            if c.snapshot is None:
                c.snapshot = snapshot_change(c)
        return cands

    @staticmethod
    def _from_mos_change(mc, proposal_requires_admin: bool = False) -> Change:
        """Translate one MOS2026 Change. Raises ValueError for a type V12 does not
        know; the caller skips it with a visible note.

        MOS2026's admin flag (on the change or its Proposal) is carried across.
        V12's ADMIN_KINDS does not cover COMMAND or ENV_VAR, so without this an
        admin-only command (an HKLM write, Add-MpPreference) would reach the ledger
        and MOS2026's executor marked as not needing elevation."""
        kt = getattr(mc.type, "value", str(mc.type))
        kind = MOS_KIND_MAP.get(kt)
        if kind is None:
            raise ValueError(f"unknown change type {kt!r} — not mapped to any V12 kind, "
                             f"so it is refused rather than run as a command")
        risk = getattr(mc.risk, "value", str(getattr(mc, "risk", "low")))
        extra = dict(getattr(mc, "extra", {}) or {})
        return Change(
            id=mc.id, kind=kind, description=mc.description, target=mc.target,
            target_value=mc.target_value, reg_value_name=mc.reg_value_name,
            reg_type=mc.reg_type, scope=getattr(mc.scope, "value", "user"),
            file_key=mc.file_key, file_sep=getattr(mc, "file_sep", "="),
            # MOS2026 carries the INI section in extra; without it two cvars with
            # the same key in different sections would share one change_key.
            section=extra.get("section"),
            snapshot_cmd=mc.snapshot_cmd, apply_cmd=mc.apply_cmd,
            undo_cmd=mc.undo_cmd, verify_cmd=mc.verify_cmd, risk=risk,
            # Imported from MOS2026's verified catalog: cited, with the catalog as
            # the source. Not measured here — V12 has not measured it.
            evidence=(EvidenceClass.GUIDED if kind is ChangeKind.GUIDED
                      else EvidenceClass.CITED),
            evidence_source="MachineOptimizer2026 verified catalog",
            benefit_claim=getattr(mc, "description", ""),
            # Change.__post_init__ only ever raises this to True (ADMIN_KINDS), so
            # an explicit True here survives.
            requires_admin=(bool(getattr(mc, "requires_admin", False))
                            or bool(proposal_requires_admin)),
            extra=extra,
        )

    @staticmethod
    def _attribute_samples(cands: Sequence[Change],
                           samples: Optional["CandidateSamples"],
                           notes: List[str]) -> Dict[str, List[float]]:
        """Decide which candidate each sample series belongs to, by change_key.

        A series is evidence about ONE change. Handing one series to every
        candidate would let a single measurement admit every MEASURED change in
        the batch, so anything that cannot be attributed unambiguously goes to
        nobody, and a MEASURED change with no samples of its own is denied.
        """
        if samples is None:
            return {}
        counts: Dict[str, int] = {}
        for c in cands:
            k = change_key(c)
            counts[k] = counts.get(k, 0) + 1
        out: Dict[str, List[float]] = {}

        if isinstance(samples, Mapping):
            for k, series in samples.items():
                n = counts.get(k, 0)
                if n == 0:
                    notes.append(f"candidate samples for '{k}' match no candidate — ignored")
                elif n > 1:
                    notes.append(f"candidate samples for '{k}' match {n} candidates — "
                                 f"cannot attribute them, so none receive them")
                else:
                    out[k] = list(series)
            return out

        if isinstance(samples, (str, bytes)):
            notes.append("candidate samples were text, not numbers — ignored")
            return {}
        measured = [c for c in cands if c.evidence is EvidenceClass.MEASURED]
        if len(measured) == 1 and counts[change_key(measured[0])] == 1:
            k = change_key(measured[0])
            out[k] = list(samples)
            notes.append(f"candidate samples attributed to the only MEASURED "
                         f"candidate, {measured[0].id} ({k})")
        elif not measured:
            notes.append("candidate samples supplied but no candidate is MEASURED — ignored")
        else:
            notes.append(f"one unkeyed sample series cannot be attributed among "
                         f"{len(measured)} MEASURED candidates — none of them receive it; "
                         f"supply samples per change_key (--candidate-for KEY=SAMPLES)")
        return out

    @staticmethod
    def _attribute_guardrails(cands: Sequence[Change],
                              samples: Optional[GuardrailSamples],
                              notes: List[str]) -> Dict[str, Dict[str, Sequence[Sequence[float]]]]:
        """Guardrail samples by change_key, with the same refusal rules as
        candidate samples: a key that matches no candidate, or more than one, goes
        to nobody."""
        if not samples:
            return {}
        counts: Dict[str, int] = {}
        for c in cands:
            k = change_key(c)
            counts[k] = counts.get(k, 0) + 1
        out: Dict[str, Dict[str, Sequence[Sequence[float]]]] = {}
        for k, by_metric in samples.items():
            n = counts.get(k, 0)
            if n != 1:
                notes.append(f"guardrail samples for '{k}' match {n} candidates — "
                             f"none receive them")
                continue
            out[k] = dict(by_metric)
        return out

    @staticmethod
    def _attribute_measurements(cands: Sequence[Change],
                                measurements: Optional[Mapping[str, baseline_mod.Measurement]],
                                notes: List[str]) -> Dict[str, baseline_mod.Measurement]:
        """Decide which candidate each measurement record belongs to, by change_key.

        Same rule as the samples: a measurement describes ONE change, so one
        that matches no candidate, or more than one, goes to nobody. Getting
        this wrong would let a single measurement's provenance authorise a
        comparison it was never part of."""
        if not measurements:
            return {}
        counts: Dict[str, int] = {}
        for c in cands:
            k = change_key(c)
            counts[k] = counts.get(k, 0) + 1
        out: Dict[str, baseline_mod.Measurement] = {}
        for k, m in measurements.items():
            n = counts.get(k, 0)
            if n == 0:
                notes.append(f"candidate measurement for '{k}' matches no candidate — ignored")
            elif n > 1:
                notes.append(f"candidate measurement for '{k}' matches {n} candidates — "
                             f"cannot attribute it, so none receive it")
            else:
                out[k] = m
        return out

    # ---- the loop ---------------------------------------------------------
    def run_cycle(self, candidates: Optional[Sequence[Change]] = None,
                  baseline_for_objective: Optional[baseline_mod.Baseline] = None,
                  candidate_samples: Optional[CandidateSamples] = None,
                  guardrail_samples: Optional[GuardrailSamples] = None,
                  candidate_measurements: Optional[Mapping[str, baseline_mod.Measurement]] = None
                  ) -> CycleResult:
        """One pass of the loop.

        `candidate_samples` is keyed by change_key. A bare series is accepted only
        when exactly one MEASURED candidate could use it; otherwise no candidate
        receives it (see _attribute_samples). `guardrail_samples` is keyed by
        change_key only; without it, an Objective that declares guardrails admits
        no MEASURED change.

        `candidate_measurements` is keyed the same way and carries the
        PROVENANCE of each candidate measurement: when it was measured, in which
        session, under what conditions, and the per-pair differences against its
        control. It is what lets the Gate refuse a comparison it cannot honestly
        make — a bare series of floats carries none of it, and so cannot be
        shown to describe the same machine state as the baseline."""
        r = CycleResult(dry_run=self.dry_run)
        r.objective_id = (f"{self.objective.workload.kind.value}:{self.objective.workload.name}"
                          if self.objective else "")

        ok, msg = self._lock.acquire() if not self.dry_run else (True, "dry-run: no lock taken")
        r.notes.append(msg)
        if not ok:
            r.blocked = True
            return r                # nothing acquired, nothing started

        self.ledger.start(note="dry-run" if self.dry_run else "live")
        # Every path out of here — the blocked returns, an exception mid-cycle, the
        # normal end — closes the session and releases the lock. Was: a blocked
        # live cycle returned holding both, so the next run in the same process
        # found its own stale lock and the ledger session never ended.
        try:
            self._cycle(r, candidates, baseline_for_objective,
                        candidate_samples, guardrail_samples,
                        candidate_measurements)
        finally:
            try:
                self.ledger.end()
            finally:
                self._lock.release()    # a no-op when it was never acquired
        return r

    def _cycle(self, r: CycleResult,
               candidates: Optional[Sequence[Change]],
               baseline_for_objective: Optional[baseline_mod.Baseline],
               candidate_samples: Optional[CandidateSamples],
               guardrail_samples: Optional[GuardrailSamples],
               candidate_measurements: Optional[Mapping[str, baseline_mod.Measurement]] = None
               ) -> None:
        """OBSERVE through REMEMBER. Fills `r`; run_cycle owns the lock and the
        ledger session around it."""
        obs = self.observe()
        r.notes += obs.notes
        r.annotations += obs.annotations

        if any(n.startswith("COMPETING OPTIMIZER") for n in obs.notes) and not self.dry_run:
            r.blocked = True
            r.notes.append("refusing to act while another optimizer is running — no "
                           "cross-process GPU partitioning exists on this hardware, so "
                           "both would be measuring a machine the other is changing")
            return
        if (any(n.startswith(guard.PROCESS_TABLE_UNREADABLE) for n in obs.notes)
                and not self.dry_run):
            r.blocked = True
            r.notes.append("refusing to act without a readable process table — another "
                           "optimizer could be running and preflight could not see it")
            return

        n0 = len(obs.notes)
        cands = list(candidates) if candidates is not None else self.gather_candidates(obs)
        r.notes += obs.notes[n0:]
        r.candidates = len(cands)

        if self.objective is None:
            # Plane never builds an opted-in Gate and the CLI has no flag for it;
            # the first branch is reachable only when code swaps in its own Gate.
            cited = ("CITED changes are admitted UNVERIFIED only because this Gate was "
                     "explicitly opted in" if getattr(self.gate, "allow_unverified", False)
                     else "CITED changes are recorded, never applied")
            r.notes.append("no Objective loaded — MEASURED and DERIVED changes are refused "
                           f"at ADMISSION, {cited}, GUIDED changes are surfaced only; "
                           "V12 will not optimize a machine it cannot measure")

        per_change = self._attribute_samples(cands, candidate_samples, r.notes)
        per_guard = self._attribute_guardrails(cands, guardrail_samples, r.notes)
        # The provenance of each candidate measurement, attributed by the same
        # rule as the samples themselves: a measurement describes ONE change, so
        # one that cannot be attributed unambiguously goes to nobody.
        per_measurement = self._attribute_measurements(cands, candidate_measurements,
                                                       r.notes)

        for c in cands:
            v = self.gate.evaluate(
                c,
                baseline=baseline_for_objective,
                candidate_samples=per_change.get(change_key(c)),
                guardrail_samples=per_guard.get(change_key(c)),
                now_conditions=obs.conditions,
                measurement=per_measurement.get(change_key(c)),
            )
            self.ledger.record_verdict(v, c)
            r.verdicts.append(v)
            if not v.admitted:
                r.denied += 1
                continue
            r.admitted += 1

            if self.dry_run:
                continue

            actuator = self._actuator_for(c)
            try:
                res = actuator.apply(c, verdict=v)
            except Exception as e:
                res = ApplyResult(change_id=c.id, applied=False, error=f"{type(e).__name__}: {e}")
            r.results.append(res)
            if res.applied:
                r.applied += 1
                c.applied = True
                self.ledger.record_applied(
                    c, v,
                    landed=res.verified,
                    landed_source=res.verification_source,
                    effect=v.detail.get("effect"),
                    p_value=v.detail.get("p_value"),
                    detail=res.detail,
                )
            else:
                r.failed += 1

    def _actuator_for(self, c: Change) -> GatedActuator:
        if c.tier is Tier.SYSTEM:
            return self.sys_actuator
        if c.tier is Tier.KERNEL:
            from .actuators import KernelActuator
            return GatedActuator(KernelActuator(dry_run=self.dry_run), Tier.KERNEL)
        raise NotImplementedError(
            f"no actuator for {c.tier.value} tier yet — APP_GRAPH changes are not "
            f"implemented in this build, and pretending otherwise would be worse "
            f"than refusing")

    # ---- rollback ---------------------------------------------------------
    def revert(self, changes: Sequence[Change]) -> List[ApplyResult]:
        """Undo. Deliberately takes no verdict — the safety path must not depend on
        evidence that may be exactly what failed."""
        out = []
        for c in changes:
            actuator = self._actuator_for(c)
            res = actuator.rollback(c)
            self.ledger.record_reverted(c, c.rollback_verb.value)
            out.append(res)
        return out


def status() -> str:
    """Everything an operator needs to know before running anything."""
    g = truth.gpu_truth()
    rl = truth.roofline()
    lines = [
        "V12 status",
        "-" * 68,
        paths.describe(),
        g.summary,
        rl.summary,
    ]
    if g.errors:
        lines.append("  could not read: " + "; ".join(g.errors))
    if g.hags_enabled is not None:
        lines.append(f"  HAGS: {'ON' if g.hags_enabled else 'OFF'}")
    table = guard._process_table()
    ok, collisions, notes = guard.preflight(table=table)
    lines.append(f"  preflight: {'CLEAR' if ok else 'BLOCKED'}")
    for n in notes:
        lines.append(f"    {n}")
    if ok:      # after the decision, never instead of it
        for a in annotate_processes(table):
            lines.append(f"    {a}")
    lines.append("")
    lines.append(baseline_mod.describe())
    return "\n".join(lines)
