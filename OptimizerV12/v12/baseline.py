"""
V12 — Baseline, and the conditions under which it means anything.

A baseline without its conditions is a number that used to be true on a different
machine. Two concrete ways that has already burned this project:

  * V9 recorded TensorRT results with `{"harness": "matmul"}` and later read them
    as if they described a diffusion UNet. The number was real; the workload
    attribution was not, and nothing in the record could catch it.
  * Houdini was running on this box during V12's own recon (PID 56920, holding
    2.84 GB of GPU). A frame-time baseline taken with Houdini resident is not
    comparable to one taken without it, because on this hardware — where there is
    no cross-process GPU partitioning of any kind, no MIG, no MPS, no TCC — the
    two processes are time-sliced against each other. The baseline would drift
    with no code change at all.

So a Baseline here carries its Conditions, and `valid_for()` is consulted BEFORE
any comparison. A stale baseline is rejected with a reason a human can read, never
silently reused.
"""
from __future__ import annotations

import enum
import json
import logging
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import paths, truth
from .objective import Objective, Workload, WorkloadKind, welch_t_test, _mean

log = logging.getLogger("v12.baseline")


#: A process committing at least this much GPU memory counts as competing for
#: the GPU. Read off this box (2026-10-01, `\GPU Process Memory(*)\Total
#: Committed`, summed per pid): ordinary desktop apps commit 150-700 MiB —
#: Spotify 488, Edge 430, obs64 330, claude.exe 701, EpicGamesLauncher 662 —
#: because the counter includes shared system memory, so Electron apps look
#: large. Houdini with a scene loaded committed 4962 MiB and dwm (the XDR's
#: compositor) 2706. At 512 the set flickers as Claude or Epic is opened, which
#: is the defect this replaced at a smaller size; 1024 separates the two groups.
HEAVY_GPU_COMMIT_MIB = 1024

#: Recorded in Conditions so a baseline states how its process set was read.
#: Baselines from the nvidia-smi name list carry no method (""), are unread on
#: that field, and are refused rather than compared across two definitions.
#: v2 marks heavy entries (`name:heavy`, `name:heavy*N`); v1 named a heavy and an
#: idle python.exe alike, so v1 baselines are refused, not compared.
COMPETING_METHOD = f"gpu-commit>={HEAVY_GPU_COMMIT_MIB}MiB+significant-apps/heavy-count-v2"

_GPU_COMMIT_COUNTER = r"\GPU Process Memory(*)\Total Committed"


def _gpu_commit_by_pid() -> Optional[Dict[int, float]]:
    """{pid: committed GPU memory in MiB}, summed across adapters, or None when
    the counter cannot be read.

    Instances are named pid_<n>_luid_<hi>_<lo>_phys_<k>; this box has two LUIDs,
    and one process can appear under both, so the per-pid figure is a sum. The
    English counter path is used so a localized Windows still resolves it.
    """
    try:
        import win32pdh  # type: ignore
    except Exception as e:
        log.info("GPU process counter unavailable (win32pdh: %s)", e)
        return None
    q = None
    try:
        q = win32pdh.OpenQuery()
        add = getattr(win32pdh, "AddEnglishCounter", win32pdh.AddCounter)
        h = add(q, _GPU_COMMIT_COUNTER)
        win32pdh.CollectQueryData(q)
        items = win32pdh.GetFormattedCounterArray(h, win32pdh.PDH_FMT_LARGE)
    except Exception as e:
        log.info("GPU process counter unreadable: %s: %s", type(e).__name__, e)
        return None
    finally:
        if q is not None:
            try:
                win32pdh.CloseQuery(q)
            except Exception:
                pass
    out: Dict[int, float] = {}
    for inst, val in (items or {}).items():
        parts = str(inst).split("_")
        if len(parts) < 2 or parts[0] != "pid":
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        out[pid] = out.get(pid, 0.0) + float(val) / 2**20
    return out


def competing_gpu_processes(table: Any = None) -> Optional[List[str]]:
    """Entries naming the OTHER processes that matter to a GPU timing: those
    committing at least HEAVY_GPU_COMMIT_MIB, plus any process whose image is in
    guard.SIGNIFICANT_APPS whatever its size.

    An entry is a lowercase image name, `name:heavy` for one heavy pid of that
    image, or `name:heavy*N` for N of them. A significant app below the threshold
    is named bare, once, however many idle copies run (23 stray python.exe on
    this box). Without the marker a ComfyUI python.exe holding 15 GB read the
    same as an idle one, and a second heavy Houdini read the same as one.

    Returns None — UNREAD, which never matches — when either the GPU memory
    counter or the process table cannot be read. [] is a reading: nothing heavy
    is resident. The two are kept apart because a failed read reported as [] is
    how two blind captures used to agree with each other.

    Was: every name from `nvidia-smi --query-compute-apps`. Under WDDM that is
    ~40 ordinary desktop apps (chrome, Spotify, explorer, Teams...), compared as
    an exact list, so opening any of them invalidated every baseline — and a
    failed nvidia-smi returned [], indistinguishable from an idle machine. There
    is deliberately no fallback to it.

    `table` is a process table the caller already read (guard._process_table()
    form); None reads one here.
    """
    from . import guard
    commit = _gpu_commit_by_pid()
    if commit is None:
        return None
    if table is None:
        table = guard._process_table()
    if table is None:
        return None                 # cannot put a name to any pid
    me = os.getpid()
    names = set()
    heavy: Dict[str, int] = {}
    for pid, name in table:
        if pid == me:
            continue                # V12's own interpreter is not a competitor
        low = name.lower()
        if commit.get(pid, 0.0) >= HEAVY_GPU_COMMIT_MIB:
            heavy[low] = heavy.get(low, 0) + 1
        elif name in guard.SIGNIFICANT_APPS:
            names.add(low)
    names.update(f"{n}:heavy" if k == 1 else f"{n}:heavy*{k}" for n, k in heavy.items())
    # A heavy pid absent from the table exited between the two reads: it is not
    # competing any more, so it is not named.
    return sorted(names)


def _jsonable(v: Any) -> Any:
    """Enums -> their value, recursively. Everything else is left as it is."""
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


@dataclass
class Conditions:
    """Everything that must match for two measurements to be comparable."""
    driver_version: str = ""
    sm_clock_max_mhz: int = 0
    mem_clock_max_mhz: int = 0
    power_limit_w: float = 0.0
    hags_enabled: Optional[bool] = None
    workload_fingerprint: str = ""
    workload_kind: str = ""
    instant_power_sensor: bool = True
    #: Heavy/significant GPU processes as competing_gpu_processes() names them
    #: (`name`, `name:heavy`, `name:heavy*N`), compared as a set. None = UNREAD (the counter or process table could not be read); []
    #: = read, and nothing is competing. See competing_gpu_processes().
    competing_processes: Optional[List[str]] = None
    #: How competing_processes was read (COMPETING_METHOD). "" = unread, which
    #: is what every baseline from the old nvidia-smi list carries.
    competing_method: str = ""

    #: Fields whose difference MUST invalidate a baseline. Deliberately excludes
    #: nothing subtle: each of these changes the machine's behaviour, not just
    #: its description.
    CRITICAL = (
        "driver_version",
        "sm_clock_max_mhz",
        "mem_clock_max_mhz",
        "power_limit_w",
        "hags_enabled",
        "workload_fingerprint",
        "competing_method",
        "competing_processes",
    )

    @classmethod
    def capture(cls, objective: Objective, sampler_available: bool = True,
                table: Any = None) -> "Conditions":
        """`table`: a process table the caller already read, so the conditions
        describe the same instant as its preflight. None reads one."""
        t = truth.gpu_truth()
        return cls(
            driver_version=t.driver_version,
            sm_clock_max_mhz=t.clock_sm_max_mhz,
            # The MAXIMUM memory clock, never `clock_mem_live_mhz`: an idle 4090
            # reads ~405 MHz live, so a live clock here would invalidate every
            # baseline on whether the card happened to be idle when sampled.
            mem_clock_max_mhz=int(t.clock_mem_max_mhz or 0),
            power_limit_w=round(t.power_limit_w, 1),
            hags_enabled=t.hags_enabled,
            workload_fingerprint=objective.workload.fingerprint,
            workload_kind=objective.workload.kind.value,
            instant_power_sensor=sampler_available,
            competing_processes=competing_gpu_processes(table),
            competing_method=COMPETING_METHOD,
        )

    #: CRITICAL fields read off the hardware. Each has an "unread" value (0, "",
    #: None) that two failed reads share, so equality alone would let a baseline
    #: match the machine only because the same read failed both times.
    #: `hags_enabled` and `competing_processes` are unread only when None:
    #: False and [] are real answers ("off", "nobody else").
    READ_FIELDS = (
        "driver_version",
        "sm_clock_max_mhz",
        "mem_clock_max_mhz",
        "power_limit_w",
        "hags_enabled",
        "competing_method",
        "competing_processes",
    )
    _NONE_IS_UNREAD = ("hags_enabled", "competing_processes")

    def unread(self) -> List[str]:
        """READ_FIELDS that hold no reading. hags_enabled=False and
        competing_processes=[] are readings."""
        return [k for k in self.READ_FIELDS
                if (getattr(self, k) is None if k in self._NONE_IS_UNREAD
                    else not getattr(self, k))]

    def differences(self, other: "Conditions",
                    ignore: Sequence[str] = ()) -> List[str]:
        """What changed between two readings of the machine.

        `ignore` names the fields the caller is DELIBERATELY varying — the ones
        a Change under test moves on purpose. Everything else still has to
        match, so a driver update or a new competing process still invalidates
        a baseline.

        This is a parameter rather than a policy because the power-ceiling
        lever needs it: raising the ceiling from 450 W to 600 W and then asking
        whether the old baseline is still applicable must not fail merely
        because the change under test was applied. Every other field still
        disqualifies the comparison, so the check keeps its teeth.
        """
        out = []
        skip = set(ignore or ())
        # Unread is never a match: "we could not tell" on either side means the
        # two machines cannot be shown to be the same.
        unread_here, unread_there = set(self.unread()), set(other.unread())
        for k in self.CRITICAL:
            if k in skip:
                continue
            a, b = getattr(self, k), getattr(other, k)
            if k in unread_here or k in unread_there:
                side = ("both" if k in unread_here and k in unread_there
                        else "baseline" if k in unread_here else "now")
                out.append(f"{k}: unread ({side})")
                continue
            if k == "competing_processes":
                # A set of image names: order and duplicates carry nothing.
                # Lowercased so a list stored by hand still compares by identity.
                sa, sb = {str(x).lower() for x in a}, {str(x).lower() for x in b}
                if sa != sb:
                    out.append(f"{k}: {sorted(sa)!r} -> {sorted(sb)!r}")
                continue
            if a == b:
                continue
            if k == "power_limit_w" and abs(float(a or 0) - float(b or 0)) < 5.0:
                continue          # within driver quantisation; not a real change
            if k in ("sm_clock_max_mhz", "mem_clock_max_mhz") and abs(int(a or 0) - int(b or 0)) <= 15:
                continue          # boost-bin jitter
            out.append(f"{k}: {a!r} -> {b!r}")
        return out


@dataclass
class Measurement:
    """The candidate side of a comparison, with the evidence of WHEN and WHERE
    it was measured.

    Why this exists
    ---------------
    The 2026-11-03 ghost was not a statistics bug. It was a bookkeeping bug:
    the candidate arrived at the gate as a bare list of five floats, carrying no
    timestamp, no conditions and no session identity, and the gate had no way to
    ask whether it had been measured alongside the baseline or 48 minutes and a
    different set of resident processes away from it. Welch's t-test on two
    INDEPENDENT series cannot tell drift from effect — that is what
    "independent" means — so the drift was charged to the candidate and came
    back ADMIT +12.34%, p=1.882e-05.

    A bare list of floats is not a measurement. It is a number with no
    provenance, and a comparison against a number with no provenance cannot be
    shown to describe one machine state.

    So the candidate carries its provenance, and the gate refuses when it is
    missing. Absence of evidence is not evidence of sameness.
    """

    #: When the candidate samples were measured (epoch seconds). 0.0 = unknown,
    #: which is NOT the same as "now" — an unknown time cannot be compared with
    #: a known one, and is refused.
    measured_at: float = 0.0
    #: The measurement session this candidate belongs to. Two measurements with
    #: the same session id were taken as one experiment.
    session_id: str = ""
    #: The machine conditions at the moment the candidate was measured. Compared
    #: against the baseline's own conditions when present, so "same machine
    #: state" is a fact rather than an assumption.
    #:
    #: Optional on its own, because the session id and the timestamp are the
    #: primary evidence that the two arms are one experiment — and because a
    #: comparison whose conditions were never recorded is still judged, by the
    #: baseline-versus-now comparison `valid_for` already makes.
    conditions: Dict[str, Any] = field(default_factory=dict)
    #: The per-pair differences, candidate[i] - baseline[i], when the two arms
    #: were measured ALTERNATELY in one session. This is what makes the
    #: comparison a paired one: the drift is common to both arms and cancels
    #: in the difference.
    paired_diffs: Optional[List[float]] = None
    #: Where the numbers came from, for the record ("husk_frame", "comfy_run").
    harness: str = ""

    def to_extra(self) -> Dict[str, Any]:
        """The JSON form, for carrying on a Change's `extra` dict."""
        out: Dict[str, Any] = {
            "measured_at": float(self.measured_at),
            "session_id": self.session_id,
            "conditions": _jsonable(dict(self.conditions)),
            "harness": self.harness,
        }
        if self.paired_diffs is not None:
            out["paired_diffs"] = [float(d) for d in self.paired_diffs]
        return out

    @classmethod
    def from_extra(cls, extra: Optional[Dict[str, Any]]) -> Optional["Measurement"]:
        """Read a Measurement back out of a Change's `extra` dict.

        Returns None when there is nothing to read — which the caller must
        treat as "this candidate carries no provenance", not as "the
        provenance says everything is fine".
        """
        if not extra:
            return None
        raw = extra.get("measurement")
        if not isinstance(raw, dict):
            return None
        diffs = raw.get("paired_diffs")
        return cls(
            measured_at=float(raw.get("measured_at") or 0.0),
            session_id=str(raw.get("session_id") or ""),
            conditions=dict(raw.get("conditions") or {}),
            paired_diffs=([float(d) for d in diffs]
                          if isinstance(diffs, (list, tuple)) else None),
            harness=str(raw.get("harness") or ""),
        )


#: How far apart two measurements may be and still be called one session.
#:
#: This is a COARSE backstop, not the real check — the real check is the
#: session id and the conditions, which say "same machine state" directly. The
#: age bound exists because a session id can be missing (an old baseline on
#: disk, recorded before this field existed), and a missing session id must
#: not become a free pass.
#:
#: The number is generous on purpose. A paired render campaign measures 5
#: repetitions of 2 arms at ~48 s each, plus a warm-up, so 15 minutes is
#: about three times a real session. It is far below the 48 minutes that
#: separated the ghost's candidate from its baseline, and far below the
#: multi-day gap of a baseline someone forgot about.
MAX_SESSION_GAP_S = 15 * 60.0


@dataclass
class Baseline:
    """A measured reference. `samples` is the raw series, kept, because a mean
    without its spread cannot be tested against anything."""
    objective_id: str
    workload: Dict[str, Any]
    metric: str
    direction: str
    samples: List[float]
    conditions: Dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    harness: str = ""
    notes: List[str] = field(default_factory=list)
    #: When the samples were MEASURED, as opposed to when this record was
    #: written. The two are usually the same moment and are not always: a
    #: baseline recorded from a measurement taken earlier carries the later
    #: `created` and the earlier `measured_at`.
    #:
    #: `created` alone cannot decide whether a comparison is honest. It says
    #: when the RECORD was written, and the record can be written long after
    #: the measurement — or, worse, written now from numbers measured in a
    #: different session entirely. The admission gate needs to know when the
    #: NUMBERS were taken.
    measured_at: float = 0.0
    #: The measurement session this baseline belongs to. Two measurements
    #: carrying the same session id were taken as one experiment, under one
    #: machine state; two measurements carrying different session ids were not,
    #: and a comparison across them is a comparison of two afternoons.
    #:
    #: This is the field that makes "measured in the same session" a fact
    #: rather than an assumption. Before it existed, the 2026-11-03 comparison
    #: had no way to say it was cross-session, so it did not.
    session_id: str = ""
    #: Baseline-side series for the objective's guardrail metrics, {metric:
    #: samples}, measured in the same session under the same Conditions as
    #: `samples`. A run supplies only the candidate side; a guardrail missing
    #: here leaves the objective with nothing to compare, and it denies.
    guardrail_samples: Dict[str, List[float]] = field(default_factory=dict)
    #: True when this baseline has been WITHDRAWN as a reference — not
    #: because the numbers were wrong, but because it was measured in a way
    #: that cannot support the comparisons it was used for.
    #:
    #: The case that forced this field: on 2026-11-03 a thread-count sweep was
    #: gated against a baseline measured 48 minutes earlier, under a different
    #: set of resident processes, and came back ADMIT +12.34% (p=1.882e-05). The
    #: candidate was not compared with the baseline; the two numbers described
    #: two different machine states and the difference between them was
    #: charged to the candidate. No statistical gate could have caught that —
    #: Welch's t-test on two INDEPENDENT series attributes the drift to the
    #: candidate, which is what "independent" means.
    #:
    #: A superseded baseline is skipped by load_latest and refused by
    #: valid_for, so it cannot gate anything, and its reason travels with it
    #: so a reader knows WHY it was withdrawn rather than finding it missing.
    superseded: bool = False
    superseded_reason: str = ""
    superseded_at: float = 0.0

    def __post_init__(self) -> None:
        # The workload is stored as plain JSON types. `asdict()` leaves an Enum
        # in place, which json.dumps cannot write — so normalise here, once, and a
        # baseline built in memory equals the same baseline read back from disk.
        self.workload = _jsonable(dict(self.workload))
        self.conditions = _jsonable(dict(self.conditions))

    # ---- stats -----------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return _mean(self.samples)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def cv(self) -> float:
        """Coefficient of variation. A baseline whose own runs vary by more than
        its effect threshold cannot support a verdict, and knowing that up front
        saves a whole measurement campaign that could never have concluded."""
        return (self.stdev / self.mean) if self.mean else float("inf")

    @property
    def usable(self) -> bool:
        return self.n >= 2 and self.cv < 0.25

    # ---- the validity gate ----------------------------------------------
    def valid_for(self, objective: Objective, now: Optional[Conditions] = None,
                  ignore: Sequence[str] = ()) -> Tuple[bool, str]:
        """May this baseline be compared against a fresh measurement?

        This is the function whose absence let a matmul harness stand in for a
        UNet. It refuses for four separate reasons, each named.

        `ignore` names condition fields the change under test deliberately
        varies, so applying it does not invalidate the baseline it is being
        tested against. See Conditions.differences.
        """
        if self.superseded:
            return False, ("this baseline was SUPERSEDED — it was measured in a way "
                           "that cannot support the comparisons it was used for, and "
                           "a verdict built on it is not a verdict. Reason: "
                           + (self.superseded_reason or "not stated"))
        if self.metric != objective.primary.name:
            return False, (f"baseline measured '{self.metric}' but the objective scores "
                           f"'{objective.primary.name}'")

        try:
            fields_ = {k: v for k, v in self.workload.items()
                       if k in Workload.__dataclass_fields__}
            # Rebuild the Enum from its stored value. An unknown kind raises and
            # the baseline is refused — never guessed into UNKNOWN.
            fields_["kind"] = WorkloadKind(fields_["kind"])
            stored_wl = Workload(**fields_)
        except Exception:
            stored_wl = None
        if stored_wl is None:
            return False, "baseline has no readable workload identity"
        ok, why = stored_wl.compatible_with(objective.workload)
        if not ok:
            return False, f"workload mismatch — {why}"

        if not self.usable:
            return False, (f"baseline is not usable as a reference (n={self.n}, "
                           f"CV={self.cv:.1%}); its own run-to-run spread is too large")

        if now is not None:
            stored_c = Conditions(**{k: v for k, v in self.conditions.items()
                                     if k in Conditions.__dataclass_fields__})
            diffs = stored_c.differences(now, ignore=ignore)
            if diffs:
                return False, ("machine changed since the baseline was taken — "
                               + "; ".join(diffs))
        return True, "valid"

    # ---- the session gate --------------------------------------------------
    def same_session_as(self, m: Optional["Measurement"],
                        ignore: Sequence[str] = ()) -> Tuple[bool, str]:
        """May a candidate measurement `m` be compared against THIS baseline?

        This is the check that closes the 2026-11-03 ghost. `valid_for` asks
        "is this baseline the right workload under the current conditions?" —
        which was satisfied on the day of the ghost, because the driver, the
        clocks, the power limit and the workload fingerprint all matched. What
        did NOT match was the moment: the candidate was measured at 19:29 and
        the baseline at 18:41, and nothing in the system was able to say so.

        So this asks the question `valid_for` cannot: were these two numbers
        taken as one experiment?

        Three facts have to line up, and each can refuse on its own:

          * the candidate must carry provenance at all — a bare list of floats
            is not a measurement and cannot be placed in time;
          * the session ids must match, when both sides have one;
          * the machine conditions must match, so "same session" means "same
            machine state", not merely "close together in the clock".

        A candidate measured in a DIFFERENT session is refused, whatever the
        arithmetic says. The arithmetic is not the thing that decides.
        """
        if m is None:
            return False, ("this candidate carries no measurement record — when it "
                           "was measured, under what conditions, and in which session "
                           "are all unknown, so it cannot be shown to describe the "
                           "same machine state as the baseline. Re-measure it with "
                           "the paired harness, which records all three.")
        if not m.measured_at:
            return False, ("this candidate carries no timestamp, so there is no way "
                           "to tell whether it was measured at the same time as the "
                           "baseline or in a different session. Re-measure it with the "
                           "paired harness, which stamps every measurement with the "
                           "time and the session it belongs to.")
        if not m.session_id:
            return False, ("this candidate carries no session identity, so there is no "
                           "way to tell whether it was measured alongside this "
                           "baseline or in a different session entirely. Re-measure it "
                           "with the paired harness, which stamps every measurement "
                           "with the session it belongs to.")
        if not self.session_id:
            return False, ("this baseline carries no session identity — it was "
                           "recorded before sessions were tracked, so it cannot be "
                           "shown to belong to the same measurement session as any "
                           "candidate. Record a fresh baseline and measure the "
                           "candidate against it in one session.")
        if m.session_id != self.session_id:
            return False, (f"the candidate was measured in session "
                           f"'{m.session_id}' and the baseline in session "
                           f"'{self.session_id}' — two different sessions are two "
                           f"different machine states, so their difference is drift, "
                           f"not the effect of the change.")

        # The age bound. Coarse, and only a backstop: the session id above is
        # the real evidence. It matters for a baseline recorded before session
        # ids existed, where the id comparison cannot be made.
        base_at = self.measured_at or self.created
        gap = abs(float(m.measured_at) - float(base_at))
        if gap > MAX_SESSION_GAP_S:
            return False, (f"the candidate was measured {gap / 60.0:.0f} minutes away "
                           f"from the baseline (limit "
                           f"{MAX_SESSION_GAP_S / 60.0:.0f} minutes) — two measurements "
                           f"that far apart describe two different machine states, "
                           f"and the difference between them would be charged to the "
                           f"candidate.")

        # The machine state, compared directly when the candidate recorded it.
        # This is the check that would have caught the ghost even if its
        # conditions had looked identical: a competing process that came or
        # went, a clock that moved.
        if m.conditions:
            try:
                cand_c = Conditions(**{k: v for k, v in m.conditions.items()
                                       if k in Conditions.__dataclass_fields__})
                base_c = Conditions(**{k: v for k, v in self.conditions.items()
                                       if k in Conditions.__dataclass_fields__})
            except Exception:
                return False, ("the candidate's machine conditions cannot be read, so "
                               "the machine state it was measured under is unknown")
            diffs = base_c.differences(cand_c, ignore=ignore)
            if diffs:
                return False, ("the machine was not in the same state when the "
                               "candidate was measured as when the baseline was taken "
                               "— " + "; ".join(diffs))
        return True, "same session"

    # ---- persistence -----------------------------------------------------
    def path(self) -> Path:
        return paths.BASELINE_DIR / f"{self.id}.json"

    def save(self) -> Path:
        paths.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        p = self.path()
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path) -> "Baseline":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_latest(objective: Objective) -> Optional[Baseline]:
    """Newest baseline for this objective that passes the workload check.
    Skips (and logs) ones that do not, rather than failing the whole lookup.

    A SUPERSEDED baseline is skipped here, so a withdrawn reference cannot
    gate anything even when it is newer than the good ones."""
    if not paths.BASELINE_DIR.exists():
        return None
    cands = sorted(paths.BASELINE_DIR.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        try:
            b = Baseline.load(p)
        except Exception as e:
            log.warning("unreadable baseline %s: %s", p.name, e)
            continue
        if b.superseded:
            log.info("skipping SUPERSEDED baseline %s: %s", p.name,
                     b.superseded_reason or "not stated")
            continue
        ok, why = b.valid_for(objective)
        if ok:
            return b
        log.debug("skipping baseline %s: %s", p.name, why)
    return None


def supersede(baseline_id: str, reason: str, when: Optional[float] = None) -> Baseline:
    """Withdraw a baseline as a reference, with the reason attached.

    This is not a deletion. The numbers stay on disk and stay readable — they
    were real measurements of a real machine. What changes is that they can no
    longer serve as the control for a verdict, because they were measured in a
    way that cannot support that comparison.

    The reason is mandatory and travels with the baseline, because "this number
    is gone" and "this number was withdrawn because it was measured against a
    stale control" are different facts, and a reader needs the second one.
    """
    if not reason or not reason.strip():
        raise ValueError("a supersede needs a reason — say WHY the baseline cannot "
                         "support the comparisons it was used for")
    path = paths.BASELINE_DIR / f"{baseline_id}.json"
    if not path.exists():
        raise ValueError(f"no baseline {baseline_id} in {paths.BASELINE_DIR}")
    b = Baseline.load(path)
    b.superseded = True
    b.superseded_reason = reason.strip()
    b.superseded_at = when if when is not None else time.time()
    b.save()
    log.info("superseded baseline %s: %s", baseline_id, b.superseded_reason)
    return b


def record(objective: Objective, samples: List[float], harness: str = "",
           notes: Optional[List[str]] = None,
           conditions: Optional[Conditions] = None,
           guardrail_samples: Optional[Dict[str, List[float]]] = None,
           session_id: str = "",
           measured_at: Optional[float] = None) -> Baseline:
    """Persist a fresh baseline. Refuses to store one it cannot later defend.

    Raises ValueError when a machine condition could not be read: such a baseline
    could never be shown to match a later machine, so it is not written at all.
    Raises ValueError, too, for a guardrail series the objective does not
    declare or that is empty: stored, it would look like evidence for a metric
    nothing ever checks.

    `session_id` names the measurement session this baseline belongs to, so a
    candidate measured in the SAME session can be shown to be same-session. It
    defaults to a fresh id, which is correct for a baseline measured on its
    own — the session exists whether or not anyone compares against it yet.
    A caller that measured both arms of a comparison together passes the id
    they used for the candidate, and the two are then provably one experiment.
    """
    guards = {k: list(v) for k, v in (guardrail_samples or {}).items()}
    for name, series in guards.items():
        if name not in objective.guardrails:
            raise ValueError(f"baseline NOT recorded — guardrail '{name}' is not declared "
                             f"by this objective (declared: "
                             f"{', '.join(sorted(objective.guardrails)) or 'none'})")
        if len(series) < 2:
            raise ValueError(f"baseline NOT recorded — guardrail '{name}' needs at least "
                             f"2 samples, got {len(series)}")
    c = conditions or Conditions.capture(objective)
    missing = c.unread()
    if missing:
        raise ValueError("baseline NOT recorded — machine conditions could not be read ("
                         + ", ".join(missing) + "); a baseline whose conditions are "
                         "unknown can never be shown to apply to a later run"
                         + ("; competing_processes is read from the GPU process memory "
                            "counter, which needs pywin32 (win32pdh) in this interpreter, "
                            "and from the process table — one of the two could not be read"
                            if "competing_processes" in missing else ""))
    b = Baseline(
        objective_id=f"{objective.workload.kind.value}:{objective.workload.name}",
        workload=_jsonable(asdict(objective.workload)),
        metric=objective.primary.name,
        direction=objective.primary.direction.value,
        samples=list(samples),
        conditions=asdict(c),
        harness=harness or objective.harness,
        notes=list(notes or []),
        guardrail_samples=guards,
        session_id=session_id or uuid.uuid4().hex[:12],
        measured_at=float(measured_at if measured_at is not None else time.time()),
    )
    if c.competing_processes:
        b.notes.append(
            "GPU was shared during measurement ("
            + ", ".join(c.competing_processes)
            + "); on this hardware processes are time-sliced, so this baseline is "
              "only comparable to another measurement with the same process mix"
        )
    b.save()
    log.info("baseline %s: %s=%.4g (n=%d, CV=%.1f%%) workload=%s",
             b.id, b.metric, b.mean, b.n, b.cv * 100, objective.workload)
    return b


# There was a `compare(baseline, candidate, objective)` here. Nothing called it,
# and it called `objective.admits` without guardrail samples — a second admission
# path beside admission.Gate that would deny every guardrailed objective, or
# admit one the day `admits` stopped requiring them. Admission has one door:
# Gate.evaluate. Removed rather than repaired.


def describe(objective: Optional[Objective] = None) -> str:
    if not paths.BASELINE_DIR.exists():
        return f"no baselines yet (dir: {paths.BASELINE_DIR})"
    rows = []
    for p in sorted(paths.BASELINE_DIR.glob("*.json"),
                    key=lambda q: q.stat().st_mtime, reverse=True)[:20]:
        try:
            b = Baseline.load(p)
        except Exception:
            continue
        age_h = (time.time() - b.created) / 3600.0
        # A superseded baseline is shown, not hidden: its numbers are still on
        # disk and a reader who finds one needs to know it was withdrawn and
        # why, rather than wondering why it is not used.
        mark = "  SUPERSEDED" if b.superseded else ""
        rows.append(f"  {b.id}  {b.metric:<14} {b.mean:>10.4g}  n={b.n:<3} "
                    f"CV={b.cv*100:>5.1f}%  {age_h:>6.1f}h  {b.workload.get('name','?')}{mark}")
        if b.superseded and b.superseded_reason:
            rows.append(f"      reason: {b.superseded_reason}")
    return "\n".join([f"baselines in {paths.BASELINE_DIR}:"] + rows)
