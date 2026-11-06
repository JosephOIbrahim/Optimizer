"""
levers — the ONE implementation of "measure a lever honestly", and the only
place a lever is applied.

Why this module exists
----------------------
The harness used to gate every candidate against a baseline measurement read
from disk — a file that might have been written in an earlier session, under a
different set of resident apps, at a different clock state. Any drift between
the two sessions was charged to the candidate.

On 2026-11-03 that produced a verdict of ADMIT +12.34% (p=1.9e-05) for a
thread-count change worth nothing: re-measured on 2026-11-05 with both arms
alternately in one session, the same change is DENY -0.60%. (The "+0.7%" that
first appeared against it — 47.710 against 48.060 — was the phantom's own
cross-session arithmetic, which is the defect, not a measurement.) The Field
Notes call it "the move that evaporated"; the verdict sat in the ledger.

No statistical gate catches that on its own. Welch's t-test on two INDEPENDENT
series cannot distinguish drift from effect — that is what "independent" means.
So the fix is in the MEASUREMENT, and this module is where it lives:

  * both arms are measured ALTERNATELY, in one session, so the drift is
    common to both and cancels in the difference;
  * the order of the arms alternates too, so whichever arm runs second cannot
    be biased by ALWAYS paying the first arm's warm-up -- that cancels
    exactly when the number of repetitions is EVEN; at an odd count the two
    arms take the second slot an unequal number of times, leaving a small
    residual of the second position's worth divided by the repetition count,
    whose sign is not established (see harness/paired.py);
  * the baseline's drift across the run is checked BEFORE the statistics, and
    a control that moved more than 5% refuses the verdict entirely — a
    verdict computed from an unstable control is not a weak verdict, it is no
    verdict;
  * the guardrail compares the candidate's frames against the BASELINE's, so
    a change that alters the picture cannot pass by being self-consistent.

Every lever here follows the same contract:

    a lever produces two ARM FUNCTIONS, not two numbers.
    arm() -> {"value": float, "images": [path, ...]}

The baseline arm renders the workload unchanged. The candidate arm applies the
change, renders, and puts the change back — so the machine is left as it was
found even if the measurement dies part way through. The put-back is CHECKED
(the setter's own result, and a read-back of the value), because a restore
that failed silently leaves the card at a limit the operator never chose while
the run reports that it left nothing applied.

    from harness import levers
    result = levers.run("frames_per_process", ctx)
    print(result.verdict)

The levers
----------
    frames_per_process   render a shot in one process instead of N per process
    aov_variant          render with fewer AOVs
    power_ceiling        raise/lower the card's power ceiling
    husk_threads         render with a fixed thread count
    manual               a change the operator applies by hand

Measured on this box (2026-11-04 and 11-05), each against its own control:

    frames_per_process  3 frames, 1 process each -> 1 process   +46.95%  p=2.4e-06
    aov_variant         full AOV set -> beauty only              +9.79%   p=3.7e-05
    power_ceiling       450 W -> 600 W (fp16 GEMM)               +8.50%   p=2.5e-07
    power_ceiling       450 W -> 600 W (Karma render)            0%   (never at the cap)
"""
from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harness import common, paired, shot, throttle  # noqa: E402

sys.path.insert(0, str(common.V12_DIR))
from v12.objective import Objective  # noqa: E402


# ---------------------------------------------------------------------------
# machine facts
# ---------------------------------------------------------------------------
def machine_spec() -> Dict[str, Any]:
    """What this machine is, read live. Never hardcoded.

    The power ceiling can only be moved within the range the card allows, and
    the allowed range is a property of the card, not of this file.
    """
    spec: Dict[str, Any] = {}
    try:
        from v12 import truth
        t = truth.gpu_truth()
        spec.update({
            "name": t.name, "driver": t.driver_version,
            "sm_count": t.sm_count, "vram_gib": t.vram_total_bytes / 2**30,
            "power_limit_w": t.power_limit_w,
            "power_limit_min_w": t.power_limit_min_w,
            "power_limit_max_w": t.power_limit_max_w,
            "hags": t.hags_enabled,
            "clock_sm_max_mhz": t.clock_sm_max_mhz,
            "mem_bandwidth_gbs": t.mem_bandwidth_gbs,
            "compute_capability": t.compute_capability,
        })
    except Exception as e:
        spec["error"] = f"{type(e).__name__}: {e}"
    return spec


def power_get() -> Optional[float]:
    """The card's current power ceiling, in watts."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def power_set(watts: float) -> Tuple[bool, str]:
    """Set the card's power ceiling. Returns (ok, message).

    NVML takes MILLIWATTS. Passing watts is not a wrong value, it is an
    invalid one: 500 (meaning 0.5 W) returns INVALID_ARGUMENT, 500000
    (meaning 500 W) succeeds. Measured on this box: 500000 -> read back
    500.0 W; 450000 -> read back 450.0 W.
    """
    try:
        import pynvml
    except ImportError:
        return False, "nvidia-ml-py not importable"
    try:
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
            if not (lo / 1000.0 <= watts <= hi / 1000.0):
                return False, (f"{watts:.0f} W is outside this card's allowed range "
                               f"({lo/1000.0:.0f}-{hi/1000.0:.0f} W)")
            pynvml.nvmlDeviceSetPowerManagementLimit(h, int(watts * 1000))
            got = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            return True, f"power ceiling {got:.0f} W (read back)"
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# measurement arms
# ---------------------------------------------------------------------------
#: The two shot arms measure per-frame wall times that MEAN DIFFERENT
#: THINGS, and they used to travel under one key — `frame_wall_s` — so a
#: reader had to know which arm produced a number before they could
#: interpret it.
#:
#: BATCHED (one process, N frames): the wall time each frame ADDED to the
#: shared process. The first frame carries the setup tax — shader compile,
#: texture load, BVH build — and the frames after it do not. The series
#: reads [expensive, cheap, cheap]. This is the series the setup tax is
#: measured FROM.
#:
#: SPLIT (N processes, one frame each): the wall time of each frame STAND
#: ALONE, in a process to itself, each paying its own setup tax. The series
#: is flat at roughly the cost of one full launch per frame.
#:
#: Both are wall seconds per frame. They are not the same measurement, and
#: reading one as the other is how a reader ends up comparing [54.70, 16.03,
#: 15.66] against [54.70, 54.70, 54.70] and concluding the batched arm
#: made frames 2 and 3 three times faster — which is true, and is the
#: lever's whole point, and is not what the two numbers measure.
FRAME_WALL_IN_PROCESS = "frame_wall_in_process_s"
FRAME_WALL_STANDALONE = "frame_wall_standalone_s"


def _ordered_series(series: Dict[int, List[float]]) -> List[List[float]]:
    """An arm's per-repetition series, in repetition order.

    The arms key their series by repetition index, because the warm-up
    repetition runs with index -1 and a repetition discarded for contention
    is retried with the same index. This puts the repetitions back in order
    and drops the warm-up.
    """
    return [series[i] for i in sorted(series) if i >= 0]


def render_frame_arm(usd: str, settings: str, samples: int, res_scale: int,
                     start_frame: int, label: str,
                     threads: int = 0) -> paired.Arm:
    """One frame, one process. The per-frame RENDER phase.

    Deliberately the render phase (husk's total minus load and update), not
    the wall time: render_s has always meant the shading phase, and changing
    what the number means would invalidate every baseline on disk. The setup
    tax is the shot objective's business.
    """
    def arm(index: int) -> Dict[str, Any]:
        r = shot.render_shot(usd, 1, start_frame, settings, samples, res_scale,
                             label=f"{label}{index}", threads=threads)
        renders = r.get("frame_render_s") or [r["wall_s"]]
        return {
            "value": float(renders[0]),
            "images": list(r.get("images") or []),
            "extra": {"wall_s": r.get("wall_s"),
                      # One frame, one process: the frame's wall time is
                      # its STANDALONE cost, and it carries the setup tax.
                      FRAME_WALL_STANDALONE: r.get("frame_wall_s", []),
                      "throttle": r.get("throttle", {})},
        }
    return arm


def render_shot_arm(usd: str, frames: int, settings: str, samples: int,
                    res_scale: int, start_frame: int, label: str) -> paired.Arm:
    """A whole shot in ONE husk process.

    The arm also accumulates its own per-repetition series of per-frame wall
    times, on `arm.frame_wall_series`. The `extra` dict cannot carry it:
    `paired.Pair` merges each repetition's extra with `.update()`, so only
    the LAST repetition's numbers survive there. The record needs the whole
    series — the setup tax is computed from these numbers, and a tax a
    reader cannot re-derive is a tax they have to trust.

    The series is keyed by repetition index, not appended to. Two things
    make an append wrong: the warm-up repetition runs with index -1 and must
    NOT appear in a series that describes the measured run, and a
    repetition that was discarded for contention is RETRIED with the same
    index, so an append would count it twice.
    """
    series: Dict[int, List[float]] = {}

    def arm(index: int) -> Dict[str, Any]:
        r = shot.render_shot(usd, frames, start_frame, settings, samples, res_scale,
                             label=f"{label}{index}")
        fw = [float(v) for v in (r.get("frame_wall_s") or [])]
        if index >= 0:
            series[index] = fw
        return {
            "value": float(r["wall_s"]),
            "images": list(r.get("images") or []),
            "extra": {FRAME_WALL_IN_PROCESS: fw,
                      "throttle": r.get("throttle", {})},
        }
    arm.frame_wall_series = series
    return arm


def render_split_arm(usd: str, frames: int, settings: str, samples: int,
                     res_scale: int, start_frame: int, label: str) -> paired.Arm:
    """The same shot, one frame per process. The control arm for the
    frames-per-process lever.

    As with the batched arm, the per-repetition series of per-frame wall
    times is accumulated on `arm.frame_wall_series`, because `extra` keeps
    only the last repetition's copy. It is keyed by repetition index for the
    same two reasons: the warm-up runs with index -1, and a repetition
    discarded for contention is retried with the same index.
    """
    series: Dict[int, List[float]] = {}

    def arm(index: int) -> Dict[str, Any]:
        r = shot.render_frames_separately(usd, frames, start_frame, settings, samples,
                                         res_scale, label=f"{label}{index}")
        fw = [float(v) for v in (r.get("frame_wall_s") or [])]
        if index >= 0:
            series[index] = fw
        return {
            "value": float(r["wall_s"]),
            "images": list(r.get("images") or []),
            "extra": {FRAME_WALL_STANDALONE: fw,
                      "throttle": r.get("throttle", {})},
        }
    arm.frame_wall_series = series
    return arm


#: How long the card is left at a new power ceiling before the GEMM is
#: measured. The card re-tunes its boost clock after a limit change, and that
#: takes longer than the three warm-up matmuls `_run_gemm` does.
#:
#: Measured on this box (2026-11-05): the first GEMM after the ceiling was
#: raised from 450 W to 600 W read 157.8 TFLOPS where the card settled at
#: 177.5 — an 11% deficit with nothing to do with the lever. With a settle,
#: the same alternating measurement reads 161.9/162.2/161.8/161.9/162.0 at
#: 450 W and 177.7/177.7/177.2/177.6/178.1 at 600 W — cv under 0.3%.
POWER_SETTLE_S = 1.5


def gemm_arm(watts: Optional[float], label: str,
             settle_s: float = 0.0) -> paired.Arm:
    """One fp16 GEMM measurement, at a given power ceiling (None = leave it).

    `settle_s` is how long the card is left at the new ceiling before the
    measurement is taken. It matters because changing the power ceiling
    changes the card's POWER STATE, and the card needs a moment to re-tune
    its boost clock to the new limit.

    Measured on this box (2026-11-05): the first GEMM after the ceiling was
    raised from 450 W to 600 W read 157.8 TFLOPS, where the card settled at
    177.5 — an 11% deficit that has nothing to do with the lever. The three
    warm-up matmuls `_run_gemm` does are worth about 20 ms, which is not
    enough for a 150 W limit change. The symptom is a control that moves
    around for no reason, which the drift gate then refuses — correctly, and
    with a stated reason ("something else was using the GPU") that is wrong.

    Only the power lever needs this: it is the only one that changes the
    card's power state between measurements.
    """
    def arm(index: int) -> Dict[str, Any]:
        if watts is not None:
            ok, msg = power_set(watts)
            if not ok:
                raise SystemExit(f"could not set the power ceiling: {msg}")
        r = _run_gemm(settle_s=settle_s)
        return {"value": r["tflops"], "extra": {"message": label,
                                                 "throttle": r["throttle"]}}
    return arm


def _run_gemm(n: int = 8192, iters: int = 150,
              settle_s: float = 0.0) -> Dict[str, Any]:
    """One fp16 GEMM measurement, in-process.

    `settle_s` runs untimed iterations at the current power state before the
    timed loop starts, so the measurement describes the card's STEADY
    behaviour at the current ceiling rather than its first second there.
    """
    import torch
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = a.clone()
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    # Settle: untimed work at the current power state. The card re-tunes its
    # boost clock after a ceiling change, and that takes longer than the
    # three warm-up matmuls above.
    if settle_s > 0:
        t_settle = time.perf_counter()
        while time.perf_counter() - t_settle < settle_s:
            for _ in range(10):
                torch.matmul(a, b)
            torch.cuda.synchronize()
    snap_a = throttle.Snapshot.take()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    snap_b = throttle.Snapshot.take()
    del a, b
    torch.cuda.empty_cache()
    return {"tflops": 2.0 * n ** 3 * iters / t / 1e12,
            "throttle": throttle.delta(snap_a, snap_b)}


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def gate(res: paired.PairResult, obj: Objective,
         guards: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Gate a paired run, refusing it if the CONTROL moved.

    The baseline is the control. If it drifted across the run, the machine's
    state changed while we measured, and the comparison describes a machine
    that existed for only part of the run.

    This is not hypothetical: during development a second render was running
    concurrently, the control drifted -33.8% across five repetitions (152.6,
    152.0, 152.3, 126.1, 101.0 TFLOPS), and the paired test still returned
    p=0.0146 — a "significant" win bought with a starved control.

    So the drift check comes FIRST. A verdict computed from an unstable
    control is not a weak verdict; it is not a verdict.

    TWO checks, because the control can move in two ways:

      * `control_spread` — the largest deviation of ANY repetition from the
        control's median. This catches a busy window ANYWHERE in the run.
        The end-to-end `baseline_drift` misses it: a control slow for
        repetitions 2-4 and fast for 1 and 5 has equal first and last values,
        so the end-to-end number reads +0.00% and the gate passes a control
        that moved enormously. Measured here, that shape produced ADMIT
        +6.75%, +10.98%, +14.86% and +23.20% at window sizes of 52, 56, 60
        and 70 seconds — verdicts that blamed the candidate for the
        machine's own change.
      * `contended_samples` — repetitions taken while other GPU work was on
        the card. It is RECORDED and it is REPORTED, but it does not refuse
        the verdict on its own, and the reason is the direction of the
        error.

        A process holding a GPU context is not the same as a process
        RENDERING. On this box a workstation with Nuke and Houdini OPEN —
        idle, drawing 12% utilisation — has both resident, and a tool that
        treated that as contention could not measure anything at all on a
        normal desk. Observed on 2026-11-06.

        What is safe to ignore is bounded. If a rival hits the CONTROL, the
        control moves, and the drift and spread checks catch it — those are
        the authority, and they are checked first. If a rival hits the
        CANDIDATE arm, the candidate reads slow and the effect is
        UNDERSTATED: a false DENY, not a false ADMIT. So contention that
        leaves the control steady can only move the verdict towards
        refusing it, which is the safe direction. A verdict that survives
        both checks was not made on a stolen card.

    The end-to-end drift is still checked, and still reported, because a
    monotone change is a real way for the machine to move and the number
    describes it better than the spread does.
    """
    if res.drift_exceeded:
        return (False,
                f"the BASELINE moved {res.baseline_drift:+.1%} across the run "
                f"(limit {paired.PairResult.DRIFT_LIMIT:.0%}), so the machine's state "
                f"changed while we measured and the comparison does not describe one "
                f"machine. Something else was using the GPU, or the machine was "
                f"otherwise busy. Close it and run this again — this is the same "
                f"failure that produced the +12% phantom on 2026-11-03.",
                {"drift": res.baseline_drift, "control_spread": res.control_spread,
                 "contended_samples": res.contended_samples, "effect": 0.0})
    if res.control_exceeded:
        return (False,
                f"the BASELINE moved {res.control_spread:.1%} away from its own "
                f"middle at some point in the run (limit "
                f"{paired.PairResult.DRIFT_LIMIT:.0%}), so the machine's state "
                f"changed for part of the measurement and the comparison does "
                f"not describe one machine. The end-to-end drift reads "
                f"{res.baseline_drift:+.1%} because the change was in the MIDDLE "
                f"of the run, not at its ends. Something else was using the GPU "
                f"for part of the campaign — close it and run this again.",
                {"drift": res.baseline_drift, "control_spread": res.control_spread,
                 "contended_samples": res.contended_samples, "effect": 0.0})
    if res.contended_samples:
        # The control held steady and the drift is inside the limit, so the
        # comparison is sound — but say plainly that the card was shared, and
        # with whom. A reader who knows a rival was on the card can judge the
        # number for themselves; a record that does not say so is asking
        # them to trust it.
        who = ", ".join(res.contenders) if res.contenders else "another process"
        shared_note = (
            f"{res.contended_samples} repetition(s) were taken while other GPU "
            f"work was on the card ({who}). The control held steady and the "
            f"drift is inside the limit, so the comparison stands; contention "
            f"that leaves the control steady can only make the effect look "
            f"SMALLER, never larger.")
        print(f"  {shared_note}")
    admitted, reason, detail = obj.admits(res.baseline.values, res.candidate.values,
                                          guards, paired_diffs=res.diffs)
    if res.contended_samples:
        detail["contended_samples"] = res.contended_samples
        detail["contenders"] = list(res.contenders)
        detail["shared_card_note"] = shared_note
        # The verdict string carries it too, so a reader of the RECORD sees
        # that the card was shared without having to know to look in the
        # detail fields.
        reason = f"{reason} — but {shared_note}"
    return admitted, reason, detail


def ssim_guardrail(res: paired.PairResult) -> Optional[Dict[str, Any]]:
    """The ssim guardrail for a paired run, scored against a SHARED reference.

    Both sides are measured against the baseline's first frame. The baseline's
    own score is therefore its self-consistency (how much the renderer varies
    between repetitions of the same frame) and the candidate's score is how far
    its frames sit from that reference. The guardrail then asks the right
    question: is the candidate's picture as close to the baseline's as the
    baseline's own repetitions are?

    Scoring the candidate against ITSELF cannot catch an altered image. That
    is not hypothetical: on this box the 16-spp data pass renders a completely
    different picture (cross-run SSIM 0.28 against the beauty pass) yet its
    repetitions are perfectly self-consistent at 1.0, so a self-comparison
    would have passed it.

    The bar is a MEASUREMENT: it is the renderer's variation between two
    DIFFERENT renders of the same frame. Scoring the reference against itself
    — which is what the bar used to do for its own repetition — is 1.0 by
    construction, and a bar made of a number that was never rendered charges
    the candidate for regression it did not cause.
    """
    notes: List[str] = []
    g_base, g_cand = paired.ssim_series(res.baseline.image_sets,
                                        res.candidate.image_sets, notes=notes)
    for n in notes:
        print(f"    guardrail: {n}")
        res.notes.append(n)
    # Either side empty means the guardrail was NOT MEASURED. A declared
    # guardrail that could not be measured denies the change (see
    # v12.objective.admits), and a guardrail with an empty side would take the
    # report down with it.
    if not g_base or not g_cand:
        return None
    return {"ssim": (g_base, g_cand)}


def _guardrail_series(guards: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The guardrail's own samples, one list per side, per repetition.

    The gate records the guardrail as a single number (guardrail_ssim: the
    fractional change in the score). That number is a VERDICT, not the
    evidence. A guardrail whose samples are not on disk cannot be re-checked,
    and a guardrail nobody can re-check is a guardrail that can be argued
    with later — the same failure the summary statistics have.

    So the samples travel with the verdict: `guardrail_ssim_baseline` and
    `guardrail_ssim_candidate`, one score per repetition.
    """
    out: Dict[str, Any] = {}
    for name, sides in (guards or {}).items():
        try:
            g_base, g_cand = sides
        except (TypeError, ValueError):
            continue
        out[f"guardrail_{name}_baseline"] = [float(v) for v in g_base]
        out[f"guardrail_{name}_candidate"] = [float(v) for v in g_cand]
    return out


# ---------------------------------------------------------------------------
# lever context and result
# ---------------------------------------------------------------------------
@dataclass
class Context:
    """Everything a lever needs to know about what it is being measured on."""

    usd: str
    settings: str = "/Render/OVOID_CPU_RENDER_V013"
    samples: int = 128
    res_scale: int = 100
    start_frame: int = 503
    frames: int = 3
    reps: int = 3
    #: How many warm-up repetitions to run and DISCARD before measuring.
    #:
    #: The default used to be 0, and `from_args` never set it, so NO lever
    #: ever warmed up through the command line. The shot lever then read the
    #: first repetition as a 5.3% drift and the drift gate refused the whole
    #: campaign — on a card that was idle, with zero contention events.
    #:
    #: What the first repetition pays for is the machine being COLD: a cold
    #: page cache, a cold shader cache, a cold BVH. The split arm pays it
    #: three times over, because it launches three husk processes, and the
    #: first one pays it hardest. That is not drift between measurements; it
    #: is the first measurement being unrepresentative of the ones after it.
    #:
    #: One warm-up repetition of each arm costs about four minutes on the
    #: shot path and buys a series that describes the machine rather than its
    #: cold start.
    warmup: int = 1
    variant: str = "beauty_only"
    threads: int = 0
    #: How long the CAMPAIGN may wait for the GPU to go quiet, in total. Long
    #: by default: a contended measurement is worse than a late one, because
    #: the drift gate refuses it and the operator has waited for nothing.
    #:
    #: This is a budget for the whole campaign, not a per-phase allowance.
    #: It used to be handed to each waiting thing separately — once before the
    #: lever and again before every repetition — so a contended machine
    #: waited the timeout, then waited it again, then again per repetition.
    #: Observed on this box: a campaign waited 900 s, announced it was
    #: measuring anyway, then waited another 900 s before its first
    #: repetition. Two full timeouts for a campaign that could not have
    #: measured anything.
    quiet_timeout_s: float = 900.0
    #: The campaign's quiet budget. Created once, when the campaign starts,
    #: and shared by every wait inside it.
    quiet_budget: Optional[common.QuietBudget] = None
    spec: Dict[str, Any] = field(default_factory=machine_spec)
    #: For the manual lever: the operator's change, applied and reverted by hand.
    #: The prompt names what they must be doing at each step.
    manual_prompt: str = ""

    def budget(self) -> common.QuietBudget:
        """The campaign's quiet budget, created on first use.

        One budget, shared by the wait before the lever and the wait before
        every repetition, so the campaign waits the timeout ONCE rather than
        once per phase.
        """
        if self.quiet_budget is None:
            self.quiet_budget = common.QuietBudget(self.quiet_timeout_s)
        return self.quiet_budget

    @classmethod
    def from_args(cls, args) -> "Context":
        return cls(usd=args.usd, settings=args.settings, samples=args.samples,
                   res_scale=args.res_scale, start_frame=args.start_frame,
                   frames=args.frames, reps=args.reps, variant=args.variant,
                   threads=getattr(args, "threads", 0),
                   # The warm-up was never wired to the command line, so every
                   # run through the .bat measured with a cold first
                   # repetition. The lever now reads it from the args when it
                   # is there, and falls back to the class default (1) when it
                   # is not — which is the point: the cold start must not be
                   # inside the measured series.
                   warmup=int(getattr(args, "warmup", cls.warmup)),
                   quiet_timeout_s=float(getattr(args, "quiet_timeout", 900)))


@dataclass
class Result:
    """The outcome of one lever, measured."""

    lever: str
    ok: bool = False
    why: str = ""
    effect: float = 0.0
    p_value: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    baseline_mean: Optional[float] = None
    candidate_mean: Optional[float] = None
    drift: float = 0.0
    #: True when the machine was NOT left the way it was found. It used to be
    #: a constant False, which meant the record asserted "nothing was left
    #: applied" whether or not the restore had actually worked — a claim the
    #: harness had never checked. Now it is set only when the restore FAILED,
    #: so the field says something a reader can trust.
    applied: bool = False
    #: Why the machine was not left as it was found. Empty when it was.
    restore_error: str = ""
    notes: List[str] = field(default_factory=list)
    #: Times the machine was busy enough to pause the run.
    contention_events: int = 0

    def summary(self) -> str:
        lines = [f"  {self.lever:<22} {'ADMIT' if self.ok else 'DENY':<6} "
                 f"{self.effect:+.2%}  {self.why}"]
        return "\n".join(lines)


def _report(res: paired.PairResult, guards: Optional[Dict[str, Any]]) -> None:
    print("\n" + res.describe())
    print(f"  control spread (largest deviation from its own middle): "
          f"{res.control_spread:.2%}")
    if res.contended_samples:
        who = f" ({', '.join(res.contenders)})" if res.contenders else ""
        print(f"  {res.contended_samples} sample(s) were taken while other GPU work "
              f"was on the card{who} — the control held steady, so the comparison "
              f"stands, but the card was shared")
    if guards:
        g_base, g_cand = guards["ssim"]
        print(f"  ssim guardrail: baseline {statistics.fmean(g_base):.4f} vs "
              f"candidate {statistics.fmean(g_cand):.4f}")
    else:
        print("  (no images to compare — the ssim guardrail cannot be evaluated)")


def _finish(lever: str, res: paired.PairResult, obj: Objective,
            guards: Optional[Dict[str, Any]],
            notes: Optional[List[str]] = None) -> Result:
    _report(res, guards)
    ok, why, detail = gate(res, obj, guards)
    # The raw series travel WITH the summary statistics, not instead of them.
    # The record is the evidence; a record that only carries the answer cannot
    # be checked by anybody but the harness that wrote it, and this repo keeps
    # retracted verdicts and superseded baselines on disk precisely so that a
    # reader can disagree with it. The same applies here: the effect, the
    # p-value and the drift are all recomputable from these three lists.
    detail.update(res.raw_series())
    detail.update(_guardrail_series(guards))
    # The control's SPREAD travels with the record, because it is the number
    # the gate now refuses on: the end-to-end drift cannot see a busy window
    # in the middle of a run, and a reader who cannot see the spread cannot
    # check why a verdict was refused.
    detail["control_spread"] = res.control_spread
    effect = detail.get("effect", 0.0)
    print(f"\n  verdict: {'ADMIT' if ok else 'DENY'}  ({effect:+.2%})")
    print(f"  {why}")
    return Result(lever=lever, ok=ok, why=why, effect=effect,
                  p_value=detail.get("p_value"), detail=detail,
                  baseline_mean=res.baseline.summary().get("mean"),
                  candidate_mean=res.candidate.summary().get("mean"),
                  drift=res.baseline_drift, notes=list(notes or []),
                  contention_events=res.contended_samples)


def _progress(msg: str) -> None:
    print(msg)


def _quiet_gate(budget: common.QuietBudget) -> "Callable[[], bool]":
    """A gate the paired engine calls before each repetition.

    Checking once, before the run, is not enough: a ten-minute measurement
    can be spoiled by a job that starts halfway through. Observed on this box
    during development: the card was quiet at the start, a 10.6 GB process
    appeared on the third repetition, and the control drifted -70.9% over the
    run. The drift gate refused the result — correctly — but the ten minutes
    were gone.

    So the engine asks again before every repetition, and waits. The cost is
    that a contended machine takes longer to measure; the benefit is that a
    measurement is not thrown away because somebody else's job started.

    TWO questions are asked here, because they catch different things and
    this is the one moment both are valid — our own render has not started:

      * WHO else holds the card (process identity). This is the question that
        survives our own load, and it is what caught a WSL benchmark holding
        the card at 98% while no Windows process accounted for it.
      * HOW HARD is the card working (utilisation). Valid only here, before
        our render starts; during a render the card is busy by design and
        this says nothing. It catches work that holds no memory at all.

    The gate is a courtesy to the operator, not a veto. If the machine never
    settles, it says so and measures anyway — the drift gate remains the
    authority on whether the result is usable.

    The gate draws on the CAMPAIGN's budget, not a fresh one. It used to be
    handed the timeout again for every repetition, so a contended machine
    waited the timeout before the lever and then the timeout again before
    each repetition. Now the budget is spent once and shared: the gate can
    only wait for what is LEFT of it.
    """
    def gate() -> bool:
        common.wait_for_no_others(timeout_s=budget.timeout_s, verbose=True,
                                  budget=budget)
        return common.wait_for_quiet_gpu(timeout_s=budget.timeout_s, verbose=True,
                                         budget=budget)
    return gate


# ---------------------------------------------------------------------------
# the levers
# ---------------------------------------------------------------------------
def lever_frames_per_process(ctx: Context, obj: Objective) -> Result:
    """Render the shot in one process instead of N, and see what that is worth.

    The setup tax is ~38 s per launch on this box — shader compile, texture
    load, BVH build — and it is paid again for every frame. The Field Notes
    name "frames per process" as the fast lever, and they are right: it is
    worth more than every card-side lever combined, and it costs nothing.
    """
    frames = ctx.frames
    print(f"\n  rendering {frames} frames one per process (today) vs {frames} frames "
          f"in ONE process,")
    print("  alternately, in one session.\n")
    base = render_split_arm(ctx.usd, frames, ctx.settings, ctx.samples,
                            ctx.res_scale, ctx.start_frame, "split")
    cand = render_shot_arm(ctx.usd, frames, ctx.settings, ctx.samples,
                           ctx.res_scale, ctx.start_frame, "batched")
    p = paired.Pair(f"frames per process (1 vs {frames})", base, cand,
                    reps=max(ctx.reps, obj.min_samples),
                    # At least one warm-up, always. The split arm launches
                    # three husk processes, so it pays the cold-cache cost
                    # three times and the first one pays it hardest. Without
                    # the warm-up that cold start landed INSIDE the measured
                    # series and the drift gate read it as the machine
                    # changing — refusing the whole campaign on an idle card
                    # with zero contention events.
                    warmup=max(ctx.warmup, 1),
                    lower_is_better=True, on_progress=_progress,
                    baseline_name="one frame per process",
                    candidate_name=f"{frames} frames per process")
    p.quiet_check = _quiet_gate(ctx.budget())
    res = p.run()
    notes = []
    tax = 0.0
    # The tax is measured from the BATCHED arm's in-process series: inside one
    # process the first frame pays the setup and the frames after it do not, so
    # the series reads [expensive, cheap, cheap] and the difference IS the tax.
    # The split arm's series is flat — every frame is alone in a process and
    # pays its own setup — so it cannot measure the tax at all. Reading the
    # wrong one of the two is how this number used to be able to come out 0.0.
    cand_extra = res.candidate.extra.get("candidate", {}) or {}
    base_extra = res.baseline.extra.get("baseline", {}) or {}
    fw = [float(v) for v in (cand_extra.get(FRAME_WALL_IN_PROCESS) or [])]
    sw = [float(v) for v in (base_extra.get(FRAME_WALL_STANDALONE) or [])]
    # The arms keep their OWN per-repetition series, because `extra` is
    # merged with `.update()` and keeps only the LAST repetition's copy.
    # The whole series is what belongs in the record.
    in_process = _ordered_series(getattr(cand, "frame_wall_series", {}))
    standalone = _ordered_series(getattr(base, "frame_wall_series", {}))
    if in_process:
        fw = in_process[0]
    if standalone:
        sw = standalone[0]
    if fw:
        from harness import constraint
        tax = constraint.setup_tax(fw)
        notes.append(f"setup tax: {tax:.1f} s per launch, paid once for {frames} "
                     f"frames instead of {frames} times")
    r = _finish("frames_per_process", res, obj, ssim_guardrail(res), notes)
    # The tax explains WHY the lever wins, and it is the only number in the
    # record that says so. It used to live in Result.notes, which nothing
    # serialized — so the 11-04 records carried it and the 11-05 ones did not.
    # A verdict whose explanation is missing from the file is a verdict a
    # reader has to take on trust, and this repo does not trade on trust.
    r.detail["setup_tax_s"] = tax
    # The series the tax came from, and the other arm's series beside it, so
    # the tax can be re-derived from the file and so the two kinds of
    # per-frame time cannot be confused for one another in it.
    if in_process:
        r.detail[FRAME_WALL_IN_PROCESS] = in_process
    if standalone:
        r.detail[FRAME_WALL_STANDALONE] = standalone
    return r


def lever_aov_variant(ctx: Context, obj: Objective) -> Result:
    """Render the frame with fewer AOVs and see what that is worth.

    The Field Notes named "an AOV pass" as one of the three wins that were
    real, and the ledger agrees — though not at the number it first claimed.
    On 2026-11-03 the beauty-only variant measured 39.93 s against a 48.06 s
    baseline, and the ledger called that +16.91%. That comparison was between
    two sessions, so it was retracted: measured properly, with both arms
    alternately in one session, the lever is worth +9.79% (p=3.7e-05). The
    win is real; the first number was drift charged to the change.

    This is a SCENE-side lever, not a card-side one, which is the point: the
    card was drawing 104 W of a 450 W budget the whole time. The pixels being
    shaded were pixels nobody had asked for.

    The guardrail matters more here than anywhere else. AOV reduction changes
    what the renderer SHADES but not what the beauty image LOOKS like, so the
    beauty PNG must still match the baseline's. Measured: 0.9948 against the
    baseline's own 0.9962 self-consistency — the same picture, shaded less.
    """
    variants = {
        "beauty_only": r"E:\OptimizerV12\scenes\ovoid_hero_v002_beauty.usd",
        "no_crypto": r"E:\OptimizerV12\scenes\ovoid_hero_v002_nocrypto.usd",
    }
    chosen = ctx.variant if ctx.variant in variants else "beauty_only"
    cand_usd = variants[chosen]
    if not Path(cand_usd).exists():
        return Result(lever="aov_variant", ok=False,
                      why=f"the {chosen} variant is not on disk ({cand_usd}) — build "
                          f"it with hython harness\\variant_usd.py first")
    print(f"\n  rendering the hero frame with the {chosen} AOV set, against the "
          f"full one,")
    print("  alternately, in one session.\n")
    base = render_frame_arm(ctx.usd, ctx.settings, ctx.samples, ctx.res_scale,
                            ctx.start_frame, "full")
    cand = render_frame_arm(cand_usd, ctx.settings, ctx.samples, ctx.res_scale,
                            ctx.start_frame, chosen)
    p = paired.Pair(f"AOV set: full vs {chosen}", base, cand,
                    reps=max(ctx.reps, obj.min_samples), warmup=max(ctx.warmup, 1),
                    lower_is_better=True, on_progress=_progress,
                    baseline_name="full AOV set", candidate_name=chosen)
    p.quiet_check = _quiet_gate(ctx.budget())
    res = p.run()
    r = _finish("aov_variant", res, obj, ssim_guardrail(res))
    r.detail["variant"] = chosen
    return r


def restore_power_ceiling(prior_w: float) -> Tuple[bool, str]:
    """Put the card's power ceiling back where it was, and PROVE it went back.

    The operator card promises that the ceiling is restored after every run
    and that the prior is read back to prove it. A promise that is never
    checked is not a promise: power_set fails when the value is outside the
    card's range, when nvidia-ml-py is missing, and when NVML errors — and
    the old code threw its return value away, so a failed restore was
    SILENT and the run still said "nothing was left applied".

    So the restore is checked twice: the setter's own result, and a read-back
    compared against the prior. A read-back that disagrees means the card is
    sitting at a power limit the operator never chose, and that has to be
    said out loud rather than buried in a return value nobody reads.

    Returns (ok, message). The message is the operator-facing text either
    way — what went wrong, or what was proven.
    """
    ok, msg = power_set(prior_w)
    if not ok:
        return False, (f"could not restore the power ceiling to {prior_w:.0f} W: {msg}")
    got = power_get()
    if got is None:
        return False, (f"the power ceiling was set back to {prior_w:.0f} W but could "
                       f"not be read back to prove it")
    if abs(got - prior_w) > 0.5:
        return False, (f"the power ceiling reads {got:.0f} W after restoring "
                       f"{prior_w:.0f} W — the card is NOT where it was found")
    return True, f"the power ceiling is back at {got:.0f} W (read back)"


def lever_power_ceiling(ctx: Context, obj: Objective) -> Result:
    """Raise the power ceiling and see whether the workload actually cares.

    The card accepts 150-600 W and ships at 450 W, so there is real headroom
    to unlock. Whether it is worth taking depends entirely on the workload:
    fp16 GEMM is pinned at the ceiling and gained +8.50%; a Karma render
    draws 104 W of 450 W and gained nothing. The ceiling is put back
    exactly afterwards, whatever happens — and the put-back is CHECKED, so
    a failure to restore is reported instead of assumed away.
    """
    # The prior is read LIVE, here, immediately before anything is changed.
    #
    # It used to come from ctx.spec, which was captured when the Context was
    # built — possibly minutes earlier, possibly before another process moved
    # the ceiling. Restoring that number would put the card at a power limit
    # the operator never chose, and would do it SILENTLY, because the
    # setter's return value was being discarded.
    prior = power_get()
    prior_source = "read live, immediately before the change"
    if prior is None:
        # No live reading. The snapshot is all there is, and the restore
        # below cannot be verified against a value nobody confirmed.
        prior = ctx.spec.get("power_limit_w")
        prior_source = ("taken from the context snapshot — the live ceiling could "
                        "not be read")
    if prior is None:
        return Result(lever="power_ceiling", ok=False,
                      why="could not read the card's power ceiling, so there is no "
                          "prior to restore — refusing to change it")
    cur = float(prior)
    hi = ctx.spec.get("power_limit_max_w") or 0.0
    if not hi or hi <= cur + 1:
        return Result(lever="power_ceiling", ok=False,
                      why=f"the card's ceiling is already at its maximum ({cur:.0f} W)")
    print(f"\n  the ceiling is {cur:.0f} W and this card accepts up to {hi:.0f} W.")
    print("  measuring the SAME workload at both ceilings, alternately, so the")
    print("  comparison cannot drift.\n")
    base = gemm_arm(cur, f"at {cur:.0f}W", settle_s=POWER_SETTLE_S)
    cand = gemm_arm(hi, f"at {hi:.0f}W", settle_s=POWER_SETTLE_S)
    restore_ok, restore_msg = True, "the power ceiling was never raised"
    try:
        p = paired.Pair(f"power ceiling {cur:.0f} W vs {hi:.0f} W", base, cand,
                        reps=max(ctx.reps, obj.min_samples), warmup=max(ctx.warmup, 1),
                        lower_is_better=False, on_progress=_progress,
                        baseline_name=f"at {cur:.0f} W",
                        candidate_name=f"at {hi:.0f} W")
        p.quiet_check = _quiet_gate(ctx.budget())
        res = p.run()
    finally:
        # The machine goes back to where it was, whatever happened — and the
        # restore is CHECKED, because a silent failure here leaves the card
        # at 600 W while the run reports that it left nothing applied.
        restore_ok, restore_msg = restore_power_ceiling(cur)
    r = _finish("power_ceiling", res, obj, None)
    r.detail["baseline_w"] = cur
    r.detail["candidate_w"] = hi
    r.detail["power_prior_w"] = cur
    r.detail["power_prior_source"] = prior_source
    r.detail["power_restored"] = restore_ok
    r.detail["power_restore_note"] = restore_msg
    if restore_ok:
        print(f"\n  {restore_msg}")
    else:
        # Loud, and in the record. The operator has to be told the machine is
        # not where they left it, and the record has to say it too.
        r.applied = True
        r.restore_error = restore_msg
        r.notes.append(f"THE POWER CEILING WAS NOT RESTORED: {restore_msg}")
        print(f"\n  !! THE POWER CEILING WAS NOT RESTORED: {restore_msg}")
        print(f"  !! Set it back by hand:  nvidia-smi -pl {cur:.0f}")
    return r


def lever_husk_threads(ctx: Context, obj: Objective) -> Result:
    """Render with a fixed thread count.

    This is the lever that produced the +12.34% phantom on 2026-11-03. It is
    kept, because a lever that cannot be measured is a lever that cannot be
    used — but now it is measured against a control taken in the SAME session,
    alternately, so the number it produces is real.

    On this box the default (all 48 threads) and -j 48 measure the same, so
    this is expected to land near zero. That is the honest answer, and it is
    the answer the old harness could not give.
    """
    n = ctx.threads or 48
    print(f"\n  rendering the hero frame with the default thread count vs -j {n},")
    print("  alternately, in one session.\n")
    base = render_frame_arm(ctx.usd, ctx.settings, ctx.samples, ctx.res_scale,
                            ctx.start_frame, "default", threads=0)
    cand = render_frame_arm(ctx.usd, ctx.settings, ctx.samples, ctx.res_scale,
                            ctx.start_frame, f"j{n}", threads=n)
    p = paired.Pair(f"husk threads (default vs -j {n})", base, cand,
                    reps=max(ctx.reps, obj.min_samples), warmup=max(ctx.warmup, 1),
                    lower_is_better=True, on_progress=_progress,
                    baseline_name="default threads", candidate_name=f"-j {n}")
    # The gate every other lever has, and this one was missing — the lever
    # that produced the +12.34% phantom was also the one lever that did not
    # wait for the card before each repetition. It has to be here: a
    # measurement of thread count is exactly the measurement that drifts
    # when something else has the card, because a busy card makes every
    # thread look slow.
    p.quiet_check = _quiet_gate(ctx.budget())
    res = p.run()
    return _finish("husk_threads", res, obj, ssim_guardrail(res))


def lever_manual(ctx: Context, obj: Objective) -> Result:
    """A change the operator applies by hand, measured the same honest way.

    The old method's move 03 was "apply one change by hand, then measure it
    and compare against the baseline on record". The comparison was the
    problem: that baseline was usually from an earlier session.

    Here the operator is asked to toggle their change between the two arms,
    so the control and the candidate are measured under the same conditions,
    alternately. It is more work than applying a change and walking away,
    and it is the whole point: the number at the end is worth something.
    """
    prompt = ctx.manual_prompt or (
        "toggle your change now: it must be UNAPPLIED for the baseline arm and "
        "APPLIED for the candidate arm")
    print(f"\n  {prompt}")
    print("  the arms alternate, so you will be asked to toggle between them.\n")

    def make_arm(applied: bool, label: str) -> paired.Arm:
        def arm(index: int) -> Dict[str, Any]:
            state = "APPLIED" if applied else "UNAPPLIED"
            print(f"    rep {index+1}: set your change to {state}, then press Enter...")
            try:
                input()
            except EOFError:
                pass
            r = shot.render_shot(ctx.usd, 1, ctx.start_frame, ctx.settings,
                                 ctx.samples, ctx.res_scale,
                                 label=f"{label}{index}")
            renders = r.get("frame_render_s") or [r["wall_s"]]
            return {"value": float(renders[0]),
                    "images": list(r.get("images") or []),
                    "extra": {"wall_s": r.get("wall_s")}}
        return arm

    base = make_arm(False, "manual_base")
    cand = make_arm(True, "manual_cand")
    p = paired.Pair("manual change", base, cand,
                    reps=max(ctx.reps, obj.min_samples),
                    # The operator is prompted to change the machine between
                    # arms, so the warm-up here doubles as the moment they
                    # learn what the change is.
                    warmup=max(ctx.warmup, 1),
                    lower_is_better=True, on_progress=_progress,
                    baseline_name="change not applied",
                    candidate_name="change applied")
    # The operator is already here and already being asked to change the
    # machine between arms, but the card can still be busy with somebody
    # else's work. Same gate as every other lever.
    p.quiet_check = _quiet_gate(ctx.budget())
    res = p.run()
    r = _finish("manual", res, obj, ssim_guardrail(res))
    # The one lever that leaves the machine alone, ON PURPOSE: the harness
    # does not know what the operator's change is, so it cannot undo it. The
    # note travels with the record and with the front door's summary, so
    # "nothing was left applied" is never said about a machine the operator
    # has to put back themselves.
    r.notes.append("the manual change was NOT reverted — the lever asks you to "
                   "toggle it between the arms and leaves it wherever you left it")
    return r


LEVERS: Dict[str, Callable[[Context, Objective], Result]] = {
    "frames_per_process": lever_frames_per_process,
    "aov_variant": lever_aov_variant,
    "power_ceiling": lever_power_ceiling,
    "husk_threads": lever_husk_threads,
    "manual": lever_manual,
}

#: The metric each lever PRODUCES — the quantity its arms actually measure.
#:
#: This is per-lever, not per-arm, because a lever's two arms are two
#: measurements of the SAME quantity: the whole point is that the candidate
#: and the control are compared like for like.
#:
#:   frames_per_process  shot_s   — a whole shot's wall time, setup included
#:   aov_variant         render_s — one frame's shading phase
#:   husk_threads        render_s — one frame's shading phase
#:   power_ceiling       tflops   — fp16 GEMM throughput
#:   manual              render_s — one frame's shading phase
#:
#: It has to be declared, because a verdict is computed from a METRIC and a
#: metric has a DIRECTION. Score one quantity against an objective that
#: scores another and the sign inverts: `render_s` is lower-is-better and
#: `tflops` is higher-is-better, so a candidate that renders 9% SLOWER
#: becomes ADMIT +9.14% against a compute objective that scores TFLOPS.
#: Nothing downstream can catch that — the ssim guardrail has no images to
#: compare for a GEMM, and the arithmetic is perfectly correct arithmetic
#: about the wrong quantity.
#:
#: "a verdict computed from the wrong quantity is not a verdict at all", so
#: the runner refuses the mismatch rather than recording a number.
LEVER_METRIC: Dict[str, str] = {
    "frames_per_process": "shot_s",
    "aov_variant": "render_s",
    "husk_threads": "render_s",
    "power_ceiling": "tflops",
    "manual": "render_s",
}


def metric_mismatch(name: str, obj: Objective) -> Optional[str]:
    """Why this lever cannot answer this objective's question, if it cannot.

    Returns None when the lever measures the quantity the objective scores,
    and otherwise the reason to refuse, naming BOTH quantities — the one the
    lever produces and the one the objective scores. A refusal that does not
    name both leaves the operator to work out which half they got wrong.
    """
    produced = LEVER_METRIC.get(name)
    if produced is None:
        return None
    scored = obj.primary.name
    if produced == scored:
        return None
    return (f"{name} measures {produced}; the {obj.workload.name} objective "
            f"scores {scored}. A verdict on one quantity cannot answer a "
            f"question about another — and the two have different directions, "
            f"so the effect's sign would be inverted.")


def run(name: str, ctx: Context, obj: Objective) -> Result:
    """Measure one lever. Nothing is left applied — or the run says so.

    The `manual` lever is the exception, and it is the exception BY DESIGN:
    it asks the operator to toggle their own change between the two arms and
    never reverts it, because only the operator knows what the change is.
    The result carries that fact in `notes`, so the front door can tell the
    operator the machine is theirs to put back.
    """
    fn = LEVERS.get(name)
    if fn is None:
        return Result(lever=name, ok=False,
                      why=f"unknown lever '{name}' (known: {', '.join(sorted(LEVERS))})")
    # The lever and the objective have to be talking about the SAME quantity.
    # This is checked BEFORE anything is measured, because the alternative is
    # to spend the campaign producing a number about the wrong thing.
    mismatch = metric_mismatch(name, obj)
    if mismatch:
        return Result(lever=name, ok=False, why=mismatch)
    # Wait for the machine to be quiet BEFORE measuring, not after. The drift
    # gate refuses a result taken on a busy card, but refusing it after the
    # operator has waited out the measurement wastes the whole campaign — and
    # the thing they are told to close is often one they did not know was
    # running.
    #
    # Both waits and every wait INSIDE the lever draw on ONE budget, created
    # here. The timeout is the CAMPAIGN's, spent once — not a fresh allowance
    # for the wait before the lever and another for the wait before each
    # repetition. On a contended machine that used to mean two full timeouts
    # before the first measurement, and more after.
    budget = ctx.budget()
    common.wait_for_no_others(timeout_s=ctx.quiet_timeout_s, verbose=True,
                              budget=budget)
    common.wait_for_quiet_gpu(timeout_s=ctx.quiet_timeout_s, verbose=True,
                              budget=budget)
    return fn(ctx, obj)
