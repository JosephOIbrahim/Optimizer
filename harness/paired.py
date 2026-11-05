"""
paired — measure two arms ALTERNATELY, in one session, so drift cancels.

Why this module exists
----------------------
On 2026-11-03 this repo gated a thread-count sweep against a baseline that had
been measured 48 minutes earlier, in a different set of resident processes. The
verdict came back ADMIT +12.34%, p=1.9e-05 — and it was meaningless. The
re-timed baseline was 48.06 s against the candidate's 47.71 s: +0.7%, nothing.
(The "+0.7%" there is the phantom's own cross-session arithmetic, not a
measurement of the lever. Measured the way this module measures — both arms
alternately in one session — the same change is DENY -0.60%, and that is the
figure to quote.) The two numbers were never a comparison of anything. They
were a comparison of two different afternoons.

No statistical gate can catch that on its own. Welch's t-test on two
INDEPENDENT series attributes the between-session drift to the candidate,
because that is exactly what an unpaired comparison cannot distinguish from a
real effect. The fix is in the MEASUREMENT, not the arithmetic: measure the two
arms alternately, in one session, so the drift is common to both and cancels in
the difference.

That is what this module does. A `Pair` runs baseline and candidate back to
back, repeatedly, and reports per-pair differences — which V12's objective
gates with a paired t-test (`v12.objective.paired_t_test`).

The arms
--------
An arm is a callable that produces ONE sample and returns it, plus whatever
artifacts the guardrail needs (here: a rendered image to compare against the
baseline's). Arms are given the same starting conditions, so anything that
varies between them is the thing under test.

    from harness import paired

    p = paired.Pair("thread count 48", arm_baseline, arm_candidate, reps=5)
    result = p.run()
    print(result.effect_pct, result.p_value)

Nothing here applies a change: the arm is responsible for putting the machine
in the state it wants and for putting it back. The Method's rule holds — the
tool measures, the operator decides.
"""
from __future__ import annotations

import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import common  # noqa: E402

#: One arm: a callable that performs one measured repetition and returns a
#: dict with at least "value" (the measured number, lower-is-better for
#: render_s / shot_s, higher-is-better for tflops) and optionally "image"
#: (a path to a rendered frame, used for the ssim guardrail).
Arm = Callable[[int], Dict[str, Any]]


@dataclass
class ArmResult:
    """What one arm produced over the whole paired run."""

    name: str
    values: List[float] = field(default_factory=list)
    #: One list of image paths per repetition. A repetition of a shot produces
    #: several frames, so this is a list of lists rather than a flat list.
    image_sets: List[List[str]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.values)

    def summary(self) -> Dict[str, float]:
        if not self.values:
            return {}
        return {
            "n": float(len(self.values)),
            "mean": statistics.fmean(self.values),
            "median": statistics.median(self.values),
            "min": min(self.values),
            "max": max(self.values),
            "stdev": statistics.stdev(self.values) if len(self.values) > 1 else 0.0,
        }


@dataclass
class PairResult:
    """A paired A/B run: both arms, their differences, and the drift check."""

    baseline: ArmResult
    candidate: ArmResult
    diffs: List[float] = field(default_factory=list)
    #: How much the BASELINE moved between the first and last repetition.
    #: This is the number that would have caught the 2026-11-03 phantom: the
    #: baseline drifted, so the comparison was between two different states.
    baseline_drift: float = 0.0
    #: Wall seconds for the whole paired run.
    elapsed_s: float = 0.0
    notes: List[str] = field(default_factory=list)
    #: How many times the machine was busy enough to pause the run. Not a
    #: failure — but it belongs in the record, because a measurement taken
    #: across a contended window is a measurement of something more than the
    #: workload.
    contention_events: int = 0
    #: How many SAMPLES were taken on a busy card. Zero is the only clean
    #: result; anything else means the series contains contention, and the
    #: drift gate is what decides whether it is still usable.
    contended_samples: int = 0
    #: The names of the competing processes seen during the run. Kept in the
    #: record so a reader can tell WHOSE job spoiled a measurement, not just
    #: that one did.
    contenders: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.diffs)

    #: The names the raw series travel under into the record. They live here,
    #: next to the data they name, so the writer and any reader of the record
    #: agree on them by construction rather than by convention.
    SERIES_KEYS = ("baseline_series", "candidate_series", "paired_diffs")

    def raw_series(self) -> Dict[str, List[float]]:
        """The raw paired series, as plain lists of floats.

        A record that keeps only summary statistics cannot be CHECKED. The
        reader has to take the harness's word for the p-value, the effect and
        the drift — which is the position the ledger was in before the
        retraction: a number nobody can re-derive is a number that can change
        to suit the present. The raw series are what make a record
        self-verifying; every statistic downstream of them can be recomputed
        from the record alone.

        Floats, not numpy, and lists, not tuples: the record is JSON, and
        JSON has one number type and one array type.
        """
        return {
            "baseline_series": [float(v) for v in self.baseline.values],
            "candidate_series": [float(v) for v in self.candidate.values],
            "paired_diffs": [float(d) for d in self.diffs],
        }

    def effect_pct(self, lower_is_better: bool = True) -> float:
        """Fractional improvement of the candidate over the baseline."""
        if not self.baseline.values or not self.candidate.values:
            return 0.0
        b = statistics.fmean(self.baseline.values)
        c = statistics.fmean(self.candidate.values)
        if b == 0:
            return 0.0
        raw = (c - b) / abs(b)
        return raw if not lower_is_better else -raw

    def describe(self) -> str:
        b = self.baseline.summary()
        c = self.candidate.summary()
        lines = [
            f"baseline  {self.baseline.name:<28} mean {b.get('mean', 0):8.2f}  "
            f"cv {(b.get('stdev', 0)/b['mean']*100 if b.get('mean') else 0):5.2f}%  n={b.get('n', 0):.0f}",
            f"candidate {self.candidate.name:<28} mean {c.get('mean', 0):8.2f}  "
            f"cv {(c.get('stdev', 0)/c['mean']*100 if c.get('mean') else 0):5.2f}%  n={c.get('n', 0):.0f}",
        ]
        if self.diffs:
            md = statistics.fmean(self.diffs)
            lines.append(f"paired difference: mean {md:+.3f}  "
                         f"(n={len(self.diffs)} matched pairs)")
        drift = self.baseline_drift
        flag = "  <-- TOO MUCH: the machine changed under the measurement" \
            if self.drift_exceeded else ""
        lines.append(f"baseline drift across the run: {drift:+.2%}  "
                     f"— the number that would have caught a stale comparison{flag}")
        return "\n".join(lines)

    #: How much the baseline may move across a run before the measurement is
    #: refused as untrustworthy.
    #:
    #: The baseline is the CONTROL. If the control moved, the machine's state
    #: changed while we measured, and the comparison describes a machine that
    #: existed only for part of the run. Measured on this box during
    #: development: a second render was running concurrently, the baseline arm
    #: drifted -33.8% across five repetitions (152.6, 152.0, 152.3, 126.1,
    #: 101.0 TFLOPS), and the paired test still returned p=0.0146 — a
    #: "significant" win bought with a starved control.
    #:
    #: 5% is the threshold. The render path's own baseline drift, measured
    #: with nothing else running, was -1.29% over three repetitions; the
    #: compute path's was -0.20%. So 5% is comfortably above the noise of a
    #: quiet machine and well below the 12% class of change that used to
    #: produce ghosts.
    DRIFT_LIMIT = 0.05

    @property
    def drift_exceeded(self) -> bool:
        return abs(self.baseline_drift) > self.DRIFT_LIMIT

    #: How far the control strayed from its own middle, as a fraction.
    #:
    #: `baseline_drift` compares the FIRST repetition with the LAST, so it
    #: sees a MONOTONE change and nothing else. A busy window in the MIDDLE
    #: of a run leaves the first and last values equal, the drift reads
    #: +0.00%, and the gate passes a control that moved enormously. Measured
    #: through the gate on this box, a control slow for repetitions 2-4 and
    #: fast for 1 and 5 produced ADMIT +6.75%, +10.98%, +14.86% and +23.20%
    #: at window sizes of 52, 56, 60 and 70 seconds — every one of them a
    #: verdict that blamed the candidate for the machine's own change.
    #:
    #: This statistic cannot miss that: it is the largest deviation of ANY
    #: repetition from the median, so a window anywhere in the run shows up.
    #: It is also robust in the way that matters — the median is not itself
    #: pulled by the busy repetitions, so one slow repetition does not drag
    #: the reference the others are measured against.
    #:
    #: The old number is KEPT, under its own name, so records written before
    #: this change stay comparable and nothing silently changes meaning.
    @property
    def control_spread(self) -> float:
        """Largest deviation of any control repetition from the median."""
        vals = self.baseline.values
        if len(vals) < 2:
            return 0.0
        med = statistics.median(vals)
        if not med:
            return 0.0
        return max(abs(v - med) for v in vals) / abs(med)

    @property
    def control_exceeded(self) -> bool:
        """Did the control move anywhere in the run, not just end to end?"""
        return self.control_spread > self.DRIFT_LIMIT


class Pair:
    """Two arms, measured alternately in one session.

    The arms are run in the order baseline -> candidate on even repetitions and
    candidate -> baseline on odd ones. That ordering matters more than it
    looks: if one arm always ran second, it would always pay whatever warm-up
    the first arm caused (page cache, shader cache, GPU clock ramp), and the
    effect would be biased by construction rather than by the thing being
    tested.

    Alternating the order cancels that position effect ONLY WHEN THE NUMBER OF
    REPETITIONS IS EVEN. With an even count each arm runs second exactly as
    often as the other, so whatever the second position is worth cancels in
    the mean paired difference.

    With an ODD count it does not cancel. The candidate is second in
    ceil(n/2) repetitions and the baseline in floor(n/2), so if the second
    position is worth `delta` seconds the mean paired difference carries a
    residual of delta/n.

    The SIGN of that residual is NOT established here. It depends on whether
    the second position is faster or slower on the workload being measured,
    and one recorded three-repetition run is not enough to tell: it has a
    single second-position sample per arm, which is not separately
    identifiable from the linear trend across the run. So this module states
    the magnitude and declines to state a direction.

    The magnitude is an UPPER BOUND on the bias in the measured effect, and
    it is small: it is the second-position offset divided by the number of
    repetitions, so it shrinks as repetitions are added. At the objective's 5%
    min_effect it cannot on its own move a verdict across the line -- it eats
    into the margin rather than deciding the outcome.

    So an EVEN number of repetitions is the honest design: it cancels the
    effect exactly rather than leaving a residual of unknown sign. An odd one
    carries that residual, which this module does NOT correct, and a caller
    that wants the position effect gone must ask for an even count.
    """

    def __init__(self, label: str, baseline: Arm, candidate: Arm, reps: int = 5,
                 warmup: int = 1, lower_is_better: bool = True,
                 on_progress: Optional[Callable[[str], None]] = None,
                 baseline_name: str = "baseline", candidate_name: str = "candidate",
                 quiet_check: Optional[Callable[[], bool]] = None):
        self.label = label
        self.baseline_arm = baseline
        self.candidate_arm = candidate
        self.reps = reps
        self.warmup = warmup
        self.lower_is_better = lower_is_better
        self.on_progress = on_progress
        self.baseline_name = baseline_name
        self.candidate_name = candidate_name
        #: Called before each repetition; returns True if the machine was
        #: quiet enough to measure on. A job that starts mid-run is waited
        #: out rather than allowed to spoil the whole campaign — the drift
        #: gate would refuse the result anyway, and ten minutes would be
        #: gone.
        self.quiet_check = quiet_check
        #: How many times a repetition may be discarded and retried because
        #: the card was busy while it ran.
        self.max_attempts = 3
        #: How many repetitions had to be discarded.
        self.contended_reps = 0

    def _run_arm(self, arm: Arm, index: int) -> Dict[str, Any]:
        out = arm(index)
        if not isinstance(out, dict):
            raise TypeError(f"an arm must return a dict, got {type(out).__name__}")
        if "value" not in out:
            raise KeyError("an arm must return a dict containing 'value'")
        return out

    def _measure_arm(self, arm: Arm, index: int) -> Optional[Dict[str, Any]]:
        """Run one arm, watching for OTHER GPU work WHILE it runs.

        Waiting for quiet before a repetition is not enough: the repetition
        takes ~48 s, and a competing job that renders in waves can start in
        the middle of one. The drift check catches that afterwards and refuses
        the whole run — correct, but the run is lost.

        So each repetition checks WHO else is on the card while it runs, and
        a repetition taken on a contended card is DISCARDED and retried. The
        alternative is a series with one poisoned sample in it and a verdict
        nobody should trust.

        The check is PROCESS IDENTITY, not utilisation. During a Karma render
        this card is busy by design — measured at 94% utilisation by our own
        workload — so a utilisation threshold cannot tell our render from
        somebody else's. That mistake was made here: a repetition that took
        148 s where a clean frame takes 48 s was reported clean, because the
        sampler saw a busy card and our own render had made it busy.

        Asking who holds the card is the question that survives our own load.
        """
        for attempt in range(self.max_attempts):
            if self.quiet_check is not None:
                self.quiet_check()
            with common.ContenderWatcher() as watcher:
                out = self._run_arm(arm, index)
            if watcher.clean:
                out["_contaminated"] = False
                out["_others"] = []
                return out
            self.contended_reps += 1
            out["_others"] = list(watcher.contenders)
            if attempt + 1 < self.max_attempts:
                print(f"    rep {index + 1}: other GPU work was on the card "
                      f"during the measurement ({', '.join(watcher.contenders)}) — "
                      f"discarding it and retrying")
        # Every attempt was contaminated. Report it rather than pretending.
        out["_contaminated"] = True
        return out

    def run(self) -> PairResult:
        res = PairResult(baseline=ArmResult(self.baseline_name),
                         candidate=ArmResult(self.candidate_name))
        t0 = time.perf_counter()

        # Warm-up: the FIRST repetition of either arm pays for a cold page
        # cache, a cold shader cache and a cold GPU clock. Those are costs of
        # the machine being cold, not of the workload, so they are excluded
        # from both arms equally.
        for i in range(self.warmup):
            self._run_arm(self.baseline_arm, -1)
            self._run_arm(self.candidate_arm, -1)

        for i in range(self.reps):
            order = (self.baseline_arm, self.candidate_arm) if i % 2 == 0 \
                else (self.candidate_arm, self.baseline_arm)
            for arm in order:
                # Wait out contention BEFORE this repetition, and sample the
                # card WHILE it runs — a job that appears mid-repetition
                # would otherwise poison the series, and the drift gate can
                # only refuse the run after the fact.
                out = self._measure_arm(arm, i)
                which = "baseline" if arm is self.baseline_arm else "candidate"
                target = res.baseline if which == "baseline" else res.candidate
                if out.get("_contaminated"):
                    res.contended_samples += 1
                    for o in (out.get("_others") or []):
                        if o not in res.contenders:
                            res.contenders.append(o)
                target.values.append(float(out["value"]))
                imgs = out.get("images")
                if imgs is None:
                    imgs = [out["image"]] if out.get("image") else []
                target.image_sets.append(list(imgs))
                if out.get("extra"):
                    target.extra.setdefault(which, {}).update(out["extra"])
            if self.on_progress:
                self.on_progress(f"  rep {i+1}/{self.reps}: "
                                 f"baseline {res.baseline.values[-1]:.2f}  "
                                 f"candidate {res.candidate.values[-1]:.2f}")

        res.elapsed_s = time.perf_counter() - t0
        # Pair the arms by index. Both ran the same number of times, so this
        # is one-to-one.
        n = min(len(res.baseline.values), len(res.candidate.values))
        res.diffs = [res.candidate.values[i] - res.baseline.values[i] for i in range(n)]

        # Drift: did the baseline itself move during the run? If it did, the
        # machine's state changed while we measured, and any single-number
        # comparison against a baseline from another time is untrustworthy.
        if len(res.baseline.values) >= 2 and res.baseline.values[0] != 0:
            res.baseline_drift = (res.baseline.values[-1] - res.baseline.values[0]) \
                / abs(res.baseline.values[0])
        return res


#: The frame number in a rendered image's FILE NAME.
#:
#: husk expands `$F4` to a four-digit frame number, and shot.py names a split
#: render `{tag}_{label}_{frame:04d}.png`, so the trailing digits of the stem
#: are the frame. Pairing on the frame NUMBER rather than on the position in
#: the list is what stops a missing frame from silently re-pairing the rest of
#: the shot: shot.py drops a frame that failed to write (`if img.exists()`),
#: and a list-index pairing would then score frame 504 against frame 503 and
#: call it a clean run.
_FRAME_IN_NAME = re.compile(r"(\d+)\s*$")


def _frame_of(path: str) -> Optional[int]:
    """The frame number in a rendered image's file name, or None.

    None means the name does not say which frame it is, and the frames
    therefore cannot be paired by number. The guardrail refuses to measure
    anything it cannot pair: a score between two frames that are not the same
    frame is not a weaker measurement, it is a different measurement.
    """
    stem = Path(str(path)).stem
    m = _FRAME_IN_NAME.search(stem)
    return int(m.group(1)) if m else None


def ssim_series(baseline_sets: Sequence[Sequence[str]],
                 candidate_sets: Sequence[Sequence[str]],
                 notes: Optional[List[str]] = None) -> Tuple[List[float], List[float]]:
    """Cross-run SSIM between two arms' frames, paired frame for frame.

    Returns (baseline_series, candidate_series) as V12's guardrail plumbing
    expects: a tuple of (baseline-side samples, candidate-side samples).

    Both sides are measured AGAINST THE SAME REFERENCE IMAGE — the baseline's
    first frame of each frame number. The baseline's own score is therefore
    its self-consistency (how much the renderer varies between repetitions of
    the same frame), and the candidate's score is how far its frames sit from
    that reference. The guardrail then asks the right question: is the
    candidate's picture as close to the baseline's as the baseline's own
    repetitions are?

    Scoring the candidate against ITSELF cannot catch an altered image. That
    is not hypothetical: on this box the 16-spp data pass renders a completely
    different picture (cross-run SSIM 0.28 against the beauty pass) yet its
    repetitions are perfectly self-consistent at 1.0, so a self-comparison
    would have passed it.

    The bar, though, has to be a MEASUREMENT. The baseline side used to be
    "the reference against each baseline repetition", which for the reference's
    own repetition is the reference against ITSELF: 1.0 by construction, not by
    measurement, and it inflated the bar with a number that was never
    rendered. Across the four frames_per_process sessions on this box that
    free 1.0 pushed the baseline side's mean from the renderer's real
    run-to-run variation (0.9949) up to 0.9964 — and because the free 1.0 was
    diluted differently by each session's repetition count, the margin the
    guard had left to catch a real picture change moved with it: 1.03% in one
    session, 0.87% in another, for the same renderer and the same lever. The
    baseline side here is measured between two DIFFERENT renders of the same
    frame: the reference repetition against every other baseline repetition.
    A bar made of measurements, and a margin that does not depend on how many
    times the arm happened to run.

    Frames are paired by frame NUMBER, not by position in the list, and every
    frame on both sides is scored — not just the first. shot.py drops a frame
    that failed to write, so a list-index pairing would score frame 504
    against the candidate's frame 503 and report it as a clean run. A frame
    that is not on both sides is a failed measurement, and it is reported as
    one (in `notes`) instead of being quietly re-paired around.

    The mode difference
    -------------------
    The two arms do not always render the frame the same way. The
    frames_per_process lever renders the baseline one frame per process and
    the candidate as the whole shot in one process, and the same picture
    rendered the two ways is not bit-identical. Measured over every pair of
    renders on this box — 156 same-mode pairs and 169 cross-mode pairs per
    frame, no file compared with itself — the renderer's own run-to-run
    variation is 0.99484 (frame 503), 0.99497 (504) and 0.99517 (505), and
    the cross-mode distance is 0.99483, 0.99491 and 0.99515. The offset is
    0.00001 to 0.00006, i.e. 0.001% to 0.006% of the score, and far inside
    the renderer's own spread (0.99380 to 0.99650). The mode difference is
    therefore NOT a separate floor that a correct render falls below: it is a
    rounding error against noise the renderer has anyway.

    It is still charged to the candidate, and that is deliberate. The only
    quantity that sees the candidate's picture is the cross-mode distance
    itself, so any offset measured from the candidate's own data would be a
    constant taken from the thing being judged. Telling the mode difference
    apart from a changed picture needs a third render — a baseline rendered in
    the candidate's mode — which would be a change to the MEASUREMENT, not to
    the scoring. So the offset is measured and reported (in `notes`) rather
    than subtracted, and the levers whose arms DO render the same way are
    unaffected by it.

    Measured on this box: the hero frame rendered standalone and the same
    frame rendered inside a sequence score 0.994 against each other — the same
    picture, different sample ordering — which is the noise floor the
    guardrail has to tolerate.

    Fails closed. If PIL or numpy is missing, if an image cannot be read, if
    a frame is missing from one side, or if there are not two baseline
    repetitions to measure the renderer against, this returns ([], []) and
    V12's objective denies the change: a declared guardrail that was not
    measured is never skipped.
    """
    def _note(msg: str) -> None:
        if notes is not None:
            notes.append(msg)

    try:
        import numpy as np
        from PIL import Image
    except Exception as e:
        _note(f"ssim guardrail NOT MEASURED: PIL or numpy is unavailable "
              f"({type(e).__name__}: {e})")
        return ([], [])

    def _load(p):
        if not p:
            return None
        try:
            return np.asarray(Image.open(p).convert("L"), dtype=np.float64)
        except Exception:
            return None

    def _ssim(x, y):
        if x is None or y is None or x.shape != y.shape:
            return None
        h, w = (x.shape[0] // 8) * 8, (x.shape[1] // 8) * 8
        if h == 0 or w == 0:
            return None
        x, y = x[:h, :w], y[:h, :w]
        bx = x.reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))
        by = y.reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))
        vx = x.reshape(h // 8, 8, w // 8, 8).var(axis=(1, 3))
        vy = y.reshape(h // 8, 8, w // 8, 8).var(axis=(1, 3))
        cov = ((x.reshape(h // 8, 8, w // 8, 8) - bx[:, None, :, None])
               * (y.reshape(h // 8, 8, w // 8, 8) - by[:, None, :, None])).mean(axis=(1, 3))
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        return float((((2 * bx * by + c1) * (2 * cov + c2))
                      / ((bx ** 2 + by ** 2 + c1) * (vx + vy + c2))).mean())

    if not baseline_sets or not candidate_sets:
        _note("ssim guardrail NOT MEASURED: one of the arms produced no images")
        return ([], [])

    def _index(sets, which):
        """frame number -> {repetition index: path}, one dict per repetition.

        None if the file names do not carry frame numbers, or if one
        repetition carries two images of the same frame: either way the frames
        cannot be paired by number and nothing can be measured honestly.
        """
        out: List[Dict[int, str]] = []
        for i, s in enumerate(sets):
            d: Dict[int, str] = {}
            for p in s:
                f = _frame_of(p)
                if f is None:
                    _note(f"ssim guardrail NOT MEASURED: the {which} image {p} does "
                          f"not name its frame, so the frames cannot be paired by "
                          f"number")
                    return None
                if f in d:
                    _note(f"ssim guardrail NOT MEASURED: the {which} repetition {i} "
                          f"holds two images of frame {f} ({d[f]} and {p})")
                    return None
                d[f] = p
            out.append(d)
        return out

    b_index = _index(baseline_sets, "baseline")
    c_index = _index(candidate_sets, "candidate")
    if b_index is None or c_index is None:
        return ([], [])

    # Every frame either arm rendered, on either side. A repetition that does
    # not hold all of them is missing a frame, and a missing frame is a failed
    # measurement — not something to re-pair around.
    frames = sorted({f for d in b_index for f in d} | {f for d in c_index for f in d})
    for which, index in (("baseline", b_index), ("candidate", c_index)):
        for i, d in enumerate(index):
            missing = [f for f in frames if f not in d]
            if missing:
                _note(f"ssim guardrail NOT MEASURED: the {which} repetition {i} is "
                      f"missing frame(s) {missing} — frames are paired by number, "
                      f"so a frame that is not on both sides is a failed "
                      f"measurement and not something to re-pair around")
                return ([], [])

    # The reference: the baseline's first repetition, frame for frame.
    refs: Dict[int, Any] = {}
    for f in frames:
        p = b_index[0][f]
        img = _load(p)
        if img is None:
            _note(f"ssim guardrail NOT MEASURED: the baseline reference frame {f} "
                  f"could not be read ({p})")
            return ([], [])
        refs[f] = img

    def _score_against_refs(paths: Dict[int, str], which: str,
                             exclude: Sequence[str] = ()) -> Optional[List[float]]:
        """One score per frame, or None if a frame cannot be read.

        A frame whose file IS the reference's file is skipped rather than
        scored: comparing a render with itself is 1.0 by construction, and
        the point of this function is that every number it returns was
        measured between two different images.
        """
        out: List[float] = []
        for f in frames:
            if paths[f] in exclude:
                continue
            img = _load(paths[f])
            if img is None:
                _note(f"ssim guardrail NOT MEASURED: the {which} image {paths[f]} "
                      f"could not be read")
                return None
            s = _ssim(refs[f], img)
            if s is None:
                _note(f"ssim guardrail NOT MEASURED: the {which} image {paths[f]} "
                      f"cannot be compared with the reference frame {f} "
                      f"(different size, or too small to score)")
                return None
            out.append(s)
        return out

    base_side: List[float] = []
    for i, d in enumerate(b_index):
        # The reference repetition against ITSELF is 1.0 by construction, not
        # by measurement. It is not a sample of the renderer's variation and
        # it is not in the bar.
        if i == 0:
            continue
        scores = _score_against_refs(d, f"baseline repetition {i}",
                                     exclude=tuple(b_index[0].values()))
        if scores is None:
            return ([], [])
        if scores:
            base_side.append(statistics.fmean(scores))

    if not base_side:
        _note("ssim guardrail NOT MEASURED: the bar is the renderer's variation "
              "between repetitions, and there is only one baseline repetition to "
              "measure it with")
        return ([], [])

    cand_side: List[float] = []
    for i, d in enumerate(c_index):
        scores = _score_against_refs(d, f"candidate repetition {i}",
                                     exclude=tuple(b_index[0].values()))
        if scores is None:
            return ([], [])
        if scores:
            cand_side.append(statistics.fmean(scores))

    if not cand_side:
        _note("ssim guardrail NOT MEASURED: the candidate produced no readable frames")
        return ([], [])

    # The candidate arm's OWN run-to-run variation. Not the bar — the bar has
    # to be independent of the candidate's picture, or a candidate that renders
    # a consistently different picture would set its own bar and pass. But it
    # is what says whether the gap between the two sides is the mode
    # difference or a changed picture.
    internal: List[float] = []
    if len(c_index) >= 2:
        c_refs: Dict[int, Any] = {}
        for f in frames:
            img = _load(c_index[0][f])
            if img is None:
                break
            c_refs[f] = img
        if len(c_refs) == len(frames):
            for i, d in enumerate(c_index):
                if i == 0:
                    continue
                scores = []
                for f in frames:
                    if d[f] in c_index[0].values():
                        continue
                    img = _load(d[f])
                    if img is None:
                        break
                    s = _ssim(c_refs[f], img)
                    if s is None:
                        break
                    scores.append(s)
                if scores and len(scores) == len(frames):
                    internal.append(statistics.fmean(scores))

    bar = statistics.fmean(base_side)
    cand = statistics.fmean(cand_side)
    msg = (f"the renderer varies by {bar:.4f} between baseline repetitions (the "
           f"bar the candidate has to clear); the candidate's frames sit "
           f"{cand:.4f} from the baseline's reference")
    if internal:
        msg += (f"; the candidate's own repetitions agree at "
                f"{statistics.fmean(internal):.4f}")
    msg += (f". The gap ({cand - bar:+.4f}) is charged to the candidate: where "
            f"the two arms render in different modes (frames_per_process "
            f"renders the baseline one frame per process and the candidate as "
            f"the whole shot in one) that offset is real but cannot be told "
            f"apart from a changed picture without a third, mode-matched "
            f"render — a change to the measurement, not to the scoring")
    _note(msg)
    return (base_side, cand_side)
