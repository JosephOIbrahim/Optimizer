"""
constraint — work out what is actually limiting the card, before proposing levers.

The Field Notes (No. 01) opened with the finding that the 4090 "was innocent":
a Karma render drew 100 W of a 450 W budget and spent 0% of its life at the
power cap. Every firmware lever aimed at the card — clock locks, power
ceiling, P2 state — was aimed at a constraint that was not binding. The real
levers were all off the card: a 32-second setup tax paid on every launch, an
AOV pass that shaded pixels nobody looked at, and a render fed from the CPU.

That finding does not generalise to "ignore firmware." Measured on this box on
2026-11-04, the same card in the same session:

    fp16 GEMM at a 450 W ceiling   159.8 TFLOPS, 439 W drawn, 2-6% at the cap
    fp16 GEMM at a 600 W ceiling   174.6 TFLOPS, 548 W drawn, 0% at the cap
                                    -> +9.3%, the card WAS the constraint

    Karma render at a 450 W ceiling  48.1 s, 104 W drawn, 0% at the cap
    Karma render at a 600 W ceiling  no change; the card was never asked
                                    -> the card was NOT the constraint

So the question is not "is firmware worth trying" but "is the card the
binding constraint for THIS workload". This module answers that from
measurements rather than from assumption, because assuming wrong is what the
whole Method exists to prevent.

The classifier
--------------
    classify(throttle_deltas) -> Constraint

reads the throttle telemetry the harness already collects and says which of
the four bounds is doing the work:

    POWER      the card is pinned at its ceiling
    SETUP      per-launch overhead dominates (the card is idle while the
               scene is compiled and loaded)
    CPU        the card is idle while the CPU feeds it
    HEADROOM   nothing is binding; the workload is simply what it is

The measurement that distinguishes SETUP from CPU is the pair
(frames per process, wall seconds per frame): if the FIRST frame in a process
costs far more than the ones after it, the difference is setup, and it is
amortizable by rendering more frames per process. If every frame costs the
same and the card is idle, the CPU is the bottleneck and no amount of GPU
power will help.
"""
from __future__ import annotations

import enum
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


class Constraint(enum.Enum):
    """What is actually limiting the workload."""

    POWER = "power"
    SETUP = "setup"
    CPU = "cpu"
    HEADROOM = "headroom"

    def describe(self) -> str:
        return {
            Constraint.POWER: (
                "the card is pinned at its power ceiling — more watts would buy "
                "more throughput, and the card's allowed range may allow it"),
            Constraint.SETUP: (
                "per-launch overhead dominates: the first frame in a process "
                "costs far more than the rest, so rendering MORE frames per "
                "process amortizes it"),
            Constraint.CPU: (
                "the card is idle while the CPU feeds it — GPU power, clocks "
                "and P2 state cannot help this workload"),
            Constraint.HEADROOM: (
                "nothing is binding — the workload is simply what it is, and "
                "the honest levers are scene-side"),
        }[self]


@dataclass
class ConstraintVerdict:
    """What binds, how strongly, and what that implies."""

    constraint: Constraint
    #: Fraction of wall time the card sat at its power ceiling.
    power_frac: float = 0.0
    #: Mean power drawn, watts.
    avg_power_w: float = 0.0
    #: The ceiling that power was drawn against, watts.
    power_limit_w: float = 0.0
    #: Per-frame wall times from a multi-frame render, in order.
    frame_times: List[float] = field(default_factory=list)
    #: First frame's cost minus the median of the rest, seconds. This is the
    #: setup tax: the part of frame one that no later frame pays.
    setup_tax_s: float = 0.0
    #: GPU utilisation over the run, percent, when known.
    util_pct: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def power_headroom_w(self) -> float:
        return max(0.0, self.power_limit_w - self.avg_power_w)

    def summary(self) -> str:
        lines = [
            f"binding constraint: {self.constraint.value.upper()} — {self.constraint.describe()}",
            f"  power: {self.avg_power_w:.0f} W drawn of a {self.power_limit_w:.0f} W ceiling "
            f"({self.power_headroom_w:.0f} W unused), {self.power_frac:.0%} of wall time at the cap",
        ]
        if self.frame_times:
            first = self.frame_times[0]
            rest = self.frame_times[1:]
            med = statistics.median(rest) if rest else 0.0
            lines.append(f"  frames in this process: {[round(t, 1) for t in self.frame_times]}")
            lines.append(f"  first frame {first:.1f} s vs median of the rest {med:.1f} s "
                         f"-> setup tax {self.setup_tax_s:.1f} s per launch")
        for n in self.notes:
            lines.append(f"  {n}")
        return "\n".join(lines)


def setup_tax(frame_times: Sequence[float]) -> float:
    """The per-launch overhead, in seconds.

    The first frame in a process pays for compiling shaders, loading textures
    and building the BVH; every later frame reuses all of it. So the tax is
    (first frame) - (median of the rest). Zero or negative when there is only
    one frame, or when the first frame was not actually the expensive one.
    """
    ts = [float(t) for t in frame_times]
    if len(ts) < 2:
        return 0.0
    rest = ts[1:]
    return max(0.0, ts[0] - statistics.median(rest))


def classify(throttle_deltas: Optional[Sequence[Dict[str, float]]] = None,
             frame_times: Optional[Sequence[float]] = None,
             util_pct: float = 0.0) -> ConstraintVerdict:
    """Say what binds, from the measurements.

    `throttle_deltas` is the per-repetition telemetry the harness already
    collects (harness/throttle.py): power_frac, avg_power_w, power_limit_w.
    `frame_times` is the per-frame wall time of a MULTI-frame render, which is
    what separates setup overhead from a CPU-fed render.
    """
    deltas = list(throttle_deltas or [])
    power_frac = 0.0
    powers: List[float] = []
    limits: List[float] = []
    for d in deltas:
        power_frac = max(power_frac, float(d.get("power_frac", 0.0) or 0.0))
        if d.get("avg_power_w") is not None:
            powers.append(float(d["avg_power_w"]))
        if d.get("power_limit_w") is not None:
            limits.append(float(d["power_limit_w"]))
    avg_power = statistics.fmean(powers) if powers else 0.0
    limit = max(limits) if limits else 0.0

    ft = [float(t) for t in (frame_times or [])]
    tax = setup_tax(ft)
    notes: List[str] = []

    # The rules, in the order they can be trusted.
    #
    # 1. If the card sat at its ceiling for a meaningful part of the run, it
    #    was the constraint. This is the strongest signal available: it is the
    #    card saying so about itself.
    if power_frac >= 0.05:
        return ConstraintVerdict(constraint=Constraint.POWER, power_frac=power_frac,
                                 avg_power_w=avg_power, power_limit_w=limit,
                                 frame_times=ft, setup_tax_s=tax, util_pct=util_pct,
                                 notes=notes)

    # 1b. The card can also be effectively pinned without the duty-cycle
    #     counter noticing. `power_frac` is read from a violation counter over
    #     ONE short window, so a workload that draws 96% of the ceiling but
    #     never trips the counter for a full sample reads as 0%. Measured on
    #     this box: fp16 GEMM at a 450 W ceiling drew 433 W (96% of it) and
    #     reported 0% at the cap on some samples and 5.8% on others.
    #
    #     So the drawn fraction of the ceiling is the second signal, and it
    #     does not depend on the counter's sampling window. A card running
    #     within 10% of its ceiling has no power headroom to give, whatever
    #     the counter says.
    if avg_power and limit and avg_power >= 0.90 * limit:
        notes.append(f"the card drew {avg_power:.0f} W of its {limit:.0f} W ceiling "
                     f"({avg_power/limit:.0%}) — it has no power headroom left to give")
        return ConstraintVerdict(constraint=Constraint.POWER, power_frac=power_frac,
                                 avg_power_w=avg_power, power_limit_w=limit,
                                 frame_times=ft, setup_tax_s=tax, util_pct=util_pct,
                                 notes=notes)

    # 2. If the first frame cost far more than the ones after it, the setup is
    #    the tax, and it is amortizable.
    if ft and len(ft) >= 2:
        rest_median = statistics.median(ft[1:])
        if rest_median > 0 and tax > 0.25 * ft[0] and tax > 3.0:
            notes.append("the setup tax is the lever: render more frames per process "
                         "to pay it once")
            return ConstraintVerdict(constraint=Constraint.SETUP, power_frac=power_frac,
                                     avg_power_w=avg_power, power_limit_w=limit,
                                     frame_times=ft, setup_tax_s=tax, util_pct=util_pct,
                                     notes=notes)

    # 3. The card is not at its ceiling and the setup is not dominant. If it is
    #    also not busy, the CPU is feeding it.
    if util_pct and util_pct < 60 and avg_power and limit and avg_power < 0.5 * limit:
        notes.append(f"the card drew {avg_power:.0f} W of {limit:.0f} W "
                     f"({avg_power/limit:.0%} of the ceiling) — more power cannot help")
        return ConstraintVerdict(constraint=Constraint.CPU, power_frac=power_frac,
                                 avg_power_w=avg_power, power_limit_w=limit,
                                 frame_times=ft, setup_tax_s=tax, util_pct=util_pct,
                                 notes=notes)

    # 4. Nothing is obviously binding.
    return ConstraintVerdict(constraint=Constraint.HEADROOM, power_frac=power_frac,
                             avg_power_w=avg_power, power_limit_w=limit,
                             frame_times=ft, setup_tax_s=tax, util_pct=util_pct,
                             notes=notes)
