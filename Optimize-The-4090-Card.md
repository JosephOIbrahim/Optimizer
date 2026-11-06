# Optimize-The-4090 — Operator's Card

**Double-click:** `Optimize-The-4090.bat`. Ask what you want faster; it measures
the machine first and only proposes levers that can pay.

Nothing the tool applied is left applied. The power ceiling is raised only for
the measurement and put back exactly afterwards — and the put-back is *checked*,
so a failed restore is reported instead of assumed away.

> `The-Method.bat` is retired and now delegates here. If you have it pinned
> to a shortcut, it still works — it just hands over.

---

## What it does

```
[1] RENDER A SHOT    N frames of a Houdini/Karma scene
[2] RENDER ONE FRAME a single hero frame
[3] COMPUTE          GEMM / tensor throughput
[4] PROFILE ONLY     what is limiting this machine right now  (no levers run)
[5] SPECS            the machine's own numbers
[Q] Quit
```

Command line, if you prefer:

```
Optimize-The-4090.bat --request render|shot|compute|profile
                       [--usd PATH] [--settings PRIM] [--frames N]
                       [--start-frame N] [--samples N] [--res-scale N]
                       [--reps N] [--warmup N] [--levers a,b]
                       [--variant beauty_only|no_crypto] [--threads N]
                       [--probe] [--quiet-timeout SECONDS]
```

`--request` defaults to render, so an operator who passes only `--usd` gets a
render request without being asked. `--reps` is floored at the objective's own
minimum — 5 per side for render and compute, 3 for shot — so raising it below
that changes nothing; `--warmup` discards repetitions before measuring, and
`--quiet-timeout` is how long the tool waits for the card to go idle before it
starts. That timeout is a budget for the **whole campaign**, not per phase:
the wait before the lever and the wait before every repetition all draw on
it, so a contended machine waits the timeout once rather than the timeout
before the lever and again before each repetition. `--probe` reports what is
binding and measures nothing. The default
`--reps 3` stays odd on purpose: every lever already runs an odd count, so an
even default would change only the shot path, and the residual it removes
measures about 0.07% of the metric against a 5% threshold — too small to flip
a verdict, and its sign is not established. See the position-bias note under
rule 1.

**Every campaign commits its measurement records.** After the run, the
records are copied into `measurements/` in the repo and committed, so the
copies this card quotes as evidence cannot go stale. It is not a flag — a
record that is cited here and not committed is a claim nobody can check.

The commit says what the campaign *found*, not that a sync happened:

```
Measure the karma-xpu workload (render)

campaign
  aov_variant  +10.85%  p=1.13e-04  n=5  drift -0.34%  spread 2.00%  (beauty_only)

New records (1):
  optimize_20261106_071113.json
```

Only `measurements/` is committed, so any source edit you have in flight is
left exactly as it was — even if you have it *staged*. To refresh the copies
without measuring anything:

```
Sync-Measurements.bat            copy the records in and commit them
Sync-Measurements.bat --check    report what would change, change nothing
```

---

## The numbers it produced on this box (2026-11-04 to 11-06)

```
RTX 4090  24 GiB  128 SM  driver 621.14
power ceiling 450 W  (allowed 150-600 W)   150 W of headroom above the current ceiling
```

(The first ten records were taken on driver 617.42; every record since
`optimize_20261104_195032.json` is 621.14. A driver's change is one of the
things a baseline records, because a baseline measured on a different driver
cannot be compared with one measured on this one.)

| Request | Lever | Result | Mechanism |
|---|---|---|---|
| compute | power ceiling 450→600 W | **ADMIT +10.57%**, p=4.4e-06 | card was pinned at 450 W (454 W drawn) |
| shot (3 frames) | frames per process 1→3 | **ADMIT +46.95%**, p=2.4e-06 | ~39 s setup tax paid once, not three times |
| render | AOV variant (beauty only) | **ADMIT +10.85%**, p=1.1e-04 | shades fewer pixels, same beauty image |
| render | AOV variant (no crypto) | **DENY -0.09%** | dropping cryptomatte alone is worth nothing |
| render | power ceiling | *not measured* — the card draws 104 W of a 450 W budget, 0% at the cap, so there is no headroom to unlock | throttle telemetry, not a lever verdict: no render+power_ceiling run exists |

Every verdict in this table was measured with both arms alternately in one
session, so these are the numbers to trust — not the 2026-11-03 ledger
figures, which were produced by the retired method and have been retracted.
The one row that carries no verdict is marked as such: no render workload was
ever run against the power lever, so the 104 W there is telemetry, not a
result.

**The card is not the constraint in every workload.** The render path never asks
the card for more than ~104 W, so a power lever there is a wasted afternoon. The
compute path is pinned at the ceiling, so the same lever is worth +10.57%. This is
why the tool asks what you want faster before it proposes anything.

Each request gets the lever whose *metric* its objective scores — a verdict
computed from the wrong quantity is not a verdict. `render` scores one frame's
shading (`render_s`), `shot` scores the whole shot (`shot_s`), `compute` scores
throughput (`tflops`).

---

## Before any measurement

```
[4] PROFILE ONLY    says what is binding, without running a lever
```

The harness also refuses to measure while another heavy app holds the GPU —
there is no MIG, no MPS and no TCC on this box, so two GPU workloads are
time-sliced against each other and any number taken then describes both.

---

## When it breaks

| You see | Do |
|---|---|
| `preflight BLOCKED` | V12 found a competing **optimizer**, or could not read the process table. Stop the other optimizer, or run the shell elevated so the table can be read. Ordinary GPU apps (Houdini, Nuke, a browser) do **not** block here — preflight runs without `--quiet`, so only a competing optimizer or an unreadable process table stops it. |
| `preflight BLOCKED` | Same thing. The process that collided is named below the line. |
| `CLEAR of optimizers — but GPU work IS resident` | A GPU workload is running and is named below the line. The measurement waits for it rather than refusing outright. |
| `other GPU work on the card` | Something else holds the card. The tool is waiting for it and says how long is left. |
| `the GPU is busy (peak` | The card was over 85% utilised before a repetition, so the tool waits for it to settle. |
| `still seeing other GPU work after` | The wait timed out and **the measurement was taken contended**. Treat the number as suspect, close the other job, and run it again. |
| `no such scene` | Check `--usd`. The default is the hero USD. |
| `variant is not on disk` | The AOV-reduced scene has not been built. `hython harness\variant_usd.py` builds it. |
| `unknown lever` | A name in `--levers` is not one of the five. The list is in the usage text. |
| `no lever is worth trying` | The card is not the constraint for this request. `--request profile` says what is. |
| `the card's ceiling is already at its maximum` | The card allows what it allows. Nothing to raise. |
| `could not set the power ceiling` | The card refused the value: it needs an elevated shell, or `nvidia-ml-py` is missing. This one **aborts the run**. |
| `this card accepts up to` but the lever did nothing | The workload is not power-bound. Check PROFILE. |
| `clears the threshold but n<` | The effect is real but there are too few samples to test it. `--reps` is floored at the objective's minimum — 5 per side for render and compute, 3 for shot — so raise it **above** that floor; below the floor it changes nothing. |
| `the BASELINE moved` | The control drifted (recorded drifts: +16.3%, -70.9%, +7.2%, -5.3%). Something else was using the GPU. Close it and run again — this check exists because a starved control once produced a "significant" p=0.0146 win. |
| `no images to compare` | The ssim guardrail could not be evaluated, so the picture is unverified. |
| `husk failed` | The scene path or the settings prim is wrong. `--settings` must exist in the USD. |
| `could not restore the power ceiling` | **The machine was not put back.** Set it by hand from an elevated shell: `nvidia-smi -pl <watts>`. The prior is in the record as `power_prior_w`. |
| `the power ceiling reads` | The ceiling was set back but reads a different value, so the card is **not** where you left it. Set it by hand. |
| `nothing was left applied` | The restore was verified and the read-back matched the prior. If you get the loud warning instead, the machine was *not* left as you found it. |

---

## The two rules this tool exists to keep

**1. Baseline and candidate are measured alternately, in one session.**
On 2026-11-03 this repo gated a thread-count sweep against a baseline
measured 48 minutes earlier, under different resident processes. It came back
ADMIT +12.34%, p=1.9e-05. Re-timed the same way, the same comparison was
**DENY -0.65%** — nothing. The tool now measures both arms alternately and gates
with a paired t-test, and prints the baseline's drift across the run — the
number that would have caught it.

Every lever also **waits for the card before each repetition**. That wait was
missing from the thread-count lever specifically — the lever that produced
the phantom was also the one with the least protection. It is now on every
lever, and a test pins it there so the gap cannot reopen.

The arm order alternates too, so neither arm is always second — but that
cancels the second position's worth exactly only at an **even** repetition
count. At an odd count one arm takes the second slot more often than the
other, leaving a residual of that offset divided by the repetition count.

**How big is that residual? Measured, not assumed.** There is exactly one run
on this box that keeps the raw series, so it is the only evidence there is:
`optimize_20261105_065602.json`, the shot lever at three repetitions. Remove
each arm's own trend first (a rising trend would otherwise be read as a
position effect), then compare the repetitions where an arm ran second
against the ones where it ran first. The second position costs the baseline
0.18 s and the candidate 0.34 s; the candidate runs second twice of three
times, so the mean paired difference carries a residual of about **0.11 s —
0.065% of the metric**, which is measured against the baseline mean the
effect is computed over.

**The sign is not established, and the card does not claim one.** Each arm's
estimate rests on a single second-position repetition, and the two disagree
by 0.16 s — more than the smaller estimate is. That is what a sign not
established looks like. What is established is the size: it is a real limit
on every verdict, it eats into the margin rather than deciding the outcome,
and it cannot by itself move a 5% effect across the line. `--reps` defaults
to 3, which is odd; an even count removes the residual entirely, and the
shot path is the only one that takes it (`--reps 4`). The derivation is
pinned in `tests/test_position_residual.py`, so the number here cannot drift
away from the record it came from.

**2. Nothing is compared against a picture of itself.**
The ssim guardrail scores the candidate's frames against the *baseline's*
frames, using the same reference on both sides. A change that alters the output
is caught. The 16-spp data pass renders a completely different picture
(cross-run SSIM 0.28) yet its repetitions are perfectly self-consistent at 1.0,
so a self-comparison would have passed it.

**3. A verdict from a moving control is refused, not reported.**
If the baseline drifts more than 5% across the run, the machine's state changed
while the measurement was taken and the comparison describes a machine that
existed for only part of it. This happened during development: a second render
was running, the control drifted **-33.8%** across five repetitions (152.6,
152.0, 152.3, 126.1, 101.0 TFLOPS), and the paired test still returned
p=0.0146 — a "significant" win bought with a starved control. The drift check
runs *before* the statistics.

It also refuses when the control's **spread** — its largest deviation from its
own middle — exceeds 5%. End-to-end drift compares the first repetition with
the last, so it sees a *monotone* change and nothing else: a busy window in the
*middle* of a run leaves the first and last values equal, the drift reads
+0.00%, and the gate passes a control that moved enormously. Reproduced
through the gate on this box, that shape produced ADMIT +6.75%, +10.98%,
+14.86% and +23.20% at window sizes of 52, 56, 60 and 70 seconds — verdicts
that blamed the candidate for the machine's own change. The spread cannot miss
a window anywhere, and a repetition known to have been taken while other GPU
work was on the card refuses the verdict outright.

Because a refusal after the fact wastes the whole campaign, the tool now
**waits for the GPU to go quiet before it starts**, and tells you when it is
waiting and why. If the machine never settles, it says so and measures anyway
— the drift gate remains the authority on whether the result is usable.

### How it decides the card is busy

**By who holds it, not by how hard it is working.** During a Karma render this
card is busy *by design* — measured at 94% utilisation by our own workload — so
a utilisation threshold cannot tell our render from somebody else's. The
detector asks **process identity**: which processes hold the card, minus our
own process tree (our renderer is a child of the harness, and it is the one
holding the card).

The check runs **during** each repetition, not just before and after, so a
rival that arrives mid-render and leaves again is still caught. It was not,
before: a repetition that took 148 s where a clean frame takes 48 s was
reported clean, because our own render had made the card busy.

Two things it knows about this box:

- **WSL counts.** GPU work inside a WSL distribution is invisible to the
  Windows process table, so the detector asks WSL directly. Found the hard
  way on 2026-11-04: the card sat at 98% and 288 W and *no* Windows process
  accounted for it — the work was a python3.12 inside a WSL Ubuntu.
- **The desktop compositor does not.** `dwm.exe` commits ~2.2 GB of GPU
  memory permanently and `csrss.exe` ~385 MB. Counting them would make every
  check a false positive, and a detector that always says "busy" is worse
  than none.

A second check — *how hard* the card is working — is used only **before** a
repetition starts, when our own workload is not holding the card. Its
threshold is calibrated to this desktop, measured over 113 samples:

```
desktop compositing    4-72% utilisation    32-70 W
a Karma render         94%+  utilisation   104 W+
a WSL benchmark        98%   utilisation   288-300 W
```

The threshold used to be 10%, which is *below* this desktop's own idle
burst — so the gate called an idle card busy and waited for a quiet that
never came. It is 85% now: clear of the desktop's worst burst, still
catching any real render or compute job.

| You see | Means |
|---|---|
| `other GPU work on the card (wsl:Ubuntu:1870: /python3.12)` | A WSL job holds the card. The tool is waiting for it. |
| `other GPU work on the card (houdinifx.exe:4962MiB)` | Houdini holds the card. The tool is waiting for it. |
| `nobody else is on the card now` | The wait is over; measuring starts. |

## The retracted verdicts

On 2026-11-03 the old method gated candidates against baselines measured in an
**earlier session**, under different resident processes, and recorded three
verdicts. None of them compared anything:

```
husk_j48        ADMIT  (+12.34%)  p=1.882e-05   baseline 48 minutes earlier
aov_beauty_only ADMIT  (+16.91%)  p=2.183e-10   baseline 2h40m earlier
aov_no_crypto   ADMIT  (+11.25%)  p=4.615e-07   baseline 2h50m earlier
```

Re-measured with both arms alternately in one session:

| Lever | Recorded | Measured properly | Verdict |
|---|---|---|---|
| husk_threads (-j 48) | +12.34% | **-0.65%** | DENY — never a win |
| aov_variant (beauty only) | +16.91% | **+10.85%** (p=1.1e-04) | ADMIT — a real win, a third smaller |
| aov_variant (no crypto) | +11.25% | **-0.09%** | DENY — not a win at all |

The two AOV reductions are **not** interchangeable: `beauty_only` removes every
extra pass and pays; dropping cryptomatte alone is worth nothing here. The
recorded +11.25% for `no_crypto` was drift between sessions.

All three were measured more than once, and the **last** time is the one to trust:

```
beauty_only     10:36   ADMIT  +9.19%   drift +1.45%   the card was contended
beauty_only     23:19   ADMIT  +9.79%   drift -1.82%   the card was verified idle
beauty_only     07:11   ADMIT  +10.85%  drift -0.34%   shared card, control steady

no_crypto       16:07   DENY   +0.85%   drift +1.59%   the card was contended
no_crypto       22:19   DENY   +0.77%   drift +0.31%   the card was verified idle
no_crypto       07:24   DENY   -0.09%   drift -0.75%   shared card, control steady

husk_threads    11:11   DENY   -0.89%   drift +1.96%   the card was contended
husk_threads    05:52   DENY   -0.60%   drift -0.28%   the card was verified idle
husk_threads    07:38   DENY   -0.65%   drift +0.11%   shared card, control steady

frames_per_proc 08:33   ADMIT  +47.23%  (no drift field)  old code, no warm-up
frames_per_proc 06:56   ADMIT  +46.95%  drift +2.12%   verified idle, paired

power_ceiling  15:23   ADMIT  +8.50%   drift -0.69%   old code, no settle
power_ceiling  09:42   ADMIT  +10.57%  drift -2.72%   verified idle, settle
```

The first of each was taken before the contention detector existed, on a
card a WSL benchmark was holding — invisible to it, and invisible to the drift
gate, which only sees the control. The second waited the competitor out by
name.

**The third round was taken on a SHARED card, and it still stands.** All three
2026-11-06 runs were taken with Nuke and Houdini **open** — idle, holding GPU
contexts, drawing ~12% utilisation — and the contention record says so in
every one: *"10 repetition(s) were taken while other GPU work was on the
card... The control held steady and the drift is inside the limit, so the
comparison stands"*.

That is the point, and it is worth being explicit about. Contention that
leaves the **control** steady can only make the effect look **smaller**,
never larger — a false DENY, not a false ADMIT. So it cannot manufacture a
win; it can only cost one. Whether the verdict stands is decided by the
control's drift and spread, and those are in the table: drift -0.34% /
spread 2.00%, drift -0.75% / spread 1.65%, drift +0.11% / spread 1.76% —
all well inside the 5% limit. This is the first measurement on this box taken
on a card with somebody else's work resident, and it is the first time the
"steady control, shared card" case has been exercised rather than avoided.

The 07:11, 07:24 and 07:38 runs are the authoritative ones for these three
levers: **beauty_only ADMIT +10.85%** (p=1.13e-04, n=5/5, drift -0.34%,
spread 2.00%), **no_crypto DENY -0.09%** (n=5/5, drift -0.75%, spread 1.65%),
**husk_threads DENY -0.65%** (n=5/5, drift +0.11%, spread 1.76%). The
conclusions do not change — beauty_only is a real win, the other two are worth
nothing — only the numbers move.

The shot lever's 08:33 run is superseded for a different reason: it was taken
under the old code, which had **no warm-up repetition**. The split arm launches
three husk processes, so its first repetition paid the cold-cache cost inside
the measured series, and that run carries no drift field to show it. The
authoritative run is 06:56 — **+46.95%, p=2.4e-06, n=3/3, paired, drift +2.12%,
zero contention events**. `Context.warmup` now defaults to 1, every lever uses
`max(ctx.warmup, 1)`, and `--warmup` is on the CLI. Without the warm-up the cold
start read as the machine changing, and the drift gate refused the campaign on
an idle card with zero contention events.

`husk_threads` was also the one lever with **no quiet gate** — it went
straight to measuring, repetition after repetition, without waiting for the
card. The lever that produced the phantom had the least protection of any
lever. It waits now, like the rest, and a test pins that.

The thread count genuinely does nothing here: the default (all 48 threads)
and `-j 48` are the same thing on a 48-thread box. Four of the five pairs
came out within 0.2 s of zero; the fifth was the renderer's own noise.

`beauty_only` is **confirmed** — two clean paired runs agreeing within about a
point (+9.79% on an idle card, +10.85% on a shared card with a steady
control), so the effect is **+10.3% ± 0.5**. `no_crypto` and `husk_threads`
reach the same conclusion three times. All three now rest on clean series:
n=5/5, a control whose drift and spread are inside the limit, and an ssim
guardrail showing the candidate's picture as close to the shared reference as
the baseline's own repetitions are.

Corrections chain. When a later measurement supersedes an earlier one, the
earlier one moves into a history chain with the reason it stopped being
current — and so do corrections to the retraction's own reason. A record that
erases its own history is the failure this whole mechanism exists to prevent.

Both measurements stay on the record; the superseded one is in the retraction's
history chain, with the reason it stopped being current.

All three are **retracted, not deleted** — the original text is still readable,
next to a note saying why it does not hold and what replaced it. The two
baselines they were gated against are **superseded**: their numbers stay on
disk (they were real measurements of a real machine) but they are skipped by
lookups and refused by the gate, so they can no longer gate anything.

```
v12 baseline                     # lists them, marked SUPERSEDED, with the reason
v12 ledger                       # same, in the honesty report
```

---

## Where things live

```
Optimize-The-4090.bat      the console
The-Method.bat              RETIRED - delegates here
Sync-Measurements.bat       copy the live records into the repo and commit them
harness/optimize.py         request -> objective -> lever -> verdict
harness/levers.py           every lever, and the ONE way to measure one
harness/paired.py           the paired A/B engine
harness/constraint.py       what is actually binding
harness/shot.py             the shot harness (frames per process)
harness/retract.py          withdraw a verdict, keeping the original visible
harness/sync_measurements.py  keeps the committed records current
harness/method.py           old Method - its candidate path now REFUSES
harness/lever_run.py        old lever runner - now REFUSES
OptimizerV12/v12/objective.py   the objectives and the paired t-test
OptimizerV12/v12/change.py       NVML_POWER_LIMIT: readable, so it restores
```

Measurements: `E:\OptimizerV12\measurements\` is the live store; the repo
keeps a committed copy of every record under `measurements/`, so a figure
quoted here can be checked against the file it came from. Run
`Sync-Measurements.bat` after a campaign to refresh those copies — it
compares content, not names, so it also catches a record edited in place.
Ledger (what was kept, what was measured): `v12 ledger` from the V12 launcher.

---

## Undo

The power ceiling is restored to its prior value after every run, and the
prior is read back to prove it. Three things make that promise real rather
than decorative:

- the prior is read **live**, immediately before the ceiling is raised — not
  from a snapshot taken when the tool started, which could already be stale;
- the restore is **checked twice**: the setter's own result, and a read-back
  compared against the prior;
- if either check fails, the run says so loudly and the record carries it
  (`applied: true`, `restore_error`). It does **not** print "nothing was left
  applied" when the restore failed, because that is the one moment the
  operator most needs the truth.

Nothing else is touched — with one exception. The `manual` lever asks you to
toggle **your** change between the two arms and never reverts it, because the
tool does not know what your change is. Put it back yourself.

### The card needs a moment to settle after a ceiling change

Raising the ceiling changes the card's **power state**, and the card re-tunes
its boost clock to the new limit. The first GEMM after the ceiling goes from
450 W to 600 W read **157.8 TFLOPS**, where the card settled at **177.5** —
an 11% deficit that has nothing to do with the lever.

So the power lever's arms now leave the card at the new ceiling for 1.5 s
before measuring. Without it the control moves around for no reason, and the
drift gate refuses the run — correctly, and with a stated reason ("something
else was using the GPU") that is wrong. A measurement that cannot tell a
settling card from a competing job is not measuring what it claims.

Only the power lever pays for this. It is the only one that changes the card's
power state between measurements.
