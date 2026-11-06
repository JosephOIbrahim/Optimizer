# Harness — Operator's Card

**Double-click:** `Measure-This-Machine.bat` — read-only measurement, nothing applied.
Numbers go to `E:\OptimizerV12\measurements\`. Code lives in `harness\`.

> **`The-Method.bat` is RETIRED (2026-11-04).** It now delegates to
> `Optimize-The-4090.bat`. See "Retired: The-Method.bat" below for why.

---

## Which door do I use?

| I want to | Use |
|---|---|
| measure, apply nothing | `Measure-This-Machine.bat` |
| make something faster, and have it proven | `Optimize-The-4090.bat` |
| see what the machine can do right now | `Optimize-The-4090.bat --request profile` |
| apply a change myself, and have it proven | `Optimize-The-4090.bat --request render --levers manual` |

---

## State as of 2026-11-06

```
render baseline   SUPERSEDED - both recorded render baselines are withdrawn:
                  f6d1f3438614 (54.43 s mean) and d87662056f00 (48.06 s mean).
                  Each was gated against candidates measured up to 2h40m later,
                  so neither can serve as a control; the numbers stay readable
                  with the reason attached. The harness takes its own control
                  in-session, so a lever no longer needs a stored baseline.
                  hero USD  E:\OptimizerV12\scenes\ovoid_hero_v002.usd  frame 503
                  settings  /Render/OVOID_CPU_RENDER_V013  (1024x429, HERO_40MM_CANDIDATE, XPU via -R)
                  conditions: Houdini open but idle. Re-record if you measure with it closed.
gemm baseline     V12 id 3fcb02b78c9b    159.90 TFLOPS mean, cv 0.17%
shot baseline     V12 id c7ceaa707f69    173.77 s mean for a 3-frame shot, cv 0.14%
diffusion         not recorded (needs ComfyUI interpreter + fixed-seed API workflow)
```

Measured levers on this box, each against its own same-session control:

```
frames_per_process   3 frames, 1 process each -> 1 process    +46.95%  p=2.4e-06
aov_variant          full AOV set -> beauty only               +10.85%  p=1.1e-04
power_ceiling        450 W -> 600 W (fp16 GEMM)               +10.57%   p=4.4e-06
power_ceiling        450 W -> 600 W (Karma render)              0%   (never at the cap)
```

---

## Before any number you intend to keep

```
[7]  Preflight            must print CLEAR. Close Houdini, Nuke, AE, Topaz, OBS, browsers.
```

---

## Record a baseline (once per workload, before touching anything)

```
[1]  Render      asks for the hero-frame USD (export once from Solaris) + frame
[2]  GEMM        no input; ~2 min
[3]  Diffusion   asks for an API-format workflow JSON with a FIXED seed; ComfyUI must be running
```
Each prints two paste lines at the end. Paste the **BASELINE** line into a terminal.

## Measure one change

**Do not compare a candidate against a baseline you measured earlier.** That is
the retired method, and it produced a ghost: see below.

```
1  Apply ONE change by hand (undervolt, P2 off, power limit, launch flag ...)
2  Optimize-The-4090.bat --request render --levers manual
3  The tool asks you to toggle the change between the two arms, so the
   control and the candidate are measured under the same conditions
4  V12 says admitted / denied, with the reason
```
Diffusion candidates also need `--reference` = rep01 PNG of the baseline run (the harness prints the path).

---

## Retired: The-Method.bat

`The-Method.bat` used to gate a candidate against a baseline read from disk —
a file usually measured in an **earlier session**, under a different set of
resident apps, at a different clock state. Drift between the two sessions was
charged to the candidate.

On 2026-11-03 that produced **ADMIT +12.34%, p=1.9e-05** for a thread-count
change worth **DENY -0.65%** when the baseline was re-timed the same way as the
candidate — nothing at all. The Field Notes call it "the move that evaporated";
the verdict sat in the ledger.

That verdict is now **retracted** (the original text is still readable, with a
note saying why it does not hold and what replaced it), and the two baselines
it was gated against are **superseded** — their numbers stay on disk, but they
are skipped by lookups and refused by the gate, so they can no longer gate
anything. See `Optimize-The-4090-Card.md` for the details.

Two more verdicts from that session were retracted the same way and for the
same reason — `aov_beauty_only` (+16.91%, gated 2h40m after its baseline)
and `aov_no_crypto` (+11.25%, gated 2h50m after its baseline). Re-measured
properly: the beauty-only AOV lever is **+10.85%** (p=1.1e-04) — still a real
win, a third smaller — while `no_crypto` alone is **-0.09%, DENY**: dropping
cryptomatte and nothing else is worth nothing on this workload. The two AOV
reductions are not interchangeable.

Both AOV figures were measured twice on 2026-11-04, the thread lever once more
on 2026-11-05, and all three once more on 2026-11-06. The first of each was
taken while a WSL benchmark held the card — invisible to the detector that did
not exist yet, and invisible to the drift gate, which only sees the control.
The second waited the competitor out by name:

```
beauty_only     10:36  ADMIT  +9.19%   drift +1.45%   contended
beauty_only     23:19  ADMIT  +9.79%   drift -1.82%   verified idle
beauty_only     07:11  ADMIT  +10.85%  drift -0.34%   shared card, control steady
no_crypto       16:07  DENY   +0.85%   drift +1.59%   contended
no_crypto       22:19  DENY   +0.77%   drift +0.31%   verified idle
no_crypto       07:24  DENY   -0.09%   drift -0.75%   shared card, control steady
husk_threads    11:11  DENY   -0.89%   drift +1.96%   contended
husk_threads    05:52  DENY   -0.60%   drift -0.28%   verified idle
husk_threads    07:38  DENY   -0.65%   drift +0.11%   shared card, control steady
frames_per_proc 08:33  ADMIT  +47.23%  (no drift field)  old code, no warm-up
frames_per_proc 06:56  ADMIT  +46.95%  drift +2.12%   verified idle, paired
```

**The 2026-11-06 runs were taken on a SHARED card, and they still stand.** All
three were taken with Nuke and Houdini **open** on the box, holding GPU
contexts (`Nuke17.1.exe x14:1559MiB`, `houdini.exe:7112MiB`). The contention
is recorded in every record, in the record's own words: *"10 repetition(s)
were taken while other GPU work was on the card... The control held steady and
the drift is inside the limit, so the comparison stands"*.

That is worth knowing, because it is the first measurement here taken on a
card with somebody else's work resident. Contention that leaves the **control**
steady can only make an effect look **smaller**, never larger — so it cannot
manufacture a win, only cost one. The drift and spread checks decide, and
those are inside the 5% limit in all three: drift -0.34% / spread 2.00%, drift
-0.75% / spread 1.65%, drift +0.11% / spread 1.76%. The authoritative runs are
the 11-06 ones — **beauty_only ADMIT +10.85%** (p=1.13e-04, n=5/5),
**no_crypto DENY -0.09%**, **husk_threads DENY -0.65%**. The conclusions do
not change: beauty_only is a real win, the other two are worth nothing. Only
the numbers move.

The shot lever's 08:33 run is superseded for a different reason: it was taken
under the old code, which had **no warm-up repetition**. The split arm launches
three husk processes, so its first repetition paid the cold-cache cost inside
the measured series, and that run carries no drift field to show it. The
authoritative run is 06:56 — **+46.95%, p=2.4e-06, n=3/3, paired, drift +2.12%,
zero contention events**. `Context.warmup` now defaults to 1, every lever uses
`max(ctx.warmup, 1)`, and `--warmup` is on the CLI. Without the warm-up the cold
start read as the machine changing, and the drift gate refused the campaign on
an idle card with zero contention events.

`beauty_only` is confirmed — two clean paired runs agreeing within about a
point (+9.79% on an idle card, +10.85% on a shared card with a steady
control), so the effect is +10.3% ± 0.5. The thread count does nothing:
default and `-j 48` are the same thing on a 48-thread box.

`husk_threads` was also the one lever with **no quiet gate** — it measured
repetition after repetition without waiting for the card. The lever that
produced the phantom had the least protection of any lever. Every lever waits
now, and `tests/test_lever_gates.py` pins it.

No statistical gate can catch that on its own — Welch's t-test on two
*independent* series cannot tell drift from effect. The fix is in the
measurement, and the retired launcher now delegates to a tool that does it:

- both arms measured **alternately in one session**, so drift cancels;
- the arm order alternates too, so neither arm is always second (that cancels exactly at an even repetition count; at an odd count
  the two arms take the second slot an unequal number of times, leaving
  a small residual whose sign is not established -- see harness/paired.py);
- the baseline's **drift across the run** is checked *before* the statistics,
  and a control that moved more than 5% **refuses** the verdict;
- the control's **spread** — its largest deviation from its own middle — is
  checked too, so a machine that changed *during* the run is caught whether the
  change was a steady slide from the first repetition to the last or a busy
  window in the middle that leaves the two ends looking identical;
- the ssim guardrail compares the candidate's frames against the **baseline's**,
  so a change that alters the picture cannot pass by being self-consistent.

The two old modules refuse to produce a verdict at all:
`harness\method.py candidate` and `harness\lever_run.py` both stop with an
explanation. The levers live in `harness\levers.py`, which every front door uses.

---

## Reading the throttle line

```
[at power cap 83%, avg 448 W, limit 450 W]   -> power-bound: undervolt / power limit can help
[at power cap 0%,  avg 300 W, limit 450 W]   -> not power-bound: those levers will do nothing here
[at power cap 0%,  avg 104 W, limit 450 W]   -> the card is idle; nothing you do to the
                                                  card will help this workload
```

## How the harness decides the card is busy

**By who holds it, not by how hard it is working.** During a Karma render this
card is busy *by design* — our own workload measured at 94% utilisation — so a
utilisation threshold cannot tell our render from somebody else's. The
detector asks process identity, minus our own process tree (our renderer is a
child of the harness; it is the one holding the card).

It polls **during** each repetition, so a rival that arrives mid-render and
leaves again is still caught. It was not, before: a repetition that took 148 s
where a clean frame takes 48 s was reported clean, because our own render had
made the card busy.

Two things it knows about this box:

- **WSL counts.** GPU work inside a WSL distribution never appears in the
  Windows process table, so the detector asks WSL directly. Found the hard
  way on 2026-11-04: the card sat at 98% and 288 W with *no* Windows process
  accounting for it — the work was a python3.12 inside a WSL Ubuntu.
- **The desktop compositor does not.** `dwm.exe` commits ~2.2 GB of GPU
  memory permanently and `csrss.exe` ~385 MB. Counting them would make every
  check a false positive.

Utilisation is a **second** check, used only *before* a repetition starts
(our own render makes the card busy by design). Its threshold is calibrated
to this desktop, measured over 113 samples:

```
desktop compositing    4-72% utilisation    32-70 W
a Karma render         94%+  utilisation   104 W+
a WSL benchmark        98%   utilisation   288-300 W
```

The threshold was 10% — *below* this desktop's own idle burst, so the gate
called an idle card busy and waited for a quiet that never came. It is 85%
now. If you ever see the harness sit at "the GPU is busy" on a card doing
nothing, this is the number to look at.

```
harness\common.py :: other_gpu_processes()   the detector
harness\common.py :: ContenderWatcher         polls during a repetition
harness\common.py :: wait_for_no_others()     the gate, before every repetition
```

## Self-test (after a driver update or if a number looks wrong)

```
C:\Python314\python.exe harness\selftest.py        expect: selftest: PASS
```

## When it breaks

| You see | Do |
|---|---|
| `preflight BLOCKED` | something heavy is on the GPU; the list is printed. Close it, rerun |
| `preflight: CLEAR of optimizers — but GPU work IS resident` | a GPU workload is running; it is named below the line |
| `other GPU work on the card (...)` | something else holds the card. The harness is waiting for it, and says how long is left |
| `the BASELINE moved -12.0% across the run` | something else was using the GPU. Close it and rerun — this check exists because a starved control once produced a "significant" p=0.0146 win |
| `REFUSING TO COMPARE ... ANOTHER SESSION` | you are on the retired path. Use `Optimize-The-4090.bat` |
| `this baseline was SUPERSEDED` | you are gating against a withdrawn reference. Record a fresh baseline — the old one cannot support the comparison |
| `torch venv not ready yet - see option [5]` | menu [5]; rebuild: `C:\...\Python312\python.exe -m venv E:\OptimizerV12\venv`, then `pip install torch --index-url https://download.pytorch.org/whl/cu130` |
| `Unable to load USD file` | path or encoding; re-export: `hython harness\export_hero.py --hip <file> --out E:\OptimizerV12\scenes\<name>.usd` |
| render_s shows `load 0.0, update 0.0` | husk did not print phase timers; render_s then = husk total incl. ~6 s load. Constant across reps, so verdicts still hold |
| `processes unreadable (not elevated)` | run from the elevated launcher for the full competing-optimizer check; preflight CLEAR still stands |
| `ComfyUI not reachable` | launch it first, the way you want measured |
| `guardrails unavailable` | lpips/open_clip missing in the venv; timing still records, V12 will deny until supplied |
| ssim/lpips wildly off between reps | the scene or workflow is not deterministic; fix the seed / disable progressive effects |

## Undo

Nothing to undo. The harness writes JSON and PNGs under `E:\OptimizerV12\measurements\`
only. The one change it can make — the power ceiling — is restored exactly
after every run, and the prior is read back to prove it.
