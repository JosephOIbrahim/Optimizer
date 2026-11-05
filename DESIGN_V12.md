# Optimizer V12 — A Gated Control Plane (one loop, one ABI, admission before action)

| Field | Value |
|---|---|
| **Disposition** | Re-architecture of the *action half* of the optimizer. The V11 substrate (measurement-driven control loop, roofline, Pareto gate, degrade-safe fallbacks) and MachineOptimizer2026's reversible executor are both KEPT as substrate. What V12 ADDS is the layer neither has: one control plane that owns the objective, the baseline, and the admission gate — and that no actuator can bypass. V9/V10/V11/MOS2026 stay intact as rollback. |
| **Authored** | 2026-09-16, ARCHITECT role. Recorded after the Mile 1–5 build rather than before it, so the numbers below are read off the machine, not quoted from a datasheet or a plan. |
| **Target HW (DETECTED)** | RTX 4090 sm_89 (Ada), 24 GB, **128 SMs**, 1008 GB/s, rated boost 2520 MHz (driver max 3105 MHz) · Threadripper PRO 7965WX 24c/48t · 128 GB · **driver 616.92** · CUDA 13.2 toolchain · HAGS ON. Read live by `v12 truth`; provenance per field. |
| **Basis** | First principles + this machine's own measurements (2026-09-16): 8192³ GEMM at four precisions under torch 2.8.0+cu126, plus direct `nvml.dll` / `nvcuda.dll` ctypes probes. No figure in §5 is taken from marketing material. |

---

## 1. The defect this exists to fix (why V12 exists)

Two working optimizers already sit on this machine. Neither is broken. Both fail in the same place.

> **Both predecessors verify the wrong thing at the moment of action.**
> MachineOptimizer2026 verifies that a change was **APPLIED** (read-back == target) and never that it was **FASTER**. V9 has a real Pareto gate and its actuator routes around it (`apply_all` is `recs[:top_n]`).

That is the whole thesis, and it is one sentence because it is one defect. MOS2026's executor is genuinely good — 37 recoverable changes, sha256-verified restore master, a real watchdog. Its verification step confirms the write landed. Nothing confirms the machine got faster. The operator reads "applied" and hears "improved." V11 went further and built the missing evidence half — a microbench-validated roofline, a Pareto gate, real backends — and then put the gate *after* the mutation: apply, measure, keep or discard. The machine is already changed by the time a verdict exists, so the verdict is a bookkeeping entry rather than a permission.

V12 inverts the order. **ADMIT sits before ACT**, and the actuator's `apply` takes the verdict as a required keyword-only argument. The refusal is in the function signature, not in a policy check that a later edit can quietly drop:

```python
def apply(self, change: Change, *, verdict: Verdict) -> ApplyResult
```

Two consequences fall out of that one line, and both are load-bearing:

- An actuator that cannot produce a verdict **cannot move the machine.** Not "should not" — cannot, because there is no argument to pass it.
- **Rollback does not require a verdict.** The asymmetry is deliberate: if a change turns out to be wrong, the evidence that justified it may be exactly what failed. Making undo depend on the evidence would make undo fail precisely when it is needed.

## 2. The loop (first principles)

```
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ OBSERVE  │─▶│  MODEL   │─▶│  DECIDE  │─▶│  ADMIT   │─▶│   ACT    │─▶│ REMEMBER │
   │ truth +  │  │ roofline │  │ eligible │  │  GATE    │  │ gated    │  │  ledger  │
   │ workload │  │ ceiling  │  │ changes  │  │ verdict? │  │ actuator │  │ + honesty│
   └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
    what is       what could    what would    is it         write it      what was it
    actually      this be       help, ranked  PROVEN        (only with    worth, and
    true?         at best?      by evidence   faster?       a verdict)    was it ever
                               class                                     MEASURED?
```

V11's loop was OBSERVE → MODEL → DECIDE → ACT → ADMIT. V12 moves ADMIT and inserts REMEMBER. `v12.plane.Plane.run_cycle` walks all six and returns a `CycleResult` that renders as a report; `Plane` is dry-run by default and `--apply` is a separate, explicit invocation.

**One loop, every layer.** The tiers differ in what they can touch, not in how they are governed:

| Tier | What it reaches | Actuator state |
|---|---|---|
| **SYSTEM** | registry, power plan, Defender exclusions, NVML clocks/offsets, NVAPI profiles | `actuators/system.py` — `Mos2026Bridge` delegates to MOS2026's proven executor |
| **APP_GRAPH** | application config (Houdini/UE5/ComfyUI) | ABI is defined (`Tier.APP_GRAPH`, `ChangeKind.APP_CONFIG`/`ENGINE`); **no actuator implemented yet — stated, not faked** |
| **KERNEL** | compiled artifacts, launch configs | `actuators/kernel.py` — probes the toolchain, compiles, caches to `D:\OptimizerV12\kernels`; blocked on `cl.exe` (see §7 Mile 6) |

## 3. `Change` — the universal ABI

Every layer's every action is one dataclass. `ChangeKind` carries 14 kinds spanning environment variables, registry, power plan, Defender, NVML clocks and offsets, NVAPI profiles, app config, runtime wrappers, engine settings, kernel artifacts, and `GUIDED` (advisory, never applied). `change_key()` gives the whole space one address format, so "what has already been tried on this machine" is a single query rather than four half-overlapping ones.

Three fields do the real work, because each encodes a distinction the predecessors collapsed:

**Observability** — can this change's effect be read back at all?

| Value | Meaning | Consequence |
|---|---|---|
| `READABLE` | a getter exists; read back and compare | rollback restores the prior value exactly |
| `RESET_ONLY` | no getter; you can only write | rollback can only reset to vendor default — **the operator's prior value is unrecoverable** |
| `LYING` | the read-back does not reflect reality | **`Change.verify_is_trustworthy()` is False — read-back==target proves nothing here, and the gate knows it** |

`RESET_ONLY` and `LYING` cost something real, so they cost a flag: `--allow-destructive-reset`. Six kinds are `ADMIN_KINDS` (`DEFENDER_EXCLUSION`, `NVAPI_PROFILE`, `NVML_CLOCK`, `NVML_OFFSET`, `POWER_PLAN`, `REGISTRY`) — the ones that reach outside the user's own session.

**Rollback** — two verbs, not one. `RESTORE_PRIOR` vs `RESET_DEFAULT` vs `REFUSE`. Collapsing these into a single "undo" is how an optimizer silently destroys an operator's hand-tuned setting while reporting success.

**EvidenceClass** — how much is actually known:

```
MEASURED  we ran it and timed it         → admissible
DERIVED   computed from the hardware model → admissible only with a measurement behind it
CITED     a source says so                → never admissible alone
GUIDED    advice for a human              → never applied; ChangeKind.GUIDED exists so it has a place to live
```

`CITED` never admits. A vendor whitepaper is not a verdict, and the `GUIDED` kind exists specifically so that unmeasurable advice has somewhere honest to go instead of being smuggled in as a change.

## 4. Admission — the gate

`v12.admission.Gate.evaluate(change, …)` returns a `Verdict` whose evidence class is explicit, and records every verdict — including every denial — to the ledger. The verdicts:

- **MEASURED** — a baseline exists, applies to this workload, and the candidate beat it by Welch's t-test with a minimum effect size. Guardrail metrics (VRAM peak, perf/watt) are checked in the same pass and can veto a speedup that costs too much elsewhere.
- **DERIVED** — the roofline predicts it; admissible, and labelled as prediction rather than proof.
- **CITED** — recorded, never applied.
- **DENIED** — with the reason, always.

A baseline is only usable if it *applies*: `Baseline.valid_for()` compares the workload fingerprint **and** the machine's `Conditions` (competing GPU processes, clock state) captured at baseline time, and refuses a comparison across a changed machine. This is what stops the classic false positive — "faster" that is actually "the other app finished." `guard.preflight()` (via `v12 preflight`) is the operator-facing half of the same idea:

```
on this machine there is no cross-process GPU partitioning. MIG is
datacenter-only, MPS is Linux/QNX-only, and TCC is unreachable on GeForce.
Two GPU workloads share the device, so any measurement taken while another
one runs describes neither.
```

Without an objective there is no baseline, and without a baseline every change is refused at ADMISSION. **V12 will not mutate a machine it cannot measure** — that is the intended behaviour, not a limitation to work around.

**The ledger's `judgments` table is not an evidence ledger.** It holds what the Jev judgment model said during a cycle's OBSERVE/DECIDE (label, probabilities, outcome, the sha256 of the redacted state it was shown). Labels printed by `v12 preflight` / `v12 status` and the relevance-drafting run are display-only and not recorded there. `Gate.evaluate` never reads it, `LedgerStats` / `honesty_report()` never count it, and `v12 ledger` prints it in its own section, *"Jev annotations - not evidence"*, after the honesty metric. A judgment cannot carry an evidence-class name as its label, at construction or at insert. See §8.

## 5. Truth — the model, and the three defects it shipped with

The gate is only as good as the numbers it reasons over, so `v12/truth.py` treats the hardware model as something to be *audited*, not just read. It queries three independent readers — NVML (`pynvml`), the CUDA driver API (`nvcuda.dll` via ctypes, always present with the driver, no pip install), and `nvidia-smi` — merges by authority precedence, and **refuses to publish any field two readers contradict.**

That structure was not defensive paranoia. It was earned. Three defects shipped in the first cut of this module, and each is worth naming because each is a *class*:

**1. SM count was 128× too high.** `nvmlDeviceGetNumGpuCores` returns **CUDA cores (16384)**, not **streaming multiprocessors (128)**. A well-named function answering a different question than the one it was asked — which is precisely why nothing complained. It now feeds the `cuda_cores / sm_count == lanes_per_SM` identity: 16384 over 512 is 32 lanes/SM, and no NVIDIA SM has 32 lanes. On that failure **both** values are dropped, not one, because a wrong-but-plausible SM count poisons every ceiling in the module and a missing one does not.

**2. Every FLOP rate was 2× too low,** halving every derived ceiling. This was refuted without any datasheet, by this machine's own measurement — which is the only kind of refutation that counts here:

```
8192³ GEMM, torch 2.8.0+cu126, 2026-09-16
  fp32    88.20 TFLOPS      tf32    90.33 TFLOPS
  fp16   165.26 TFLOPS      bf16   168.63 TFLOPS
```

The old table's fp32 rate of 128 FLOP/SM/clk would require a **5385 MHz** clock across 128 SMs to reach the 88.20 TFLOPS that was actually achieved. The card tops out at 3105. *A measurement cannot require a clock the silicon does not have.* Corrected to Ada's real per-SM rates (fp32 256, fp16/bf16 1024, tf32 512, fp8/int8 2048), with the fp16 rate pinned by Ada's fixed precision ratios once fp32 was settled. The check is written as a *required clock* rather than a pass/fail, because that is what is actually true.

**And a note on the honest limit of that check:** the fp16 half is **not** refutable by measurement — 512 FLOP/SM/clk needs only 2521 MHz, which the card can hold. `test_the_check_does_not_pretend_to_refute_the_old_fp16_rate` asserts the weakness explicitly so it cannot be quietly forgotten. Claiming more would credit the check with more than it tests, which is the same failure as a number that looks more authoritative than it is.

**3. A live clock was used where a maximum was required.** An idle 4090 sits at ~405 MHz on the memory clock. The module would have reported **39 GB/s** of bandwidth and a ridge point **26× too high** — while looking entirely reasonable. Live clocks are now never candidates for a peak, at any precedence: `sm_clock_for_peak_mhz` returns the *rated* boost (2520 MHz, CUDA attr 13), because no sustained kernel holds the driver's 3105 MHz maximum.

The result, read live and consistent across interpreters:

```
NVIDIA GeForce RTX 4090 | driver 616.92 | CC 8.9 | 128 SM @ 2520 MHz | 24.0 GiB | 1008 GB/s
peak @ 2520 MHz rated boost (driver max 3105 MHz, which no sustained kernel holds):
  fp32_cuda         82.6 TFLOPS ceiling  — measured  88.20 = 106.8% of ceiling  (over 100%:
                    the card boosts above its rated clock — the rated figure is a
                    guaranteed minimum, not a cap)
  fp16_tensor      330.3 TFLOPS ceiling  — measured 165.26 = 50.0% of ceiling
  fp8_tensor       660.6 TFLOPS ceiling  — no measurement on record; treat as a bound
  bandwidth       1008.1 GB/s    (384-bit x 10501 MHz x2)
  ridge fp16       327.6 FLOP/byte — below this a kernel cannot be compute-bound
```

That `106.8%` row is the one to read carefully. It is not an error and it is not rounded away: **GPU Boost lets a real card exceed its rated clock**, so a naive "rated clock = ceiling" assertion would fire a false alarm on healthy silicon. `BOOST_HEADROOM = 1.25` encodes that, because a false alarm is its own bug.

A fourth defect was caught in the same family and lives in its own module. `v12/power.py` exists because power was once sampled at 400 ms and reported as a "421 W average" — while `nvmlDeviceGetPowerUsage()` returns a **trailing 1-second boxcar average**, not an instantaneous reading. The series looked smooth and high-resolution and was neither. `PowerSampler` now prefers `NVML_FI_DEV_POWER_INSTANT` (field 186, ~100 ms floor on Ada), always records which sensor answered, and refuses to let a point sample and a 1-second mean be confused.

## 6. Act + Remember — three tiers, one ledger

Actuators are thin on purpose. `GatedActuator` wraps any inner actuator and is the only path to `apply`; it stamps the tier, refuses without a verdict, and records `landed` separately from `admitted` — because "the gate said yes" and "the write took" are different facts and MOS2026's defect was conflating a write's success with a performance gain.

`v12.ledger` is SQLite at `D:\OptimizerV12\ledger.db` and records verdicts (including denials), applications, reversions, measurements, and baseline indices. Its output is the honesty metric, which is the one number that should be read before any other:

```
Applied N | measured M | derived X | cited Y | denied Z
```

`LedgerStats.honest_note()` states plainly what the ratio does and does not mean. An optimizer that reports "12 optimizations applied" and has measured one of them is a different instrument from one that reports "12 applied, 1 measured, 9 derived, 2 cited," and only the second is usable for deciding whether to trust it.

`v12 rollback` deliberately takes **no verdict**. It lists what is currently applied with its `change_key` and evidence class, and it does not reconstruct a `Change` from the ledger by guessing — for the SYSTEM tier it hands off to MOS2026's own restore master (37 recoverable changes, sha256-verified), which is the artifact that actually has the prior values.

## 7. Phasing (mile markers)

- **V12-Mile 1 — `Change` ABI + the Gate. ✓ DONE.** `change.py` (14 kinds, observability/rollback/evidence axes, `change_key`), `admission.py` (`Gate`, `Verdict`, `GatedActuator` with the keyword-only verdict), `objective.py` (Welch's t-test with a hand-rolled Student-t CDF — no scipy), `baseline.py` (fingerprint + conditions validity). **The actuator cannot be called without a verdict, by signature.**
- **V12-Mile 2 — Truth. ✓ DONE + AUDITED.** `truth.py`, 1027 lines: three readers, precedence merge, contradiction refusal, physical bands, the lanes-per-SM identity, the FLOP table checked against measurement, `EXACT_FIELDS` so identities compare exactly and measurements get 2%. Found and fixed the three defects in §5 — **two of which were mine, found by measuring rather than by re-reading.** 45 tests in `test_truth.py`.
- **V12-Mile 3 — The loop. ✓ DONE.** `plane.py`: `observe` → `gather_candidates` (builtin catalog + MOS2026 catalog + kernel candidates) → `Gate.evaluate` → gated `apply` → ledger. Dry run by default; `--apply` is explicit; `CycleResult.report()` prints the whole six-stage walk with per-change evidence class.
- **V12-Mile 4 — Remember / honesty. ✓ DONE.** `ledger.py` with `honesty_report()` and `recent_applied()`; `guard.py` with `SingleInstance` (one loop at a time), collision detection, and coupling validation.
- **V12-Mile 5 — Phase 0a: state off OneDrive. ✓ DONE.** `paths.py` resolves the data root away from OneDrive to **`D:\OptimizerV12`** (ledger, baselines, kernels, locks, logs); `migrate.py` moves MOS2026's state with a journal, `Repair`, and a `verify()` that re-hashes — including a check for malformed identity holds. OneDrive sync on a live SQLite ledger is a corruption vector and a measurement confounder both.
- **V12-Mile 6 — KERNEL tier: compile path. ☐ PARTIAL, AND IT SAYS SO.** `actuators/kernel.py` probes the toolchain and reports readiness honestly rather than optimistically. Current live state: **`toolchain incomplete: cl.exe (MSVC) not on PATH — nvcc needs it for host compilation`**, 0 cached artifacts. It also enumerates what is unreachable *on this hardware, as distinct from misconfigured* — CuTe DSL (POSIX-only wheels), FP4/NVFP4 (sm_100+), thread-block clusters (CC 9.0+; hard-blocked on sm_89), cuTile+FP8 (`TileUnsupportedFeatureError` on the sm80 family). That list is the honest version of "NVIDIA Labs latest": half of it does not run on Ada, and a tool that claims otherwise is lying about its own coverage.
- **V12-Mile 7 — APP_GRAPH actuator. ☐ DEFERRED.** The tier and kinds exist in the ABI; nothing implements them yet. Documented, not faked.
- **V12-Mile 8 — `.exe`. ☐ DEFERRED.** `.bat` launcher is working (`V12-Optimize-This-Machine.bat`). The PyInstaller route needs network to install, and it carries a specific hazard worth writing down before anyone attempts it: **the actuator registry builds itself with `pkgutil.iter_modules` + `importlib.import_module`, which static analysis cannot see.** A naive PyInstaller build will succeed, launch, and then fail at the first actuator — which is the worst possible failure shape. Hidden-import collection must be explicit.

Dependency: **ABI ✓ → Gate ✓ → Truth ✓ → Plane ✓ → {Ledger ✓ ‖ Migrate ✓} → actuators (M6 partial, M7 open).** The loop closes today for the SYSTEM tier.

**Correction pass, 2026-10-01.** Several of the miles above were marked DONE while still carrying real bugs. Steps 1–3 fixed them. They are recorded here so the ✓ marks above are read with that history:

- **Step 1 — admission, baseline, ledger, bridge and guard bugs. ✓ 264 tests.** One example, from `tests/test_ledger.py`: the ledger's `recent_applied` selected six columns, but `v12 ledger` and `v12 rollback` indexed columns it had left out. That worked on an empty ledger and crashed on the first real application, which is exactly when rollback is needed. The full list is in that step's report, not in this document.
- **Step 2a — hardware facts. ✓ config and comments only.** The root `config.yaml`, `CLAUDE.md`, `v9/config.yaml` and MOS2026 `hardware_profile.py` now carry driver 616.92 Studio and CUDA UMD 13.4 (dated), no P/E-core split (CPU pinning is disabled under both V9 keys; it was latent, not active, because of a key mismatch), cooperative VRAM budgets that sum under ~22 GB because of the XDR compositor, and `thread_block_clusters: false`. None of these keys had a Python reader, so no behaviour changed.
- **Step 2b — measurement validity. ✓ 309 tests.** Baseline `Conditions` now name competing GPU processes from the per-process GPU memory counter (≥ 1024 MiB committed, or a significant app). Before, it used the whole nvidia-smi list, which changed whenever a desktop app opened. An unread list is `None`, never `[]`, and refuses the baseline. Guardrails now have a CLI path on both sides (`baseline --guardrail`, `run --guardrail-for`), and `baseline.compare()` (a second admission path that skipped guardrails) is gone. `run_cycle` always ends its session and releases its lock. `preflight` and `observe` share one process-table read.
- **Step 3 — Jev, isolated. ✓ 374 tests.** `v12/jev.py` is the one opt-in integration point for a judgment model (§8). It is off unless `V12_JEV=on`, pinned to `jev-1.13.0`, has a 3 s timeout, no retries, and a 40-question cap per process. Every response is validated (model, labels, finite probabilities that sum to 1). State is redacted before it is hashed or sent. Every printed field is sanitized against terminal escapes. There is an additive `judgments` ledger table that the honesty metric never reads. `tests/test_invariants.py` imports the gate's modules in a fresh interpreter and fails if `v12.jev` or `typesafe_sdk` loads. At Step 3 nothing consumed a judgment; Step 4 wired in the display uses below.
- **Step 4 — Jev display uses, and command-line optimizer detection. ✓ 442 tests.** Three things, and only the third can decide anything. (1) *Process labels:* python.exe rows the deterministic rules leave unresolved (`guard.unresolved_python`) get a Jev label, asked only after the preflight decision and not at all when it blocked. Display only: a label never moves a process in or out of `gpu_apps`, a collision, a note or `ok`. Labels asked in a cycle's OBSERVE are recorded in `judgments`; those printed by `v12 preflight` and `v12 status` are not recorded. (2) *`v12 plan` ordering:* `v12 plan` orders candidates by `v12/data/catalog_relevance.json`, which Jev drafted (`tools/draft_relevance.py`; its judgments live in that file and its cache, not the ledger). Only `cli.py` and `relevance.py` read it; the Gate never does, and the order changes what is timed next, never what admits. (3) *Competing optimizers by command line:* `guard.OPTIMIZER_COMMAND_LINE` reads cmd.exe/python.exe command lines, because a .bat or Python optimizer is only `cmd.exe`/`python.exe` by image name. It is deterministic, involves no Jev, and blocks a live cycle.
- **Repair pass after Step 4. ✓ 454 tests.** Baseline `competing_processes` now marks heavy entries (`python.exe:heavy`, `houdini.exe:heavy*2`), so a 15 GB ComfyUI job no longer reads the same as an idle stray python.exe; the method string changed, so earlier baselines are refused. The competing-optimizer check no longer exempts a reused pid as V12's ancestor (CreationDate is checked on each parent link). The redactor masks quoted JSON keys, bearer tokens and short password flags (`-p`, `--pass`, `--pw`).

## 8. Negative space (do NOT build)

- **A `config.toml`.** Nothing in the CLI reads one. Adding a config file that looks load-bearing and is not reproduces exactly the defect class this version exists to fix — a thing that appears to govern behaviour and doesn't. If one is ever added, something must load it, and the loading must be visible in the run report.
- **Datacenter research.** MIG, NVLink fabric, multi-node, collective comms. Irrelevant to one 4090, and §4's preflight note already says why MIG in particular is a dead end here.
- **Blackwell-only and Hopper-only features.** FP4/NVFP4, 2nd-gen Transformer Engine, thread-block clusters. Not "coming later on this box" — hard-blocked on sm_89 by the hardware.
- **A fourth optimizer.** There are already three. V12 is the layer over them, not a replacement for them, and any new code that reimplements a backend V9 already has is a regression in disguise.
- **Automatic remediation of a failed measurement.** If the numbers are bad, the answer is a worse-reported number, not a guess at a fix.
- **No judgment-model output is evidence.** Jev annotates OBSERVE/DECIDE only, opt-in via `V12_JEV=on`, enforced by `tests/test_invariants.py`. `v12/jev.py` is the only module that names `typesafe_sdk`, and imports it lazily. A judgment is a display annotation or an ordering hint. It never reaches `Gate.evaluate`, a `Verdict`, a `Baseline`, `Conditions`, an `Objective`, an `Observation` field used for control flow, a `Change`, or an actuator. A Choice's confidence is how concentrated the distribution is, not a probability of being right — the same reason `CITED` never admits. An API key being present is not consent; only the switch is.

## 9. Open items to verify at build

- **`cl.exe` / MSVC on PATH** to make the KERNEL tier real. Until then the tier is a probe, and it reports itself that way.
- **APP_GRAPH actuator design** — Houdini/UE5/ComfyUI config edits are line-oriented and mostly not read-back-verifiable, so several will land as `RESET_ONLY` or `LYING` and need the honest labels rather than optimistic ones.
- **`nvmlDeviceGetFieldValues` availability** on the installed driver for the power instant path; `power.py` prefers it and falls back, but the fallback's window should be visible in every report that uses it.
- **Ledger `Change` reconstruction** for `rollback --change <id>`. Today it refuses and hands off to MOS2026's restore master, which is correct-but-limited; a prior-value column in the ledger is the real fix.
- **PyInstaller hidden imports** for the dynamic actuator registry (§7 Mile 8), before any `.exe` is shipped as a convenience.
- **`DESIGN_V11.md`'s own hardware row** carries `driver 610.62`, the same stale value that `MachineOptimizer2026/optimizer/hardware_profile.py` was holding as if live until this session corrected it to 616.92. The V11 row is a dated record and is left as written; the profile file was not dated, and that difference is the whole defect class in one line.

---
*Gate: the ABI (§3) and the gate itself (§4) are the architecture — everything else is replaceable. Confirm those two, and the remaining miles (M6 compiler, M7 APP_GRAPH, M8 packaging) are additive rather than load-bearing.*
