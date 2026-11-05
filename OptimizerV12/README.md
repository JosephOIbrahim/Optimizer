# OptimizerV12

One control plane over every actuator on this machine. V12 does not reimplement the
physics or the acceleration backends — V9 has the microbench-validated roofline and
four real backends, MachineOptimizer2026 has the reversible executor and the watchdog.
Both are kept. V12 is the layer neither has: **the one that owns the objective, the
baseline, and the admission gate — and that refuses to let any actuator move state
without a verdict.**

The defect it exists to fix, stated once:

> Both predecessors verify the wrong thing at the moment of action.
> MachineOptimizer2026 verifies that a change was **APPLIED** (read-back == target) and
> never that it was **FASTER**. V9 has a real Pareto gate and its actuator routes around
> it (`apply_all` is `recs[:top_n]`).

The gate is enforced by a **function signature**, not a policy check:

```python
def apply(self, change: Change, *, verdict: Verdict) -> ApplyResult
```

No verdict, no argument to pass, no mutation. Rollback is deliberately exempt — if a
change turns out to be wrong, the evidence that justified it may be exactly what failed.

Architecture and rationale: [`../DESIGN_V12.md`](../DESIGN_V12.md).

---

## Run it

```
..\V12-Optimize-This-Machine.bat
```

Launched by absolute path (a bare filename will not resolve). The launcher self-elevates,
picks an interpreter that can actually read the GPU, shows the machine and a preflight,
then offers the loop. Recording a baseline (`baseline --record`, launcher option 5) also
needs **pywin32** (`win32pdh`) in that interpreter: the competing-process reading comes from
the per-process GPU memory counter, and without it the baseline is refused as unread, never
recorded blind. `C:\Python314` has it; `py -3.13` on this box does not. To drive it directly:

```bat
python -m v12 <subcommand>
```

| Subcommand | Answers |
|---|---|
| `status` | what machine am I on, and what can it do |
| `preflight` | is anything else running that would corrupt a measurement |
| `truth` | live hardware truth, with per-field provenance (`--json`) |
| `kernels` | is the compile toolchain ready, and what is unreachable here |
| `baseline` | record / list measurement baselines (`--record '41.2,40.9,41.5'`) |
| `plan` | which candidate to time next for an objective (`--objective diffusion\|vellum\|render`); read-only |
| `run` | the loop — **dry run unless `--apply`** |
| `ledger` | what has been done, and how much of it was actually measured |
| `rollback` | undo. Requires no verdict. |
| `migrate` | Phase 0a — move MachineOptimizer2026's state off OneDrive |

**Nothing mutates without being told twice.** `run` is a dry run; `--apply` is a separate
invocation. Without `--objective` there is no baseline, and without a baseline every change
is refused at ADMISSION — V12 will not mutate a machine it cannot measure.

## Where it lives

| Path | What |
|---|---|
| `v12/` | the package |
| `v12/truth.py` | three readers (NVML · CUDA driver API · nvidia-smi), precedence merge, contradiction refusal |
| `v12/change.py` | `Change`, 14 kinds, observability/rollback/evidence axes, `change_key()` |
| `v12/admission.py` | `Gate`, `Verdict`, `GatedActuator` |
| `v12/plane.py` | the six-stage loop |
| `D:\OptimizerV12\` | ledger.db · baselines · kernels · locks · logs (deliberately **off OneDrive**) |

## Read the honesty metric first

```
Applied N | measured M | derived X | denied Z
```

Zero counts are left out. "12 applied" and "12 applied, 3 measured, 9 derived" are different
instruments. Only the second is usable for deciding whether to trust it. `CITED` evidence is
never admissible (there is no cited count: no operator-facing opt-in exists in this build);
`GUIDED` changes are never applied at all. A derived count appears only once something
supplies a per-change prediction, which the plane does not compute yet.

## Test it

```bat
python -m unittest discover -s tests -q
```

454 tests. `tests/test_truth.py` is the one to read if you are changing anything about the
hardware model — it asserts, among other things, the *weakness* of its own FLOP-table check
so the limit cannot be quietly forgotten. `tests/test_invariants.py` is the one to read before
touching anything near Jev (below). No test touches the network or `D:\OptimizerV12`.

## Jev annotations (opt-in, never evidence)

`v12/jev.py` can ask TypeSafe's Jev model (pinned `jev-1.13.0`) what a piece of OBSERVE or
DECIDE state *looks like*. The answer is a display annotation or an ordering hint. **It is
never evidence.** It never reaches the Gate, a Verdict, a Baseline, Conditions, an Objective,
a Change or an actuator, and the honesty metric never counts it. `tests/test_invariants.py`
enforces this. It imports the gate's modules in a fresh interpreter and fails if `v12.jev`
or `typesafe_sdk` loads, and it fails if any module other than `jev.py` names the SDK.

| | |
|---|---|
| Switch | `set V12_JEV=on`. Off otherwise. An API key being present is **not** consent. |
| Key | `TYPESAFE_API_KEY`, read by the SDK. V12 never passes or prints it. |
| Interpreter | the SDK (`typesafe-sdk` 0.7.0) is installed only for `C:\Python314`, the interpreter the launcher picks. |
| Limits | 3 s timeout, no retries, at most 40 questions per process; over the cap reads `unlabelled (cap)`. |
| Sent | redacted state only: secret-looking keys and flags (including `-p`, `--pass`, `--pw`), `NAME=value` and quoted `"key": "value"` secrets, the token after `Bearer`/`Basic`, URL `user:pw@`, long base64/hex runs and your profile path are masked before hashing or sending. |
| Recorded | judgments asked during a cycle's OBSERVE (`plane.observe`) go to the ledger's own `judgments` table, printed by `v12 ledger` under *Jev annotations - not evidence*, after the honesty metric. Process labels printed by `v12 preflight` and `v12 status` are display-only and **not recorded** (those commands never write the ledger). The relevance-drafting tool (`tools/draft_relevance.py`) records nothing in the ledger either: its judgments live only in its cache file under `D:\OptimizerV12` and in `v12/data/catalog_relevance.json`. |

`v12 status` and `v12 preflight` each print one line saying which state Jev is in:

```
jev: off (set V12_JEV=on to enable)
jev: on (model jev-1.13.0)
jev: unavailable (<reason>)      e.g. no API key, or SDK not importable in this interpreter
```

The jev line plays no part in preflight's CLEAR/BLOCKED decision.

**Process labels (display only).** Process identity is decided deterministically first: native
GPU-app images, launcher command lines, and a ComfyUI install found on disk. Only the
`python.exe`/`pythonw.exe` rows those rules leave unresolved, outside V12's own process lineage,
are sent to Jev, never `cmd` or PowerShell rows. Labels print after preflight's decision, prefixed
`jev (not evidence):`, and are skipped entirely when preflight blocks. At most 16 rows are asked
per call, and asking stops at the first offline answer. Competing-optimizer detection reads
`cmd.exe`/`python.exe` command lines from Win32_Process. That check is deterministic and can block.
Jev only labels what is left over.

## Measurement order (`v12 plan`)

Timing one change honestly takes minutes of quiet GPU, so the order matters. `v12 plan
--objective X` prints the dry catalog as a measure-next list: change_key, title, and an
**order-hint** 0-3 for that objective. It sorts by hint (highest first), then by catalog order.
Changes with no hint for X come last, in catalog order. Advisory-only changes are counted but
not listed, because they are never applied. `plan` builds no Plane and no Gate, takes no
snapshot and writes nothing. `run` still evaluates in catalog order: the hint orders what *you*
time first and nothing else.

The hints live in `v12/data/catalog_relevance.json`. They are static and reviewable, one entry
per change id, with `diffusion`/`vellum`/`render` levels (0-3 or `null` = unsure), a
`rationale` and a `source` of `jev-draft` or `human`. `v12/relevance.py` validates the file
whole. A bad level, a bool posing as a level, an unknown key or source, or a wrong schema is
refused, and `plan` falls back to catalog order with the reason printed. While the file says
`"reviewed": false`, every plan prints:

```
order: DRAFT, unreviewed (jev-drafted, not evidence)
```

Drafting the file is an offline step:

```bat
set V12_JEV=on
C:\Python314\python.exe tools\draft_relevance.py --show   &rem no API: question count + redacted states
C:\Python314\python.exe tools\draft_relevance.py          &rem ask, write reviewed=false, print a review table
```

It sends one Jev Score per (change, objective) on four concrete levels, from *does not act on this
workload's critical path* to *directly acts on the dominant cost of this workload*. A confidence
under 0.6 is written as `null`. An offline or error answer aborts the run without writing. Entries
a human wrote (`source: "human"`) are kept. Answers are cached in
`D:\OptimizerV12\cache\relevance_drafts.json`, keyed by (redacted-state hash, objective, model),
so a re-run asks only for what changed. The run raises jev's 40-question cap to exactly the
pre-counted number for its own process, and says so. To review: correct the levels, set `source`
to `human` where you changed one, then set `"reviewed": true`.

## What is not built

Stated rather than faked — see `DESIGN_V12.md` §7:

- **APP_GRAPH actuator** — tier and kinds exist in the ABI; nothing implements them.
- **KERNEL compile path** — probes honestly; currently blocked on `cl.exe` (MSVC) not on PATH.
- **`.exe` packaging** — needs network for PyInstaller, and the actuator registry builds
  itself with `pkgutil`/`importlib`, which **static analysis cannot see**. A naive build
  succeeds, launches, and fails at the first actuator.
