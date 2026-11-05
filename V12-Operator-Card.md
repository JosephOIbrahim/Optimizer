# V12 — Operator's Card

**Double-click:** `V12-Optimize-This-Machine.bat` → Yes on the admin prompt → menu.
Everything in the menu is read-only.

---

## Daily

```
[1]  Dry run, every change          what V12 would do, and why not
[7]  Preflight                      is anything heavy on the GPU?
[6]  Ledger                         what was applied, how much was measured
[9]  Self-test                      expect: Ran 454 tests ... OK
```

## To get a change applied

```
1  Close heavy GPU apps            menu [7] must say CLEAR
2  Time the workload, today        menu [5] → type the timings
3  Apply ONE change by hand, time again
4  V12-Optimize-This-Machine.bat run --objective vellum ^
       --candidate-for KEY=41.0,40.8,40.9
5  Admitted? add  --apply          a separate, deliberate step
```

`KEY` is printed under each decision in a dry run.
Vellum needs only timings. Diffusion and render also need a quality
guardrail on both sides (menu [5] does not ask for it; use the command line):

```
diffusion   guardrails: lpips, clip     render   guardrail: ssim
baseline --objective diffusion --record S --guardrail lpips=S --guardrail clip=S
run --objective diffusion --candidate-for KEY=S --guardrail-for KEY:lpips=S --guardrail-for KEY:clip=S
```

## What to measure first

```
V12-Optimize-This-Machine.bat plan --objective vellum
```

The order comes from `OptimizerV12\v12\data\catalog_relevance.json`.
**It is a Jev DRAFT until you review it.** Edit levels (0–3),
set `"reviewed": true`, set `"source": "human"` on rows you change.

## Jev

Off by default. On for one window: `set V12_JEV=on`, then run.
Labels print as `jev (not evidence)` and never decide anything.

## When it breaks

| You see | Do |
|---|---|
| `MOS2026 catalog NOT loaded (...)` | the reason is on that line |
| `process table unreadable` | tasklist failed. Fail-closed on purpose; rerun |
| `competing_processes: unread` | the GPU-memory counter needs pywin32. Launcher's python has it |
| Baseline refused after the update | old baselines used the old process rule. Record again |

## Undo

`D:\OptimizerV12\snapshots\` — `v12-pre-refactor-2026-10-01` (before everything),
`post-step1-2026-10-01` (before steps 2-4, incl. CLAUDE.md, config.yaml, v9 config).
