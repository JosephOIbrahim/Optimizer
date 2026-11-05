# Crucible — Adversarial Verification (Correctness + Performance/Quality)

**Read first:** `briefs/SHARED_CONTEXT.md`, all 4 `_handoff_*` files (Forge-1..4 deliverables).

You are split into two sub-roles. Same agent, but enforce role separation in your tests:

- **Crucible-1** → correctness. Edge cases, regressions, race conditions, fallback paths.
- **Crucible-2** → performance + quality. Measured speedup, LPIPS / CLIP quality gates.

## Authority boundary (Commandment 5 + 7)
- **You implement:** test files under `tests/` plus the `data/quality_ledger.db` schema and the per-model eval prompt sets.
- **You do NOT touch:** any production code under `acceleration/`, `core/`, `config.yaml`. If a test reveals a bug, write the test, file the failure as a blocker (`briefs/_blockers_crucible.md`), and **do not** weaken the test to make it pass.
- **You may not** be the same logical pass as the implementing Forge — the brief author who wrote the code does not write its tests in the same role.

## Crucible-1: Correctness suite (mandatory cases — Commandment 7)

Place under `tests/v8/`. Use pytest + pytest-asyncio.

### C1.1 — Cache invariants
- `lookup(unknown_key)` → None.
- Store→lookup roundtrip with 100MB plan bytes intact.
- Eviction frees ≥ requested bytes; refuses to evict in-use entry (last_used < 60s).
- `invalidate_on_driver_change("999.99")` evicts every entry with prior driver.
- Hash collision impossible across apps: cache keys for ComfyUI vs Houdini with otherwise-identical inputs produce different keys.
- Stale model invalidation: touch a model file; expect `invalidate_stale_models` to evict its engines.
- Concurrent store + lookup → no torn writes (use a partial-file injector test).

### C1.2 — BackgroundCompiler invariants
- Does not start a compile while GPU > idle threshold.
- Cancel mid-compile leaves no `.plan.tmp` orphans.
- Queue persists across simulated restart.
- Pause on app → resume restores progress.
- 3 backend exceptions in a row → job marked failed, doesn't loop forever.

### C1.3 — Specialist invariants
- ComfyUI specialist:
  - Process not running → empty recommendations.
  - Process running, all custom nodes installed, no flag → emits R1b only (`SET_LAUNCH_FLAG`).
  - API unreachable → returns empty list, does not raise.
  - Hot-shape histogram with cache hit → no R3 emitted.
- Houdini specialist:
  - ML SOP ONNX missing on disk → no R1; logs but no raise.
  - Karma XPU inactive → no R2.
- UE5 specialist:
  - NIM plugin marketplace URL missing from registry → R1 still emits with `notification` only, no install attempt.
  - DLSS DLL missing → no R2 emitted (only emits when present and old).
- CrossAppCoordinator:
  - All apps idle, history empty → R3 not emitted (no candidate to predict).
  - 3 apps simultaneously active → VRAM budget = 11/8/3 + 2GB system; sums to 24GB.

### C1.4 — Orchestrator V8 integration
- `python core/gpu_optimizer_v8.py --once --config config.yaml` exits 0 with no apps running.
- V7 baseline `python core/gpu_optimizer_v6.py --once` still works.
- ComfyUI running locally → at least one ComfyUI recommendation appears in cycle log.
- All 5 V7 + 4 V8 specialists present in `_initialize_experts` after init (with V7 ue5/houdini disabled and V8 versions enabled per Forge-3's config diff).

### C1.5 — Failure-injection (deliberately adversarial)
- Replace `pynvml` with a stub that raises → `gather_context` returns safe defaults (V7 invariant must hold).
- Inject `ConnectionError` on ComfyUI HTTP poll → no exception propagates.
- Truncate a `.plan` file mid-bytes → `lookup` returns None and self-evicts.
- Set cache_dir to a non-writable path → `EngineCacheManager` raises a clear blocker, doesn't silently succeed.

## Crucible-2: Performance + Quality (Pareto admission)

### C2.1 — Quality ledger schema (`data/quality_ledger.db`)

```sql
CREATE TABLE quality_evals (
  id INTEGER PRIMARY KEY,
  ts_utc TEXT NOT NULL,
  app TEXT NOT NULL,
  model_class TEXT NOT NULL,         -- "sdxl" | "flux" | "cosmos" | "wan" | "houdini_ml" | "ue5_nim"
  model_sha256 TEXT,
  backend_name TEXT,
  precision TEXT,
  shape_w INTEGER, shape_h INTEGER,
  reference TEXT,                    -- "torch_fp32" baseline
  speedup_vs_reference REAL,
  lpips_delta REAL,
  clip_delta REAL,
  fid_delta REAL,
  admitted INTEGER NOT NULL DEFAULT 0
);
```

### C2.2 — Eval set construction
For each model class, build a fixed set of:
- 8 prompts × 4 seeds = 32 generations per (backend, precision) combination.
- Same sampler, same steps as user's median historical config.
- Reference: torch fp32 baseline (computed once per model, cached).

### C2.3 — Pareto admission rule
A backend admitted only if:
```
speedup_vs_reference >= 1.30
AND lpips_delta <= 0.04
AND clip_delta >= -0.02
```

Backends below threshold are flagged in ledger as `admitted=0` and **must not** be recommended-default by ComfyUISpecialist (it queries `quality_ledger.fp8_admitted(...)` before emitting FP8 recommendations).

### C2.4 — Performance regression watchdog
- After every batch of 100 telemetry rows, run Welch's t-test (existing scipy dep) comparing the latest 30 vs the prior 70 for `samples_per_sec` per (model, shape).
- If `p < 0.05` AND mean delta < -10% → emit a regression alert. ComfyUISpecialist responds by enqueueing a `EVICT_ENGINE_CACHE` recommendation for that engine to force recompile.

### C2.5 — Phase A acceptance benchmarks (must run before declaring Phase A done)
Per the user's hardware (RTX 4090 sm_89), the following targets must be **measured**, not estimated:

| Workload | Baseline (torch fp16) | Phase A target | Pass criterion |
|---|---|---|---|
| ComfyUI SDXL @ 1024² × 25 steps | measured | ≥ 1.5× speedup | ≥ 1.4× actual |
| ComfyUI Flux @ 1024² × 25 steps | measured | ≥ 1.4× speedup | ≥ 1.3× actual |
| Houdini ML SOP (256³ inference) | measured | ≥ 1.5× speedup | ≥ 1.4× actual |
| UE5 Path Tracer 4K denoiser frame | measured | ≥ 1.1× speedup | ≥ 1.05× actual |
| V7 baseline regressions | n/a | None permitted | All V7 tests pass |

Failures here block Phase A sign-off; surface in `briefs/_blockers_crucible.md`.

## Sequence
1. Scout: read all four handoff files; map every public symbol.
2. Build C1.1 (cache) tests first — most isolated.
3. Build C1.2 (compiler) — depends on cache tests already green.
4. Build C1.3 (specialists) — uses fixtures with stub `SystemContext` instances.
5. Build C1.4 (orchestrator integration) — needs Forge-3 done.
6. Build C1.5 (adversarial) — final correctness pass.
7. C2 quality ledger + admission tests.
8. C2.5 acceptance benchmarks last.

## Bounded failure
- A test that won't run (missing fixture, missing handoff) 3× → blocker. Do not stub the missing piece to "make the test runnable" — that's exactly what Commandment 7 forbids.

## Done criteria
- All C1 tests green.
- All C2.5 measured speedups meet pass criterion, OR explicit blockers filed for failures.
- Quality ledger populated for every (app, model_class, backend) combination Forge-2 can recommend.
- V7 regression suite green (no regressions introduced by V8).

## Handoff (final)
Write `briefs/_handoff_crucible.md` reporting:
- Test counts (passed / failed / skipped per suite).
- Measured speedups vs targets.
- List of admitted backends per (app, model_class).
- Any open blockers.

This is the deliverable Joe reads to clear the V8 release.
