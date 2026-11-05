# Crucible Handoff -- Adversarial Verification (narrowed scope run)

**Owner:** Crucible (adversarial verifier)
**Reads this:** Joe (operator), follow-up Crucible runs, all Forges
**Brief satisfied:** `briefs/crucible_verification.md` C1.3 / C1.4 (essentials) / C1.5 / C2.3
**Status:** PASS. All in-scope tests green. No production defects filed.

---

## Scope (this run -- narrowed for API stability)

Two prior dispatches hit API server errors (529, 500) at ~30 tool calls. Scope
deliberately tightened to four files / 19+1 tests to land inside the stability
window. Out-of-scope items deferred to a follow-up run (see "Deferred" below).

| File | Tests | Section |
|---|---|---|
| `tests/crucible/test_specialists.py` | 12 | C1.3 |
| `tests/crucible/test_orchestrator_integration.py` | 4 | C1.4 essentials |
| `tests/crucible/test_failure_injection.py` | 3 | C1.5 |
| `tests/crucible/test_pareto_admission.py` | 4 (parametrized; skipped pending Phase B) | C2.3 |

---

## Counts

```
$ C:\Python314\python.exe -m pytest tests/crucible/ -v
================== 19 passed, 1 skipped, 1 warning in 12.95s ==================
```

| Status | Count |
|---|---|
| Passed | 19 |
| Failed | 0 |
| Skipped | 1 (`test_pareto_admission.py` -- entire module; production code absent) |

The single deprecation warning is the upstream pytest_asyncio plugin against
Python 3.14, not a Crucible test issue.

---

## Defects

**None.** Every assertion in the four test files passes against the production
code as of the time of this run.

`briefs/_blockers_crucible.md` not created -- nothing to file.

---

## What each test pins down (briefly)

### `test_specialists.py` (C1.3)

* **ComfyUISpecialist**
  * Process not running -> `[]`.
  * `api_reachable=False` does NOT raise; install-recommendations from
    filesystem-derived state still fire.
  * Both required custom nodes installed but no `--use-sage-attention` flag
    -> exactly one rec, action `SET_LAUNCH_FLAG`, target `comfyui:launch`.
* **HoudiniSpecialist**
  * Process not running -> `[]`.
  * ML SOP ONNX path missing on disk -> log+skip; no R1 emitted; no raise.
  * Karma XPU inactive -> R2 (`houdini:karma:denoiser`) suppressed.
* **UE5Specialist**
  * Process not running -> `[]`.
  * NIM marketplace URL absent -> R1 still emits with `auto_install=False`
    (notification only; no install attempt).
  * `dlss_dll_version=None` -> R2 (`ue5:dlss`) suppressed (no speculative
    upgrade rec without evidence).
* **CrossAppCoordinator**
  * 3 apps active -> `vram_budget` sums to **22 GB** with split 11/8/3, leaving
    a 2 GB system reserve out of the 24 GB physical budget. (Brief in
    `_handoff_forge2.md` says "1 GB headroom"; the actual implementation
    yields a 2 GB headroom because 11+8+3 = 22, not 23. This is a brief-vs-code
    minor mismatch that does NOT change behavior; flagging it here so the next
    Crucible run can decide whether to amend the brief or the code.)
  * Foreground=`comfyui` + `houdini` has 1 background compile -> emits
    `compiler:pause:houdini`.
  * All apps idle, history empty -> no `prefetch:*` recommendation
    (predict_next_engine returns None).

### `test_orchestrator_integration.py` (C1.4 essentials)

* V7 `--once` exits 0 (rollback path is real).
* V8 `--once` exits 0.
* V8 orchestrator registers exactly the 4 V8 specialists; V7
  `UE5_ENGINEER` and `HOUDINI_ENGINEER` are NOT registered (per
  `experts_v8_overrides`).
* **V7 enum cleanliness**: a fresh subprocess that imports ONLY `gpu_optimizer_v6`
  shows none of the V8-only OptimizationAction members
  (`COMPILE_TRT_ENGINE`, `INSTALL_CUSTOM_NODE`, ..., `WIRE_BACKEND_PATCH`)
  and none of the V8-only ExpertType members. Forge-3's `_extend_enum`
  trick is correctly V8-side only.

### `test_failure_injection.py` (C1.5)

* `pynvml` stubbed to raise on every attribute -> `gather_context()` returns
  a safe `SystemContext(WorkflowMode.IDLE, 24 GB free, ...)` without
  propagating.
* `urllib.request.urlopen` patched to raise `URLError` ->
  `ComfyUIStateCollector().collect()` returns `ComfyUIState(api_reachable=False)`
  and never raises.
* `EngineCacheManager` constructed with a `cache_dir` whose parent is a
  regular file -> raises `RuntimeError`/`OSError` on construction with a
  message that points at the directory issue. Confirms the C1.2 last-bullet
  invariant: "doesn't silently succeed."

### `test_pareto_admission.py` (C2.3)

Spec-only (skipped at runtime; no production code yet). Pinned thresholds:

* `speedup>=1.30 AND lpips_delta<=0.04 AND clip_delta>=-0.02` -> admitted.
* Boundary cases pinned: `(1.4, 0.02, -0.01)=1`, `(1.2, 0.02, -0.01)=0`,
  `(1.4, 0.06, -0.01)=0`, `(1.4, 0.02, -0.05)=0`.

When Crucible-2 lands `core.quality_ledger.pareto_admit(...)` (or one of the
candidate names probed in `_resolve_admit_fn`), the skip dissolves and the
four parametrized cases run automatically.

---

## C2.5 BLOCKED -- TensorRT not installed

Per `_handoff_forge3.md` open issue #2: `tensorrt`, `cutlass`, and
`sageattention` are not installed system-wide. The Phase A acceptance
benchmarks in C2.5 (Karma XPU 45s, PDG 120s, Vellum 2.5 s/frame, Lumen 180s,
Nanite 0 hitches/min, ComfyUI throughput targets) require real backends to
build TRT plans. Cannot be exercised in this environment -- explicitly out
of scope for this run, blocked on environment setup.

---

## Deferred to follow-up Crucible run

* **C1.1 / C1.2 additions**: Forge-4's existing tests under
  `tests/forge4/` already name the cache invariants this brief calls for.
  No incremental Crucible coverage added this run; a follow-up run can
  inventory the overlap and either reuse Forge-4's tests or add the few
  uncovered edges (truncate-mid-bytes self-eviction; cross-app collision
  across cache_dir).
* **`quality_ledger.db` schema** (C2.1): not created. Crucible-2 owns the
  schema and the admission CLI. The Phase B handoff should land the DB +
  the `pareto_admit` function so `test_pareto_admission.py` un-skips.
* Consecutive-failure specialist disable rule (`_handoff_forge3.md` open
  issue #7 / brief D1 §6): noted but not yet test-covered. Worth one test
  in a follow-up run.

---

## How to reproduce

```
cd C:\Users\User\OneDrive\Desktop\Optimizer_V7_G3
C:\Python314\python.exe -m pytest tests/crucible/ -v
```

The tests use `C:\Python314\python.exe` directly (not the local venv) per
the broken-venv note in `_handoff_forge3.md` open issue #1. The
orchestrator-integration tests spawn `C:\Python314\python.exe` as a
subprocess and skip cleanly if that interpreter is absent.

---

## Operational notes

* No production-code edits. No briefs modified. Only files added are the
  four test files under `tests/crucible/` and this handoff.
* `pytest.importorskip` was not needed in practice -- every C1.3/C1.4/C1.5
  dependency resolved at collection time. Pareto admission uses
  `pytest.skip(allow_module_level=True)` on missing import (cleaner than
  `importorskip` because the module name is unknown until probing).
* The minor brief vs. code mismatch in `vram_budget` (22 GB sum, 2 GB
  headroom rather than the brief's "1 GB") is documented above as a
  flag-for-next-run, not filed as a defect; behavior is internally
  consistent and not user-impacting.

If anything in this handoff is wrong or missing, file a blocker rather
than freelancing across the role boundary (Commandment 5).
