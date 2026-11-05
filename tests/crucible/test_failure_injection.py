"""Crucible C1.5 — Failure injection.

Verifies the system's graceful-degradation invariants hold under the three
adversarial scenarios from the verification brief:

1. pynvml broken (raises on every call) -> V7 `gather_context()` must still
   return a safe `SystemContext` rather than crashing the optimizer cycle.
2. ComfyUI HTTP poll raises ConnectionError -> `ComfyUIStateCollector.collect()`
   returns a `ComfyUIState` with `api_reachable=False` and no exception
   propagates.
3. EngineCacheManager constructed with a non-writable cache_dir -> raises a
   clear blocker on construction (or first store), does NOT silently succeed.
"""
from __future__ import annotations

import io
import sys
import types
from datetime import datetime
from pathlib import Path
from urllib.error import URLError

import pytest


# ===========================================================================
# 1. pynvml stubbed to raise -> safe SystemContext
# ===========================================================================

def test_pynvml_stub_returns_safe_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """V7 invariant from SHARED_CONTEXT.md: `gather_context()` graceful-degrade
    contract requires returning a safe SystemContext on exception."""
    import core.gpu_optimizer_v6 as v6
    from core.gpu_optimizer_v6 import SystemContext, WorkflowMode

    # Replace v6.pynvml with a stub whose every attribute access raises.
    class _ExplodingPynvml:
        class _Boom(Exception):
            pass

        def __getattr__(self, name: str):
            def _raise(*a, **k):
                raise RuntimeError(f"pynvml.{name} stub: synthetic failure")
            return _raise

    monkeypatch.setattr(v6, "pynvml", _ExplodingPynvml(), raising=True)

    # Build the orchestrator just enough to call gather_context. We avoid the
    # full __init__ (it spins up experts) by using a placeholder class with
    # only the attributes gather_context reads.
    fake = types.SimpleNamespace()
    fake.logger = v6.logging.getLogger("CrucibleStub")  # type: ignore[attr-defined]

    # Bind the unbound method.
    ctx = v6.OptimizerOrchestrator.gather_context(fake)  # type: ignore[arg-type]

    assert isinstance(ctx, SystemContext)
    # Safe defaults from the except branch (lines 1107-1119 of v6).
    assert ctx.workflow_mode == WorkflowMode.IDLE
    assert ctx.gpu_utilization_percent == 0.0
    assert ctx.vram_used_gb == 0.0
    assert ctx.vram_free_gb == 24.0
    assert ctx.active_processes == []
    assert ctx.process_vram == {}


# ===========================================================================
# 2. ComfyUI HTTP raising ConnectionError -> api_reachable=False, no raise
# ===========================================================================

def test_comfyui_collector_http_failure_marks_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every HTTP probe raises, the collector must return a state with
    api_reachable=False and NOT propagate the exception."""
    from core.experts import comfyui_state_collector as cscol

    # Force `_find_comfy_process` to short-circuit so we don't depend on the
    # real psutil enumeration (process_running can stay False; what we are
    # testing is the HTTP path).
    def _no_proc(_self):
        return None
    monkeypatch.setattr(cscol.ComfyUIStateCollector, "_find_comfy_process", _no_proc)

    # Force `_urlrequest.urlopen` (used inside _http_get_json) to raise.
    def _explode(*args, **kwargs):
        raise URLError("synthetic: connection refused")
    monkeypatch.setattr(cscol._urlrequest, "urlopen", _explode)

    collector = cscol.ComfyUIStateCollector()
    state = collector.collect()  # MUST NOT RAISE

    assert isinstance(state, cscol.ComfyUIState)
    assert state.api_reachable is False
    # process_running is False because the stub returned no proc; that's OK.
    assert state.process_running is False


# ===========================================================================
# 3. Non-writable cache_dir -> clear blocker, not silent success
# ===========================================================================

def test_engine_cache_non_writable_raises_blocker(tmp_path: Path) -> None:
    """Per crucible_verification.md C1.2 last bullet: non-writable cache_dir
    must raise (clear blocker), not silently succeed.

    We force this by passing a cache_dir whose parent is an existing FILE,
    which makes mkdir(parents=True) fail on every platform.
    """
    from core.engine_cache import EngineCacheManager

    # Create a regular file at the path we'll try to use as a parent.
    blocker_file = tmp_path / "blocker.txt"
    blocker_file.write_text("not a directory")

    # cache_dir attempts to live INSIDE the file -> mkdir must fail.
    bad_cache_dir = blocker_file / "EngineCache"
    bad_db_path = tmp_path / "engine_cache.db"

    with pytest.raises((RuntimeError, OSError)) as excinfo:
        EngineCacheManager(cache_dir=bad_cache_dir, db_path=bad_db_path)

    # Error message must mention either 'cache' or the failing path so the
    # operator can diagnose; "silently succeed" is the failure mode we're
    # ruling out.
    msg = str(excinfo.value).lower()
    assert any(token in msg for token in ("cache", "dir", str(blocker_file).lower())), (
        f"Error message lacks a clear pointer at the cache_dir issue: {excinfo.value!r}"
    )
