"""Tests for core.background_compiler.BackgroundCompiler (Forge-4 D2).

Crucible-1's done criteria:
- BackgroundCompiler does not start a compile while GPU > 5% utilized.
- Cancel mid-compile leaves no .plan.tmp files behind.
- Queue persists across simulated restart.

We use a stub AccelerationBackend + a stub _GPUIdleProbe so tests don't
depend on a live GPU.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pytest

from acceleration.backend import (
    AccelerationBackend,
    BackendSource,
    BuildArtifactDescriptor,
    CancelToken,
    CompileCancelled,
    OnnxPath,
    ShapeRange,
    TorchModulePickle,
    TrtPlanBytes,
)
from core.background_compiler import (
    BackendRegistry,
    BackgroundCompiler,
    CompileJob,
    _GPUIdleProbe,
)
from core.engine_cache import EngineCacheManager


# ============================================================================
# Stubs
# ============================================================================


class StubBackend(AccelerationBackend):
    name = "stub_backend"
    version = "0.0.1"

    def __init__(
        self,
        sleep_seconds: float = 0.05,
        artifact_bytes: bytes = b"STUB_PLAN",
        check_cancel_per_iter: int = 5,
        explode: bool = False,
    ) -> None:
        self.sleep_seconds = sleep_seconds
        self.artifact_bytes = artifact_bytes
        self.check_cancel_per_iter = check_cancel_per_iter
        self.explode = explode
        self.enabled = True
        self.compile_calls = 0

    def applies_to(
        self, model_kind: str, shape: Tuple[int, ...], precision: str
    ) -> bool:
        return True

    def estimated_compile_seconds(self, model_kind: str, shape) -> float:
        return self.sleep_seconds

    def estimated_speedup(self, model_kind: str, shape) -> float:
        return 1.5

    def estimated_quality_delta(self) -> float:
        return 0.0

    def compile(
        self,
        source: BackendSource,
        shape_range: ShapeRange,
        precision: str,
        cancel_token: CancelToken,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> BuildArtifactDescriptor:
        self.compile_calls += 1
        if self.explode:
            raise RuntimeError("stub_explode")

        # Simulate a long-running compile in small slices so cancel can
        # interrupt cleanly. Brief: cancel mid-compile must leave no orphans.
        slice_s = max(0.005, self.sleep_seconds / 20)
        n_slices = max(1, int(self.sleep_seconds / slice_s))
        for i in range(n_slices):
            cancel_token.raise_if_cancelled()
            if progress_cb is not None:
                progress_cb((i + 1) / n_slices)
            time.sleep(slice_s)
        cancel_token.raise_if_cancelled()
        return BuildArtifactDescriptor(
            backend_name=self.name,
            backend_version=self.version,
            artifact_bytes=self.artifact_bytes,
            artifact_format="trt_plan",
            precision=precision,
            shape_range=shape_range,
            compile_seconds=self.sleep_seconds,
            notes={"stub": True},
        )

    def load(self, artifact_path: Path):
        from acceleration.backend import LoadedEngine
        return LoadedEngine(
            backend_name=self.name,
            backend_version=self.version,
            runner=None,
            metadata={},
        )


class FakeIdleProbe:
    """Settable utilisation. Behaves like _GPUIdleProbe.utilization_pct()."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def utilization_pct(self) -> float:
        return self.value


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cache(tmp_path: Path) -> EngineCacheManager:
    return EngineCacheManager(
        cache_dir=tmp_path / "EngineCache",
        db_path=tmp_path / "engine_cache.db",
        max_disk_bytes=1 * 1024 ** 3,
    )


def _make_compiler(
    cache: EngineCacheManager,
    tmp_path: Path,
    sleep: float = 0.05,
    idle_dwell: int = 0,
    poll_interval: float = 0.05,
    util: float = 0.0,
) -> Tuple[BackgroundCompiler, StubBackend, FakeIdleProbe]:
    reg = BackendRegistry()
    backend = StubBackend(sleep_seconds=sleep)
    reg.register(backend)
    probe = FakeIdleProbe(value=util)
    bc = BackgroundCompiler(
        engine_cache=cache,
        idle_threshold_pct=5.0,
        idle_dwell_seconds=idle_dwell,
        poll_interval_s=poll_interval,
        registry=reg,
        idle_probe=probe,
        queue_path=tmp_path / "compile_queue.json",
        telemetry_path=tmp_path / "telemetry.db",
    )
    return bc, backend, probe


def _onnx_compile_params(tmp_path: Path, app: str = "comfyui") -> dict:
    onnx_path = tmp_path / "fake_model.onnx"
    onnx_path.write_bytes(b"fake")
    return {
        "app": app,
        "backend_name": "stub_backend",
        "backend_ver": "0.0.1",
        "model_sha256": "a" * 64,
        "shape_w": 64,
        "shape_h": 64,
        "batch": 1,
        "precision": "fp16",
        "model_kind": "unet",
        "source": {"kind": "onnx", "path": str(onnx_path)},
        "shape_range": {
            "min": [1, 4, 64, 64],
            "opt": [1, 4, 64, 64],
            "max": [1, 4, 64, 64],
        },
    }


# ============================================================================
# Enqueue / status
# ============================================================================


def test_enqueue_status_reflects_queue(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    bc, _, _ = _make_compiler(cache, tmp_path)
    jid = bc.enqueue(_onnx_compile_params(tmp_path))
    assert isinstance(jid, str) and len(jid) > 0
    s = bc.status()
    assert s["pending_total"] == 1
    assert s["queue_depth_per_app"]["comfyui"] == 1


def test_enqueue_persists_queue_to_disk(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    bc, _, _ = _make_compiler(cache, tmp_path)
    bc.enqueue(_onnx_compile_params(tmp_path))
    persisted = json.loads((tmp_path / "compile_queue.json").read_text())
    assert persisted["pending"][0]["compile_params"]["app"] == "comfyui"


def test_enqueue_requires_app_and_backend(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    bc, _, _ = _make_compiler(cache, tmp_path)
    with pytest.raises(ValueError):
        bc.enqueue({"backend_name": "stub_backend"})
    with pytest.raises(ValueError):
        bc.enqueue({"app": "comfyui"})


# ============================================================================
# Idle gating
# ============================================================================


def test_does_not_start_compile_while_gpu_busy(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    """Crucible-1: BackgroundCompiler does not start a compile while GPU > 5%.

    We set util=80% (above 5% threshold), enqueue a job, start the loop,
    sleep more than the dwell window, and verify the backend was never
    invoked.
    """
    bc, backend, probe = _make_compiler(
        cache, tmp_path, idle_dwell=1, poll_interval=0.05, util=80.0,
    )
    bc.enqueue(_onnx_compile_params(tmp_path))
    bc.start()
    try:
        time.sleep(1.5)  # would be plenty of time if util were low
        assert backend.compile_calls == 0
        s = bc.status()
        # The dwell counter must remain at 0 because util > threshold.
        assert s["idle_state"]["dwell_seconds_observed"] == 0.0
        assert s["pending_total"] == 1
    finally:
        bc.stop(timeout=5)


def test_compile_runs_when_idle(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    bc, backend, probe = _make_compiler(
        cache, tmp_path, sleep=0.05, idle_dwell=0, poll_interval=0.02, util=0.0,
    )
    bc.enqueue(_onnx_compile_params(tmp_path))
    bc.start()
    try:
        # Wait for both: the call AND the bookkeeping that follows.
        deadline = time.time() + 5
        while time.time() < deadline and bc.status()["completed_total"] == 0:
            time.sleep(0.05)
        assert backend.compile_calls >= 1, "backend.compile was not invoked"
        assert bc.status()["completed_total"] == 1
    finally:
        bc.stop(timeout=5)

    # And the cache should now contain the artifact.
    completed = bc._completed[-1]  # type: ignore[attr-defined]
    assert completed.result_hash_key is not None
    art = cache.lookup(completed.result_hash_key)
    assert art is not None
    assert art.read_bytes() == b"STUB_PLAN"


# ============================================================================
# Cancel mid-compile
# ============================================================================


def test_cancel_before_idle_dwell_never_compiles(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    """Brief sequence step 6: enqueue + cancel before idle dwell elapses
    -> never compiles."""
    bc, backend, _ = _make_compiler(
        cache, tmp_path, idle_dwell=10, poll_interval=0.05, util=0.0,
    )
    jid = bc.enqueue(_onnx_compile_params(tmp_path))
    bc.start()
    try:
        time.sleep(0.2)
        assert bc.cancel(jid) is True
        time.sleep(0.5)
        assert backend.compile_calls == 0
    finally:
        bc.stop(timeout=5)


def test_cancel_mid_compile_leaves_no_tmp_orphans(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    """Crucible-1: cancel mid-compile leaves no .plan.tmp files behind."""
    bc, backend, _ = _make_compiler(
        cache, tmp_path, sleep=2.0, idle_dwell=0, poll_interval=0.02, util=0.0,
    )
    jid = bc.enqueue(_onnx_compile_params(tmp_path))
    bc.start()
    try:
        # Wait for the compile to actually start.
        deadline = time.time() + 3
        while time.time() < deadline and not bc._current:  # type: ignore[attr-defined]
            time.sleep(0.02)
        assert bc._current, "compile didn't start in time"  # type: ignore[attr-defined]

        # Cancel it.
        assert bc.cancel(jid) is True

        # Wait for the worker to settle.
        deadline = time.time() + 5
        while time.time() < deadline and bc._current:  # type: ignore[attr-defined]
            time.sleep(0.02)
        assert not bc._current, "compile didn't unwind on cancel"  # type: ignore[attr-defined]
    finally:
        bc.stop(timeout=5)

    # No .plan.tmp orphans anywhere under cache_dir.
    for path in cache.cache_dir.rglob("*.plan.tmp"):
        pytest.fail(f"orphaned tmp file: {path}")


def test_pause_app_cancels_running_and_requeues(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    bc, backend, _ = _make_compiler(
        cache, tmp_path, sleep=2.0, idle_dwell=0, poll_interval=0.02, util=0.0,
    )
    bc.enqueue(_onnx_compile_params(tmp_path))
    bc.start()
    try:
        deadline = time.time() + 3
        while time.time() < deadline and not bc._current:  # type: ignore[attr-defined]
            time.sleep(0.02)
        assert bc._current, "compile didn't start"  # type: ignore[attr-defined]

        n_paused = bc.pause_app("comfyui")
        assert n_paused == 1

        deadline = time.time() + 5
        while time.time() < deadline and bc._current:  # type: ignore[attr-defined]
            time.sleep(0.02)
        assert not bc._current

        # Job must be requeued, paused.
        s = bc.status()
        assert s["pending_total"] >= 1
        # Resume + give it a second; it should complete now.
        bc.resume_app("comfyui")
        deadline = time.time() + 5
        while time.time() < deadline and backend.compile_calls < 2:
            time.sleep(0.05)
        # NB: the requeued attempt counts as a 2nd .compile call ->
        # backend.compile_calls should be >= 2 (first was cancelled).
        assert backend.compile_calls >= 2
    finally:
        bc.stop(timeout=5)


# ============================================================================
# Queue persistence across restart
# ============================================================================


def test_queue_persists_across_restart(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    """Crucible-1: queue persists across simulated restart.

    Enqueue 3 jobs, drop the BackgroundCompiler instance without starting
    the loop, construct a fresh one pointing at the same queue file, and
    verify the new instance sees the 3 pending jobs.
    """
    bc1, _, _ = _make_compiler(cache, tmp_path)
    j1 = bc1.enqueue(_onnx_compile_params(tmp_path, app="comfyui"))
    j2 = bc1.enqueue(_onnx_compile_params(tmp_path, app="houdini"))
    j3 = bc1.enqueue(_onnx_compile_params(tmp_path, app="ue5"))
    # Don't start; just abandon the instance.
    del bc1

    bc2, _, _ = _make_compiler(cache, tmp_path)
    s = bc2.status()
    assert s["pending_total"] == 3
    apps = {j.app() for j in bc2._pending}  # type: ignore[attr-defined]
    assert apps == {"comfyui", "houdini", "ue5"}


def test_running_jobs_are_requeued_after_restart(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    """A "running" job in the persisted file must be picked back up as
    pending on the next start."""
    queue_file = tmp_path / "compile_queue.json"
    queue_file.write_text(json.dumps({
        "version": 1,
        "saved_utc": "1970-01-01T00:00:00.000000+0000",
        "pending": [],
        "running": [
            {
                "job_id": "abc",
                "created_utc": "1970-01-01T00:00:00.000000+0000",
                "state": "running",
                "compile_params": {
                    "app": "comfyui",
                    "backend_name": "stub_backend",
                    "model_sha256": "x" * 64,
                    "shape_w": 64, "shape_h": 64, "batch": 1,
                    "precision": "fp16",
                    "source": {"kind": "onnx", "path": str(tmp_path / "f.onnx")},
                    "shape_range": {
                        "min": [1, 4, 64, 64],
                        "opt": [1, 4, 64, 64],
                        "max": [1, 4, 64, 64],
                    },
                },
                "attempts": 1,
                "last_error": None,
                "result_hash_key": None,
                "pause_reason": None,
            }
        ],
        "completed_recent": [],
        "failed_recent": [],
        "paused_apps": [],
    }))

    bc, _, _ = _make_compiler(cache, tmp_path)
    s = bc.status()
    assert s["pending_total"] == 1
    j = bc._pending[0]  # type: ignore[attr-defined]
    assert j.last_error == "interrupted_by_restart"
    assert j.state == "pending"


# ============================================================================
# Backend registration / failure modes
# ============================================================================


def test_unregistered_backend_hard_fails(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    bc, _, _ = _make_compiler(
        cache, tmp_path, idle_dwell=0, poll_interval=0.02, util=0.0,
    )
    params = _onnx_compile_params(tmp_path)
    params["backend_name"] = "no_such_backend"
    bc.enqueue(params)
    bc.start()
    try:
        deadline = time.time() + 3
        while time.time() < deadline and bc.status()["failed_total"] == 0:
            time.sleep(0.05)
        assert bc.status()["failed_total"] == 1
    finally:
        bc.stop(timeout=5)


def test_disabled_backend_hard_fails(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    reg = BackendRegistry()
    backend = StubBackend()
    backend.enabled = False
    reg.register(backend)
    bc = BackgroundCompiler(
        engine_cache=cache,
        idle_threshold_pct=5.0,
        idle_dwell_seconds=0,
        poll_interval_s=0.02,
        registry=reg,
        idle_probe=FakeIdleProbe(0.0),
        queue_path=tmp_path / "compile_queue.json",
        telemetry_path=tmp_path / "telemetry.db",
    )
    bc.enqueue(_onnx_compile_params(tmp_path))
    bc.start()
    try:
        deadline = time.time() + 3
        while time.time() < deadline and bc.status()["failed_total"] == 0:
            time.sleep(0.05)
        assert bc.status()["failed_total"] == 1
    finally:
        bc.stop(timeout=5)


def test_backend_exception_retries_then_fails(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    reg = BackendRegistry()
    backend = StubBackend(explode=True)
    reg.register(backend)
    bc = BackgroundCompiler(
        engine_cache=cache,
        idle_threshold_pct=5.0,
        idle_dwell_seconds=0,
        poll_interval_s=0.02,
        registry=reg,
        idle_probe=FakeIdleProbe(0.0),
        queue_path=tmp_path / "compile_queue.json",
        telemetry_path=tmp_path / "telemetry.db",
    )
    bc.enqueue(_onnx_compile_params(tmp_path))
    bc.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and bc.status()["failed_total"] == 0:
            time.sleep(0.05)
        assert bc.status()["failed_total"] == 1
        # Should have tried 3 times then given up.
        assert backend.compile_calls == 3
    finally:
        bc.stop(timeout=5)


# ============================================================================
# Idle probe degrades gracefully
# ============================================================================


def test_idle_probe_unavailable_treats_as_idle(
    cache: EngineCacheManager, tmp_path: Path, monkeypatch,
) -> None:
    """If pynvml import fails, _GPUIdleProbe should report 0% util after
    its 3-attempt budget is exhausted (graceful degrade)."""
    probe = _GPUIdleProbe()

    def boom():
        raise ImportError("simulated")
    # Force the lazy-init path to fail by stuffing in a fake pynvml module.
    import sys
    saved = sys.modules.get("pynvml")
    sys.modules["pynvml"] = None  # type: ignore[assignment]
    try:
        # First three calls each register a failed init attempt, eventually
        # disabling the probe.
        v1 = probe.utilization_pct()
        v2 = probe.utilization_pct()
        v3 = probe.utilization_pct()
        v4 = probe.utilization_pct()
        for v in (v1, v2, v3, v4):
            assert v == 0.0
    finally:
        if saved is not None:
            sys.modules["pynvml"] = saved
        else:
            sys.modules.pop("pynvml", None)


# ============================================================================
# Telemetry write
# ============================================================================


def test_compile_telemetry_row_written(
    cache: EngineCacheManager, tmp_path: Path,
) -> None:
    bc, backend, _ = _make_compiler(
        cache, tmp_path, sleep=0.05, idle_dwell=0, poll_interval=0.02, util=0.0,
    )
    bc.enqueue(_onnx_compile_params(tmp_path))
    bc.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and bc.status()["completed_total"] == 0:
            time.sleep(0.05)
        assert bc.status()["completed_total"] == 1
    finally:
        bc.stop(timeout=5)

    from core.comfyui_telemetry import fetch_recent_compiles
    rows = fetch_recent_compiles(tmp_path / "telemetry.db", limit=10)
    assert len(rows) == 1
    assert rows[0]["app"] == "comfyui"
    assert rows[0]["backend_name"] == "stub_backend"
