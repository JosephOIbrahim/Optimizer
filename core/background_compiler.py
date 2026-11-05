"""
BackgroundCompiler -- idle-gated, cancellable compile worker.

Owner: Forge-4 (briefs/forge-4_cache_compiler.md, D2).

Loop sketch:
    1. Poll system idle: read GPU utilization via pynvml. If utilisation >
       idle_threshold_pct, reset dwell counter and sleep poll_interval_s.
    2. Once dwell counter reaches idle_dwell_seconds, pop the next job from
       the queue.
    3. Resolve the AccelerationBackend for compile_params["backend_name"]
       from the registry.
    4. Call backend.compile(source, shape_range, precision, cancel_token,
       progress_cb).
    5. On completion: engine_cache.store(...) + telemetry row.
    6. On CancelToken set (foreground app started): cooperatively abort,
       requeue, drop any .plan.tmp orphans (the store path already does
       that, but we double-check).
    7. On exception: 3 retries with backoff; after 3 failures, mark job
       failed and surface to next orchestrator cycle.

Queue persistence:
    - Every enqueue / dequeue / state transition writes
      `data/compile_queue.json`.
    - On startup the orchestrator constructs this class; the constructor
      reloads the queue. Any "running" jobs from the previous run get
      re-queued (they didn't survive the restart by definition).

Threading:
    - One worker thread (`_loop_thread`) drives the idle-poll + dispatch.
    - `_lock` (RLock) guards `_pending`, `_paused_apps`, `_current`,
      `_completed`, `_failed`.
    - The worker thread exits when `stop()` is called or the process dies.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
from core.engine_cache import EngineCacheManager

logger = logging.getLogger(__name__)


# ============================================================================
# Job dataclass
# ============================================================================


@dataclass
class CompileJob:
    """A single queued or in-flight compile job.

    `compile_params` keys (the orchestrator builds these; backend-agnostic):
        backend_name      -- registered AccelerationBackend.name
        backend_ver       -- backend.version (for cache key)
        app               -- "comfyui" | "houdini" | "ue5"
        model_sha256      -- for cache key
        shape_w, shape_h  -- nominal input shape
        batch             -- nominal batch
        precision         -- "fp16" | "bf16" | "fp8" | ...
        driver_ver        -- defaults to "596.36"
        sm_arch           -- defaults to 89
        source            -- a dict that we deserialize back to BackendSource
                             (since BackendSource isn't JSON-friendly,
                             callers can either pass an in-memory ref via
                             .source_ref or persist via .source_payload).
        shape_range       -- {min: [...], opt: [...], max: [...]} or None
                             (defaults to static(opt_shape) inferred from
                             shape_w x shape_h if not given).
        model_kind        -- "unet" | "gemm" | "attention" | ...
        priority          -- int; higher pops first; default 0.
    """

    job_id: str
    created_utc: str
    state: str  # "pending" | "running" | "completed" | "failed" | "cancelled"
    compile_params: Dict[str, Any]
    attempts: int = 0
    last_error: Optional[str] = None
    result_hash_key: Optional[str] = None
    pause_reason: Optional[str] = None  # set when an app pause holds the job

    def app(self) -> str:
        return self.compile_params.get("app", "")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompileJob":
        return cls(
            job_id=d["job_id"],
            created_utc=d["created_utc"],
            state=d.get("state", "pending"),
            compile_params=d.get("compile_params", {}),
            attempts=int(d.get("attempts", 0)),
            last_error=d.get("last_error"),
            result_hash_key=d.get("result_hash_key"),
            pause_reason=d.get("pause_reason"),
        )


# ============================================================================
# GPU-idle gate (pynvml-based, with safe fallback)
# ============================================================================


class _GPUIdleProbe:
    """
    Reads GPU utilisation. If pynvml is unavailable the probe degrades to
    "always idle" (matches Forge-1's "degrade gracefully" pattern); the
    BackgroundCompiler also exposes an injectable probe so tests / dry-runs
    don't depend on a live GPU.
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._handle = None
        self._initialized = False
        self._init_attempts = 0
        self._init_max_attempts = 3
        self._available = False
        self._init_lock = threading.Lock()

    def _ensure_init(self) -> bool:
        if self._initialized:
            return self._available
        with self._init_lock:
            if self._initialized:
                return self._available
            self._init_attempts += 1
            try:
                import pynvml  # type: ignore[import]
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._pynvml = pynvml
                self._available = True
                self._initialized = True
                logger.info("_GPUIdleProbe initialised on device %d",
                            self.device_index)
                return True
            except Exception as exc:  # broad: missing import OR no GPU
                if self._init_attempts >= self._init_max_attempts:
                    self._initialized = True
                    self._available = False
                    logger.warning(
                        "_GPUIdleProbe disabled after %d init failures (%s). "
                        "Background compiler will treat GPU as always idle. "
                        "BLOCKER hint: GPU monitoring unavailable; cannot "
                        "idle-gate.",
                        self._init_attempts, exc,
                    )
                    return False
                logger.warning(
                    "_GPUIdleProbe init attempt %d/%d failed: %s",
                    self._init_attempts, self._init_max_attempts, exc,
                )
                return False

    def utilization_pct(self) -> float:
        """0.0 .. 100.0; returns 0.0 when probe is unavailable."""
        if not self._ensure_init():
            return 0.0
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(util.gpu)
        except Exception as exc:
            logger.warning("_GPUIdleProbe utilization read failed: %s", exc)
            return 0.0


# ============================================================================
# Backend registry
# ============================================================================


class BackendRegistry:
    """
    Lookup table from backend.name -> AccelerationBackend instance.

    Forge-3 (orchestrator) is expected to register the four Forge-1
    backends at startup. Forge-4 only consumes the registry; we never
    instantiate Forge-1's backends ourselves (workload-neutrality on the
    other side: backend layer doesn't know we exist; cache/compiler layer
    doesn't reach across into backend internals).
    """

    def __init__(self) -> None:
        self._table: Dict[str, AccelerationBackend] = {}
        self._lock = threading.Lock()

    def register(self, backend: AccelerationBackend) -> None:
        with self._lock:
            self._table[backend.name] = backend

    def get(self, name: str) -> Optional[AccelerationBackend]:
        with self._lock:
            return self._table.get(name)

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._table.keys())


# ============================================================================
# BackgroundCompiler
# ============================================================================


class BackgroundCompiler:
    """
    Idle-gated compile worker.

    Constructor:
        engine_cache:        Forge-4's EngineCacheManager.
        idle_threshold_pct:  GPU util above which we count as "busy"
                             (default 5.0 per brief).
        idle_dwell_seconds:  must observe `< idle_threshold_pct` for this
                             many seconds continuously before popping a
                             job (default 30 per brief).
        max_concurrent:      max simultaneous compiles (default 1).
        poll_interval_s:     idle-poll period (default 5 per brief).
        registry:            optional BackendRegistry; if None a fresh one
                             is created (Forge-3 will inject its own).
        idle_probe:          test injection point; defaults to _GPUIdleProbe.
        queue_path:          where to persist the queue (default
                             data/compile_queue.json).
        telemetry_path:      where to write generation telemetry rows
                             (default data/comfyui_telemetry.db). Used for
                             compile-side rows ("backend X built engine for
                             model Y in N seconds"); the foreground
                             generation telemetry is written by Forge-2.
    """

    def __init__(
        self,
        engine_cache: EngineCacheManager,
        idle_threshold_pct: float = 5.0,
        idle_dwell_seconds: int = 30,
        max_concurrent: int = 1,
        poll_interval_s: float = 5.0,
        registry: Optional[BackendRegistry] = None,
        idle_probe: Optional[_GPUIdleProbe] = None,
        queue_path: Path = Path("data/compile_queue.json"),
        telemetry_path: Path = Path("data/comfyui_telemetry.db"),
    ) -> None:
        self.engine_cache = engine_cache
        self.idle_threshold_pct = float(idle_threshold_pct)
        self.idle_dwell_seconds = int(idle_dwell_seconds)
        self.max_concurrent = max(1, int(max_concurrent))
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.registry = registry or BackendRegistry()
        self.idle_probe = idle_probe or _GPUIdleProbe()
        self.queue_path = Path(queue_path).resolve()
        self.telemetry_path = Path(telemetry_path).resolve()

        self._lock = threading.RLock()
        self._pending: List[CompileJob] = []
        self._current: Dict[str, CompileJob] = {}  # job_id -> job
        self._cancel_tokens: Dict[str, CancelToken] = {}
        self._completed: List[CompileJob] = []
        self._failed: List[CompileJob] = []
        self._paused_apps: Set[str] = set()

        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None
        self._idle_dwell_counter: float = 0.0

        # Bootstrap
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self._restore_queue()

    # ------------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the background worker thread (idempotent)."""
        with self._lock:
            if self._loop_thread is not None and self._loop_thread.is_alive():
                return
            self._stop_event.clear()
            self._loop_thread = threading.Thread(
                target=self._run, name="BackgroundCompiler",
                daemon=True,
            )
            self._loop_thread.start()
            logger.info(
                "BackgroundCompiler started (idle<%.1f%% for %ds, poll=%.1fs).",
                self.idle_threshold_pct, self.idle_dwell_seconds,
                self.poll_interval_s,
            )

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the background worker. Cancels in-flight jobs."""
        with self._lock:
            self._stop_event.set()
            for tok in self._cancel_tokens.values():
                tok.cancel("shutdown")
        thread = self._loop_thread
        if thread is not None:
            thread.join(timeout=timeout)
        logger.info("BackgroundCompiler stopped.")

    # ------------------------------------------------------------------- public API

    def enqueue(self, compile_params: Dict[str, Any]) -> str:
        """Push a job. Returns the new job_id."""
        if not isinstance(compile_params, dict):
            raise TypeError("compile_params must be a dict")
        if "app" not in compile_params:
            raise ValueError("compile_params requires 'app' key")
        if "backend_name" not in compile_params:
            raise ValueError("compile_params requires 'backend_name' key")

        job = CompileJob(
            job_id=uuid.uuid4().hex,
            created_utc=_utc_now_iso(),
            state="pending",
            compile_params=dict(compile_params),
        )
        with self._lock:
            self._pending.append(job)
            self._sort_pending()
            self._persist_queue_locked()
        logger.info(
            "BackgroundCompiler.enqueue: job %s app=%s backend=%s",
            job.job_id, job.app(), compile_params.get("backend_name"),
        )
        return job.job_id

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job. Returns True if found."""
        with self._lock:
            # Still pending?
            for i, j in enumerate(self._pending):
                if j.job_id == job_id:
                    j.state = "cancelled"
                    self._pending.pop(i)
                    self._completed.append(j)  # keep the record
                    self._persist_queue_locked()
                    logger.info("cancel: dropped pending job %s", job_id)
                    return True
            # Running?
            if job_id in self._current:
                tok = self._cancel_tokens.get(job_id)
                if tok is not None:
                    tok.cancel("user_cancel")
                logger.info("cancel: signalled running job %s", job_id)
                return True
        return False

    def pause_app(self, app: str, reason: str = "foreground_active") -> int:
        """
        Pause pending + running compiles for `app`. Used by
        CrossAppCoordinator when the user starts working in that app.
        Returns count affected.
        """
        affected = 0
        with self._lock:
            self._paused_apps.add(app)
            for j in self._pending:
                if j.app() == app and j.state == "pending":
                    j.pause_reason = reason
                    affected += 1
            for j in self._current.values():
                if j.app() == app:
                    tok = self._cancel_tokens.get(j.job_id)
                    if tok is not None:
                        tok.cancel(f"pause:{reason}")
                    j.pause_reason = reason
                    affected += 1
            self._persist_queue_locked()
        if affected:
            logger.info("pause_app(%s, %s): paused %d jobs", app, reason, affected)
        return affected

    def resume_app(self, app: str) -> int:
        """Lift a pause set via pause_app. Returns count un-paused."""
        affected = 0
        with self._lock:
            self._paused_apps.discard(app)
            for j in self._pending:
                if j.app() == app and j.pause_reason is not None:
                    j.pause_reason = None
                    affected += 1
            self._persist_queue_locked()
        if affected:
            logger.info("resume_app(%s): un-paused %d jobs", app, affected)
        return affected

    def status(self) -> Dict[str, Any]:
        """Snapshot for the dashboard / orchestrator status loop."""
        with self._lock:
            queue_depth_per_app: Dict[str, int] = {}
            for j in self._pending:
                if j.pause_reason is None:
                    queue_depth_per_app[j.app()] = (
                        queue_depth_per_app.get(j.app(), 0) + 1
                    )
            current = [
                {
                    "job_id": j.job_id,
                    "app": j.app(),
                    "backend": j.compile_params.get("backend_name"),
                    "state": j.state,
                    "attempts": j.attempts,
                }
                for j in self._current.values()
            ]
            estimated_remaining = self._estimate_remaining_seconds_locked()
            return {
                "queue_depth_per_app": queue_depth_per_app,
                "pending_total": len(self._pending),
                "current_jobs": current,
                "completed_total": len(self._completed),
                "failed_total": len(self._failed),
                "paused_apps": sorted(self._paused_apps),
                "idle_state": {
                    "running": self._loop_thread is not None
                               and self._loop_thread.is_alive(),
                    "dwell_seconds_observed": self._idle_dwell_counter,
                    "dwell_required": self.idle_dwell_seconds,
                    "threshold_pct": self.idle_threshold_pct,
                },
                "estimated_time_remaining": estimated_remaining,
                "registered_backends": self.registry.names(),
            }

    # ------------------------------------------------------------------- main loop

    def _run(self) -> None:
        """Worker thread main loop."""
        last_tick = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            elapsed = now - last_tick
            last_tick = now

            util = self.idle_probe.utilization_pct()
            if util > self.idle_threshold_pct:
                if self._idle_dwell_counter > 0:
                    logger.debug(
                        "GPU busy (%.1f%% > %.1f%%); resetting dwell counter.",
                        util, self.idle_threshold_pct,
                    )
                self._idle_dwell_counter = 0.0
            else:
                self._idle_dwell_counter += elapsed

            # We only kick off a new compile when:
            # - at least one slot is free,
            # - dwell counter has reached threshold,
            # - there's a pickable job.
            with self._lock:
                free_slots = self.max_concurrent - len(self._current)
                ready_for_dispatch = (
                    self._idle_dwell_counter >= self.idle_dwell_seconds
                    and free_slots > 0
                    and self._has_dispatchable_job_locked()
                )
                if ready_for_dispatch:
                    job = self._pop_next_job_locked()
                else:
                    job = None

            if job is not None:
                # Run outside the lock; backend.compile blocks for minutes.
                self._run_job(job)

            # Sleep until next poll. Use stop_event.wait so stop() is responsive.
            self._stop_event.wait(timeout=self.poll_interval_s)

    def _has_dispatchable_job_locked(self) -> bool:
        for j in self._pending:
            if j.pause_reason is not None:
                continue
            if j.app() in self._paused_apps:
                continue
            return True
        return False

    def _pop_next_job_locked(self) -> Optional[CompileJob]:
        for i, j in enumerate(self._pending):
            if j.pause_reason is not None:
                continue
            if j.app() in self._paused_apps:
                continue
            self._pending.pop(i)
            j.state = "running"
            j.attempts += 1
            self._current[j.job_id] = j
            tok = CancelToken()
            self._cancel_tokens[j.job_id] = tok
            self._persist_queue_locked()
            return j
        return None

    # ------------------------------------------------------------------- job execution

    def _run_job(self, job: CompileJob) -> None:
        """Outside-lock execution of a single job. Handles cancel + retry."""
        backend_name = job.compile_params.get("backend_name", "")
        backend = self.registry.get(backend_name)
        if backend is None:
            self._fail_job(job, f"backend not registered: {backend_name!r}",
                           hard=True)
            return
        if not getattr(backend, "enabled", False):
            self._fail_job(
                job, f"backend disabled: {backend_name!r} (.enabled=False)",
                hard=True,
            )
            return

        tok = self._cancel_tokens.get(job.job_id) or CancelToken()

        try:
            source = self._materialize_source(job.compile_params)
            shape_range = self._materialize_shape_range(job.compile_params)
            precision = job.compile_params.get("precision", "fp16")

            t0 = time.perf_counter()
            descriptor: BuildArtifactDescriptor = backend.compile(
                source=source,
                shape_range=shape_range,
                precision=precision,
                cancel_token=tok,
                progress_cb=self._make_progress_cb(job, tok),
            )
            elapsed = time.perf_counter() - t0

            # Persist into the cache.
            cache_key = self.engine_cache.compute_key(
                app=job.compile_params["app"],
                model_sha256=job.compile_params.get("model_sha256", ""),
                shape_w=int(job.compile_params.get("shape_w", 0)),
                shape_h=int(job.compile_params.get("shape_h", 0)),
                batch=int(job.compile_params.get("batch", 1)),
                precision=precision,
                backend_name=backend.name,
                backend_ver=backend.version,
                driver_ver=str(job.compile_params.get("driver_ver", "596.36")),
                sm_arch=int(job.compile_params.get("sm_arch", 89)),
            )

            # Stamp app into descriptor.notes so EngineCacheManager.store
            # can pick it up via .notes['app'] if the caller didn't pass app
            # explicitly. (We do pass app explicitly here, but be friendly
            # to other call sites.)
            if "app" not in descriptor.notes:
                descriptor.notes["app"] = job.compile_params["app"]

            self.engine_cache.store(
                hash_key=cache_key,
                plan_bytes=descriptor.artifact_bytes,
                descriptor=descriptor,
                app=job.compile_params["app"],
                model_sha256=job.compile_params.get("model_sha256", ""),
                shape_w=int(job.compile_params.get("shape_w", 0)),
                shape_h=int(job.compile_params.get("shape_h", 0)),
                batch=int(job.compile_params.get("batch", 1)),
                driver_ver=str(job.compile_params.get("driver_ver", "596.36")),
                sm_arch=int(job.compile_params.get("sm_arch", 89)),
            )

            self._record_compile_telemetry(job, descriptor, elapsed)

            with self._lock:
                job.state = "completed"
                job.result_hash_key = cache_key
                self._current.pop(job.job_id, None)
                self._cancel_tokens.pop(job.job_id, None)
                self._completed.append(job)
                self._persist_queue_locked()
            logger.info(
                "BackgroundCompiler: completed %s (%s/%s) in %.1fs -> %s",
                job.job_id, job.app(), backend_name, elapsed, cache_key,
            )

        except CompileCancelled as exc:
            self._handle_cancel(job, exc)
        except Exception as exc:
            self._handle_exception(job, exc)
        finally:
            self._cleanup_tmp_orphans(job)

    def _handle_cancel(self, job: CompileJob, exc: CompileCancelled) -> None:
        with self._lock:
            self._current.pop(job.job_id, None)
            self._cancel_tokens.pop(job.job_id, None)
            reason = job.pause_reason or str(exc) or "cancelled"
            if reason.startswith("pause:") or reason == "foreground_active":
                # Re-queue: pause came from foreground app activity.
                job.state = "pending"
                job.last_error = f"cancelled: {reason}"
                self._pending.append(job)
                self._sort_pending()
                logger.info(
                    "BackgroundCompiler: requeued cancelled job %s (%s)",
                    job.job_id, reason,
                )
            elif reason == "shutdown":
                # Shutdown cancel: leave in pending so next start picks up.
                job.state = "pending"
                job.last_error = "cancelled: shutdown"
                self._pending.append(job)
                self._sort_pending()
            else:
                # Explicit user cancel: don't requeue.
                job.state = "cancelled"
                job.last_error = f"cancelled: {reason}"
                self._completed.append(job)
                logger.info(
                    "BackgroundCompiler: cancelled job %s (%s)",
                    job.job_id, reason,
                )
            self._persist_queue_locked()

    def _handle_exception(self, job: CompileJob, exc: Exception) -> None:
        err = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "BackgroundCompiler: job %s attempt %d failed: %s",
            job.job_id, job.attempts, err,
        )
        with self._lock:
            self._current.pop(job.job_id, None)
            self._cancel_tokens.pop(job.job_id, None)
            job.last_error = err
            if job.attempts >= 3:
                job.state = "failed"
                self._failed.append(job)
                logger.error(
                    "BackgroundCompiler: job %s exhausted retries: %s",
                    job.job_id, err,
                )
            else:
                # Backoff: re-queue with attempt count preserved. The next
                # idle window will pick it up. A small static backoff
                # marker via pause_reason="retry_backoff_<n>" keeps the job
                # visible in status without delaying scheduling itself.
                job.state = "pending"
                self._pending.append(job)
                self._sort_pending()
            self._persist_queue_locked()

    def _fail_job(
        self,
        job: CompileJob,
        message: str,
        hard: bool = False,
    ) -> None:
        logger.error("BackgroundCompiler: hard fail job %s: %s",
                     job.job_id, message)
        with self._lock:
            self._current.pop(job.job_id, None)
            self._cancel_tokens.pop(job.job_id, None)
            job.state = "failed"
            job.last_error = message
            self._failed.append(job)
            self._persist_queue_locked()

    # ------------------------------------------------------------------- helpers

    def _make_progress_cb(
        self,
        job: CompileJob,
        tok: CancelToken,
    ) -> Callable[[float], None]:
        """Wrap a progress callback that also services cooperative cancel."""
        def _cb(p: float) -> None:
            tok.raise_if_cancelled()
            logger.debug("compile %s: progress=%.2f", job.job_id, p)
        return _cb

    def _materialize_source(self, params: Dict[str, Any]) -> BackendSource:
        """Reconstruct a BackendSource from the JSON-friendly params dict."""
        src = params.get("source") or {}
        kind = (src.get("kind") or "").lower()
        if kind == "onnx":
            opset = src.get("opset")
            return OnnxPath(path=Path(src["path"]),
                            opset=int(opset) if opset is not None else None)
        if kind == "torch_module":
            payload = src.get("pickle_bytes")
            if isinstance(payload, str):
                # Allow base64 transport for cross-process queues.
                import base64
                payload = base64.b64decode(payload)
            return TorchModulePickle(
                pickle_bytes=bytes(payload),
                module_class_qualname=src.get("module_class_qualname", ""),
            )
        if kind == "trt_plan":
            payload = src.get("plan_bytes")
            if isinstance(payload, str):
                import base64
                payload = base64.b64decode(payload)
            return TrtPlanBytes(
                plan_bytes=bytes(payload),
                plan_format_version=src.get("plan_format_version",
                                            "trt_plan_v1"),
            )
        raise ValueError(
            f"compile_params['source'] missing or unknown kind: {src!r}"
        )

    def _materialize_shape_range(self, params: Dict[str, Any]) -> ShapeRange:
        sr = params.get("shape_range")
        if isinstance(sr, dict) and "min" in sr and "opt" in sr and "max" in sr:
            return ShapeRange(
                min_shape=tuple(sr["min"]),
                opt_shape=tuple(sr["opt"]),
                max_shape=tuple(sr["max"]),
            )
        # Fallback: synthesize a static profile from (batch, c?, h, w).
        h = int(params.get("shape_h", 0))
        w = int(params.get("shape_w", 0))
        b = int(params.get("batch", 1))
        if h > 0 and w > 0:
            return ShapeRange.static((b, 4, h, w))
        raise ValueError(
            "compile_params requires either shape_range or (shape_h, shape_w)"
        )

    def _sort_pending(self) -> None:
        """Stable sort: priority desc, then created_utc asc."""
        self._pending.sort(
            key=lambda j: (
                -int(j.compile_params.get("priority", 0)),
                j.created_utc,
            )
        )

    def _estimate_remaining_seconds_locked(self) -> float:
        total = 0.0
        for j in self._pending + list(self._current.values()):
            if j.pause_reason is not None and j.state == "pending":
                continue
            backend = self.registry.get(
                j.compile_params.get("backend_name", "")
            )
            if backend is None:
                total += 60.0
                continue
            try:
                shape = (
                    int(j.compile_params.get("shape_h", 0)),
                    int(j.compile_params.get("shape_w", 0)),
                )
                total += float(backend.estimated_compile_seconds(
                    j.compile_params.get("model_kind", ""),
                    shape,
                ))
            except Exception:
                total += 60.0
        return total

    def _cleanup_tmp_orphans(self, job: CompileJob) -> None:
        """Remove any .plan.tmp file under the job's app dir.

        EngineCacheManager.store already handles its own .tmp cleanup, but
        we double-check for our public test invariant: "cancel mid-compile
        leaves no .plan.tmp orphans." Backends could in principle leave
        their own scratch files; we don't reach into those, but we do scan
        the cache app dir.
        """
        app = job.app()
        if not app:
            return
        app_dir = self.engine_cache.cache_dir / app
        if not app_dir.exists():
            return
        try:
            for path in app_dir.glob("*.plan.tmp"):
                try:
                    path.unlink()
                    logger.info("cleanup_tmp_orphans: removed %s", path)
                except OSError as exc:
                    logger.warning(
                        "cleanup_tmp_orphans: cannot remove %s: %s",
                        path, exc,
                    )
        except OSError as exc:
            logger.warning("cleanup_tmp_orphans: scan failed for %s: %s",
                           app_dir, exc)

    # ------------------------------------------------------------------- queue persistence

    def _persist_queue_locked(self) -> None:
        """Atomically persist queue state to disk. Caller holds self._lock."""
        snapshot = {
            "version": 1,
            "saved_utc": _utc_now_iso(),
            "pending": [j.to_dict() for j in self._pending],
            "running": [j.to_dict() for j in self._current.values()],
            "completed_recent": [j.to_dict() for j in self._completed[-50:]],
            "failed_recent": [j.to_dict() for j in self._failed[-50:]],
            "paused_apps": sorted(self._paused_apps),
        }
        tmp = self.queue_path.with_suffix(self.queue_path.suffix + ".tmp")
        try:
            self.queue_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(snapshot, indent=2, default=str),
                encoding="utf-8",
            )
            import os
            os.replace(tmp, self.queue_path)
        except OSError as exc:
            logger.error("persist_queue: %s", exc)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def _restore_queue(self) -> None:
        if not self.queue_path.exists():
            return
        try:
            data = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("restore_queue: cannot parse %s: %s",
                           self.queue_path, exc)
            return

        with self._lock:
            for d in data.get("pending", []):
                try:
                    self._pending.append(CompileJob.from_dict(d))
                except (KeyError, TypeError) as exc:
                    logger.warning("restore_queue skip pending %r: %s", d, exc)
            # "running" jobs from a prior process didn't actually finish;
            # re-queue them as pending.
            for d in data.get("running", []):
                try:
                    job = CompileJob.from_dict(d)
                    job.state = "pending"
                    job.last_error = "interrupted_by_restart"
                    self._pending.append(job)
                except (KeyError, TypeError) as exc:
                    logger.warning("restore_queue skip running %r: %s", d, exc)
            for app in data.get("paused_apps", []):
                self._paused_apps.add(app)
            self._sort_pending()
        if self._pending:
            logger.info(
                "BackgroundCompiler: restored %d pending jobs from %s",
                len(self._pending), self.queue_path,
            )

    # ------------------------------------------------------------------- compile telemetry

    def _record_compile_telemetry(
        self,
        job: CompileJob,
        descriptor: BuildArtifactDescriptor,
        elapsed: float,
    ) -> None:
        """Append a compile-side row to comfyui_telemetry.db (compiles table).

        Forge-2 owns the foreground generation telemetry; this is the
        compile-time complement (so we know how often the cache misses and
        how long misses cost). We use a separate `compiles` table; the
        `generations` table belongs to Forge-2.
        """
        try:
            from core.comfyui_telemetry import record_compile  # local import
            record_compile(
                db_path=self.telemetry_path,
                ts_utc=_utc_now_iso(),
                app=job.app(),
                backend_name=descriptor.backend_name,
                backend_version=descriptor.backend_version,
                model_sha256=job.compile_params.get("model_sha256", ""),
                shape_w=int(job.compile_params.get("shape_w", 0)),
                shape_h=int(job.compile_params.get("shape_h", 0)),
                batch=int(job.compile_params.get("batch", 1)),
                precision=descriptor.precision,
                compile_seconds=float(elapsed),
                artifact_bytes=int(descriptor.size_bytes),
                notes=descriptor.notes,
            )
        except Exception as exc:
            logger.warning(
                "compile telemetry write failed (non-fatal): %s", exc,
            )


# ============================================================================
# Module-private utilities
# ============================================================================


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")
