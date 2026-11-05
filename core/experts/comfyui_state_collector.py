"""
ComfyUI State Collector (Forge-2)

Collects runtime state from a local ComfyUI instance for the V8 specialist.

Responsibilities (per brief Section A):
    - Polls http://127.0.0.1:8188/system_stats, /queue, /history?max_items=20.
    - Subscribes (best-effort) to /ws for completion events; on completion,
      appends a workload tuple to an in-memory ring buffer that Forge-4's
      telemetry layer is expected to persist.
    - Reads ComfyUI's extra_model_paths.yaml if discoverable to find model
      directories.
    - Detects custom nodes by listing <install_dir>/custom_nodes/.
    - Parses launch flags from the ComfyUI process cmdline via psutil.
    - All HTTP calls have 1-second timeouts; failures degrade
      ComfyUIState.api_reachable=False rather than raise.

Bounded failure: 3 consecutive HTTP polling failures -> mark
api_reachable=False; do not retry inside this cycle.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib import error as _urlerror
from urllib import request as _urlrequest

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - environment fallback
    psutil = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - environment fallback
    yaml = None  # type: ignore

logger = logging.getLogger("collector.comfyui")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188
HTTP_TIMEOUT_S = 1.0
RING_BUFFER_MAX = 512
MAX_HTTP_FAILURES_PER_CYCLE = 3


# ---------------------------------------------------------------------------
# Workload event for the ring buffer
# ---------------------------------------------------------------------------

@dataclass
class WorkloadEvent:
    """Single completion event captured from ComfyUI history/ws."""

    workflow_hash: str
    model_sha: str
    shape: Tuple[int, int]            # (width, height)
    batch: int
    sampler: str
    steps: int
    wall_seconds: float
    timestamp: float                  # epoch seconds


# ---------------------------------------------------------------------------
# State exposed to the specialist
# ---------------------------------------------------------------------------

@dataclass
class ComfyUIState:
    """
    Snapshot of a local ComfyUI instance.

    Wired into SystemContext.comfyui by Forge-3.
    """

    process_running: bool = False
    api_reachable: bool = False
    install_dir: Optional[Path] = None
    model_dirs: List[Path] = field(default_factory=list)
    custom_nodes_installed: List[str] = field(default_factory=list)
    launch_flags: List[str] = field(default_factory=list)
    queue_pending: int = 0
    queue_running: int = 0
    system_stats: Dict[str, Any] = field(default_factory=dict)

    # Ring buffer of recent completions. Forge-4 telemetry mirrors this to
    # data/comfyui_telemetry.db. Specialists read it via
    # workload_histogram_hot().
    recent_workloads: Deque[WorkloadEvent] = field(
        default_factory=lambda: deque(maxlen=RING_BUFFER_MAX)
    )

    # ----- Helpers consumed by ComfyUISpecialist (R3) -----------------------
    def workload_histogram_hot(
        self,
        min_count: int = 3,
        lookback_days: int = 14,
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Return (model_sha, shape) pairs that appear at least `min_count` times
        in the last `lookback_days` of recent_workloads.
        """
        if not self.recent_workloads:
            return []

        horizon = time.time() - (lookback_days * 86400)
        counts: Dict[Tuple[str, Tuple[int, int]], int] = {}
        for ev in self.recent_workloads:
            if ev.timestamp < horizon:
                continue
            key = (ev.model_sha, ev.shape)
            counts[key] = counts.get(key, 0) + 1

        return [k for k, c in counts.items() if c >= min_count]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class ComfyUIStateCollector:
    """Polls a local ComfyUI instance and produces a ComfyUIState snapshot."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        process_names: Optional[List[str]] = None,
        telemetry_db_path: Optional[Path] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.process_names = process_names or ["python.exe", "pythonw.exe", "python"]
        self._http_failures_this_cycle = 0
        self._cached_install_dir: Optional[Path] = None
        self._cached_state: Optional[ComfyUIState] = None
        # Persistent ring buffer carries across collect() calls.
        self._ring: Deque[WorkloadEvent] = deque(maxlen=RING_BUFFER_MAX)
        # Dedup set: ComfyUI's /history returns the same prompt_id every poll
        # until it ages out, so without this every cycle re-appended its full
        # 20-entry window and lookback_days became meaningless. Bounded by ring.
        self._seen_prompt_ids: set[str] = set()
        # Durable telemetry DB. Restored on init so workload history survives
        # Optimizer restarts; per-event writes happen in _ingest_history.
        self._telemetry_db_path: Path = (
            Path(telemetry_db_path) if telemetry_db_path is not None
            else Path("data/comfyui_telemetry.db")
        )
        self._restore_from_db()

    # ------------------------------------------------------------------
    # Durable workload history (Path B)
    # ------------------------------------------------------------------
    def _restore_from_db(self) -> None:
        """Repopulate the ring from `data/comfyui_telemetry.db` on startup.

        Failure is non-fatal: the ring stays empty and the next /history poll
        repopulates from ComfyUI's own (volatile) history instead.
        """
        try:
            from core.comfyui_telemetry import fetch_recent_generations
            rows = fetch_recent_generations(
                self._telemetry_db_path, limit=RING_BUFFER_MAX,
            )
        except Exception as exc:
            logger.debug("ring restore from %s failed: %s",
                         self._telemetry_db_path, exc)
            return

        # Newest-first from the query; reverse so deque order matches the
        # original insertion order (oldest at index 0, newest at -1).
        from datetime import datetime
        for row in reversed(rows):
            try:
                ts_iso = row.get("ts_utc") or ""
                ts_clean = ts_iso[:-1] if ts_iso.endswith("Z") else ts_iso
                try:
                    ts_epoch = datetime.fromisoformat(ts_clean).timestamp()
                except (ValueError, TypeError):
                    ts_epoch = 0.0
                workflow_hash = row.get("workflow_hash") or ""
                ev = WorkloadEvent(
                    workflow_hash=workflow_hash,
                    model_sha=row.get("model_sha256") or "",
                    shape=(int(row.get("shape_w") or 0),
                           int(row.get("shape_h") or 0)),
                    batch=int(row.get("batch") or 1),
                    sampler=row.get("sampler") or "",
                    steps=int(row.get("steps") or 0),
                    wall_seconds=float(row.get("wall_seconds") or 0.0),
                    timestamp=ts_epoch,
                )
            except Exception as exc:
                logger.debug("ring restore: skipped malformed row: %s", exc)
                continue
            self._ring.append(ev)
            if workflow_hash:
                self._seen_prompt_ids.add(workflow_hash)

    def _persist_event(self, ev: WorkloadEvent) -> None:
        """Append one row to `generations` for the new event.

        Collector-sourced rows leave precision/backend_set/vram_peak null --
        those columns are reserved for the optional in-process patch (Path C).
        R3 only keys on (model_sha, shape) so this is enough.
        """
        try:
            from core.comfyui_telemetry import record_generation
            from datetime import datetime, timezone
            ts_utc = datetime.fromtimestamp(
                ev.timestamp, tz=timezone.utc,
            ).isoformat()
            sps = 0.0
            if ev.steps > 0 and ev.wall_seconds > 0.0:
                sps = float(ev.steps) / float(ev.wall_seconds)
            record_generation(
                db_path=self._telemetry_db_path,
                ts_utc=ts_utc,
                workflow_hash=ev.workflow_hash or None,
                model_sha256=ev.model_sha or None,
                shape_w=int(ev.shape[0]),
                shape_h=int(ev.shape[1]),
                batch=int(ev.batch),
                sampler=ev.sampler or None,
                steps=int(ev.steps) if ev.steps else None,
                precision=None,
                backend_set=[],
                wall_seconds=float(ev.wall_seconds),
                samples_per_sec=sps,
                vram_peak_gb=0.0,
            )
        except Exception as exc:
            logger.debug("telemetry persist failed for %s: %s",
                         ev.workflow_hash, exc)

    # ------------------------------------------------------------------
    # Top-level entry
    # ------------------------------------------------------------------
    def collect(self) -> ComfyUIState:
        """Build a fresh ComfyUIState snapshot. Never raises."""
        self._http_failures_this_cycle = 0
        state = ComfyUIState()
        # Carry the persistent ring buffer into the new snapshot.
        state.recent_workloads = self._ring

        try:
            proc = self._find_comfy_process()
            if proc is not None:
                state.process_running = True
                state.launch_flags = self._extract_launch_flags(proc)
                state.install_dir = self._infer_install_dir(proc)
                self._cached_install_dir = state.install_dir
            else:
                state.install_dir = self._cached_install_dir

            if state.install_dir is not None:
                state.custom_nodes_installed = self._list_custom_nodes(state.install_dir)
                state.model_dirs = self._read_extra_model_paths(state.install_dir)

            # HTTP probes (graceful degrade)
            state.system_stats = self._http_get_json("/system_stats") or {}
            queue = self._http_get_json("/queue") or {}
            try:
                state.queue_running = len(queue.get("queue_running", []) or [])
                state.queue_pending = len(queue.get("queue_pending", []) or [])
            except Exception:
                pass

            history = self._http_get_json("/history?max_items=20") or {}
            self._ingest_history(history)

            state.api_reachable = (
                self._http_failures_this_cycle < MAX_HTTP_FAILURES_PER_CYCLE
                and bool(state.system_stats)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ComfyUIStateCollector.collect failed: %s", exc)

        self._cached_state = state
        return state

    # ------------------------------------------------------------------
    # Process discovery
    # ------------------------------------------------------------------
    def _find_comfy_process(self):
        if psutil is None:
            return None
        try:
            for p in psutil.process_iter(attrs=["name", "cmdline", "exe"]):
                try:
                    name = (p.info.get("name") or "").lower()
                    if name not in {n.lower() for n in self.process_names}:
                        continue
                    cmdline = p.info.get("cmdline") or []
                    joined = " ".join(cmdline).lower()
                    if "comfyui" in joined and "main.py" in joined:
                        return p
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as exc:  # pragma: no cover
            logger.debug("psutil.process_iter failed: %s", exc)
        return None

    def _extract_launch_flags(self, proc) -> List[str]:
        try:
            cmdline = proc.info.get("cmdline") or []
        except Exception:
            return []
        return [tok for tok in cmdline if tok.startswith("--")]

    def _infer_install_dir(self, proc) -> Optional[Path]:
        try:
            cmdline = proc.info.get("cmdline") or []
        except Exception:
            return None
        for tok in cmdline:
            tok_l = tok.lower()
            if tok_l.endswith("main.py") and "comfyui" in tok_l:
                p = Path(tok).parent
                if p.exists():
                    return p
        return None

    # ------------------------------------------------------------------
    # File-system probes
    # ------------------------------------------------------------------
    def _list_custom_nodes(self, install_dir: Path) -> List[str]:
        nodes_root = install_dir / "custom_nodes"
        if not nodes_root.exists():
            return []
        out: List[str] = []
        try:
            for child in nodes_root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    out.append(child.name)
        except OSError:
            pass
        return out

    def _read_extra_model_paths(self, install_dir: Path) -> List[Path]:
        cfg = install_dir / "extra_model_paths.yaml"
        if not cfg.exists() or yaml is None:
            return []
        try:
            with cfg.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
        except Exception as exc:
            logger.debug("Failed to parse extra_model_paths.yaml: %s", exc)
            return []
        out: List[Path] = []
        if isinstance(data, dict):
            for section in data.values():
                if not isinstance(section, dict):
                    continue
                base_path = section.get("base_path", "")
                for key, val in section.items():
                    if key == "base_path":
                        continue
                    if isinstance(val, str):
                        for piece in val.splitlines():
                            piece = piece.strip()
                            if piece:
                                out.append(Path(base_path) / piece if base_path else Path(piece))
        return out

    # ------------------------------------------------------------------
    # HTTP probes
    # ------------------------------------------------------------------
    def _http_get_json(self, path: str) -> Optional[Dict[str, Any]]:
        if self._http_failures_this_cycle >= MAX_HTTP_FAILURES_PER_CYCLE:
            return None
        url = f"http://{self.host}:{self.port}{path}"
        try:
            req = _urlrequest.Request(url, headers={"User-Agent": "OptimizerV8/2.0"})
            with _urlrequest.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                raw = resp.read()
                return json.loads(raw)
        except (_urlerror.URLError, _urlerror.HTTPError, json.JSONDecodeError, OSError) as exc:
            self._http_failures_this_cycle += 1
            logger.debug("ComfyUI HTTP %s failed (%d/%d): %s",
                         path, self._http_failures_this_cycle,
                         MAX_HTTP_FAILURES_PER_CYCLE, exc)
            return None
        except Exception as exc:  # pragma: no cover
            self._http_failures_this_cycle += 1
            logger.debug("ComfyUI HTTP %s unexpected error: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # History ingest -> ring buffer
    # ------------------------------------------------------------------
    def _ingest_history(self, history: Dict[str, Any]) -> None:
        if not isinstance(history, dict):
            return
        now = time.time()
        for prompt_id, entry in history.items():
            if prompt_id in self._seen_prompt_ids:
                continue
            if not isinstance(entry, dict):
                continue
            try:
                ev = self._parse_history_entry(prompt_id, entry, now)
            except Exception:
                ev = None
            if ev is None:
                continue
            # Bound _seen_prompt_ids to ring size: when the deque is about to
            # evict its oldest entry, drop that prompt_id from the set first.
            if (
                self._ring.maxlen is not None
                and len(self._ring) == self._ring.maxlen
                and self._ring
            ):
                evicted_id = self._ring[0].workflow_hash
                if evicted_id:
                    self._seen_prompt_ids.discard(evicted_id)
            self._ring.append(ev)
            self._seen_prompt_ids.add(prompt_id)
            self._persist_event(ev)

    @staticmethod
    def _parse_history_entry(
        prompt_id: str,
        entry: Dict[str, Any],
        ts: float,
    ) -> Optional[WorkloadEvent]:
        prompt = entry.get("prompt") or []
        graph = {}
        if isinstance(prompt, list) and len(prompt) >= 3 and isinstance(prompt[2], dict):
            graph = prompt[2]
        elif isinstance(prompt, dict):
            graph = prompt
        if not isinstance(graph, dict) or not graph:
            return None

        model_sha = ""
        shape: Tuple[int, int] = (0, 0)
        batch = 1
        sampler = ""
        steps = 0
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            class_type = (node.get("class_type") or "").lower()
            inputs = node.get("inputs") or {}
            if "ckptloader" in class_type or "checkpointloader" in class_type:
                ckpt = inputs.get("ckpt_name") or inputs.get("model_name") or ""
                if ckpt and not model_sha:
                    model_sha = str(ckpt)
            if "ksampler" in class_type or "sampler" in class_type:
                sampler = str(inputs.get("sampler_name") or sampler)
                try:
                    steps = int(inputs.get("steps") or steps)
                except (TypeError, ValueError):
                    pass
            if "emptylatent" in class_type or "latentimage" in class_type:
                try:
                    w = int(inputs.get("width") or shape[0] or 0)
                    h = int(inputs.get("height") or shape[1] or 0)
                    if w and h:
                        shape = (w, h)
                except (TypeError, ValueError):
                    pass
                try:
                    batch = int(inputs.get("batch_size") or batch)
                except (TypeError, ValueError):
                    pass

        wall_seconds = 0.0
        status = entry.get("status") or {}
        if isinstance(status, dict):
            try:
                wall_seconds = float(status.get("execution_time") or 0.0)
            except (TypeError, ValueError):
                pass

        if not model_sha and shape == (0, 0):
            return None

        return WorkloadEvent(
            workflow_hash=str(prompt_id),
            model_sha=model_sha,
            shape=shape,
            batch=batch,
            sampler=sampler,
            steps=steps,
            wall_seconds=wall_seconds,
            timestamp=ts,
        )
