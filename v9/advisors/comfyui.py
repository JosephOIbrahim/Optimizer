"""V9 — ComfyUI diffusion advisor.

Ports core/experts/comfyui_specialist.py AND its state collector
(comfyui_state_collector.py), inlined and cleaned. The advisor collects its
own ComfyUI state inside _analyze (gated on ctx.running["comfyui"]) — context.py
stays free of app coupling per the frozen contract — then applies the decision
rules:

  R1  SageAttention 2: install custom node, or add --use-sage-attention flag
  R2  ComfyUI-TensorRT custom node install
  R3  speculative TRT compile on hot (model, shape) workloads (idle-gated)
  R4  cache eviction when the engine cache nears its disk budget

Build artifacts (TRT compiles) are looked up via self.cache; FP8 promotion is
gated on self.quality_ledger. Both handles may be None -> permissive fallback
(treat as "not cached" / "not admitted") so the advisor still surfaces work.
The advisor never imports the build/telemetry layer directly.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Optional
from urllib import error as _urlerror
from urllib import request as _urlrequest

from v9.advisors.base import Advisor
from v9.recommend import Action, Recommendation

if TYPE_CHECKING:
    from v9.context import SystemContext

log = logging.getLogger("v9.advisor.comfyui")

_HTTP_TIMEOUT_S = 1.0
_MAX_HTTP_FAILURES = 3
_RING_MAX = 512

_NODE_REPOS = {
    "ComfyUI-SageAttention": ("thu-ml/SageAttention", "v2.0"),
    "ComfyUI_TensorRT": ("comfyanonymous/ComfyUI_TensorRT", "main"),
}
_SAGE_FLAG = "--use-sage-attention"


# ---------------------------------------------------------------------------
# Workload event + in-memory hot-shape histogram
# ---------------------------------------------------------------------------
@dataclass
class _WorkloadEvent:
    workflow_hash: str
    model_sha: str
    shape: tuple[int, int]            # (width, height)
    batch: int
    sampler: str
    steps: int
    wall_seconds: float
    timestamp: float


@dataclass
class _ComfyUIState:
    process_running: bool = False
    api_reachable: bool = False
    install_dir: Optional[Path] = None
    custom_nodes_installed: list[str] = field(default_factory=list)
    launch_flags: list[str] = field(default_factory=list)
    queue_pending: int = 0
    queue_running: int = 0
    recent_workloads: Deque[_WorkloadEvent] = field(
        default_factory=lambda: deque(maxlen=_RING_MAX))

    def hot_shapes(self, min_count: int = 3,
                   lookback_days: int = 14) -> list[tuple[str, tuple[int, int]]]:
        if not self.recent_workloads:
            return []
        horizon = time.time() - lookback_days * 86400
        counts: dict[tuple[str, tuple[int, int]], int] = {}
        for ev in self.recent_workloads:
            if ev.timestamp < horizon:
                continue
            counts[(ev.model_sha, ev.shape)] = \
                counts.get((ev.model_sha, ev.shape), 0) + 1
        return [k for k, c in counts.items() if c >= min_count]


def _derive_shape_range(shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """(W,H) -> (min,opt,max) as 4D (B,C,H,W) for TRT dynamic-shape profiles."""
    w, h = shape
    if w <= 0 or h <= 0:
        w, h = 512, 512
    return {
        "min": (1, 4, max(h - 64, 64), max(w - 64, 64)),
        "opt": (1, 4, h, w),
        "max": (2, 4, h + 64, w + 64),
    }


class ComfyUIAdvisor(Advisor):
    name = "comfyui"
    min_confidence = 0.80

    def __init__(self, config=None, cache_handle=None, quality_ledger=None):
        super().__init__(config, cache_handle, quality_ledger)
        self.host = str(self.config.get("api_host", "127.0.0.1"))
        self.port = int(self.config.get("api_port", 8188))
        trt = self.config.get("tensorrt") or {}
        self.trt_version = str(trt.get("version", "11.x"))
        self.cache_size_gb = float(trt.get("cache_size_gb", 50.0))
        spec = self.config.get("speculative_compile") or {}
        self.spec_min_count = int(spec.get("min_usage_count", 3))
        self.spec_lookback_days = int(spec.get("lookback_days", 14))
        self.phase_b = bool(self.config.get("phase_b_enabled", False))
        self.cli_scan_enabled = bool(self.config.get("cli_scan_enabled", True))
        self._cfg_install_dir = self.config.get("install_dir") or ""
        self._cfg_output_dir = self.config.get("output_dir") or ""
        # persistent ring across cycles
        self._ring: Deque[_WorkloadEvent] = deque(maxlen=_RING_MAX)
        self._seen_prompt_ids: set[str] = set()
        self._seen_images: set[str] = set()
        self._http_failures = 0
        self._cached_install_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # cache / quality handle wrappers (permissive fallback when None)
    # ------------------------------------------------------------------
    def _cache_key(self, **kw: Any) -> str:
        if self.cache is None:
            return "uncached:" + "|".join(f"{k}={v}" for k, v in sorted(kw.items()))
        try:
            return self.cache.engine_cache_key(**kw)
        except Exception:
            return "uncached:error"

    def _cache_has(self, key: str) -> bool:
        if self.cache is None:
            return False
        try:
            return bool(self.cache.engine_cache_lookup(key))
        except Exception:
            return False

    def _cache_disk_gb(self, app: str) -> float:
        if self.cache is None:
            return 0.0
        try:
            return float(self.cache.engine_cache_disk_used_gb(app))
        except Exception:
            return 0.0

    def _fp8_admitted(self, app: str, model_sha: str) -> bool:
        if self.quality_ledger is None:
            return False
        try:
            return bool(self.quality_ledger.fp8_admitted(app, model_sha))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------
    def _analyze(self, ctx: "SystemContext") -> list[Recommendation]:
        # Frozen contract: app advisors are gated on ctx.running[app]. During a
        # `comfy run` CLI render the ComfyUI python process IS live, so this gate
        # still fires for CLI workloads — _collect() additionally scans the
        # output dir so hot-shape detection survives an empty /history.
        if not ctx.running.get("comfyui"):
            return []

        state = self._collect(ctx)
        ctx.comfyui = state          # publish for the orchestrator/coordinator
        if not state.process_running:
            return []

        recs: list[Recommendation] = []
        installed = set(state.custom_nodes_installed)
        flags = state.launch_flags

        # ---- R1: SageAttention 2 -----------------------------------------
        if "ComfyUI-SageAttention" not in installed:
            repo, ref = _NODE_REPOS["ComfyUI-SageAttention"]
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.INSTALL_CUSTOM_NODE,
                target="comfyui:sage_attention",
                confidence=0.92,
                expected_impact_pct=50.0,
                rationale=(
                    "ComfyUI-SageAttention not installed. SageAttention 2 gives "
                    "~1.5x SDPA speedup on Ada FP16/BF16 kernels; install and "
                    "relaunch."
                ),
                parameters={"node_repo": repo, "ref": ref},
            ))
        elif _SAGE_FLAG not in flags:
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.SET_LAUNCH_FLAG,
                target="comfyui:launch_flag:sage",
                confidence=0.92,
                expected_impact_pct=50.0,
                rationale=(
                    "ComfyUI-SageAttention is installed but ComfyUI was started "
                    f"without {_SAGE_FLAG}; the SDPA patch only activates with "
                    "the flag present."
                ),
                parameters={"add_flag": _SAGE_FLAG, "restart_required": True},
            ))

        # ---- R2: ComfyUI-TensorRT node -----------------------------------
        if "ComfyUI_TensorRT" not in installed:
            repo, ref = _NODE_REPOS["ComfyUI_TensorRT"]
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.INSTALL_CUSTOM_NODE,
                target="comfyui:tensorrt_node",
                confidence=0.88,
                expected_impact_pct=85.0,
                rationale=(
                    "ComfyUI_TensorRT node not installed; required to compile + "
                    "load TRT engines for diffusion UNets (up to ~1.85x vs torch)."
                ),
                parameters={"node_repo": repo, "ref": ref},
            ))

        recs.extend(self._speculative_and_evict(state))
        return recs

    # ------------------------------------------------------------------
    # R3 + R4: usable from both the live-server and CLI-render paths
    # ------------------------------------------------------------------
    def _speculative_and_evict(self, state: "_ComfyUIState"
                               ) -> list[Recommendation]:
        recs: list[Recommendation] = []
        # ---- R3: speculative TRT compile on hot shapes -------------------
        for model_sha, shape in state.hot_shapes(
                self.spec_min_count, self.spec_lookback_days):
            shape_range = _derive_shape_range(shape)
            key = self._cache_key(app="comfyui", model=model_sha, shape=shape,
                                  precision="fp16", backend="tensorrt",
                                  backend_ver=self.trt_version)
            if not self._cache_has(key):
                recs.append(Recommendation(
                    advisor=self.name,
                    action=Action.COMPILE_TRT_ENGINE,
                    target=f"comfyui:{model_sha}:{shape[0]}x{shape[1]}",
                    confidence=0.85,
                    expected_impact_pct=70.0,
                    rationale=(
                        f"Hot workload model={model_sha} shape={shape} "
                        f"(>= {self.spec_min_count} runs / "
                        f"{self.spec_lookback_days}d) has no cached TRT plan; "
                        "speculative idle-gated compile."
                    ),
                    gated_on_idle=True,
                    parameters={"precision": "fp16", "shape_range": shape_range,
                                "model_kind": "unet", "cache_key": key,
                                "app": "comfyui", "model_sha": model_sha},
                ))

            # FP8 promotion (Phase B, quality-gated)
            if self.phase_b and self._fp8_admitted("comfyui", model_sha):
                fp8_key = self._cache_key(app="comfyui", model=model_sha,
                                          shape=shape, precision="fp8",
                                          backend="tensorrt",
                                          backend_ver=self.trt_version)
                if not self._cache_has(fp8_key):
                    recs.append(Recommendation(
                        advisor=self.name,
                        action=Action.COMPILE_TRT_ENGINE,
                        target=f"comfyui:{model_sha}:{shape[0]}x{shape[1]}:fp8",
                        confidence=0.80,
                        expected_impact_pct=120.0,
                        rationale=(
                            "Phase B: quality ledger admitted FP8 for this "
                            "model; ~2x on Ada FP8 tensor cores vs FP16."
                        ),
                        gated_on_idle=True,
                        parameters={"precision": "fp8", "shape_range": shape_range,
                                    "model_kind": "unet", "cache_key": fp8_key,
                                    "app": "comfyui", "model_sha": model_sha},
                    ))

        # ---- R4: cache eviction ------------------------------------------
        used_gb = self._cache_disk_gb("comfyui")
        if used_gb > self.cache_size_gb * 0.90:
            recs.append(Recommendation(
                advisor=self.name,
                action=Action.EVICT_ENGINE_CACHE,
                target="comfyui:cache",
                confidence=0.95,
                expected_impact_pct=0.0,
                rationale=(
                    f"Engine cache {used_gb:.1f} GB > 90% of "
                    f"{self.cache_size_gb:.1f} GB budget; evict LRU plans."
                ),
                parameters={"app": "comfyui", "used_gb": used_gb,
                            "limit_gb": self.cache_size_gb, "immediate": True},
            ))

        return recs

    # ------------------------------------------------------------------
    # state collection (inlined collector; never raises)
    # ------------------------------------------------------------------
    def _collect(self, ctx: "SystemContext") -> _ComfyUIState:
        self._http_failures = 0
        state = _ComfyUIState(process_running=True)
        state.recent_workloads = self._ring
        try:
            proc = self._find_proc()
            if proc is not None:
                state.launch_flags = self._launch_flags(proc)
                state.install_dir = self._install_dir(proc) or self._cached_install_dir
                self._cached_install_dir = state.install_dir
            else:
                state.install_dir = self._cached_install_dir

            # robust install-dir resolution: proc cmdline, then config /
            # comfy-cli workspace / common locations.
            if state.install_dir is None:
                state.install_dir = self._detect_install_dir()
            if state.install_dir is not None:
                state.custom_nodes_installed = self._list_custom_nodes(state.install_dir)
                # CLI renders: recover (model, shape) from output PNGs so hot
                # shapes are known even when /history is empty (fresh server).
                if self.cli_scan_enabled:
                    try:
                        self._scan_output_dir(state.install_dir)
                    except Exception:
                        log.debug("comfyui output scan degraded", exc_info=True)

            queue = self._http_json("/queue") or {}
            try:
                state.queue_running = len(queue.get("queue_running") or [])
                state.queue_pending = len(queue.get("queue_pending") or [])
            except Exception:
                pass
            stats = self._http_json("/system_stats") or {}
            self._ingest_history(self._http_json("/history?max_items=20") or {})
            state.api_reachable = (self._http_failures < _MAX_HTTP_FAILURES
                                   and bool(stats))
        except Exception:
            log.debug("comfyui state collect degraded", exc_info=True)
        return state

    # ------------------------------------------------------------------
    # CLI-render path: detect install, scan outputs (no live server needed)
    # ------------------------------------------------------------------
    def _detect_install_dir(self) -> Optional[Path]:
        if self._cached_install_dir and self._cached_install_dir.exists():
            return self._cached_install_dir
        candidates: list[Path] = []
        if self._cfg_install_dir:
            candidates.append(Path(self._cfg_install_dir))
        # comfy-cli default workspace (best-effort across config layouts)
        home = Path(os.path.expanduser("~"))
        for rel in (".config/comfy-cli/config.ini", ".comfy/config.ini"):
            cfg = home / rel
            ws = self._comfy_cli_workspace(cfg)
            if ws:
                candidates.append(ws)
        candidates += [Path("C:/ComfyUI"), home / "ComfyUI"]
        for c in candidates:
            try:
                if c.exists() and (c / "main.py").exists():
                    self._cached_install_dir = c
                    return c
            except OSError:
                continue
        return None

    @staticmethod
    def _comfy_cli_workspace(cfg_path: Path) -> Optional[Path]:
        try:
            if not cfg_path.exists():
                return None
            import configparser
            cp = configparser.ConfigParser()
            cp.read(cfg_path, encoding="utf-8")
            for sect in cp.sections():
                for key in ("default_workspace", "workspace", "recent_workspace"):
                    if cp.has_option(sect, key):
                        p = Path(cp.get(sect, key))
                        if p.exists():
                            return p
        except Exception:  # noqa: BLE001
            pass
        return None

    def _scan_output_dir(self, install_dir: Path, max_images: int = 64) -> None:
        out_dir = (Path(self._cfg_output_dir) if self._cfg_output_dir
                   else install_dir / "output")
        if not out_dir.exists():
            return
        try:
            pngs = sorted((p for p in out_dir.rglob("*.png") if p.is_file()),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return
        now = time.time()
        for png in pngs[:max_images]:
            key = f"{png}:{int(png.stat().st_mtime)}"
            if key in self._seen_images:
                continue
            self._seen_images.add(key)
            graph = self._read_png_prompt(png)
            if not graph:
                continue
            ev = self._graph_to_event(str(png.name), graph, now)
            if ev is not None:
                self._ring.append(ev)

    @staticmethod
    def _read_png_prompt(path: Path) -> Optional[dict]:
        """Extract ComfyUI's embedded 'prompt' graph from a PNG tEXt chunk
        using stdlib only (no PIL dependency)."""
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        idx = 8
        n = len(data)
        while idx + 8 <= n:
            length = int.from_bytes(data[idx:idx + 4], "big")
            ctype = data[idx + 4:idx + 8]
            body = data[idx + 8:idx + 8 + length]
            idx += 12 + length  # 4 len + 4 type + body + 4 crc
            if ctype in (b"tEXt", b"iTXt"):
                try:
                    text = body.split(b"\x00", 1)
                    keyword = text[0].decode("latin-1", "ignore")
                    if keyword.lower() in ("prompt", "workflow") and len(text) > 1:
                        raw = text[1]
                        # iTXt has extra null-separated fields before the value
                        if ctype == b"iTXt":
                            raw = raw.split(b"\x00")[-1]
                        obj = json.loads(raw.decode("utf-8", "ignore"))
                        if isinstance(obj, dict):
                            return obj
                except Exception:  # noqa: BLE001
                    continue
            if ctype == b"IEND":
                break
        return None

    def _graph_to_event(self, ident: str, graph: dict,
                        ts: float) -> Optional[_WorkloadEvent]:
        model_sha = ""
        shape: tuple[int, int] = (0, 0)
        batch, sampler, steps = 1, "", 0
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            ct = (node.get("class_type") or "").lower()
            inputs = node.get("inputs") or {}
            if "ckptloader" in ct or "checkpointloader" in ct:
                ckpt = inputs.get("ckpt_name") or inputs.get("model_name") or ""
                if ckpt and not model_sha:
                    model_sha = str(ckpt)
            if "ksampler" in ct or "sampler" in ct:
                sampler = str(inputs.get("sampler_name") or sampler)
                try:
                    steps = int(inputs.get("steps") or steps)
                except (TypeError, ValueError):
                    pass
            if "emptylatent" in ct or "latentimage" in ct:
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
        if not model_sha and shape == (0, 0):
            return None
        return _WorkloadEvent(workflow_hash=ident, model_sha=model_sha,
                              shape=shape, batch=batch, sampler=sampler,
                              steps=steps, wall_seconds=0.0, timestamp=ts)

    def _find_proc(self):
        try:
            import psutil  # type: ignore
        except Exception:
            return None
        try:
            for p in psutil.process_iter(attrs=["name", "cmdline"]):
                try:
                    name = (p.info.get("name") or "").lower()
                    if "python" not in name:
                        continue
                    joined = " ".join(p.info.get("cmdline") or []).lower()
                    if "comfyui" in joined and "main.py" in joined:
                        return p
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            return None
        return None

    @staticmethod
    def _launch_flags(proc) -> list[str]:
        try:
            return [t for t in (proc.info.get("cmdline") or []) if t.startswith("--")]
        except Exception:
            return []

    @staticmethod
    def _install_dir(proc) -> Optional[Path]:
        try:
            for tok in proc.info.get("cmdline") or []:
                tl = tok.lower()
                if tl.endswith("main.py") and "comfyui" in tl:
                    p = Path(tok).parent
                    if p.exists():
                        return p
        except Exception:
            pass
        return None

    @staticmethod
    def _list_custom_nodes(install_dir: Path) -> list[str]:
        root = install_dir / "custom_nodes"
        if not root.exists():
            return []
        out: list[str] = []
        try:
            for child in root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    out.append(child.name)
        except OSError:
            pass
        return out

    def _http_json(self, path: str) -> Optional[dict]:
        if self._http_failures >= _MAX_HTTP_FAILURES:
            return None
        url = f"http://{self.host}:{self.port}{path}"
        try:
            req = _urlrequest.Request(url, headers={"User-Agent": "OptimizerV9"})
            with _urlrequest.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except (_urlerror.URLError, _urlerror.HTTPError,
                json.JSONDecodeError, OSError):
            self._http_failures += 1
            return None
        except Exception:
            self._http_failures += 1
            return None

    def _ingest_history(self, history: dict) -> None:
        if not isinstance(history, dict):
            return
        now = time.time()
        for prompt_id, entry in history.items():
            if prompt_id in self._seen_prompt_ids or not isinstance(entry, dict):
                continue
            ev = self._parse_history_entry(prompt_id, entry, now)
            if ev is None:
                continue
            if (self._ring.maxlen is not None
                    and len(self._ring) == self._ring.maxlen and self._ring):
                evicted = self._ring[0].workflow_hash
                if evicted:
                    self._seen_prompt_ids.discard(evicted)
            self._ring.append(ev)
            self._seen_prompt_ids.add(prompt_id)

    @staticmethod
    def _parse_history_entry(prompt_id: str, entry: dict,
                             ts: float) -> Optional[_WorkloadEvent]:
        prompt = entry.get("prompt") or []
        graph: dict = {}
        if isinstance(prompt, list) and len(prompt) >= 3 and isinstance(prompt[2], dict):
            graph = prompt[2]
        elif isinstance(prompt, dict):
            graph = prompt
        if not graph:
            return None
        model_sha = ""
        shape: tuple[int, int] = (0, 0)
        batch = 1
        sampler = ""
        steps = 0
        for node in graph.values():
            if not isinstance(node, dict):
                continue
            ct = (node.get("class_type") or "").lower()
            inputs = node.get("inputs") or {}
            if "ckptloader" in ct or "checkpointloader" in ct:
                ckpt = inputs.get("ckpt_name") or inputs.get("model_name") or ""
                if ckpt and not model_sha:
                    model_sha = str(ckpt)
            if "ksampler" in ct or "sampler" in ct:
                sampler = str(inputs.get("sampler_name") or sampler)
                try:
                    steps = int(inputs.get("steps") or steps)
                except (TypeError, ValueError):
                    pass
            if "emptylatent" in ct or "latentimage" in ct:
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
        wall = 0.0
        status = entry.get("status") or {}
        if isinstance(status, dict):
            try:
                wall = float(status.get("execution_time") or 0.0)
            except (TypeError, ValueError):
                pass
        if not model_sha and shape == (0, 0):
            return None
        return _WorkloadEvent(workflow_hash=str(prompt_id), model_sha=model_sha,
                              shape=shape, batch=batch, sampler=sampler,
                              steps=steps, wall_seconds=wall, timestamp=ts)
