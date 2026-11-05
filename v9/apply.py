"""V9 — Applier: turn a Recommendation into a real-world effect.

Two families, matching recommend.Action.is_build:

  * CONFIG family (instant, reversible side-effects):
      SET_ENV_VAR        -> os.environ + persistent user env (setx on Windows)
      REGISTRY_TWEAK     -> winreg write (e.g. HAGS HwSchMode)
      PROCESS_PRIORITY   -> psutil nice/priority class on matching processes
      CPU_AFFINITY       -> psutil cpu_affinity on matching processes
      VRAM_ALLOCATION    -> advisory budget record (no kernel-level fence here)
      ADJUST_QUALITY     -> write cvars into UE5 Engine.ini ([section])
      SET_LAUNCH_FLAG    -> persist relaunch advice + NOTIFY (never kills a proc)
      NOTIFY             -> surface info, no mutation

  * BUILD family (expensive, cached, background):
      COMPILE_TRT_ENGINE -> enqueue a CompileJob on the BackgroundCompiler
      ENABLE_BACKEND     -> compiler control (pause/resume) or enable advice
      INSTALL_CUSTOM_NODE / INSTALL_PLUGIN -> persist install advice + NOTIFY
      WIRE_BACKEND_PATCH -> persist wiring advice + NOTIFY
      EVICT_ENGINE_CACHE -> EngineCacheManager.evict_lru()

Hard invariant (DESIGN_V9 §4.2): an applier NEVER raises out of `apply()`.
Every dispatch returns an `ApplyResult(applied, message)`; a failure is a
False/`message`, not an exception that kills the cycle.

Honest capability (§4.4): `applied=True` is set ONLY when something actually
changed (an env var was set, a job was enqueued, a row was evicted). Advice-only
actions (SET_LAUNCH_FLAG, INSTALL_*, WIRE_*) report applied=True for "advice
persisted + surfaced" but their message makes clear no process was touched.
"""
from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from v9 import paths
from v9.recommend import Action, Recommendation

log = logging.getLogger("v9.apply")

_IS_WINDOWS = os.name == "nt"

# Windows priority-class names psutil understands.
_PRIORITY_MAP = {
    "IDLE": "IDLE_PRIORITY_CLASS",
    "BELOW_NORMAL": "BELOW_NORMAL_PRIORITY_CLASS",
    "NORMAL": "NORMAL_PRIORITY_CLASS",
    "ABOVE_NORMAL": "ABOVE_NORMAL_PRIORITY_CLASS",
    "HIGH": "HIGH_PRIORITY_CLASS",
    "REALTIME": "REALTIME_PRIORITY_CLASS",
}

# Process-name substrings per logical app (for priority/affinity matching).
_APP_PROC_MATCH = {
    "comfyui": ("python",),     # refined by cmdline "comfyui" + "main.py"
    "houdini": ("houdini", "hindie", "husk"),
    "ue5": ("unrealeditor", "ue5editor", "ue4editor", "unreal"),
}


@dataclass
class ApplyResult:
    applied: bool
    message: str
    action: Optional[Action] = None
    target: str = ""
    advisory: bool = False          # True when this only recorded/surfaced advice
    details: dict[str, Any] = field(default_factory=dict)


class Applier:
    """Dispatches Recommendations to side-effects. Build-family actions need a
    BackgroundCompiler + EngineCacheManager; pass them in. Any may be None ->
    that family degrades to advice-only, never crashes."""

    def __init__(
        self,
        compiler: Any = None,            # v9.build.compiler.BackgroundCompiler
        cache: Any = None,               # v9.build.cache.EngineCacheManager
        registry: Any = None,            # v9.accel.registry.BackendRegistry
        config: Any = None,              # v9.config.Config (for env scope etc.)
        dry_run: bool = False,
    ) -> None:
        self.compiler = compiler
        self.cache = cache
        self.registry = registry
        self.config = config
        self.dry_run = bool(dry_run)
        # advice ledger (relaunch flags, install hints) — persisted off-OneDrive
        self._advice_path = paths.queue_path("pending_advice")

    # ------------------------------------------------------------------
    # public entry point — never raises
    # ------------------------------------------------------------------
    def apply(self, rec: Recommendation) -> ApplyResult:
        handler = self._DISPATCH.get(rec.action)
        if handler is None:
            return ApplyResult(False, f"no applier for {rec.action.value}",
                               rec.action, rec.target)
        try:
            return handler(self, rec)
        except Exception as exc:  # noqa: BLE001 - applier must never crash loop
            log.warning("applier for %s/%s raised: %s",
                        rec.action.value, rec.target, exc, exc_info=True)
            return ApplyResult(False, f"applier error: {exc}",
                               rec.action, rec.target)

    def apply_all(self, recs: list[Recommendation],
                  top_n: Optional[int] = None) -> list[ApplyResult]:
        chosen = recs[:top_n] if top_n is not None else recs
        return [self.apply(r) for r in chosen]

    # ==================================================================
    # CONFIG family
    # ==================================================================
    def _apply_set_env_var(self, rec: Recommendation) -> ApplyResult:
        p = rec.parameters
        var = p.get("variable") or rec.target
        value = str(p.get("value", ""))
        scope = p.get("scope", "user")
        if self.dry_run:
            return ApplyResult(True, f"[dry-run] would set {var}={value}",
                               rec.action, rec.target, advisory=True)
        # process env (immediate, this process + children we spawn)
        os.environ[var] = value
        persisted = False
        if _IS_WINDOWS and scope in ("user", "machine"):
            persisted = self._setx(var, value, machine=(scope == "machine"))
        msg = (f"set {var}={value} for this session"
               + ("; persisted to user env (new processes only)"
                  if persisted else
                  "; not persisted (set for session only)"))
        return ApplyResult(True, msg, rec.action, rec.target,
                           details={"variable": var, "persisted": persisted})

    @staticmethod
    def _setx(var: str, value: str, machine: bool = False) -> bool:
        """Persist an env var via setx. Returns True on success. Never raises."""
        cmd = ["setx", var, value] + (["/M"] if machine else [])
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            return out.returncode == 0
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("setx %s failed: %s", var, exc)
            return False

    def _apply_registry_tweak(self, rec: Recommendation) -> ApplyResult:
        p = rec.parameters
        key = p.get("key", "")
        value_name = p.get("value_name", "")
        value_data = p.get("value_data")
        value_type = p.get("value_type", "REG_DWORD")
        if not _IS_WINDOWS:
            return ApplyResult(False, "registry tweaks are Windows-only",
                               rec.action, rec.target)
        if self.dry_run:
            return ApplyResult(True,
                               f"[dry-run] would set {key}\\{value_name}={value_data}",
                               rec.action, rec.target, advisory=True)
        ok, msg = self._winreg_set(key, value_name, value_data, value_type)
        if ok and p.get("requires_reboot"):
            msg += " (reboot required to take effect)"
        return ApplyResult(ok, msg, rec.action, rec.target)

    @staticmethod
    def _winreg_set(key: str, value_name: str, value_data: Any,
                    value_type: str) -> tuple[bool, str]:
        try:
            import winreg  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return False, f"winreg unavailable: {exc}"
        roots = {
            "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
            "HKLM": winreg.HKEY_LOCAL_MACHINE,
            "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
            "HKCU": winreg.HKEY_CURRENT_USER,
        }
        root_name, _, subkey = key.replace("/", "\\").partition("\\")
        root = roots.get(root_name.upper())
        if root is None:
            return False, f"unknown registry root: {root_name!r}"
        reg_type = getattr(winreg, value_type, winreg.REG_DWORD)
        data = (int(value_data) if reg_type == winreg.REG_DWORD
                else str(value_data))
        try:
            handle = winreg.CreateKeyEx(root, subkey, 0, winreg.KEY_SET_VALUE)
            try:
                winreg.SetValueEx(handle, value_name, 0, reg_type, data)
            finally:
                winreg.CloseKey(handle)
            return True, f"set {key}\\{value_name}={value_data}"
        except PermissionError:
            return False, (f"access denied writing {key}\\{value_name}; "
                           "run as Administrator")
        except OSError as exc:
            return False, f"registry write failed: {exc}"

    def _apply_process_priority(self, rec: Recommendation) -> ApplyResult:
        p = rec.parameters
        app = p.get("app", "")
        priority = str(p.get("priority", "NORMAL")).upper()
        procs = self._find_procs(app, p.get("match"))
        if not procs:
            return ApplyResult(False, f"no running process matched {app!r}",
                               rec.action, rec.target)
        if self.dry_run:
            return ApplyResult(True,
                               f"[dry-run] would set {len(procs)} {app} proc(s) "
                               f"to {priority}", rec.action, rec.target,
                               advisory=True)
        changed = self._set_priority(procs, priority)
        if changed == 0:
            return ApplyResult(False,
                               f"could not set priority on {app} (access denied?)",
                               rec.action, rec.target)
        return ApplyResult(True, f"set {changed} {app} process(es) to {priority}",
                           rec.action, rec.target,
                           details={"app": app, "count": changed})

    @staticmethod
    def _set_priority(procs: list, priority: str) -> int:
        try:
            import psutil  # type: ignore
        except Exception:
            return 0
        cls_name = _PRIORITY_MAP.get(priority, "NORMAL_PRIORITY_CLASS")
        prio_value = getattr(psutil, cls_name, None)
        changed = 0
        for proc in procs:
            try:
                if prio_value is not None:
                    proc.nice(prio_value)
                changed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue
        return changed

    def _apply_cpu_affinity(self, rec: Recommendation) -> ApplyResult:
        p = rec.parameters
        app = p.get("app", "")
        cores = list(p.get("cores") or [])
        if not cores:
            return ApplyResult(False, "no cores specified", rec.action, rec.target)
        procs = self._find_procs(app, p.get("match"))
        if not procs:
            return ApplyResult(False, f"no running process matched {app!r}",
                               rec.action, rec.target)
        if self.dry_run:
            return ApplyResult(True,
                               f"[dry-run] would pin {app} to cores "
                               f"{cores[0]}-{cores[-1]}", rec.action, rec.target,
                               advisory=True)
        changed = self._set_affinity(procs, cores)
        if changed == 0:
            return ApplyResult(False, f"could not pin {app} affinity",
                               rec.action, rec.target)
        return ApplyResult(True,
                           f"pinned {changed} {app} proc(s) to cores "
                           f"{cores[0]}-{cores[-1]}", rec.action, rec.target,
                           details={"app": app, "cores": cores})

    @staticmethod
    def _set_affinity(procs: list, cores: list[int]) -> int:
        try:
            import psutil  # type: ignore
        except Exception:
            return 0
        changed = 0
        for proc in procs:
            try:
                proc.cpu_affinity(cores)
                changed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue
        return changed

    def _apply_vram_allocation(self, rec: Recommendation) -> ApplyResult:
        """No kernel-level VRAM fence here — record the advisory budget so the
        coordinator/monitor and downstream consumers can honor it. Honest: we
        do not pretend to have hard-fenced VRAM."""
        p = rec.parameters
        if p.get("trigger_gc"):
            return ApplyResult(True,
                               "advised GC / cache shrink to relieve VRAM "
                               "pressure (advisory; apps own their allocators)",
                               rec.action, rec.target, advisory=True)
        limit = p.get("limit_gb")
        app = p.get("app", rec.target)
        return ApplyResult(True,
                           f"recorded VRAM budget {limit} GB for {app} "
                           "(advisory budget; no hard fence)",
                           rec.action, rec.target, advisory=True,
                           details={"app": app, "limit_gb": limit})

    def _apply_adjust_quality(self, rec: Recommendation) -> ApplyResult:
        """Write cvars into a UE5 Engine.ini under [section]. If no ini path is
        known, persist as advice."""
        p = rec.parameters
        ini_path = p.get("engine_ini_path")
        section = p.get("section", "/Script/Engine.RendererSettings")
        cvars: dict[str, Any] = p.get("cvars") or {}
        if not cvars:
            return ApplyResult(True, f"quality guidance: {rec.rationale}",
                               rec.action, rec.target, advisory=True)
        if not ini_path:
            self._persist_advice(rec, kind="adjust_quality")
            return ApplyResult(True,
                               "no Engine.ini located; recorded cvar advice "
                               f"({len(cvars)} cvars) for next session",
                               rec.action, rec.target, advisory=True)
        if self.dry_run:
            return ApplyResult(True,
                               f"[dry-run] would write {len(cvars)} cvars to "
                               f"{ini_path}", rec.action, rec.target, advisory=True)
        ok, msg = self._write_ini_cvars(Path(ini_path), section, cvars)
        return ApplyResult(ok, msg, rec.action, rec.target,
                           details={"ini": ini_path, "cvars": list(cvars)})

    @staticmethod
    def _write_ini_cvars(ini: Path, section: str,
                         cvars: dict[str, Any]) -> tuple[bool, str]:
        import configparser
        parser = configparser.ConfigParser(strict=False, interpolation=None)
        parser.optionxform = str  # cvars are case-sensitive
        try:
            if ini.exists():
                parser.read(ini, encoding="utf-8")
        except (configparser.Error, OSError, UnicodeDecodeError) as exc:
            return False, f"could not read {ini}: {exc}"
        if not parser.has_section(section):
            try:
                parser.add_section(section)
            except (ValueError, configparser.Error) as exc:
                return False, f"could not add section {section}: {exc}"
        for k, v in cvars.items():
            parser.set(section, k, str(v))
        try:
            ini.parent.mkdir(parents=True, exist_ok=True)
            tmp = ini.with_suffix(ini.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                parser.write(fh)
            os.replace(tmp, ini)
        except OSError as exc:
            return False, f"could not write {ini}: {exc}"
        return True, f"wrote {len(cvars)} cvars to [{section}] in {ini.name}"

    def _apply_set_launch_flag(self, rec: Recommendation) -> ApplyResult:
        """Persist relaunch advice + surface it. NEVER kills a running process
        (DESIGN: SET_LAUNCH_FLAG/INSTALL_* persist advice + notify only)."""
        flag = rec.parameters.get("add_flag", "")
        self._persist_advice(rec, kind="launch_flag")
        return ApplyResult(True,
                           f"advice recorded: relaunch with {flag} "
                           "(running process left untouched)",
                           rec.action, rec.target, advisory=True,
                           details={"flag": flag})

    def _apply_notify(self, rec: Recommendation) -> ApplyResult:
        log.info("NOTIFY [%s] %s: %s", rec.advisor, rec.target, rec.rationale)
        return ApplyResult(True, rec.rationale or rec.target,
                           rec.action, rec.target, advisory=True)

    # ==================================================================
    # BUILD family
    # ==================================================================
    def _apply_compile_trt_engine(self, rec: Recommendation) -> ApplyResult:
        if self.compiler is None:
            self._persist_advice(rec, kind="compile_trt")
            return ApplyResult(True,
                               "no background compiler wired; recorded compile "
                               "request as advice (advisory)",
                               rec.action, rec.target, advisory=True)
        params = self._build_compile_params(rec)
        if self.dry_run:
            return ApplyResult(True,
                               f"[dry-run] would enqueue TRT compile for "
                               f"{rec.target}", rec.action, rec.target,
                               advisory=True)
        try:
            job_id = self.compiler.enqueue(params)
        except Exception as exc:  # noqa: BLE001
            return ApplyResult(False, f"enqueue failed: {exc}",
                               rec.action, rec.target)
        return ApplyResult(True,
                           f"enqueued background TRT compile (job {job_id[:8]}, "
                           "idle-gated)", rec.action, rec.target,
                           details={"job_id": job_id})

    def _build_compile_params(self, rec: Recommendation) -> dict[str, Any]:
        """Map a COMPILE_TRT_ENGINE Recommendation to a BackgroundCompiler
        compile_params dict. Shapes come from the advisor's shape_range; the
        backend defaults to tensorrt. driver/arch are NOT passed (the cache
        owns those via HardwareProfile)."""
        p = dict(rec.parameters)
        app = p.get("app", "comfyui")
        shape_range = p.get("shape_range")
        shape_w = shape_h = 0
        if isinstance(shape_range, dict) and shape_range.get("opt"):
            opt = list(shape_range["opt"])
            # opt is (B,C,H,W); W last, H second-last
            if len(opt) >= 2:
                shape_w, shape_h = int(opt[-1]), int(opt[-2])
        params: dict[str, Any] = {
            "app": app,
            "backend_name": p.get("backend_name", "tensorrt"),
            "backend_ver": p.get("backend_ver", ""),
            "model_sha256": p.get("model_sha") or p.get("model_sha256", ""),
            "shape_w": shape_w,
            "shape_h": shape_h,
            "batch": int(p.get("batch", 1)),
            "precision": p.get("precision", "fp16"),
            "model_kind": p.get("model_kind", "unet"),
            "priority": int(p.get("priority", 0)),
        }
        if shape_range:
            params["shape_range"] = shape_range
        # carry an ONNX source through if the advisor found one (Houdini ML SOP)
        src = p.get("source_onnx")
        if src:
            params["source"] = {"kind": "onnx", "path": str(src)}
        return params

    def _apply_enable_backend(self, rec: Recommendation) -> ApplyResult:
        """Two meanings collapse onto ENABLE_BACKEND:
          * coordinator pause/resume of background compiles for an app
          * an advisor advising a backend be enabled (e.g. cuTensor Vellum)."""
        p = rec.parameters
        control = p.get("action")
        if control == "pause_background_compiles":
            app = p.get("app", "")
            if self.compiler is None:
                return ApplyResult(False, "no compiler wired to pause",
                                   rec.action, rec.target)
            if self.dry_run:
                return ApplyResult(True, f"[dry-run] would pause {app} compiles",
                                   rec.action, rec.target, advisory=True)
            n = self.compiler.pause_app(app, reason="foreground_active")
            return ApplyResult(True,
                               f"paused {n} background compile job(s) for {app}",
                               rec.action, rec.target, details={"app": app})
        if control == "resume_background_compiles":
            app = p.get("app", "")
            if self.compiler is None:
                return ApplyResult(False, "no compiler wired to resume",
                                   rec.action, rec.target)
            n = self.compiler.resume_app(app)
            return ApplyResult(True, f"resumed {n} compile job(s) for {app}",
                               rec.action, rec.target, details={"app": app})
        # generic "enable a backend" -> advice (install/wiring is user-gated)
        self._persist_advice(rec, kind="enable_backend")
        return ApplyResult(True,
                           f"recorded request to enable backend "
                           f"{p.get('backend', rec.target)} (advisory)",
                           rec.action, rec.target, advisory=True)

    def _apply_install_custom_node(self, rec: Recommendation) -> ApplyResult:
        return self._advice_only_install(rec, "ComfyUI custom node")

    def _apply_install_plugin(self, rec: Recommendation) -> ApplyResult:
        return self._advice_only_install(rec, "plugin")

    def _advice_only_install(self, rec: Recommendation,
                             what: str) -> ApplyResult:
        """INSTALL_* persists advice + notifies; it never auto-installs or kills
        a process. The user (or a separate gated activation step) performs the
        install."""
        repo = (rec.parameters.get("node_repo")
                or rec.parameters.get("plugin")
                or rec.parameters.get("marketplace_url", ""))
        self._persist_advice(rec, kind="install")
        return ApplyResult(True,
                           f"{what} install advised: {repo} "
                           "(recorded; no process touched)",
                           rec.action, rec.target, advisory=True,
                           details={"target": repo})

    def _apply_wire_backend_patch(self, rec: Recommendation) -> ApplyResult:
        self._persist_advice(rec, kind="wire_patch")
        return ApplyResult(True,
                           f"backend wiring advised for {rec.target} "
                           "(recorded; relaunch to take effect)",
                           rec.action, rec.target, advisory=True)

    def _apply_evict_engine_cache(self, rec: Recommendation) -> ApplyResult:
        if self.cache is None:
            return ApplyResult(False, "no engine cache wired to evict",
                               rec.action, rec.target)
        p = rec.parameters
        app = p.get("app")
        limit_gb = p.get("limit_gb")
        target_bytes = (int(float(limit_gb) * 0.9 * 1024 ** 3)
                        if limit_gb else None)
        if self.dry_run:
            return ApplyResult(True, f"[dry-run] would evict LRU for {app}",
                               rec.action, rec.target, advisory=True)
        try:
            freed = self.cache.evict_lru(target_bytes=target_bytes,
                                         target_app=app)
        except Exception as exc:  # noqa: BLE001
            return ApplyResult(False, f"eviction failed: {exc}",
                               rec.action, rec.target)
        freed_gb = freed / 1024 ** 3
        if freed == 0:
            return ApplyResult(True,
                               "cache already within budget (nothing evicted)",
                               rec.action, rec.target, advisory=True)
        return ApplyResult(True, f"evicted {freed_gb:.2f} GB of LRU engines "
                           f"for {app or 'all apps'}", rec.action, rec.target,
                           details={"freed_gb": freed_gb})

    # ==================================================================
    # helpers
    # ==================================================================
    @staticmethod
    def _find_procs(app: str, match: Any = None) -> list:
        """Return live psutil.Process objects for the app. Never raises."""
        try:
            import psutil  # type: ignore
        except Exception:
            return []
        needles = tuple(n.lower() for n in
                        (match or _APP_PROC_MATCH.get(app, (app,))))
        out = []
        try:
            for p in psutil.process_iter(attrs=["name", "cmdline"]):
                try:
                    name = (p.info.get("name") or "").lower()
                    if app == "comfyui":
                        cmd = " ".join(p.info.get("cmdline") or []).lower()
                        if "python" in name and "main.py" in cmd and "comfyui" in cmd:
                            out.append(p)
                        continue
                    if any(n in name for n in needles):
                        out.append(p)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            return out
        return out

    def _persist_advice(self, rec: Recommendation, kind: str) -> None:
        """Append an advisory entry to the off-OneDrive advice ledger. Never
        raises — advice persistence is best-effort."""
        try:
            import json
            entry = {
                "kind": kind,
                "advisor": rec.advisor,
                "action": rec.action.value,
                "target": rec.target,
                "rationale": rec.rationale,
                "parameters": {k: _jsonable(v)
                               for k, v in rec.parameters.items()},
            }
            existing = []
            if self._advice_path.exists():
                try:
                    existing = json.loads(
                        self._advice_path.read_text(encoding="utf-8"))
                    if not isinstance(existing, list):
                        existing = []
                except (OSError, json.JSONDecodeError):
                    existing = []
            # de-dup on (action, target)
            existing = [e for e in existing
                        if not (e.get("action") == entry["action"]
                                and e.get("target") == entry["target"])]
            existing.append(entry)
            tmp = self._advice_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(existing[-200:], indent=2, default=str),
                           encoding="utf-8")
            os.replace(tmp, self._advice_path)
        except Exception as exc:  # noqa: BLE001
            log.debug("persist_advice failed (non-fatal): %s", exc)

    # populated below the class body
    _DISPATCH: dict = {}


def _jsonable(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(v)).decode("ascii")
    if isinstance(v, Path):
        return str(v)
    return v


# Dispatch table — built after the class so the bound methods exist.
Applier._DISPATCH = {
    Action.SET_ENV_VAR: Applier._apply_set_env_var,
    Action.REGISTRY_TWEAK: Applier._apply_registry_tweak,
    Action.PROCESS_PRIORITY: Applier._apply_process_priority,
    Action.CPU_AFFINITY: Applier._apply_cpu_affinity,
    Action.VRAM_ALLOCATION: Applier._apply_vram_allocation,
    Action.ADJUST_QUALITY: Applier._apply_adjust_quality,
    Action.SET_LAUNCH_FLAG: Applier._apply_set_launch_flag,
    Action.NOTIFY: Applier._apply_notify,
    Action.COMPILE_TRT_ENGINE: Applier._apply_compile_trt_engine,
    Action.ENABLE_BACKEND: Applier._apply_enable_backend,
    Action.INSTALL_CUSTOM_NODE: Applier._apply_install_custom_node,
    Action.INSTALL_PLUGIN: Applier._apply_install_plugin,
    Action.WIRE_BACKEND_PATCH: Applier._apply_wire_backend_patch,
    Action.EVICT_ENGINE_CACHE: Applier._apply_evict_engine_cache,
}
