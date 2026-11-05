"""V10 — torch.compile acceleration backend (zero-install; uses the live torch).

The cheapest real win on the existing stack: `torch.compile` with Inductor
(``mode="max-autotune"`` + CUDA graphs) fuses the UNet/DiT graph and autotunes
kernels for the detected GPU. No SDK to install, no wheel ABI to match — if
torch imports, this backend is live.

Like SageAttention (and unlike TensorRT), this is a **runtime-wrap** backend, not
a compile-once-to-a-portable-plan backend. torch.compile is lazy: the actual
kernel compilation happens on the first forward pass and is cached by torch's
own Inductor cache. So here:

  * ``compile()`` is light — it records the compile decision (mode/options) as a
    small artifact and redirects Inductor's on-disk cache OFF OneDrive (principle
    3: OneDrive churns the thousands of small files Inductor writes). The heavy,
    lazy work is reported honestly via ``estimated_compile_seconds``.
  * ``load()`` returns a runner whose ``.wrap(module)`` returns a compiled module.

Workload-agnostic: imports nothing from v9.advisors / v9.build / v9.context. It
recognises a module by ``model_kind`` only. Degrade-safe: torch is probed in
__init__; missing -> enabled=False and compile()/load() raise a clear hint.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from v9.accel.backend import (
    AccelerationBackend,
    BackendSource,
    BuildArtifactDescriptor,
    CancelToken,
    LoadedEngine,
    ShapeRange,
)

logger = logging.getLogger("v9.accel.torchcompile")

_INSTALL_HINT = "torch is required (it is the runtime); this should never fire."

# model_kinds this backend will wrap (whole-module graphs, not bare attention —
# SageAttention owns the attention swap).
_APPLICABLE_KINDS = {
    "unet", "dit", "mmdit", "transformer", "diffusion", "diffusion_model", "vae",
}
_APPLICABLE_PREC = {"fp16", "bf16", "fp32", "mixed", ""}

# compile-mode priors (Ada diffusion). Measured numbers replace these via the
# benchmark/quality ledger; these are only the scheduling estimate.
_MODE_SPEEDUP = {"max-autotune": 1.35, "reduce-overhead": 1.25, "default": 1.15}
_MODE_COMPILE_S = {"max-autotune": 90.0, "reduce-overhead": 25.0, "default": 15.0}


def _try_import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        return None, f"torch import failed: {exc}"
    except Exception as exc:  # pragma: no cover - broken install
        return None, f"torch import error: {exc}"
    return torch, None


def _apply_windows_inductor_fixes() -> None:
    """Windows-only torch.compile/Inductor workarounds, learned on this rig.

    The static CUDA launcher passes 64-bit kernel pointers through a C ``long``,
    which is 32-bit on Windows (LLP64) -> ``OverflowError: Python int too large
    to convert to C long`` during max-autotune benchmarking. Disable it so
    Inductor falls back to the triton launcher (correct, marginally slower
    dispatch). Never raises; an absent knob just means another torch handled it.
    """
    if os.name != "nt":
        return
    os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
    try:
        import torch._inductor.config as _ind  # type: ignore
        if hasattr(_ind, "use_static_cuda_launcher"):
            _ind.use_static_cuda_launcher = False
    except Exception:  # noqa: BLE001 - best-effort; backend stays degrade-safe
        pass


def _redirect_inductor_cache_off_onedrive() -> Optional[str]:
    """Point TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR at the off-OneDrive data
    root (only if the user hasn't set them). Returns the dir used, or None if it
    couldn't be resolved. Never raises."""
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        return os.environ["TORCHINDUCTOR_CACHE_DIR"]
    try:
        from v9 import paths  # infra import (stdlib-only module); no workload coupling
        cache = paths.DATA_ROOT / "torchinductor_cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache))
        os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
        return str(cache)
    except Exception:  # noqa: BLE001 - cache redirect is best-effort
        return None


class TorchCompileRunner:
    """Returned inside LoadedEngine.runner. ``wrap(module)`` returns a compiled
    module; calling the runner directly is an alias for wrap()."""

    def __init__(self, torch_module: Any, mode: str, dynamic: bool,
                 fullgraph: bool) -> None:
        self._torch = torch_module
        self.mode = mode
        self.dynamic = dynamic
        self.fullgraph = fullgraph

    def wrap(self, module: Any) -> Any:
        return self._torch.compile(
            module, mode=self.mode, dynamic=self.dynamic,
            fullgraph=self.fullgraph,
        )

    def __call__(self, module: Any) -> Any:
        return self.wrap(module)


class TorchCompileBackend(AccelerationBackend):
    """torch.compile / Inductor backend. compile() records the decision; the
    real (lazy) kernel build happens at runtime and is cached by Inductor."""

    name: str = "torch_compile"
    version: str = "1.0.0"

    def __init__(self, mode: str = "max-autotune", dynamic: bool = False,
                 fullgraph: bool = False) -> None:
        super().__init__()
        self.mode = mode if mode in _MODE_SPEEDUP else "max-autotune"
        self.dynamic = bool(dynamic)
        self.fullgraph = bool(fullgraph)
        self._torch = None
        self._cache_dir: Optional[str] = None

        torch_mod, reason = _try_import_torch()
        if torch_mod is None:
            self._unavailable_reason = reason
            self.enabled = False
            logger.info("TorchCompileBackend disabled (%s).", reason)
            return

        # torch.compile needs a working dynamo/inductor; guard the attribute.
        if not hasattr(torch_mod, "compile"):
            self._unavailable_reason = "torch has no .compile (torch < 2.0)"
            self.enabled = False
            logger.info("TorchCompileBackend disabled (%s).",
                        self._unavailable_reason)
            return

        self._torch = torch_mod
        _apply_windows_inductor_fixes()
        self._cache_dir = _redirect_inductor_cache_off_onedrive()
        self._unavailable_reason = None
        self.enabled = True
        self.version = f"1.0.0+torch{torch_mod.__version__}"
        logger.info("TorchCompileBackend ready (torch=%s, mode=%s, cache=%s).",
                    torch_mod.__version__, self.mode, self._cache_dir)

    # ----------------------------------------------------------- ABC impl

    def applies_to(self, model_kind: str, shape: Tuple[int, ...],
                   precision: str) -> bool:
        if not self.enabled:
            return False
        if (model_kind or "").lower() not in _APPLICABLE_KINDS:
            return False
        return (precision or "").lower() in _APPLICABLE_PREC

    def estimated_compile_seconds(self, model_kind: str,
                                  shape: Tuple[int, ...]) -> float:
        # The lazy first-forward cost. max-autotune autotunes kernels -> slow
        # first call, fast thereafter. Reported so the scheduler idle-gates it.
        return _MODE_COMPILE_S.get(self.mode, 90.0)

    def estimated_speedup(self, model_kind: str,
                          shape: Tuple[int, ...]) -> float:
        return _MODE_SPEEDUP.get(self.mode, 1.3)

    def estimated_quality_delta(self) -> float:
        # torch.compile preserves semantics (fusion + autotune of the same math);
        # numerically near-identical. Treated as lossless.
        return 0.0

    def compile(self, source: BackendSource, shape_range: ShapeRange,
                precision: str, cancel_token: CancelToken,
                progress_cb: Optional[Callable[[float], None]] = None,
                ) -> BuildArtifactDescriptor:
        """Light: record the compile decision + ensure the off-OneDrive Inductor
        cache exists. The actual kernel compile is lazy (first forward) and lives
        in torch's Inductor cache, not in our artifact bytes."""
        if not self.enabled:
            raise RuntimeError(
                f"TorchCompileBackend not available: {self._unavailable_reason}. "
                f"{_INSTALL_HINT}")
        cancel_token.raise_if_cancelled()
        blob = json.dumps({
            "mode": self.mode, "dynamic": self.dynamic,
            "fullgraph": self.fullgraph, "cache_dir": self._cache_dir,
        }).encode("utf-8")
        if progress_cb:
            progress_cb(1.0)
        return BuildArtifactDescriptor(
            backend_name=self.name,
            backend_version=self.version,
            artifact_bytes=blob,
            artifact_format="torch_compiled",
            precision=(precision or "fp16").lower(),
            shape_range=shape_range,
            compile_seconds=0.0,   # decision only; lazy build is at runtime
            notes={"kind": "runtime_torch_compile", "mode": self.mode,
                   "lazy_compile_s_estimate":
                       self.estimated_compile_seconds("", shape_range.opt_shape)},
        )

    def load(self, artifact_path: Path) -> LoadedEngine:
        if not self.enabled:
            raise RuntimeError(
                f"TorchCompileBackend not available: {self._unavailable_reason}. "
                f"{_INSTALL_HINT}")
        # Reconstruct mode/options from the artifact if present; else instance.
        mode, dynamic, fullgraph = self.mode, self.dynamic, self.fullgraph
        try:
            if artifact_path and Path(artifact_path).exists():
                cfg = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
                mode = cfg.get("mode", mode)
                dynamic = bool(cfg.get("dynamic", dynamic))
                fullgraph = bool(cfg.get("fullgraph", fullgraph))
        except Exception:  # noqa: BLE001 - artifact is advisory; fall back clean
            pass
        runner = TorchCompileRunner(self._torch, mode, dynamic, fullgraph)
        return LoadedEngine(
            backend_name=self.name,
            backend_version=self.version,
            runner=runner,
            metadata={"kind": "runtime_wrap", "mode": mode,
                      "cache_dir": self._cache_dir,
                      "usage": "runner.wrap(module) -> compiled module"},
        )


if __name__ == "__main__":
    be = TorchCompileBackend()
    print(f"name={be.name} version={be.version} enabled={be.enabled} "
          f"mode={be.mode}")
    if not be.enabled:
        print(f"unavailable_reason: {be._unavailable_reason}")
