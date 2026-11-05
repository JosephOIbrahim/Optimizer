"""V9 — BackendRegistry: register acceleration backends and report an HONEST
capability map of what is actually installed vs dormant. This is what
`doctor` prints.

Principle 4 (honest capability): every backend probed its SDK in __init__ and
set `enabled` accordingly. The registry never claims success when nothing is
installed; `capability_map()` reports {enabled, version, reason} per backend,
where `reason` explains *why* a dormant backend is dormant.

Degrade-safe: instantiating any backend can never crash the registry — a
backend whose __init__ raises is recorded as a dead entry with the exception
text as its reason, so doctor still prints a complete picture.

No workload knowledge, no hardware literals. The optional HardwareProfile is
used only to hand CUTLASS the detected sm_arch.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from v9.accel.backend import AccelerationBackend

logger = logging.getLogger("v9.accel.registry")


class BackendRegistry:
    """Holds the live backend instances and answers capability questions."""

    def __init__(self) -> None:
        self._backends: Dict[str, AccelerationBackend] = {}
        # name -> (reason text) for backends that failed to even construct
        self._dead: Dict[str, str] = {}

    # ----------------------------------------------------------- mutation

    def register(self, backend: AccelerationBackend) -> None:
        """Register a constructed backend instance."""
        self._backends[backend.name] = backend
        self._dead.pop(backend.name, None)

    def register_factory(self, name: str,
                         factory: Callable[[], AccelerationBackend]) -> None:
        """Construct via `factory` and register. If construction raises, record
        a dead entry instead of propagating (degrade, never crash)."""
        try:
            backend = factory()
        except Exception as exc:  # noqa: BLE001 - registry must not crash
            self._dead[name] = f"construction failed: {exc}"
            logger.warning("Backend %r failed to construct: %s", name, exc)
            return
        self.register(backend)

    # ----------------------------------------------------------- queries

    def get(self, name: str) -> Optional[AccelerationBackend]:
        return self._backends.get(name)

    def all_backends(self) -> List[AccelerationBackend]:
        return list(self._backends.values())

    def enabled_backends(self) -> List[AccelerationBackend]:
        return [b for b in self._backends.values() if b.enabled]

    def candidates_for(self, model_kind: str, shape, precision
                       ) -> List[AccelerationBackend]:
        """Enabled backends whose applies_to() accepts the triple. Each
        applies_to() call is shielded so one misbehaving backend can't break
        candidate selection."""
        out: List[AccelerationBackend] = []
        for b in self._backends.values():
            if not b.enabled:
                continue
            try:
                if b.applies_to(model_kind, shape, precision):
                    out.append(b)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s.applies_to raised: %s", b.name, exc)
        return out

    def capability_map(self) -> Dict[str, Dict[str, Any]]:
        """Honest report: {name: {enabled, version, reason}}.

        `reason` is the install/disable explanation for a dormant backend, or
        None when enabled. Dead (failed-to-construct) backends appear with
        enabled=False and the construction error as the reason.
        """
        report: Dict[str, Dict[str, Any]] = {}
        for name, b in self._backends.items():
            reason = None
            if not b.enabled:
                reason = getattr(b, "_unavailable_reason", None) or "disabled"
            report[name] = {
                "enabled": bool(b.enabled),
                "version": getattr(b, "version", "unknown"),
                "reason": reason,
            }
        for name, reason in self._dead.items():
            report[name] = {
                "enabled": False,
                "version": "unavailable",
                "reason": reason,
            }
        return report


def build_default_registry(hardware: Optional[Any] = None,
                           config: Optional[Any] = None) -> BackendRegistry:
    """Construct the standard V9 backend set. `hardware` is an optional
    HardwareProfile; if given, its gpu.sm_arch is handed to CUTLASS. `config`
    is the optional v9.config.Config (or a plain dict); its
    `acceleration.cutlass.enabled` flag opts the CUTLASS backend into the
    recipe. Every backend is constructed through register_factory so a broken
    SDK can never abort registry construction."""
    from v9.accel.tensorrt import TensorRTBackend
    from v9.accel.sage import SageAttentionV2Backend
    from v9.accel.cutlass import CutlassFP8GemmBackend
    from v9.accel.torchcompile import TorchCompileBackend

    sm_arch: Optional[int] = None
    if hardware is not None:
        try:
            sm_arch = getattr(getattr(hardware, "gpu", None), "sm_arch", None)
        except Exception:  # pragma: no cover - defensive
            sm_arch = None

    cutlass_enabled = _read_cutlass_enabled(config)
    tc_mode = _read_torch_compile_mode(config)

    reg = BackendRegistry()
    reg.register_factory("tensorrt", TensorRTBackend)
    reg.register_factory("sage_attention_v2", SageAttentionV2Backend)
    reg.register_factory("torch_compile",
                         lambda: TorchCompileBackend(mode=tc_mode))
    reg.register_factory(
        "cutlass_fp8_gemm",
        lambda: CutlassFP8GemmBackend(sm_arch=sm_arch, enabled=cutlass_enabled),
    )
    return reg


def _read_cutlass_enabled(config: Optional[Any]) -> bool:
    """Pull acceleration.cutlass.enabled out of a Config/dict, default False.
    Never raises — a malformed config just leaves CUTLASS opted out."""
    if config is None:
        return False
    try:
        accel = config.get("acceleration", {}) if hasattr(config, "get") else {}
        cutlass = (accel or {}).get("cutlass", {}) if isinstance(accel, dict) else {}
        return bool(cutlass.get("enabled", False))
    except Exception:  # noqa: BLE001
        return False


def _read_torch_compile_mode(config: Optional[Any]) -> str:
    """Pull acceleration.torch_compile.mode out of a Config/dict, default
    'max-autotune'. Never raises — a malformed config just uses the default.
    torch.compile is zero-install (uses the live torch), so unlike CUTLASS it
    has no enabled flag; the backend self-disables only if torch is absent."""
    default = "max-autotune"
    if config is None:
        return default
    try:
        accel = config.get("acceleration", {}) if hasattr(config, "get") else {}
        tc = (accel or {}).get("torch_compile", {}) if isinstance(accel, dict) else {}
        return str(tc.get("mode", default)) or default
    except Exception:  # noqa: BLE001
        return default


if __name__ == "__main__":
    reg = build_default_registry()
    print("=== V9 acceleration capability map ===")
    for name, info in reg.capability_map().items():
        state = "ENABLED" if info["enabled"] else "dormant"
        print(f"  {name:20s} {state:8s} v={info['version']}"
              + (f"  ({info['reason']})" if info["reason"] else ""))
