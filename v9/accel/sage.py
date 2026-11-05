"""V9 — SageAttention v2 acceleration backend.

Wraps the `sageattention` PyPI package (v2.x). Unlike TRT, SageAttention is a
*runtime* swap, not a compile-once-then-load artifact. compile() is a no-op
that returns a sentinel BuildArtifactDescriptor; the real work happens in
load().runner — a context manager that monkeypatches
torch.nn.functional.scaled_dot_product_attention only for the duration of its
`with` block (no global mutation across the orchestrator).

Workload-agnostic: imports nothing from v9.advisors / v9.build / v9.context.
It recognises attention inputs only by tensor rank / head structure.

Degrade-safe: both sageattention and torch are probed in __init__; either
missing -> enabled=False. compile()/load() raise a clear install hint.
"""
from __future__ import annotations

import logging
from contextlib import AbstractContextManager
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

logger = logging.getLogger("v9.accel.sage")

_INSTALL_HINT = "Install with `pip install sageattention`."


def _try_import_sage():
    """Returns (module, version, entrypoint_fn) or (None, reason, None)."""
    try:
        import sageattention  # type: ignore
    except ImportError as exc:
        return None, f"sageattention import failed: {exc}", None
    except Exception as exc:  # pragma: no cover - broken install
        return None, f"sageattention import error: {exc}", None

    fn = None
    for candidate in ("sageattn", "sage_attn", "sageattn_qk_int8_pv_fp16",
                      "sageattn_varlen"):
        if hasattr(sageattention, candidate):
            fn = getattr(sageattention, candidate)
            break
    if fn is None:
        return None, "sageattention present but no sageattn entrypoint found", None

    version = getattr(sageattention, "__version__", None)
    if not version or version == "unknown":
        # sage 1.0.x doesn't expose __version__; ask the installed package
        # metadata so doctor never reports a fake/"unknown" version (the
        # honest-capability invariant). pip knows the real version even when
        # the module doesn't advertise it.
        try:
            import importlib.metadata as _im
            version = _im.version("sageattention")
        except Exception:
            version = "unknown"
    return sageattention, version, fn


def _try_import_torch():
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except ImportError as exc:
        return None, None, f"torch import failed: {exc}"
    except Exception as exc:  # pragma: no cover - broken install
        return None, None, f"torch import error: {exc}"
    return torch, F, None


class SageSDPAPatch(AbstractContextManager):
    """Context manager that monkeypatches
    ``torch.nn.functional.scaled_dot_product_attention`` to call into
    sageattention for the duration of the ``with`` block, restoring the
    original on exit.

    Each instance snapshots the *current* SDPA on enter, so nested patches
    from separate instances stack and unwind correctly. No global state lives
    outside the saved attribute.
    """

    def __init__(self, sage_fn: Callable[..., Any], torch_module: Any,
                 functional_module: Any) -> None:
        self._sage_fn = sage_fn
        self._torch = torch_module
        self._F = functional_module
        self._saved: Optional[Callable[..., Any]] = None
        self._patched: bool = False

    def _sdpa_replacement(self, query, key, value, attn_mask=None,
                          dropout_p: float = 0.0, is_causal: bool = False,
                          scale=None, enable_gqa: bool = False):
        # sageattention's signature varies by version; forward the canonical
        # (q, k, v, is_causal, scale) and fall back to positional.
        try:
            return self._sage_fn(query, key, value, is_causal=is_causal,
                                 sm_scale=scale)
        except TypeError:
            return self._sage_fn(query, key, value, is_causal, scale)

    def __enter__(self) -> "SageSDPAPatch":
        if self._patched:
            raise RuntimeError(
                "SageSDPAPatch is not re-enterable on the same instance; "
                "create a new one."
            )
        self._saved = self._F.scaled_dot_product_attention
        self._F.scaled_dot_product_attention = self._sdpa_replacement
        self._patched = True
        logger.debug("SageSDPAPatch: SDPA monkeypatched.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._patched and self._saved is not None:
            self._F.scaled_dot_product_attention = self._saved
            self._patched = False
            logger.debug("SageSDPAPatch: SDPA restored.")
        return None

    @property
    def is_active(self) -> bool:
        return self._patched


class SageAttentionV2Backend(AccelerationBackend):
    """Wraps sageattention v2.x. compile() is a no-op; all acceleration is
    expressed at load() time as a context-managed SDPA monkeypatch."""

    name: str = "sage_attention_v2"
    version: str = "2.0.0"

    def __init__(self) -> None:
        super().__init__()
        sage_mod, ver_or_reason, fn = _try_import_sage()
        torch_mod, F_mod, torch_err = _try_import_torch()

        self._sage = None
        self._sage_fn = None
        self._sage_version = "unavailable"
        self._torch = None
        self._F = None

        if sage_mod is None:
            self._unavailable_reason = ver_or_reason
            self.enabled = False
            logger.info("SageAttentionV2Backend disabled (%s). %s",
                        ver_or_reason, _INSTALL_HINT)
            return

        if torch_mod is None:
            self._sage = sage_mod
            self._sage_fn = fn
            self._sage_version = ver_or_reason
            self._unavailable_reason = torch_err
            self.enabled = False
            logger.info("SageAttentionV2Backend disabled (no torch: %s).",
                        torch_err)
            return

        major = (ver_or_reason or "").split(".", 1)[0]
        if major and major not in {"2", "unknown"}:
            logger.warning(
                "SageAttentionV2Backend: sageattention version %s != 2.x; "
                "running anyway but this backend targets v2.", ver_or_reason,
            )

        self._sage = sage_mod
        self._sage_fn = fn
        self._sage_version = ver_or_reason
        self._torch = torch_mod
        self._F = F_mod
        self._unavailable_reason = None
        self.enabled = True
        # Honest: report the REAL installed sage version, never a hardcoded
        # "2.0.0". When a v1.x lib is running under this v2-targeting backend,
        # say so in the version string so `doctor` surfaces the upgrade path
        # (v2 adds FP8/INT8 attention; v1 is INT8 QK only).
        is_v2 = major == "2"
        self.version = (ver_or_reason if is_v2
                        else f"{ver_or_reason} (v1; upgrade for v2 FP8 attn)")
        logger.info("SageAttentionV2Backend ready (sageattention=%s, is_v2=%s).",
                    ver_or_reason, is_v2)

    # ----------------------------------------------------------- ABC impl

    def applies_to(self, model_kind: str, shape: Tuple[int, ...],
                   precision: str) -> bool:
        if not self.enabled:
            return False
        kind = (model_kind or "").lower()
        if kind not in {"attention", "sdpa", "transformer_block",
                        "self_attention", "cross_attention", "mha"}:
            return False
        # Attention tensors are typically (B, H, S, D) or (B, S, H, D).
        if len(shape) < 3:
            return False
        prec = (precision or "").lower()
        if prec not in {"fp16", "bf16", "fp8", "int8", "mixed", ""}:
            return False
        return True

    def estimated_compile_seconds(self, model_kind: str,
                                  shape: Tuple[int, ...]) -> float:
        return 0.0  # no-op compile (runtime swap)

    def estimated_speedup(self, model_kind: str,
                          shape: Tuple[int, ...]) -> float:
        # SageAttention v2 reports ~1.6-2.7x over flash-attn-2 on Ada; scale
        # the prior with sequence length (the -2 axis of (.., S, D)).
        seq = max(int(shape[-2]), 1) if len(shape) >= 3 else 1
        if seq >= 4096:
            return 2.5
        if seq >= 1024:
            return 2.0
        return 1.6

    def estimated_quality_delta(self) -> float:
        # Small INT8/FP8 noise on the QK matmul; near-imperceptible for
        # diffusion / attention blocks.
        return 0.08

    def compile(self, source: BackendSource, shape_range: ShapeRange,
                precision: str, cancel_token: CancelToken,
                progress_cb: Optional[Callable[[float], None]] = None,
                ) -> BuildArtifactDescriptor:
        # No build step. Return a sentinel descriptor so the cache layer can
        # record the backend choice without persisting a real plan.
        cancel_token.raise_if_cancelled()
        if progress_cb:
            progress_cb(1.0)
        return BuildArtifactDescriptor(
            backend_name=self.name,
            backend_version=self.version,
            artifact_bytes=b"",
            artifact_format="noop",
            precision=(precision or "fp16").lower(),
            shape_range=shape_range,
            compile_seconds=0.0,
            notes={
                "kind": "runtime_monkeypatch",
                "sage_version": self._sage_version,
            },
        )

    def load(self, artifact_path: Path) -> LoadedEngine:
        if not self.enabled:
            raise RuntimeError(
                f"SageAttentionV2Backend not available: "
                f"{self._unavailable_reason}. {_INSTALL_HINT}"
            )
        # artifact_path is unused (no-op compile); accepted for ABC conformance.
        runner = SageSDPAPatch(
            sage_fn=self._sage_fn,
            torch_module=self._torch,
            functional_module=self._F,
        )
        return LoadedEngine(
            backend_name=self.name,
            backend_version=self.version,
            runner=runner,
            metadata={
                "kind": "context_manager",
                "sage_version": self._sage_version,
                "patches": "torch.nn.functional.scaled_dot_product_attention",
            },
        )

    # ------------------------------------------------------------- helpers

    def run_patched(self, callable_fn: Callable[..., Any], *args, **kwargs):
        """Convenience SDPA-monkeypatch context runner: execute callable_fn
        with sage SDPA active and restore afterward. Raises if disabled."""
        if not self.enabled:
            raise RuntimeError(
                f"SageAttentionV2Backend not available: "
                f"{self._unavailable_reason}. {_INSTALL_HINT}"
            )
        with SageSDPAPatch(self._sage_fn, self._torch, self._F):
            return callable_fn(*args, **kwargs)


if __name__ == "__main__":
    be = SageAttentionV2Backend()
    print(f"name={be.name} version={be.version} enabled={be.enabled}")
    if not be.enabled:
        print(f"unavailable_reason: {be._unavailable_reason}")
