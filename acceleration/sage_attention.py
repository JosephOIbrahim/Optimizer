"""
SageAttention v2 acceleration backend for Optimizer_V8.

Wraps the `sageattention` PyPI package (v2.x). Unlike TRT, SageAttention is
a *runtime* swap, not a compile-once-then-load artifact. compile() is a
no-op that returns a sentinel BuildArtifactDescriptor; the real work
happens in load().runner, which is a context manager that monkeypatches
torch.nn.functional.scaled_dot_product_attention only for the duration of
its `with` block (no global mutation across the orchestrator).

Workload-agnostic: this file imports nothing from `core/experts/`. It only
recognises attention-shape inputs by looking at rank and head structure.

Public API:
    SageAttentionV2Backend()
        .applies_to(model_kind, shape, precision) -> bool
        .estimated_compile_seconds(model_kind, shape) -> float
        .estimated_speedup(model_kind, shape) -> float
        .estimated_quality_delta() -> float
        .compile(...) -> BuildArtifactDescriptor   # no-op, compile_seconds=0
        .load(artifact_path) -> LoadedEngine       # runner = context manager

Use:
    be = SageAttentionV2Backend()
    if be.enabled:
        engine = be.load(Path("ignored.bin"))   # path unused
        with engine.runner:
            output = my_model(...)              # SDPA is sage-accelerated
        # SDPA restored on exit.
"""

from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from .backend import (
    AccelerationBackend,
    BackendSource,
    BuildArtifactDescriptor,
    CancelToken,
    LoadedEngine,
    ShapeRange,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _try_import_sage():
    try:
        import sageattention  # type: ignore
    except ImportError as exc:
        return None, f"sageattention import failed: {exc}", None

    fn = None
    for candidate in ("sageattn", "sage_attn", "sageattn_qk_int8_pv_fp16",
                      "sageattn_varlen"):
        if hasattr(sageattention, candidate):
            fn = getattr(sageattention, candidate)
            break

    if fn is None:
        return None, "sageattention package present but no sageattn entrypoint found", None

    version = getattr(sageattention, "__version__", "unknown")
    return sageattention, version, fn


def _try_import_torch():
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except ImportError as exc:
        return None, None, f"torch import failed: {exc}"
    return torch, F, None


# ---------------------------------------------------------------------------
# The context manager returned via LoadedEngine.runner
# ---------------------------------------------------------------------------

class SageSDPAPatch(AbstractContextManager):
    """
    Context manager that monkeypatches
    ``torch.nn.functional.scaled_dot_product_attention`` to call into
    sageattention for the duration of the ``with`` block, and restores the
    original implementation on exit.

    Re-entrancy: nested ``with`` blocks are stacked safely (each save/restore
    its own snapshot of the *current* SDPA, which may be the previous
    patch). No global state outside the stack.
    """

    def __init__(self, sage_fn: Callable[..., Any], torch_module: Any,
                 functional_module: Any) -> None:
        self._sage_fn = sage_fn
        self._torch = torch_module
        self._F = functional_module
        self._saved: Optional[Callable[..., Any]] = None
        self._patched: bool = False

    # The signature mirrors torch.nn.functional.scaled_dot_product_attention
    # so callers can swap transparently.
    def _sdpa_replacement(
        self,
        query, key, value,
        attn_mask=None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale=None,
        enable_gqa: bool = False,
    ):
        # Defer to sageattention. Its signature varies by package version;
        # we forward the canonical (q, k, v, is_causal, scale) and let the
        # callee accept-or-error on extras.
        try:
            return self._sage_fn(
                query, key, value,
                is_causal=is_causal,
                sm_scale=scale,
            )
        except TypeError:
            # Older / different sage variants accept positional args only.
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
        # Do not suppress exceptions.
        return None

    @property
    def is_active(self) -> bool:
        return self._patched


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------

class SageAttentionV2Backend(AccelerationBackend):
    """
    Wraps sageattention v2.x. compile() is a no-op; all acceleration is
    expressed at load() time as a context-managed SDPA monkeypatch.
    """

    name: str = "sage_attention_v2"
    version: str = "2.0.0"

    def __init__(self) -> None:
        sage_mod, ver_or_reason, fn = _try_import_sage()
        torch_mod, F_mod, torch_err = _try_import_torch()

        if sage_mod is None:
            self._sage = None
            self._sage_fn = None
            self._sage_version = "unavailable"
            self._unavailable_reason = ver_or_reason
            self.enabled = False
            logger.warning(
                "SageAttentionV2Backend disabled (%s).", ver_or_reason,
            )
            return

        if torch_mod is None:
            self._sage = sage_mod
            self._sage_fn = fn
            self._sage_version = ver_or_reason
            self._unavailable_reason = torch_err
            self.enabled = False
            logger.warning(
                "SageAttentionV2Backend disabled (no torch: %s).", torch_err,
            )
            return

        # Pin to v2.x.
        major = (ver_or_reason or "").split(".", 1)[0]
        if major and major != "2" and major != "unknown":
            logger.warning(
                "SageAttentionV2Backend: sageattention version %s != 2.x; "
                "running anyway but the brief pins to v2.",
                ver_or_reason,
            )

        self._sage = sage_mod
        self._sage_fn = fn
        self._sage_version = ver_or_reason
        self._torch = torch_mod
        self._F = F_mod
        self._unavailable_reason = None
        self.enabled = True
        self.version = f"2.0.0+sage{ver_or_reason}"
        logger.info(
            "SageAttentionV2Backend ready (sageattention=%s).",
            ver_or_reason,
        )

    # ----------------------------------------------------------- ABC impl

    def applies_to(
        self,
        model_kind: str,
        shape: Tuple[int, ...],
        precision: str,
    ) -> bool:
        if not self.enabled:
            return False
        kind = (model_kind or "").lower()
        if kind not in {"attention", "sdpa", "transformer_block",
                        "self_attention", "cross_attention", "mha"}:
            return False
        # Attention tensors are typically (B, H, S, D) or (B, S, H, D).
        # Reject obvious non-attention shapes.
        if len(shape) < 3:
            return False
        prec = (precision or "").lower()
        # Sage v2 supports fp16 / bf16 / fp8-int8 mixed paths.
        if prec not in {"fp16", "bf16", "fp8", "int8", "mixed", ""}:
            return False
        return True

    def estimated_compile_seconds(
        self,
        model_kind: str,
        shape: Tuple[int, ...],
    ) -> float:
        # No-op compile.
        return 0.0

    def estimated_speedup(
        self,
        model_kind: str,
        shape: Tuple[int, ...],
    ) -> float:
        # SageAttention v2 reports ~1.6-2.7x over flash-attn-2 on Ada.
        # Use a conservative midpoint as the prior.
        seq = 1
        if len(shape) >= 3:
            seq = max(int(shape[-2]), 1)
        if seq >= 4096:
            return 2.5
        if seq >= 1024:
            return 2.0
        return 1.6

    def estimated_quality_delta(self) -> float:
        # SageAttention v2 introduces small INT8/FP8 noise on the QK matmul;
        # empirically near-imperceptible for diffusion / attention blocks.
        return 0.08

    def compile(
        self,
        source: BackendSource,
        shape_range: ShapeRange,
        precision: str,
        cancel_token: CancelToken,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> BuildArtifactDescriptor:
        # No build step. Return a sentinel descriptor so the cache layer
        # can record the backend choice without writing a real plan.
        cancel_token.raise_if_cancelled()
        if progress_cb:
            progress_cb(1.0)

        return BuildArtifactDescriptor(
            backend_name=self.name,
            backend_version=self.version,
            artifact_bytes=b"",   # nothing to persist
            artifact_format="noop",
            precision=(precision or "fp16").lower(),
            shape_range=shape_range,
            compile_seconds=0.0,
            notes={
                "kind": "runtime_monkeypatch",
                "sage_version": self._sage_version
                if self.enabled else "unavailable",
            },
        )

    def load(self, artifact_path: Path) -> LoadedEngine:
        if not self.enabled:
            raise RuntimeError(
                f"SageAttentionV2Backend not available: "
                f"{self._unavailable_reason}. "
                "Install with `pip install sageattention`."
            )
        # artifact_path is unused for this backend (no-op compile). We
        # accept it for ABC conformance.
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


# ---------------------------------------------------------------------------
# Module CLI (sanity print only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    be = SageAttentionV2Backend()
    print("=" * 60)
    print("SageAttentionV2Backend")
    print("=" * 60)
    print(f"name        : {be.name}")
    print(f"version     : {be.version}")
    print(f"enabled     : {be.enabled}")
    print(f"sage_version: {be._sage_version}")
    if not be.enabled:
        print(f"unavailable : {be._unavailable_reason}")
    else:
        # Demonstrate context-manager toggle without invoking sage on real
        # tensors (which would need CUDA). We only verify SDPA gets
        # swapped + restored.
        engine = be.load(Path("ignored.bin"))
        import torch.nn.functional as F  # type: ignore
        before = F.scaled_dot_product_attention
        with engine.runner:
            inside = F.scaled_dot_product_attention
            print("inside `with`: SDPA patched ->", inside is not before)
        after = F.scaled_dot_product_attention
        print("after  `with`: SDPA restored ->", after is before)
    print("=" * 60)
