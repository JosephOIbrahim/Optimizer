"""Crucible C2.3 — Pareto admission rule.

Per `briefs/crucible_verification.md` Section C2.3, a backend is admitted iff:

    speedup_vs_reference >= 1.30
    AND lpips_delta       <= 0.04
    AND clip_delta        >= -0.02

This file is the executable specification for that rule. The production
quality-ledger module that actually evaluates admission is owned by Crucible-2
(see `_handoff_forge2.md` permissive-fallback note: ComfyUISpecialist already
calls `quality_ledger.fp8_admitted(...)` and treats `None` as not-admitted).

If the ledger module is not yet implemented, the test file remains as the
spec but skips at runtime. The test scope -- the four boundary cases -- is
fixed and normative; do not weaken it.
"""
from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Resolve the production admission function -- skip cleanly if absent.
# ---------------------------------------------------------------------------

def _resolve_admit_fn():
    """Try a few likely module paths for the production admission function.
    Returns a callable that takes (speedup, lpips_delta, clip_delta) and
    returns 0/1 or False/True; or None if the production code is absent."""
    candidates = [
        ("core.quality_ledger", "pareto_admit"),
        ("core.quality_ledger", "admit"),
        ("core.quality_ledger", "is_admitted"),
        ("core.experts.quality_ledger", "pareto_admit"),
    ]
    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return fn
    return None


_admit = _resolve_admit_fn()
if _admit is None:
    pytest.skip(
        "Pareto admission logic not yet implemented; Phase B "
        "(Crucible-2 owns `quality_ledger`).",
        allow_module_level=True,
    )


def _to_bool(x) -> bool:
    """Normalize 0/1, False/True, etc. to a Python bool."""
    return bool(int(x)) if not isinstance(x, bool) else x


# ---------------------------------------------------------------------------
# Boundary cases (normative -- do not weaken)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "speedup, lpips_delta, clip_delta, expected_admitted, label",
    [
        (1.4,  0.02, -0.01, 1, "all-pass"),
        (1.2,  0.02, -0.01, 0, "speedup-below-threshold"),
        (1.4,  0.06, -0.01, 0, "lpips-above-threshold"),
        (1.4,  0.02, -0.05, 0, "clip-below-threshold"),
    ],
)
def test_pareto_admission(
    speedup: float,
    lpips_delta: float,
    clip_delta: float,
    expected_admitted: int,
    label: str,
) -> None:
    """C2.3 admission must be the AND of all three thresholds.

    speedup>=1.30 AND lpips_delta<=0.04 AND clip_delta>=-0.02
    """
    got = _to_bool(_admit(speedup, lpips_delta, clip_delta))
    assert got is bool(expected_admitted), (
        f"[{label}] speedup={speedup} lpips={lpips_delta} clip={clip_delta}: "
        f"expected admitted={expected_admitted}, got {got}"
    )
