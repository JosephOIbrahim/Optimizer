"""
OptimizerV12 — one control plane over every actuator on this machine.

V12 does not reimplement the physics or the acceleration backends. V9 already has
a real, microbench-validated roofline (`v9/hwmodel.py`) and four real backends;
MachineOptimizer2026 already has a real reversible-change executor and a watchdog.
Both are KEPT. V12 is the layer neither of them has: the one that owns the
objective, the baseline, and the admission gate — and that refuses to let any
actuator move state without a verdict.

The defect V12 exists to fix, stated once:

    Both predecessors verify the wrong thing at the moment of action.
    MachineOptimizer2026 verifies that a change was APPLIED (read-back == target)
    and never that it was FASTER. V9 has a real Pareto gate and its actuator,
    apply.py, routes around it (apply_all is recs[:top_n]).

V12's actuator cannot be called without a Verdict. That is enforced by the
function signature, not by a policy check that a future edit can drop.
"""
from __future__ import annotations

__version__ = "12.0.0"

from .change import (
    Change,
    ChangeKind,
    EvidenceClass,
    Observability,
    Snapshot,
    Tier,
    change_key,
)
from .admission import Gate, GatedActuator, UnadmittedChange, Verdict

__all__ = [
    "__version__",
    "Change",
    "ChangeKind",
    "EvidenceClass",
    "Observability",
    "Snapshot",
    "Tier",
    "change_key",
    "Gate",
    "GatedActuator",
    "UnadmittedChange",
    "Verdict",
]
