"""
Core contracts shared by the orchestrator, executor, safety layer and agents.

The whole system is built on ONE idea: every optimization is expressed as a list
of atomic, reversible `Change` objects. Agents never touch the machine directly —
they only *propose* Changes. The Executor is the only thing that mutates state, and
because each Change carries how to read/apply/revert itself, backup and rollback
come for free.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ChangeType(enum.Enum):
    ENV_VAR = "env_var"                 # [Environment]::SetEnvironmentVariable (User/Machine)
    REGISTRY = "registry"               # winreg typed value
    POWER_PLAN = "power_plan"           # powercfg active scheme / setting
    DEFENDER_EXCLUSION = "defender_exclusion"  # Add/Remove-MpPreference -ExclusionPath
    FILE_LINE = "file_line"             # upsert key=value in a config file (houdini.env, Engine.ini)
    COMMAND = "command"                 # generic apply cmd + explicit undo cmd
    GUIDED = "guided"                   # cannot be automated safely -> show instructions only


class Scope(enum.Enum):
    USER = "user"
    MACHINE = "machine"


class WorkloadMode(enum.Enum):
    IDLE = "idle"
    HOUDINI_HEAVY = "houdini_heavy"
    UE5_HEAVY = "ue5_heavy"
    COMFYUI_HEAVY = "comfyui_heavy"
    PARALLEL = "parallel"


class Risk(enum.Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Change:
    """One atomic, reversible mutation. `current_value` is filled by the Executor
    at plan time (by reading the live value) so the restore point can undo it."""
    id: str
    type: ChangeType
    description: str
    target: str                          # env name / reg path / file path / plan GUID / exclusion path
    target_value: Any = None             # desired value
    # registry specifics
    reg_value_name: Optional[str] = None
    reg_type: Optional[str] = None       # "REG_DWORD" | "REG_SZ" | "REG_QWORD" ...
    # env specifics
    scope: Scope = Scope.USER
    requires_admin: bool = False
    # file_line specifics
    file_key: Optional[str] = None       # e.g. "HOUDINI_MAXTHREADS" or "r.Streaming.PoolSize"
    file_sep: str = "="                  # "=" for env files, "=" for ini cvars too
    # command specifics
    snapshot_cmd: Optional[str] = None   # prints prior state -> captured into current_value
    apply_cmd: Optional[str] = None      # may reference {prior} = snapshot output
    undo_cmd: Optional[str] = None       # may reference {prior} to restore captured state
    verify_cmd: Optional[str] = None     # prints value used to confirm the change took
    # bookkeeping (filled at runtime)
    current_value: Any = None            # snapshot before apply; None => value was absent
    existed_before: bool = True
    risk: Risk = Risk.LOW
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """A named optimization an agent recommends. May carry several Changes that
    must be applied together (e.g. an env var + its NVIDIA per-app profile)."""
    key: str
    title: str
    domain: str
    why: str
    impact: str
    confidence: float                    # 0..1; orchestrator filters below threshold
    changes: List[Change]
    applies_when: List[WorkloadMode] = field(default_factory=lambda: list(WorkloadMode))
    requires_admin: bool = False
    note_2026: str = ""

    def gated(self) -> bool:
        return self.requires_admin or any(c.requires_admin for c in self.changes)


@dataclass
class Context:
    """Live machine state produced by probe.py and handed to every agent."""
    is_admin: bool = False
    workload: WorkloadMode = WorkloadMode.IDLE
    running: Dict[str, bool] = field(default_factory=dict)   # {"houdini": bool, "unreal": bool, "comfyui": bool}
    gpu_util: int = 0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 24.0
    temp_c: int = 0
    power_w: float = 0.0
    pcie_gen: int = 0
    throttle_reasons: List[str] = field(default_factory=list)
    physical_cores: int = 24
    logical_cores: int = 48
    ccds: int = 4
    fastest_nvme: Optional[str] = None
    python_exe: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def change_key(c: Change) -> str:
    """Stable identity for a Change target — used to snapshot each original value
    exactly once, even when the watchdog re-applies it every cycle."""
    if c.type is ChangeType.ENV_VAR:
        return f"env|{c.scope.value}|{c.target}"
    if c.type is ChangeType.REGISTRY:
        return f"reg|{c.target}|{c.reg_value_name}"
    if c.type is ChangeType.DEFENDER_EXCLUSION:
        return f"def|{c.target}"
    if c.type is ChangeType.FILE_LINE:
        return f"file|{c.target}|{c.extra.get('section', '')}|{c.file_key}"
    return f"cmd|{c.id}"


class Agent:
    """Base class for a domain optimization agent. Subclasses live in optimizer/agents/
    and are auto-discovered by the orchestrator. Keep propose() pure — no side effects."""
    name: str = "agent"
    domain: str = "generic"

    def propose(self, ctx: Context) -> List[Proposal]:  # pragma: no cover - interface
        raise NotImplementedError
