"""Terse factory helpers so each agent reads as a list of optimizations, not boilerplate.
Agents build `Change` objects with these and wrap them in `prop(...)`."""
from __future__ import annotations

from typing import List, Optional

from .base import Change, ChangeType, Proposal, Risk, Scope, WorkloadMode


def env(id: str, name: str, value, *, scope: Scope = Scope.USER, desc: str = None,
        risk: Risk = Risk.LOW, admin: bool = False) -> Change:
    return Change(id=id, type=ChangeType.ENV_VAR, description=desc or f"env {name}={value}",
                  target=name, target_value=str(value), scope=scope,
                  requires_admin=admin, risk=risk)


def reg(id: str, path: str, name: str, value, *, vtype: str = "REG_DWORD",
        desc: str = None, risk: Risk = Risk.MEDIUM, admin: bool = True) -> Change:
    return Change(id=id, type=ChangeType.REGISTRY, description=desc or f"{name}={value}",
                  target=path, reg_value_name=name, reg_type=vtype, target_value=value,
                  requires_admin=admin, risk=risk)


def defender(id: str, path: str, *, desc: str = None) -> Change:
    return Change(id=id, type=ChangeType.DEFENDER_EXCLUSION,
                  description=desc or f"Defender exclusion: {path}",
                  target=path, target_value=True, requires_admin=True, risk=Risk.LOW)


def file_line(id: str, path: str, key: str, value, *, section: Optional[str] = None,
              sep: str = "=", desc: str = None, risk: Risk = Risk.LOW) -> Change:
    return Change(id=id, type=ChangeType.FILE_LINE,
                  description=desc or f"{key}{sep}{value}", target=path,
                  file_key=key, target_value=str(value), file_sep=sep,
                  extra={"section": section} if section else {}, risk=risk)


def command(id: str, desc: str, apply_cmd: str, *, undo_cmd: str = None,
            snapshot_cmd: str = None, verify_cmd: str = None, admin: bool = True,
            risk: Risk = Risk.MEDIUM, power_plan: bool = False) -> Change:
    return Change(id=id, type=ChangeType.POWER_PLAN if power_plan else ChangeType.COMMAND,
                  description=desc, target=id, apply_cmd=apply_cmd, undo_cmd=undo_cmd,
                  snapshot_cmd=snapshot_cmd, verify_cmd=verify_cmd,
                  requires_admin=admin, risk=risk)


def guided(id: str, desc: str, instructions: str, *, risk: Risk = Risk.NONE) -> Change:
    return Change(id=id, type=ChangeType.GUIDED, description=desc, target=id,
                  extra={"instructions": instructions}, risk=risk)


def prop(key: str, title: str, domain: str, why: str, impact: str, confidence: float,
         changes: List[Change], *, applies_when: List[WorkloadMode] = None,
         admin: bool = False, note: str = "") -> Proposal:
    return Proposal(key=key, title=title, domain=domain, why=why, impact=impact,
                    confidence=confidence, changes=changes,
                    applies_when=applies_when or list(WorkloadMode),
                    requires_admin=admin, note_2026=note)
