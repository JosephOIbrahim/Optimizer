"""
The Executor is the ONLY component that mutates machine state.

Every Change is snapshotted (current value read + stored) before it is applied, so
safety.py can write a restore point and `--restore` can undo everything. Honors a
dry_run flag: in dry-run nothing is written, but reads/snapshots still happen so the
user sees exactly what *would* change and from what.
"""
from __future__ import annotations

import subprocess
import winreg
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .base import Change, ChangeType, Scope

_ROOTS = {
    "HKLM": winreg.HKEY_LOCAL_MACHINE,
    "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
    "HKCU": winreg.HKEY_CURRENT_USER,
    "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
}
_REG_TYPES = {
    "REG_DWORD": winreg.REG_DWORD,
    "REG_QWORD": winreg.REG_QWORD,
    "REG_SZ": winreg.REG_SZ,
    "REG_EXPAND_SZ": winreg.REG_EXPAND_SZ,
}


@dataclass
class ApplyResult:
    ok: bool
    changed: bool
    detail: str = ""


def run_ps(script: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Run a PowerShell snippet. Returns (rc, stdout, stderr)."""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def run_cmd(cmd: str, timeout: int = 60) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _split_reg(path: str) -> Tuple[int, str]:
    root, _, sub = path.replace("/", "\\").partition("\\")
    return _ROOTS[root.upper()], sub


class Executor:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run

    # ---- read / snapshot -------------------------------------------------
    def snapshot(self, c: Change) -> None:
        """Read the live value into c.current_value / c.existed_before."""
        try:
            if c.type is ChangeType.ENV_VAR:
                scope = "User" if c.scope is Scope.USER else "Machine"
                rc, out, _ = run_ps(f"[Environment]::GetEnvironmentVariable('{c.target}','{scope}')")
                c.existed_before = out != ""
                c.current_value = out if out != "" else None
            elif c.type is ChangeType.REGISTRY:
                c.current_value, c.existed_before = self._reg_read(c)
            elif c.type is ChangeType.DEFENDER_EXCLUSION:
                rc, out, _ = run_ps(
                    f"[bool]((Get-MpPreference).ExclusionPath -contains '{c.target}')")
                c.existed_before = out.lower() == "true"
                c.current_value = c.existed_before
            elif c.type is ChangeType.FILE_LINE:
                c.current_value, c.existed_before = self._file_read(c)
                c.extra["file_existed"] = Path(c.target).exists()
            elif c.type in (ChangeType.COMMAND, ChangeType.POWER_PLAN):
                if c.snapshot_cmd:
                    rc, out, _ = run_ps(c.snapshot_cmd)
                    c.current_value = out
                    c.existed_before = out != ""
            elif c.type is ChangeType.GUIDED:
                c.existed_before = True
        except Exception as e:  # snapshot must never crash the run
            c.current_value = None
            c.extra["snapshot_error"] = str(e)

    # ---- apply -----------------------------------------------------------
    def apply(self, c: Change) -> ApplyResult:
        if c.current_value is None and c.existed_before is True and c.type is not ChangeType.GUIDED:
            self.snapshot(c)
        if c.type is ChangeType.GUIDED:
            return ApplyResult(True, False, "guided: manual step, not auto-applied")
        if self.dry_run:
            return ApplyResult(True, False, f"[dry-run] would set -> {c.target_value}")
        try:
            if c.type is ChangeType.ENV_VAR:
                return self._env_apply(c)
            if c.type is ChangeType.REGISTRY:
                return self._reg_apply(c)
            if c.type is ChangeType.DEFENDER_EXCLUSION:
                if c.existed_before:
                    return ApplyResult(True, False, "exclusion already present")
                rc, out, err = run_ps(f"Add-MpPreference -ExclusionPath '{c.target}'")
                return ApplyResult(rc == 0, rc == 0, err or "added")
            if c.type is ChangeType.FILE_LINE:
                return self._file_apply(c)
            if c.type in (ChangeType.COMMAND, ChangeType.POWER_PLAN):
                cmd = (c.apply_cmd or "").replace("{prior}", str(c.current_value or ""))
                rc, out, err = run_ps("$ErrorActionPreference='Stop';" + cmd)
                return ApplyResult(rc == 0, rc == 0, (out or err)[:200])
        except Exception as e:
            return ApplyResult(False, False, f"error: {e}")
        return ApplyResult(False, False, "unhandled change type")

    # ---- verify ----------------------------------------------------------
    def verify(self, c: Change) -> bool:
        if self.dry_run or c.type is ChangeType.GUIDED:
            return True
        try:
            if c.type is ChangeType.ENV_VAR:
                scope = "User" if c.scope is Scope.USER else "Machine"
                _, out, _ = run_ps(f"[Environment]::GetEnvironmentVariable('{c.target}','{scope}')")
                return out == str(c.target_value)
            if c.type is ChangeType.REGISTRY:
                val, existed = self._reg_read(c)
                return existed and str(val) == str(c.target_value)
            if c.type is ChangeType.DEFENDER_EXCLUSION:
                _, out, _ = run_ps(f"[bool]((Get-MpPreference).ExclusionPath -contains '{c.target}')")
                return out.lower() == "true"
            if c.type is ChangeType.FILE_LINE:
                val, existed = self._file_read(c)
                return existed and str(val) == str(c.target_value)
            if c.type in (ChangeType.COMMAND, ChangeType.POWER_PLAN):
                if not c.verify_cmd:
                    return True
                rc, out, _ = run_ps(c.verify_cmd)
                return rc == 0
        except Exception:
            return False
        return False

    # ---- revert ----------------------------------------------------------
    def revert(self, c: Change) -> ApplyResult:
        if self.dry_run:
            return ApplyResult(True, False, "[dry-run] would revert")
        try:
            if c.type is ChangeType.ENV_VAR:
                scope = "User" if c.scope is Scope.USER else "Machine"
                if c.existed_before and c.current_value is not None:
                    run_ps(f"[Environment]::SetEnvironmentVariable('{c.target}','{c.current_value}','{scope}')")
                else:
                    run_ps(f"[Environment]::SetEnvironmentVariable('{c.target}',$null,'{scope}')")
                return ApplyResult(True, True, "env reverted")
            if c.type is ChangeType.REGISTRY:
                return self._reg_revert(c)
            if c.type is ChangeType.DEFENDER_EXCLUSION:
                if not c.existed_before:
                    run_ps(f"Remove-MpPreference -ExclusionPath '{c.target}'")
                return ApplyResult(True, True, "exclusion reverted")
            if c.type is ChangeType.FILE_LINE:
                return self._file_revert(c)
            if c.type in (ChangeType.COMMAND, ChangeType.POWER_PLAN):
                if c.undo_cmd:
                    cmd = c.undo_cmd.replace("{prior}", str(c.current_value or ""))
                    rc, out, err = run_ps("$ErrorActionPreference='Stop';" + cmd)
                    return ApplyResult(rc == 0, rc == 0, (out or err)[:200])
                return ApplyResult(False, False, "no undo_cmd")
            if c.type is ChangeType.GUIDED:
                return ApplyResult(True, False, "guided: nothing to revert")
        except Exception as e:
            return ApplyResult(False, False, f"revert error: {e}")
        return ApplyResult(False, False, "unhandled change type")

    # ---- env -------------------------------------------------------------
    def _env_apply(self, c: Change) -> ApplyResult:
        scope = "User" if c.scope is Scope.USER else "Machine"
        rc, out, err = run_ps(
            f"[Environment]::SetEnvironmentVariable('{c.target}','{c.target_value}','{scope}')")
        return ApplyResult(rc == 0, rc == 0, err or f"{c.target}={c.target_value}")

    # ---- registry --------------------------------------------------------
    def _reg_read(self, c: Change) -> Tuple[Optional[object], bool]:
        root, sub = _split_reg(c.target)
        try:
            with winreg.OpenKey(root, sub, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
                val, _ = winreg.QueryValueEx(k, c.reg_value_name)
                return val, True
        except FileNotFoundError:
            return None, False

    def _reg_apply(self, c: Change) -> ApplyResult:
        root, sub = _split_reg(c.target)
        rtype = _REG_TYPES.get(c.reg_type or "REG_DWORD", winreg.REG_DWORD)
        value = int(c.target_value) if rtype in (winreg.REG_DWORD, winreg.REG_QWORD) else str(c.target_value)
        with winreg.CreateKeyEx(root, sub, 0, winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY) as k:
            winreg.SetValueEx(k, c.reg_value_name, 0, rtype, value)
        return ApplyResult(True, True, f"{c.reg_value_name}={value}")

    def _reg_revert(self, c: Change) -> ApplyResult:
        root, sub = _split_reg(c.target)
        rtype = _REG_TYPES.get(c.reg_type or "REG_DWORD", winreg.REG_DWORD)
        try:
            with winreg.OpenKey(root, sub, 0, winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY) as k:
                if c.existed_before and c.current_value is not None:
                    winreg.SetValueEx(k, c.reg_value_name, 0, rtype, c.current_value)
                else:
                    try:
                        winreg.DeleteValue(k, c.reg_value_name)
                    except FileNotFoundError:
                        pass
            return ApplyResult(True, True, "registry reverted")
        except FileNotFoundError:
            return ApplyResult(True, False, "key gone; nothing to revert")

    # ---- config files (houdini.env, Engine.ini) --------------------------
    def _file_read(self, c: Change) -> Tuple[Optional[str], bool]:
        p = Path(c.target)
        if not p.exists():
            return None, False
        section = c.extra.get("section")
        in_section = section is None
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if section is not None:
                if s.startswith("[") and s.endswith("]"):
                    in_section = s[1:-1].strip().lower() == section.strip().lower()
                    continue
            if in_section and s and not s.startswith((";", "#")):
                key = s.split(c.file_sep, 1)[0].strip()
                if key == c.file_key:
                    return s.split(c.file_sep, 1)[1].strip(), True
        return None, False

    def _file_apply(self, c: Change) -> ApplyResult:
        p = Path(c.target)
        bak = p.with_suffix(p.suffix + ".mo2bak")
        if p.exists() and not bak.exists():
            bak.write_text(p.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        elif not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")
        line = f"{c.file_key}{c.file_sep}{c.target_value}"
        text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        out, done = [], False
        for ln in text:
            key = ln.strip().split(c.file_sep, 1)[0].strip() if c.file_sep in ln else None
            if key == c.file_key and not ln.strip().startswith((";", "#")):
                out.append(line)
                done = True
            else:
                out.append(ln)
        if not done:
            out.append(line)
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
        return ApplyResult(True, True, f"{c.file_key} set in {p.name}")

    def _file_revert(self, c: Change) -> ApplyResult:
        p = Path(c.target)
        bak = p.with_suffix(p.suffix + ".mo2bak")
        if bak.exists():
            p.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
            bak.unlink()
            return ApplyResult(True, True, f"{p.name} restored from backup")
        # The optimizer created this file (it did not exist pre-apply): remove it entirely.
        if not c.extra.get("file_existed", True):
            if p.exists():
                p.unlink()
            return ApplyResult(True, True, f"{p.name} removed (created by optimizer)")
        return ApplyResult(True, False, "no backup; left as-is")
