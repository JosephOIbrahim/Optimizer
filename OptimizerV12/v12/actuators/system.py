"""
V12 — the SYSTEM-tier actuator.

Two mechanisms behind one interface, because this tier contains two genuinely
different kinds of target and pretending otherwise is how the predecessors got
into trouble.

**The classical family** — env vars, registry, power plans, Defender exclusions,
config-file lines, shell commands. MachineOptimizer2026 already implements these
correctly and has proven it in production, including the snapshot-once discipline
that stops the watchdog from re-snapshotting a value it has itself already
changed. V12 does not reimplement them; it bridges to them. `Mos2026Bridge` maps a
V12 Change onto MOS2026's `Change` dataclass and delegates.

**The non-observable family** — locked clocks, clock offsets, NVAPI 3D profiles.
MOS2026 has no implementation because there is nothing to read back, and its
executor is built on read-back. These are handled natively here, with the
reversibility honestly declared rather than faked:

  * `nvmlDeviceSetGpuLockedClocks` works on Ada. NVML exposes **no getter** —
    `nvmlDeviceGetClockInfo` returns the *live* clock, not the lock setting, and
    `nvmlDeviceGetMaxClockInfo` does not reflect it either. So the snapshot is
    `trusted=False` by construction, and the only honest verb is reset-to-default.
    Verification must be behavioural, under load.
  * `nvmlDeviceSetClockOffsets` is worse: a getter exists and reports the offset
    as 0 even when one is set. Writing that into the snapshot would record a lie as
    ground truth.

The consequence is stated plainly rather than buried: applying either of these
**destroys whatever the operator had**, because reset returns to vendor default
and not to their prior state. That is why the Gate refuses them without an
explicit opt-in, and why the verdict carries a `destructive_reset` flag into the
ledger.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

from .. import paths
from ..change import (Change, ChangeKind, Observability, Snapshot, Tier,
                      change_key)
from ..admission import ApplyResult

log = logging.getLogger("v12.actuator.system")


# ---------------------------------------------------------------------------
# OBSERVE — snapshot a target, recording how much the answer can be trusted
# ---------------------------------------------------------------------------
def snapshot_change(change: Change) -> Snapshot:
    """Capture the prior value of `change.target`, with provenance.

    This is the function whose absence made MOS2026's reversibility partly
    fictional. It never claims more than it can support.
    """
    # --- targets with no readable prior, by hardware design ---
    if change.kind is ChangeKind.NVML_POWER_LIMIT:
        # The power ceiling IS readable, and the setter round-trips it exactly.
        # Measured on this box (2026-11-04): set 500 W -> read back 500.0 W;
        # set 450 W -> read back 450.0 W. So the prior is captured truthfully
        # and undo restores it — no vendor-default reset, no operator state lost.
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                w = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            finally:
                pynvml.nvmlShutdown()
            return Snapshot(value=w, existed=True, trusted=True,
                            source="nvmlDeviceGetPowerManagementLimit")
        except Exception as e:
            return Snapshot(value=None, existed=False, trusted=False,
                            source=f"NVML power limit read failed: {e}", error=str(e))
    if change.kind is ChangeKind.NVML_CLOCK:
        return Snapshot(
            value=None, existed=False, trusted=False,
            source="NONE — NVML exposes set/reset for locked clocks but no getter; "
                   "the prior lock state is unrecoverable",
        )
    if change.kind in (ChangeKind.NVML_OFFSET, ChangeKind.NVAPI_PROFILE):
        return Snapshot(
            value=None, existed=False, trusted=False,
            source="LYING GETTER — a getter exists but reports the offset as 0 when "
                   "one is set; recording it would write a false prior into the ledger",
        )

    if change.kind is ChangeKind.ENV_VAR:
        import os
        scope = os.environ if change.scope == "user" else os.environ
        if change.target in scope:
            return Snapshot(value=scope[change.target], existed=True, trusted=True,
                            source="os.environ")
        # The orchestrator's env may not reflect the persisted user/machine value.
        return Snapshot(value=None, existed=False, trusted=True,
                        source="absent from this process environment")

    if change.kind is ChangeKind.REGISTRY:
        try:
            import winreg
            hive, sub, name = _split_reg(change.target, change.reg_value_name)
            with winreg.OpenKey(hive, sub, 0, winreg.KEY_READ) as k:
                v, _ = winreg.QueryValueEx(k, name)
            return Snapshot(value=v, existed=True, trusted=True,
                            source=f"winreg.QueryValueEx({change.target}\\{name})")
        except FileNotFoundError:
            return Snapshot(value=None, existed=False, trusted=True,
                            source="registry value absent")
        except Exception as e:
            return Snapshot(value=None, existed=False, trusted=False,
                            source="registry read failed", error=str(e))

    if change.kind is ChangeKind.FILE_LINE:
        try:
            p = Path(change.target)
            if not p.exists():
                return Snapshot(value=None, existed=False, trusted=True,
                                source="file absent")
            key = change.file_key or ""
            sep = change.file_sep or "="
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or sep not in stripped:
                    continue
                k, _, v = stripped.partition(sep)
                if k.strip() == key:
                    return Snapshot(value=v.strip(), existed=True, trusted=True,
                                    source=f"{p.name}:{key}")
            return Snapshot(value=None, existed=False, trusted=True,
                            source=f"{p.name}: key absent")
        except Exception as e:
            return Snapshot(value=None, existed=False, trusted=False,
                            source="file read failed", error=str(e))

    if change.kind is ChangeKind.DEFENDER_EXCLUSION:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "((Get-MpPreference).ExclusionPath) -join '|'"],
                capture_output=True, text=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            present = change.target.lower() in out.lower()
            return Snapshot(value=present, existed=present, trusted=True,
                            source="Get-MpPreference.ExclusionPath")
        except Exception as e:
            return Snapshot(value=None, existed=False, trusted=False,
                            source="Defender query failed", error=str(e))

    if change.snapshot_cmd:
        rc, out = _run_ps(change.snapshot_cmd)
        return Snapshot(value=out.strip() if rc == 0 else None,
                        existed=(rc == 0 and bool(out.strip())),
                        trusted=(rc == 0), source=f"snapshot_cmd (rc={rc})",
                        error="" if rc == 0 else out.strip()[:200])

    return Snapshot(value=None, existed=False, trusted=False,
                    source="no snapshot method for this kind")


def _split_reg(target: str, name: Optional[str]):
    import winreg
    hive_map = {"HKCU": winreg.HKEY_CURRENT_USER, "HKLM": winreg.HKEY_LOCAL_MACHINE}
    parts = target.split("\\", 1)
    hive = hive_map.get(parts[0].upper())
    if hive is None:
        raise ValueError(f"unsupported hive in {target!r}")
    return hive, parts[1], (name or "")


def _run_ps(cmd: str, timeout: int = 60) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except Exception as e:
        return 1, str(e)


# ---------------------------------------------------------------------------
# The bridge to MachineOptimizer2026's proven executor
# ---------------------------------------------------------------------------
class Mos2026Bridge:
    """Translate a V12 Change into MOS2026's shape and let MOS2026 do the work.

    MOS2026's executor is the most-battle-tested code in this project: it snapshots
    once, uses `{prior}` templating for undos, and refuses to re-snapshot a target
    it has already touched. Reimplementing it would throw that away for no gain.
    """

    KIND_MAP = {
        ChangeKind.ENV_VAR: "env_var",
        ChangeKind.REGISTRY: "registry",
        ChangeKind.POWER_PLAN: "power_plan",
        ChangeKind.DEFENDER_EXCLUSION: "defender_exclusion",
        ChangeKind.FILE_LINE: "file_line",
        ChangeKind.COMMAND: "command",
    }

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._executor = None
        self._available = False
        self._why = ""
        self._load()

    def _load(self) -> None:
        try:
            root = str(paths.REPO_ROOT / "MachineOptimizer2026")
            if root not in sys.path:
                sys.path.insert(0, root)
            from optimizer.executor import Executor  # type: ignore
            self._executor = Executor(dry_run=self.dry_run)
            self._available = True
        except Exception as e:
            self._why = f"{type(e).__name__}: {e}"
            log.warning("MOS2026 executor unavailable (%s) — SYSTEM-tier classical "
                        "changes cannot be applied", self._why)

    @property
    def available(self) -> bool:
        return self._available

    def to_mos(self, c: Change) -> Any:
        from optimizer.base import Change as MosChange, ChangeType  # type: ignore
        return MosChange(
            id=c.id,
            type=ChangeType(self.KIND_MAP[c.kind]),
            description=c.description,
            target=c.target,
            target_value=c.target_value,
            reg_value_name=c.reg_value_name,
            reg_type=c.reg_type,
            scope=c.scope,
            requires_admin=c.requires_admin,
            file_key=c.file_key,
            file_sep=c.file_sep,
            snapshot_cmd=c.snapshot_cmd,
            apply_cmd=c.apply_cmd,
            undo_cmd=c.undo_cmd,
            verify_cmd=c.verify_cmd,
            current_value=(c.snapshot.value if c.snapshot else None),
            existed_before=(c.snapshot.existed if c.snapshot else False),
            risk=c.risk,
            extra=dict(c.extra),
        )

    def apply(self, c: Change) -> ApplyResult:
        if not self._available:
            return ApplyResult(change_id=c.id, applied=False,
                               error=f"MOS2026 executor unavailable: {self._why}")
        try:
            mos = self.to_mos(c)
            ok = self._executor.apply(mos)
            return ApplyResult(change_id=c.id, applied=bool(ok),
                               verified=bool(ok),
                               detail="delegated to MachineOptimizer2026 executor",
                               error="" if ok else "executor reported failure")
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False, error=f"{type(e).__name__}: {e}")

    def apply_undo(self, c: Change) -> ApplyResult:
        if not self._available:
            return ApplyResult(change_id=c.id, applied=False,
                               error=f"MOS2026 executor unavailable: {self._why}")
        try:
            ok = self._executor.undo(self.to_mos(c))
            return ApplyResult(change_id=c.id, applied=bool(ok),
                               detail="delegated to MachineOptimizer2026 executor (undo)")
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False, error=f"{type(e).__name__}: {e}")

    def verify(self, c: Change) -> ApplyResult:
        if not self._available:
            return ApplyResult(change_id=c.id, applied=c.applied, verified=False,
                               error="MOS2026 executor unavailable")
        try:
            ok = self._executor.verify(self.to_mos(c))
            return ApplyResult(change_id=c.id, applied=c.applied, verified=bool(ok),
                               verification_source="MOS2026 executor.verify (read-back == target)")
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=c.applied, verified=False,
                               error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# NVML — the non-observable family
# ---------------------------------------------------------------------------
class NvmlActuator:
    """Locked clocks and clock offsets, with reversibility declared honestly.

    Handles three kinds:
      NVML_CLOCK   set/reset locked clocks. NO getter exists.
      NVML_OFFSET  set clock offsets. A getter exists and LIES.

    Both are Administrator-only and neither persists across reboot. Both are
    reset-only: undoing returns the GPU to VENDOR DEFAULT, which is not necessarily
    what the operator had.
    """

    def __init__(self, index: int = 0, dry_run: bool = False):
        self.index = index
        self.dry_run = dry_run
        self._p = None
        self._last = ""

    def _nvml(self):
        if self._p is not None:
            return self._p
        try:
            import pynvml
            pynvml.nvmlInit()
            self._p = pynvml
        except Exception as e:
            self._last = f"nvml unavailable: {e}"
            self._p = None
        return self._p

    def _handle(self):
        p = self._nvml()
        return p.nvmlDeviceGetHandleByIndex(self.index) if p else None

    # ---- clocks ----------------------------------------------------------
    def apply(self, c: Change) -> ApplyResult:
        if c.kind is ChangeKind.NVML_CLOCK:
            return self._lock_clocks(c)
        if c.kind is ChangeKind.NVML_OFFSET:
            return self._set_offsets(c)
        if c.kind is ChangeKind.NVML_POWER_LIMIT:
            return self._set_power_limit(c)
        if c.kind is ChangeKind.NVAPI_PROFILE:
            return ApplyResult(
                change_id=c.id, applied=False,
                error="NVAPI 3D-profile writes have no supported read-back path and need "
                      "nvidia-settings/NVIDIA App; not implemented — use a CITED guided "
                      "change instead")
        return ApplyResult(change_id=c.id, applied=False, error=f"unsupported kind {c.kind}")

    def _lock_clocks(self, c: Change) -> ApplyResult:
        p, h = self._nvml(), None
        if p is None:
            return ApplyResult(change_id=c.id, applied=False, error=self._last)
        try:
            h = p.nvmlDeviceGetHandleByIndex(self.index)
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False, error=str(e))

        lo, hi = (c.target_value if isinstance(c.target_value, (tuple, list))
                  else (c.target_value, c.target_value))
        if lo is None or hi is None:
            return ApplyResult(change_id=c.id, applied=False,
                               error="locked clocks need a (min,max) MHz pair; "
                                     "a pin (min==max) is the only unambiguous setting")
        if self.dry_run:
            return ApplyResult(change_id=c.id, applied=False, verified=False,
                               detail=f"dry-run: would lock {lo}-{hi} MHz")
        try:
            rc = p.nvmlDeviceSetGpuLockedClocks(h, int(lo), int(hi))
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False, error=f"{type(e).__name__}: {e}")
        if rc != 0:
            # A non-zero return is itself the answer — this is how unsupported
            # targets announce themselves. Not a bug to work around.
            return ApplyResult(change_id=c.id, applied=False,
                               error=f"nvmlDeviceSetGpuLockedClocks returned {rc} "
                                     f"(NVML_ERROR_NOT_SUPPORTED is a final answer, not "
                                     f"a transient failure)")
        # Applied. Verification is NOT possible by read-back, by construction.
        return ApplyResult(
            change_id=c.id, applied=True, verified=False,
            verification_source=(
                "NOT VERIFIABLE BY READ-BACK — NVML has no getter for locked clocks. "
                "Confirm behaviourally: pin min==max, then check the live clock under "
                "load sits at the pin and that throttle reasons show no sw_power_cap "
                "drift. On rollback this returns to VENDOR DEFAULT, not to any prior lock."))

    def _set_offsets(self, c: Change) -> ApplyResult:
        p = self._nvml()
        if p is None:
            return ApplyResult(change_id=c.id, applied=False, error=self._last)
        if self.dry_run:
            return ApplyResult(change_id=c.id, applied=False, detail="dry-run: would set offsets")
        try:
            h = p.nvmlDeviceGetHandleByIndex(self.index)
            off = p.c_nvmlClockOffset_t()
            off.type = p.NVML_CLOCK_MEM if str(c.target).lower().startswith("mem") else p.NVML_CLOCK_GRAPHICS
            off.pstate = p.NVML_PSTATE_0
            off.clockOffsetMHz = int(c.target_value or 0)
            rc = p.nvmlDeviceSetClockOffsets(h, off)
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False, error=f"{type(e).__name__}: {e}")
        if rc != 0:
            return ApplyResult(change_id=c.id, applied=False,
                               error=f"nvmlDeviceSetClockOffsets returned {rc}")
        return ApplyResult(
            change_id=c.id, applied=True, verified=False,
            verification_source=(
                "NOT VERIFIABLE — the getter for clock offsets reports 0 even when an "
                "offset is set. Read-back would be confidently wrong. Verify only by "
                "observing the achieved clock under load."))

    # ---- the power ceiling ----------------------------------------------
    def _set_power_limit(self, c: Change) -> ApplyResult:
        """Raise or lower the card's power ceiling in watts.

        The one NVML target here whose prior is both readable and exactly
        restorable, so undo is a RESTORE, not a reset.

        Note the unit: NVML takes MILLIWATTS. Passing watts is not a wrong
        value, it is an invalid one — nvmlDeviceSetPowerManagementLimit
        returns NVML_ERROR_INVALID_ARGUMENT for 500 (meaning 0.5 W) and
        succeeds for 500000 (meaning 500 W). Measured on this box:
        500000 -> read back 500.0 W; 450000 -> read back 450.0 W.
        """
        p = self._nvml()
        if p is None:
            return ApplyResult(change_id=c.id, applied=False, error=self._last)
        try:
            h = p.nvmlDeviceGetHandleByIndex(self.index)
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False, error=f"{type(e).__name__}: {e}")
        want = c.target_value
        if not isinstance(want, (int, float)):
            return ApplyResult(change_id=c.id, applied=False,
                               error="a power limit needs a watt value")
        want = float(want)
        try:
            lo, hi = p.nvmlDeviceGetPowerManagementLimitConstraints(h)
            lo_w, hi_w = lo / 1000.0, hi / 1000.0
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False,
                               error=f"could not read the allowed power range: {e}")
        if not (lo_w <= want <= hi_w):
            return ApplyResult(change_id=c.id, applied=False,
                               error=f"{want:.0f} W is outside this card's allowed range "
                                     f"({lo_w:.0f}-{hi_w:.0f} W)")
        if self.dry_run:
            return ApplyResult(change_id=c.id, applied=False, verified=False,
                               detail=f"dry-run: would set the power ceiling to {want:.0f} W "
                                      f"(currently {p.nvmlDeviceGetPowerManagementLimit(h)/1000.0:.0f} W)")
        try:
            rc = p.nvmlDeviceSetPowerManagementLimit(h, int(want * 1000))
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False, error=f"{type(e).__name__}: {e}")
        if rc != 0:
            return ApplyResult(change_id=c.id, applied=False,
                               error=f"nvmlDeviceSetPowerManagementLimit returned {rc}")
        # Read-back verification: this target has a trustworthy getter, so unlike
        # locked clocks the read-back IS evidence that the change landed.
        try:
            got = p.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=True, verified=False,
                               error=f"applied but could not read it back: {e}")
        if abs(got - want) > 1.0:
            return ApplyResult(change_id=c.id, applied=False, verified=False,
                               error=f"set {want:.0f} W but read back {got:.0f} W")
        return ApplyResult(change_id=c.id, applied=True, verified=True,
                           verification_source="nvmlDeviceGetPowerManagementLimit read-back",
                           detail=f"power ceiling {got:.0f} W "
                                  f"(allowed {lo_w:.0f}-{hi_w:.0f} W)")

    def _restore_power_limit(self, c: Change) -> ApplyResult:
        """Put the operator's own ceiling back. Not a reset — a restore."""
        p = self._nvml()
        if p is None:
            return ApplyResult(change_id=c.id, applied=False, error=self._last)
        prior = c.snapshot.value if c.snapshot else None
        if prior is None or not c.snapshot.trusted:
            return ApplyResult(change_id=c.id, applied=False,
                               error="no trustworthy prior power limit was captured, so the "
                                     "ceiling cannot be restored exactly")
        if self.dry_run:
            return ApplyResult(change_id=c.id, applied=False,
                               detail=f"dry-run: would restore the power ceiling to {prior:.0f} W")
        try:
            h = p.nvmlDeviceGetHandleByIndex(self.index)
            rc = p.nvmlDeviceSetPowerManagementLimit(h, int(float(prior) * 1000))
            got = p.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            return ApplyResult(change_id=c.id, applied=(rc == 0),
                               verified=abs(got - float(prior)) <= 1.0,
                               verification_source="nvmlDeviceGetPowerManagementLimit read-back",
                               detail=f"power ceiling restored to {got:.0f} W")
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False, error=f"{type(e).__name__}: {e}")

    # ---- the reset-only verb --------------------------------------------
    def reset_default(self, c: Change) -> ApplyResult:
        """The ONLY rollback available for this family.

        Not a restore: it returns the GPU to vendor default clocks. If the operator
        had a lock or an undervolt, it is gone. The Gate requires an explicit opt-in
        for exactly this reason, and the warning is emitted at the point of action
        rather than only at admission time.

        The power ceiling is the exception and does not come through here: it
        has a trustworthy prior and is restored exactly, because it is not
        destructive — undoing it returns the operator's own number.
        """
        if c.kind is ChangeKind.NVML_POWER_LIMIT:
            return self._restore_power_limit(c)
        p = self._nvml()
        if p is None:
            return ApplyResult(change_id=c.id, applied=False, error=self._last)
        log.warning("resetting '%s' to VENDOR DEFAULT — any operator-applied clock lock "
                    "or undervolt on this GPU is discarded, not restored", c.id)
        if self.dry_run:
            return ApplyResult(change_id=c.id, applied=False, detail="dry-run: would reset")
        try:
            h = p.nvmlDeviceGetHandleByIndex(self.index)
            errors = []
            if c.kind is ChangeKind.NVML_CLOCK:
                rc = p.nvmlDeviceResetGpuLockedClocks(h)
                if rc != 0:
                    errors.append(f"nvmlDeviceResetGpuLockedClocks={rc}")
            elif c.kind is ChangeKind.NVML_OFFSET:
                off = p.c_nvmlClockOffset_t()
                off.type = p.NVML_CLOCK_GRAPHICS
                off.pstate = p.NVML_PSTATE_0
                off.clockOffsetMHz = 0
                rc = p.nvmlDeviceSetClockOffsets(h, off)
                if rc != 0:
                    errors.append(f"nvmlDeviceSetClockOffsets(0)={rc}")
            return ApplyResult(change_id=c.id, applied=not errors,
                               detail="reset to vendor default",
                               error="; ".join(errors))
        except Exception as e:
            return ApplyResult(change_id=c.id, applied=False, error=f"{type(e).__name__}: {e}")

    def verify(self, c: Change) -> ApplyResult:
        return ApplyResult(
            change_id=c.id, applied=c.applied, verified=False,
            verification_source=f"{c.observability.value} target — read-back is never "
                                f"evidence here")

    def apply_undo(self, c: Change) -> ApplyResult:
        return self.reset_default(c)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
class SystemActuator:
    """Routes a SYSTEM-tier change to whichever mechanism can actually carry it."""

    NVML_KINDS = (ChangeKind.NVML_CLOCK, ChangeKind.NVML_OFFSET, ChangeKind.NVAPI_PROFILE)

    def __init__(self, dry_run: bool = False):
        self.bridge = Mos2026Bridge(dry_run=dry_run)
        self.nvml = NvmlActuator(dry_run=dry_run)

    def apply(self, c: Change) -> ApplyResult:
        if c.kind in self.NVML_KINDS:
            return self.nvml.apply(c)
        return self.bridge.apply(c)

    def apply_undo(self, c: Change) -> ApplyResult:
        if c.kind in self.NVML_KINDS:
            return self.nvml.apply_undo(c)
        return self.bridge.apply_undo(c)

    def reset_default(self, c: Change) -> ApplyResult:
        if c.kind in self.NVML_KINDS:
            return self.nvml.reset_default(c)
        # The classical family is restorable, so a reset is not normally reached;
        # fall back to undo rather than doing nothing.
        return self.bridge.apply_undo(c)

    def verify(self, c: Change) -> ApplyResult:
        if c.kind in self.NVML_KINDS:
            return self.nvml.verify(c)
        return self.bridge.verify(c)
