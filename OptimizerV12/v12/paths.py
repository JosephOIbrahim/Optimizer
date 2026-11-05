"""
V12 — path resolution. Anything mutable lives OFF OneDrive.

This project tree sits under C:\\Users\\User\\OneDrive\\Desktop. OneDrive sync
churns open SQLite/WAL files and can corrupt a transaction log mid-write. V9
already solved this (`v9/paths.py`); MachineOptimizer2026 did not, and its
data/transactions.jsonl — the file that describes how to UNDO every change the
optimizer has ever made — has been syncing from OneDrive since July.

Resolution order for the V12 data root:
  1. $OPTIMIZER_V12_DATA        (explicit override)
  2. D:\\OptimizerV12           (the 3.6 TB SSD, if writable)
  3. %LOCALAPPDATA%\\OptimizerV12
  4. <repo>/data_v12            (last resort; warns loudly if under OneDrive)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("v12.paths")

# OptimizerV12/v12/paths.py -> parent.parent = OptimizerV12 -> parent = the Optimizer root
PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

MOS2026_DIR = REPO_ROOT / "MachineOptimizer2026"
V9_PKG_DIR = REPO_ROOT / "v9"


def is_under_onedrive(p: Path) -> bool:
    return any("onedrive" in seg.lower() for seg in Path(p).parts)


def _writable(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".v12_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except (OSError, PermissionError):
        return False


def _resolve_data_root() -> Path:
    env = os.environ.get("OPTIMIZER_V12_DATA")
    if env:
        return Path(env)
    candidates = [Path("D:/OptimizerV12")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "OptimizerV12")
    for c in candidates:
        if _writable(c):
            return c
    fallback = REPO_ROOT / "data_v12"
    if is_under_onedrive(fallback):
        log.warning(
            "V12 data root falling back to %s, which is under OneDrive. Set "
            "$OPTIMIZER_V12_DATA to an off-OneDrive path — OneDrive sync on an "
            "open SQLite/WAL file can corrupt the transaction log.",
            fallback,
        )
    return fallback


DATA_ROOT = _resolve_data_root()
LEDGER_DB = DATA_ROOT / "ledger.db"
BASELINE_DIR = DATA_ROOT / "baselines"
LOCK_FILE = DATA_ROOT / "v12.lock"
LOG_DIR = DATA_ROOT / "logs"

# Where MachineOptimizer2026's reversible state MOVED to (Phase 0a). Its old
# location — under OneDrive — is kept as a read-only migration source so an
# in-flight restore point is never lost.
MOS2026_DATA = DATA_ROOT / "mos2026"
MOS2026_DATA_LEGACY = MOS2026_DIR / "data"


def ensure_dirs() -> None:
    for d in (DATA_ROOT, BASELINE_DIR, LOG_DIR, MOS2026_DATA):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    flag = "  [WARNING: UNDER ONEDRIVE]" if is_under_onedrive(DATA_ROOT) else ""
    return f"V12 data root: {DATA_ROOT}{flag}"
