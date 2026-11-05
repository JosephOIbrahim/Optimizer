"""
MachineOptimizer2026 — path resolution. Mutable state lives OFF OneDrive.

Why this file now exists
------------------------
`data/` sat under `C:\\Users\\User\\OneDrive\\Desktop\\Optimizer\\MachineOptimizer2026`.
OneDrive syncs it. On 2026-09-16 an integrity check of `data/transactions.jsonl`
found **9 torn records out of 58,570** — lines split mid-object, leaving fragments
like `}`, `rue}`, `e}` and `verified": true}` on their own lines. That is the
signature of a sync client flushing a partial write while another process appends.

`restore_points/master.json` — the file that actually decides whether this machine
can be un-optimized — was intact, all 33 entries well-formed. So no recovery
capability was lost this time. That was luck, not design: the same failure one
directory over would have destroyed the undo record for 33 registry keys, service
settings and file edits.

V9 already solved this (`v9/paths.py` resolves to `D:\\OptimizerV9` first). This
mirrors that resolution order, and deliberately imports nothing from V12, so the
two agree on the data root without being coupled.

Resolution order:
  1. $MACHINEOPTIMIZER_DATA          (explicit override)
  2. D:\\OptimizerV12\\mos2026        (the 3.6 TB SSD, if writable)
  3. %LOCALAPPDATA%\\MachineOptimizer2026
  4. <package>/data                  (last resort; warns loudly if under OneDrive)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("mos2026.paths")

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent

#: Where the data used to live, under OneDrive. Kept as a read-only migration
#: source so an in-flight restore point is never lost.
LEGACY_DATA = PROJECT_DIR / "data"


def is_under_onedrive(p: Path) -> bool:
    return any("onedrive" in seg.lower() for seg in Path(p).parts)


def _writable(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".mos_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except (OSError, PermissionError):
        return False


def _resolve_data_root() -> Path:
    env = os.environ.get("MACHINEOPTIMIZER_DATA")
    if env:
        return Path(env)
    candidates = [Path("D:/OptimizerV12/mos2026")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "MachineOptimizer2026")
    for c in candidates:
        if _writable(c):
            return c
    if is_under_onedrive(LEGACY_DATA):
        log.warning(
            "MachineOptimizer2026 data root falling back to %s, which is UNDER ONEDRIVE. "
            "This is the arrangement that tore 9 records out of transactions.jsonl. "
            "Set $MACHINEOPTIMIZER_DATA to an off-OneDrive path.", LEGACY_DATA)
    return LEGACY_DATA


DATA = _resolve_data_root()
RESTORE_POINTS = DATA / "restore_points"
TRANSACTIONS = DATA / "transactions.jsonl"
MASTER = RESTORE_POINTS / "master.json"


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    RESTORE_POINTS.mkdir(parents=True, exist_ok=True)


def migration_needed() -> bool:
    """True when state is still only at the legacy location."""
    if DATA == LEGACY_DATA:
        return False
    return LEGACY_DATA.exists() and not MASTER.exists()


def load_master_with_fallback() -> dict:
    """Read the restore master from the new location, else the legacy one.

    Falling back matters: a restore point written before the migration describes
    changes that are still applied to the machine. Refusing to read it because the
    file moved would strand the machine in its optimized state with no way back.
    """
    for p in (MASTER, LEGACY_DATA / "restore_points" / "master.json"):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                log.error("master at %s is unreadable: %s", p, e)
    return {"created": None, "changes": {}}


def describe() -> str:
    flag = "  [WARNING: UNDER ONEDRIVE]" if is_under_onedrive(DATA) else ""
    extra = "  (migration pending)" if migration_needed() else ""
    return f"MOS2026 data: {DATA}{flag}{extra}"
