"""V9 — path resolution. Anything mutable (caches, DBs, queues, logs) must
live OFF OneDrive, because this project tree sits under
C:\\Users\\User\\OneDrive\\Desktop and OneDrive sync churns SQLite/WAL files.

Resolution order for the data root:
  1. $OPTIMIZER_V9_DATA (explicit override)
  2. D:\\OptimizerV9   (the 3.6 TB SSD, if D: is writable)
  3. %LOCALAPPDATA%\\OptimizerV9
  4. <repo>/data_v9   (last resort; warns if under OneDrive)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("v9.paths")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_under_onedrive(p: Path) -> bool:
    parts = {seg.lower() for seg in p.parts}
    return "onedrive" in parts or any("onedrive" in seg.lower() for seg in p.parts)


def _writable(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".v9_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except (OSError, PermissionError):
        return False


def _resolve_data_root() -> Path:
    env = os.environ.get("OPTIMIZER_V9_DATA")
    if env:
        return Path(env)
    candidates = [Path("D:/OptimizerV9")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "OptimizerV9")
    for c in candidates:
        if _writable(c):
            return c
    fallback = REPO_ROOT / "data_v9"
    if _is_under_onedrive(fallback):
        log.warning("V9 data root falling back to %s which is under OneDrive; "
                    "set $OPTIMIZER_V9_DATA to an off-OneDrive path to avoid "
                    "sync churn on SQLite files.", fallback)
    return fallback


DATA_ROOT = _resolve_data_root()


def cache_dir(app: str = "") -> Path:
    """Engine-cache artifact directory, optionally per-app."""
    d = DATA_ROOT / "engine_cache"
    if app:
        d = d / app
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path(name: str) -> Path:
    """SQLite path (e.g. 'engine_cache', 'telemetry', 'quality')."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT / f"{name}.db"


def queue_path(name: str) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT / f"{name}.json"


def log_path(name: str = "optimizer") -> Path:
    d = DATA_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.log"


def info() -> dict:
    return {
        "data_root": str(DATA_ROOT),
        "under_onedrive": _is_under_onedrive(DATA_ROOT),
        "repo_root": str(REPO_ROOT),
    }


if __name__ == "__main__":
    print(info())
