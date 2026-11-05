"""V9 — quality ledger + Pareto admission gate.

A build-and-cache acceleration is only worth keeping if it is BOTH faster AND
not visibly worse. We measure three numbers per candidate:

  * speedup       -- accelerated wall time / baseline wall time, as a ratio
                     (>1.0 means faster; 1.30 == 30% faster).
  * lpips_delta   -- perceptual distance from the baseline output (LPIPS).
                     Lower is better; 0.0 == identical.
  * clip_delta    -- change in CLIP score vs baseline (prompt adherence).
                     Higher is better; negative means it drifted off-prompt.

The admission gate (DESIGN_V9 §1, the Pareto quality gate) is::

    speedup >= 1.30  AND  lpips_delta <= 0.04  AND  clip_delta >= -0.02

All three must hold. A candidate that is fast but degrades the image, or
preserves the image but isn't fast enough, is rejected.

The ledger records every measured candidate (admitted or not) in an
off-OneDrive SQLite DB (paths.db_path('quality')) so we can audit decisions
and never claim a speedup we didn't measure (DESIGN_V9 principle 5).
Degrade-safe: writes/reads swallow errors rather than crash the caller.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from v9 import paths

logger = logging.getLogger("v9.quality")

# --- the gate thresholds (the only quality constants in the system) --------
MIN_SPEEDUP = 1.30
MAX_LPIPS_DELTA = 0.04
MIN_CLIP_DELTA = -0.02


@dataclass(frozen=True)
class ParetoVerdict:
    """Result of a Pareto admission check."""

    admitted: bool
    speedup: float
    lpips_delta: float
    clip_delta: float
    reasons: tuple[str, ...]  # which sub-criteria failed (empty if admitted)

    def __bool__(self) -> bool:
        return self.admitted


def pareto_admit(
    speedup: float, lpips_delta: float, clip_delta: float
) -> ParetoVerdict:
    """Apply the Pareto admission gate.

        speedup >= 1.30 AND lpips_delta <= 0.04 AND clip_delta >= -0.02

    Returns a ParetoVerdict that is truthy iff all three criteria hold and
    lists the failing sub-criteria otherwise.
    """
    speedup = float(speedup)
    lpips_delta = float(lpips_delta)
    clip_delta = float(clip_delta)

    reasons: List[str] = []
    if not (speedup >= MIN_SPEEDUP):
        reasons.append(f"speedup {speedup:.3f} < {MIN_SPEEDUP:.2f}")
    if not (lpips_delta <= MAX_LPIPS_DELTA):
        reasons.append(f"lpips_delta {lpips_delta:.4f} > {MAX_LPIPS_DELTA:.2f}")
    if not (clip_delta >= MIN_CLIP_DELTA):
        reasons.append(f"clip_delta {clip_delta:.4f} < {MIN_CLIP_DELTA:.2f}")

    return ParetoVerdict(
        admitted=not reasons,
        speedup=speedup,
        lpips_delta=lpips_delta,
        clip_delta=clip_delta,
        reasons=tuple(reasons),
    )


# ============================================================================
# Ledger
# ============================================================================


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS quality (
    id              INTEGER PRIMARY KEY,
    ts_utc          TEXT NOT NULL,
    app             TEXT,
    backend_name    TEXT,
    backend_version TEXT,
    model_sha256    TEXT,
    shape_w         INTEGER,
    shape_h         INTEGER,
    batch           INTEGER,
    precision       TEXT,
    speedup         REAL NOT NULL,
    lpips_delta     REAL NOT NULL,
    clip_delta      REAL NOT NULL,
    admitted        INTEGER NOT NULL,
    reasons_json    TEXT,
    notes_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_quality_workload
    ON quality(model_sha256, shape_w, shape_h);
CREATE INDEX IF NOT EXISTS idx_quality_backend
    ON quality(backend_name, admitted);
"""

_init_lock = threading.Lock()
_initialized_paths: set[str] = set()


def _db() -> Path:
    return paths.db_path("quality")


def ensure_schema(db_path: Optional[Path] = None) -> Optional[Path]:
    """Idempotently create the quality schema. Returns the resolved path, or
    None if it couldn't be initialised (degrade-safe)."""
    p = Path(db_path).resolve() if db_path else _db()
    key = str(p)
    with _init_lock:
        if key in _initialized_paths:
            return p
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with _open(p) as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
            _initialized_paths.add(key)
            return p
        except (sqlite3.DatabaseError, OSError) as exc:
            logger.error("quality ensure_schema failed for %s: %s", p, exc)
            return None


def _open(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(db_path), timeout=10.0, isolation_level=None, check_same_thread=False
    )
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except sqlite3.DatabaseError:
        pass
    conn.row_factory = sqlite3.Row
    return conn


def record_quality(
    ts_utc: str,
    speedup: float,
    lpips_delta: float,
    clip_delta: float,
    app: Optional[str] = None,
    backend_name: Optional[str] = None,
    backend_version: Optional[str] = None,
    model_sha256: Optional[str] = None,
    shape_w: int = 0,
    shape_h: int = 0,
    batch: int = 1,
    precision: Optional[str] = None,
    notes: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> ParetoVerdict:
    """Evaluate the Pareto gate, persist the measurement + verdict, and return
    the verdict. Recording is degrade-safe (a DB failure logs and is ignored);
    the verdict is always returned so the caller can act on it."""
    verdict = pareto_admit(speedup, lpips_delta, clip_delta)
    p = ensure_schema(db_path)
    if p is None:
        return verdict
    try:
        with _open(p) as conn:
            conn.execute(
                "INSERT INTO quality ("
                " ts_utc, app, backend_name, backend_version, model_sha256,"
                " shape_w, shape_h, batch, precision,"
                " speedup, lpips_delta, clip_delta, admitted,"
                " reasons_json, notes_json"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ts_utc, app, backend_name, backend_version, model_sha256,
                    int(shape_w), int(shape_h), int(batch), precision,
                    verdict.speedup, verdict.lpips_delta, verdict.clip_delta,
                    1 if verdict.admitted else 0,
                    json.dumps(list(verdict.reasons), default=str),
                    json.dumps(notes or {}, default=str),
                ),
            )
    except (sqlite3.DatabaseError, OSError) as exc:
        logger.warning("record_quality failed (non-fatal): %s", exc)
    return verdict


def fetch_recent_quality(
    limit: int = 50,
    backend_name: Optional[str] = None,
    admitted_only: bool = False,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    p = ensure_schema(db_path)
    if p is None:
        return []
    clauses: List[str] = []
    params: List[Any] = []
    if backend_name:
        clauses.append("backend_name = ?")
        params.append(backend_name)
    if admitted_only:
        clauses.append("admitted = 1")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(int(limit))
    try:
        with _open(p) as conn:
            rows = conn.execute(
                f"SELECT * FROM quality{where} ORDER BY ts_utc DESC LIMIT ?",
                tuple(params),
            ).fetchall()
            return [dict(r) for r in rows]
    except (sqlite3.DatabaseError, OSError) as exc:
        logger.warning("fetch_recent_quality failed: %s", exc)
        return []
