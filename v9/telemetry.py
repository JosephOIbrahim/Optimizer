"""V9 — per-generation timing ledger.

Two tables in one off-OneDrive SQLite DB (paths.db_path('telemetry')):

  * `generations` — one row per foreground generation (the thing we are
    trying to make faster). Collectors call record_generation() after each
    run; backend_set records which acceleration backends were active so we
    can attribute a speedup.

  * `compiles` — one row per successful background compile (the cache-miss
    cost). BackgroundCompiler calls record_compile() after each build.

Schema::

    CREATE TABLE generations (
        id INTEGER PRIMARY KEY,
        ts_utc TEXT NOT NULL,
        workflow_hash TEXT,
        model_sha256 TEXT,
        shape_w INTEGER, shape_h INTEGER, batch INTEGER,
        sampler TEXT, steps INTEGER,
        precision TEXT,
        backend_set TEXT,          -- comma-separated: "sage_attn,trt_unet"
        wall_seconds REAL,
        samples_per_sec REAL,
        vram_peak_gb REAL
    );

    CREATE TABLE compiles (
        id INTEGER PRIMARY KEY,
        ts_utc TEXT NOT NULL,
        app TEXT NOT NULL,
        backend_name TEXT, backend_version TEXT,
        model_sha256 TEXT,
        shape_w INTEGER, shape_h INTEGER, batch INTEGER,
        precision TEXT,
        compile_seconds REAL,
        artifact_bytes INTEGER,
        notes_json TEXT
    );

Degrade-safe (DESIGN_V9 principle 2): the record/fetch helpers swallow
SQLite/OS errors and return a benign value rather than raising into the
caller's hot path. The path resolves off-OneDrive through v9.paths.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from v9 import paths

logger = logging.getLogger("v9.telemetry")


_GENERATIONS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS generations (
    id              INTEGER PRIMARY KEY,
    ts_utc          TEXT NOT NULL,
    workflow_hash   TEXT,
    model_sha256    TEXT,
    shape_w         INTEGER,
    shape_h         INTEGER,
    batch           INTEGER,
    sampler         TEXT,
    steps           INTEGER,
    precision       TEXT,
    backend_set     TEXT,
    wall_seconds    REAL,
    samples_per_sec REAL,
    vram_peak_gb    REAL
);
CREATE INDEX IF NOT EXISTS idx_workload
    ON generations(model_sha256, shape_w, shape_h);
"""

_COMPILES_SCHEMA = """\
CREATE TABLE IF NOT EXISTS compiles (
    id               INTEGER PRIMARY KEY,
    ts_utc           TEXT NOT NULL,
    app              TEXT NOT NULL,
    backend_name     TEXT,
    backend_version  TEXT,
    model_sha256     TEXT,
    shape_w          INTEGER,
    shape_h          INTEGER,
    batch            INTEGER,
    precision        TEXT,
    compile_seconds  REAL,
    artifact_bytes   INTEGER,
    notes_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_compile_workload
    ON compiles(model_sha256, shape_w, shape_h);
"""


_init_lock = threading.Lock()
_initialized_paths: set[str] = set()


def _db() -> Path:
    """The telemetry DB path (off-OneDrive)."""
    return paths.db_path("telemetry")


def ensure_schema(db_path: Optional[Path] = None) -> Optional[Path]:
    """Idempotently create the schema. Returns the resolved path, or None if
    the DB couldn't be initialised (degrade-safe -- never raises)."""
    p = Path(db_path).resolve() if db_path else _db()
    key = str(p)
    with _init_lock:
        if key in _initialized_paths:
            return p
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with _open(p) as conn:
                conn.executescript(_GENERATIONS_SCHEMA)
                conn.executescript(_COMPILES_SCHEMA)
                conn.commit()
            _initialized_paths.add(key)
            return p
        except (sqlite3.DatabaseError, OSError) as exc:
            logger.error("telemetry ensure_schema failed for %s: %s", p, exc)
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


# ============================================================================
# Write client
# ============================================================================


def record_generation(
    ts_utc: str,
    workflow_hash: Optional[str],
    model_sha256: Optional[str],
    shape_w: int,
    shape_h: int,
    batch: int,
    sampler: Optional[str],
    steps: Optional[int],
    precision: Optional[str],
    backend_set: List[str],
    wall_seconds: float,
    samples_per_sec: float,
    vram_peak_gb: float,
    db_path: Optional[Path] = None,
) -> Optional[int]:
    """Insert one `generations` row. Returns the new row id, or None on a
    degraded write (never raises into the caller's hot path).

    `backend_set` is stored comma-joined ("sage_attn,trt_unet")."""
    p = ensure_schema(db_path)
    if p is None:
        return None
    backend_str = ",".join(sorted(set(backend_set or [])))
    try:
        with _open(p) as conn:
            cur = conn.execute(
                "INSERT INTO generations ("
                " ts_utc, workflow_hash, model_sha256, shape_w, shape_h, batch,"
                " sampler, steps, precision, backend_set,"
                " wall_seconds, samples_per_sec, vram_peak_gb"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ts_utc, workflow_hash, model_sha256,
                    int(shape_w), int(shape_h), int(batch),
                    sampler, int(steps) if steps is not None else None,
                    precision, backend_str,
                    float(wall_seconds), float(samples_per_sec), float(vram_peak_gb),
                ),
            )
            return int(cur.lastrowid)
    except (sqlite3.DatabaseError, OSError) as exc:
        logger.warning("record_generation failed (non-fatal): %s", exc)
        return None


def record_compile(
    ts_utc: str,
    app: str,
    backend_name: str,
    backend_version: str,
    model_sha256: str,
    shape_w: int,
    shape_h: int,
    batch: int,
    precision: str,
    compile_seconds: float,
    artifact_bytes: int,
    notes: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> Optional[int]:
    """Insert one `compiles` row. Returns the new row id, or None on a
    degraded write."""
    p = ensure_schema(db_path)
    if p is None:
        return None
    notes_json = json.dumps(notes or {}, default=str)
    try:
        with _open(p) as conn:
            cur = conn.execute(
                "INSERT INTO compiles ("
                " ts_utc, app, backend_name, backend_version, model_sha256,"
                " shape_w, shape_h, batch, precision,"
                " compile_seconds, artifact_bytes, notes_json"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ts_utc, app, backend_name, backend_version, model_sha256,
                    int(shape_w), int(shape_h), int(batch), precision,
                    float(compile_seconds), int(artifact_bytes), notes_json,
                ),
            )
            return int(cur.lastrowid)
    except (sqlite3.DatabaseError, OSError) as exc:
        logger.warning("record_compile failed (non-fatal): %s", exc)
        return None


# ============================================================================
# Read helpers
# ============================================================================


def fetch_recent_generations(
    limit: int = 50,
    model_sha256: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    p = ensure_schema(db_path)
    if p is None:
        return []
    try:
        with _open(p) as conn:
            if model_sha256:
                rows = conn.execute(
                    "SELECT * FROM generations WHERE model_sha256 = ? "
                    "ORDER BY ts_utc DESC LIMIT ?",
                    (model_sha256, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM generations ORDER BY ts_utc DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            return [dict(r) for r in rows]
    except (sqlite3.DatabaseError, OSError) as exc:
        logger.warning("fetch_recent_generations failed: %s", exc)
        return []


def fetch_recent_compiles(
    limit: int = 50,
    app: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    p = ensure_schema(db_path)
    if p is None:
        return []
    try:
        with _open(p) as conn:
            if app:
                rows = conn.execute(
                    "SELECT * FROM compiles WHERE app = ? "
                    "ORDER BY ts_utc DESC LIMIT ?",
                    (app, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM compiles ORDER BY ts_utc DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            return [dict(r) for r in rows]
    except (sqlite3.DatabaseError, OSError) as exc:
        logger.warning("fetch_recent_compiles failed: %s", exc)
        return []
