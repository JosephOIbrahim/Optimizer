"""
ComfyUI / cross-app telemetry persistence client.

Owner: Forge-4 (briefs/forge-4_cache_compiler.md, D3).

Forge-4 owns:
  - the schema (`generations` and `compiles` tables),
  - the schema-migration code,
  - the thin write client (`record_generation`, `record_compile`).

Forge-2 (specialists / collectors) calls `record_generation()` after every
foreground generation. Forge-4's BackgroundCompiler calls
`record_compile()` after every successful compile.

The `generations` table schema is locked by the brief:

    CREATE TABLE generations (
        id INTEGER PRIMARY KEY,
        ts_utc TEXT NOT NULL,
        workflow_hash TEXT,
        model_sha256 TEXT,
        shape_w INTEGER, shape_h INTEGER, batch INTEGER,
        sampler TEXT, steps INTEGER,
        precision TEXT,
        backend_set TEXT,    -- comma-separated: "sage_attn,trt_unet"
        wall_seconds REAL,
        samples_per_sec REAL,
        vram_peak_gb REAL
    );
    CREATE INDEX idx_workload ON generations(model_sha256, shape_w, shape_h);

We also add a `compiles` table for compile-side rows (cache miss costs):

    CREATE TABLE compiles (
        id INTEGER PRIMARY KEY,
        ts_utc TEXT NOT NULL,
        app TEXT NOT NULL,
        backend_name TEXT,
        backend_version TEXT,
        model_sha256 TEXT,
        shape_w INTEGER, shape_h INTEGER, batch INTEGER,
        precision TEXT,
        compile_seconds REAL,
        artifact_bytes INTEGER,
        notes_json TEXT
    );
    CREATE INDEX idx_compile_workload ON compiles(model_sha256, shape_w, shape_h);

Naming: the file is called `comfyui_telemetry.db` per the brief, but we
write rows for any app via `app` column on `compiles`. The `generations`
table is ComfyUI-specific by historical convention; Houdini / UE5 collectors
will get sibling tables in a later phase.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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


def ensure_schema(db_path: Path) -> None:
    """Idempotently create the schema for `db_path`.

    We cache the (resolved) path in a module-level set so we don't pay the
    schema-create round-trip on every record_generation() call.
    """
    db_path = Path(db_path).resolve()
    key = str(db_path)
    with _init_lock:
        if key in _initialized_paths:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with _open(db_path) as conn:
            conn.executescript(_GENERATIONS_SCHEMA)
            conn.executescript(_COMPILES_SCHEMA)
            conn.commit()
        _initialized_paths.add(key)


def _open(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(db_path),
        timeout=10.0,
        isolation_level=None,
        check_same_thread=False,
    )
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except sqlite3.DatabaseError:
        pass
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================
# Public client
# ============================================================================


def record_generation(
    db_path: Path,
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
) -> int:
    """Insert one row into `generations`. Returns the new row id.

    `backend_set` is stored as a comma-joined string ("sage_attn,trt_unet").
    """
    ensure_schema(db_path)
    backend_str = ",".join(sorted(set(backend_set or [])))
    with _open(Path(db_path)) as conn:
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


def record_compile(
    db_path: Path,
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
) -> int:
    """Insert one row into `compiles`. Returns the new row id."""
    ensure_schema(db_path)
    notes_json = json.dumps(notes or {}, default=str)
    with _open(Path(db_path)) as conn:
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


# ============================================================================
# Read helpers (used by the dashboard, optional)
# ============================================================================


def fetch_recent_generations(
    db_path: Path,
    limit: int = 50,
    model_sha256: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ensure_schema(db_path)
    with _open(Path(db_path)) as conn:
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


def fetch_recent_compiles(
    db_path: Path,
    limit: int = 50,
    app: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ensure_schema(db_path)
    with _open(Path(db_path)) as conn:
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
