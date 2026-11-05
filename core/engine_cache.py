"""
EngineCacheManager -- persistent cache of compiled acceleration artifacts.

Owner: Forge-4 (briefs/forge-4_cache_compiler.md, D1).
Reads from: acceleration.backend (Forge-1's ABC + dataclasses).
Read by:    core/background_compiler.py, core/gpu_optimizer_v8.py (Forge-3),
            core/experts/* (Forge-2, lookup-only).

The cache key formula MUST match SHARED_CONTEXT.md exactly:

    blake2b(
        f"{app}|{model_sha256}|{shape_w}x{shape_h}|{batch}|{precision}|"
        f"{backend_name}@{backend_ver}|drv={driver_ver}|sm={89}"
    ).hexdigest()[:32]

`app` is FIRST in the digest input, so two apps with otherwise-identical
parameters produce different keys -- collisions across apps are impossible
by construction (Crucible-1 will hammer this).

Storage layout:
    <cache_dir>/<app>/<hash_key>.plan       -- raw artifact bytes
    <db_path>                               -- SQLite metadata
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Forge-1 surface (read-only).
from acceleration.backend import BuildArtifactDescriptor, ShapeRange

logger = logging.getLogger(__name__)


# ============================================================================
# Public dataclasses
# ============================================================================


@dataclass
class BuildArtifact:
    """
    Hydrated cache row returned by EngineCacheManager.lookup() / store().

    Fields:
        hash_key:        32-char blake2b hex (cache key).
        app:             "comfyui" | "houdini" | "ue5".
        artifact_path:   absolute path to the on-disk .plan file.
        artifact_bytes:  the actual bytes (loaded lazily; populated when the
                         caller asks for them via .read_bytes()).
        descriptor:      BuildArtifactDescriptor reconstructed from row +
                         (optionally) on-disk bytes.
        disk_bytes:      size of the .plan file on disk.
        compile_seconds: wall time the original compile took.
        runtime_seconds_p50: optional measured median runtime (Crucible-2
                         fills this in).
        created_utc:     ISO8601 timestamp of original store() call.
        last_used_utc:   ISO8601 timestamp updated on every lookup() hit.
        use_count:       monotonically increasing usage counter.
    """

    hash_key: str
    app: str
    artifact_path: Path
    descriptor: BuildArtifactDescriptor
    disk_bytes: int
    compile_seconds: Optional[float]
    runtime_seconds_p50: Optional[float]
    created_utc: str
    last_used_utc: str
    use_count: int

    def read_bytes(self) -> bytes:
        """Read the artifact bytes from disk on demand."""
        return Path(self.artifact_path).read_bytes()


# ============================================================================
# Schema
# ============================================================================


_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS artifacts (
    hash_key            TEXT PRIMARY KEY,
    app                 TEXT NOT NULL,
    model_sha256        TEXT NOT NULL,
    shape_w             INTEGER,
    shape_h             INTEGER,
    batch               INTEGER,
    precision           TEXT,
    backend_name        TEXT,
    backend_ver         TEXT,
    driver_ver          TEXT,
    sm_arch             INTEGER,
    artifact_path       TEXT NOT NULL,
    artifact_format     TEXT NOT NULL DEFAULT 'trt_plan',
    disk_bytes          INTEGER NOT NULL,
    compile_seconds     REAL,
    runtime_seconds_p50 REAL,
    notes_json          TEXT,
    shape_min_json      TEXT,
    shape_opt_json      TEXT,
    shape_max_json      TEXT,
    created_utc         TEXT NOT NULL,
    last_used_utc       TEXT NOT NULL,
    use_count           INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_app_lru ON artifacts(app, last_used_utc);
CREATE INDEX IF NOT EXISTS idx_app_model ON artifacts(app, model_sha256);
"""


# ============================================================================
# Manager
# ============================================================================


class EngineCacheManager:
    """
    Filesystem + SQLite cache for compiled acceleration artifacts.

    Threading model:
        - All public methods are safe to call from multiple threads.
        - We open a fresh sqlite3 connection per call (sqlite is OK with
          this; the alternative -- one connection -- needs check_same_thread
          gymnastics).
        - A single `_fs_lock` serialises the small critical section between
          "write .plan.tmp" and "atomic rename to .plan".

    Eviction policy:
        - LRU by `last_used_utc`, optionally scoped to a single `app`.
        - Refuse to evict an artifact whose `last_used_utc` is within the
          last `_IN_USE_WINDOW_S` seconds (default 60). Prevents evicting
          an engine that the foreground app is actively executing.
        - Greedy: evict oldest first until either target met or all
          eligible candidates exhausted.

    Bounded failure (per brief):
        - SQLite write 3 retries, then raise. Caller (BackgroundCompiler /
          orchestrator) is responsible for surfacing the blocker.
        - TRT plan deserialization failure on lookup hit -> invalidate the
          row, return None, do NOT raise. (Backends own deserialization;
          we just guarantee the row goes away when the file is unreadable.)
    """

    _IN_USE_WINDOW_S = 60
    _DEFAULT_MAX_DISK = 200 * 1024 ** 3  # 200 GiB
    _SQLITE_RETRIES = 3
    _SQLITE_RETRY_BACKOFF_S = 0.1
    _APPS = ("comfyui", "houdini", "ue5")

    def __init__(
        self,
        cache_dir: Path,
        db_path: Path = Path("data/engine_cache.db"),
        max_disk_bytes: int = _DEFAULT_MAX_DISK,
        in_use_window_s: int = _IN_USE_WINDOW_S,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.db_path = Path(db_path).resolve()
        self.max_disk_bytes = int(max_disk_bytes)
        self._in_use_window_s = int(in_use_window_s)

        self._fs_lock = threading.Lock()
        self._db_lock = threading.Lock()  # serialises schema-create + writes

        self._ensure_dirs()
        self._init_db()

    # ------------------------------------------------------------------- bootstrap

    def _ensure_dirs(self) -> None:
        """Create <cache_dir>/<app>/ subtree and the SQLite parent dir."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for app in self._APPS:
                (self.cache_dir / app).mkdir(parents=True, exist_ok=True)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"EngineCacheManager could not create dir tree at "
                f"{self.cache_dir} or {self.db_path.parent}: {exc}"
            ) from exc

    def _init_db(self) -> None:
        with self._db_lock:
            with self._connect() as conn:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Open a sqlite3 connection with reasonable defaults."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            isolation_level=None,  # autocommit; we manage txn explicitly
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        # WAL is friendlier under concurrent readers + a writer.
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except sqlite3.DatabaseError:
            # Some test envs ban WAL; safe to ignore.
            pass
        return conn

    # ------------------------------------------------------------------- key

    def compute_key(
        self,
        app: str,
        model_sha256: str,
        shape_w: int,
        shape_h: int,
        batch: int,
        precision: str,
        backend_name: str,
        backend_ver: str,
        driver_ver: str = "596.36",
        sm_arch: int = 89,
    ) -> str:
        """
        Compute the 32-char blake2b cache key.

        Matches SHARED_CONTEXT.md format exactly. `app` is first in the
        digest input so cross-app collisions are impossible.
        """
        if app not in self._APPS:
            raise ValueError(
                f"app must be one of {self._APPS}; got {app!r}"
            )
        payload = (
            f"{app}|{model_sha256}|{shape_w}x{shape_h}|{batch}|{precision}|"
            f"{backend_name}@{backend_ver}|drv={driver_ver}|sm={sm_arch}"
        )
        return hashlib.blake2b(payload.encode("utf-8")).hexdigest()[:32]

    def artifact_path_for(self, app: str, hash_key: str) -> Path:
        """Resolve the canonical on-disk path for (app, hash_key)."""
        return self.cache_dir / app / f"{hash_key}.plan"

    # ------------------------------------------------------------------- lookup

    def lookup(self, hash_key: str) -> Optional[BuildArtifact]:
        """
        Return the BuildArtifact for `hash_key`, or None if missing /
        unreadable.

        On hit we update `last_used_utc` and bump `use_count`. On the file
        having vanished from disk under us, we delete the row and return
        None (self-healing).
        """
        if not isinstance(hash_key, str) or not hash_key:
            return None

        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM artifacts WHERE hash_key = ?",
                    (hash_key,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            logger.error(
                "EngineCacheManager.lookup db error for key=%s: %s",
                hash_key, exc,
            )
            return None

        if row is None:
            return None

        artifact_path = Path(row["artifact_path"])
        if not artifact_path.exists():
            logger.warning(
                "EngineCacheManager.lookup: row exists but file missing: "
                "%s -- self-healing the row.",
                artifact_path,
            )
            self._delete_row(hash_key)
            return None

        # Touch LRU + bump counter. Do this in a single transaction.
        now = _utc_now_iso()
        try:
            with self._db_lock, self._connect() as conn:
                conn.execute(
                    "UPDATE artifacts "
                    "SET last_used_utc = ?, use_count = use_count + 1 "
                    "WHERE hash_key = ?",
                    (now, hash_key),
                )
        except sqlite3.DatabaseError as exc:
            # Don't fail the lookup just because we couldn't update LRU --
            # the artifact is still usable.
            logger.warning(
                "EngineCacheManager.lookup: failed to update LRU for %s: %s",
                hash_key, exc,
            )

        return self._row_to_artifact(row, override_last_used=now,
                                     override_use_count=row["use_count"] + 1)

    def lookup_by_params(
        self,
        app: str,
        model_sha256: str,
        shape_w: int,
        shape_h: int,
        batch: int,
        precision: str,
        backend_name: str,
        backend_ver: str,
        driver_ver: str = "596.36",
        sm_arch: int = 89,
    ) -> Optional[BuildArtifact]:
        """Convenience wrapper: compute key + lookup."""
        key = self.compute_key(
            app, model_sha256, shape_w, shape_h, batch, precision,
            backend_name, backend_ver, driver_ver, sm_arch,
        )
        return self.lookup(key)

    # ------------------------------------------------------------------- store

    def store(
        self,
        hash_key: str,
        plan_bytes: bytes,
        descriptor: BuildArtifactDescriptor,
        app: Optional[str] = None,
        model_sha256: str = "",
        shape_w: int = 0,
        shape_h: int = 0,
        batch: int = 1,
        driver_ver: str = "596.36",
        sm_arch: int = 89,
    ) -> BuildArtifact:
        """
        Persist `plan_bytes` and the corresponding metadata row.

        `app` may be passed explicitly OR inferred from
        `descriptor.notes["app"]` if present. We REQUIRE one of those; we
        will not silently default an app, since that would make the cache
        layout ambiguous.

        Atomic write: bytes go to <path>.tmp first, then os.replace() onto
        the final name. If anything fails before rename, the .tmp is
        cleaned up (try/finally).
        """
        if app is None:
            app = descriptor.notes.get("app") if descriptor.notes else None
        if app not in self._APPS:
            raise ValueError(
                f"store() requires app in {self._APPS}; got {app!r}. "
                f"Pass app=... or set descriptor.notes['app']."
            )
        if not isinstance(plan_bytes, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"plan_bytes must be bytes-like, got {type(plan_bytes)!r}"
            )
        plan_bytes = bytes(plan_bytes)

        artifact_path = self.artifact_path_for(app, hash_key)
        tmp_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")

        # Ensure dir exists (cheap, idempotent).
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- atomic file write ------------------------------------------
        with self._fs_lock:
            try:
                with open(tmp_path, "wb") as fh:
                    fh.write(plan_bytes)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
                os.replace(tmp_path, artifact_path)
            finally:
                # Defensive: if .tmp survived (rare; replace failed), nuke it.
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        logger.warning(
                            "EngineCacheManager.store: leftover tmp %s",
                            tmp_path,
                        )

        # ---- DB row -----------------------------------------------------
        now = _utc_now_iso()
        sr = descriptor.shape_range
        precision = descriptor.precision
        backend_name = descriptor.backend_name
        backend_ver = descriptor.backend_version
        compile_seconds = descriptor.compile_seconds
        notes_json = _json_dumps(descriptor.notes or {})
        shape_min_json = _json_dumps(list(sr.min_shape))
        shape_opt_json = _json_dumps(list(sr.opt_shape))
        shape_max_json = _json_dumps(list(sr.max_shape))
        disk_bytes = len(plan_bytes)
        artifact_format = descriptor.artifact_format

        last_exc: Optional[BaseException] = None
        for attempt in range(1, self._SQLITE_RETRIES + 1):
            try:
                with self._db_lock, self._connect() as conn:
                    conn.execute(
                        "INSERT INTO artifacts ("
                        " hash_key, app, model_sha256, shape_w, shape_h, batch,"
                        " precision, backend_name, backend_ver,"
                        " driver_ver, sm_arch, artifact_path, artifact_format,"
                        " disk_bytes, compile_seconds, runtime_seconds_p50,"
                        " notes_json, shape_min_json, shape_opt_json, shape_max_json,"
                        " created_utc, last_used_utc, use_count"
                        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                        " ON CONFLICT(hash_key) DO UPDATE SET"
                        "  app=excluded.app,"
                        "  model_sha256=excluded.model_sha256,"
                        "  shape_w=excluded.shape_w, shape_h=excluded.shape_h,"
                        "  batch=excluded.batch, precision=excluded.precision,"
                        "  backend_name=excluded.backend_name,"
                        "  backend_ver=excluded.backend_ver,"
                        "  driver_ver=excluded.driver_ver, sm_arch=excluded.sm_arch,"
                        "  artifact_path=excluded.artifact_path,"
                        "  artifact_format=excluded.artifact_format,"
                        "  disk_bytes=excluded.disk_bytes,"
                        "  compile_seconds=excluded.compile_seconds,"
                        "  notes_json=excluded.notes_json,"
                        "  shape_min_json=excluded.shape_min_json,"
                        "  shape_opt_json=excluded.shape_opt_json,"
                        "  shape_max_json=excluded.shape_max_json,"
                        "  last_used_utc=excluded.last_used_utc",
                        (
                            hash_key, app, model_sha256, shape_w, shape_h, batch,
                            precision, backend_name, backend_ver,
                            driver_ver, sm_arch, str(artifact_path), artifact_format,
                            disk_bytes, compile_seconds, None,
                            notes_json, shape_min_json, shape_opt_json, shape_max_json,
                            now, now, 0,
                        ),
                    )
                last_exc = None
                break
            except sqlite3.DatabaseError as exc:
                last_exc = exc
                logger.warning(
                    "EngineCacheManager.store sqlite attempt %d/%d failed: %s",
                    attempt, self._SQLITE_RETRIES, exc,
                )
                time.sleep(self._SQLITE_RETRY_BACKOFF_S * attempt)

        if last_exc is not None:
            # Roll back the file write to keep DB+FS consistent.
            try:
                if artifact_path.exists():
                    artifact_path.unlink()
            except OSError:
                pass
            raise RuntimeError(
                "EngineCacheManager.store: SQLite write failed after "
                f"{self._SQLITE_RETRIES} attempts. data/ permissions; "
                f"need user resolution. Last error: {last_exc}"
            ) from last_exc

        return BuildArtifact(
            hash_key=hash_key,
            app=app,
            artifact_path=artifact_path,
            descriptor=descriptor,
            disk_bytes=disk_bytes,
            compile_seconds=compile_seconds,
            runtime_seconds_p50=None,
            created_utc=now,
            last_used_utc=now,
            use_count=0,
        )

    # ------------------------------------------------------------------- eviction

    def evict_lru(
        self,
        target_bytes: Optional[int] = None,
        target_app: Optional[str] = None,
    ) -> int:
        """
        Evict LRU entries until disk usage drops below `target_bytes`
        (default: 90% of `max_disk_bytes`). Returns total bytes freed.

        Refuses to evict an entry whose `last_used_utc` is within the last
        `_in_use_window_s` seconds (default 60s). This prevents the
        background compiler from reaping an engine that the foreground
        app is actively executing against.

        If `target_app` is specified, only artifacts within that app are
        considered for eviction. Otherwise eviction is global LRU across
        all apps.
        """
        if target_bytes is None:
            target_bytes = int(self.max_disk_bytes * 0.9)
        target_bytes = max(0, int(target_bytes))

        # Fast exit: nothing to do.
        current = self.disk_used_bytes(app=target_app)
        if current <= target_bytes:
            return 0

        cutoff_iso = _utc_iso(datetime.now(timezone.utc) -
                              timedelta(seconds=self._in_use_window_s))

        # Pull candidates oldest-first.
        try:
            with self._connect() as conn:
                if target_app is not None:
                    rows = conn.execute(
                        "SELECT hash_key, app, artifact_path, disk_bytes, last_used_utc "
                        "FROM artifacts WHERE app = ? "
                        "ORDER BY last_used_utc ASC",
                        (target_app,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT hash_key, app, artifact_path, disk_bytes, last_used_utc "
                        "FROM artifacts ORDER BY last_used_utc ASC",
                    ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.error("evict_lru: cannot read candidate rows: %s", exc)
            return 0

        freed = 0
        skipped_in_use = 0
        for row in rows:
            if current - freed <= target_bytes:
                break
            if row["last_used_utc"] >= cutoff_iso:
                # In-use; skip.
                skipped_in_use += 1
                continue
            n = self._evict_one(row["hash_key"], Path(row["artifact_path"]))
            freed += n
            logger.info(
                "evict_lru: dropped %s (%s, %d bytes, last_used %s)",
                row["hash_key"], row["app"], n, row["last_used_utc"],
            )

        if current - freed > target_bytes:
            logger.warning(
                "evict_lru could not reach target: freed=%d wanted<=%d "
                "(skipped %d in-use entries within %ds window)",
                freed, target_bytes, skipped_in_use, self._in_use_window_s,
            )
        return freed

    def _evict_one(self, hash_key: str, artifact_path: Path) -> int:
        """Delete file + row. Returns bytes freed (0 on best-effort failure)."""
        bytes_freed = 0
        try:
            if artifact_path.exists():
                bytes_freed = artifact_path.stat().st_size
                artifact_path.unlink()
        except OSError as exc:
            logger.warning("evict_lru: cannot delete %s: %s", artifact_path, exc)
        self._delete_row(hash_key)
        return bytes_freed

    def _delete_row(self, hash_key: str) -> None:
        try:
            with self._db_lock, self._connect() as conn:
                conn.execute(
                    "DELETE FROM artifacts WHERE hash_key = ?",
                    (hash_key,),
                )
        except sqlite3.DatabaseError as exc:
            logger.warning("delete_row(%s): %s", hash_key, exc)

    # ------------------------------------------------------------------- inspection

    def disk_used_bytes(self, app: Optional[str] = None) -> int:
        """Total on-disk footprint, optionally scoped to one app."""
        try:
            with self._connect() as conn:
                if app is None:
                    cur = conn.execute(
                        "SELECT COALESCE(SUM(disk_bytes), 0) AS s FROM artifacts"
                    )
                else:
                    cur = conn.execute(
                        "SELECT COALESCE(SUM(disk_bytes), 0) AS s "
                        "FROM artifacts WHERE app = ?",
                        (app,),
                    )
                row = cur.fetchone()
                return int(row["s"]) if row else 0
        except sqlite3.DatabaseError as exc:
            logger.error("disk_used_bytes: %s", exc)
            return 0

    def disk_used_gb(self, app: Optional[str] = None) -> float:
        return self.disk_used_bytes(app=app) / (1024 ** 3)

    def stats(self) -> Dict[str, Any]:
        """Summary counts + bytes per app, for the dashboard."""
        out: Dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "db_path": str(self.db_path),
            "max_disk_bytes": self.max_disk_bytes,
            "max_disk_gb": self.max_disk_bytes / (1024 ** 3),
            "total_bytes": self.disk_used_bytes(),
            "total_gb": self.disk_used_gb(),
            "per_app": {},
        }
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT app, COUNT(*) AS n, COALESCE(SUM(disk_bytes), 0) AS s "
                    "FROM artifacts GROUP BY app",
                ).fetchall()
            for r in rows:
                out["per_app"][r["app"]] = {
                    "count": int(r["n"]),
                    "bytes": int(r["s"]),
                    "gb": int(r["s"]) / (1024 ** 3),
                }
        except sqlite3.DatabaseError as exc:
            logger.error("stats: %s", exc)
        return out

    # ------------------------------------------------------------------- invalidation

    def invalidate_app(self, app: str) -> int:
        """Drop every cached artifact for `app`. Returns count removed."""
        if app not in self._APPS:
            raise ValueError(f"app must be one of {self._APPS}; got {app!r}")
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT hash_key, artifact_path FROM artifacts WHERE app = ?",
                    (app,),
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.error("invalidate_app: %s", exc)
            return 0

        n = 0
        for r in rows:
            self._evict_one(r["hash_key"], Path(r["artifact_path"]))
            n += 1
        return n

    def invalidate_on_driver_change(self, current_driver: str) -> int:
        """
        Drop every artifact whose `driver_ver` does not match
        `current_driver`. Returns count removed.

        Driver upgrades can invalidate previously-built TRT plans (NVIDIA
        does NOT promise plan portability across major driver versions);
        the orchestrator calls this on startup if it detects a driver
        delta.
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT hash_key, artifact_path FROM artifacts "
                    "WHERE driver_ver IS NOT NULL AND driver_ver != ?",
                    (current_driver,),
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.error("invalidate_on_driver_change: %s", exc)
            return 0

        n = 0
        for r in rows:
            self._evict_one(r["hash_key"], Path(r["artifact_path"]))
            n += 1
        if n:
            logger.info(
                "invalidate_on_driver_change: dropped %d entries "
                "(driver != %s)", n, current_driver,
            )
        return n

    def invalidate_stale_models(self, model_dir: Path) -> int:
        """
        Compare model file mtimes vs cache entries' created_utc. If the
        model file's mtime is newer than the cache entry, the cached engine
        is stale and gets evicted. Returns count removed.

        Matching strategy: we look up every distinct model_sha256 in the
        cache. For each, we search `model_dir` recursively for a file
        whose sha256 matches; if found and mtime > created_utc, evict.

        This is intentionally conservative: if we can't find the model
        file, we leave the entry alone (the engine still works for that
        sha; only when the underlying file changed do we evict).
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            logger.warning(
                "invalidate_stale_models: model_dir does not exist: %s",
                model_dir,
            )
            return 0

        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT model_sha256 FROM artifacts "
                    "WHERE model_sha256 != ''",
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.error("invalidate_stale_models: select: %s", exc)
            return 0

        if not rows:
            return 0

        # Build sha -> latest mtime for files in model_dir.
        # We hash each candidate file once; this can be slow for huge
        # model directories, so we cap at 64 MiB scan budget per file (we
        # hash the full file -- any heuristic less than that risks false
        # negatives). Caller invokes this on startup, not per-cycle.
        sha_to_mtime: Dict[str, float] = {}
        for path in model_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                sha = _sha256_file(path)
            except OSError:
                continue
            mtime = path.stat().st_mtime
            prev = sha_to_mtime.get(sha)
            if prev is None or mtime > prev:
                sha_to_mtime[sha] = mtime

        n = 0
        for r in rows:
            sha = r["model_sha256"]
            on_disk_mtime = sha_to_mtime.get(sha)
            if on_disk_mtime is None:
                continue  # model file not present; leave cache alone
            try:
                with self._connect() as conn:
                    crows = conn.execute(
                        "SELECT hash_key, artifact_path, created_utc "
                        "FROM artifacts WHERE model_sha256 = ?",
                        (sha,),
                    ).fetchall()
            except sqlite3.DatabaseError as exc:
                logger.warning("invalidate_stale_models: %s", exc)
                continue
            for cr in crows:
                created_ts = _iso_to_epoch(cr["created_utc"])
                if on_disk_mtime > created_ts:
                    self._evict_one(cr["hash_key"], Path(cr["artifact_path"]))
                    n += 1
        if n:
            logger.info("invalidate_stale_models: dropped %d entries", n)
        return n

    # ------------------------------------------------------------------- helpers

    def _row_to_artifact(
        self,
        row: sqlite3.Row,
        override_last_used: Optional[str] = None,
        override_use_count: Optional[int] = None,
    ) -> BuildArtifact:
        sr = ShapeRange(
            min_shape=tuple(_json_loads(row["shape_min_json"]) or ()),
            opt_shape=tuple(_json_loads(row["shape_opt_json"]) or ()),
            max_shape=tuple(_json_loads(row["shape_max_json"]) or ()),
        ) if row["shape_min_json"] else ShapeRange.static((0,))

        descriptor = BuildArtifactDescriptor(
            backend_name=row["backend_name"] or "",
            backend_version=row["backend_ver"] or "",
            artifact_bytes=b"",  # caller uses .read_bytes() to lazy-load
            artifact_format=row["artifact_format"] or "trt_plan",
            precision=row["precision"] or "",
            shape_range=sr,
            compile_seconds=float(row["compile_seconds"] or 0.0),
            notes=_json_loads(row["notes_json"]) or {},
        )
        return BuildArtifact(
            hash_key=row["hash_key"],
            app=row["app"],
            artifact_path=Path(row["artifact_path"]),
            descriptor=descriptor,
            disk_bytes=int(row["disk_bytes"]),
            compile_seconds=row["compile_seconds"],
            runtime_seconds_p50=row["runtime_seconds_p50"],
            created_utc=row["created_utc"],
            last_used_utc=override_last_used or row["last_used_utc"],
            use_count=int(override_use_count if override_use_count is not None
                          else row["use_count"]),
        )


# ============================================================================
# Module-private utilities
# ============================================================================


def _utc_now_iso() -> str:
    return _utc_iso(datetime.now(timezone.utc))


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Compact ISO8601 with tz; sortable lexicographically for LRU.
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")


def _iso_to_epoch(iso: str) -> float:
    """Parse one of our ISO timestamps back to an epoch float."""
    if not iso:
        return 0.0
    # Tolerate both "+0000" and "+00:00" zones.
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(iso, fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0


def _json_dumps(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return "{}"


def _json_loads(s: Optional[str]) -> Any:
    import json
    if not s:
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ============================================================================
# CLI sanity
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EngineCacheManager CLI")
    parser.add_argument("--cache-dir", default=r"D:\OptimizerV8_EngineCache")
    parser.add_argument("--db", default="data/engine_cache.db")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    mgr = EngineCacheManager(Path(args.cache_dir), Path(args.db))
    if args.stats:
        import json
        print(json.dumps(mgr.stats(), indent=2))
    else:
        print(f"EngineCacheManager ready. cache_dir={mgr.cache_dir}")
