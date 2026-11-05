"""EngineCacheManager -- persistent cache of compiled acceleration artifacts.

The build-and-cache half of the optimizer produces artifacts (TRT plans,
CUTLASS .so, ...) that take minutes to compile and must survive across runs.
This module persists them on disk + in SQLite, keyed so that a change to the
machine that would invalidate a plan also invalidates the cache row.

Cache-key formula::

    blake2b(
        f"{app}|{model_sha256}|{shape_w}x{shape_h}|{batch}|{precision}|"
        f"{backend_name}@{backend_ver}|drv={driver_ver}|sm={sm_arch}"
    ).hexdigest()[:32]

CRITICAL (DESIGN_V9 principle 1 -- detect, don't hardcode): `driver_ver` and
`sm_arch` are NOT literals. They come from the live HardwareProfile via
`HardwareProfile.cache_key_fields()`. This kills the class of bug where the
config said one driver, the code defaulted to another, and the real driver
was a third. `invalidate_on_driver_change` likewise compares against the
DETECTED driver.

`app` is FIRST in the digest input so two apps with otherwise-identical
parameters produce different keys -- cross-app collisions are impossible by
construction.

Storage layout::

    paths.cache_dir(app)/<hash_key>.plan   -- raw artifact bytes (off-OneDrive)
    paths.db_path('engine_cache')          -- SQLite metadata (off-OneDrive)
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from v9 import paths
from v9.accel.backend import BuildArtifactDescriptor, ShapeRange
from v9.hardware import HardwareProfile

logger = logging.getLogger("v9.build.cache")


# ============================================================================
# Public dataclass
# ============================================================================


@dataclass
class BuildArtifact:
    """Hydrated cache row returned by EngineCacheManager.lookup() / store().

    `descriptor.artifact_bytes` is left empty on a lookup; call
    `read_bytes()` to lazy-load the on-disk plan.
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
    """Filesystem + SQLite cache for compiled acceleration artifacts.

    All paths resolve off-OneDrive through v9.paths. The driver/arch facts
    that make up a cache key come from a detected HardwareProfile -- never a
    literal.

    Threading model:
        - All public methods are safe to call from multiple threads.
        - A fresh sqlite3 connection is opened per call.
        - `_fs_lock` serialises the .plan.tmp -> atomic-replace critical
          section.

    Eviction policy:
        - LRU by `last_used_utc`, optionally scoped to one `app`.
        - Refuses to evict an artifact whose `last_used_utc` is within the
          last `in_use_window_s` seconds (default 60) -- prevents reaping an
          engine the foreground app is actively executing.

    Bounded failure:
        - SQLite write: 3 retries then raise (caller surfaces the blocker).
        - On a lookup hit whose file vanished: drop the row, return None --
          never raise (self-heal, DESIGN_V9 principle 2).
    """

    _IN_USE_WINDOW_S = 60
    _DEFAULT_MAX_DISK = 200 * 1024 ** 3  # 200 GiB
    _SQLITE_RETRIES = 3
    _SQLITE_RETRY_BACKOFF_S = 0.1
    _APPS = ("comfyui", "houdini", "ue5")

    def __init__(
        self,
        hardware: HardwareProfile,
        cache_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
        max_disk_bytes: int = _DEFAULT_MAX_DISK,
        in_use_window_s: int = _IN_USE_WINDOW_S,
    ) -> None:
        """
        Args:
            hardware:        detected HardwareProfile -- the single source of
                             the driver/arch facts baked into cache keys.
            cache_dir:       artifact root; defaults to paths.cache_dir()
                             (off-OneDrive). Per-app subdirs created lazily.
            db_path:         SQLite metadata path; defaults to
                             paths.db_path('engine_cache') (off-OneDrive).
            max_disk_bytes:  eviction high-water mark.
            in_use_window_s: protect-from-eviction window.
        """
        self.hardware = hardware
        # paths.cache_dir() / db_path() already resolve off-OneDrive and never
        # raise on a missing dir (they mkdir). We degrade rather than crash.
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else paths.cache_dir()
        self.db_path = (
            Path(db_path).resolve() if db_path else paths.db_path("engine_cache")
        )
        self.max_disk_bytes = int(max_disk_bytes)
        self._in_use_window_s = int(in_use_window_s)

        self._fs_lock = threading.Lock()
        self._db_lock = threading.Lock()

        self._ensure_dirs()
        self._init_db()

    # ---- hardware-derived key defaults ---------------------------------

    def _hw_driver(self) -> str:
        return str(self.hardware.cache_key_fields().get("driver_ver", "unknown"))

    def _hw_sm_arch(self) -> int:
        return int(self.hardware.cache_key_fields().get("sm_arch", 0) or 0)

    # ---- bootstrap ------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create the cache root + per-app subtree + SQLite parent dir.

        Degrade-safe: a failure here is logged, not raised. The cache simply
        becomes a no-op miss-everything cache rather than crashing the cycle.
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for app in self._APPS:
                (self.cache_dir / app).mkdir(parents=True, exist_ok=True)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(
                "EngineCacheManager could not create dir tree at %s / %s: %s "
                "-- cache will operate degraded (miss-only).",
                self.cache_dir, self.db_path.parent, exc,
            )

    def _init_db(self) -> None:
        try:
            with self._db_lock:
                with self._connect() as conn:
                    conn.executescript(_SCHEMA_SQL)
                    conn.commit()
        except sqlite3.DatabaseError as exc:
            logger.error("EngineCacheManager schema init failed: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except sqlite3.DatabaseError:
            pass
        return conn

    # ---- key ------------------------------------------------------------

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
        driver_ver: Optional[str] = None,
        sm_arch: Optional[int] = None,
    ) -> str:
        """Compute the 32-char blake2b cache key.

        `driver_ver` / `sm_arch` default to the DETECTED HardwareProfile
        values -- callers should normally NOT pass them. `app` is first in
        the digest input so cross-app collisions are impossible.
        """
        if app not in self._APPS:
            raise ValueError(f"app must be one of {self._APPS}; got {app!r}")
        drv = driver_ver if driver_ver is not None else self._hw_driver()
        sm = sm_arch if sm_arch is not None else self._hw_sm_arch()
        payload = (
            f"{app}|{model_sha256}|{shape_w}x{shape_h}|{batch}|{precision}|"
            f"{backend_name}@{backend_ver}|drv={drv}|sm={sm}"
        )
        return hashlib.blake2b(payload.encode("utf-8")).hexdigest()[:32]

    def artifact_path_for(self, app: str, hash_key: str) -> Path:
        return self.cache_dir / app / f"{hash_key}.plan"

    # ---- lookup ---------------------------------------------------------

    def lookup(self, hash_key: str) -> Optional[BuildArtifact]:
        """Return the BuildArtifact for `hash_key`, or None if missing /
        unreadable. On hit, touch LRU + bump use_count. If the row exists but
        the file vanished, drop the row and return None (self-heal)."""
        if not isinstance(hash_key, str) or not hash_key:
            return None

        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM artifacts WHERE hash_key = ?",
                    (hash_key,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            logger.error("lookup db error for key=%s: %s", hash_key, exc)
            return None

        if row is None:
            return None

        artifact_path = Path(row["artifact_path"])
        if not artifact_path.exists():
            logger.warning(
                "lookup: row exists but file missing: %s -- self-healing.",
                artifact_path,
            )
            self._delete_row(hash_key)
            return None

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
            logger.warning("lookup: failed to update LRU for %s: %s", hash_key, exc)

        return self._row_to_artifact(
            row, override_last_used=now, override_use_count=row["use_count"] + 1
        )

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
        driver_ver: Optional[str] = None,
        sm_arch: Optional[int] = None,
    ) -> Optional[BuildArtifact]:
        """Convenience: compute key (from detected HW unless overridden) + lookup."""
        key = self.compute_key(
            app, model_sha256, shape_w, shape_h, batch, precision,
            backend_name, backend_ver, driver_ver, sm_arch,
        )
        return self.lookup(key)

    # ---- store ----------------------------------------------------------

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
        driver_ver: Optional[str] = None,
        sm_arch: Optional[int] = None,
    ) -> BuildArtifact:
        """Persist `plan_bytes` + metadata row.

        `app` may be passed or inferred from `descriptor.notes['app']`; one of
        those is REQUIRED (no silent default -- it would make the layout
        ambiguous). The driver/arch columns default to the detected HW.

        Atomic write: bytes go to <path>.tmp then os.replace() onto the final
        name; the .tmp is cleaned up on any failure.
        """
        if app is None:
            app = descriptor.notes.get("app") if descriptor.notes else None
        if app not in self._APPS:
            raise ValueError(
                f"store() requires app in {self._APPS}; got {app!r}. "
                f"Pass app=... or set descriptor.notes['app']."
            )
        if not isinstance(plan_bytes, (bytes, bytearray, memoryview)):
            raise TypeError(f"plan_bytes must be bytes-like, got {type(plan_bytes)!r}")
        plan_bytes = bytes(plan_bytes)

        drv = driver_ver if driver_ver is not None else self._hw_driver()
        sm = sm_arch if sm_arch is not None else self._hw_sm_arch()

        artifact_path = self.artifact_path_for(app, hash_key)
        tmp_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
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
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        logger.warning("store: leftover tmp %s", tmp_path)

        # ---- DB row -----------------------------------------------------
        now = _utc_now_iso()
        sr = descriptor.shape_range
        notes_json = _json_dumps(descriptor.notes or {})
        shape_min_json = _json_dumps(list(sr.min_shape))
        shape_opt_json = _json_dumps(list(sr.opt_shape))
        shape_max_json = _json_dumps(list(sr.max_shape))
        disk_bytes = len(plan_bytes)

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
                            descriptor.precision, descriptor.backend_name,
                            descriptor.backend_version,
                            drv, sm, str(artifact_path), descriptor.artifact_format,
                            disk_bytes, descriptor.compile_seconds, None,
                            notes_json, shape_min_json, shape_opt_json, shape_max_json,
                            now, now, 0,
                        ),
                    )
                last_exc = None
                break
            except sqlite3.DatabaseError as exc:
                last_exc = exc
                logger.warning(
                    "store sqlite attempt %d/%d failed: %s",
                    attempt, self._SQLITE_RETRIES, exc,
                )
                time.sleep(self._SQLITE_RETRY_BACKOFF_S * attempt)

        if last_exc is not None:
            try:
                if artifact_path.exists():
                    artifact_path.unlink()
            except OSError:
                pass
            raise RuntimeError(
                "EngineCacheManager.store: SQLite write failed after "
                f"{self._SQLITE_RETRIES} attempts; need user resolution. "
                f"Last error: {last_exc}"
            ) from last_exc

        return BuildArtifact(
            hash_key=hash_key,
            app=app,
            artifact_path=artifact_path,
            descriptor=descriptor,
            disk_bytes=disk_bytes,
            compile_seconds=descriptor.compile_seconds,
            runtime_seconds_p50=None,
            created_utc=now,
            last_used_utc=now,
            use_count=0,
        )

    # ---- eviction -------------------------------------------------------

    def evict_lru(
        self,
        target_bytes: Optional[int] = None,
        target_app: Optional[str] = None,
    ) -> int:
        """Evict LRU entries until disk usage drops below `target_bytes`
        (default: 90% of max_disk_bytes). Returns bytes freed. Refuses to
        evict entries used within the in-use window. Scoped to `target_app`
        if given, else global LRU."""
        if target_bytes is None:
            target_bytes = int(self.max_disk_bytes * 0.9)
        target_bytes = max(0, int(target_bytes))

        current = self.disk_used_bytes(app=target_app)
        if current <= target_bytes:
            return 0

        cutoff_iso = _utc_iso(
            datetime.now(timezone.utc) - timedelta(seconds=self._in_use_window_s)
        )

        try:
            with self._connect() as conn:
                if target_app is not None:
                    rows = conn.execute(
                        "SELECT hash_key, app, artifact_path, disk_bytes, last_used_utc "
                        "FROM artifacts WHERE app = ? ORDER BY last_used_utc ASC",
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
                conn.execute("DELETE FROM artifacts WHERE hash_key = ?", (hash_key,))
        except sqlite3.DatabaseError as exc:
            logger.warning("delete_row(%s): %s", hash_key, exc)

    # ---- inspection -----------------------------------------------------

    def disk_used_bytes(self, app: Optional[str] = None) -> int:
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
        """Summary counts + bytes per app, for the dashboard / doctor."""
        out: Dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "db_path": str(self.db_path),
            "max_disk_bytes": self.max_disk_bytes,
            "max_disk_gb": self.max_disk_bytes / (1024 ** 3),
            "total_bytes": self.disk_used_bytes(),
            "total_gb": self.disk_used_gb(),
            "driver_ver": self._hw_driver(),
            "sm_arch": self._hw_sm_arch(),
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

    # ---- invalidation ---------------------------------------------------

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

    def invalidate_on_driver_change(self, current_driver: Optional[str] = None) -> int:
        """Drop every artifact whose `driver_ver` != the current driver.
        Returns count removed.

        `current_driver` defaults to the DETECTED HardwareProfile driver, so
        the orchestrator can call this with no args on startup and have it do
        the right thing. NVIDIA does not promise TRT plan portability across
        major driver versions.
        """
        if current_driver is None:
            current_driver = self._hw_driver()
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
                "invalidate_on_driver_change: dropped %d entries (driver != %s)",
                n, current_driver,
            )
        return n

    def invalidate_stale_models(self, model_dir: Path) -> int:
        """Evict cache entries whose underlying model file's mtime is newer
        than the cache entry's created_utc. Conservative: if the model file
        isn't found, the entry is left alone. Returns count removed.
        Intended for startup, not per-cycle (it hashes model files)."""
        model_dir = Path(model_dir)
        if not model_dir.exists():
            logger.warning(
                "invalidate_stale_models: model_dir does not exist: %s", model_dir
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
                continue
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

    # ---- helpers --------------------------------------------------------

    def _row_to_artifact(
        self,
        row: sqlite3.Row,
        override_last_used: Optional[str] = None,
        override_use_count: Optional[int] = None,
    ) -> BuildArtifact:
        sr = (
            ShapeRange(
                min_shape=tuple(_json_loads(row["shape_min_json"]) or ()),
                opt_shape=tuple(_json_loads(row["shape_opt_json"]) or ()),
                max_shape=tuple(_json_loads(row["shape_max_json"]) or ()),
            )
            if row["shape_min_json"]
            else ShapeRange.static((0,))
        )

        descriptor = BuildArtifactDescriptor(
            backend_name=row["backend_name"] or "",
            backend_version=row["backend_ver"] or "",
            artifact_bytes=b"",  # lazy-load via BuildArtifact.read_bytes()
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
            use_count=int(
                override_use_count
                if override_use_count is not None
                else row["use_count"]
            ),
        )


# ============================================================================
# Module-private utilities
# ============================================================================


def _utc_now_iso() -> str:
    return _utc_iso(datetime.now(timezone.utc))


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")


def _iso_to_epoch(iso: str) -> float:
    if not iso:
        return 0.0
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
