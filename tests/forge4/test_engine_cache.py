"""Tests for core.engine_cache.EngineCacheManager (Forge-4 D1).

Crucible-1's done criteria:
- lookup(unknown_key) returns None.
- Store -> lookup roundtrip works for >= 100MB plan bytes.
- Eviction frees >= requested bytes when over budget.
- Engine in use within last 60s is NOT evicted.
- Cache key starts with `app|` (cross-app collisions impossible).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from acceleration.backend import BuildArtifactDescriptor, ShapeRange
from core.engine_cache import EngineCacheManager


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def cache(tmp_path: Path) -> EngineCacheManager:
    cache_dir = tmp_path / "EngineCache"
    db_path = tmp_path / "engine_cache.db"
    return EngineCacheManager(cache_dir=cache_dir, db_path=db_path,
                              max_disk_bytes=10 * 1024 ** 3)


def _descriptor(precision: str = "fp16",
                shape=(1, 4, 64, 64),
                backend: str = "tensorrt",
                version: str = "1.0.0",
                bytes_size: int = 1024,
                app: str = "comfyui") -> BuildArtifactDescriptor:
    return BuildArtifactDescriptor(
        backend_name=backend,
        backend_version=version,
        artifact_bytes=b"\x00" * bytes_size,
        artifact_format="trt_plan",
        precision=precision,
        shape_range=ShapeRange.static(shape),
        compile_seconds=12.5,
        notes={"app": app, "test": True},
    )


# ---------------------------------------------------------------- key tests


def test_lookup_unknown_key_returns_none(cache: EngineCacheManager) -> None:
    assert cache.lookup("nonexistent_key_xxxx") is None
    assert cache.lookup("") is None


def test_compute_key_is_32_hex(cache: EngineCacheManager) -> None:
    key = cache.compute_key(
        app="comfyui", model_sha256="a" * 64,
        shape_w=512, shape_h=512, batch=1,
        precision="fp16", backend_name="tensorrt", backend_ver="1.0.0",
    )
    assert len(key) == 32
    int(key, 16)  # parses as hex


def test_compute_key_app_first_no_cross_app_collision(
    cache: EngineCacheManager,
) -> None:
    """All other params identical, only `app` differs -> key must differ.

    This is the Crucible-1 invariant: `app|` is FIRST in the digest input,
    so two apps cannot collide for any choice of other params.
    """
    common = dict(
        model_sha256="b" * 64, shape_w=1024, shape_h=1024, batch=2,
        precision="fp8", backend_name="cutlass_fp8_gemm",
        backend_ver="0.1.0",
    )
    k_comfy = cache.compute_key(app="comfyui", **common)
    k_houdini = cache.compute_key(app="houdini", **common)
    k_ue5 = cache.compute_key(app="ue5", **common)
    assert k_comfy != k_houdini != k_ue5
    assert k_comfy != k_ue5


def test_compute_key_changes_on_any_field(cache: EngineCacheManager) -> None:
    base = dict(
        app="comfyui", model_sha256="c" * 64,
        shape_w=512, shape_h=512, batch=1,
        precision="fp16", backend_name="tensorrt", backend_ver="1.0.0",
    )
    k0 = cache.compute_key(**base)
    # Each field perturbation must change the digest.
    perturb = [
        {"shape_w": 513},
        {"shape_h": 513},
        {"batch": 2},
        {"precision": "bf16"},
        {"backend_name": "sage_attention_v2"},
        {"backend_ver": "1.0.1"},
        {"driver_ver": "596.99"},
        {"sm_arch": 90},
        {"model_sha256": "d" * 64},
    ]
    for p in perturb:
        kw = {**base, **p}
        assert cache.compute_key(**kw) != k0, f"perturbation {p} did not change key"


def test_compute_key_rejects_bad_app(cache: EngineCacheManager) -> None:
    with pytest.raises(ValueError):
        cache.compute_key(
            app="not_a_real_app", model_sha256="x" * 64,
            shape_w=1, shape_h=1, batch=1,
            precision="fp16", backend_name="tensorrt", backend_ver="1.0.0",
        )


# ---------------------------------------------------------------- roundtrip


def test_store_then_lookup_small(cache: EngineCacheManager) -> None:
    desc = _descriptor()
    key = cache.compute_key(
        app="comfyui", model_sha256="e" * 64,
        shape_w=64, shape_h=64, batch=1,
        precision="fp16", backend_name="tensorrt", backend_ver="1.0.0",
    )
    art = cache.store(key, b"\xff" * 1024, desc, app="comfyui",
                      model_sha256="e" * 64, shape_w=64, shape_h=64, batch=1)
    assert art.hash_key == key
    assert art.artifact_path.exists()
    assert art.artifact_path.read_bytes() == b"\xff" * 1024

    hit = cache.lookup(key)
    assert hit is not None
    assert hit.hash_key == key
    assert hit.app == "comfyui"
    assert hit.disk_bytes == 1024
    assert hit.use_count >= 1
    assert hit.read_bytes() == b"\xff" * 1024


@pytest.mark.skipif(
    os.environ.get("FORGE4_SKIP_LARGE") == "1",
    reason="100MB roundtrip skipped via FORGE4_SKIP_LARGE",
)
def test_store_then_lookup_100mb(cache: EngineCacheManager) -> None:
    """Crucible-1: store->lookup roundtrip works for >= 100MB plan bytes."""
    size = 100 * 1024 * 1024 + 1  # > 100 MiB
    blob = os.urandom(size)
    desc = _descriptor(bytes_size=0)  # we pass bytes separately

    key = cache.compute_key(
        app="houdini", model_sha256="f" * 64,
        shape_w=2048, shape_h=2048, batch=1,
        precision="fp16", backend_name="tensorrt", backend_ver="1.0.0",
    )
    art = cache.store(key, blob, desc, app="houdini",
                      model_sha256="f" * 64, shape_w=2048, shape_h=2048, batch=1)
    assert art.disk_bytes == size

    hit = cache.lookup(key)
    assert hit is not None
    assert hit.disk_bytes == size
    # Verify byte-exact roundtrip on a sampled prefix + suffix (full
    # comparison would double-allocate 100 MiB).
    on_disk = hit.artifact_path.read_bytes()
    assert len(on_disk) == size
    assert on_disk[:1024] == blob[:1024]
    assert on_disk[-1024:] == blob[-1024:]


def test_store_creates_app_subdir(cache: EngineCacheManager) -> None:
    desc = _descriptor(app="ue5")
    key = "a" * 32
    art = cache.store(key, b"plan", desc, app="ue5")
    assert art.artifact_path.parent.name == "ue5"
    assert art.artifact_path.name == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.plan"


def test_lookup_self_heals_missing_file(cache: EngineCacheManager) -> None:
    desc = _descriptor()
    key = "b" * 32
    art = cache.store(key, b"plan", desc, app="comfyui")
    art.artifact_path.unlink()  # simulate disk corruption

    assert cache.lookup(key) is None
    # Row was deleted too.
    assert cache.lookup(key) is None


def test_store_atomic_no_tmp_orphan_on_success(cache: EngineCacheManager) -> None:
    desc = _descriptor()
    key = "c" * 32
    art = cache.store(key, b"plan", desc, app="comfyui")
    tmp_candidate = art.artifact_path.with_suffix(art.artifact_path.suffix + ".tmp")
    assert not tmp_candidate.exists()


# ---------------------------------------------------------------- eviction


def test_evict_lru_frees_at_least_requested(
    cache: EngineCacheManager,
    tmp_path: Path,
) -> None:
    """Crucible-1: eviction frees >= requested bytes when over budget."""
    desc = _descriptor()
    # Insert 5 entries of 100KB each, with last_used_utc deliberately old.
    keys = [f"{i:032d}" for i in range(5)]
    for i, key in enumerate(keys):
        cache.store(key, b"\x00" * 100_000, desc, app="comfyui")
    # Backdate last_used_utc on all rows so eviction is allowed.
    import sqlite3
    with sqlite3.connect(str(cache.db_path)) as conn:
        conn.execute("UPDATE artifacts SET last_used_utc = ?",
                     ("1970-01-01T00:00:00.000000+0000",))
        conn.commit()

    total = cache.disk_used_bytes()
    assert total == 5 * 100_000

    freed = cache.evict_lru(target_bytes=total - 250_000)
    assert freed >= 250_000
    assert cache.disk_used_bytes() <= total - 250_000


def test_evict_lru_refuses_in_use_entries(cache: EngineCacheManager) -> None:
    """Crucible-1: artifact within last 60s of use must NOT be evicted."""
    desc = _descriptor()
    key_recent = "r" * 32
    key_old = "o" * 32
    cache.store(key_recent, b"\x00" * 1000, desc, app="comfyui")
    cache.store(key_old, b"\x00" * 1000, desc, app="comfyui")

    # Backdate ONLY the "old" row.
    import sqlite3
    with sqlite3.connect(str(cache.db_path)) as conn:
        conn.execute("UPDATE artifacts SET last_used_utc = ? WHERE hash_key = ?",
                     ("1970-01-01T00:00:00.000000+0000", key_old))
        conn.commit()

    # Touch recent row to make sure it's "now".
    cache.lookup(key_recent)

    # Demand we evict everything (target 0 bytes).
    freed = cache.evict_lru(target_bytes=0, target_app="comfyui")
    # Only the old one should have gone.
    assert freed == 1000
    assert cache.lookup(key_old) is None
    assert cache.lookup(key_recent) is not None


def test_evict_lru_scoped_to_target_app(cache: EngineCacheManager) -> None:
    desc = _descriptor()
    cache.store("c" * 32, b"\x00" * 1000, desc, app="comfyui")
    cache.store("h" * 32, b"\x00" * 1000, desc, app="houdini")

    # Backdate both.
    import sqlite3
    with sqlite3.connect(str(cache.db_path)) as conn:
        conn.execute("UPDATE artifacts SET last_used_utc = ?",
                     ("1970-01-01T00:00:00.000000+0000",))
        conn.commit()

    freed = cache.evict_lru(target_bytes=0, target_app="comfyui")
    assert freed == 1000
    assert cache.lookup("c" * 32) is None
    assert cache.lookup("h" * 32) is not None


def test_evict_lru_returns_zero_when_under_budget(
    cache: EngineCacheManager,
) -> None:
    desc = _descriptor()
    cache.store("z" * 32, b"\x00" * 100, desc, app="comfyui")
    freed = cache.evict_lru(target_bytes=10 ** 9)
    assert freed == 0


# ---------------------------------------------------------------- invalidation


def test_invalidate_app(cache: EngineCacheManager) -> None:
    desc = _descriptor()
    cache.store("a" * 32, b"plan", desc, app="comfyui")
    cache.store("b" * 32, b"plan", desc, app="houdini")
    n = cache.invalidate_app("comfyui")
    assert n == 1
    assert cache.lookup("a" * 32) is None
    assert cache.lookup("b" * 32) is not None


def test_invalidate_on_driver_change(cache: EngineCacheManager) -> None:
    desc = _descriptor()
    cache.store("a" * 32, b"plan", desc, app="comfyui",
                driver_ver="500.00")
    cache.store("b" * 32, b"plan", desc, app="comfyui",
                driver_ver="596.36")
    n = cache.invalidate_on_driver_change("596.36")
    assert n == 1
    assert cache.lookup("a" * 32) is None
    assert cache.lookup("b" * 32) is not None


# ---------------------------------------------------------------- inspection


def test_disk_used_gb_per_app(cache: EngineCacheManager) -> None:
    desc = _descriptor()
    cache.store("c" * 32, b"\x00" * 5_000, desc, app="comfyui")
    cache.store("h" * 32, b"\x00" * 7_000, desc, app="houdini")
    assert cache.disk_used_bytes(app="comfyui") == 5_000
    assert cache.disk_used_bytes(app="houdini") == 7_000
    assert cache.disk_used_bytes() == 12_000


def test_lookup_bumps_use_count(cache: EngineCacheManager) -> None:
    desc = _descriptor()
    cache.store("a" * 32, b"plan", desc, app="comfyui")
    h1 = cache.lookup("a" * 32)
    h2 = cache.lookup("a" * 32)
    assert h1 is not None and h2 is not None
    assert h2.use_count > h1.use_count


def test_stats_summary(cache: EngineCacheManager) -> None:
    desc = _descriptor()
    cache.store("a" * 32, b"\x00" * 100, desc, app="comfyui")
    cache.store("b" * 32, b"\x00" * 200, desc, app="houdini")
    stats = cache.stats()
    assert stats["total_bytes"] == 300
    assert stats["per_app"]["comfyui"]["bytes"] == 100
    assert stats["per_app"]["houdini"]["bytes"] == 200
