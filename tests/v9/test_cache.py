"""CRUCIBLE — adversarial suite for EngineCacheManager (v9/build/cache.py).

Covered failure modes / invariants (per task brief + DESIGN_V9 §4):
  * store -> lookup roundtrip preserves the artifact and metadata.
  * invalidate_on_driver_change(<different driver>) evicts prior-driver rows
    but keeps current-driver rows.
  * cross-app keys with identical inputs differ (app is first in the digest;
    cross-app collisions impossible by construction).
  * a .plan truncated / deleted mid-bytes -> lookup() returns None AND
    self-evicts the orphan row (self-heal, principle 2 — never raise).

Every cache is built with explicit cache_dir/db_path under tmp_path so tests
are hermetic and never touch the real off-OneDrive data root.
"""
from __future__ import annotations

import pytest

from v9.hardware import HardwareProfile, GPUInfo
from v9.accel.backend import BuildArtifactDescriptor, ShapeRange
from v9.build.cache import EngineCacheManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _hw(driver: str = "610.47", sm_arch: int = 89) -> HardwareProfile:
    return HardwareProfile(
        gpu=GPUInfo(name="RTX 4090", driver_version=driver, sm_arch=sm_arch,
                    vram_total_mb=24564)
    )


@pytest.fixture
def mgr(tmp_path):
    return EngineCacheManager(
        hardware=_hw(),
        cache_dir=tmp_path / "engine_cache",
        db_path=tmp_path / "engine_cache.db",
    )


def _descriptor(app: str = "comfyui", precision: str = "fp16",
                nbytes: int = 256) -> BuildArtifactDescriptor:
    return BuildArtifactDescriptor(
        backend_name="tensorrt",
        backend_version="1.0.0+trt11",
        artifact_bytes=b"\x00" * nbytes,
        artifact_format="trt_plan",
        precision=precision,
        shape_range=ShapeRange(min_shape=(1, 4, 64, 64),
                               opt_shape=(1, 4, 64, 64),
                               max_shape=(2, 4, 128, 128)),
        compile_seconds=12.5,
        notes={"app": app},
    )


def _store(mgr, *, app="comfyui", model="abc123", w=1024, h=1024, batch=1,
           precision="fp16", payload=b"PLAN-BYTES-PAYLOAD"):
    key = mgr.compute_key(app, model, w, h, batch, precision,
                          "tensorrt", "11.x")
    desc = _descriptor(app=app, precision=precision, nbytes=len(payload))
    desc.artifact_bytes = payload  # descriptor bytes are advisory; plan_bytes wins
    art = mgr.store(key, payload, desc, app=app, model_sha256=model,
                    shape_w=w, shape_h=h, batch=batch)
    return key, art


# ===========================================================================
# store -> lookup roundtrip
# ===========================================================================
def test_store_lookup_roundtrip_intact(mgr):
    payload = b"the-actual-serialized-trt-plan-bytes-0123456789"
    key, stored = _store(mgr, payload=payload)

    got = mgr.lookup(key)
    assert got is not None
    assert got.hash_key == key
    assert got.app == "comfyui"
    assert got.disk_bytes == len(payload)
    # bytes survive the disk roundtrip exactly
    assert got.read_bytes() == payload
    # descriptor metadata preserved
    assert got.descriptor.backend_name == "tensorrt"
    assert got.descriptor.precision == "fp16"
    assert got.descriptor.shape_range.max_shape == (2, 4, 128, 128)
    # lookup bumps use_count and refreshes LRU
    assert got.use_count == 1


def test_lookup_miss_returns_none(mgr):
    assert mgr.lookup("definitely-not-a-real-key") is None


def test_lookup_rejects_empty_key(mgr):
    assert mgr.lookup("") is None
    assert mgr.lookup(None) is None  # type: ignore[arg-type]


def test_store_requires_known_app(mgr):
    desc = _descriptor(app="comfyui")
    with pytest.raises(ValueError):
        mgr.store("k", b"x", desc, app="not_a_real_app")


# ===========================================================================
# invalidate_on_driver_change
# ===========================================================================
def test_invalidate_on_driver_change_evicts_prior_driver_rows(tmp_path):
    """A cache built under the OLD driver, then queried after a driver bump,
    must drop the stale-driver rows and keep current-driver rows."""
    old = EngineCacheManager(
        hardware=_hw(driver="596.36"),
        cache_dir=tmp_path / "ec",
        db_path=tmp_path / "ec.db",
    )
    # Two artifacts compiled under the OLD driver.
    k_old1, _ = _store(old, model="model-A", payload=b"A" * 32)
    k_old2, _ = _store(old, app="houdini", model="model-B", payload=b"B" * 32)
    assert old.lookup(k_old1) is not None
    assert old.lookup(k_old2) is not None

    # Now the machine reports a NEW driver. Open a manager pointing at the same
    # store but with the detected new driver, add a current-driver row.
    new = EngineCacheManager(
        hardware=_hw(driver="610.47"),
        cache_dir=tmp_path / "ec",
        db_path=tmp_path / "ec.db",
    )
    k_new, _ = _store(new, model="model-C", payload=b"C" * 32)

    # invalidate against the DETECTED (new) driver — no arg => uses HW profile.
    removed = new.invalidate_on_driver_change()
    assert removed == 2, "both old-driver rows should be evicted"

    # Old-driver rows are gone (row + file), new-driver row survives.
    assert new.lookup(k_old1) is None
    assert new.lookup(k_old2) is None
    assert new.lookup(k_new) is not None
    # the orphaned plan files were unlinked
    assert not new.artifact_path_for("comfyui", k_old1).exists()


def test_invalidate_on_driver_change_keeps_matching_driver(mgr):
    """Invalidating against the SAME driver evicts nothing."""
    k, _ = _store(mgr)
    removed = mgr.invalidate_on_driver_change(current_driver="610.47")
    assert removed == 0
    assert mgr.lookup(k) is not None


def test_invalidate_on_driver_change_explicit_arg(tmp_path):
    """Explicit current_driver arg drives eviction independent of HW profile."""
    mgr = EngineCacheManager(
        hardware=_hw(driver="610.47"),
        cache_dir=tmp_path / "ec",
        db_path=tmp_path / "ec.db",
    )
    k, _ = _store(mgr)
    # Pretend the live driver is now something else entirely.
    removed = mgr.invalidate_on_driver_change(current_driver="999.99")
    assert removed == 1
    assert mgr.lookup(k) is None


# ===========================================================================
# cross-app key isolation
# ===========================================================================
def test_cross_app_keys_differ_for_identical_inputs(mgr):
    """Same model/shape/precision/backend across two apps must produce
    DIFFERENT keys — app is first in the digest, so collisions are impossible
    by construction."""
    common = dict(model_sha256="same-model", shape_w=1024, shape_h=1024,
                  batch=1, precision="fp16", backend_name="tensorrt",
                  backend_ver="11.x")
    k_comfy = mgr.compute_key(app="comfyui", **common)
    k_houdini = mgr.compute_key(app="houdini", **common)
    k_ue5 = mgr.compute_key(app="ue5", **common)
    assert len({k_comfy, k_houdini, k_ue5}) == 3


def test_cross_app_store_does_not_collide(mgr):
    """Storing identical inputs under two apps yields two independent rows
    with distinct artifact files and bytes."""
    k1, _ = _store(mgr, app="comfyui", model="m", payload=b"COMFY")
    k2, _ = _store(mgr, app="houdini", model="m", payload=b"HOUDINI-BYTES")
    assert k1 != k2
    a1, a2 = mgr.lookup(k1), mgr.lookup(k2)
    assert a1 is not None and a2 is not None
    assert a1.read_bytes() == b"COMFY"
    assert a2.read_bytes() == b"HOUDINI-BYTES"
    assert a1.app == "comfyui" and a2.app == "houdini"


def test_key_changes_with_each_distinguishing_field(mgr):
    """Driver, sm_arch, precision, shape, model each change the key."""
    base = dict(app="comfyui", model_sha256="m", shape_w=1024, shape_h=1024,
                batch=1, precision="fp16", backend_name="tensorrt",
                backend_ver="11.x")
    k0 = mgr.compute_key(**base)
    assert mgr.compute_key(**{**base, "precision": "fp8"}) != k0
    assert mgr.compute_key(**{**base, "shape_w": 512}) != k0
    assert mgr.compute_key(**{**base, "model_sha256": "n"}) != k0
    assert mgr.compute_key(**{**base, "driver_ver": "1.0"}) != k0
    assert mgr.compute_key(**{**base, "sm_arch": 120}) != k0


# ===========================================================================
# truncated / vanished .plan -> lookup returns None and self-evicts
# ===========================================================================
def test_vanished_plan_self_heals(mgr):
    """Row exists but the .plan file was deleted out from under us -> lookup()
    returns None AND drops the orphan row (self-heal, never raise)."""
    key, _ = _store(mgr, payload=b"X" * 64)
    plan = mgr.artifact_path_for("comfyui", key)
    assert plan.exists()

    plan.unlink()  # simulate the file vanishing (sync conflict, manual delete)

    assert mgr.lookup(key) is None          # no raise, returns None
    # row self-evicted: a second lookup is still a clean miss and the row is
    # actually gone from the DB.
    assert mgr.lookup(key) is None
    assert mgr.disk_used_bytes() == 0       # row removed from metadata


def test_truncated_plan_midbytes_lookup_still_returns_artifact(mgr):
    """A plan truncated mid-bytes still EXISTS on disk. lookup() must not raise;
    it returns the artifact (the file is present). read_bytes() reflects the
    truncated content — the cache layer's job is existence + self-heal on
    vanish, not byte-integrity verification (that's the backend's deserialize).

    This test pins the CURRENT contract: a present-but-corrupt file is a hit,
    a missing file is a self-healing miss. If a future change adds checksum
    verification, update this test deliberately — do not let it silently flip.
    """
    payload = b"FULL-PLAN-BYTES-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    key, _ = _store(mgr, payload=payload)
    plan = mgr.artifact_path_for("comfyui", key)

    # Truncate the file mid-bytes.
    with open(plan, "r+b") as fh:
        fh.truncate(8)

    got = mgr.lookup(key)              # must not raise
    assert got is not None             # file still exists -> hit
    assert got.read_bytes() == payload[:8]


def test_truncated_to_zero_then_deleted_self_heals(mgr):
    """Truncate to zero bytes then delete -> the deleted file path self-heals
    to a None lookup. (Pairs with the above to cover the empty-file edge.)"""
    key, _ = _store(mgr, payload=b"PLAN")
    plan = mgr.artifact_path_for("comfyui", key)
    with open(plan, "r+b") as fh:
        fh.truncate(0)
    # zero-byte file still exists -> still a hit (no integrity check)
    got = mgr.lookup(key)
    assert got is not None
    assert got.read_bytes() == b""
    # now remove it entirely -> self-heal miss
    plan.unlink()
    assert mgr.lookup(key) is None


# ===========================================================================
# misc invariants
# ===========================================================================
def test_disk_accounting_and_invalidate_app(mgr):
    _store(mgr, app="comfyui", model="m1", payload=b"1" * 100)
    _store(mgr, app="comfyui", model="m2", payload=b"2" * 200)
    _store(mgr, app="houdini", model="m3", payload=b"3" * 300)

    assert mgr.disk_used_bytes(app="comfyui") == 300
    assert mgr.disk_used_bytes(app="houdini") == 300
    assert mgr.disk_used_bytes() == 600

    n = mgr.invalidate_app("comfyui")
    assert n == 2
    assert mgr.disk_used_bytes(app="comfyui") == 0
    assert mgr.disk_used_bytes() == 300


def test_store_lookup_by_params_roundtrip(mgr):
    """lookup_by_params recomputes the same key and finds the stored row."""
    key, _ = _store(mgr, app="ue5", model="modelX", w=1280, h=720,
                    precision="fp16", payload=b"UE5-PLAN")
    got = mgr.lookup_by_params(
        app="ue5", model_sha256="modelX", shape_w=1280, shape_h=720,
        batch=1, precision="fp16", backend_name="tensorrt", backend_ver="11.x",
    )
    assert got is not None
    assert got.hash_key == key
    assert got.read_bytes() == b"UE5-PLAN"
