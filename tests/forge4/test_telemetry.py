"""Tests for core.comfyui_telemetry (Forge-4 D3)."""

from __future__ import annotations

from pathlib import Path

from core.comfyui_telemetry import (
    ensure_schema,
    fetch_recent_compiles,
    fetch_recent_generations,
    record_compile,
    record_generation,
)


def test_record_generation_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    rid = record_generation(
        db_path=db,
        ts_utc="2026-05-06T10:00:00.000000+0000",
        workflow_hash="wf_abc",
        model_sha256="m" * 64,
        shape_w=1024, shape_h=1024, batch=1,
        sampler="euler", steps=20,
        precision="fp16",
        backend_set=["sage_attn", "trt_unet"],
        wall_seconds=4.5,
        samples_per_sec=4.4,
        vram_peak_gb=14.2,
    )
    assert rid > 0
    rows = fetch_recent_generations(db, limit=10)
    assert len(rows) == 1
    assert rows[0]["sampler"] == "euler"
    assert rows[0]["backend_set"] == "sage_attn,trt_unet"


def test_record_compile_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    rid = record_compile(
        db_path=db,
        ts_utc="2026-05-06T10:00:00.000000+0000",
        app="comfyui",
        backend_name="tensorrt",
        backend_version="1.0.0",
        model_sha256="m" * 64,
        shape_w=1024, shape_h=1024, batch=1,
        precision="fp16",
        compile_seconds=42.0,
        artifact_bytes=12345678,
        notes={"ok": True, "kind": "unet"},
    )
    assert rid > 0
    rows = fetch_recent_compiles(db, limit=10)
    assert len(rows) == 1
    assert rows[0]["app"] == "comfyui"
    assert rows[0]["compile_seconds"] == 42.0


def test_ensure_schema_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    ensure_schema(db)
    ensure_schema(db)
    record_generation(
        db_path=db, ts_utc="t", workflow_hash=None, model_sha256=None,
        shape_w=0, shape_h=0, batch=0, sampler=None, steps=None,
        precision=None, backend_set=[], wall_seconds=0.0,
        samples_per_sec=0.0, vram_peak_gb=0.0,
    )
    assert len(fetch_recent_generations(db)) == 1


def test_filter_by_app(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    record_compile(
        db, "t", "comfyui", "tensorrt", "1.0", "x" * 64,
        1, 1, 1, "fp16", 1.0, 100, {},
    )
    record_compile(
        db, "t", "houdini", "tensorrt", "1.0", "x" * 64,
        1, 1, 1, "fp16", 2.0, 200, {},
    )
    cu = fetch_recent_compiles(db, app="comfyui")
    assert len(cu) == 1
    assert cu[0]["app"] == "comfyui"
