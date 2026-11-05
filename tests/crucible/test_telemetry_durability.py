"""Crucible — ComfyUIStateCollector telemetry durability + dedup (Path B).

The collector now persists ring events to `data/comfyui_telemetry.db` and
restores from it on init. Three asserts:

  1. _ingest_history writes one row per unique prompt_id.
  2. Re-ingesting the same prompt_id is a no-op (no second row, no second
     ring entry) -- this fixes the pre-existing duplicate-stuffing bug
     where every poll re-appended the full /history window.
  3. A fresh collector pointing at a populated DB rebuilds its ring and
     pre-populates _seen_prompt_ids so the next /history poll doesn't
     re-insert events that already survived restart.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.experts.comfyui_state_collector import ComfyUIStateCollector
from core.comfyui_telemetry import fetch_recent_generations


def _fake_history(prompt_id: str) -> dict:
    """Synthesize one /history-style entry the parser can consume."""
    return {
        prompt_id: {
            "prompt": [
                0,
                "client",
                {
                    "1": {
                        "class_type": "CheckpointLoaderSimple",
                        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
                    },
                    "2": {
                        "class_type": "EmptyLatentImage",
                        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
                    },
                    "3": {
                        "class_type": "KSampler",
                        "inputs": {"sampler_name": "euler", "steps": 20},
                    },
                },
            ],
            "status": {"execution_time": 4.2},
        },
    }


def test_persist_writes_one_row_per_prompt(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    coll = ComfyUIStateCollector(telemetry_db_path=db)
    coll._ingest_history(_fake_history("p1"))

    rows = fetch_recent_generations(db)
    assert len(rows) == 1
    assert rows[0]["workflow_hash"] == "p1"
    assert rows[0]["shape_w"] == 1024
    assert rows[0]["shape_h"] == 1024
    assert rows[0]["sampler"] == "euler"
    assert rows[0]["steps"] == 20
    # samples_per_sec derived from steps/wall_seconds = 20/4.2 ≈ 4.76
    assert rows[0]["samples_per_sec"] == pytest.approx(20.0 / 4.2, rel=1e-3)


def test_dedup_blocks_repeated_ingest(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    coll = ComfyUIStateCollector(telemetry_db_path=db)
    coll._ingest_history(_fake_history("p1"))
    coll._ingest_history(_fake_history("p1"))
    coll._ingest_history(_fake_history("p1"))

    assert len(coll._ring) == 1
    rows = fetch_recent_generations(db)
    assert len(rows) == 1


def test_restore_rebuilds_ring_and_blocks_redups(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"

    # First collector lifetime: write two distinct prompts.
    coll1 = ComfyUIStateCollector(telemetry_db_path=db)
    coll1._ingest_history(_fake_history("p1"))
    coll1._ingest_history(_fake_history("p2"))
    assert len(coll1._ring) == 2

    # Fresh collector, same DB: ring repopulates, set is primed.
    coll2 = ComfyUIStateCollector(telemetry_db_path=db)
    assert len(coll2._ring) == 2
    assert "p1" in coll2._seen_prompt_ids
    assert "p2" in coll2._seen_prompt_ids

    # Re-ingesting the same prompts after restart must not duplicate.
    coll2._ingest_history(_fake_history("p1"))
    coll2._ingest_history(_fake_history("p2"))
    assert len(coll2._ring) == 2
    rows = fetch_recent_generations(db)
    assert len(rows) == 2
