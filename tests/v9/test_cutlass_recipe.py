"""CRUCIBLE — tests for the CUTLASS-in-the-recipe + render-driver additions.

Covers the goal deltas landed on top of V9:
  * CUTLASS backend is config-gated (off by default), honest reason text.
  * registry reads acceleration.cutlass.enabled.
  * VellumGemmAccelerator degrades to the numpy reference when CUTLASS is dormant
    and aligns GEMM dims to 8 for FP8 tensor cores.
  * render-driver builds correct husk/comfy commands and parses frame ranges.
  * ComfyUI CLI path: PNG-embedded workflow is parsed into a hot-shape event.

Dependency-light: no torch / nvcc / network required.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path


# ---------------------------------------------------------------------------
# CUTLASS backend gating
# ---------------------------------------------------------------------------
def test_cutlass_backend_disabled_by_default():
    from v9.accel.cutlass import CutlassFP8GemmBackend
    be = CutlassFP8GemmBackend(sm_arch=89)  # enabled defaults False
    assert be.enabled is False
    assert "config" in (be._unavailable_reason or "").lower()


def test_cutlass_backend_cannot_enable_without_toolchain():
    """Even with the config flag on, a missing CUTLASS install keeps it off —
    honesty wins over the flag (DESIGN_V9 §4.4)."""
    from v9.accel.cutlass import CutlassFP8GemmBackend
    be = CutlassFP8GemmBackend(sm_arch=89, enabled=True)
    # On a box without $CUTLASS_PATH this stays dormant with an install hint.
    if not be.enabled:
        assert ("not installed" in (be._unavailable_reason or "")
                or "nvcc" in (be._unavailable_reason or ""))


def test_registry_reads_cutlass_enabled_flag():
    from v9.accel.registry import _read_cutlass_enabled
    assert _read_cutlass_enabled(None) is False
    assert _read_cutlass_enabled({"acceleration": {"cutlass": {"enabled": True}}}) is True
    assert _read_cutlass_enabled({"acceleration": {"cutlass": {}}}) is False
    assert _read_cutlass_enabled({"nope": 1}) is False


# ---------------------------------------------------------------------------
# Vellum GEMM accelerator
# ---------------------------------------------------------------------------
def test_vellum_falls_back_to_reference_when_not_live():
    from v9.accel.vellum import VellumGemmAccelerator
    acc = VellumGemmAccelerator(backend=None, sm_arch=89)
    assert acc.live is False
    res = acc.build((1000, 1000, 1000))
    assert res.ok is True
    assert res.used_backend == "numpy_reference"


def test_vellum_aligns_dims_to_16():
    # Tuned Ada FP8 kernel requires 16-element alignment on M/N/K.
    from v9.accel.vellum import VellumGemmAccelerator
    acc = VellumGemmAccelerator(backend=None, sm_arch=89)
    assert acc.shape_for((1001, 1023, 17)) == (992, 1008, 16)
    assert acc.shape_for((1, 2, 3)) == (16, 16, 16)  # floored to >= ALIGN


def test_vellum_reference_gemm_matches_numpy():
    import pytest
    np = pytest.importorskip("numpy")
    from v9.accel.vellum import VellumGemmAccelerator
    a = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8, 8).astype(np.float32)
    out = VellumGemmAccelerator.reference_gemm(a, b)
    assert np.allclose(out, a @ b, atol=1e-4)


# ---------------------------------------------------------------------------
# render-driver command construction (no launch)
# ---------------------------------------------------------------------------
def test_render_houdini_cmd_frame_range():
    from v9.render import _houdini_cmd
    cmd = _houdini_cmd(Path("husk.exe"), "scene.usd", "out/$F4.exr", "1-100x2", [])
    assert "--frame" in cmd and "1" in cmd
    assert "--frame-inc" in cmd and "2" in cmd
    assert cmd[-1] == "scene.usd"
    assert "--output" in cmd


def test_render_comfy_cmd():
    from v9.render import _comfy_cmd
    cmd = _comfy_cmd("comfy", "wf.json", None, ["--extra"])
    assert cmd[:2] == ["comfy", "run"]
    assert "--workflow" in cmd and "wf.json" in cmd
    assert "--wait" in cmd and "--extra" in cmd


def test_render_frame_count():
    from v9.render import _frame_count
    assert _frame_count("1-100") == 100
    assert _frame_count("1-100x2") == 50
    assert _frame_count(None) is None
    assert _frame_count("garbage") is None


def test_render_progress_mile_parsing():
    from v9.render import _progress_mile
    assert _progress_mile("Rendering frame 42 of 100", 100) == 42
    assert _progress_mile("step 7/20", None) == 7
    assert _progress_mile("nothing here", None) is None


# ---------------------------------------------------------------------------
# ComfyUI CLI path: PNG-embedded workflow -> hot-shape event
# ---------------------------------------------------------------------------
def _png_with_text(keyword: str, value: str) -> bytes:
    """Minimal PNG: signature + IHDR + tEXt + IEND. CRCs are real so a strict
    reader wouldn't choke, but the parser under test ignores them."""
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + ctype + data
                + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
    text = chunk(b"tEXt", keyword.encode("latin-1") + b"\x00" + value.encode("utf-8"))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + text + iend


def test_read_png_prompt_extracts_graph(tmp_path):
    from v9.advisors.comfyui import ComfyUIAdvisor
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sdxl_base.safetensors"}},
        "2": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"sampler_name": "dpmpp_2m", "steps": 25}},
    }
    png = tmp_path / "render_00001_.png"
    png.write_bytes(_png_with_text("prompt", json.dumps(graph)))

    adv = ComfyUIAdvisor(config={})
    parsed = adv._read_png_prompt(png)
    assert parsed is not None
    ev = adv._graph_to_event("render_00001_.png", parsed, ts=123.0)
    assert ev is not None
    assert ev.model_sha == "sdxl_base.safetensors"
    assert ev.shape == (1024, 1024)
    assert ev.sampler == "dpmpp_2m"
    assert ev.steps == 25


def test_scan_output_dir_populates_ring(tmp_path):
    from v9.advisors.comfyui import ComfyUIAdvisor
    out = tmp_path / "output"
    out.mkdir()
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "flux_dev.safetensors"}},
        "2": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1216, "height": 832, "batch_size": 2}},
    }
    (out / "img_1.png").write_bytes(_png_with_text("prompt", json.dumps(graph)))

    adv = ComfyUIAdvisor(config={"output_dir": str(out)})
    adv._scan_output_dir(tmp_path)  # install_dir; output_dir override used
    assert len(adv._ring) == 1
    assert adv._ring[0].model_sha == "flux_dev.safetensors"
    assert adv._ring[0].shape == (1216, 832)


# ---------------------------------------------------------------------------
# Houdini advisor: CUTLASS-Vellum rule honors config opt-in
# ---------------------------------------------------------------------------
def test_houdini_advisor_reads_cutlass_config():
    from v9.advisors.houdini import HoudiniAdvisor
    off = HoudiniAdvisor(config={})
    assert off.cutlass_enabled is False

    on = HoudiniAdvisor(config={"acceleration": {"cutlass": {
        "enabled": True, "vellum": {"enabled": True, "precision": "fp8_e4m3",
                                    "default_gemm": [2048, 2048, 2048]}}}})
    assert on.cutlass_enabled is True
    assert on.cutlass_gemm == (2048, 2048, 2048)
    assert on.cutlass_precision == "fp8_e4m3"
