"""Crucible C1.3 — V8 specialist verification.

Each of 4 specialists exercised through the contract documented in
`briefs/_handoff_forge2.md`. Tests are normative and adversarial: they assert
specific Recommendation shapes / counts so a future Forge can't silently weaken
them without breaking these tests.

All specialists must:
  * return [] when their target app is not running
  * never raise out of analyze() even when downstream collectors / handles
    misbehave
  * emit recommendations that match the Section A/B/C/D decision rules in
    `briefs/forge-2_specialists.md` (and the contract table in
    `_handoff_forge2.md`)

These tests construct synthetic state objects directly and pass them through
SystemContext attribute-setting; they do NOT spin up the real collectors. That
is the right scope: collector behavior is a separate concern (failure-injection
tests cover collector graceful-degrade).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Importing v8 first triggers the enum extension so V8 enum members exist
# for the resolution helpers inside the specialists. (The fallback path also
# works on a fresh V7 import; we're exercising the production wiring.)
import core.gpu_optimizer_v8  # noqa: F401  -- side-effect: extends enums

from core.gpu_optimizer_v6 import (
    OptimizationAction,
    Recommendation,
    SystemContext,
    WorkflowMode,
)
from core.experts.comfyui_specialist import ComfyUISpecialist
from core.experts.comfyui_state_collector import ComfyUIState, WorkloadEvent
from core.experts.houdini_specialist import HoudiniSpecialist
from core.experts.houdini_state_collector import HoudiniState
from core.experts.ue5_specialist import UE5Specialist
from core.experts.ue5_state_collector import UE5State
from core.experts.cross_app_coordinator import (
    CrossAppCoordinator,
    CrossAppState,
    UsageHistoryEntry,
    vram_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx() -> SystemContext:
    """Minimal V7 SystemContext; V8 fields are attached as attributes."""
    return SystemContext(
        timestamp=datetime.now(),
        workflow_mode=WorkflowMode.IDLE,
        gpu_utilization_percent=0.0,
        vram_used_gb=0.0,
        vram_free_gb=24.0,
        gpu_temp_c=0.0,
        gpu_power_w=0.0,
        gpu_clock_mhz=0,
        active_processes=[],
        process_vram={},
    )


def _comfy_cfg() -> Dict[str, Any]:
    return {
        "min_confidence": 0.50,  # accept all rec confidences in tests
        "tensorrt": {"version": "11.x", "cache_size_gb": 50.0},
        "phase_b_enabled": False,
    }


def _houd_cfg() -> Dict[str, Any]:
    return {
        "min_confidence": 0.50,
        "tensorrt": {"version": "11.x"},
        "phase_b_enabled": False,
    }


def _ue5_cfg() -> Dict[str, Any]:
    return {
        "min_confidence": 0.50,
        "ue5": {"nim_marketplace_url": "https://example.invalid/nim"},
    }


def _cross_cfg() -> Dict[str, Any]:
    return {
        "min_confidence": 0.50,
        "cross_app": {"idle_window_seconds": 60.0},
    }


# ===========================================================================
# ComfyUISpecialist
# ===========================================================================

class TestComfyUISpecialist:
    def test_process_not_running_returns_empty(self) -> None:
        ctx = _ctx()
        ctx.comfyui = ComfyUIState(process_running=False)
        spec = ComfyUISpecialist(_comfy_cfg())
        assert spec.analyze(ctx) == []

    def test_api_unreachable_does_not_raise(self) -> None:
        """api_reachable=False is the documented degraded mode (collector
        sets it after 3 consecutive HTTP failures). The specialist must
        still produce recommendations from filesystem-derived state without
        raising."""
        ctx = _ctx()
        ctx.comfyui = ComfyUIState(
            process_running=True,
            api_reachable=False,             # degraded
            install_dir=Path("C:/comfyui"),
            custom_nodes_installed=[],       # neither node installed
            launch_flags=[],
        )
        spec = ComfyUISpecialist(_comfy_cfg())
        # Must not raise; should emit R1+R2 (both nodes missing).
        recs = spec.analyze(ctx)
        targets = {r.target for r in recs}
        assert "comfyui:sage_attention" in targets
        assert "comfyui:tensorrt" in targets

    def test_all_nodes_installed_no_flag_emits_set_launch_flag_only(self) -> None:
        ctx = _ctx()
        ctx.comfyui = ComfyUIState(
            process_running=True,
            api_reachable=True,
            install_dir=Path("C:/comfyui"),
            custom_nodes_installed=["ComfyUI-SageAttention", "ComfyUI_TensorRT"],
            launch_flags=[],  # no --use-sage-attention
        )
        spec = ComfyUISpecialist(_comfy_cfg())
        recs = spec.analyze(ctx)
        # R1 (install) suppressed, R2 (install) suppressed; only the
        # SET_LAUNCH_FLAG branch should fire. R3 needs hot workloads -- ring
        # buffer is empty so no compile rec. R5 needs disk usage -- 0.
        assert len(recs) == 1
        rec = recs[0]
        assert rec.action == OptimizationAction.SET_LAUNCH_FLAG
        assert rec.target == "comfyui:launch"
        assert rec.parameters.get("add_flag") == "--use-sage-attention"


# ===========================================================================
# HoudiniSpecialist
# ===========================================================================

class TestHoudiniSpecialist:
    def test_process_not_running_returns_empty(self) -> None:
        ctx = _ctx()
        ctx.houdini_v8 = HoudiniState(process_running=False)
        spec = HoudiniSpecialist(_houd_cfg())
        assert spec.analyze(ctx) == []

    def test_ml_sop_onnx_missing_no_raise_no_r1(self, tmp_path: Path) -> None:
        """ONNX path that doesn't exist on disk: brief Section B says log+skip,
        no blocker. R1 must not emit for that ref and analyze must not raise."""
        missing = tmp_path / "definitely_not_here.onnx"
        assert not missing.exists()
        ctx = _ctx()
        ctx.houdini_v8 = HoudiniState(
            process_running=True,
            ml_sop_onnx_paths=[missing],     # missing on disk
            karma_xpu_active=False,          # suppress R2
            tops_active=False,               # suppress R3
            vellum_solver_active=False,      # suppress R4
        )
        spec = HoudiniSpecialist(_houd_cfg())
        recs = spec.analyze(ctx)              # must not raise
        # No R1 emitted for the missing onnx ref; no other rules fire.
        assert recs == []

    def test_karma_xpu_inactive_no_r2(self) -> None:
        ctx = _ctx()
        ctx.houdini_v8 = HoudiniState(
            process_running=True,
            karma_xpu_active=False,           # XPU inactive
            optix_denoiser_enabled=False,     # would otherwise trigger R2
        )
        spec = HoudiniSpecialist(_houd_cfg())
        recs = spec.analyze(ctx)
        # R2 must NOT emit when Karma XPU is inactive even though the
        # denoiser env var would otherwise be a candidate.
        assert all(r.target != "houdini:karma:denoiser" for r in recs)


# ===========================================================================
# UE5Specialist
# ===========================================================================

class TestUE5Specialist:
    def test_process_not_running_returns_empty(self) -> None:
        ctx = _ctx()
        ctx.ue5_v8 = UE5State(process_running=False)
        spec = UE5Specialist(_ue5_cfg())
        assert spec.analyze(ctx) == []

    def test_nim_marketplace_url_missing_emits_notification_only(self) -> None:
        """When nim_marketplace_url is not configured, R1 should still emit
        but as a notification-only payload (auto_install=False); no install
        attempt is made."""
        cfg = {"min_confidence": 0.50, "ue5": {}}  # no marketplace_url -> default
        ctx = _ctx()
        ctx.ue5_v8 = UE5State(
            process_running=True,
            nim_plugin_installed=False,       # triggers R1
            dlss_dll_version=None,            # suppress R2
            path_tracer_in_use=False,         # suppress R3
            cvars={},
        )
        spec = UE5Specialist(cfg)
        recs = spec.analyze(ctx)
        nim_recs = [r for r in recs if r.target == "ue5:nim"]
        assert len(nim_recs) == 1
        rec = nim_recs[0]
        # auto_install=False is the contract: notification only.
        assert rec.parameters.get("auto_install") is False
        assert rec.action == OptimizationAction.INSTALL_PLUGIN
        # Notification text + marketplace url present.
        assert "notification" in rec.parameters
        assert "marketplace_url" in rec.parameters

    def test_dlss_dll_missing_no_r2(self) -> None:
        """dlss_dll_version=None means we have no evidence. R2 must not emit
        speculatively (the version-comparison helper returns False on None)."""
        ctx = _ctx()
        ctx.ue5_v8 = UE5State(
            process_running=True,
            nim_plugin_installed=True,        # suppress R1
            dlss_dll_version=None,            # missing DLL
            path_tracer_in_use=False,         # suppress R3
            cvars={},
        )
        spec = UE5Specialist(_ue5_cfg())
        recs = spec.analyze(ctx)
        assert all(r.target != "ue5:dlss" for r in recs)


# ===========================================================================
# CrossAppCoordinator
# ===========================================================================

class TestCrossAppCoordinator:
    def test_three_apps_active_vram_sum_22gb(self) -> None:
        """vram_budget for 3 apps must total 22 GB (1 GB headroom in the 24 GB
        physical card budget). Document & forge-2 contract: 11/8/3."""
        budget = vram_budget(["comfyui", "houdini", "ue5"])
        assert sum(budget.values()) == pytest.approx(22.0)
        # Spot-check the per-app split per Forge-2 brief.
        assert budget["comfyui"] == 11.0
        assert budget["houdini"] == 8.0
        assert budget["ue5"] == 3.0

    def test_foreground_pauses_background_compiles(self) -> None:
        """foreground=comfyui, houdini has a background compile in flight =>
        coordinator must emit a 'pause background compiles for houdini' rec."""
        ctx = _ctx()
        ctx.cross_app = CrossAppState(
            foreground_pid_app="comfyui",
            app_running={"comfyui": True, "houdini": True, "ue5": False},
            compile_jobs_active={"comfyui": 0, "houdini": 1, "ue5": 0},
            idle_dwell_seconds=0.0,
            usage_history=deque(),
        )
        coord = CrossAppCoordinator(_cross_cfg())
        recs = coord.analyze(ctx)
        pause_recs = [r for r in recs if r.target == "compiler:pause:houdini"]
        assert len(pause_recs) == 1
        rec = pause_recs[0]
        assert rec.parameters.get("action") == "pause_background_compiles"
        assert rec.parameters.get("app") == "houdini"
        assert rec.parameters.get("foreground_app") == "comfyui"

    def test_idle_with_empty_history_no_r3_prefetch(self) -> None:
        """All apps idle, history empty: predict_next_engine returns None,
        so R3 must not emit a COMPILE_TRT_ENGINE prefetch."""
        ctx = _ctx()
        ctx.cross_app = CrossAppState(
            foreground_pid_app="none",
            app_running={"comfyui": False, "houdini": False, "ue5": False},
            compile_jobs_active={"comfyui": 0, "houdini": 0, "ue5": 0},
            idle_dwell_seconds=600.0,         # well past idle window
            usage_history=deque(),            # empty history
        )
        coord = CrossAppCoordinator(_cross_cfg())
        recs = coord.analyze(ctx)
        # No compile-prefetch recommendation when history is empty.
        assert all(
            not r.target.startswith("prefetch:") for r in recs
        ), "R3 must not emit prefetch when history is empty"
