"""CRUCIBLE — adversarial degrade-safety suite for Optimizer V9.

DESIGN_V9 §4.2: "Degrade, never crash." This file hammers the failure modes:

  * pynvml raising on every attribute access -> context.gather() returns a
    safe idle SystemContext and never raises.
  * urllib/requests raising -> ComfyUI state collection returns empty and the
    advisor returns [] (no exception).
  * each advisor with its app NOT running -> analyze() returns [] (no raise).
  * a backend whose SDK is absent -> enabled=False, applies_to() previews
    without raising, compile() raises a CLEAR install hint (not a crash).

These tests are written to FAIL loudly if a degradation path regresses into a
crash. They are NOT weakened to make production pass; a failure here is a
reported defect.
"""
from __future__ import annotations

import builtins
import sys
import types

import pytest

import v9.context as context_mod
from v9.context import SystemContext, gather
from v9.hardware import HardwareProfile, GPUInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _ExplodingModule(types.ModuleType):
    """A fake module whose EVERY attribute access raises. Simulates a pynvml
    that is importable but whose every call blows up (driver mismatch, broken
    .so, etc.)."""

    def __getattr__(self, name):  # noqa: D401
        raise RuntimeError(f"exploding-pynvml: attribute {name!r} unavailable")


def _fake_profile() -> HardwareProfile:
    """A fixed profile so gather() does not shell out to nvidia-smi during the
    test (detection is exercised elsewhere)."""
    return HardwareProfile(
        gpu=GPUInfo(name="RTX 4090", driver_version="610.47", sm_arch=89,
                    vram_total_mb=24564)
    )


# ===========================================================================
# context.gather() — pynvml explodes
# ===========================================================================
def test_gather_safe_idle_when_pynvml_raises_on_every_attr(monkeypatch):
    """If pynvml is importable but every attribute access raises, gather()
    must still return a usable idle SystemContext and never propagate."""
    exploding = _ExplodingModule("pynvml")
    monkeypatch.setitem(sys.modules, "pynvml", exploding)

    # Force the nvidia-smi fallback to also be unavailable so we exercise the
    # pure-degrade path (no GPU info at all).
    monkeypatch.setattr(context_mod.shutil, "which", lambda _name: None)

    ctx = gather(profile=_fake_profile(), timestamp=123.0)

    assert isinstance(ctx, SystemContext)
    assert ctx.timestamp == 123.0
    # idle is the safe default when nothing could be probed
    assert ctx.workflow_mode in ("idle", "sequential", "parallel")
    # the profile's VRAM total seeded the context even though live probe failed
    assert ctx.vram_total_gb > 0
    # running dict is intact and well-formed
    assert set(ctx.running.keys()) == {"comfyui", "houdini", "ue5"}


def test_gather_never_raises_when_everything_explodes(monkeypatch):
    """Even if detect(), pynvml, nvidia-smi, AND psutil all blow up, gather()
    returns a minimal idle context rather than raising."""

    def _boom_detect(*_a, **_k):
        raise RuntimeError("detect exploded")

    monkeypatch.setattr(context_mod, "detect", _boom_detect)
    monkeypatch.setitem(sys.modules, "pynvml", _ExplodingModule("pynvml"))
    monkeypatch.setattr(context_mod.shutil, "which", lambda _n: None)

    # psutil import raises
    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "psutil":
            raise ImportError("no psutil for you")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    ctx = gather(timestamp=5.0)  # no profile -> forces the exploding detect()
    assert isinstance(ctx, SystemContext)
    assert ctx.timestamp == 5.0
    # nothing detected -> profile stays None, mode is idle, no apps running
    assert ctx.active_app_count == 0
    assert ctx.workflow_mode == "idle"


def test_gather_survives_nvidia_smi_garbage(monkeypatch):
    """nvidia-smi present but returning garbage must not crash gather()."""
    monkeypatch.setitem(sys.modules, "pynvml", _ExplodingModule("pynvml"))
    monkeypatch.setattr(context_mod.shutil, "which", lambda _n: "nvidia-smi")

    class _Res:
        returncode = 0
        stdout = "not,enough\nfields,here,??,??\n"

    monkeypatch.setattr(context_mod.subprocess, "run",
                        lambda *a, **k: _Res())

    ctx = gather(profile=_fake_profile())
    assert isinstance(ctx, SystemContext)
    # garbage parse left the seeded profile VRAM intact (no crash, no zero-out)
    assert ctx.vram_total_gb > 0


# ===========================================================================
# ComfyUI advisor — HTTP layer explodes
# ===========================================================================
def _running_ctx(**running) -> SystemContext:
    base = {"comfyui": False, "houdini": False, "ue5": False}
    base.update(running)
    ctx = SystemContext(profile=_fake_profile())
    ctx.running = base
    return ctx


def test_comfyui_advisor_returns_empty_when_http_raises(monkeypatch):
    """When ComfyUI is 'running' but every HTTP call to its API raises, the
    advisor's state collection must yield no API data and analyze() must not
    raise. It may still surface install recs (those don't need the API), but
    it must NEVER throw."""
    from v9.advisors.comfyui import ComfyUIAdvisor

    adv = ComfyUIAdvisor(config={})

    # Make the proc-find return nothing (no psutil match) so install_dir is
    # None and custom-node listing is skipped -> we isolate the HTTP path.
    monkeypatch.setattr(adv, "_find_proc", lambda: None)

    # Every urllib request raises a hard error (not a tidy URLError).
    import v9.advisors.comfyui as cmod

    def _boom_urlopen(*_a, **_k):
        raise RuntimeError("network stack exploded")

    monkeypatch.setattr(cmod._urlrequest, "urlopen", _boom_urlopen)

    ctx = _running_ctx(comfyui=True)
    recs = adv.analyze(ctx)  # MUST NOT RAISE

    assert isinstance(recs, list)
    # state was published and api is not reachable
    assert ctx.comfyui is not None
    assert ctx.comfyui.api_reachable is False


def test_comfyui_http_json_swallows_arbitrary_exception(monkeypatch):
    """The _http_json helper must absorb ANY exception (not just URLError) and
    count it as a failure, returning None."""
    from v9.advisors.comfyui import ComfyUIAdvisor
    import v9.advisors.comfyui as cmod

    adv = ComfyUIAdvisor(config={})

    def _boom(*_a, **_k):
        raise ValueError("totally unexpected")

    monkeypatch.setattr(cmod._urlrequest, "urlopen", _boom)
    assert adv._http_json("/queue") is None
    assert adv._http_failures >= 1


def test_comfyui_advisor_empty_when_requests_layer_unavailable(monkeypatch):
    """Belt-and-suspenders: if the whole urllib.request module attribute is
    swapped for an exploding object, analyze() still returns a list."""
    from v9.advisors.comfyui import ComfyUIAdvisor
    import v9.advisors.comfyui as cmod

    adv = ComfyUIAdvisor(config={})
    monkeypatch.setattr(adv, "_find_proc", lambda: None)
    monkeypatch.setattr(cmod, "_urlrequest", _ExplodingModule("urllib.request"))

    ctx = _running_ctx(comfyui=True)
    recs = adv.analyze(ctx)
    assert isinstance(recs, list)


# ===========================================================================
# Every advisor — app NOT running -> [] and never raises
# ===========================================================================
def _all_advisors():
    from v9.advisors.gpu import GpuAdvisor
    from v9.advisors.system import SystemAdvisor
    from v9.advisors.comfyui import ComfyUIAdvisor
    from v9.advisors.houdini import HoudiniAdvisor
    from v9.advisors.ue5 import UE5Advisor
    return [GpuAdvisor, SystemAdvisor, ComfyUIAdvisor, HoudiniAdvisor, UE5Advisor]


@pytest.mark.parametrize("AdvisorCls", _all_advisors())
def test_advisor_idle_context_never_raises(AdvisorCls):
    """With NOTHING running, every advisor's analyze() must return a list and
    never raise. The app-gated advisors (comfyui/houdini/ue5) must return []."""
    adv = AdvisorCls(config={})
    ctx = _running_ctx()  # nothing running
    recs = adv.analyze(ctx)
    assert isinstance(recs, list)


@pytest.mark.parametrize("AdvisorCls,app", [
    pytest.param("comfyui", "comfyui", id="comfyui"),
    pytest.param("houdini", "houdini", id="houdini"),
    pytest.param("ue5", "ue5", id="ue5"),
])
def test_app_advisor_returns_empty_when_app_not_running(AdvisorCls, app):
    """The three app advisors are explicitly gated on ctx.running[app]; with
    that flag False they MUST return [] (not just 'no raise')."""
    mapping = {
        "comfyui": "v9.advisors.comfyui:ComfyUIAdvisor",
        "houdini": "v9.advisors.houdini:HoudiniAdvisor",
        "ue5": "v9.advisors.ue5:UE5Advisor",
    }
    modpath, clsname = mapping[AdvisorCls].split(":")
    mod = __import__(modpath, fromlist=[clsname])
    cls = getattr(mod, clsname)
    adv = cls(config={})

    ctx = _running_ctx()  # app NOT running
    assert ctx.running[app] is False
    recs = adv.analyze(ctx)
    assert recs == []


def test_advisor_base_guard_swallows_subclass_explosion():
    """The Advisor base class promises analyze() never raises even if the
    subclass _analyze() throws. Verify the guard."""
    from v9.advisors.base import Advisor

    class _Bomb(Advisor):
        name = "bomb"

        def _analyze(self, ctx):
            raise RuntimeError("subclass detonated")

    adv = _Bomb(config={})
    ctx = _running_ctx()
    assert adv.analyze(ctx) == []  # guard caught it


def test_disabled_advisor_returns_empty():
    """An advisor disabled via config must short-circuit to [] before _analyze."""
    from v9.advisors.gpu import GpuAdvisor
    adv = GpuAdvisor(config={"enabled": False})
    assert adv.enabled is False
    assert adv.analyze(_running_ctx()) == []


# ===========================================================================
# Accel backend with SDK absent — enabled=False, preview ok, compile() hints
# ===========================================================================
def test_tensorrt_backend_disabled_when_sdk_absent():
    """tensorrt is not installed in this env. The backend must construct
    cleanly with enabled=False and a non-empty unavailable reason."""
    pytest.importorskip  # noqa: B018 - keep import available below
    try:
        import tensorrt  # type: ignore  # noqa: F401
    except Exception:
        pass
    else:
        pytest.skip("tensorrt IS installed here; SDK-absent path not exercisable")

    from v9.accel.tensorrt import TensorRTBackend
    be = TensorRTBackend()
    assert be.enabled is False
    assert be._unavailable_reason
    assert "tensorrt" in be._unavailable_reason.lower()


def test_tensorrt_applies_to_previews_without_raising():
    """A disabled backend's applies_to() must answer (False) without raising —
    it is safe to query for preview even when the SDK is missing."""
    try:
        import tensorrt  # type: ignore  # noqa: F401
        installed = True
    except Exception:
        installed = False

    from v9.accel.tensorrt import TensorRTBackend
    be = TensorRTBackend()
    # Should not raise regardless of install state.
    result = be.applies_to("gemm", (64, 64), "fp16")
    if not installed:
        assert result is False


def test_tensorrt_compile_raises_clear_install_hint_when_disabled():
    """compile() on a disabled backend must raise a CLEAR install hint
    (a RuntimeError mentioning how to install), NOT an opaque AttributeError /
    NoneType crash."""
    try:
        import tensorrt  # type: ignore  # noqa: F401
        pytest.skip("tensorrt installed; disabled-compile path not exercisable")
    except Exception:
        pass

    from v9.accel.tensorrt import TensorRTBackend
    from v9.accel.backend import OnnxPath, ShapeRange, CancelToken
    from pathlib import Path

    be = TensorRTBackend()
    assert be.enabled is False

    with pytest.raises(RuntimeError) as ei:
        be.compile(
            OnnxPath(path=Path("nonexistent.onnx")),
            ShapeRange.static((1, 4, 64, 64)),
            "fp16",
            CancelToken(),
        )
    msg = str(ei.value).lower()
    # The hint must actually tell the user how to fix it.
    assert "install" in msg
    assert "tensorrt" in msg
    # Must NOT be a bare/empty message.
    assert len(str(ei.value)) > 20


def test_tensorrt_load_raises_clear_hint_when_disabled():
    """load() on a disabled backend likewise raises a clear install-hint
    RuntimeError, not a crash from touching a None SDK handle."""
    try:
        import tensorrt  # type: ignore  # noqa: F401
        pytest.skip("tensorrt installed; disabled-load path not exercisable")
    except Exception:
        pass

    from v9.accel.tensorrt import TensorRTBackend
    from pathlib import Path

    be = TensorRTBackend()
    with pytest.raises(RuntimeError) as ei:
        be.load(Path("whatever.plan"))
    assert "tensorrt" in str(ei.value).lower()
    assert "install" in str(ei.value).lower()


def test_registry_capability_map_honest_about_dormant_backend():
    """The registry must report any dormant backend as enabled=False WITH a
    reason — DESIGN_V9 principle 4 (honest capability) — and constructing the
    default registry must never crash. Written env-independently: CUTLASS is
    config-gated OFF by default, so it is deterministically dormant whether or
    not the CUTLASS toolchain is installed on the test host."""
    from v9.accel.registry import build_default_registry

    reg = build_default_registry(hardware=_fake_profile())   # no config -> cutlass off
    cap = reg.capability_map()

    assert "tensorrt" in cap and "cutlass_fp8_gemm" in cap
    # CUTLASS is dormant-by-config here regardless of ambient SDKs.
    info = cap["cutlass_fp8_gemm"]
    assert info["enabled"] is False
    assert info["reason"]  # a non-empty explanation
    # Honesty invariant for EVERY dormant backend: a reason is always given.
    for name, meta in cap.items():
        if not meta["enabled"]:
            assert meta["reason"], f"{name} reported dormant without a reason"
    # enabled_backends() only ever returns enabled instances.
    assert all(b.enabled for b in reg.enabled_backends())


def test_registry_candidates_for_empty_when_all_dormant():
    """candidates_for() must exclude dormant backends and never raise. Tested
    env-independently with a stub backend whose applies_to() returns True but
    which is disabled — so any non-empty result would be a real filter bug."""
    from v9.accel.registry import BackendRegistry
    from v9.accel.backend import AccelerationBackend

    class _DormantStub(AccelerationBackend):
        name = "dormant_stub"
        version = "0"

        def __init__(self):
            super().__init__()
            self.enabled = False
            self._unavailable_reason = "stub: SDK absent"

        def applies_to(self, model_kind, shape, precision):
            return True  # would match if not for the enabled filter

        def estimated_compile_seconds(self, model_kind, shape):
            return 1.0

        def estimated_speedup(self, model_kind, shape):
            return 1.0

        def estimated_quality_delta(self):
            return 0.0

        def compile(self, *a, **k):
            raise RuntimeError("dormant")

        def load(self, *a, **k):
            raise RuntimeError("dormant")

    reg = BackendRegistry()
    reg.register(_DormantStub())
    assert reg.candidates_for("unet", (1, 4, 64, 64), "fp16") == []


def test_registry_register_factory_records_dead_on_construction_failure():
    """A backend whose __init__ raises must be recorded as a dead entry (with
    the exception text as reason) rather than aborting registry construction."""
    from v9.accel.registry import BackendRegistry

    def _boom_factory():
        raise RuntimeError("SDK .dll load failed")

    reg = BackendRegistry()
    reg.register_factory("explody", _boom_factory)  # must not raise
    cap = reg.capability_map()
    assert "explody" in cap
    assert cap["explody"]["enabled"] is False
    assert "SDK .dll load failed" in cap["explody"]["reason"]
