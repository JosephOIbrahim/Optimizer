"""CRUCIBLE — Correctness contract tests for Optimizer V9.

Adversarial verification of the DESIGN_V9 §4 invariants:
  1. Detect, don't hardcode  -> NO-STALE-CONSTANT grep (headline principle).
  2/6. One enum, clean imports -> every v9 module imports under this interpreter.
  3. Off-OneDrive for mutable  -> paths.DATA_ROOT is not under OneDrive.
  + the recommend.py contract: aggregate() dedup semantics + build gating.

These tests must NOT be weakened to pass. A failure is a reported defect.
"""
from __future__ import annotations

import importlib
import re
import warnings
from pathlib import Path

import pytest

# v9 lives one directory up from tests/v9/ (repo root / v9).
REPO_ROOT = Path(__file__).resolve().parents[2]
V9_DIR = REPO_ROOT / "v9"


def _all_v9_module_names() -> list[str]:
    names: list[str] = []
    for p in sorted(V9_DIR.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(REPO_ROOT).with_suffix("")
        dotted = ".".join(rel.parts)
        if dotted.endswith(".__init__"):
            dotted = dotted[: -len(".__init__")]
        names.append(dotted)
    return names


def _all_v9_source_files() -> list[Path]:
    files: list[Path] = []
    for ext in ("*.py", "*.yaml", "*.yml", "*.bat", "*.cfg", "*.ini", "*.txt"):
        for p in V9_DIR.rglob(ext):
            if "__pycache__" in p.parts:
                continue
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# §2/§6 — every module imports clean under C:/Python314/python.exe
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("modname", _all_v9_module_names())
def test_module_imports_clean(modname: str) -> None:
    """Every v9 module must import without error on this interpreter."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{modname} failed to import: {type(exc).__name__}: {exc}")


def test_v9_package_discovers_modules() -> None:
    """Sanity: the discovery itself found a real tree (not zero files)."""
    names = _all_v9_module_names()
    assert "v9.cli" in names
    assert "v9.hardware" in names
    assert "v9.recommend" in names
    assert len(names) >= 20, f"only discovered {len(names)} modules: {names}"


# ---------------------------------------------------------------------------
# §1 — NO-STALE-CONSTANT GREP (the headline principle: fail loudly).
# ---------------------------------------------------------------------------
# The known stale driver literals from the v8-era three-way contradiction
# (config:581.80, code:596.36) plus the DESIGN doc's live example 610.47.
# DESIGN_V9 §4.1: "Zero driver/arch literals elsewhere." Any occurrence in the
# v9/ source tree (tests excluded) is a violation, even in comments/docstrings,
# because the principle is that the value is DETECTED, never written down.
_STALE_LITERALS = ("596.36", "581.80")

# A generic "NNN.NN" driver-version-shaped literal. We allow benign decimals
# (versions like CUDA 12.0, ratios, thresholds) but flag anything matching the
# NVIDIA driver shape (3 digits . 2 digits), which is exactly the format of
# 596.36 / 581.80 / 610.47.
_DRIVER_SHAPE = re.compile(r"\b\d{3}\.\d{2}\b")


def test_no_known_stale_driver_literals_in_v9_source() -> None:
    """581.80 and 596.36 must not appear anywhere under v9/ (tests excluded)."""
    offenders: list[str] = []
    for f in _all_v9_source_files():
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lit in _STALE_LITERALS:
            if lit in text:
                for i, line in enumerate(text.splitlines(), 1):
                    if lit in line:
                        rel = f.relative_to(REPO_ROOT)
                        offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "Stale driver literals found under v9/ (DESIGN_V9 §4.1 — detect, "
        "don't hardcode). Even in comments these violate the no-stale-constant "
        "principle:\n  " + "\n  ".join(offenders)
    )


def test_no_hardcoded_driver_version_literal_in_v9_source() -> None:
    """No NVIDIA-driver-shaped literal (NNN.NN) anywhere under v9/.

    This is the strict form of the headline principle: the driver version is
    DETECTED at runtime (hardware.detect()), so no NNN.NN literal — including
    the live 610.47 — should be written into the source tree at all.
    """
    offenders: list[str] = []
    for f in _all_v9_source_files():
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if _DRIVER_SHAPE.search(line):
                rel = f.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "Driver-version-shaped literals (NNN.NN) found under v9/. The driver "
        "version must be detected, never written down (DESIGN_V9 §4.1):\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# recommend.py contract — aggregate() dedup + build gating defaults
# ---------------------------------------------------------------------------
def test_aggregate_dedups_by_action_target_keeping_highest_confidence() -> None:
    from v9.recommend import Action, Recommendation, aggregate

    low = Recommendation("gpu", Action.SET_ENV_VAR, "CUDA_X", 0.40,
                         expected_impact_pct=5.0)
    high = Recommendation("system", Action.SET_ENV_VAR, "CUDA_X", 0.90,
                          expected_impact_pct=5.0)
    other = Recommendation("gpu", Action.SET_ENV_VAR, "CUDA_Y", 0.50)

    out = aggregate([low, high, other])
    # CUDA_X collapses to one entry; CUDA_Y stays separate.
    keyset = {(r.action, r.target) for r in out}
    assert (Action.SET_ENV_VAR, "CUDA_X") in keyset
    assert (Action.SET_ENV_VAR, "CUDA_Y") in keyset
    assert len(out) == 2, f"expected 2 deduped recs, got {len(out)}"

    kept = next(r for r in out if r.target == "CUDA_X")
    assert kept.confidence == 0.90, "dedup must keep the highest-confidence rec"
    assert kept.advisor == "system"


def test_aggregate_distinct_keys_when_action_or_target_differ() -> None:
    from v9.recommend import Action, Recommendation, aggregate

    # Same target, different action -> NOT a duplicate.
    r1 = Recommendation("gpu", Action.SET_ENV_VAR, "thing", 0.5)
    r2 = Recommendation("gpu", Action.REGISTRY_TWEAK, "thing", 0.5)
    assert len(aggregate([r1, r2])) == 2


def test_aggregate_drops_below_min_confidence() -> None:
    from v9.recommend import Action, Recommendation, aggregate

    weak = Recommendation("gpu", Action.SET_ENV_VAR, "weak", 0.10)
    strong = Recommendation("gpu", Action.SET_ENV_VAR, "strong", 0.80)
    out = aggregate([weak, strong], min_confidence=0.5)
    targets = {r.target for r in out}
    assert targets == {"strong"}


def test_aggregate_sorts_by_confidence_times_impact_descending() -> None:
    from v9.recommend import Action, Recommendation, aggregate

    # higher conf*impact must come first
    big = Recommendation("a", Action.SET_ENV_VAR, "big", 0.9,
                         expected_impact_pct=50.0)
    small = Recommendation("b", Action.SET_ENV_VAR, "small", 0.9,
                           expected_impact_pct=2.0)
    out = aggregate([small, big])
    assert out[0].target == "big"


def test_build_actions_default_gated_on_idle() -> None:
    from v9.recommend import Action, Recommendation, _BUILD_ACTIONS

    for action in _BUILD_ACTIONS:
        rec = Recommendation("comfyui", action, f"target-{action.value}", 0.8)
        assert rec.action.is_build is True
        assert rec.gated_on_idle is True, (
            f"build action {action} must default gated_on_idle=True "
            "(DESIGN_V9: build-and-cache is idle-gated)")


def test_build_action_immediate_param_disables_gating() -> None:
    from v9.recommend import Action, Recommendation

    rec = Recommendation("comfyui", Action.COMPILE_TRT_ENGINE, "sdxl", 0.8,
                         parameters={"immediate": True})
    assert rec.gated_on_idle is False


def test_config_actions_not_gated_by_default() -> None:
    from v9.recommend import Action, Recommendation

    for action in (Action.SET_ENV_VAR, Action.REGISTRY_TWEAK,
                   Action.PROCESS_PRIORITY, Action.CPU_AFFINITY,
                   Action.VRAM_ALLOCATION, Action.ADJUST_QUALITY,
                   Action.SET_LAUNCH_FLAG):
        rec = Recommendation("gpu", action, "t", 0.5)
        assert rec.action.is_build is False
        assert rec.gated_on_idle is False, (
            f"config action {action} must not be idle-gated by default")


def test_confidence_clamped_to_unit_interval() -> None:
    from v9.recommend import Action, Recommendation

    assert Recommendation("a", Action.NOTIFY, "t", 5.0).confidence == 1.0
    assert Recommendation("a", Action.NOTIFY, "t", -3.0).confidence == 0.0


# ---------------------------------------------------------------------------
# §3 — paths: mutable data root must NOT be under OneDrive
# ---------------------------------------------------------------------------
def test_data_root_not_under_onedrive() -> None:
    from v9 import paths

    root = Path(paths.DATA_ROOT)
    parts_lower = [seg.lower() for seg in root.parts]
    assert not any("onedrive" in seg for seg in parts_lower), (
        f"DATA_ROOT {root} is under OneDrive (DESIGN_V9 §4.3 — mutable state "
        "must live off OneDrive to avoid SQLite/WAL sync churn).")
    assert paths.info()["under_onedrive"] is False


def test_paths_helpers_resolve_off_onedrive() -> None:
    from v9 import paths

    for p in (paths.db_path("telemetry"), paths.cache_dir("comfyui"),
              paths.queue_path("compile"), paths.log_path("optimizer")):
        parts_lower = [seg.lower() for seg in Path(p).parts]
        assert not any("onedrive" in seg for seg in parts_lower), (
            f"path helper resolved under OneDrive: {p}")
