"""
Characterization tests for the control plane — gather_candidates and run_cycle,
dry-run only.

Nothing here touches the machine. Every boundary the plane reaches through is
replaced:

  * truth reads     -> `plane.Observation` is patched to a factory with stub gpu /
                       roofline. (Patching `truth.gpu_truth` would do nothing: the
                       dataclass captured the function object as its default_factory
                       at class-definition time.)
  * tasklist        -> `guard._process_table` returns a fixed table.
  * nvidia-smi      -> `Conditions.capture` returns a fixed Conditions.
  * MOS2026         -> `SystemActuator` is a stub (the real one imports MOS2026's
                       executor in its constructor) and the MOS2026 catalog is a set
                       of fake modules in `sys.modules`.
  * ledger          -> a Ledger on a temp dir, never paths.LEDGER_DB.

Bugs fixed in a later step keep their test, with the comment in the past tense
("Was bug (a): ...") so the history stays readable.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12 import guard, plane  # noqa: E402
from v12.admission import Verdict  # noqa: E402
from v12.baseline import COMPETING_METHOD, Baseline, Conditions, Measurement  # noqa: E402
from v12.change import (Change, ChangeKind, EvidenceClass, Snapshot,  # noqa: E402
                        change_key)
from v12.ledger import Ledger  # noqa: E402
from v12.objective import diffusion_objective  # noqa: E402

_ORIGINAL_OBSERVATION = plane.Observation
_ABSENT_VAR = "V12_TEST_PLANE_VAR_THAT_IS_NEVER_SET"
#: What the stubbed Conditions.capture returns at OBSERVE time. Baselines are
#: recorded under the same conditions so valid_for() accepts them. Every read
#: field is populated: an unread field never matches (Conditions.differences).
_STUB_CONDITIONS = Conditions(driver_version="stub", sm_clock_max_mhz=3105,
                              mem_clock_max_mhz=10501, power_limit_w=450.0,
                              hags_enabled=True, competing_processes=[],
                              competing_method=COMPETING_METHOD)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _snap(value="0") -> Snapshot:
    return Snapshot(value=value, existed=True, trusted=True, source="test")


def _change(cid, *, kind=ChangeKind.ENV_VAR, target=None,
            evidence=EvidenceClass.CITED, source="nvidia docs", **kw) -> Change:
    return Change(id=cid, kind=kind, description="d", target=target or f"VAR_{cid}",
                  evidence=evidence, evidence_source=source,
                  snapshot=kw.pop("snapshot", _snap()), **kw)


#: Unchanged guardrail series for every guardrail diffusion_objective declares.
_FLAT_GUARDS = {"lpips": ([0.20] * 5, [0.20] * 5), "clip": ([0.30] * 5, [0.30] * 5)}


def _readable_conditions() -> Conditions:
    """Conditions that were actually READ, so that "same machine state" is a
    fact the comparison can rely on rather than a pair of blanks.

    Conditions.differences() treats an unread field on either side as a
    difference: "we could not tell" is not "the same". A default
    Conditions() is unread on every field, so a fixture built from it would
    be refused for the wrong reason before it reached the statistics.
    """
    return Conditions(
        driver_version="999.99",
        sm_clock_max_mhz=3105,
        mem_clock_max_mhz=14001,
        power_limit_w=450.0,
        hags_enabled=True,
        workload_fingerprint="test-fingerprint",
        workload_kind="render",
        competing_processes=[],
        competing_method="test",
    )


def _measurement(candidate, baseline=None, session_id: str = "test-session") -> Measurement:
    """A candidate carrying the provenance the gate requires.

    Its conditions are the SAME as the baseline's, because that is what a
    paired session looks like: both arms were measured under one machine
    state. They have to match exactly, because the gate compares them — a
    candidate claiming a different machine state than its control is not a
    comparison of one experiment.
    """
    diffs = None
    if baseline is not None:
        diffs = [float(c) - float(b) for c, b in zip(candidate, baseline)]
    return Measurement(measured_at=1000.0, session_id=session_id,
                       conditions=asdict(_STUB_CONDITIONS),
                       paired_diffs=diffs, harness="test")


def _with_measurement(change, candidate, baseline=None):
    """A change carrying the measurement that describes its candidate."""
    extra = dict(getattr(change, "extra", {}) or {})
    extra["measurement"] = _measurement(candidate, baseline).to_extra()
    return replace(change, extra=extra)


def _baseline(obj, samples, session_id: str = "test-session") -> Baseline:
    """A baseline as `record()` would have written it: the machine state it was
    measured under, and the session it belongs to.

    The session id matters because it is what lets the gate tell a comparison
    made in ONE session from a comparison made across two — and the latter is
    what produced the +12.34% phantom on 2026-11-03.
    """
    return Baseline(
        objective_id=f"{obj.workload.kind.value}:{obj.workload.name}",
        workload=asdict(obj.workload),
        metric=obj.primary.name,
        direction=obj.primary.direction.value,
        samples=list(samples),
        conditions=asdict(_STUB_CONDITIONS),
        session_id=session_id,
        measured_at=1000.0,
    )


class _StubActuator:
    """Stands in for SystemActuator. Any call is a test failure: dry-run must
    never reach an actuator."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def _boom(self, *a, **k):
        raise AssertionError("dry-run reached the system actuator")

    apply = apply_undo = reset_default = verify = _boom


def _fake_obs(conditions=None):
    return _ORIGINAL_OBSERVATION(
        gpu=SimpleNamespace(summary="stub gpu", errors=[]),
        roofline=SimpleNamespace(summary="stub roofline"),
        conditions=conditions,
    )


class _MosChange(SimpleNamespace):
    """Carries every attribute Plane._from_mos_change reads."""


def _mos_change(cid, type_value="env_var", target=_ABSENT_VAR):
    return _MosChange(
        id=cid, type=SimpleNamespace(value=type_value), risk=SimpleNamespace(value="low"),
        scope=SimpleNamespace(value="user"), description=f"mos {cid}", target=target,
        target_value="1", reg_value_name=None, reg_type=None, file_key=None,
        file_sep="=", snapshot_cmd=None, apply_cmd=None, undo_cmd=None,
        verify_cmd=None, extra={},
    )


def _fake_mos_modules(changes, probe_accepts_mode=False):
    """A fake MachineOptimizer2026 `optimizer` package.

    By default `probe()` takes NO arguments, exactly like the real
    MachineOptimizer2026/optimizer/probe.py:132 `def probe() -> Context`.
    """
    pkg = types.ModuleType("optimizer")
    pkg.__path__ = []

    base = types.ModuleType("optimizer.base")
    base.WorkloadMode = SimpleNamespace(IDLE="idle")

    probe_mod = types.ModuleType("optimizer.probe")
    if probe_accepts_mode:
        def probe(mode=None):
            return SimpleNamespace(mode=mode)
    else:
        def probe():
            return SimpleNamespace(mode=None)
    probe_mod.probe = probe

    orch_mod = types.ModuleType("optimizer.orchestrator")

    class Orchestrator:
        # Like the real one (orchestrator.py:101), gather() reads
        # cfg['min_confidence'], so an empty cfg KeyErrors here too.
        def __init__(self, cfg, dry_run=True):
            assert dry_run is True, "plane must build MOS2026's orchestrator dry-run"
            self.cfg = cfg

        def gather(self, ctx):
            self.cfg["min_confidence"]
            return [SimpleNamespace(changes=list(changes))]

    orch_mod.Orchestrator = Orchestrator
    orch_mod.load_config = lambda: {"min_confidence": 0.6}
    pkg.base, pkg.probe, pkg.orchestrator = base, probe_mod, orch_mod
    return {"optimizer": pkg, "optimizer.base": base,
            "optimizer.probe": probe_mod, "optimizer.orchestrator": orch_mod}


class _PlaneCase(unittest.TestCase):
    """A Plane on a temp ledger with every machine boundary stubbed."""

    process_table = []

    def setUp(self):
        self.assertNotIn(_ABSENT_VAR, os.environ)
        self.tmp = tempfile.mkdtemp(prefix="v12_plane_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), saved_path))

        for p in (
            mock.patch.object(plane, "SystemActuator", _StubActuator),
            mock.patch.object(guard, "_process_table", return_value=list(self.process_table)),
            mock.patch.object(Conditions, "capture",
                              return_value=_STUB_CONDITIONS),
            mock.patch.object(plane, "Observation", side_effect=lambda: _fake_obs()),
            # Default: no MOS2026 reachable. Every submodule is blocked too, because
            # an already-cached submodule is returned without consulting its parent.
            # Individual tests override this.
            mock.patch.dict(sys.modules, {"optimizer": None, "optimizer.base": None,
                                          "optimizer.probe": None,
                                          "optimizer.orchestrator": None}),
        ):
            p.start()
            self.addCleanup(p.stop)

    def make_plane(self, objective=None, **kw) -> plane.Plane:
        led = Ledger(db_path=Path(self.tmp) / "ledger.db")
        pl = plane.Plane(objective=objective, dry_run=True, ledger=led, **kw)
        # Closed before the temp dir is removed (cleanups run LIFO): Windows
        # refuses to delete an open WAL file.
        self.addCleanup(led.close)
        # Belt and braces: no actuator may be selected in dry-run either.
        pl._actuator_for = mock.Mock(side_effect=AssertionError("actuator selected"))
        return pl

    def verdict_rows(self, pl):
        return pl.ledger.conn.execute(
            "SELECT change_id, admitted FROM verdicts ORDER BY id").fetchall()


# ---------------------------------------------------------------------------
# gather_candidates
# ---------------------------------------------------------------------------
class TestGatherCandidates(_PlaneCase):
    def test_without_mos2026_it_returns_the_builtin_catalog(self):
        pl = self.make_plane()
        cands = pl.gather_candidates(_fake_obs())
        # nvml_power_limit joined the catalog on 2026-11-04: the power ceiling
        # is the only NVML target on this card whose prior is readable and
        # whose setter round-trips, so it can be applied and restored exactly.
        self.assertEqual([c.id for c in cands],
                         ["nvml_pin_clocks", "nvml_power_limit", "kernel_vellum_gemm"])

    def test_every_candidate_is_snapshotted_before_admission(self):
        pl = self.make_plane()
        seen = []

        def rec(c):
            seen.append(c.id)
            return _snap()
        with mock.patch.object(plane, "snapshot_change", side_effect=rec):
            cands = pl.gather_candidates(_fake_obs())
        self.assertTrue(all(c.snapshot is not None for c in cands))
        self.assertEqual(seen, [c.id for c in cands])

    def test_an_existing_snapshot_is_not_retaken(self):
        pl = self.make_plane()
        pre = _snap("kept")
        cat = [_change("a", snapshot=pre)]
        with mock.patch.object(plane, "builtin_catalog", return_value=cat), \
             mock.patch.object(plane, "snapshot_change",
                               side_effect=AssertionError("re-snapshotted")):
            cands = pl.gather_candidates(_fake_obs())
        self.assertIs(cands[0].snapshot, pre)

    def test_a_broken_mos2026_falls_back_to_the_builtin_catalog(self):
        """Any exception in the MOS2026 block is logged and the builtin catalog is
        still returned (the visible run note is covered in TestMos2026Bridge)."""
        # Tolerant probe, so the failure comes from the agent and not from bug (a).
        mods = _fake_mos_modules([_mos_change("mos_a")], probe_accepts_mode=True)

        def explode(self, ctx):
            raise RuntimeError("agent crashed")
        mods["optimizer.orchestrator"].Orchestrator.gather = explode
        with mock.patch.dict(sys.modules, mods):
            pl = self.make_plane()
            with self.assertLogs("v12.plane", level="DEBUG") as cm:
                cands = pl.gather_candidates(_fake_obs())
        self.assertEqual([c.id for c in cands],
                         ["nvml_pin_clocks", "nvml_power_limit", "kernel_vellum_gemm"])
        self.assertTrue(any("agent crashed" in m for m in cm.output))

    def test_mos2026_changes_are_imported_as_cited(self):
        """Pins _from_mos_change: MOS2026 changes arrive as CITED (GUIDED kind ->
        GUIDED), sourced to the catalog. The fake Orchestrator reads
        cfg['min_confidence'] like the real one, so this also needs MOS2026's own
        config to reach it."""
        mods = _fake_mos_modules([_mos_change("mos_env"),
                                  _mos_change("mos_guided", type_value="guided")],
                                 probe_accepts_mode=True)
        with mock.patch.dict(sys.modules, mods):
            pl = self.make_plane()
            cands = pl.gather_candidates(_fake_obs())
        by_id = {c.id: c for c in cands}
        self.assertIn("mos_env", by_id)
        self.assertIs(by_id["mos_env"].kind, ChangeKind.ENV_VAR)
        self.assertIs(by_id["mos_env"].evidence, EvidenceClass.CITED)
        self.assertEqual(by_id["mos_env"].evidence_source,
                         "MachineOptimizer2026 verified catalog")
        self.assertIs(by_id["mos_guided"].evidence, EvidenceClass.GUIDED)
        self.assertIsNotNone(by_id["mos_env"].snapshot)

    def test_mos2026_catalog_arrives_with_the_real_probe_signature(self):
        # Was bug (a): the plane called probe(mode=WorkloadMode.IDLE), but MOS2026's
        # probe() takes no arguments (MachineOptimizer2026/optimizer/probe.py:132).
        # The TypeError was swallowed at DEBUG, so the MOS2026 catalog never
        # arrived and the plane silently ran on 2 builtin changes.
        mods = _fake_mos_modules([_mos_change("mos_env")], probe_accepts_mode=False)
        with mock.patch.dict(sys.modules, mods):
            pl = self.make_plane()
            cands = pl.gather_candidates(_fake_obs())
        self.assertIn("mos_env", [c.id for c in cands])


# ---------------------------------------------------------------------------
# run_cycle (dry-run)
# ---------------------------------------------------------------------------
class TestRunCycleDryRun(_PlaneCase):
    def test_explicit_candidates_are_used_verbatim(self):
        pl = self.make_plane()
        cands = [_change("a"), _change("b"), _change("c")]
        with mock.patch.object(pl, "gather_candidates",
                               side_effect=AssertionError("gathered anyway")):
            r = pl.run_cycle(candidates=cands)
        self.assertEqual(r.candidates, 3)
        self.assertEqual([v.change_id for v in r.verdicts], ["a", "b", "c"])

    def test_dry_run_takes_no_lock_and_applies_nothing(self):
        pl = self.make_plane()
        with mock.patch.object(pl._lock, "acquire",
                               side_effect=AssertionError("lock taken in dry-run")):
            r = pl.run_cycle(candidates=[_change("a"), _change("b")])
        self.assertTrue(r.dry_run)
        self.assertFalse(r.blocked)
        self.assertIn("dry-run: no lock taken", r.notes)
        self.assertEqual((r.applied, r.failed, r.results), (0, 0, []))

    def test_every_verdict_is_recorded_in_the_ledger(self):
        pl = self.make_plane()
        cands = [_change("a"), _change("g", kind=ChangeKind.GUIDED,
                                       evidence=EvidenceClass.GUIDED)]
        r = pl.run_cycle(candidates=cands)
        rows = self.verdict_rows(pl)
        self.assertEqual([row[0] for row in rows], ["a", "g"])
        self.assertEqual(r.admitted + r.denied, len(cands))
        self.assertEqual(sum(row[1] for row in rows), r.admitted)

    def test_guided_is_denied(self):
        pl = self.make_plane()
        r = pl.run_cycle(candidates=[_change("g", kind=ChangeKind.GUIDED,
                                             evidence=EvidenceClass.GUIDED)])
        self.assertEqual((r.admitted, r.denied), (0, 1))

    def test_no_objective_is_stated_in_the_report(self):
        pl = self.make_plane()
        r = pl.run_cycle(candidates=[])
        self.assertEqual(r.objective_id, "")
        self.assertTrue(any("no Objective loaded" in n for n in r.notes))
        self.assertIn("NONE — no objective loaded", r.report())

    def test_measured_without_a_baseline_is_denied(self):
        pl = self.make_plane(objective=diffusion_objective())
        r = pl.run_cycle(candidates=[_change("m", evidence=EvidenceClass.MEASURED)],
                         candidate_samples=[80.0] * 5)
        self.assertEqual((r.admitted, r.denied), (0, 1))
        self.assertIn("no baseline", r.verdicts[0].reason)

    def test_one_measured_change_with_a_clean_win_is_admitted(self):
        """The sane single-candidate path: baseline + winning samples admits."""
        obj = diffusion_objective()
        pl = self.make_plane(objective=obj)
        m = _change("m", evidence=EvidenceClass.MEASURED)
        cand = [80.0, 80.1, 79.9, 80.0, 80.05]
        m = _with_measurement(m, cand, [100.0, 100.1, 99.9, 100.0, 100.05])
        r = pl.run_cycle(
            candidates=[m],
            baseline_for_objective=_baseline(obj, [100.0, 100.1, 99.9, 100.0, 100.05]),
            candidate_samples=cand,
            guardrail_samples={change_key(m): _FLAT_GUARDS})
        self.assertEqual(r.admitted, 1, r.verdicts[0].reason)
        self.assertIs(r.verdicts[0].evidence, EvidenceClass.MEASURED)
        self.assertEqual(r.applied, 0)          # dry-run

    def test_a_win_without_its_declared_guardrails_is_denied(self):
        """Was admitted: run_cycle never passed guardrail samples and the
        Objective skipped any guardrail it had none for, so diffusion's declared
        lpips/clip were never checked on the plane path."""
        obj = diffusion_objective()
        pl = self.make_plane(objective=obj)
        base = [100.0, 100.1, 99.9, 100.0, 100.05]
        cand = [80.0, 80.1, 79.9, 80.0, 80.05]
        m = _with_measurement(_change("m", evidence=EvidenceClass.MEASURED),
                               cand, base)
        r = pl.run_cycle(
            candidates=[m],
            baseline_for_objective=_baseline(obj, base),
            candidate_samples=cand)
        self.assertEqual((r.admitted, r.denied), (0, 1))
        self.assertIn("guardrail", r.verdicts[0].reason)

    def test_guardrail_samples_reach_only_their_own_change(self):
        obj = diffusion_objective()
        pl = self.make_plane(objective=obj)
        base = [100.0, 100.1, 99.9, 100.0, 100.05]
        win = [80.0, 80.1, 79.9, 80.0, 80.05]
        cands = [_with_measurement(_change("m1", evidence=EvidenceClass.MEASURED),
                                   win, base),
                 _with_measurement(_change("m2", evidence=EvidenceClass.MEASURED),
                                   win, base)]
        r = pl.run_cycle(
            candidates=cands,
            baseline_for_objective=_baseline(obj, base),
            candidate_samples={change_key(c): win for c in cands},
            guardrail_samples={change_key(cands[0]): _FLAT_GUARDS,
                               "system|env|user|NOT_A_CANDIDATE": _FLAT_GUARDS})
        by_id = {v.change_id: v for v in r.verdicts}
        self.assertTrue(by_id["m1"].admitted, by_id["m1"].reason)
        self.assertFalse(by_id["m2"].admitted)
        self.assertIn("guardrail", by_id["m2"].reason)
        self.assertTrue(any("NOT_A_CANDIDATE" in n for n in r.notes), r.notes)

    def test_the_session_is_opened_and_closed(self):
        pl = self.make_plane()
        pl.run_cycle(candidates=[])
        started, ended, note = pl.ledger.conn.execute(
            "SELECT started, ended, note FROM sessions").fetchone()
        self.assertIsNotNone(started)
        self.assertIsNotNone(ended)
        self.assertEqual(note, "dry-run")

    def test_one_sample_series_cannot_admit_every_measured_change(self):
        # Was bug (b): run_cycle handed the SAME candidate_samples list to every
        # candidate's gate.evaluate, so one measurement admitted every MEASURED
        # change in the batch. Fixed by Plane._attribute_samples. Correct (fail
        # closed): at most one of them can be supported by a single series.
        # Outcome-based on purpose, so it does not depend on the parameter shape.
        obj = diffusion_objective()
        pl = self.make_plane(objective=obj)
        cands = [_change("m1", evidence=EvidenceClass.MEASURED),
                 _change("m2", evidence=EvidenceClass.MEASURED)]
        r = pl.run_cycle(
            candidates=cands,
            baseline_for_objective=_baseline(obj, [100.0, 100.1, 99.9, 100.0, 100.05]),
            candidate_samples=[80.0, 80.1, 79.9, 80.0, 80.05],
            # Guardrails satisfied, so only attribution can deny.
            guardrail_samples={change_key(c): _FLAT_GUARDS for c in cands})
        self.assertLessEqual(r.admitted, 1)

    def test_now_conditions_reach_the_gate(self):
        # Was bug (b), second half: run_cycle never passed now_conditions to
        # gate.evaluate, so Baseline.valid_for(objective, None) skipped the
        # machine-conditions comparison and a baseline from a different machine
        # state was accepted. Correct: the conditions OBSERVE captured are passed.
        captured = Conditions(driver_version="observed-now")
        obj = diffusion_objective()
        pl = self.make_plane(objective=obj)
        seen = []

        def evaluate(c, **kw):
            seen.append(kw)
            return Verdict(admitted=False, change_key=change_key(c), change_id=c.id,
                           reason="stub", evidence=c.evidence)
        pl.gate = SimpleNamespace(evaluate=evaluate)
        with mock.patch.object(plane, "Observation",
                               side_effect=lambda: _fake_obs()), \
             mock.patch.object(Conditions, "capture", return_value=captured):
            pl.run_cycle(candidates=[_change("m", evidence=EvidenceClass.MEASURED)])
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0].get("now_conditions"), captured)


# ---------------------------------------------------------------------------
# observe (via run_cycle)
# ---------------------------------------------------------------------------
class TestObserveWithAGpuApp(_PlaneCase):
    process_table = [(31337, "houdini.exe")]

    def test_a_resident_gpu_app_is_noted_and_listed(self):
        """Was a gap: with require_quiet=False, Observation.gpu_apps was built
        from collisions, which carry gpu-app entries only in quiet mode — so a
        normal run with Houdini resident reported gpu_apps=[]. It is now read off
        the same process table in both modes."""
        pl = self.make_plane()
        obs = pl.observe()
        self.assertTrue(any(n.startswith("GPU-resident: houdini.exe") for n in obs.notes))
        self.assertEqual(obs.gpu_apps, ["houdini.exe"])

    def test_listing_it_does_not_make_a_normal_run_blocking(self):
        # gpu_apps is informational: preflight's ok/collisions contract decides.
        pl = self.make_plane()
        pl.dry_run = False
        pl._lock = SimpleNamespace(acquire=lambda: (True, "stub lock"),
                                   release=lambda: None)
        r = pl.run_cycle(candidates=[_change("a")])
        self.assertFalse(r.blocked)
        self.assertEqual(r.candidates, 1)

    def test_quiet_mode_lists_the_gpu_app(self):
        pl = self.make_plane(require_quiet=True)
        obs = pl.observe()
        self.assertEqual(obs.gpu_apps, ["houdini.exe"])

    def test_one_table_read_serves_preflight_gpu_apps_and_conditions(self):
        """The conditions a verdict is judged under describe the same instant as
        the preflight that cleared it."""
        pl = self.make_plane(objective=diffusion_objective())
        with mock.patch.object(guard, "_process_table",
                               return_value=list(self.process_table)) as tbl, \
             mock.patch.object(Conditions, "capture",
                               return_value=_STUB_CONDITIONS) as cap:
            pl.observe()
        self.assertEqual(tbl.call_count, 1)
        self.assertEqual(cap.call_args.kwargs.get("table"), list(self.process_table))


class TestObserveWithACompetingOptimizer(_PlaneCase):
    process_table = [(4040, "machineoptimizer2026")]

    def test_dry_run_notes_the_competitor_but_still_evaluates(self):
        """Today: a competing optimizer blocks only a LIVE cycle; a dry run reports
        it and carries on, which is sane — nothing is mutated."""
        pl = self.make_plane()
        r = pl.run_cycle(candidates=[_change("a")])
        self.assertFalse(r.blocked)
        self.assertTrue(any(n.startswith("COMPETING OPTIMIZER: machineoptimizer2026")
                            for n in r.notes))
        self.assertEqual(r.candidates, 1)


class TestObserveWithAnUnreadableProcessTable(_PlaneCase):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(guard, "_process_table", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def test_a_dry_run_reports_it(self):
        r = self.make_plane().run_cycle(candidates=[_change("a")])
        self.assertFalse(r.blocked)
        self.assertTrue(any(n.startswith("process table unreadable") for n in r.notes),
                        r.notes)

    def test_a_live_cycle_refuses_to_act(self):
        """Was fail-open: an unreadable tasklist looked like an empty machine, so
        a live cycle could not see a competing optimizer and went ahead."""
        pl = self.make_plane()
        pl.dry_run = False
        pl._lock = SimpleNamespace(acquire=lambda: (True, "stub lock"),
                                   release=lambda: None)
        pl.gate = SimpleNamespace(evaluate=mock.Mock(
            side_effect=AssertionError("evaluated without a process table")))
        r = pl.run_cycle(candidates=[_change("a")])
        self.assertTrue(r.blocked)
        self.assertTrue(any("readable process table" in n for n in r.notes), r.notes)

    def test_gpu_apps_is_empty_not_guessed(self):
        self.assertEqual(self.make_plane().observe().gpu_apps, [])


class _LockSpy:
    """Stands in for guard.SingleInstance and counts what was done to it."""

    def __init__(self):
        self.acquired = self.released = 0

    def acquire(self):
        self.acquired += 1
        return True, "spy lock"

    def release(self):
        self.released += 1


class TestRunCycleCleansUp(_PlaneCase):
    """Was: a live cycle blocked after acquiring the lock and starting the ledger
    session returned holding both — the lock stayed taken for the rest of the
    process and the session never got an end time."""

    process_table = [(4040, "machineoptimizer2026")]

    def live_plane(self, **kw):
        pl = self.make_plane(**kw)
        pl.dry_run = False
        pl._lock = _LockSpy()
        return pl

    def session_ended(self, pl):
        row = pl.ledger.conn.execute("SELECT ended FROM sessions WHERE id=?",
                                     (pl.ledger.session,)).fetchone()
        return row is not None and row[0] is not None

    def test_a_cycle_blocked_by_a_competing_optimizer_releases_and_ends(self):
        pl = self.live_plane()
        r = pl.run_cycle(candidates=[_change("a")])
        self.assertTrue(r.blocked)
        self.assertEqual((pl._lock.acquired, pl._lock.released), (1, 1))
        self.assertTrue(self.session_ended(pl))

    def test_a_cycle_blocked_by_an_unreadable_table_releases_and_ends(self):
        pl = self.live_plane()
        with mock.patch.object(guard, "_process_table", return_value=None):
            r = pl.run_cycle(candidates=[_change("a")])
        self.assertTrue(r.blocked)
        self.assertEqual(pl._lock.released, 1)
        self.assertTrue(self.session_ended(pl))

    def test_an_exception_mid_cycle_still_releases_and_ends(self):
        pl = self.live_plane()
        with mock.patch.object(guard, "_process_table", return_value=[]):
            pl.gate = SimpleNamespace(evaluate=mock.Mock(side_effect=RuntimeError("boom")))
            with self.assertRaises(RuntimeError):
                pl.run_cycle(candidates=[_change("a")])
        self.assertEqual(pl._lock.released, 1)
        self.assertTrue(self.session_ended(pl))

    def test_a_lock_that_was_not_acquired_is_not_released_or_started(self):
        pl = self.live_plane()
        pl._lock.acquire = lambda: (False, "held by pid 1")
        with mock.patch.object(pl.ledger, "start",
                               side_effect=AssertionError("session started")):
            r = pl.run_cycle(candidates=[_change("a")])
        self.assertTrue(r.blocked)
        self.assertEqual(pl._lock.released, 0)

    def test_a_dry_run_also_ends_its_session(self):
        pl = self.make_plane()
        pl.run_cycle(candidates=[_change("a")])
        self.assertTrue(self.session_ended(pl))


# ---------------------------------------------------------------------------
# the MOS2026 bridge
# ---------------------------------------------------------------------------
class TestMos2026Bridge(_PlaneCase):
    def gather(self, mods):
        with mock.patch.dict(sys.modules, mods):
            pl = self.make_plane()
            obs = _fake_obs()
            cands = pl.gather_candidates(obs)
        return cands, obs

    def test_the_orchestrator_is_built_with_mos2026s_own_config(self):
        # Orchestrator(cfg={}) KeyErrors on cfg['min_confidence'] at gather time.
        mods = _fake_mos_modules([_mos_change("mos_env")])
        seen = []
        mods["optimizer.orchestrator"].load_config = (
            lambda: seen.append(1) or {"min_confidence": 0.6, "marker": "mos-own"})
        cands, _ = self.gather(mods)
        self.assertEqual(seen, [1])
        self.assertIn("mos_env", [c.id for c in cands])

    def test_a_loaded_catalog_is_stated_with_its_size(self):
        mods = _fake_mos_modules([_mos_change("a"), _mos_change("b")])
        _, obs = self.gather(mods)
        self.assertIn("MOS2026 catalog loaded: 2 change(s)", obs.notes)

    def test_the_loaded_catalog_reaches_the_run_report(self):
        mods = _fake_mos_modules([_mos_change("a")])
        with mock.patch.dict(sys.modules, mods):
            pl = self.make_plane()
            r = pl.run_cycle()
        self.assertIn("note: MOS2026 catalog loaded: 1 change(s)", r.report())

    def test_a_broken_catalog_is_a_visible_note_not_a_debug_line(self):
        mods = _fake_mos_modules([_mos_change("a")])

        def explode(self, ctx):
            raise RuntimeError("agent crashed")
        mods["optimizer.orchestrator"].Orchestrator.gather = explode
        with mock.patch.dict(sys.modules, mods):
            pl = self.make_plane()
            r = pl.run_cycle()
        self.assertTrue(any(n.startswith("MOS2026 catalog NOT loaded (RuntimeError: "
                                         "agent crashed)") for n in r.notes), r.notes)

    def test_the_config_source_and_filters_are_stated(self):
        """Was silent: MOS2026's config.toml sets min_confidence, gather() keeps
        only IDLE proposals, and the report said neither."""
        mods = _fake_mos_modules([_mos_change("a")])
        mods["optimizer.orchestrator"].load_config = lambda: {"min_confidence": 0.9}
        _, obs = self.gather(mods)
        note = next((n for n in obs.notes if n.startswith("MOS2026 catalog filter:")),
                    None)
        self.assertIsNotNone(note, obs.notes)
        self.assertIn("min_confidence=0.9", note)
        self.assertIn("config.toml", note)
        self.assertIn("workload=IDLE regardless of the objective", note)

    def test_mos2026_warnings_reach_the_report_and_warn_is_restored(self):
        """Was swallowed: MOS2026 prints agent import/propose() failures through
        its own ui.warn, so a catalog with dead agents read as fully loaded."""
        mods = _fake_mos_modules([_mos_change("a")])
        orch_mod = mods["optimizer.orchestrator"]
        console = []
        orch_mod.ui = SimpleNamespace(warn=console.append)
        original = orch_mod.ui.warn
        real_gather = orch_mod.Orchestrator.gather

        def gather(self, ctx):
            orch_mod.ui.warn("agent gpu propose() failed: boom")
            return real_gather(self, ctx)
        orch_mod.Orchestrator.gather = gather
        orch_mod.Orchestrator.agents = ["gpu", "cpu"]
        with mock.patch.dict(sys.modules, mods):
            pl = self.make_plane()
            r = pl.run_cycle()
        self.assertIn("note: MOS2026 warned: agent gpu propose() failed: boom", r.report())
        self.assertTrue(any("2 agent(s) discovered" in n for n in r.notes), r.notes)
        self.assertEqual(console, ["agent gpu propose() failed: boom"])
        self.assertIs(orch_mod.ui.warn, original)

    def test_an_unknown_change_type_is_refused_not_run_as_a_command(self):
        mods = _fake_mos_modules([_mos_change("ok"),
                                  _mos_change("weird", type_value="kernel_driver_poke")])
        cands, obs = self.gather(mods)
        ids = [c.id for c in cands]
        self.assertIn("ok", ids)
        self.assertNotIn("weird", ids)
        self.assertFalse(any(c.kind is ChangeKind.COMMAND for c in cands))
        self.assertIn("MOS2026 catalog loaded: 1 change(s), 1 refused", obs.notes)
        self.assertTrue(any(n.startswith("refused MOS2026 change 'weird'")
                            and "kernel_driver_poke" in n for n in obs.notes), obs.notes)

    def test_an_admin_only_mos_command_arrives_requiring_admin(self):
        """Was dropped: MOS2026 marks HKLM/Defender COMMANDs admin=True, V12's
        ADMIN_KINDS does not include COMMAND, and the bridge never read the flag,
        so they arrived (and went back to MOS2026's executor) as requires_admin=False."""
        mc = _mos_change("gpu_msi_mode", type_value="command")
        mc.requires_admin = True
        self.assertTrue(plane.Plane._from_mos_change(mc).requires_admin)

    def test_a_proposal_level_admin_flag_is_carried_too(self):
        mods = _fake_mos_modules([_mos_change("cmd", type_value="command"),
                                  _mos_change("env")])
        orch = mods["optimizer.orchestrator"].Orchestrator
        orch.gather = lambda self, ctx: [
            SimpleNamespace(changes=[_mos_change("cmd", type_value="command")],
                            requires_admin=True),
            SimpleNamespace(changes=[_mos_change("env")], requires_admin=False)]
        cands, _ = self.gather(mods)
        by_id = {c.id: c for c in cands}
        self.assertTrue(by_id["cmd"].requires_admin)
        self.assertFalse(by_id["env"].requires_admin)

    def test_ini_section_keeps_same_key_cvars_apart(self):
        def ini(cid, section):
            mc = _mos_change(cid, type_value="file_line", target=r"C:\fake\Engine.ini")
            mc.file_key = "r.Streaming.PoolSize"
            mc.extra = {"section": section}
            return mc
        a = plane.Plane._from_mos_change(ini("a", "/Script/Engine.RendererSettings"))
        b = plane.Plane._from_mos_change(ini("b", "SystemSettings"))
        self.assertEqual(a.section, "/Script/Engine.RendererSettings")
        self.assertNotEqual(change_key(a), change_key(b))


# ---------------------------------------------------------------------------
# candidate samples are attributed per change
# ---------------------------------------------------------------------------
class TestSampleAttribution(_PlaneCase):
    BASE = [100.0, 100.1, 99.9, 100.0, 100.05]
    WIN = [80.0, 80.1, 79.9, 80.0, 80.05]

    def run_two(self, samples):
        obj = diffusion_objective()
        pl = self.make_plane(objective=obj)
        cands = [_change("m1", evidence=EvidenceClass.MEASURED),
                 _change("m2", evidence=EvidenceClass.MEASURED)]
        r = pl.run_cycle(candidates=cands,
                         baseline_for_objective=_baseline(obj, self.BASE),
                         candidate_samples=samples,
                         # Guardrails satisfied, so only attribution can deny.
                         guardrail_samples={change_key(c): _FLAT_GUARDS for c in cands})
        return r, cands

    def test_keyed_samples_admit_only_their_own_change(self):
        obj = diffusion_objective()
        pl = self.make_plane(objective=obj)
        cands = [_with_measurement(_change("m1", evidence=EvidenceClass.MEASURED),
                                   self.WIN, self.BASE),
                 _with_measurement(_change("m2", evidence=EvidenceClass.MEASURED),
                                   self.WIN, self.BASE)]
        r = pl.run_cycle(candidates=cands,
                         baseline_for_objective=_baseline(obj, self.BASE),
                         candidate_samples={change_key(cands[0]): self.WIN},
                         guardrail_samples={change_key(c): _FLAT_GUARDS for c in cands})
        by_id = {v.change_id: v for v in r.verdicts}
        self.assertTrue(by_id["m1"].admitted, by_id["m1"].reason)
        self.assertFalse(by_id["m2"].admitted)
        self.assertIn("no candidate samples", by_id["m2"].reason)

    def test_a_bare_series_with_two_measured_candidates_admits_neither(self):
        r, _ = self.run_two(self.WIN)
        self.assertEqual((r.admitted, r.denied), (0, 2))
        self.assertTrue(any("cannot be attributed" in n for n in r.notes), r.notes)

    def test_keyed_samples_for_an_unknown_key_are_ignored_and_noted(self):
        r, _ = self.run_two({"system|env|user|NOT_A_CANDIDATE": self.WIN})
        self.assertEqual(r.admitted, 0)
        self.assertTrue(any("NOT_A_CANDIDATE" in n and "match no candidate" in n
                            for n in r.notes), r.notes)

    def test_samples_for_a_key_shared_by_two_candidates_go_to_neither(self):
        obj = diffusion_objective()
        pl = self.make_plane(objective=obj)
        twins = [_change("t1", evidence=EvidenceClass.MEASURED, target="SAME"),
                 _change("t2", evidence=EvidenceClass.MEASURED, target="SAME")]
        r = pl.run_cycle(candidates=twins,
                         baseline_for_objective=_baseline(obj, self.BASE),
                         candidate_samples={change_key(twins[0]): self.WIN})
        self.assertEqual(r.admitted, 0)
        self.assertTrue(any("cannot attribute" in n for n in r.notes), r.notes)

    def test_the_no_objective_note_does_not_claim_cited_changes_apply(self):
        pl = self.make_plane()
        r = pl.run_cycle(candidates=[])
        note = next(n for n in r.notes if "no Objective loaded" in n)
        self.assertIn("CITED changes are recorded, never applied", note)


# ---------------------------------------------------------------------------
# cli: --candidate / --candidate-for
# ---------------------------------------------------------------------------
class TestCliSamples(unittest.TestCase):
    def test_keyed_samples_split_on_the_last_equals(self):
        from v12 import cli
        got = cli._parse_keyed_samples(["system|env|user|A=1,2,3",
                                        r"system|file|C:\x.ini|S|k=4 5"])
        self.assertEqual(got, {"system|env|user|A": [1.0, 2.0, 3.0],
                               r"system|file|C:\x.ini|S|k": [4.0, 5.0]})

    def test_keyed_samples_without_a_key_are_refused(self):
        from v12 import cli
        with self.assertRaises(SystemExit):
            cli._parse_keyed_samples(["=1,2,3"])

    def test_run_refuses_both_sample_forms_at_once(self):
        from v12 import cli
        args = cli.build_parser().parse_args(
            ["run", "--candidate", "1,2", "--candidate-for", "k=1,2"])
        with mock.patch("v12.plane.Plane",
                        side_effect=AssertionError("plane built anyway")):
            with self.assertRaises(SystemExit) as cm:
                cli.cmd_run(args)
        self.assertIn("not both", str(cm.exception))

    def test_run_hands_keyed_samples_to_the_plane(self):
        from v12 import cli
        args = cli.build_parser().parse_args(
            ["run", "--candidate-for", "system|env|user|A=1,2,3"])
        fake = mock.Mock()
        fake.return_value.run_cycle.return_value = SimpleNamespace(
            report=lambda: "", admitted=0)
        with mock.patch("v12.plane.Plane", fake), \
             mock.patch("builtins.print"):
            cli.cmd_run(args)
        self.assertEqual(fake.return_value.run_cycle.call_args.kwargs["candidate_samples"],
                         {"system|env|user|A": [1.0, 2.0, 3.0]})
        self.assertTrue(fake.call_args.kwargs.get("dry_run", False))


if __name__ == "__main__":
    unittest.main()
