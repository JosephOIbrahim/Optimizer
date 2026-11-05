"""
Tests for baseline — the conditions a measurement was taken under, and the file
that carries them from one session to the next.

Two defects, both of which crashed `run --objective` and `baseline --record`:

  * `Conditions.capture` read `t.mem_clock_max_mhz or t.mem_clock_mhz`. GpuTruth
    has neither attribute; the maximum memory clock is `clock_mem_max_mhz`. The
    live clock (`clock_mem_live_mhz`) must never be used — an idle 4090 reads
    ~405 MHz, so a baseline would be invalidated by whether the card was idle.
  * The workload was stored via `asdict()`, which leaves `WorkloadKind` as an
    Enum that json.dumps cannot write; and a stored string kind was never turned
    back into an Enum on load, so `compatible_with()`'s identity check failed.

No real GPU, registry, nvidia-smi, or D:\\OptimizerV12 is touched: the truth
reader and the process lister are faked, and BASELINE_DIR points at a temp dir.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12 import baseline, guard, paths  # noqa: E402
from v12.baseline import Baseline, Conditions  # noqa: E402
from v12.objective import WorkloadKind, diffusion_objective  # noqa: E402
from v12.truth import GpuTruth  # noqa: E402


def _fake_truth() -> GpuTruth:
    """A complete read, with a live memory clock deliberately far from the max
    so a capture that used the live one is caught."""
    return GpuTruth(
        name="NVIDIA GeForce RTX 4090", driver_version="999.99",
        sm_count=128, mem_bus_width_bits=384,
        clock_sm_max_mhz=3105, clock_mem_max_mhz=10501,
        clock_sm_live_mhz=210, clock_mem_live_mhz=405,
        clock_sm_rated_mhz=2520, mem_clock_used_mhz=10501,
        power_limit_w=450.04, hags_enabled=True,
    )


class _Isolated(unittest.TestCase):
    """Fakes every machine boundary baseline.py reaches for."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "baselines"
        for patcher in (
            mock.patch.object(paths, "BASELINE_DIR", self.dir),
            mock.patch.object(baseline.truth, "gpu_truth", side_effect=_fake_truth),
            mock.patch.object(baseline, "competing_gpu_processes", return_value=[]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.obj = diffusion_objective()


class TestConditionsCapture(_Isolated):

    def test_capture_reads_a_fake_truth_without_crashing(self):
        c = Conditions.capture(self.obj)
        self.assertEqual(c.driver_version, "999.99")
        self.assertEqual(c.sm_clock_max_mhz, 3105)
        self.assertEqual(c.power_limit_w, 450.0)
        self.assertIs(c.hags_enabled, True)
        self.assertEqual(c.workload_fingerprint, self.obj.workload.fingerprint)
        self.assertEqual(c.workload_kind, WorkloadKind.DIFFUSION.value)
        self.assertEqual(c.competing_processes, [])

    def test_capture_uses_the_max_memory_clock_not_the_live_one(self):
        c = Conditions.capture(self.obj)
        self.assertEqual(c.mem_clock_max_mhz, 10501)
        self.assertNotEqual(c.mem_clock_max_mhz, 405)

    def test_an_idle_and_a_busy_card_capture_the_same_conditions(self):
        idle = Conditions.capture(self.obj)
        busy = _fake_truth()
        busy.clock_mem_live_mhz = 10501
        with mock.patch.object(baseline.truth, "gpu_truth", return_value=busy):
            loaded = Conditions.capture(self.obj)
        self.assertEqual(idle.differences(loaded), [])


class TestPersistence(_Isolated):

    SAMPLES = [10.0, 10.2, 9.9, 10.1, 10.05]

    def test_record_then_load_round_trips_to_an_equal_usable_baseline(self):
        b = baseline.record(self.obj, self.SAMPLES, harness="test")
        p = self.dir / f"{b.id}.json"
        self.assertTrue(p.exists())
        on_disk = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["workload"]["kind"], "diffusion")

        back = Baseline.load(p)
        self.assertEqual(back, b)
        ok, why = back.valid_for(self.obj, Conditions.capture(self.obj))
        self.assertTrue(ok, why)

    def test_a_baseline_built_with_an_enum_kind_saves_and_loads_equal(self):
        b = Baseline(
            objective_id="diffusion:x", workload=asdict(self.obj.workload),
            metric=self.obj.primary.name, direction=self.obj.primary.direction.value,
            samples=list(self.SAMPLES), conditions=asdict(Conditions.capture(self.obj)),
        )
        back = Baseline.load(b.save())
        self.assertEqual(back, b)
        self.assertEqual(back.valid_for(self.obj), (True, "valid"))

    def test_load_latest_finds_the_recorded_baseline(self):
        b = baseline.record(self.obj, self.SAMPLES)
        latest = baseline.load_latest(self.obj)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.id, b.id)

    def test_a_loaded_baseline_still_refuses_a_different_workload_kind(self):
        b = Baseline.load(baseline.record(self.obj, self.SAMPLES).path())
        other = diffusion_objective()
        other.workload.kind = WorkloadKind.RENDER
        ok, why = b.valid_for(other)
        self.assertFalse(ok)
        self.assertIn("workload", why)

    def test_an_unknown_stored_kind_is_refused_not_guessed(self):
        b = baseline.record(self.obj, self.SAMPLES)
        raw = json.loads(b.path().read_text(encoding="utf-8"))
        raw["workload"]["kind"] = "not-a-kind"
        b.path().write_text(json.dumps(raw), encoding="utf-8")
        ok, why = Baseline.load(b.path()).valid_for(self.obj)
        self.assertFalse(ok)
        self.assertIn("no readable workload identity", why)


class TestUnreadConditionsNeverMatch(_Isolated):
    """An unread field is stored as 0, "" or None, and two unread values compare
    equal. Matching on that would admit a MEASURED change against a baseline that
    agrees with the machine only because the same read failed both times."""

    SAMPLES = [10.0, 10.2, 9.9, 10.1, 10.05]

    def test_two_unread_conditions_do_not_match(self):
        diffs = Conditions().differences(Conditions())
        for k in ("driver_version", "sm_clock_max_mhz", "mem_clock_max_mhz",
                  "power_limit_w", "hags_enabled"):
            self.assertTrue(any(d.startswith(f"{k}: unread") for d in diffs), (k, diffs))

    def test_one_unread_side_does_not_match(self):
        full = Conditions.capture(self.obj)
        blind = Conditions.capture(self.obj)
        blind.mem_clock_max_mhz = 0
        self.assertIn("mem_clock_max_mhz: unread (now)", full.differences(blind))
        self.assertIn("mem_clock_max_mhz: unread (baseline)", blind.differences(full))

    def test_hags_off_is_a_reading_not_an_unread_field(self):
        a, b = Conditions.capture(self.obj), Conditions.capture(self.obj)
        a.hags_enabled = b.hags_enabled = False
        self.assertEqual(a.differences(b), [])

    def test_no_competing_processes_is_a_reading(self):
        c = Conditions.capture(self.obj)
        self.assertEqual(c.competing_processes, [])
        self.assertEqual(c.unread(), [])

    def test_a_baseline_with_unread_conditions_is_refused_by_valid_for(self):
        b = Baseline(
            objective_id="diffusion:x", workload=asdict(self.obj.workload),
            metric=self.obj.primary.name, direction=self.obj.primary.direction.value,
            samples=list(self.SAMPLES), conditions=asdict(Conditions()),
        )
        ok, why = b.valid_for(self.obj, Conditions())
        self.assertFalse(ok)
        self.assertIn("unread", why)

    def test_record_refuses_an_incomplete_capture(self):
        blind = _fake_truth()
        blind.clock_mem_max_mhz = 0
        with mock.patch.object(baseline.truth, "gpu_truth", return_value=blind):
            with self.assertRaises(ValueError) as cm:
                baseline.record(self.obj, self.SAMPLES)
        self.assertIn("mem_clock_max_mhz", str(cm.exception))
        self.assertFalse(self.dir.exists() and any(self.dir.glob("*.json")))

    def test_record_refuses_explicit_unread_conditions(self):
        with self.assertRaises(ValueError):
            baseline.record(self.obj, self.SAMPLES, conditions=Conditions())


# ---------------------------------------------------------------------------
# competing GPU processes — heavy or significant only, and UNREAD kept distinct
# ---------------------------------------------------------------------------
ME = 4242

#: The shape of this box's table on 2026-10-01: a resident Houdini, the XDR's
#: compositor, and ordinary desktop apps the old nvidia-smi list named.
_TABLE = [(51896, "houdini.exe"), (2584, "dwm.exe"), (13332, "spotify.exe"),
          (75552, "msedge.exe"), (61388, "obs64.exe"), (34020, "claude.exe"),
          (ME, "python.exe")]
_COMMIT = {51896: 4961.9, 2584: 2706.3, 13332: 488.2, 75552: 429.6,
           61388: 330.0, 34020: 701.4, ME: 2048.0}

#: The real reader, kept before _Isolated stubs it out for capture().
_read_competing = baseline.competing_gpu_processes


class TestCompetingGpuProcesses(unittest.TestCase):
    """Was: every name `nvidia-smi --query-compute-apps` printed (~40 desktop
    apps under WDDM), compared as an exact list, and [] when nvidia-smi failed."""

    def setUp(self):
        p = mock.patch.object(baseline.os, "getpid", return_value=ME)
        p.start()
        self.addCleanup(p.stop)
        # No test here may reach the old source, silently or otherwise.
        p = mock.patch("subprocess.run",
                       side_effect=AssertionError("a subprocess (nvidia-smi?) was run"))
        p.start()
        self.addCleanup(p.stop)

    def read(self, commit=_COMMIT, table=_TABLE):
        with mock.patch.object(baseline, "_gpu_commit_by_pid", return_value=commit), \
             mock.patch.object(guard, "_process_table",
                               return_value=None if table is None else list(table)):
            return baseline.competing_gpu_processes()

    def test_only_heavy_and_significant_processes_are_named(self):
        self.assertEqual(self.read(), ["dwm.exe:heavy", "houdini.exe:heavy"])

    def test_desktop_apps_coming_and_going_do_not_change_the_reading(self):
        more = _TABLE + [(9001, "chrome.exe"), (9002, "teams.exe")]
        commit = dict(_COMMIT)
        commit.update({9001: 600.0, 9002: 350.0})
        self.assertEqual(self.read(), self.read(commit=commit, table=more))

    def test_a_significant_app_is_named_whatever_its_size(self):
        # Houdini just launched with nothing committed yet; ComfyUI under python.exe.
        table = [(1, "houdini.exe"), (2, "python.exe")]
        self.assertEqual(self.read(commit={}, table=table), ["houdini.exe", "python.exe"])

    def test_a_heavy_python_job_arriving_beside_a_stray_one_is_seen(self):
        """A ComfyUI/torch job under python.exe committing 15 GB used to read the
        same as an idle stray python.exe: both were the bare name."""
        before = self.read(commit={1: 0.0}, table=[(1, "python.exe")])
        after = self.read(commit={1: 0.0, 2: 15000.0},
                          table=[(1, "python.exe"), (2, "python.exe")])
        self.assertEqual(before, ["python.exe"])
        self.assertEqual(after, ["python.exe", "python.exe:heavy"])

    def test_idle_strays_are_named_once_however_many(self):
        table = [(i, "python.exe") for i in range(1, 24)]
        self.assertEqual(self.read(commit={}, table=table), ["python.exe"])

    def test_a_second_heavy_instance_is_counted(self):
        table = [(1, "houdini.exe"), (2, "houdini.exe")]
        self.assertEqual(self.read(commit={1: 5000.0}, table=table),
                         ["houdini.exe", "houdini.exe:heavy"])
        self.assertEqual(self.read(commit={1: 5000.0, 2: 3000.0}, table=table),
                         ["houdini.exe:heavy*2"])

    def test_the_threshold_is_a_named_constant_and_inclusive(self):
        cut = baseline.HEAVY_GPU_COMMIT_MIB
        table = [(1, "big.exe"), (2, "small.exe")]
        self.assertEqual(self.read(commit={1: cut, 2: cut - 0.1}, table=table),
                         ["big.exe:heavy"])

    def test_v12s_own_process_is_not_a_competitor(self):
        self.assertNotIn("python.exe", self.read())

    def test_nothing_heavy_is_an_honest_empty_reading(self):
        self.assertEqual(self.read(commit={13332: 488.2}, table=[(13332, "spotify.exe")]), [])

    def test_an_unreadable_counter_is_unread_not_empty(self):
        self.assertIsNone(self.read(commit=None))

    def test_an_unreadable_process_table_is_unread_not_empty(self):
        self.assertIsNone(self.read(table=None))

    def test_a_table_the_caller_already_read_is_used(self):
        with mock.patch.object(baseline, "_gpu_commit_by_pid", return_value=_COMMIT), \
             mock.patch.object(guard, "_process_table",
                               side_effect=AssertionError("read the table twice")):
            self.assertEqual(baseline.competing_gpu_processes(list(_TABLE)),
                             ["dwm.exe:heavy", "houdini.exe:heavy"])

    def test_counter_instances_are_summed_per_pid_across_adapters(self):
        fake = mock.Mock()
        fake.GetFormattedCounterArray.return_value = {
            "pid_10_luid_0x00000000_0x00016C55_phys_0": 600 * 2**20,
            "pid_10_luid_0x00000000_0x0001818D_phys_0": 500 * 2**20,
            "pid_11_luid_0x00000000_0x00016C55_phys_0": 100 * 2**20,
            "_Total": 1,
        }
        with mock.patch.dict(sys.modules, {"win32pdh": fake}):
            self.assertEqual(baseline._gpu_commit_by_pid(), {10: 1100.0, 11: 100.0})

    def test_a_counter_error_is_unread(self):
        fake = mock.Mock()
        fake.CollectQueryData.side_effect = OSError("PDH_NO_DATA")
        with mock.patch.dict(sys.modules, {"win32pdh": fake}):
            self.assertIsNone(baseline._gpu_commit_by_pid())

    def test_no_win32pdh_is_unread(self):
        with mock.patch.dict(sys.modules, {"win32pdh": None}):
            self.assertIsNone(baseline._gpu_commit_by_pid())


class TestCompetingConditions(_Isolated):
    SAMPLES = [10.0, 10.2, 9.9, 10.1, 10.05]

    def test_capture_records_how_the_set_was_read(self):
        self.assertEqual(Conditions.capture(self.obj).competing_method,
                         baseline.COMPETING_METHOD)

    def test_unread_competing_processes_never_match(self):
        a, b = Conditions.capture(self.obj), Conditions.capture(self.obj)
        a.competing_processes = b.competing_processes = None
        self.assertIn("competing_processes", a.unread())
        self.assertIn("competing_processes: unread (both)", a.differences(b))
        b.competing_processes = []
        self.assertIn("competing_processes: unread (baseline)", a.differences(b))

    def test_empty_is_a_reading_and_matches_empty(self):
        a, b = Conditions.capture(self.obj), Conditions.capture(self.obj)
        self.assertEqual((a.competing_processes, b.competing_processes), ([], []))
        self.assertNotIn("competing_processes", a.unread())
        self.assertEqual(a.differences(b), [])

    def test_the_process_set_is_compared_as_a_set(self):
        a, b = Conditions.capture(self.obj), Conditions.capture(self.obj)
        a.competing_processes = ["houdini.exe", "dwm.exe"]
        b.competing_processes = ["dwm.exe", "Houdini.exe"]
        self.assertEqual(a.differences(b), [])

    def test_a_heavy_app_arriving_invalidates(self):
        a, b = Conditions.capture(self.obj), Conditions.capture(self.obj)
        a.competing_processes, b.competing_processes = ["dwm.exe"], ["dwm.exe", "houdini.exe"]
        self.assertTrue(any(d.startswith("competing_processes:") for d in a.differences(b)))

    def test_a_heavy_python_job_invalidates_a_baseline_taken_beside_a_stray(self):
        a, b = Conditions.capture(self.obj), Conditions.capture(self.obj)
        with mock.patch.object(baseline.os, "getpid", return_value=ME),              mock.patch.object(baseline, "_gpu_commit_by_pid",
                               return_value={1: 0.0, 2: 15000.0}):
            a.competing_processes = _read_competing([(1, "python.exe")])
            b.competing_processes = _read_competing([(1, "python.exe"), (2, "python.exe")])
        self.assertTrue(any(d.startswith("competing_processes:") for d in a.differences(b)))

    def test_a_baseline_from_the_first_commit_method_is_refused(self):
        """v1 stored heavy and idle python.exe alike; its set cannot be compared."""
        old = asdict(Conditions.capture(self.obj))
        old["competing_method"] = f"gpu-commit>={baseline.HEAVY_GPU_COMMIT_MIB}MiB+significant-apps"
        b = Baseline(objective_id="diffusion:x", workload=asdict(self.obj.workload),
                     metric=self.obj.primary.name,
                     direction=self.obj.primary.direction.value,
                     samples=list(self.SAMPLES), conditions=old)
        ok, why = b.valid_for(self.obj, Conditions.capture(self.obj))
        self.assertFalse(ok)
        self.assertIn("competing_method", why)

    def test_record_refusal_names_the_missing_counter_module(self):
        with mock.patch.object(baseline, "competing_gpu_processes", return_value=None):
            with self.assertRaises(ValueError) as cm:
                baseline.record(self.obj, self.SAMPLES)
        self.assertIn("win32pdh", str(cm.exception))

    def test_a_baseline_from_the_old_nvidia_smi_list_is_refused(self):
        old = asdict(Conditions.capture(self.obj))
        del old["competing_method"]
        old["competing_processes"] = []          # what a failed nvidia-smi stored
        b = Baseline(objective_id="diffusion:x", workload=asdict(self.obj.workload),
                     metric=self.obj.primary.name,
                     direction=self.obj.primary.direction.value,
                     samples=list(self.SAMPLES), conditions=old)
        ok, why = b.valid_for(self.obj, Conditions.capture(self.obj))
        self.assertFalse(ok)
        self.assertIn("competing_method: unread (baseline)", why)

    def test_record_refuses_when_the_process_set_is_unread(self):
        with mock.patch.object(baseline, "competing_gpu_processes", return_value=None):
            with self.assertRaises(ValueError) as cm:
                baseline.record(self.obj, self.SAMPLES)
        self.assertIn("competing_processes", str(cm.exception))
        self.assertFalse(self.dir.exists() and any(self.dir.glob("*.json")))

    def test_capture_passes_a_shared_table_through(self):
        seen = []
        with mock.patch.object(baseline, "competing_gpu_processes",
                               side_effect=lambda table=None: seen.append(table) or []):
            Conditions.capture(self.obj, table=_TABLE)
        self.assertEqual(seen, [_TABLE])


class TestGuardrailBaseline(_Isolated):
    SAMPLES = [10.0, 10.2, 9.9, 10.1, 10.05]

    def test_guardrail_series_are_recorded_and_round_trip(self):
        b = baseline.record(self.obj, self.SAMPLES,
                            guardrail_samples={"lpips": [0.2, 0.21], "clip": [0.3, 0.31]})
        back = Baseline.load(b.path())
        self.assertEqual(back.guardrail_samples,
                         {"lpips": [0.2, 0.21], "clip": [0.3, 0.31]})
        self.assertEqual(back, b)

    def test_a_baseline_file_without_guardrails_still_loads(self):
        b = baseline.record(self.obj, self.SAMPLES)
        raw = json.loads(b.path().read_text(encoding="utf-8"))
        raw.pop("guardrail_samples", None)
        b.path().write_text(json.dumps(raw), encoding="utf-8")
        self.assertEqual(Baseline.load(b.path()).guardrail_samples, {})

    def test_an_undeclared_guardrail_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            baseline.record(self.obj, self.SAMPLES, guardrail_samples={"ssim": [0.9, 0.9]})
        self.assertIn("not declared", str(cm.exception))
        self.assertFalse(self.dir.exists() and any(self.dir.glob("*.json")))

    def test_a_one_sample_guardrail_is_refused(self):
        with self.assertRaises(ValueError):
            baseline.record(self.obj, self.SAMPLES, guardrail_samples={"lpips": [0.2]})

    def test_the_guardrail_blind_compare_path_is_gone(self):
        # Was: baseline.compare() called objective.admits without guardrails — a
        # second admission path beside Gate.evaluate that nothing called.
        self.assertFalse(hasattr(baseline, "compare"))


if __name__ == "__main__":
    unittest.main()
