"""
Characterization tests for guard — collision detection and preflight.

These pin what `v12/guard.py` does TODAY against fixed fake process tables, so a
refactor of the matching logic cannot silently change who blocks a run. Tests
named `..._today` pin current behaviour, known gaps included, so changing it is
a deliberate act; bugs fixed in a later step say so in the past tense ("was ...").

`_process_table()` is the only thing patched, except in the one test of its own
failure path, which patches `subprocess.run` beneath it. It returns `(pid, lowercase image
name)` — the real one lowercases tasklist output — so every fake row here is
lowercase too. A row with an uppercase name would test a state the real function
never produces.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12 import guard  # noqa: E402

ME = 4242          # stands in for os.getpid() in every test


class _GuardCase(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(guard.os, "getpid", return_value=ME)
        p.start()
        self.addCleanup(p.stop)
        # The WSL probe reads the LIVE machine unless it is stubbed, and this
        # box has a WSL distribution that sometimes holds the GPU. A test
        # must not read whatever the machine happens to be doing — that is
        # exactly the variable these tests exist to control.
        self._orig_wsl_gpu_apps = guard.wsl_gpu_apps
        w = mock.patch.object(guard, "wsl_gpu_apps", return_value=[])
        w.start()
        self.addCleanup(w.stop)

    def run_with(self, rows, fn, *a, **kw):
        with mock.patch.object(guard, "_process_table", return_value=list(rows)):
            return fn(*a, **kw)


# ---------------------------------------------------------------------------
# find_gpu_apps
# ---------------------------------------------------------------------------
class TestFindGpuApps(_GuardCase):
    def test_empty_table_finds_nothing(self):
        self.assertEqual(self.run_with([], guard.find_gpu_apps), [])

    def test_houdini_is_a_gpu_app(self):
        out = self.run_with([(100, "houdinifx.exe")], guard.find_gpu_apps)
        self.assertEqual([(c.kind, c.name, c.pid) for c in out],
                         [("gpu-app", "houdinifx.exe", 100)])

    def test_bare_python_counts_as_a_gpu_app_today(self):
        """Today: ANY python.exe is treated as GPU-resident, because 'python.exe'
        is in SIGNIFICANT_APPS. Pinned as-is; it is over-broad (an idle REPL
        blocks a quiet baseline), but over-broad here fails closed."""
        out = self.run_with([(200, "python.exe")], guard.find_gpu_apps)
        self.assertEqual([c.name for c in out], ["python.exe"])

    def test_unreal_editor_is_a_gpu_app(self):
        # was a typo bug ('unrealditor.exe'), fixed in step 1.
        out = self.run_with([(300, "unrealeditor.exe")], guard.find_gpu_apps)
        self.assertEqual([c.name for c in out], ["unrealeditor.exe"])

    def test_houdini_batch_and_unreal_helper_images_are_gpu_apps(self):
        """hython/husk (Houdini batch + USD render) and the Unreal commandlet and
        shader-compile workers all hold the GPU; each must be seen on its own."""
        for image in ("hython.exe", "husk.exe", "unrealeditor-cmd.exe",
                      "shadercompileworker.exe", "comfyui.exe"):
            with self.subTest(image=image):
                out = self.run_with([(310, image)], guard.find_gpu_apps)
                self.assertEqual([c.name for c in out], [image])

    def test_every_entry_is_an_image_name(self):
        """Matching is exact against tasklist image names, so an entry without an
        extension (the old bare 'comfyui') can never match anything."""
        for entry in guard.SIGNIFICANT_APPS:
            with self.subTest(entry=entry):
                self.assertEqual(entry, entry.lower())
                self.assertTrue(entry.endswith(".exe"))

    def test_an_explicit_table_is_used_instead_of_tasklist(self):
        with mock.patch.object(guard, "_process_table",
                               side_effect=AssertionError("table was fetched")):
            out = guard.find_gpu_apps([(320, "houdini.exe")])
        self.assertEqual([c.pid for c in out], [320])

    def test_own_pid_is_excluded(self):
        """V12 itself runs under python.exe; it must not count itself."""
        out = self.run_with([(ME, "python.exe"), (201, "python.exe")], guard.find_gpu_apps)
        self.assertEqual([c.pid for c in out], [201])

    def test_unrelated_processes_are_ignored(self):
        out = self.run_with([(1, "explorer.exe"), (2, "svchost.exe")], guard.find_gpu_apps)
        self.assertEqual(out, [])


# ---------------------------------------------------------------------------
# find_competing_optimizers
# ---------------------------------------------------------------------------
class TestFindCompetingOptimizers(_GuardCase):
    def test_empty_table_finds_nothing(self):
        self.assertEqual(self.run_with([], guard.find_competing_optimizers), [])

    def test_a_named_optimizer_image_is_found(self):
        out = self.run_with([(500, "machineoptimizer2026")], guard.find_competing_optimizers)
        self.assertEqual([(c.kind, c.pid) for c in out], [("optimizer", 500)])

    def test_substring_match_catches_variants(self):
        out = self.run_with([(501, "optimize_rtx4090_vfx_v2.exe")],
                            guard.find_competing_optimizers)
        self.assertEqual([c.pid for c in out], [501])

    def test_cmd_without_a_command_line_is_still_not_an_optimizer(self):
        """Was a gap: matching was on IMAGE name only, and tasklist reports a .bat
        optimizer as 'cmd.exe', so watch-this-machine.bat in a console was
        invisible. The command line is now read (tests/test_proclabels.py). A
        bare (pid, name) table carries none, and a cmd.exe with nothing to read
        is not guessed to be one."""
        out = self.run_with([(600, "cmd.exe")], guard.find_competing_optimizers)
        self.assertEqual(out, [])

    def test_cmd_hosting_a_bat_optimizer_is_seen_by_its_command_line(self):
        table = guard.ProcessTable([guard.ProcessRow(
            601, "cmd.exe", ppid=1,
            cmdline=r'C:\WINDOWS\system32\cmd.exe /c ""C:\Opt\Watch-This-Machine.bat" "')])
        out = guard.find_competing_optimizers(table)
        self.assertEqual([(c.kind, c.name, c.pid) for c in out],
                         [("optimizer", "cmd.exe", 601)])

    def test_own_pid_is_excluded(self):
        out = self.run_with([(ME, "machineoptimizer2026"), (502, "machineoptimizer2026")],
                            guard.find_competing_optimizers)
        self.assertEqual([c.pid for c in out], [502])

    def test_a_gpu_app_is_not_an_optimizer(self):
        out = self.run_with([(503, "houdini.exe")], guard.find_competing_optimizers)
        self.assertEqual(out, [])


# ---------------------------------------------------------------------------
# find_v9_processes
# ---------------------------------------------------------------------------
class TestFindV9Processes(_GuardCase):
    def test_empty_table_finds_nothing(self):
        self.assertEqual(self.run_with([], guard.find_v9_processes), [])

    def test_build_and_profiler_tools_are_found(self):
        rows = [(700, "nvcc.exe"), (701, "ncu.exe"), (702, "trtexec.exe"), (703, "cl.exe")]
        out = self.run_with(rows, guard.find_v9_processes)
        self.assertEqual([(c.kind, c.name) for c in out],
                         [("v9-tooling", "nvcc.exe"), ("v9-tooling", "ncu.exe"),
                          ("v9-tooling", "trtexec.exe")])

    def test_own_pid_is_not_excluded_today(self):
        """Today: unlike the other two finders, this one does not skip os.getpid().
        Harmless in practice (V12 is never nvcc.exe), pinned so a refactor that
        unifies the loops does it on purpose."""
        out = self.run_with([(ME, "nvcc.exe")], guard.find_v9_processes)
        self.assertEqual([c.pid for c in out], [ME])


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# WSL GPU work
# ---------------------------------------------------------------------------
class TestWslGpuApps(_GuardCase):
    """GPU work inside WSL is invisible to the Windows process table.

    Found the hard way on 2026-11-04: preflight reported CLEAR — nothing is
    competing for the GPU — on a card at 98% utilisation and 288 W. The work
    belonged to a python3.12 process inside a WSL Ubuntu, whose GPU memory is
    allocated through the vGPU path and never appears in the Windows process
    table that every other check here reads.

    A preflight that says CLEAR on a pinned card is worse than one that says
    nothing: it tells the operator the machine is free when it is not.
    """

    def setUp(self):
        super().setUp()
        # The base class stubs the WSL probe so no other test reads the live
        # machine. THESE tests are about the probe, so they get the real one
        # back — with subprocess.run mocked, never the actual wsl.exe.
        self._restore = mock.patch.object(guard, "wsl_gpu_apps",
                                          self._orig_wsl_gpu_apps)
        self._restore.start()
        self.addCleanup(self._restore.stop)

    def _wsl(self, stdout):
        return mock.Mock(returncode=0, stdout=stdout, stderr="")

    def test_a_wsl_compute_app_is_found(self):
        with mock.patch.object(guard.subprocess, "run", side_effect=[
                self._wsl("Ubuntu\n"),          # --list --running
                self._wsl("1870, /python3.12\n")  # nvidia-smi inside it
        ]):
            out = guard.wsl_gpu_apps()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, "wsl-gpu-app")
        self.assertIn("python3.12", out[0].name)
        self.assertIn("Ubuntu", out[0].name)

    def test_the_utf16_distribution_list_is_decoded(self):
        """`wsl --list` returns UTF-16 on this box, so the name arrives with
        its bytes separated by NULs. Read naively it looks like three short
        names, or like 'Ub'. Neither is the distribution."""
        raw = "U\x00b\x00u\x00n\x00t\x00u\x00\n\x00\n\x00"
        with mock.patch.object(guard.subprocess, "run", side_effect=[
                self._wsl(raw),
                self._wsl("1870, /python3.12\n")
        ]):
            out = guard.wsl_gpu_apps()
        self.assertEqual(len(out), 1)
        self.assertIn("Ubuntu", out[0].name)

    def test_a_stopped_distribution_is_never_started(self):
        """Invoking `wsl -d <name>` on a STOPPED distribution STARTS it. A
        check must never change the state it is checking, so only the
        RUNNING distributions are enumerated and queried."""
        calls = []

        def fake_run(cmd, *a, **kw):
            calls.append(cmd)
            if "--list" in cmd:
                return self._wsl("Ubuntu\n")
            return self._wsl("1870, /python3.12\n")

        with mock.patch.object(guard.subprocess, "run", side_effect=fake_run):
            guard.wsl_gpu_apps()
        self.assertTrue(any("--list" in c for c in calls))
        self.assertTrue(any("--running" in c for c in calls))
        for c in calls:
            if "-d" in c:
                # It may only name a distribution that was listed as running.
                self.assertIn(c[c.index("-d") + 1], ["Ubuntu"])

    def test_no_wsl_work_means_no_collisions(self):
        with mock.patch.object(guard.subprocess, "run", side_effect=[
                self._wsl("Ubuntu\n"),
                self._wsl("")            # nvidia-smi reports nothing
        ]):
            self.assertEqual(guard.wsl_gpu_apps(), [])

    def test_wsl_work_blocks_a_quiet_baseline(self):
        """require_quiet=True is what a baseline run uses. A card held by a
        WSL process is not a card you can take a comparable baseline on."""
        with mock.patch.object(guard, "_process_table", return_value=[]), \
             mock.patch.object(guard.subprocess, "run", side_effect=[
                 self._wsl("Ubuntu\n"),
                 self._wsl("1870, /python3.12\n")
             ]):
            ok, collisions, notes = guard.preflight(require_quiet=True)
        self.assertFalse(ok)
        self.assertTrue(any("wsl" in n.lower() for n in notes), notes)

    def test_wsl_work_is_named_even_when_not_blocking(self):
        """The non-quiet path is context, not a verdict — but the operator
        has to be told, because nothing else on this machine can see it."""
        with mock.patch.object(guard, "_process_table", return_value=[]), \
             mock.patch.object(guard.subprocess, "run", side_effect=[
                 self._wsl("Ubuntu\n"),
                 self._wsl("1870, /python3.12\n")
             ]):
            ok, collisions, notes = guard.preflight(require_quiet=False)
        self.assertTrue(any("WSL" in n for n in notes), notes)

    def test_we_do_not_probe_wsl_from_inside_wsl(self):
        """If V12 itself runs in WSL, the probe would find its own work."""
        with mock.patch.dict(guard.os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}):
            self.assertEqual(guard.wsl_gpu_apps(), [])


class TestPreflight(_GuardCase):
    def test_empty_table_is_clear_in_both_modes(self):
        for quiet in (False, True):
            with self.subTest(require_quiet=quiet):
                self.assertEqual(
                    self.run_with([], guard.preflight, require_quiet=quiet),
                    (True, [], []))

    def test_competing_optimizer_blocks_in_both_modes(self):
        for quiet in (False, True):
            with self.subTest(require_quiet=quiet):
                ok, coll, notes = self.run_with([(800, "machineoptimizer2026")],
                                                guard.preflight, require_quiet=quiet)
                self.assertFalse(ok)
                self.assertEqual([c.kind for c in coll], ["optimizer"])
                self.assertTrue(any("Another optimizer is running" in n for n in notes))

    def test_gpu_app_is_only_a_note_when_not_quiet(self):
        """Today: in default mode a resident GPU app does NOT block and is NOT
        returned as a collision — only named in a note. (The plane no longer
        derives Observation.gpu_apps from collisions; it calls find_gpu_apps on
        the same table, so this contract and gpu_apps are independent.)"""
        ok, coll, notes = self.run_with([(900, "houdini.exe")],
                                        guard.preflight, require_quiet=False)
        self.assertTrue(ok)
        self.assertEqual(coll, [])
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].startswith("GPU-resident: houdini.exe"))

    def test_a_table_the_caller_read_is_judged_without_rereading(self):
        with mock.patch.object(guard, "_process_table",
                               side_effect=AssertionError("re-read the table")):
            ok, coll, notes = guard.preflight(require_quiet=True,
                                              table=[(900, "houdini.exe")])
        self.assertFalse(ok)
        self.assertEqual([c.name for c in coll], ["houdini.exe"])

    def test_a_failed_read_passed_in_fails_closed(self):
        # None is "the read failed", not "read it for me".
        with mock.patch.object(guard, "_process_table", return_value=[]):
            ok, coll, notes = guard.preflight(table=None)
        self.assertFalse(ok)
        self.assertTrue(notes[0].startswith(guard.PROCESS_TABLE_UNREADABLE))

    def test_gpu_app_blocks_a_quiet_baseline(self):
        ok, coll, notes = self.run_with([(900, "houdini.exe")],
                                        guard.preflight, require_quiet=True)
        self.assertFalse(ok)
        self.assertEqual([(c.kind, c.name) for c in coll], [("gpu-app", "houdini.exe")])
        self.assertTrue(any("not quiet for a baseline" in n for n in notes))

    def test_bare_python_blocks_a_quiet_baseline_today(self):
        ok, coll, _ = self.run_with([(901, "python.exe")],
                                    guard.preflight, require_quiet=True)
        self.assertFalse(ok)
        self.assertEqual([c.name for c in coll], ["python.exe"])

    def test_own_python_does_not_block_a_quiet_baseline(self):
        self.assertEqual(
            self.run_with([(ME, "python.exe")], guard.preflight, require_quiet=True),
            (True, [], []))

    def test_v9_tooling_blocks_only_a_quiet_baseline(self):
        ok, coll, notes = self.run_with([(950, "nsys.exe")],
                                        guard.preflight, require_quiet=False)
        self.assertTrue(ok)
        self.assertEqual(coll, [])
        self.assertEqual(notes, [])
        ok, coll, notes = self.run_with([(950, "nsys.exe")],
                                        guard.preflight, require_quiet=True)
        self.assertFalse(ok)
        self.assertEqual([c.kind for c in coll], ["v9-tooling"])
        self.assertTrue(any("Build/profiler tooling" in n for n in notes))

    def test_unreal_editor_blocks_a_quiet_baseline(self):
        # was a typo bug ('unrealditor.exe'), fixed in step 1.
        ok, coll, _ = self.run_with([(960, "unrealeditor.exe")],
                                    guard.preflight, require_quiet=True)
        self.assertFalse(ok)
        self.assertEqual([c.name for c in coll], ["unrealeditor.exe"])

    def test_new_images_block_a_quiet_baseline(self):
        for image in ("hython.exe", "husk.exe", "unrealeditor-cmd.exe",
                      "shadercompileworker.exe"):
            with self.subTest(image=image):
                ok, coll, _ = self.run_with([(970, image)],
                                            guard.preflight, require_quiet=True)
                self.assertFalse(ok)
                self.assertEqual([c.name for c in coll], [image])

    def test_the_process_table_is_fetched_once_per_preflight(self):
        """All three finders must judge the same snapshot; three tasklist calls
        cost three process spawns and can each see a different machine."""
        rows = [(980, "machineoptimizer2026"), (981, "houdini.exe"), (982, "nsys.exe")]
        for quiet in (False, True):
            with self.subTest(require_quiet=quiet):
                with mock.patch.object(guard, "_process_table",
                                       return_value=list(rows)) as table:
                    guard.preflight(require_quiet=quiet)
                self.assertEqual(table.call_count, 1)

    def test_the_finders_still_fetch_their_own_table_when_called_bare(self):
        with mock.patch.object(guard, "_process_table",
                               return_value=[(990, "nvcc.exe")]) as table:
            self.assertEqual([c.pid for c in guard.find_v9_processes()], [990])
            self.assertEqual(guard.find_competing_optimizers(), [])
            self.assertEqual(guard.find_gpu_apps(), [])
        self.assertEqual(table.call_count, 3)

    def test_an_unreadable_process_table_blocks_instead_of_reading_clear(self):
        """Was fail-open: a tasklist failure returned [] and preflight said CLEAR
        about a machine it never saw. Fail closed: BLOCKED, with the reason."""
        for failure in (OSError("tasklist missing"),
                        mock.Mock(returncode=1, stdout=""),
                        mock.Mock(returncode=0, stdout="")):
            for quiet in (False, True):
                with self.subTest(failure=failure, require_quiet=quiet):
                    kw = ({"side_effect": failure} if isinstance(failure, Exception)
                          else {"return_value": failure})
                    with mock.patch.object(guard.subprocess, "run", **kw), \
                         self.assertLogs("v12.guard", level="WARNING"):
                        ok, coll, notes = guard.preflight(require_quiet=quiet)
                    self.assertFalse(ok)
                    self.assertEqual(coll, [])
                    self.assertTrue(any("process table unreadable" in n for n in notes),
                                    notes)

    def test_the_process_table_is_read_without_a_real_tasklist(self):
        """Guard against a test in this file shelling out: every finder above
        goes through the patched table."""
        with mock.patch.object(guard.subprocess, "run",
                               side_effect=AssertionError("tasklist was called")):
            self.run_with([], guard.preflight, require_quiet=True)


if __name__ == "__main__":
    unittest.main()
