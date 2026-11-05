"""
Process identity: the Win32_Process reader, the deterministic rules, and the
narrowed Jev residue (display only).

Order of authority, which these tests pin:

  1. the reader — Win32_Process (pid, parent, path, command line, start time),
     tasklist when that fails, None when both fail (fail closed);
  2. deterministic rules — image names, ComfyUI resolved on disk, competing
     optimizers read off cmd.exe / python.exe command lines. These MAY block;
  3. Jev — only python.exe / pythonw.exe rows the rules left unresolved, never
     V12's own lineage, never a cmd.exe or PowerShell command line. A label is a
     'jev (not evidence):' display line and changes nothing: not gpu_apps, not a
     collision, not a note, not `ok`.

No network: jev's SDK import seam is a fake whose client records what it was
sent. No real reader: `subprocess.run` / `guard._process_table` are patched in
every test, and every Ledger lives in a temp dir.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12 import cli, guard, jev, plane  # noqa: E402
from v12.baseline import COMPETING_METHOD, Conditions  # noqa: E402
from v12.change import Change, ChangeKind, EvidenceClass, Snapshot  # noqa: E402
from v12.ledger import Ledger  # noqa: E402

ME, PARENT = 4242, 4000
T0 = "2026-10-01T10:00:00.0000000Z"
DUMMY_KEY = "dummy-not-a-real-key"


def row(pid, name, cmd=None, *, ppid=1, exe=None, created=T0) -> guard.ProcessRow:
    return guard.ProcessRow(pid=pid, name=name, ppid=ppid, exe=exe, cmdline=cmd,
                            created=created)


def table(*rows, source="cim", elevated=True) -> guard.ProcessTable:
    return guard.ProcessTable(rows, source=source, elevated=elevated)


#: V12 itself, under the launcher cmd.exe, as the reader would see it.
SELF_ROWS = (
    row(PARENT, "cmd.exe", r'C:\WINDOWS\system32\cmd.exe /c ""C:\Opt\V12-Optimize-This-Machine.bat""'),
    row(ME, "python.exe", r"C:\Python314\python.exe -m v12 preflight", ppid=PARENT,
        exe=r"C:\Python314\python.exe"),
)
TRAIN = row(5000, "python.exe", r"C:\Python314\python.exe train.py --epochs 3",
            exe=r"C:\Python314\python.exe")


# ---------------------------------------------------------------------------
# a fake SDK: the names jev uses, no transport
# ---------------------------------------------------------------------------
class _Q:
    def __init__(self, **kw):
        self.kw = kw


class _ConnectionError(Exception):
    pass


class _FakeClient:
    """Answers every question with one label and records every state it is sent."""

    def __init__(self, label="general_utility", conf=0.99, model=jev.PINNED_MODEL, exc=None):
        self.label, self.conf, self.model, self.exc = label, conf, model, exc
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((json.loads(json.dumps(state)), dict(questions)))
        if self.exc is not None:
            raise self.exc
        ans = SimpleNamespace(choice=self.label, confidence=self.conf,
                              probabilities={self.label: 0.99, "none": 0.01})
        return SimpleNamespace(model=self.model, request_id="req-1", nouls={}, scores={},
                               choices={q: ans for q in questions})


def _sdk(client):
    class SDK:
        Choice = type("Choice", (_Q,), {})
        Noul = type("Noul", (_Q,), {})
        Score = type("Score", (_Q,), {})

        class RetryPolicy:
            def __init__(self, max_retries=2):
                self.max_retries = max_retries

        @staticmethod
        def TypeSafeClient(**kw):
            return client
    return SDK


class _Case(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(guard.os, "getpid", return_value=ME),
                  mock.patch.object(guard.os, "getppid", return_value=PARENT)):
            p.start()
            self.addCleanup(p.stop)
        jev._reset()
        self.addCleanup(jev._reset)
        self.tmp = tempfile.mkdtemp(prefix="v12_proclabels_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def jev_on(self, client):
        """Jev switched on with a dummy key and the fake SDK. Returns the client."""
        for p in (mock.patch.dict(os.environ, {jev.SWITCH_ENV: "on", jev.KEY_ENV: DUMMY_KEY}),
                  mock.patch.object(jev, "_import_sdk", return_value=_sdk(client)),
                  mock.patch.object(jev, "_sdk_importable", return_value=True)):
            p.start()
            self.addCleanup(p.stop)
        return client

    def jev_off(self):
        """Switch unset; the SDK seam fails the test if anything reaches it."""
        env = {k: v for k, v in os.environ.items() if k != jev.SWITCH_ENV}
        for p in (mock.patch.dict(os.environ, env, clear=True),
                  mock.patch.object(jev, "_import_sdk",
                                    side_effect=AssertionError("SDK reached with Jev off"))):
            p.start()
            self.addCleanup(p.stop)

    def preflight_out(self, tbl, quiet=False):
        buf = io.StringIO()
        with mock.patch.object(guard, "_process_table", return_value=tbl), \
                contextlib.redirect_stdout(buf):
            rc = cli.cmd_preflight(argparse.Namespace(quiet=quiet))
        return rc, buf.getvalue()

    def ledger(self) -> Ledger:
        led = Ledger(db_path=Path(self.tmp) / "ledger.db")
        self.addCleanup(led.close)
        return led


# ---------------------------------------------------------------------------
# 1. the reader
# ---------------------------------------------------------------------------
def _ps(stdout, rc=0):
    return mock.Mock(returncode=rc, stdout=stdout)


CIM_ROWS = [
    {"Name": "System", "ProcessId": 4, "ParentProcessId": 0, "ExecutablePath": None,
     "CommandLine": None, "CreationDate": T0},
    {"Name": "Python.exe", "ProcessId": 5000, "ParentProcessId": 77,
     "ExecutablePath": r"C:\Python314\python.exe",
     "CommandLine": r'"C:\Python314\python.exe" train.py --tag "a, b"', "CreationDate": T0},
]


class TestReader(_Case):
    def test_win32_process_rows_carry_parent_path_command_line_and_start(self):
        out = ("\ufeff" + json.dumps(CIM_ROWS)).encode("utf-8")
        with mock.patch.object(guard.subprocess, "run", return_value=_ps(out)) as run, \
                mock.patch.object(guard, "_is_elevated", return_value=True):
            t = guard._process_table()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "powershell")
        self.assertIn("Win32_Process", run.call_args.args[0][-1])
        self.assertEqual(list(t), [(4, "system"), (5000, "python.exe")])
        self.assertEqual(t.source, "cim")
        self.assertIs(t.elevated, True)
        r = guard.rows_of(t)[5000]
        self.assertEqual((r.ppid, r.exe, r.created), (77, r"C:\Python314\python.exe", T0))
        self.assertEqual(r.cmdline, r'"C:\Python314\python.exe" train.py --tag "a, b"')
        self.assertIsNone(guard.rows_of(t)[4].cmdline)

    def test_one_process_printed_as_an_object_is_still_a_table(self):
        with mock.patch.object(guard.subprocess, "run",
                               return_value=_ps(json.dumps(CIM_ROWS[1]).encode())), \
                mock.patch.object(guard, "_is_elevated", return_value=True):
            self.assertEqual(list(guard._process_table()), [(5000, "python.exe")])

    def test_tasklist_is_the_fallback_and_says_what_it_cannot_see(self):
        tasklist = '"cmd.exe","600","Console","1","4,000 K"\n"houdini.exe","700","Console","1","9 K"\n'
        with mock.patch.object(guard.subprocess, "run",
                               side_effect=[_ps(b"", rc=1), _ps(tasklist)]), \
                mock.patch.object(guard, "_is_elevated", return_value=True), \
                self.assertLogs("v12.guard", level="WARNING"):
            t = guard._process_table()
        self.assertEqual(list(t), [(600, "cmd.exe"), (700, "houdini.exe")])
        self.assertEqual(t.source, "tasklist")
        self.assertIsNone(guard.rows_of(t)[600].cmdline)
        ok, _, notes = guard.preflight(table=t)
        self.assertTrue(ok)
        self.assertIn("command lines unread (Win32_Process unavailable, tasklist fallback) "
                      "- competing-optimizer check incomplete", notes)

    def test_both_readers_failing_is_none_and_blocks(self):
        with mock.patch.object(guard.subprocess, "run", side_effect=OSError("gone")), \
                self.assertLogs("v12.guard", level="WARNING"):
            self.assertIsNone(guard._process_table())
            ok, coll, notes = guard.preflight()
        self.assertFalse(ok)
        self.assertTrue(notes[0].startswith(guard.PROCESS_TABLE_UNREADABLE))

    def test_garbage_from_powershell_falls_back_rather_than_reading_empty(self):
        with mock.patch.object(guard.subprocess, "run",
                               side_effect=[_ps(b"not json"), _ps("")]), \
                self.assertLogs("v12.guard", level="WARNING"):
            self.assertIsNone(guard._process_table())


class TestUnelevatedNote(_Case):
    NOTE = "3 processes unreadable (not elevated) - competing-optimizer check incomplete"

    def rows(self):
        return (row(10, "cmd.exe"), row(11, "python.exe"), row(12, "svchost.exe"),
                row(13, "notepad.exe", "notepad.exe a.txt"))

    def test_unelevated_with_unreadable_command_lines_says_so(self):
        for quiet in (False, True):
            with self.subTest(require_quiet=quiet):
                ok, _, notes = guard.preflight(require_quiet=quiet,
                                               table=table(*self.rows(), elevated=False))
                self.assertIn(self.NOTE, notes)

    def test_the_note_changes_no_verdict(self):
        ok, coll, _ = guard.preflight(table=table(*self.rows(), elevated=False))
        self.assertTrue(ok)
        self.assertEqual(coll, [])

    def test_no_note_when_elevated_or_unknown_or_nothing_unreadable(self):
        for t in (table(*self.rows(), elevated=True), table(*self.rows(), elevated=None),
                  table(row(13, "notepad.exe", "notepad.exe"), elevated=False),
                  [(10, "cmd.exe")]):
            with self.subTest(t=t):
                _, _, notes = guard.preflight(table=t)
                self.assertFalse([n for n in notes if "not elevated" in n], notes)


# ---------------------------------------------------------------------------
# 2. deterministic rules
# ---------------------------------------------------------------------------
class TestCommandLineOptimizers(_Case):
    FOUND = {
        "watch bat": ("cmd.exe", r'C:\WINDOWS\system32\cmd.exe /c ""C:\Opt\Watch-This-Machine.bat""'),
        "mos2026 bat": ("cmd.exe", r'cmd.exe /c "C:\Opt\MachineOptimizer2026\Optimize-This-Machine.bat"'),
        "v11 bat": ("cmd.exe", r'cmd.exe /c "C:\Opt\V11-Optimize-This-Machine.bat"'),
        "legacy bat": ("cmd.exe", r"cmd /c optimize_rtx4090_vfx.bat"),
        "launch_optimizer": ("cmd.exe", r"cmd /c launch_optimizer.bat --fast"),
        "v9_optimizer": ("cmd.exe", r'cmd /c "C:\Opt\v9_optimizer.bat"'),
        "mos2026.py": ("python.exe", r"python C:\Opt\MachineOptimizer2026\mos2026.py --watch"),
        "mos watchdog": ("python.exe", r"venv\Scripts\python.exe -m optimizer.orchestrator --watch"),
        "v9 apply": ("pythonw.exe", r"pythonw.exe -X utf8 -m v9 apply --top 3"),
    }
    NOT_FOUND = {
        "v12 launcher": ("cmd.exe", r'cmd.exe /c "C:\Opt\V12-Optimize-This-Machine.bat"'),
        "v9 read-only": ("python.exe", r"python -m v9 status"),
        "mcp shell": ("cmd.exe", r"cmd.exe /d /s /c mcp-remote https://x --header X-API-Key:${K}"),
        "not a host": ("notepad.exe", r"notepad.exe C:\Opt\Watch-This-Machine.bat"),
    }

    def test_a_bat_or_python_optimizer_is_found_and_blocks(self):
        for label, (image, cmd) in self.FOUND.items():
            for quiet in (False, True):
                with self.subTest(label, require_quiet=quiet):
                    ok, coll, notes = guard.preflight(
                        require_quiet=quiet, table=table(*SELF_ROWS, row(900, image, cmd)))
                    self.assertFalse(ok)
                    self.assertEqual([(c.kind, c.pid) for c in coll if c.kind == "optimizer"],
                                     [("optimizer", 900)])
                    self.assertTrue(any(n.startswith("Another optimizer is running")
                                        for n in notes))

    def test_look_alikes_and_v12s_own_launcher_are_not(self):
        for label, (image, cmd) in self.NOT_FOUND.items():
            with self.subTest(label):
                self.assertEqual(guard.find_competing_optimizers(
                    table(*SELF_ROWS, row(901, image, cmd))), [])

    def test_v12s_own_launcher_chain_is_excluded(self):
        """An ancestor waits on V12; it is not running beside it."""
        rows = (row(3000, "cmd.exe", r"cmd /c launch_optimizer.bat"),
                row(PARENT, "cmd.exe", r"cmd /c call v12.bat", ppid=3000),
                row(ME, "python.exe", r"python -m v12 run", ppid=PARENT))
        self.assertEqual(guard.find_competing_optimizers(table(*rows)), [])

    def test_a_reused_ancestor_pid_running_an_optimizer_is_not_exempt(self):
        """V12's launcher chain ends at a dead pid (explorer's parent, on this
        box). A cmd.exe that later took that pid is not V12's ancestor: it was
        created after its supposed child."""
        later = "2026-10-01T11:00:00.0000000Z"
        rows = (row(3000, "cmd.exe", r'cmd /c "C:\Opt\Watch-This-Machine.bat"', created=later),
                row(PARENT, "cmd.exe", r"cmd /c call v12.bat", ppid=3000),
                row(ME, "python.exe", r"python -m v12 run", ppid=PARENT))
        self.assertEqual([(c.kind, c.pid) for c in guard.find_competing_optimizers(table(*rows))],
                         [("optimizer", 3000)])

    def test_a_reused_parent_pid_from_getppid_is_not_exempt(self):
        later = "2026-10-01T11:00:00.0000000Z"
        rows = (row(PARENT, "cmd.exe", r'cmd /c "C:\Opt\Watch-This-Machine.bat"', created=later),
                row(ME, "python.exe", r"python -m v12 run", ppid=PARENT))
        self.assertEqual([(c.kind, c.pid) for c in guard.find_competing_optimizers(table(*rows))],
                         [("optimizer", PARENT)])

    def test_an_ancestor_without_a_creation_date_is_not_exempt(self):
        rows = (row(3000, "cmd.exe", r"cmd /c launch_optimizer.bat", created=None),
                row(PARENT, "cmd.exe", r"cmd /c call v12.bat", ppid=3000),
                row(ME, "python.exe", r"python -m v12 run", ppid=PARENT))
        self.assertEqual([c.pid for c in guard.find_competing_optimizers(table(*rows))], [3000])

    def test_a_cmd_hosted_optimizer_blocks_a_live_cycle(self):
        pl = _plane(self, table(*SELF_ROWS, row(902, "cmd.exe",
                                                r'cmd /c "C:\Opt\Watch-This-Machine.bat"')))
        pl.dry_run = False
        pl._lock = SimpleNamespace(acquire=lambda: (True, "stub lock"), release=lambda: None)
        r = pl.run_cycle(candidates=[_change("a")])
        self.assertTrue(r.blocked)
        self.assertIn("COMPETING OPTIMIZER: cmd.exe (pid 902)", r.notes)


class TestComfyUIResolution(_Case):
    def layout(self, app="studio", nodes=True):
        root = Path(self.tmp) / "portable"
        (root / "python_embeded").mkdir(parents=True)
        (root / app / "comfy").mkdir(parents=True)
        (root / app / "main.py").write_text("")
        if nodes:
            (root / app / "nodes.py").write_text("")
        return root

    def py(self, root, cmd):
        exe = str(root / "python_embeded" / "python.exe")
        return row(6000, "python.exe", cmd, exe=exe)

    def test_portable_relative_main_py_is_resolved(self):
        """The portable build's own launcher form. The folder is deliberately not
        called ComfyUI, so only resolution on disk can identify it."""
        root = self.layout()
        for cmd in (r".\python_embeded\python.exe -s studio\main.py --windows-standalone-build",
                    r"python_embeded\python.exe python_embeded\..\studio\main.py",
                    f'"{root}\\python_embeded\\python.exe" "{root}\\studio\\main.py"'):
            with self.subTest(cmd=cmd):
                r = self.py(root, cmd)
                self.assertEqual(guard.comfyui_dir(r), str(root / "studio"))
                self.assertEqual(guard.identify(r), f"ComfyUI ({root / 'studio'})")
                self.assertEqual(guard.unresolved_python(table(r)), [])

    def test_a_main_py_without_comfy_and_nodes_py_is_not_comfyui(self):
        root = self.layout(nodes=False)
        r = self.py(root, r".\python_embeded\python.exe -s studio\main.py")
        self.assertIsNone(guard.comfyui_dir(r))
        self.assertIsNone(guard.identify(r))
        self.assertEqual(guard.unresolved_python(table(r)), [r])

    def test_mos2026s_command_line_rule_is_the_fallback(self):
        r = row(6001, "python.exe", r"python Z:\nowhere\ComfyUI\main.py --listen")
        self.assertEqual(guard.identify(r), "ComfyUI (command line)")

    def test_a_resolved_comfyui_python_is_still_a_gpu_app_by_image(self):
        """Identity routes the Jev residue; it does not rename gpu_apps."""
        root = self.layout()
        r = self.py(root, r".\python_embeded\python.exe -s studio\main.py")
        self.assertEqual([c.name for c in guard.find_gpu_apps(table(r))], ["python.exe"])


class TestResidue(_Case):
    def test_only_unresolved_python_outside_v12s_lineage(self):
        child = row(ME + 1, "python.exe", "python worker.py", ppid=ME)
        grandchild = row(ME + 2, "pythonw.exe", "pythonw helper.py", ppid=ME + 1)
        rows = (*SELF_ROWS, child, grandchild, TRAIN,
                row(7001, "cmd.exe", "cmd /c build.bat"),
                row(7002, "powershell.exe", "powershell -c Get-Date"),
                row(7003, "pwsh.exe", "pwsh -c Get-Date"),
                row(7004, "python.exe", None),                      # unreadable
                row(7005, "houdini.exe", "houdini.exe scene.hip"),
                row(7006, "python.exe", r"python C:\x\ComfyUI\main.py"))
        self.assertEqual([r.pid for r in guard.unresolved_python(table(*rows))], [5000])

    def test_a_bare_pair_table_has_no_residue(self):
        self.assertEqual(guard.unresolved_python([(5000, "python.exe")]), [])


# ---------------------------------------------------------------------------
# 3. the Jev residue — display only
# ---------------------------------------------------------------------------
class TestAnnotate(_Case):
    def test_off_asks_nothing_and_prints_nothing(self):
        self.jev_off()
        self.assertEqual(plane.annotate_processes(table(*SELF_ROWS, TRAIN)), [])

    def test_a_label_is_bound_to_pid_and_start_time_and_says_not_evidence(self):
        client = self.jev_on(_FakeClient())
        led = self.ledger()
        out = plane.annotate_processes(table(*SELF_ROWS, TRAIN), ledger=led)
        self.assertEqual(out, [f"jev (not evidence): python.exe pid 5000 (started {T0}): "
                               f"general_utility (0.99)"])
        self.assertEqual(len(client.calls), 1)
        recent = led.judgments_summary()["recent"]
        self.assertEqual([(r["subject_key"], r["label"], r["consumed_as"]) for r in recent],
                         [(f"process:5000@{T0}", "general_utility", "display")])

    def test_no_cmd_or_powershell_content_ever_reaches_the_client(self):
        client = self.jev_on(_FakeClient())
        rows = (*SELF_ROWS,
                row(7001, "cmd.exe", "cmd /c MARK_CMD_SECRET.bat"),
                row(7002, "powershell.exe", "powershell -c MARK_PS_SECRET"),
                row(7003, "pwsh.exe", "pwsh -c MARK_PWSH_SECRET"),
                row(5000, "python.exe", "python train.py", ppid=7001,
                    exe=r"C:\Python314\python.exe"))
        plane.annotate_processes(table(*rows))
        self.assertEqual(len(client.calls), 1)
        sent = json.dumps([state for state, _ in client.calls])
        for mark in ("MARK_CMD", "MARK_PS", "MARK_PWSH", "v12 preflight", "V12-Optimize"):
            self.assertNotIn(mark, sent)
        self.assertEqual(client.calls[0][0]["image_name"], "python.exe")
        self.assertEqual(set(client.calls[0][0]), {"image_name", "executable_path",
                                                    "command_line"})

    def test_the_command_line_is_redacted_before_it_is_sent(self):
        client = self.jev_on(_FakeClient())
        plane.annotate_processes(table(row(5000, "python.exe",
                                           "python train.py --api-key sk-live-123")))
        self.assertNotIn("sk-live-123", json.dumps(client.calls[0][0]))

    def test_offline_stops_after_the_first_row(self):
        client = self.jev_on(_FakeClient(exc=_ConnectionError("down")))
        rows = [row(5000 + i, "python.exe", f"python job{i}.py") for i in range(3)]
        out = plane.annotate_processes(table(*rows))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(out, [f"jev (not evidence): python.exe pid 5000 (started {T0}): "
                               f"unlabelled (offline: _ConnectionError)"])

    def test_rows_beyond_the_limit_are_not_asked_and_are_counted(self):
        client = self.jev_on(_FakeClient())
        n = plane.MAX_ANNOTATED_PROCESSES + 4
        out = plane.annotate_processes(
            table(*[row(5000 + i, "python.exe", f"python job{i}.py") for i in range(n)]))
        self.assertEqual(len(client.calls), plane.MAX_ANNOTATED_PROCESSES)
        self.assertEqual(out[-1], f"jev (not evidence): 4 more python process(es) not "
                                  f"asked (limit {plane.MAX_ANNOTATED_PROCESSES})")

    def test_no_label_can_read_like_evidence(self):
        self.assertFalse(set(plane.PROCESS_KINDS) & jev.RESERVED_LABELS)
        self.assertIn("none", plane.PROCESS_KINDS)


class TestCliPreflight(_Case):
    def test_require_quiet_with_unresolved_python_still_blocks_whatever_jev_says(self):
        client = self.jev_on(_FakeClient("general_utility", 0.99))
        rc, out = self.preflight_out(table(*SELF_ROWS, TRAIN), quiet=True)
        self.assertEqual(rc, 1)
        self.assertIn("BLOCKED", out)
        self.assertIn("[gpu-app] python.exe (pid 5000)", out)
        self.assertNotIn("jev (not evidence)", out)
        self.assertEqual(client.calls, [], "Jev was asked about a blocked preflight")

    def test_a_clear_preflight_prints_the_label_after_the_decision(self):
        client = self.jev_on(_FakeClient("general_utility", 0.99))
        rc, out = self.preflight_out(table(*SELF_ROWS, TRAIN))
        self.assertEqual(rc, 0)
        lines = out.splitlines()
        self.assertIn(f"  jev (not evidence): python.exe pid 5000 (started {T0}): "
                      f"general_utility (0.99)", lines)
        # python.exe still counts as GPU-resident, exactly as without Jev.
        self.assertIn("  GPU-resident: python.exe (recorded in baseline conditions)", lines)
        self.assertEqual(len(client.calls), 1)

    def test_jev_adds_lines_and_changes_none(self):
        """With the switch unset the output is the pre-Jev output plus the jev
        status line; switched on, it is that plus annotation lines and nothing
        else changes."""
        t = table(*SELF_ROWS, TRAIN, row(700, "houdini.exe"))
        self.jev_off()
        rc_off, off = self.preflight_out(t)
        self.assertNotIn("jev (not evidence)", off)
        self.assertIn("  jev: off (set V12_JEV=on to enable)", off.splitlines())
        stop = mock.patch.dict(os.environ, {jev.SWITCH_ENV: "on", jev.KEY_ENV: DUMMY_KEY})
        client = _FakeClient()
        with stop, mock.patch.object(jev, "_import_sdk", return_value=_sdk(client)), \
                mock.patch.object(jev, "_sdk_importable", return_value=True):
            rc_on, on = self.preflight_out(t)
        self.assertEqual(rc_on, rc_off)
        kept = [ln for ln in on.splitlines()
                if not ln.startswith("  jev (not evidence)") and not ln.startswith("  jev: ")]
        self.assertEqual(kept, [ln for ln in off.splitlines() if not ln.startswith("  jev: ")])

    def test_terminal_escapes_in_a_row_never_reach_the_terminal(self):
        evil = row(5000, "python.exe", "python a.py \x1b[2K\x1b]0;pwned\x07",
                   created="2026\x1b[1A\x1b[2K preflight: CLEAR\r\x9b2J")
        for client in (_FakeClient(), _FakeClient(model="jev-1.13.0\x1b[2K")):
            with self.subTest(model=client.model):
                jev._reset()
                self.jev_on(client)
                rc, out = self.preflight_out(table(*SELF_ROWS, evil))
                line = [ln for ln in out.splitlines() if "jev (not evidence)" in ln]
                self.assertEqual(len(line), 1, out)
                for ch in ("\x1b", "\x07", "\r", "\x9b"):
                    self.assertNotIn(ch, out)


# ---------------------------------------------------------------------------
# the plane: OBSERVE fixes everything before a label exists
# ---------------------------------------------------------------------------
_STUB_CONDITIONS = Conditions(driver_version="stub", sm_clock_max_mhz=3105,
                              mem_clock_max_mhz=10501, power_limit_w=450.0,
                              hags_enabled=True, competing_processes=[],
                              competing_method=COMPETING_METHOD)


class _NoActuator:
    def __init__(self, dry_run=False):
        pass

    def _boom(self, *a, **k):
        raise AssertionError("an actuator was reached")

    apply = apply_undo = reset_default = verify = _boom


def _change(cid) -> Change:
    return Change(id=cid, kind=ChangeKind.ENV_VAR, description="d", target=f"VAR_{cid}",
                  evidence=EvidenceClass.CITED, evidence_source="docs",
                  snapshot=Snapshot(value="0", existed=True, trusted=True, source="test"))


_ORIGINAL_OBSERVATION = plane.Observation


def _plane(case: _Case, tbl, **kw) -> plane.Plane:
    obs = _ORIGINAL_OBSERVATION
    for p in (mock.patch.object(plane, "SystemActuator", _NoActuator),
              mock.patch.object(guard, "_process_table", return_value=tbl),
              mock.patch.object(Conditions, "capture", return_value=_STUB_CONDITIONS),
              mock.patch.object(plane, "Observation", side_effect=lambda: obs(
                  gpu=SimpleNamespace(summary="stub gpu", errors=[]),
                  roofline=SimpleNamespace(summary="stub roofline"))),
              mock.patch.dict(sys.modules, {"optimizer": None, "optimizer.base": None,
                                            "optimizer.probe": None,
                                            "optimizer.orchestrator": None})):
        p.start()
        case.addCleanup(p.stop)
    pl = plane.Plane(dry_run=True, ledger=case.ledger(), **kw)
    return pl


class TestObserve(_Case):
    def observe(self, on: bool, **kw):
        jev._reset()
        client = _FakeClient()
        ctx = (mock.patch.dict(os.environ, {jev.SWITCH_ENV: "on", jev.KEY_ENV: DUMMY_KEY})
               if on else mock.patch.dict(os.environ, {jev.SWITCH_ENV: ""}))
        with ctx, mock.patch.object(jev, "_import_sdk", return_value=_sdk(client)), \
                mock.patch.object(jev, "_sdk_importable", return_value=True):
            pl = _plane(self, table(*SELF_ROWS, TRAIN), **kw)
            return pl, pl.observe(), client

    def test_labels_change_no_control_flow_field(self):
        _, off, _ = self.observe(False)
        pl, on, client = self.observe(True)
        self.assertEqual(off.annotations, [])
        self.assertEqual(len(on.annotations), 1)
        self.assertTrue(on.annotations[0].startswith("jev (not evidence): python.exe pid 5000"))
        self.assertEqual((on.gpu_apps, on.notes, on.conditions),
                         (off.gpu_apps, off.notes, off.conditions))
        self.assertEqual(on.gpu_apps, ["python.exe"])
        # recorded in the judgments table, at OBSERVE, as display
        rows = pl.ledger.judgments_summary()["recent"]
        self.assertEqual([r["consumed_as"] for r in rows], ["display"])

    def test_a_quiet_observe_that_blocks_asks_nothing(self):
        _, obs, client = self.observe(True, require_quiet=True)
        self.assertEqual(obs.annotations, [])
        self.assertEqual(client.calls, [])
        self.assertEqual(obs.gpu_apps, ["python.exe"])

    def test_the_run_report_shows_labels_apart_from_notes(self):
        jev._reset()
        self.jev_on(_FakeClient())
        pl = _plane(self, table(*SELF_ROWS, TRAIN))
        r = pl.run_cycle(candidates=[_change("a")])
        self.assertEqual(len(r.annotations), 1)
        self.assertFalse([n for n in r.notes if "jev" in n])
        self.assertIn("  " + r.annotations[0], r.report().splitlines())


if __name__ == "__main__":
    unittest.main()
