"""
Tests for the ledger's read side — the rows the CLI prints and rolls back from.

The bug these exist for: `recent_applied` SELECTed six hand-picked columns, while
`v12 ledger` indexes r['reverted'] and `v12 rollback` indexes r['reverted'],
r['tier'] and r['change_key']. sqlite3.Row raises IndexError on a key the SELECT
left out, so both commands worked on an empty ledger and crashed on the first
real application — the exact moment an operator needs rollback to work.

Every ledger here lives in a temp dir. Nothing touches D:\\OptimizerV12\\ledger.db.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12 import cli  # noqa: E402
from v12.admission import Verdict  # noqa: E402
from v12.change import ChangeKind, EvidenceClass, Tier, change_key, make  # noqa: E402
from v12.ledger import Ledger  # noqa: E402

#: Every key cli.py reads off a recent_applied row (cmd_ledger + cmd_rollback).
CLI_KEYS = ("change_id", "evidence", "effect", "reverted", "tier", "change_key")


def _change(cid: str = "env_cuda_cache", target: str = "CUDA_CACHE_MAXSIZE"):
    return make(ChangeKind.ENV_VAR, cid, "test change", target,
                target_value="4294967296", tier=Tier.SYSTEM)


def _verdict(c) -> Verdict:
    return Verdict(admitted=True, change_key=change_key(c), change_id=c.id,
                   reason="test", evidence=EvidenceClass.CITED)


class _TempLedger(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="v12ledger_"))
        self.lg = Ledger(db_path=self.dir / "ledger.db", session="test-session")

    def tearDown(self) -> None:
        self.lg.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _record(self, cid: str = "env_cuda_cache", effect=None,
                target: str = "CUDA_CACHE_MAXSIZE"):
        c = _change(cid, target)
        self.lg.record_applied(c, _verdict(c), landed=True, landed_source="read-back",
                               effect=effect)
        return c


class TestRecentAppliedColumns(_TempLedger):
    def test_every_key_the_cli_reads_is_present(self) -> None:
        c = self._record()
        rows = self.lg.recent_applied()
        self.assertEqual(len(rows), 1)
        r = rows[0]
        for k in CLI_KEYS:
            with self.subTest(key=k):
                self.assertIn(k, r.keys())
        self.assertEqual(r["change_id"], c.id)
        self.assertEqual(r["tier"], "system")
        self.assertEqual(r["change_key"], change_key(c))
        self.assertEqual(r["evidence"], "cited")
        self.assertEqual(r["reverted"], 0)
        self.assertIsNone(r["effect"])

    def test_reverted_flag_is_visible_after_a_revert(self) -> None:
        c = self._record()
        self.lg.record_reverted(c, "restore")
        r = self.lg.recent_applied()[0]
        self.assertEqual(r["reverted"], 1)
        self.assertEqual(r["revert_verb"], "restore")

    def test_rollback_filter_runs_and_excludes_reverted_rows(self) -> None:
        # The exact expression cmd_rollback evaluates.
        kept = self._record("env_kept", target="CUDA_CACHE_MAXSIZE")
        gone = self._record("env_gone", target="CUDA_MODULE_LOADING")
        self.lg.record_reverted(gone, "restore")
        rows = [r for r in self.lg.recent_applied(limit=500) if not r["reverted"]]
        self.assertEqual([r["change_id"] for r in rows], [kept.id])


class TestCliCommandsOverRealRows(_TempLedger):
    """Drive cmd_ledger and cmd_rollback --list against a populated temp ledger.
    open_ledger is replaced so neither command can reach the real DB; rollback
    with --list only prints, it never builds a Plane or touches the machine."""

    def _run(self, fn, args) -> str:
        @contextlib.contextmanager
        def fake_open():
            yield self.lg
        buf = io.StringIO()
        with mock.patch.object(cli, "open_ledger", lambda *a, **k: fake_open()), \
                contextlib.redirect_stdout(buf):
            rc = fn(args)
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_cmd_ledger_prints_applied_rows(self) -> None:
        self._record(effect=0.05)
        out = self._run(cli.cmd_ledger, argparse.Namespace(session=False, limit=25))
        self.assertIn("env_cuda_cache", out)
        self.assertIn("effect +0.050", out)

    def test_cmd_rollback_list_prints_tier_and_key(self) -> None:
        c = self._record()
        out = self._run(cli.cmd_rollback,
                        argparse.Namespace(list=True, change=None, all=False))
        self.assertIn(c.id, out)
        self.assertIn(f"key={change_key(c)}", out)


if __name__ == "__main__":
    unittest.main()
