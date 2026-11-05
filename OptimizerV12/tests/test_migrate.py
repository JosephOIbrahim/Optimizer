"""
Tests for Phase 0a — the migration and, more importantly, the repair rule.

The bug these tests exist for is worth stating plainly, because it was not a crash.
The first repair rule asked *"is this fragment a suffix of the line above?"* Every
`applied` record ends `"ok": true, "verified": true}`, so `}`, `rue}`, `e}` and
`true}` satisfy that question **by coincidence**. The rule reported 7 of 9 damaged
lines as repaired. Nothing had been repaired. A tool that says "healed" when it has
done nothing is more dangerous than one that says nothing at all, so the assertion
that matters is not "did it run" but "did it claim a repair it did not perform".
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12 import migrate  # noqa: E402


def _repair(content: bytes):
    """Run repair_journal over `content` in a temp dir; return (report, output lines)."""
    d = Path(tempfile.mkdtemp(prefix="v12mig_"))
    src = d / "transactions.jsonl"
    dst = d / "out.jsonl"
    src.write_bytes(content)
    rep = migrate.MigrationReport()
    migrate.repair_journal(src, dst, rep)
    out = [l for l in dst.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rep, out, dst


#: A well-formed `applied` record. Its ending is what caused the false positive.
APPLIED = '{"ts":"t","kind":"applied","id":"%s","ok":true,"verified":true}'


class TestCoincidentalSuffix(unittest.TestCase):
    """The regression. A fragment that is a suffix of the line above is NOT evidence
    of a torn write — the line above parsed, so nothing was torn there."""

    def test_suffix_of_a_complete_record_is_not_called_a_repair(self):
        content = (
            (APPLIED % "a").encode() + b"\n"
            + b"e}\n"                                   # coincidental suffix
            + (APPLIED % "b").encode() + b"\n"
        )
        rep, out, _ = _repair(content)

        self.assertEqual(rep.journal_malformed, 1)
        self.assertEqual(rep.journal_reconstructed, 0,
                         "a fragment following a VALID line was reported as a repair; "
                         "this is the false-assurance bug")
        self.assertEqual(rep.journal_unattributable, 1)
        self.assertEqual(rep.journal_valid_lines, 2)

    def test_every_coincidental_suffix_shape_is_rejected(self):
        # Each of these is a literal suffix of a complete `applied` record.
        for frag in (b"}", b"e}", b"rue}", b"true}", b'verified": true}'):
            with self.subTest(frag=frag):
                content = (APPLIED % "a").encode() + b"\n" + frag + b"\n"
                rep, _, _ = _repair(content)
                self.assertEqual(rep.journal_reconstructed, 0,
                                 f"{frag!r} was reported as a repair")
                self.assertEqual(rep.journal_unattributable, 1)

    def test_nothing_is_deleted_when_it_cannot_be_attributed(self):
        content = (APPLIED % "a").encode() + b"\n" + b"true}\n"
        rep, out, _ = _repair(content)
        # Two source lines, two output lines: the fragment is preserved, not dropped.
        self.assertEqual(len(out), 2)
        self.assertIn("true}", out[1])
        self.assertEqual(json.loads(out[1])["_v12_migration"], "unattributable_fragment")
        self.assertEqual(rep.unattributable[0]["fragment"], "true}")


class TestGenuineTear(unittest.TestCase):
    """A real torn write splits mid-record, so the tail is the head's CONTINUATION.
    The tail is therefore NOT a suffix of the head — which is precisely why the
    suffix test could not detect the case it was written for."""

    def test_torn_record_is_reconstructed_by_joining(self):
        full = (APPLIED % "scratch_dirs").encode()
        head, tail = full[:40], full[40:]
        self.assertFalse(head.endswith(tail), "sanity: a real tear is not a suffix match")
        with self.assertRaises(json.JSONDecodeError):
            json.loads(head)                            # sanity: the head is incomplete

        rep, out, _ = _repair(head + b"\n" + tail + b"\n")

        self.assertEqual(rep.journal_reconstructed, 1)
        self.assertEqual(rep.journal_unattributable, 0)
        self.assertEqual(len(out), 1, "head+tail should collapse into one record")
        self.assertEqual(json.loads(out[0]), json.loads(full))

    def test_rejoin_requires_the_preceding_line_to_have_failed(self):
        """The precondition. If the preceding line parsed, it was never torn."""
        complete = APPLIED % "a"
        self.assertIsNone(migrate._try_rejoin("x", complete, True),
                          "a fragment after a VALID line must never be joined onto it")

    def test_rejoin_checks_the_join_rather_than_assuming_it(self):
        # A join that happens to be valid JSON is a real reconstruction.
        self.assertEqual(migrate._try_rejoin('"b":2}', '{"a":1,', False), '{"a":1,"b":2}')
        # A join that is not valid JSON is not a reconstruction — it is two pieces
        # of unrelated damage, and guessing at it would be inventing history.
        self.assertIsNone(migrate._try_rejoin('"b":2}', '{"a":1', False))
        self.assertIsNone(migrate._try_rejoin("garbage", '{"a":1', False))
        self.assertIsNone(migrate._try_rejoin("", '{"a":1', False))

    def test_a_suffix_that_does_not_rebuild_is_not_reconstructed(self):
        rep, _, _ = _repair((APPLIED % "a").encode() + b"\n" + b"true}\n")
        self.assertEqual(rep.journal_reconstructed, 0)


class TestAccounting(unittest.TestCase):
    """Every source line must be accounted for. The failure mode of a repair routine
    is silent loss, so this is asserted rather than assumed."""

    def test_every_source_line_is_accounted_for(self):
        content = (
            (APPLIED % "a").encode() + b"\n"
            + b"e}\n"
            + (APPLIED % "b").encode() + b"\n"
            + b"true}\n"
            + b"\n"                                     # a blank line
            + (APPLIED % "c").encode() + b"\n"
        )
        rep, out, _ = _repair(content)
        self.assertEqual(rep.journal_source_lines, 6)
        self.assertEqual(rep.journal_valid_lines, 3)
        self.assertEqual(rep.journal_malformed, 2)
        self.assertEqual(rep.journal_blank_lines, 1)
        # The identity that proves nothing was silently dropped.
        self.assertEqual(rep.journal_total_accounted, rep.journal_source_lines)
        self.assertTrue(rep.journal_malformed_identity_holds)
        self.assertEqual(len(out), 5)                   # 3 records + 2 preserved tails

    def test_a_reconstructed_record_consumes_two_lines_and_yields_one(self):
        full = (APPLIED % "scratch_dirs").encode()
        rep, out, _ = _repair(full[:40] + b"\n" + full[40:] + b"\n")
        self.assertEqual(rep.journal_source_lines, 2)
        self.assertEqual(rep.journal_malformed, 2)
        self.assertEqual(rep.journal_reconstructed, 1)
        self.assertEqual(rep.journal_unattributable, 0,
                         "the unfinished head was counted as unattributable before its "
                         "continuation arrived and the classification was not withdrawn")
        self.assertEqual(len(out), 1)
        self.assertEqual(rep.journal_total_accounted, rep.journal_source_lines)
        self.assertTrue(rep.journal_malformed_identity_holds)

    def test_no_malformed_line_is_dropped_without_a_record(self):
        content = (
            (APPLIED % "a").encode() + b"\n"
            + b"rue}\n"
            + (APPLIED % "b").encode() + b"\n"
            + b'": true, "verified": true}\n'
        )
        rep, _, dst = _repair(content)
        recorded = {r["fragment"] for r in rep.unattributable} | {
            r["fragment"] for r in rep.repairs}
        self.assertEqual(recorded, {"rue}", '": true, "verified": true}'})
        # and the report is written out in full, not summarised
        self.assertTrue((dst.parent / "out.jsonl").exists())


class TestOkSemantics(unittest.TestCase):
    """`ok` means the migration succeeded — not that the file was clean. Damage that
    predates the migration by weeks must not be reported as a migration failure, and
    an unverified restore master must never be reported as success."""

    def test_unattributable_fragments_do_not_make_the_migration_fail(self):
        rep = migrate.MigrationReport(master_verified=True)
        rep.journal_unattributable = 9
        self.assertTrue(rep.ok)

    def test_an_unverified_master_is_a_failure(self):
        rep = migrate.MigrationReport(master_verified=False)
        self.assertFalse(rep.ok)

    def test_any_error_is_a_failure(self):
        rep = migrate.MigrationReport(master_verified=True, errors=["hash mismatch"])
        self.assertFalse(rep.ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
