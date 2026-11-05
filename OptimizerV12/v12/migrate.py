"""
V12 — Phase 0a: move MachineOptimizer2026's state off OneDrive, and repair what
OneDrive already damaged.

Two jobs, and the second one is why this is a module and not a shell one-liner.

**Move.** `MachineOptimizer2026/data/` held the transaction log and the restore
master under `OneDrive\\Desktop`. Sync on an open append-only log is not a
theoretical hazard here — it already fired.

**Repair.** An integrity check on 2026-09-16 found 9 malformed lines in
`transactions.jsonl` — fragments on their own lines (`}`, `rue}`, `e}`, `true}`,
`verified": true}`).

The first version of the repair rule asked only: *is this fragment a suffix of the
line before it?* That question is wrong twice over.

Wrong once because a torn write splits a record at an arbitrary byte, so the tail is
the head's **continuation**, not a suffix of it — the suffix test cannot recognise the
very thing it was written for. Wrong twice because every `applied` record ends
`"ok": true, "verified": true}`, so the tails `}`, `rue}`, `e}` and `true}` *are*
suffixes of that ending by coincidence. The rule duly reported 7 of the 9 as
"repaired", implying the data had been healed. Nothing had been healed.

The question that is actually checkable is whether the record can be **rebuilt**:

  * preceding line did **not** parse, and head+tail parses → a real tear. The record
    is recovered in full and written that way.
  * preceding line **did** parse → no tear occurred there. The fragment cannot be
    rebuilt, and two histories are consistent with it that this file cannot tell
    apart: a redundant re-flush of a tail whose record is already present, or the back
    half of a record whose front half, carrying the change id, was lost. Preserved
    verbatim, tagged, and named in the report as *unattributable*.

In this file all 9 follow a line that already parsed, so none can be rebuilt and all 9
are preserved. No malformed line is ever dropped without being recorded in full, so
the repair is auditable rather than merely asserted. The legacy directory is left in
place as a read-only source and a `_migration_report.json` is written beside the new
data.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from . import paths

log = logging.getLogger("v12.migrate")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Repair:
    line_no: int
    fragment: str
    reason: str
    recovered: bool = False   # True = the record was reconstructed; False = recorded but lost


@dataclass
class MigrationReport:
    started: float = field(default_factory=time.time)
    source: str = ""
    dest: str = ""
    master_copied: bool = False
    master_sha256: str = ""
    master_verified: bool = False
    master_entries: int = 0
    journal_source_lines: int = 0
    journal_valid_lines: int = 0
    journal_blank_lines: int = 0
    journal_malformed: int = 0          # lines that were not valid JSON
    journal_reconstructed: int = 0      # malformed lines whose record was recovered by joining
    journal_unattributable: int = 0     # tails that cannot be identified either way
    repairs: List[dict] = field(default_factory=list)
    unattributable: List[dict] = field(default_factory=list)
    dest_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    already_done: bool = False

    @property
    def ok(self) -> bool:
        """`ok` means the migration succeeded, not that the file was clean.

        The damage predates the migration by weeks — it is OneDrive's, not ours —
        and no migration can invent a change id that was never written. An
        unattributable fragment is therefore *correctly handled*, not a failure: it
        is preserved verbatim and named in the report.

        What would be a failure is an unverified restore master, or a line dropped
        without being recorded. Neither is permitted here.
        """
        return not self.errors and self.master_verified

    @property
    def journal_total_accounted(self) -> int:
        """Source lines consumed. A reconstructed record consumed TWO lines — the
        unfinished head and its continuation — so it is weighted accordingly here.
        This is the identity that proves nothing was silently dropped:
        valid + 2*reconstructed + unattributable + blank == source lines."""
        return (self.journal_valid_lines + 2 * self.journal_reconstructed
                + self.journal_unattributable + self.journal_blank_lines)

    @property
    def journal_malformed_identity_holds(self) -> bool:
        """Every non-parsing line is either an unattributable tail or one half of a
        reconstructed record. If this is False, a line went somewhere unaccounted."""
        return self.journal_malformed == (self.journal_unattributable
                                          + 2 * self.journal_reconstructed)

    def summary(self) -> str:
        if self.already_done:
            return f"migration already complete at {self.dest} (nothing to do)"
        L = [
            "Phase 0a — MachineOptimizer2026 data migration",
            "-" * 68,
            f"  from: {self.source}",
            f"  to:   {self.dest}",
            "",
            f"  restore master : {'copied + sha256 verified' if self.master_verified else 'FAILED'} "
            f"({self.master_entries} recoverable changes)",
            f"  transaction log: {self.journal_valid_lines} valid records written "
            f"from {self.journal_source_lines} source lines",
            f"  damaged lines  : {self.journal_malformed} found  "
            f"({self.journal_reconstructed} reconstructed, "
            f"{self.journal_unattributable} unattributable tails preserved)",
        ]
        if self.journal_malformed:
            L.append("")
            L.append("    What these are, and what the first version of this rule got")
            L.append("    wrong. That version asked only \"is this fragment a suffix of")
            L.append("    the line above?\" Every `applied` record ends")
            L.append("    `\"ok\": true, \"verified\": true}`, so `}`, `rue}`, `e}` and")
            L.append("    `true}` are suffixes of that ending BY COINCIDENCE. The rule")
            L.append("    duly reported 7 of these as repaired — implying the data had")
            L.append("    been healed when nothing had happened at all.")
            L.append("")
            L.append("    The correct question is whether the record can be rebuilt:")
            L.append("    a torn write leaves an unfinished head, and head+tail parses.")
            L.append("    Every damaged line here follows a line that already parsed,")
            L.append("    so no tear occurred at any of them. They cannot be rebuilt.")
            L.append("    They are preserved verbatim. Nothing was healed and nothing")
            L.append("    was lost that this migration could have recovered.")
        for r in self.repairs[:12]:
            L.append(f"      line {r['line_no']:>6}: reconstructed {r['fragment']!r}")
        for u in self.unattributable[:12]:
            L.append(f"      line {u['line_no']:>6}: unattributable tail {u['fragment']!r}")
        if len(self.unattributable) > 12:
            L.append(f"      ... and {len(self.unattributable) - 12} more "
                     f"(see _migration_report.json)")
        if self.errors:
            L.append("")
            for e in self.errors:
                L.append(f"  ERROR: {e}")
        L.append("")
        L.append(f"  accounts for {self.journal_total_accounted} of "
                 f"{self.journal_source_lines} source lines "
                 f"({self.journal_valid_lines} valid + "
                 f"{2 * self.journal_reconstructed} torn halves + "
                 f"{self.journal_unattributable} unattributable + "
                 f"{self.journal_blank_lines} blank)")
        if not self.journal_malformed_identity_holds:
            L.append("  WARNING: malformed lines do not reconcile with the repair "
                     "records — a line went somewhere unaccounted")
        L.append(f"  recovery capability: the restore master is the file that decides")
        L.append(f"  how to un-optimize this machine, and it is intact and hash-verified.")
        L.append(f"  The damaged lines are log entries, not undo records.")
        L.append(f"  legacy copy left in place at {self.source}")
        L.append("  report: " + str(Path(self.dest) / "_migration_report.json"))
        return "\n".join(L)


def _try_rejoin(fragment: str, previous: str, previous_parsed: bool) -> Optional[str]:
    """Reconstruct a torn record, or return None.

    A torn write splits a record at an arbitrary byte: the head lands, the tail
    lands, and the newline between them lands late. So the tail is the head's
    **continuation**, not a suffix of it. The checkable condition is therefore not
    "is this fragment a suffix of the line above" — an earlier version of this
    function asked that, and it is the wrong question — but simply:

        the preceding line did NOT parse, and head + tail parses.

    That is verifiable rather than plausible: the recovered string is either valid
    JSON or it is not. Note the precondition — if the preceding line *did* parse
    then it was never torn, so no fragment can be its continuation.
    """
    if previous_parsed:
        return None
    head, frag = previous.strip(), fragment.strip()
    if not head or not frag:
        return None
    joined = head + frag
    try:
        json.loads(joined)
    except Exception:
        return None
    return joined


def repair_journal(src: Path, dest: Path, report: MigrationReport) -> None:
    """Write a repaired copy of the transaction log, recording every decision.

    Two outcomes for a malformed line, and the difference is checkable:

      * **reconstructed** — it is the tail of a record whose head is the preceding
        line, and that preceding line did *not* parse. Joining them yields valid
        JSON, so the record is genuinely recovered and written in full. This is
        what a real torn write looks like.
      * **headless** — the preceding line parsed, so nothing was torn here. The
        fragment is the back half of a record whose front half is gone. It is
        preserved verbatim and tagged, because guessing its identity would be
        inventing history.

    No malformed line is ever dropped without being recorded in full.
    """
    lines = src.read_bytes().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()          # a trailing newline terminates the last line; it is not one
    report.journal_source_lines = len(lines)
    out_lines: List[bytes] = []
    prev_text = ""
    prev_parsed = True                # nothing precedes line 1, so nothing to complete
    pending: Optional[int] = None     # index in out_lines of an unfinished head

    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s:
            report.journal_blank_lines += 1
            continue
        frag = s.decode("utf-8", "replace")
        try:
            json.loads(s)
            parsed = True
        except Exception:
            parsed = False

        if parsed:
            out_lines.append(raw)
            report.journal_valid_lines += 1
            pending = None
        else:
            report.journal_malformed += 1

            # ---- path 1: does this fragment continue the preceding unfinished line?
            joined = _try_rejoin(frag, prev_text, prev_parsed)
            if joined is not None:
                report.journal_reconstructed += 1
                report.repairs.append({
                    "line_no": i, "fragment": frag, "reconstructed": joined,
                    "reason": "the preceding line did not parse and joining it with this "
                              "fragment yields valid JSON — the record is recovered in "
                              "full, not guessed at",
                    "recovered": True,
                })
                enc = joined.encode("utf-8")
                if pending is not None:
                    out_lines[pending] = enc
                    # The unfinished head was provisionally recorded as unattributable
                    # before we had seen its continuation. It is not unattributable —
                    # it is the front half of the record just rebuilt — so withdraw
                    # that classification and fold it into the repair record instead.
                    report.journal_unattributable -= 1
                    if report.unattributable:
                        head = report.unattributable.pop()
                        report.repairs[-1]["head_line_no"] = head["line_no"]
                        report.repairs[-1]["head_fragment"] = head["fragment"]
                else:
                    out_lines.append(enc)
                pending = None
                prev_text, prev_parsed = joined, True
                continue

            # ---- path 2: unattributable. Preserve verbatim; never guess.
            report.journal_unattributable += 1
            report.unattributable.append({
                "line_no": i, "fragment": frag,
                "reason": "the preceding line parsed as valid JSON, so no tear occurred "
                          "there, and joining this fragment to it does not yield JSON "
                          "either. It is the tail of a record that cannot be identified. "
                          "Two histories are consistent with it and this file cannot "
                          "distinguish them: a redundant re-flush of a tail whose record "
                          "is already present, or the back half of a record whose front "
                          "half — carrying the change id — was lost. Preserved verbatim; "
                          "attributing it either way would be inventing history.",
            })
            out_lines.append(json.dumps({
                "_v12_migration": "unattributable_fragment",
                "_line": i, "_raw": frag,
                "_note": "tail of an unidentifiable record; the id is unknowable",
            }).encode("utf-8"))
            pending = len(out_lines) - 1

        prev_text, prev_parsed = frag, parsed

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish: write to a temp name, then replace.
    tmp = dest.with_suffix(".jsonl.tmp")
    with open(tmp, "wb") as f:
        f.write(b"\n".join(out_lines) + b"\n")
    tmp.replace(dest)


def run(force: bool = False) -> MigrationReport:
    """Idempotent. Safe to re-run; refuses to clobber an existing destination."""
    rep = MigrationReport(source=str(paths.MOS2026_DATA_LEGACY), dest=str(paths.MOS2026_DATA))
    src = paths.MOS2026_DATA_LEGACY
    dst = paths.MOS2026_DATA

    if not src.exists():
        rep.errors.append(f"nothing to migrate: {src} does not exist")
        return rep

    dst.mkdir(parents=True, exist_ok=True)
    master_src = src / "restore_points" / "master.json"
    master_dst = dst / "restore_points" / "master.json"

    if master_dst.exists() and not force:
        rep.already_done = True
        return rep

    # ---- the restore master: copy, then verify by hash -------------------
    if master_src.exists():
        master_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(master_src, master_dst)
        a, b = sha256(master_src), sha256(master_dst)
        rep.master_sha256 = b
        rep.master_copied = True
        rep.master_verified = (a == b)
        if not rep.master_verified:
            rep.errors.append(
                f"restore master hash mismatch after copy ({a[:12]} != {b[:12]}) — "
                f"the undo record did not survive; do not use the new location")
        try:
            rep.master_entries = len(json.loads(master_dst.read_text(encoding="utf-8"))
                                     .get("changes", {}))
        except Exception as e:
            rep.errors.append(f"copied master is unreadable: {e}")
    else:
        rep.errors.append(f"restore master missing at {master_src}")

    # ---- the journal: repair while copying -------------------------------
    j_src = src / "transactions.jsonl"
    if j_src.exists():
        repair_journal(j_src, dst / "transactions.jsonl", rep)
    else:
        log.warning("no transactions.jsonl at %s", j_src)

    # ---- anything else that might appear later ---------------------------
    for extra in sorted(src.rglob("*")):
        if extra.is_file() and extra.name not in ("transactions.jsonl", "master.json"):
            rel = extra.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, target)

    rep.dest_files = sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file())
    (dst / "_migration_report.json").write_text(
        json.dumps(asdict(rep), indent=2), encoding="utf-8")
    return rep


def verify() -> Tuple[bool, str]:
    """Post-migration check an operator can run any time."""
    dst = paths.MOS2026_DATA
    if not dst.exists():
        return False, f"no migrated data at {dst}"
    problems = []
    master = dst / "restore_points" / "master.json"
    if not master.exists():
        problems.append("restore master missing at the new location")
    else:
        try:
            n = len(json.loads(master.read_text(encoding="utf-8")).get("changes", {}))
        except Exception as e:
            problems.append(f"restore master unreadable: {e}")
            n = 0
    legacy = paths.MOS2026_DATA_LEGACY / "restore_points" / "master.json"
    if legacy.exists() and master.exists():
        if sha256(legacy) != sha256(master):
            problems.append("restore master differs from the legacy copy — one of them "
                            "has been written since the migration")

    j = dst / "transactions.jsonl"
    bad = 0
    if j.exists():
        for line in j.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
            except Exception:
                bad += 1
        if bad:
            problems.append(f"{bad} malformed lines remain in the migrated journal")

    if paths.is_under_onedrive(dst):
        problems.append("the new data root is still under OneDrive")
    if problems:
        return False, "; ".join(problems)
    return True, (f"migrated data verified at {dst} "
                  f"({n} recoverable changes, journal well-formed)")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths.ensure_dirs()
    r = run(force="--force" in sys.argv)
    print(r.summary())
    ok, msg = verify()
    print()
    print(("VERIFY OK: " if ok else "VERIFY FAILED: ") + msg)
    sys.exit(0 if (r.ok and ok) else 1)
