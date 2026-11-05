"""
V12 — the ledger, and the honesty metric.

MachineOptimizer2026 reports, correctly and uselessly: "Applied 40, verified 40."
Every one of those 40 was verified in the only sense it knows — read-back equals
target. A change that landed and made the machine slower passes that test. The
number is true and it answers the wrong question, which is the most durable kind
of misleading.

So this ledger does not report "verified". It reports the breakdown by how the
benefit was KNOWN (zero counts are left out):

    Applied 3  |  measured 3  |  denied 33

That single line is the thesis of V12 in a form a human reads in two seconds.
There is no cited count because a source is not a measurement: the Gate refuses
CITED, and no operator-facing opt-in exists in this build (Gate(allow_unverified=
True) is a programmatic switch only). A derived count appears only once something
supplies a per-change prediction, which the plane does not compute yet; until
then a run without measurements reads "Applied 0  |  denied N". A large denied
count is not a failure. It is an honest
description of what has been proven on this machine, and it is the number that
should shrink as baselines accumulate.

WAL mode, off OneDrive. OneDrive syncing an open WAL file is how a transaction log
gets corrupted mid-write, which is precisely the file you least want corrupted.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import paths
from .admission import Verdict
from .change import Change, EvidenceClass, change_key

log = logging.getLogger("v12.ledger")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started     REAL NOT NULL,
    ended       REAL,
    version     TEXT,
    note        TEXT
);

-- Every verdict the Gate issued, admitted or not. Denials are the interesting
-- half: they are the record of what V12 declined to do and why.
CREATE TABLE IF NOT EXISTS verdicts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session      TEXT,
    ts           REAL NOT NULL,
    change_id    TEXT NOT NULL,
    change_key   TEXT NOT NULL,
    tier         TEXT,
    kind         TEXT,
    observability TEXT,
    admitted     INTEGER NOT NULL,
    evidence     TEXT NOT NULL,
    reason       TEXT,
    objective    TEXT,
    baseline_id  TEXT,
    destructive_reset INTEGER DEFAULT 0,
    detail       TEXT
);
CREATE INDEX IF NOT EXISTS idx_verdicts_key ON verdicts(change_key, ts DESC);
CREATE INDEX IF NOT EXISTS idx_verdicts_admitted ON verdicts(admitted, evidence);

-- Changes that actually reached an actuator.
CREATE TABLE IF NOT EXISTS applied (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session       TEXT,
    ts            REAL NOT NULL,
    change_id     TEXT NOT NULL,
    change_key    TEXT NOT NULL,
    tier          TEXT,
    kind          TEXT,
    evidence      TEXT NOT NULL,
    baseline_id   TEXT,
    objective     TEXT,
    landed        INTEGER,          -- read-back == target
    landed_source TEXT,             -- how that was established, or why it wasn't
    effect        REAL,             -- measured fractional improvement, when known
    p_value       REAL,
    reverted      INTEGER DEFAULT 0,
    reverted_ts   REAL,
    revert_verb   TEXT,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS idx_applied_key ON applied(change_key, ts DESC);

-- Raw measurement series, kept because a mean without its spread cannot be tested.
CREATE TABLE IF NOT EXISTS measurements (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session    TEXT,
    ts         REAL NOT NULL,
    objective  TEXT NOT NULL,
    workload   TEXT NOT NULL,
    metric     TEXT NOT NULL,
    samples    TEXT NOT NULL,       -- JSON array
    conditions TEXT,                -- JSON Conditions
    harness    TEXT,
    label      TEXT                 -- 'baseline' | 'candidate' | 'ad-hoc'
);
CREATE INDEX IF NOT EXISTS idx_meas_obj ON measurements(objective, ts DESC);

-- Baselines are files on disk; this is the index so lookups don't stat the dir.
-- `superseded` mirrors the file's own flag: a withdrawn baseline stays
-- queryable but is visibly not a usable reference.
CREATE TABLE IF NOT EXISTS baselines (
    id         TEXT PRIMARY KEY,
    ts         REAL NOT NULL,
    objective  TEXT NOT NULL,
    workload   TEXT,
    metric     TEXT,
    direction  TEXT,
    mean       REAL,
    stdev      REAL,
    n          INTEGER,
    cv         REAL,
    usable     INTEGER,
    conditions TEXT,
    path       TEXT,
    superseded INTEGER DEFAULT 0,
    superseded_reason TEXT
);

-- Jev annotations (v12/jev.py). NOT an evidence table: nothing here is a
-- verdict, a measurement or a reason to act, and stats()/honesty_report() never
-- read it. It exists so an operator can see what the judgment model said and
-- what it was shown (by hash), next to - never inside - the honesty metric.
CREATE TABLE IF NOT EXISTS judgments (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session            TEXT,
    ts                 REAL NOT NULL,
    stage              TEXT NOT NULL,   -- 'observe' | 'decide'
    question_id        TEXT NOT NULL,
    subject_key        TEXT,
    state_sha256       TEXT,            -- of the REDACTED state that was sent
    model              TEXT,
    request_id         TEXT,
    label              TEXT,            -- NULL unless outcome = 'ok'
    probabilities      TEXT,            -- JSON object
    confidence_or_noul REAL,
    outcome            TEXT NOT NULL,   -- ok | offline | rejected | error | cap
    consumed_as        TEXT NOT NULL,   -- 'display' | 'ordering'
    cache_hit          INTEGER DEFAULT 0,
    offline_reason     TEXT
);
CREATE INDEX IF NOT EXISTS idx_judgments_session ON judgments(session, ts DESC);
"""

#: The only uses a judgment row may declare. Mirrors jev.CONSUMED_AS without
#: importing it: plane imports this module, and the gate's import closure must
#: not reach v12.jev (tests/test_invariants.py).
JUDGMENT_USES = ("display", "ordering")
#: A judgment label may not read like an evidence class (mirrors jev.RESERVED_LABELS).
_EVIDENCE_WORDS = frozenset({"measured", "derived", "cited", "guided",
                             "admitted", "denied", "verdict", "evidence"})


@dataclass
class LedgerStats:
    applied_total: int = 0
    by_evidence: Dict[str, int] = None            # type: ignore[assignment]
    denied_total: int = 0
    deny_reasons: Dict[str, int] = None           # type: ignore[assignment]
    landed: int = 0
    landed_untrusted: int = 0                     # applied but read-back not evidence
    measured_wins: int = 0                        # applied WITH a real effect recorded
    reverted: int = 0
    destructive: int = 0

    def __post_init__(self) -> None:
        self.by_evidence = self.by_evidence or {}
        self.deny_reasons = self.deny_reasons or {}

    def headline(self) -> str:
        ev = self.by_evidence
        parts = [f"Applied {self.applied_total}"]
        for name in ("measured", "derived", "cited", "guided"):
            if ev.get(name):
                parts.append(f"{name} {ev[name]}")
        parts.append(f"denied {self.denied_total}")
        return "  |  ".join(parts)

    def honest_note(self) -> str:
        m = self.by_evidence.get("measured", 0)
        if self.applied_total == 0:
            return "nothing applied this session"
        pct = 100.0 * m / self.applied_total
        if pct == 0:
            return ("NOTHING here was measured on this machine. Every applied change "
                    "rests on a model prediction or vendor documentation. None of them "
                    "has been shown to be faster.")
        if pct < 50:
            return (f"{pct:.0f}% of applied changes were measured on this machine; the "
                    f"rest are unverified here.")
        return f"{pct:.0f}% of applied changes were measured on this machine."


class Ledger:
    def __init__(self, db_path=None, session: Optional[str] = None):
        self.path = db_path or paths.LEDGER_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if paths.is_under_onedrive(self.path):
            log.warning(
                "ledger at %s is under OneDrive — an open WAL file being synced can be "
                "corrupted mid-write. Set $OPTIMIZER_V12_DATA to an off-OneDrive path.",
                self.path)
        self.conn = sqlite3.connect(str(self.path), timeout=15)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()
        self.session = session or time.strftime("s%Y%m%d-%H%M%S")

    def _migrate(self) -> None:
        """Add columns that newer builds expect, to an existing ledger.

        SCHEMA uses CREATE TABLE IF NOT EXISTS, which never adds a column to a
        table that already exists — so a ledger written by an older build
        would keep the old shape and the newer INSERT would fail with
        "table baselines has no column named superseded".

        The migration is additive and idempotent: a column is added only when
        it is missing, and existing rows get the default, which is the
        conservative reading (not superseded). Nothing is rewritten.
        """
        for table, column, decl in (("baselines", "superseded", "INTEGER DEFAULT 0"),
                                   ("baselines", "superseded_reason", "TEXT"),):
            cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                log.info("ledger: adding %s.%s (schema was older)", table, column)
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # ---- lifecycle -------------------------------------------------------
    def start(self, version: str = "", note: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions(id, started, version, note) VALUES (?,?,?,?)",
            (self.session, time.time(), version, note))
        self.conn.commit()

    def end(self) -> None:
        self.conn.execute("UPDATE sessions SET ended=? WHERE id=?",
                          (time.time(), self.session))
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- writes ----------------------------------------------------------
    def record_verdict(self, v: Verdict, change: Optional[Change] = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO verdicts(session, ts, change_id, change_key, tier, kind,
                                    observability, admitted, evidence, reason, objective,
                                    baseline_id, destructive_reset, detail)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.session, v.issued, v.change_id, v.change_key,
             change.tier.value if change else None,
             change.kind.value if change else None,
             change.observability.value if change else None,
             1 if v.admitted else 0, v.evidence.value, v.reason,
             v.objective_id, v.baseline_id, 1 if v.destructive_reset else 0,
             json.dumps(v.detail)),
        )
        self.conn.commit()
        return cur.lastrowid or 0

    def record_applied(self, change: Change, v: Verdict, *, landed: bool,
                       landed_source: str = "", effect: Optional[float] = None,
                       p_value: Optional[float] = None,
                       detail: str = "") -> int:
        cur = self.conn.execute(
            """INSERT INTO applied(session, ts, change_id, change_key, tier, kind,
                                   evidence, baseline_id, objective, landed, landed_source,
                                   effect, p_value, detail)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.session, time.time(), change.id, change_key(change),
             change.tier.value, change.kind.value, v.evidence.value, v.baseline_id,
             v.objective_id, None if landed is None else (1 if landed else 0),
             landed_source, effect, p_value, detail),
        )
        self.conn.commit()
        return cur.lastrowid or 0

    def record_reverted(self, change: Change, verb: str) -> None:
        self.conn.execute(
            """UPDATE applied SET reverted=1, reverted_ts=?, revert_verb=?
               WHERE change_key=? AND reverted=0""",
            (time.time(), verb, change_key(change)))
        self.conn.commit()

    def record_measurement(self, objective_id: str, workload: str, metric: str,
                           samples: Sequence[float], conditions: Any = None,
                           harness: str = "", label: str = "ad-hoc") -> int:
        cur = self.conn.execute(
            """INSERT INTO measurements(session, ts, objective, workload, metric, samples,
                                        conditions, harness, label)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (self.session, time.time(), objective_id, workload, metric,
             json.dumps(list(samples)),
             json.dumps(conditions, default=str) if conditions else None,
             harness, label))
        self.conn.commit()
        return cur.lastrowid or 0

    def index_baseline(self, b: Any) -> None:
        """Mirror a baseline into the queryable index.

        The JSON file in paths.BASELINE_DIR is the source of truth; this table
        is a denormalised copy so an operator can query it with SQL. It must
        not DISAGREE with the file — a baseline that was withdrawn would
        otherwise still look usable here, which is exactly the kind of quiet
        disagreement this project exists to avoid.
        """
        self.conn.execute(
            """INSERT OR REPLACE INTO baselines(id, ts, objective, workload, metric,
                                                direction, mean, stdev, n, cv, usable,
                                                conditions, path, superseded,
                                                superseded_reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (b.id, b.created, b.objective_id, b.workload.get("name", ""), b.metric,
             b.direction, b.mean, b.stdev, b.n, b.cv, 1 if b.usable else 0,
             json.dumps(b.conditions), str(b.path()),
             1 if getattr(b, "superseded", False) else 0,
             getattr(b, "superseded_reason", "") or None))
        self.conn.commit()

    def record_judgment(self, j: Any, *, stage: str) -> int:
        """One Jev judgment, into its own table. Duck-typed on v12.jev.Judgment so
        this module never imports jev. Refuses a use other than display/ordering
        and a label that reads like an evidence class: a row that could be
        mistaken for a verdict is worse than no row."""
        if j.consumed_as not in JUDGMENT_USES:
            raise ValueError(f"judgment consumed_as {j.consumed_as!r} is not one of "
                             f"{JUDGMENT_USES}")
        if j.label is not None and str(j.label).strip().lower() in _EVIDENCE_WORDS:
            raise ValueError(f"judgment label {j.label!r} reads like an evidence class")
        cur = self.conn.execute(
            """INSERT INTO judgments(session, ts, stage, question_id, subject_key,
                                     state_sha256, model, request_id, label,
                                     probabilities, confidence_or_noul, outcome,
                                     consumed_as, cache_hit, offline_reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.session, time.time(), stage, j.question_id, j.subject_key,
             j.state_sha256, j.model, j.request_id, j.label,
             json.dumps({str(k): v for k, v in (j.probabilities or ())}),
             j.confidence_or_noul, j.outcome, j.consumed_as,
             1 if getattr(j, "cache_hit", False) else 0,
             getattr(j, "offline_reason", "") or None))
        self.conn.commit()
        return cur.lastrowid or 0

    def judgments_summary(self, session_only: bool = False, limit: int = 10
                          ) -> Dict[str, Any]:
        """{'total', 'by_outcome', 'recent'} over the judgments table. Read by the
        ledger CLI's separate 'Jev annotations' section only — never by stats()."""
        where, args = ("WHERE session=?", (self.session,)) if session_only else ("", ())
        by_outcome: Dict[str, int] = {}
        for row in self.conn.execute(
                f"SELECT outcome, COUNT(*) c FROM judgments {where} GROUP BY outcome", args):
            by_outcome[row["outcome"]] = row["c"]
        recent = list(self.conn.execute(
            f"SELECT stage, question_id, subject_key, label, confidence_or_noul, "
            f"outcome, consumed_as, cache_hit, offline_reason FROM judgments {where} "
            f"ORDER BY ts DESC, id DESC LIMIT ?", args + (limit,)))
        return {"total": sum(by_outcome.values()), "by_outcome": by_outcome,
                "recent": recent}

    # ---- the honesty metric ---------------------------------------------
    # stats() and honesty_report() read verdicts and applied ONLY. The judgments
    # table is not evidence and must never be counted here.
    def stats(self, session_only: bool = False) -> LedgerStats:
        s = LedgerStats()
        where, args = ("WHERE session=?", (self.session,)) if session_only else ("", ())

        for row in self.conn.execute(
                f"SELECT evidence, COUNT(*) c FROM applied {where} GROUP BY evidence", args):
            s.by_evidence[row["evidence"]] = row["c"]
            s.applied_total += row["c"]

        for row in self.conn.execute(
                f"SELECT reason, COUNT(*) c FROM verdicts WHERE admitted=0 "
                + ("AND session=? " if session_only else "") + "GROUP BY reason", args):
            s.denied_total += row["c"]
            # Bucket by the leading clause so the histogram is readable rather
            # than one bucket per unique string.
            key = (row["reason"] or "unknown").split("—")[0].strip()[:70]
            s.deny_reasons[key] = s.deny_reasons.get(key, 0) + row["c"]

        q = lambda sql: self.conn.execute(  # noqa: E731
            f"{sql} " + ("AND session=?" if session_only and "WHERE" in sql else
                         ("WHERE session=?" if session_only else "")), args).fetchone()[0]

        s.landed = q("SELECT COUNT(*) FROM applied WHERE landed=1")
        s.landed_untrusted = q("SELECT COUNT(*) FROM applied WHERE landed IS NULL")
        s.measured_wins = q("SELECT COUNT(*) FROM applied WHERE effect IS NOT NULL AND effect>0")
        s.reverted = q("SELECT COUNT(*) FROM applied WHERE reverted=1")
        s.destructive = q("SELECT COUNT(*) FROM verdicts WHERE destructive_reset=1 AND admitted=1")
        return s

    def honesty_report(self, session_only: bool = False) -> str:
        s = self.stats(session_only=session_only)
        scope = "this session" if session_only else "all time"
        lines = [f"V12 ledger — {scope}  ({self.path})", "-" * 68, s.headline(), ""]
        lines.append(s.honest_note())
        lines.append("")
        lines.append(f"  landed (read-back == target) : {s.landed}")
        lines.append(f"  applied, read-back NOT trusted: {s.landed_untrusted}  "
                     f"(reset-only / lying targets — landed is unknown by design)")
        lines.append(f"  applied with a measured effect: {s.measured_wins}")
        lines.append(f"  reverted                      : {s.reverted}")
        lines.append(f"  destructive resets permitted  : {s.destructive}")
        if s.deny_reasons:
            lines.append("")
            lines.append("  refused, by reason:")
            for k, c in sorted(s.deny_reasons.items(), key=lambda kv: -kv[1])[:10]:
                lines.append(f"    {c:>4}x  {k}")
        # Withdrawn baselines. They are counted here rather than hidden,
        # because a baseline that was used to gate a verdict and then
        # withdrawn is part of the honest picture: the verdicts built on it
        # did not hold, and a reader should be able to see that.
        sup = self.superseded_baselines()
        if sup:
            lines.append("")
            lines.append(f"  baselines SUPERSEDED (withdrawn as references): {len(sup)}")
            for r in sup:
                lines.append(f"    {r['id']}  {r['metric']:<14} mean {r['mean']:>9.4g}  "
                             f"{r['workload']}")
                # sqlite3.Row has no .get(); index by key or read the value
                # directly. A reason is mandatory on a supersede, so it is
                # present, but it is read defensively rather than assumed.
                reason = r["reason"] if "reason" in r.keys() else None
                if reason:
                    lines.append(f"        reason: {reason}")
        return "\n".join(lines)

    def superseded_baselines(self) -> List[sqlite3.Row]:
        """Baselines withdrawn as references, newest first."""
        return list(self.conn.execute(
            "SELECT id, objective, workload, metric, mean, n, path, "
            "superseded_reason AS reason FROM baselines "
            "WHERE superseded=1 ORDER BY ts DESC"))

    def recent_applied(self, limit: int = 25) -> List[sqlite3.Row]:
        # Every column of `applied`, not a hand-picked subset: `v12 ledger` reads
        # `reverted` and `v12 rollback` reads `tier`/`change_key`, and a Row raises
        # IndexError on a key the SELECT left out. `rollback` filters on `reverted`,
        # so omitting it is not cosmetic — undo would crash the moment rows exist.
        return list(self.conn.execute(
            "SELECT id, session, ts, change_id, change_key, tier, kind, evidence, "
            "baseline_id, objective, landed, landed_source, effect, p_value, "
            "reverted, reverted_ts, revert_verb, detail FROM applied "
            "ORDER BY ts DESC, id DESC LIMIT ?", (limit,)))


def open_ledger(session: Optional[str] = None) -> Ledger:
    paths.ensure_dirs()
    return Ledger(session=session)


if __name__ == "__main__":
    with open_ledger() as led:
        print(led.honesty_report())
