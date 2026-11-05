"""
V12 — measurement ORDER, nothing else.

The catalog holds ~50 changes and measuring one honestly takes minutes of quiet
GPU. So the order they are measured in matters, and nothing in the Gate has an
opinion about it. This module supplies one: a static, reviewable file of
per-objective relevance tags (v12/data/catalog_relevance.json), 0-3 per change
per objective, that sorts the `v12 plan` list.

What it may and may not touch, stated once:

  * It reads a JSON file and returns indices. It never imports the Gate, a
    Verdict, a Baseline, a Change, the plane or jev (tests/test_relevance.py
    scans the source), and it never changes what is a candidate — only the order
    an operator is shown.
  * An order-hint is not evidence. A level-3 tag on an untimed change admits
    nothing; the Gate never sees this file. `Plane.run_cycle` evaluates in catalog
    order regardless.
  * The tags may be drafted offline by Jev (tools/draft_relevance.py). A drafted
    file says `reviewed: false` until a human flips it, and every plan printed
    from it says so on its own line (DRAFT_LINE).

Anything malformed is refused whole (RelevanceError): a half-trusted ordering
file is worse than catalog order, which is what the caller falls back to.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "v12.catalog_relevance/1"
#: The objectives `v12 plan --objective` accepts; one tag field each.
OBJECTIVES = ("diffusion", "vellum", "render")
MAX_LEVEL = 3
SOURCES = frozenset({"jev-draft", "human"})
TOP_KEYS = frozenset({"schema", "model", "drafted_ts", "reviewed", "entries"})
ENTRY_KEYS = frozenset(OBJECTIVES) | {"rationale", "source"}

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "catalog_relevance.json"

#: Printed by `v12 plan` whenever the file it ordered from is not reviewed.
DRAFT_LINE = "order: DRAFT, unreviewed (jev-drafted, not evidence)"


class RelevanceError(ValueError):
    """The relevance file is missing, unreadable or malformed. Callers list in
    catalog order and say why."""


@dataclass(frozen=True)
class Tag:
    """One change's levels. None means 'unsure' / not tagged for that objective."""
    levels: Tuple[Tuple[str, Optional[int]], ...]
    rationale: str = ""
    source: str = "human"

    def level(self, objective: str) -> Optional[int]:
        return dict(self.levels).get(objective)


@dataclass(frozen=True)
class RelevanceFile:
    model: str
    drafted_ts: Optional[str]
    reviewed: bool
    entries: Mapping[str, Tag] = field(default_factory=dict)

    def level(self, change_id: str, objective: str) -> Optional[int]:
        t = self.entries.get(change_id)
        return t.level(objective) if t is not None else None


def _level(v: Any, where: str) -> Optional[int]:
    if v is None:
        return None
    # bool is an int subclass; True must not read as level 1.
    if isinstance(v, bool) or not isinstance(v, int):
        raise RelevanceError(f"{where}: level must be an integer 0-{MAX_LEVEL} or null, "
                             f"not {v!r}")
    if not 0 <= v <= MAX_LEVEL:
        raise RelevanceError(f"{where}: level {v} outside 0-{MAX_LEVEL}")
    return v


def validate(doc: Any) -> RelevanceFile:
    """The parsed file, or RelevanceError. Strict: unknown keys are refused, so a
    field that looks load-bearing and is not cannot creep in."""
    if not isinstance(doc, dict):
        raise RelevanceError("top level must be an object")
    extra = set(doc) - TOP_KEYS
    missing = TOP_KEYS - set(doc)
    if extra or missing:
        raise RelevanceError(f"top-level keys: missing {sorted(missing)}, "
                             f"unknown {sorted(extra)}")
    if doc["schema"] != SCHEMA:
        raise RelevanceError(f"schema {doc['schema']!r} is not {SCHEMA!r}")
    if not isinstance(doc["model"], str):
        raise RelevanceError("model must be a string")
    if doc["drafted_ts"] is not None and not isinstance(doc["drafted_ts"], str):
        raise RelevanceError("drafted_ts must be a string or null")
    if not isinstance(doc["reviewed"], bool):
        raise RelevanceError("reviewed must be true or false")
    if not isinstance(doc["entries"], dict):
        raise RelevanceError("entries must be an object")
    entries: Dict[str, Tag] = {}
    for cid, e in doc["entries"].items():
        where = f"entry {cid!r}"
        if not isinstance(cid, str) or not cid.strip():
            raise RelevanceError("entry ids must be non-empty strings")
        if not isinstance(e, dict):
            raise RelevanceError(f"{where} must be an object")
        extra = set(e) - ENTRY_KEYS
        missing = ENTRY_KEYS - set(e)
        if extra or missing:
            raise RelevanceError(f"{where}: missing {sorted(missing)}, "
                                 f"unknown {sorted(extra)}")
        if e["source"] not in SOURCES:
            raise RelevanceError(f"{where}: source {e['source']!r} is not one of "
                                 f"{sorted(SOURCES)}")
        if not isinstance(e["rationale"], str):
            raise RelevanceError(f"{where}: rationale must be a string")
        levels = tuple((o, _level(e[o], f"{where}.{o}")) for o in OBJECTIVES)
        entries[cid] = Tag(levels=levels, rationale=e["rationale"], source=e["source"])
    return RelevanceFile(model=doc["model"], drafted_ts=doc["drafted_ts"],
                         reviewed=doc["reviewed"], entries=entries)


def _no_constants(name: str) -> Any:
    raise RelevanceError(f"non-finite number {name} in relevance file")


def load(path: Optional[Path] = None) -> RelevanceFile:
    """Read and validate. RelevanceError on anything wrong, including absence."""
    p = Path(path) if path is not None else DEFAULT_PATH
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise RelevanceError(f"cannot read {p}: {type(e).__name__}") from None
    try:
        doc = json.loads(text, parse_constant=_no_constants)
    except ValueError as e:
        raise RelevanceError(f"{p} is not valid JSON: {e}") from None
    return validate(doc)


def to_document(rf: RelevanceFile) -> Dict[str, Any]:
    """The JSON shape of a RelevanceFile (inverse of validate)."""
    return {
        "schema": SCHEMA, "model": rf.model, "drafted_ts": rf.drafted_ts,
        "reviewed": rf.reviewed,
        "entries": {cid: {**{o: t.level(o) for o in OBJECTIVES},
                          "rationale": t.rationale, "source": t.source}
                    for cid, t in rf.entries.items()},
    }


def order(change_ids: Sequence[str], rf: Optional[RelevanceFile],
          objective: str) -> List[Tuple[int, Optional[int]]]:
    """(original index, level) pairs in measure-next order: level descending,
    then original index. Entries with no level for this objective (absent, or
    null = unsure) come after every tagged one, in catalog order. With no file,
    that is catalog order throughout. Unreviewed files order the same way; the
    caller prints DRAFT_LINE."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective {objective!r} is not one of {OBJECTIVES}")
    rows = [(i, rf.level(cid, objective) if rf is not None else None)
            for i, cid in enumerate(change_ids)]
    return sorted(rows, key=lambda r: (r[1] is None, -(r[1] or 0), r[0]))


def is_draft(rf: Optional[RelevanceFile]) -> bool:
    """True when the order shown came from a file nobody has reviewed."""
    return rf is not None and not rf.reviewed

