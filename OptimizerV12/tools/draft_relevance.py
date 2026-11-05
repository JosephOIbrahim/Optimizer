"""
Draft v12/data/catalog_relevance.json with Jev — offline, opt-in, never evidence.

    set V12_JEV=on
    C:\\Python314\\python.exe tools\\draft_relevance.py            # ask, write, print review table
    C:\\Python314\\python.exe tools\\draft_relevance.py --show     # no API: what would be sent

One Jev Score per (change, objective): how directly does this change act on the
cost of that workload, on four concrete levels (LEVELS). The answer is a DRAFT:
the file is written with `reviewed: false` and every entry `source: "jev-draft"`,
so `v12 plan` prints 'order: DRAFT, unreviewed (jev-drafted, not evidence)' until a
human reviews it and flips `reviewed`. Entries a human already wrote
(`source: "human"`) are kept as they are.

Nothing here reaches the Gate. The output orders the `v12 plan` display list and
nothing else (v12/relevance.py). Confidence below MIN_CONFIDENCE is written as
null ('unsure'): Score confidence is how concentrated the answer is, not how
likely it is to be right, and a hint nobody should trust is better absent.

Fail-closed: an offline/error/cap answer aborts the run and writes no catalog
file. A rejected answer for one entry leaves that level null and says why.
Answers that came back ok are cached by (sha256 of the redacted state, objective,
model) in the V12 data root, so a re-run after a failure, or after a catalog edit,
asks only for what changed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from v12 import jev, paths, relevance  # noqa: E402

#: The workload each objective names, as a situation rather than a metric name.
#: Kept beside the v12.objective makers they describe (diffusion_objective,
#: vellum_objective, render_objective).
WORKLOADS = {
    "diffusion": ("ComfyUI diffusion inference on the RTX 4090: an SDXL-class model "
                  "sampling 1024x1024 images, 20 steps, bf16. Almost all of the time is "
                  "GPU tensor math in the UNet/attention forward passes, bounded by "
                  "VRAM and memory bandwidth; model loading happens once per session."),
    "vellum": ("A Houdini Vellum cloth/softbody solve on the RTX 4090: OpenCL constraint "
               "iterations every substep, plus writing the sim cache to disk each frame. "
               "Almost all of the time is the GPU solver kernels; cache I/O is second."),
    "render": ("A Houdini Karma XPU render at 1920x1080, 64 samples per pixel, on the "
               "RTX 4090: GPU path tracing through OptiX/CUDA. Almost all of the time "
               "is ray tracing and shading on the GPU; scene and texture load comes first."),
}

#: Four levels, each a concrete situation (0-based, matching Jev's Score legend).
LEVELS = (
    "does not act on this workload's critical path (it changes something this "
    "workload never waits on)",
    "acts only on setup or side work around this workload (startup, loading, "
    "caching, launch, stability), not on the work repeated every step or frame",
    "acts on a secondary cost of this workload (part of the repeated per-step or "
    "per-frame work, but not the largest part)",
    "directly acts on the dominant cost of this workload",
)
INSTRUCTIONS = ("`workload` describes one GPU workload on this machine. `title`, `why`, "
                "`impact` and `applies_when` describe one machine setting change. How "
                "directly does that change act on the cost of the workload in `workload`?")

MIN_CONFIDENCE = 0.6
STAGE = "decide"
CONSUMED_AS = "ordering"
DEFAULT_CACHE = paths.DATA_ROOT / "cache" / "relevance_drafts.json"


class DraftAborted(RuntimeError):
    """Jev was unreachable or refused mid-run. Nothing was written."""


def question(objective: str) -> jev.Question:
    return jev.Question.score(f"relevance_{objective}", INSTRUCTIONS, LEVELS)


def catalog_entries(notes: List[str]) -> List[Tuple[str, Dict[str, str]]]:
    """(change id, state without the workload) for every change `v12 plan` lists:
    the dry catalog minus the advisory-only kind. An id that appears twice with
    different text is skipped with a note, as the plane's side map does."""
    from v12.change import ChangeKind, change_key
    from v12.plane import gather_catalog
    cands, meta = gather_catalog(notes)
    out: Dict[str, Dict[str, str]] = {}
    clash = set()
    for c in cands:
        if c.kind is ChangeKind.GUIDED:
            continue
        m = meta.get((change_key(c), c.id))
        state = ({"title": m.title, "why": m.why, "impact": m.impact,
                  "applies_when": ", ".join(m.applies_when)} if m is not None else
                 {"title": c.description, "why": c.benefit_claim or "",
                  "impact": "", "applies_when": ""})
        if c.id in out and out[c.id] != state:
            clash.add(c.id)
        out.setdefault(c.id, state)
    for cid in sorted(clash):
        notes.append(f"change id '{cid}' appears twice with different text - not drafted")
        out.pop(cid, None)
    return list(out.items())


def state_for(base: Dict[str, str], objective: str) -> Dict[str, str]:
    return {**base, "workload": WORKLOADS[objective]}


def cache_key(state: Dict[str, str], objective: str) -> str:
    return f"{jev.state_sha256(jev.redact(dict(state)))}|{objective}|{jev.PINNED_MODEL}"


def _load_cache(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def level_from(j: jev.Judgment) -> Tuple[Optional[int], Optional[float]]:
    """(level, confidence). Level is the most probable legend entry — not the
    label, which is the SDK's score and may be an expectation between levels.
    None when confidence is under MIN_CONFIDENCE."""
    probs = dict(j.probabilities)
    if not probs:
        return None, j.confidence_or_noul
    best = max(sorted(probs), key=lambda k: probs[k])
    conf = j.confidence_or_noul
    if conf is None or conf < MIN_CONFIDENCE:
        return None, conf
    return int(best), conf


def draft(*, out_path: Path = relevance.DEFAULT_PATH, cache_path: Path = DEFAULT_CACHE,
          entries: Optional[List[Tuple[str, Dict[str, str]]]] = None,
          sleep_s: float = 0.2, client: Any = None,
          log=print) -> Dict[str, Any]:
    """Ask, validate, write. Returns a summary dict. Raises DraftAborted (and
    writes no catalog file) when Jev is off, unreachable, capped or erroring."""
    if not jev.switch_on():
        raise DraftAborted(f"Jev is off - set {jev.SWITCH_ENV}=on to draft")
    ok, why = jev.available()
    if not ok:
        raise DraftAborted(f"Jev unavailable: {why}")
    notes: List[str] = []
    if entries is None:
        entries = catalog_entries(notes)
    for n in notes:
        log(f"note: {n}")

    cache = _load_cache(cache_path)
    todo_keys = {cache_key(state_for(b, o), o)
                 for _, b in entries for o in relevance.OBJECTIVES}
    to_send = len(todo_keys - set(cache))
    log(f"{len(entries)} change(s) x {len(relevance.OBJECTIVES)} objective(s); "
        f"{to_send} question(s) to send ({len(todo_keys) - to_send} from the file cache)")

    # jev caps a process at MAX_QUESTIONS_PER_PROCESS so an unattended run cannot
    # turn into unbounded calls. This run is operator-invoked and pre-counted, so
    # the cap is raised to exactly what was counted, for this process only.
    old_cap = jev.MAX_QUESTIONS_PER_PROCESS
    new_cap = max(old_cap, to_send)
    if new_cap != old_cap:
        log(f"jev question cap raised {old_cap} -> {new_cap} for this drafting run only")
    jev.MAX_QUESTIONS_PER_PROCESS = new_cap
    sent0 = jev._sent
    results: Dict[str, Dict[str, Any]] = {}
    unsure = rejected = 0
    probed = False
    try:
        for cid, base in entries:
            levels: Dict[str, Optional[int]] = {}
            why_parts: List[str] = []
            for o in relevance.OBJECTIVES:
                st = state_for(base, o)
                k = cache_key(st, o)
                hit = cache.get(k)
                if hit is None:
                    js = jev.judge(STAGE, f"catalog:{cid}", st, [question(o)],
                                   consumed_as=CONSUMED_AS, ledger=None, client=client)
                    if not js:
                        raise DraftAborted("jev.judge returned nothing (switched off?)")
                    j = js[0]
                    if j.outcome in ("offline", "error", "cap"):
                        raise DraftAborted(f"{cid}/{o}: {j.display()}")
                    if not j.cache_hit:
                        # The first question sent is the probe: if the SDK's
                        # answer shape (legend, model pin) does not fit, every
                        # later one would be rejected the same way.
                        if not probed and j.outcome == "rejected":
                            raise DraftAborted(f"first question rejected "
                                               f"({j.offline_reason}) - the answer "
                                               f"shape does not fit; nothing drafted")
                        probed = True
                        if sleep_s:
                            time.sleep(sleep_s)
                    if j.outcome == "rejected":
                        rejected += 1
                        levels[o] = None
                        why_parts.append(f"{o[0]}=rejected({j.offline_reason})")
                        continue
                    lvl, conf = level_from(j)
                    hit = {"level": lvl, "confidence": conf,
                           "probabilities": dict(j.probabilities),
                           "request_id": j.request_id}
                    cache[k] = hit
                lvl, conf = hit["level"], hit["confidence"]
                levels[o] = lvl
                if lvl is None:
                    unsure += 1
                why_parts.append(f"{o[0]}={'unsure' if lvl is None else lvl}"
                                 + (f"@{conf:.2f}" if isinstance(conf, (int, float)) else ""))
            results[cid] = {**levels,
                            "rationale": f"{jev.PINNED_MODEL} Score, confidence "
                                         + " ".join(why_parts),
                            "source": "jev-draft"}
    finally:
        jev.MAX_QUESTIONS_PER_PROCESS = old_cap
        if cache:
            _save_json(cache_path, cache)
    requests = jev._sent - sent0

    # Keep what a human wrote; replace every earlier draft.
    try:
        prior = relevance.load(out_path)
        for cid, t in prior.entries.items():
            if t.source == "human":
                results[cid] = {**{o: t.level(o) for o in relevance.OBJECTIVES},
                                "rationale": t.rationale, "source": "human"}
    except relevance.RelevanceError:
        pass
    doc = {"schema": relevance.SCHEMA, "model": jev.PINNED_MODEL,
           "drafted_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "reviewed": False, "entries": results}
    relevance.validate(doc)          # never write a file plan would refuse
    _save_json(out_path, doc)
    return {"entries": len(results), "unsure": unsure, "rejected": rejected,
            "requests": requests, "path": str(out_path), "doc": doc}


def review_table(doc: Dict[str, Any], titles: Optional[Dict[str, str]] = None) -> str:
    """Plain table for a human reviewer: id, the three levels, rationale."""
    titles = titles or {}
    L = [f"{'change id':<30} {'diff':>4} {'vell':>4} {'rend':>4}  title / rationale",
         "-" * 100]
    for cid, e in sorted(doc["entries"].items()):
        lv = ["-" if e[o] is None else str(e[o]) for o in relevance.OBJECTIVES]
        L.append(f"{cid:<30} {lv[0]:>4} {lv[1]:>4} {lv[2]:>4}  "
                 + jev.sanitize_for_terminal(titles.get(cid, ""))[:60])
        L.append(f"{'':<30} {'':>4} {'':>4} {'':>4}  "
                 + jev.sanitize_for_terminal(e["rationale"])[:120])
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--show", action="store_true",
                    help="no API call: print the question count and redacted states")
    ap.add_argument("--out", type=Path, default=relevance.DEFAULT_PATH)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--sleep", type=float, default=0.2,
                    help="seconds between sent questions (no retries are configured)")
    args = ap.parse_args(argv)

    notes: List[str] = []
    entries = catalog_entries(notes)
    titles = {cid: b["title"] for cid, b in entries}
    if args.show:
        for n in notes:
            print(f"note: {n}")
        uniq = {cache_key(state_for(b, o), o) for _, b in entries
                for o in relevance.OBJECTIVES}
        cached = len(uniq & set(_load_cache(args.cache)))
        print(f"{len(entries)} change(s); {len(uniq)} distinct question(s), "
              f"{cached} already cached, {len(uniq) - cached} would be sent")
        for cid, b in entries[:3]:
            print(f"\n[{cid}] redacted state as sent:")
            print(json.dumps(jev.redact(state_for(b, "diffusion")), indent=2))
        return 0
    try:
        s = draft(out_path=args.out, cache_path=args.cache, entries=entries,
                  sleep_s=args.sleep)
    except DraftAborted as e:
        print(f"ABORTED - nothing written: {e}")
        return 2
    print(review_table(s["doc"], titles))
    print()
    print(f"wrote {s['path']}: {s['entries']} entries, {s['unsure']} unsure level(s), "
          f"{s['rejected']} rejected, {s['requests']} request(s); reviewed=false")
    print(relevance.DRAFT_LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
