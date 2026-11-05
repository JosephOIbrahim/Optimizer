"""
Tests for measurement ORDER: v12/relevance.py, `v12 plan`, the plane's proposal
metadata side map, and the offline draft tool (tools/draft_relevance.py).

What these hold:
  * the order is level-descending then catalog index; untagged/unsure rows sort
    last, in catalog order; an unreviewed file prints the DRAFT line;
  * the relevance file is validated whole — a bad level, a bool posing as a
    level, an unknown key or source is refused, and `plan` falls back to catalog
    order rather than half-trusting it;
  * `plan` is read-only: it builds no Plane and no Gate, evaluates nothing,
    records nothing and snapshots nothing (each is patched to raise);
  * the side map is keyed by (change_key, id), never copied into a Change, and a
    (key, id) delivered by two proposals with different text is dropped;
  * relevance.py imports nothing that can admit (source scan);
  * the draft tool sends one Score per (entry, objective), writes reviewed=false
    and source='jev-draft', nulls a low-confidence level, aborts without writing
    on offline/error, restores jev's cap, and its file cache stops a resend.

No network: jev's SDK seam is a fake with no transport, the client is scripted,
and the real TYPESAFE_API_KEY is replaced with a dummy. Files go to temp dirs;
nothing touches D:\\OptimizerV12 or v12/data.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))

from v12 import cli, jev, plane, relevance  # noqa: E402
from v12.admission import Gate  # noqa: E402
from v12.change import Change, ChangeKind, EvidenceClass, Snapshot, change_key  # noqa: E402
from v12.ledger import Ledger  # noqa: E402

import draft_relevance as draft_mod  # noqa: E402

DUMMY_KEY = "dummy-not-a-real-key"
EVIDENCE_WORDS = {"measured", "derived", "cited", "guided"}


def _change(cid, kind=ChangeKind.ENV_VAR, desc=None):
    return Change(id=cid, kind=kind, description=desc or f"desc {cid}",
                  target=f"VAR_{cid}", evidence=EvidenceClass.CITED,
                  evidence_source="test", snapshot=Snapshot(value="0", existed=True,
                                                            trusted=True, source="t"))


def _doc(entries, reviewed=False, **top):
    d = {"schema": relevance.SCHEMA, "model": "jev-1.13.0",
         "drafted_ts": "2026-10-01T00:00:00Z", "reviewed": reviewed, "entries": entries}
    d.update(top)
    return d


def _entry(d=None, v=None, r=None, source="jev-draft", rationale="r"):
    return {"diffusion": d, "vellum": v, "render": r, "rationale": rationale,
            "source": source}


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v12_relevance_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def write(self, doc, name="rel.json") -> Path:
        p = self.tmp / name
        p.write_text(json.dumps(doc) if not isinstance(doc, str) else doc,
                     encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------
class TestOrder(unittest.TestCase):
    def rf(self, entries, reviewed=False):
        return relevance.validate(_doc(entries, reviewed))

    def test_level_descending_then_original_index(self):
        rf = self.rf({"a": _entry(d=1), "b": _entry(d=3), "c": _entry(d=1),
                      "d": _entry(d=3), "e": _entry(d=0)})
        got = relevance.order(["a", "b", "c", "d", "e"], rf, "diffusion")
        self.assertEqual(got, [(1, 3), (3, 3), (0, 1), (2, 1), (4, 0)])

    def test_the_objective_selects_the_field(self):
        rf = self.rf({"a": _entry(d=3, r=0), "b": _entry(d=0, r=3)})
        self.assertEqual([i for i, _ in relevance.order(["a", "b"], rf, "diffusion")], [0, 1])
        self.assertEqual([i for i, _ in relevance.order(["a", "b"], rf, "render")], [1, 0])

    def test_absent_and_unsure_sort_after_every_tagged_row_in_catalog_order(self):
        rf = self.rf({"tagged0": _entry(d=0), "unsure": _entry(d=None),
                      "tagged2": _entry(d=2)})
        got = relevance.order(["absent1", "unsure", "tagged0", "absent2", "tagged2"],
                              rf, "diffusion")
        self.assertEqual(got, [(4, 2), (2, 0), (0, None), (1, None), (3, None)])

    def test_no_file_is_catalog_order(self):
        self.assertEqual(relevance.order(["x", "y", "z"], None, "vellum"),
                         [(0, None), (1, None), (2, None)])

    def test_unknown_objective_is_refused(self):
        with self.assertRaises(ValueError):
            relevance.order(["x"], None, "game")

    def test_draft_flag_follows_reviewed(self):
        self.assertTrue(relevance.is_draft(self.rf({}, reviewed=False)))
        self.assertFalse(relevance.is_draft(self.rf({}, reviewed=True)))
        self.assertFalse(relevance.is_draft(None))


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class TestValidation(_TmpCase):
    def test_a_valid_file_loads_and_round_trips(self):
        doc = _doc({"a": _entry(d=3, v=0, r=None, source="human")}, reviewed=True)
        rf = relevance.load(self.write(doc))
        self.assertEqual(rf.level("a", "diffusion"), 3)
        self.assertIsNone(rf.level("a", "render"))
        self.assertEqual(relevance.to_document(rf), doc | {"drafted_ts": doc["drafted_ts"]})

    def test_bad_levels_are_rejected(self):
        for bad in (4, -1, 1.5, True, False, "2", [1]):
            with self.subTest(level=bad), self.assertRaises(relevance.RelevanceError):
                relevance.validate(_doc({"a": _entry(d=bad)}))

    def test_entry_shape_is_strict(self):
        e = _entry(d=1)
        cases = {
            "unknown key": {**e, "confidence": 0.9},
            "missing key": {k: v for k, v in e.items() if k != "render"},
            "bad source": {**e, "source": "jev"},
            "rationale not text": {**e, "rationale": 3},
            "not an object": [1, 2, 3],
        }
        for name, entry in cases.items():
            with self.subTest(name), self.assertRaises(relevance.RelevanceError):
                relevance.validate(_doc({"a": entry}))

    def test_top_level_shape_is_strict(self):
        cases = {
            "reviewed as text": _doc({}, reviewed="false"),
            "schema mismatch": _doc({}, schema="v12.catalog_relevance/0"),
            "unknown key": _doc({}, verdict="admit"),
            "entries a list": _doc([]),
            "not an object": [],
        }
        for name, doc in cases.items():
            with self.subTest(name), self.assertRaises(relevance.RelevanceError):
                relevance.validate(doc)

    def test_load_refuses_missing_file_bad_json_and_nan(self):
        with self.assertRaises(relevance.RelevanceError):
            relevance.load(self.tmp / "absent.json")
        with self.assertRaises(relevance.RelevanceError):
            relevance.load(self.write("{not json", "bad.json"))
        nan = json.dumps(_doc({"a": _entry(d=1)})).replace('"diffusion": 1',
                                                             '"diffusion": NaN')
        with self.assertRaises(relevance.RelevanceError):
            relevance.load(self.write(nan, "nan.json"))


# ---------------------------------------------------------------------------
# relevance.py is display-order only
# ---------------------------------------------------------------------------
class TestRelevanceSource(unittest.TestCase):
    FORBIDDEN = {"admission", "baseline", "change", "plane", "jev", "ledger",
                 "actuators", "objective", "typesafe_sdk"}

    def test_relevance_imports_nothing_that_can_admit(self):
        tree = ast.parse((PROJECT / "v12" / "relevance.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[-1] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[-1])
                names |= {a.name for a in node.names}
        self.assertFalse(names & self.FORBIDDEN, sorted(names & self.FORBIDDEN))

    def test_the_cycle_does_not_consult_relevance(self):
        src = (PROJECT / "v12" / "plane.py").read_text(encoding="utf-8")
        self.assertNotIn("relevance", src)


# ---------------------------------------------------------------------------
# v12 plan
# ---------------------------------------------------------------------------
_CANDS = [_change("alpha", desc="Alpha lever"),
          _change("beta", desc="Beta lever"),
          _change("advice", kind=ChangeKind.GUIDED, desc="Advice only"),
          _change("gamma", desc="Gamma lever"),
          _change("delta", desc="Delta lever")]


def _fake_gather(notes):
    notes.append("catalog stub")
    meta = {(change_key(_CANDS[1]), "beta"): plane.ProposalMeta(title="Beta title")}
    return list(_CANDS), meta


class TestPlan(_TmpCase):
    def setUp(self):
        super().setUp()
        boom = AssertionError("plan reached the gate/ledger/plane/snapshot")
        for p in (
            mock.patch.object(plane, "gather_catalog", side_effect=_fake_gather),
            mock.patch.object(Gate, "__init__", side_effect=boom),
            mock.patch.object(Gate, "evaluate", side_effect=boom),
            mock.patch.object(Ledger, "__init__", side_effect=boom),
            mock.patch.object(Ledger, "record_verdict", side_effect=boom),
            mock.patch.object(plane.Plane, "__init__", side_effect=boom),
            mock.patch.object(plane, "snapshot_change", side_effect=boom),
            mock.patch.object(cli, "open_ledger", side_effect=boom),
        ):
            p.start()
            self.addCleanup(p.stop)

    def plan(self, rel_path, objective="diffusion"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cli.main(["plan", "--objective", objective, "--relevance", str(rel_path)])
        self.assertEqual(rc, 0)
        return out.getvalue()

    def ids_in_order(self, text):
        return [ln.rsplit("[", 1)[1].rstrip("]") for ln in text.splitlines()
                if ln.rstrip().endswith("]") and "  [" in ln]

    def test_orders_by_level_and_never_touches_the_gate(self):
        p = self.write(_doc({"alpha": _entry(d=1), "gamma": _entry(d=3),
                             "delta": _entry(d=3), "advice": _entry(d=3)}))
        text = self.plan(p)
        self.assertEqual(self.ids_in_order(text), ["gamma", "delta", "alpha", "beta"])
        self.assertIn("system|env|user|VAR_gamma", text)
        self.assertIn("3/3", text)
        self.assertIn("Beta title  [beta]", text)        # side-map title
        self.assertIn("Alpha lever  [alpha]", text)       # description fallback

    def test_unreviewed_prints_the_draft_line_and_reviewed_does_not(self):
        text = self.plan(self.write(_doc({"alpha": _entry(d=2)}, reviewed=False)))
        self.assertIn("order: DRAFT, unreviewed (jev-drafted, not evidence)", text)
        text = self.plan(self.write(_doc({"alpha": _entry(d=2, source="human")},
                                         reviewed=True)))
        self.assertNotIn("DRAFT", text)

    def test_untagged_count_and_advisory_rows_are_reported(self):
        text = self.plan(self.write(_doc({"alpha": _entry(d=2), "beta": _entry(d=None)})))
        self.assertIn("untagged: 3 of 4", text)
        self.assertIn("advisory-only (never applied): 1 not listed", text)
        self.assertNotIn("[advice]", text)

    def test_an_invalid_file_falls_back_to_catalog_order(self):
        text = self.plan(self.write(_doc({"gamma": _entry(d=7)})))
        self.assertIn("none usable", text)
        self.assertNotIn("DRAFT", text)
        self.assertEqual(self.ids_in_order(text), ["alpha", "beta", "gamma", "delta"])
        self.assertIn("untagged: 4 of 4", text)

    def test_plan_wording_names_no_evidence_class(self):
        for doc in (_doc({"alpha": _entry(d=2)}), _doc({}, reviewed=True)):
            text = self.plan(self.write(doc)).lower()
            words = set(text.replace("(", " ").replace(")", " ").replace(":", " ")
                        .replace(",", " ").split())
            self.assertFalse(words & EVIDENCE_WORDS, sorted(words & EVIDENCE_WORDS))
            self.assertIn("order-hint", text)

    def test_plan_requires_an_objective(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["plan"])


class TestPlanEndToEndWithoutMos(unittest.TestCase):
    """The real gather_catalog with MOS2026 unreachable: builtins only, and still
    no snapshot, Gate or ledger."""

    def test_builtin_catalog_lists_without_snapshots(self):
        boom = AssertionError("plan reached the gate/ledger/snapshot")
        with mock.patch.dict(sys.modules, {"optimizer": None, "optimizer.base": None,
                                           "optimizer.probe": None,
                                           "optimizer.orchestrator": None}), \
             mock.patch.object(plane, "snapshot_change", side_effect=boom), \
             mock.patch.object(Gate, "__init__", side_effect=boom), \
             mock.patch.object(Ledger, "__init__", side_effect=boom):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = cli.main(["plan", "--objective", "vellum", "--relevance",
                               str(PROJECT / "no_such_relevance.json")])
        self.assertEqual(rc, 0)
        self.assertIn("[kernel_vellum_gemm]", out.getvalue())
        self.assertIn("MOS2026 catalog NOT loaded", out.getvalue())


# ---------------------------------------------------------------------------
# the side map
# ---------------------------------------------------------------------------
def _mos_change(cid, target):
    return SimpleNamespace(
        id=cid, type=SimpleNamespace(value="env_var"), risk=SimpleNamespace(value="low"),
        scope=SimpleNamespace(value="user"), description=f"mos {cid}", target=target,
        target_value="1", reg_value_name=None, reg_type=None, file_key=None,
        file_sep="=", snapshot_cmd=None, apply_cmd=None, undo_cmd=None,
        verify_cmd=None, extra={})


def _proposal(title, changes, why="w"):
    return SimpleNamespace(title=title, domain="gpu", why=why, impact="i",
                           applies_when=[SimpleNamespace(value="idle")],
                           note_2026="", requires_admin=False, changes=changes)


def _mos_modules(proposals):
    pkg = types.ModuleType("optimizer")
    pkg.__path__ = []
    base = types.ModuleType("optimizer.base")
    base.WorkloadMode = SimpleNamespace(IDLE="idle")
    probe_mod = types.ModuleType("optimizer.probe")
    probe_mod.probe = lambda: SimpleNamespace()
    orch_mod = types.ModuleType("optimizer.orchestrator")

    class Orchestrator:
        def __init__(self, cfg, dry_run=True):
            assert dry_run is True
            self.cfg = cfg

        def gather(self, ctx):
            return list(proposals)

    orch_mod.Orchestrator = Orchestrator
    orch_mod.load_config = lambda: {"min_confidence": 0.6}
    pkg.base, pkg.probe, pkg.orchestrator = base, probe_mod, orch_mod
    return {"optimizer": pkg, "optimizer.base": base,
            "optimizer.probe": probe_mod, "optimizer.orchestrator": orch_mod}


class TestSideMap(unittest.TestCase):
    def gather(self, proposals):
        saved = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), saved))
        notes = []
        with mock.patch.dict(sys.modules, _mos_modules(proposals)):
            cands, meta = plane.gather_catalog(notes)
        return cands, meta, notes

    def test_keyed_by_change_key_and_id_and_never_copied_into_a_change(self):
        a = _mos_change("mos_a", "VAR_A")
        cands, meta, _ = self.gather([_proposal("Proposal A", [a], why="because")])
        c = next(c for c in cands if c.id == "mos_a")
        m = meta[(change_key(c), "mos_a")]
        self.assertEqual((m.title, m.why, m.applies_when), ("Proposal A", "because", ("idle",)))
        # Not in any Change field: the Gate and actuators cannot see it.
        flat = json.dumps({k: str(v) for k, v in vars(c).items()})
        self.assertNotIn("Proposal A", flat)
        self.assertNotIn("because", flat)
        self.assertNotIn("title", c.extra)

    def test_same_key_from_two_proposals_with_different_text_is_dropped(self):
        a1, a2 = _mos_change("mos_a", "VAR_A"), _mos_change("mos_a", "VAR_A")
        b = _mos_change("mos_b", "VAR_B")
        cands, meta, notes = self.gather([_proposal("First", [a1, b]),
                                          _proposal("Second", [a2])])
        ids = {k[1] for k in meta}
        self.assertNotIn("mos_a", ids)
        self.assertIn("mos_b", ids)
        self.assertTrue(any("'mos_a'" in n and "more than one proposal" in n
                            for n in notes), notes)
        # A third, agreeing delivery does not resurrect the dropped entry.
        cands, meta, _ = self.gather([_proposal("First", [a1]), _proposal("Second", [a2]),
                                      _proposal("First", [_mos_change("mos_a", "VAR_A")])])
        self.assertNotIn("mos_a", {k[1] for k in meta})

    def test_same_key_with_identical_text_is_kept(self):
        a1, a2 = _mos_change("mos_a", "VAR_A"), _mos_change("mos_a", "VAR_A")
        _, meta, notes = self.gather([_proposal("Same", [a1]), _proposal("Same", [a2])])
        self.assertIn("mos_a", {k[1] for k in meta})
        self.assertFalse(any("more than one proposal" in n for n in notes))

    def test_same_id_on_different_targets_are_two_entries(self):
        _, meta, _ = self.gather([_proposal("One", [_mos_change("mos_a", "VAR_1")]),
                                  _proposal("Two", [_mos_change("mos_a", "VAR_2")])])
        self.assertEqual(sorted(m.title for k, m in meta.items() if k[1] == "mos_a"),
                         ["One", "Two"])

    def test_plane_gather_candidates_keeps_the_map(self):
        tmp = tempfile.mkdtemp(prefix="v12_relevance_plane_")
        self.addCleanup(shutil.rmtree, tmp, True)
        led = Ledger(db_path=Path(tmp) / "ledger.db")
        self.addCleanup(led.close)
        with mock.patch.object(plane, "SystemActuator", lambda dry_run=True: None), \
             mock.patch.object(plane, "snapshot_change",
                               return_value=Snapshot(value=None, existed=False,
                                                     trusted=True, source="t")), \
             mock.patch.dict(sys.modules, _mos_modules(
                 [_proposal("Kept", [_mos_change("mos_k", "VAR_K")])])):
            pl = plane.Plane(dry_run=True, ledger=led)
            obs = SimpleNamespace(notes=[])
            cands = pl.gather_candidates(obs)
        c = next(c for c in cands if c.id == "mos_k")
        self.assertEqual(pl.proposal_meta[(change_key(c), "mos_k")].title, "Kept")
        self.assertTrue(any("MOS2026 catalog loaded" in n for n in obs.notes))


# ---------------------------------------------------------------------------
# the draft tool (mocked client)
# ---------------------------------------------------------------------------
class _FakeSDK:
    class _Q:
        def __init__(self, **kw):
            self.kw = kw
    Choice = type("Choice", (_Q,), {})
    Noul = type("Noul", (_Q,), {})
    Score = type("Score", (_Q,), {})

    class RetryPolicy:
        def __init__(self, max_retries=2):
            self.max_retries = max_retries

    class TypeSafeClient:
        def __init__(self, **kw):
            raise AssertionError("a real TypeSafeClient was constructed in a test")


class _ScoreClient:
    """Answers each Score with a level chosen from the state, or raises."""

    def __init__(self, pick, conf=0.9, exc=None, model=jev.PINNED_MODEL):
        self.pick, self.conf, self.exc, self.model = pick, conf, exc, model
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((dict(state), dict(questions)))
        if self.exc is not None:
            raise self.exc
        scores = {}
        for qid, q in questions.items():
            lvl = self.pick(state, qid)
            probs = {i: (0.85 if i == lvl else 0.05) for i in range(4)}
            conf = self.conf(state, qid) if callable(self.conf) else self.conf
            scores[qid] = SimpleNamespace(score=float(lvl), confidence=conf,
                                          probabilities=probs,
                                          legend={i: f"l{i}" for i in range(4)})
        return SimpleNamespace(model=self.model, request_id="req", choices={},
                               nouls={}, scores=scores)


_ENTRIES = [
    ("comfy_launcher", {"title": "Tuned ComfyUI launcher", "why": "attention",
                        "impact": "faster diffusion", "applies_when": "idle"}),
    ("houdini_temp", {"title": "Houdini temp -> NVMe", "why": "sim cache",
                      "impact": "faster caching", "applies_when": "idle"}),
    ("houdini_temp_twin", {"title": "Houdini temp -> NVMe", "why": "sim cache",
                           "impact": "faster caching", "applies_when": "idle"}),
]


def _pick(state, qid):
    if "ComfyUI" in state["title"]:
        return 3 if qid == "relevance_diffusion" else 0
    return 1


class TestDraft(_TmpCase):
    def setUp(self):
        super().setUp()
        jev._reset()
        self.addCleanup(jev._reset)
        for p in (mock.patch.dict(os.environ, {jev.SWITCH_ENV: "on",
                                               jev.KEY_ENV: DUMMY_KEY}),
                  mock.patch.object(jev, "_import_sdk", return_value=_FakeSDK),
                  mock.patch.object(jev, "_sdk_importable", return_value=True)):
            p.start()
            self.addCleanup(p.stop)
        self.out = self.tmp / "catalog_relevance.json"
        self.cache = self.tmp / "cache.json"

    def run_draft(self, client, entries=_ENTRIES):
        return draft_mod.draft(out_path=self.out, cache_path=self.cache,
                               entries=entries, sleep_s=0, client=client,
                               log=lambda *_: None)

    def test_one_score_per_entry_and_objective_written_as_an_unreviewed_draft(self):
        client = _ScoreClient(_pick)
        s = self.run_draft(client)
        rf = relevance.load(self.out)
        self.assertFalse(rf.reviewed)
        self.assertEqual(rf.model, jev.PINNED_MODEL)
        self.assertEqual({t.source for t in rf.entries.values()}, {"jev-draft"})
        self.assertEqual(rf.level("comfy_launcher", "diffusion"), 3)
        self.assertEqual(rf.level("comfy_launcher", "render"), 0)
        self.assertEqual(rf.level("houdini_temp_twin", "vellum"), 1)
        # The twin has the same text, so it is a cache hit, not a request.
        self.assertEqual(s["requests"], 6)
        self.assertEqual(len(client.calls), 6)
        for state, qs in client.calls:
            self.assertEqual(set(state), {"title", "why", "impact", "applies_when",
                                          "workload"})
            (qid, q), = qs.items()
            self.assertIsInstance(q, _FakeSDK.Score)
            self.assertEqual(len(q.kw["criteria"]), 4)
            self.assertIn(state["workload"], draft_mod.WORKLOADS.values())
        self.assertIn("comfy_launcher", draft_mod.review_table(s["doc"]))

    def test_low_confidence_is_written_as_unsure(self):
        conf = lambda st, qid: 0.59 if qid == "relevance_vellum" else 0.6  # noqa: E731
        s = self.run_draft(_ScoreClient(_pick, conf=conf))
        rf = relevance.load(self.out)
        self.assertIsNone(rf.level("comfy_launcher", "vellum"))
        self.assertEqual(rf.level("comfy_launcher", "diffusion"), 3)
        self.assertEqual(s["unsure"], 3)
        self.assertIn("v=unsure@0.59", rf.entries["comfy_launcher"].rationale)

    def test_offline_aborts_and_writes_nothing(self):
        class APIConnectionError(Exception):
            pass
        with self.assertRaises(draft_mod.DraftAborted):
            self.run_draft(_ScoreClient(_pick, exc=APIConnectionError()))
        self.assertFalse(self.out.exists())
        self.assertEqual(jev.MAX_QUESTIONS_PER_PROCESS, 40)

    def test_a_rejected_first_answer_aborts_and_writes_nothing(self):
        with self.assertRaises(draft_mod.DraftAborted):
            self.run_draft(_ScoreClient(_pick, model="jev-9.9.9"))
        self.assertFalse(self.out.exists())

    def test_off_means_no_call_and_no_file(self):
        client = _ScoreClient(_pick)
        with mock.patch.dict(os.environ, {jev.SWITCH_ENV: "off"}):
            with self.assertRaises(draft_mod.DraftAborted):
                self.run_draft(client)
        self.assertEqual(client.calls, [])
        self.assertFalse(self.out.exists())

    def test_the_cap_is_raised_only_for_the_run_and_restored(self):
        many = [(f"c{i}", {"title": f"t{i}", "why": "", "impact": "",
                           "applies_when": ""}) for i in range(20)]
        s = self.run_draft(_ScoreClient(lambda st, q: 2), entries=many)
        self.assertEqual(s["requests"], 60)          # over the 40-per-process cap
        self.assertEqual(jev.MAX_QUESTIONS_PER_PROCESS, 40)

    def test_the_file_cache_stops_a_resend(self):
        self.run_draft(_ScoreClient(_pick))
        jev._reset()                      # a new process: only the file cache remains
        again = _ScoreClient(_pick)
        s = self.run_draft(again)
        self.assertEqual(again.calls, [])
        self.assertEqual(s["requests"], 0)
        self.assertEqual(relevance.load(self.out).level("comfy_launcher", "diffusion"), 3)

    def test_human_entries_are_kept(self):
        self.out.write_text(json.dumps(_doc({"comfy_launcher": _entry(
            d=0, v=0, r=0, source="human", rationale="checked by hand")},
            reviewed=True)), encoding="utf-8")
        self.run_draft(_ScoreClient(_pick))
        rf = relevance.load(self.out)
        self.assertEqual(rf.entries["comfy_launcher"].source, "human")
        self.assertEqual(rf.level("comfy_launcher", "diffusion"), 0)
        self.assertFalse(rf.reviewed)

    def test_levels_never_carry_an_evidence_label(self):
        q = draft_mod.question("diffusion")
        self.assertEqual(q.kind, "score")
        text = " ".join(q.criteria).lower() + q.instructions.lower()
        self.assertFalse(set(text.split()) & EVIDENCE_WORDS)


if __name__ == "__main__":
    unittest.main()
