"""
Tests that the SOURCE TREE publishes the current figure for every lever.

Why this exists
---------------
The operator cards are where a reader checks a claim against a record. But
the source tree publishes figures too — in module docstrings that explain
WHY a piece of code exists — and those docstrings had gone stale.

Concretely: on 2026-11-06 three levers were re-measured and the cards were
swept to the new figures. The code was not. `harness/levers.py` still said
"the lever is worth +9.79%" and its table still carried `+8.50%` for the
power ceiling; `harness/paired.py` and `OptimizerV12/v12/objective.py` still
said "DENY -0.60%". Every one of those was a real figure that a real record
holds, so the cards' own trace check could not see them — they were
traceable AND superseded, which is the exact combination the cards' checks
each miss on their own.

The rule
--------
A sentence that publishes a lever's verdict must publish that lever's CURRENT
effect. A sentence that quotes a figure as history must quote one that a
WITHDRAWN record holds. A sentence that quotes a documented counter-example
is exempt — but it has to be declared as one.

The shape of the test
---------------------
The verdict figures in the source tree are enumerable — there are a handful,
and they are declared below with what each one means. Declaring them
exactly, rather than pattern-matching prose, is what makes the check
trustworthy: there is no heuristic deciding whether a sentence is history
or a current claim.

The catch-all is the other half. A line carrying a verdict word and a figure
that matches NO declared site FAILS. So a new publication site cannot slip
in undeclared, and a figure added to an existing one cannot hide behind
that one's declaration.

The test files are excluded. Their figures are fixtures — arbitrary strings
like `replacement_verdict="DENY (-0.89%)"` — chosen to exercise a code
path, not to state a measurement.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MEASUREMENTS = REPO / "measurements"

#: The source that publishes figures. The harness, and V12's library.
#: NOT tests/ - a test's figures are fixtures.
SOURCE_GLOBS = [
    ("harness", "*.py"),
    ("OptimizerV12/v12", "*.py"),
]

#: A word that makes a line publish a verdict.
VERDICT_WORD = re.compile(r"\b(ADMIT|DENY)\b|is worth|\bworth\b")

#: Every percentage on a line.
#:
#: The lookbehind matters: Python's format specs carry a percentage sign -
#: `{effect:+.2%}` - and matching inside one would report the PLACES as a
#: published figure. A real figure never starts with a dot.
PERCENT = re.compile(r"(?<!\.)[+-]?\d+\.?\d*%")

#: The declared publication sites.
#:
#: Each entry is (file, regex, kind):
#:
#:   * kind is a LEVER name - the figure the regex captures must be that
#:     lever's CURRENT effect. `("aov_variant", "beauty_only")` names the
#:     variant too.
#:   * kind is "history" - the figure must be one that a WITHDRAWN record
#:     holds, because the sentence is quoting what was once measured.
#:   * kind is "arithmetic" - a figure that is arithmetic on two OTHER
#:     numbers, not a measurement. The phantom's "+0.7%" is one: it is
#:     47.710 against 48.060, and the sentence says so. It has to be
#:     declared like anything else, so a reader can see that it is not a
#:     claim.
#:   * kind is "exempt" - not a measurement at all: a documented
#:     counter-example of a bug, or an illustrative figure. The line is
#:     exempt from the per-figure check, but it is declared, so it stays
#:     visible rather than invisible.
SOURCE_VERDICT_SITES = [
    # --- the CURRENT verdicts. These move when a lever is re-measured. ----
    (r"harness/levers\.py",
     r"the lever is worth ([+-]?\d+\.?\d*)%", ("aov_variant", "beauty_only")),
    (r"harness/levers\.py",
     r"the same change is DENY ([+-]?\d+\.?\d*)%", ("husk_threads", None)),
    (r"harness/paired\.py",
     r"the same change is DENY ([+-]?\d+\.?\d*)%", ("husk_threads", None)),
    (r"harness/sync_measurements\.py",
     r'"DENY ([+-]?\d+\.?\d*)%"', ("husk_threads", None)),
    (r"harness/sync_measurements\.py",
     r'"([+-]?\d+\.?\d*)%, p=2\.4e-06"', ("frames_per_process", None)),
    (r"OptimizerV12/v12/objective\.py",
     r"same change is DENY ([+-]?\d+\.?\d*)%", ("husk_threads", None)),

    # --- HISTORY. The 2026-11-03 phantom, quoted as what it was. --------
    (r"harness/lever_run\.py", r"produced ADMIT ([+-]?\d+\.?\d*)%", "history"),
    (r"harness/__init__\.py", r"an ADMIT of ([+-]?\d+\.?\d*)%", "history"),
    (r"harness/apply_retractions\.py", r"was: ADMIT \(([+-]?\d+\.?\d*)%\)",
     "history"),
    (r"harness/levers\.py",
     r"a verdict of ADMIT ([+-]?\d+\.?\d*)%", "history"),
    (r"harness/method\.py", r"produced a verdict of ADMIT ([+-]?\d+\.?\d*)%",
     "history"),
    (r"harness/method\.py", r"produced ADMIT ([+-]?\d+\.?\d*)%", "history"),
    (r"harness/optimize\.py", r'produced a ([+-]?\d+\.?\d*)% "ADMIT"',
     "history"),
    (r"harness/paired\.py", r"came back ADMIT ([+-]?\d+\.?\d*)%", "history"),
    (r"harness/retract\.py", r"ADMIT  \(([+-]?\d+\.?\d*)%\)", "history"),
    (r"harness/retract\.py", r"DENY, ([+-]?\d+\.?\d*)%", "history"),
    (r"OptimizerV12/v12/baseline\.py", r"back ADMIT ([+-]?\d+\.?\d*)%",
     "history"),
    (r"OptimizerV12/v12/baseline\.py", r"came back ADMIT ([+-]?\d+\.?\d*)%",
     "history"),
    (r"OptimizerV12/v12/objective\.py", r"ADMIT ([+-]?\d+\.?\d*)%, p=",
     "history"),

    # --- ARITHMETIC. A figure computed from two other numbers, not a
    # --- measurement. The phantom's cross-session arithmetic. ------------
    (r"harness/levers\.py", r'"(\+0\.7)%"', "arithmetic"),
    (r"harness/paired\.py", r's: (\+0\.7)%, nothing', "arithmetic"),
    (r"harness/paired\.py", r'"(\+0\.7)%" there', "arithmetic"),
    (r"OptimizerV12/v12/objective\.py", r'"(\+0\.7)%" that first appeared',
     "arithmetic"),

    # --- EXEMPT. Not a measurement: a documented counter-example of a bug,
    # --- or an illustrative figure. Declared so it stays visible. --------
    (r"harness/levers\.py",
     r"becomes ADMIT ([+-]?\d+\.?\d*)% against a compute objective", "exempt"),
    (r"harness/paired\.py",
     r"produced ADMIT ([+-]?\d+\.?\d*)%, \+[+-]?\d+\.?\d*%", "exempt"),
    (r"OptimizerV12/v12/objective\.py",
     r"samples a ([+-]?\d+\.?\d*)% win is statistically real", "exempt"),
]


def _records():
    out = []
    for p in sorted(MEASUREMENTS.glob("*.json")):
        try:
            out.append((p.name, json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return out


def _blocks(doc):
    res = doc.get("results")
    if isinstance(res, dict):
        return [(k, v) for k, v in res.items() if isinstance(v, dict)]
    if isinstance(res, list):
        return [(b.get("lever"), b) for b in res if isinstance(b, dict)]
    return []


def _withdrawn_records():
    """The records the repo has withdrawn: superseded, or holding a retraction."""
    out = {}
    for name, doc in _records():
        if doc.get("superseded") or any(
                b.get("retracted") for _, b in _blocks(doc)):
            out[name] = doc
    return out


def _current_effects():
    """The CURRENT effect for every lever, from the newest live record."""
    withdrawn = _withdrawn_records()
    current = {}
    for name, doc in _records():
        if name in withdrawn:
            continue
        stamp = doc.get("recorded_utc") or ""
        for lever, block in _blocks(doc):
            if block.get("retracted"):
                continue
            key = (lever, block.get("variant"))
            if key not in current or stamp > current[key][0]:
                current[key] = (stamp, block.get("effect"))
    return {k: v[1] for k, v in current.items()}


def _withdrawn_figures():
    """Every figure a withdrawn record holds, in the forms prose prints it.

    Two sources, and both matter. A measured `effect` is one. But the
    2026-11-03 ledger records hold their original claim in PROSE - the
    verdict string and the reason the verdict was withdrawn - and never had
    an `effect` field at all. The phantom's +12.34% is one of those, and it
    is the figure the source tree quotes as history.
    """
    figs = set()
    for doc in _withdrawn_records().values():
        for _, block in _blocks(doc):
            eff = block.get("effect")
            if isinstance(eff, (int, float)) and not isinstance(eff, bool):
                for places in (1, 2):
                    figs.add(f"{eff * 100:+.{places}f}")
                    figs.add(f"{eff * 100:.{places}f}")
        for m in PERCENT.finditer(json.dumps(doc, ensure_ascii=False)):
            figs.add(m.group(0).rstrip("%"))
    return figs


def _source_files():
    files = []
    for sub, pat in SOURCE_GLOBS:
        files.extend(sorted((REPO / sub).glob(pat)))
    return [f for f in files if "__pycache__" not in str(f)]


class TestTheSourceTreePublishesCurrentVerdicts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = _current_effects()
        cls.withdrawn_figures = _withdrawn_figures()
        cls.files = _source_files()
        cls.lines = []
        for f in cls.files:
            rel = f.relative_to(REPO).as_posix()
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                cls.lines.append((rel, n, line))

    def _declared(self, rel: str, line: str):
        """The declared sites this line matches, with the figure captured.

        A declared site is (file, regex, kind): the file says WHERE the
        publication lives, the regex says WHAT it looks like, and the kind
        says what the figure there MEANS.
        """
        out = []
        for filepat, pattern, kind in SOURCE_VERDICT_SITES:
            if not re.search(filepat, rel):
                continue
            m = re.search(pattern, line)
            if not m:
                continue
            fig = None
            if kind != "exempt" and m.groups():
                fig = m.group(1)
            out.append((kind, fig))
        return out

    def test_every_declared_site_still_matches_something(self):
        """The declaration list must not rot.

        A declared site that matches nothing is a stale declaration, and it
        means the check has quietly stopped covering the place it was
        written to cover.
        """
        unmatched = []
        for filepat, pattern, kind in SOURCE_VERDICT_SITES:
            hits = [1 for rel, n, line in self.lines
                    if re.search(filepat, rel) and re.search(pattern, line)]
            if not hits:
                unmatched.append(f"{filepat} :: {pattern}")
        self.assertEqual(
            unmatched, [],
            "these declared publication sites match nothing in the source "
            "any more: " + "; ".join(unmatched))

    def test_every_verdict_figure_in_the_source_is_declared(self):
        """The catch-all.

        A line carrying a verdict word and a figure that matches no declared
        site is an UNDECLARED publication: nobody has said what the figure
        means, so nobody can tell whether it is current. A figure added to
        an already-declared line is caught the same way - the declaration
        covers one figure, and the extra one is still unaccounted for.
        """
        undeclared = []
        for rel, n, line in self.lines:
            figs = [m.group(0).rstrip("%") for m in PERCENT.finditer(line)]
            if not figs:
                continue
            if not VERDICT_WORD.search(line):
                continue
            kinds = self._declared(rel, line)
            if any(kind == "exempt" for kind, _ in kinds):
                continue          # a declared non-measurement exempts its line
            covered = {fig for kind, fig in kinds if fig is not None}
            missing = [f for f in figs if f not in covered]
            if missing:
                undeclared.append((rel, n, missing, line.strip()[:80]))
        self.assertEqual(
            undeclared, [],
            "these figures in the source are not declared, so nothing says "
            "whether they are current: "
            + "; ".join(f"{r}:{n} {miss} ({c!r})"
                        for r, n, miss, c in undeclared))

    def test_a_current_verdict_publishes_the_current_figure(self):
        """The check that catches a lever that was re-measured and the code
        was not."""
        wrong = []
        for rel, n, line in self.lines:
            for kind, fig in self._declared(rel, line):
                if not isinstance(kind, tuple) or fig is None:
                    continue
                lever, variant = kind
                eff = self.current.get((lever, variant))
                if eff is None:
                    continue
                try:
                    published = float(fig)
                except ValueError:
                    wrong.append((rel, n, fig, "not a number"))
                    continue
                if abs(published - eff * 100) > 0.005:
                    wrong.append((rel, n, fig,
                                  f"the current figure for {lever} is "
                                  f"{eff * 100:+.2f}%"))
        self.assertEqual(
            wrong, [],
            "the source tree publishes a figure that is not the lever's "
            "current one: "
            + "; ".join(f"{r}:{n} {g}% ({why})" for r, n, g, why in wrong))

    def test_a_history_quote_quotes_a_withdrawn_figure(self):
        """A sentence that says 'this is what it was' must quote a figure
        that really was measured."""
        wrong = []
        for rel, n, line in self.lines:
            for kind, fig in self._declared(rel, line):
                if kind != "history" or fig is None:
                    continue
                if fig not in self.withdrawn_figures:
                    wrong.append((rel, n, fig))
        self.assertEqual(
            wrong, [],
            "these sentences quote history, but the figure they quote is not "
            "one a withdrawn record holds: "
            + "; ".join(f"{r}:{n} {g}%" for r, n, g in wrong))

    def test_the_withdrawn_set_is_not_empty(self):
        """The history check needs something to check against."""
        self.assertTrue(self.withdrawn_figures,
                        "no withdrawn figures found - the records are missing "
                        "their retractions")


if __name__ == "__main__":
    unittest.main(verbosity=2)
