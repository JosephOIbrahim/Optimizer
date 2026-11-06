"""
Tests that every figure the cards publish traces to a measurement record.

The failure this exists to prevent
----------------------------------
A measurement record is the evidence. The cards are what a reader actually
holds, and every number in them is a CLAIM about a measurement. When a lever
is re-measured, the records move and the cards must move with them — but
until this test existed, nothing made them move together. The cards were
swept by hand, once, by whoever happened to be looking, and every future
re-measurement re-opened the same hole: a card could publish a figure no
record supports and no test would fail.

The sweep by hand found real defects. It published a thread lever at +0.7%
when the authoritative figure was -0.60% — the wrong sign, and the wrong
number. It published an AOV lever at +7.49% from a record that had been
RETRACTED. It published -12.0% as a drift the code had never printed. It
claimed a utilisation figure that was in no record at all.

So the sweep is now a property of the repo rather than of whoever happened
to be looking. Every percentage in both cards must be one of:

  * a figure the committed measurement records actually hold - an effect, a
    drift, a spread, a p-value or a mean, at the precision the card prints;
  * a threshold the code defines (the 5% effect floor, the 85% utilisation
    limit, and so on);
  * a figure the harness's own code computes and the tests pin - the
    position-bias residual, the busy-window reproductions;
  * or a HISTORICAL number, quoted as history, that the records also hold -
    the retracted +12.34%, +16.91% and +11.25% verdicts.

Anything else is a figure nobody can check, and it fails here.

Reading the records, not the live store
---------------------------------------
The source of truth for this test is the committed `measurements/` directory.
It is in the repo, it is what the cards cite, and it is what a reader can
check without a live box. The live store is deliberately off OneDrive and
is not available everywhere, so a test that needed it would be a test that
only runs on one machine.

Nothing here renders, touches the GPU, or writes anywhere.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: The committed measurement records - the evidence the cards cite.
MEASUREMENTS = REPO / "measurements"

#: The cards. Both are operator-facing documents that quote figures.
CARDS = ["Optimize-The-4090-Card.md", "Harness-Operator-Card.md"]

#: The document the repo's own instructions live in. It quotes the same
#: figures, so the same rule applies to it.
CLAUDE = "CLAUDE.md"

#: The README. It is the front door for anyone arriving from GitHub, so
#: the same rule applies to it — and it now leads with the harness's own
#: figures rather than the V6 design targets that used to fill it.
README = "README.md"

#: Every document the figure rules apply to.
DOCUMENTS = CARDS + [CLAUDE, README]


def _records():
    """Every committed measurement record, parsed."""
    out = []
    for p in sorted(MEASUREMENTS.glob("*.json")):
        try:
            out.append((p.name, json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return out


def _percent_forms(value: float) -> set:
    """Every way a card might legitimately print one number.

    A card writes +10.85% or -0.09% or 5%; a record holds 0.10846649376725509.
    The card rounds, and this test has to allow the roundings a card
    actually uses without becoming so loose that any number passes.

    The forms carry the `%` because that is how the cards print them, and a
    token that does not match the way a card prints it is not evidence of
    anything.
    """
    pct = value * 100
    forms = set()
    for places in (0, 1, 2, 3):
        forms.add(f"{pct:+.{places}f}%")
        forms.add(f"{pct:.{places}f}%")
    return forms


def _record_facts():
    """Every figure the committed records hold, in the forms a card prints.

    An effect, a drift, a spread, a p-value and the two means of each
    measurement - and the ORIGINAL verdicts the retractions preserve, which
    are history a card is entitled to quote.
    """
    facts = set()

    def add(value):
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            facts.update(_percent_forms(float(value)))

    def add_p(value):
        """A p-value, printed the ways a card prints one.

        A card writes p=1.13e-04 or p=2.5e-07 - one or two decimals in the
        exponent, because that is how a person reads a small number. The
        record holds 0.00011321489170426711.
        """
        if value is None:
            return
        v = float(value)
        for places in (1, 2, 3):
            facts.add(f"p={v:.{places}e}")
        facts.add(f"p={v:.3g}")
        facts.add(f"p={v:.4g}")

    def add_text(text):
        """Every percentage a piece of record prose holds.

        A retraction preserves the ORIGINAL claim it withdrew, and the reason
        it was withdrawn. Both are in the record, and a card quotes them as
        history - so their figures are figures the records hold.
        """
        for m in re.finditer(r"[+-]?\d+\.?\d*%", str(text)):
            facts.add(m.group(0))

    def add_result(block):
        """One measurement block, in either record shape."""
        if not isinstance(block, dict):
            return
        for key in ("effect", "drift", "control_spread"):
            add(block.get(key))
        add_p(block.get("p_value"))
        for key in ("baseline_mean", "candidate_mean"):
            val = block.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                facts.add(f"{val:.2f}")
                facts.add(f"{val:.3f}")
        # The claim a retraction preserves, and the reason it was withdrawn.
        ret = block.get("retracted")
        if isinstance(ret, dict):
            add_text(ret.get("original_verdict", ""))
            add_text(ret.get("replacement_verdict", ""))
            add_text(ret.get("reason", ""))
            corr = ret.get("reason_corrected")
            if isinstance(corr, dict):
                add_text(corr.get("reason", ""))
            hist = ret.get("replacement_history")
            if isinstance(hist, list):
                for h in hist:
                    if isinstance(h, dict):
                        add_text(h.get("verdict", ""))
                        add_text(h.get("superseded_because", ""))
        # A superseded record's own reason, and any correction to it.
        sup = block.get("superseded")
        if isinstance(sup, dict):
            add_text(sup.get("reason", ""))
            corr = sup.get("reason_corrected")
            if isinstance(corr, dict):
                add_text(corr.get("reason", ""))

    for name, doc in _records():
        results = doc.get("results")
        if isinstance(results, dict):
            # optimize_*.json: keyed by lever name.
            for block in results.values():
                add_result(block)
        elif isinstance(results, list):
            # levers_*.json: a list of results, each with its own label.
            for block in results:
                add_result(block)
                if isinstance(block, dict):
                    # The verdict string itself, withdrawn or not.
                    add_text(block.get("verdict", ""))
        # A record's own superseded block carries the reason and its figures.
        sup = doc.get("superseded")
        if isinstance(sup, dict):
            add_text(sup.get("reason", ""))
            corr = sup.get("reason_corrected")
            if isinstance(corr, dict):
                add_text(corr.get("reason", ""))
    return facts


#: Figures that are NOT measurement results but are true, and are stated in
#: the code or pinned by a test. Each carries the reason it is allowed.
#:
#: A threshold is a threshold: the 5% effect floor every objective uses, the
#: utilisation limits the contention detector compares against, and so on.
#: The list is short on purpose - every entry is a claim somebody has to
#: stand behind, and a figure that does not need to be here is a figure
#: that should be traceable to a record instead.
ALLOWED_NON_RECORD = {
    # --- thresholds and limits the code defines --------------------------
    "5%": "the objective's minimum effect threshold",
    "85%": "the current utilisation threshold",
    "90%": "the GPU is busy line, quoted from the telemetry format",
    "95%": "a utilisation reading, quoted from the telemetry format",
    # --- telemetry readings quoted as examples ---------------------------
    "83%": "the throttle line's power-bound example reading",
    "94%": "our own render's utilisation, quoted as a measurement example",
    "-72%": "the desktop's own burst ceiling, quoted as a measurement example",
    "-25%": "a power figure quoted as an example",
    "-40%": "a power figure quoted as an example",
    "-60%": "a power figure quoted as an example",
    "20%": "a mutation rate quoted as an example",
    # --- computed by the harness and pinned by a test -------------------
    "+6.75%": "the 52s busy window's effect - pinned in tests/test_drift_gate.py",
    "+10.98%": "the 56s busy window's effect - pinned in tests/test_drift_gate.py",
    "+14.86%": "the 60s busy window's effect - pinned in tests/test_drift_gate.py",
    "+23.20%": "the 70s busy window's effect - pinned in tests/test_drift_gate.py",
    "0.065%": "the position-bias residual - pinned in tests/test_position_residual.py",
    "0.07%": "the position-bias residual, rounded - pinned in tests/test_position_residual.py",
    "+10.3%": "the midpoint of the two live beauty_only runs - checked by "
             "TestTheDerivedFiguresAreDerived below",
    # --- baseline coefficients, from the V12 baseline store ---------------
    "0.17%": "the gemm baseline's cv - a V12 baseline record, not a measurement",
    "0.14%": "the shot baseline's cv - a V12 baseline record, not a measurement",
    # --- the README's own figures ---------------------------------------
    "0.5%": "the README's argument that a small win is indistinguishable "
            "from drift - an illustrative size, not a measurement",
    "-15%": "a V6 design target in the README's archived section, never "
            "measured by this harness",
    "-30%": "a V6 design target in the README's archived section, never "
            "measured by this harness",
    "40%": "a V6 design target in the README's archived section, never "
           "measured by this harness",
    "30%": "a V6 design target in the README's archived section, never "
           "measured by this harness",
    "-8%": "a V6 design target in the README's archived section, never "
           "measured by this harness",
    "-100%": "a V6 design target in the README's archived section, never "
             "measured by this harness",
}


#: The limit the drift and control-spread checks use.
CONTROL_LIMIT = 0.05


def _is_clean(block: dict) -> bool:
    """Is this run one whose numbers can be trusted as the lever's effect?

    Cleanliness is NOT the same as currency. A record marked `superseded`
    says a later measurement replaced this one - which is true of a run
    that was simply older, and also of one that was invalid - so
    supersession alone cannot mean "not evidence", or the cards' own
    history rows could not be quoted.

    Cleanliness IS in the data:

      * a result carrying a retraction is not evidence;
      * a run with no `contention_events` was taken before the
        per-repetition contention check existed, so nothing certified it;
      * a run with zero contention events is clean;
      * a run with contention is clean only if the record says the
        comparison stands AND the control's drift and spread are inside
        the limit - the rivals were on the card, and the control held.
    """
    if block.get("retracted"):
        return False
    cont = block.get("contention_events")
    if cont is None:
        return False          # the check did not exist yet - uncertified
    if cont > 0 and not block.get("shared_card_note"):
        return False          # rivals were present and the record does not
                              # say the comparison stands
    # The control is the last word, whether or not anything else was on the
    # card: a control that drifted past the limit makes the comparison
    # meaningless, and no amount of quiet certifies it.
    for key in ("drift", "control_spread"):
        val = block.get(key)
        if val is not None and abs(val) > CONTROL_LIMIT:
            return False      # the control moved
    return True


#: Where the documents live. Overridable so the mutation test can plant a
#: defect into a COPY of the card rather than into the real one.
#:
#: This is not a convenience. The mutation test used to write the defect
#: into the shared card on disk, and the card was left holding the very
#: defect the test exists to catch when the run was killed mid-plant. A
#: restore in a `finally` is not enough: a finally does not run when the
#: process is killed.
DOCS_DIR = Path(os.environ.get("PUBLISHED_FIGURES_DOCS", str(REPO)))


def _card_text(name: str) -> str:
    return (DOCS_DIR / name).read_text(encoding="utf-8")


def _withdrawn_records():
    """The records the repo has withdrawn: superseded, or holding a retraction.

    A withdrawn record is real evidence of what was once measured, and a card
    is entitled to quote it AS HISTORY. What it may not do is publish it as
    the current answer, because the current answer is a different record.
    """
    out = {}
    for name, doc in _records():
        results = doc.get("results")
        blocks = []
        if isinstance(results, dict):
            blocks = list(results.values())
        elif isinstance(results, list):
            blocks = list(results)
        if doc.get("superseded") or any(
                isinstance(b, dict) and b.get("retracted") for b in blocks):
            out[name] = doc
    return out


def _live_figures():
    """The figures the LIVE records hold — the current answers."""
    figures = set()
    withdrawn = _withdrawn_records()
    for name, doc in _records():
        if name in withdrawn:
            continue
        results = doc.get("results")
        blocks = []
        if isinstance(results, dict):
            blocks = list(results.values())
        elif isinstance(results, list):
            blocks = list(results)
        for block in blocks:
            if not isinstance(block, dict):
                continue
            for key in ("effect", "drift", "control_spread"):
                val = block.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    pct = val * 100
                    for places in (1, 2):
                        figures.add(f"{pct:+.{places}f}%")
                        figures.add(f"{pct:.{places}f}%")
    return figures


def _history_rows(text: str):
    """The lines that are quoting history rather than publishing a verdict.

    A card keeps the record of what each measurement said, so a withdrawn
    figure belongs there — on a row that is plainly history.
    """
    history = []
    for line in text.splitlines():
        stripped = line.strip()
        # A row in a "measured twice"-style block: a lever name, a time, and
        # a verdict. Those rows ARE the history.
        if re.match(r"^[a-z_]+\s+\d\d:\d\d\s", stripped):
            history.append(line)
    return history


class TestEveryPublishedFigureTracesToARecord(unittest.TestCase):
    """The sweep, as a property of the repo rather than of whoever is looking."""

    @classmethod
    def setUpClass(cls):
        cls.facts = _record_facts()
        if not cls.facts:
            raise unittest.SkipTest("no committed measurement records to check against")

    def _unsourced(self, text: str, label: str):
        """Every percentage in the text that no record holds."""
        missing = []
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in re.finditer(r"[+-]?\d+\.?\d*%", line):
                tok = m.group(0)
                if tok in self.facts:
                    continue
                if tok in ALLOWED_NON_RECORD:
                    continue
                if tok.lstrip("+-") in ALLOWED_NON_RECORD:
                    continue
                missing.append((lineno, tok, line.strip()[:90]))
        self.assertEqual(
            missing, [],
            f"{label} publishes figures that no committed record holds and that "
            f"are not on the allowlist of thresholds and pinned computations. "
            f"Every number in a card must be checkable: "
            + "; ".join(f"line {n}: {t} ({c!r})" for n, t, c in missing))

    def test_the_optimize_card_publishes_only_traceable_figures(self):
        self._unsourced(_card_text(CARDS[0]), CARDS[0])

    def test_the_harness_card_publishes_only_traceable_figures(self):
        self._unsourced(_card_text(CARDS[1]), CARDS[1])

    def test_claude_md_publishes_only_traceable_figures(self):
        """The repo's own instructions quote the same figures."""
        self._unsourced(_card_text(CLAUDE), CLAUDE)

    def test_the_readme_publishes_only_traceable_figures(self):
        """The README is the front door for anyone arriving from GitHub."""
        self._unsourced(_card_text(README), README)


class TestTheCardsDoNotPublishWithdrawnFigures(unittest.TestCase):
    """A withdrawn figure may be quoted as HISTORY. It may not be published
    as the CURRENT ANSWER.

    This is the check the trace-only test above cannot make, and it is the
    one that matters most. A retracted or superseded figure is STILL IN THE
    RECORDS - as history - so "does a record hold this number?" answers
    yes and the card sails through. But the hand-sweep found exactly this: a
    card publishing an AOV lever at +7.49% from a record that had been
    RETRACTED.

    So the question is not "is this figure in a record" but "is this figure
    in a LIVE record". A figure that only a withdrawn record holds is
    stale, and a stale figure has no business in a row that states what a
    lever is worth.

    NARRATION IS NOT A VERDICT. The cards legitimately explain the
    phantom - "it came back ADMIT +12.34%, p=1.9e-05" - because that is
    the story the mechanism exists to tell. What must not happen is a stale
    figure in a row that publishes what a lever is WORTH: the results
    table, the retraction table's "measured properly" column, or a
    measured-twice block's current row.
    """

    @classmethod
    def setUpClass(cls):
        cls.withdrawn = _withdrawn_records()
        cls.live = _live_figures()
        # The figures that ONLY withdrawn records hold.
        cls.stale = set()
        for name, doc in cls.withdrawn.items():
            for m in re.finditer(r"[+-]\d+\.\d+%", json.dumps(doc)):
                tok = m.group(0)
                if tok not in cls.live:
                    cls.stale.add(tok)

    #: The rows that publish what a lever is WORTH RIGHT NOW.
    #:
    #: A results table's "Result" column, and a retraction table's
    #: "Measured properly" column. Both are the CURRENT answer for a lever.
    #: The retraction table's OTHER column - "Recorded" - is the phantom
    #: being quoted as history, and the error table is a lookup of messages
    #: the code prints, neither of which states a lever's worth.
    _CURRENT_ROW = re.compile(r"^\|")

    def _published_as_current(self, text: str):
        """The stale figures in rows that publish a lever's current worth."""
        bad = []
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not self._CURRENT_ROW.match(stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not cells:
                continue
            # A results table: "| render | AOV variant (beauty only) | ADMIT
            # +10.85% | ..." - the lever's worth is in the THIRD cell.
            # A retraction table: "| lever | Recorded | Measured properly |
            # Verdict |" - the current worth is in the THIRD cell too.
            # Both are "the third cell is the answer" tables, and in both
            # the first cell names the lever.
            if len(cells) < 3:
                continue
            head = cells[0].lower()
            if not any(k in head for k in ("lever", "request", "variant",
                                            "render", "compute", "shot")):
                continue
            # The error table is "| you see | do |" - its first cell is a
            # quoted message, not a lever.
            if "you see" in head or "|" == head:
                continue
            current = cells[2]
            for tok in self.stale:
                if tok in current:
                    bad.append((lineno, tok, stripped[:88]))
        return bad

    def test_the_optimize_card_publishes_no_withdrawn_figure_as_current(self):
        bad = self._published_as_current(_card_text(CARDS[0]))
        self.assertEqual(bad, [],
                         f"{CARDS[0]} publishes a figure that only a WITHDRAWN "
                         f"record holds, in a row that states what a lever is "
                         f"worth. A retracted or superseded measurement may be "
                         f"quoted as history, never as the current answer: "
                         + "; ".join(f"line {n}: {t} ({c!r})" for n, t, c in bad))

    def test_the_harness_card_publishes_no_withdrawn_figure_as_current(self):
        bad = self._published_as_current(_card_text(CARDS[1]))
        self.assertEqual(bad, [],
                         f"{CARDS[1]} publishes a figure that only a WITHDRAWN "
                         f"record holds, in a row that states what a lever is "
                         f"worth: "
                         + "; ".join(f"line {n}: {t} ({c!r})" for n, t, c in bad))

    def test_claude_md_publishes_no_withdrawn_figure_as_current(self):
        bad = self._published_as_current(_card_text(CLAUDE))
        self.assertEqual(bad, [],
                         f"{CLAUDE} publishes a figure that only a WITHDRAWN "
                         f"record holds, in a row that states what a lever is "
                         f"worth: "
                         + "; ".join(f"line {n}: {t} ({c!r})" for n, t, c in bad))

    def test_there_are_stale_figures_to_catch(self):
        """The check above is only worth anything if it CAN fire.

        If no figure is stale, the test passes vacuously and a card could
        publish a retracted number forever. This pins that the withdrawn
        records really do hold figures the live ones do not.
        """
        self.assertTrue(self.stale,
                        "no figure is held only by a withdrawn record, so the "
                        "withdrawn-figure check cannot fire")


class TestTheCardsPublishTheCurrentFigureForEachLever(unittest.TestCase):
    """A lever's CURRENT answer is its most recent live measurement.

    This is the third thing a card can get wrong, and the subtlest: the
    figure is real, it is in a LIVE record, and it is simply not the
    answer any more. A card that publishes +9.79% for beauty_only when the
    live answer is +10.85% is quoting a superseded number as if it were
    current — and nothing else would catch it, because the old figure is
    still perfectly traceable.

    The rule is simple and the repo already states it: a later measurement
    supersedes an earlier one. So for each lever, the figure a card
    publishes must be the one the NEWEST live record holds.
    """

    @classmethod
    def setUpClass(cls):
        cls.withdrawn = _withdrawn_records()
        # The newest live record per lever, by its own timestamp.
        cls.current = {}
        for name, doc in _records():
            if name in cls.withdrawn:
                continue
            stamp = doc.get("recorded_utc") or ""
            results = doc.get("results")
            blocks = {}
            if isinstance(results, dict):
                blocks = {k: v for k, v in results.items()
                          if isinstance(v, dict) and not v.get("retracted")}
            for lever, block in blocks.items():
                key = (lever, block.get("variant"))
                if key not in cls.current or stamp > cls.current[key][0]:
                    cls.current[key] = (stamp, block, name)

    def _current_effect(self, lever, variant=None):
        key = (lever, variant)
        if key not in self.current:
            return None
        return self.current[key][1].get("effect")

    def _figures_for(self, lever, variant=None):
        """The effect figures a card may publish for this lever."""
        eff = self._current_effect(lever, variant)
        if eff is None:
            return set()
        pct = eff * 100
        forms = set()
        for places in (1, 2):
            forms.add(f"{pct:+.{places}f}%")
            forms.add(f"{pct:.{places}f}%")
        return forms

    def test_the_optimize_card_publishes_the_current_figure_for_aov_variant(self):
        """Both AOV variants are named in that card's results table."""
        allowed = self._figures_for("aov_variant", "beauty_only")
        self.assertTrue(allowed, "no live beauty_only measurement to check against")
        self._check_current(CARDS[0], "aov_variant", "beauty only", allowed)

    def test_the_optimize_card_publishes_the_current_figure_for_no_crypto(self):
        allowed = self._figures_for("aov_variant", "no_crypto")
        self.assertTrue(allowed, "no live no_crypto measurement to check against")
        self._check_current(CARDS[0], "aov_variant", "no crypto", allowed)

    def test_the_harness_card_publishes_the_current_figure_for_aov_variant(self):
        allowed = self._figures_for("aov_variant", "beauty_only")
        self.assertTrue(allowed)
        self._check_current(CARDS[1], "aov_variant", "beauty only", allowed)

    def _check_current(self, doc, lever, label, allowed):
        """Every row that publishes this lever's worth must hold a current figure.

        The two cards publish that answer in different shapes: the Optimize
        card uses a markdown table, the Harness card a fixed-width block.
        Both are handled.

        A row can name the lever in any cell — the Optimize card's results
        table puts the REQUEST first and the lever second, and its retraction
        table puts the lever first and the recorded phantom second. So the
        row is matched on the lever appearing anywhere in it, and every
        figure in it is checked against the live measurement, except the
        "Recorded" column of the retraction table, which is the phantom
        quoted as history.
        """
        text = _card_text(doc)
        checked = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("|"):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cells) < 3:
                    continue
                if label.lower() not in " ".join(cells[:2]).lower():
                    continue
                # The retraction table's SECOND column is "Recorded" - the
                # phantom being quoted as history, not the current answer.
                if cells[1].lstrip("*").lstrip("`") in ("+12.34%", "+16.91%",
                                                         "+11.25%"):
                    continue
                published_text = cells[2]
            elif stripped.startswith(lever):
                if label.lower() not in stripped.lower():
                    continue
                published_text = stripped
            else:
                continue
            checked += 1
            published = set(re.findall(r"[+-]\d+\.\d+%", published_text))
            if not published:
                continue
            self.assertTrue(
                published & allowed,
                f"{doc} line {lineno} publishes {sorted(published)} for "
                f"{label!r}, but the live measurement says "
                f"{sorted(allowed)} - a superseded figure is being published as "
                f"the current answer")
        self.assertTrue(checked, f"{doc} has no row publishing {label!r}")


class TestTheDerivedFiguresAreDerived(unittest.TestCase):
    """The card states a MIDPOINT and a half-width across runs.

    "the effect is +10.3% ± 0.5" is two claims: that the two clean runs
    of the beauty-only lever agree to about a point, and what their
    midpoint is. Both are computed FROM the records, so both can be checked
    here rather than taken on trust.

    Which runs count as clean is not a judgement call, and it is NOT the same
    question as "which run is the current answer".

    A record marked `superseded` says "a later measurement replaced this
    one" — which is true of a run that was simply older, and also of one
    that was invalid. The repo marks BOTH: `optimize_20261104_083355.json`
    is superseded because a later, cleaner run of the same lever replaced
    it, and its own numbers are called "real measurements of a real machine"
    that "stay readable". So supersession alone cannot mean "not evidence",
    or the card's own history rows could not be quoted.

    Cleanliness is a different property, and it IS in the data:

      * a result carrying a retraction is not evidence;
      * a run with no `contention_events` was taken before the per-repetition
        contention check existed, so nothing certified it — not evidence;
      * a run with zero contention events is clean;
      * a run with contention is clean only if the record says the
        comparison stands (`shared_card_note`) AND the control's drift and
        spread are inside the limit — the rivals were on the card, and the
        control did not move.

    That is what "two clean paired runs" means, and it selects exactly the
    runs the card names.
    """

    @classmethod
    def setUpClass(cls):
        cls.runs = []
        for name, doc in _records():
            results = doc.get("results")
            if not isinstance(results, dict):
                continue
            for block in results.values():
                if not isinstance(block, dict):
                    continue
                if block.get("retracted"):
                    continue      # withdrawn: not evidence
                if not _is_clean(block):
                    continue
                variant = block.get("variant")
                if not variant:
                    continue
                if block.get("effect") is None:
                    continue
                cls.runs.append((variant, block["effect"], name))

    def _clean_effects(self, variant):
        return sorted(e for v, e, _ in self.runs if v == variant)

    def test_the_beauty_only_midpoint_and_half_width_are_the_records_figures(self):
        effects = self._clean_effects("beauty_only")
        self.assertEqual(len(effects), 2,
                         f"expected exactly two live beauty_only runs, got {effects}")
        mid = (effects[0] + effects[1]) / 2
        half = (effects[1] - effects[0]) / 2
        expected = f"{mid * 100:+.1f}% ± {half * 100:.1f}"
        for name in CARDS:
            with self.subTest(document=name):
                text = _card_text(name)
                self.assertIn(expected, text,
                              f"{name} should state the beauty_only effect as "
                              f"{expected!r}, computed from the two live runs")

    def test_the_two_clean_runs_really_do_agree(self):
        """The claim the midpoint rests on: the runs agree to about a point."""
        effects = self._clean_effects("beauty_only")
        spread = (effects[1] - effects[0]) * 100
        self.assertLess(spread, 1.5,
                        f"the two clean beauty_only runs differ by {spread:.2f} "
                        f"points - the card says they agree to about a point")


@unittest.skipIf(
    os.environ.get("PUBLISHED_FIGURES_MUTATING"),
    "this IS the mutation run - the checks are proved here, not re-proved "
    "inside it")
class TestTheChecksCanFire(unittest.TestCase):
    """A test that cannot fail is not a test.

    Each check above is proved against a PLANTED defect - the three ways a
    card goes wrong, planted one at a time in a copy of the card, and the
    check is required to catch it. Without this, a check that silently
    stopped matching (a row format changed, a column moved) would pass
    forever and the whole file would be decoration.
    """

    #: The three defects, as (what it is, the text to plant, its replacement).
    PLANTS = [
        ("a figure from a RETRACTED record",
         "**ADMIT +10.85%**, p=1.1e-04",
         "**ADMIT +7.49%**, p=1.8e-05"),
        ("a figure nobody can check",
         "450 W budget, 0% at the cap",
         "450 W budget, 12% at the cap"),
        ("a superseded figure published as current",
         "**ADMIT +10.85%**, p=1.1e-04",
         "**ADMIT +9.79%**, p=3.7e-05"),
    ]

    def _run(self, docs_dir):
        """Run the checks in this file against the documents in `docs_dir`.

        Two things matter here, and both were learned the hard way.

        The subprocess must NOT run this file's mutation test again, or the
        run recurses: each level plants a defect and spawns the next level,
        and the card on disk is shared between them, so a nested level can
        see a card the level above has already changed. The environment
        marker below is what stops it.

        And the documents it reads must be the COPY in `docs_dir`, not the
        real ones. The test used to write the defect into the shared card,
        and when the run was killed mid-plant the card was left holding the
        very defect the test exists to catch. A copy cannot be corrupted
        that way, and it needs no restore.
        """
        import subprocess
        env = dict(os.environ, PUBLISHED_FIGURES_MUTATING="1",
                   PUBLISHED_FIGURES_DOCS=str(docs_dir))
        return subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_published_figures.py",
             "-q", "--no-header", "-x"],
            cwd=str(REPO), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=180, env=env)

    def test_each_planted_defect_is_caught(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory(prefix="published_figures_") as tmp:
            docs = Path(tmp)
            for name in DOCUMENTS:
                shutil.copy2(REPO / name, docs / name)
            card = docs / CARDS[0]
            original = card.read_text(encoding="utf-8")

            # The UNPLANTED copy has to PASS first. Two things fall out of
            # that: the copy is complete, so a failure below cannot be a
            # missing document; and the failure below is caused by the
            # PLANT, not by something else in the copy. Without this, a
            # subprocess that failed for any reason at all would satisfy
            # the assertion, and the test would prove nothing.
            clean = self._run(docs)
            self.assertEqual(
                clean.returncode, 0,
                f"the checks failed on the UNPLANTED copy, so a failure on "
                f"a planted one would prove nothing:\n{clean.stdout[-2000:]}")

            for what, old, new in self.PLANTS:
                with self.subTest(defect=what):
                    self.assertIn(
                        old, original,
                        f"the plant for {what!r} no longer matches the card, "
                        f"so this defect is no longer being proved against. "
                        f"Update the plant to the card's current wording - a "
                        f"plant that silently does not apply is a proof that "
                        f"quietly proves nothing")
                    card.write_text(original.replace(old, new), encoding="utf-8")
                    r = self._run(docs)
                    self.assertNotEqual(
                        r.returncode, 0,
                        f"planting {what} did not fail the checks - a check "
                        f"that cannot catch this defect is not a check")
                    card.write_text(original, encoding="utf-8")


class TestTheAllowlistIsHonest(unittest.TestCase):
    """The allowlist is the one place a figure can escape the records.

    So it is checked too: an entry that is really a measurement result in
    disguise is a hole, and a stale entry is a figure the cards are allowed
    to publish that nothing supports.
    """

    def test_every_allowlist_entry_is_still_used_by_a_document(self):
        """An entry nothing quotes is dead weight - and dead allowlist
        entries are how a real figure sneaks in later."""
        text = "\n".join(_card_text(n) for n in DOCUMENTS)
        for token in ALLOWED_NON_RECORD:
            with self.subTest(token=token):
                self.assertIn(token, text,
                              f"{token!r} is allowlisted but no document quotes "
                              f"it any more - remove it")

    def test_the_allowlist_is_not_a_place_to_hide_measurements(self):
        """The allowlist is for figures NO record holds.

        A record's own figure on the allowlist would be a hole: the record
        could change and the card would not have to follow, which is the
        whole failure this test exists to close.
        """
        facts = _record_facts()
        for token in ALLOWED_NON_RECORD:
            with self.subTest(token=token):
                self.assertNotIn(token, facts,
                                 f"{token!r} is on the allowlist but a record "
                                 f"holds it - it belongs to the records, not "
                                 f"to the exceptions")


class TestTheCardsCiteTheRecordsTheyQuote(unittest.TestCase):
    """A card that names a record must name one that exists."""

    def test_every_record_named_by_a_card_is_committed(self):
        named = set()
        for name in DOCUMENTS:
            text = _card_text(name)
            named.update(re.findall(r"optimize_\d+_\d+\.json", text))
            named.update(re.findall(r"levers_\d+_\d+\.json", text))
            named.update(re.findall(r"render_\d+_\d+\.json", text))
        for rec in sorted(named):
            with self.subTest(record=rec):
                self.assertTrue((MEASUREMENTS / rec).exists(),
                                f"a card cites {rec}, which is not committed "
                                f"under measurements/ - a reader cannot check "
                                f"a figure whose evidence is not in the repo")


if __name__ == "__main__":
    unittest.main(verbosity=2)
