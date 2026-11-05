"""
Tests for the shot objective's sample floor and its justification.

The failure this prevents
--------------------------
`shot_objective` carries `min_samples=3` instead of the default 5, and the
comment that justified it cited "173.8 s vs 89.9 s, a 48% gap" — the figures
of a run the gate had REFUSED.

That was not an accident of arithmetic. On 2026-11-04 the shot threshold was
5; a run measured at 08:16 was denied for exactly that reason ("n<5 per side
(baseline n=3, candidate n=3)"); the objective file was edited at 08:18, in
the gap between that refusal and the 08:33 run that was then admitted at
n=3/3; and the justification written into the file for the lower floor was
that same refused run's arithmetic — 173.8 x (1 - 0.4825) = 89.9. So the
threshold was lowered to admit a result, and the reasoning offered for the
lowering was the rejected result's own numbers.

The repo's governing principle is that a record must never change to suit
the present: the original claim stays visible and marked, and superseded
things stay visible in a chain. A THRESHOLD is a record of that kind too.
It can be changed, but the justification for the change has to be one that
stands on its own — not the arithmetic of the thing it is being used to
admit.

So the justification now cites a run that was actually admitted and actually
had a drift check (optimize_20261105_065602.json), and this test pins the
docstring's cited figures to that record. If the docstring drifts away from
the record it claims to describe — or starts citing a refused run again —
the test fails.

The test reads the real measurement store, read-only, and skips when the
store is not on this box (the same pattern as tests/test_ssim_guardrail.py).
Nothing here renders, touches the GPU, or writes anywhere.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent))

from v12.objective import shot_objective  # noqa: E402

#: The measurement store. Read-only, and absent on a box that never ran a
#: campaign — hence the skip below.
STORE = Path(r"E:\OptimizerV12\measurements")

#: The run the docstring is allowed to cite: it was ADMITTED, and it carried
#: a drift check, so the verdict rests on a control that held still.
ADMITTED = STORE / "optimize_20261105_065602.json"

#: The run the docstring must NOT cite. It was REFUSED — for n<5 per side.
REFUSED = STORE / "optimize_20261104_081645.json"

#: The label the frame-count lever is recorded under.
LABEL = "frames_per_process"


def _docstring() -> str:
    return shot_objective.__doc__ or ""


def _result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["results"][LABEL]


@unittest.skipUnless(STORE.is_dir(), "no measurement store on this box")
class TestTheShotFloorIsJustifiedByAnAdmittedRun(unittest.TestCase):
    """The docstring's numbers must be the numbers of the admitted record.

    This is the pin. The justification for a threshold is the one thing about
    a threshold that can quietly rot: the code keeps working, the verdict
    keeps getting written, and the sentence explaining why the number is
    what it is drifts into describing some other measurement entirely.
    """

    def setUp(self):
        if not ADMITTED.is_file():
            self.skipTest(f"no admitted shot run on disk: {ADMITTED}")
        self.doc = _docstring()
        self.admitted = _result(ADMITTED)

    # -- the docstring cites the ADMITTED run, and its real figures ---------

    def test_the_cited_effect_is_the_admitted_runs_effect(self):
        """+46.95%, from the run that was admitted."""
        self.assertIn("+46.95%", self.doc)
        self.assertAlmostEqual(self.admitted["effect"], 0.46952254108257346,
                               places=12)
        self.assertAlmostEqual(round(self.admitted["effect"] * 100, 2), 46.95,
                               places=9)

    def test_the_cited_means_are_the_admitted_runs_means(self):
        """176.23 s against 93.49 s, from the run that was admitted."""
        self.assertIn("176.23", self.doc)
        self.assertIn("93.49", self.doc)
        self.assertAlmostEqual(round(self.admitted["baseline_mean"], 2), 176.23,
                               places=9)
        self.assertAlmostEqual(round(self.admitted["candidate_mean"], 2), 93.49,
                               places=9)

    def test_the_cited_drift_is_the_admitted_runs_drift(self):
        """+2.12% — the control held still, which is the point of citing it."""
        self.assertIn("+2.12%", self.doc)
        self.assertAlmostEqual(round(self.admitted["drift"] * 100, 2), 2.12,
                               places=9)

    def test_the_cited_sample_count_is_the_admitted_runs(self):
        """n=3/3 — the floor and the run that was admitted agree."""
        self.assertIn("n=3/3", self.doc)
        self.assertEqual(self.admitted["n_baseline"], 3.0)
        self.assertEqual(self.admitted["n_candidate"], 3.0)

    def test_the_cited_differences_are_the_admitted_runs(self):
        """-82.66, -83.09 and -82.48 s — the raw paired differences."""
        for diff in self.admitted["paired_diffs"]:
            self.assertIn(f"{diff:.2f}", self.doc,
                          "the docstring must cite the paired differences "
                          "the record actually holds")

    def test_the_cited_record_is_named(self):
        """A reader has to be able to go and check it."""
        self.assertIn(ADMITTED.name, self.doc)

    # -- the docstring does NOT cite the REFUSED run ------------------------

    def test_the_refused_runs_figures_are_not_cited(self):
        """89.9 s and the 48% gap are the refused run's arithmetic.

        173.8 x (1 - 0.4825) = 89.94. Those are the numbers of the run the
        gate REFUSED, and citing them as the justification for the floor
        that admitted the next one is the circularity this test exists to
        catch.
        """
        refused = _result(REFUSED)
        self.assertFalse(refused["ok"], "this run was refused; if it now reads "
                                        "as admitted the fixture is stale")
        self.assertNotIn("89.9", self.doc,
                         "89.9 s is the refused run's arithmetic — the "
                         "justification cannot rest on it")
        self.assertNotIn("+48%", self.doc,
                         "+48% is the refused run's effect — the justification "
                         "cannot rest on it")
        self.assertNotIn("48.25", self.doc,
                         "+48.25% is the refused run's effect, to two decimals")

    def test_the_refused_runs_name_is_not_the_justification(self):
        """The refusal may be named as history; it may not be the source of
        the numbers that justify the floor."""
        if REFUSED.name not in self.doc:
            self.skipTest("the docstring does not name the refused run at all")
        # It may appear, but only as the thing the floor moved AWAY from —
        # never as the measurement that supports the floor.
        self.assertNotIn(f"{REFUSED.name} measured", self.doc)

    # -- the docstring does not cite any run but the admitted one ----------

    def test_every_record_the_docstring_cites_is_an_admitted_run(self):
        """A justification may not rest on a run the gate refused.

        The docstring is allowed to name the refused run as the thing the
        floor moved away from — that is history, and history belongs in the
        record. What it may not do is quote a REFUSED run's figures as
        support for the floor. So: every record named as support must be a
        run that was admitted, and the one refused run may appear only as
        the thing that was refused.
        """
        named = set(re.findall(r"optimize_\d+_\d+\.json", self.doc))
        self.assertTrue(named, "the docstring names no measurement record, so "
                               "nothing in it can be checked")
        for name in sorted(named):
            path = STORE / name
            self.assertTrue(path.is_file(),
                            f"the docstring cites {name}, which is not on disk")
            block = _result(path)
            if name == REFUSED.name:
                continue      # named as the refusal, which is the point
            self.assertTrue(
                block["ok"],
                f"the docstring cites {name} as support, but that run was "
                "REFUSED — a justification may not rest on a refusal")

    def test_the_setup_tax_cited_is_the_setup_tax_that_record_holds(self):
        """37.42 s comes from the record named beside it."""
        other = STORE / "optimize_20261104_083355.json"
        if not other.is_file():
            self.skipTest("no setup-tax record on disk")
        if other.name not in self.doc:
            self.skipTest("the docstring does not cite the setup-tax record")
        self.assertAlmostEqual(round(_result(other)["setup_tax_s"], 2), 37.42,
                               places=9)
        self.assertIn("37.42", self.doc)

    # -- the floor itself ---------------------------------------------------

    def test_the_floor_is_still_three(self):
        """The floor is a decision, not a computation; this test pins that
        the decision has not been changed behind the docstring's back."""
        self.assertEqual(shot_objective().min_samples, 3)

    def test_the_admitted_run_would_still_be_admitted_at_this_floor(self):
        """The record the docstring cites has to be one this floor admits.

        If the floor moved above 3, the docstring would be citing a run the
        objective can no longer admit — and the justification would be
        describing a measurement the gate would refuse.
        """
        obj = shot_objective()
        self.assertLessEqual(self.admitted["n_baseline"], obj.min_samples)
        self.assertLessEqual(self.admitted["n_candidate"], obj.min_samples)


if __name__ == "__main__":
    unittest.main(verbosity=2)
