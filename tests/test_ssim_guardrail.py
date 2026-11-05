"""
Tests for the ssim guardrail: the bar must be a MEASUREMENT.

What is being protected here
-----------------------------
The guardrail answers one question: is the candidate's rendered PICTURE the
same picture as the baseline's? It answers it with two numbers scored against
a shared reference — the baseline's first frame.

  * the baseline side is the renderer's own run-to-run variation (the bar);
  * the candidate side is how far the candidate's frames sit from that
    reference.

Three defects made that pair mean something other than what it says.

  1. The bar was measured by scoring the reference against ITSELF for its own
     repetition. That is 1.0 by construction, not by measurement. Measured on
     the four frames_per_process sessions on this box, the free 1.0 pushed
     the baseline side's mean from the renderer's real variation (0.9949) up
     to 0.9964 — and because the free 1.0 was diluted differently by each
     session's repetition count, the margin the guard had left to catch a real
     picture change moved with it: 1.03% in one session, 0.87% in another,
     for the same renderer and the same lever. With the bar measured, the
     margin is a steady 1.09% to 1.13%.

  2. Frames were paired by POSITION in the list, and only the first frame of
     each set was scored. shot.py drops a frame that failed to write
     (`if img.exists(): images.append(...)`), so the list shifts and frame 504
     gets scored against the candidate's frame 503 — silently, and reported
     as a clean run.

  3. A missing frame, an unreadable image, or a missing PIL/numpy left the
     guardrail half-measured instead of unmeasured.

A guardrail that cannot be measured must DENY, not pass: v12.objective refuses
a declared guardrail with no samples, and that refusal is the thing these
tests protect.

Nothing here renders, touches the GPU, or writes outside a temp dir. The
images are synthetic PNGs written into a temporary directory.
"""
from __future__ import annotations

import re
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import common, levers, paired  # noqa: E402

sys.path.insert(0, str(common.V12_DIR))
from v12.objective import render_objective  # noqa: E402


#: A control that stays put and a candidate that is genuinely faster, with
#: enough spread in the differences for the paired test to have something to
#: chew on. Same numbers the record-serialisation tests use.
BASE_SERIES = [48.20, 48.10, 48.35, 48.05, 48.15]
CAND_SERIES = [43.90, 43.55, 43.85, 43.60, 43.50]
DIFFS = [c - b for b, c in zip(BASE_SERIES, CAND_SERIES)]


# ---------------------------------------------------------------------------
# synthetic frames
# ---------------------------------------------------------------------------
def _frame(seed: int, width: int = 96, height: int = 64, shift: float = 0.0):
    """A deterministic frame.

    Two frames built from different seeds are genuinely different pictures
    (cross-SSIM ~0.03 on this box); two built from the same seed with a
    little noise on them are the same picture, which is what two renders of
    the same frame by the same renderer look like (cross-SSIM ~0.998).
    """
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(int(seed))
    coarse = rng.integers(0, 256, size=(8, 12)).astype("float64")
    img = np.asarray(Image.fromarray(coarse.astype("uint8"))
                     .resize((width, height), Image.BILINEAR), dtype="float64")
    if shift:
        noise = np.random.default_rng(int(seed) * 7919 + int(shift * 100))
        img = img + noise.normal(0.0, shift, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))


class _Frames:
    """A temp directory of rendered frames, named so they carry their frame."""

    def __init__(self, tmp: Path):
        self.tmp = tmp

    def write(self, name: str, seed: int, shift: float = 0.0,
              frame: int = 503) -> str:
        p = self.tmp / f"{name}_{frame:04d}.png"
        _frame(seed, shift=shift).save(p)
        return str(p)

    def shot(self, name: str, seeds, shift: float = 0.0,
             frames=(503, 504, 505)) -> list:
        """A whole shot: one file per frame number."""
        return [self.write(name, seed, shift=shift, frame=f)
                for f, seed in zip(frames, seeds)]


# ---------------------------------------------------------------------------
# 1 + 2. the bar is a measurement
# ---------------------------------------------------------------------------
class TestTheBarIsAMeasurement(unittest.TestCase):
    """The baseline side must contain only real measurements."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.f = _Frames(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_baseline_side_is_not_the_reference_scored_against_itself(self):
        """Two DIFFERENT renders: the bar must not be a free 1.0.

        The old code scored the reference against the reference's own
        repetition, which is 1.0 by construction. A bar that contains a
        number nobody rendered is not a bar.
        """
        a = self.f.write("rep0", seed=1)
        b = self.f.write("rep1", seed=2)
        base, cand = paired.ssim_series([[a], [b]], [[a], [b]])
        self.assertTrue(base, "the baseline side must carry a measurement")
        for s in base:
            self.assertLess(s, 1.0,
                            "the baseline side contains a self-comparison (1.0 by "
                            "construction), not a measurement")
        self.assertGreater(min(base), 0.0)

    def test_two_renders_of_the_same_frame_agree_closely(self):
        """The bar is the renderer's variation: two renders, one picture."""
        a = self.f.write("rep0", seed=7)
        b = self.f.write("rep1", seed=7, shift=1.0)
        base, cand = paired.ssim_series([[a], [b]], [[a], [b]])
        self.assertTrue(base)
        self.assertGreater(statistics.fmean(base), 0.95,
                           "two renders of the same frame by the same renderer "
                           "must score high against the shared reference")
        self.assertLess(statistics.fmean(base), 1.0)

    def test_the_bar_needs_a_second_baseline_repetition(self):
        """One render cannot measure the renderer's variation.

        The old code reported a bar of [1.0] here — a number that was never
        rendered. There is no measurement to be had from a single render, so
        the guardrail is unmeasured and the gate denies.
        """
        a = self.f.write("rep0", seed=3)
        base, cand = paired.ssim_series([[a]], [[a]])
        self.assertEqual(base, [])
        self.assertEqual(cand, [])

    def test_the_guardrail_is_unmeasured_with_one_baseline_repetition(self):
        """Fail closed: a declared guardrail that was not measured denies."""
        a = self.f.write("rep0", seed=3)
        res = paired.PairResult(baseline=paired.ArmResult("baseline"),
                                 candidate=paired.ArmResult("candidate"))
        res.baseline.image_sets = [[a]]
        res.candidate.image_sets = [[a]]
        self.assertIsNone(levers.ssim_guardrail(res))


# ---------------------------------------------------------------------------
# 3. the guard still bites
# ---------------------------------------------------------------------------
class TestTheGuardStillBites(unittest.TestCase):
    """A candidate that renders a different PICTURE must be denied."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.f = _Frames(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_corrupted_candidate_scores_far_below_the_bar(self):
        good = self.f.write("rep0", seed=11)
        good2 = self.f.write("rep1", seed=11, shift=1.0)
        bad = self.f.write("rep0bad", seed=99)
        base, cand = paired.ssim_series([[good], [good2]], [[bad], [bad]])
        self.assertTrue(base and cand)
        self.assertLess(statistics.fmean(cand), statistics.fmean(base) * 0.989,
                        "a candidate that renders a different picture must sit "
                        "below the bar, or the guardrail guards nothing")

    def test_the_gate_denies_a_corrupted_candidate(self):
        """End to end: the objective's own gate refuses the change."""
        good = self.f.write("rep0", seed=11)
        good2 = self.f.write("rep1", seed=11, shift=1.0)
        bad = self.f.write("rep0bad", seed=99)
        base, cand = paired.ssim_series([[good], [good2]], [[bad], [bad]])

        obj = render_objective()
        admitted, why, detail = obj.admits(
            BASE_SERIES, CAND_SERIES,
            guardrail_samples={"ssim": (base, cand)},
            paired_diffs=DIFFS)
        self.assertFalse(admitted, "the gate admitted a candidate that renders "
                                   "a different picture")
        self.assertIn("ssim", why)

    def test_a_clean_candidate_is_admitted(self):
        """The other direction: the fix must not blunt the guard."""
        a = self.f.write("rep0", seed=21)
        b = self.f.write("rep1", seed=21, shift=1.0)
        c = self.f.write("repc", seed=21, shift=1.0)
        base, cand = paired.ssim_series([[a], [b]], [[c], [c]])
        obj = render_objective()
        admitted, why, detail = obj.admits(
            BASE_SERIES, CAND_SERIES,
            guardrail_samples={"ssim": (base, cand)},
            paired_diffs=DIFFS)
        self.assertTrue(admitted, f"a clean candidate was denied: {why}")


# ---------------------------------------------------------------------------
# 4. frames are paired by number, and every frame is checked
# ---------------------------------------------------------------------------
class TestFramesArePairedByNumber(unittest.TestCase):
    """A missing frame is a failed measurement, not something to re-pair."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.f = _Frames(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_missing_frame_on_the_candidate_side_is_refused(self):
        base0 = self.f.shot("rep0", [5, 6, 7])
        base1 = self.f.shot("rep1", [5, 6, 7], shift=1.0)
        # The candidate's frame 504 never landed. shot.py drops it silently,
        # so the list would be [503, 505] and an index pairing would score
        # the baseline's 504 against the candidate's 505.
        cand0 = [self.f.write("cand0", 5, frame=503),
                 self.f.write("cand0", 7, frame=505)]
        cand1 = [self.f.write("cand1", 5, frame=503),
                 self.f.write("cand1", 7, frame=505)]
        notes: list = []
        base, cand = paired.ssim_series([base0, base1], [cand0, cand1],
                                        notes=notes)
        self.assertEqual(base, [], "a missing frame must leave the guardrail "
                                   "unmeasured, not half-measured")
        self.assertEqual(cand, [])
        self.assertTrue(any("504" in n for n in notes),
                        f"the missing frame must be visible: {notes}")

    def test_a_missing_frame_on_the_baseline_side_is_refused(self):
        base0 = [self.f.write("rep0", 5, frame=503),
                 self.f.write("rep0", 6, frame=504)]
        base1 = [self.f.write("rep1", 5, frame=503),
                 self.f.write("rep1", 6, frame=504)]
        cand0 = self.f.shot("cand0", [5, 6, 7])
        cand1 = self.f.shot("cand1", [5, 6, 7], shift=1.0)
        notes: list = []
        base, cand = paired.ssim_series([base0, base1], [cand0, cand1],
                                        notes=notes)
        self.assertEqual((base, cand), ([], []))
        self.assertTrue(any("505" in n for n in notes),
                        f"the missing frame must be visible: {notes}")

    def test_frames_are_paired_by_number_not_by_position(self):
        """The candidate's frames in a different ORDER still pair correctly.

        Position pairing would compare the baseline's 503 with the
        candidate's 505 and call a re-ordered shot a changed picture.
        """
        base0 = self.f.shot("rep0", [5, 6, 7])
        base1 = self.f.shot("rep1", [5, 6, 7], shift=1.0)
        cand0 = [self.f.write("cand0", 7, frame=505),
                 self.f.write("cand0", 5, frame=503),
                 self.f.write("cand0", 6, frame=504)]
        cand1 = [self.f.write("cand1", 7, frame=505),
                 self.f.write("cand1", 5, frame=503),
                 self.f.write("cand1", 6, frame=504)]
        base, cand = paired.ssim_series([base0, base1], [cand0, cand1])
        self.assertTrue(base and cand)
        self.assertGreater(statistics.fmean(cand), 0.95,
                           "the frames must be paired by number: a re-ordered "
                           "shot is the same shot")

    def test_every_frame_is_scored_not_just_the_first(self):
        """A changed frame in the middle of the shot must be caught."""
        base0 = self.f.shot("rep0", [5, 6, 7])
        base1 = self.f.shot("rep1", [5, 6, 7], shift=1.0)
        # The candidate's middle frame is a different picture.
        cand0 = [self.f.write("cand0", 5, frame=503),
                 self.f.write("cand0", 99, frame=504),
                 self.f.write("cand0", 7, frame=505)]
        cand1 = [self.f.write("cand1", 5, frame=503),
                 self.f.write("cand1", 99, frame=504),
                 self.f.write("cand1", 7, frame=505)]
        base, cand = paired.ssim_series([base0, base1], [cand0, cand1])
        self.assertTrue(cand)
        self.assertLess(statistics.fmean(cand), 0.95,
                        "only the first frame was scored before; a changed "
                        "middle frame has to move the number")

    def test_a_frame_that_does_not_name_itself_is_refused(self):
        """No frame number means no pairing, so no measurement."""
        p = self.tmp / "render.png"
        _frame(1).save(p)
        notes: list = []
        base, cand = paired.ssim_series([[str(p)], [str(p)]], [[str(p)], [str(p)]],
                                        notes=notes)
        self.assertEqual((base, cand), ([], []))
        self.assertTrue(any("frame" in n for n in notes), notes)


# ---------------------------------------------------------------------------
# 5. it still fails closed
# ---------------------------------------------------------------------------
class TestTheGuardrailFailsClosed(unittest.TestCase):
    """No images, no PIL, no numpy: unmeasured, and the gate denies."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.f = _Frames(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_images_leave_the_guardrail_unmeasured(self):
        notes: list = []
        base, cand = paired.ssim_series([["nope_0503.png"], ["nope_0503.png"]],
                                        [["nope_0503.png"], ["nope_0503.png"]],
                                        notes=notes)
        self.assertEqual((base, cand), ([], []))
        self.assertTrue(notes, "the refusal must say why")

    def test_no_pil_leaves_the_guardrail_unmeasured(self):
        a = self.f.write("rep0", seed=1)
        b = self.f.write("rep1", seed=2)
        with mock.patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            base, cand = paired.ssim_series([[a], [b]], [[a], [b]])
        self.assertEqual((base, cand), ([], []))

    def test_no_numpy_leaves_the_guardrail_unmeasured(self):
        a = self.f.write("rep0", seed=1)
        b = self.f.write("rep1", seed=2)
        with mock.patch.dict(sys.modules, {"numpy": None}):
            base, cand = paired.ssim_series([[a], [b]], [[a], [b]])
        self.assertEqual((base, cand), ([], []))

    def test_the_guardrail_is_none_when_it_could_not_be_measured(self):
        res = paired.PairResult(baseline=paired.ArmResult("baseline"),
                                 candidate=paired.ArmResult("candidate"))
        res.baseline.image_sets = [["missing_0503.png"]]
        res.candidate.image_sets = [["missing_0503.png"]]
        self.assertIsNone(levers.ssim_guardrail(res))

    def test_the_gate_denies_a_declared_but_unmeasured_guardrail(self):
        """V12's own rule: a declared guardrail is never skipped."""
        obj = render_objective()
        admitted, why, detail = obj.admits(
            BASE_SERIES, CAND_SERIES,
            guardrail_samples={"ssim": ([], [])},
            paired_diffs=DIFFS)
        self.assertFalse(admitted)
        self.assertIn("ssim", why)


# ---------------------------------------------------------------------------
# the real renders on disk
# ---------------------------------------------------------------------------
SHOTS = Path(r"E:\OptimizerV12\measurements\husk_shots")
_NAME = re.compile(r"(\d+)_(\d+)_(.+)_(\d+)\.png$")


def _real_sessions():
    """The frames_per_process sessions on disk, as (baseline_sets, cand_sets).

    A session is a run of repetitions: the harness labels them split0,
    split1, split2 and batched0, batched1, batched2, and a new session starts
    at the next split0. Read-only — these are the real records.
    """
    reps = {}
    for p in sorted(SHOTS.glob("*.png")):
        m = _NAME.match(p.name)
        if not m:
            continue
        label, frame = m.group(3), m.group(4)
        if not label.startswith(("split", "batched")):
            continue
        reps.setdefault((m.group(1) + "_" + m.group(2), label), {})[frame] = str(p)
    sessions, cur = [], []
    for key in sorted(reps):
        if key[1] == "split0" and cur:
            sessions.append(cur)
            cur = []
        cur.append(key)
    if cur:
        sessions.append(cur)

    def index(label):
        return int(re.sub(r"^[a-z]+", "", label))

    out = []
    for s in sessions:
        split = {k: reps[k] for k in s if k[1].startswith("split") and index(k[1]) >= 0}
        batched = {k: reps[k] for k in s if k[1].startswith("batched") and index(k[1]) >= 0}
        if len(split) < 2 or len(batched) < 2:
            continue
        base_sets = [[split[k][f] for f in sorted(split[k])]
                     for k in sorted(split, key=lambda k: index(k[1]))]
        cand_sets = [[batched[k][f] for f in sorted(batched[k])]
                     for k in sorted(batched, key=lambda k: index(k[1]))]
        out.append((base_sets, cand_sets))
    return out


@unittest.skipUnless(SHOTS.is_dir(), "no measurement root on this box")
class TestTheRealRenders(unittest.TestCase):
    """The fix, checked against the frames that were actually rendered.

    Read-only: these are the real records of the frames_per_process runs on
    this box, and the point of the test is that the guardrail now reports the
    renderer's real variation instead of a 1.0 it never measured.
    """

    def test_a_real_session_has_no_free_one_and_sits_near_the_floor(self):
        sessions = _real_sessions()
        if not sessions:
            self.skipTest("no frames_per_process session on disk")
        for base_sets, cand_sets in sessions:
            base, cand = paired.ssim_series(base_sets, cand_sets)
            self.assertTrue(base and cand)
            for s in base:
                self.assertLess(s, 1.0,
                                "the bar contains the reference scored against "
                                "itself — a number that was never rendered")
            # The candidate renders the same shot in a different mode. On this
            # box that lands at ~0.9949, and it must not be far below the bar.
            self.assertGreater(statistics.fmean(base), 0.99)
            self.assertGreater(statistics.fmean(cand), 0.99)
            self.assertLess(abs(statistics.fmean(cand) - statistics.fmean(base)),
                            0.002,
                            "the cross-mode offset must stay small — the mode "
                            "difference is inside the renderer's own noise")

    def test_the_real_candidate_side_sits_at_the_documented_floor(self):
        """0.994 is the documented floor for a cross-mode render."""
        sessions = _real_sessions()
        if not sessions:
            self.skipTest("no frames_per_process session on disk")
        for base_sets, cand_sets in sessions:
            base, cand = paired.ssim_series(base_sets, cand_sets)
            self.assertGreater(statistics.fmean(cand), 0.9935)
            self.assertLess(statistics.fmean(cand), 0.9965)


if __name__ == "__main__":
    unittest.main(verbosity=2)
