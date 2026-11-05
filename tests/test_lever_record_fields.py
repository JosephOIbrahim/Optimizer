"""
Tests for the two record fields the frames-per-process lever used to lose.

What this prevents
------------------
Two separate drops, both found by reading the ledger rather than the code.

1. `setup_tax_s` — the number that EXPLAINS the lever's win. The setup tax
   is the ~38 s per husk launch that the shot path pays for shader compile,
   texture load and BVH build. Inside one process the first frame pays it
   and the frames after it do not; the split arm pays it once per frame. The
   difference between the arms IS the tax, and the tax is the reason a 47%
   win is not a mystery.

   It lived in `Result.notes`, and `optimize.result_block` did not serialize
   `notes` — so the 11-04 records carried `setup_tax_s` and the 11-05 ones
   did not. The number was computed, printed to the console, and dropped on
   the way to the file. A reader of the 11-05 record could see THAT the
   lever won and not WHY, and had to take the harness's word for the
   mechanism.

2. `frame_wall_s` — one key, two meanings. The batched arm's per-frame wall
   times are INCREMENTAL (what each frame added to the shared process, so
   the series reads [54.70, 16.03, 15.66]); the split arm's are STANDALONE
   (what each frame cost in a process to itself, so the series reads
   [54.70, 54.70, 54.70]). They travelled under the same key in the same
   `extra` dict, so a reader had to know which arm produced a number before
   they could interpret it — and the natural mistake, comparing one series
   against the other, reads as "the batched arm made frames 2 and 3 three
   times faster", which is true and is the lever's whole point and is not
   what either number measures.

Both are the same failure: a field that is computed, used, and then dropped
before it reaches the file. A record that omits its own explanation is a
claim rather than a measurement.

Nothing here renders, touches the GPU, or writes outside a temp dir: the
arms are stubs and the record is written into a temporary directory.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import common, constraint, levers, optimize, paired  # noqa: E402

sys.path.insert(0, str(common.V12_DIR))
from v12.objective import shot_objective  # noqa: E402


#: The two kinds of per-frame wall time, as the harness actually measured them
#: on this box: the batched arm's in-process series (first frame pays the
#: setup) against the split arm's standalone series (every frame pays it).
BATCHED_IN_PROCESS = [54.70, 16.03, 15.66]
SPLIT_STANDALONE = [54.70, 54.70, 54.70]

#: What the setup tax must come out as, from the batched series: the first
#: frame's cost minus the median of the frames after it.
EXPECTED_TAX = 54.70 - 15.845


def _no_gpu_probe(*a, **kw):
    """A contention probe that sees nobody on the card."""
    return []


class _ShotStub:
    """Stands in for `harness.shot` so the arms can run without husk.

    `render_shot` is the BATCHED arm's source and `render_frames_separately`
    is the SPLIT arm's. Each hands back the per-frame wall series its real
    counterpart produces, which is the whole point: the two series differ,
    and the code has to keep them apart.

    The images are REAL PNGs on disk, named with their frame number, because
    the guardrail pairs frames by number: a stub that returned no images at
    all would leave the guardrail unmeasured, and the point of these tests
    is the record, not the guardrail.
    """

    def __init__(self, tmpdir, in_process=BATCHED_IN_PROCESS,
                 standalone=SPLIT_STANDALONE):
        self.tmpdir = Path(tmpdir)
        self.in_process = list(in_process)
        self.standalone = list(standalone)
        self.calls = []

    def _write(self, label, frames, start_frame):
        """Real PNGs, one per frame, named so the frame number is in the name."""
        from PIL import Image
        out = []
        for i in range(frames):
            f = start_frame + i
            # A different flat value per arm, so a cross-arm comparison is a
            # real comparison rather than a comparison of two identical files.
            v = 60 + (40 if "batched" in label else 0) + i
            p = self.tmpdir / f"{label}_{f:04d}.png"
            Image.new("L", (64, 64), color=v).save(p)
            out.append(str(p))
        return out

    def render_shot(self, usd, frames, start_frame=503, settings="",
                   pixel_samples=128, res_scale=100, renderer="", husk="",
                   out_dir=None, label="shot", threads=0):
        self.calls.append(("batched", label))
        return {
            "wall_s": sum(self.in_process),
            "frame_wall_s": list(self.in_process),
            "frame_render_s": [16.0] * frames,
            "images": self._write(label, frames, start_frame),
            "throttle": {},
        }

    def render_frames_separately(self, usd, frames, start_frame=503, settings="",
                                 pixel_samples=128, res_scale=100, renderer="",
                                 husk="", out_dir=None, label="shot_split",
                                 threads=0):
        self.calls.append(("split", label))
        return {
            "wall_s": sum(self.standalone),
            "frame_wall_s": list(self.standalone),
            "frame_render_s": [16.0] * frames,
            "images": self._write(label, frames, start_frame),
            "throttle": {},
        }


class _Arms:
    """The two arms of the frames-per-process lever, built for real.

    The arms resolve `shot` at CALL time, so the stub has to stay patched
    for as long as the arm is called — hence the context manager.
    """

    def __init__(self, stub=None, tmpdir=None):
        self._tmp = None
        if stub is None:
            self._tmp = tempfile.TemporaryDirectory()
            tmpdir = self._tmp.name
            stub = _ShotStub(tmpdir)
        self.stub = stub
        self.base = None
        self.cand = None
        self._patch = None

    def __enter__(self):
        self._patch = mock.patch.object(levers, "shot", self.stub)
        self._patch.start()
        self.base = levers.render_split_arm("u", 3, "", 128, 100, 503, "split")
        self.cand = levers.render_shot_arm("u", 3, "", 128, 100, 503, "batched")
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        if self._tmp is not None:
            self._tmp.cleanup()
        return False


class TestTheSetupTaxReachesTheRecord(unittest.TestCase):
    """`setup_tax_s` must be in the file, not just on the screen."""

    def test_the_tax_is_measured_from_the_batched_series(self):
        """The tax is the first frame's cost minus what the later frames pay.

        Measured from the BATCHED arm's in-process series, because that is
        the series where the setup shows up as a difference between the
        first frame and the rest. The split arm's series is flat — every
        frame is alone in a process — so a tax read from it is 0.0, and a
        tax of 0.0 is a plausible-looking number that is simply wrong.
        """
        tax = constraint.setup_tax(BATCHED_IN_PROCESS)
        self.assertAlmostEqual(tax, EXPECTED_TAX, places=3)

        # The split arm's series, read the wrong way, gives zero.
        wrong = constraint.setup_tax(SPLIT_STANDALONE)
        self.assertEqual(wrong, 0.0,
                         "the split series is flat — it cannot measure a tax")

    def test_the_record_carries_the_tax(self):
        """End to end: a lever result with a tax becomes a record with it.

        This is the test that would have failed on 2026-11-05, when the
        record was written without `setup_tax_s` while the console printed
        it.
        """
        result = levers.Result(
            lever="frames_per_process", ok=True, effect=0.4695,
            detail={"setup_tax_s": EXPECTED_TAX})
        block = optimize.result_block(result)
        self.assertIn("setup_tax_s", block,
                      "the setup tax must reach the record — it is the only "
                      "number in it that explains WHY the lever won")
        self.assertAlmostEqual(block["setup_tax_s"], EXPECTED_TAX, places=3)

    def test_the_written_file_carries_the_tax(self):
        """Not just the dict — the FILE, as JSON on disk."""
        result = levers.Result(
            lever="frames_per_process", ok=True, effect=0.4695,
            detail={"setup_tax_s": EXPECTED_TAX})
        record = optimize.build_record("shot", _obj(), {"name": "test card"},
                                       [("frames_per_process", result)])
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with mock.patch.object(common, "MEASURE_ROOT", tmp), \
                    mock.patch.object(common, "CONFIG_PATH", tmp / "config.json"):
                path = common.write_measurement("optimize", record)
                written = json.loads(path.read_text(encoding="utf-8"))
        block = written["results"]["frames_per_process"]
        self.assertIn("setup_tax_s", block)
        self.assertAlmostEqual(block["setup_tax_s"], EXPECTED_TAX, places=3)

    def test_the_tax_is_re_derivable_from_the_recorded_series(self):
        """The series the tax came from travels too, so the tax is checkable.

        A tax a reader cannot re-derive is a tax they have to take on trust.
        This repo's whole reason for keeping raw series in the record.
        """
        result = levers.Result(
            lever="frames_per_process", ok=True, effect=0.4695,
            detail={"setup_tax_s": EXPECTED_TAX,
                    levers.FRAME_WALL_IN_PROCESS: [BATCHED_IN_PROCESS]})
        record = optimize.build_record("shot", _obj(), {"name": "test card"},
                                       [("frames_per_process", result)])
        block = record["results"]["frames_per_process"]
        series = block[levers.FRAME_WALL_IN_PROCESS][0]
        self.assertAlmostEqual(constraint.setup_tax(series),
                               block["setup_tax_s"], places=6)


class TestTheLeverPutsTheTaxInTheRecord(unittest.TestCase):
    """The lever itself, run against stubbed shot calls.

    Everything here is the production path: the production arms, the
    production paired engine, the production gate. Only husk is stubbed.
    """

    def _run_lever(self):
        ctx = levers.Context(usd="u", frames=3, reps=3, warmup=0)
        obj = shot_objective(resolution="1024x429", samples=128, frames=3)
        with _Arms() as arms:
            with mock.patch.object(levers, "_quiet_gate", lambda t: (lambda: True)), \
                    mock.patch.object(common, "other_gpu_processes", _no_gpu_probe):
                result = levers.lever_frames_per_process(ctx, obj)
        return result

    def test_the_tax_is_in_the_detail(self):
        result = self._run_lever()
        self.assertIn("setup_tax_s", result.detail,
                      "the lever must put the tax where the record reads it")
        self.assertAlmostEqual(result.detail["setup_tax_s"], EXPECTED_TAX, places=3)

    def test_the_tax_is_not_zero(self):
        """A tax of 0.0 is a plausible-looking number that is simply wrong.

        It is what you get from reading the FLAT series — the split arm's,
        where every frame is alone in a process and pays its own setup. The
        lever must read the batched arm's series, where the setup shows up
        as the difference between the first frame and the rest.
        """
        result = self._run_lever()
        self.assertGreater(result.detail["setup_tax_s"], 30.0,
                           "the setup tax on this box is ~38 s — a zero means "
                           "the wrong series was read")

    def test_the_note_says_what_the_tax_is(self):
        result = self._run_lever()
        self.assertTrue(any("setup tax" in n for n in result.notes),
                        f"the note explaining the tax is missing: {result.notes}")

    def test_both_series_reach_the_record(self):
        """Both kinds of per-frame time, under names that say which is which."""
        result = self._run_lever()
        self.assertIn(levers.FRAME_WALL_IN_PROCESS, result.detail)
        self.assertIn(levers.FRAME_WALL_STANDALONE, result.detail)
        self.assertEqual(result.detail[levers.FRAME_WALL_IN_PROCESS][0],
                         BATCHED_IN_PROCESS)
        self.assertEqual(result.detail[levers.FRAME_WALL_STANDALONE][0],
                         SPLIT_STANDALONE)

    def test_the_record_block_carries_the_tax(self):
        """The whole chain: lever -> result -> record block -> file."""
        result = self._run_lever()
        record = optimize.build_record("shot", _obj(), {"name": "test card"},
                                       [("frames_per_process", result)])
        block = record["results"]["frames_per_process"]
        self.assertIn("setup_tax_s", block)
        self.assertAlmostEqual(block["setup_tax_s"], EXPECTED_TAX, places=3)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with mock.patch.object(common, "MEASURE_ROOT", tmp), \
                    mock.patch.object(common, "CONFIG_PATH", tmp / "config.json"):
                path = common.write_measurement("optimize", record)
                written = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("setup_tax_s", written["results"]["frames_per_process"])


class TestTheNotesReachTheRecord(unittest.TestCase):
    """`Result.notes` is the harness's explanation; it was being dropped."""

    def test_the_notes_are_in_the_block(self):
        result = levers.Result(
            lever="frames_per_process", ok=True,
            notes=["setup tax: 38.9 s per launch, paid once for 3 frames "
                   "instead of 3 times"])
        block = optimize.result_block(result)
        self.assertIn("notes", block)
        self.assertEqual(len(block["notes"]), 1)
        self.assertIn("setup tax", block["notes"][0])

    def test_applied_is_in_the_block(self):
        """The standing assertion that nothing was left changed.

        The 11-04 records carry `"applied": false`. A reader checking that
        the machine was left alone should not have to take it on trust.
        """
        result = levers.Result(lever="power_ceiling", ok=True, applied=False)
        block = optimize.result_block(result)
        self.assertIn("applied", block)
        self.assertFalse(block["applied"])

    def test_the_notes_survive_a_json_dump(self):
        result = levers.Result(lever="frames_per_process", ok=True,
                               notes=["setup tax: 38.9 s per launch"])
        block = optimize.result_block(result)
        self.assertEqual(json.loads(json.dumps(block))["notes"],
                         ["setup tax: 38.9 s per launch"])


class TestTheTwoFrameWallSeriesStayApart(unittest.TestCase):
    """One key, two meanings, was the trap. Now they are two keys."""

    def test_the_two_keys_are_different(self):
        self.assertNotEqual(levers.FRAME_WALL_IN_PROCESS,
                            levers.FRAME_WALL_STANDALONE)
        self.assertNotEqual(levers.FRAME_WALL_IN_PROCESS, "frame_wall_s")
        self.assertNotEqual(levers.FRAME_WALL_STANDALONE, "frame_wall_s")

    def test_the_batched_arm_reports_the_in_process_series(self):
        with _Arms() as arms:
            out = arms.cand(0)
        self.assertIn(levers.FRAME_WALL_IN_PROCESS, out["extra"])
        self.assertNotIn(levers.FRAME_WALL_STANDALONE, out["extra"])
        self.assertNotIn("frame_wall_s", out["extra"])
        self.assertEqual(out["extra"][levers.FRAME_WALL_IN_PROCESS],
                         BATCHED_IN_PROCESS)

    def test_the_split_arm_reports_the_standalone_series(self):
        with _Arms() as arms:
            out = arms.base(0)
        self.assertIn(levers.FRAME_WALL_STANDALONE, out["extra"])
        self.assertNotIn(levers.FRAME_WALL_IN_PROCESS, out["extra"])
        self.assertNotIn("frame_wall_s", out["extra"])
        self.assertEqual(out["extra"][levers.FRAME_WALL_STANDALONE],
                         SPLIT_STANDALONE)

    def test_neither_arm_publishes_the_ambiguous_key(self):
        """`frame_wall_s` meant two things; nothing should publish it now."""
        with _Arms() as arms:
            outs = (arms.base(0), arms.cand(0))
        for out in outs:
            self.assertNotIn("frame_wall_s", out["extra"])

    def test_the_record_can_carry_both_without_confusion(self):
        """Both series in the record, under names that say which is which."""
        result = levers.Result(
            lever="frames_per_process", ok=True,
            detail={"setup_tax_s": EXPECTED_TAX,
                    levers.FRAME_WALL_IN_PROCESS: [BATCHED_IN_PROCESS],
                    levers.FRAME_WALL_STANDALONE: [SPLIT_STANDALONE]})
        record = optimize.build_record("shot", _obj(), {"name": "test card"},
                                       [("frames_per_process", result)])
        block = record["results"]["frames_per_process"]
        self.assertIn(levers.FRAME_WALL_IN_PROCESS, block)
        self.assertIn(levers.FRAME_WALL_STANDALONE, block)
        # The two series are DIFFERENT numbers, and the record keeps them
        # apart rather than merging them into one list.
        self.assertNotEqual(block[levers.FRAME_WALL_IN_PROCESS],
                            block[levers.FRAME_WALL_STANDALONE])


class TestTheArmsKeepTheirOwnSeries(unittest.TestCase):
    """`extra` keeps only the LAST repetition's numbers; the arms keep all.

    `paired.Pair` merges each repetition's `extra` with `.update()`, so the
    per-frame series in `extra` is the last repetition's and nothing else.
    The record needs the whole series, so the arms accumulate their own.
    """

    def test_the_series_accumulates_across_repetitions(self):
        with _Arms() as arms:
            for i in range(3):
                arms.cand(i)
                arms.base(i)
            self.assertEqual(len(arms.cand.frame_wall_series), 3)
            self.assertEqual(len(arms.base.frame_wall_series), 3)

    def test_the_warm_up_repetition_is_not_in_the_series(self):
        """The warm-up runs with index -1 and is discarded by the engine.

        It must not appear in a series that describes the measured run.
        """
        with _Arms() as arms:
            arms.cand(-1)
            self.assertEqual(arms.cand.frame_wall_series, {})

    def test_a_retried_repetition_is_counted_once(self):
        """A repetition discarded for contention is retried with the same index.

        Appending would count it twice; keying by index does not.
        """
        with _Arms() as arms:
            arms.cand(0)
            arms.cand(0)
            self.assertEqual(len(arms.cand.frame_wall_series), 1)

    def test_the_series_comes_back_in_repetition_order(self):
        with _Arms() as arms:
            for i in (2, 0, 1):
                arms.cand(i)
            ordered = levers._ordered_series(arms.cand.frame_wall_series)
        self.assertEqual(len(ordered), 3)
        # Every entry is one repetition's frame series, in index order.
        for entry in ordered:
            self.assertEqual(entry, BATCHED_IN_PROCESS)

    def test_the_ordered_series_drops_the_warm_up(self):
        with _Arms() as arms:
            arms.cand(-1)
            arms.cand(0)
            self.assertEqual(len(levers._ordered_series(arms.cand.frame_wall_series)), 1)


def _obj():
    """A throwaway objective; these tests never gate on it."""
    return shot_objective(resolution="1024x429", samples=128, frames=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
