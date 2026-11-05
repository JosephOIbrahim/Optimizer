"""
Tests for the promises the harness makes, and for the refusals it must keep.

Two things are protected here.

THE UNDO PROMISE. The operator card says the power ceiling "is restored to
its prior value after every run, and the prior is read back to prove it", and
that "nothing was left applied". The code did not keep that promise:

  * the prior was read from the CONTEXT SNAPSHOT (ctx.spec), captured when
    the Context was built, so a ceiling moved in between was overwritten
    with the wrong number;
  * the setter's return value was DISCARDED, so a restore that failed
    (out of range, nvidia-ml-py missing, NVML error) was silent — and the
    run still printed "nothing was left applied" while the card sat at 600 W;
  * the value was never read back, so the card's claim that the prior is
    "read back to prove it" was not true either.

THE STALE-BASELINE REFUSAL. harness/method.py and harness/lever_run.py refuse
to produce a verdict from a baseline measured in another session. That
refusal used to live only in front of the helpers, which meant an importer
could call the helper directly and get the stale baseline anyway. The
refusal now lives inside the reading helpers, and the helper that computed
a verdict from any two dicts has been removed.

And the ERROR TABLE. The card's "When it breaks" table is a contract with
the operator: every string in it must be a string the code actually prints,
and every remedy must actually work. A table that quotes strings the code
never prints is worse than no table — it sends the operator looking for a
message that will never appear. So the table is checked against the source.

Nothing here touches the GPU, renders anything, or writes outside a temp dir.
"""
from __future__ import annotations

import io
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harness import levers, optimize  # noqa: E402

CARD = REPO / "Optimize-The-4090-Card.md"
HARNESS_CARD = REPO / "Harness-Operator-Card.md"


# ---------------------------------------------------------------------------
# fakes: a lever run that never touches the GPU
# ---------------------------------------------------------------------------
class _FakeArm:
    def __init__(self, values):
        self.values = list(values)

    def summary(self):
        return {"mean": sum(self.values) / len(self.values)}


class _FakePairResult:
    """Just enough of a paired result for the gate and the report."""

    def __init__(self):
        self.baseline = _FakeArm([10.0] * 5)
        self.candidate = _FakeArm([12.0] * 5)
        self.diffs = [2.0] * 5
        self.baseline_drift = 0.01
        self.drift_exceeded = False
        # control_spread / control_exceeded are the robust "did the control
        # move ANYWHERE in the run" pair, alongside the end-to-end drift.
        self.control_spread = 0.01
        self.control_exceeded = False
        self.contended_samples = 0
        self.contenders = []
        self.notes = []
        self.image_sets = []

    def describe(self):
        return "fake pair result"

    def raw_series(self):
        return {}


class _FakePair:
    def __init__(self, *args, **kwargs):
        self.quiet_check = None

    def run(self):
        return _FakePairResult()


def _ctx(spec):
    """A Context whose spec is given, so machine_spec() never reads the GPU."""
    return levers.Context(usd=r"E:\OptimizerV12\scenes\ovoid_hero_v002.usd", spec=spec)


def _obj():
    from v12.objective import render_objective
    return render_objective()


def _run_power_lever(spec, power_get, power_set):
    """Run the power_ceiling lever with the GPU and the pair engine faked.

    Returns (result, stdout, power_set_calls).
    """
    calls: list = []

    def _set(watts):
        calls.append(watts)
        return power_set(watts)

    ctx = _ctx(spec)
    buf = io.StringIO()
    # The stub takes `settle_s` because the real arm does: the card is left
    # at the new ceiling for a moment before the measurement, so the reading
    # describes its steady behaviour rather than the first second at the new
    # limit.
    def _gemm_arm(watts, label, settle_s=0.0):
        return (lambda index: {})

    with mock.patch.object(levers, "power_get", power_get), \
         mock.patch.object(levers, "power_set", _set), \
         mock.patch.object(levers, "gemm_arm", _gemm_arm), \
         mock.patch.object(levers.paired, "Pair", _FakePair):
        with redirect_stdout(buf):
            r = levers.lever_power_ceiling(ctx, _obj())
    return r, buf.getvalue(), calls


# ---------------------------------------------------------------------------
# 1. the undo promise
# ---------------------------------------------------------------------------
class TestTheUndoPromiseIsKept(unittest.TestCase):
    """The card promises the machine is left as it was found. Prove it."""

    SPEC = {"power_limit_w": 450.0, "power_limit_min_w": 150.0,
            "power_limit_max_w": 600.0}

    def test_a_failed_restore_is_not_reported_as_success(self):
        """The restore fails -> the result says so, loudly."""
        r, out, _ = _run_power_lever(
            self.SPEC,
            lambda: 450.0,
            lambda watts: (False, "nvidia-ml-py not importable"))

        self.assertTrue(r.restore_error,
                        "a failed restore must be reported on the result")
        self.assertIn("could not restore the power ceiling", r.restore_error)
        self.assertTrue(r.applied,
                        "the machine WAS left changed, so `applied` must say so")
        self.assertFalse(r.detail["power_restored"])
        self.assertIn("NOT RESTORED", out)

    def test_a_failed_restore_does_not_print_the_success_message(self):
        """The success message must not survive a failed restore."""
        r, _, _ = _run_power_lever(
            self.SPEC,
            lambda: 450.0,
            lambda watts: (False, "nvidia-ml-py not importable"))
        report = optimize.closure_report([("power_ceiling", r)], ["power_ceiling"])
        self.assertNotIn("nothing was left applied", report)
        self.assertIn("NOT LEFT AS YOU FOUND IT", report)

    def test_a_failed_restore_is_in_the_record(self):
        """The written record has to carry it too, not just the screen."""
        r, _, _ = _run_power_lever(
            self.SPEC,
            lambda: 450.0,
            lambda watts: (False, "nvidia-ml-py not importable"))
        block = optimize.result_block(r)
        self.assertTrue(block["applied"])
        self.assertIn("could not restore the power ceiling", block["restore_error"])

    def test_the_prior_restored_is_the_live_value_not_the_snapshot(self):
        """The snapshot is captured when the Context is built. The prior has
        to be read LIVE, immediately before the change."""
        spec = {"power_limit_w": 300.0,     # a STALE snapshot
                "power_limit_min_w": 150.0,
                "power_limit_max_w": 600.0}
        r, _, calls = _run_power_lever(spec, lambda: 450.0,
                                       lambda watts: (True, "ok"))

        self.assertIn(450.0, calls,
                      "the restore must write back the LIVE prior (450 W)")
        self.assertNotIn(300.0, calls,
                         "the restore must not write back the stale snapshot")
        self.assertEqual(r.detail["power_prior_w"], 450.0)
        self.assertIn("live", r.detail["power_prior_source"])

    def test_the_ceiling_is_read_back_after_restoring(self):
        """A restore that cannot be read back is not a proven restore."""
        # live prior 450, read-back after the restore says 600 -> it did not
        # go back where it was.
        reads = iter([450.0, 600.0])
        r, out, _ = _run_power_lever(self.SPEC, lambda: next(reads),
                                     lambda watts: (True, "ok"))
        self.assertFalse(r.detail["power_restored"])
        self.assertIn("NOT where it was found", r.restore_error)
        self.assertIn("NOT RESTORED", out)

    def test_a_verified_restore_says_so(self):
        reads = iter([450.0, 450.0])
        r, out, calls = _run_power_lever(self.SPEC, lambda: next(reads),
                                         lambda watts: (True, "ok"))
        self.assertTrue(r.detail["power_restored"])
        self.assertEqual(r.restore_error, "")
        self.assertFalse(r.applied)
        self.assertIn("read back", out)
        report = optimize.closure_report([("power_ceiling", r)], ["power_ceiling"])
        self.assertIn("nothing was left applied", report)

    def test_the_manual_lever_does_not_claim_nothing_was_left_applied(self):
        """The manual lever asks the operator to toggle their own change and
        never reverts it. The closing message must not deny that."""
        r = levers.Result(lever="manual", ok=True, why="")
        report = optimize.closure_report([("manual", r)], ["manual"])
        self.assertNotIn("nothing was left applied", report)
        self.assertIn("does not revert YOUR change", report)


# ---------------------------------------------------------------------------
# 2. the stale-baseline refusal cannot be bypassed
# ---------------------------------------------------------------------------
class TestTheStaleRefusalCannotBeBypassed(unittest.TestCase):
    """Importing the helpers must not be a way around the refusal."""

    def test_method_latest_baseline_refuses(self):
        from harness import method
        with self.assertRaises(SystemExit) as ctx:
            method.latest_baseline()
        msg = str(ctx.exception)
        self.assertIn("ANOTHER SESSION", msg)
        self.assertIn("12.34", msg,
                      "the refusal must name the failure it prevents")

    def test_lever_run_latest_baseline_measurement_refuses(self):
        from harness import lever_run
        with self.assertRaises(SystemExit) as ctx:
            lever_run.latest_baseline_measurement("render")
        msg = str(ctx.exception)
        self.assertIn("ANOTHER SESSION", msg)
        self.assertIn("12.34", msg)

    def test_the_verdict_helper_is_gone(self):
        """It computed an ADMIT/DENY from any two dicts, so it was a bypass
        around the refusal rather than a gate."""
        from harness import lever_run
        self.assertFalse(hasattr(lever_run, "verdict"),
                         "lever_run.verdict must not exist: it gated a candidate "
                         "against a baseline from another session")

    def test_neither_module_can_produce_a_verdict_from_a_stale_baseline(self):
        """The strongest form of the claim: put a stale baseline ON DISK, and
        prove that the helpers still refuse to hand it over.

        The old code returned that dict. A refusal that only holds when the
        file is missing is not a refusal.
        """
        import json
        import tempfile

        from harness import common, lever_run, method

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / "render_20260101_000000.json"
            stale.write_text(json.dumps({
                "label": "baseline", "render_s": [1.0, 2.0, 3.0],
                "summary": {"mean": 2.0},
            }), encoding="utf-8")

            with mock.patch.object(common, "MEASURE_ROOT", root):
                with self.assertRaises(SystemExit):
                    method.latest_baseline()
                with self.assertRaises(SystemExit):
                    lever_run.latest_baseline_measurement("render")

    def test_the_cli_still_refuses(self):
        """The refusal at the door has not been softened."""
        from harness import lever_run, method

        class _Args:
            label = "anything"
            force = False
            usd = ""
            frame = 503
            settings = ""

        with self.assertRaises(SystemExit) as ctx:
            method.cmd_candidate(_Args())
        self.assertIn("ANOTHER SESSION", str(ctx.exception))
        with self.assertRaises(SystemExit):
            lever_run.refuse_stale_comparison()


# ---------------------------------------------------------------------------
# 3. the card's error table must quote strings the code prints
# ---------------------------------------------------------------------------
def _source_text() -> str:
    """Every line of the code the card's error table is supposed to quote."""
    parts = []
    for pattern in ("harness/*.py", "OptimizerV12/v12/*.py"):
        for p in sorted(REPO.glob(pattern)):
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _card_error_rows(card: Path):
    """The first cell of every row of the card's "When it breaks" table."""
    text = card.read_text(encoding="utf-8", errors="replace")
    marker = "## When it breaks"
    if marker not in text:
        return []
    section = text[text.index(marker):]
    end = section.find("\n## ", 1)
    if end != -1:
        section = section[:end]
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if set(line) <= set("|- :"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0].lower() in ("you see", "---"):
            continue
        rows.append((cells[0], re.findall(r"`([^`]+)`", cells[0])))
    return rows


class TestTheErrorTableIsTrue(unittest.TestCase):
    """Every quoted string must be a string the code actually prints."""

    def test_the_table_has_rows(self):
        rows = _card_error_rows(CARD)
        self.assertGreaterEqual(len(rows), 8,
                                "the error table must actually list errors")

    def test_the_card_has_a_when_it_breaks_section(self):
        text = CARD.read_text(encoding="utf-8", errors="replace")
        self.assertIn("## When it breaks", text)

    def test_every_quoted_string_exists_in_the_source(self):
        source = _source_text()
        missing = []
        for cell, quoted in _card_error_rows(CARD):
            for s in quoted:
                if s not in source:
                    missing.append((cell, s))
        self.assertEqual(missing, [],
                         "the card quotes strings the code never prints: "
                         f"{missing}")

    def test_the_strings_that_do_not_exist_are_gone(self):
        """The old table quoted strings that were never printed."""
        text = CARD.read_text(encoding="utf-8", errors="replace")
        for ghost in ("permanently the machine is faster",
                      "Power limit will not set",
                      "n<3 per side"):
            self.assertNotIn(ghost, text,
                             f"the card still quotes a string the code never "
                             f"prints: {ghost!r}")

    def test_the_drift_figure_quoted_is_a_recorded_one(self):
        """The old table quoted -12.0%, which no record has."""
        text = CARD.read_text(encoding="utf-8", errors="replace")
        self.assertNotIn("-12.0%", text)
        for recorded in ("+16.3%", "-70.9%", "+7.2%", "-5.3%"):
            self.assertIn(recorded, text,
                          f"the card should quote a recorded drift ({recorded})")

    def test_the_quiet_remedy_is_the_right_one(self):
        """Closing Houdini and friends does NOT unblock the preflight: it runs
        without --quiet, so only a competing OPTIMIZER (or an unreadable
        process table) blocks it."""
        rows = _card_error_rows(CARD)
        remedy = ""
        for cell, quoted in rows:
            if any("preflight BLOCKED" in s for s in quoted):
                remedy = cell
        self.assertTrue(remedy, "the preflight row must still exist")
        text = CARD.read_text(encoding="utf-8", errors="replace")
        section = text[text.index("## When it breaks"):]
        end = section.find("\n## ", 1)
        if end != -1:
            section = section[:end]
        line = [l for l in section.splitlines()
                if "preflight BLOCKED" in l]
        self.assertTrue(line)
        self.assertNotIn("Close Houdini, Nuke, AfterFX, Chrome, OBS", line[0],
                         "that remedy is wrong: those apps do not block preflight")

    def test_the_harness_card_quotes_the_real_torch_string(self):
        """The real message is 'torch venv not ready yet - see option [5]'."""
        text = HARNESS_CARD.read_text(encoding="utf-8", errors="replace")
        self.assertIn("torch venv not ready yet - see option [5]", text)
        self.assertNotIn("`torch venv not ready`", text,
                         "the card must quote the whole string the code prints")


# ---------------------------------------------------------------------------
# 4. the card's flags, defaults and menu
# ---------------------------------------------------------------------------
class TestTheCardDescribesTheTool(unittest.TestCase):
    """A card that omits a flag is a card that sends the operator to the
    usage text when they could have been told."""

    def setUp(self):
        self.text = CARD.read_text(encoding="utf-8", errors="replace")

    def test_every_real_flag_is_documented(self):
        for flag in ("--variant", "--threads", "--probe", "--quiet-timeout",
                     "--reps", "--levers", "--usd", "--settings", "--frames",
                     "--start-frame", "--samples", "--res-scale"):
            self.assertIn(flag, self.text, f"the card never mentions {flag}")

    def test_the_default_request_is_stated(self):
        self.assertRegex(self.text, r"--request.*defaults? to `?render",
                         "the card must say that --request defaults to render")

    def test_the_menu_lists_quit_and_the_profile_note(self):
        self.assertIn("[Q] Quit", self.text, "the menu list must include [Q] Quit")
        self.assertIn("(no levers run)", self.text,
                      "option [4] must say it runs no levers")

    def test_the_undo_promise_is_scoped_to_the_manual_lever(self):
        """The manual lever never reverts the operator's change, so 'nothing
        else is touched' cannot be true of it."""
        tail = self.text[self.text.index("## Undo"):]
        self.assertIn("manual", tail.lower(),
                      "the Undo section must say the manual lever is the "
                      "exception")

    def test_the_card_does_not_overstate_the_undo(self):
        """The promise is kept by the code now — but only for the power
        ceiling, and only because the restore is checked."""
        tail = self.text[self.text.index("## Undo"):]
        self.assertIn("read back", tail.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
