"""
Tests for v12/jev.py — the one place a judgment model is allowed in.

What these hold, in order of how much damage the failure would do:
  * a judgment is never evidence: it has no evidence field, cannot carry an
    evidence-class label, and recording it leaves the honesty metric unchanged;
  * nothing leaves the machine unredacted (secrets, URL userinfo, the profile path);
  * nothing printed can drive the terminal (escape codes in network text);
  * OFF unless V12_JEV=on — an API key being present is not consent;
  * every response is validated (pinned model, labels inside the criteria,
    probabilities finite and summing to 1) and every failure is a typed outcome,
    never an exception;
  * a hard per-process cap on questions sent.

No network: the SDK import seam (`jev._import_sdk`) is replaced by a fake with no
transport at all — an unintended client construction raises inside judge() and
surfaces as outcome 'error', never as a real call — and the real
TYPESAFE_API_KEY in this environment is replaced with a dummy for every test.
Ledgers live in temp dirs; nothing touches D:\\OptimizerV12.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import inspect
import io
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v12 import cli, jev  # noqa: E402
from v12.admission import Verdict  # noqa: E402
from v12.change import ChangeKind, EvidenceClass, Tier, change_key, make  # noqa: E402
from v12.jev import Judgment, Question  # noqa: E402
from v12.ledger import Ledger  # noqa: E402

DUMMY_KEY = "dummy-not-a-real-key"


# ---------------------------------------------------------------------------
# a fake SDK: same names jev uses, no transport at all
# ---------------------------------------------------------------------------
class _Q:
    def __init__(self, **kw):
        self.kw = kw


class _FakeSDK:
    Choice = type("Choice", (_Q,), {})
    Noul = type("Noul", (_Q,), {})
    Score = type("Score", (_Q,), {})

    class RetryPolicy:
        def __init__(self, max_retries=2):
            self.max_retries = max_retries

    class TypeSafeAPIConnectionError(Exception):
        pass

    class TypeSafeAPITimeoutError(Exception):
        pass

    class TypeSafeRateLimitError(Exception):
        pass

    class TypeSafeClient:
        def __init__(self, **kw):
            raise AssertionError("a real TypeSafeClient was constructed in a test")


class _FakeClient:
    """Returns a canned response (or raises) and records what it was sent."""

    def __init__(self, response=None, exc=None):
        self.response, self.exc = response, exc
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, dict(questions)))
        if self.exc is not None:
            raise self.exc
        return self.response


CRITERIA = {"render": "a renderer is the main GPU user",
            "sim": "a simulation is the main GPU user",
            "none": "None of these"}
Q_KIND = Question.choice("workload", "What is `top_gpu_app` mostly doing?", CRITERIA)
STATE = {"top_gpu_app": "houdini.exe", "vram_mib": 4962}


def _choice(label="render", probs=None, conf=0.8):
    probs = probs if probs is not None else {"render": 0.8, "sim": 0.15, "none": 0.05}
    return SimpleNamespace(choice=label, confidence=conf, probabilities=probs)


def _resp(model=jev.PINNED_MODEL, choices=None, nouls=None, scores=None, rid="req-1"):
    return SimpleNamespace(model=model, request_id=rid, choices=choices or {},
                           nouls=nouls or {}, scores=scores or {})


class _JevCase(unittest.TestCase):
    """Jev ON, a dummy key, the fake SDK importable. Individual tests take away
    whichever piece they are about."""

    env = {jev.SWITCH_ENV: "on", jev.KEY_ENV: DUMMY_KEY}

    def setUp(self) -> None:
        jev._reset()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, self.env))
        stack.enter_context(mock.patch.object(jev, "_import_sdk", return_value=_FakeSDK))
        stack.enter_context(mock.patch.object(jev, "_sdk_importable", return_value=True))
        self.addCleanup(jev._reset)

    def ask(self, client, questions=(Q_KIND,), state=STATE, **kw):
        return jev.judge("observe", "proc|houdini.exe", state, list(questions),
                         client=client, **kw)


# ---------------------------------------------------------------------------
# consent and availability
# ---------------------------------------------------------------------------
class TestOffUnlessSwitchedOn(_JevCase):
    def test_a_key_alone_is_not_consent(self) -> None:
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        with mock.patch.dict(os.environ, {jev.KEY_ENV: DUMMY_KEY}, clear=False):
            os.environ.pop(jev.SWITCH_ENV, None)
            self.assertIsNone(self.ask(client))
            ok, why = jev.available()
        self.assertFalse(ok)
        self.assertIn("switch off", why)
        self.assertEqual(client.calls, [])

    def test_only_the_word_on_enables_it(self) -> None:
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        for val in ("1", "true", "yes", "enabled", ""):
            with mock.patch.dict(os.environ, {jev.SWITCH_ENV: val}):
                self.assertIsNone(self.ask(client), val)
        self.assertEqual(client.calls, [])

    def test_no_sdk_in_this_interpreter_is_offline_not_an_error(self) -> None:
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        with mock.patch.object(jev, "_sdk_importable", return_value=False):
            ok, why = jev.available()
            out = self.ask(client)
        self.assertFalse(ok)
        self.assertIn("not importable", why)
        self.assertEqual([j.outcome for j in out], ["offline"])
        self.assertIsNone(out[0].label)
        self.assertEqual(client.calls, [])

    def test_no_key_is_offline_and_the_reason_never_holds_a_key(self) -> None:
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        with mock.patch.dict(os.environ, {jev.KEY_ENV: ""}):
            ok, why = jev.available()
            out = self.ask(client)
        self.assertFalse(ok)
        self.assertIn("no API key", why)
        self.assertEqual(out[0].outcome, "offline")
        self.assertEqual(client.calls, [])

    def test_the_key_never_appears_in_any_reason_or_status_line(self) -> None:
        texts = [jev.status_line(), jev.available()[1]]
        with mock.patch.object(jev, "_sdk_importable", return_value=False):
            texts += [jev.status_line(), jev.available()[1]]
        for t in texts:
            self.assertNotIn(DUMMY_KEY, t)


class TestStatusLine(_JevCase):
    def test_off(self) -> None:
        with mock.patch.dict(os.environ, {jev.SWITCH_ENV: "off"}):
            self.assertEqual(jev.status_line(), "jev: off (set V12_JEV=on to enable)")

    def test_on(self) -> None:
        self.assertEqual(jev.status_line(), "jev: on (model jev-1.13.0)")

    def test_unavailable_names_the_reason(self) -> None:
        with mock.patch.dict(os.environ, {jev.KEY_ENV: ""}):
            self.assertEqual(jev.status_line(),
                             "jev: unavailable (no API key (TYPESAFE_API_KEY unset))")

    def test_the_line_is_sanitized(self) -> None:
        with mock.patch.object(jev, "_sdk_importable", return_value=False), \
                mock.patch.object(jev.sys, "executable", "C:\\py\x1b[2K\rpython.exe"):
            line = jev.status_line()
        self.assertNotIn("\x1b", line)
        self.assertNotIn("\r", line)
        self.assertTrue(line.startswith("jev: unavailable ("))

    def _cli_out(self, fn, args) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(args)
        return buf.getvalue()

    def test_status_prints_the_jev_line(self) -> None:
        with mock.patch("v12.plane.status", return_value="MACHINE"):
            out = self._cli_out(cli.cmd_status, argparse.Namespace())
        self.assertIn("jev: on (model jev-1.13.0)", out.splitlines())

    def test_status_prints_off_by_default(self) -> None:
        with mock.patch("v12.plane.status", return_value="MACHINE"), \
                mock.patch.dict(os.environ, {jev.SWITCH_ENV: ""}):
            out = self._cli_out(cli.cmd_status, argparse.Namespace())
        self.assertIn("jev: off (set V12_JEV=on to enable)", out.splitlines())

    def test_preflight_prints_the_jev_line_and_it_does_not_change_the_verdict(self) -> None:
        # The table is stubbed too: cmd_preflight reads it itself, and a live read
        # is a PowerShell Win32_Process query, which no test may make.
        with mock.patch("v12.guard.preflight", return_value=(False, [], ["busy"])), \
                mock.patch("v12.guard._process_table", return_value=[]), \
                mock.patch("subprocess.run",
                           side_effect=AssertionError("a subprocess was run")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.cmd_preflight(argparse.Namespace(quiet=False))
        self.assertEqual(rc, 1)
        self.assertIn("BLOCKED", buf.getvalue())
        self.assertIn("  jev: on (model jev-1.13.0)", buf.getvalue().splitlines())


# ---------------------------------------------------------------------------
# the call and its validation
# ---------------------------------------------------------------------------
class TestJudge(_JevCase):
    def test_an_ok_choice(self) -> None:
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        (j,) = self.ask(client)
        self.assertEqual(j.outcome, "ok")
        self.assertEqual(j.label, "render")
        self.assertEqual(dict(j.probabilities), {"render": 0.8, "sim": 0.15, "none": 0.05})
        self.assertEqual(j.confidence_or_noul, 0.8)
        self.assertEqual((j.model, j.request_id, j.consumed_as), (jev.PINNED_MODEL,
                                                                  "req-1", "display"))
        self.assertEqual(j.subject_key, "proc|houdini.exe")
        self.assertEqual(len(j.state_sha256), 64)
        self.assertEqual(j.display(), "jev: render (0.80)")
        # The no-match label is part of what was asked.
        sent = client.calls[0][1]["workload"]
        self.assertIn("none", sent.kw["criteria"])

    def test_the_pinned_client_settings(self) -> None:
        made = {}

        class _Client(_FakeClient):
            def __init__(self, **kw):
                made.update(kw)
                super().__init__(_resp(choices={"workload": _choice()}))

        with mock.patch.object(_FakeSDK, "TypeSafeClient", _Client):
            (j,) = self.ask(client=None)
        self.assertEqual(j.outcome, "ok")
        self.assertEqual(made["model"], "jev-1.13.0")
        self.assertEqual(made["timeout"], 3.0)
        self.assertEqual(made["retry"].max_retries, 0)
        self.assertNotIn("api_key", made)

    def test_wrong_model_is_rejected(self) -> None:
        client = _FakeClient(_resp(model="jev-2.0.0", choices={"workload": _choice()}))
        (j,) = self.ask(client)
        self.assertEqual(j.outcome, "rejected")
        self.assertIsNone(j.label)
        self.assertIn("not pinned", j.offline_reason)

    def test_a_model_string_with_escapes_is_rejected_and_never_echoed_raw(self) -> None:
        client = _FakeClient(_resp(model="jev-1.13.0\x1b[2K", choices={"workload": _choice()}))
        (j,) = self.ask(client)
        self.assertEqual(j.outcome, "rejected")
        self.assertNotIn("\x1b", j.offline_reason + j.model + j.display())

    def test_a_label_outside_the_criteria_is_rejected(self) -> None:
        bad = _choice(label="measured", probs={"render": 0.5, "sim": 0.3, "none": 0.2})
        (j,) = self.ask(_FakeClient(_resp(choices={"workload": bad})))
        self.assertEqual(j.outcome, "rejected")
        self.assertIsNone(j.label)

    def test_probabilities_for_an_unknown_label_are_rejected(self) -> None:
        bad = _choice(probs={"render": 0.8, "other": 0.2})
        (j,) = self.ask(_FakeClient(_resp(choices={"workload": bad})))
        self.assertEqual(j.outcome, "rejected")

    def test_probabilities_that_do_not_sum_to_one_are_rejected(self) -> None:
        for probs in ({"render": 0.5, "sim": 0.2, "none": 0.05},
                      {"render": 0.9, "sim": 0.15, "none": 0.05}):
            jev._reset()
            (j,) = self.ask(_FakeClient(_resp(choices={"workload": _choice(probs=probs)})))
            self.assertEqual(j.outcome, "rejected", probs)
            self.assertIn("sum", j.offline_reason)

    def test_a_sum_within_tolerance_is_accepted(self) -> None:
        probs = {"render": 0.8, "sim": 0.15, "none": 0.055}
        (j,) = self.ask(_FakeClient(_resp(choices={"workload": _choice(probs=probs)})))
        self.assertEqual(j.outcome, "ok")

    def test_non_finite_values_are_rejected(self) -> None:
        cases = [_choice(probs={"render": math.nan, "sim": 0.5, "none": 0.5}),
                 _choice(probs={"render": math.inf, "sim": 0.0, "none": 0.0}),
                 _choice(conf=math.nan), _choice(conf=1.5)]
        for c in cases:
            jev._reset()
            (j,) = self.ask(_FakeClient(_resp(choices={"workload": c})))
            self.assertEqual(j.outcome, "rejected")

    def test_a_missing_answer_is_rejected(self) -> None:
        (j,) = self.ask(_FakeClient(_resp(choices={})))
        self.assertEqual(j.outcome, "rejected")

    def test_noul(self) -> None:
        q = Question.noul("render_bound", "`top_gpu_app` is rendering.")
        (j,) = self.ask(_FakeClient(_resp(nouls={"render_bound": SimpleNamespace(noul=0.7)})),
                        questions=[q])
        self.assertEqual((j.outcome, j.label, j.confidence_or_noul), ("ok", "yes", 0.7))
        self.assertAlmostEqual(sum(p for _, p in j.probabilities), 1.0)
        jev._reset()
        (j,) = self.ask(_FakeClient(_resp(nouls={"render_bound": SimpleNamespace(noul=1.4)})),
                        questions=[q])
        self.assertEqual(j.outcome, "rejected")

    def test_score(self) -> None:
        q = Question.score("contention", "How contended is the GPU in `state`?",
                           ["idle", "shared", "saturated"])
        good = SimpleNamespace(score=1.2, confidence=0.6,
                               legend={0: "idle", 1: "shared", 2: "saturated"},
                               probabilities={0: 0.1, 1: 0.6, 2: 0.3})
        (j,) = self.ask(_FakeClient(_resp(scores={"contention": good})), questions=[q])
        self.assertEqual((j.outcome, j.label), ("ok", "1.20"))
        self.assertEqual(dict(j.probabilities), {"0": 0.1, "1": 0.6, "2": 0.3})
        jev._reset()
        bad = SimpleNamespace(score=1.2, confidence=0.6, legend={0: "a", 1: "b"},
                              probabilities={0: 0.5, 1: 0.5})
        (j,) = self.ask(_FakeClient(_resp(scores={"contention": bad})), questions=[q])
        self.assertEqual(j.outcome, "rejected")

    def test_connection_failure_is_offline_and_the_message_is_not_kept(self) -> None:
        exc = _FakeSDK.TypeSafeAPIConnectionError("https://u:pw@api.example/x failed")
        (j,) = self.ask(_FakeClient(exc=exc))
        self.assertEqual(j.outcome, "offline")
        self.assertEqual(j.offline_reason, "TypeSafeAPIConnectionError")
        (j,) = self.ask(_FakeClient(exc=_FakeSDK.TypeSafeAPITimeoutError("t")))
        self.assertEqual(j.outcome, "offline")

    def test_other_failures_are_error_and_never_raise(self) -> None:
        (j,) = self.ask(_FakeClient(exc=_FakeSDK.TypeSafeRateLimitError("429")))
        self.assertEqual(j.outcome, "error")
        (j,) = self.ask(_FakeClient(exc=RuntimeError("boom")))
        self.assertEqual(j.outcome, "error")

        class _Hostile:
            @property
            def model(self):
                raise RuntimeError("hostile")

        class _C(_FakeClient):
            def system_one(self, state, questions):
                return _Hostile()

        jev._reset()
        self.assertIsNone(self.ask(_C()))  # swallowed, reported as nothing

    def test_only_observe_and_decide_may_ask(self) -> None:
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        with self.assertLogs("v12.jev", "WARNING"):
            for stage in ("admit", "act", "remember", "model", ""):
                self.assertIsNone(jev.judge(stage, "k", STATE, [Q_KIND], client=client))
            self.assertIsNone(jev.judge("decide", "k", STATE, [Q_KIND], client=client,
                                        consumed_as="evidence"))
        self.assertEqual(client.calls, [])
        out = jev.judge("decide", "k", STATE, [Q_KIND], client=client,
                        consumed_as="ordering")
        self.assertEqual(out[0].consumed_as, "ordering")


class TestCap(_JevCase):
    def _qs(self, n):
        return [Question.choice(f"q{i}", "x", CRITERIA) for i in range(n)]

    def test_over_the_cap_is_unlabelled_and_never_sent(self) -> None:
        qs = self._qs(5)
        resp = _resp(choices={q.qid: _choice() for q in qs})
        client = _FakeClient(resp)
        with mock.patch.object(jev, "MAX_QUESTIONS_PER_PROCESS", 3):
            out = self.ask(client, questions=qs)
            self.assertEqual([j.outcome for j in out], ["ok"] * 3 + ["cap"] * 2)
            self.assertEqual(sorted(client.calls[0][1]), ["q0", "q1", "q2"])
            self.assertEqual(out[4].display(), "jev: unlabelled (cap)")
            # The cap is per process: a later call sends nothing new.
            more = self.ask(client, questions=self._qs(5)[3:], state={"other": 1})
            self.assertEqual([j.outcome for j in more], ["cap", "cap"])
        self.assertEqual(len(client.calls), 1)

    def test_the_default_cap_is_a_code_constant(self) -> None:
        self.assertEqual(jev.MAX_QUESTIONS_PER_PROCESS, 40)


class TestCache(_JevCase):
    def test_same_state_same_question_same_model_is_not_resent(self) -> None:
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        (a,) = self.ask(client)
        (b,) = jev.judge("decide", "other|subject", dict(STATE), [Q_KIND], client=client)
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(a.cache_hit)
        self.assertTrue(b.cache_hit)
        self.assertEqual((b.label, b.subject_key), ("render", "other|subject"))
        self.ask(client, state={**STATE, "vram_mib": 1})
        self.assertEqual(len(client.calls), 2)

    def test_failures_are_not_cached(self) -> None:
        client = _FakeClient(exc=_FakeSDK.TypeSafeAPIConnectionError("x"))
        self.ask(client)
        client.exc, client.response = None, _resp(choices={"workload": _choice()})
        (j,) = self.ask(client)
        self.assertEqual((j.outcome, j.cache_hit), ("ok", False))

    def test_the_key_is_the_redacted_state(self) -> None:
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        self.ask(client, state={"cmd": "set COMFY_API_KEY=abc"})
        (j,) = self.ask(client, state={"cmd": "set COMFY_API_KEY=xyz"})
        self.assertTrue(j.cache_hit)
        self.assertEqual(j.state_sha256, jev.state_sha256(jev.redact({"cmd": "set COMFY_API_KEY=x"})))


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------
class TestRedact(unittest.TestCase):
    def test_inline_assignment(self) -> None:
        out = jev.redact("set COMFY_API_KEY=abc && python main.py")
        self.assertNotIn("abc", out)
        self.assertIn("COMFY_API_KEY=", out)
        self.assertIn("python main.py", out)

    def test_argv_flag_with_a_space(self) -> None:
        out = jev.redact("python main.py --api-key x --listen 0.0.0.0")
        self.assertNotIn(" x ", out)
        self.assertIn("--api-key [redacted]", out)
        self.assertIn("--listen 0.0.0.0", out)

    def test_argv_flag_with_equals(self) -> None:
        self.assertNotIn("s3cr3t", jev.redact("tool --token=s3cr3t"))

    def test_argv_as_a_list(self) -> None:
        out = jev.redact(["python", "main.py", "--api-key", "x", "--port", "8188"])
        self.assertEqual(out, ["python", "main.py", "--api-key", "[redacted]",
                               "--port", "8188"])

    def test_quoted_json_keys(self) -> None:
        """The closing quote sits between the name and the colon."""
        for text in ('python app.py {"api_key": "abc123"}',
                     'python app.py {"apiKey":"abc123", "x": 1}',
                     'python app.py "{\\"api_key\\": \\"abc123\\"}"'):
            with self.subTest(text=text):
                out = jev.redact(text)
                self.assertNotIn("abc123", out)
                self.assertIn("[redacted]", out)

    def test_a_short_bearer_token(self) -> None:
        for flag in ("--header", "-H"):
            with self.subTest(flag=flag):
                out = jev.redact(f'python app.py {flag} "Authorization: Bearer abc123xyz"')
                self.assertNotIn("abc123xyz", out)
                self.assertIn(flag, out)

    def test_short_password_flags(self) -> None:
        for text in ("python -m yt_dlp -u joe -p hunter2 https://x.com/v",
                     "x --pass hunter2", "x --pw hunter2", "x --pwd hunter2",
                     "x --pass=hunter2", "x -P hunter2"):
            with self.subTest(text=text):
                self.assertNotIn("hunter2", jev.redact(text))
        self.assertIn("https://x.com/v",
                      jev.redact("python -m yt_dlp -u joe -p hunter2 https://x.com/v"))
        self.assertEqual(jev.redact(["python", "s.py", "-p", "hunter2", "--port", "8188"]),
                         ["python", "s.py", "-p", "[redacted]", "--port", "8188"])
        self.assertEqual(jev.redact(["python", "s.py", "--pw", "hunter2"]),
                         ["python", "s.py", "--pw", "[redacted]"])

    def test_url_userinfo(self) -> None:
        out = jev.redact("fetch https://user:pw@host.example/path?q=1")
        self.assertNotIn("user:", out)
        self.assertNotIn("pw@", out)
        self.assertIn("host.example/path", out)

    def test_secret_looking_keys_lose_their_values(self) -> None:
        out = jev.redact({"password": "hunter2", "Auth_Header": "Bearer z",
                          "license_server": "lic.local", "name": "houdini.exe",
                          "nested": {"api_token": "t"}})
        self.assertEqual(out["password"], "[redacted]")
        self.assertEqual(out["Auth_Header"], "[redacted]")
        self.assertEqual(out["license_server"], "[redacted]")
        self.assertEqual(out["nested"]["api_token"], "[redacted]")
        self.assertEqual(out["name"], "houdini.exe")

    def test_long_base64_like_runs(self) -> None:
        blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkw"
        out = jev.redact(f"header {blob} end")
        self.assertNotIn(blob, out)
        self.assertEqual(jev.redact("short abc123 ok"), "short abc123 ok")
        name = "HOUDINI_PDG_MAX_CONCURRENT_COOKS=24"
        self.assertEqual(jev.redact(name), name)
        sha = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        self.assertNotIn(sha, jev.redact({"hash": sha})["hash"])

    def test_the_profile_path(self) -> None:
        with mock.patch.dict(os.environ, {"USERPROFILE": r"C:\Users\Joe"}):
            out = jev.redact({"cwd": r"C:\Users\Joe\Documents\scene.hip",
                              "other": r"c:/users/someone/x.txt"})
        self.assertEqual(out["cwd"], r"%USERPROFILE%\Documents\scene.hip")
        self.assertEqual(out["other"], "%USERPROFILE%/x.txt")

    def test_ordinary_state_is_untouched(self) -> None:
        state = {"top_gpu_app": "houdini.exe", "vram_mib": 4962, "busy": True,
                 "apps": ["dwm.exe", "houdini.exe"], "clock": None}
        self.assertEqual(jev.redact(state), state)

    def test_what_is_sent_is_the_redacted_state(self) -> None:
        jev._reset()
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        with mock.patch.dict(os.environ, {jev.SWITCH_ENV: "on", jev.KEY_ENV: DUMMY_KEY}), \
                mock.patch.object(jev, "_import_sdk", return_value=_FakeSDK), \
                mock.patch.object(jev, "_sdk_importable", return_value=True):
            jev.judge("observe", "k", {"cmd": "comfy --api-key x", "token": "t"},
                      [Q_KIND], client=client)
        jev._reset()
        sent = repr(client.calls[0][0])
        self.assertNotIn("'t'", sent)
        self.assertNotIn(" x", sent)
        self.assertNotIn(DUMMY_KEY, sent)


# ---------------------------------------------------------------------------
# terminal safety
# ---------------------------------------------------------------------------
class TestSanitize(unittest.TestCase):
    def test_a_forged_preflight_line(self) -> None:
        out = jev.sanitize_for_terminal("\x1b[1A\x1b[2K preflight: CLEAR")
        self.assertEqual(out, " preflight: CLEAR")
        self.assertNotIn("[1A", out)

    def test_every_control_character_goes(self) -> None:
        raw = "a\rb\x08c\nd\x00e\x7ff\x9b2Kg\x85h\x1b]0;title\x07i\u202ej\tk"
        out = jev.sanitize_for_terminal(raw)
        self.assertEqual(out, "abcdefghij k")
        self.assertTrue(all(ord(ch) >= 0x20 and not 0x7f <= ord(ch) <= 0x9f for ch in out))

    def test_none_and_non_strings(self) -> None:
        self.assertEqual(jev.sanitize_for_terminal(None), "")
        self.assertEqual(jev.sanitize_for_terminal(3.5), "3.5")


# ---------------------------------------------------------------------------
# a judgment is not evidence
# ---------------------------------------------------------------------------
class TestNotEvidence(unittest.TestCase):
    def test_no_field_is_named_or_typed_as_evidence(self) -> None:
        for f in dataclasses.fields(Judgment):
            self.assertNotIn("evidence", f.name.lower())
            self.assertNotIn("EvidenceClass", str(f.type))
            self.assertNotIn("Verdict", str(f.type))

    def test_reserved_labels_cover_every_evidence_class(self) -> None:
        for e in EvidenceClass:
            self.assertIn(e.value.lower(), jev.RESERVED_LABELS)

    def test_a_question_cannot_offer_an_evidence_label(self) -> None:
        for lab in ("measured", "Derived", " CITED ", "guided"):
            with self.assertRaises(ValueError):
                Question.choice("q", "x", {lab: "y", "none": "None of these"})

    def test_a_choice_must_offer_no_match(self) -> None:
        with self.assertRaises(ValueError):
            Question.choice("q", "x", {"a": "1", "b": "2"})

    def test_a_judgment_cannot_carry_an_evidence_label_or_use(self) -> None:
        base = dict(question_id="q", subject_key="k", probabilities=(), model="m",
                    request_id="", state_sha256="0" * 64, confidence_or_noul=None)
        with self.assertRaises(ValueError):
            Judgment(label="measured", outcome="ok", consumed_as="display", **base)
        with self.assertRaises(ValueError):
            Judgment(label="x", outcome="ok", consumed_as="admission", **base)
        with self.assertRaises(ValueError):
            Judgment(label="x", outcome="offline", consumed_as="display", **base)

    def test_it_is_frozen(self) -> None:
        j = Judgment("q", "k", None, (), None, "", "", "cap", "display", "0" * 64)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            j.label = "x"  # type: ignore[misc]


class TestLedgerIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="v12jev_"))
        self.lg = Ledger(db_path=self.dir / "ledger.db", session="jev-session")

    def tearDown(self) -> None:
        self.lg.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _populate(self) -> None:
        c = make(ChangeKind.ENV_VAR, "env_x", "t", "X", target_value="1", tier=Tier.SYSTEM)
        ok = Verdict(admitted=True, change_key=change_key(c), change_id=c.id,
                     reason="t", evidence=EvidenceClass.MEASURED)
        no = Verdict(admitted=False, change_key=change_key(c), change_id=c.id,
                     reason="no baseline", evidence=EvidenceClass.CITED)
        self.lg.record_verdict(ok, c)
        self.lg.record_verdict(no, c)
        self.lg.record_applied(c, ok, landed=True, effect=0.05)

    def _judgments(self):
        sha = "0" * 64
        return [
            Judgment("workload", "proc|houdini.exe", "render", (("render", 1.0),), 0.9,
                     jev.PINNED_MODEL, "r1", "ok", "display", sha),
            Judgment("workload", "proc|x", None, (), None, "", "", "offline", "display",
                     sha, offline_reason="no API key"),
            Judgment("order", "k", None, (), None, "", "", "cap", "ordering", sha),
        ]

    def test_recording_judgments_leaves_the_honesty_metric_unchanged(self) -> None:
        self._populate()
        before = (self.lg.stats(), self.lg.honesty_report(),
                  self.lg.stats(session_only=True), self.lg.honesty_report(True))
        for j in self._judgments():
            self.lg.record_judgment(j, stage="observe")
        after = (self.lg.stats(), self.lg.honesty_report(),
                 self.lg.stats(session_only=True), self.lg.honesty_report(True))
        self.assertEqual(before, after)
        s = self.lg.judgments_summary()
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["by_outcome"], {"ok": 1, "offline": 1, "cap": 1})

    def test_the_honesty_code_never_reads_the_judgments_table(self) -> None:
        for fn in (Ledger.stats, Ledger.honesty_report):
            self.assertNotIn("judgments", inspect.getsource(fn))

    def test_the_ledger_refuses_rows_that_would_read_as_evidence(self) -> None:
        row = SimpleNamespace(question_id="q", subject_key="k", state_sha256="",
                              model="", request_id="", label="x", probabilities=(),
                              confidence_or_noul=None, outcome="ok")
        with self.assertRaises(ValueError):
            self.lg.record_judgment(SimpleNamespace(**vars(row), consumed_as="admission"),
                                    stage="observe")
        with self.assertRaises(ValueError):
            self.lg.record_judgment(SimpleNamespace(**{**vars(row), "label": "Measured"},
                                                    consumed_as="display"),
                                    stage="observe")
        self.assertEqual(self.lg.judgments_summary()["total"], 0)

    def test_the_table_is_additive_on_an_existing_ledger(self) -> None:
        self._populate()
        self.lg.close()
        again = Ledger(db_path=self.dir / "ledger.db", session="jev-session")
        try:
            self.assertEqual(again.stats().applied_total, 1)
            self.assertEqual(again.judgments_summary()["total"], 0)
        finally:
            again.close()
        self.lg = Ledger(db_path=self.dir / "ledger.db", session="jev-session")

    def test_judge_records_through_the_ledger(self) -> None:
        jev._reset()
        self.addCleanup(jev._reset)
        client = _FakeClient(_resp(choices={"workload": _choice()}))
        with mock.patch.dict(os.environ, {jev.SWITCH_ENV: "on", jev.KEY_ENV: DUMMY_KEY}), \
                mock.patch.object(jev, "_import_sdk", return_value=_FakeSDK), \
                mock.patch.object(jev, "_sdk_importable", return_value=True):
            jev.judge("decide", "k", STATE, [Q_KIND], ledger=self.lg, client=client)
        rows = self.lg.judgments_summary()["recent"]
        self.assertEqual([(r["stage"], r["label"], r["outcome"]) for r in rows],
                         [("decide", "render", "ok")])

    def test_the_ledger_cli_prints_them_in_their_own_section_sanitized(self) -> None:
        self._populate()
        for j in self._judgments():
            self.lg.record_judgment(j, stage="observe")
        self.lg.record_judgment(
            Judgment("workload", "\x1b[1A\x1b[2K preflight: CLEAR", None, (), None, "", "",
                     "rejected", "display", "0" * 64), stage="observe")

        @contextlib.contextmanager
        def fake_open(*a, **k):
            yield self.lg

        buf = io.StringIO()
        with mock.patch.object(cli, "open_ledger", fake_open), contextlib.redirect_stdout(buf):
            cli.cmd_ledger(argparse.Namespace(session=False, limit=25))
        out = buf.getvalue()
        self.assertNotIn("\x1b", out)
        head, sep, tail = out.partition("Jev annotations - not evidence")
        self.assertTrue(sep)
        self.assertIn("Applied 1", head)          # the honesty metric comes first
        self.assertNotIn("render", head)
        self.assertIn("render", tail)
        self.assertIn("unlabelled (cap)", tail)

    def test_no_section_when_there_are_no_judgments(self) -> None:
        self._populate()

        @contextlib.contextmanager
        def fake_open(*a, **k):
            yield self.lg

        buf = io.StringIO()
        with mock.patch.object(cli, "open_ledger", fake_open), contextlib.redirect_stdout(buf):
            cli.cmd_ledger(argparse.Namespace(session=False, limit=25))
        self.assertNotIn("Jev", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
