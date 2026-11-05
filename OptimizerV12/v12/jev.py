"""
V12 — Jev, the one place a judgment model is allowed in.

Jev (TypeSafe's System One model) turns a small JSON state plus a labelled
question into a typed judgment with probabilities. That is useful for telling an
operator what a process table or a decision list LOOKS like. It is not useful as
evidence, and this module exists to keep the two apart.

THE INVARIANT. A Jev judgment is never evidence. Nothing returned from here may
reach Gate.evaluate, a Verdict, a Baseline, Conditions, an Objective, the
control-flow fields of an Observation, a Change, or any actuator. A judgment is
a display annotation or an ordering hint, recorded in its own ledger table when
a ledger is given (a cycle's OBSERVE; `v12 preflight`/`status` labels and the
relevance-drafting tool record none), and never counted in the honesty metric. The reason is the same one CITED is never
admitted: a model saying "this looks like a render workload" has timed nothing on
this machine. A Choice's confidence measures how concentrated the distribution
is, not whether the answer is right.

How the invariant is held, not just stated:
  * this is the ONLY module that names `typesafe_sdk`, and it imports it lazily
    inside functions — tests/test_invariants.py asserts both, by AST and by
    importing the gate's modules in a fresh interpreter;
  * Judgment has no evidence field, and a question whose labels are evidence
    class names is refused at construction, so a judgment cannot be MISREAD as
    a verdict in a report or a ledger row;
  * OFF unless V12_JEV=on. An API key being present is not consent — the key is
    there for other tools too.

Every failure is quiet and typed: judge() never raises. Off returns None; an
unreachable SDK or network gives outcome 'offline'; a response that fails
validation gives 'rejected'; anything else 'error'; over the per-process cap
'cap'. A caller that ignores the outcome still shows nothing, because only an
'ok' judgment carries a label.

State is redacted BEFORE it is hashed or sent: values under secret-looking keys,
secret-looking argv flags and NAME=value pairs, URL userinfo, long base64-like
runs, and the user's profile path. Anything printed goes through
sanitize_for_terminal(), because a label is text from a network service and a
terminal will execute the escape codes in it.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

log = logging.getLogger("v12.jev")

#: The opt-in switch. Only the exact value 'on' (any case) enables Jev.
SWITCH_ENV = "V12_JEV"
#: Read by the SDK itself; this module only checks it is non-empty. Never printed.
KEY_ENV = "TYPESAFE_API_KEY"

#: Pinned. A response from any other model is rejected, not reinterpreted.
PINNED_MODEL = "jev-1.13.0"
TIMEOUT_S = 3.0
MAX_RETRIES = 0
#: Hard cap on questions SENT per process invocation (cache hits are free).
#: One `v12 run` should never turn into a meter running in the background.
MAX_QUESTIONS_PER_PROCESS = 40
#: Probabilities must sum to 1 within this.
PROB_SUM_TOL = 0.01

#: Stages Jev may annotate. ADMIT / ACT / REMEMBER are deliberately absent.
STAGES = frozenset({"observe", "decide"})
#: What a judgment may be consumed as. There is no third use.
CONSUMED_AS = frozenset({"display", "ordering"})
OUTCOMES = ("ok", "offline", "rejected", "error", "cap")

#: Evidence-class names (and the verdict words next to them). A Jev label that
#: reads like one of these would print like a verdict, so it is refused at
#: construction. Kept as literals so this module does not import the gate's
#: types; tests/test_jev.py checks it still covers EvidenceClass.
RESERVED_LABELS = frozenset({"measured", "derived", "cited", "guided",
                             "admitted", "denied", "verdict", "evidence"})


# ---------------------------------------------------------------------------
# terminal safety
# ---------------------------------------------------------------------------
_CSI = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]")
_OSC = re.compile(r"(?:\x1b\]|\x9d).*?(?:\x07|\x1b\\|\x9c)", re.S)
_ESC_OTHER = re.compile(r"\x1b[@-_]?")
_CONTROL = re.compile(r"[\x00-\x08\x0a-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def sanitize_for_terminal(text: Any) -> str:
    """One printable line. Removes whole CSI/OSC escape sequences first (so
    '\\x1b[2K' leaves nothing rather than '[2K'), then every remaining C0/C1
    control character — ESC, CR, LF, BS, DEL — and the bidi overrides that can
    reorder what a reader sees. Tab becomes a space."""
    s = "" if text is None else str(text)
    s = _OSC.sub("", s)
    s = _CSI.sub("", s)
    s = _ESC_OTHER.sub("", s)
    s = s.replace("\t", " ")
    return _CONTROL.sub("", s)


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------
_SECRET_WORD = r"(?:key|token|secret|password|passwd|auth|cred|license)"
SECRET_KEY = re.compile(rf"(?i){_SECRET_WORD}")
REDACTED = "[redacted]"

#: Flags only: the secret words plus the short password spellings tools use
#: (--pass, --pw, --pwd). Not used for dict keys or NAME=value, where 'pass'
#: would mask every 'passes: 3'. A bare -p / -P value is masked too: yt-dlp's
#: password flag is -p, and a port masked by mistake is the closed side.
_FLAG_WORD = r"(?:key|token|secret|password|passwd|auth|cred|license|pass|pwd|pw)"
SECRET_FLAG = re.compile(rf"(?i)^(?:--?[\w.-]*{_FLAG_WORD}[\w.-]*|-[pP])$")

# --api-key x  /  -token x  /  --pass=x  /  -p x
_ARGV_SPACE = re.compile(
    rf"(?i)(?<![\w-])(--?[\w.-]*{_FLAG_WORD}[\w.-]*|-[pP])(\s+|=)(?!-)(\"[^\"]*\"|'[^']*'|\S+)")
# Authorization: Bearer <token> — the token is the second word, so _INLINE alone
# would mask only the scheme. Runs before _INLINE.
_AUTH_SCHEME = re.compile(
    r"(?i)\b(bearer|basic|digest|token|apikey)(\s+)(?!\[redacted\])([^\s\"',;&|\\]+)")
# set COMFY_API_KEY=abc  /  --api-key=abc  /  password: hunter2  /  "api_key": "abc"
# and the escaped-quote argv form {\"api_key\": \"abc\"}: an optional closing
# quote (escaped or not) may sit between the name and the separator.
_INLINE = re.compile(
    rf"(?i)((?<![\w-])-{{0,2}}[\w.-]*{_SECRET_WORD}[\w.-]*)(\\?[\"']?\s*[=:]\s*)"
    r"(\\\"(?:[^\"\\]|\\(?!\"))*\\\"|\"[^\"]*\"|'[^']*'|[^\s\"',;&|\\]+)")
# scheme://user:pw@host
_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@")
# long base64 / base64url / hex runs: tokens, hashes, keys. A run must mix
# letters and digits, so a long env var name (HOUDINI_PDG_MAX_CONCURRENT_COOKS)
# is not mistaken for a secret.
_B64 = r"[A-Za-z0-9+/_-]"
_BLOB = re.compile(rf"(?<!{_B64})(?={_B64}*\d)(?={_B64}*[A-Za-z]){_B64}{{32,}}={{0,2}}")
_PROFILE = re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[^\\/\s\"']+")


def _redact_str(s: str) -> str:
    s = _USERINFO.sub(lambda m: m.group(1) + REDACTED + "@", s)
    s = _AUTH_SCHEME.sub(lambda m: m.group(1) + m.group(2) + REDACTED, s)
    s = _ARGV_SPACE.sub(lambda m: m.group(1) + m.group(2) + REDACTED, s)
    s = _INLINE.sub(lambda m: m.group(1) + m.group(2) + REDACTED, s)
    home = os.environ.get("USERPROFILE", "")
    if home:
        s = re.sub(re.escape(home), "%USERPROFILE%", s, flags=re.I)
    s = _PROFILE.sub("%USERPROFILE%", s)
    s = _BLOB.sub("[blob]", s)
    return s


def redact(value: Any) -> Any:
    """A copy of `value` safe to send off this machine. Fails toward removing too
    much: `change_key` loses its value too, because 'key' matches. Subjects are
    passed to judge() as subject_key, never inside the state."""
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            ks = _redact_str(str(k))
            out[ks] = REDACTED if SECRET_KEY.search(str(k)) else redact(v)
        return out
    if isinstance(value, (list, tuple)):
        items = list(value)
        out_list: List[Any] = []
        mask_next = False
        for item in items:
            if mask_next:
                out_list.append(REDACTED)
                mask_next = False
                continue
            # argv as a list: ['--api-key', 'x'] — the secret is the NEXT element.
            if (isinstance(item, str) and item.startswith("-") and "=" not in item
                    and (SECRET_KEY.search(item) or SECRET_FLAG.match(item))):
                mask_next = True
            out_list.append(redact(item))
        return out_list
    if isinstance(value, str):
        return _redact_str(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _redact_str(str(value))


def state_sha256(state: Any) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=True,
                                     default=str).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# questions and judgments
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Question:
    """One question, described without the SDK so callers never import it.

    kind 'choice': criteria = ((label, description), ...) and MUST include a
    no-match label (NO_MATCH). kind 'score': criteria = (level0, level1, ...).
    kind 'noul': criteria unused. Labels that read like evidence are refused."""
    qid: str
    kind: str
    instructions: str
    criteria: Tuple[Any, ...] = ()

    NO_MATCH = "none"

    def __post_init__(self) -> None:
        if not self.qid or not re.fullmatch(r"[A-Za-z0-9_]+", self.qid):
            raise ValueError(f"question id {self.qid!r} must be [A-Za-z0-9_]+")
        if self.kind not in ("choice", "noul", "score"):
            raise ValueError(f"unknown question kind {self.kind!r}")
        if self.kind == "choice":
            labels = [lab for lab, _ in self.criteria]
            if len(set(labels)) != len(labels) or len(labels) < 2:
                raise ValueError(f"{self.qid}: choice needs >= 2 distinct labels")
            if self.NO_MATCH not in labels:
                raise ValueError(f"{self.qid}: choice must include the no-match "
                                 f"label {self.NO_MATCH!r}")
            bad = [lab for lab in labels if str(lab).strip().lower() in RESERVED_LABELS]
            if bad:
                raise ValueError(f"{self.qid}: labels {bad} read like evidence classes; "
                                 f"a judgment must never print like a verdict")
        if self.kind == "score" and len(self.criteria) < 2:
            raise ValueError(f"{self.qid}: score needs >= 2 levels")

    @classmethod
    def choice(cls, qid: str, instructions: str, criteria: Mapping[str, str]) -> "Question":
        return cls(qid, "choice", instructions, tuple(criteria.items()))

    @classmethod
    def noul(cls, qid: str, instructions: str) -> "Question":
        return cls(qid, "noul", instructions, ())

    @classmethod
    def score(cls, qid: str, instructions: str, levels: Sequence[str]) -> "Question":
        return cls(qid, "score", instructions, tuple(levels))

    def labels(self) -> Tuple[str, ...]:
        if self.kind == "choice":
            return tuple(lab for lab, _ in self.criteria)
        if self.kind == "score":
            return tuple(str(i) for i in range(len(self.criteria)))
        return ("yes", "no")

    def spec(self) -> str:
        """Canonical text of the question, for the cache key."""
        return json.dumps([self.qid, self.kind, self.instructions, list(self.criteria)],
                          sort_keys=True, ensure_ascii=True, default=str)


@dataclass(frozen=True)
class Judgment:
    """What Jev said about one subject, and how it may be used.

    Deliberately has no evidence field and no EvidenceClass type anywhere: it is
    an annotation. `label` is None unless outcome is 'ok'. `probabilities` is a
    tuple of (label, p) pairs so the dataclass stays hashable.
    `confidence_or_noul` is a Choice/Score concentration, or a Noul's P(yes) —
    in neither case a probability of being right."""
    question_id: str
    subject_key: str
    label: Optional[str]
    probabilities: Tuple[Tuple[str, float], ...]
    confidence_or_noul: Optional[float]
    model: str
    request_id: str
    outcome: str
    consumed_as: str
    state_sha256: str
    cache_hit: bool = False
    offline_reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}")
        if self.consumed_as not in CONSUMED_AS:
            raise ValueError(f"consumed_as must be one of {sorted(CONSUMED_AS)}")
        if self.outcome != "ok" and self.label is not None:
            raise ValueError("only an 'ok' judgment carries a label")
        if self.label is not None and self.label.strip().lower() in RESERVED_LABELS:
            raise ValueError(f"label {self.label!r} reads like an evidence class")

    def display(self) -> str:
        """The text an operator sees. Never names an evidence class."""
        if self.outcome == "ok":
            conf = ("" if self.confidence_or_noul is None
                    else f" ({self.confidence_or_noul:.2f})")
            text = f"jev: {self.label}{conf}"
        elif self.outcome == "cap":
            text = "jev: unlabelled (cap)"
        else:
            why = f": {self.offline_reason}" if self.offline_reason else ""
            text = f"jev: unlabelled ({self.outcome}{why})"
        return sanitize_for_terminal(text)


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------
def switch_on() -> bool:
    return os.environ.get(SWITCH_ENV, "").strip().lower() == "on"


def _sdk_importable() -> bool:
    # find_spec, not import: `v12 status` must not load the SDK just to ask.
    try:
        return importlib.util.find_spec("typesafe_sdk") is not None
    except (ImportError, ValueError):
        return False


def available() -> Tuple[bool, str]:
    """(usable, reason). Checks consent first, so 'off' is reported as off even
    on an interpreter without the SDK."""
    if not switch_on():
        return False, f"switch off (set {SWITCH_ENV}=on to enable)"
    if not _sdk_importable():
        return False, f"typesafe_sdk not importable in this interpreter ({sys.executable})"
    if not os.environ.get(KEY_ENV, "").strip():
        return False, f"no API key ({KEY_ENV} unset)"
    return True, f"model {PINNED_MODEL}"


def status_line() -> str:
    """The one line `v12 status` and `v12 preflight` print."""
    if not switch_on():
        return sanitize_for_terminal(f"jev: off (set {SWITCH_ENV}=on to enable)")
    ok, why = available()
    if ok:
        return sanitize_for_terminal(f"jev: on (model {PINNED_MODEL})")
    return sanitize_for_terminal(f"jev: unavailable ({why})")


# ---------------------------------------------------------------------------
# the call
# ---------------------------------------------------------------------------
_cache: Dict[Tuple[str, str, str], Judgment] = {}
_sent = 0


def _reset() -> None:
    """Clear the cache and the cap counter. For tests."""
    global _sent
    _cache.clear()
    _sent = 0


def _import_sdk():
    """The SDK module, or None. The ONE import seam; tests replace this."""
    try:
        return importlib.import_module("typesafe_sdk")
    except Exception:  # noqa: BLE001 — any import failure means offline
        return None


def _sdk_question(sdk, q: Question):
    if q.kind == "choice":
        return sdk.Choice(instructions=q.instructions, criteria=dict(q.criteria))
    if q.kind == "score":
        return sdk.Score(instructions=q.instructions, criteria=list(q.criteria))
    return sdk.Noul(instructions=q.instructions)


def _probs_ok(probs: Mapping[str, float], allowed: Iterable[str]) -> Optional[str]:
    allowed = set(allowed)
    if not probs:
        return "no probabilities"
    unknown = set(probs) - allowed
    if unknown:
        return f"probabilities for unknown labels {sorted(unknown)[:3]}"
    vals = list(probs.values())
    if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0.0 <= p <= 1.0
               for p in vals):
        return "probability not finite in [0,1]"
    if abs(sum(vals) - 1.0) > PROB_SUM_TOL:
        return f"probabilities sum to {sum(vals):.4f}"
    return None


def _finite01(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x) and 0.0 <= x <= 1.0


def _validate(q: Question, r: Any) -> Tuple[Optional[str], Tuple[Tuple[str, float], ...],
                                            Optional[float], str]:
    """(label, probabilities, confidence_or_noul, why_rejected). why == '' is ok."""
    try:
        if q.kind == "choice":
            a = r.choices[q.qid]
            label = a.choice
            probs = {str(k): float(v) for k, v in dict(a.probabilities).items()}
            if label not in q.labels():
                return None, (), None, "label outside criteria"
            why = _probs_ok(probs, q.labels())
            if why:
                return None, (), None, why
            if not _finite01(a.confidence):
                return None, (), None, "confidence not finite in [0,1]"
            return label, tuple(sorted(probs.items())), float(a.confidence), ""
        if q.kind == "score":
            a = r.scores[q.qid]
            probs = {str(k): float(v) for k, v in dict(a.probabilities).items()}
            legend = {str(k) for k in dict(a.legend)}
            if legend != set(q.labels()):
                return None, (), None, "score legend does not match the levels asked"
            why = _probs_ok(probs, q.labels())
            if why:
                return None, (), None, why
            score = a.score
            if not (isinstance(score, (int, float)) and math.isfinite(score)
                    and 0.0 <= score <= len(q.criteria) - 1):
                return None, (), None, "score outside the levels asked"
            if not _finite01(a.confidence):
                return None, (), None, "confidence not finite in [0,1]"
            return f"{float(score):.2f}", tuple(sorted(probs.items())), float(a.confidence), ""
        a = r.nouls[q.qid]
        p = a.noul
        if not _finite01(p):
            return None, (), None, "noul not finite in [0,1]"
        p = float(p)
        return ("yes" if p >= 0.5 else "no"), (("no", 1.0 - p), ("yes", p)), p, ""
    except (KeyError, AttributeError, TypeError, ValueError):
        return None, (), None, "answer missing or malformed"


def _unlabelled(q: Question, subject_key: str, sha: str, consumed_as: str,
                outcome: str, reason: str, model: str = "", request_id: str = "") -> Judgment:
    return Judgment(q.qid, subject_key, None, (), None, model, request_id, outcome,
                    consumed_as, sha, offline_reason=sanitize_for_terminal(reason)[:120])


def _offline_kind(exc: BaseException) -> str:
    name = type(exc).__name__
    if any(t in name for t in ("Connection", "Timeout")):
        return "offline"
    return "error"


def judge(stage: str, subject_key: str, state: Mapping[str, Any],
          questions: Sequence[Question], *, consumed_as: str = "display",
          ledger: Any = None, client: Any = None) -> Optional[List[Judgment]]:
    """Ask Jev about one subject. Returns None when Jev is off or the call is
    malformed (wrong stage, bad consumed_as); otherwise one Judgment per
    question, in order. Never raises.

    `state` is redacted before it is hashed or sent. `ledger`, when given, gets
    every judgment through record_judgment — into its own table, which the
    honesty metric never reads."""
    try:
        return _judge(stage, subject_key, state, questions, consumed_as, ledger, client)
    except Exception:  # noqa: BLE001 — a display annotation must not break a run
        log.debug("jev.judge failed", exc_info=True)
        return None


def _judge(stage, subject_key, state, questions, consumed_as, ledger, client):
    global _sent
    if not switch_on():
        return None
    if stage not in STAGES or consumed_as not in CONSUMED_AS:
        log.warning("jev: refused stage=%r consumed_as=%r", stage, consumed_as)
        return None
    questions = list(questions)
    if not questions or not all(isinstance(q, Question) for q in questions):
        return None

    clean = redact(dict(state))
    sha = state_sha256(clean)
    results: Dict[str, Judgment] = {}
    todo: List[Question] = []
    for q in questions:
        hit = _cache.get((sha, q.spec(), PINNED_MODEL))
        if hit is not None:
            results[q.qid] = replace(hit, subject_key=subject_key, consumed_as=consumed_as,
                                     cache_hit=True)
        else:
            todo.append(q)

    if todo:
        ok, why = available()
        sdk = _import_sdk() if ok else None
        if not ok or sdk is None:
            reason = why if not ok else "typesafe_sdk failed to import"
            for q in todo:
                results[q.qid] = _unlabelled(q, subject_key, sha, consumed_as,
                                             "offline", reason)
            todo = []

    if todo:
        room = max(0, MAX_QUESTIONS_PER_PROCESS - _sent)
        for q in todo[room:]:
            results[q.qid] = _unlabelled(q, subject_key, sha, consumed_as, "cap",
                                         f"{MAX_QUESTIONS_PER_PROCESS} questions per process")
        todo = todo[:room]

    if todo:
        _sent += len(todo)
        try:
            if client is None:
                client = sdk.TypeSafeClient(
                    model=PINNED_MODEL, timeout=TIMEOUT_S,
                    retry=sdk.RetryPolicy(max_retries=MAX_RETRIES))
            r = client.system_one(clean, {q.qid: _sdk_question(sdk, q) for q in todo})
        except Exception as e:  # noqa: BLE001 — classified, never re-raised
            # The class name only: an SDK message can carry a URL or a header.
            kind = _offline_kind(e)
            for q in todo:
                results[q.qid] = _unlabelled(q, subject_key, sha, consumed_as,
                                             kind, type(e).__name__)
        else:
            # Compare the RAW string: sanitizing first would let 'jev-1.13.0[2K'
            # pass as the pinned model. Only the sanitized form is ever stored.
            raw_model = getattr(r, "model", "")
            model = sanitize_for_terminal(raw_model)[:64]
            try:
                rid = sanitize_for_terminal(getattr(r, "request_id", "") or "")[:128]
            except Exception:  # noqa: BLE001 — a missing header is not fatal
                rid = ""
            for q in todo:
                if raw_model != PINNED_MODEL:
                    results[q.qid] = _unlabelled(q, subject_key, sha, consumed_as,
                                                 "rejected", f"model {model!r} is not pinned",
                                                 model, rid)
                    continue
                label, probs, conf, why = _validate(q, r)
                if why:
                    results[q.qid] = _unlabelled(q, subject_key, sha, consumed_as,
                                                 "rejected", why, model, rid)
                    continue
                try:
                    j = Judgment(q.qid, subject_key, label, probs, conf, model, rid,
                                 "ok", consumed_as, sha)
                except ValueError as e:
                    results[q.qid] = _unlabelled(q, subject_key, sha, consumed_as,
                                                 "rejected", str(e), model, rid)
                    continue
                _cache[(sha, q.spec(), PINNED_MODEL)] = j
                results[q.qid] = j

    out = [results[q.qid] for q in questions]
    if ledger is not None:
        for j in out:
            try:
                ledger.record_judgment(j, stage=stage)
            except Exception:  # noqa: BLE001 — recording an annotation is best-effort
                log.debug("jev: record_judgment failed", exc_info=True)
    return out
