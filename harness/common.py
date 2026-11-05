"""Shared plumbing: where measurements go, preflight, and the V12 paste lines."""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
V12_DIR = REPO_ROOT / "OptimizerV12"

#: Measurements live off OneDrive. E:\ by request (1.7 TB free on 2026-11-03).
MEASURE_ROOT = Path(os.environ.get("OPTIMIZER_MEASURE_ROOT", r"E:\OptimizerV12\measurements"))
CONFIG_PATH = Path(os.environ.get("OPTIMIZER_HARNESS_CONFIG", r"E:\OptimizerV12\harness_config.json"))

#: Interpreter that can read the GPU AND has pywin32 (V12's preflight needs win32pdh).
V12_PYTHON = os.environ.get("V12_PYTHON", r"C:\Python314\python.exe")


def ensure_dirs() -> None:
    MEASURE_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def save_config(cfg: Dict) -> None:
    ensure_dirs()
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def preflight(quiet: bool = False) -> bool:
    """Run V12's own preflight. True only when it prints CLEAR (exit 0).
    V12 is the authority on what counts as a competing GPU process; this does
    not re-implement that rule."""
    cmd = [V12_PYTHON, "-m", "v12", "preflight"] + (["--quiet"] if quiet else [])
    try:
        r = subprocess.run(cmd, cwd=str(V12_DIR), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"preflight could not run ({e}); refusing to measure blind")
        return False
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        print("\nharness: preflight BLOCKED — close the competing apps and rerun.")
        return False
    return True


def summarize(xs: Sequence[float]) -> Dict[str, float]:
    xs = list(xs)
    if not xs:
        return {}
    out = {"n": float(len(xs)), "mean": statistics.fmean(xs),
           "median": statistics.median(xs), "min": min(xs), "max": max(xs)}
    if len(xs) > 1:
        sd = statistics.stdev(xs)
        out["stdev"] = sd
        out["cv"] = sd / out["mean"] if out["mean"] else float("nan")
    return out


def series(xs: Sequence[float], nd: int = 4) -> str:
    return ",".join(f"{x:.{nd}f}" for x in xs)


#: The utilisation above which this card counts as BUSY rather than idle.
#:
#: It used to be 10%, which was wrong by an order of magnitude for this
#: machine: the DESKTOP alone drives the card to 4-72% (measured over 113
#: samples on 2026-11-04, median 24%, p95 41%) while drawing only 32-70 W.
#: A 10% threshold therefore declared an idle card busy, and the gate waited
#: for a quiet that never came — on a card doing nothing but compositing a
#: desktop.
#:
#: The number that separates the two states is not utilisation alone but
#: POWER. Measured on this box:
#:
#:     desktop compositing        4-72% util    32-70 W
#:     a Karma render             94%+  util   104 W+
#:     a WSL vLLM benchmark       98%  util   288-300 W
#:
#: 85% clears the desktop's worst burst with room to spare and still catches
#: any real render or compute job, which sit above 90% while they work.
DESKTOP_UTIL_CEILING_PCT = 85.0


def busy_from_readings(util_vals: Sequence[float], vram_used: int, vram_total: int,
                      util_limit_pct: float = DESKTOP_UTIL_CEILING_PCT
                      ) -> Tuple[bool, float, int, int]:
    """The decision, separated from the reading so it can be tested.

    (busy, peak utilisation %, VRAM used MiB, VRAM total MiB)

    UTILISATION is the signal, and it is the only one.

    An earlier version also treated high VRAM occupancy as busy, on the theory
    that a frameserver holding 16 GB could wake up mid-measurement. That
    theory was wrong about the number it was reading: on WDDM, nvidia-smi's
    memory.used counts shared-system-memory allocations, so it reported 17.8 GB
    "used" while the sum of per-process GPU commit was 5.5 GB. A 12.4 GB
    phantom. It blocked a measurement for an hour on a card that was idle.

    VRAM is still REPORTED, because it is useful context, but it does not
    decide. The per-process commit counter (v12.baseline._gpu_commit_by_pid)
    is the honest measure of occupancy, and it is what V12's own preflight
    already reasons about.

    The threshold itself is set by measurement, not by taste — see
    DESKTOP_UTIL_CEILING_PCT. A threshold of 10% was below this desktop's
    own idle burst, so it called an idle card busy forever.

    And the decision that matters is not made here at all: only the drift
    check on the control can say whether a measurement is usable. This avoids
    walking into a contended run; it does not replace the gate.
    """
    vals = [float(v) for v in util_vals]
    peak = max(vals) if vals else 0.0
    return peak > util_limit_pct, peak, int(vram_used), int(vram_total)


def gpu_busy(util_limit_pct: float = DESKTOP_UTIL_CEILING_PCT, samples: int = 3,
             interval: float = 1.0) -> Tuple[bool, float, int, int]:
    """Is somebody ELSE using the GPU right now?

    Returns (busy, peak utilisation %, VRAM used MiB, VRAM total MiB).

    The drift gate already REFUSES a measurement taken on a busy machine —
    during development a stray job drove the baseline +16.3% across a run and
    the verdict came back DENY. That is the correct outcome, but it wastes the
    whole campaign: the operator waits out the measurement and then is told to
    close the thing they did not know was running.

    So this asks the question first. See busy_from_readings for the decision.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return False, 0.0, 0, 0
    util_vals: List[float] = []
    vram_used = 0
    vram_total = 0
    for i in range(max(1, samples)):
        line = out.strip().splitlines()[0].strip() if out.strip() else "0, 0, 0"
        parts = [p.strip() for p in line.split(",")]
        try:
            util_vals.append(float(parts[0]))
        except (ValueError, IndexError):
            util_vals.append(0.0)
        if i == 0:
            try:
                vram_used = int(float(parts[1]))
                vram_total = int(float(parts[2]))
            except (ValueError, IndexError):
                vram_used = vram_total = 0
        if i + 1 < samples:
            time.sleep(interval)
    return busy_from_readings(util_vals, vram_used, vram_total, util_limit_pct)


def wait_for_quiet_gpu(timeout_s: float = 300.0,
                       util_limit_pct: float = DESKTOP_UTIL_CEILING_PCT,
                       poll_s: float = 5.0, verbose: bool = True) -> bool:
    """Wait until the GPU is quiet enough to measure on. True if it went quiet.

    Times out rather than waiting forever: a machine that stays busy is a fact
    about the machine, and the caller should be told rather than left waiting.

    This is the UTILISATION check, and it is only valid at a moment when our
    own workload is NOT holding the card — before a repetition starts. During
    a render the card is busy by design and this says nothing. That is why
    the contention detector is process identity (see other_gpu_processes), and
    why the two are used at different moments rather than one replacing the
    other.

    The threshold is DESKTOP_UTIL_CEILING_PCT — 85%, not the 10% it used to
    be. The desktop alone drives this card to 72% while doing nothing but
    compositing, so a 10% threshold called an idle card busy and the gate
    waited for a quiet that never came.
    """
    deadline = time.time() + max(0.0, timeout_s)
    attempt = 0
    while True:
        attempt += 1
        busy, peak, used, total = gpu_busy(util_limit_pct=util_limit_pct)
        if not busy:
            if attempt > 1 and verbose:
                print(f"  the GPU is quiet now (peak {peak:.0f}% utilisation, "
                      f"{used} MiB of {total} MiB in use)")
            return True
        if time.time() >= deadline:
            if verbose:
                print(f"  still busy after {timeout_s:.0f}s (peak {peak:.0f}% "
                      f"utilisation, {used} MiB of {total} MiB held) — measuring "
                      f"anyway; the drift gate will refuse the result if the "
                      f"machine does not hold still")
            return False
        if verbose:
            print(f"  the GPU is busy (peak {peak:.0f}% utilisation, "
                  f"{used} MiB of {total} MiB held) — waiting for it to settle "
                  f"({max(0, deadline - time.time()):.0f}s left)")
        time.sleep(poll_s)


#: A process holding at least this much GPU memory counts as a competitor.
#: V12 uses the same threshold (v12.baseline.HEAVY_GPU_COMMIT_MIB = 1024 MiB):
#: the desktop's own compositor and apps sit far below it, while a real render
#: or a frameserver sits well above.
HEAVY_GPU_COMMIT_MIB = 1024

#: Always-resident Windows graphics processes. dwm.exe (the desktop window
#: manager) commits ~2.2 GB of GPU memory on this box and csrss.exe ~385 MB —
#: both permanently, and neither competes for the card in any way that would
#: move a render's timing. Counting them would make every check a false
#: positive and the detector worthless.
OS_GRAPHICS_EXEMPT = {
    "dwm.exe", "csrss.exe", "system", "wininit.exe", "sihost.exe",
    "fontdrvhost.exe", "wudfhost.exe", "runtimebroker.exe",
}

#: Applications that ARE render tools. If one of these is resident at all, it
#: is either rendering now or about to, and it competes. V12 keeps the same
#: list (v12.guard.SIGNIFICANT_APPS) for the same reason.
#:
#: "python.exe" is deliberately NOT here. This box runs ~23 stray python
#: processes; naming them would be noise. A python process that actually holds
#: the card is caught by the heavy rule anyway.
RENDER_APP_EXACT = {
    "houdini.exe", "houdinifx.exe", "husk.exe", "hython.exe", "karma.exe",
    "redshift4houdini.exe", "shadercompileworker.exe", "ue5editor.exe",
    "unrealeditor.exe", "unrealeditor-cmd.exe", "comfyui.exe",
}


def _render_app(name: str) -> bool:
    """Is this image a render tool?"""
    low = name.lower()
    if low in RENDER_APP_EXACT:
        return True
    # Nuke's frameserver and Houdini's batch tools both carry a version in
    # their image name ("Nuke14.0.exe", "hbatch.exe").
    return low.startswith("nuke") or low.startswith("hbatch")


def my_process_tree() -> set:
    """This process and every descendant of it.

    Used to tell MY GPU work apart from somebody else's. During a render the
    card is busy by design, so "is the GPU busy" cannot answer the question
    that matters: "is somebody ELSE on the card". Process identity can.

    The descendants matter as much as this process: the renderer (husk) is a
    CHILD of this process, and it is the one holding the GPU.
    """
    mine = {os.getpid()}
    try:
        import psutil
        me = psutil.Process(os.getpid())
        for child in me.children(recursive=True):
            mine.add(child.pid)
    except Exception:
        pass
    return mine


def _wsl_running_distros() -> List[str]:
    """The WSL distributions that are RUNNING right now. Names only.

    Uses `wsl --list --running --quiet`, which never starts a stopped
    distribution — invoking `wsl -d <name>` on a stopped one DOES start it,
    and a detector must never change the state it is measuring.

    The output is UTF-16 on this box, which arrives already decoded by text
    mode, so the ASCII characters are separated by NULs: "U\\x00b\\x00u\\x00"
    is the name "Ubuntu" with its bytes split, not the name "Ub". Decoding it
    as UTF-16-LE is the only correct reading.
    """
    try:
        out = subprocess.run(
            ["wsl.exe", "--list", "--running", "--quiet"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    raw = out.stdout or ""
    if "\x00" in raw:
        try:
            raw = raw.encode("utf-8", "surrogatepass").decode(
                "utf-16-le", errors="replace")
        except Exception:
            raw = raw.replace("\x00", "")
    return [n.strip() for n in raw.replace("\x00", "").splitlines() if n.strip()]


#: The WSL probe spawns a process inside the distribution, which costs ~0.2 s
#: and adds a little load to a machine we are trying to measure. The
#: ContenderWatcher polls every few seconds, so the answer is cached for a
#: moment: long enough to spare the repeated spawns, short enough that a
#: rival which starts is still caught.
_WSL_CACHE_TTL_S = 5.0
_wsl_cache: Dict[str, Any] = {"at": 0.0, "value": ()}


def _wsl_gpu_processes() -> List[str]:
    """GPU processes running inside WSL distributions.

    Found the hard way on 2026-11-04: the card was at 98% utilisation and
    288 W, and NO Windows process accounted for it. The work was in a WSL
    Ubuntu, whose GPU memory is allocated through the vGPU path and never
    appears in the Windows per-process commit counter. The Windows-side
    detector saw an idle card on a card that was pinned.

    So the probe asks WSL directly, per RUNNING distribution. `wsl --list
    --running` is used rather than naming a distribution, because invoking
    `wsl -d <name>` on a STOPPED distribution STARTS it — a detector must
    never change the state it is measuring.

    A WSL compute app counts unconditionally: there is no desktop compositor
    in there, and the harness does not run its measurements inside WSL.
    """
    if os.environ.get("WSL_DISTRO_NAME"):
        return []                      # we are the WSL process; the Windows
                                       # side of this probe owns the question
    now = time.time()
    if now - _wsl_cache["at"] < _WSL_CACHE_TTL_S:
        return list(_wsl_cache["value"])
    found: List[str] = []
    distros = _wsl_running_distros()
    if not distros:
        return found
    for d in distros:
        try:
            r = subprocess.run(
                ["wsl.exe", "-d", d, "--", "nvidia-smi",
                 "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
        except Exception:
            continue
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            found.append(f"wsl:{d}:{line.replace(',', ':')}")
    _wsl_cache["at"] = time.time()
    _wsl_cache["value"] = tuple(found)
    return found


def other_gpu_processes(threshold_mib: int = HEAVY_GPU_COMMIT_MIB) -> List[str]:
    """Heavy GPU processes that are NOT ours. The names, sorted.

    This is the contention signal that works DURING a render. V12's baseline
    module already reads the per-process GPU commit counter
    (`\\GPU Process Memory(*)\\Total Committed`), which is the honest measure —
    nvidia-smi's own process list is unusable under WDDM, where it enumerates
    every desktop app on the machine.

    A process counts when it holds a serious amount of the card, or when it is
    a render tool at all. The always-resident OS graphics processes are
    exempt — see OS_GRAPHICS_EXEMPT for why they cannot be counted.

    WSL is probed separately — see _wsl_gpu_processes for why the Windows
    side cannot see it.
    """
    try:
        if str(V12_DIR) not in sys.path:
            sys.path.insert(0, str(V12_DIR))
        from v12.baseline import _gpu_commit_by_pid
        commit = _gpu_commit_by_pid()
    except Exception:
        commit = None
    mine = my_process_tree()
    out = []
    if commit:
        for pid, mib in commit.items():
            if pid in mine:
                continue
            try:
                import psutil
                name = psutil.Process(pid).name()
            except Exception:
                name = f"pid{pid}"
            if name.lower() in OS_GRAPHICS_EXEMPT:
                continue
            if mib < threshold_mib and not _render_app(name):
                continue
            out.append(f"{name}:{mib:.0f}MiB")
    out.extend(_wsl_gpu_processes())
    return sorted(out)


def wait_for_no_others(timeout_s: float = 300.0, poll_s: float = 5.0,
                       verbose: bool = True) -> bool:
    """Wait until nobody else is on the card. True when it is clear.

    Unlike the utilisation check, this one is valid DURING a render: it looks
    at WHO holds the card, not at how hard it is working. My own render is
    excluded by process identity, so what is left is somebody else's.

    The waiting is reported once a minute, not once a poll. A contended
    machine can stay contended for hours — a benchmark job on the card, say —
    and a line every five seconds is a log nobody can read and a terminal
    nobody can scroll.
    """
    deadline = time.time() + max(0.0, timeout_s)
    attempt = 0
    last_report = 0.0
    while True:
        attempt += 1
        others = other_gpu_processes()
        if not others:
            if attempt > 1 and verbose:
                print("  nobody else is on the card now")
            return True
        now = time.time()
        if now >= deadline:
            if verbose:
                print(f"  still seeing other GPU work after {timeout_s:.0f}s "
                      f"({', '.join(others)}) — measuring anyway; the drift "
                      f"gate will refuse the result if the machine does not "
                      f"hold still")
            return False
        if verbose and now - last_report >= 60.0:
            last_report = now
            print(f"  other GPU work on the card ({', '.join(others)}) — "
                  f"waiting for it to finish "
                  f"({max(0, deadline - now):.0f}s left)")
        time.sleep(poll_s)


class ContenderWatcher:
    """Watches for OTHER GPU work WHILE something else runs.

    Checking only before and after a measurement misses the case that matters
    most: a rival that arrives mid-render and LEAVES again before the render
    finishes. Observed on this box — a Houdini watch job that woke up, took
    the card for a few frames and dropped it, leaving a repetition that took
    148 s where a clean frame takes 48 s, and reported clean.

    This polls in the background and remembers everything it saw, so the
    verdict is made against what happened DURING the work, not just at its
    edges.

    The poll is process identity, so our own renderer — a child of this
    process, and the thing actually holding the card — is never counted.

        with ContenderWatcher() as w:
            render_the_frame()
        if w.contenders:
            # somebody else was on the card
    """

    def __init__(self, interval: float = 3.0):
        self.interval = interval
        self.contenders: List[str] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for name in other_gpu_processes():
                    if name not in self.contenders:
                        self.contenders.append(name)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "ContenderWatcher":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # One last look, in case a rival arrived during the final interval.
        try:
            for name in other_gpu_processes():
                if name not in self.contenders:
                    self.contenders.append(name)
        except Exception:
            pass

    @property
    def clean(self) -> bool:
        """Was the card ours alone for the whole of this work?"""
        return not self.contenders


def jsonable(value: Any) -> Any:
    """Coerce a value into something `json.dumps` accepts, without losing it.

    A record is written AFTER the measurement, so a TypeError at write time
    throws away an hour of GPU work and leaves nothing on disk to argue with.
    The types that actually cause that are few:

      * numpy scalars — np.float64 is a float, but np.int64 and np.bool_ are
        not int and bool, and json refuses them;
      * numpy arrays;
      * non-finite floats, which json writes as `NaN` and `Infinity` — not
        valid JSON, so a record written that way cannot be re-read by a
        strict parser, and a record nobody can re-read is a record nobody
        can check.

    So numpy becomes Python, tuples become lists, and a non-finite float
    becomes null — which says "there is no value here" honestly, where `NaN`
    says something no strict reader will accept. Anything else that is not
    JSON-native becomes its string form rather than taking the whole record
    down with it.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    # numpy, by duck type: the harness must not depend on numpy being
    # installed to be able to write a record down.
    if hasattr(value, "tolist"):          # an array
        return jsonable(value.tolist())
    if hasattr(value, "item"):            # a numpy scalar
        return jsonable(value.item())
    return str(value)


def write_measurement(name: str, payload: Dict) -> Path:
    ensure_dirs()
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = MEASURE_ROOT / f"{name}_{ts}.json"
    payload = dict(payload)
    payload.setdefault("recorded_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    p.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    return p


def paste_lines(objective: str, primary: Sequence[float], harness: str,
                guardrails: Optional[Dict[str, Sequence[float]]] = None,
                candidate: bool = False) -> List[str]:
    """The lines an operator pastes into V12.

    What changed, and why
    ---------------------
    This used to print a CANDIDATE line — `v12 run --candidate-for "KEY=..."` —
    for every measurement. That command compares the pasted series against
    whatever baseline happens to be on disk, and the baseline on disk is usually
    from an EARLIER SESSION. So the line it printed was the exact command that
    produced the 2026-11-03 phantom: a candidate measured at 19:29, gated
    against a baseline measured at 18:41, admitted at +12.34% with p=1.882e-05.

    V12 now REFUSES that comparison. The gate requires the candidate to carry
    its provenance — when it was measured, in which session, and the per-pair
    differences against its control — and refuses when it cannot show that the
    two arms are one experiment. So printing the candidate line here would be
    printing a command that cannot succeed, and worse, printing one that LOOKS
    like the way to gate a candidate.

    So this prints the procedure that actually works: measure both arms
    alternately in ONE session, with the paired harness, and gate the result
    there. The baseline line is still printed, because recording where the
    machine stands is a legitimate thing to do — it is just no longer the
    first half of a comparison that cannot be made across sessions.

    `candidate` is kept for callers that asked for the old shape; it now
    changes only what is printed, and never re-introduces the stale path.
    """
    guardrails = guardrails or {}
    v12 = f'"{V12_PYTHON}" -m v12'
    lines = []
    base = f'{v12} baseline --objective {objective} --record "{series(primary)}" --harness {harness}'
    for k, v in guardrails.items():
        base += f' --guardrail "{k}={series(v)}"'
    lines.append("# as a BASELINE (no change applied) — records where the machine stands:")
    lines.append(base)
    lines.append("#")
    lines.append("# to COMPARE a change, both arms must be measured ALTERNATELY in ONE")
    lines.append("# session. A candidate measured in a different session cannot be gated")
    lines.append("# against this baseline at all: the drift between the two sessions is")
    lines.append("# indistinguishable from the effect (that is how a +12.34% phantom was")
    lines.append("# admitted on 2026-11-03), so V12 now refuses the comparison.")
    lines.append("#")
    lines.append("# measure both arms together, in one session:")
    lines.append("#     python harness\\optimize.py --request render --levers manual")
    lines.append("#   (the manual lever times the change-applied and change-not-applied arms")
    lines.append("#    alternately; a control that drifts more than 5% refuses the verdict)")
    lines.append(f"# both run from: {V12_DIR}")
    return lines
