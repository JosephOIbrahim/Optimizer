"""
V12 — guards: one instance, no collisions, intact couplings.

Three distinct hazards, each of which produces a silent wrong answer rather than
an error, which is why they need a module instead of a convention.

1. **Two optimizers at once.** There is no cross-process GPU partitioning of any
   kind on this box — no MIG (datacenter only), no MPS (Linux/QNX only), no TCC
   (unreachable on GeForce). Separate processes are *time-sliced*. So if V12 runs
   while MachineOptimizer2026's watchdog is also running, both are measuring a
   machine the other is mutating, and every number either produces is worthless
   while looking perfectly plausible. A lock, not a hope.

2. **The documented but unvalidated coupling.** MachineOptimizer2026 consumes V9's
   virtualenv (`D:\\OptimizerV9\\venv\\Scripts\\python.exe`) and V9's engine cache
   (`D:\\OptimizerV9\\engine_cache\\comfyui`). The audit confirmed this is real and
   that **nothing validates it** — `v9/` does not know MOS2026 exists, and a path
   move on either side silently breaks the other at some later date. V12 checks.

3. **Measuring while the machine is busy.** Houdini was running during V12's own
   recon, holding 2.84 GB. A baseline taken in that state is not comparable to one
   taken without it. `MeasurementGuard` refuses to record one without the operator
   saying the machine is quiet.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from . import paths

log = logging.getLogger("v12.guard")

#: GPU-resident applications whose presence changes what a measurement means.
#: Not a security list — a comparability list. Entries are lowercase image names,
#: compared exactly against `_process_table()`. Bare python.exe stays on purpose:
#: ComfyUI runs under it, and an over-broad match only blocks a quiet baseline,
#: which is the closed side. (There was a bare "comfyui" here; no image name can
#: equal it, so it matched nothing. ComfyUI Desktop's image is comfyui.exe.)
SIGNIFICANT_APPS = {
    # Houdini: interactive, batch (hython) and the USD renderer (husk)
    "houdini.exe", "houdinifx.exe", "hython.exe", "husk.exe", "karma.exe",
    "redshift4houdini.exe",
    # Unreal: editor, headless commandlet runs, and its shader compile farm
    "unrealeditor.exe", "unrealeditor-cmd.exe", "shadercompileworker.exe",
    "ue5editor.exe",
    # ComfyUI
    "comfyui.exe", "python.exe",
}

#: Other optimizers. If any of these is running, V12 must not also be running.
COMPETING_OPTIMIZERS = {
    "watch-this-machine.bat",
    "optimize_rtx4090_vfx.bat",
    "launch_optimizer.bat",
    "v11-optimize-this-machine.bat",
    "machineoptimizer2026",
}

#: Images that host a launcher script. tasklist and Win32_Process both name a
#: .bat optimizer 'cmd.exe' and a Python one 'python.exe', so the image-name match
#: above cannot see them; their command lines can.
LAUNCHER_HOSTS = frozenset({"cmd.exe", "python.exe", "pythonw.exe"})

#: Another optimizer, read off a launcher host's command line. Deterministic, so
#: it MAY decide a block. Names, not directories, where a name exists; the bare
#: MachineOptimizer2026 entry is kept because the operator asked for it, and an
#: over-broad match here only blocks, which is the closed side. V12's own launcher
#: is V12-Optimize-This-Machine.bat, hence the lookbehind; its host cmd.exe is an
#: ancestor of this process and is excluded as well.
OPTIMIZER_COMMAND_LINE = re.compile(
    r"(?i)(watch-this-machine"
    r"|(?<!v12-)optimize-this-machine"
    r"|optimize_rtx4090_vfx"
    r"|launch_optimizer"
    r"|v9_optimizer"
    r"|machineoptimizer2026"
    r"|mos2026\.py"
    r"|-m\s+optimizer\.orchestrator\b"     # MOS2026's watchdog, as Watch-This-Machine.bat runs it
    r"|-m\s+v9\s+apply\b)")

#: Python images. The only rows whose command lines ever leave this machine (as
#: Jev annotations, via plane.annotate_processes) — never cmd.exe or PowerShell.
PYTHON_IMAGES = frozenset({"python.exe", "pythonw.exe"})


# ---------------------------------------------------------------------------
# 1. single instance
# ---------------------------------------------------------------------------
class SingleInstance:
    """Exclusive lock, with stale-lock recovery.

    A PID-based lock file alone is not enough — a crashed process leaves the file
    behind and every later run refuses. And an OS lock alone is not enough either,
    because it gives no diagnostic. So: an OS-level byte lock (which the kernel
    releases on process death, so crashes self-heal) plus a PID written inside for
    the error message.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or paths.LOCK_FILE)
        self.fh = None
        self.acquired = False

    def acquire(self) -> Tuple[bool, str]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = self._read_holder()
            return False, (f"another V12 instance holds {self.path}"
                           + (f" (pid {holder})" if holder else "")
                           + " — refusing to run two optimizers against one GPU")
        self.acquired = True
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"{os.getpid()}\n{time.time()}\n")
        self.fh.flush()
        atexit.register(self.release)
        return True, f"lock acquired (pid {os.getpid()})"

    def _read_holder(self) -> str:
        try:
            with open(self.path) as f:
                return f.readline().strip()
        except Exception:
            return ""

    def release(self) -> None:
        if not self.acquired or self.fh is None:
            return
        try:
            self.fh.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self.fh.close()
        except Exception:
            pass
        self.acquired = False


# ---------------------------------------------------------------------------
# 2. collision detection
# ---------------------------------------------------------------------------
@dataclass
class Collision:
    kind: str                    # "optimizer" | "gpu-app"
    name: str
    pid: int = 0
    detail: str = ""


#: Note prefix preflight uses when tasklist could not be read. The plane refuses a
#: live cycle on it, the same way it refuses on a competing optimizer.
PROCESS_TABLE_UNREADABLE = "process table unreadable"


@dataclass(frozen=True)
class ProcessRow:
    """One process as Win32_Process reports it. `cmdline` None means it could not
    be read (an elevated process seen from an unelevated reader, a protected
    process, or the tasklist fallback) — never "no arguments"."""
    pid: int
    name: str                          # lowercase image name
    ppid: Optional[int] = None
    exe: Optional[str] = None          # ExecutablePath
    cmdline: Optional[str] = None      # CommandLine
    created: Optional[str] = None      # CreationDate, UTC ISO-8601


class ProcessTable(list):
    """(pid, lowercase image name) pairs — the shape every finder, the baseline's
    competing-process reading and Conditions iterate — carrying the full rows
    beside them. A plain list of pairs is still a valid table: everything that
    needs a command line asks `rows_of()` and gets {} for one."""

    def __init__(self, rows: Iterable[ProcessRow] = (), *, source: str = "",
                 elevated: Optional[bool] = None):
        rows = list(rows)
        super().__init__((r.pid, r.name) for r in rows)
        self.rows: Dict[int, ProcessRow] = {r.pid: r for r in rows}
        self.source = source            # "cim" | "tasklist"
        self.elevated = elevated        # None = could not tell


def rows_of(table) -> Dict[int, ProcessRow]:
    """pid -> ProcessRow, or {} for a table without rows (a bare pair list)."""
    return getattr(table, "rows", None) or {}


def _is_elevated() -> Optional[bool]:
    """Whether this process runs as Administrator; None when it cannot tell."""
    try:
        if sys.platform == "win32":
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except Exception:
        return None


#: Win32_Process through Get-CimInstance, the way MOS2026's probe reads it, but as
#: JSON rather than CSV: command lines carry commas and quotes. -InputObject @()
#: so one process still prints an array; UTF-8 so a non-ASCII path survives.
_CIM_SCRIPT = (
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
    "ConvertTo-Json -Compress -Depth 2 -InputObject @(Get-CimInstance Win32_Process | "
    "Select-Object Name,ProcessId,ParentProcessId,ExecutablePath,CommandLine,"
    "@{n='CreationDate';e={if($_.CreationDate){"
    "$_.CreationDate.ToUniversalTime().ToString('o')}}})")


def _text(out) -> str:
    if isinstance(out, bytes):
        return out.decode("utf-8-sig", errors="replace")
    return (out or "").lstrip("﻿")


def _str_or_none(v) -> Optional[str]:
    return v if isinstance(v, str) else None


def _cim_table() -> Optional["ProcessTable"]:
    """The full table from Win32_Process, or None when it could not be read."""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _CIM_SCRIPT],
            capture_output=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as e:
        log.warning("Win32_Process read failed: %s: %s", type(e).__name__, e)
        return None
    if proc.returncode != 0:
        log.warning("Win32_Process read exited %s", proc.returncode)
        return None
    try:
        data = json.loads(_text(proc.stdout).strip() or "null")
    except ValueError:
        log.warning("Win32_Process read returned no JSON")
        return None
    if isinstance(data, dict):
        data = [data]
    rows = []
    for d in data if isinstance(data, list) else []:
        try:
            pid = int(d["ProcessId"])
            name = str(d["Name"]).lower()
        except (KeyError, TypeError, ValueError):
            continue
        ppid = d.get("ParentProcessId")
        rows.append(ProcessRow(
            pid=pid, name=name,
            ppid=int(ppid) if isinstance(ppid, int) and not isinstance(ppid, bool) else None,
            exe=_str_or_none(d.get("ExecutablePath")),
            cmdline=_str_or_none(d.get("CommandLine")),
            created=_str_or_none(d.get("CreationDate")),
        ))
    if not rows:
        log.warning("Win32_Process read returned no parseable rows")
        return None
    return ProcessTable(rows, source="cim", elevated=_is_elevated())


def _process_table() -> Optional[List[Tuple[int, str]]]:
    """(pid, lowercase image name), as a ProcessTable that also carries each
    row's parent, path, command line and start time. Win32_Process first; tasklist
    when that cannot be read (names and pids only). One read serves everything a
    preflight or an OBSERVE derives.

    Returns None when neither could be read. An empty list would mean "no
    processes at all", which a working reader never reports, and reading a
    failure as an empty table would let preflight say CLEAR about a machine it
    never looked at."""
    table = _cim_table()
    if table is not None:
        return table
    return _tasklist_table()


def _tasklist_table() -> Optional["ProcessTable"]:
    """The fallback. tasklist is available everywhere on Windows and cheap, but
    gives names and pids only, so the command-line checks cannot run on it."""
    try:
        proc = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as e:
        log.warning("tasklist failed: %s: %s", type(e).__name__, e)
        return None
    if proc.returncode != 0:
        log.warning("tasklist exited %s", proc.returncode)
        return None
    out = proc.stdout or ""
    rows = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2:
            try:
                rows.append((int(parts[1]), parts[0].lower()))
            except ValueError:
                continue
    if not rows:
        log.warning("tasklist returned no parseable rows")
        return None
    return ProcessTable((ProcessRow(pid, name) for pid, name in rows),
                        source="tasklist", elevated=_is_elevated())


# ---------------------------------------------------------------------------
# deterministic process identity
# ---------------------------------------------------------------------------
_ARG = re.compile(r'"([^"]*)"|(\S+)')


def _argv(cmdline: str) -> List[str]:
    """Windows-style split, good enough to find a script path: a quoted run is
    one argument. Not CommandLineToArgvW — no escape rules are needed here."""
    return [a or b for a, b in _ARG.findall(cmdline)]


def comfyui_dir(row: ProcessRow) -> Optional[str]:
    r"""The ComfyUI install a python row runs, resolved from its command line.

    main.py is taken as written when absolute, else relative to the interpreter's
    directory and the two above it — Win32_Process has no cwd, and the portable
    build runs `python_embeded\python.exe -s ComfyUI\main.py` from the folder
    above the interpreter, a venv from two above. A directory counts only if it
    holds `comfy\` and `nodes.py`: a main.py alone is any script at all."""
    if row.name not in PYTHON_IMAGES or not row.cmdline:
        return None
    exe_dir = Path(row.exe).parent if row.exe else None
    for arg in _argv(row.cmdline)[1:]:
        a = arg.replace("/", "\\")
        if a.lower() != "main.py" and not a.lower().endswith("\\main.py"):
            continue
        if os.path.isabs(a):
            cands = [Path(a)]
        elif exe_dir is not None:
            cands = [exe_dir / a, exe_dir.parent / a, exe_dir.parent.parent / a]
        else:
            cands = []
        for c in cands:
            d = Path(os.path.normpath(str(c))).parent
            try:
                if (d / "comfy").is_dir() and (d / "nodes.py").is_file():
                    return str(d)
            except OSError:
                continue
    return None


def identify(row: ProcessRow) -> Optional[str]:
    """What a row deterministically is, or None when the rules cannot say. These
    rules always win: a row they resolve is never sent to Jev. Identity is for
    that routing and for display — gpu_apps still lists python.exe by image."""
    if row.name in SIGNIFICANT_APPS and row.name not in PYTHON_IMAGES:
        return row.name
    if row.name in LAUNCHER_HOSTS and row.cmdline:
        m = OPTIMIZER_COMMAND_LINE.search(row.cmdline)
        if m:
            return f"optimizer launcher ({m.group(0)})"
    d = comfyui_dir(row)
    if d:
        return f"ComfyUI ({d})"
    # MOS2026's probe rule, for a ComfyUI that cannot be resolved on disk.
    low = (row.cmdline or "").lower()
    if row.name in PYTHON_IMAGES and "main.py" in low and "comfyui" in low:
        return "ComfyUI (command line)"
    return None


def _is_parent(rows: Dict[int, ProcessRow], parent: int, child: int) -> bool:
    """Whether the live row at `parent` can be `child`'s parent. Windows keeps a
    dead parent's pid in ParentProcessId and reuses pids, so a live row at that
    pid created AFTER the child is an unrelated process — possibly another
    optimizer. A missing CreationDate cannot be shown to be the parent, so it is
    not (fails closed: an unproven ancestor is not exempted from anything).
    CreationDate is the 'o' format in UTC, fixed width, so it orders as text."""
    p, c = rows.get(parent), rows.get(child)
    if p is None or c is None or not p.created or not c.created:
        return False
    return p.created <= c.created


def _lineage(table) -> Tuple[Set[int], Set[int]]:
    """(this process and its ancestors, its descendants), by pid. Ancestors come
    from the rows' parent links plus os.getppid(), so a table without rows still
    knows the parent. Each link is checked against CreationDate (_is_parent);
    the walk stops at the first one a reused pid could have forged."""
    rows = rows_of(table)
    me = os.getpid()
    up: Set[int] = {me}
    try:
        ppid = os.getppid()
        # Without rows nothing can be matched by command line, so the bare pid is
        # harmless; with rows, the same reuse check as the walk below.
        if not rows or ppid not in rows or _is_parent(rows, ppid, me):
            up.add(ppid)
    except Exception:
        pass
    seen: Set[int] = set()
    pid = me
    while pid in rows and pid not in seen and rows[pid].ppid not in (None, 0):
        seen.add(pid)                   # a pid-reuse loop ends here
        parent = rows[pid].ppid
        if parent in rows and not _is_parent(rows, parent, pid):
            break                       # dead parent, pid since reused
        pid = parent
        up.add(pid)
    down: Set[int] = set()
    frontier = {me}
    while frontier:
        kids = {r.pid for r in rows.values()
                if r.ppid in frontier and r.pid not in down and r.pid not in up}
        down |= kids
        frontier = kids
    return up, down


def unresolved_python(table) -> List[ProcessRow]:
    """python.exe / pythonw.exe rows with a readable command line that the
    deterministic rules leave unresolved — the only rows Jev is ever shown.
    V12 itself, its ancestors and its descendants are left out."""
    up, down = _lineage(table)
    return [r for _, r in sorted(rows_of(table).items())
            if r.name in PYTHON_IMAGES and r.cmdline
            and r.pid not in up and r.pid not in down
            and identify(r) is None]


def table_notes(table) -> List[str]:
    """What the reader itself could not see. Deterministic; decides nothing."""
    source = getattr(table, "source", "")
    if source == "cim" and getattr(table, "elevated", None) is False:
        n = sum(1 for r in rows_of(table).values() if r.cmdline is None)
        if n:
            return [f"{n} processes unreadable (not elevated) - "
                    f"competing-optimizer check incomplete"]
    if source == "tasklist":
        return ["command lines unread (Win32_Process unavailable, tasklist fallback) - "
                "competing-optimizer check incomplete"]
    return []


def find_competing_optimizers(table: Optional[ProcessTable] = None) -> List[Collision]:
    """Another optimizer's process. This is the dangerous one: it mutates the
    machine while V12 measures it, and there is no GPU partitioning to keep them
    apart."""
    me = os.getpid()
    found = []
    if table is None:
        table = _process_table() or []
    rows = rows_of(table)
    ancestors: Set[int] = _lineage(table)[0] if rows else set()
    for pid, name in table:
        if pid == me:
            continue
        if name in COMPETING_OPTIMIZERS or any(
                k in name for k in ("watch-this-machine", "optimize_rtx4090", "machineoptimizer")):
            found.append(Collision("optimizer", name, pid,
                                   "another optimizer is running — it will mutate the "
                                   "machine underneath V12's measurements"))
            continue
        # A .bat or Python optimizer is 'cmd.exe' / 'python.exe' by image name;
        # its command line says which. V12's own launcher chain is excluded.
        row = rows.get(pid)
        if (name in LAUNCHER_HOSTS and pid not in ancestors
                and row is not None and row.cmdline):
            m = OPTIMIZER_COMMAND_LINE.search(row.cmdline)
            if m:
                found.append(Collision("optimizer", name, pid,
                                       f"command line runs {m.group(0)!r} — another "
                                       f"optimizer will mutate the machine underneath "
                                       f"V12's measurements"))
    return found


def find_gpu_apps(table: Optional[ProcessTable] = None) -> List[Collision]:
    """GPU-resident apps, for measurement comparability. Not an error — context."""
    me = os.getpid()
    out = []
    for pid, name in ((_process_table() or []) if table is None else table):
        if pid == me:
            continue
        if name in SIGNIFICANT_APPS:
            out.append(Collision("gpu-app", name, pid, "GPU-resident; affects comparability"))
    return out


def wsl_gpu_apps() -> List[Collision]:
    """GPU work running inside WSL distributions.

    Found the hard way on 2026-11-04: preflight said CLEAR — nothing is
    competing for the GPU — on a card that was at 98% utilisation and 288 W.
    The work belonged to a python3.12 process inside a WSL Ubuntu, whose GPU
    memory is allocated through the vGPU path and never appears in the
    Windows process table this function reads.

    A preflight that reports CLEAR on a pinned card is worse than one that
    reports nothing: it tells the operator the machine is free when it is
    not, and every measurement taken then describes two workloads at once.

    The running distributions are enumerated with `wsl --list --running`,
    because `wsl -d <name>` on a STOPPED distribution STARTS it — a check
    must never change the state it is checking.
    """
    if os.environ.get("WSL_DISTRO_NAME"):
        return []                        # we are the WSL process; the Windows
                                         # side owns the question
    try:
        out = subprocess.run(
            ["wsl.exe", "--list", "--running", "--quiet"],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    raw = out.stdout or ""
    if "\x00" in raw:
        # UTF-16 on this box, already decoded by text mode: the ASCII
        # characters arrive separated by NULs.
        try:
            raw = raw.encode("utf-8", "surrogatepass").decode(
                "utf-16-le", errors="replace")
        except Exception:
            raw = raw.replace("\x00", "")
    distros = [n.strip() for n in raw.replace("\x00", "").splitlines() if n.strip()]

    found: List[Collision] = []
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
            parts = [p.strip() for p in line.split(",")]
            pid = parts[0] if parts else "?"
            name = parts[1] if len(parts) > 1 else line
            found.append(Collision(
                "wsl-gpu-app", f"{name} (wsl:{d})", pid,
                "GPU-resident inside WSL; the Windows process table cannot see it, "
                "so it is invisible to every other check here"))
    return found


def find_v9_processes(table: Optional[ProcessTable] = None) -> List[Collision]:
    """V9's own tooling, if it is mid-run. Separate from 'optimizer' because V9's
    CLI is read-only unless you ask it to apply — worth naming distinctly."""
    found = []
    for pid, name in ((_process_table() or []) if table is None else table):
        if name in ("nvcc.exe", "ptxas.exe", "ncu.exe", "nsys.exe", "trtexec.exe"):
            found.append(Collision("v9-tooling", name, pid,
                                   "a build/profiler tool is running; it will occupy the GPU"))
    return found


#: preflight's "read the table yourself" default. Not None: None is what
#: `_process_table()` returns when the read FAILED, and a caller that passes the
#: table it already read must be able to pass that failure through as-is.
_READ_TABLE = object()


def preflight(require_quiet: bool = False, *,
              table: object = _READ_TABLE) -> Tuple[bool, List[Collision], List[str]]:
    """Everything that must be true before V12 acts or measures.

    Returns (ok, collisions, notes). `require_quiet=True` is what a baseline run
    uses — it demands the machine actually be idle, because a baseline taken
    during someone else's render describes that render, not this machine.

    `table` lets a caller (the plane's OBSERVE) judge preflight on a process
    table it already read, so everything it derives sees one snapshot. Passing
    None means that read failed, and preflight fails closed exactly as if its own
    read had.
    """
    collisions: List[Collision] = []
    notes: List[str] = []
    ok = True

    # One tasklist call, one snapshot: three calls cost three process spawns and
    # could each see a different machine.
    if table is _READ_TABLE:
        table = _process_table()
    if table is None:
        # Fail closed: without the table nothing below can be judged, and every
        # finder would report "nothing found" about a machine it never saw.
        return False, collisions, [
            f"{PROCESS_TABLE_UNREADABLE} (tasklist failed) — cannot confirm that no "
            f"other optimizer or GPU application is running"]

    opt = find_competing_optimizers(table)
    collisions += opt
    if opt:
        ok = False
        notes.append("Another optimizer is running. Stop it first — with no MIG, no MPS "
                     "and no TCC on this hardware, the two processes are time-sliced and "
                     "every measurement either one takes is meaningless.")
    # What the reader could not see, said rather than read as "nothing there".
    notes += table_notes(table)

    gpu_apps = find_gpu_apps(table)
    wsl_apps = wsl_gpu_apps()
    v9tools = find_v9_processes(table)
    if require_quiet:
        collisions += gpu_apps + wsl_apps + v9tools
        if gpu_apps:
            ok = False
            names = ", ".join(sorted({c.name for c in gpu_apps}))
            notes.append(f"Machine is not quiet for a baseline: {names} resident. A "
                         f"baseline taken now will not be comparable to one taken on an "
                         f"idle machine.")
        if wsl_apps:
            ok = False
            names = ", ".join(sorted({c.name for c in wsl_apps}))
            notes.append(f"Machine is not quiet for a baseline: {names} resident. "
                         f"GPU work inside WSL does not appear in the Windows process "
                         f"table, so nothing else here can see it — and two GPU "
                         f"workloads on this card are time-sliced against each other.")
        if v9tools:
            ok = False
            notes.append("Build/profiler tooling is running: "
                         + ", ".join(sorted({c.name for c in v9tools})))
    else:
        if gpu_apps:
            notes.append("GPU-resident: " + ", ".join(sorted({c.name for c in gpu_apps}))
                         + " (recorded in baseline conditions)")
        if wsl_apps:
            names = ", ".join(sorted({c.name for c in wsl_apps}))
            notes.append("GPU-resident inside WSL: " + names
                         + " — invisible to the Windows process table")

    return ok, collisions, notes


# ---------------------------------------------------------------------------
# 3. coupling validation
# ---------------------------------------------------------------------------
@dataclass
class CouplingCheck:
    name: str
    path: Path
    ok: bool
    detail: str = ""
    is_link: bool = False
    target: str = ""


def _link_target(p: Path) -> Tuple[bool, str]:
    try:
        t = os.readlink(p)
        return True, t
    except OSError:
        return False, ""


def validate_couplings() -> List[CouplingCheck]:
    """The MOS2026 -> V9 couplings that nothing currently validates.

    Per the audit: `v9/` imports nothing from MOS2026 and has no knowledge it
    exists. The coupling is purely file-level, so it can only be checked at the
    filesystem, which is exactly what this does.
    """
    checks: List[CouplingCheck] = []

    venv_py = Path(r"D:\OptimizerV9\venv\Scripts\python.exe")
    checks.append(CouplingCheck(
        "MOS2026 -> V9 venv (python)", venv_py, venv_py.exists(),
        "MOS2026 shells out to this interpreter" if venv_py.exists()
        else "MISSING — every MOS2026 agent that shells out to V9's interpreter will fail",
    ))

    cache = Path(r"D:\OptimizerV9\engine_cache\comfyui")
    is_link, target = _link_target(cache) if cache.exists() else (False, "")
    checks.append(CouplingCheck(
        "MOS2026 -> V9 engine cache", cache, cache.exists(),
        ("junction -> " + target) if is_link else
        ("plain directory" if cache.exists() else
         "MISSING — compiled engines will be rebuilt from scratch"),
        is_link=is_link, target=target,
    ))

    v9_data = Path(r"D:\OptimizerV9")
    checks.append(CouplingCheck(
        "V9 data root", v9_data, v9_data.exists(),
        "V9 resolves here via v9/paths.py after $OPTIMIZER_V9_DATA",
    ))

    for name, p in (("V12 data root", paths.DATA_ROOT),
                    ("MOS2026 data (migrated)", paths.MOS2026_DATA)):
        ok = p.exists()
        detail = str(p)
        if ok and paths.is_under_onedrive(p):
            ok = False
            detail += "  [UNDER ONEDRIVE — move it]"
        checks.append(CouplingCheck(name, p, ok, detail))

    legacy = paths.MOS2026_DATA_LEGACY
    if legacy.exists():
        checks.append(CouplingCheck(
            "MOS2026 data (legacy, pre-migration)", legacy, True,
            "still present; verify the migration before deleting",
        ))
    return checks


def free_space_gb(p: Path) -> Optional[float]:
    try:
        return shutil.disk_usage(str(p if p.exists() else p.parent)).free / 2**30
    except Exception:
        return None


def report() -> str:
    ok, collisions, notes = preflight()
    lines = ["V12 preflight", "-" * 68,
             f"  {'CLEAR' if ok else 'BLOCKED'}: {len(collisions)} process(es) found"]
    for c in collisions:
        lines.append(f"    [{c.kind}] {c.name} (pid {c.pid}) — {c.detail}")
    for n in notes:
        lines.append(f"    note: {n}")
    lines.append("")
    lines.append("  couplings:")
    for c in validate_couplings():
        mark = "OK  " if c.ok else "FAIL"
        lines.append(f"    [{mark}] {c.name}: {c.detail}")
    fs = free_space_gb(paths.DATA_ROOT)
    if fs is not None:
        lines.append(f"    free space at {paths.DATA_ROOT}: {fs:.1f} GiB")
    return "\n".join(lines)


if __name__ == "__main__":
    print(paths.describe())
    print(report())
