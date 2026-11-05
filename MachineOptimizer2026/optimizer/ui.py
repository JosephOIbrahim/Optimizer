"""Terminal presentation. Uses `rich` if installed, else a plain-text fallback so the
tool works before dependencies are installed."""
from __future__ import annotations

import sys
from typing import List

from .base import Change, ChangeType, Context, Proposal

# Make Windows consoles (cp1252 by default) tolerate box-drawing / arrows / marks.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.markup import escape as _escape
    from rich import box
    # legacy_windows=False forces ANSI VT output (Win11 conhost supports it) and avoids
    # rich's legacy Windows renderer, which crashes on non-UTF8 code pages.
    _c = Console(legacy_windows=False, highlight=False)
    _RICH = True
except Exception:  # pragma: no cover
    _RICH = False
    _c = None


def _e(s) -> str:
    """Escape dynamic text so brackets (e.g. UE [SystemSettings]) aren't parsed as rich markup."""
    s = str(s)
    return _escape(s) if _RICH else s


def _p(msg: str = "") -> None:
    if _RICH:
        _c.print(msg)
    else:
        print(_strip(msg))


def _strip(s: str) -> str:
    import re
    return re.sub(r"\[/?[a-z0-9_ #]+\]", "", s)


def banner(ctx: Context, mode: str) -> None:
    from .hardware_profile import PROFILE
    bar = "=" * 64
    _p()
    _p(f"[bold cyan]{bar}[/bold cyan]")
    _p("[bold]  MachineOptimizer 2026 - RTX 4090 . Threadripper PRO 7965WX . VFX[/bold]")
    _p(f"[bold cyan]{bar}[/bold cyan]")
    admin = "[green]admin[/green]" if ctx.is_admin else "[yellow]non-admin (system tweaks skipped)[/yellow]"
    _p(f"  GPU {ctx.gpu_util}%  ·  VRAM {ctx.vram_used_gb}/{ctx.vram_total_gb} GB  ·  "
       f"{ctx.temp_c}°C  ·  {ctx.power_w} W  ·  PCIe Gen{ctx.pcie_gen}")
    _p(f"  Workload: [bold]{ctx.workload.value}[/bold]  ·  cores {ctx.physical_cores}p/{ctx.logical_cores}t  ·  "
       f"NVMe {ctx.fastest_nvme or '?'}  ·  {admin}")
    if ctx.throttle_reasons:
        _p(f"  [yellow]throttling: {', '.join(ctx.throttle_reasons)}[/yellow]")
    _p(f"  Mode: [bold magenta]{mode}[/bold magenta]")
    _p()


_RISK_COLOR = {"none": "green", "low": "green", "medium": "yellow", "high": "red"}


def plan_table(proposals: List[Proposal]) -> None:
    if _RICH:
        t = Table(box=box.SIMPLE_HEAVY, show_lines=False, expand=True)
        for col in ("Domain", "Optimization", "Change", "Current → Target", "Risk", "Conf"):
            t.add_column(col, overflow="fold")
        for p in proposals:
            for i, ch in enumerate(p.changes):
                dom = _e(p.domain) if i == 0 else ""
                title = _e(p.title) if i == 0 else ""
                conf = f"{p.confidence:.0%}" if i == 0 else ""
                gate = " [red]*admin[/red]" if ch.requires_admin else ""
                cur = "(absent)" if ch.current_value is None else _e(ch.current_value)
                arrow = f"{cur} → [bold]{_e(ch.target_value)}[/bold]" if ch.type is not ChangeType.GUIDED else "[dim]manual[/dim]"
                rc = _RISK_COLOR.get(ch.risk.value, "white")
                t.add_row(dom, title, f"{_e(ch.description)}{gate}", arrow, f"[{rc}]{ch.risk.value}[/{rc}]", conf)
        _c.print(t)
    else:
        for p in proposals:
            print(f"\n[{p.domain}] {p.title}  ({p.confidence:.0%})")
            for ch in p.changes:
                cur = "(absent)" if ch.current_value is None else ch.current_value
                tgt = "manual" if ch.type is ChangeType.GUIDED else ch.target_value
                gate = " *admin" if ch.requires_admin else ""
                print(f"    - {ch.description}{gate}: {cur} -> {tgt}")


def result_line(ok: bool, applied: bool, verified: bool, label: str, detail: str) -> None:
    label, detail = _e(label), _e(detail)
    if not applied:
        _p(f"  [dim]· {label}: {detail}[/dim]")
    elif ok and verified:
        _p(f"  [green]✓[/green] {label}")
    elif ok and not verified:
        _p(f"  [yellow]✓ applied but unverified[/yellow] {label} — {detail}")
    else:
        _p(f"  [red]✗[/red] {label} — {detail}")


def guided(label: str, instructions: str) -> None:
    _p(f"  [magenta]> MANUAL[/magenta] {_e(label)}")
    for line in (instructions or "").splitlines():
        if line.strip():
            _p(f"      [dim]{_e(line)}[/dim]")


def info(msg: str) -> None:
    _p(f"[cyan]{_e(msg)}[/cyan]")


def warn(msg: str) -> None:
    _p(f"[yellow]⚠ {_e(msg)}[/yellow]")


def error(msg: str) -> None:
    _p(f"[red]✗ {_e(msg)}[/red]")


def ask(prompt: str) -> str:
    try:
        return input(_strip(prompt))
    except EOFError:
        return ""
