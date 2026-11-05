"""
Orchestrator: the loop that turns live machine state into applied, reversible
optimizations.

    probe → detect workload → gather agent proposals → resolve conflicts →
    snapshot originals → (preview | apply+verify+rollback | watch)

Modes:
    (default)   dry-run preview — reads everything, changes nothing
    --apply     one-shot apply with per-batch approval (skip with --yes)
    --watch     continuous watchdog: re-plans on workload switch, thermal back-off
    --restore   revert every recorded original, back to the machine's pre-optimizer state
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from . import agents as agents_pkg
from . import paths as paths_mod
from . import ui, workflows
from .base import Agent, Change, ChangeType, Context, Proposal, WorkloadMode, change_key
from .executor import Executor, run_ps
from .hardware_profile import PROFILE
from .probe import probe
from .safety import Safety
from .thermal import ThermalGovernor

# ROOT is the package directory — it holds config.toml, which is version-controlled
# configuration and belongs with the code.
ROOT = Path(__file__).resolve().parent.parent
# DATA is mutable state — transaction logs and restore points. It must NOT live
# under OneDrive: sync on an open append-only log tore 9 records out of
# transactions.jsonl before this moved. Resolution lives in paths.py.
DATA = paths_mod.DATA

DEFAULTS = {
    "min_confidence": 0.6,
    "interval_seconds": 20,
    "thermal_warn": PROFILE.thermal_c_warn,
    "thermal_reduce": PROFILE.thermal_c_reduce,
    "thermal_critical": PROFILE.thermal_c_critical,
    "vram_hard_pct": 0.96,
}

_RISK_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    p = ROOT / "config.toml"
    if p.exists():
        try:
            import tomllib
            data = tomllib.loads(p.read_text(encoding="utf-8"))
            for section in data.values():
                if isinstance(section, dict):
                    cfg.update({k: v for k, v in section.items() if k in cfg})
        except Exception as e:
            ui.warn(f"config.toml ignored: {e}")
    return cfg


def discover_agents() -> List[Agent]:
    found: List[Agent] = []
    for m in pkgutil.iter_modules(agents_pkg.__path__):
        try:
            mod = importlib.import_module(f".agents.{m.name}", __package__)
        except Exception as e:
            ui.warn(f"agent {m.name} failed to import: {e}")
            continue
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, Agent) and obj is not Agent:
                found.append(obj())
    return found


class Orchestrator:
    def __init__(self, cfg: dict, dry_run: bool = True):
        self.cfg = cfg
        self.executor = Executor(dry_run=dry_run)
        self.safety = Safety(DATA)
        self.agents = discover_agents()
        # seed originals so the watchdog never re-snapshots a target we already changed
        self.origins: Dict[str, Tuple[object, bool]] = self.safety.load_originals()

    # ---- planning --------------------------------------------------------
    def gather(self, ctx: Context) -> List[Proposal]:
        props: List[Proposal] = []
        for a in self.agents:
            try:
                props.extend(a.propose(ctx))
            except Exception as e:
                ui.warn(f"agent {a.name} propose() failed: {e}")
        props = [p for p in props
                 if p.confidence >= self.cfg["min_confidence"]
                 and ctx.workload in p.applies_when]
        return self._resolve(props)

    def _resolve(self, proposals: List[Proposal]) -> List[Proposal]:
        """Highest-confidence proposal claims a target; dedupe overlapping changes."""
        ordered = sorted(proposals, key=lambda p: p.confidence, reverse=True)
        seen: set = set()
        kept: List[Proposal] = []
        for p in ordered:
            fresh = [c for c in p.changes if change_key(c) not in seen]
            if not fresh:
                continue
            for c in fresh:
                seen.add(change_key(c))
            p.changes = fresh
            kept.append(p)
        # apply order: non-admin before admin, low risk before high
        kept.sort(key=lambda p: (p.gated(), max((_RISK_RANK[c.risk.value] for c in p.changes), default=0)))
        return kept

    def _snapshot(self, changes: List[Change]) -> None:
        for c in changes:
            k = change_key(c)
            if not self.executor.dry_run and k in self.origins:
                c.current_value, c.existed_before = self.origins[k]
            else:
                self.executor.snapshot(c)
                if not self.executor.dry_run:
                    self.origins[k] = (c.current_value, c.existed_before)

    # ---- apply -----------------------------------------------------------
    def apply(self, proposals: List[Proposal], ctx: Context) -> None:
        changes = [c for p in proposals for c in p.changes]
        self._snapshot(changes)
        self.safety.record_originals(changes, self.executor.dry_run)
        applied = verified = failed = 0
        for p in proposals:
            for c in p.changes:
                if c.type is ChangeType.GUIDED:
                    ui.guided(f"[{p.domain}] {c.description}", c.extra.get("instructions", ""))
                    self.safety.event("guided_shown", {"id": c.id})
                    continue
                res = self.executor.apply(c)
                if not res.changed:
                    ui.result_line(res.ok, False, True, f"[{p.domain}] {c.description}", res.detail)
                    continue
                ok_v = self.executor.verify(c)
                if res.ok and not ok_v:
                    rb = self.executor.revert(c)
                    ui.result_line(False, True, False, f"[{p.domain}] {c.description}",
                                   f"verify failed, rolled back ({rb.detail})")
                    failed += 1
                else:
                    ui.result_line(res.ok, True, ok_v, f"[{p.domain}] {c.description}", res.detail)
                    applied += 1
                    verified += int(ok_v)
                self.safety.event("applied", {"id": c.id, "ok": res.ok, "verified": ok_v})
        ui.info(f"\nApplied {applied} change(s), {verified} verified, {failed} rolled back.")

    # ---- run modes -------------------------------------------------------
    def preview(self, ctx: Context) -> List[Proposal]:
        proposals = self.gather(ctx)
        self._snapshot([c for p in proposals for c in p.changes])
        ui.plan_table(proposals)
        guided_changes = [(p, c) for p in proposals for c in p.changes if c.type is ChangeType.GUIDED]
        if guided_changes:
            ui.info("\nManual steps (surfaced, never auto-applied):")
            for p, c in guided_changes:
                ui.guided(f"[{p.domain}] {c.description}", c.extra.get("instructions", ""))
        gated = sum(1 for p in proposals if p.gated())
        if gated and not ctx.is_admin:
            ui.warn(f"{gated} proposal(s) need Administrator — re-run elevated to include them.")
        return proposals

    def watch(self, ctx: Context) -> None:
        interval = int(self.cfg["interval_seconds"])
        gov = ThermalGovernor(self.cfg, self.safety, self.executor.dry_run, ctx.is_admin)
        ui.info(f"Watchdog started — re-checking every {interval}s. Ctrl+C to stop.")
        ui.banner(ctx, f"watch → apply ({ctx.workload.value})")
        self.apply(self.gather(ctx), ctx)      # initial apply (writes the master restore point)
        last_mode = ctx.workload
        try:
            while True:
                time.sleep(interval)
                ctx = build_context()
                status = gov.step(ctx)
                if status and status.startswith("CRITICAL"):
                    ui.error(f"GPU {status} — pausing optimization dispatch until it cools.")
                    continue
                if status:
                    ui.warn(f"thermal: {status}")
                self._guardrails(ctx)
                if ctx.workload != last_mode:
                    ui.banner(ctx, f"watch → replan ({ctx.workload.value})")
                    self.apply(self.gather(ctx), ctx)
                    last_mode = ctx.workload
        except KeyboardInterrupt:
            ui.info("\nWatchdog stopped. Machine keeps the applied optimizations.")
            ui.info("Run  Restore-This-Machine.bat  to revert everything.")
        finally:
            gov.release_on_exit()   # always release the clock cap, even on crash

    def _guardrails(self, ctx: Context) -> None:
        if ctx.vram_used_gb >= ctx.vram_total_gb * self.cfg["vram_hard_pct"]:
            ui.warn(f"VRAM {ctx.vram_used_gb}/{ctx.vram_total_gb} GB — sysmem-fallback risk; "
                    "drop tile size / switch ComfyUI to INT4, or lower Karma samples.")
            self.safety.event("vram_pressure", {"used": ctx.vram_used_gb})


def build_context(force_mode: str = None) -> Context:
    ctx = probe()
    ctx.workload = WorkloadMode(force_mode) if force_mode else workflows.detect(ctx)
    return ctx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="MachineOptimizer2026",
                                 description="Machine-specific RTX 4090 / Threadripper VFX optimizer")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="preview only (default)")
    mode.add_argument("--apply", action="store_true", help="apply once, with approval")
    mode.add_argument("--watch", action="store_true", help="continuous watchdog")
    mode.add_argument("--restore", action="store_true", help="revert everything to original")
    mode.add_argument("--status", action="store_true", help="print machine state and exit")
    ap.add_argument("--yes", action="store_true", help="skip the apply confirmation")
    ap.add_argument("--interval", type=int, help="watchdog interval seconds")
    ap.add_argument("--min-confidence", type=float, help="drop proposals below this")
    ap.add_argument("--workload", choices=[m.value for m in WorkloadMode],
                    help="force a workload profile for preview")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.interval:
        cfg["interval_seconds"] = args.interval
    if args.min_confidence is not None:
        cfg["min_confidence"] = args.min_confidence

    # --restore must run a LIVE executor (dry-run would revert nothing yet consume the
    # restore point) — it is a mutating operation just like --apply.
    dry = not (args.apply or args.watch or args.restore)
    orch = Orchestrator(cfg, dry_run=dry)

    if args.restore:
        ui.info("Reverting all optimizations to the machine's original state...")
        for line in orch.safety.restore(orch.executor):
            print("  " + line)
        try:                       # clear any leaked GPU clock cap from a killed watchdog
            run_ps("nvidia-smi -rgc")
        except Exception:
            pass
        return 0

    ctx = build_context(args.workload)

    if args.status:
        ui.banner(ctx, "status")
        return 0

    if not orch.agents:
        ui.banner(ctx, "no-agents")
        ui.warn("No optimization agents installed yet in optimizer/agents/. "
                "Nothing to propose.")
        return 0

    if args.watch:
        orch.watch(ctx)
        return 0

    # dry-run or apply: always preview first
    ui.banner(ctx, "apply" if args.apply else "dry-run (preview)")
    proposals = orch.preview(ctx)
    if not args.apply:
        ui.info("\nPreview only. Re-run with --apply (or launch the .bat) to apply.")
        return 0
    if not proposals:
        ui.info("Nothing to apply.")
        return 0
    if not args.yes:
        ans = ui.ask(f"\nApply {sum(len(p.changes) for p in proposals)} change(s)? A restore "
                     f"point will be written first. [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            ui.info("Aborted. No changes made.")
            return 0
    orch.executor.dry_run = False
    orch.apply(proposals, ctx)
    ui.info("Done. Run  Restore-This-Machine.bat  to revert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
