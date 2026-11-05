"""
Restore spine: an accumulating *master* restore point + a transaction log.

Because the watchdog re-applies the same targets across cycles, we must capture each
target's ORIGINAL value exactly once. The master restore point is keyed by change_key
and never overwrites an existing key, so it always describes the machine's pre-optimizer
state. `--restore` reverts every recorded original (newest-first) and archives the file.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import Change, ChangeType, Risk, Scope, change_key
from .executor import Executor


def _to_dict(c: Change) -> dict:
    d = asdict(c)
    d["type"] = c.type.value
    d["scope"] = c.scope.value
    d["risk"] = c.risk.value
    return d


def _from_dict(d: dict) -> Change:
    d = dict(d)
    d["type"] = ChangeType(d["type"])
    d["scope"] = Scope(d["scope"])
    d["risk"] = Risk(d.get("risk", "low"))
    return Change(**d)


class Safety:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.rp_dir = self.data_dir / "restore_points"
        self.rp_dir.mkdir(parents=True, exist_ok=True)
        self.master = self.rp_dir / "master.json"
        self.log_path = self.data_dir / "transactions.jsonl"

    def _load_master(self) -> Dict[str, dict]:
        if self.master.exists():
            return json.loads(self.master.read_text(encoding="utf-8")).get("changes", {})
        return {}

    def load_originals(self) -> Dict[str, Tuple[object, bool]]:
        """Seed the orchestrator so it won't re-snapshot targets we already changed."""
        out: Dict[str, Tuple[object, bool]] = {}
        for key, d in self._load_master().items():
            out[key] = (d.get("current_value"), d.get("existed_before", True))
        return out

    def record_originals(self, changes: List[Change], dry_run: bool) -> int:
        """Persist first-seen original snapshots. Call AFTER snapshot(), so current_value
        holds the true pre-apply value. Returns count of newly recorded targets."""
        if dry_run:
            return 0
        master = self._load_master()
        added = 0
        for c in changes:
            if c.type is ChangeType.GUIDED:
                continue
            k = change_key(c)
            if k not in master:
                master[k] = _to_dict(c)
                added += 1
        self.master.write_text(json.dumps({
            "created": datetime.now().isoformat(timespec="seconds"),
            "changes": master,
        }, indent=2), encoding="utf-8")
        if added:
            self.event("originals_recorded", {"added": added, "total": len(master)})
        return added

    def restore(self, executor: Executor) -> List[str]:
        master = self._load_master()
        if not master:
            return ["No master restore point — machine is at its original state (nothing recorded)."]
        results: List[str] = []
        failed: List[str] = []
        for k in reversed(list(master.keys())):
            c = _from_dict(master[k])
            res = executor.revert(c)
            results.append(f"[{'OK ' if res.ok else 'ERR'}] {c.id}: {res.detail}")
            self.event("reverted", {"id": c.id, "ok": res.ok})
            if not res.ok:
                failed.append(c.id)
        # Only consume the restore point when EVERYTHING reverted — otherwise keep it so
        # the user can re-run (after elevation/reboot) without losing the record of originals.
        if failed:
            results.append(f"{len(failed)} change(s) failed to revert ({', '.join(failed)}). "
                           "Restore point kept — re-run Restore-This-Machine.bat (as admin) to retry.")
        else:
            archived = self.rp_dir / f"master.restored_{datetime.now():%Y%m%d_%H%M%S}.json"
            self.master.rename(archived)
            results.append(f"All changes reverted. Restore point consumed -> {archived.name}")
        return results

    def has_restore_point(self) -> bool:
        return self.master.exists() and bool(self._load_master())

    def event(self, kind: str, payload: dict) -> None:
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, **payload}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
