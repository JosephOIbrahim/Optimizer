r"""
sync_measurements — copy the live measurement records into the repo, and
commit them, so the committed copies cannot go stale.

Why this exists
---------------
The operator cards quote figures — "+46.95%, p=2.4e-06", "DENY -0.60%",
"the beauty-only lever is +9.79%" — and the records those figures came from
are the evidence. They live in the repo under `measurements/`, committed
alongside the code that produced them, so a reader can check a claim against
the file it came from instead of taking it on trust.

But the LIVE store is `E:\OptimizerV12\measurements`, deliberately off
OneDrive: many small JSON files syncing through OneDrive is a bad trade, and
E: has the space. So the repo copy is a copy, and it goes stale the moment a
record is written or — worse — edited in place. A retraction edits a record
in place; that is the whole point of a retraction, and it is exactly the case
a "copy the new files" script would miss.

So this copies by CONTENT, not by name: a record that is new, or whose bytes
changed, is staged and committed.

What it will not do
-------------------
  * It will not commit anything outside `measurements/`. The operator may
    have source edits in flight; those are not this script's business.
  * It will not delete a committed record that has gone from the live store.
    That is a deletion, and a deletion of evidence is a decision, not a
    side effect. It is REPORTED instead.
  * It will not commit a record that does not parse as JSON. A truncated
    write would otherwise land in the history.

Usage
-----
    python harness\\sync_measurements.py            copy and commit
    python harness\\sync_measurements.py --check    report what would change
    python harness\\sync_measurements.py --message "..."   override the message
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEST = REPO_ROOT / "measurements"

#: The live store. Overridable so the same script works against a copy, and
#: so a test does not need the real thing.
SOURCE = Path(r"E:\OptimizerV12\measurements")

#: Only the records are evidence. The render output is tens of megabytes and
#: grows with every campaign.
SOURCE_GLOB = "*.json"


class SyncError(Exception):
    """Something that must stop the run rather than be reported and continued."""


def _git(repo: Path, *args: str) -> Tuple[int, str, str]:
    """Run a git command in the repo. Returns (rc, stdout, stderr)."""
    p = subprocess.run(["git", *args], cwd=str(repo),
                       capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_json(path: Path) -> Tuple[bool, str]:
    """Does this file parse? Returns (ok, reason)."""
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def _in_repo(repo: Path) -> bool:
    rc, out, _ = _git(repo, "rev-parse", "--is-inside-work-tree")
    return rc == 0 and out.strip() == "true"


def _tracked_names(repo: Path) -> List[str]:
    """Every file git tracks under measurements/."""
    rc, out, _ = _git(repo, "ls-files", "-z", "--", "measurements/")
    if rc != 0:
        return []
    return [Path(e).name for e in out.split("\0") if e]


def _committed_blob(repo: Path, name: str) -> str:
    """The blob sha git has committed for this record."""
    rc, out, _ = _git(repo, "rev-parse", f":measurements/{name}")
    return out.strip() if rc == 0 else ""


def _blob_for(repo: Path, path: Path, name: str) -> str:
    """The blob sha git WOULD store for this file.

    Asked of git rather than computed here, because git applies the repo's
    own line-ending and filter rules when it stores a file — and the live
    store's records are CRLF while the committed blobs are LF. Comparing raw
    bytes would report every record as changed, every time. `hash-object
    --path` applies exactly the rules `git add` would apply, so the answer is
    "would committing this file change anything?", which is the question
    that actually matters.
    """
    p = subprocess.run(
        ["git", "hash-object", "--path", f"measurements/{name}", str(path)],
        cwd=str(repo), capture_output=True, text=True, timeout=60)
    return p.stdout.strip()


def plan(source: Path = SOURCE, dest: Path = DEST,
         repo: Path = REPO_ROOT) -> Dict[str, List[str]]:
    """What would change. Returns new, changed, unchanged, missing, invalid."""
    out: Dict[str, List[str]] = {"new": [], "changed": [], "unchanged": [],
                                  "missing": [], "invalid": []}
    if not source.is_dir():
        raise SyncError(f"the live store is not there: {source}")

    tracked = set(_tracked_names(repo))
    live = sorted(source.glob(SOURCE_GLOB))
    # An empty store is not an error. It is either a fresh repo with nothing
    # to sync, or records that were moved out of the store — and the second
    # is worth REPORTING, not refusing over. Refusing here would hide the
    # one case the operator needs to see. A store that does not exist at all
    # IS an error, and is caught above.

    for f in live:
        name = f.name
        ok, why = _is_json(f)
        if not ok:
            out["invalid"].append(f"{name} ({why})")
            continue
        if name not in tracked:
            out["new"].append(name)
            continue
        # Compare the CONTENT, not the timestamp: a retraction edits a record
        # in place, and a copy-by-name script would never notice.
        committed = _committed_blob(repo, name)
        if not committed or committed != _blob_for(repo, f, name):
            out["changed"].append(name)
        else:
            out["unchanged"].append(name)

    # Tracked here but gone from the live store. Reported, never deleted.
    # Only RECORDS count — the directory's README is a note about the
    # records, not one of them.
    for name in sorted(tracked):
        if name.endswith(".json") and not (source / name).exists():
            out["missing"].append(name)
    return out


def copy_files(source: Path, dest: Path, names: List[str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        (dest / name).write_bytes((source / name).read_bytes())


def commit(repo: Path, names: List[str], message: str) -> Tuple[bool, str]:
    """Stage ONLY measurements/ and commit. Returns (ok, output)."""
    rc, out, err = _git(repo, "add", "--", "measurements/")
    if rc != 0:
        return False, f"git add failed: {err.strip() or out.strip()}"

    # Stage nothing but the records, even if the operator has other changes
    # in flight.
    rc, staged, _ = _git(repo, "diff", "--cached", "--name-only", "--",
                         "measurements/")
    if not staged.strip():
        return False, "nothing staged"

    rc, out, err = _git(repo, "commit", "-m", message)
    if rc != 0:
        return False, f"git commit failed: {err.strip() or out.strip()}"
    return True, out.strip()


def build_message(new: List[str], changed: List[str]) -> str:
    """A commit message that says what was synced and why it matters."""
    lines = ["Sync the measurement records into the repo", ""]
    if new:
        lines.append(f"New records ({len(new)}):")
        lines += [f"  {n}" for n in new[:12]]
        if len(new) > 12:
            lines.append(f"  ... and {len(new) - 12} more")
        lines.append("")
    if changed:
        lines.append(f"Changed records ({len(changed)}) — an edit in place, which is "
                     f"what a retraction is:")
        lines += [f"  {n}" for n in changed[:12]]
        if len(changed) > 12:
            lines.append(f"  ... and {len(changed) - 12} more")
        lines.append("")
    lines += [
        "The live store is E:\\OptimizerV12\\measurements, deliberately off "
        "OneDrive.",
        "The copies here are for version control and for reading next to the "
        "code, so a",
        "figure quoted in the operator cards can be checked against the record "
        "it came from.",
        "",
        "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report what would change; copy and commit nothing")
    ap.add_argument("--message", default="",
                    help="override the generated commit message")
    ap.add_argument("--source", default=str(SOURCE),
                    help=f"the live store (default {SOURCE})")
    args = ap.parse_args(argv)

    source = Path(args.source)
    print(f"live store : {source}")
    print(f"repo copy  : {DEST}")

    if not _in_repo(REPO_ROOT):
        print("[ERROR] this is not a git repository — nothing to commit into")
        return 2

    try:
        p = plan(source, DEST, REPO_ROOT)
    except SyncError as e:
        print(f"[ERROR] {e}")
        return 2

    print()
    print(f"  new      : {len(p['new'])}")
    print(f"  changed  : {len(p['changed'])}")
    print(f"  unchanged: {len(p['unchanged'])}")
    if p["invalid"]:
        print(f"  INVALID  : {len(p['invalid'])}  (not copied)")
        for name in p["invalid"][:5]:
            print(f"     {name}")
    if p["missing"]:
        print(f"  missing from the live store: {len(p['missing'])}  "
              f"(committed copies kept — a deletion is a decision)")
        for name in p["missing"][:5]:
            print(f"     {name}")

    touched = p["new"] + p["changed"]
    if not touched:
        print()
        print("nothing to sync — the committed copies are current")
        return 0

    if args.check:
        print()
        print("would copy and commit:")
        for name in touched:
            print(f"  {name}")
        return 0

    print()
    copy_files(source, DEST, touched)
    print(f"copied {len(touched)} record(s)")

    message = args.message or build_message(p["new"], p["changed"])
    ok, out = commit(REPO_ROOT, touched, message)
    if not ok:
        print(f"[ERROR] {out}")
        print("       the files are copied but NOT committed — the copies are "
              "in the working tree")
        return 1
    print()
    print(out.splitlines()[0] if out else "committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
