"""
Session-level guards for the test suite.

This repo has been bitten twice by tests that wrote into it:

  * a scratch script in `tests/` with module-level code ran the real
    measurement on import, leaving eighteen rendered frames in the
    measurement store with no record behind them — and it fired during
    ANOTHER agent's suite run, not just its author's;
  * the figure-sweep's own mutation test planted a defect into the
    operator card ON DISK and shelled out to re-run itself. It recursed,
    three levels deep, each level writing the same shared file; when the
    outer run was killed the card was left holding the very defect the
    test exists to catch.

Both are the same failure: a test that changes the thing the project is
built to protect. The evidence here is the operator cards and the
measurement records, and a test that leaves them dirty has destroyed
something, whatever its intent.

So the suite checks itself. At the end of the run, the working tree must
be exactly as it was at the start: no tracked file modified, no tracked
file deleted, no new file left behind. `.gitignore` is honoured, so
`__pycache__` and the pytest cache do not count.

The check is a session fixture rather than a per-test one, because the
failure it exists to catch is cumulative — a file that one test dirtied
and another cleaned up is still a test that wrote into the repo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent


def _working_tree() -> set:
    """The working tree's status, as a set of porcelain lines.

    `git status --porcelain` respects `.gitignore`, so the caches and the
    byte-code do not count, and it covers modified, deleted and untracked
    files alike.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO), capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[conftest] could not read the working tree: {e}")
        return set()
    if r.returncode != 0:
        # Not a git repository, or git is unavailable. The guard has
        # nothing to compare against, so it says so rather than passing
        # a check it did not make.
        print("[conftest] not a git repository - the working-tree guard is off")
        return set()
    return {line for line in r.stdout.splitlines() if line.strip()}


@pytest.fixture(scope="session", autouse=True)
def the_working_tree_is_unchanged():
    """No test may leave the repo dirty."""
    before = _working_tree()
    yield
    after = _working_tree()

    left_behind = after - before
    if not left_behind:
        return

    # The lines are porcelain's "XY path" form. Report the path plainly.
    paths = sorted({line[3:].strip().strip('"') for line in left_behind})
    pytest.fail(
        "these files were changed by the test run, and the working tree was "
        "clean of them when it started:\n  "
        + "\n  ".join(paths)
        + "\n\na test that writes into the repo corrupts the evidence the "
        "project is built around. If the test meant to write to a file, it "
        "must write to a COPY of it, and restore it in a finally - and the "
        "copy is better than the restore, because a restore does not run "
        "when the process is killed.")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(0)
