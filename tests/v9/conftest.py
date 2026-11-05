"""CRUCIBLE — make the REAL `v9` source package importable under pytest.

Problem: this test directory is itself named `v9` and contains an
`__init__.py`, so pytest's default (prepend) import mode puts the `tests/`
directory on sys.path and `import v9` resolves to THIS test package
(tests/v9/__init__.py), shadowing the real source package at <repo>/v9.

Fix: put the repo root at the FRONT of sys.path, and evict any already-cached
`v9`/`v9.*` modules that were imported from anywhere other than <repo>/v9 so a
fresh, correct import wins. conftest.py is imported early enough that this runs
before the test modules' own imports resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_V9 = REPO_ROOT / "v9"

# 1. Repo root must come first so `import v9` finds <repo>/v9.
repo_str = str(REPO_ROOT)
while repo_str in sys.path:
    sys.path.remove(repo_str)
sys.path.insert(0, repo_str)

# 2. Drop any cached v9 modules that did NOT come from the real source tree
#    (e.g. the empty tests/v9 package picked up during collection).
for name in list(sys.modules):
    if name == "v9" or name.startswith("v9."):
        mod = sys.modules.get(name)
        file = getattr(mod, "__file__", None)
        try:
            from_real = file is not None and Path(file).resolve().is_relative_to(REAL_V9)
        except (ValueError, OSError):
            from_real = False
        if not from_real:
            del sys.modules[name]
