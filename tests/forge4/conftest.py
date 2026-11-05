"""Pytest config for Forge-4 cache + background-compiler tests.

Injects the repo root onto sys.path so `from core.* import ...` and
`from acceleration.* import ...` work without an installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
