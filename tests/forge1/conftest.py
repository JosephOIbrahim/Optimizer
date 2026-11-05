"""Pytest config for Forge-1 backend tests.

Injects the repo root onto sys.path so `from acceleration.* import ...`
works without an installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
