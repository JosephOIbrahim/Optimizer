"""sys.path injector so `from core.* import *` and `from acceleration.* import *`
resolve from the repo root regardless of pytest invocation directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
