"""`python -m v12` — same entry point as `python -m v12.cli`."""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
