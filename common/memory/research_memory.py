#!/usr/bin/env python3
"""Backward-compatibility shim."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from common.memory.factory_brain import *  # noqa: F401,F403
from common.memory.factory_brain import main

if __name__ == "__main__":
    sys.exit(main())
