#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License.

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tests" / "test_speed.py"), run_name="__main__")
