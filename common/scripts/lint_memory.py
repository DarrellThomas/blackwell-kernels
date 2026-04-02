#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_FILES = [
    ROOT / "common/memory/factory_brain.py",
    ROOT / "common/memory/generate_summaries.py",
    ROOT / "common/memory/memory_cli.py",
    ROOT / "common/memory/memory_config.py",
    ROOT / "common/memory/memory_embeddings.py",
    ROOT / "common/memory/memory_experiments.py",
    ROOT / "common/memory/memory_helpers.py",
    ROOT / "common/memory/memory_ingest.py",
    ROOT / "common/memory/memory_issues.py",
    ROOT / "common/memory/memory_jobs.py",
    ROOT / "common/memory/memory_maintain.py",
    ROOT / "common/memory/memory_messages.py",
    ROOT / "common/memory/memory_search.py",
    ROOT / "common/memory/memory_server.py",
    ROOT / "common/memory/memory_stats.py",
    ROOT / "common/memory/memory_workers.py",
    ROOT / "common/memory/research_memory.py",
    ROOT / "common/tests/conftest.py",
    ROOT / "common/tests/test_memory_refactor.py",
    ROOT / "common/tests/test_memory_edge_cases.py",
    ROOT / "common/scripts/test_memory.py",
    ROOT / "common/scripts/lint_memory.py",
]
MODULES = [
    "common.memory.factory_brain",
    "common.memory.generate_summaries",
    "common.memory.memory_cli",
    "common.memory.memory_config",
    "common.memory.memory_embeddings",
    "common.memory.memory_experiments",
    "common.memory.memory_helpers",
    "common.memory.memory_ingest",
    "common.memory.memory_issues",
    "common.memory.memory_jobs",
    "common.memory.memory_maintain",
    "common.memory.memory_messages",
    "common.memory.memory_search",
    "common.memory.memory_server",
    "common.memory.memory_stats",
    "common.memory.memory_workers",
    "common.memory.research_memory",
]
UNQUALIFIED_FACTORY_IMPORT = re.compile(r"^\s*(?:from|import)\s+factory_brain\b")


def _lint_file_text(path: Path) -> list[str]:
    findings: list[str] = []
    text = path.read_text(errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if path.name != "lint_memory.py" and "/data/src/bwk" in line and not stripped.startswith("#"):
            findings.append(f"{path}:{lineno}: hardcoded repo root path")
        if UNQUALIFIED_FACTORY_IMPORT.search(line):
            findings.append(f"{path}:{lineno}: unqualified factory_brain import")
        if path.name not in {"research_memory.py", "lint_memory.py"} and "from common.memory.factory_brain import *" in line:
            findings.append(f"{path}:{lineno}: wildcard factory_brain import outside compatibility shim")
    return findings


def _lint_reject_files() -> list[str]:
    findings: list[str] = []
    for folder in (ROOT / "common/memory", ROOT / "common/tests", ROOT / "common/scripts"):
        for path in sorted(folder.glob("*.rej")):
            findings.append(f"{path}: patch reject file must be removed")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint common/memory Python modules and tests")
    parser.add_argument("--imports-only", action="store_true", help="Skip file text and py_compile checks")
    args = parser.parse_args(argv)

    findings: list[str] = []

    if not args.imports_only:
        for path in TARGET_FILES:
            if not path.is_file():
                findings.append(f"{path}: missing target file")
                continue
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                findings.append(f"{path}: py_compile failed: {exc.msg}")
            findings.extend(_lint_file_text(path))
        findings.extend(_lint_reject_files())

    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - reported via CLI
            findings.append(f"{name}: import failed: {exc.__class__.__name__}: {exc}")

    if findings:
        print("common/memory lint findings:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("common/memory lint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
