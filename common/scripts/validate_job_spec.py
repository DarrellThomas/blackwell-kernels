#!/usr/bin/env python3
"""Validate a repo-local job spec against the shared factory schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
COMMON_DIR = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = COMMON_DIR / "docs" / "job_specs" / "job_spec_schema.json"


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON in {path}: {exc}")


def validate_document(document: dict[str, Any], schema: dict[str, Any]) -> tuple[list[str], str | None]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return [], "missing dependency 'jsonschema'. Install with: pip install jsonschema"

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda err: list(err.path))
    issues = []
    for err in errors:
        path = ".".join(str(part) for part in err.path) or "<root>"
        issues.append(f"path={path}: {err.message}")
    return issues, None


def validate_job_spec_document(job_doc: dict[str, Any], schema_doc: dict[str, Any]) -> list[str]:
    issues, unavailable = validate_document(job_doc, schema_doc)
    if unavailable:
        return [unavailable]

    weighted_mix = job_doc.get("contracts", {}).get("performance", {}).get("weighted_mix")
    if isinstance(weighted_mix, dict) and weighted_mix:
        total = sum(float(value) for value in weighted_mix.values())
        if abs(total - 1.0) > 1e-9:
            issues.append(
                "path=contracts.performance.weighted_mix: values must sum to 1.0 "
                f"but got {total:.12f}"
            )
    return issues


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python3 common/scripts/validate_job_spec.py <job_spec.json> [schema.json]"
        )

    job_path = Path(sys.argv[1]).resolve()
    schema_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_SCHEMA_PATH.resolve()

    job = load_json(job_path)
    schema = load_json(schema_path)
    if not isinstance(job, dict):
        raise SystemExit(f"ERROR: expected top-level object in {job_path}")
    if not isinstance(schema, dict):
        raise SystemExit(f"ERROR: expected top-level object in {schema_path}")

    issues = validate_job_spec_document(job, schema)
    if issues:
        print("INVALID: job spec failed validation")
        for idx, issue in enumerate(issues, start=1):
            print(f"[{idx}] {issue}")
        return 1

    print("VALID: job spec passed schema validation")
    print(f"job_id={job['job_id']}")
    print(f"title={job['title']}")
    print(f"hardware_target={job['hardware_target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
