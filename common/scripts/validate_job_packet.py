#!/usr/bin/env python3
"""Validate a generated watchdog job packet."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON in {path}: {exc}")


def main() -> int:
    script_path = Path(__file__).resolve()
    common_dir = script_path.parent.parent
    default_schema = common_dir / "docs" / "job_packets" / "job_packet_schema.json"

    packet_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("job_packet.json")
    schema_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_schema

    packet = load_json(packet_path)
    schema = load_json(schema_path)

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise SystemExit(
            "ERROR: missing dependency 'jsonschema'. Install with: pip install jsonschema"
        ) from exc

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(packet), key=lambda err: list(err.path))
    if errors:
        print("INVALID: watchdog job packet failed validation")
        for index, err in enumerate(errors, start=1):
            path = ".".join(str(part) for part in err.path) or "<root>"
            print(f"[{index}] path={path}")
            print(f"    message={err.message}")
        return 1

    worker = packet.get("worker", {}).get("name", "")
    status = packet.get("assignment_status", "")
    job = packet.get("job") or {}
    print("VALID: watchdog job packet passed schema validation")
    print(f"worker={worker}")
    print(f"assignment_status={status}")
    if job:
        print(f"job_id={job.get('id')}")
        print(f"title={job.get('title', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
