#!/usr/bin/env python3
"""Build a validated watchdog job packet for a worker slot."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_job_spec import DEFAULT_SCHEMA_PATH as DEFAULT_JOB_SPEC_SCHEMA_PATH
from validate_job_spec import validate_job_spec_document


SCRIPT_PATH = Path(__file__).resolve()
COMMON_DIR = SCRIPT_PATH.parent.parent
BWK_ROOT = COMMON_DIR.parent
DEFAULT_SCHEMA_PATH = COMMON_DIR / "docs" / "job_packets" / "job_packet_schema.json"
DEFAULT_JOB_SPEC_VALIDATOR_PATH = COMMON_DIR / "scripts" / "validate_job_spec.py"
DEFAULT_WORKTREE_ROOT = Path(
    os.environ.get("WATCHDOG_WORKTREE_ROOT", str(BWK_ROOT / "data" / "watchdog-worktrees"))
)
TERMINAL_STATES = {"shipped", "converged", "parked", "abandoned"}
PRIMARY_FILES_HEADING_RE = re.compile(r"^(#+\s*)?primary files\s*:?\s*$", re.IGNORECASE)
SECTION_RE = re.compile(r"^(#+\s*)?[A-Za-z][A-Za-z0-9 _/()'\-]*\s*:?\s*$")
PATTERN_TOKENS = ("...", "*", "?", "[")

sys.path.insert(0, str(COMMON_DIR / "memory"))

from factory_brain import (  # noqa: E402
    ResearchMemory,
    describe_job_shipping,
    resolve_job_project_dir,
    resolve_job_source_path,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def git_branch(path: Path) -> str | None:
    try:
        return git_output(path, "symbolic-ref", "--quiet", "--short", "HEAD")
    except Exception:
        return None


def git_repo_root(path: Path) -> Path | None:
    try:
        return Path(git_output(path, "rev-parse", "--show-toplevel")).resolve()
    except Exception:
        return None


def job_sort_key(job: dict[str, Any]) -> tuple[int, str, int]:
    raw_priority = str(job.get("priority", "")).strip()
    priority = int(raw_priority) if raw_priority.isdigit() else 99
    return priority, str(job.get("updated_at", "")), int(job.get("id", 0))


def choose_job(mem: ResearchMemory, worker: str, explicit_job_id: int | None) -> dict[str, Any] | None:
    if explicit_job_id is not None:
        return mem.get_job(explicit_job_id)
    rows = [job for job in mem.get_jobs(execution_lane="active", assigned_to=worker) if job["state"] not in TERMINAL_STATES]
    rows.sort(key=job_sort_key)
    return rows[0] if rows else None


def as_int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def as_text(value: Any) -> str:
    return "" if value is None else str(value)


def stringify_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def rebase_into_worktree(path: Path | None, shared_repo_root: Path | None, worktree_root: Path | None) -> Path | None:
    if path is None or shared_repo_root is None or worktree_root is None:
        return path
    try:
        rel = path.resolve().relative_to(shared_repo_root.resolve())
    except Exception:
        return path
    return worktree_root / rel


def clean_shipping_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(info.get("ok")),
        "mode": as_text(info.get("mode")),
        "detail": as_text(info.get("detail")),
        "error": as_text(info.get("error")),
        "project_dir": stringify_path(info.get("project_dir")),
    }


def load_json_with_error(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = load_json(path)
    except FileNotFoundError:
        return None, f"file not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"expected top-level object in {path}"
    return payload, None


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


def build_repo_local_spec(
    job: dict[str, Any] | None,
    shared_project_dir: Path | None,
    shared_repo_root: Path | None,
    worktree_root: Path | None,
) -> dict[str, Any]:
    base = {
        "present": False,
        "job_json_path": None,
        "schema_path": None,
        "validator_path": None,
        "validation_status": "not_present",
        "validation_errors": [],
        "document": None,
    }
    if job is None or shared_project_dir is None:
        return base

    shared_job_json = shared_project_dir / f"docs/Job{job['id']}" / f"job_{job['id']}.json"
    if not shared_job_json.exists():
        return base

    shared_schema = DEFAULT_JOB_SPEC_SCHEMA_PATH
    shared_validator = DEFAULT_JOB_SPEC_VALIDATOR_PATH
    job_json_path = rebase_into_worktree(shared_job_json, shared_repo_root, worktree_root)
    schema_path = shared_schema.resolve()
    validator_path = shared_validator.resolve()

    actual_job_json = job_json_path if job_json_path and job_json_path.exists() else shared_job_json
    actual_schema = schema_path if schema_path.exists() else shared_schema

    payload, payload_error = load_json_with_error(actual_job_json)
    if payload_error is not None:
        return {
            **base,
            "present": True,
            "job_json_path": stringify_path(job_json_path or shared_job_json),
            "schema_path": stringify_path(schema_path if shared_schema.exists() else None),
            "validator_path": stringify_path(validator_path if shared_validator.exists() else None),
            "validation_status": "invalid_json",
            "validation_errors": [payload_error],
            "document": None,
        }

    if not actual_schema.exists():
        return {
            **base,
            "present": True,
            "job_json_path": stringify_path(job_json_path or shared_job_json),
            "schema_path": None,
            "validator_path": stringify_path(validator_path if shared_validator.exists() else None),
            "validation_status": "missing_schema",
            "validation_errors": [f"schema not found: {shared_schema}"],
            "document": payload,
        }

    schema_doc, schema_error = load_json_with_error(actual_schema)
    if schema_error is not None:
        return {
            **base,
            "present": True,
            "job_json_path": stringify_path(job_json_path or shared_job_json),
            "schema_path": stringify_path(schema_path or shared_schema),
            "validator_path": stringify_path(validator_path if shared_validator.exists() else None),
            "validation_status": "invalid_schema",
            "validation_errors": [schema_error],
            "document": payload,
        }

    issues = validate_job_spec_document(payload, schema_doc)
    status = "valid"
    if issues:
        status = "validator_unavailable" if len(issues) == 1 and issues[0].startswith("missing dependency") else "invalid"
    return {
        **base,
        "present": True,
        "job_json_path": stringify_path(job_json_path or shared_job_json),
        "schema_path": stringify_path(schema_path or shared_schema),
        "validator_path": stringify_path(validator_path if shared_validator.exists() else None),
        "validation_status": status,
        "validation_errors": issues,
        "document": payload,
    }


def strip_repo_prefix(declared_path: str, shared_repo_root: Path | None) -> str:
    cleaned = declared_path.strip().strip("`").strip()
    if shared_repo_root is None:
        return cleaned
    repo_name = shared_repo_root.name + "/"
    return cleaned[len(repo_name):] if cleaned.startswith(repo_name) else cleaned


def extract_markdown_primary_files(spec_text: str) -> list[str]:
    if not spec_text.strip():
        return []
    paths: list[str] = []
    capture = False
    for raw_line in spec_text.splitlines():
        stripped = raw_line.strip()
        if not capture:
            if PRIMARY_FILES_HEADING_RE.match(stripped):
                capture = True
            continue

        if not stripped:
            if paths:
                break
            continue
        if stripped.startswith("#") or SECTION_RE.match(stripped):
            break

        match = re.match(r"^[-*]\s+(.*)$", stripped) or re.match(r"^\d+\.\s+(.*)$", stripped)
        if not match:
            if paths:
                break
            continue
        candidate = match.group(1).strip().strip("`")
        if candidate:
            paths.append(candidate)
    return paths


def build_file_hint_entry(
    declared_path: str,
    source: str,
    shared_repo_root: Path | None,
    worktree_root: Path | None,
) -> dict[str, Any]:
    normalized = strip_repo_prefix(declared_path, shared_repo_root)
    kind = "pattern" if any(token in normalized for token in PATTERN_TOKENS) else "file"
    shared_path: Path | None = None
    worktree_path: Path | None = None
    relative_path: str | None = None

    raw_path = Path(normalized)
    if raw_path.is_absolute():
        shared_path = raw_path
    elif kind == "file" and shared_repo_root is not None:
        relative_path = normalized
        shared_path = shared_repo_root / normalized
    else:
        relative_path = normalized if not raw_path.is_absolute() else None

    if shared_path is not None and worktree_root is not None and shared_repo_root is not None:
        worktree_path = rebase_into_worktree(shared_path, shared_repo_root, worktree_root)

    return {
        "declared_path": declared_path,
        "relative_path": relative_path,
        "kind": kind,
        "shared_path": stringify_path(shared_path) if kind == "file" else None,
        "worktree_path": stringify_path(worktree_path) if kind == "file" else None,
        "exists_in_shared_repo": bool(shared_path and shared_path.exists()),
        "exists_in_worktree": bool(worktree_path and worktree_path.exists()),
        "sources": [source],
    }


def build_local_file_hints(
    job: dict[str, Any] | None,
    shared_repo_root: Path | None,
    worktree_root: Path | None,
    repo_local_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    sources: list[tuple[str, list[str]]] = []

    if job is not None:
        sources.append(("job_spec_markdown", extract_markdown_primary_files(as_text(job.get("spec")))))

    if repo_local_spec.get("present") and isinstance(repo_local_spec.get("document"), dict):
        primary_files = repo_local_spec["document"].get("primary_files")
        if isinstance(primary_files, list):
            sources.append(("repo_local_spec.primary_files", [as_text(item) for item in primary_files if as_text(item).strip()]))

    for source, declared_paths in sources:
        for declared_path in declared_paths:
            entry = build_file_hint_entry(declared_path, source, shared_repo_root, worktree_root)
            key = entry["relative_path"] or entry["declared_path"]
            existing = merged.get(key)
            if existing is None:
                merged[key] = entry
                continue
            if source not in existing["sources"]:
                existing["sources"].append(source)
            existing["exists_in_shared_repo"] = existing["exists_in_shared_repo"] or entry["exists_in_shared_repo"]
            existing["exists_in_worktree"] = existing["exists_in_worktree"] or entry["exists_in_worktree"]

    hints = list(merged.values())
    hints.sort(key=lambda entry: (entry["relative_path"] or entry["declared_path"]))
    for entry in hints:
        entry["sources"].sort()
    return hints


def build_protocol(
    worker: str,
    job: dict[str, Any] | None,
    common_dir: Path,
    recent: int,
    repo_local_spec: dict[str, Any],
) -> dict[str, Any]:
    if job is None:
        return {
            "heartbeat_working": None,
            "refresh_commands": {
                "job_show": None,
                "messages": None,
                "experiment_summary": None,
                "validate_repo_local_spec": None,
            },
            "done": {"heartbeat": None, "message": None},
            "check_my_work": {"heartbeat": None, "message": None},
            "problem": {
                "heartbeat": None,
                "message": None,
                "guidance": "No active assignment. Wait for watchdog/manual reassignment instead of editing shared checkouts.",
            },
            "research_session": "researcher",
            "research_guidance": "Ping the researcher tmux session or open a research_request message whenever you need bounded external context.",
        }

    job_id = int(job["id"])
    kernel = as_text(job.get("kernel_type")) or worker
    refresh_validate = None
    if repo_local_spec.get("present") and repo_local_spec.get("job_json_path") and repo_local_spec.get("schema_path"):
        validator_path = repo_local_spec.get("validator_path")
        if validator_path:
            refresh_validate = (
                f"python3 {validator_path} {repo_local_spec['job_json_path']} {repo_local_spec['schema_path']}"
            )

    brain = common_dir / "memory" / "factory_brain.py"
    return {
        "heartbeat_working": (
            f"python3 {brain} heartbeat {worker} --job {job_id} --state working "
            "--task 'resuming active job and reading spec'"
        ),
        "refresh_commands": {
            "job_show": f"python3 {brain} job-show {job_id}",
            "messages": f"python3 {brain} messages --job {job_id}",
            "experiment_summary": f"python3 {brain} experiment-summary --kernel {kernel} --recent {recent}",
            "validate_repo_local_spec": refresh_validate,
        },
        "done": {
            "heartbeat": (
                f"python3 {brain} heartbeat {worker} --job {job_id} --state complete "
                "--task 'done: <summary>; commit <hash>'"
            ),
            "message": (
                f"python3 {brain} message-create --from {worker} --job {job_id} "
                "--subject 'done' --body 'commit=<hash>; summary=<summary>' --type info"
            ),
        },
        "check_my_work": {
            "heartbeat": (
                f"python3 {brain} heartbeat {worker} --job {job_id} --state complete "
                "--task 'check my work: <summary>; commit <hash>'"
            ),
            "message": (
                f"python3 {brain} message-create --from {worker} --job {job_id} "
                "--subject 'check my work' --body 'commit=<hash>; summary=<summary>' --type feedback"
            ),
        },
        "problem": {
            "heartbeat": (
                f"python3 {brain} heartbeat {worker} --job {job_id} --state working "
                "--task 'problem: <summary>'"
            ),
            "message": (
                f"python3 {brain} message-create --from {worker} --job {job_id} "
                "--subject '<problem>' --body '<details>' --type blocker"
            ),
            "guidance": "If you are blocked or need research, do not guess. Leave heartbeat=working and open a blocker or question message.",
        },
        "research_session": "researcher",
        "research_guidance": "Ping the researcher tmux session 'researcher' or open a research_request message whenever you hit stuck_needs_research.",
    }


def build_instructions(job: dict[str, Any] | None, repo_local_spec: dict[str, Any]) -> dict[str, Any]:
    if job is None:
        return {
            "summary": "No active job is assigned to this worker slot. Stay in the dedicated worktree, avoid shared checkout edits, and wait for reassignment.",
            "constraints": [
                "Assume zero prior model context.",
                "Do not edit the shared repo root while idle.",
                "Do not start generic exploration without an active assignment."
            ],
            "refresh_order": [
                "Read this packet.",
                "Wait for watchdog/manual reassignment.",
                "Do not modify repo state until a job is assigned."
            ],
            "handoff_signals": ["done", "check my work", "problem"],
        }

    constraints = [
        "Assume zero prior model context.",
        "Stay inside the dedicated watchdog worktree; do not edit the shared checkout.",
        "Do not begin with generic codebase exploration.",
        "Make the smallest change needed for the active job.",
        "Refresh heartbeat during long work and report status back to the DB.",
        "Use common/csrc and common/docs for shared primitives and reference material."
    ]
    if repo_local_spec.get("present") and repo_local_spec.get("validation_status") == "valid":
        constraints.append("Treat the validated repo-local structured spec as the bounded contract before editing.")

    return {
        "summary": "Read this packet first. It is the structured handoff for the active watchdog assignment.",
        "constraints": constraints,
        "refresh_order": [
            "Run protocol.heartbeat_working before substantial exploration.",
            "Read job, context.open_messages, and context.experiment_summary.",
            "Open the files named in context.local_file_hints.",
            "If repo_local_spec.present and validation_status is valid, use that spec as the bounded contract.",
            "Use protocol.refresh_commands for any DB refresh."
        ],
        "handoff_signals": ["done", "check my work", "problem"],
    }


def validate_packet(packet: dict[str, Any], schema_path: Path) -> list[str]:
    schema = load_json(schema_path)
    issues, unavailable = validate_document(packet, schema)
    if unavailable:
        return [unavailable]
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, help="Worker slot name, e.g. cx3")
    parser.add_argument("--job", type=int, default=None, help="Optional explicit job id")
    parser.add_argument("--worktree", default=None, help="Dedicated worktree path for the worker")
    parser.add_argument("--branch", default=None, help="Optional explicit branch name")
    parser.add_argument("--shared-repo-root", default=None, help="Shared repo root that owns the job")
    parser.add_argument("--output", default=None, help="Output packet path")
    parser.add_argument("--recent", type=int, default=8, help="Recent experiment count for summary")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Packet schema path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worker = args.worker
    worktree_root = Path(args.worktree).resolve() if args.worktree else (DEFAULT_WORKTREE_ROOT / worker / "current").resolve()
    output_path = Path(args.output).resolve() if args.output else (DEFAULT_WORKTREE_ROOT / worker / "job_packet.json").resolve()
    shared_repo_root = Path(args.shared_repo_root).resolve() if args.shared_repo_root else None
    branch = args.branch
    if branch is None and worktree_root.exists():
        branch = git_branch(worktree_root)

    mem = ResearchMemory()
    try:
        job = choose_job(mem, worker, args.job)
        if args.job is not None and job is None:
            raise SystemExit(f"ERROR: job not found: {args.job}")

        shared_project_dir = resolve_job_project_dir(job) if job is not None else None
        if shared_repo_root is None and shared_project_dir is not None:
            shared_repo_root = git_repo_root(shared_project_dir) or shared_project_dir.resolve()
        if shared_repo_root is None and worktree_root.exists():
            shared_repo_root = git_repo_root(worktree_root) or worktree_root
        if shared_repo_root is not None:
            if shared_project_dir is None:
                shared_project_dir = shared_repo_root
            else:
                try:
                    shared_project_dir.resolve().relative_to(shared_repo_root.resolve())
                except Exception:
                    shared_project_dir = shared_repo_root

        shared_source_path = resolve_job_source_path(job) if job is not None else None
        worktree_project_dir = rebase_into_worktree(shared_project_dir, shared_repo_root, worktree_root)
        worktree_source_path = rebase_into_worktree(shared_source_path, shared_repo_root, worktree_root)
        shipping = clean_shipping_info(describe_job_shipping(job)) if job is not None else {
            "ok": True,
            "mode": "idle",
            "detail": "no active assignment",
            "error": "",
            "project_dir": stringify_path(shared_project_dir),
        }

        repo_local_spec = build_repo_local_spec(job, shared_project_dir, shared_repo_root, worktree_root)
        local_file_hints = build_local_file_hints(job, shared_repo_root, worktree_root, repo_local_spec)
        open_messages = mem.get_messages(status="open", job_id=job["id"]) if job is not None else []
        recent_messages = mem.get_messages(job_id=job["id"])[:20] if job is not None else []
        experiment_summary = (
            mem.summarize_experiments(kernel_type=as_text(job.get("kernel_type")) or worker, job_id=job["id"], recent=args.recent)
            if job is not None else None
        )

        packet = {
            "schema_version": "1.0",
            "packet_type": "watchdog_job_packet",
            "generated_at": iso_now(),
            "assignment_status": "active" if job is not None else "idle",
            "worker": {
                "name": worker,
                "assigned_job_id": int(job["id"]) if job is not None else None,
                "shared_repo_root": stringify_path(shared_repo_root),
                "worktree_root": str(worktree_root),
                "branch": branch,
                "packet_path": str(output_path),
            },
            "job": None if job is None else {
                "id": int(job["id"]),
                "title": as_text(job.get("title")),
                "description": as_text(job.get("description")),
                "state": as_text(job.get("state")),
                "phase": as_text(job.get("phase")),
                "job_type": as_text(job.get("job_type") or "kernel"),
                "kernel_type": as_text(job.get("kernel_type")),
                "assigned_to": as_text(job.get("assigned_to")),
                "execution_lane": as_text(job.get("execution_lane")),
                "priority": as_int_or_none(job.get("priority")),
                "version": as_text(job.get("version")),
                "factory_mode": as_text(job.get("factory_mode")),
                "optimization_scope": as_text(job.get("optimization_scope")),
                "hardware_target": as_text(job.get("hardware_target")),
                "reference_label": as_text(job.get("reference_label")),
                "shared_project_dir": stringify_path(shared_project_dir),
                "worktree_project_dir": stringify_path(worktree_project_dir),
                "source_file": as_text(job.get("source_file")),
                "shared_source_path": stringify_path(shared_source_path),
                "worktree_source_path": stringify_path(worktree_source_path),
                "db_spec_markdown": as_text(job.get("spec")),
                "acceptance_gates_text": as_text(job.get("acceptance_gates")),
                "shipping": shipping,
            },
            "context": {
                "open_messages": open_messages,
                "recent_messages": recent_messages,
                "experiment_summary": experiment_summary,
                "local_file_hints": local_file_hints,
            },
            "repo_local_spec": repo_local_spec,
            "protocol": build_protocol(worker, job, COMMON_DIR, args.recent, repo_local_spec),
            "instructions": build_instructions(job, repo_local_spec),
        }
    finally:
        mem.close()

    schema_path = Path(args.schema).resolve()
    issues = validate_packet(packet, schema_path)
    if issues:
        print("INVALID: watchdog job packet failed validation", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(packet, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
