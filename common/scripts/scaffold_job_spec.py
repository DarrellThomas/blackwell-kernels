#!/usr/bin/env python3
"""Scaffold a shared-format repo-local job spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_job_spec import DEFAULT_SCHEMA_PATH, load_json, validate_job_spec_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=int, required=True, help="Factory job id")
    parser.add_argument("--repo", required=True, help="Repository root that will own docs/Job<id>/job_<id>.json")
    parser.add_argument("--output", default=None, help="Optional output path; defaults to the canonical repo-local path")
    return parser.parse_args()


def canonical_job_spec_path(repo_root: Path, job_id: int) -> Path:
    return repo_root / "docs" / f"Job{job_id}" / f"job_{job_id}.json"


def build_template(job_id: int, repo_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "job_id": job_id,
        "title": f"TODO: define job {job_id} title",
        "state": "not_started",
        "version": "unshipped",
        "type": "research",
        "priority": 3,
        "kernel": repo_root.name,
        "factory_mode": "TODO",
        "optimization_scope": "measurement",
        "hardware_target": "TODO",
        "reference_label": "TODO",
        "description": "TODO: replace this scaffold with the bounded contract for the repo-local job.",
        "notes": [
            "Replace every TODO field before using this spec as an editing contract."
        ],
        "architecture": {
            "current_backend": ["TODO"],
            "plugin_surface": "TODO",
            "authoritative_behavior": ["TODO"]
        },
        "objective_vector": {
            "correctness": "gate",
            "coverage": "gate",
            "weighted_performance": "primary",
            "dependency_reduction": "secondary",
            "numerical_quality": "secondary",
            "complexity": "secondary"
        },
        "scope": {
            "in_scope": ["TODO"],
            "out_of_scope": ["TODO"]
        },
        "contracts": {
            "numerical": {
                "residual_metric": "TODO",
                "residual_threshold": 1e-12,
                "output_shape": "TODO",
                "diagonal_requirement": "TODO",
                "non_spd_behavior": "TODO"
            },
            "coverage": {
                "required_classes": ["TODO"],
                "explicit_padding_sizes": [64],
                "durable_tests_required": True
            },
            "performance": {
                "primary_metric": "TODO",
                "gpu_resident_only": True,
                "exclude_host_transfer": True,
                "weighted_mix": {
                    "small_64_256": 0.4,
                    "medium_256_2048": 0.4,
                    "large_gt_2048": 0.2
                },
                "decision_basis": "TODO"
            },
            "memory": {
                "max_peak_memory_multiplier_vs_baseline": 1.1,
                "new_persistent_buffers_require_justification": True,
                "temporary_allocations_should_be_reused_or_amortized": True
            },
            "reproducibility": {
                "baseline_reproduction_required": True,
                "deterministic_fixed_input_runs_required": True,
                "benchmark_variance_max_fraction": 0.02,
                "record_math_modes": True
            }
        },
        "acceptance_gates": ["TODO"],
        "keep_rule": "TODO",
        "benchmark_set": {
            "baseline_artifact": "TODO",
            "required_scripts": ["TODO"],
            "new_sweep_requirements": ["TODO"],
            "artifact_output_dir": "results/"
        },
        "profiling_requirements": {
            "must_quantify": ["TODO"],
            "escalate_if": ["TODO"]
        },
        "dependency_policy": {
            "retain_third_party_when_empirically_right": True,
            "replacement_allowed_only_if": {
                "correctness_and_coverage_pass": True,
                "numerical_stability_not_degraded": True,
                "plugin_end_to_end_data_justifies_change": True,
                "min_weighted_latency_improvement_fraction": 0.1,
                "or_meaningful_dependency_reduction": True
            },
            "forbid_purity_rewrites": True
        },
        "iteration_logging": {
            "required": True,
            "fields": [
                "hypothesis",
                "change_applied",
                "files_changed",
                "correctness_result",
                "performance_delta",
                "benchmark_artifact_path",
                "keep_or_discard",
                "rationale"
            ]
        },
        "failure_budget": {
            "discard_on_any_hard_gate_correctness_failure": True,
            "stop_after_repeated_non_improving_iterations": True,
            "escalate_if_baseline_not_reproducible": True,
            "escalate_if_no_actionable_bottleneck": True
        },
        "stop_condition": {
            "min_improvement_fraction": 0.02,
            "window_iterations": 5,
            "alternate_condition": "TODO"
        },
        "primary_files": ["path/to/primary/file"],
        "done_criteria": ["TODO"]
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo).resolve()
    source_path = canonical_job_spec_path(repo_root, args.job)
    output_path = Path(args.output).resolve() if args.output else source_path

    if source_path.exists():
        document = load_json(source_path)
        mode = f"copied existing spec from {source_path}"
    else:
        document = build_template(args.job, repo_root)
        mode = "generated starter scaffold"

    if not isinstance(document, dict):
        raise SystemExit(f"ERROR: expected top-level object in {source_path}")

    schema_path = DEFAULT_SCHEMA_PATH.resolve()
    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        raise SystemExit(f"ERROR: expected top-level object in {schema_path}")

    issues = validate_job_spec_document(document, schema)
    if issues:
        print("INVALID: scaffolded job spec failed validation")
        for idx, issue in enumerate(issues, start=1):
            print(f"[{idx}] {issue}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(str(output_path))
    print(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
