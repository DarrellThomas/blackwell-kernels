#!/bin/bash
# test-all.sh — Run edge case tests across all factory projects
#
# Usage:
#   ./test-all.sh              # run all projects
#   ./test-all.sh linalg gemm  # run specific projects
#
# Each project must have tests/test_edge_cases.py.
# Results are saved to results/edge_case_results.tsv in each project.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
PROJECTS=(linalg gemm numerical qr spmv cuquantum)
RESULTS_FILE="$LOG_DIR/edge_case_results_$(date +%Y%m%d).log"

if [ $# -gt 0 ]; then
    PROJECTS=("$@")
fi

echo "=== Factory Edge Case Test Suite ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Projects: ${PROJECTS[*]}"
echo ""

total_pass=0
total_fail=0

for project in "${PROJECTS[@]}"; do
    dir="$REPO_ROOT/$project"
    test_file="$dir/tests/test_edge_cases.py"

    if [ ! -f "$test_file" ]; then
        echo "[$project] SKIP — no test_edge_cases.py"
        echo ""
        continue
    fi

    echo "[$project] Running edge case tests..."
    cd "$dir"

    # Build if needed
    if [ -f setup.py ]; then
        CUDA_HOME=/usr/local/cuda-13 python3 setup.py build_ext --inplace > /dev/null 2>&1 || true
    fi

    # Run tests
    output=$(CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 "$test_file" 2>&1)
    pass=$(echo "$output" | grep -c "PASS" || true)
    fail=$(echo "$output" | grep -c "FAIL" || true)

    total_pass=$((total_pass + pass))
    total_fail=$((total_fail + fail))

    echo "$output" | grep -E "PASS|FAIL|RESULTS:"
    echo ""
done

echo "==========================================="
echo "FACTORY TOTALS: $total_pass passed, $total_fail failed"
echo "==========================================="

# Save to log
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) | pass=$total_pass fail=$total_fail projects=${PROJECTS[*]}" >> "$RESULTS_FILE"
