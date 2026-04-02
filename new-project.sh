#!/bin/bash
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Create a new project from the template.
#
# Usage:
#   ./new-project.sh <project-name>
#
# Example:
#   ./new-project.sh conv2d
#   ./new-project.sh octave-numerical
#
# This will:
#   1. Create a new subdirectory at $BWK_ROOT/<project-name>/
#   2. Copy the lean project template
#   3. Register a job in the factory database
#   4. Print next steps
#
# Note: New projects are tracked in the bwk/ factory repo (not as git worktrees).
# The worktree model (main/ + per-kernel branches) is legacy — new work goes here.

set -euo pipefail

BWK_ROOT="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_DIR="${BWK_ROOT}/template"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <project-name>"
    echo ""
    echo "Examples:"
    echo "  $0 conv2d"
    echo "  $0 optimize"
    exit 1
fi

PROJECT_NAME="$1"
PROJECT_DIR="${BWK_ROOT}/${PROJECT_NAME}"

if [[ -d "$PROJECT_DIR" ]]; then
    echo "Error: ${PROJECT_DIR} already exists"
    exit 1
fi

echo "Creating project: ${PROJECT_NAME}"
echo "  Directory: ${PROJECT_DIR}"
echo ""

# Create project directory from template
mkdir -p "$PROJECT_DIR"
cp -a "${TEMPLATE_DIR}/." "${PROJECT_DIR}/"

# Make scripts executable
chmod +x "${PROJECT_DIR}/eval.sh" "${PROJECT_DIR}/build.sh" 2>/dev/null || true

# Rename template files to project-specific names
echo "Renaming template files for ${PROJECT_NAME}..."
cd "$PROJECT_DIR"
mv -n tests/test_kernel.py "tests/test_${PROJECT_NAME}.py" 2>/dev/null || true
mv -n benchmarks/bench_kernel.py "benchmarks/bench_${PROJECT_NAME}.py" 2>/dev/null || true
mv -n profiles/profile_kernel.py "profiles/profile_${PROJECT_NAME}.py" 2>/dev/null || true
mv -n program_kernel.md "program_${PROJECT_NAME}.md" 2>/dev/null || true
mv -n docs/kernel_agent_state.md "docs/${PROJECT_NAME}_agent_state.md" 2>/dev/null || true

# Create kernel source directory
mkdir -p "csrc/${PROJECT_NAME}"

# Register the new job in the factory database
echo "Registering job in factory database..."
TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 python3 "${BWK_ROOT}/common/memory/factory_brain.py" \
    job-create "${PROJECT_NAME}" "${PROJECT_NAME} kernel" \
    --type kernel --kernel "${PROJECT_NAME}" \
    --state not_started --priority medium \
    --assigned "${PROJECT_NAME}" --by ops 2>/dev/null \
    && echo "  Job registered." \
    || echo "  WARNING: Could not register job (DB may be locked or name exists)"

echo ""
echo "=== Project ${PROJECT_NAME} created ==="
echo ""
echo "Next steps:"
echo "  1. cd ${PROJECT_DIR}"
echo "  2. Edit .claude/CLAUDE.md — keep only project-local facts there"
echo "  3. Put the real project spec in the factory DB (source of truth)"
echo "     factory_brain.py job-update <id> --factory-mode ... --objective-vector ... --acceptance-gates ..."
echo "  4. Edit program_${PROJECT_NAME}.md only for project-specific contract details"
echo "  5. Edit setup.py — add your .cu source files"
echo "  6. Edit eval.sh — uncomment example case, rename to '${PROJECT_NAME}'"
echo "  7. Write csrc/${PROJECT_NAME}/${PROJECT_NAME}_sm120a.cu — your kernel"
echo "  8. Fill in tests/test_${PROJECT_NAME}.py — correctness tests"
echo "  9. Fill in benchmarks/bench_${PROJECT_NAME}.py — benchmarks"
echo " 10. Fill in profiles/profile_${PROJECT_NAME}.py — ncu profile launch"
echo " 11. git add ${PROJECT_NAME}/ && git commit -m 'init: ${PROJECT_NAME} from template'"
echo ""
echo "Shared infrastructure (single source of truth in common/ and factory_brain):"
echo "  common/csrc/common/  → CUDA headers (mma, ldmatrix, cp_async, swizzle, fp8)"
echo "  common/scripts/      → build, watchdog, gate processing"
echo "  common/memory/       → factory_brain DB + search"
echo "  common/claude/       → shared worker instruction docs"
echo ""
echo "Factory DB commands:"
echo "  factory_brain.py jobs                    — see all jobs"
echo "  factory_brain.py messages --status open  — see open messages"
echo "  factory_brain.py job-history <id>        — audit trail"
