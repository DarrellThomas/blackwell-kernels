# watchdog_git.sh — Dedicated git worktree helpers for watchdog-managed workers
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Sourced by watchdog.sh. Requires REPO_ROOT and COMMON_DIR exported.

WATCHDOG_WORKTREE_ROOT="${WATCHDOG_WORKTREE_ROOT:-$REPO_ROOT/data/watchdog-worktrees}"
WATCHDOG_PACKET_NAME="${WATCHDOG_PACKET_NAME:-job_packet.json}"

sanitize_ref_component() {
    local raw="${1:-}"
    local cleaned
    cleaned="$(printf '%s' "$raw" | tr -cs 'A-Za-z0-9._-' '-')"
    cleaned="$(printf '%s' "$cleaned" | sed 's/^-*//; s/-*$//')"
    [[ -n "$cleaned" ]] || cleaned="repo"
    printf '%s' "$cleaned"
}

worker_packet_path() {
    local worker="$1"
    printf '%s\n' "$WATCHDOG_WORKTREE_ROOT/$worker/$WATCHDOG_PACKET_NAME"
}

refresh_worker_packet() {
    local worker="$1" worktree_path="$2" shared_repo_root="${3:-}" branch="${4:-}"
    local packet_path
    packet_path="$(worker_packet_path "$worker")"
    mkdir -p "$(dirname "$packet_path")"

    local build_cmd=(python3 "$COMMON_DIR/scripts/build_job_packet.py" --worker "$worker" --worktree "$worktree_path" --output "$packet_path")
    if [[ -n "$shared_repo_root" ]]; then
        build_cmd+=(--shared-repo-root "$shared_repo_root")
    fi
    if [[ -n "$branch" ]]; then
        build_cmd+=(--branch "$branch")
    fi

    if ! "${build_cmd[@]}" >/dev/null; then
        log_watchdog attention "$worker" "failed to refresh job packet at $packet_path"
        return 1
    fi

    printf '%s\n' "$packet_path"
}

_worker_context_lines() {
    local worker="$1" fallback_cwd="$2"
    python3 - "$worker" "$fallback_cwd" <<'PYCTX'
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory, get_active_worker_jobs, resolve_job_project_dir

worker = sys.argv[1]
fallback_cwd = sys.argv[2]
terminal = {'shipped', 'converged', 'parked', 'abandoned'}
mem = ResearchMemory()
rows = get_active_worker_jobs(mem, worker, exclude_done_handoffs=True)
rows.sort(key=lambda j: (int(j['priority']) if str(j['priority']).isdigit() else 99, j['updated_at'], j['id']))
job = rows[0] if rows else None
project_dir = None
if job is not None:
    resolved = resolve_job_project_dir(job)
    if resolved is not None:
        project_dir = str(resolved)
if not project_dir:
    project_dir = fallback_cwd
repo_top = project_dir
try:
    repo_top = subprocess.check_output(
        ['git', '-C', project_dir, 'rev-parse', '--show-toplevel'],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    repo_top = project_dir
repo_name = Path(repo_top).name if repo_top else Path(project_dir).name
print(job['id'] if job else '')
print((job or {}).get('state', ''))
print((job or {}).get('title', ''))
print(repo_top)
print(project_dir)
print(repo_name)
mem.close()
PYCTX
}

prepare_worker_workspace() {
    local worker="$1" fallback_cwd="${2:-$REPO_ROOT/$worker}"
    local ctx=()
    local job_id repo_top project_dir repo_name repo_key branch
    local worktree_path desired_common current_common current_branch

    mapfile -t ctx < <(_worker_context_lines "$worker" "$fallback_cwd")
    job_id="${ctx[0]:-}"
    repo_top="${ctx[3]:-$fallback_cwd}"
    project_dir="${ctx[4]:-$fallback_cwd}"
    repo_name="${ctx[5]:-$(basename "$repo_top")}"

    if ! git -C "$repo_top" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        refresh_worker_packet "$worker" "$project_dir" "$repo_top" >/dev/null 2>&1 || true
        printf '%s\n' "$project_dir"
        return 0
    fi

    repo_key="$(sanitize_ref_component "$repo_name")"
    if [[ -n "$job_id" ]]; then
        branch="watchdog/$worker/job-${job_id}-${repo_key}"
    else
        branch="watchdog/$worker/idle-${repo_key}"
    fi
    worktree_path="$WATCHDOG_WORKTREE_ROOT/$worker/current"
    mkdir -p "$(dirname "$worktree_path")"

    desired_common="$(cd "$repo_top" && readlink -f "$(git rev-parse --git-common-dir)")"

    if [[ -e "$worktree_path" ]]; then
        if git -C "$worktree_path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            current_common="$(cd "$worktree_path" && readlink -f "$(git rev-parse --git-common-dir)")"
            current_branch="$(git -C "$worktree_path" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
            if [[ "$current_common" == "$desired_common" && "$current_branch" == "$branch" ]]; then
                refresh_worker_packet "$worker" "$worktree_path" "$repo_top" "$branch" >/dev/null 2>&1 || true
                printf '%s\n' "$worktree_path"
                return 0
            fi
            if git -C "$worktree_path" status --porcelain | grep -q .; then
                log_watchdog attention "$worker" "dedicated worktree $worktree_path is dirty; refusing automatic reset"
                refresh_worker_packet "$worker" "$worktree_path" "$repo_top" "$current_branch" >/dev/null 2>&1 || true
                printf '%s\n' "$worktree_path"
                return 0
            fi
            git -C "$worktree_path" worktree remove --force "$worktree_path" >/dev/null 2>&1 || true
        fi
        rm -rf "$worktree_path"
    fi

    git -C "$repo_top" worktree prune >/dev/null 2>&1 || true
    if git -C "$repo_top" show-ref --verify --quiet "refs/heads/$branch"; then
        git -C "$repo_top" worktree add "$worktree_path" "$branch" >/dev/null
    else
        git -C "$repo_top" worktree add -b "$branch" "$worktree_path" HEAD >/dev/null
    fi

    log_watchdog git "$worker" "workspace=$worktree_path repo=$repo_top branch=$branch job=${job_id:-idle}"
    refresh_worker_packet "$worker" "$worktree_path" "$repo_top" "$branch" >/dev/null 2>&1 || true
    printf '%s\n' "$worktree_path"
}

reset_worker_workspace() {
    local worker="$1" fallback_cwd="${2:-$REPO_ROOT/$worker}"
    local worktree_path="$WATCHDOG_WORKTREE_ROOT/$worker/current"

    if [[ -e "$worktree_path" ]]; then
        if git -C "$worktree_path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            if git -C "$worktree_path" status --porcelain | grep -q .; then
                log_watchdog attention "$worker" "cannot reset dirty dedicated worktree $worktree_path"
                printf '%s\n' "$worktree_path"
                return 1
            fi
            git -C "$worktree_path" worktree remove --force "$worktree_path" >/dev/null 2>&1 || true
        fi
        rm -rf "$worktree_path"
    fi

    prepare_worker_workspace "$worker" "$fallback_cwd"
}
