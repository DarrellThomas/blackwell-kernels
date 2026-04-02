#!/bin/bash
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# gpu-sched.sh — Shared GPU scheduler for autokernel loops
#
# Provides file-lock based GPU reservation so multiple autokernel instances
# can share GPUs without contention. Lives at ${REPO_ROOT}/ (outside any
# git worktree) so all projects can source it.
#
# Usage in eval.sh:
#   source "$(dirname "$0")/../gpu-sched.sh"
#   gpu_acquire 1          # prefer GPU 1, fall back to GPU 0
#   # ... CUDA_VISIBLE_DEVICES is now set, lock held until exit ...
#   # gpu_release called automatically via EXIT trap
#
# How it works:
#   - Each GPU gets a lock file: /tmp/bwk-gpu-{0,1}.lock
#   - flock(1) provides atomic advisory locking
#   - On acquire: try preferred GPU, then fallback, then wait
#   - On exit/crash: release via trap (EXIT, ERR, INT, TERM)
#   - Lock files contain: PID, project name, timestamp (for debugging)
#
# GPU policy:
#   - GPU 0: water-cooled (heavy training; ComfyUI intermittent)
#   - GPU 1: primary dev GPU
#   - Both are RTX 5090 (32 GB each), kernel jobs use <1 GB

GPU_LOCK_DIR="/tmp"
GPU_LOCK_PREFIX="bwk-gpu"
GPU_ACQUIRED=""
GPU_LOCK_FD=""
GPU_MAX_WAIT=${GPU_MAX_WAIT:-600}      # Max seconds to wait for a GPU (default: 10 min)
GPU_POLL_INTERVAL=${GPU_POLL_INTERVAL:-5}  # Seconds between retry attempts
GPU_PROJECT=${GPU_PROJECT:-$(basename "$(pwd)")}  # Project name for logging

# Write lock metadata (for debugging stale locks)
_gpu_write_lock_info() {
    local gpu_id=$1
    local lockfile="${GPU_LOCK_DIR}/${GPU_LOCK_PREFIX}-${gpu_id}.lock"
    echo "pid=$$  project=${GPU_PROJECT}  acquired=$(date -Iseconds)" > "${lockfile}.info"
}

# Remove lock metadata
_gpu_remove_lock_info() {
    local gpu_id=$1
    rm -f "${GPU_LOCK_DIR}/${GPU_LOCK_PREFIX}-${gpu_id}.lock.info"
}

# Try to acquire a specific GPU (non-blocking)
# Returns 0 on success, 1 on failure
_gpu_try_acquire() {
    local gpu_id=$1
    local lockfile="${GPU_LOCK_DIR}/${GPU_LOCK_PREFIX}-${gpu_id}.lock"

    # Open lock file on a free file descriptor
    local fd
    case $gpu_id in
        0) fd=200 ;;
        1) fd=201 ;;
        *) echo "gpu-sched: invalid GPU id: $gpu_id" >&2; return 1 ;;
    esac

    eval "exec ${fd}>${lockfile}"

    if flock --nonblock $fd 2>/dev/null; then
        GPU_ACQUIRED=$gpu_id
        GPU_LOCK_FD=$fd
        export CUDA_VISIBLE_DEVICES=$gpu_id
        _gpu_write_lock_info $gpu_id
        return 0
    else
        eval "exec ${fd}>&-"
        return 1
    fi
}

# Acquire a GPU. Try preferred first, then fallback, then wait.
# Usage: gpu_acquire [preferred_gpu_id]
gpu_acquire() {
    local prefer=${1:-1}
    local fallback=$((1 - prefer))

    # Install release trap (accumulate with existing traps)
    trap 'gpu_release' EXIT

    # Try preferred GPU
    if _gpu_try_acquire $prefer; then
        echo "gpu-sched: acquired GPU ${GPU_ACQUIRED} (preferred) for ${GPU_PROJECT}" >&2
        return 0
    fi

    # Try fallback GPU
    if _gpu_try_acquire $fallback; then
        echo "gpu-sched: acquired GPU ${GPU_ACQUIRED} (fallback) for ${GPU_PROJECT}" >&2
        return 0
    fi

    # Both busy — wait for preferred GPU with polling
    echo "gpu-sched: GPUs busy, waiting for GPU ${prefer} (max ${GPU_MAX_WAIT}s)..." >&2
    local waited=0
    while [ $waited -lt $GPU_MAX_WAIT ]; do
        sleep $GPU_POLL_INTERVAL
        waited=$((waited + GPU_POLL_INTERVAL))

        # Try preferred first, then fallback
        if _gpu_try_acquire $prefer; then
            echo "gpu-sched: acquired GPU ${GPU_ACQUIRED} after ${waited}s wait for ${GPU_PROJECT}" >&2
            return 0
        fi
        if _gpu_try_acquire $fallback; then
            echo "gpu-sched: acquired GPU ${GPU_ACQUIRED} (fallback) after ${waited}s wait for ${GPU_PROJECT}" >&2
            return 0
        fi
    done

    echo "gpu-sched: TIMEOUT waiting for GPU after ${GPU_MAX_WAIT}s" >&2
    return 1
}

# Release the currently held GPU
gpu_release() {
    if [ -n "$GPU_ACQUIRED" ]; then
        _gpu_remove_lock_info $GPU_ACQUIRED
        # Close the file descriptor to release flock
        if [ -n "$GPU_LOCK_FD" ]; then
            eval "exec ${GPU_LOCK_FD}>&-" 2>/dev/null || true
        fi
        echo "gpu-sched: released GPU ${GPU_ACQUIRED} for ${GPU_PROJECT}" >&2
        GPU_ACQUIRED=""
        GPU_LOCK_FD=""
    fi
}

# Show current GPU lock status (for debugging)
gpu_status() {
    echo "GPU Lock Status:"
    for gpu_id in 0 1; do
        local lockfile="${GPU_LOCK_DIR}/${GPU_LOCK_PREFIX}-${gpu_id}.lock"
        local infofile="${lockfile}.info"
        if [ -f "$infofile" ]; then
            local info=$(cat "$infofile" 2>/dev/null)
            local pid=$(echo "$info" | grep -oP 'pid=\K\d+')
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "  GPU $gpu_id: LOCKED  ($info)"
            else
                echo "  GPU $gpu_id: STALE   ($info) — process $pid not running"
            fi
        else
            echo "  GPU $gpu_id: FREE"
        fi
    done
}
