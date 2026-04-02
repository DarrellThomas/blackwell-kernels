#!/bin/bash
# Copyright (c) 2026 Darrell Thomas. MIT License.
# watchdog.sh — Periodic clear+restart for autokernel loops
#
# Monitors iteration count in factory_brain. After N experiments since last
# restart, waits for a safe moment (no eval running), then clears and
# restarts the loop. Also detects idle workers and restarts them.
#
# Usage: nohup ./watchdog.sh > $LOG_DIR/watchdog.log 2>&1 &
#
# Modules (sourced below):
#   watchdog_db.sh      — DB/state helpers (get_tick_epoch, touch_tick, etc.)
#   watchdog_tmux.sh    — tmux/session helpers (classify_pane_state, etc.)
#   watchdog_workers.sh — worker lifecycle (ensure_loop_session, restart_loop)
#   watchdog_ticks.sh   — periodic ticks (tick_short, tick_medium, tick_long)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON_DIR="$REPO_ROOT/common"
LOG_DIR="$REPO_ROOT/logs"
export REPO_ROOT COMMON_DIR
export PYTHONPATH="$COMMON_DIR/memory:$COMMON_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"

# --- Configuration ---
MAX_ITERS=1                     # Clear after every iteration (fresh context each cycle)
STALL_TIMEOUT=9000              # Restart if no new iteration in N seconds (2.5 hrs)
IDLE_TIMEOUT=900                # Restart if worker idle at prompt for N seconds (15 min)
ACTIVE_SIT_TIMEOUT=300          # Flag active-lane jobs with no live heartbeat for N seconds
SHORT_LOOP_INTERVAL=60          # scheduler / gate / worker nudges
MEDIUM_LOOP_INTERVAL=1800       # ingest changed content / metadata hygiene
LONG_LOOP_INTERVAL=21600        # summary/audit/promotion work
RESTART_PAUSE=5                 # Seconds between clear and restart
PROMPT_CONFIRM_RETRIES=3        # How many times to resubmit a staged Codex prompt
PROMPT_CONFIRM_WAIT=2           # Seconds to wait between prompt submission checks

# ===== Lane configuration =====
WORKER_LANES_CONFIG="${WORKER_LANES_CONFIG:-$COMMON_DIR/scripts/worker_lanes.conf}"
DEFAULT_MANAGED_WORKER_SLOTS="gemm:1 octave-gpu:2"

load_managed_worker_slots() {
    local override="${MANAGED_WORKER_SLOTS:-}"
    if [[ -n "$override" ]]; then
        for entry in $override; do echo "$entry"; done
        return
    fi
    if [[ -f "$WORKER_LANES_CONFIG" ]]; then
        while IFS= read -r line; do
            line="${line%%#*}"
            line="${line//[[:space:]]/}"
            [[ -n "$line" ]] && echo "$line"
        done < "$WORKER_LANES_CONFIG"
        return
    fi
    for entry in $DEFAULT_MANAGED_WORKER_SLOTS; do echo "$entry"; done
}

# Loop definitions
#
# Legacy format:
#   name|tmux_session|legacy_results_tsv|resume_cmd
#
# Codex-aware format:
#   name|tmux_session|legacy_results_tsv|resume_cmd|launch_cmd|cwd
#
# Notes:
# - Iteration counts are sourced from factory_brain experiment rows for `name`.
#   `legacy_results_tsv` is only a fallback while old loops still mirror TSV.
# - `resume_cmd` is what watchdog sends after `/clear` when the session is alive.
# - `launch_cmd` is optional. Use it when recreating a dead session requires
#   launching Codex rather than sending a plain prompt into an existing shell.
# - `cwd` defaults to `$REPO_ROOT/<name>` when omitted.
# Add new projects here when onboarding.
LOOPS=(
    # Worker launch_cmd — either Claude or Codex:
    #   claude --dangerously-skip-permissions
    #   codex --dangerously-bypass-approvals-and-sandbox --no-alt-screen
    # Format: name|tmux_session|legacy_results_tsv|resume_cmd|launch_cmd|cwd
    "gemm|gemm|$REPO_ROOT/gemm/results/gemm.tsv|@active-prompt:gemm|claude --dangerously-skip-permissions|$REPO_ROOT/gemm"
    "octave-gpu|octave-gpu||@active-prompt:octave-gpu|claude --dangerously-skip-permissions|$REPO_ROOT/octave-gpu"
    #
    # --- 3 Codex workers (each owns a distinct job so file conflicts are rare) ---
    # cx1=dgetrs #50  cx2=dgesv #33  cx3=dnrm2 #31  (all in octave-gpu/)
    "cx1|cx1||@active-prompt:cx1|codex --dangerously-bypass-approvals-and-sandbox --no-alt-screen|$REPO_ROOT/octave-gpu"
    "cx2|cx2||@active-prompt:cx2|codex --dangerously-bypass-approvals-and-sandbox --no-alt-screen|$REPO_ROOT/octave-gpu"
    "cx3|cx3||@active-prompt:cx3|codex --dangerously-bypass-approvals-and-sandbox --no-alt-screen|$REPO_ROOT/octave-gpu"

    # --- NEVER add this — chess-training is Darrell's live ML training, not a worker ---
    # chess-training: RNN training in progress, not a kernel optimization loop
)

# Staff loops
#
# Legacy format:
#   name|tmux_session|resume_cmd
#
# Codex-aware format:
#   name|tmux_session|resume_cmd|launch_cmd|cwd
#
# Researcher gets a nudge if idle — it's pull-based (only runs when kicked).
STAFF_LOOPS=(
    # Same claude/codex choice applies here
    "researcher|researcher|Read your open messages in the factory DB and handle only those bounded research requests. Stay pull-based, keep outputs concise, and do not proactively fan work out to workers.|claude --dangerously-skip-permissions|$REPO_ROOT/foreman-staff/researcher"
)

# --- Source helper modules ---
_SCRIPTS_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$_SCRIPTS_DIR/watchdog_db.sh"
source "$_SCRIPTS_DIR/watchdog_tmux.sh"
source "$_SCRIPTS_DIR/watchdog_workers.sh"
source "$_SCRIPTS_DIR/watchdog_ticks.sh"

# --- Phase context directory (used by watchdog_workers.sh) ---
PHASES_DIR="$COMMON_DIR/claude/phases"

# --- Per-worker state tracking (indexed by worker name) ---
declare -A LAST_LINE_COUNT
declare -A LAST_CHANGE_TIME
declare -A LAST_IDLE_TIME

init_counts() {
    local now
    now=$(date +%s)
    for loop in "${LOOPS[@]}"; do
        IFS='|' read -r name session tsv resume_cmd launch_cmd cwd <<< "$loop"
        LAST_LINE_COUNT[$name]=$(get_experiment_count "$name" "$tsv")
        LAST_CHANGE_TIME[$name]=$now
        LAST_IDLE_TIME[$name]=0
        log_watchdog init "$name" "starting experiment count = ${LAST_LINE_COUNT[$name]}"
    done
    for entry in "${STAFF_LOOPS[@]}"; do
        IFS='|' read -r name session resume_cmd launch_cmd cwd <<< "$entry"
        LAST_IDLE_TIME[$name]=0
        log_watchdog init "$name" "staff loop registered (pull-only)"
    done
}

# --- Main loop ---
mkdir -p "$LOG_DIR"
trap 'set_daemon_state down "watchdog exiting"' EXIT INT TERM
log_watchdog daemon start "starting watchdog (short=${SHORT_LOOP_INTERVAL}s, medium=${MEDIUM_LOOP_INTERVAL}s, long=${LONG_LOOP_INTERVAL}s)"
set_daemon_state up "watchdog starting"
init_counts

while true; do
    set_daemon_state up "watchdog loop alive"

    if tick_due "watchdog_short" "$SHORT_LOOP_INTERVAL"; then
        if tick_short; then
            touch_tick "watchdog_short" "ok" "scheduler"
        else
            log_watchdog attention short "tick_short failed, will retry next cycle"
        fi
    fi

    if tick_due "watchdog_medium" "$MEDIUM_LOOP_INTERVAL"; then
        if tick_medium; then
            touch_tick "watchdog_medium" "ok" "ingest_hygiene"
        else
            log_watchdog attention medium "tick_medium failed, will retry next cycle"
        fi
    fi

    if tick_due "watchdog_long" "$LONG_LOOP_INTERVAL"; then
        tick_long &
        touch_tick "watchdog_long" "ok" "summaries_audit"
    fi

    sleep 15
done
