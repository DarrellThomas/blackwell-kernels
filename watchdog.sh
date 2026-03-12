#!/bin/bash
# Copyright (c) 2026 Darrell Thomas. MIT License.
# watchdog.sh — Periodic clear+restart for autokernel loops
#
# Monitors iteration count in TSV files. After N iterations since last
# restart, waits for a safe moment (no eval running), then clears and
# restarts the loop.
#
# Usage: nohup ./watchdog.sh &

set -euo pipefail

# --- Configuration ---
MAX_ITERS=1                     # Clear after every iteration (fresh context each cycle)
STALL_TIMEOUT=2700              # Restart if no new iteration in N seconds (45 min)
CHECK_INTERVAL=120              # Check every N seconds
RESTART_PAUSE=5                 # Seconds between clear and restart

# Loop definitions: name|tmux_session|tsv_file|autokernel_cmd
LOOPS=(
    "gemm|gemm|../gemm/results/gemm.tsv|/autokernel gemm mar12"
    "attention|attention|results/attention.tsv|/autokernel attention mar12"
)

# --- State tracking ---
declare -A LAST_LINE_COUNT
declare -A LAST_CHANGE_TIME

init_counts() {
    local now
    now=$(date +%s)
    for loop in "${LOOPS[@]}"; do
        IFS='|' read -r name session tsv cmd <<< "$loop"
        if [[ -f "$tsv" ]]; then
            LAST_LINE_COUNT[$name]=$(wc -l < "$tsv")
        else
            LAST_LINE_COUNT[$name]=0
        fi
        LAST_CHANGE_TIME[$name]=$now
        echo "[watchdog] $name: starting line count = ${LAST_LINE_COUNT[$name]}"
    done
}

is_eval_running() {
    local session="$1"
    # Check if eval.sh is running in the session's working directory
    tmux capture-pane -t "$session" -p 2>/dev/null | grep -q "eval.sh" && return 0
    return 1
}

is_session_alive() {
    local session="$1"
    tmux has-session -t "$session" 2>/dev/null
}

restart_loop() {
    local name="$1" session="$2" cmd="$3"

    echo "[watchdog] $(date '+%H:%M:%S') Restarting $name loop..."

    # Interrupt any current operation
    tmux send-keys -t "$session" Escape 2>/dev/null
    sleep 2

    # Clear context
    tmux send-keys -t "$session" '/clear' Enter 2>/dev/null
    sleep "$RESTART_PAUSE"

    # Restart the autokernel command
    tmux send-keys -t "$session" "$cmd" Enter 2>/dev/null

    echo "[watchdog] $(date '+%H:%M:%S') $name loop restarted"
}

# --- Main loop ---
echo "[watchdog] $(date '+%H:%M:%S') Starting watchdog (max_iters=$MAX_ITERS, check_interval=${CHECK_INTERVAL}s)"
init_counts

while true; do
    sleep "$CHECK_INTERVAL"

    for loop in "${LOOPS[@]}"; do
        IFS='|' read -r name session tsv cmd <<< "$loop"

        # Skip if session is dead
        if ! is_session_alive "$session"; then
            continue
        fi

        # Count current lines
        if [[ -f "$tsv" ]]; then
            current=$(wc -l < "$tsv")
        else
            current=0
        fi

        new_iters=$((current - LAST_LINE_COUNT[$name]))
        now=$(date +%s)
        reason=""

        # Update last-change time if progress was made
        if (( new_iters > 0 )); then
            LAST_CHANGE_TIME[$name]=$now
        fi

        # Check triggers
        stall_secs=$(( now - LAST_CHANGE_TIME[$name] ))
        if (( new_iters >= MAX_ITERS )); then
            reason="iteration limit ($new_iters >= $MAX_ITERS)"
        elif (( stall_secs >= STALL_TIMEOUT )); then
            reason="stall timeout (${stall_secs}s with no new iteration)"
        fi

        if [[ -n "$reason" ]]; then
            echo "[watchdog] $(date '+%H:%M:%S') $name: restarting — $reason"

            # Wait for safe moment — no eval running
            wait_count=0
            while is_eval_running "$session" && (( wait_count < 30 )); do
                echo "[watchdog] $name: eval running, waiting..."
                sleep 10
                ((wait_count++))
            done

            restart_loop "$name" "$session" "$cmd"
            LAST_LINE_COUNT[$name]=$current
            LAST_CHANGE_TIME[$name]=$(date +%s)
        fi
    done
done
