#!/bin/bash
# Copyright (c) 2026 Darrell Thomas. MIT License.
# watchdog_wrapper.sh -- Self-restarting wrapper for watchdog_main.py.
# If the watchdog exits for any reason, wait 10 seconds and restart.
# State lives in the DB, not in process memory, so restarts are clean.
#
# Usage: nohup bash watchdog_wrapper.sh > /dev/null 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/../../logs/watchdog.log"
mkdir -p "$(dirname "$LOG")"

while true; do
    echo "[wrapper] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting watchdog_main.py (pid=$$)" >> "$LOG"
    python3 "$SCRIPT_DIR/watchdog_main.py" >> "$LOG" 2>&1
    EXIT_CODE=$?
    echo "[wrapper] $(date -u +%Y-%m-%dT%H:%M:%SZ) watchdog_main.py exited with code $EXIT_CODE, restarting in 10s" >> "$LOG"
    sleep 10
done
