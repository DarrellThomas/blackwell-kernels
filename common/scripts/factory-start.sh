#!/bin/bash
# factory-start.sh — Start all factory services
#
# Brings up everything the factory needs to run:
#   1. Memory server (port 8421) — research DB API
#   2. UI dashboard (port 8420) — optimization dashboard
#   3. Watchdog — worker management, TSV ingest, worker state
#
# Safe to run multiple times — skips services that are already up.
#
# Usage:
#   ./factory-start.sh          # start everything
#   ./factory-start.sh status   # show what's running
#   ./factory-start.sh stop     # stop all services
#
# Auto-start on boot:
#   crontab -e → add: @reboot ${REPO_ROOT}/common/scripts/factory-start.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON_DIR="$REPO_ROOT/common"
UI_DIR="$REPO_ROOT/ui"
LOG_DIR="$REPO_ROOT/logs"
SCRIPTS="$COMMON_DIR/scripts"
MEMORY="$COMMON_DIR/memory"
UI="$UI_DIR"

mkdir -p "$LOG_DIR"

# --- Helpers ---

is_port_up() {
    curl -sf "http://localhost:$1/" > /dev/null 2>&1 || \
    curl -sf "http://localhost:$1/api/stats" > /dev/null 2>&1
}

is_process_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

log() {
    echo "[factory] $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# --- Status ---

if [[ "${1:-}" == "status" ]]; then
    echo "=== Factory Services ==="
    is_port_up 8421 && echo "  Memory server (8421): UP" || echo "  Memory server (8421): DOWN"
    is_port_up 8420 && echo "  UI dashboard  (8420): UP" || echo "  UI dashboard  (8420): DOWN"
    tmux has-session -t watchdog 2>/dev/null && echo "  Watchdog:             UP (tmux attach -t watchdog)" || echo "  Watchdog:             DOWN"
    echo ""
    echo "=== Worker Sessions ==="
    tmux list-sessions 2>/dev/null || echo "  (no tmux sessions)"
    exit 0
fi

# --- Stop ---

if [[ "${1:-}" == "stop" ]]; then
    log "Stopping all factory services..."
    # Memory server
    if [[ -f "$MEMORY/.server.pid" ]]; then
        kill "$(cat "$MEMORY/.server.pid")" 2>/dev/null && log "Memory server stopped" || true
        rm -f "$MEMORY/.server.pid"
    fi
    # Dashboard
    pkill -f "python3 dashboard.py" 2>/dev/null && log "Dashboard stopped" || true
    # Watchdog
    tmux kill-session -t watchdog 2>/dev/null && log "Watchdog stopped" || true
    log "All services stopped."
    exit 0
fi

# --- Start ---

log "Starting factory services..."

# 1. Memory server
if is_port_up 8421; then
    log "Memory server already running on port 8421"
else
    log "Starting memory server..."
    "$MEMORY/start-server.sh" --daemon
fi

# 2. UI dashboard
if is_port_up 8420; then
    log "UI dashboard already running on port 8420"
else
    log "Starting UI dashboard..."
    cd "$UI"
    nohup python3 dashboard.py > "$LOG_DIR/dashboard.log" 2>&1 &
    sleep 2
    if is_port_up 8420; then
        log "UI dashboard started on port 8420"
    else
        log "WARNING: UI dashboard failed to start — check $LOG_DIR/dashboard.log"
    fi
fi

# 3. Watchdog
if tmux has-session -t watchdog 2>/dev/null; then
    log "Watchdog already running (tmux attach -t watchdog)"
else
    log "Starting watchdog in tmux session 'watchdog'..."
    tmux new-session -d -s watchdog -c "$REPO_ROOT" \
        "bash $SCRIPTS/watchdog.sh 2>&1 | tee -a $LOG_DIR/watchdog.log; echo '[watchdog exited — attach to inspect]'"
    log "Watchdog started (tmux attach -t watchdog to follow)"
fi

log "Factory services ready."
echo ""
# Show status
$0 status
