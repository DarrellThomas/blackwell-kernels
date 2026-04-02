#!/bin/bash
# Start the research memory server on port 8421
# Usage: ./start-server.sh          (foreground)
#        ./start-server.sh --daemon  (background, logs to /data/src/bwk/logs/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/data/src/bwk/logs"
PORT=8421

# Check if already running
if curl -sf "http://localhost:${PORT}/api/stats" > /dev/null 2>&1; then
    echo "Memory server already running on port ${PORT}."
    exit 0
fi

export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "--daemon" ]]; then
    echo "Starting memory server (daemon) on port ${PORT}..."
    nohup python3 "${SCRIPT_DIR}/factory_brain.py" serve "${PORT}" \
        > "${LOG_DIR}/memory-server.log" 2>&1 &
    echo $! > "${SCRIPT_DIR}/.server.pid"
    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PORT}/api/stats" > /dev/null 2>&1; then
            echo "Memory server ready (PID $(cat "${SCRIPT_DIR}/.server.pid"))."
            exit 0
        fi
        sleep 1
    done
    echo "WARNING: Server started but not responding after 30s. Check ${LOG_DIR}/memory-server.log"
else
    echo "Starting memory server (foreground) on port ${PORT}..."
    python3 "${SCRIPT_DIR}/factory_brain.py" serve "${PORT}"
fi
