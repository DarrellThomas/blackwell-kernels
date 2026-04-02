#!/bin/bash
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Worker status: combines tmux pane, process liveness, and heartbeat
# into a single JSON per worker.
#
# Usage:
#   ./worker-status.sh                    # default workers
#   ./worker-status.sh attention gemm     # specific workers
#
# Output: JSON array to stdout

if [ $# -gt 0 ]; then
    SESSIONS=("$@")
else
    # Auto-detect: all tmux sessions that aren't foreman staff
    SESSIONS=()
    while IFS= read -r name; do
        # Skip non-worker sessions
        [[ "$name" == "researcher" ]] && continue
        SESSIONS+=("$name")
    done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null)
fi

echo "["
first=true
for sess in "${SESSIONS[@]}"; do
    $first || echo ","
    first=false

    # 1. Check tmux session exists
    pane_pid=$(tmux list-panes -t "$sess" -F '#{pane_pid}' 2>/dev/null || true)
    if [ -z "$pane_pid" ]; then
        echo "  {\"worker\": \"$sess\", \"alive\": false, \"state\": \"no_session\", \"duration\": \"\", \"tokens\": \"\", \"thinking\": false, \"heartbeat_age_sec\": -1}"
        continue
    fi

    # 2. Check if worker process is running (claude or deepseek agent)
    claude_pid=$(pgrep -P "$pane_pid" -x claude 2>/dev/null || true)
    if [ -z "$claude_pid" ]; then
        comm=$(ps -p "$pane_pid" -o comm= 2>/dev/null || true)
        if echo "$comm" | grep -q claude 2>/dev/null; then
            claude_pid="$pane_pid"
        fi
    fi
    # Also detect python-based agents (deepseek_agent.py, etc.)
    if [ -z "$claude_pid" ]; then
        agent_pid=$(pstree -a "$pane_pid" 2>/dev/null | grep -m1 "python3.*agent" | grep -oE '^[^,]+' | grep -oE '[0-9]+' | head -1 || true)
        [ -n "$agent_pid" ] && claude_pid="$agent_pid"
    fi
    process_alive="false"
    [ -n "$claude_pid" ] && process_alive="true"

    # 3. Parse tmux pane for activity status
    pane_text=$(tmux capture-pane -t "$sess" -p 2>/dev/null || true)
    status_line=$(echo "$pane_text" | grep -oE '[A-Za-zéè]+…[^│]+' | grep -E '(tokens|thinking|thought|Crunched)' | tail -1 || true)

    state="unknown"
    duration=""
    tokens=""
    thinking="false"

    if [ -n "$status_line" ]; then
        state="working"
        duration=$(echo "$status_line" | grep -oE '[0-9]+h [0-9]+m [0-9]+s|[0-9]+m [0-9]+s|[0-9]+s' | head -1 || true)
        tokens=$(echo "$status_line" | grep -oE '↓ [0-9.]+k tokens' | head -1 || true)
        echo "$status_line" | grep -q "thinking" 2>/dev/null && thinking="true"
    else
        # Check if at idle prompt
        if echo "$pane_text" | grep -qE '^❯ $' 2>/dev/null; then
            state="idle"
        fi
    fi

    # 4. Heartbeat file
    hb_age=-1
    for dir in "$REPO_ROOT/$sess" "$REPO_ROOT/main"; do
        hb_file=$(find "$dir" -maxdepth 1 -name ".autokernel.*.alive" 2>/dev/null | head -1 || true)
        if [ -n "$hb_file" ] && [ -f "$hb_file" ]; then
            hb_age=$(( $(date +%s) - $(stat -c %Y "$hb_file") ))
            break
        fi
    done

    echo "  {\"worker\": \"$sess\", \"alive\": $process_alive, \"state\": \"$state\", \"duration\": \"$duration\", \"tokens\": \"$tokens\", \"thinking\": $thinking, \"heartbeat_age_sec\": $hb_age}"
done
echo "]"
