#!/bin/bash
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Foreman patrol — runs every 10 minutes, handles routine checks,
# nudges foreman-claude for judgment calls.
#
# Usage: Run in a tmux loop or cron
#   while true; do ./foreman-patrol.sh; sleep 600; done

set -euo pipefail
BWK_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$BWK_ROOT/common/scripts"
FOREMAN_SESSION="foreman"
LOG="$BWK_ROOT/.claude/patrol.log"

echo "[$(date '+%H:%M:%S')] === PATROL ===" >> "$LOG"

# Skip non-worker sessions
SKIP_SESSIONS="patrol|foreman|researcher|debrief|ui"

# 1. Check for dead workers (alive=false) and restart them
DEAD_WORKERS=()
while IFS= read -r line; do
    worker=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('worker',''))" 2>/dev/null)
    alive=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('alive',True))" 2>/dev/null)
    # Skip non-worker sessions
    echo "$worker" | grep -qE "^($SKIP_SESSIONS)$" && continue
    if [ "$alive" = "False" ]; then
        DEAD_WORKERS+=("$worker")
    fi
done < <("$SCRIPT_DIR/worker-status.sh" 2>/dev/null | python3 -c "
import json,sys
for entry in json.load(sys.stdin):
    print(json.dumps(entry))
" 2>/dev/null)

for w in "${DEAD_WORKERS[@]}"; do
    echo "[$(date '+%H:%M:%S')] DEAD: $w — attempting restart" >> "$LOG"
    if [ -d "$BWK_ROOT/$w" ]; then
        tmux new-session -d -s "$w" -c "$BWK_ROOT/$w" "claude --dangerously-skip-permissions" 2>/dev/null || true
        sleep 3
        tmux send-keys -t "$w" "Continue. You were restarted by the patrol watchdog. Read your .claude/CLAUDE.md and docs/*_agent_state.md to pick up where you left off. You run 24/7." Enter 2>/dev/null || true
        echo "[$(date '+%H:%M:%S')] RESTARTED: $w" >> "$LOG"
    fi
done

# 2. Check for idle workers (at prompt, not thinking)
IDLE_WORKERS=()
while IFS= read -r line; do
    worker=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('worker',''))" 2>/dev/null)
    state=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('state',''))" 2>/dev/null)
    # Skip non-worker sessions
    echo "$worker" | grep -qE "^($SKIP_SESSIONS)$" && continue
    if [ "$state" = "idle" ]; then
        IDLE_WORKERS+=("$worker")
    fi
done < <("$SCRIPT_DIR/worker-status.sh" 2>/dev/null | python3 -c "
import json,sys
for entry in json.load(sys.stdin):
    print(json.dumps(entry))
" 2>/dev/null)

for w in "${IDLE_WORKERS[@]}"; do
    # Check if there's a halt note — if so, foreman needs to review (don't auto-kick)
    notes=$(find "$BWK_ROOT/$w/for_foreman-claude" -name "*.md" ! -name ".gitkeep" 2>/dev/null)
    if [ -n "$notes" ]; then
        echo "[$(date '+%H:%M:%S')] IDLE+NOTE: $w — has halt note, needs foreman review" >> "$LOG"
    else
        echo "[$(date '+%H:%M:%S')] IDLE: $w — nudging" >> "$LOG"
        tmux send-keys -t "$w" "Continue optimizing. Read your agent_state.md for current status. You run 24/7 — do not stop without writing a halt note in for_foreman-claude/." Enter 2>/dev/null || true
    fi
done

# 3. Check for new foreman notes (needs judgment — log for foreman review)
NEW_NOTES=()
for d in "$BWK_ROOT"/*/for_foreman-claude; do
    project=$(basename "$(dirname "$d")")
    notes=$(find "$d" -name "*.md" ! -name ".gitkeep" 2>/dev/null)
    if [ -n "$notes" ]; then
        NEW_NOTES+=("$project")
        echo "[$(date '+%H:%M:%S')] NOTE: $project has unread notes" >> "$LOG"
    fi
done

# 4. Check dashboard is alive
if ! curl -s -o /dev/null -w "" http://localhost:8420/ 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] DASHBOARD DOWN — restarting" >> "$LOG"
    cd "$BWK_ROOT/ui" && python3 dashboard.py > /dev/null 2>&1 &
fi

# 5. Summary
DEAD_COUNT=${#DEAD_WORKERS[@]}
IDLE_COUNT=${#IDLE_WORKERS[@]}
NOTE_COUNT=${#NEW_NOTES[@]}

if [ "$DEAD_COUNT" -gt 0 ] || [ "$NOTE_COUNT" -gt 0 ]; then
    echo "[$(date '+%H:%M:%S')] ACTION NEEDED: $DEAD_COUNT dead, $IDLE_COUNT idle, $NOTE_COUNT notes" >> "$LOG"
else
    echo "[$(date '+%H:%M:%S')] All clear: $IDLE_COUNT idle nudged, 0 dead, 0 notes" >> "$LOG"
fi

# 6. Nudge foreman-claude (if running in tmux)
if tmux has-session -t foreman 2>/dev/null; then
    # Check if foreman is idle (at prompt)
    foreman_state=$(tmux capture-pane -t foreman -p 2>/dev/null | tail -10 | grep -c '❯' || echo "0")
    if [ "$foreman_state" -gt 0 ]; then
        SUMMARY="Patrol: $DEAD_COUNT dead, $IDLE_COUNT idle, $NOTE_COUNT notes."
        [ ${#DEAD_WORKERS[@]} -gt 0 ] && SUMMARY="$SUMMARY Restarted: ${DEAD_WORKERS[*]}."
        [ ${#NEW_NOTES[@]} -gt 0 ] && SUMMARY="$SUMMARY Notes from: ${NEW_NOTES[*]}."
        tmux send-keys -t foreman "Patrol nudge. $SUMMARY Do your rounds: check action_log.md, scan pipeline, review any notes in for_foreman-claude/ folders, check researcher/debrief inbox at foreman-staff/*/inbox/. Check dashboard at http://localhost:8420/api/foreman-status." Enter 2>/dev/null || true
        echo "[$(date '+%H:%M:%S')] Nudged foreman" >> "$LOG"
    else
        echo "[$(date '+%H:%M:%S')] Foreman busy, skipping nudge" >> "$LOG"
    fi
fi
