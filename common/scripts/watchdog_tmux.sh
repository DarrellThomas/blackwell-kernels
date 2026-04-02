# helper functions for watchdog tmux/session management

capture_pane_tail() {
    local session="$1" lines="${2:-40}"
    python3 - <<PY
import subprocess
session = ${session@Q}
limit = int(${lines@Q})
text = subprocess.check_output(['tmux', 'capture-pane', '-pt', session]).decode('utf-8', 'ignore').splitlines()
nonempty = [ln for ln in text if ln.strip()]
for line in nonempty[-limit:]:
    print(line)
PY
}

classify_pane_state() {
    local session="$1"
    local pane_text
    pane_text="$(capture_pane_tail "$session" 50)"

    # Codex upgrade selection menu (GPT-5.4 prompt)
    if echo "$pane_text" | grep -Eq 'Try new model|Use existing model|Introducing GPT-5\.4'; then
        echo "codex_upgrade"
        return 0
    fi

    if echo "$pane_text" | grep -Eq 'Working \(|esc to interrupt|Ran |Updated |Reading |thinking|Tokens? usage:'; then
        echo "active"
        return 0
    fi

    if echo "$pane_text" | grep -Eq 'To continue this session, run codex resume '; then
        echo "expired"
        return 0
    fi

    if echo "$pane_text" | grep -Eq "Tip: .*Codex|OpenAI Codex \(v|model: .* /model to change|directory: ${REPO_ROOT}"; then
        if echo "$pane_text" | grep -Eq "Implement \{feature\}|Write tests for @filename|Explain this codebase|gpt-5\.4 .* ${REPO_ROOT}"; then
            echo "codex_welcome"
            return 0
        fi
    fi

    if echo "$pane_text" | grep -Eq '^[^[:space:]@]+@[^[:space:]]+:.+[#$] ?$'; then
        echo "shell"
        return 0
    fi

    if echo "$pane_text" | grep -Fq '› '; then
        echo "composer"
        return 0
    fi

    if echo "$pane_text" | grep -Fq '❯ '; then
        echo "claude_idle"
        return 0
    fi

    echo "unknown"
}

confirm_execution_started() {
    local session="$1" prompt="$2"
    local pane_text
    pane_text="$(capture_pane_tail "$session" 50)"

    if echo "$pane_text" | grep -Eq 'Try new model|Use existing model|Introducing GPT-5\.4'; then
        tmux send-keys -t "$session" "1" C-m 2>/dev/null
        return 1
    fi

    if echo "$pane_text" | grep -Eq 'To continue this session, run codex resume '; then
        return 1
    fi

    if echo "$pane_text" | grep -Eq 'Implement \{feature\}|Write tests for @filename|Explain this codebase'; then
        return 1
    fi

    if echo "$pane_text" | grep -Fq "$prompt"; then
        return 1
    fi

    if echo "$pane_text" | grep -Fq '› '; then
        return 1
    fi

    if echo "$pane_text" | grep -Eq 'Working \(|esc to interrupt|Ran |Updated |Reading |thinking'; then
        return 0
    fi

    return 0
}

submit_prompt_and_confirm() {
    local session="$1" prompt="$2"
    local tries=0 state=""

    while (( tries < PROMPT_CONFIRM_RETRIES )); do
        state="$(classify_pane_state "$session")"
        log_watchdog prompt "$session" "state=$state try=$((tries + 1))"

        case "$state" in
            codex_upgrade)
                tmux send-keys -t "$session" "1" C-m 2>/dev/null
                sleep 1
                continue
                ;;
            active)
                return 0
                ;;
            shell|claude_idle)
                tmux send-keys -t "$session" '/clear' C-m 2>/dev/null
                sleep 1
                tmux set-buffer -- "$prompt"
                tmux paste-buffer -t "$session" 2>/dev/null
                tmux send-keys -t "$session" C-m 2>/dev/null
                ;;
            codex_welcome|unknown)
                tmux set-buffer -- "$prompt"
                tmux paste-buffer -t "$session" 2>/dev/null
                tmux send-keys -t "$session" C-m 2>/dev/null
                ;;
            composer)
                if ! capture_pane_tail "$session" 20 | grep -Fq "$prompt"; then
                    tmux set-buffer -- "$prompt"
                    tmux paste-buffer -t "$session" 2>/dev/null
                    sleep 1
                fi
                tmux send-keys -t "$session" C-m 2>/dev/null
                ;;
            expired)
                return 1
                ;;
        esac

        sleep "$PROMPT_CONFIRM_WAIT"
        if confirm_execution_started "$session" "$prompt"; then
            log_watchdog prompt "$session" "execution confirmed"
            return 0
        fi

        ((tries++))
    done

    log_watchdog attention "$session" "execution not confirmed after prompt submission"
    return 1
}

is_codex_launch_cmd() {
    local launch_cmd="${1:-}"
    [[ "$launch_cmd" == codex* ]]
}

launch_codex_fresh() {
    local session="$1" launch_cmd="$2" prompt="$3"
    local launch_line
    printf -v launch_line '%s %q' "$launch_cmd" "$prompt"
    tmux send-keys -t "$session" "$launch_line" C-m 2>/dev/null
}

is_session_alive() {
    local session="$1"
    tmux has-session -t "$session" 2>/dev/null
}

is_eval_running() {
    local session="$1"
    tmux capture-pane -t "$session" -p 2>/dev/null | tail -3 | grep -q "eval.sh" && return 0
    return 1
}

is_worker_idle() {
    local session="$1"
    local pane_text
    pane_text=$(capture_pane_tail "$session" 20)
    # Codex always shows › in its UI even while actively executing — check active
    # markers first. "Working (" and "esc to interrupt" mean Codex is mid-task.
    if echo "$pane_text" | grep -Eq 'Working \(|esc to interrupt|Ran |Updated |Reading '; then
        return 1  # not idle
    fi
    if echo "$pane_text" | grep -Eq '[❯›]'; then
        return 0  # idle at prompt
    fi
    if echo "$pane_text" | grep -Eq '^[^[:space:]@]+@[^[:space:]]+:.+[#$] ?$'; then
        return 0  # idle at shell prompt
    fi
    return 1
}

get_active_job_label() {
    local worker="$1"
    python3 - <<PY
import sys
import os
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
mem = ResearchMemory()
worker = ${worker@Q}
rows = mem.get_jobs(execution_lane='active', assigned_to=worker)
rows = [j for j in rows if j['state'] not in ('shipped','converged','parked','abandoned')]
rows.sort(key=lambda j: (int(j['priority']) if str(j['priority']).isdigit() else 99, j['updated_at'], j['id']))
job = rows[0] if rows else None
if job:
    print(f"{worker} #{job['id']}")
else:
    print(worker)
mem.close()
PY
}

set_session_window_label() {
    local worker="$1" session="$2"
    local label
    label="$(get_active_job_label "$worker")"
    tmux rename-window -t "$session:0" "$label" 2>/dev/null || true
}
