# watchdog_workers.sh — Worker lifecycle management for watchdog.sh
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Sourced by watchdog.sh. Requires watchdog_db.sh and watchdog_tmux.sh sourced first.

update_phase_context() {
    local kernel="$1" state="$2"
    local project_dir="$REPO_ROOT/$kernel"
    local target="$project_dir/.claude/phase_context.md"

    local phase_file=""
    case "$state" in
        not_started|algo_building|algo_optimizing|hw_optimizing|stuck_needs_research|research_available)
            phase_file="development.md" ;;
        compiles_ok|tests_writing|testing|testing_pass|testing_fail|edge_testing|edge_pass|edge_fail)
            phase_file="validation.md" ;;
        rework|rework_complete|retesting|retest_pass|retest_fail)
            phase_file="rework.md" ;;
        linting|lint_pass|lint_fail)
            phase_file="quality.md" ;;
        ready_to_ship|shipping|shipped)
            phase_file="shipping.md" ;;
        *)
            phase_file="development.md" ;;
    esac

    if [[ -f "$PHASES_DIR/$phase_file" && -d "$project_dir/.claude" ]]; then
        cp "$PHASES_DIR/$phase_file" "$target"
        log_watchdog phase "$kernel" "context → $phase_file (state=$state)"
    fi
}

ensure_loop_session() {
    local name="$1" session="$2" cwd="${3:-$REPO_ROOT/$name}" launch_cmd="${4:-}" resume_cmd="${5:-}"

    if is_session_alive "$session"; then
        return 0
    fi

    log_watchdog session "$name" "session dead, recreating"
    tmux new-session -d -s "$session" -c "$cwd" bash
    sleep 2
    set_session_window_label "$name" "$session"

    if [[ -n "$launch_cmd" ]]; then
        local expanded_resume=""
        [[ -n "$resume_cmd" ]] && expanded_resume="$(expand_resume_cmd "$name" "$resume_cmd")"
        if is_codex_launch_cmd "$launch_cmd" && [[ -n "$expanded_resume" ]]; then
            launch_codex_fresh "$session" "$launch_cmd" "$expanded_resume"
            sleep "$RESTART_PAUSE"
            if ! confirm_execution_started "$session" "$expanded_resume"; then
                log_watchdog attention "$name" "Codex launch not confirmed; retrying after upgrade prompt handling"
                submit_prompt_and_confirm "$session" "$expanded_resume" || \
                    log_watchdog attention "$name" "prompt submission unconfirmed after launch"
                sleep "$RESTART_PAUSE"
            fi
        else
            tmux send-keys -t "$session" "$launch_cmd" C-m 2>/dev/null
            sleep "$RESTART_PAUSE"
            if [[ -n "$expanded_resume" ]]; then
                submit_prompt_and_confirm "$session" "$expanded_resume" || \
                    log_watchdog attention "$name" "prompt submission unconfirmed after launch"
                sleep "$RESTART_PAUSE"
            fi
        fi
    elif [[ -n "$resume_cmd" ]]; then
        local expanded_resume
        expanded_resume="$(expand_resume_cmd "$name" "$resume_cmd")"
        submit_prompt_and_confirm "$session" "$expanded_resume" || \
            log_watchdog attention "$name" "prompt submission unconfirmed"
        sleep "$RESTART_PAUSE"
    fi
}

restart_loop() {
    local name="$1" session="$2" resume_cmd="$3" launch_cmd="${4:-}" cwd="${5:-$REPO_ROOT/$name}"

    log_watchdog restart "$name" "restarting loop"

    local state
    state=$(get_job_state "$name")
    if [[ -n "$state" ]]; then
        update_phase_context "$name" "$state"
    fi

    if is_codex_launch_cmd "$launch_cmd"; then
        tmux kill-session -t "$session" 2>/dev/null || true
        ensure_loop_session "$name" "$session" "$cwd" "$launch_cmd" "$resume_cmd"
        set_session_window_label "$name" "$session"
        log_watchdog restart "$name" "fresh Codex loop restarted"
        return 0
    fi

    ensure_loop_session "$name" "$session" "$cwd" "$launch_cmd" "$resume_cmd"

    if [[ "$(classify_pane_state "$session")" == "expired" && -n "$launch_cmd" ]]; then
        log_watchdog restart "$name" "expired Codex session detected; relaunching"
        tmux kill-session -t "$session" 2>/dev/null || true
        ensure_loop_session "$name" "$session" "$cwd" "$launch_cmd" "$resume_cmd"
    fi

    tmux send-keys -t "$session" Escape 2>/dev/null
    sleep 2
    tmux send-keys -t "$session" '/clear' C-m 2>/dev/null
    sleep "$RESTART_PAUSE"

    local expanded_resume
    expanded_resume="$(expand_resume_cmd "$name" "$resume_cmd")"
    submit_prompt_and_confirm "$session" "$expanded_resume" || \
        log_watchdog attention "$name" "prompt submission unconfirmed after restart"

    log_watchdog restart "$name" "loop restarted"
}
