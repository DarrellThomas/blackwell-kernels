# watchdog_ticks.sh — Periodic tick functions for watchdog.sh
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Sourced by watchdog.sh. Requires watchdog_db.sh, watchdog_tmux.sh,
# and watchdog_workers.sh sourced first.

tick_short() {
    local now current new_iters reason wait_count idle_secs stall_secs

    for loop in "${LOOPS[@]}"; do
        IFS='|' read -r name session tsv resume_cmd launch_cmd cwd <<< "$loop"
        cwd="${cwd:-$REPO_ROOT/$name}"

        ensure_loop_session "$name" "$session" "$cwd" "$launch_cmd" "$resume_cmd"

        current=$(get_experiment_count "$name" "$tsv")
        new_iters=$((current - LAST_LINE_COUNT[$name]))
        now=$(date +%s)
        reason=""

        if (( new_iters > 0 )); then
            LAST_CHANGE_TIME[$name]=$now
            LAST_IDLE_TIME[$name]=0
        fi

        if is_worker_idle "$session"; then
            if (( LAST_IDLE_TIME[$name] == 0 )); then
                LAST_IDLE_TIME[$name]=$now
            fi
            idle_secs=$(( now - LAST_IDLE_TIME[$name] ))
            if (( idle_secs >= IDLE_TIMEOUT )); then
                reason="idle at prompt (${idle_secs}s)"
            fi
        else
            LAST_IDLE_TIME[$name]=0
        fi

        if [[ -z "$reason" ]] && (( new_iters >= MAX_ITERS )); then
            reason="iteration limit ($new_iters >= $MAX_ITERS)"
        fi

        if [[ -z "$reason" ]]; then
            stall_secs=$(( now - LAST_CHANGE_TIME[$name] ))
            if (( stall_secs >= STALL_TIMEOUT )); then
                reason="stall timeout (${stall_secs}s with no new iteration)"
            fi
        fi

        if [[ -n "$reason" ]]; then
            log_watchdog restart "$name" "$reason"
            wait_count=0
            while is_eval_running "$session" && (( wait_count < 30 )); do
                log_watchdog wait "$name" "eval running, waiting"
                sleep 10
                ((wait_count++))
            done
            restart_loop "$name" "$session" "$resume_cmd" "$launch_cmd" "$cwd"
            LAST_LINE_COUNT[$name]=$current
            LAST_CHANGE_TIME[$name]=$(date +%s)
            LAST_IDLE_TIME[$name]=0
        fi
    done

    if TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 \
        python3 "$COMMON_DIR/scripts/watchdog_scheduler.py" >/dev/null 2>&1; then
        log_watchdog tick short "job gate processing complete"
    else
        log_watchdog tick short "job gate processing error (non-fatal)"
    fi

    if ! curl -sf "http://localhost:8421/api/stats" >/dev/null 2>&1; then
        log_watchdog service memory "server down; restarting"
        "$COMMON_DIR/memory/start-server.sh" --daemon
    fi

    for entry in "${STAFF_LOOPS[@]}"; do
        IFS='|' read -r name session resume_cmd launch_cmd cwd <<< "$entry"
        cwd="${cwd:-$REPO_ROOT/$name}"
        if [[ "$(staff_has_open_work "$name")" != "1" ]]; then
            continue
        fi
        ensure_loop_session "$name" "$session" "$cwd" "$launch_cmd" "$resume_cmd"

        now=$(date +%s)
        if is_worker_idle "$session"; then
            if (( ${LAST_IDLE_TIME[$name]:-0} == 0 )); then
                LAST_IDLE_TIME[$name]=$now
            fi
            idle_secs=$(( now - LAST_IDLE_TIME[$name] ))
            if (( idle_secs >= IDLE_TIMEOUT )); then
                log_watchdog nudge "$name" "idle at prompt (${idle_secs}s) with open research work"
                tmux send-keys -t "$session" '/clear' C-m 2>/dev/null
                sleep "$RESTART_PAUSE"
                local expanded_resume
                expanded_resume="$(expand_resume_cmd "$name" "$resume_cmd")"
                if submit_prompt_and_confirm "$session" "$expanded_resume"; then
                    log_watchdog nudge "$name" "resume prompt confirmed"
                else
                    log_watchdog attention "$name" "research prompt submission unconfirmed"
                fi
                LAST_IDLE_TIME[$name]=0
            fi
        else
            LAST_IDLE_TIME[$name]=0
        fi
    done
}

tick_medium() {
    # Ingest new content and refresh worker state
    TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 python3 - <<'PY' >/dev/null 2>&1 || true
import os, sys
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
mem = ResearchMemory()
mem.ingest_all()
mem.ingest_all_tsv()
mem.refresh_worker_state()
mem.close()
PY
    log_watchdog tick medium "ingest + worker refresh complete"

    # Metadata hygiene: log-only. These are operational noise — not DB messages.
    # (Previously used create_message() which spammed the DB every 15s due to
    # the get_tick_epoch bug. Now just goes to the watchdog log.)
    TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 python3 - <<'PY' 2>&1 \
        | sed 's/^/[watchdog][hygiene] /' || true
import os, sys
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
mem = ResearchMemory()
for job in mem.get_jobs():
    mode = job.get('factory_mode', '')
    if not mode:
        print(f"Job #{job['id']} missing factory mode")
        continue
    scope = (job.get('optimization_scope') or '').strip()
    if not scope:
        print(f"Job #{job['id']} missing optimization scope")
        continue
    required = ['objective_vector', 'acceptance_gates', 'keep_rule', 'benchmark_set']
    missing = [f for f in required if not (job.get(f) or '').strip()]
    if scope in ('hardware_tuned', 'hybrid'):
        for field in ('hardware_target', 'retarget_policy'):
            if not (job.get(field) or '').strip():
                missing.append(field)
    if missing:
        print(f"Job #{job['id']} missing objective metadata: {', '.join(missing)}")
mem.close()
PY
    log_watchdog tick medium "metadata hygiene complete"
}

tick_long() {
    if pgrep -af "generate_summaries.py --limit 50" >/dev/null 2>&1; then
        log_watchdog tick long "summary generation already running"
        return 0
    fi
    log_watchdog tick long "starting summary generation batch"
    python3 "$COMMON_DIR/memory/generate_summaries.py" --limit 50 >/dev/null 2>&1 || true
    log_watchdog tick long "summary generation batch complete"

    # Promotion candidates: ensure_open_message so we don't re-create each long tick
    TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3 python3 - <<'PY' >/dev/null 2>&1 || true
import os, sys
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
mem = ResearchMemory()
candidates = mem.conn.execute("""
    SELECT d.id, d.title
    FROM documents d
    WHERE d.provenance = 'research'
      AND d.is_empirical = 1
      AND d.doc_type NOT IN ('experiment', 'dead_end')
    ORDER BY d.kernel_type, d.title
    LIMIT 20
""").fetchall()
if candidates:
    body = "\n".join(f"#{r['id']}: {r['title']}" for r in candidates)
    mem.ensure_open_message('watchdog', 'Promotion candidates available',
                            body=body, message_type='info', priority='normal')
mem.close()
PY
    log_watchdog tick long "research audit complete"
}
