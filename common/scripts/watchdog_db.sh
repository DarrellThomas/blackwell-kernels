# watchdog_db.sh — DB/state helpers for watchdog.sh
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Sourced by watchdog.sh. Requires REPO_ROOT, COMMON_DIR exported.
# No tmux or worker dependencies — pure DB/state operations.

log_watchdog() {
    local category="$1" scope="$2" message="$3"
    echo "[watchdog][$category][$scope] $(date '+%H:%M:%S') $message"
}

get_job_state() {
    local worker_or_kernel="$1"
    python3 - "$worker_or_kernel" <<'PY' 2>/dev/null
import os, sys
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
key = sys.argv[1]
terminal = {'shipped', 'converged', 'parked', 'abandoned'}
mem = ResearchMemory()
rows = [j for j in mem.get_jobs(execution_lane='active', assigned_to=key) if j['state'] not in terminal]
if not rows:
    rows = [j for j in mem.get_jobs(kernel_type=key) if j['state'] not in terminal]
rows.sort(key=lambda j: (int(j['priority']) if str(j['priority']).isdigit() else 99, j['updated_at'], j['id']))
if rows:
    print(rows[0]['state'])
mem.close()
PY
}

get_worker_completion_context() {
    local worker="$1"
    python3 - "$worker" <<'PY' 2>/dev/null
import os, sys
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
worker = sys.argv[1]
terminal = {'shipped', 'converged', 'parked', 'abandoned'}
mem = ResearchMemory()
row = mem.conn.execute(
    "SELECT process_state, job_id FROM worker_state WHERE kernel_type = ?",
    (worker,),
).fetchone()
rows = [j for j in mem.get_jobs(execution_lane='active', assigned_to=worker) if j['state'] not in terminal]
rows.sort(key=lambda j: (int(j['priority']) if str(j['priority']).isdigit() else 99, j['updated_at'], j['id']))
active_job = rows[0] if rows else None
print((row['process_state'] if row else '').strip())
print(str((row['job_id'] if row and row['job_id'] is not None else '')))
print(str(active_job['id']) if active_job else '')
mem.close()
PY
}

get_experiment_count() {
    local kernel="$1" fallback_tsv="${2:-}"
    local count
    count=$(python3 - <<PY 2>/dev/null || true
import os, sys
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
mem = ResearchMemory()
try:
    row = mem.conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE kernel_type = ?",
        (${kernel@Q},)
    ).fetchone()
    print(int(row[0] if row else 0))
finally:
    mem.close()
PY
)
    if [[ "$count" =~ ^[0-9]+$ && "$count" -gt 0 ]]; then
        echo "$count"
        return 0
    fi
    if [[ -n "$fallback_tsv" && -f "$fallback_tsv" ]]; then
        wc -l < "$fallback_tsv"
    else
        echo 0
    fi
}

staff_has_open_work() {
    local agent="$1"
    python3 - <<PY
import os, sys
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
mem = ResearchMemory()
rows = mem.get_messages(status='open', to_agent=${agent@Q}, message_type='research_request')
print('1' if rows else '0')
mem.close()
PY
}

expand_resume_cmd() {
    local worker="$1" raw="$2"
    if [[ "$raw" != @active-prompt:* ]]; then
        printf '%s' "$raw"
        return 0
    fi
    python3 - "$worker" <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory, resolve_job_project_dir

worker = sys.argv[1]
terminal = {'shipped', 'converged', 'parked', 'abandoned'}
mem = ResearchMemory()
rows = mem.get_jobs(execution_lane='active', assigned_to=worker)
rows = [j for j in rows if j['state'] not in terminal]
rows.sort(key=lambda j: (int(j['priority']) if str(j['priority']).isdigit() else 99, j['updated_at'], j['id']))
job = rows[0] if rows else None
if not job:
    print(f"No active DB job is assigned to worker family {worker}. Stay in your dedicated worktree, do not touch shared repo roots, and wait for the next pull from hopper.")
else:
    root = os.environ.get('REPO_ROOT', '')
    common = os.environ.get('COMMON_DIR', root + '/common')
    project_dir = resolve_job_project_dir(job)
    structured_spec = None
    if project_dir is not None:
        candidate = Path(project_dir) / f"docs/Job{job['id']}" / f"job_{job['id']}.json"
        if candidate.exists():
            structured_spec = candidate
    spec_clause = ''
    if structured_spec is not None:
        validator = structured_spec.with_name('validate_job.py')
        schema = structured_spec.with_name('job_schema.json')
        parts = [f"If structured spec files exist at {structured_spec}"]
        if validator.exists() and schema.exists():
            parts.append(f"validate them with python3 {validator} {structured_spec} {schema}")
        parts.append("and treat that JSON packet as the authoritative bounded contract before editing")
        spec_clause = '. '.join(parts) + '. '
    print(
        f"Resume in this dedicated watchdog worktree on active job #{job['id']}: {job['title']}. Assume zero prior model context. "
        f"This project is one subdirectory inside the shared {root} factory, but you are expected to stay inside this isolated git worktree rather than editing the shared checkout. "
        f"The factory database is external to the project tree at {common}/memory/research.db; do not search for a local project database. "
        f"Before substantial exploration, run python3 {common}/memory/factory_brain.py heartbeat {worker} --job {job['id']} --state working --task 'resuming active job and reading spec'. "
        f"Then rebuild context only from python3 {common}/memory/factory_brain.py job-show {job['id']}, python3 {common}/memory/factory_brain.py messages --job {job['id']}, python3 {common}/memory/factory_brain.py experiment-summary --kernel {(job.get('kernel_type') or worker)} --recent 8, and the local files named by the spec. "
        f"{spec_clause}"
        f"Do not begin with generic codebase exploration. Do not rely on prior chat context. Make the smallest change needed for this job, refresh heartbeat during long work, and report status back to the DB. "
        f"Git discipline: commit only on this worker branch before handoff, never edit the shared repo root, and keep ownership to the files implied by the active job. "
        f"When you are done and want the slot reset, create a commit, run python3 {common}/memory/factory_brain.py heartbeat {worker} --job {job['id']} --state complete --task 'done: <summary>; commit <hash>', and send python3 {common}/memory/factory_brain.py message-create --from {worker} --job {job['id']} --subject 'done' --body 'commit=<hash>; summary=<summary>' --type info. "
        f"If you want review before the job moves, use the same complete heartbeat but send subject 'check my work' with message type feedback. "
        f"If you have a problem, keep the heartbeat state as working with a brief task summary and open a blocker or question message instead of guessing. "
        f"Use {common}/csrc and {common}/docs for the shared primitives library and reference material. Ping the researcher (tmux session 'researcher') or open a research request whenever you need additional context or hit 'stuck_needs_research'."
    )
mem.close()
PY
}

touch_tick() {
    local tick_name="$1" status="${2:-ok}" notes="${3:-}"
    python3 - <<PY >/dev/null 2>&1
import os, sys
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
mem = ResearchMemory()
mem.touch_watchdog_state(${tick_name@Q}, status=${status@Q}, notes=${notes@Q})
mem.close()
PY
}

set_daemon_state() {
    local status="$1" notes="${2:-}"
    if ! python3 - <<PY >/dev/null 2>&1
import os, socket, sys
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
mem = ResearchMemory()
mem.set_watchdog_daemon_state(${status@Q}, notes=${notes@Q}, pid=os.getppid(), host=socket.gethostname())
mem.close()
PY
    then
        log_watchdog attention daemon "set_watchdog_daemon_state failed status=$status notes=$notes"
    fi
}

# BUG FIX: was missing `import os` — caused NameError on every call,
# returned empty string, tick_due treated empty as 0, all ticks fired
# every 15s instead of at their scheduled intervals.
get_tick_epoch() {
    local tick_name="$1"
    python3 - <<PY 2>/dev/null
import os, sys, calendar
sys.path.insert(0, os.environ['COMMON_DIR'] + '/memory')
from factory_brain import ResearchMemory
mem = ResearchMemory()
row = mem.get_watchdog_state(${tick_name@Q})
mem.close()
ts = (row or {}).get('last_run_at', '')
if not ts:
    print(0)
else:
    print(calendar.timegm(__import__('time').strptime(ts, '%Y-%m-%dT%H:%M:%SZ')))
PY
}

tick_due() {
    local tick_name="$1" interval="$2"
    local last now
    last="$(get_tick_epoch "$tick_name")"
    now="$(date +%s)"
    (( now - ${last:-0} >= interval ))
}
