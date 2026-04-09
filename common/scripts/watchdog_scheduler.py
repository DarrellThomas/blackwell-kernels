import argparse
import os
from pathlib import Path
import re
import subprocess
import time
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = REPO_ROOT / "common"
sys.path.insert(0, str(COMMON_DIR / "memory"))
from factory_brain import ResearchMemory, detect_worker_job_signal, get_active_worker_jobs  # noqa: E402
sys.path.insert(0, str(COMMON_DIR / "scripts"))
from gate_process import gate_process_job  # noqa: E402

TERMINAL_STATES = {'shipped', 'converged', 'parked', 'abandoned'}
DONE_SUBJECT_RE = re.compile(r'^(done|.*ready for handoff.*)$', re.IGNORECASE)
COMMIT_RE = re.compile(r'\bcommit(?:=|:|\s+)([0-9a-f]{7,40})\b', re.IGNORECASE)


def lane_entries():
    override = os.environ.get('MANAGED_WORKER_SLOTS', '').strip()
    if override:
        return override.split()
    cfg_path = Path(os.environ.get('WORKER_LANES_CONFIG', str(COMMON_DIR / 'scripts' / 'worker_lanes.conf')))
    if cfg_path.is_file():
        entries = []
        for line in cfg_path.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                entries.append(line)
        return entries
    return os.environ.get('DEFAULT_MANAGED_WORKER_SLOTS', 'gemm:1 octave-gpu:2').split()


def build_managed_workers():
    managed_workers = {}
    for entry in lane_entries():
        if ':' not in entry:
            continue
        worker, slot_value = entry.split(':', 1)
        try:
            slots = max(1, int(slot_value))
        except ValueError:
            slots = 1
        managed_workers[worker.strip()] = slots
    if not managed_workers:
        managed_workers = {'gemm': 1, 'octave-gpu': 2}
    return managed_workers


def _open_worker_messages(mem: ResearchMemory, worker: str, job_id: int | None):
    if job_id is None:
        return []
    return mem.get_messages(status='open', job_id=job_id, from_agent=worker)


def _extract_commit_hint(worker_state: dict, messages: list[dict]) -> str:
    texts = [(worker_state or {}).get('current_task', '')]
    for row in messages:
        texts.append(row.get('subject') or '')
        texts.append(row.get('body') or '')
    for text in texts:
        match = COMMIT_RE.search(text or '')
        if match:
            return match.group(1)
    return ''


def _verify_committed_worktree(worktree_path: str, commit_hint: str) -> tuple[bool, str, str]:
    if not worktree_path:
        return False, 'worktree path missing', ''
    worktree = Path(worktree_path)
    if not worktree.exists():
        return False, f'worktree missing: {worktree}', ''
    try:
        head = subprocess.check_output(
            ['git', '-C', str(worktree), 'rev-parse', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        return False, f'git rev-parse HEAD failed: {exc}', ''
    try:
        dirty = subprocess.check_output(
            ['git', '-C', str(worktree), 'status', '--porcelain'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        return False, f'git status failed: {exc}', head
    # Only flag modified/added/deleted tracked files as dirty.
    # Untracked (??) files are harmless worker artifacts (docs, build outputs).
    tracked_dirty = [l for l in dirty.splitlines() if not l.startswith('?? ')]
    if tracked_dirty:
        snippet = '\n'.join(tracked_dirty[:12])
        return False, f'worktree dirty:\n{snippet}', head
    if commit_hint:
        try:
            resolved = subprocess.check_output(
                ['git', '-C', str(worktree), 'rev-parse', '--verify', f'{commit_hint}^{{commit}}'],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return False, f'referenced commit not found in worktree: {commit_hint}', head
        if resolved != head and not head.startswith(commit_hint):
            return False, f'referenced commit {commit_hint} does not match HEAD {head[:12]}', head
    return True, f'HEAD {head[:12]} verified', head


def _clear_worker_completion(mem: ResearchMemory, worker: str):
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    mem.conn.execute(
        """
        UPDATE worker_state
        SET process_state = '', current_task = '', job_id = NULL, updated_at = ?
        WHERE kernel_type = ?
        """,
        (now, worker),
    )
    mem.conn.commit()


def _resolve_inactive_done_messages(mem: ResearchMemory):
    for row in mem.get_messages(status='open'):
        subject = (row.get('subject') or '').strip()
        if not DONE_SUBJECT_RE.match(subject):
            continue
        job_id = row.get('job_id')
        job = mem.get_job(job_id) if job_id is not None else None
        if not job or (job.get('execution_lane') or '').strip() != 'active':
            mem.resolve_message(row['id'], by='watchdog')


def consume_worker_handoff(mem: ResearchMemory, worker: str, worktree_path: str) -> dict:
    worker_row = mem.conn.execute(
        "SELECT kernel_type, heartbeat_at, current_task, process_state, job_id FROM worker_state WHERE kernel_type = ?",
        (worker,),
    ).fetchone()
    worker_state = dict(worker_row) if worker_row else {}
    reported_job_id = worker_state.get('job_id')
    active_jobs = get_active_worker_jobs(mem, worker, exclude_done_handoffs=True)
    active_job = active_jobs[0] if active_jobs else None
    open_messages = _open_worker_messages(mem, worker, reported_job_id)
    signal = detect_worker_job_signal(mem, worker, reported_job_id, worker_state=worker_state)

    if signal == 'problem':
        detail = open_messages[0].get('subject') if open_messages else worker_state.get('current_task', 'problem reported')
        if reported_job_id is not None:
            body = (
                f"Worker {worker} reported a problem on job #{reported_job_id}.\n"
                f"task={worker_state.get('current_task', '')}\n"
                f"latest={detail}"
            )
            mem.ensure_open_message(
                'watchdog',
                f'Problem reported for job #{reported_job_id}',
                body=body,
                job_id=reported_job_id,
                message_type='blocker',
                priority='normal',
            )
        return {'action': 'hold', 'detail': detail or 'worker problem reported'}

    if worker_state.get('process_state') != 'complete':
        return {'action': 'none', 'detail': ''}

    if reported_job_id is None:
        _clear_worker_completion(mem, worker)
        return {'action': 'reset', 'detail': 'stale complete heartbeat with no job id'}

    job = mem.get_job(reported_job_id)
    if not job or (job.get('execution_lane') or '').strip() != 'active':
        _resolve_inactive_done_messages(mem)
        _clear_worker_completion(mem, worker)
        return {'action': 'reset', 'detail': f'completed handoff job #{reported_job_id} is no longer active'}

    if signal == 'check_my_work':
        commit_hint = _extract_commit_hint(worker_state, open_messages)
        ok, reason, head = _verify_committed_worktree(worktree_path, commit_hint)
        if not ok:
            mem.ensure_open_message(
                'watchdog',
                f'Review handoff blocked for job #{reported_job_id}',
                body=reason,
                job_id=reported_job_id,
                message_type='blocker',
                priority='normal',
            )
            return {'action': 'hold', 'detail': f'check-my-work blocked: {reason}'}
        body = (
            f"Worker {worker} requested review for job #{reported_job_id}.\n"
            f"commit={head}\n"
            f"task={worker_state.get('current_task', '')}"
        )
        mem.ensure_open_message(
            'watchdog',
            f'Review requested for job #{reported_job_id}',
            body=body,
            job_id=reported_job_id,
            message_type='feedback',
            priority='normal',
        )
        return {'action': 'hold', 'detail': f'review requested for job #{reported_job_id} at {head[:12]}'}

    if signal == 'done':
        commit_hint = _extract_commit_hint(worker_state, open_messages)
        ok, reason, head = _verify_committed_worktree(worktree_path, commit_hint)
        if not ok:
            mem.ensure_open_message(
                'watchdog',
                f'Done handoff blocked for job #{reported_job_id}',
                body=reason,
                job_id=reported_job_id,
                message_type='blocker',
                priority='normal',
            )
            return {'action': 'hold', 'detail': f'done blocked: {reason}'}
        # If the worker reports done but left the job in an intermediate
        # validation state, advance to testing_pass so gate processing can
        # pick it up and run the compliance/edge/lint suite.
        job = mem.get_job(reported_job_id)
        pre_gate_states = {'compiles_ok', 'tests_writing', 'testing'}
        if job and job.get('state') in pre_gate_states:
            mem.update_job_state(
                reported_job_id, 'testing_pass', 'watchdog',
                f'worker {worker} reported done from {job["state"]}; advancing to testing_pass for gate processing',
            )
        gate_process_job(reported_job_id)
        body = (
            f"Worker {worker} completed a done handoff for job #{reported_job_id}.\n"
            f"commit={head}\n"
            f"task={worker_state.get('current_task', '')}\n"
            f"Slot reset is allowed; the open done message keeps this job out of worker selection until gate/manual follow-up finishes."
        )
        mem.ensure_open_message(
            'watchdog',
            f'Handoff accepted for job #{reported_job_id}',
            body=body,
            job_id=reported_job_id,
            message_type='info',
            priority='normal',
        )
        _resolve_inactive_done_messages(mem)
        _clear_worker_completion(mem, worker)
        return {'action': 'reset', 'detail': f'done handoff accepted for job #{reported_job_id} at {head[:12]}'}

    if active_job and reported_job_id == active_job.get('id'):
        mem.ensure_open_message(
            'watchdog',
            f'Completion missing handoff signal for job #{reported_job_id}',
            body=(
                f"Worker {worker} reported process_state=complete for active job #{reported_job_id} "
                f"but did not send a recognized done/check-my-work/problem signal.\n"
                f"task={worker_state.get('current_task', '')}"
            ),
            job_id=reported_job_id,
            message_type='info',
            priority='normal',
        )
        return {'action': 'hold', 'detail': f'complete heartbeat for active job #{reported_job_id} is missing a handoff signal'}

    _clear_worker_completion(mem, worker)
    return {'action': 'reset', 'detail': f'completed job #{reported_job_id} no longer owns the active slot'}


def fill_active_lanes(mem: ResearchMemory, managed_workers):
    for worker, slot_count in managed_workers.items():
        active_list = get_active_worker_jobs(mem, worker, exclude_done_handoffs=True)

        normalized = []
        for j in active_list:
            if (j.get('state') or '').strip() in ('wishlist', 'planning'):
                mem.update_job_state(j['id'], 'not_started', 'watchdog', reason='promoted into active lane')
                j = dict(j)
                j['state'] = 'not_started'
                j['phase'] = 'development'
            normalized.append(j)
        active_list = normalized

        while len(active_list) < slot_count:
            hopper_candidates = [j for j in mem.get_jobs(execution_lane='hopper') if j['state'] not in TERMINAL_STATES]
            hopper = [j for j in hopper_candidates if (j.get('assigned_to') or j.get('kernel_type') or '').strip() == worker]
            if not hopper:
                break
            hopper.sort(key=lambda j: (
                int(j['priority']) if str(j.get('priority', '')).isdigit() else 99,
                j.get('updated_at') or '',
                j.get('id') or 0,
            ))
            nxt = dict(hopper[0])
            mem.update_job(nxt['id'], updated_by='watchdog', execution_lane='active')
            if (nxt.get('state') or '').strip() in ('wishlist', 'planning'):
                mem.update_job_state(nxt['id'], 'not_started', 'watchdog', reason='promoted into active lane')
                nxt['state'] = 'not_started'
                nxt['phase'] = 'development'
            mem.create_message('watchdog', f"Pulled job #{nxt['id']} into active lane",
                               body=(f"Worker family '{worker}' runs {slot_count} concurrent slots. "
                                     f"Promoted hopper job #{nxt['id']} ({nxt['title']}) into active lane."),
                               job_id=nxt['id'], message_type='info', priority='normal')
            active_list.append(nxt)


def gate_process_active(mem: ResearchMemory):
    gate_states = ['testing_pass', 'edge_pass', 'lint_pass', 'ready_to_ship',
                   'testing_fail', 'edge_fail', 'lint_fail',
                   'stuck_needs_research', 'research_available']
    for job in mem.get_jobs(execution_lane='active'):
        if job['state'] in gate_states:
            gate_process_job(job['id'])


def flag_sitting_jobs(mem: ResearchMemory, active_sit_timeout: int):
    worker_rows = {w['kernel_type']: w for w in mem.get_worker_state()}
    for job in mem.get_jobs(execution_lane='active'):
        if job['state'] in TERMINAL_STATES:
            continue
        worker_key = (job.get('assigned_to') or job.get('kernel_type') or '').strip()
        if detect_worker_job_signal(mem, worker_key, job.get('id'), worker_state=worker_rows.get(worker_key) or {}) == 'done':
            continue
        worker = worker_rows.get(worker_key) or {}
        live = (worker.get('live_status') or '').strip()
        activity_at = (worker.get('activity_at') or worker.get('heartbeat_at') or '').strip()
        age_s = None
        if activity_at:
            try:
                import calendar
                age_s = int(time.time() - calendar.timegm(time.strptime(activity_at, '%Y-%m-%dT%H:%M:%SZ')))
            except Exception:
                age_s = None
        if live == 'active':
            continue
        if age_s is not None and age_s < active_sit_timeout:
            continue
        detail = worker.get('live_reason') or worker.get('diagnosis') or 'no active worker heartbeat'
        body = (
            f"Active-lane job #{job['id']} ({job['title']}) is sitting without an active worker heartbeat. "
            f"Worker family: {worker_key or '-'}; live={live or 'historical'}; detail={detail}. "
            f"Check the tmux session, nudge the worker, or demote the job if it should not block the hopper."
        )
        mem.ensure_open_message('watchdog',
                                f"Active lane sitting: job #{job['id']}",
                                body=body, job_id=job['id'],
                                message_type='info', priority='normal')
        print(f"[watchdog][attention][{worker_key or job['id']}] active job #{job['id']} sitting")


def run_scheduler_tick():
    mem = ResearchMemory()
    try:
        mem.refresh_worker_state()
        managed_workers = build_managed_workers()
        fill_active_lanes(mem, managed_workers)
        gate_process_active(mem)
        _resolve_inactive_done_messages(mem)
        timeout = int(os.environ.get('ACTIVE_SIT_TIMEOUT', '300'))
        flag_sitting_jobs(mem, timeout)
    finally:
        mem.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Watchdog lane scheduler and handoff consumer')
    parser.add_argument('--consume-worker-handoff', dest='worker', help='Worker slot to classify/consume')
    parser.add_argument('--worktree', help='Dedicated worktree path for the worker handoff check')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        mem = ResearchMemory()
        try:
            mem.refresh_worker_state()
            result = consume_worker_handoff(mem, args.worker, args.worktree or '')
        finally:
            mem.close()
        print(result['action'])
        print(result['detail'])
        return
    run_scheduler_tick()


if __name__ == '__main__':
    main()
