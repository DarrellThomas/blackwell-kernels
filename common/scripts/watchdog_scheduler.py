import os
from pathlib import Path
import time
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = REPO_ROOT / "common"
sys.path.insert(0, str(COMMON_DIR / "memory"))
from factory_brain import ResearchMemory  # noqa: E402
sys.path.insert(0, str(COMMON_DIR / "scripts"))
from gate_process import gate_process_job  # noqa: E402

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


def fill_active_lanes(mem: ResearchMemory, managed_workers):
    terminal = {'shipped', 'converged', 'parked', 'abandoned'}

    for worker, slot_count in managed_workers.items():
        active_list = [j for j in mem.get_jobs(execution_lane='active', assigned_to=worker) if j['state'] not in terminal]

        # Normalize any legacy promotions that were left in wishlist/planning state
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
            hopper_candidates = [j for j in mem.get_jobs(execution_lane='hopper') if j['state'] not in terminal]
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
    terminal = {'shipped', 'converged', 'parked', 'abandoned'}
    worker_rows = {w['kernel_type']: w for w in mem.get_worker_state()}
    for job in mem.get_jobs(execution_lane='active'):
        if job['state'] in terminal:
            continue
        worker_key = (job.get('assigned_to') or job.get('kernel_type') or '').strip()
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


def main():
    mem = ResearchMemory()
    try:
        mem.refresh_worker_state()
        managed_workers = build_managed_workers()
        fill_active_lanes(mem, managed_workers)
        gate_process_active(mem)
        timeout = int(os.environ.get('ACTIVE_SIT_TIMEOUT', '300'))
        flag_sitting_jobs(mem, timeout)
    finally:
        mem.close()


if __name__ == '__main__':
    main()
