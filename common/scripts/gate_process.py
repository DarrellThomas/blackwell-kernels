# gate_process.py — Shared gate logic for watchdog.sh and fb nudge
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# Single source of truth for job state transitions.
# Called by watchdog.sh (all jobs) and fb nudge (one job).

import glob
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = REPO_ROOT / "common"
sys.path.insert(0, str(COMMON_DIR / "memory"))
from factory_brain import ResearchMemory

SCRIPTS = str(COMMON_DIR / "scripts")
PHASES_DIR = str(COMMON_DIR / "claude" / "phases")


TEST_CATEGORY_PATTERNS = {
    'stress': [
        'tests/test_stress_cases.py',
        'tests/test_stress_{kernel}.py',
        'tests/test_stress_{kernel_slug}.py',
        'tests/test_stress.py',
    ],
    'edge': [
        'tests/test_edge_cases.py',
        'tests/test_edge_{kernel}.py',
        'tests/test_edge_{kernel_slug}.py',
        'tests/test_{kernel}.py',
    ],
    'corner': [
        'tests/test_corner_cases.py',
        'tests/test_corner_{kernel}.py',
        'tests/test_corner_{kernel_slug}.py',
    ],
    'security': [
        'tests/test_security.py',
        'tests/test_security_{kernel}.py',
        'tests/test_security_{kernel_slug}.py',
    ],
    'accuracy': [
        'tests/test_accuracy.py',
        'tests/test_accuracy_{kernel}.py',
        'tests/test_accuracy_{kernel_slug}.py',
        'tests/test_{kernel}_accuracy.py',
    ],
    'speed': [
        'benchmarks/bench_{kernel}.py',
        'benchmarks/bench_{kernel_slug}.py',
        'benchmarks/bench_speed.py',
    ],
}

ALL_TEST_CATEGORIES = list(TEST_CATEGORY_PATTERNS.keys())


_PHASE_MAP = {
    'development': ['not_started', 'algo_building', 'algo_optimizing', 'hw_optimizing', 'stuck_needs_research', 'research_available'],
    'validation': ['compiles_ok', 'tests_writing', 'testing', 'testing_pass', 'testing_fail', 'edge_testing', 'edge_pass', 'edge_fail'],
    'rework': ['rework', 'rework_complete', 'retesting', 'retest_pass', 'retest_fail'],
    'quality': ['linting', 'lint_pass', 'lint_fail'],
    'shipping': ['ready_to_ship', 'shipping', 'shipped'],
}

def format_command(command):
    return ' '.join(shlex.quote(part) for part in command)


def run_gate_command(mem, job_id, kernel, project_dir, env, category, command, timeout=600):
    cmd_desc = format_command(command)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=project_dir, env=env)
        output = (result.stdout or '') + (result.stderr or '')
        status = 'pass' if result.returncode == 0 else 'fail'
        success = result.returncode == 0
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or '') + (exc.stderr or '') or str(exc)
        status = 'timeout'
        success = False
    except Exception as exc:
        output = str(exc)
        status = 'error'
        success = False
    trace = output.strip()[-1200:]
    mem.record_test_run(job_id, kernel, category, cmd_desc, status, trace)
    tag = 'PASSED' if success else 'FAILED'
    print(f'  → {category} {cmd_desc} {tag}')
    return success, trace


def find_test_file(project_dir, kernel, category, executed):
    kernel_slug = kernel.replace('_', '-')
    for pattern in TEST_CATEGORY_PATTERNS.get(category, []):
        candidate = Path(project_dir) / pattern.format(kernel=kernel, kernel_slug=kernel_slug)
        if candidate.is_file() and str(candidate) not in executed:
            return candidate
    return None


def ensure_repo_clean(project_dir):
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd=project_dir)
    dirty = result.stdout.strip()
    if dirty:
        return False, dirty
    return True, ''

def resolve_job_project_dir(job):
    assigned = (job.get('assigned_to') or '').strip()
    if assigned in ('octave-gpu', 'cx1', 'cx2', 'cx3', 'cx4'):
        candidate = REPO_ROOT / 'octave-gpu'
        if candidate.is_dir():
            return candidate

    kernel = (job.get('kernel_type') or '').strip()
    if kernel:
        candidate = REPO_ROOT / kernel
        if candidate.is_dir():
            return candidate

    return None


def update_phase_context(project_dir, new_state):
    """Copy the right phase context file into the worker's .claude/ directory."""
    target = str(Path(project_dir) / '.claude/phase_context.md')
    phase = 'development'
    for p, states in _PHASE_MAP.items():
        if new_state in states:
            phase = p
            break
    src = f'{PHASES_DIR}/{phase}.md'
    if os.path.isfile(src) and os.path.isdir(os.path.dirname(target)):
        shutil.copy2(src, target)
        print(f'[gate] {Path(project_dir).name}: phase context -> {phase}.md')


def gate_process_job(job_id):
    """Run gate logic for a single job. Same code path as watchdog."""
    mem = ResearchMemory()
    job = mem.get_job(job_id)
    if not job:
        print(f'Job #{job_id} not found.', file=sys.stderr)
        mem.close()
        return

    jid, name, state, kernel = job['id'], job['name'], job['state'], job['kernel_type']
    project_path = resolve_job_project_dir(job)
    project_dir = str(project_path) if project_path else None
    print(f'Job #{jid} ({name}): {state} [{job["phase"]}]')


    # --- testing_pass → run compliance + extended gate suite ---
    if state == 'testing_pass' and project_dir:
        print(f'  → running edge tests and compliance suite...')
        mem.update_job_state(jid, 'edge_testing', 'gate', 'running compliance + edge tests')
        if project_dir: update_phase_context(project_dir, 'edge_testing')

        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONPATH=f'{project_dir}/python')
        all_passed = True
        gate_output = []
        executed_files = set()

        compliance_script = f'{SCRIPTS}/blas_compliance.py'
        compliance_config = f'{project_dir}/.claude/compliance_args.txt'
        if os.path.isfile(compliance_script) and os.path.isfile(compliance_config):
            with open(compliance_config) as f:
                compliance_args = [line.strip() for line in f if line.strip()]
            for args_line in compliance_args:
                args = args_line.split()
                command = ['python3', compliance_script] + args
                ok, trace = run_gate_command(mem, jid, kernel, project_dir, env, 'compliance', command, timeout=900)
                all_passed &= ok
                if not ok:
                    gate_output.append(f'BLAS compliance FAILED ({" ".join(args)}): {trace}')
        else:
            note = 'Compliance configuration missing; add .claude/compliance_args.txt'
            mem.record_test_run(jid, kernel, 'compliance', '<missing>', 'skipped', note)
            gate_output.append(note)
            all_passed = False

        missing_categories = []
        for category in ALL_TEST_CATEGORIES:
            candidate = find_test_file(project_dir, kernel, category, executed_files)
            if candidate:
                executed_files.add(str(candidate))
                command = ['python3', str(candidate)]
                ok, trace = run_gate_command(mem, jid, kernel, project_dir, env, category, command, timeout=600)
                all_passed &= ok
                if not ok:
                    gate_output.append(f'{candidate.name} FAILED ({category}): {trace}')
            else:
                mem.record_test_run(jid, kernel, category, '<missing>', 'skipped', f'No {category} test script found')
                missing_categories.append(category)

        if missing_categories and all_passed:
            # Existing tests all pass but some suites haven't been written yet.
            # This is normal for a newly minted algo — route to tests_writing so
            # the worker builds them rather than treating absence as failure.
            cats = ', '.join(missing_categories)
            body = (
                f"Write missing test suites: {cats}.\n"
                f"Use tests/ dir. For Octave work, create a thin Python wrapper in tests/ "
                f"that invokes octave-cli and exits with its return code.\n"
                f"Reference: linalg/tests/test_edge_cases.py, gemm/tests/test_edge_cases.py\n"
                f"When all suites exist and pass locally, advance job to testing_pass."
            )
            mem.update_job_state(jid, 'tests_writing', 'gate', f'missing test suites: {cats}')
            if project_dir: update_phase_context(project_dir, 'tests_writing')
            mem.ensure_open_message('gate', f'Test suites needed for {name}',
                body=body, job_id=jid, message_type='directive', priority='normal')
            print(f'  → tests_writing: missing {cats}')
        elif all_passed:
            mem.update_job_state(jid, 'edge_pass', 'gate', 'compliance + gate tests passed')
            print(f'  → ALL tests PASSED')
        else:
            mem.update_job_state(jid, 'edge_fail', 'gate', 'tests failed')
            mem.ensure_open_message('gate', f'Gate tests failed for {name}',
                body='\n'.join(gate_output)[-1200:],
                job_id=jid, message_type='info', priority='normal')
            print(f'  → tests FAILED')
    # Reload state
    job = mem.get_job(jid)
    state = job['state']

    # --- edge_pass → linter ---
    if state == 'edge_pass' and project_dir:
        print(f'  → running linter...')
        mem.update_job_state(jid, 'linting', 'gate', 'running linter')
        if project_dir: update_phase_context(project_dir, 'linting')

        lint_script = f'{SCRIPTS}/lint_cuda.py'
        cu_files = glob.glob(f'{project_dir}/csrc/{kernel}/*.cu') if project_dir else []
        if os.path.isfile(lint_script) and cu_files:
            all_clean = True
            lint_output = []
            for cu in cu_files:
                r = subprocess.run(['python3', lint_script, cu],
                    capture_output=True, text=True, timeout=60)
                if r.returncode != 0:
                    all_clean = False
                    lint_output.append(r.stdout + r.stderr)
            if all_clean:
                mem.update_job_state(jid, 'lint_pass', 'gate', 'lint clean')
                print(f'  → lint PASSED')
            else:
                mem.update_job_state(jid, 'lint_fail', 'gate', 'lint issues found')
                mem.ensure_open_message('gate', f'Lint failed for {name}',
                    body='\n'.join(lint_output)[-500:],
                    job_id=jid, message_type='info', priority='normal')
                print(f'  → lint FAILED')
        else:
            mem.update_job_state(jid, 'lint_pass', 'gate', 'no .cu files or no linter, skipped')
            print(f'  → no lint target, skipped to lint_pass')

    # Reload state
    job = mem.get_job(jid)
    state = job['state']


    # --- lint_pass / ready_to_ship → ship ---
    if state in ('lint_pass', 'ready_to_ship'):
        if state == 'lint_pass':
            mem.update_job_state(jid, 'ready_to_ship', 'gate', 'lint passed')
            if project_dir: update_phase_context(project_dir, 'ready_to_ship')
        if project_dir:
            clean, dirty = ensure_repo_clean(project_dir)
            if not clean:
                mem.ensure_open_message('gate', f'Shipping blocked for {name}: dirty repository',
                    body=f'git status --porcelain:\n{dirty}',
                    job_id=jid, message_type='info', priority='normal')
                print('  → shipping blocked: working tree dirty (commit before shipping)')
                mem.close()
                return
        mem.update_job_state(jid, 'shipping', 'gate', 'shipping primitives')
        results = mem.auto_ship_job(jid, shipped_by='gate')
        shipped = [r for r in results if r.get('action') not in ('error', None)]
        errors = [r for r in results if r.get('action') == 'error']
        for r in shipped:
            print(f"  SHIPPED: {r.get('shelf_path')} v{r.get('version')} hash={r.get('hash')}")
        for r in errors:
            print(f"  ERROR: {r.get('file')}: {r.get('error')}")
        if shipped or not errors:
            files = ', '.join(r.get('shelf_path','') for r in shipped) or '(infrastructure)'
            mem.update_job_state(jid, 'shipped', 'gate', f'shipped: {files}')
            print(f'  → shipped')
            if project_dir:
                try:
                    commit_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=project_dir, check=True).stdout.strip()
                except subprocess.CalledProcessError:
                    commit_hash = '<git_rev_parse_failed>'
                notes = (job.get('notes') or '').strip()
                note_lines = [notes] if notes else []
                shipment_note = f'Shipped from {commit_hash}: {files}'
                note_lines.append(shipment_note)
                mem.update_job(jid, notes='\n'.join(line for line in note_lines if line), updated_by='gate')

    # Reload state
    job = mem.get_job(jid)
    state = job['state']

    # --- Fail states → rework ---
    if state in ('testing_fail', 'edge_fail', 'lint_fail'):
        mem.update_job_state(jid, 'rework', 'gate', f'{state} → rework')
        if project_dir: update_phase_context(project_dir, 'rework')
        mem.ensure_open_message('gate', f'{name} sent back for rework ({state})',
            job_id=jid, message_type='directive', priority='normal')
        print(f'  → sent back to rework')

    # --- stuck_needs_research → kick researcher ---
    elif state == 'stuck_needs_research':
        msgs = mem.get_messages(job_id=jid, message_type='question')
        subject = msgs[0]['subject'] if msgs else f'Research needed for {name}'
        body = msgs[0]['body'] if msgs else ''
        try:
            subprocess.run(['tmux', 'has-session', '-t', 'researcher'], check=True,
                capture_output=True, timeout=5)
            subprocess.run(['tmux', 'send-keys', '-t', 'researcher', '/clear', 'Enter'], timeout=5)
            import time; time.sleep(3)
            prompt = f'Research request for job #{jid} ({name}): {subject}. {body[:200]}'
            subprocess.run(['tmux', 'send-keys', '-t', 'researcher', prompt, 'Enter'], timeout=5)
            print(f'  → kicked researcher: {subject[:60]}')
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            mem.ensure_open_message('gate', f'No researcher session — {name} needs research',
                body=f'Job #{jid} stuck. Needs: {subject}',
                job_id=jid, message_type='blocker', priority='urgent')
            print(f'  → no researcher session, posted urgent message')

    # --- research_available → resume worker ---
    elif state == 'research_available' and kernel:
        try:
            subprocess.run(['tmux', 'has-session', '-t', kernel], check=True,
                capture_output=True, timeout=5)
            subprocess.run(['tmux', 'send-keys', '-t', kernel, '/clear', 'Enter'], timeout=5)
            import time; time.sleep(3)
            prompt = f'New research available. Search: msearch "your problem" --kernel {kernel} -k 5. Resume optimization.'
            subprocess.run(['tmux', 'send-keys', '-t', kernel, prompt, 'Enter'], timeout=5)
            mem.update_job_state(jid, 'hw_optimizing', 'gate', 'research delivered, worker resumed')
            print(f'  → research delivered, worker resumed')
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            mem.update_job_state(jid, 'hw_optimizing', 'gate', 'research available, no worker session')
            print(f'  → research available, no worker session, state advanced')

    mem.close()
