#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License.
# watchdog_tick.py -- Tick functions for the watchdog main loop.
# Replaces watchdog_ticks.sh + watchdog_workers.sh.
# Each worker/staff operation is wrapped in try/except so one failure
# never blocks the rest of the tick.

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))

from watchdog_tmux import (
    classify_pane_state,
    create_session,
    is_eval_running,
    is_session_alive,
    is_worker_idle,
    launch_cli_fresh,
    send_keys,
    set_window_label,
    submit_prompt_and_confirm,
    RESTART_PAUSE,
)
from watchdog_git import (
    prepare_worker_workspace,
    get_worker_context,
)

log = logging.getLogger("watchdog.tick")

# -- Configuration dataclasses --

@dataclass
class WorkerConfig:
    name: str
    tmux_session: str
    legacy_tsv: str
    launch_cmd: str
    cwd: str


@dataclass
class StaffConfig:
    name: str
    tmux_session: str
    resume_prompt: str
    launch_cmd: str
    cwd: str


@dataclass
class WatchdogConfig:
    workers: list[WorkerConfig] = field(default_factory=list)
    staff: list[StaffConfig] = field(default_factory=list)
    repo_root: str = ""
    common_dir: str = ""
    phases_dir: str = ""
    max_iters: int = 1
    stall_timeout: int = 9000
    idle_timeout: int = 900
    active_sit_timeout: int = 300


@dataclass
class WorkerState:
    """Per-worker state that persists across ticks (in-memory only)."""
    last_experiment_count: int = 0
    last_change_time: float = 0.0
    last_idle_time: float = 0.0


# -- DB helper functions (replace inline python3 heredocs) --

def _get_experiment_count(kernel: str) -> int:
    """Get experiment count from DB."""
    try:
        from factory_brain import ResearchMemory
        mem = ResearchMemory()
        try:
            row = mem.conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE kernel_type = ?",
                (kernel,),
            ).fetchone()
            return int(row[0]) if row else 0
        finally:
            mem.close()
    except Exception as e:
        log.warning(f"get_experiment_count({kernel}) failed: {e}")
        return 0


def _get_worker_completion_context(worker: str) -> tuple[str, str, str]:
    """Get (process_state, reported_job_id, active_job_id) for a worker."""
    try:
        from factory_brain import ResearchMemory, get_active_worker_jobs
        mem = ResearchMemory()
        try:
            row = mem.conn.execute(
                "SELECT process_state, job_id FROM worker_state WHERE kernel_type = ?",
                (worker,),
            ).fetchone()
            rows = get_active_worker_jobs(mem, worker, exclude_done_handoffs=False)
            rows.sort(key=lambda j: (
                int(j["priority"]) if str(j["priority"]).isdigit() else 99,
                j["updated_at"], j["id"],
            ))
            active_job = rows[0] if rows else None

            process_state = (row["process_state"] if row else "").strip()
            reported_job_id = str(row["job_id"] if row and row["job_id"] is not None else "")
            active_job_id = str(active_job["id"]) if active_job else ""
            return process_state, reported_job_id, active_job_id
        finally:
            mem.close()
    except Exception as e:
        log.warning(f"get_worker_completion_context({worker}) failed: {e}")
        return "", "", ""


def _staff_has_open_work(agent: str) -> bool:
    """Check if a staff agent has open research_request messages."""
    try:
        from factory_brain import ResearchMemory
        mem = ResearchMemory()
        try:
            rows = mem.get_messages(status="open", to_agent=agent, message_type="research_request")
            return bool(rows)
        finally:
            mem.close()
    except Exception as e:
        log.warning(f"staff_has_open_work({agent}) failed: {e}")
        return False


def _consume_worker_handoff(worker: str, worktree_path: str) -> tuple[str, str]:
    """Consume a worker handoff signal. Returns (action, detail)."""
    try:
        from watchdog_scheduler import consume_worker_handoff
        from factory_brain import ResearchMemory
        mem = ResearchMemory()
        try:
            result = consume_worker_handoff(mem, worker, worktree_path)
            action = result.get("action", "none")
            detail = result.get("detail", "")
            return action, detail
        finally:
            mem.close()
    except Exception as e:
        log.warning(f"consume_worker_handoff({worker}) failed: {e}")
        return "none", ""


def _get_active_job_label(worker: str) -> str:
    """Get a label like 'comfy_render #65' for tmux window title."""
    try:
        from factory_brain import ResearchMemory, get_active_worker_jobs
        mem = ResearchMemory()
        try:
            rows = get_active_worker_jobs(mem, worker, exclude_done_handoffs=False)
            rows = [j for j in rows if j["state"] not in
                    ("shipped", "converged", "parked", "abandoned")]
            rows.sort(key=lambda j: (
                int(j["priority"]) if str(j["priority"]).isdigit() else 99,
                j["updated_at"], j["id"],
            ))
            if rows:
                return f"{worker} #{rows[0]['id']}"
            return worker
        finally:
            mem.close()
    except Exception:
        return worker


def _expand_resume_cmd(worker: str, repo_root: str, common_dir: str) -> str:
    """Generate the resume prompt for a worker based on its active job."""
    try:
        from factory_brain import ResearchMemory, get_active_worker_jobs
        mem = ResearchMemory()
        try:
            rows = get_active_worker_jobs(mem, worker, exclude_done_handoffs=False)
            rows.sort(key=lambda j: (
                int(j["priority"]) if str(j["priority"]).isdigit() else 99,
                j["updated_at"], j["id"],
            ))
            job = rows[0] if rows else None
            if not job:
                return (
                    f"No active DB job is assigned to worker family {worker}. "
                    "Stay in your dedicated worktree, do not touch shared repo roots, "
                    "and wait for the next pull from hopper."
                )
            wt_root = Path(repo_root, "data", "watchdog-worktrees")
            packet_path = wt_root / worker / "job_packet.json"
            return (
                f"Resume in this dedicated watchdog worktree on active job #{job['id']}: {job['title']}. "
                f"Assume zero prior model context. "
                f"This project is one subdirectory inside the shared {repo_root} factory, "
                f"but you are expected to stay inside this isolated git worktree rather than editing the shared checkout. "
                f"The factory database is external to the project tree at {common_dir}/memory/research.db; "
                f"do not search for a local project database. "
                f"Before substantial exploration, run python3 {common_dir}/memory/factory_brain.py heartbeat "
                f"{worker} --job {job['id']} --state working --task 'resuming active job and reading spec'. "
                f"Then read the generated worker packet at {packet_path} first. "
                f"It is the authoritative structured packet generated from the DB job, open messages, "
                f"experiment summary, local file hints, and any repo-local structured spec. "
                f"If that packet reports repo_local_spec.present=true with validation_status='valid', "
                f"treat the validated repo-local spec as the bounded contract before editing. "
                f"Use only the packet's refresh commands plus the local files it names; "
                f"do not begin with generic codebase exploration. "
                f"Do not rely on prior chat context. Make the smallest change needed for this job, "
                f"refresh heartbeat during long work, and report status back to the DB. "
                f"Git discipline: commit only on this worker branch before handoff, "
                f"never edit the shared repo root, and keep ownership to the files implied by the active job. "
                f"When you hand off, follow the packet protocol exactly and signal one of "
                f"'done', 'check my work', or 'problem'. "
                f"Use {common_dir}/csrc and {common_dir}/docs for the shared primitives library "
                f"and reference material. Ping the researcher (tmux session 'researcher') or "
                f"open a research request whenever you need additional context or hit 'stuck_needs_research'."
            )
        finally:
            mem.close()
    except Exception as e:
        log.error(f"expand_resume_cmd({worker}) failed: {e}")
        return f"Resume work on your active job. Worker: {worker}."


# -- Phase context --

_STATE_TO_PHASE = {
    "not_started": "development", "algo_building": "development",
    "algo_optimizing": "development", "hw_optimizing": "development",
    "stuck_needs_research": "development", "research_available": "development",
    "compiles_ok": "validation", "tests_writing": "validation",
    "testing": "validation", "testing_pass": "validation",
    "testing_fail": "validation", "edge_testing": "validation",
    "edge_pass": "validation", "edge_fail": "validation",
    "rework": "rework", "rework_complete": "rework",
    "retesting": "rework", "retest_pass": "rework", "retest_fail": "rework",
    "linting": "quality", "lint_pass": "quality", "lint_fail": "quality",
    "ready_to_ship": "shipping", "shipping": "shipping", "shipped": "shipping",
}


def _update_phase_context(worker: str, state: str, project_dir: str, phases_dir: str) -> None:
    """Copy the appropriate phase context file into the worker's .claude/ dir."""
    phase = _STATE_TO_PHASE.get(state, "development")
    phase_file = Path(phases_dir) / f"{phase}.md"
    target = Path(project_dir) / ".claude" / "phase_context.md"

    if phase_file.is_file() and target.parent.is_dir():
        try:
            shutil.copy2(str(phase_file), str(target))
            log.info(f"[{worker}] phase context -> {phase}.md (state={state})")
        except Exception as e:
            log.warning(f"[{worker}] phase context copy failed: {e}")


# -- Worker lifecycle --

def _is_cli_launch_cmd(launch_cmd: str) -> bool:
    return launch_cmd.startswith("claude") or launch_cmd.startswith("codex")


def _ensure_loop_session(
    worker: WorkerConfig, cwd: str, config: WatchdogConfig,
) -> None:
    """Ensure the worker's tmux session exists. Create + launch if needed."""
    if is_session_alive(worker.tmux_session):
        return

    log.info(f"[{worker.name}] session dead, recreating")
    create_session(worker.tmux_session, cwd)
    time.sleep(2)
    label = _get_active_job_label(worker.name)
    set_window_label(worker.tmux_session, label)

    resume = _expand_resume_cmd(worker.name, config.repo_root, config.common_dir)

    if worker.launch_cmd:
        if _is_cli_launch_cmd(worker.launch_cmd):
            launch_cli_fresh(worker.tmux_session, worker.launch_cmd, resume)
            time.sleep(RESTART_PAUSE)
            if not submit_prompt_and_confirm(worker.tmux_session, resume):
                log.warning(f"[{worker.name}] launch not confirmed; retrying")
                submit_prompt_and_confirm(worker.tmux_session, resume)
                time.sleep(RESTART_PAUSE)
        else:
            send_keys(worker.tmux_session, worker.launch_cmd, "C-m")
            time.sleep(RESTART_PAUSE)
            if resume:
                submit_prompt_and_confirm(worker.tmux_session, resume)
                time.sleep(RESTART_PAUSE)
    elif resume:
        submit_prompt_and_confirm(worker.tmux_session, resume)
        time.sleep(RESTART_PAUSE)


def _restart_loop(
    worker: WorkerConfig, cwd: str, config: WatchdogConfig,
) -> None:
    """Restart a worker's optimization loop (clear + re-prompt, or kill + relaunch)."""
    log.info(f"[{worker.name}] restarting loop")

    ctx = get_worker_context(worker.name, cwd)
    state = ctx.get("state", "")
    if state:
        _update_phase_context(worker.name, state, cwd, config.phases_dir)

    resume = _expand_resume_cmd(worker.name, config.repo_root, config.common_dir)

    if _is_cli_launch_cmd(worker.launch_cmd):
        # Kill and recreate for CLI-based workers
        try:
            subprocess.run(["tmux", "kill-session", "-t", worker.tmux_session],
                           capture_output=True, timeout=5)
        except Exception:
            pass
        _ensure_loop_session(worker, cwd, config)
        label = _get_active_job_label(worker.name)
        set_window_label(worker.tmux_session, label)
        log.info(f"[{worker.name}] fresh CLI loop restarted")
        return

    _ensure_loop_session(worker, cwd, config)

    pane_state = classify_pane_state(worker.tmux_session)
    if pane_state == "expired" and worker.launch_cmd:
        try:
            subprocess.run(["tmux", "kill-session", "-t", worker.tmux_session],
                           capture_output=True, timeout=5)
        except Exception:
            pass
        _ensure_loop_session(worker, cwd, config)

    send_keys(worker.tmux_session, "Escape")
    time.sleep(2)
    send_keys(worker.tmux_session, "/clear", "C-m")
    time.sleep(RESTART_PAUSE)

    submit_prompt_and_confirm(worker.tmux_session, resume)
    log.info(f"[{worker.name}] loop restarted")


# -- Tick functions --

def tick_short(config: WatchdogConfig, worker_states: dict[str, WorkerState]) -> None:
    """Short-interval tick: worker lifecycle, gate processing, service health."""
    now = time.time()

    # -- Process each worker --
    for worker in config.workers:
        try:
            _process_worker(worker, config, worker_states, now)
        except Exception as e:
            log.error(f"[{worker.name}] tick_short failed: {e}", exc_info=True)

    # -- Gate processing (scheduler) --
    try:
        from watchdog_scheduler import run_scheduler_tick
        run_scheduler_tick()
        log.info("job gate processing complete")
    except Exception as e:
        log.error(f"gate processing failed: {e}")

    # -- Memory server health check --
    try:
        _check_memory_server(config)
    except Exception as e:
        log.warning(f"memory server check failed: {e}")

    # -- Staff loops --
    for staff in config.staff:
        try:
            _process_staff(staff, config, worker_states, now)
        except Exception as e:
            log.error(f"[{staff.name}] staff tick failed: {e}")


def _process_worker(
    worker: WorkerConfig, config: WatchdogConfig,
    worker_states: dict[str, WorkerState], now: float,
) -> None:
    """Process a single worker: handoff, completion, idle/stall detection, restart."""
    ws = worker_states.setdefault(worker.name, WorkerState(last_change_time=now))
    cwd = worker.cwd or os.path.join(config.repo_root, worker.name)
    cwd = prepare_worker_workspace(worker.name, cwd, config.repo_root, config.common_dir)

    # Check for handoff signals
    action, detail = _consume_worker_handoff(worker.name, cwd)
    if action == "reset":
        log.info(f"[{worker.name}] handoff: {detail}")
        _restart_loop(worker, cwd, config)
        ws.last_idle_time = 0.0
        ws.last_change_time = now
        return
    if action == "hold":
        log.info(f"[{worker.name}] hold: {detail}")
        ws.last_idle_time = 0.0
        ws.last_change_time = now
        return

    # Check completion state
    process_state, reported_job_id, active_job_id = _get_worker_completion_context(worker.name)
    if process_state == "complete":
        if active_job_id and reported_job_id == active_job_id:
            log.info(f"[{worker.name}] worker complete for active job #{active_job_id}; waiting for handoff")
            return
        _restart_loop(worker, cwd, config)
        ws.last_idle_time = 0.0
        ws.last_change_time = now
        return

    # Ensure session exists
    _ensure_loop_session(worker, cwd, config)

    # Check experiment progress
    current_count = _get_experiment_count(worker.name)
    new_iters = current_count - ws.last_experiment_count
    reason = ""

    if new_iters > 0:
        ws.last_change_time = now
        ws.last_idle_time = 0.0

    # Idle detection
    if is_worker_idle(worker.tmux_session):
        if ws.last_idle_time == 0.0:
            ws.last_idle_time = now
        idle_secs = now - ws.last_idle_time
        if idle_secs >= config.idle_timeout:
            reason = f"idle at prompt ({idle_secs:.0f}s)"
    else:
        ws.last_idle_time = 0.0

    # Iteration limit
    if not reason and new_iters >= config.max_iters:
        reason = f"iteration limit ({new_iters} >= {config.max_iters})"

    # Stall timeout
    if not reason:
        stall_secs = now - ws.last_change_time
        if stall_secs >= config.stall_timeout:
            reason = f"stall timeout ({stall_secs:.0f}s with no new iteration)"

    if reason:
        log.info(f"[{worker.name}] restart: {reason}")
        # Wait for eval to finish
        wait_count = 0
        while is_eval_running(worker.tmux_session) and wait_count < 30:
            log.info(f"[{worker.name}] eval running, waiting")
            time.sleep(10)
            wait_count += 1

        _restart_loop(worker, cwd, config)
        ws.last_experiment_count = current_count
        ws.last_change_time = time.time()
        ws.last_idle_time = 0.0


def _process_staff(
    staff: StaffConfig, config: WatchdogConfig,
    worker_states: dict[str, WorkerState], now: float,
) -> None:
    """Process a staff loop (e.g., researcher): nudge if idle with open work."""
    if not _staff_has_open_work(staff.name):
        return

    cwd = staff.cwd or os.path.join(config.repo_root, staff.name)
    _ensure_loop_session_staff(staff, cwd, config)

    ws = worker_states.setdefault(staff.name, WorkerState(last_change_time=now))

    if is_worker_idle(staff.tmux_session):
        if ws.last_idle_time == 0.0:
            ws.last_idle_time = now
        idle_secs = now - ws.last_idle_time
        if idle_secs >= config.idle_timeout:
            log.info(f"[{staff.name}] idle ({idle_secs:.0f}s) with open research work, nudging")
            send_keys(staff.tmux_session, "/clear", "C-m")
            time.sleep(RESTART_PAUSE)
            resume = staff.resume_prompt
            if submit_prompt_and_confirm(staff.tmux_session, resume):
                log.info(f"[{staff.name}] resume prompt confirmed")
            else:
                log.warning(f"[{staff.name}] research prompt submission unconfirmed")
            ws.last_idle_time = 0.0
    else:
        ws.last_idle_time = 0.0


def _ensure_loop_session_staff(
    staff: StaffConfig, cwd: str, config: WatchdogConfig,
) -> None:
    """Ensure staff tmux session exists."""
    if is_session_alive(staff.tmux_session):
        return

    log.info(f"[{staff.name}] session dead, recreating")
    create_session(staff.tmux_session, cwd)
    time.sleep(2)

    if staff.launch_cmd:
        if _is_cli_launch_cmd(staff.launch_cmd):
            launch_cli_fresh(staff.tmux_session, staff.launch_cmd, staff.resume_prompt)
            time.sleep(RESTART_PAUSE)
        else:
            send_keys(staff.tmux_session, staff.launch_cmd, "C-m")
            time.sleep(RESTART_PAUSE)
            if staff.resume_prompt:
                submit_prompt_and_confirm(staff.tmux_session, staff.resume_prompt)
                time.sleep(RESTART_PAUSE)


def _check_memory_server(config: WatchdogConfig) -> None:
    """Check if the memory server is running, restart if down."""
    try:
        result = subprocess.run(
            ["curl", "-sf", "http://localhost:8421/api/stats"],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return
    except subprocess.TimeoutExpired:
        log.warning("memory server health check timed out")
    except Exception:
        pass

    log.info("memory server down; restarting")
    try:
        start_script = os.path.join(config.common_dir, "memory", "start-server.sh")
        subprocess.run([start_script, "--daemon"], capture_output=True, timeout=15)
    except Exception as e:
        log.warning(f"memory server restart failed: {e}")


# -- Medium tick --

def tick_medium(config: WatchdogConfig) -> None:
    """Medium-interval tick: ingest, worker state refresh, metadata hygiene, message dedup."""

    # Ingest + refresh
    try:
        from factory_brain import ResearchMemory
        mem = ResearchMemory()
        try:
            mem.ingest_all()
            mem.ingest_all_tsv()
            mem.refresh_worker_state()
        finally:
            mem.close()
        log.info("ingest + worker refresh complete")
    except Exception as e:
        log.error(f"ingest/refresh failed: {e}")

    # Metadata hygiene (log-only)
    try:
        from factory_brain import ResearchMemory
        mem = ResearchMemory()
        try:
            for job in mem.get_jobs():
                mode = job.get("factory_mode", "")
                if not mode:
                    log.debug(f"Job #{job['id']} missing factory mode")
                    continue
                scope = (job.get("optimization_scope") or "").strip()
                if not scope:
                    log.debug(f"Job #{job['id']} missing optimization scope")
                    continue
                required = ["objective_vector", "acceptance_gates", "keep_rule", "benchmark_set"]
                missing = [f for f in required if not (job.get(f) or "").strip()]
                if scope in ("hardware_tuned", "hybrid"):
                    for fld in ("hardware_target", "retarget_policy"):
                        if not (job.get(fld) or "").strip():
                            missing.append(fld)
                if missing:
                    log.debug(f"Job #{job['id']} missing objective metadata: {', '.join(missing)}")
        finally:
            mem.close()
        log.info("metadata hygiene complete")
    except Exception as e:
        log.error(f"metadata hygiene failed: {e}")

    # Message dedup
    try:
        import sqlite3
        db_path = os.path.join(config.common_dir, "memory", "research.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM messages
            WHERE status = 'open'
            AND id NOT IN (
                SELECT MAX(id)
                FROM messages
                WHERE status = 'open'
                GROUP BY from_agent, subject, COALESCE(job_id, -1)
            )
        """)
        n = cur.rowcount
        conn.commit()
        conn.close()
        if n:
            log.info(f"deduped {n} open messages")
        log.info("message dedup complete")
    except Exception as e:
        log.error(f"message dedup failed: {e}")


# -- Long tick --

def tick_long(config: WatchdogConfig) -> None:
    """Long-interval tick: summary generation, promotion candidates."""

    # Summary generation
    try:
        gen_script = os.path.join(config.common_dir, "memory", "generate_summaries.py")
        # Check if already running
        check = subprocess.run(
            ["pgrep", "-af", "generate_summaries.py --limit 50"],
            capture_output=True, timeout=5,
        )
        if check.returncode == 0:
            log.info("summary generation already running")
        else:
            log.info("starting summary generation batch")
            subprocess.run(
                [sys.executable, gen_script, "--limit", "50"],
                capture_output=True, timeout=1800,  # 30 min timeout
            )
            log.info("summary generation batch complete")
    except subprocess.TimeoutExpired:
        log.warning("summary generation timed out after 30 min")
    except Exception as e:
        log.error(f"summary generation failed: {e}")

    # Promotion candidates
    try:
        from factory_brain import ResearchMemory
        mem = ResearchMemory()
        try:
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
                mem.ensure_open_message(
                    "watchdog", "Promotion candidates available",
                    body=body, message_type="info", priority="normal",
                )
        finally:
            mem.close()
        log.info("research audit complete")
    except Exception as e:
        log.error(f"research audit failed: {e}")
