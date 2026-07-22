#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License.
# watchdog_main.py -- Main watchdog loop with error isolation and structured logging.
# Replaces watchdog.sh + watchdog_db.sh.
#
# Usage: python3 watchdog_main.py
#        (or via watchdog_wrapper.sh for auto-restart)

import calendar
import logging
import os
import sys
import time
from pathlib import Path

# -- Path setup --
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = str(SCRIPT_DIR.parent.parent)
COMMON_DIR = str(SCRIPT_DIR.parent)
LOG_DIR = os.path.join(REPO_ROOT, "logs")

os.environ["REPO_ROOT"] = REPO_ROOT
os.environ["COMMON_DIR"] = COMMON_DIR
sys.path.insert(0, os.path.join(COMMON_DIR, "memory"))
sys.path.insert(0, str(SCRIPT_DIR))

from watchdog_tick import (
    WatchdogConfig,
    WorkerConfig,
    StaffConfig,
    WorkerState,
    tick_short,
    tick_medium,
    tick_long,
    _get_experiment_count,
)


# -- Logging setup --

class WatchdogFormatter(logging.Formatter):
    """Format logs as [watchdog][level][name] HH:MM:SS message"""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        # Extract scope from logger name: watchdog.tick -> tick, watchdog.tmux -> tmux
        parts = record.name.split(".")
        scope = parts[-1] if len(parts) > 1 else "main"
        level = record.levelname.lower()
        if level == "warning":
            level = "attention"
        elif level == "info":
            level = "tick"

        msg = record.getMessage()

        # Add traceback for errors
        result = f"[watchdog][{level}][{scope}] {ts} {msg}"
        if record.exc_info and record.exc_info[1]:
            import traceback
            tb = "".join(traceback.format_exception(*record.exc_info))
            result += "\n" + tb.rstrip()
        return result


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(WatchdogFormatter())
    root = logging.getLogger("watchdog")
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -- DB helpers --

def _db_path() -> str:
    return os.path.join(COMMON_DIR, "memory", "research.db")


def set_daemon_state(status: str, notes: str = "") -> None:
    """Update watchdog daemon state in the DB. Errors are logged, not fatal."""
    try:
        from factory_brain import ResearchMemory
        mem = ResearchMemory()
        try:
            import socket
            mem.set_watchdog_daemon_state(
                status, notes=notes, pid=os.getpid(), host=socket.gethostname(),
            )
        finally:
            mem.close()
    except Exception as e:
        log.warning(f"set_daemon_state({status}) failed: {e}")


def touch_tick(tick_name: str, status: str = "ok", notes: str = "") -> None:
    """Record a tick timestamp in the DB. Errors are logged, not fatal."""
    try:
        from factory_brain import ResearchMemory
        mem = ResearchMemory()
        try:
            mem.touch_watchdog_state(tick_name, status=status, notes=notes)
        finally:
            mem.close()
    except Exception as e:
        log.warning(f"touch_tick({tick_name}) failed: {e}")


def tick_due(tick_name: str, interval: int) -> bool:
    """Check if a tick is due. Returns False on any error (never fire-storms)."""
    try:
        from factory_brain import ResearchMemory
        mem = ResearchMemory()
        try:
            row = mem.get_watchdog_state(tick_name)
            ts = (row or {}).get("last_run_at", "")
            if not ts:
                return True
            last_epoch = calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
            return (time.time() - last_epoch) >= interval
        finally:
            mem.close()
    except Exception as e:
        log.warning(f"tick_due({tick_name}) failed: {e}, defaulting to not-due")
        return False


# -- Configuration loading --

def load_config() -> WatchdogConfig:
    """Load watchdog configuration from worker_lanes.conf and environment."""
    config = WatchdogConfig(
        repo_root=REPO_ROOT,
        common_dir=COMMON_DIR,
        phases_dir=os.path.join(COMMON_DIR, "claude", "phases"),
        max_iters=1,
        stall_timeout=9000,
        idle_timeout=900,
        active_sit_timeout=300,
    )

    lanes_file = os.path.join(COMMON_DIR, "scripts", "worker_lanes.conf")
    section = "workers"

    if os.path.isfile(lanes_file):
        with open(lanes_file) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if line == "[staff]":
                    section = "staff"
                    continue
                if line == "[workers]":
                    section = "workers"
                    continue

                parts = line.split(":")
                name = parts[0].strip()
                if not name:
                    continue

                if section == "workers":
                    # Format: name:slots[:launch_cmd[:cwd]]
                    launch_cmd = parts[2].strip() if len(parts) > 2 else "claude --dangerously-skip-permissions"
                    cwd = parts[3].strip() if len(parts) > 3 else os.path.join(REPO_ROOT, name)
                    config.workers.append(WorkerConfig(
                        name=name,
                        tmux_session=name,
                        legacy_tsv="",
                        launch_cmd=launch_cmd,
                        cwd=cwd,
                    ))
                elif section == "staff":
                    launch_cmd = parts[2].strip() if len(parts) > 2 else "claude --dangerously-skip-permissions"
                    cwd = parts[3].strip() if len(parts) > 3 else os.path.join(REPO_ROOT, name)
                    resume = (
                        "Read your open messages in the factory DB and handle only those "
                        "bounded research requests. Stay pull-based, keep outputs concise, "
                        "and do not proactively fan work out to workers."
                    )
                    config.staff.append(StaffConfig(
                        name=name,
                        tmux_session=name,
                        resume_prompt=resume,
                        launch_cmd=launch_cmd,
                        cwd=cwd,
                    ))

    # Fallback: if no workers loaded from config, use defaults
    if not config.workers:
        log.warning("No workers in config, using hardcoded defaults")
        for name in ("comfy_render", "cuquantum"):
            config.workers.append(WorkerConfig(
                name=name,
                tmux_session=name,
                legacy_tsv="",
                launch_cmd="claude --dangerously-skip-permissions",
                cwd=os.path.join(REPO_ROOT, name),
            ))

    if not config.staff:
        config.staff.append(StaffConfig(
            name="researcher",
            tmux_session="researcher",
            resume_prompt=(
                "Read your open messages in the factory DB and handle only those "
                "bounded research requests. Stay pull-based, keep outputs concise, "
                "and do not proactively fan work out to workers."
            ),
            launch_cmd="claude --dangerously-skip-permissions",
            cwd=os.path.join(REPO_ROOT, "foreman-staff", "researcher"),
        ))

    return config


# -- Main loop --

SHORT_INTERVAL = 60
MEDIUM_INTERVAL = 1800
LONG_INTERVAL = 21600
SLEEP_INTERVAL = 15

log = logging.getLogger("watchdog.main")


def main() -> None:
    setup_logging()
    log.info(f"starting watchdog (short={SHORT_INTERVAL}s, medium={MEDIUM_INTERVAL}s, long={LONG_INTERVAL}s)")

    config = load_config()
    log.info(f"loaded {len(config.workers)} workers, {len(config.staff)} staff")
    for w in config.workers:
        log.info(f"  worker: {w.name} cwd={w.cwd}")
    for s in config.staff:
        log.info(f"  staff: {s.name} cwd={s.cwd}")

    set_daemon_state("up", "watchdog starting")

    # Initialize per-worker state
    worker_states: dict[str, WorkerState] = {}
    now = time.time()
    for w in config.workers:
        count = _get_experiment_count(w.name)
        worker_states[w.name] = WorkerState(
            last_experiment_count=count,
            last_change_time=now,
        )
        log.info(f"  [{w.name}] starting experiment count = {count}")

    try:
        while True:
            try:
                set_daemon_state("up", "watchdog loop alive")
            except Exception:
                pass

            # Short tick
            if tick_due("watchdog_short", SHORT_INTERVAL):
                try:
                    tick_short(config, worker_states)
                    touch_tick("watchdog_short", "ok", "scheduler")
                except Exception as e:
                    log.error(f"tick_short failed: {e}", exc_info=True)
                    try:
                        touch_tick("watchdog_short", "error", str(e)[:200])
                    except Exception:
                        pass

            # Medium tick
            if tick_due("watchdog_medium", MEDIUM_INTERVAL):
                try:
                    tick_medium(config)
                    touch_tick("watchdog_medium", "ok", "ingest_hygiene")
                except Exception as e:
                    log.error(f"tick_medium failed: {e}", exc_info=True)
                    try:
                        touch_tick("watchdog_medium", "error", str(e)[:200])
                    except Exception:
                        pass

            # Long tick (in a thread with timeout)
            if tick_due("watchdog_long", LONG_INTERVAL):
                import threading
                t = threading.Thread(target=_run_tick_long, args=(config,), daemon=True)
                t.start()
                touch_tick("watchdog_long", "ok", "summaries_audit")

            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        log.info("watchdog interrupted by user")
    finally:
        set_daemon_state("down", "watchdog exiting")


def _run_tick_long(config: WatchdogConfig) -> None:
    """Run tick_long in a thread so it doesn't block the main loop."""
    try:
        tick_long(config)
    except Exception as e:
        log.error(f"tick_long failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
