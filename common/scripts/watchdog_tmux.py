#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License.
# watchdog_tmux.py -- tmux interaction with timeouts and error handling.
# Replaces watchdog_tmux.sh. Every subprocess call gets a timeout.

import logging
import re
import subprocess
import time

log = logging.getLogger("watchdog.tmux")

SUBPROCESS_TIMEOUT = 10  # seconds for any single tmux call
PROMPT_CONFIRM_RETRIES = 3
PROMPT_CONFIRM_WAIT = 2  # seconds between retries
RESTART_PAUSE = 5  # seconds between /clear and resume prompt


def _run_tmux(*args: str, timeout: int = SUBPROCESS_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a tmux command with timeout. Returns CompletedProcess or raises."""
    return subprocess.run(
        ["tmux", *args],
        capture_output=True, text=True, timeout=timeout,
    )


def capture_pane(session: str, lines: int = 40) -> list[str]:
    """Capture visible pane text. Returns non-empty lines, last N."""
    try:
        result = _run_tmux("capture-pane", "-pt", session)
        all_lines = result.stdout.splitlines()
        nonempty = [ln for ln in all_lines if ln.strip()]
        return nonempty[-lines:]
    except subprocess.TimeoutExpired:
        log.warning(f"capture_pane({session}) timed out")
        return []
    except Exception as e:
        log.warning(f"capture_pane({session}) failed: {e}")
        return []


def is_session_alive(session: str) -> bool:
    """Check if a tmux session exists."""
    try:
        result = _run_tmux("has-session", "-t", session)
        return result.returncode == 0
    except Exception:
        return False


# -- Pane state classification --
# Patterns that indicate Claude/Codex is actively working
_ACTIVE_PATTERNS = re.compile(
    r"[\u280B\u2819\u2839\u2838\u283C\u2834\u2826\u2827\u2807\u280F]"  # spinners
    r"|Tokens?"
    r"|esc to interrupt"
    r"|Running "
    r"|Reading "
    r"|Writing "
    r"|Executing "
    r"|Working \("
    r"|Ran "
    r"|Updated "
    r"|Smooshing"
)

_CLAUDE_IDLE = re.compile(r"\u276F ")  # ❯  (Claude prompt)
_SHELL_IDLE = re.compile(r"^[^\s@]+@[^\s]+:.+[#$] ?$", re.MULTILINE)
_COMPOSER_IDLE = re.compile(r"\u203A ")  # ›  (Codex composer)


def classify_pane_state(session: str) -> str:
    """Classify tmux pane as: active, claude_idle, shell, composer, unknown."""
    text_lines = capture_pane(session, 50)
    if not text_lines:
        return "unknown"

    text = "\n".join(text_lines)

    if _ACTIVE_PATTERNS.search(text):
        return "active"
    if _CLAUDE_IDLE.search(text):
        return "claude_idle"
    if _SHELL_IDLE.search(text):
        return "shell"
    if _COMPOSER_IDLE.search(text):
        return "composer"
    return "unknown"


def is_worker_idle(session: str) -> bool:
    """Check if the worker is idle at a prompt (Claude, Codex, or shell)."""
    text_lines = capture_pane(session, 20)
    if not text_lines:
        return False

    text = "\n".join(text_lines)

    # Active markers take priority
    if re.search(r"Working \(|esc to interrupt|Ran |Updated |Reading |Smooshing", text):
        return False
    # Idle prompt characters
    if re.search(r"[\u276F\u203A]", text):
        return True
    # Shell prompt
    if _SHELL_IDLE.search(text):
        return True
    return False


def is_eval_running(session: str) -> bool:
    """Check if eval.sh is visible in the last 3 lines."""
    text_lines = capture_pane(session, 3)
    return any("eval.sh" in line for line in text_lines)


def send_keys(session: str, *keys: str) -> bool:
    """Send keys to a tmux session. Returns True on success."""
    try:
        _run_tmux("send-keys", "-t", session, *keys)
        return True
    except Exception as e:
        log.warning(f"send_keys({session}) failed: {e}")
        return False


def create_session(session: str, cwd: str) -> bool:
    """Create a new tmux session with bash."""
    try:
        _run_tmux("new-session", "-d", "-s", session, "-c", cwd, "bash")
        return True
    except Exception as e:
        log.error(f"create_session({session}) failed: {e}")
        return False


def set_window_label(session: str, label: str) -> None:
    """Rename the first window in a session."""
    try:
        _run_tmux("rename-window", "-t", f"{session}:0", label)
    except Exception:
        pass


def paste_prompt(session: str, prompt: str) -> bool:
    """Set tmux buffer and paste into session, then press Enter."""
    try:
        _run_tmux("set-buffer", "--", prompt)
        _run_tmux("paste-buffer", "-t", session)
        _run_tmux("send-keys", "-t", session, "C-m")
        return True
    except Exception as e:
        log.warning(f"paste_prompt({session}) failed: {e}")
        return False


def _confirm_execution_started(session: str) -> bool:
    """Check if Claude/Codex has started processing (not still at idle prompt)."""
    text_lines = capture_pane(session, 50)
    if not text_lines:
        return False
    text = "\n".join(text_lines)

    # Still at idle prompt -- not started
    if _CLAUDE_IDLE.search(text):
        return False
    # Active indicators
    if _ACTIVE_PATTERNS.search(text):
        return True
    # Default: assume started (same as original bash logic)
    return True


def submit_prompt_and_confirm(session: str, prompt: str) -> bool:
    """Submit a prompt to a session and confirm execution started.

    Handles claude_idle (sends /clear first), shell, and unknown states.
    Retries up to PROMPT_CONFIRM_RETRIES times.
    """
    for attempt in range(PROMPT_CONFIRM_RETRIES):
        state = classify_pane_state(session)
        log.debug(f"submit_prompt({session}) state={state} try={attempt + 1}")

        if state == "active":
            return True

        if state == "claude_idle":
            send_keys(session, "/clear", "C-m")
            time.sleep(1)
            paste_prompt(session, prompt)
        elif state in ("shell", "unknown"):
            paste_prompt(session, prompt)
        elif state == "composer":
            # Check if prompt is already pasted
            text_lines = capture_pane(session, 20)
            text = "\n".join(text_lines)
            if prompt[:50] not in text:
                paste_prompt(session, prompt)
            else:
                send_keys(session, "C-m")

        time.sleep(PROMPT_CONFIRM_WAIT)
        if _confirm_execution_started(session):
            log.debug(f"submit_prompt({session}) execution confirmed")
            return True

    log.warning(f"submit_prompt({session}) not confirmed after {PROMPT_CONFIRM_RETRIES} tries")
    return False


def launch_cli_fresh(session: str, launch_cmd: str, prompt: str) -> None:
    """Launch Claude/Codex fresh in a session with an initial prompt."""
    import shlex
    cmd_line = f"{launch_cmd} {shlex.quote(prompt)}"
    send_keys(session, cmd_line, "C-m")
