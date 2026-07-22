#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License.
# watchdog_git.py -- Dedicated git worktree management for watchdog workers.
# Replaces watchdog_git.sh. All git calls get timeouts and lock awareness.

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))

log = logging.getLogger("watchdog.git")

GIT_TIMEOUT = 30  # seconds for any single git call
STALE_LOCK_AGE = 300  # remove index.lock older than 5 min


def _git(repo: str, *args: str, timeout: int = GIT_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a git command with timeout. Returns CompletedProcess."""
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _git_output(repo: str, *args: str, timeout: int = GIT_TIMEOUT) -> str:
    """Run a git command and return stripped stdout. Raises on failure."""
    result = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    result.check_returncode()
    return result.stdout.strip()


def _sanitize_ref_component(raw: str) -> str:
    """Sanitize a string for use as a git ref component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", raw)
    cleaned = cleaned.strip("-")
    return cleaned or "repo"


def _check_git_lock(repo_top: str) -> bool:
    """Check for stale git lock. Removes if stale. Returns True if lock is held (skip ops)."""
    lock = Path(repo_top) / ".git" / "index.lock"
    if not lock.exists():
        return False
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return False
    if age > STALE_LOCK_AGE:
        try:
            lock.unlink()
            log.warning(f"removed stale git lock ({age:.0f}s old) at {lock}")
        except OSError as e:
            log.warning(f"could not remove stale git lock: {e}")
        return False
    log.info(f"git lock held ({age:.0f}s), skipping worktree ops")
    return True


def _is_git_repo(path: str) -> bool:
    """Check if path is inside a git work tree."""
    try:
        result = _git(path, "rev-parse", "--is-inside-work-tree", timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def _get_git_common_dir(path: str) -> str | None:
    """Get the resolved git common dir for a repo/worktree."""
    try:
        common = _git_output(path, "rev-parse", "--git-common-dir", timeout=10)
        return str(Path(path, common).resolve())
    except Exception:
        return None


def _get_current_branch(path: str) -> str | None:
    """Get the current branch name, or None if detached/missing."""
    try:
        return _git_output(path, "symbolic-ref", "--quiet", "--short", "HEAD", timeout=10)
    except Exception:
        return None


def _has_dirty_tracked_files(path: str) -> bool:
    """Check if there are modified tracked files (ignores untracked)."""
    try:
        result = _git(path, "status", "--porcelain", timeout=10)
        for line in result.stdout.splitlines():
            if not line.startswith("?? "):
                return True
        return False
    except Exception:
        return False


def _branch_exists(repo: str, branch: str) -> bool:
    """Check if a branch exists in the repo."""
    try:
        result = _git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", timeout=10)
        return result.returncode == 0
    except Exception:
        return False


# -- Worker context from DB --

def get_worker_context(worker: str, fallback_cwd: str) -> dict:
    """Get the active job context for a worker from the factory DB.

    Returns dict with keys: job_id, state, title, repo_top, project_dir, repo_name.
    """
    try:
        from factory_brain import ResearchMemory, get_active_worker_jobs, resolve_job_project_dir

        mem = ResearchMemory()
        try:
            rows = get_active_worker_jobs(mem, worker, exclude_done_handoffs=False)
            rows.sort(key=lambda j: (
                int(j["priority"]) if str(j["priority"]).isdigit() else 99,
                j["updated_at"], j["id"],
            ))
            job = rows[0] if rows else None

            project_dir = None
            if job is not None:
                resolved = resolve_job_project_dir(job)
                if resolved is not None:
                    project_dir = str(resolved)
            if not project_dir:
                project_dir = fallback_cwd

            repo_top = project_dir
            try:
                repo_top = _git_output(project_dir, "rev-parse", "--show-toplevel", timeout=10)
            except Exception:
                pass

            repo_name = Path(repo_top).name if repo_top else Path(project_dir).name

            return {
                "job_id": str(job["id"]) if job else "",
                "state": (job or {}).get("state", ""),
                "title": (job or {}).get("title", ""),
                "repo_top": repo_top,
                "project_dir": project_dir,
                "repo_name": repo_name,
            }
        finally:
            mem.close()
    except Exception as e:
        log.warning(f"get_worker_context({worker}) failed: {e}")
        return {
            "job_id": "", "state": "", "title": "",
            "repo_top": fallback_cwd, "project_dir": fallback_cwd,
            "repo_name": Path(fallback_cwd).name,
        }


# -- Packet management --

def worker_packet_path(worker: str, worktree_root: str) -> str:
    """Return the path to a worker's job packet."""
    return os.path.join(worktree_root, worker, "job_packet.json")


def refresh_worker_packet(
    worker: str, worktree_path: str, shared_repo_root: str = "",
    branch: str = "", common_dir: str = "",
) -> str | None:
    """Regenerate the job packet for a worker. Returns packet path or None on failure."""
    if not common_dir:
        common_dir = os.environ.get("COMMON_DIR", "")
    worktree_root = os.environ.get(
        "WATCHDOG_WORKTREE_ROOT",
        os.path.join(os.environ.get("REPO_ROOT", ""), "data", "watchdog-worktrees"),
    )
    packet_path = worker_packet_path(worker, worktree_root)
    os.makedirs(os.path.dirname(packet_path), exist_ok=True)

    cmd = [
        sys.executable, os.path.join(common_dir, "scripts", "build_job_packet.py"),
        "--worker", worker,
        "--worktree", worktree_path,
        "--output", packet_path,
    ]
    if shared_repo_root:
        cmd.extend(["--shared-repo-root", shared_repo_root])
    if branch:
        cmd.extend(["--branch", branch])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log.warning(f"refresh_worker_packet({worker}) failed: {result.stderr.strip()}")
            return None
        return packet_path
    except subprocess.TimeoutExpired:
        log.warning(f"refresh_worker_packet({worker}) timed out")
        return None
    except Exception as e:
        log.warning(f"refresh_worker_packet({worker}) error: {e}")
        return None


# -- Worktree lifecycle --

def prepare_worker_workspace(
    worker: str, fallback_cwd: str, repo_root: str, common_dir: str,
) -> str:
    """Ensure a dedicated worktree exists for the worker. Returns worktree path.

    Creates or reuses a worktree at <worktree_root>/<worker>/current.
    If the worktree points to the wrong branch/repo, it is recreated.
    If the worktree has dirty tracked files, it is left alone.
    """
    worktree_root = os.path.join(repo_root, "data", "watchdog-worktrees")
    ctx = get_worker_context(worker, fallback_cwd)
    job_id = ctx["job_id"]
    repo_top = ctx["repo_top"]
    project_dir = ctx["project_dir"]
    repo_name = ctx["repo_name"]

    if not _is_git_repo(repo_top):
        refresh_worker_packet(worker, project_dir, repo_top, common_dir=common_dir)
        return project_dir

    if _check_git_lock(repo_top):
        # Git lock held, skip worktree ops but return existing if available
        existing = os.path.join(worktree_root, worker, "current")
        if os.path.isdir(existing) and _is_git_repo(existing):
            return existing
        return project_dir

    repo_key = _sanitize_ref_component(repo_name)
    if job_id:
        branch = f"watchdog/{worker}/job-{job_id}-{repo_key}"
    else:
        branch = f"watchdog/{worker}/idle-{repo_key}"

    worktree_path = os.path.join(worktree_root, worker, "current")
    os.makedirs(os.path.dirname(worktree_path), exist_ok=True)

    desired_common = _get_git_common_dir(repo_top)

    # Check existing worktree
    if os.path.exists(worktree_path):
        if _is_git_repo(worktree_path):
            current_common = _get_git_common_dir(worktree_path)
            current_branch = _get_current_branch(worktree_path)

            # Already correct -- just refresh packet
            if current_common == desired_common and current_branch == branch:
                refresh_worker_packet(worker, worktree_path, repo_top, branch, common_dir)
                return worktree_path

            # Wrong branch/repo but has dirty files -- leave it alone
            if _has_dirty_tracked_files(worktree_path):
                log.warning(f"[{worker}] worktree {worktree_path} has dirty tracked files; refusing reset")
                refresh_worker_packet(worker, worktree_path, repo_top, current_branch or "", common_dir)
                return worktree_path

            # Wrong branch/repo, clean -- remove it
            try:
                _git(repo_top, "worktree", "remove", "--force", worktree_path, timeout=15)
            except Exception:
                pass

        # Clean up the directory
        try:
            shutil.rmtree(worktree_path, ignore_errors=True)
        except Exception:
            pass

    # Prune stale worktree entries
    try:
        _git(repo_top, "worktree", "prune", timeout=15)
    except Exception:
        pass

    # Create the worktree
    try:
        if _branch_exists(repo_top, branch):
            _git(repo_top, "worktree", "add", worktree_path, branch, timeout=30)
        else:
            _git(repo_top, "worktree", "add", "-b", branch, worktree_path, "HEAD", timeout=30)
    except subprocess.TimeoutExpired:
        log.error(f"[{worker}] git worktree add timed out")
        return project_dir
    except Exception as e:
        log.error(f"[{worker}] git worktree add failed: {e}")
        return project_dir

    log.info(f"[{worker}] workspace={worktree_path} repo={repo_top} branch={branch} job={job_id or 'idle'}")
    refresh_worker_packet(worker, worktree_path, repo_top, branch, common_dir)
    return worktree_path


def reset_worker_workspace(worker: str, fallback_cwd: str, repo_root: str, common_dir: str) -> str:
    """Force-reset a worker's worktree. Returns new worktree path."""
    worktree_root = os.path.join(repo_root, "data", "watchdog-worktrees")
    worktree_path = os.path.join(worktree_root, worker, "current")

    if os.path.exists(worktree_path):
        if _is_git_repo(worktree_path):
            if _has_dirty_tracked_files(worktree_path):
                log.warning(f"[{worker}] cannot reset worktree with dirty tracked files")
                return worktree_path
            try:
                repo_top = _git_output(worktree_path, "rev-parse", "--show-toplevel", timeout=10)
                _git(repo_top, "worktree", "remove", "--force", worktree_path, timeout=15)
            except Exception:
                pass
        shutil.rmtree(worktree_path, ignore_errors=True)

    return prepare_worker_workspace(worker, fallback_cwd, repo_root, common_dir)
