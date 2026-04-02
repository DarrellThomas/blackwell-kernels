"""Worker state helpers extracted from factory_brain.
Expect `self` with .conn, get_job, ensure_open_message, BWK_ROOT.
"""
import time


def attach_worker_methods(cls):
    cls.worker_heartbeat = worker_heartbeat
    cls.refresh_worker_state = refresh_worker_state
    cls.get_worker_state = get_worker_state
    return cls

# -- Worker State --

def worker_heartbeat(self, kernel_type: str, current_task: str = "",
                     process_state: str = "working", job_id: int = None):
    """Worker calls this to report it's alive and what it's doing.

    Workers self-report: heartbeat (alive), current_task, process_state (working/complete).
    The system computes: stuck (from discard streaks), idle (from stale heartbeat).
    If a job id is supplied and that job has a kernel_type, the job-owned kernel identity
    wins over the caller-provided label. This keeps project/job heartbeats from being
    accidentally attributed to the wrong worker family.
    """
    canonical_kernel = (kernel_type or "").strip()
    if job_id is not None:
        job = self.get_job(job_id)
        job_worker = ((job or {}).get("assigned_to") or "").strip()
        job_kernel = ((job or {}).get("kernel_type") or "").strip()
        if job_worker:
            canonical_kernel = job_worker
        elif job_kernel:
            canonical_kernel = job_kernel
    if not canonical_kernel:
        raise ValueError("heartbeat requires a kernel type or a job with kernel_type set")

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cursor = self.conn.execute("""
        UPDATE worker_state SET heartbeat_at = ?, current_task = ?,
            process_state = ?, job_id = ?, updated_at = ?
        WHERE kernel_type = ?
    """, (now, current_task[:200], process_state, job_id, now, canonical_kernel))
    if cursor.rowcount == 0:
        self.conn.execute("""
            INSERT OR IGNORE INTO worker_state (kernel_type, heartbeat_at,
                current_task, process_state, job_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (canonical_kernel, now, current_task[:200], process_state, job_id, now))
    self.conn.commit()
    return canonical_kernel

def refresh_worker_state(self) -> dict:
    """Compute worker state from structured experiment history + heartbeat."""

    existing = {}
    for row in self.conn.execute(
        "SELECT kernel_type, tsv_path, heartbeat_at, current_task, job_id, process_state FROM worker_state"
    ).fetchall():
        existing[row[0]] = dict(row)

    kernels = set(existing.keys())
    kernels.update(
        r[0] for r in self.conn.execute(
            "SELECT DISTINCT kernel_type FROM experiments WHERE kernel_type != ''"
        ).fetchall()
    )
    kernels.update(
        r[0] for r in self.conn.execute(
            "SELECT DISTINCT kernel_type FROM jobs WHERE kernel_type != ''"
        ).fetchall()
    )

    results = {}

    for kernel in sorted(kernels):
        prior = existing.get(kernel, {})
        rows = [
            dict(r) for r in self.conn.execute("""
                SELECT * FROM experiments
                WHERE kernel_type = ?
                ORDER BY COALESCE(timestamp, recorded_at) ASC, id ASC
            """, (kernel,)).fetchall()
        ]

        state = {
            "kernel_type": kernel,
            "tsv_path": prior.get("tsv_path", ""),
            "total_experiments": 0,
            "kept": 0,
            "discarded": 0,
            "best_vsref": None,
            "best_duration_us": None,
            "top_stall": "",
            "current_discard_streak": 0,
            "max_discard_streak": 0,
            "last_kept_description": "",
            "last_experiment_time": "",
            "has_halt_note": 0,
            "status": "idle",
            "diagnosis": "no experiment data",
            "heartbeat_at": prior.get("heartbeat_at", ""),
            "current_task": prior.get("current_task", ""),
            "job_id": prior.get("job_id"),
            "process_state": prior.get("process_state", ""),
            "live_status": "historical",
            "live_reason": "no worker heartbeat recorded",
            "activity_at": "",
        }

        if not rows:
            heartbeat_recent = False
            if state.get("heartbeat_at"):
                try:
                    import calendar
                    hb_epoch = calendar.timegm(time.strptime(state["heartbeat_at"], "%Y-%m-%dT%H:%M:%SZ"))
                    heartbeat_recent = (time.time() - hb_epoch) <= 600
                except (ValueError, OverflowError):
                    heartbeat_recent = False
            if state.get("process_state") == "complete":
                state["status"] = "complete"
                state["diagnosis"] = f"worker self-reported complete. task: {state.get('current_task', '')[:60]}"
                state["live_status"] = "complete"
                state["live_reason"] = f"worker self-reported complete ({state.get('heartbeat_at') or 'no heartbeat'})"
                state["activity_at"] = state.get("heartbeat_at", "")
            elif heartbeat_recent:
                state["status"] = "producing"
                state["diagnosis"] = f"active worker heartbeat. task: {state.get('current_task', '')[:80]}"
                state["live_status"] = "active"
                state["live_reason"] = f"recent heartbeat ({state['heartbeat_at']})"
                state["activity_at"] = state.get("heartbeat_at", "")
            results[kernel] = state
            continue

        kept_rows = []
        streak = 0
        max_streak = 0
        for row in rows:
            row_status = (row.get("status") or "").strip().lower()
            if row_status in ("keep", "kept"):
                kept_rows.append(row)
                streak = 0
            elif row_status in ("discard", "discarded"):
                streak += 1
                max_streak = max(max_streak, streak)

        tail_streak = 0
        for row in reversed(rows):
            row_status = (row.get("status") or "").strip().lower()
            if row_status in ("discard", "discarded"):
                tail_streak += 1
            else:
                break

        best_vsref = None
        best_duration = None
        for row in kept_rows:
            vr = row.get("vs_ref")
            if vr is not None and (best_vsref is None or vr > best_vsref):
                best_vsref = vr
            dur = row.get("duration_us")
            if dur is not None and (best_duration is None or dur < best_duration):
                best_duration = dur

        last_row = rows[-1]
        top_stall = (last_row.get("top_stall") or "").strip()
        if top_stall in ("-", "none", ""):
            top_stall = ""

        last_kept_desc = ""
        if kept_rows:
            last_kept_desc = (kept_rows[-1].get("description") or "").strip()

        last_time = (last_row.get("timestamp") or "").strip() or (last_row.get("recorded_at") or "").strip()

        # Determine status (hybrid: self-reported + computed)
        # 1. Worker self-reported "complete" → trust it
        # 2. Heartbeat stale (>30 min) → dead/idle (computed)
        # 3. Discard streaks → stuck/grinding (computed, worker can't see this)
        # 4. Otherwise → producing
        heartbeat_stale = False
        heartbeat_recent = False
        hb_epoch = None
        last_exp_epoch = None
        if state.get("heartbeat_at"):
            try:
                import calendar
                hb_epoch = calendar.timegm(time.strptime(state["heartbeat_at"], "%Y-%m-%dT%H:%M:%SZ"))
                heartbeat_recent = (time.time() - hb_epoch) <= 600   # 10 min
                heartbeat_stale = (time.time() - hb_epoch) > 1800    # 30 min
            except (ValueError, OverflowError):
                pass
        try:
            import calendar
            last_exp_epoch = calendar.timegm(time.strptime(last_time, "%Y-%m-%dT%H:%M:%SZ"))
        except (ValueError, OverflowError):
            last_exp_epoch = None

        if state.get("process_state") == "complete":
            status = "complete"
            diagnosis = f"worker self-reported complete. task: {state.get('current_task', '')[:60]}"
        elif heartbeat_recent:
            status = "producing"
            if tail_streak >= 5:
                diagnosis = f"active worker investigating prior {tail_streak}-discard streak. task: {state.get('current_task', '')[:60]}"
            elif kept_rows:
                diagnosis = f"active worker heartbeat. last keep: {last_kept_desc[:80]}"
            else:
                diagnosis = f"active worker heartbeat. task: {state.get('current_task', '')[:60]}"
        elif heartbeat_stale and state.get("heartbeat_at"):
            status = "idle"
            diagnosis = f"heartbeat stale (last: {state['heartbeat_at']})"
        elif tail_streak >= 10:
            status = "stalled"
            diagnosis = f"{tail_streak} consecutive discards — likely exhausted current approach"
        elif tail_streak >= 5:
            status = "grinding"
            diagnosis = f"{tail_streak} consecutive discards — spinning without progress"
        elif len(kept_rows) >= 3:
            # Check convergence: last 3 keeps within 2%
            recent_vsrefs = []
            for kr in kept_rows[-3:]:
                try:
                    recent_vsrefs.append(float(kr.get("vs_ref", "0")))
                except (ValueError, TypeError):
                    pass
            if len(recent_vsrefs) == 3:
                spread = max(recent_vsrefs) - min(recent_vsrefs)
                avg = sum(recent_vsrefs) / 3
                if avg > 0 and spread / avg < 0.02:
                    status = "converged"
                    diagnosis = f"last 3 keeps within 2% ({min(recent_vsrefs):.2f}–{max(recent_vsrefs):.2f}x)"
                else:
                    status = "producing"
                    diagnosis = f"last keep: {last_kept_desc[:80]}"
            else:
                status = "producing"
                diagnosis = f"last keep: {last_kept_desc[:80]}"
        else:
            status = "producing"
            diagnosis = f"{len(kept_rows)} kept so far, tail streak={tail_streak}"

        if state.get("process_state") == "complete":
            live_status = "complete"
            live_reason = f"worker self-reported complete ({state.get('heartbeat_at') or 'no heartbeat'})"
        elif heartbeat_recent:
            live_status = "active"
            live_reason = f"recent heartbeat ({state['heartbeat_at']})"
        elif state.get("heartbeat_at"):
            live_status = "stale"
            live_reason = f"stale heartbeat ({state['heartbeat_at']})"
        elif last_exp_epoch and (time.time() - last_exp_epoch) <= 21600:
            live_status = "untracked_recent"
            live_reason = f"recent experiment results without heartbeat ({last_time})"
        elif last_time:
            live_status = "historical"
            live_reason = f"last experiment {last_time}"
        else:
            live_status = "historical"
            live_reason = "no recent activity"

        activity_candidates = [t for t in (state.get("heartbeat_at", ""), last_time) if t]
        activity_at = max(activity_candidates) if activity_candidates else ""

        state.update({
            "total_experiments": len(rows),
            "kept": len(kept_rows),
            "discarded": len(rows) - len(kept_rows),
            "best_vsref": best_vsref,
            "best_duration_us": best_duration,
            "top_stall": top_stall,
            "current_discard_streak": tail_streak,
            "max_discard_streak": max_streak,
            "last_kept_description": last_kept_desc[:200],
            "last_experiment_time": last_time,
            "status": status,
            "diagnosis": diagnosis,
            "live_status": live_status,
            "live_reason": live_reason,
            "activity_at": activity_at,
        })

        results[kernel] = state

    # Write to DB
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for kernel, s in results.items():
        self.conn.execute("""
            INSERT INTO worker_state (
                kernel_type, tsv_path, total_experiments, kept, discarded,
                best_vsref, best_duration_us, top_stall,
                current_discard_streak, max_discard_streak,
                last_kept_description, last_experiment_time,
                has_halt_note, status, diagnosis, updated_at,
                heartbeat_at, current_task, job_id, process_state,
                live_status, live_reason, activity_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(kernel_type) DO UPDATE SET
                tsv_path=excluded.tsv_path,
                total_experiments=excluded.total_experiments,
                kept=excluded.kept, discarded=excluded.discarded,
                best_vsref=excluded.best_vsref,
                best_duration_us=excluded.best_duration_us,
                top_stall=excluded.top_stall,
                current_discard_streak=excluded.current_discard_streak,
                max_discard_streak=excluded.max_discard_streak,
                last_kept_description=excluded.last_kept_description,
                last_experiment_time=excluded.last_experiment_time,
                has_halt_note=excluded.has_halt_note,
                status=excluded.status,
                diagnosis=excluded.diagnosis,
                updated_at=excluded.updated_at,
                heartbeat_at=excluded.heartbeat_at,
                current_task=excluded.current_task,
                job_id=excluded.job_id,
                process_state=excluded.process_state,
                live_status=excluded.live_status,
                live_reason=excluded.live_reason,
                activity_at=excluded.activity_at
        """, (kernel, s["tsv_path"], s["total_experiments"], s["kept"],
              s["discarded"], s["best_vsref"], s["best_duration_us"],
              s["top_stall"], s["current_discard_streak"],
              s["max_discard_streak"], s["last_kept_description"],
              s["last_experiment_time"], s["has_halt_note"],
              s["status"], s["diagnosis"], now,
              s.get("heartbeat_at", ""), s.get("current_task", ""),
              s.get("job_id"), s.get("process_state", ""),
              s.get("live_status", ""), s.get("live_reason", ""),
              s.get("activity_at", "")))

    self.conn.commit()

    for kernel, s in results.items():
        if s.get("status") not in ("stalled", "grinding"):
            continue
        job_id = s.get("job_id")
        if not job_id:
            row = self.conn.execute("""
                SELECT id FROM jobs
                WHERE kernel_type = ?
                  AND state NOT IN ('shipped', 'converged', 'parked', 'abandoned')
                ORDER BY CAST(priority AS INTEGER), updated_at DESC
                LIMIT 1
            """, (kernel,)).fetchone()
            job_id = row[0] if row else None
        if not job_id:
            continue
        root = str(getattr(self, "BWK_ROOT", "") or "").rstrip("/")
        search_bin = f"{root}/common/memory/msearch" if root else "common/memory/msearch"
        query = f"{search_bin} \"{kernel} {s.get('top_stall') or 'bottleneck'}\" --kernel {kernel} -k 5"
        body = (
            f"Worker/progress status is {s.get('status')} for kernel '{kernel}'. Before continuing, run a research checkpoint against the DB and post the useful findings back into the work. Suggested query: {query}. If the playbook is empty, widen the search and then mark the job stuck_needs_research."
        )
        self.ensure_open_message('watchdog',
                                 f"Research checkpoint required for job #{job_id}",
                                 body=body, job_id=job_id,
                                 message_type='info', priority='normal')

    return results

def get_worker_state(self, kernel_type: str = None) -> list[dict]:
    """Query worker state. If kernel_type is None, returns all workers."""
    if kernel_type:
        rows = self.conn.execute(
            "SELECT * FROM worker_state WHERE kernel_type = ?", (kernel_type,)
        ).fetchall()
    else:
        rows = self.conn.execute(
            "SELECT * FROM worker_state ORDER BY "
            "CASE live_status "
            "  WHEN 'active' THEN 1 "
            "  WHEN 'stale' THEN 2 "
            "  WHEN 'untracked_recent' THEN 3 "
            "  WHEN 'complete' THEN 4 "
            "  ELSE 5 END, "
            "CASE status "
            "  WHEN 'stalled' THEN 1 "
            "  WHEN 'grinding' THEN 2 "
            "  WHEN 'halted' THEN 3 "
            "  WHEN 'producing' THEN 4 "
            "  WHEN 'converged' THEN 5 "
            "  ELSE 6 END, "
            "current_discard_streak DESC"
        ).fetchall()
    return [dict(r) for r in rows]

def _remove_document(self, doc_id: int):
    """Remove a document and all its chunks/vectors."""
    chunk_ids = [r["id"] for r in self.conn.execute(
        "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()]
    for cid in chunk_ids:
        self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
    self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

