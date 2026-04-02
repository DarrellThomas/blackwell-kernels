"""Job lifecycle helpers extracted from factory_brain.
Expect `self` with .conn, ensure_open_message, validate_transition, constants set on class.
"""
import time, json, hashlib


def attach_job_methods(cls):
    # Caller (factory_brain) pre-populates constants to avoid import cycles.
    cls.validate_transition = staticmethod(getattr(cls, 'validate_transition'))
    cls.create_job = create_job
    cls.update_job_state = update_job_state
    cls.update_job = update_job
    cls.get_job = get_job
    cls.get_job_by_name = get_job_by_name
    cls.get_jobs = get_jobs
    cls.get_job_history = get_job_history
    cls.get_watchdog_state = get_watchdog_state
    cls.touch_watchdog_state = touch_watchdog_state
    cls.set_watchdog_daemon_state = set_watchdog_daemon_state
    cls.sync_job_vsref = sync_job_vsref
    cls.BWK_ROOT = getattr(cls, "BWK_ROOT", None)
    cls.ALL_JOB_STATES = getattr(cls, "ALL_JOB_STATES", set())
    cls.STATE_TO_PHASE = getattr(cls, "STATE_TO_PHASE", {})
    cls.FACTORY_MODES = getattr(cls, "FACTORY_MODES", set())
    cls.OPTIMIZATION_SCOPES = getattr(cls, "OPTIMIZATION_SCOPES", set())
    cls.EXECUTION_LANES = getattr(cls, "EXECUTION_LANES", set())
# -- Jobs (workpiece lifecycle tracking) --

def create_job(self, name, title, description="", job_type="kernel", kernel_type="",
               parent_job_id=None, state="wishlist", priority="3", assigned_to="",
               execution_lane="", target_vs_ref=1.0, tags="", created_by="ops", notes="",
               source_file="", factory_mode="", objective_vector="",
               acceptance_gates="", keep_rule="", benchmark_set="",
               failure_budget="", crossover_policy="", optimization_scope="",
               hardware_target="", retarget_policy="", reference_label="") -> int:
    if state not in self.ALL_JOB_STATES:
        raise ValueError(f"Unknown state '{state}'. Valid: {sorted(self.ALL_JOB_STATES)}")
    if job_type not in self.JOB_TYPES:
        raise ValueError(f"Unknown job type '{job_type}'. Valid: {sorted(self.JOB_TYPES)}")
    if priority not in self.JOB_PRIORITIES:
        raise ValueError(f"Unknown priority '{priority}'. Valid: {sorted(self.JOB_PRIORITIES)}")
    if factory_mode and factory_mode not in self.FACTORY_MODES:
        raise ValueError(f"Unknown factory mode '{factory_mode}'. Valid: {sorted(self.FACTORY_MODES)}")
    if optimization_scope and optimization_scope not in self.OPTIMIZATION_SCOPES:
        raise ValueError(
            f"Unknown optimization scope '{optimization_scope}'. "
            f"Valid: {sorted(self.OPTIMIZATION_SCOPES)}"
        )
    if execution_lane and execution_lane not in self.EXECUTION_LANES:
        raise ValueError(
            f"Unknown execution lane '{execution_lane}'. Valid: {sorted(self.EXECUTION_LANES)}"
        )
    phase = self.STATE_TO_PHASE[state]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cursor = self.conn.execute("""
        INSERT INTO jobs (name, title, description, job_type, kernel_type,
                          parent_job_id, state, phase, priority, assigned_to, execution_lane,
                          target_vs_ref, tags, created_at, updated_at,
                          created_by, updated_by, notes, source_file,
                          factory_mode, objective_vector, acceptance_gates,
                          keep_rule, benchmark_set, failure_budget,
                          crossover_policy, optimization_scope,
                          hardware_target, retarget_policy, reference_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, title, description, job_type, kernel_type,
          parent_job_id, state, phase, priority, assigned_to, execution_lane,
          target_vs_ref, tags, now, now, created_by, created_by, notes, source_file,
          factory_mode, objective_vector, acceptance_gates, keep_rule,
          benchmark_set, failure_budget, crossover_policy, optimization_scope,
          hardware_target, retarget_policy, reference_label))
    job_id = cursor.lastrowid
    self.conn.execute("""
        INSERT INTO job_transitions (job_id, from_state, to_state, changed_by, reason, timestamp)
        VALUES (?, '', ?, ?, 'created', ?)
    """, (job_id, state, created_by, now))
    self.conn.commit()
    return job_id

def update_job_state(self, job_id, to_state, changed_by, reason=""):
    job = self.get_job(job_id)
    if not job:
        raise ValueError(f"Job #{job_id} not found")
    from_state = job["state"]
    valid, err = self.validate_transition(from_state, to_state)
    if not valid:
        raise ValueError(err)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new_phase = self.STATE_TO_PHASE[to_state]

    # Version bump: every time a job hits 'shipped' internally, minor version increments.
    # Internal versions are 0.x (0.1, 0.2, ...). Public release will be 1.0.
    lane_update = ""
    lane_params = []
    if to_state in ('shipped', 'converged', 'abandoned', 'parked'):
        lane_update = ", execution_lane = ?"
        lane_params = ['parked']
    if to_state == 'shipped':
        cur = job.get("version", 0) or 0
        version = round(cur + 0.1, 1)
        reason = f"[v{version}] {reason}" if reason else f"[v{version}] shipped"
        self.conn.execute(
            f"UPDATE jobs SET state = ?, phase = ?, version = ?, updated_at = ?, updated_by = ?{lane_update} WHERE id = ?",
            (to_state, new_phase, version, now, changed_by, *lane_params, job_id))
    else:
        self.conn.execute(
            f"UPDATE jobs SET state = ?, phase = ?, updated_at = ?, updated_by = ?{lane_update} WHERE id = ?",
            (to_state, new_phase, now, changed_by, *lane_params, job_id))
    self.conn.execute("""
        INSERT INTO job_transitions (job_id, from_state, to_state, changed_by, reason, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (job_id, from_state, to_state, changed_by, reason, now))
    self.conn.commit()

    if to_state == 'rework':
        kernel = job.get('kernel_type') or '<kernel>'
        root = str(getattr(self, "BWK_ROOT", "") or "").rstrip("/")
        search_bin = f"{root}/common/memory/msearch" if root else "common/memory/msearch"
        query = (
            f"{search_bin} \"{kernel} failure root cause\" --kernel {kernel} -k 5"
            if kernel != "<kernel>" else
            f"{search_bin} \"failure root cause\" -k 5"
        )
        self.ensure_open_message("watchdog",
                                 f"Research checkpoint required for job #{job_id}",
                                 body=(
                                     "Job entered rework. Before continuing, read the failure messages, run a research checkpoint against the DB, and use the result to guide the next fix. Suggested query: " + query
                                 ),
                                 job_id=job_id, message_type='info', priority='normal')

    return self.get_job(job_id)

def update_job(self, job_id, updated_by="ops", **kwargs):
    allowed = {"title", "description", "priority", "assigned_to", "execution_lane", "vs_ref",
                "target_vs_ref", "tags", "notes", "kernel_type", "spec",
                "source_file", "factory_mode", "objective_vector",
                "acceptance_gates", "keep_rule", "benchmark_set",
                "failure_budget", "crossover_policy", "optimization_scope",
                "hardware_target", "retarget_policy", "reference_label"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return self.get_job(job_id)
    if "factory_mode" in updates and updates["factory_mode"] and updates["factory_mode"] not in self.FACTORY_MODES:
        raise ValueError(f"Unknown factory mode '{updates['factory_mode']}'. Valid: {sorted(self.FACTORY_MODES)}")
    if "optimization_scope" in updates and updates["optimization_scope"] and updates["optimization_scope"] not in self.OPTIMIZATION_SCOPES:
        raise ValueError(
            f"Unknown optimization scope '{updates['optimization_scope']}'. "
            f"Valid: {sorted(self.OPTIMIZATION_SCOPES)}"
        )
    if "execution_lane" in updates and updates["execution_lane"] and updates["execution_lane"] not in self.EXECUTION_LANES:
        raise ValueError(
            f"Unknown execution lane '{updates['execution_lane']}'. Valid: {sorted(self.EXECUTION_LANES)}"
        )
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    updates["updated_at"] = now
    updates["updated_by"] = updated_by
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [job_id]
    self.conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
    self.conn.commit()
    return self.get_job(job_id)

def get_job(self, job_id):
    row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None

def get_job_by_name(self, name):
    row = self.conn.execute("SELECT * FROM jobs WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None

def get_jobs(self, state=None, phase=None, job_type=None, kernel_type=None,
             assigned_to=None, parent_job_id=None, priority=None, execution_lane=None):
    where, params = [], []
    if state:
        where.append("state = ?"); params.append(state)
    if phase:
        where.append("phase = ?"); params.append(phase)
    if job_type:
        where.append("job_type = ?"); params.append(job_type)
    if kernel_type:
        where.append("kernel_type = ?"); params.append(kernel_type)
    if assigned_to:
        where.append("assigned_to = ?"); params.append(assigned_to)
    if execution_lane:
        where.append("execution_lane = ?"); params.append(execution_lane)
    if parent_job_id is not None:
        where.append("parent_job_id = ?"); params.append(parent_job_id)
    if priority:
        where.append("priority = ?"); params.append(priority)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    rows = self.conn.execute(f"""
        SELECT * FROM jobs {where_sql}
        ORDER BY CASE execution_lane WHEN 'active' THEN 0 WHEN 'hopper' THEN 1 WHEN 'incubating' THEN 2 WHEN 'parked' THEN 3 ELSE 4 END,
                 CAST(priority AS INTEGER), updated_at DESC
    """, params).fetchall()
    return [dict(r) for r in rows]

def get_job_history(self, job_id):
    rows = self.conn.execute(
        "SELECT * FROM job_transitions WHERE job_id = ? ORDER BY timestamp ASC",
        (job_id,)).fetchall()
    return [dict(r) for r in rows]

def get_watchdog_state(self, name: str) -> dict | None:
    row = self.conn.execute(
        "SELECT * FROM watchdog_state WHERE name = ?",
        (name,)
    ).fetchone()
    return dict(row) if row else None

def touch_watchdog_state(self, name: str, status: str = "", notes: str = "") -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    self.conn.execute("""
        INSERT INTO watchdog_state (name, last_run_at, last_status, notes)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            last_run_at=excluded.last_run_at,
            last_status=excluded.last_status,
            notes=excluded.notes
    """, (name, now, status, notes))
    self.conn.commit()
    return self.get_watchdog_state(name)

def set_watchdog_daemon_state(self, status: str, notes: str = "", pid: int | None = None, host: str = "") -> dict:
    parts = []
    if pid is not None:
        parts.append(f"pid={pid}")
    if host:
        parts.append(f"host={host}")
    if notes:
        parts.append(notes)
    return self.touch_watchdog_state("watchdog_daemon", status=status, notes=" | ".join(parts))

def sync_job_vsref(self, job_id):
    job = self.get_job(job_id)
    if not job or not job.get("kernel_type"):
        return job
    row = self.conn.execute(
        "SELECT best_vsref FROM worker_state WHERE kernel_type = ?",
        (job["kernel_type"],)).fetchone()
    if row and row[0] is not None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute("UPDATE jobs SET vs_ref = ?, updated_at = ? WHERE id = ?",
                          (row[0], now, job_id))
        self.conn.commit()
    return self.get_job(job_id)

