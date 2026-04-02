"""Issue helpers extracted from factory_brain.
Expect `self` with .conn.
"""
import time

def attach_issue_methods(cls):
    cls.file_issue = file_issue
    cls.assign_issue = assign_issue
    cls.rework_issue = rework_issue
    cls.close_issue = close_issue
    cls.reopen_issue = reopen_issue
    cls.get_issues = get_issues
    return cls

# -- Issues --

def file_issue(self, title: str, severity: str, kernel_type: str,
               source_file: str, description: str, reproduce: str = "",
               filed_by: str = "tester") -> int:
    """File a new issue. Returns issue ID."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    self.conn.execute("""
        INSERT INTO issues (title, severity, status, kernel_type, source_file,
                            filed_by, description, reproduce, filed_at, updated_at)
        VALUES (?, ?, 'open', ?, ?, ?, ?, ?, ?, ?)
    """, (title, severity, kernel_type, source_file, filed_by,
          description, reproduce, now, now))
    self.conn.commit()
    issue_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return issue_id

def assign_issue(self, issue_id: int, assigned_to: str):
    """Foreman assigns an issue to a worker."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    self.conn.execute(
        "UPDATE issues SET assigned_to = ?, status = 'assigned', updated_at = ? WHERE id = ?",
        (assigned_to, now, issue_id)
    )
    self.conn.commit()

def rework_issue(self, issue_id: int, fix_description: str):
    """Worker submits a fix for re-testing."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    self.conn.execute(
        "UPDATE issues SET fix_description = ?, status = 'retest', updated_at = ? WHERE id = ?",
        (fix_description, now, issue_id)
    )
    self.conn.commit()

def close_issue(self, issue_id: int):
    """Tester verifies fix and closes the issue."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    self.conn.execute(
        "UPDATE issues SET status = 'closed', closed_at = ?, updated_at = ? WHERE id = ?",
        (now, now, issue_id)
    )
    self.conn.commit()

def reopen_issue(self, issue_id: int, reason: str):
    """Tester reopens if fix didn't work."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    self.conn.execute(
        "UPDATE issues SET status = 'open', fix_description = ?, updated_at = ? WHERE id = ?",
        (f"REOPENED: {reason}", now, issue_id)
    )
    self.conn.commit()

def get_issues(self, status: str = None, kernel_type: str = None) -> list[dict]:
    """Query issues. Sorted: open first, then by severity."""
    where = []
    params = []
    if status:
        where.append("status = ?")
        params.append(status)
    if kernel_type:
        where.append("kernel_type = ?")
        params.append(kernel_type)

    where_sql = ""
    if where:
        where_sql = "WHERE " + " AND ".join(where)

    rows = self.conn.execute(f"""
        SELECT * FROM issues {where_sql}
        ORDER BY
            CASE status
                WHEN 'open' THEN 1
                WHEN 'assigned' THEN 2
                WHEN 'retest' THEN 3
                WHEN 'closed' THEN 4
            END,
            CASE severity
                WHEN 'blocking' THEN 1
                WHEN 'correctness' THEN 2
                WHEN 'warning' THEN 3
            END,
            filed_at DESC
    """, params).fetchall()
    return [dict(r) for r in rows]

# -- Jobs (workpiece lifecycle tracking) --

# -- Messages --

# -- Experiments --

# -- Stats & Quality --

