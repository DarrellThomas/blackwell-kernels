"""Message helpers extracted from factory_brain.
These functions expect `self` with .conn.
"""
import time

def attach_message_methods(cls):
    cls.create_message = create_message
    cls.ensure_open_message = ensure_open_message
    cls.acknowledge_message = acknowledge_message
    cls.resolve_message = resolve_message
    cls.get_message = get_message
    cls.get_messages = get_messages
    return cls

# -- Messages --

def create_message(self, from_agent, subject, body="", to_agent="",
                   job_id=None, message_type="info", priority="normal"):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cursor = self.conn.execute("""
        INSERT INTO messages (job_id, from_agent, to_agent, message_type,
                              subject, body, status, priority, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)
    """, (job_id, from_agent, to_agent, message_type, subject, body, priority, now))
    self.conn.commit()
    return cursor.lastrowid

def ensure_open_message(self, from_agent, subject, body="", to_agent="",
                        job_id=None, message_type="info", priority="normal"):
    row = self.conn.execute("""
        SELECT id FROM messages
        WHERE status = 'open' AND from_agent = ? AND subject = ?
          AND COALESCE(job_id, -1) = COALESCE(?, -1)
        ORDER BY id DESC
        LIMIT 1
    """, (from_agent, subject, job_id)).fetchone()
    if row:
        return row[0]
    return self.create_message(from_agent=from_agent, subject=subject, body=body,
                               to_agent=to_agent, job_id=job_id,
                               message_type=message_type, priority=priority)

def acknowledge_message(self, message_id, by="foreman"):
    self.conn.execute("UPDATE messages SET status = 'acknowledged' WHERE id = ?", (message_id,))
    self.conn.commit()
    return self.get_message(message_id)

def resolve_message(self, message_id, by="foreman"):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    self.conn.execute(
        "UPDATE messages SET status = 'resolved', resolved_at = ?, resolved_by = ? WHERE id = ?",
        (now, by, message_id))
    self.conn.commit()
    return self.get_message(message_id)

def get_message(self, message_id):
    row = self.conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    return dict(row) if row else None

def get_messages(self, status=None, job_id=None, from_agent=None,
                 to_agent=None, message_type=None):
    where, params = [], []
    if status:
        where.append("status = ?"); params.append(status)
    if job_id is not None:
        where.append("job_id = ?"); params.append(job_id)
    if from_agent:
        where.append("from_agent = ?"); params.append(from_agent)
    if to_agent:
        where.append("to_agent = ?"); params.append(to_agent)
    if message_type:
        where.append("message_type = ?"); params.append(message_type)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    rows = self.conn.execute(f"""
        SELECT * FROM messages {where_sql}
        ORDER BY
            CASE status WHEN 'open' THEN 1 WHEN 'acknowledged' THEN 2 ELSE 3 END,
            CASE priority WHEN 'urgent' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
            created_at DESC
    """, params).fetchall()
    return [dict(r) for r in rows]

