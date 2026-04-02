"""HTTP server for ResearchMemory APIs."""
from __future__ import annotations

import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# We intentionally keep this module lightweight; caller passes a ResearchMemory instance
# to avoid a circular import.


def serve(mem, port: int = 8421):
    def _parse_int_param(params, name: str, default: int) -> int:
        raw = params.get(name, [str(default)])[0]
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid integer for '{name}': {raw}") from exc

    class Handler(BaseHTTPRequestHandler):
        def _json_response(self, payload, status: int = 200):
            import json
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                if parsed.path == "/api/search":
                    query = params.get("q", [""])[0]
                    k = _parse_int_param(params, "k", 10)
                    mode = params.get("mode", ["hybrid"])[0]
                    level = _parse_int_param(params, "level", 0)
                    kernel = params.get("kernel", [None])[0]
                    doc_type = params.get("type", [None])[0]
                    stall = params.get("stall", [None])[0]
                    technique = params.get("technique", [None])[0]

                    if level > 0:
                        results = mem.search_summaries(query, k, kernel, doc_type, stall, technique, level=level)
                    elif mode == "fts":
                        results = mem.search_fts(query, k, kernel, doc_type, stall, technique)
                    elif mode == "semantic":
                        results = mem.search_semantic(query, k, kernel, doc_type, stall, technique)
                    else:
                        has_summaries = mem.conn.execute("SELECT COUNT(*) FROM vec_summaries").fetchone()[0]
                        if has_summaries > 0:
                            results = mem.search_summaries(query, k, kernel, doc_type, stall, technique, level=2)
                        else:
                            results = mem.search_hybrid(query, k, kernel, doc_type, stall, technique)

                    self._json_response({"results": results, "query": query, "count": len(results)})

                elif parsed.path == "/api/stats":
                    self._json_response(mem.stats())

                elif parsed.path == "/api/quality":
                    self._json_response({"report": mem.quality_report()})

                elif parsed.path == "/api/workers":
                    if params.get("refresh", ["0"])[0] in ("1", "true", "yes"):
                        mem.refresh_worker_state()
                    workers = mem.get_worker_state()
                    self._json_response({"workers": workers, "count": len(workers)})

                elif parsed.path == "/api/issues":
                    status = params.get("status", [None])[0]
                    kernel = params.get("kernel", [None])[0]
                    issues = mem.get_issues(status=status, kernel_type=kernel)
                    open_count = sum(1 for i in issues if i["status"] in ("open", "assigned"))
                    self._json_response({"issues": issues, "count": len(issues), "open": open_count})

                elif parsed.path.startswith("/api/issues/"):
                    parts = parsed.path.split("/")
                    if len(parts) >= 4:
                        try:
                            issue_id = int(parts[3])
                        except ValueError:
                            self._json_response({"error": "Invalid issue ID"}, 400)
                            return
                        action = parts[4] if len(parts) > 4 else "get"
                        if action == "assign":
                            to = params.get("to", [""])[0]
                            mem.assign_issue(issue_id, to)
                            self._json_response({"ok": True, "issue_id": issue_id, "assigned_to": to})
                        elif action == "rework":
                            fix = params.get("fix", [""])[0]
                            mem.rework_issue(issue_id, fix)
                            self._json_response({"ok": True, "issue_id": issue_id, "status": "retest"})
                        elif action == "close":
                            mem.close_issue(issue_id)
                            self._json_response({"ok": True, "issue_id": issue_id, "status": "closed"})
                        elif action == "reopen":
                            reason = params.get("reason", [""])[0]
                            mem.reopen_issue(issue_id, reason)
                            self._json_response({"ok": True, "issue_id": issue_id, "status": "open"})
                        else:
                            issues = mem.get_issues()
                            match = [i for i in issues if i["id"] == issue_id]
                            self._json_response(match[0] if match else {"error": "Not found"}, 200 if match else 404)
                    else:
                        self._json_response({"error": "Not found"}, 404)

                elif parsed.path == "/api/jobs":
                    jobs = mem.get_jobs(state=params.get("state", [None])[0],
                                        phase=params.get("phase", [None])[0],
                                        job_type=params.get("type", [None])[0],
                                        kernel_type=params.get("kernel", [None])[0],
                                        assigned_to=params.get("assigned", [None])[0],
                                        priority=params.get("priority", [None])[0])
                    active = sum(1 for j in jobs if j["state"] not in ("shipped", "converged", "parked", "abandoned"))
                    self._json_response({"jobs": jobs, "count": len(jobs), "active": active})

                elif parsed.path.startswith("/api/jobs/"):
                    parts = parsed.path.split("/")
                    if len(parts) >= 4:
                        if parts[3] == "new":
                            try:
                                jid = mem.create_job(
                                    name=params.get("name", [""])[0], title=params.get("title", [""])[0],
                                    description=params.get("description", [""])[0],
                                    job_type=params.get("type", ["kernel"])[0],
                                    kernel_type=params.get("kernel", [""])[0],
                                    priority=params.get("priority", ["3"])[0],
                                    factory_mode=params.get("factory_mode", [""])[0],
                                    optimization_scope=params.get("scope", [""])[0],
                                    objective_vector=params.get("objective", [""])[0],
                                    acceptance_gates=params.get("gates", [""])[0],
                                    keep_rule=params.get("keep_rule", [""])[0],
                                    benchmark_set=params.get("bench", [""])[0],
                                    hardware_target=params.get("hardware_target", [""])[0],
                                    retarget_policy=params.get("retarget_policy", [""])[0],
                                )
                                self._json_response({"ok": True, "job_id": jid})
                            except Exception as exc:  # pylint: disable=broad-except
                                self._json_response({"error": str(exc)}, 400)
                        else:
                            try:
                                job_id = int(parts[3])
                            except ValueError:
                                self._json_response({"error": "Invalid job ID"}, 400)
                                return
                            action = parts[4] if len(parts) > 4 else "get"
                            if action == "state":
                                new_state = params.get("to", [""])[0]
                                mem.update_job_state(job_id, new_state, "api")
                                self._json_response({"ok": True, "job_id": job_id, "state": new_state})
                            else:
                                job = mem.get_job(job_id)
                                self._json_response(job or {"error": "Not found"}, 404 if not job else 200)
                    else:
                        self._json_response({"error": "Not found"}, 404)

                else:
                    self._json_response({"error": "Not found"}, 404)
            except ValueError as exc:
                self._json_response({"error": str(exc)}, 400)
            except sqlite3.OperationalError as exc:
                self._json_response({"error": str(exc)}, 500)
            except Exception as exc:  # pragma: no cover - defensive server fallback
                self._json_response({"error": f"Internal server error: {exc.__class__.__name__}: {exc}"}, 500)

        def log_message(self, fmt, *args):  # noqa: D401
            # Silence default logging
            return

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving ResearchMemory HTTP API on :{port}")
    try:
        server.serve_forever()
    finally:
        mem.close()

__all__ = ["serve"]
