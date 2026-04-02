from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from common.memory import factory_brain as fb
from common.memory.factory_brain import ResearchMemory

ROOT = Path(__file__).resolve().parents[2]
MEMORY_SCRIPT = ROOT / "common/memory/factory_brain.py"


def _script_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not current else f"{ROOT}{os.pathsep}{current}"
    if extra:
        env.update(extra)
    return env


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_json(url: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1.0) as response:
                return json.loads(response.read().decode())
        except Exception as exc:  # pragma: no cover - retry path
            last_error = exc
            time.sleep(0.1)
    raise AssertionError(f"HTTP endpoint never became ready: {url} ({last_error})")


def _fetch_json(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=2.0) as response:
            return response.status, json.loads(response.read().decode())
    except HTTPError as exc:
        payload = exc.read().decode()
        return exc.code, json.loads(payload) if payload else {}


def test_dedup_and_incremental_reingest_track_canonical_docs(
    stub_memory_embeddings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary_dir = tmp_path / "validated"
    secondary_dir = tmp_path / "research"
    primary_dir.mkdir()
    secondary_dir.mkdir()

    duplicate_text = (
        "# Shared Kernel Note\n\n"
        "This swizzle write-up is long enough for indexing and covers bank conflict reduction, "
        "shared memory layout, and occupancy tuning. " * 3
    )
    primary = primary_dir / "shared.md"
    secondary = secondary_dir / "shared_copy.md"
    primary.write_text(duplicate_text)
    secondary.write_text(duplicate_text)

    monkeypatch.setattr(fb, "SOURCE_PRIORITY", [
        (primary_dir, "research", "validated", 1),
        (secondary_dir, "research", "research", 9),
    ])
    monkeypatch.setattr(fb, "CODE_SOURCES", [])

    mem = ResearchMemory(tmp_path / "dedup.db")
    try:
        first = mem.ingest_all()
        assert first["files"] == 1
        assert first["dedup_skipped"] == 1
        assert first["removed"] == 0

        docs = mem.conn.execute(
            "SELECT source_file, also_at, provenance, content_hash FROM documents"
        ).fetchall()
        assert len(docs) == 1
        assert docs[0]["source_file"] == str(primary.resolve())
        assert str(secondary.resolve()) in docs[0]["also_at"]
        assert docs[0]["provenance"] == "validated"
        first_hash = docs[0]["content_hash"]

        primary.write_text(
            "# Mutated Kernel Note\n\n"
            "This now describes an independently evolved kernel with register reuse, vectorized loads, "
            "and a different swizzle schedule. " * 4
        )

        second = mem.ingest_all()
        assert second["files"] == 2
        assert second["dedup_skipped"] == 0

        docs = mem.conn.execute(
            "SELECT source_file, also_at, title, content_hash FROM documents ORDER BY source_file"
        ).fetchall()
        assert len(docs) == 2
        assert docs[0]["also_at"] == ""
        assert docs[1]["also_at"] == ""
        mutated = [row for row in docs if row["source_file"] == str(primary.resolve())][0]
        assert mutated["title"] == "Mutated Kernel Note"
        assert mutated["content_hash"] != first_hash

        third = mem.ingest_all()
        assert third["files"] == 0
        assert third["skipped"] == 2
    finally:
        mem.close()


def test_search_fts_handles_blank_and_operator_heavy_queries(
    stub_memory_embeddings,
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "swizzle.md").write_text(
        "# Swizzle Note\n\n"
        "This note explains swizzle layouts, bank conflict reduction, and shared memory staging. " * 4
    )

    mem = ResearchMemory(tmp_path / "fts.db")
    try:
        stats = mem.ingest_directory(str(docs))
        assert stats["files"] == 1

        for query in ['"swizzle"', "swizzle)(", "swizzle OR bank", "AND", "   "]:
            results = mem.search_fts(query, k=5)
            assert isinstance(results, list)

        assert mem.search_fts("   ", k=5) == []
        assert mem.search_fts("AND", k=5) == []
        assert mem.search_fts('"swizzle"', k=5)[0]["title"] == "Swizzle Note"
    finally:
        mem.close()


def test_ingest_directory_skips_empty_and_tiny_docs_but_indexes_malformed_utf8(
    stub_memory_embeddings,
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "empty.md").write_text("")
    (docs / "tiny.md").write_text("# Tiny\n\nToo short.")
    (docs / "broken.md").write_bytes(
        b"# Broken Kernel Note\n\nThis malformed byte stream still mentions swizzle, scoreboard stalls, "
        b"and vectorized loads for a long enough ingest path. "
        b"\xff\xfe\x80"
        b"This file should still be indexed because read_text(errors='replace') is used."
    )

    mem = ResearchMemory(tmp_path / "broken.db")
    try:
        stats = mem.ingest_directory(str(docs))
        assert stats["files"] == 1
        assert stats["skipped"] == 2
        assert mem.stats()["documents"] == 1
        assert mem.search_fts("swizzle", k=3)[0]["title"] == "Broken Kernel Note"
    finally:
        mem.close()


def test_job_validation_rejects_invalid_metadata_and_illegal_transitions(tmp_path: Path) -> None:
    mem = ResearchMemory(tmp_path / "jobs.db")
    try:
        job_id = mem.create_job("job-valid", "Valid Job", kernel_type="gemm", execution_lane="active")

        with pytest.raises(ValueError, match="Unknown state"):
            mem.create_job("job-bad-state", "Bad State", state="bogus")
        with pytest.raises(ValueError, match="Unknown job type"):
            mem.create_job("job-bad-type", "Bad Type", job_type="bogus")
        with pytest.raises(ValueError, match="Unknown priority"):
            mem.create_job("job-bad-priority", "Bad Priority", priority="99")
        with pytest.raises(ValueError, match="Unknown factory mode"):
            mem.create_job("job-bad-mode", "Bad Mode", factory_mode="bogus")
        with pytest.raises(ValueError, match="Unknown execution lane"):
            mem.create_job("job-bad-lane", "Bad Lane", execution_lane="bogus")
        with pytest.raises(ValueError, match="Unknown execution lane"):
            mem.update_job(job_id, execution_lane="bogus")
        with pytest.raises(ValueError, match="Unknown target state"):
            mem.update_job_state(job_id, "bogus", "tester")

        shipped = mem.update_job_state(job_id, "shipped", "tester", reason="complete")
        assert shipped["state"] == "shipped"
        assert shipped["execution_lane"] == "parked"
        assert shipped["version"] == pytest.approx(0.1)

        with pytest.raises(ValueError, match="Cannot go backward from 'shipped'"):
            mem.update_job_state(job_id, "planning", "tester")
    finally:
        mem.close()


def test_refresh_worker_state_handles_grinding_watchdog_path(tmp_path: Path) -> None:
    mem = ResearchMemory(tmp_path / "workers.db")
    try:
        job_id = mem.create_job("watchdog-job", "Watchdog Job", kernel_type="gemm", execution_lane="active")
        for i in range(5):
            mem.record_experiment("gemm", "discard", f"discard {i}", job_id=job_id, top_stall="barrier")

        refreshed = mem.refresh_worker_state()
        assert refreshed["gemm"]["status"] == "grinding"

        messages = mem.get_messages(job_id=job_id, from_agent="watchdog")
        assert messages
        assert "Research checkpoint required" in messages[0]["subject"]
        assert "common/memory/msearch" in messages[0]["body"]
    finally:
        mem.close()


def test_hybrid_search_respects_technique_filter_in_fts_branch(
    stub_memory_embeddings,
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "swizzle.md").write_text(
        "# Swizzle Note\n\n"
        "This kernel uses swizzle to reduce bank conflicts in shared memory. " * 4
    )
    (docs / "occupancy.md").write_text(
        "# Occupancy Note\n\n"
        "This kernel improves occupancy and latency hiding without the target technique. " * 4
    )

    mem = ResearchMemory(tmp_path / "hybrid.db")
    try:
        stats = mem.ingest_directory(str(docs))
        assert stats["files"] == 2

        results = mem.search_hybrid("kernel", k=5, technique="swizzle")
        assert results
        assert [r["title"] for r in results] == ["Swizzle Note"]
        assert all("swizzle" in (r.get("techniques") or "") for r in results)
    finally:
        mem.close()


def test_reingest_clears_stale_summary_and_summary_vector(
    stub_memory_embeddings,
    tmp_path: Path,
) -> None:
    doc = tmp_path / "note.md"
    doc.write_text(
        "# Mutable Note\n\n"
        "This document is long enough to index and mentions swizzle and bank conflicts. " * 4
    )

    mem = ResearchMemory(tmp_path / "summary.db")
    try:
        assert mem.ingest_file(str(doc)) > 0
        doc_id = mem.conn.execute("SELECT id FROM documents").fetchone()[0]
        mem.conn.execute(
            "UPDATE documents SET summary = 'old summary', signal = 'old signal', has_summary = 1 WHERE id = ?",
            (doc_id,)
        )
        mem.conn.execute(
            "INSERT INTO vec_summaries (doc_id, embedding) VALUES (?, ?)",
            (doc_id, fb.serialize_f32([0.0] * fb.EMBEDDING_DIM))
        )
        mem.conn.commit()

        doc.write_text(
            "# Mutable Note\n\n"
            "This document changed materially and is still long enough for indexing. " * 4
        )
        assert mem.ingest_file(str(doc), force=True) > 0

        row = mem.conn.execute(
            "SELECT summary, signal, has_summary FROM documents WHERE id = ?",
            (doc_id,)
        ).fetchone()
        assert row["summary"] == ""
        assert row["signal"] == ""
        assert row["has_summary"] == 0
        assert mem.conn.execute(
            "SELECT COUNT(*) FROM vec_summaries WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()[0] == 0
    finally:
        mem.close()


def test_tsv_ingest_skips_bad_rows_and_refreshes_experiment_rows(
    stub_memory_embeddings,
    tmp_path: Path,
) -> None:
    mem = ResearchMemory(tmp_path / "tsv.db")
    missing_desc = tmp_path / "missing_description.tsv"
    missing_desc.write_text("timestamp\tstatus\tvs_ref\n2026-01-01T00:00:00Z\tkeep\t1.2\n")
    tsv_path = tmp_path / "demo.tsv"

    try:
        missing_stats = mem.ingest_tsv(str(missing_desc), kernel_type="gemm")
        assert missing_stats["skipped"] == 1
        assert mem.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
        assert mem.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0

        tsv_path.write_text(
            "timestamp\tcommit\tdescription\tstatus\tvs_ref\tduration_us\ttop_stall\tstall_math\n"
            "2026-01-01T00:00:00Z\tabc123\tshort\tkeep\t1.2\t10\tbarrier\t12\n"
            "2026-01-02T00:00:00Z\tdef456\tThis keep row is long enough and documents a swizzle win with reduced bank conflicts.\tkeep\t1.27\t9.5\tbarrier\t11\n"
            "2026-01-03T00:00:00Z\tghi789\tThis discard row is still valid input even though numeric fields are partial and messy.\tdiscard\tabc\t-\tlong_scoreboard\txyz\n"
            "2026-01-04T00:00:00Z\tjkl012\t    \tdiscard\t-\t-\t-\t-\n"
        )

        first = mem.ingest_tsv(str(tsv_path), kernel_type="gemm")
        assert first["files"] == 1
        assert first["chunks"] == 2
        assert first["kept"] == 1
        assert first["discarded"] == 1

        rows = mem.conn.execute(
            "SELECT status, vs_ref, duration_us, top_stall FROM experiments WHERE source_path = ? ORDER BY experiment_index",
            (str(tsv_path.resolve()),)
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["status"] == "keep"
        assert rows[0]["vs_ref"] == pytest.approx(1.27)
        assert rows[1]["status"] == "discard"
        assert rows[1]["vs_ref"] is None
        assert rows[1]["duration_us"] is None
        assert rows[1]["top_stall"] == "long_scoreboard"

        tsv_path.write_text(
            "timestamp\tcommit\tdescription\tstatus\tvs_ref\tduration_us\n"
            "2026-01-05T00:00:00Z\tmno345\tThis refreshed TSV keeps only one valid experiment row with a new result.\tkeep\t1.33\t8.8\n"
        )

        second = mem.ingest_tsv(str(tsv_path), kernel_type="gemm")
        assert second["files"] == 1
        assert second["chunks"] == 1
        assert mem.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE source_path = ?",
            (str(tsv_path.resolve()),)
        ).fetchone()[0] == 1
    finally:
        mem.close()


def test_memory_server_returns_json_for_bad_requests(tmp_path: Path) -> None:
    db_path = tmp_path / "server.db"
    mem = ResearchMemory(db_path)
    try:
        job_id = mem.create_job("server-job", "Server Job", kernel_type="gemm", execution_lane="active")
        mem.file_issue("server issue", "warning", "gemm", "foo.cu", "desc")
        assert job_id == 1
    finally:
        mem.close()

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(MEMORY_SCRIPT), "--db", str(db_path), "serve", "--port", str(port)],
        cwd=ROOT,
        env=_script_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_json(f"http://127.0.0.1:{port}/api/stats")

        status, payload = _fetch_json(f"http://127.0.0.1:{port}/api/jobs/not-an-int")
        assert status == 400
        assert payload["error"] == "Invalid job ID"

        status, payload = _fetch_json(f"http://127.0.0.1:{port}/api/issues/not-an-int")
        assert status == 400
        assert payload["error"] == "Invalid issue ID"

        status, payload = _fetch_json(f"http://127.0.0.1:{port}/api/search?q=&mode=fts")
        assert status == 200
        assert payload["results"] == []
        assert payload["count"] == 0

        status, payload = _fetch_json(f"http://127.0.0.1:{port}/api/search?q=foo&k=abc")
        assert status == 400
        assert payload["error"] == "Invalid integer for 'k': abc"

        status, payload = _fetch_json(
            f"http://127.0.0.1:{port}/api/jobs/new?name=x&title=y&factory_mode=bogus"
        )
        assert status == 400
        assert "Unknown factory mode" in payload["error"]

        status, payload = _fetch_json(f"http://127.0.0.1:{port}/api/jobs/1/state?to=bogus")
        assert status == 400
        assert "Unknown target state 'bogus'" in payload["error"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - hard cleanup fallback
            proc.kill()
            proc.wait(timeout=5)


def test_add_source_registers_runtime_ingest_source(
    stub_memory_embeddings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    docs = tmp_path / "runtime-source"
    docs.mkdir()
    (docs / "note.md").write_text(
        "# Runtime Source\n\n"
        "This document is long enough to be indexed through add_source. " * 4
    )

    monkeypatch.setattr(fb, "SOURCE_PRIORITY", [])
    monkeypatch.setattr(fb, "CODE_SOURCES", [])

    mem = ResearchMemory(tmp_path / "add-source.db")
    try:
        mem.add_source(str(docs), provenance="validated")
        stats = mem.ingest_all()
        assert stats["files"] == 1
        row = mem.conn.execute("SELECT source_file, provenance FROM documents").fetchone()
        assert row["source_file"] == str((docs / "note.md").resolve())
        assert row["provenance"] == "validated"
    finally:
        mem.close()
