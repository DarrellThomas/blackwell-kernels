from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

from common.memory.factory_brain import ResearchMemory

ROOT = Path(__file__).resolve().parents[2]
MEMORY_SCRIPT = ROOT / "common/memory/factory_brain.py"
ENTRYPOINTS = (
    "common/memory/factory_brain.py",
    "common/memory/research_memory.py",
    "common/memory/memory_cli.py",
)
MODULES = (
    "common.memory.factory_brain",
    "common.memory.memory_cli",
    "common.memory.memory_config",
    "common.memory.memory_embeddings",
    "common.memory.memory_experiments",
    "common.memory.memory_helpers",
    "common.memory.memory_ingest",
    "common.memory.memory_issues",
    "common.memory.memory_jobs",
    "common.memory.memory_maintain",
    "common.memory.memory_messages",
    "common.memory.memory_search",
    "common.memory.memory_server",
    "common.memory.memory_stats",
    "common.memory.memory_workers",
    "common.memory.research_memory",
    "common.memory.generate_summaries",
)


def _script_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not current else f"{ROOT}{os.pathsep}{current}"
    if extra:
        env.update(extra)
    return env


def _run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=_script_env(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_sample_docs(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    (base / "gemm_swizzle_notes.md").write_text(
        "# GEMM Swizzle Notes\n\n"
        "## Findings\n"
        "This benchmark note describes a GEMM kernel that uses XOR swizzle in shared memory "
        "to reduce bank conflicts. Measured result: 1.23x vs_ref relative to baseline with "
        "lower memory latency and reduced long_scoreboard stalls. The kernel also uses double "
        "buffer staging and tile size tuning for better occupancy and vectorized load behavior.\n\n"
        "## Evidence\n"
        "Experiment 7 measured duration_us = 12.5 and profiler output showed shared memory "
        "bank conflicts dropping substantially.\n"
    )
    (base / "attention_barrier_note.md").write_text(
        "# Attention Barrier Note\n\n"
        "Warp divergence and synchronization overhead dominated this attention kernel. "
        "A barrier-heavy schedule increased barrier stalls and hurt throughput.\n"
    )
    return base


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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_memory_entrypoint_scripts_show_help() -> None:
    for relpath in ENTRYPOINTS:
        result = _run_script(str(ROOT / relpath), "--help")
        assert result.returncode == 0, result.stderr
        assert "Research Memory Database for blackwell-kernels" in result.stdout


def test_memory_modules_import_cleanly() -> None:
    for name in MODULES:
        mod = importlib.import_module(name)
        assert mod is not None


def test_cli_round_trip_with_stubbed_embeddings(stub_memory_embeddings, tmp_path: Path, capsys) -> None:
    fb = stub_memory_embeddings
    docs = _write_sample_docs(tmp_path / "docs")
    db_path = tmp_path / "cli.db"

    assert fb.main(["--db", str(db_path), "ingest", str(docs)]) == 0
    out = capsys.readouterr().out
    assert "Docs: 2 files" in out

    assert fb.main(["--db", str(db_path), "stats"]) == 0
    out = capsys.readouterr().out
    assert "Documents: 2" in out
    assert "Chunks:" in out
    assert "gemm: 1" in out
    assert "attention: 1" in out

    assert fb.main(["--db", str(db_path), "fts", "swizzle"]) == 0
    out = capsys.readouterr().out
    assert "GEMM Swizzle Notes" in out

    assert fb.main(["--db", str(db_path), "search", "bank", "conflict", "swizzle", "--mode", "semantic", "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "GEMM Swizzle Notes" in out

    assert fb.main(["--db", str(db_path), "quality"]) == 0
    out = capsys.readouterr().out
    assert "RESEARCH MEMORY QUALITY REPORT" in out
    assert "DUPLICATES:" in out


def test_ingest_directory_and_search_methods(stub_memory_embeddings, tmp_path: Path) -> None:
    docs = _write_sample_docs(tmp_path / "docs")
    mem = ResearchMemory(tmp_path / "ingest.db")
    try:
        stats = mem.ingest_directory(str(docs), "research", "**/*.md", False)
        assert stats["files"] == 2
        assert stats["chunks"] >= 2
        assert mem.stats()["documents"] == 2

        fts = mem.search_fts("swizzle", k=3)
        semantic = mem.search_semantic("bank conflict swizzle", k=3)
        hybrid = mem.search_hybrid("memory latency swizzle", k=3)

        assert fts
        assert semantic
        assert hybrid
        assert fts[0]["title"] == "GEMM Swizzle Notes"
        assert semantic[0]["kernel_type"] == "gemm"
        assert hybrid[0]["title"] == "GEMM Swizzle Notes"

        row = mem.conn.execute(
            "SELECT kernel_type, is_empirical, techniques FROM documents WHERE title = ?",
            ("GEMM Swizzle Notes",),
        ).fetchone()
        assert row["kernel_type"] == "gemm"
        assert row["is_empirical"] == 1
        assert "swizzle" in row["techniques"]
    finally:
        mem.close()


def test_fresh_db_supports_core_refactored_methods(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    mem = ResearchMemory(db_path)
    try:
        stats = mem.stats()
        assert stats["documents"] == 0
        assert stats["experiments_total"] == 0

        job_id = mem.create_job(
            "job1",
            "Job 1",
            kernel_type="gemm",
            execution_lane="active",
            factory_mode="fixed_shape_kernel",
            optimization_scope="algorithmic",
            reference_label="ref",
        )
        message_id = mem.create_message("ops", "hello", job_id=job_id)
        issue_id = mem.file_issue("bug", "warning", "gemm", "foo.cu", "desc")
        experiment = mem.record_experiment("gemm", "keep", "first keep", vs_ref=1.2, duration_us=10.0)
        worker_kernel = mem.worker_heartbeat("gemm", "testing", job_id=job_id)
        refresh = mem.refresh_worker_state()
        watchdog = mem.touch_watchdog_state("watchdog", "ok")

        job = mem.get_job(job_id)
        assert job["execution_lane"] == "active"
        assert job["factory_mode"] == "fixed_shape_kernel"
        assert mem.get_message(message_id)["subject"] == "hello"
        assert mem.get_message(message_id)["job_id"] == job_id
        assert mem.get_issues()[0]["id"] == issue_id
        assert experiment["kernel_type"] == "gemm"
        assert experiment["status"] == "keep"
        assert worker_kernel == "gemm"
        assert "gemm" in refresh
        assert mem.get_worker_state("gemm")[0]["kernel_type"] == "gemm"
        assert watchdog["name"] == "watchdog"
        assert mem.get_watchdog_state("watchdog")["last_status"] == "ok"
    finally:
        mem.close()


def test_memory_server_stats_jobs_issues_and_workers_endpoints(tmp_path: Path) -> None:
    db_path = tmp_path / "server.db"
    mem = ResearchMemory(db_path)
    try:
        job_id = mem.create_job("server-job", "Server Job", kernel_type="gemm", execution_lane="active")
        mem.file_issue("server bug", "warning", "gemm", "foo.cu", "desc")
        mem.worker_heartbeat("gemm", "server-test", job_id=job_id)
        mem.refresh_worker_state()
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
        stats = _wait_for_json(f"http://127.0.0.1:{port}/api/stats")
        jobs = _wait_for_json(f"http://127.0.0.1:{port}/api/jobs")
        issues = _wait_for_json(f"http://127.0.0.1:{port}/api/issues")
        workers = _wait_for_json(f"http://127.0.0.1:{port}/api/workers")

        assert stats["jobs_total"] == 1
        assert jobs["count"] == 1
        assert jobs["jobs"][0]["name"] == "server-job"
        assert issues["count"] == 1
        assert issues["issues"][0]["title"] == "server bug"
        assert workers["count"] >= 1
        assert any(worker["kernel_type"] == "gemm" for worker in workers["workers"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - hard cleanup fallback
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.integration
def test_real_embedding_cli_smoke(tmp_path: Path) -> None:
    if os.environ.get("FB_RUN_REAL_EMBEDDING") != "1":
        pytest.skip("set FB_RUN_REAL_EMBEDDING=1 to run the real-model smoke test")

    docs = _write_sample_docs(tmp_path / "docs")
    db_path = tmp_path / "real.db"

    ingest = _run_script(str(MEMORY_SCRIPT), "--db", str(db_path), "ingest", str(docs))
    assert ingest.returncode == 0, ingest.stderr
    assert "Docs: 2 files" in ingest.stdout

    semantic = _run_script(
        str(MEMORY_SCRIPT), "--db", str(db_path), "search", "bank", "conflict", "swizzle",
        "--mode", "semantic", "-k", "1",
    )
    assert semantic.returncode == 0, semantic.stderr
    assert "GEMM Swizzle Notes" in semantic.stdout

    quality = _run_script(str(MEMORY_SCRIPT), "--db", str(db_path), "quality")
    assert quality.returncode == 0, quality.stderr
    assert "RESEARCH MEMORY QUALITY REPORT" in quality.stdout
