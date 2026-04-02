from __future__ import annotations

from pathlib import Path

from common.memory.factory_brain import ResearchMemory
from ui import dashboard


def test_message_log_surfaces_relevant_messages_and_collapses_duplicates(
    stub_memory_embeddings,
    tmp_path: Path,
) -> None:
    mem = ResearchMemory(tmp_path / "dashboard.db")
    try:
        active_job = mem.create_job(
            name="active-job",
            title="Active Job",
            state="algo_building",
            execution_lane="active",
            priority="3",
        )
        hopper_job = mem.create_job(
            name="hopper-job",
            title="Hopper Job",
            state="wishlist",
            execution_lane="hopper",
            priority="3",
        )
        shipped_job = mem.create_job(
            name="shipped-job",
            title="Shipped Job",
            state="shipped",
            priority="3",
        )

        mem.create_message("watchdog", "Active warning", body="still working", job_id=active_job)
        mem.create_message("watchdog", "Active warning", body="still working", job_id=active_job)
        mem.create_message("watchdog", "Hopper info", body="queued", job_id=hopper_job)
        mem.create_message(
            "gate",
            "Critical shipped blocker",
            body="needs intervention",
            job_id=shipped_job,
            message_type="blocker",
            priority="urgent",
        )
        mem.create_message(
            "watchdog",
            "Historic shipped info",
            body="old noise",
            job_id=shipped_job,
        )

        log = dashboard._collect_relevant_message_log(mem, limit=10)
        subjects = [entry["subject"] for entry in log["entries"]]

        assert subjects == ["Critical shipped blocker", "Active warning", "Hopper info"]
        assert log["total_entries"] == 3
        assert log["raw_matches"] == 4
        assert log["deduped_repeats"] == 1

        active_entry = next(entry for entry in log["entries"] if entry["subject"] == "Active warning")
        assert active_entry["repeat_count"] == 2
        assert active_entry["execution_lane"] == "active"
        assert "Historic shipped info" not in subjects
    finally:
        mem.close()
