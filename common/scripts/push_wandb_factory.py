#!/usr/bin/env python3
"""
Push a lightweight factory snapshot into Weights & Biases so progress is visible at wandb.redshed.ai.

Usage:
    WANDB_API_KEY=xxx python push_wandb_factory.py \
        --entity your-entity --project factory --interval 120

Use --once to send a single snapshot and exit.
The snapshot mirrors the dashboard overview: lane fill, worker state, job counts, top alerts.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = REPO_ROOT / 'common'
import sys
sys.path.insert(0, str(COMMON_DIR / 'memory'))
from factory_brain import ResearchMemory

# ---------------- lane config helpers ----------------

def _lane_entries() -> List[str]:
    override = os.environ.get("MANAGED_WORKER_SLOTS", "").strip()
    if override:
        return override.split()
    cfg_path = Path(os.environ.get("WORKER_LANES_CONFIG", str(COMMON_DIR / 'scripts' / 'worker_lanes.conf')))
    if cfg_path.is_file():
        entries = []
        for line in cfg_path.read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                entries.append(line)
        return entries
    return os.environ.get("DEFAULT_MANAGED_WORKER_SLOTS", "gemm:1 octave-gpu:2").split()

def load_managed_slots() -> Dict[str, int]:
    managed = {}
    for entry in _lane_entries():
        if ':' not in entry:
            continue
        worker, slot_value = entry.split(':', 1)
        try:
            slots = max(1, int(slot_value))
        except ValueError:
            slots = 1
        managed[worker.strip()] = slots
    if not managed:
        managed = {"gemm": 1, "octave-gpu": 2}
    return managed

# ---------------- snapshot builder ----------------

def build_snapshot():
    mem = ResearchMemory()
    jobs = mem.get_jobs()
    workers = mem.get_worker_state()
    open_msgs = mem.get_messages(status="open")
    watchdog_rows = [dict(r) for r in mem.conn.execute("SELECT * FROM watchdog_state").fetchall()]
    mem.close()

    managed = load_managed_slots()

    active_jobs = [j for j in jobs if (j.get("execution_lane") or "") == "active"]
    hopper = [j for j in jobs if (j.get("execution_lane") or "") == "hopper"]
    incubating = [j for j in jobs if (j.get("execution_lane") or "") == "incubating"]
    shipped = [j for j in jobs if j.get("state") == "shipped"]

    # lane fill
    active_counts, hopper_counts = {}, {}
    for j in active_jobs:
        key = (j.get("assigned_to") or j.get("kernel_type") or "").strip()
        active_counts[key] = active_counts.get(key, 0) + 1
    for j in hopper:
        key = (j.get("assigned_to") or j.get("kernel_type") or "").strip()
        hopper_counts[key] = hopper_counts.get(key, 0) + 1

    lane_rows = []
    total_slots = 0
    slots_filled = 0
    slots_idle = 0
    for worker, slots in managed.items():
        total_slots += slots
        used = active_counts.get(worker, 0)
        slots_filled += min(slots, used)
        idle = max(0, slots - used)
        slots_idle += idle
        lane_rows.append({
            "worker": worker,
            "slots": slots,
            "active": used,
            "idle": idle,
            "hopper": hopper_counts.get(worker, 0),
        })
    lane_rows.sort(key=lambda r: r["worker"])

    summary = {
        "active_jobs": len(active_jobs),
        "hopper_jobs": len(hopper),
        "incubating_jobs": len(incubating),
        "shipped_jobs": len(shipped),
        "workers_active": sum(1 for w in workers if w.get("live_status") == "active"),
        "workers_stale": sum(1 for w in workers if w.get("live_status") == "stale"),
        "workers_blocked": sum(1 for w in workers if (w.get("status") or "") in ("stalled", "grinding")),
        "open_messages": len(open_msgs),
        "slots_total": total_slots,
        "slots_filled": slots_filled,
        "slots_idle": slots_idle,
    }

    return {
        "summary": summary,
        "lanes": lane_rows,
        "active_jobs": active_jobs,
        "workers": workers,
        "watchdog": watchdog_rows,
        "open_messages": open_msgs,
    }

# ---------------- wandb push ----------------

def push_once(args):
    snap = build_snapshot()

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        job_type="factory_overview",
        config={"slots": load_managed_slots()},
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )

    s = snap["summary"]
    wandb.log({
        "factory/active_jobs": s["active_jobs"],
        "factory/hopper_jobs": s["hopper_jobs"],
        "factory/incubating_jobs": s["incubating_jobs"],
        "factory/shipped_jobs": s["shipped_jobs"],
        "factory/workers_active": s["workers_active"],
        "factory/workers_stale": s["workers_stale"],
        "factory/workers_blocked": s["workers_blocked"],
        "factory/open_messages": s["open_messages"],
        "factory/slots_total": s["slots_total"],
        "factory/slots_filled": s["slots_filled"],
        "factory/slots_idle": s["slots_idle"],
    })

    lane_table = wandb.Table(columns=["worker", "slots", "active", "idle", "hopper"])
    for row in snap["lanes"]:
        lane_table.add_data(row["worker"], row["slots"], row["active"], row["idle"], row["hopper"])
    wandb.log({"lanes": lane_table})

    active_table = wandb.Table(columns=["id", "title", "kernel", "lane", "priority", "state", "assigned_to", "updated_at"])
    for j in snap["active_jobs"]:
        active_table.add_data(j["id"], j.get("title"), j.get("kernel_type"), j.get("execution_lane"), j.get("priority"), j.get("state"), j.get("assigned_to"), j.get("updated_at"))
    wandb.log({"active_jobs": active_table})

    wandb.finish()


def main():
    ap = argparse.ArgumentParser(description="Push factory snapshot to W&B")
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--interval", type=int, default=180, help="Seconds between pushes")
    ap.add_argument("--once", action="store_true", help="Send one snapshot and exit")
    args = ap.parse_args()

    if args.once:
        push_once(args)
        return

    while True:
        push_once(args)
        time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
