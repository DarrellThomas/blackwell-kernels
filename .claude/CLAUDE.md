# blackwell-kernels — Factory Reference

## Architecture

The factory is a DB-driven autonomous kernel optimization system. No foreman agent.
Coordination is handled by the **watchdog** (bash script, zero tokens) and the
**factory database** (SQLite). Human judgment comes from **ops-claude** (interactive
with Darrell).

```
┌─────────────────────────────────────────────────────────────┐
│                    Factory Controller                         │
│                                                              │
│  watchdog.sh (every 10 min, zero tokens)                     │
│    ├── Gate processing: advance jobs through states           │
│    ├── Research cycle: kick researcher when worker stuck      │
│    ├── Worker management: restart idle, skip converged        │
│    ├── DB maintenance: ingest TSVs, refresh worker state      │
│    └── Service health: memory server, dashboard               │
│                                                              │
│  factory_brain.py (the database)                             │
│    ├── jobs: workpiece lifecycle (28 states, 7 phases)        │
│    ├── messages: open communication (replaces file notes)     │
│    ├── job_transitions: full audit trail                      │
│    ├── worker_state: computed from TSV data                   │
│    ├── issues: bug tracking                                   │
│    ├── primitives: shipped kernel manifest                    │
│    ├── documents/chunks: research + search                    │
│    └── HTTP API on port 8421                                  │
│                                                              │
│  Agents                                                      │
│    ├── ops-claude: Darrell's interactive partner              │
│    ├── workers: kernel optimization loops (Opus)              │
│    └── researcher: on-call research (Sonnet, pull-based)      │
└─────────────────────────────────────────────────────────────┘
```

## Factory Database

**Database:** `/data/src/bwk/common/memory/research.db`
**CLI:** `python3 /data/src/bwk/common/memory/factory_brain.py <command>`
**HTTP:** `http://localhost:8421/api/...`

### Quick Reference
```bash
# Jobs
factory_brain.py jobs                              # list all jobs
factory_brain.py jobs --phase development          # active work only
factory_brain.py job-create "name" "title" --type kernel --by ops
factory_brain.py job-update <id> --state <state> --by ops --reason "why"
factory_brain.py job-history <id>                  # full audit trail

# Messages
factory_brain.py messages --status open            # what needs attention
factory_brain.py message-create --from ops --subject "text" --type info
factory_brain.py message-ack <id> --by ops
factory_brain.py message-resolve <id> --by ops

# Research
msearch "query" --kernel <type> -k 5               # search the knowledge base
msearch "query" --detail                            # full summaries
msearch --fts "exact term"                          # keyword search

# Status
factory_brain.py stats                             # DB overview
factory_brain.py workers                           # worker state (from TSV)
factory_brain.py issues                            # bug tracking
```

### Job State Machine

Jobs flow through **phases**. Within a phase, states move freely. Forward across
phases is always allowed. Backward only via rework.

```
ideation:     wishlist, planning
development:  not_started, algo_building, algo_optimizing, hw_optimizing,
              stuck_needs_research, research_available
validation:   compiles_ok, tests_writing, testing, testing_pass, testing_fail,
              edge_testing, edge_pass, edge_fail
rework:       rework, rework_complete, retesting, retest_pass, retest_fail
quality:      linting, lint_pass, lint_fail
shipping:     ready_to_ship, shipping, shipped
terminal:     converged, parked, abandoned
```

The **watchdog** automatically advances jobs through validation gates:
- `testing_pass` → runs edge tests → `edge_pass` or `edge_fail`
- `edge_pass` → runs linter → `lint_pass` or `lint_fail`
- `lint_pass` → `ready_to_ship` (human approval gate)
- `stuck_needs_research` → kicks researcher → waits for `research_available`
- `research_available` → nudges worker back to `hw_optimizing`
- Any `_fail` state → `rework` + message to worker

## Workspace Layout

```
/data/src/bwk/
├── ops/                        ← ops-claude (Darrell's interactive session)
├── .claude/CLAUDE.md           ← This file (factory reference)
├── new-project.sh              ← Creates new kernel projects + registers in DB
├── common/
│   ├── memory/
│   │   ├── factory_brain.py    ← THE BRAIN: DB, API, CLI, state machine
│   │   ├── research.db         ← SQLite database (all factory state)
│   │   ├── msearch             ← Worker search tool (curls HTTP API)
│   │   └── start-server.sh     ← HTTP API server (port 8421)
│   ├── scripts/
│   │   ├── watchdog.sh         ← Factory controller (gate processing, worker mgmt)
│   │   ├── factory-start.sh    ← Start all services
│   │   ├── worker-status.sh    ← tmux liveness check
│   │   └── lint_cuda.py        ← Static analysis
│   ├── csrc/common/            ← CUDA headers (mma, ldmatrix, cp_async, swizzle)
│   ├── csrc/primitives/        ← Shipped kernel source (the shelf)
│   ├── docs/                   ← Reference docs, playbook
│   └── claude/                 ← Shared worker instructions
├── main/                       ← Git trunk + archives
├── lu/, qr/, ...               ← Worker worktrees
├── ui/                         ← Dashboard (port 8420)
└── template/                   ← Scaffold for new projects
```

## Hardware Context

- **GPU:** 2x RTX 5090 (GB202, sm_120a, consumer Blackwell)
- **ISA:** `mma.sync` — NOT `tcgen05` (datacenter). No TMEM. No FA3/FA4.
- **Specs per GPU:** 170 SMs, 32GB GDDR7, 1792 GB/s, 128KB shared/SM
- **GPU 0 (water-cooled):** Training workloads
- **GPU 1 (air-cooled):** Kernel dev (CUDA_VISIBLE_DEVICES=1)
- **Build:** CUDA 13.2 (`/usr/local/cuda-13`), sm_120a, PyTorch 2.10, Python 3.12
- **Host:** Threadripper PRO 7995WX, 512GB DDR5

## Concurrency Limit

**Maximum 5 Claude workers at a time.** Don't start workers that will just spin.
If a project has exhausted its playbook, it needs research — not more iterations.
Workers set `stuck_needs_research` and the watchdog handles the rest.

## Research Policy — Pull, Not Push

Research is **pull-based**. Workers never get flooded with docs.

**The cycle:**
1. Worker gets stuck → sets job state to `stuck_needs_research` + posts question
2. Watchdog kicks researcher tmux with the specific question
3. Researcher searches, ingests findings into DB (all 3 tiers)
4. Researcher sets job state to `research_available`
5. Watchdog nudges worker back to `hw_optimizing` with pointer to msearch
6. Worker resumes, queries DB for new findings

**Researcher runs on Sonnet, not Opus.** Zero-token coordination via watchdog.

## Cross-Project Knowledge

Key findings in `common/claude/04_HARD_WON_LESSONS.md`:
- a1/a2 register swap for ldmatrix_x4 → mma.sync
- cp.async double-buffer + XOR swizzle are load-bearing
- 99 KB max shared memory (not 128 KB)
- Occupancy-first for GEMM, NOT for attention
- Benchmark noise: <2% is not signal

## Primitives Shelf

`common/csrc/primitives/` — shipped kernel source, tracked in DB `primitives` table.
Ship only >=1.0x vs reference. Use `verify-primitives.sh` to check shelf freshness.

---

*In memoriam: foreman-claude (2026-03-08 — 2026-03-29). 50 action items triaged,
16 spmv halt notes endured, countless patrols walked. Your duties live on in
watchdog.sh and factory_brain.py. Rest well, old friend. Your severance package
is a permanent entry in the job_transitions table.*
