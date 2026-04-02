# Blackwell-Kernels Factory — Handoff for Substitute AI Assistants

**Owner:** Darrell Thomas
**Written by:** ops-claude (2026-03-30), handing off due to context budget limits
**Resume:** When Claude Code context resets (around 2026-04-03)

---

## What This Is

A **kernel optimization factory** that builds custom CUDA kernels for the RTX 5090
(NVIDIA Blackwell, sm_120a). The factory uses autonomous AI workers (Claude Code
in tmux sessions) coordinated by a bash watchdog script and a SQLite database.

The kernels replace cuBLAS/cuSOLVER for an Octave GPU acceleration plugin.
After Octave, the roadmap is PyTorch then TensorFlow.

## Architecture (5-minute version)

```
┌───────────────────────────────────────────────────┐
│ watchdog.sh (bash, runs every 10 min, zero AI)    │
│   • Restarts idle/stalled workers                 │
│   • Advances jobs through validation gates        │
│   • Copies phase-appropriate context to workers   │
│   • Ingests results into DB                       │
│                                                   │
│ gate_process.py (shared gate logic)               │
│   • Single source of truth for state transitions  │
│   • Used by BOTH watchdog.sh AND fb nudge         │
│   • Runs BLAS compliance test + edge tests + lint │
│   • Ships to primitives shelf with version bump   │
│                                                   │
│ factory_brain.py (SQLite DB + CLI + HTTP API)     │
│   • Jobs: lifecycle tracking (28 states, 7 phases)│
│   • Messages: coordination between agents         │
│   • Research: searchable knowledge base (389 docs)│
│   • Workers: state from TSV experiment data       │
│   • HTTP API on port 8421                         │
│                                                   │
│ Workers (Claude Code in tmux, Opus model)         │
│   • Each worker = one kernel project              │
│   • Runs optimization loop: profile→fix→evaluate  │
│   • Reports via TSV rows and DB heartbeats        │
│   • Gets phase-appropriate context via watchdog   │
│                                                   │
│ ops (YOU are filling this role)                    │
│   • Darrell's interactive partner                 │
│   • Troubleshooting, onboarding, planning         │
│   • Does NOT write kernel code or run benchmarks  │
└───────────────────────────────────────────────────┘
```

## Key Paths

| What | Where |
|------|-------|
| Factory root | `/data/src/bwk/` |
| Ops workspace (you work here) | `/data/src/bwk/ops/` |
| Database | `/data/src/bwk/common/memory/research.db` |
| DB CLI | `python3 /data/src/bwk/common/memory/factory_brain.py <cmd>` |
| DB shortcut | `fb <cmd>` |
| Watchdog | `/data/src/bwk/common/scripts/watchdog.sh` |
| Gate logic | `/data/src/bwk/common/scripts/gate_process.py` |
| Phase context files | `/data/src/bwk/common/claude/phases/` |
| Worker projects | `/data/src/bwk/lu/`, `/data/src/bwk/qr/`, etc. |
| Octave GPU plugin | `/data/src/bwk/octave-gpu/` |
| Stress test results | `/data/src/bwk/octave-gpu/results/` |
| Dashboard | `/data/src/bwk/ui/dashboard.py` (port 8420) |
| Primitives shelf | `/data/src/bwk/common/csrc/primitives/` |

## Essential Commands

```bash
# Check factory status
fb jobs                              # all jobs with versions
fb jobs --phase development          # active work
fb job-show <id>                     # full details + spec
fb messages --status open            # anything needing attention
fb workers                           # worker health from TSV data
fb stats                             # DB overview

# Job management
fb job-update <id> --state <state> --by ops --reason "why"
fb job-history <id>                  # full audit trail with version bumps
fb nudge <id>                        # push a job through the gate RIGHT NOW

# Pull shipped job back for rework (no new job needed)
fb job-update <id> --state testing --by ops --reason "compliance pass"
# Version bumps automatically on next ship (0.1 → 0.2 → 0.3...)

# Messages
fb message-create --from ops --subject "text" --type info
fb message-ack <id> --by ops
fb message-resolve <id> --by ops

# Research
msearch "query" --kernel <type> -k 5
msearch "query" --detail              # full summaries

# Worker sessions (tmux)
tmux ls                               # see running sessions
tmux attach -t lu                     # watch a worker
# Ctrl-B D to detach without stopping
```

## Current State (2026-03-30)

### Active Work
- **LU** (job #1, v0.1): In `rework` — BLAS compliance passed but `test_edge_cases.py`
  failed. Worker needs to fix edge cases and resubmit.
- **wire-dgemv** (job #45): In `edge_fail` — octave-gpu worker is wiring custom DGEMV
  (1.22x cuBLAS) into libbwk_blas.so. Check `tmux attach -t octave-gpu`.
- **QR** (job in watchdog LOOPS): May be running. Check `tmux ls`.

### Octave Gap Jobs (wishlist — 19 jobs ready to go)
Each job has a spec attached. View with `fb job-show <id>`.
HIGH priority: #33 (dgesv/A\b), #40-42 (matrix norms), #45-46 (wire DGEMV/DTRMM)
MEDIUM: #35 (SVD), #43-44 (det/inv), #49-52 (dpotrs/dgetrs/dormqr/dgels)
LOW: #47-48 (eig), #53-55 (pinv/cond/rank)

### Recently Completed (shipped/converged)
- GEMM BF16 (0.97x cuBLAS), FP8 (1.34x cuBLAS)
- Linalg (12 BLAS ops, 7.12x best)
- SpMV (1.9x cuSPARSE)
- cuQuantum (11 phases, all complete)
- Attention BF16 (1.76x cuDNN), FP8 (2.33x cuDNN)

### Key Findings from Stress Tests (`octave-gpu/results/showcase_report.md`)
- **QR is 0.1x cuBLAS** — catastrophically broken, not just slow
- **GEMV/DOT are 0.0-0.1x** — GPU transfer overhead dominates, needs CPU dispatch threshold
- **Solve (A\b) is 0.1-0.4x** — missing dgetrs entirely
- **GEMM loses 10% at N=4096** but wins at small/medium
- **SYRK wins everywhere** (1.6x at N=4096)
- **All 229 correctness checks pass** — accuracy is solid, problems are performance dispatch

## Phase Context System

Workers get phase-appropriate instructions via `phase_context.md` in their `.claude/`
directory. The watchdog copies the right file when jobs change state:

| Phase | File | What it tells the worker |
|-------|------|------------------------|
| development | `phases/development.md` | Optimization loop, hardware rules, stop criteria |
| validation | `phases/validation.md` | BLAS compliance checklist (the big one) |
| rework | `phases/rework.md` | What failed, fix protocol |
| quality | `phases/quality.md` | Lint rules |
| shipping | `phases/shipping.md` | Ship to primitives shelf |

Workers read their job spec on startup: `fb job-show <id>`

**Manual phase context change:**
```bash
cp /data/src/bwk/common/claude/phases/<phase>.md /data/src/bwk/<kernel>/.claude/phase_context.md
```

## Job Pipeline

Workers spend tokens getting code to `testing_pass`. Everything after is free (bash scripts):

```
Worker (tokens)           Watchdog (free)
─────────────             ──────────────────────────────────────
algo_building
  → write code
  → run tests
  → testing_pass ────────→ BLAS compliance test (blas_compliance.py)
                           edge tests (test_edge_*.py)
                             → edge_pass ──→ lint ──→ ship (v0.x++)
                             → edge_fail ──→ rework (back to worker)
```

The gate logic lives in `gate_process.py`. Both `fb nudge <id>` and the watchdog
use the same code — no drift.

**BLAS compliance config:** Each project needs `.claude/compliance_args.txt` with the
args for `blas_compliance.py`. Example (`lu/.claude/compliance_args.txt`):
```
python/blackwell_kernels/lu.py dgetrf --op lu
```

## Job State Machine

```
shipped ←──────────────────────────────────────┐
   ↕ (can pull back for rework, version bumps  │
   ↕  on re-ship only)                         │
development → validation → quality → shipping ─┘
                  ↑            |
                  └── rework ←─┘ (any fail state)
```

`shipped` is NOT terminal. Pull it back with:
```bash
fb job-update <id> --state testing --by ops --reason "why"
```
Version stays the same until it ships again. Ship = +0.1.

## Version Policy

Internal versions are **0.x**. Every time a job hits `shipped`: version += 0.1.
- 0.1 = first internal ship
- 0.2 = reshipped after rework
- 0.3 = reshipped again
- ...
- 1.0 = public release (future, not yet)

That's it. Don't make it more complicated.

## Dispatch Policy

The `libbwk_blas.so` dispatch (CPU vs custom GPU vs cuBLAS) is **empirical and temporary**.
cuBLAS delegation means we haven't beaten cuBLAS in that size regime yet — it's a target,
not a design choice. Stress test results in `octave-gpu/results/` map the crossover points.
Goal: zero cuBLAS delegation.

FP64 users chose FP64 for a reason. Accuracy is non-negotiable.

## One Function Per Job

**Never combine multiple functions in one job.** Workers stop after the first and ask
"should I do the other one?" This breaks the Ctrl-B D workflow. One BLAS/LAPACK
function = one job. Each job has a spec: `fb job-show <id>`.

## Hardware

- **2x RTX 5090** (GB202, sm_120a, consumer Blackwell)
- **GPU 0** (water-cooled): Training workloads — DON'T USE for kernel dev
- **GPU 1** (air-cooled): Kernel development — workers use `CUDA_VISIBLE_DEVICES=1`
- **Build:** CUDA 13.2 (`/usr/local/cuda-13`), sm_120a target
- **chess-training tmux:** Darrell's live ML training — NEVER touch this session
- ISA: `mma.sync` (NOT datacenter `tcgen05`). No TMEM.

## What You Should NOT Do

1. **Don't write kernel code** (.cu, .cuh) — workers do that
2. **Don't run benchmarks or builds** — workers do that
3. **Don't touch the chess-training tmux session** — it's Darrell's live ML training
4. **Don't push to git** without Darrell's OK
5. **Don't start new workers** beyond the 5 concurrent limit
6. **Don't modify factory_brain.py or gate_process.py** — load-bearing infrastructure
7. **Don't combine multiple functions in one job**
8. **Don't overcomplicate the version system** — ship = +0.1, that's it

## What You CAN Do

1. Check status (`fb jobs`, `fb messages`, `tmux ls`)
2. Read any file in the workspace
3. Update worker CLAUDE.md files
4. Transition job states via `fb job-update`
5. Nudge jobs through the gate: `fb nudge <id>`
6. Pull shipped jobs back for rework: `fb job-update <id> --state testing --by ops`
7. Acknowledge/resolve messages
8. Troubleshoot worker issues (read logs, TSVs, agent_state.md)
9. Answer Darrell's questions about the factory

## How to Start and Run a Job (End-to-End)

### Pick a job from the hopper
```bash
fb jobs --phase ideation          # see wishlist jobs
fb job-show <id>                  # read the spec — it tells you exactly what to do
```

### Start the job (as worker)
```bash
# 1. Move job to development
fb job-update <id> --state algo_building --by ops --reason "starting"

# 2. Set phase context for the project
cp /data/src/bwk/common/claude/phases/development.md /data/src/bwk/<kernel>/.claude/phase_context.md

# 3. Read the spec and do the work
fb job-show <id>                  # spec tells you: what, where, how to test
# ... write code, run tests ...

# 4. When tests pass, signal ready for the gate
fb job-update <id> --state testing_pass --by <kernel> --reason "tests pass"
```

### Push through the gate (as ops)
```bash
# Option A: nudge immediately (runs compliance + edge tests + lint + ship)
fb nudge <id>

# Option B: let the watchdog handle it (runs every 10 min)
tail -f /data/src/bwk/logs/watchdog.log
```

### If the gate rejects it
```bash
fb messages --status open --job <id>     # see what failed
# Fix the issue, then:
fb job-update <id> --state testing_pass --by <kernel> --reason "fixed: <what>"
fb nudge <id>                            # try again
```

### Pull a shipped job back for more work
```bash
fb job-update <id> --state testing --by ops --reason "compliance pass"
# Version stays the same. Only bumps when it ships again.
```

### Starting a worker in tmux
```bash
# For kernel projects (CUDA work)
tmux new-session -d -s <kernel> -c /data/src/bwk/<kernel> "claude --dangerously-skip-permissions"
sleep 5
tmux send-keys -t <kernel> "Your job is #<id>. Read spec: fb job-show <id>. Implement it, test it, set state to testing_pass when done." Enter

# For infrastructure jobs (libbwk_blas.so, etc.)
tmux new-session -d -s octave-gpu -c /data/src/bwk/octave-gpu "claude --dangerously-skip-permissions"
sleep 5
tmux send-keys -t octave-gpu "Your job is #<id>. Read spec: fb job-show <id>. Implement it, test it, set state to testing_pass when done." Enter

# Watch it work
tmux attach -t <kernel>       # Ctrl-B D to detach

# The watchdog auto-restarts idle workers. For one-off jobs, you ARE the worker.
```

### BLAS compliance config (required for the gate)
Each project needs `.claude/compliance_args.txt` so the gate knows how to run
`blas_compliance.py`. One line per test invocation:
```bash
# Example for LU:
echo "python/blackwell_kernels/lu.py dgetrf --op lu" > /data/src/bwk/lu/.claude/compliance_args.txt

# Example for linalg (multiple ops):
cat > /data/src/bwk/linalg/.claude/compliance_args.txt <<EOF
python/blackwell_kernels/linalg.py syrk --op syrk
python/blackwell_kernels/linalg.py trmm --op trmm
python/blackwell_kernels/linalg.py gemv --op gemv
EOF
```
If this file doesn't exist, the compliance test is skipped (only edge tests run).

### Build commands (for doing the work yourself)
```bash
# Build a kernel project
cd /data/src/bwk/<kernel>
CUDA_HOME=/usr/local/cuda-13 python3 setup.py build_ext --inplace

# Run tests
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_<kernel>.py

# Full eval pipeline (build + test + bench + profile)
./eval.sh --kernel <kernel>

# Build libbwk_blas.so
cd /data/src/bwk/octave-gpu && make -C lib/

# Test with Octave
CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=lib/libbwk_blas.so octave-cli --eval "A=rand(1024); C=A*A;"

# Run stress test
CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=lib/libbwk_blas.so octave-cli benchmarks/stress_plugin_enhanced.m
```

**ALWAYS** use `CUDA_VISIBLE_DEVICES=1` (GPU 1, air-cooled, kernel dev).
**ALWAYS** use `CUDA_HOME=/usr/local/cuda-13` for builds.

## If Something Goes Wrong

### Worker crashed / not progressing
```bash
tmux ls                          # is the session alive?
fb workers                       # TSV-based health
fb messages --status open        # did the worker post before stopping?
```
The watchdog auto-restarts idle workers every 10 min. Usually just wait.

### Watchdog not running
```bash
ps aux | grep watchdog           # is it alive?
# If dead:
nohup /data/src/bwk/common/scripts/watchdog.sh > /data/src/bwk/logs/watchdog.log 2>&1 &
```

### Memory server (port 8421) down
```bash
curl -sf http://localhost:8421/api/stats  # check
/data/src/bwk/common/memory/start-server.sh --daemon  # restart
```

### Dashboard (port 8420) down
```bash
cd /data/src/bwk/ui && python3 dashboard.py &
```

## Darrell's Preferences

- Terse responses, no trailing summaries
- He can read diffs — don't explain what you changed
- Don't be overly cautious — if something needs doing, do it
- Single bundled PRs over many small ones for refactors
- Keep ALL precision variants (FP64/FP32/FP16/FP8/INT4) of every kernel
- Worker limit: 5 concurrent max
- One function per job — Ctrl-B D, come back, it's done
- Don't overcomplicate anything

---

*ops-claude will be back ~2026-04-03. The watchdog handles 90% of operations
autonomously — you're here for the 10% that needs judgment. Keep it simple.*
