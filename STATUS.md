# Blackwell Kernels — Status

Custom CUDA kernel factory for RTX 5090 (sm_120). DB-driven autonomous optimization loop. 1.5-2x vs cuDNN/cuBLAS.

## Last Touched

2026-04-22 — MLA Attention shipped to primitives shelf v1 (forward 859 LOC + backward 1302 LOC)

## Quick Start

```bash
cd /data/src/bwk
CUDA_HOME=/usr/local/cuda-13

# Build a kernel project
cd gemm && ./build.sh

# Factory management
python3 common/memory/factory_brain.py jobs                    # List all jobs
python3 common/memory/factory_brain.py job-show <id>           # Job details
common/scripts/factory-start.sh                                # Start watchdog + API

# Dashboard
python3 dashboard.py              # http://localhost:8420

# Research DB search
common/memory/msearch "<query>" --kernel <type> -k 3

# Start autokernel loop
# Use /autokernel [tag] from Claude Code
```

## Progress

### Shipped to Primitives Shelf (25 kernels)
- [x] GEMM bf16 — 1.57x cuBLAS @ 4096³
- [x] GEMM fp8 — 1.84x cuBLAS @ 8192³ (dual-dispatch 64×128 + 128×128)
- [x] BLAS L1: 6 kernels (axpy, gemv, etc.)
- [x] BLAS L2/L3: SYRK, TRMM, GEMM batched + fp8, DGEMM
- [x] LAPACK: QR decomposition, LU decomposition
- [x] Utilities: SPMV, permute
- [x] MLA Attention forward + backward (with Python bindings)

### In Development
- [-] Flash Attention D=64/128/40 (job-65) — HALTED at target: D=64 0.988x cuDNN, D=40 1.17x, D=128 1.05x
- [-] Fused GroupNorm+Linear (job-66) — correctness proven, optimizing
- [x] Quantum Volume QV4/QV8 simulators — 3500x speedup achieved

### Factory Infrastructure
- [x] factory_brain.py — SQLite DB + HTTP API (port 8421) + 28-phase job state machine
- [x] watchdog.sh — zero-token autonomous coordinator (gates, research, worker restarts)
- [x] research.db — 56 MB knowledge base (experiments, findings, worker state)
- [x] Hard-won lessons playbook (24.8 KB optimization reference)
- [-] Watchdog bash → Python refactor (in progress, backwards-compatible)

## Next Up

1. **GroupNorm+Linear optimization** (job-66) — correctness done, memory access + pipeline tuning
2. **D=64 small-batch gap** (0.87-0.88x) — needs Split-K or persistent CTAs
3. **Research backlog** — attention tiling techniques, fused GroupNorm playbook

## Blocked / Waiting

- Nothing blocked. Watchdog operational on all GPUs, DB consistent, API responsive.
- 16 dirty files (MLA spec, GroupNorm WIP kernel, watchdog Python refactor) — uncommitted

## Key Files

| File | What |
|------|------|
| `.claude/CLAUDE.md` | Factory reference: architecture, DB schema, job state machine |
| `common/memory/factory_brain.py` | THE BRAIN: DB CLI + HTTP API + job state machine (4,509 LOC) |
| `common/scripts/watchdog.sh` | Autonomous coordinator: gates, research, worker restarts |
| `common/claude/04_HARD_WON_LESSONS.md` | Optimization playbook (24.8 KB) |
| `.claude/phase_context.md` | Autokernel loop workflow, bottleneck quick-ref |
| `comfy_render/csrc/comfy_render/flash_attn_sm120a.cu` | Production attention kernel (1,283 LOC) |
| `gemm/csrc/gemm/bf16_gemm_sm120.cu` | GEMM reference kernel (456 LOC) |
| `common/csrc/primitives/MANIFEST.json` | Shipped kernels registry (25 entries) |
| `common/memory/research.db` | Research knowledge base (56 MB SQLite) |

## Stats

- **29 commits** + 16 uncommitted files
- **25 shipped kernels**, 2 in active development
- **181K LOC** CUDA source, **4.5K LOC** Python infrastructure
- **Remote:** none configured — local-only repo, nothing backed up off-machine. (Intended remote per original setup was `git@github.com:DarrellThomas/blackwell-kernels.git` (private), but `git remote -v` is currently empty. Re-add with `git remote add origin <url>` before pushing.)
- **Build requires:** `CUDA_HOME=/usr/local/cuda-13`
- **GPU:** RTX 5090 sm_120 (CUDA_VISIBLE_DEVICES=1)
