# autokernel — Quick Reference

Automated, profile-driven optimization loop for CUDA kernels on RTX 5090 (sm_120).

---

## How It Works

```
    ┌─────────┐
    │ PROFILE │ ← ncu tells us the #1 bottleneck
    └────┬────┘
         ▼
    ┌─────────┐
    │ ANALYZE │ ← map bottleneck to optimization strategy
    └────┬────┘
         ▼
    ┌─────────┐
    │  CODE   │ ← modify kernel source (one focused change)
    └────┬────┘
         ▼
    ┌─────────┐
    │  BUILD  │ ← compile CUDA kernel
    └────┬────┘
         ▼
    ┌─────────┐
    │  TEST   │ ← 6 correctness tests must pass
    └────┬────┘
         ▼
    ┌─────────┐
    │  BENCH  │ ← measure duration, compare to cuDNN SDPA
    └────┬────┘
         ▼
    ┌──────────────┐
    │ KEEP/DISCARD │ ← faster + correct → keep. else → git reset
    └──────┬───────┘
           │
           └──→ loop back to PROFILE
```

Each iteration takes ~60-90 seconds. Over 8-10 hours you'll get ~400-500 experiments.

The profiler drives every decision — no guessing, no hand-tuned priority list.

---

## Starting a Run

### 1. Start the dashboard (separate terminal)

```bash
cd /data/src/blackwell-kernels
python3 dashboard.py
```

Open **http://localhost:8420** in a browser. Leave it open — it auto-refreshes every 30 seconds.

### 2. Start the optimization loop

In a Claude Code session in the `blackwell-kernels` directory:

```
Follow the instructions in program.md to start an autokernel run.
```

The agent will:
- Create a branch (`autokernel/<tag>`)
- Run a baseline eval
- Begin the autonomous loop
- Stop when it hits convergence (or you interrupt it)

### 3. Walk away

Check the dashboard whenever you want. That's it.

---

## What to Look At

### Dashboard (http://localhost:8420)

| Section | What It Shows |
|---------|---------------|
| **Top cards** | Iteration count, best duration, vs SDPA ratio, SM throughput, current bottleneck |
| **Duration chart** | Kernel speed over time — green dots = kept, red = discarded |
| **vs SDPA chart** | Speedup ratio with 1.5x target line |
| **Stall bar chart** | Current bottleneck breakdown |
| **Stall evolution** | How bottlenecks shift over iterations — **this is the key chart** for spotting plateaus |
| **Experiment log** | Every experiment, newest first, with descriptions |

### Key metrics

| Metric | Current | Target | Ceiling |
|--------|---------|--------|---------|
| vs cuDNN SDPA | ~1.33x | >1.50x | ~1.0x (cuDNN is 97% peak) |
| SM Throughput | ~49% | >70% | 97% (cuDNN) |
| Duration (D=64 N=2048) | ~115 us | <90 us | depends on throughput |

### Files you can check

```bash
# Live results (tab-separated)
cat results.tsv

# Specific iteration's full eval output
cat logs/eval_42.log

# Current git state (what the agent has changed)
git log --oneline -20
git diff HEAD~1
```

---

## When It Stops

The agent stops when it detects convergence:

1. **Bottleneck oscillation** — same two stalls trading #1 for 3+ rounds
2. **Diminishing returns** — last 5 kept experiments each < 2% improvement
3. **Plateau** — structural change attempted and failed to break through

It writes a summary to `results_summary.md` and notifies you.

You can also interrupt it manually anytime (Ctrl+C the Claude session). All progress is committed to git on the branch.

---

## Files

| File | Purpose | Modified by agent? |
|------|---------|-------------------|
| `program.md` | Agent instructions (the "skill") | No |
| `eval.sh` | Build → test → bench → profile pipeline | No |
| `dashboard.py` | Live web dashboard | No |
| `results.tsv` | Experiment log | Yes (append only) |
| `logs/eval_N.log` | Per-iteration eval output | Yes (created) |
| `csrc/attention/flash_attn_v2_sm120.cu` | The kernel being optimized | Yes |
| `csrc/common/*.cuh` | Shared CUDA primitives | Sometimes |
| `.claude/04_HARD_WON_LESSONS.md` | Guardrails + discoveries — wins, failures, structural insights | Yes (append) |
| `docs/nvidia_blackwell_tuning_guide_sm120.md` | sm_120 hardware specs | No |

---

## Beyond Flash Attention — Generalizing the Loop

The autokernel loop is designed for one kernel right now (flash attention v2), but the infrastructure is kernel-agnostic. The plan is to build a library of optimized sm_120 kernels, each going through the same two-phase process.

### Phase 1: Get It Correct (human-guided)

Design and implement a new kernel to the point where it produces correct results:

1. **Design** — choose tile sizes, data flow, MMA mapping, shared memory layout
2. **Implement** — write a functional kernel, not a fast one
3. **Test** — build correctness tests against a PyTorch reference
4. **Pass** — all tests green, output matches reference within BF16 tolerance

This is where accumulated knowledge pays off. The primitives in `csrc/common/` (mma wrappers, ldmatrix, cp.async, swizzle) are reusable. Hard-won lessons (a1/a2 register swap, sm_120 ISA constraints, bank conflict patterns) apply to every kernel. Each new kernel starts further ahead because Phase 1 incorporates patterns that took weeks to discover on earlier kernels.

### Phase 2: Make It Fast (autokernel loop, autonomous)

Hand the correct-but-slow kernel to the loop:

1. Write a test script, bench script, and profile script for the new kernel
2. Point eval.sh at them
3. `/autokernel` — walk away
4. Come back to a converged, optimized kernel

### What generalizing requires (after flash attention converges)

**Parameterize eval.sh** — each kernel needs its own test, bench, and profile scripts. A config per kernel:

```
csrc/attention/eval.yaml     →  test: test_attention.py, bench: bench_attention.py, ...
csrc/gemm/eval.yaml          →  test: test_gemm.py, bench: bench_gemm.py, ...
```

Invoke as `./eval.sh --kernel gemm` instead of hardcoding flash attention.

**Split the lessons** — `04_HARD_WON_LESSONS.md` currently mixes universal sm_120 facts (register layout, ISA constraints) with flash-attention-specific decisions (P conversion strategy, Q reuse pattern). Split into:

- `04_HARD_WON_LESSONS.md` — universal sm_120 kernel development lessons
- Per-kernel convergence summaries — what worked, what didn't, final bottleneck state

**Create a design playbook** — `docs/kernel_design_playbook.md`, updated after each kernel converges. This is the accumulating knowledge base:

```
After flash attention:
  "cp.async gave 2.54x — use from the start on memory-bound kernels"
  "XOR swizzle eliminates bank conflicts — apply by default"
  "Register-only datapath beats shared memory round-trip when possible"

After GEMM:
  "Tile size M=128 N=128 K=32 hit sweet spot for register pressure"
  "Double-buffer K tiles, not output tiles"
  ...each kernel adds to the playbook
```

**Update the skill** — `/autokernel gemm mar15` to select which kernel to optimize.

### The kernel roadmap

| Kernel | Status | Purpose |
|--------|--------|---------|
| Flash Attention (BF16) | Optimizing now | Training bottleneck |
| BF16 GEMM | Stub exists | Linear layers |
| Flash Attention (FP8) | Not started | 2x throughput from `mma.sync.m16n8k32` |
| Fused MLP | Not started | Linear + activation + linear in one kernel |
| RMSNorm + Attention | Not started | Eliminate memory round-trip between norm and attn |

### The flywheel

```
 Kernel 1 (attention)
   └─ converges → lessons extracted
        └─ Kernel 2 (GEMM) starts at higher baseline
             └─ converges → more lessons
                  └─ Kernel 3 starts even higher
                       └─ ...
```

Each kernel's Phase 1 gets shorter because `csrc/common/` grows and the playbook gets richer. Each kernel's Phase 2 converges faster because the starting point is already partially optimized. The loop compounds.
