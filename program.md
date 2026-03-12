# autokernel

Profile-driven autonomous optimization of CUDA kernels for RTX 5090 (sm_120).

## Setup

To set up a new optimization run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autokernel/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autokernel/<tag>` from current HEAD.
3. **Read the in-scope files**: Read these for full context:
   - `.claude/CLAUDE.md` — project principles, build commands, MMA register layout.
   - `.claude/03_PROJECT_SPECIFICATION.md` — kernel specs, API, phases.
   - `.claude/04_HARD_WON_LESSONS.md` — current bottleneck profile, completed and remaining optimizations.
   - `docs/nvidia_blackwell_tuning_guide_sm120.md` — sm_120 hardware limits (shared memory, registers, occupancy).
   - `csrc/attention/flash_attn_v2_sm120.cu` — the kernel you optimize.
   - `csrc/common/*.cuh` — shared primitives (mma, ldmatrix, cp_async, swizzle).
4. **Run baseline**: `./eval.sh > eval.log 2>&1`, then extract results. Record as baseline in `results.tsv`.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the optimization loop.

## The Optimization Loop

This is a **profile-driven** loop. Unlike blind search, every iteration starts by reading the profiler output to identify the #1 bottleneck, then targets that specific bottleneck. The profiler tells you what to fix — you don't guess.

LOOP FOREVER:

1. **HEARTBEAT**: Signal the dashboard that this kernel's loop is active. On the first iteration, write the start time: `date +%s > .autokernel.<kernel>.alive`. On subsequent iterations, just update mtime: `touch .autokernel.<kernel>.alive`. (The file content is the loop start epoch; mtime is the last heartbeat.)
2. **PROFILE**: Run `./eval.sh > eval.log 2>&1` (full pipeline: build → test → bench → profile).
2. **READ RESULTS**: `grep -E "^(build|test|bench|profile|primary_|ncu_|top_)" eval.log`
   - If build or test FAIL: read the tail of eval.log, attempt fix (max 3 attempts), then re-run.
   - If bench FAIL: same — read log, attempt fix.
3. **ANALYZE BOTTLENECK**: The output includes a ranked stall analysis. Read the `top_bottleneck:` line. This is what you optimize next. Common bottlenecks and their fixes:

   | Bottleneck | What It Means | Optimization Strategy |
   |------------|---------------|----------------------|
   | `math_throttle` | Tensor core MMA instructions arrive in bursts, saturating the pipe | Spread MMA over time: interleave loads with compute, increase tile size for more MMA per launch, pipeline stages |
   | `wait` | Threads waiting on `cp.async.wait` or `__syncthreads` barriers | Overlap loads with compute: double-buffer, increase pipeline depth, prefetch further ahead |
   | `long_scoreboard` | Waiting on global memory loads (cache misses) | Use `cp.async` for global→shared, increase prefetch distance, improve data locality |
   | `short_scoreboard` | Waiting on shared memory or MIO operations | Reduce shared memory round-trips (register-only paths), reduce bank conflicts |
   | `barrier` | Waiting at `__syncthreads` for other warps | Reduce sync points, use warp-level primitives instead of block-level where possible |
   | `not_selected` | Warps ready but not scheduled (low occupancy) | Reduce register pressure, reduce shared memory per block, increase blocks/SM |
   | `lg_throttle` | LSU input queue full | Reduce global memory traffic, coalesce accesses, use shared memory staging |
   | `bank_conflicts_store` / `bank_conflicts_load` | Shared memory bank conflicts | Apply XOR swizzle, pad shared memory, restructure access patterns |

4. **DESIGN FIX**: Based on the bottleneck analysis, design a targeted optimization. Consult:
   - `docs/nvidia_blackwell_tuning_guide_sm120.md` for hardware limits
   - `.claude/04_HARD_WON_LESSONS.md` for previously planned optimizations
   - The kernel source for current structure
5. **IMPLEMENT**: Modify kernel source files. Keep changes focused — one optimization per iteration.
6. **COMMIT**: `git commit -m "autokernel: <short description of optimization>"`
7. **EVALUATE**: Run `./eval.sh > eval.log 2>&1` again. Extract results.
8. **DECIDE**:
   - If `primary_vs_sdpa` improved (higher) AND tests pass → **KEEP** the commit.
   - If `primary_vs_sdpa` is worse or equal → `git reset --hard HEAD~1` (discard).
   - If tests fail but the idea is sound → attempt fix (max 3 tries), else discard.
9. **RECORD**: Log results to `results.tsv` (see format below).
10. **LOOP**: Go to step 1. The new profile will reveal a new #1 bottleneck.

## What You CAN Modify

- `csrc/attention/flash_attn_v2_sm120.cu` — the main kernel
- `csrc/common/*.cuh` — shared CUDA primitives (mma, ldmatrix, cp_async, swizzle)
- New `.cuh` headers in `csrc/common/` if needed for new primitives

## What You CANNOT Modify

- `tests/test_attention.py` — correctness gate, immutable
- `benchmarks/bench_attention.py` — benchmark harness, immutable
- `profiles/profile_v2.py` — profile harness, immutable
- `eval.sh` — evaluation script, immutable
- `csrc/attention/flash_attn_sm120.cu` — v1 reference kernel, immutable
- `python/` — Python bindings (change only if kernel signature changes)

## Hardware Constraints (sm_120 / RTX 5090)

These are hard limits from the NVIDIA tuning guide. Violating them will crash or silently degrade:

- **Shared memory per block: 99 KB max** (CUDA reserves 1 KB from 128 KB/SM)
- **Shared memory per SM: 128 KB** — constrains multi-block occupancy
- **Registers per thread: 255 max** — but high register usage kills occupancy
- **Max warps per SM: 48** (fewer than datacenter's 64)
- **Max thread blocks per SM: 32**
- **Tensor core ISA: `mma.sync`** (NOT `tcgen05`). Always swap a1/a2 from ldmatrix_x4.
- **Static shared memory: 48 KB** — use dynamic allocation for more
- See `docs/nvidia_blackwell_tuning_guide_sm120.md` for occupancy calculations.

## Primary Metric

**`primary_vs_sdpa`** — the speedup ratio of our v2 kernel vs cuDNN SDPA on the primary config (B=2 H=8 N=2048 D=64, causal). Higher is better. Current: ~1.3x. Target: >1.5x.

Secondary: `ncu_duration_us`, `ncu_sm_throughput_pct`, `ncu_tensor_pipe_pct`.

## Simplicity Criterion

All else being equal, simpler is better. A small speedup that adds ugly complexity is not worth it. Conversely, simplifying the kernel while maintaining speed is a great outcome. When evaluating whether to keep a change, weigh complexity cost against improvement magnitude.

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT committed to git).

Header and columns:

```
commit	duration_us	vs_sdpa	sm_pct	stall_math	stall_wait	stall_scoreboard	stall_barrier	top_stall	status	description
```

1. git commit hash (short, 7 chars)
2. kernel duration from ncu (us)
3. vs_sdpa ratio (e.g. 1.33)
4. SM throughput % from ncu
5. math_throttle stall % (e.g. 29.0)
6. wait stall % (e.g. 20.0)
7. scoreboard stall % (long + short combined, e.g. 16.0)
8. barrier stall % (e.g. 6.0)
9. top stall reason (e.g. math_throttle)
10. status: `keep`, `discard`, or `crash`
11. short text description of what this experiment tried

Example:

```
commit	duration_us	vs_sdpa	sm_pct	stall_math	stall_wait	stall_scoreboard	stall_barrier	top_stall	status	description
a1b2c3d	115.5	1.33	48.5	29.0	20.0	16.0	6.0	math_throttle	keep	baseline v7
b2c3d4e	98.2	1.52	55.1	18.0	25.0	12.0	5.0	wait	keep	interleave MMA with K prefetch
c3d4e5f	120.1	1.25	42.3	31.0	15.0	18.0	8.0	math_throttle	discard	BLOCK_Q=128 (too much register pressure)
```

## Dashboard & Progress Monitoring

A live dashboard runs at **http://localhost:8420** during optimization. Start it before the loop:

```bash
python3 dashboard.py &
```

The dashboard reads `results.tsv` and auto-refreshes every 30 seconds. It shows:
- Summary cards: iteration count, best duration, vs SDPA ratio, SM throughput, current bottleneck
- Charts: duration over time, vs SDPA over time (with 1.5x target line), SM throughput, stall breakdown bar chart
- **Stall evolution chart**: shows how math_throttle, wait, scoreboard, and barrier stalls shift across iterations — this is the key chart for spotting bottleneck oscillation (Signal 1)
- Full experiment log table (newest first), color-coded by keep/discard/crash

The human can check the dashboard from a browser at any point during the run without interrupting the agent.

Also preserve **per-iteration logs**: after each eval run, copy eval.log to `logs/eval_<iteration_number>.log` so the human can review any specific iteration's full output. Create the `logs/` directory on first use.

## Build & Test Commands

```bash
# Full evaluation (build + test + bench + profile)
./eval.sh > eval.log 2>&1

# Quick check (no profiling, faster)
./eval.sh --quick > eval.log 2>&1

# Profile only (skip build/test/bench)
./eval.sh --profile > eval.log 2>&1

# Extract key results
grep -E "^(build|test|bench|profile|primary_|ncu_|top_)" eval.log
```

**Environment**: Always inherited from eval.sh — `CUDA_VISIBLE_DEVICES=1`, `CUDA_HOME=/usr/local/cuda-13`. Do NOT run these manually.

## Recording Hard-Won Lessons

When an experiment reveals something **structural** — not just a number changing — append it to `.claude/04_HARD_WON_LESSONS.md`. This file persists across compressions and future runs, so what you write here directly improves future iterations.

### When to write

**Big wins (kept experiments):** When an optimization gives >5% improvement, record what you did and why it worked. Focus on the insight, not the code change. Example: "cp.async with zero-fill gave 2.54x because it eliminated register-mediated global→shared copies AND avoided OOB guard branches."

**Hard failures (discarded experiments):** When something you expected to help made things worse, record what you tried, what happened, and why it failed. These are often more valuable than wins because they prevent future iterations from repeating the same mistake. Example: "BLOCK_Q=128 increased register pressure to 180 regs/thread, dropped to 1 block/SM, and the occupancy loss outweighed the larger tile benefit."

**Structural discoveries:** When you learn something about the hardware or ISA that isn't in the docs. Example: "On sm_120, interleaving 2 MMAs with 1 cp.async gives better pipe utilization than batching all MMAs then all loads."

### When NOT to write

- Small incremental improvements (<5%)
- Routine discards where the reason is obvious (typo, build failure, wrong constant)
- Anything already recorded in the file

### Format

Append to the appropriate section in `04_HARD_WON_LESSONS.md`. Use the existing style — bold the key insight, then explain briefly. Keep entries to 1-3 lines. If a new section heading is needed, add one.

## Convergence Detection & Exit Criteria

Track these signals across iterations to detect when optimization has plateaued:

### Signal 1: Bottleneck Oscillation
If the `top_bottleneck` alternates between the same two stalls for **3+ consecutive iterations** without duration improvement, you've hit a local equilibrium.

**Action**: Don't keep grinding the same two knobs. Instead, try a **structural change** — different BLOCK_Q/BLOCK_KV, different warp count, different pipeline depth, different loop structure — that resets the bottleneck distribution entirely. If one structural change fails, try a different one. There are many dimensions to explore.

### Signal 2: Small Gains
If recent kept experiments show <2% improvement each, incremental tuning on the current approach is getting diminishing returns.

**Action**: This is NOT a reason to stop. Small gains compound. Keep them. But also start mixing in **bold experiments** alongside incremental ones — alternate between safe micro-optimizations and high-risk structural changes. Try things that might not work. The discard cost is one iteration (~90 seconds).

### Halt Criterion: 95% of Theoretical Ceiling

The optimization loop targets **95% of the achievable theoretical ceiling** for each kernel. These ceilings come from roofline analysis in `docs/theoretical_limits.md`:

| Kernel | Achievable Ceiling | 95% Target | Hard Floor |
|--------|-------------------|------------|------------|
| **Attention** (B=2 H=8 N=2048 D=64 causal) | ~53 μs | **≤56 μs** | 38 μs |
| **GEMM** (M=N=K=4096 BF16) | ~614 μs | **≤646 μs** | 614 μs |

The dashboard's "vs Theory" card tracks this in real time.

**When to actually stop:**

1. **Target reached**: Duration ≤ 95% target. Write convergence report and notify the human.
2. **90% reached + exhausted**: If you reach 90% of ceiling and stall for **10+ consecutive iterations** with all structural approaches below exhausted — the ceiling estimate may be wrong. Update the ceiling in `docs/theoretical_limits.md`, write convergence report, notify the human.
3. **Human interrupts** (Ctrl+C).

**The ceiling estimates are living numbers.** As you learn more about the hardware through optimization, update them. If a structural discovery reveals the real ceiling is different, adjust `docs/theoretical_limits.md` AND the `KERNEL_CONFIG` in `dashboard.py`.

### Before Stopping: Exhaust These Approaches

Do NOT stop until you've tried all of the following:

1. Re-read the profiler output from scratch — look at metrics you haven't focused on
2. Re-read `docs/theoretical_limits.md` for the gap analysis and where headroom lives
3. Re-read the reference docs (`docs/*.md`) for techniques you haven't tried yet
4. Study reference implementations (gau-nernst uses BLOCK_Q=128, 94.4% peak)
5. Combine two previous near-miss optimizations that were each discarded individually
6. Try the opposite of what you've been doing (if you've been reducing shared memory, try using more)
7. Try radically different tile geometries, warp counts, or loop organizations
8. Look at the stall breakdown for the 2nd and 3rd bottlenecks, not just #1
9. Consider register pressure vs occupancy tradeoffs from a fresh angle
10. Try coalescing patterns you haven't explored (stores, loads, different vector widths)
11. Revisit a previously discarded idea with a twist

If you truly exhaust all of these AND meet the halt criterion above, write a convergence report to `results_summary.md` with final metrics, what worked, what didn't, and remaining bottleneck analysis. Then notify the human.

### Reference Ceiling

From the tuning guide, reference implementations, and our roofline analysis:
- **RTX 5090 sustained BF16 peak**: ~224 TFLOPS (verified via cuBLAS at 99.9%)
- **cuDNN SDPA**: 97.2% of peak on large configs (slower than our kernel on small configs)
- **gau-nernst custom FA**: 94.4% of 209.5 TFLOPS peak on sm_120 with BLOCK_Q=128
- **cuBLAS GEMM**: 99.9% of sustained peak on M=N=K=4096 — effectively at the physical limit
- **Attention hard floor**: 38 μs (tensor math only, unreachable due to softmax serialization)
- **Attention achievable ceiling**: ~53 μs (~70% tensor utilization, accounting for fundamental overheads)
- **GEMM achievable ceiling**: 614 μs (= cuBLAS, = physical limit)

Always keep `docs/theoretical_limits.md` and `docs/*.md` in mind — they contain the full analysis and may inspire new approaches.

## Autonomous Operation

Once the optimization loop has begun, do NOT pause to ask the human if you should continue — run autonomously until manual interruption. Keep optimizing. Keep experimenting. Small gains are still gains. Novel approaches that fail still teach you something (record it in hard-won lessons).

**Persistence is the strategy.** The loop is cheap to run. Every discarded experiment costs ~90 seconds but might reveal an insight that leads to the next breakthrough. Don't give up on an approach after one failure — try variations. Don't stop exploring because gains are small — compound them.

## Scope

This is a test run to validate the autokernel loop on the existing flash attention v2 kernel. The broader goal is to build a library of optimized kernels (GEMM, fused ops, FP8), each going through this same profile-driven optimization loop. The infrastructure (`eval.sh`, `program.md`, `results.tsv` format) is being designed to generalize to future kernels.
