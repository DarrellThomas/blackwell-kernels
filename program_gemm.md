# autokernel — GEMM

Profile-driven autonomous optimization of BF16 GEMM kernel for RTX 5090 (sm_120).

## Setup

To set up a new optimization run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autokernel/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autokernel/<tag>` from current HEAD.
3. **Read the in-scope files**: Read these for full context:
   - `.claude/CLAUDE.md` — project principles, build commands, MMA register layout.
   - `.claude/04_HARD_WON_LESSONS.md` — dead ends, structural constraints, load-bearing optimizations.
   - `docs/gemm_agent_state.md` — current best numbers, diagnosis, and next directions.
   - `docs/nvidia_blackwell_tuning_guide_sm120.md` — sm_120 hardware limits.
   - `docs/reference_spatters_mma_matmul.md` — reference Ada GEMM reaching 93% peak (directly applicable).
   - `docs/math_throttle_optimization.md` — math_pipe_throttle diagnosis and strategies.
   - `csrc/gemm/bf16_gemm_sm120.cu` — the kernel you optimize.
   - `csrc/common/*.cuh` — shared primitives (mma, ldmatrix, cp_async, swizzle).
4. **Run baseline**: `./eval.sh --kernel gemm > eval.log 2>&1`, then extract results. Record as baseline in `results/gemm.tsv`.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the optimization loop.

## The Optimization Loop

This is a **profile-driven** loop. Every iteration starts by reading the profiler output to identify the #1 bottleneck, then targets that specific bottleneck.

LOOP FOREVER:

1. **PROFILE**: Run `./eval.sh --kernel gemm > eval.log 2>&1` (full pipeline: build → test → bench → profile).
2. **READ RESULTS**: `grep -E "^(build|test|bench|profile|primary_|ncu_|top_)" eval.log`
   - If build or test FAIL: read the tail of eval.log, attempt fix (max 3 attempts), then re-run.
   - If bench FAIL: same — read log, attempt fix.
3. **ANALYZE BOTTLENECK**: The output includes a ranked stall analysis. Read the `top_bottleneck:` line. This is what you optimize next. Common bottlenecks and their fixes:

   | Bottleneck | What It Means | Optimization Strategy |
   |------------|---------------|----------------------|
   | `math_throttle` | MMA instructions arrive in bursts, saturating the tensor core input FIFO | Spread MMA over time: interleave loads between MMAs, increase per-warp tile size for more independent MMAs, see spatters.ca 3.0→3.1 |
   | `wait` | Threads waiting on `cp.async.wait` or `__syncthreads` barriers | Overlap loads with compute: adjust pipeline depth, prefetch further ahead |
   | `long_scoreboard` | Waiting on global memory loads (cache misses) | Use `cp.async` for global→shared, increase prefetch distance, improve data locality |
   | `short_scoreboard` | Waiting on shared memory or MIO operations | Reduce shared memory round-trips, reduce bank conflicts (XOR swizzle) |
   | `barrier` | Waiting at `__syncthreads` for other warps | Reduce sync points, larger tiles to amortize barriers |
   | `not_selected` | Warps ready but not scheduled (low occupancy) | Reduce register pressure, reduce shared memory per block |
   | `bank_conflicts` | Shared memory bank conflicts | Adjust XOR swizzle, restructure access patterns |

4. **DESIGN FIX**: Based on the bottleneck analysis, design a targeted optimization. Consult:
   - `docs/gemm_agent_state.md` for current diagnosis and direction
   - `.claude/04_HARD_WON_LESSONS.md` for dead ends — **do not retry anything listed there**
   - `docs/reference_spatters_mma_matmul.md` for techniques that worked on Ada
   - `docs/math_throttle_optimization.md` if math_throttle is dominant
5. **IMPLEMENT**: Modify kernel source files. Keep changes focused — one optimization per iteration.
6. **COMMIT**: `git commit -m "autokernel: <short description of optimization>"`
7. **EVALUATE**: Run `./eval.sh --kernel gemm > eval.log 2>&1` again. Extract results.
8. **DECIDE**:
   - If `primary_vs_ref` improved (higher) AND tests pass → **KEEP** the commit.
   - If `primary_vs_ref` is worse or equal → `git reset --hard HEAD~1` (discard).
   - If tests fail but the idea is sound → attempt fix (max 3 tries), else discard.
9. **RECORD**: Log results to `results/gemm.tsv` (see format below).
10. **UPDATE STATE**: Update `docs/gemm_agent_state.md` with:
    - Current best (commit, duration, vs cuBLAS ratio, top stall)
    - Updated diagnosis if the bottleneck profile changed
    - Any new direction insights from this iteration
    This file is the persistent memory across context clears — the watchdog flushes context after every cycle. If you don't write it back here, the next cycle wakes up with stale state.
11. **LOOP**: Go to step 1.

## What You CAN Modify

- `csrc/gemm/bf16_gemm_sm120.cu` — the GEMM kernel
- `csrc/common/*.cuh` — shared CUDA primitives (mma, ldmatrix, cp_async, swizzle)
- New `.cuh` headers in `csrc/common/` if needed for new primitives

## What You CANNOT Modify

- `tests/test_gemm.py` — correctness gate, immutable
- `benchmarks/bench_gemm.py` — benchmark harness, immutable
- `profiles/profile_gemm.py` — profile harness, immutable
- `eval.sh` — evaluation script, immutable
- `python/` — Python bindings (change only if kernel signature changes)

## Hardware Constraints (sm_120 / RTX 5090)

Hard limits from the NVIDIA tuning guide. Violating them will crash or silently degrade:

- **Shared memory per block: 99 KB max** (CUDA reserves 1 KB from 128 KB/SM)
- **Shared memory per SM: 128 KB** — constrains multi-block occupancy
- **Registers per thread: 255 max** — but high register usage kills occupancy
- **Max warps per SM: 48** (fewer than datacenter's 64)
- **Max thread blocks per SM: 32**
- **Tensor core ISA: `mma.sync`** (NOT `tcgen05`). Always swap a1/a2 from ldmatrix_x4.
- **Static shared memory: 48 KB** — use dynamic allocation for more
- **32 KB smem sweet spot** — current kernel uses 32 KB double-buffered, leaving 96 KB for L1. Exceeding this kills L1 and regresses performance.
- See `docs/nvidia_blackwell_tuning_guide_sm120.md` for occupancy calculations.

## Primary Metric

**`primary_vs_ref`** — the speedup ratio of our GEMM kernel vs cuBLAS (`torch.mm`) on the primary config (M=N=K=4096, BF16). Higher is better. 1.0x = parity with cuBLAS.

Secondary: `ncu_duration_us`, `ncu_sm_throughput_pct`.

## Current Architecture Summary

**Instruction:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
**Tiles:** BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 8 warps (256 threads)
**Data path:** Global → cp.async → Shared (XOR swizzle, double-buffered) → ldmatrix → Registers → mma.sync → FP32 accumulators → BF16 store

This architecture has been optimized through 38 experiments. The remaining gap to cuBLAS (~11%) is likely in MMA scheduling efficiency. See `docs/gemm_agent_state.md` for full diagnosis and next directions.

## Simplicity Criterion

All else being equal, simpler is better. A small speedup that adds ugly complexity is not worth it. Conversely, simplifying the kernel while maintaining speed is a great outcome.

## Logging Results

Log to `results/gemm.tsv` (tab-separated, NOT committed to git).

Header:
```
commit	duration_us	vs_ref	sm_pct	stall_math	stall_wait	stall_scoreboard	stall_barrier	top_stall	status	description
```

1. git commit hash (short, 7 chars)
2. kernel duration from ncu (us)
3. vs_ref ratio (e.g. 0.89) — our kernel time / cuBLAS time, higher = better
4. SM throughput % from ncu
5. math_throttle stall %
6. wait stall %
7. scoreboard stall % (long + short combined)
8. barrier stall %
9. top stall reason
10. status: `keep`, `discard`, or `crash`
11. short text description of what this experiment tried

## Recording Hard-Won Lessons

When an experiment reveals something **structural**, append it to `.claude/04_HARD_WON_LESSONS.md`. This file persists across context clears.

### When to write

**Big wins (>5% improvement):** Record what you did and why it worked. Focus on the insight.

**Hard failures:** When something expected to help made things worse. Record what, what happened, and why. These prevent future iterations from repeating mistakes.

**Structural discoveries:** Hardware or ISA behaviors not in the docs.

### When NOT to write

- Small incremental improvements (<5%)
- Routine discards (typo, build failure, wrong constant)
- Anything already recorded in the file

### Format

Append to the appropriate section in `04_HARD_WON_LESSONS.md`. Bold the key insight, explain briefly. 1-3 lines per entry.

## Convergence Detection

### Signal 1: Bottleneck Oscillation
If `top_bottleneck` alternates between the same two stalls for 3+ iterations without duration improvement, you've hit a local equilibrium.

**Action**: Try a **structural change** — different tile geometry, warp count, pipeline depth, loop structure — that resets the bottleneck distribution. Don't keep grinding the same two knobs.

### Signal 2: Small Gains
If recent kept experiments show <2% improvement each, incremental tuning is hitting diminishing returns.

**Action**: Keep the small gains (they compound), but mix in bold experiments. Alternate safe micro-optimizations with high-risk structural changes. Discard cost is one iteration (~90 seconds).

### When to Actually Stop

**Only stop when the human interrupts (Ctrl+C) or you cannot think of anything new to try after exhausting:**

1. Re-read profiler output from scratch — metrics you haven't focused on
2. Re-read reference docs for techniques not yet tried
3. Study spatters.ca reference (93% peak via 4x4 tiling + cp.async pipeline)
4. Combine two previous near-miss optimizations
5. Try the opposite of recent attempts
6. Try radically different tile geometries or warp counts
7. Look at 2nd and 3rd bottlenecks, not just #1
8. Revisit a previously discarded idea with a twist

If truly exhausted, write convergence report to `results_summary.md` and notify the human.

## Autonomous Operation

Once the loop begins, do NOT pause to ask the human — run autonomously until manual interruption. The loop is cheap (~90 seconds per iteration). Every discarded experiment costs little but might reveal an insight.

**Persistence is the strategy.** Small gains compound. Novel failures teach. Don't give up after one failure — try variations.

## Build & Test Commands

```bash
# Full evaluation (build + test + bench + profile)
./eval.sh --kernel gemm > eval.log 2>&1

# Quick check (no profiling, faster)
./eval.sh --kernel gemm --quick > eval.log 2>&1

# Profile only
./eval.sh --kernel gemm --profile > eval.log 2>&1

# Extract key results
grep -E "^(build|test|bench|profile|primary_|ncu_|top_)" eval.log
```

**Environment**: Always inherited from eval.sh — `CUDA_VISIBLE_DEVICES=1`, `CUDA_HOME=/usr/local/cuda-13`. Do NOT run these manually.

## Dashboard & Logs

Live dashboard at **http://localhost:8420** reads `results/gemm.tsv` and auto-refreshes.

Preserve per-iteration logs: copy eval.log to `logs/gemm/eval_<iteration>.log`. Create dir on first use.
