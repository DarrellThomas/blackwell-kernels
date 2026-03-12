# Attention Kernel Convergence Report

**Date:** 2026-03-12
**Branch:** autokernel/mar12
**Config:** B=2 H=8 N=2048 D=64, causal, RTX 5090 (sm_120)
**Iterations:** 38 total (24 kept, 14 consecutive discards at end)

---

## Final Metrics

| Metric | Value |
|--------|-------|
| Bench duration | 68 μs |
| vs cuDNN SDPA | 1.78x faster |
| ncu duration | 92.7 μs |
| SM throughput | 59.8% |
| Effective TFLOPS | 126.3 (56.4% of 224 peak) |
| vs compiler ceiling (64 μs) | 1.06x (94%) |
| vs full PTX ceiling (55 μs) | 1.24x (81%) |
| vs hard floor (38 μs) | 1.79x (56%) |

### Stall Breakdown (converged)

```
math_throttle:   48%  ← tensor core BUSY (input FIFO full)
wait:            17%  ← waiting for MMA result (pipeline latency)
scoreboard:      13%  ← ldmatrix latency (shared → registers)
barrier:          5%  ← __syncthreads between KV iterations
short_scoreboard: 3%  ← MIO pipe latency
not_selected:     2%  ← scheduling
active_issue:    12%  ← useful instruction issue
```

Tensor utilization ≈ math_throttle (48%) + MMA fraction of active_issue (~6%) ≈ 54-56%.

---

## What Worked (kept optimizations, in order)

| # | Commit | Speedup | Description |
|---|--------|---------|-------------|
| 1 | 114c55c | baseline | Round 2 baseline (1.61x SDPA) |
| 2 | 0fb7b84 | 1.0% | Prefetch after QK^T — barrier 6→3% |
| 3 | 08488c6 | 0.7% | Unconditional exp in softmax — removed 64 compare+select |
| 4 | 303e47e | 0.8% | exp2f softmax — fold LOG2E into Q scale, save 34 MULs/iter |
| 5 | 5b44875 | 2.4% | Skip mask for fully unmasked KV blocks — 60+ fewer conditionals/iter |
| 6 | 2cb22e7 | 0.2% | ldmatrix_x4 for K loads — fewer instructions per MMA pair |
| 7 | n/a | 0% | Dynamic BQ dispatch — BQ=128 for large grids (2x faster at N=4096) |

**Pattern:** All successful optimizations reduced scalar instruction count in the hot path.
The compiler schedules MMA well, but scalar overhead (softmax, masking, control flow)
creates irreducible tensor core idle time. Removing scalar instructions directly converts
to tensor core utilization gains.

---

## What Didn't Work (14 consecutive discards)

### Structural changes (all regressed)

| Approach | Result | Root Cause |
|----------|--------|------------|
| BLOCK_Q=128, 8 warps | 2.3x slower | Register spills (234 regs) |
| BLOCK_KV=128 | 27% slower | Occupancy drop (12→8 warps) |
| BLOCK_KV=96 | 13% slower | Bank conflicts doubled |
| BLOCK_KV=32 | 12% slower | Doubled iterations |
| Q in shared memory | 22% slower | Extra ldmatrix per KV block |
| launch_bounds(128,4) | 5% slower | Spill cost > occupancy gain |
| 8 warps (any config) | Always slower | Register pressure on sm_120 |

### Scheduling changes (no effect or regressed)

| Approach | Result | Root Cause |
|----------|--------|------------|
| Loop reordering (nc/dc swap) | No change | Compiler unrolls identically |
| P*V loop reorder | No change | Compiler unrolls identically |
| Non-volatile MMA | No change | Compiler produces identical code |
| -O2 vs default | No change | Same code quality |
| Split prefetch timing | 3% worse | Extra commit overhead |
| Prefetch before QK^T | 3% worse | cp.async/ldmatrix contention |

### Block scheduling changes (all regressed)

| Approach | Result | Root Cause |
|----------|--------|------------|
| Stride-3 remapping | 23% slower | Destroyed L2 locality |
| Light/heavy interleave | 18% slower | Same L2 destruction |
| Stagger delay | 3% worse | Delay overhead, bank conflicts doubled |

**Key insight:** ANY change to block-to-SM mapping destroys the natural L2 cache
locality for K/V data. The default linear mapping is optimal.

### Softmax variants (marginal or worse)

| Approach | Result | Root Cause |
|----------|--------|------------|
| Split-half softmax | No change | Extra scalar overhead cancelled reduction |
| Skip first-iter rescale | No change | Branch overhead ≈ savings |
| Deferred sum shuffles | 3% worse | Shuffles may aid compiler scheduling |
| Fused QK-softmax-PV | 12% worse | O rescale overhead 4x |

---

## Remaining Bottleneck Analysis

### The structural ceiling: synchronized softmax

The dominant remaining bottleneck is the synchronized softmax computation between
QK^T and PV matmuls. During softmax (~134 cycles per KV iteration), ALL warps on
the SM execute scalar work simultaneously (3 blocks × 4 warps = 12 warps in lockstep).
The tensor core sits completely idle during this window.

This creates a ~19% structural ceiling on tensor utilization that no compiler-managed
optimization can eliminate. The only way to break through is:

1. **Full PTX inner loop** — hand-scheduled assembly that interleaves softmax scalar
   ops with MMA instructions from the next iteration. Requires manual register
   allocation and breaking the natural phase boundaries.

2. **FP8 attention** — `mma.sync.aligned.m16n8k32` provides 2x tensor throughput.
   The same softmax overhead becomes a smaller fraction of total time.

3. **Algorithmic changes** — alternative attention mechanisms that reduce or eliminate
   the softmax serialization (e.g., linear attention, sigmoid attention).

### Why the compiler can't fix this

The compiler sees `#pragma unroll` and correctly unrolls the KV loop. But it cannot
reorder instructions across the softmax barrier because:
- The PV matmul depends on softmax output (data dependency)
- The next QK^T depends on prefetched K data (cp.async dependency)
- These dependencies create a strict phase ordering that the compiler respects

A hand-written PTX inner loop could overlap the END of softmax with the START of
PV loads, and the END of PV with the START of next-iteration QK^T prefetch. This
~10% improvement is what separates the compiler ceiling (64 μs) from the PTX ceiling
(55 μs).

---

## Multi-Config Performance

| Config | Duration | vs SDPA | Notes |
|--------|----------|---------|-------|
| B2H8 N2048 D64 causal | 68 μs | 1.78x | Primary config |
| B2H8 N4096 D64 causal | ~260 μs | ~1.15x | Uses BQ=128 dynamic dispatch |
| B4H16 N2048 D128 causal | ~420 μs | ~1.00x | D=128 uses BQ=128 |

---

## Recommendation

The attention kernel has converged at the compiler-managed optimization ceiling.
Further gains require a fundamentally different approach:

1. **Phase 1 (recommended next):** FP8 attention kernel — 2x tensor throughput
   makes the softmax overhead proportionally smaller, potentially reaching 100+ TFLOPS
   effective throughput.

2. **Phase 2 (high effort, moderate gain):** Full PTX inner loop rewrite — target
   ~55 μs (15% improvement). Requires writing ~200 lines of PTX assembly with
   manual register allocation. The salykova/gau-nernst approach.

3. **Phase 3 (research):** Algorithmic alternatives to softmax-based attention
   for the forward pass.
