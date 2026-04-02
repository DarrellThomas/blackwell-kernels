# Sawtooth Wavefront Reordering: 50% L2 Miss Reduction in Attention Kernels

**Source:** https://arxiv.org/abs/2601.16032
**Relevant to:** attention worker
**Worker's current problem:** FP8 attention at 2.33x SDPA (52 us), SM 43.8%. Math_pipe_throttle 48%.
**Date:** 2026-03-15

---

## What This Is

A new paper (arxiv 2601.16032, January 2026) introducing "Sawtooth Wavefront
Reordering," a CTA scheduling technique that reduces L2 cache misses by 50% and
improves attention throughput by up to 60% on NVIDIA GB10 (Grace Blackwell).

The technique reorders how threadblocks access KV tiles to improve L2 cache
locality, specifically for Flash Attention workloads.

---

## Why It Matters for Us

### L2 Cache Miss Pattern Problem

In standard Flash Attention, each threadblock processes a row of Q tiles against
all KV columns. With multiple threadblocks active simultaneously, they may access
different KV columns, causing L2 cache thrashing:

```
Standard ordering (simplified):
  CTA 0: K[0] K[1] K[2] K[3] ...
  CTA 1: K[0] K[1] K[2] K[3] ...
  CTA 2: K[0] K[1] K[2] K[3] ...
```

If CTAs are at different points in their KV iteration, they access different K
tiles, exceeding L2 capacity.

### Sawtooth Solution

The "sawtooth" pattern reorders CTA assignments so that adjacent CTAs process
the same or nearby KV columns simultaneously:

```
Sawtooth ordering (simplified):
  CTA 0: K[0] K[1] K[2] ...
  CTA 1: K[1] K[2] K[3] ...  (offset by 1)
  CTA 2: K[2] K[3] K[4] ...  (offset by 2)
```

This creates a "wavefront" pattern where all active CTAs share K tiles in the
L2 cache. The "sawtooth" name comes from the repeating wave pattern when
visualized.

### Results on GB10

- **50% or greater reduction in L2 cache misses**
- **Up to 60% throughput improvement**
- Validated in both CUDA C++ and cuTile implementations

---

## Applicability to sm_120 (RTX 5090)

**Potentially high impact.** The technique is architecture-agnostic -- it's a
grid launch optimization (how threadblocks are assigned to KV tiles), not a
kernel-internal change. It should work on any GPU with an L2 cache.

The RTX 5090 has a 96 MB L2 cache. For our primary config (B=2, H=8, D=64):
- K tile size: BKV * D * 2 bytes = 64 * 64 * 2 = 8 KB
- Total KV per head: seq_len * D * 2 = 4096 * 64 * 2 = 512 KB
- With 3 blocks/SM * 170 SMs = 510 active CTAs, if each processes different K
  tiles, working set = 510 * 8KB = ~4 MB (fits in L2)

At our current config, L2 capacity isn't the bottleneck. But at longer sequences
or larger head dimensions (D=128), the working set grows and L2 misses become
significant.

**Where this helps most:** Large batch sizes or long sequences where the KV cache
exceeds L2 capacity. If our worker benchmarks at longer sequence lengths, this
technique could show significant improvement.

---

## Key Technique (Implementation Sketch)

The core idea modifies the mapping from `blockIdx.x` to (q_tile, kv_start):

```cuda
// Standard: each CTA starts at KV tile 0
int q_tile = blockIdx.x;
int kv_start = 0;

// Sawtooth: stagger KV start based on CTA index
int q_tile = blockIdx.x;
int kv_start = (blockIdx.x * stride) % num_kv_tiles;
```

The stride and pattern depend on the number of active SMs and L2 cache size.
The paper likely provides optimal stride calculations.

---

## Caveats

1. **Benchmarked on GB10 (Grace Blackwell SoC), not RTX 5090.** GB10 has a
   different L2 cache size and memory hierarchy. Results may differ on desktop
   Blackwell.

2. **The 60% improvement is likely for L2-miss-dominated configs.** At our
   primary small-batch config, L2 pressure is low, so improvement would be
   minimal.

3. **Full paper PDF needed.** The arxiv abstract provides the headline results
   but not the implementation details. The full paper would reveal the exact
   stride calculation and interaction with occupancy settings.

4. **Interaction with occupancy.** Staggering KV access may conflict with our
   current 3-blocks/SM occupancy strategy if blocks within the same SM need
   synchronized KV access.

---

## Recommendation

**Medium priority.** Read the full paper (arxiv 2601.16032) for implementation
details. This technique is most relevant for:
- Long sequence benchmarks (8K+ tokens)
- Large batch sizes (B >= 8)
- D=128 configurations where KV tiles are larger

For our primary D=64, B=2 config, impact is likely small. But as a free grid-level
optimization (no kernel changes needed), it's worth testing once the implementation
is understood.
