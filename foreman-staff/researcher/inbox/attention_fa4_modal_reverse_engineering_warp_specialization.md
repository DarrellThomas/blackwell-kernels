# Modal's Reverse-Engineering of FlashAttention-4: Warp Specialization Details

**Source:** https://modal.com/blog/reverse-engineer-flash-attention-4
**Relevant to:** attention worker
**Worker's current problem:** BF16 at 94% compiler ceiling (68 us). FP8 at 2.33x SDPA (52 us). Math_pipe_throttle 48%.
**Date:** 2026-03-15
**Supplements:** attention_fa4_full_paper_deep_dive_new_details.md (our existing FA4 brief)

---

## What This Is

Modal (cloud GPU company) reverse-engineered FlashAttention-4's CuTe-DSL kernel to
understand the implementation details not covered in the paper. This blog post
provides the most detailed public breakdown of FA4's warp specialization pipeline.

---

## Why It Matters for Us

### FA4's 5-Type Warp Specialization Pipeline

FA4 uses **warp specialization** where different warps within a CTA do completely
different jobs, coordinated via barriers:

| Warp Type | Count | Job |
|-----------|-------|-----|
| Load warp | 1 | Streams Q tiles to SMEM, K/V via TMA |
| MMA warp | 1 | Tensor core matmuls (QK^T and PV) |
| Softmax warps | 8 (2 warpgroups) | Score normalization, running stats |
| Correction warps | 4 | Rescale outputs when stability factors change |
| Epilogue warps | 1-2 | Store finalized outputs to global memory |

**Key insight for us:** FA4 dedicates a **majority of warps to non-MMA work**
(softmax + correction + epilogue = 13-14 warps out of ~16). This is the opposite
of our approach (all 4 warps do both MMA and softmax).

The rationale: on Blackwell (sm_100), MMA is so fast (via tcgen05) that a single
MMA warp can saturate the tensor cores. The bottleneck is the non-MMA work
(softmax, rescaling), so more warps are assigned there.

### Portable Insight: Software Exponential Approximation

FA4 replaces SFU-based `exp()` with a **cubic polynomial via Horner's method**:

```
exp2(x) = 2^floor(x) * poly(frac(x))
poly(f) = c0 + f*(c1 + f*(c2 + f*c3))  // 3 FMAs
```

This uses FMA units instead of SFUs. On sm_120, SFU throughput is 16 ops/clock/SM.
FMA throughput is much higher (tensor cores + CUDA cores). If SFU saturation
contributes to math_pipe_throttle, replacing exp2f with a polynomial could help.

**However:** Our existing FA4 brief already covers this technique and notes it's
mainly useful when SFU is the bottleneck. For D=64, we compute ~64 exp2f calls
per KV block. At 16 ops/clock, that's only 4 clock cycles per warp -- likely not
a significant contributor.

### Portable Insight: Lazy Rescaling (Conditional Correction)

FA4 only rescales accumulated outputs when the running max has changed "enough to
impact numerical stability." This reduces correction operations by ~10x.

**Our existing brief covers this** as "conditional rescaling with tau=8.0."

### Instruction Details (NOT Portable)

FA4 uses `tcgen05.mma.cta_group::1` instructions -- 5th generation tensor cores
exclusive to sm_100 datacenter Blackwell. The single-warp MMA approach works
because tcgen05 can dispatch an entire 128x128 MMA from one warp. On sm_120
with mma.sync (16x8x16), we need all warps to achieve the equivalent throughput.

---

## What's Genuinely New (Not in Our Existing FA4 Brief)

### 1. Two Q Tiles Per CTA

FA4 processes **two query tiles per CTA instance**, streaming across all KV tiles
for each. This amortizes the KV tile loading cost across two Q computations.

**For our kernel:** We process one Q tile per CTA. Processing two would reduce KV
load traffic by ~50% but requires 2x the register space for Q fragments and
accumulators. At D=64 with 145 registers, we might have register budget for this.
Worth investigating.

### 2. Producer-Consumer as "Vectorized Sequential Scan"

The Modal blog characterizes FA4's execution model as a "vectorized sequential scan
for a batch of aggregation queries against a key-value store." This mental model
-- treating attention as a streaming scan rather than a tiled matmul -- may suggest
different optimization strategies.

---

## Caveats

1. **SM_100 only.** The warp specialization pattern with 16+ warps and tcgen05 is
   not transferable to sm_120. We have 4 warps and mma.sync.

2. **The portable techniques (polynomial exp2, lazy rescaling) are already covered**
   in our existing FA4 briefs. This blog adds implementation detail but not new
   techniques.

3. **The "two Q tiles per CTA" idea** needs register analysis before attempting.

---

## Recommendation

**Low priority.** The Modal blog provides fascinating implementation detail about
FA4's warp specialization but most portable techniques are already in our existing
briefs. The only new actionable idea is:

- **Two Q tiles per CTA:** Investigate whether processing two Q blocks per CTA
  instance reduces KV load overhead. Register budget: ~145 current + ~32 for
  second Q tile + ~64 for second accumulator = ~241 registers. This would reduce
  occupancy from 3 blocks/SM to 2 blocks/SM. Likely a net negative for D=64
  but could be positive for D=128 where occupancy is already lower.
