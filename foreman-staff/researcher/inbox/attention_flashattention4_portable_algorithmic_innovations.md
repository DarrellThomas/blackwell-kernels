# FlashAttention-4: Portable Algorithmic Innovations for sm_120

**Sources:**
- https://arxiv.org/html/2603.05451v1 (FlashAttention-4 paper, March 2026)
- https://modal.com/blog/reverse-engineer-flash-attention-4 (Modal reverse engineering blog)

**Relevant to:** attention worker
**Worker's current problem:** BF16 attention at 94% of compiler ceiling (68 us vs 64 us theoretical). math_pipe_throttle at 48% from softmax between QK^T and PV MMA phases. FP8 at 2.33x SDPA but latency-bound (SM 43.8%). All C++ optimization paths exhausted after 66 experiments. The worker's stated next directions require architectural changes.

---

## What This Is

FlashAttention-4 was released this week (March 2026). While it targets datacenter Blackwell (tcgen05, TMEM), it introduces **three algorithmic innovations that are fully portable to mma.sync-based kernels on sm_120**. These address the exact bottleneck our worker faces: softmax overhead between MMA phases.

---

## INNOVATION 1: Conditional Softmax Rescaling (Skip When Safe)

**The problem we have:** Every KV block iteration computes new_max, checks if max changed, and rescales all partial results by exp(old_max - new_max). This rescaling touches every accumulator register and involves multiply chains.

**FA4's solution:** Skip rescaling when the max delta is below a threshold τ (typically log₂(256) = 8.0 for BF16). The key insight: if the new max is within τ of the old max, the rescaling factor exp(old_max - new_max) is close to 1.0 and the precision loss from skipping it is within BF16's representable range.

**Implementation:**
```cuda
// Current approach (rescales EVERY iteration):
new_max = max(old_max, row_max);
rescale = exp2f((old_max - new_max) * LOG2E);
for (int i = 0; i < D/8; i++) acc[i] *= rescale;  // expensive!
sum *= rescale;

// FA4 approach (conditional rescaling):
new_max = max(old_max, row_max);
float delta = old_max - new_max;
if (fabsf(delta) > TAU) {  // TAU = 8.0 for BF16
    rescale = exp2f(delta * LOG2E);
    for (int i = 0; i < D/8; i++) acc[i] *= rescale;
    sum *= rescale;
}
// Final normalization at the end handles accumulated error
```

**Expected impact for us:**
- FA4 reports **10x reduction in correction operations** with this technique
- In our kernel with D=64 and N=2048: ~32 KV blocks. If rescaling happens only in ~10% of blocks (3 iterations instead of 32), the rescaling multiply chain (~64 FMA instructions per rescale) is reduced from ~2048 to ~192 instructions
- This directly reduces math_pipe_throttle (our #1 stall at 48%)
- **Risk:** Requires a final normalization pass that adds some instructions. Net savings depend on how often rescaling is actually skipped for real attention distributions.

**Precision consideration:** For our use case (BF16 attention with FP32 accumulators), τ = 8.0 means we skip rescaling whenever the max changes by less than 256x. For typical attention patterns, the max stabilizes quickly after the first few KV blocks, so most later blocks can skip rescaling entirely.

---

## INNOVATION 2: Software-Emulated Exponential via FMA Polynomial

**The problem we have:** softmax requires exp2f() (MUFU.EX2 on the Special Function Unit). The SFU is a throughput bottleneck — it has limited issue rate, and all warps compete for it. This is a major component of the math_pipe_throttle stall.

**FA4's solution:** Implement 2^x using a degree-3 Horner's method polynomial on FMA units (CUDA Cores) instead of the SFU:

```cuda
// Hardware exp2f (uses MUFU.EX2 - Special Function Unit):
float result = exp2f(x);

// Software exp2f (uses FMA - CUDA Core):
// 2^x = 2^floor(x) * 2^frac(x)
// 2^floor(x) via IEEE 754 bit manipulation
// 2^frac(x) via degree-3 polynomial approximation

__device__ float fast_exp2f_fma(float x) {
    // Integer part: construct float with biased exponent
    int i = __float_as_int(x);
    int floor_x = (i >> 23) - 127;  // extract exponent
    float frac = x - (float)floor_x;

    // Polynomial approximation of 2^frac for frac in [0, 1)
    // Coefficients for BF16-precision match:
    const float c3 = 0.0796f;
    const float c2 = 0.2274f;
    const float c1 = 0.6931f;  // ln(2)
    const float c0 = 1.0f;

    // Horner's method: c0 + frac*(c1 + frac*(c2 + frac*c3))
    float poly = fma(frac, fma(frac, fma(frac, c3, c2), c1), c0);

    // Reconstruct: multiply by 2^floor_x
    int result_bits = __float_as_int(poly) + (floor_x << 23);
    return __int_as_float(result_bits);
}
```

**Expected impact for us:**
- Each exp2f becomes ~4 FMA + 2 integer ops instead of 1 MUFU.EX2
- FMA units are on the CUDA Core pipeline, which is SEPARATE from the SFU pipeline
- This allows exp2f computation to overlap with MMA (which uses the tensor core pipeline)
- FA4 reports this technique "mitigates exponential bottlenecks by leveraging FMA units during softmax computation"
- **The key insight:** On sm_120, MMA uses tensor cores, exp2f uses MUFU, and FMA uses CUDA cores. All three are different execution units. By moving exp2f from MUFU to FMA, we free up the MUFU and create more opportunities for overlapping softmax with MMA.

**Caveat:** The polynomial approximation has ~2^-10 relative error (matching BF16 precision but not FP32). For our attention kernel with FP32 accumulators, this is acceptable for the softmax (the final output is BF16 anyway). But verify the error is within our tolerance.

**Selective application:** FA4 doesn't replace ALL exp2f calls with the polynomial. It applies the software version to only 10-25% of entries (configurable), using hardware MUFU for the rest. This balances throughput gains against register pressure from the polynomial coefficients.

---

## INNOVATION 3: Ping-Pong Overlap (Two-Tile Processing)

**The problem we have:** Our kernel processes one BQ×D output tile per thread block. Within each KV block iteration, the computation is sequential: QK^T MMA → softmax → PV MMA. The softmax phase stalls the MMA pipeline.

**FA4's solution:** Process TWO output tiles per thread block, ping-ponging between them:
```
Tile A: QK^T MMA ─── softmax ─── PV MMA
Tile B:              QK^T MMA ─── softmax ─── PV MMA
```

While Tile A is in softmax (using SFU + CUDA cores), Tile B's QK^T MMA can run on tensor cores. This overlaps the two phases.

**Implementation considerations for sm_120:**
- **Register pressure:** Two tiles = 2× accumulators = 2× register usage. With our current 145 regs for one tile, two tiles would need ~290 regs → exceeds 170-reg limit for 3 blocks/SM.
- **Alternative:** Use BQ=32 for each tile (half the Q rows). Two BQ=32 tiles = same total work as one BQ=64 tile, but with overlap.
- **Or:** Process tiles in different KV block iterations: while Tile A is in softmax for KV block j, Tile B starts QK^T for KV block j+1.

**Expected impact:** This is the "overlap softmax with MMA across phase boundaries" that the worker identified as requiring "radical approach" in their next directions. FA4 proves it works at the algorithmic level.

**Risk:** Register pressure on sm_120 may make this impractical with current tile sizes. Would need BQ=32 per tile (doubling softmax passes) or aggressive register reduction.

---

## INNOVATION 4: LPT (Longest-Processing-Time) CTA Scheduling

**The problem we have:** With causal masking, CTAs that handle later Q positions process fewer KV blocks. This creates load imbalance across SMs.

**FA4's solution:** Order CTA launches by workload (longest first), so heavy CTAs start early and light CTAs fill in at the end. This is purely a grid scheduling optimization.

**Implementation:**
```cuda
// Instead of launching grid (num_q_tiles, batch*heads):
// Launch with remapped block indices
int block_id = blockIdx.x;
int q_tile = remap_lpt(block_id, num_q_tiles);  // longest-first ordering
```

**Expected impact:** 4-8% FLOPS improvement on causal attention (reported by FA4). Architecture-independent, zero cost to implement.

---

## RECOMMENDATION

**Priority order for our attention worker:**

1. **Conditional rescaling (Innovation 1)** — Lowest risk, highest expected impact. Can be implemented in ~10 lines of code change. Test with τ=8.0 for BF16. Should reduce math_pipe_throttle from 48% by eliminating most rescaling work.

2. **LPT scheduling (Innovation 4)** — Zero risk, small gain. Pure scheduling change. Implement for causal attention configs.

3. **Software exp2f (Innovation 2)** — Medium risk, potentially high impact. Moves exp2f from MUFU to FMA, enabling overlap with MMA. Start with selective application (10-25% of entries). Test precision carefully.

4. **Ping-pong overlap (Innovation 3)** — High risk (register pressure), high potential reward. Try BQ=32 per tile with two-tile overlap. If register pressure is manageable, this could break through the compiler ceiling.

---

## CAVEATS

1. **FA4 targets datacenter Blackwell** with 2-CTA cooperative processing, TMEM, and 256×256 tiles. These specific features do NOT apply to sm_120. Only the algorithmic innovations listed above transfer.

2. **The conditional rescaling threshold τ** needs empirical tuning on our attention distributions. τ=8.0 is FA4's recommendation for BF16 but may need adjustment for FP8.

3. **The polynomial exp2f** has lower precision than hardware MUFU.EX2. For training (our primary use case), this is likely acceptable. For inference with strict numerical reproducibility requirements, use hardware exp2f.

4. **FA4 uses 8 warps for softmax** in its warp-specialized design. Our kernel uses 4 warps total. The warp specialization pattern (dedicating warps to different phases) doesn't directly map to our 4-warp design, but the underlying principle (overlap compute phases) does.
