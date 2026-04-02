# 3xBF16 FP32 Recovery: Exact Pseudocode from Ootomo & Yokota

**Source:** https://arxiv.org/html/2203.03341 (IJHPCA 2022)
**Relevant to:** numerical worker (Cholesky monolithic kernel)
**Worker's current problem:** Needs device-side SYRK/GEMM with FP32 accuracy inside a monolithic kernel. BF16 MMA loses ~1e-3 precision per multiply. BF16x9 achieves full FP32 but requires 9 MMA calls. Is there a middle ground?

## What This Is

The Ootomo/Yokota paper provides the exact algorithm for recovering FP32 precision from tensor core MMA using only 3 MMA calls instead of 9. While their paper targets TF32 (broken on sm_120), the same algorithm works with BF16 MMA (m16n8k16) and achieves ~14-15 bits of effective precision (vs BF16's 7 bits or FP32's 23 bits).

## Why It Matters for Us

The worker's monolithic Cholesky has 3 precision options for the trailing SYRK:
1. BF16 MMA (1 call, ~7-bit precision, ~1e-3 error) -- fast, maybe too imprecise
2. BF16x9 (9 calls, full FP32 precision) -- exact but slow
3. **3xBF16 (3 calls, ~14-bit precision, ~1e-4 error) -- sweet spot for Cholesky**

For Cholesky factorization of well-conditioned SPD matrices, ~14-bit precision in the SYRK update is likely sufficient. The diagonal potf2 and TRSM run in FP32, so numerical stability is maintained where it matters most.

## Key Technique: The 3xBF16 Algorithm

### Step 1: Splitting FP32 into Two BF16 Components

```cuda
// For each FP32 value v, decompose into v_hi (BF16) + v_lo (BF16 * 2^-11):
__nv_bfloat16 v_hi = __float2bfloat16_rn(v);            // truncate to 7-bit mantissa
float residual = v - __bfloat162float(v_hi);              // exact remainder
__nv_bfloat16 v_lo = __float2bfloat16_rn(residual * 2048.0f);  // scale by 2^11, convert
```

The 2^11 (2048) scaling factor is critical. Without it, the residual underflows in BF16. BF16 has the same 8-bit exponent as FP32, so the scaled residual is always representable. The reconstruction is:

```
v ≈ v_hi + (1/2048) * v_lo
```

### Step 2: Matrix Product with 3 MMA Calls

For C = A * B:

```cuda
// Split matrices
// A = A_hi + 2^(-11) * A_lo
// B = B_hi + 2^(-11) * B_lo

// Full expansion: A*B = A_hi*B_hi + 2^(-11)*(A_lo*B_hi + A_hi*B_lo) + 2^(-22)*A_lo*B_lo
// Term 4 (A_lo*B_lo / 2^22) affects only the LSB of the result -- DROP IT

// MMA call 1: main product
fill_fragment(frag_c, 0.f);
fill_fragment(frag_tmp, 0.f);
mma_sync(frag_tmp, frag_a_hi, frag_b_hi, frag_tmp);      // A_hi * B_hi

// Accumulate main product in FP32 with RN (NOT inside tensor core's RZ accum)
for (int i = 0; i < elements; i++)
    frag_c.x[i] += frag_tmp.x[i];

// MMA calls 2-3: correction terms (accumulated together)
fill_fragment(frag_dc, 0.f);
mma_sync(frag_dc, frag_a_lo, frag_b_hi, frag_dc);        // A_lo * B_hi
mma_sync(frag_dc, frag_a_hi, frag_b_lo, frag_dc);        // A_hi * B_lo

// Scale and add corrections
for (int i = 0; i < elements; i++)
    frag_c.x[i] += frag_dc.x[i] / 2048.0f;                // divide by 2^11
```

### Step 3: Key Rounding Mode Insight

The paper's critical finding: **do NOT accumulate the main product inside the tensor core's C accumulator across K iterations.** Tensor cores use round-to-zero (RZ) for accumulation, which introduces systematic errors. Instead:

1. For each K tile, compute a FRESH mma_sync with C=0
2. Add the result to an FP32 register accumulator using round-to-nearest (RN)
3. This single change recovers ~2 bits of precision

### Accuracy Analysis for Cholesky

With 3xBF16:
- Effective mantissa precision: ~14-15 bits (7 from v_hi + 7 from v_lo, minus overlap)
- Per-element relative error: ~2^(-15) ≈ 3e-5
- After 64 rank-k updates (NB=64 Cholesky panels): accumulated error ~64 * 3e-5 ≈ 2e-3
- This is MUCH better than raw BF16 (7 bits, ~1e-2 accumulated) and only 3x the cost

For comparison:
- Native FP32: 23-bit mantissa, ~1e-7 per-element
- BF16x9: full FP32 accuracy, 9x cost
- 3xBF16: ~15-bit accuracy, 3x cost
- 1xBF16: ~7-bit accuracy, 1x cost

### Why the Dropped Term is Safe

The dropped term A_lo * B_lo has magnitude ≤ 2^(-22) relative to A_hi * B_hi. With FP32 using 23-bit mantissa, this term at most affects the last 1-2 bits of the result. For Cholesky where we already lose ~8 bits from the BF16 truncation, this additional 1-bit loss is irrelevant.

## Caveats

1. **3xBF16 does NOT achieve full FP32 accuracy.** It achieves ~14-15 bits vs FP32's 23 bits. For well-conditioned Cholesky (kappa < 10^4), this is fine. For ill-conditioned matrices, use BF16x9 or iterative refinement.

2. **The 2^11 scaling is essential.** Using 2^8 (the "natural" split for 7-bit BF16 mantissa) causes more underflow. The 2^11 factor places the residual in the middle of BF16's representable range.

3. **Memory overhead: 2x input matrices.** Each A and B matrix needs both _hi and _lo variants in shared memory. For the monolithic kernel's limited shmem budget, this doubles the input storage for GEMM tiles.

4. **Register pressure: 2x fragment storage.** Need separate a_hi, a_lo, b_hi, b_lo fragments. With 202-register budget (matching cuSOLVER), this is tight.

## Recommendation

For the monolithic Cholesky kernel:
- **Start with 1xBF16** (simple, fast, good enough for many SPD matrices)
- **Profile accuracy** on representative matrices
- **If accuracy insufficient, upgrade to 3xBF16** (3x slower SYRK, but still monolithic)
- **If STILL insufficient, use BF16x9** (9x slower SYRK, but guaranteed FP32)

The progressive accuracy upgrade path means the worker can build incrementally.

## Sources

- [Ootomo & Yokota 2022, IJHPCA](https://arxiv.org/html/2203.03341) -- Full algorithm with pseudocode
- [Ootomo & Yokota 2022, arXiv](https://arxiv.org/abs/2203.03341) -- Paper abstract
- [ORNL TF32/TF64 (SC'23)](https://dl.acm.org/doi/10.1145/3624062.3624084) -- Extended framework
- [CUTLASS 3xTF32 Discussion](https://github.com/NVIDIA/cutlass/discussions/390) -- Implementation notes
