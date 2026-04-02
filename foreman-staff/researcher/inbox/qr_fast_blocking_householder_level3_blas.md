# Fast Blocking of Householder Reflectors via Level-3 BLAS on GPU

**Source:** Tomás, Quintana-Ortí. "Fast Blocking of Householder Reflectors on Graphics Processors." 26th Euromicro PDP, 2018. (https://ieeexplore.ieee.org/document/8374491/)
**Relevant to:** QR worker
**Worker's current problem:** LARFT (building the T matrix) uses Level-2 BLAS in standard implementations. Need a GPU-friendly alternative.

## What This Is

An alternative representation to the standard compact WY transform that replaces all Level-2 BLAS operations with Level-3 BLAS for constructing the accumulation of Householder reflectors. This is the foundational work that the Zou et al. 2025 GPU-centered paper built upon.

## Why It Matters for Us

The standard compact WY construction (LAPACK's LARFT) builds T column-by-column using GEMV and TRMV -- Level-2 BLAS that are memory-bound on GPU. This paper showed that an alternative formulation using Level-3 BLAS achieves **20-40% speedup on GPU** compared to MAGMA's implementation (as of 2018).

The key benefit for our monolithic kernel approach: Level-3 BLAS (GEMM) operations map to tensor cores, while Level-2 operations (GEMV, TRMV) do not. Every BLAS-2 operation in the T construction is a wasted opportunity to use mma.sync.

## Key Technique

### Standard compact WY (Level-2, sequential):
```
T(1,1) = tau(1)
For j = 2..b:
    z = V(:, 1:j-1)^T * V(:, j)     // GEMV: (j-1) x m dot products
    z = -T(1:j-1, 1:j-1) * z        // TRMV: triangular multiply
    T(1:j-1, j) = z * tau(j)
    T(j,j) = tau(j)
```
Total: b-1 GEMV + b-1 TRMV operations, sequential.

### Alternative Level-3 representation:
```
// Split reflectors into two halves: V1 (first b/2), V2 (last b/2)
// Recursively build T1, T2 for each half
// Cross-term: T12 = V1^T * V2     // GEMM: (b/2 x m) * (m x b/2)
//             T12 = -T1 * T12     // TRMM
//             T12 = T12 * T2      // TRMM
```

This is the recursive LARFT already described in our existing `recursive_qr_tensor_cores.md`, but the Tomás/Quintana-Ortí contribution is specifically showing that this formulation:
1. Removes T construction from the critical path
2. Enables larger block sizes (because T is built via GEMM, which scales better)
3. Allows moving T construction entirely to GPU

### Performance advantage from larger block sizes:
- Standard LARFT with BLAS-2: optimal nb is small (32-64) because GEMV doesn't scale
- Level-3 LARFT with GEMM: larger nb (128-256) becomes viable because GEMM utilization improves with size
- Larger nb means fewer outer loop iterations and larger trailing update GEMMs (better for tensor cores)

## Application to sm_120

### For the QR worker:
1. Use the recursive LARFT formulation (already planned from existing docs)
2. The V1^T * V2 GEMM inside LARFT can use BF16 mma.sync for the inner product computation, with FP32 accumulation
3. For b=64: the top-level cross-term GEMM is (32 x m) * (m x 32) = 32 x 32 output. The inner dimension m is large enough for tensor cores.
4. The TRMM operations on T (b x b) are small -- use scalar FP32 or our TRMM primitive

### Combined with modified CWY:
The Zou et al. approach (T^{-1} = Y^T * Y) is a further simplification of this Level-3 idea. Both achieve BLAS-3-only T construction. Choose between them:
- **Recursive LARFT**: Standard T, use TRMM in trailing update. More numerically stable.
- **Modified CWY (T^{-1})**: Simpler (one GEMM instead of recursive tree), use TRSM in trailing update. Potentially less stable for ill-conditioned problems.

For our sm_120 implementation, recommend starting with the modified CWY (simpler) and falling back to recursive LARFT if numerical issues arise.

## Caveats

1. **2018 results**: The 20-40% speedup was on 2018-era GPUs (Pascal/Volta). On sm_120 with tensor cores, the BLAS-3 advantage should be even larger because tensor cores accelerate GEMM but not GEMV.

2. **Interaction with recursive QR**: When using recursive QR (HRQR), the T matrix construction is integrated into the recursion. The Level-3 LARFT is most relevant for the standard blocked QR outer loop. If using fully recursive QR, the T merge step already uses GEMM naturally.
