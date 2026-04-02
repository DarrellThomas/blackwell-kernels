# Modified CWY: GPU-Centered T-Inverse Formulation for QR

**Source:** Zou, Leng, Wang, Wu, Zhang. "Efficient GPU-Centered SVD Using Divide-and-Conquer." arXiv:2508.11467, May 2025. (https://arxiv.org/html/2508.11467v1)
**Relevant to:** QR worker
**Worker's current problem:** Building blocked Householder QR. Needs efficient LARFT (T matrix construction) that works on GPU without CPU-GPU transfers.

## What This Is

The Zou et al. 2025 paper introduces a modified Compact WY (CWY) formulation that replaces the standard LARFT construction with a purely BLAS-3 approach. Instead of building T column-by-column via sequential BLAS-2 operations (GEMV + TRMV), they compute T^{-1} directly as a single GEMM, then use TRSM in the trailing update.

## Why It Matters for Us

Standard LARFT (as in LAPACK) builds T using (b-1) iterations of GEMV and TRMV -- these are BLAS-2, memory-bound, and sequential. For a monolithic kernel pattern (single thread block), every BLAS-2 operation is a serialization point. The modified CWY eliminates this entirely.

The paper reports **outperforming both cuSOLVER and MAGMA for geqrf across all tested matrix sizes** on V100 and AMD MI210. The speedup over MAGMA is especially large for tall-skinny matrices (where panel fraction is high).

## Key Technique

### Standard LARFT (what LAPACK does):
```
For each reflector j = 1..b:
    T(1:j-1, j) = -T(1:j-1, 1:j-1) * V(:, 1:j-1)^T * V(:, j)   // GEMV + TRMV
    T(j, j) = tau(j)
```
This is b sequential steps of BLAS-2 operations.

### Modified CWY (what the paper does):
```
Step 1: Compute T^{-1} = Y_b^T * Y_b           // GEMM: (b x m) * (m x b) -> b x b
Step 2: Set diag(T^{-1}) = 1/tau_1, ..., 1/tau_b  // Fix diagonal
```

### Modified trailing update:
```
Z = Y^T * A_trailing        // GEMM (b x m_remain) * (m_remain x n_trail)
Z = T * Z                   // Solve via TRSM on T^{-1}: T^{-1} * Z_new = Z_old
A_trailing -= Y * Z         // GEMM (m_remain x b) * (b x n_trail)
```

The key insight: replacing TRMM(T, W) with TRSM(T^{-1}, Z) has the same complexity for the small b x n_trail system, but the T^{-1} computation itself is a single b x b GEMM instead of sequential BLAS-2.

### Why T^{-1} = Y^T * Y works:
For Householder reflectors H_i = I - tau_i * v_i * v_i^T stored as columns of Y with unit lower-triangular structure, the product T^{-1} = Y^T * Y naturally produces the correct inverse of the standard T matrix. The diagonal must be corrected to 1/tau_i because Y's diagonal entries are 1 (unit reflectors) not tau_i.

## Performance Impact

From the paper (Figure 14, m=20000):
- **Consistently outperforms cuSOLVER and MAGMA for geqrf across all n values**
- Speedup over MAGMA decreases as n increases (MAGMA's trailing GEMM catches up for square matrices)
- **Most suitable for taller-and-skinnier matrices** where the panel fraction is higher
- On MI210: better BLAS-3 performance makes the advantage larger

The paper identifies a key MAGMA bottleneck: for tall-skinny matrices, MAGMA transfers the trailing part of Q (size (n%b + (m-n))^2) to CPU for panel factorization, incurring "significant overhead when m >> n."

## Application to sm_120

### What to implement:
1. **T^{-1} computation**: A b x b GEMM of Y^T * Y. For b=64 with m=4096, this is (64 x 4096) * (4096 x 64) = 64 x 64 output. Small but compute-intensive inner dimension.
   - Can use our BF16 MMA for this GEMM if precision is acceptable (T is small, FP32 may be safer)
   - Alternatively, compute in shared memory with FP32 -- 64x64 FP32 is only 16 KB

2. **Trailing update with TRSM**: Replace TRMM(T, W) with TRSM(T^{-1}, Z). T^{-1} is b x b upper triangular. Our TRSM primitive (from linalg/) can handle this.

3. **GPU-only panel**: The paper does the entire panel factorization on GPU, eliminating CPU-GPU transfers. This matches our monolithic kernel strategy.

### Block size flexibility:
The paper notes an important advantage: "the optimal block size for geqrf is smaller than that for orgqr, which limits orgqr's performance." With modified CWY, orgqr can recompute T^{-1} with a larger block size, decoupling the two.

For our QR worker, this means:
- Use b=32-64 for geqrf (matching cuSOLVERDx panel size limits)
- Can reblock to larger b for trailing updates if needed

## Caveats

1. **TRSM replaces TRMM**: TRSM on a b x n_trail system is slightly more expensive than TRMM (same O() but higher constant). For b=64, the difference is negligible compared to the GEMM savings.

2. **T^{-1} accuracy**: Computing T^{-1} via GEMM and then solving T^{-1} * Z = ... introduces a matrix inverse. For well-conditioned T (which Householder T matrices typically are), this is fine. For very ill-conditioned problems, the standard LARFT may be more stable.

3. **Tested on V100 and MI210, not sm_120**: The algorithmic approach is architecture-independent, but the relative benefit depends on the GPU's BLAS-2 vs BLAS-3 performance gap. On sm_120, BLAS-3 is even more dominant (tensor cores), so the benefit should be at least as large.

4. **This is a refinement, not a revolution**: The modified CWY helps most when the panel (LARFT) fraction is significant. For large square matrices where trailing GEMMs dominate, the impact is smaller. Combined with recursive QR (which already converts tall-skinny GEMMs to square), this addresses the remaining LARFT bottleneck.
