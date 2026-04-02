# Mixed-Precision Recursive Cholesky with Tensor Cores

**Source:** https://arxiv.org/html/2601.08082v1 | https://link.springer.com/chapter/10.1007/978-3-030-50417-5_18
**Relevant to:** cholesky worker (new kernel)
**Worker's current problem:** How to leverage sm_120 tensor cores (mma.sync) for Cholesky, given that the core algorithm (potf2) is not GEMM-shaped.

## What This Is

A 2026 paper showing how to recursively decompose ALL three Cholesky sub-operations (POTRF, TRSM, SYRK) to maximize GEMM content, then use tensor cores (FP16/BF16 MMA) for the GEMM-heavy parts while keeping critical diagonal operations in FP32/FP64. Achieves 5.3x speedup over cuSOLVER FP64 Cholesky on H200.

## Why It Matters for Us

Standard blocked Cholesky only uses tensor cores for the trailing SYRK+GEMM update (~85% of FLOPs). This paper shows how to get tensor cores into TRSM and even parts of POTRF through recursive decomposition — pushing tensor core utilization to 90%+ of total FLOPs. On sm_120, this means our mma.sync BF16 GEMM (0.97x cuBLAS) and FP8 GEMM (1.34x cuBLAS) become the workhorses for almost the entire factorization.

## Key Technique

### Recursive Decomposition

Instead of standard blocked POTRF → TRSM → SYRK, recursively split each operation:

**Recursive POTRF:**
```
function potrf(A, n):
    if n <= nb_base:          // base case: unblocked potf2
        return potf2(A)
    split A into [A11, A21; _, A22]  // half-size blocks
    potrf(A11)                        // recurse on top-left
    trsm(A11, A21)                    // solve bottom-left
    syrk(A21, A22)                    // update bottom-right: A22 -= A21 * A21^T
    potrf(A22)                        // recurse on bottom-right
```

**Recursive TRSM (novel):**
```
function trsm(L, B):
    if n <= nb_base:          // base case: vendor TRSM
        return cublas_trsm(L, B)
    split L, B into halves
    trsm(L11, B1)             // solve top half
    gemm(B2 -= L21 * B1)     // ← TENSOR CORE GEMM
    trsm(L22, B2)             // solve bottom half
```

**Recursive SYRK (first GPU implementation):**
```
function syrk(A, C):
    if n <= nb_base:
        return cublas_syrk(A, C)
    split A, C into halves
    syrk(A1, C11)             // recurse diagonal
    gemm(C21 -= A2 * A1^T)   // ← TENSOR CORE GEMM (off-diagonal)
    syrk(A2, C22)             // recurse diagonal
```

### Mixed Precision Strategy

| Operation | Precision | Why |
|-----------|-----------|-----|
| POTRF diagonal (potf2 base case) | FP32 | Numerical stability — sqrt of near-zero values |
| TRSM base case | FP32 | Triangular solve accuracy |
| GEMM updates (from recursive TRSM/SYRK) | BF16 → FP32 accumulation | Tensor core throughput |
| Off-diagonal GEMM (main trailing update) | BF16 → FP32 accumulation | Tensor core throughput |

**Per-block scaling for BF16:** Before converting FP32 → BF16 for GEMM, compute `α = max(1, ||B||_∞ / 65504)` and scale down to prevent overflow. Scale back after GEMM. This is similar to FP8 per-tensor scaling.

### Performance Results (H200)

| Component | Speedup vs cuSOLVER FP64 |
|-----------|--------------------------|
| Recursive SYRK (FP64) | 14x |
| Recursive SYRK (mixed FP16) | 27x |
| Recursive TRSM (FP64) | 6x |
| Recursive TRSM (mixed FP16) | 5.3x |
| **Full Cholesky** | **5.3x** |

## Adaptation for sm_120

On our RTX 5090 (sm_120, mma.sync):

1. **Replace GEMM calls with our mma.sync kernel.** The recursive decomposition generates many small-to-medium GEMMs — our 64×64 tile GEMM with 6 blocks/SM is well-suited.

2. **BF16 for off-diagonal updates.** Our BF16 GEMM at 0.97x cuBLAS means the recursive GEMM calls run at near-peak tensor throughput.

3. **FP8 for off-diagonal updates (aggressive).** If accuracy permits, our FP8 GEMM at 1.34x cuBLAS would give even more speedup. The per-block scaling in MXFP8 could help maintain accuracy.

4. **FP32 for diagonal/panel.** Use standard FP32 arithmetic for potf2 and TRSM base cases. No tensor cores needed for these (~10% of total FLOPs).

5. **SYRK as half-GEMM.** The recursive SYRK generates standard GEMM calls for off-diagonal blocks. The diagonal blocks are symmetric rank-k updates that can use our GEMM kernel with only the lower-triangle output tiles computed (skip upper-triangle blocks in the grid).

## Caveats

- **This paper targets H200/MI300X datacenter GPUs.** The recursive decomposition generates many concurrent GEMMs that benefit from massive SM counts (132 on H200 vs 170 on RTX 5090). Should work on sm_120 — we actually have MORE SMs.
- **Recursion depth matters.** Too deep = too many tiny GEMMs with launch overhead. Too shallow = not enough tensor core utilization. Need to tune `nb_base` (the recursion base case size) empirically. Start with nb_base=32 or 64.
- **The accuracy tradeoff:** Mixed-precision Cholesky with BF16 updates is NOT exact. The residual ||A - L*L^T|| is larger than FP64 Cholesky. For most ML applications (covariance matrices, kernel methods, GP regression), this is fine. For scientific computing requiring FP64 accuracy, iterative refinement is needed.
- **No existing sm_120 implementation.** We would be the first (that we found) to implement recursive mixed-precision Cholesky with mma.sync tensor cores on consumer Blackwell. High novelty but also high risk.
- **TRSM kernel needed.** We don't currently have a triangular solve kernel. The recursive approach converts most of TRSM into GEMM, but the base case still needs a TRSM implementation (simpler than a full optimized TRSM since it's only for small tiles).
