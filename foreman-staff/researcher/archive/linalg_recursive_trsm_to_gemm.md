# Recursive TRSM: Converting Triangular Solve to GEMM Calls

**Source:** https://arxiv.org/html/2504.13821v1 | https://github.com/ecrc/kblas-gpu | https://arxiv.org/html/2601.08082v1
**Relevant to:** linalg worker (linalg/)
**Worker's current problem:** TRSM at 0.82x reference (cuBLAS F32). Next direction listed: "Native BF16 TRSM — multi-SM blocked approach (significant complexity)."

## What This Is

Recursive TRSM converts most of the triangular solve computation into GEMM calls, which are the best-optimized operation on GPU. The recursion: split L and B into halves, solve the top half (small TRSM), update the bottom half via GEMM, then solve the bottom half (small TRSM). At the base case, use a simple column-by-column solver. The GEMM calls dominate compute for large matrices.

## Why It Matters for Us

The linalg worker's TRSM is at 0.82x using `torch.linalg.solve_triangular` with F32 + cast. The recursive approach would:

1. **Leverage our BF16 GEMM (0.97x cuBLAS)** for the dominant compute phase
2. **Leverage our FP8 GEMM (1.34x cuBLAS)** for even more throughput if accuracy permits
3. **Only need a small base-case TRSM** (e.g., 32×32 or 64×64 in shared memory)
4. Avoid the cuBLAS overhead of a general-purpose TRSM implementation

A 2025 paper shows recursive TRSM matching or surpassing cuBLAS on A100, and KBLAS uses the same approach on older architectures.

## Key Technique

### Recursive algorithm (lower triangular, L * X = B):

```
function trsm(L, B, n):
    if n <= nb_base:           // base case: small direct solve
        return small_trsm(L, B)

    // Split into halves
    L11 = L[0:n/2, 0:n/2]     // upper-left triangle
    L21 = L[n/2:n, 0:n/2]     // lower-left rectangle
    L22 = L[n/2:n, n/2:n]     // lower-right triangle
    B1 = B[0:n/2, :]          // top half of RHS
    B2 = B[n/2:n, :]          // bottom half of RHS

    // Step 1: Solve top half (recursive)
    X1 = trsm(L11, B1, n/2)

    // Step 2: Update bottom half via GEMM (tensor cores!)
    B2 = B2 - L21 * X1        // ← THIS IS A GEMM (dominates compute)

    // Step 3: Solve bottom half (recursive)
    X2 = trsm(L22, B2, n/2)

    return [X1; X2]
```

### GEMM fraction of total work:
- At each recursion level, the GEMM is (n/2 × n/2) × (n/2 × nrhs)
- Total GEMM FLOPs ≈ 75-85% of total for large n
- The base-case TRSM (sequential) is only ~15-25%

### Base case implementation (nb_base = 32 or 64):
- Load the small triangular matrix into shared memory
- Column-by-column forward/backward substitution
- Each column: dot product of known values, subtract, divide by diagonal
- Similar structure to potf2 in Cholesky (but simpler — no sqrt)

### Performance expectations:
- The recursive TRSM paper reports matching or surpassing cuBLAS on A100
- With our GEMM at 0.97x cuBLAS (BF16) or 1.34x (FP8), the recursive approach should beat the current 0.82x
- Key advantage: cuBLAS TRSM doesn't use tensor cores for the solve itself. Our approach does (for the GEMM portion)

## Caveats

- **Recursion depth.** Each level halves the problem. For n=4096 with nb_base=64: 6 recursion levels. Each generates a GEMM call — that's 6 kernel launches (or 6 cuBLAS GEMM calls). The launch overhead may matter for small n.
- **Mixed precision accuracy.** If using BF16 GEMM for the update (B2 = B2 - L21*X1), the subtraction in BF16 loses precision. The standard fix: accumulate in FP32 (which our GEMM already does), then write back in BF16.
- **nrhs (number of right-hand sides) matters.** For single-vector TRSM (nrhs=1), the GEMM becomes GEMV and the recursive approach has less advantage. Best for multiple RHS (nrhs >= 16).
- **The base case TRSM kernel is new code.** But it's simpler than potf2 — just forward substitution with shared memory, no sqrt.
- **For the linalg worker's current F32 TRSM path:** The worker delegates to `torch.linalg.solve_triangular` which calls cuSOLVER under the hood. The recursive approach would replace this entirely with our custom code + GEMM kernel.
