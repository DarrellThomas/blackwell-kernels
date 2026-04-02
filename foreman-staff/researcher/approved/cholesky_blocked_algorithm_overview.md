# Blocked GPU Cholesky Factorization — Algorithm Overview

**Source:** https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html | https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__potrf.html | https://arxiv.org/html/2601.03754v1
**Relevant to:** cholesky worker (new kernel)
**Worker's current problem:** Starting from scratch — needs algorithmic foundation for GPU Cholesky on sm_120.

## What This Is

Dense Cholesky factorization (potrf) decomposes a symmetric positive-definite matrix A into A = L * L^T (lower triangular). On GPU, this is implemented as a **blocked algorithm** that decomposes the work into four BLAS operations on tiles, maximizing tensor core utilization for the compute-heavy parts.

## Why It Matters for Us

This is the fundamental algorithm the worker needs to implement. Every high-performance GPU Cholesky (cuSOLVER, MAGMA, KBLAS) uses this structure. Understanding it is prerequisite to writing the kernel.

## Key Technique

### The Blocked Algorithm (tile Cholesky)

The matrix is divided into Nt × Nt tiles of size nb × nb. The algorithm proceeds:

```
for k = 0 to Nt-1:
    // 1. POTRF: Cholesky factorize the diagonal tile
    A[k][k] = potrf(A[k][k])                          // nb×nb unblocked Cholesky

    // 2. TRSM: Solve triangular system for column tiles below diagonal
    for m = k+1 to Nt-1:
        A[m][k] = trsm(A[k][k], A[m][k])              // A[m][k] = A[m][k] * L[k][k]^{-T}

    // 3. SYRK + GEMM: Update trailing matrix
    for m = k+1 to Nt-1:
        A[m][m] = syrk(A[m][k], A[m][m])               // A[m][m] -= A[m][k] * A[m][k]^T
        for n = k+1 to m-1:
            A[m][n] = gemm(A[m][k], A[n][k], A[m][n])   // A[m][n] -= A[m][k] * A[n][k]^T
```

### Operation Breakdown per Step

| Operation | Compute | Parallelism | GPU Mapping |
|-----------|---------|-------------|-------------|
| POTRF (diagonal tile) | nb³/3 FLOPs | Sequential (data-dependent) | Single SM, shared memory |
| TRSM (column tiles) | nb³ FLOPs each | Independent across rows | One block per tile |
| SYRK (diagonal updates) | nb³ FLOPs each | Independent across tiles | Tensor core GEMM |
| GEMM (off-diagonal updates) | 2×nb³ FLOPs each | Fully parallel | Tensor core GEMM |

### Left-Looking vs Right-Looking

- **Right-looking (LAPACK-style):** After factorizing panel k, immediately update ALL trailing tiles. Exposes maximum parallelism in SYRK/GEMM. Better for GPU.
- **Left-looking (cuSOLVERDx-style):** Before factorizing panel k, apply ALL pending updates to it. Better data locality, less memory traffic. Used for out-of-core/batched.

**For a single large matrix on GPU, right-looking is preferred** because the trailing update (SYRK+GEMM) is the compute-heavy part that maps to tensor cores — and right-looking maximizes its parallelism.

### Where the Time Goes

For an N×N matrix with tile size nb:
- **POTRF panels:** O(N × nb²) total — sequential, ~5% of time for large N
- **TRSM solves:** O(N² × nb) total — moderate parallelism, ~10%
- **SYRK+GEMM updates:** O(N³) total — massive parallelism, **~85% of time**

The trailing matrix update dominates and is pure GEMM — exactly what tensor cores excel at. The panel factorization (potrf/potf2) is the serial bottleneck.

### Typical Tile Sizes

| Implementation | Tile nb | Notes |
|---------------|---------|-------|
| MAGMA (FP64) | 64-256 | Auto-tuned per matrix size |
| MAGMA (FP32) | 128-256 | Larger tiles for f32 |
| cuSOLVERDx | configurable | Left-looking blocked |
| Block tridiagonal (2026) | 8-32 | Smaller for fused kernels |
| Our GEMM tile | 64 | Natural starting point |

**Recommendation for sm_120:** Start with nb=64 (matches our GEMM tile infrastructure). The GEMM/SYRK trailing updates reuse our existing GEMM kernel. The critical new code is the potf2 panel kernel and the TRSM solve.

## Caveats

- **The panel factorization (potf2) is fundamentally sequential.** Each column depends on all previous columns. This is the Amdahl's law bottleneck for GPU Cholesky. All optimization efforts focus on (a) making potf2 fast despite being sequential, and (b) overlapping it with the parallel trailing update.
- **SYRK is not the same as GEMM.** SYRK computes C -= A * A^T (symmetric output). It does roughly half the work of a full GEMM because only the lower triangle is computed. Implementing SYRK as a full GEMM wastes ~50% of tensor core cycles. A dedicated SYRK kernel or a GEMM that skips upper-triangle tiles is needed.
- **Numerical stability requires FP32 or FP64 for the diagonal.** BF16/FP8 may work for off-diagonal GEMM updates (the mixed-precision approach) but the diagonal potf2 must use at least FP32 to avoid negative pivots from rounding.
- **For small matrices (N < 512),** a single-SM fused kernel that keeps the entire matrix in shared memory is faster than the blocked approach. The crossover depends on shared memory capacity (99 KB on sm_120).
