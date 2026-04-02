# Batched Small Cholesky: Where We Can Beat cuSOLVER

**Source:** MAGMA potrf_batched (https://icl.utk.edu/magma/), "A Fast Batched Cholesky Factorization on a GPU" (IEEE 2014), "Fast Cholesky factorization on GPUs for batch and native modes in MAGMA" (2017)
**Relevant to:** numerical worker (Cholesky)
**Worker's current problem:** 0.55x cuSOLVER at N=4096 for single large matrix. The monolithic kernel gap is insurmountable. Batched small Cholesky may be a winnable target.

## What This Is

For batches of many small SPD matrices (N=16-64), cuSOLVER's overhead-per-matrix
becomes the dominant cost. A single fused kernel that processes all matrices in one
launch can dramatically beat cuSOLVER. MAGMA achieves within 90% of optimal for
batched factorizations of matrices N≤32.

## Why It Matters for Us

Our panel kernel (64×64, 1 block, 256 threads, sub-blocked IB=16) is already
competitive for small matrices — the factorization time at N=64 is ~33 μs. The
problem was always the trailing TRSM+SYRK overhead. For batched operations, we
can assign one block per matrix and eliminate ALL inter-kernel launch overhead.

Applications that need batched small Cholesky:
- **Gaussian Processes:** Kernel matrix Cholesky for GP inference/training
- **Kalman filters:** State covariance updates (N=6 to N=100)
- **Block-diagonal preconditioners:** Incomplete Cholesky for iterative solvers
- **Monte Carlo methods:** Sampling from multivariate normals
- **Computer vision:** Covariance estimation in feature spaces

## Key Technique: Register-Resident Batched Cholesky

### Architecture

```
Grid: batch_count blocks
Block: N threads (one thread per matrix row)
Each thread holds one row of the matrix in registers (N values)
```

### Algorithm (in-register)

```
for j = 0 to N-1:
  // Step 1: Compute L[j,j] = sqrt(A[j,j] - sum(L[j,0:j]^2))
  // Thread j computes the diagonal element
  // Broadcast via __shfl_sync to all threads

  // Step 2: Compute L[i,j] for i > j
  // L[i,j] = (A[i,j] - sum(L[i,0:j]*L[j,0:j])) / L[j,j]
  // Each thread i>j computes its own L[i,j] (has both rows in registers)

  // Step 3: Update trailing matrix (implicit — values stay in registers)
```

### Register Budget

Each thread holds N FP32 values in registers:
- N=32: 32 regs for data + ~10 control = 42 regs/thread → 255 max → OK
- N=64: 64 regs for data + ~10 control = 74 regs/thread → OK but lower occupancy
- N=128: 128 regs + control = ~140 regs → still fits but 1 block/SM

### Communication

All communication is within a warp (N≤32) or block:
- **Pivot broadcast:** Not needed for Cholesky (no pivoting for SPD matrices!)
- **Diagonal broadcast:** Thread j broadcasts L[j,j] to all threads via `__shfl_sync`
- **Row broadcast:** Thread j broadcasts L[j,0:j] for the dot product
  - For N≤32: `__shfl_sync` (warp-level, no shared memory needed)
  - For N>32: shared memory broadcast (one __syncthreads)

### Why This Beats cuSOLVER

| Factor | cuSOLVER batched | Our kernel |
|--------|-----------------|------------|
| Kernel launches | 1 per batch, but generic | 1, specialized |
| Matrix in memory | Global memory, cached | **Registers** (zero latency) |
| Inter-step sync | Implicit (single thread per col) | Warp shuffle or block sync |
| Occupancy | Low (generic kernel) | High (small register count per thread) |

For N=32, batch=10000:
- cuSOLVER: ~0.5-1.0 ms (dominated by per-matrix overhead within kernel)
- Fused register-resident: estimated ~0.05-0.1 ms (10-20x speedup possible)

## Implementation Steps

1. **Start with N=32 (warp-level).** One warp per matrix. All communication via
   shuffles. No shared memory. This is the simplest and should show the largest
   speedup.

2. **Extend to N=64 (block-level).** Two warps per matrix. Use shared memory for
   cross-warp communication (diagonal broadcast, row broadcast).

3. **Benchmark against `cusolverDnSpotrfBatched`.** Target: 5x cuSOLVER for
   N=32, batch=1000-10000.

4. **Consider mixed precision.** BF16 MMA for the SYRK-equivalent step (row dot
   products) with FP32 accumulators. For N=32, the dot products are short
   (≤32 elements) so the MMA advantage is limited.

## Caveats

1. **SPD requirement.** Cholesky only works for symmetric positive definite matrices.
   Non-SPD inputs cause sqrt of negative number → NaN. Need to handle this
   (return error flag per matrix, or clamp to zero).

2. **Numerical stability.** Register-resident FP32 Cholesky is numerically stable
   for well-conditioned matrices. For ill-conditioned matrices (condition number
   > 10^6), consider iterative refinement or double precision.

3. **The N≤32 sweet spot.** For N>32, warp shuffles can't reach all threads.
   Need shared memory, which adds latency. The MAGMA paper shows the performance
   advantage drops significantly for N>32.

4. **Our panel kernel (NB=64, IB=16) might already be competitive for N=64.**
   Measure cuSOLVER batched at N=64 before building a new kernel. The win may
   be smaller than for N=32.
