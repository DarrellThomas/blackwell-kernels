# MAGMA Batched LU Panel Factorization Techniques

**Source:** "Progressive Optimization of Batched LU Factorization on GPUs" (Abdelfattah, Tomov, Dongarra, 2018), https://www.netlib.org/utk/people/JackDongarra/PAPERS/icl-utk-1237-2018.pdf
**Relevant to:** LU worker
**Worker's current problem:** Building v1 blocked LU. Need to understand panel factorization techniques for GPU to eventually build a monolithic kernel that beats cuSOLVER.

## What This Is

MAGMA's progressive optimization of batched LU factorization, showing three levels of
blocking with increasing data reuse. While the paper targets batched small matrices,
the panel factorization techniques directly apply to our single-matrix N=4096 case
because the panel kernel is reused at every step of the blocked algorithm.

## Why It Matters for Us

Our Cholesky experience (0.55x cuSOLVER) showed that many-kernel-launch approaches
cannot beat cuSOLVER's monolithic kernel. The LU panel factorization is the critical
component — it's where pivoting happens and where most of the memory-bound work lives.
Getting the panel kernel right is prerequisite to building a competitive monolithic
kernel.

## Key Technique: Three Levels of Blocking

### Level 1: Column Blocking (10-40% panel speedup)

Cache one column of the panel in shared memory. Fuse IDAMAX + DSWAP + DSCAL + DGER
into a single kernel per column.

```
For each column i in panel:
  1. Read column i from global → shared memory
  2. IDAMAX: find pivot (max absolute value) in column
  3. DSWAP: exchange pivot row with row i (in shared memory)
  4. DSCAL: scale column below diagonal by 1/A[i,i]
  5. DGER: rank-1 update of trailing submatrix
  6. Write column back to global
```

**Limitation:** One thread per element, so limited by max threads (1024). Shared memory
is O(m) per column. Works for any panel height.

### Level 2: Panel Blocking (1.5-4.8x over column blocking)

Cache the ENTIRE m×nb panel in **registers**. Each thread holds one complete row
of the panel (nb values). All DGETF2 operations fused into ONE kernel — the panel
is read once and written once to global memory.

```
Architecture:
  - m threads per thread block (one per row)
  - Each thread holds nb register values (one row of panel)
  - nb is a compile-time template parameter (C++ templates)
  - Recursive panel (DGETRF2): sub-divide wide panels

Key detail: "lazy pivoting" — delay ALL row interchanges to the very end
of the kernel. Write the factored panel to global memory only once, with
all pivots already applied.
```

**Register budget:** nb values per thread in FP32 = nb×4 bytes. For nb=32, that's
128 bytes = 32 registers. Maximum panel size limited by register file:
- V100: 255 regs/thread → nb ≤ ~60 (leaving room for control)
- RTX 5090: 255 regs/thread → similar limit
- Largest panel empirically: 512×32 (512 rows × 32 columns)

**This is the key technique for our panel kernel.**

### Level 3: Matrix Blocking (for tiny matrices only, M=N≤32)

Cache the ENTIRE matrix in registers. One thread per row, each holds N values.
The full LU factorization runs in a single kernel with zero global memory traffic
during computation. ~2x faster than panel blocking for N≤32.

**Limited to N ≤ warp size (32)** because the pivot search (IDAMAX) requires
cross-thread communication, which is efficient only within a warp via shuffles.
For N>32, need __syncthreads which breaks the register-only model.

## Critical Implementation Details

### Lazy Pivoting

Standard LU applies row swaps immediately at each step. Lazy pivoting defers ALL
swaps to the end:

```
Standard:              Lazy:
for i in 0..nb:        for i in 0..nb:
  find pivot             find pivot
  SWAP rows(i, piv)      record piv[i]    ← just record, don't swap
  DSCAL, DGER            apply piv to REGISTERS only (not global)
                          DSCAL, DGER (using pivoted register state)
                        // End of kernel:
                        apply ALL pivots when writing back to global
```

This eliminates nb global memory row swaps during panel factorization. Only one
read and one write of the panel to global memory.

### IDAMAX (Pivot Search) on GPU

Finding the maximum absolute value in a column requires a parallel reduction:
- Within a warp: use `__shfl_xor_sync` for butterfly reduction (fast, no shared memory)
- Across warps: use shared memory reduction (need __syncthreads)
- For register-resident panel: each thread has the column value in its register,
  reduce across threads

**Critical detail:** IDAMAX must return both the VALUE and the INDEX (row position)
of the maximum. The shuffle-based reduction carries both.

### Parallel Row Swaps (DLASWP)

Row swaps outside the panel (swap-left and swap-right in Algorithm 2) are done via
a "parallel swapping" kernel: multiple row pairs swapped simultaneously by different
thread groups.

### Recursive Panel (DGETRF2)

For nb > some threshold (e.g., nb > 8), recursively split the panel:
```
DGETRF2(A[0:m, 0:nb]):
  DGETRF2(A[0:m, 0:nb/2])          // factor left half
  DLASWP(A[0:m, nb/2:nb], pivots)  // apply pivots to right half
  DTRSM(A[0:nb/2, nb/2:nb])        // triangular solve
  DGEMM(A[nb/2:m, nb/2:nb] -= ...)  // rank-k update
  DGETRF2(A[nb/2:m, nb/2:nb])      // factor right half (recurse)
```

This makes the compute-bound GEMM/TRSM a larger fraction of the work.

## Performance Results (V100, batch=500)

| Matrix size | vs generic LAPACK-style | vs cuBLAS |
|-------------|------------------------|-----------|
| N=32 | 21.1x (single) | 8.72x (single) |
| N=100 | 3.28x | ~4x |
| N=500 | 1.6x | ~2x |
| N=1600 | ~1.1x | ~1.5x |

Key insight: **the smaller the matrix, the larger the speedup** because panel
factorization (memory-bound) dominates for small sizes, and register-caching
eliminates nearly all memory traffic.

## Time Breakdown by Component (Fig. 4 in paper)

| Matrix size | GEMM % | TRSM % | Rank-1 % | Swap % | IDAMAX % |
|-------------|--------|--------|----------|--------|----------|
| 50 | ~5% | ~5% | ~40% | ~30% | ~20% |
| 100 | ~10% | ~5% | ~35% | ~30% | ~20% |
| 500 | ~40% | ~10% | ~25% | ~15% | ~10% |
| 2000 | ~70% | ~10% | ~10% | ~5% | ~5% |

**For N=4096 (our case): GEMM dominates (~80%+).** The panel factorization is a
smaller fraction, but it's still the serializing bottleneck (panel must complete
before the trailing GEMM update can start).

## Applicability to Our N=4096 Single-Matrix Case

The paper targets **batched** small matrices. For our **single large** N=4096 case:

1. **Panel kernel techniques apply directly.** The blocked LU algorithm's inner
   panel (m×nb where m shrinks each step and nb=32-64) uses exactly these techniques.
   Register-resident panel with lazy pivoting for the DGETF2 step.

2. **GEMM dominates.** At N=4096, the trailing update DGEMM is ~80% of compute.
   We already have a 0.97x cuBLAS GEMM and 1.29x FP8 GEMM — these are assets.

3. **The monolithic kernel problem remains.** Even with optimal panel + our fast
   GEMM, the kernel launch overhead of blocked LU (panel → swap → TRSM → GEMM,
   repeated ~64 times for NB=64) will likely exceed cuSOLVER's monolithic approach.

4. **Path forward:** Build the blocked approach first (v1-v3), measure where time
   goes, then consider a monolithic kernel that fuses the panel + TRSM + GEMM
   loop into a single persistent kernel.

## Caveats

1. **V100 results, not RTX 5090.** The relative speedups may differ on sm_120
   due to different memory hierarchy and register file sizes.

2. **FP32 required for pivoting.** BF16 loses too much precision for pivot
   selection. The panel must be FP32. The trailing GEMM update could potentially
   use mixed precision (BF16 MMA with FP32 accumulators) for speedup.

3. **TF32 MMA B fragment broadcasting on sm_120** (from Cholesky lessons) means
   TF32 tensor cores can't be used for the trailing GEMM. Use BF16 MMA with
   FP32→BF16 conversion instead.
