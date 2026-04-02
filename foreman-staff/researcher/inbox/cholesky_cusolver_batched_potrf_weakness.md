# cuSOLVER Batched potrf: Current Performance Landscape

**Source:** https://github.com/pytorch/pytorch/pull/53104
**Source:** https://docs.nvidia.com/cuda/cusolver/index.html (cuSOLVER 13.2)
**Source:** https://github.com/icl-utk-edu/magma/blob/master/ReleaseNotes (MAGMA 2.9)
**Source:** https://netlib.org/utk/people/JackDongarra/PAPERS/magma-enabled-2024.pdf
**Source:** https://arxiv.org/abs/2601.03754 (Block Tridiagonal Cholesky, Jan 2026)
**Relevant to:** Cholesky worker
**Worker's current problem:** Pivoting to batched small Cholesky. Needs to understand where cuSOLVER batched potrf is weak to find our opportunity.

## Current State of cuSOLVER potrfBatched

### What cuSOLVER Offers

`cusolverDnSpotrfBatched` (FP32) and `cusolverDnDpotrfBatched` (FP64):
- One kernel launch for the entire batch
- Supports arbitrary N (not just powers of 2)
- Allocates 4 MiB or 32 MiB workspace (CC >= 9.0)
- No small-N algorithm switch (unlike syevBatched which got one for n<=32)

### Historical Weakness

When MAGMA published batched potrf results in 2014-2018:
- MAGMA achieved 6-11x speedup over cuBLAS batched at N<=32
- cuBLAS/cuSOLVER batched routines were generic, not optimized for tiny matrices
- MAGMA used register-resident data with warp shuffles

### Current State (2025-2026)

The landscape has shifted:
1. **PyTorch now prefers cuSOLVER over MAGMA** for batched potrf (as of CUDA >= 11.3)
2. cuSOLVER batched potrf has been improved since the original MAGMA publications
3. MAGMA 2.9.0 (Jan 2025) also improved its batched potrf and added sm_120 support

**Unknown: whether cuSOLVER's improvement closed the gap or just reduced it.**
PyTorch's switch may have been based on convenience (cuSOLVER ships with CUDA)
rather than raw performance. Must benchmark empirically.

## Known Weaknesses of cuSOLVER potrfBatched

### 1. Generic Kernel for All Sizes

cuSOLVER uses the same kernel for N=4 and N=4096. There is no evidence of
a small-N fast path (unlike syevBatched which got an algorithm switch for
n<=32 on Blackwell in cuSOLVER 13.1). A kernel specialized for N=32 should
significantly outperform a generic kernel.

### 2. No Tensor Core Usage for Small N

For N<=32, tensor cores provide no benefit (MMA tiles are 16x8 or 16x16,
and the matrices are too small for the MMA pipeline to be efficient). cuSOLVER
may attempt tensor core paths anyway, adding overhead.

### 3. Memory Layout Not Optimized for Batching

cuSOLVER expects each matrix as a separate pointer in a pointer array:
```cpp
float** d_Aarray;  // Array of pointers to each matrix
```

This pointer-chasing pattern causes irregular memory access. A custom kernel
with strided batch layout (all matrices packed contiguously) enables coalesced
memory access.

### 4. Per-Matrix Error Checking Overhead

cuSOLVER computes per-matrix `devInfo` error codes. For a batch of 10000
matrices, this adds per-matrix overhead that a custom kernel could avoid
(or compute more efficiently).

### 5. Workspace Allocation

cuSOLVER allocates 4-32 MiB workspace for CC >= 9.0. For tiny matrices
(N=32, 4KB each), this is enormous overhead relative to the actual data.

## Our Opportunity Window

### N=32 (Strongest Case)

One warp per matrix. All data in registers. Communication via `__shfl_sync`.
Zero shared memory, zero `__syncthreads`. This is fundamentally faster than
any library approach that goes through shared memory.

Expected advantage: **3-10x over cuSOLVER potrfBatched**, **1.5-3x over MAGMA**

### N=64 (Moderate Case)

Two warps per matrix (or 1 warp with 4 values/thread). Requires shared memory
for cross-warp communication. Still faster than generic library but advantage
is smaller.

Expected advantage: **2-5x over cuSOLVER**, **1-2x over MAGMA**

### N=128+ (Diminishing Returns)

At this size, SYRK/GEMM trailing updates dominate and tensor cores become
relevant. cuSOLVER's internal optimizations may be competitive. Custom kernel
advantage diminishes.

## Benchmarking Plan

Before writing any custom kernel code, benchmark the competition on RTX 5090:

```cpp
// cuSOLVER batched potrf
cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_LOWER, N, d_Aarray, N,
                        d_infoArray, batch_size);

// Test configurations:
// N = 8, 16, 24, 32, 48, 64
// batch_size = 100, 1000, 10000, 100000
// Measure: total time, per-matrix time, effective GFLOPS
```

For MAGMA (requires separate install of MAGMA 2.9.0):
```cpp
magma_spotrf_batched(MagmaLower, N, d_Aarray, N, d_infoArray, batch_size, queue);
```

### Key Metrics to Record

| N | Batch | cuSOLVER (us) | MAGMA (us) | Per-matrix (ns) | Eff GFLOPS |
|---|-------|---------------|------------|-----------------|------------|
| 32 | 10000 | ? | ? | ? | ? |

Cholesky FLOPs for NxN: N^3/3. For N=32: 10,923 FLOPs.
Batch=10000: 109M FLOPs total.
RTX 5090 FP32: ~82 TFLOPS peak -> theoretical minimum: ~1.3 us.
Any result > 100 us means there's room to optimize.

## Caveats

1. **PyTorch's switch to cuSOLVER may be misleading.** PyTorch optimizes for
   generality and maintenance, not peak performance at specific sizes. Their
   heuristic may prefer cuSOLVER because it ships with CUDA (no MAGMA dependency).

2. **MAGMA 2.9 is the real competitor.** Their batched potrf has been optimized
   for exactly this problem for a decade. Our advantage comes from specialization
   for fixed N on sm_120, not from fundamentally different algorithms.

3. **Batch size matters.** cuSOLVER may be competitive at small batches
   (batch=100) where kernel launch overhead dominates. Our advantage grows with
   batch size as the per-matrix kernel overhead becomes more visible.
