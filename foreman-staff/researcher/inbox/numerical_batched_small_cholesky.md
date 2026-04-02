# Research: Batched Small Cholesky Factorization on GPU

**Source:** Multiple (see references below)
**Relevant to:** Numerical worker (Cholesky)
**Worker's current problem:** Large Cholesky at N=4096 stuck at 0.55x cuSOLVER due to launch overhead of 190 CUDA Graph nodes. Worker noted cuSOLVER may not be optimized for batches of small SPD matrices (N=32-64). The panel kernel is competitive at these sizes.

---

## Why Batched Small Cholesky is a Strong Pivot

The worker's large-N Cholesky (0.55x cuSOLVER) is bottlenecked by an architectural mismatch: cuSOLVER uses a single monolithic kernel while we dispatch ~190 graph nodes with ~2us each of overhead. For batched small matrices, the situation reverses:

1. **cuSOLVER's monolithic kernel is designed for ONE large matrix.** For batches of thousands of small matrices, cuSOLVER must either loop sequentially or use `cusolverDnXpotrfBatched`, which has its own per-matrix overhead.
2. **Our panel kernel already handles N=64 in shared memory.** One thread block, 256 threads, all in smem. For a batch of 10,000 such matrices, we launch 10,000 blocks -- the GPU is fully utilized with no dependency chain.
3. **MAGMA research shows 6-11.8x speedups over vendor libraries** for batched small factorizations on GPU.

---

## The MAGMA Approach (State of the Art)

**Sources:**
- Haidar et al., "Fast Cholesky factorization on GPUs for batch and native modes in MAGMA" (2017)
- Haidar et al., "A Guide for Achieving High Performance with Very Small Matrices on GPU" (IEEE TPDS 2018)
- Haidar et al., "Batched one-sided factorizations of tiny matrices using GPUs" (2018)

### Architecture

MAGMA's batched Cholesky (`magma_spotrf_batched` / `magma_dpotrf_batched`) uses:

- **One matrix per thread block.** Each independent Cholesky runs in its own block. The GPU handles parallelism across the batch automatically via the hardware scheduler.
- **Right-looking algorithm.** After factoring column j, immediately update the trailing submatrix. This maximizes data reuse within the block.
- **Register blocking for tiny matrices (N <= 32).** For matrices that fit in registers, the entire factorization runs register-resident with no shared memory. Each thread holds one or more columns. The factorization is fully unrolled at compile time.
- **Shared memory blocking for small matrices (32 < N <= 64).** The matrix lives in shared memory. Panel factorization uses a sub-blocked approach (similar to our NB=64, IB=16) with shared memory TRSM and SYRK.

### Key Optimizations for Small Matrices

1. **Compile-time unrolling.** The factorization loop is completely unrolled for a specific matrix size. The code is "compiled for a specific size of matrices in the batch, and heavily or completely unrolled using preprocessors."

2. **Register-resident factorization (N <= 32).** For matrices smaller than a warp width (32), the entire matrix is distributed across thread registers. Each thread owns elements from one or more rows. The factorization proceeds column-by-column with warp shuffles for communication. No shared memory needed, no bank conflicts possible.

3. **Coalesced batch memory layout.** Instead of each matrix stored contiguously (AoS), matrices are stored in a strided/interleaved layout (SoA) where element (i,j) of matrix k is at offset `k + batch_count * (i * lda + j)`. This ensures coalesced global memory access when all thread blocks load the same element index from their respective matrices simultaneously.

4. **Avoiding sqrt in inner loop.** The diagonal element requires `sqrt` for Cholesky. For tiny matrices, some implementations use `rsqrt` (reciprocal sqrt, faster on GPU) and restructure the algorithm to use multiplications instead of divisions.

5. **Warp-level parallelism for sub-warp matrices.** For N < 32, multiple matrices can be packed into a single warp. E.g., for N=16, two 16x16 matrices share one warp, each half-warp processing one matrix.

### Performance

| Matrix Size | Speedup vs cuBLAS batched | GPU | Precision |
|-------------|---------------------------|-----|-----------|
| N=16 | ~6x | P100 | double |
| N=32 | ~4x | P100 | double |
| N=64 | ~2x | P100 | double |
| N=16 | ~11.8x | V100 | double |
| N=32 | ~8x | V100 | double |

Batch sizes: 10,000 - 1,000,000 matrices.

The speedup is largest for tiny matrices because vendor libraries have the most overhead per matrix there. At N=64, the gap narrows but MAGMA still wins significantly.

---

## cuSOLVER Batched Cholesky (`cusolverDnXpotrfBatched`)

**Source:** https://docs.nvidia.com/cuda/cusolver/index.html

cuSOLVER provides `cusolverDnXpotrfBatched` for batched Cholesky. Key characteristics:

- **API:** `cusolverDnXpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)`
- **Limitation:** Designed as a convenience wrapper. Internally, it may dispatch individual `potrf` calls per matrix or use a generic batched kernel.
- **Performance:** Research consistently shows cuSOLVER's batched path is 2-6x slower than MAGMA's hand-optimized batched kernels for small N. cuSOLVER's strength is large single-matrix factorization (the monolithic kernel approach).
- **cuSOLVER handle overhead:** Creating a cuSOLVER handle adds ~0.2ms fixed cost. For small matrices where the factorization itself takes microseconds, this overhead is proportionally large.

**Bottom line:** cuSOLVER batched Cholesky is a weak baseline for small N. A custom kernel should beat it substantially.

---

## IA-Chol: Input-Aware Cholesky (ICS 2025)

**Source:** Deng & Wang, "IA-Chol: Input-Aware Cholesky Decomposition on CPU and GPU" (ACM ICS 2025)

### What This Is

IA-Chol automatically determines the optimal tile size based on input matrix size. It also fuses operators within the blocked Cholesky algorithm to reduce kernel launch overhead.

### Key Technique

- **Operator fusion:** Instead of separate kernels for panel factorization, TRSM, and SYRK update, IA-Chol fuses them into fewer, larger kernels. This reduces launch overhead (our exact bottleneck at large N).
- **Tile size prediction:** A lightweight model predicts the best NB for a given N, avoiding the need to sweep tile sizes manually.
- **Result on A100:** 85.1% efficiency vs cuSOLVER's 75.8%.

### Why It Matters for Us

The operator fusion idea is directly relevant to our large-N Cholesky bottleneck (190 graph nodes). Fusing panel + TRSM + SYRK into fewer kernels would reduce node count dramatically. However, for the batched small Cholesky pivot, IA-Chol's approach is less relevant -- at N=32-64, everything fits in one kernel anyway.

---

## Applications That Need Batched Small Cholesky

This section establishes that batched small Cholesky is a real workload with demand:

### 1. Gaussian Process Inference (GPyTorch)

**Source:** Gardner et al., "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration" (NeurIPS 2018)

GP inference requires computing `L = cholesky(K + sigma^2 * I)` where K is the kernel matrix. For Vecchia approximations (the scalable GP approach), K is decomposed into many small conditional covariance matrices (typically 30-150 elements per side). Each location requires its own small Cholesky.

- **Batch sizes:** 100,000 - 1,000,000 small matrices
- **Matrix sizes:** 30x30 to 150x150 (conditioning set size)
- **GPU-accelerated Vecchia on H100:** 1380x faster than exact MLE

### 2. Kalman Filtering and Smoothing

**Source:** Multiple (Kalman filter literature)

Extended Kalman filters require Cholesky of the innovation covariance matrix S = H*P*H^T + R at each timestep. For systems with state dimension 4-64, this means batched Cholesky of 4x4 to 64x64 matrices across timesteps.

- Code generators have achieved 90x acceleration for 4x4 F32 Kalman filter Cholesky on GPU
- Block-tridiagonal SPD systems from Rauch-Tung-Striebel smoothers need batched small Cholesky

### 3. Sparse Multifrontal Solvers

Sparse direct solvers (STRUMPACK, MUMPS) decompose a large sparse factorization into many small dense factorizations at the frontal matrices. These fronts are typically 16-128 elements per side.

- STRUMPACK uses "batch kernel restricted to matrix blocks smaller than 32x32"
- Larger fronts use standard dense factorization

### 4. Preconditioners

Block-Jacobi and block-ILU preconditioners require independent factorizations of many small diagonal blocks. Typical block sizes: 8-64.

---

## Recommended Implementation Strategy

### Target

- **Matrix sizes:** N = 16, 32, 48, 64 (template-specialized)
- **Batch sizes:** 1,000 - 1,000,000
- **Precision:** FP32 (single precision, matching cuSOLVER default for batched)
- **Reference baseline:** `cusolverDnSpotrfBatched` and MAGMA `magma_spotrf_batched`

### Architecture: One Matrix Per Thread Block

```
Grid: batch_size blocks
Block: 256 threads (for N=64) or 128 threads (for N=32)
Shared memory: N*N*sizeof(float) bytes per block
```

**For N=32 (register-resident):**
- 32 threads, each thread owns one column (32 floats = 32 registers)
- Factorization uses warp shuffles for broadcast/reduction
- No shared memory needed
- Full loop unrolling at compile time
- Pack multiple matrices per warp if N < 32

**For N=64 (shared memory):**
- 256 threads, matrix in shared memory (64*64*4 = 16KB)
- Sub-blocked panel (IB=16, same as current panel kernel)
- TRSM and SYRK in shared memory using thread-parallel operations
- XOR swizzle for bank-conflict-free access (we already have this pattern)

### Memory Layout

Use **strided batch layout** for coalesced access:
```c
// Instead of: A[matrix_idx][row][col]
// Use:        A[row * lda + col + matrix_idx * stride]
// Where stride = lda * N (each matrix contiguous)
```

Or for maximum coalescing with many tiny matrices, interleave:
```c
// A[matrix_idx + batch_size * (row * N + col)]
// All threads access element (0,0) of their matrix at consecutive addresses
```

### Key Optimizations

1. **Template on N.** Compile separate kernels for N=16, 32, 48, 64. Full unroll.
2. **rsqrt instead of sqrt.** `rsqrtf()` maps to a single instruction on sm_120. Restructure: instead of `L[j][j] = sqrt(A[j][j])`, compute `inv_ljj = rsqrtf(A[j][j])` and multiply.
3. **Warp shuffle reductions.** For the column update `A[i][j] -= sum(L[i][k]*L[j][k], k<j)`, distribute across threads and reduce with `__shfl_down_sync`.
4. **XOR swizzle for smem.** Reuse the existing swizzle pattern from our GEMM/attention kernels to avoid bank conflicts in the shared memory path.
5. **FP32 throughout.** No need for tensor cores -- at N=32-64 the factorization is too small for MMA to help. Pure FP32 scalar/vectorized math.
6. **Skip positive-definiteness check in hot path.** cuSOLVER checks for negative diagonals. For known-SPD input (e.g., kernel matrices with jitter), skip the branch.

### Expected Performance

Based on MAGMA's published results, a well-optimized batched small Cholesky should achieve:

| N | Estimated speedup vs cuSOLVER batched |
|---|---------------------------------------|
| 16 | 4-8x |
| 32 | 3-6x |
| 64 | 2-4x |

The speedup comes from: (1) eliminating per-matrix launch overhead, (2) register/smem-resident factorization, (3) compile-time specialization, (4) coalesced batch memory layout.

---

## KBLAS: Another Reference Implementation

**Source:** https://arxiv.org/html/2403.07412v1

KBLAS (KAUST BLAS) provides batched Cholesky (`kblas_potrf_batch`) used in GPU-accelerated Vecchia approximations for geostatistics. It uses a strided data layout for contiguous GPU memory allocation, with custom CUDA kernels for the small-matrix case. Performance on A100/H100 enables processing up to 1 million locations.

The strided layout ("contiguous data storage, enhancing GPU memory use efficiency") is preferred over pointer-array approaches for fixed-size batches.

---

## References

- MAGMA batched Cholesky: https://www.sciencedirect.com/science/article/abs/pii/S1877750316305154
- MAGMA batched small matrices guide: https://ieeexplore.ieee.org/document/8214236/
- MAGMA batched tiny matrices: https://www.sciencedirect.com/science/article/abs/pii/S1877750317311456
- MAGMA library: https://icl.utk.edu/magma/
- MAGMA potrf_batched API: https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__potrf__batched.html
- IA-Chol (ICS 2025): https://dl.acm.org/doi/10.1145/3721145.3725756
- cuSOLVER potrfBatched sample: https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSOLVER/potrfBatched
- cuSOLVER 13.2 docs: https://docs.nvidia.com/cuda/cusolver/index.html
- GPyTorch (Gaussian processes): https://ar5iv.labs.arxiv.org/html/1809.11165
- GPU-accelerated Vecchia approximations: https://arxiv.org/html/2403.07412v1
- Batched block-tridiagonal Cholesky: https://arxiv.org/html/2601.03754
- Autotuning batch Cholesky (Dongarra): https://www.netlib.org/utk/people/JackDongarra/PAPERS/autotuning-batch-cholesky-ipdps-2017.pdf
