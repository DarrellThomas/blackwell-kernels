# Batched Small Cholesky: New Findings (March 2026)

**Source:** https://github.com/icl-utk-edu/magma/blob/master/ReleaseNotes
**Source:** https://arxiv.org/abs/2601.03754
**Source:** https://arxiv.org/abs/2509.03015
**Source:** https://github.com/pytorch/pytorch/pull/53104
**Source:** https://docs.nvidia.com/cuda/cusolver/index.html
**Relevant to:** Cholesky worker
**Worker's current problem:** Monolithic large Cholesky blocked by TF32 MMA defect. Next direction is batched small Cholesky (N=32-64) where cuSOLVER may be weak.

## New Finding 1: MAGMA 2.9.0 Adds sm_120 Support and Improved Batched potrf

MAGMA 2.9.0 (released January 23, 2025) includes:
- **Performance improvements for `magma_?potrf_batched`** (batched Cholesky)
- **Performance improvements for `magma_?trsv_batched`** (batched triangular solve)
- **Variable-size batched Cholesky:** `magma_[sdcz]potrf_vbatched` for batches
  where each matrix can have a different size
- **NVIDIA Blackwell GPU support** (sm_100 and sm_120)

**Why it matters:** MAGMA is our primary competitor for batched small Cholesky.
Their optimized batched potrf for sm_120 sets the bar. We need to benchmark
`magma_spotrf_batched` at N=32 and N=64 on RTX 5090 before building a custom
kernel. If MAGMA already handles this well, our effort is better spent
elsewhere.

**Action:** Install MAGMA 2.9.0 and benchmark:
```
magma_spotrf_batched(lower, N=32, batch=1000, ...)
magma_spotrf_batched(lower, N=32, batch=10000, ...)
magma_spotrf_batched(lower, N=64, batch=1000, ...)
```
Compare against `cusolverDnSpotrfBatched` at the same sizes.

## New Finding 2: PyTorch Switched FROM MAGMA TO cuSOLVER for Batched potrf

As of PyTorch's CUDA >= 11.3 heuristics, the dispatch for batched Cholesky
switched from MAGMA to cuSOLVER:
- cuSOLVER `potrfBatched` now preferred for `batch_size > 1`
- This suggests cuSOLVER has caught up with or surpassed MAGMA for batched
  Cholesky in recent CUDA versions

**Implication:** cuSOLVER's batched potrf may no longer be the easy target it
was in 2017-2018 when MAGMA showed 6-11x speedups. The worker should benchmark
cuSOLVER batched potrf on RTX 5090 at small sizes to calibrate expectations.

## New Finding 3: cuSOLVER 13.1 Internal Algorithm Switch for n<=32 on Blackwell

cuSOLVER 13.1 Update 1 introduced a performance improvement for
`cusolverDnXsyevBatched()` with an **internal algorithm switch on Blackwell
GPUs for matrices of size n <= 32**. Users can revert via
`cusolverDnSetAdvOptions()`.

**This is for eigenvalue only, NOT potrf.** However, it signals that NVIDIA
is actively optimizing small-matrix paths on Blackwell. A similar internal
optimization for batched potrf may exist or may be coming. Check whether
cuSOLVER 13.2's potrf_batched has a similar small-N fast path.

## New Finding 4: Block Tridiagonal Cholesky on RTX 5090 (arXiv 2601.03754)

Schwan, Kuhn & Jones (January 2026) demonstrated GPU-accelerated block
tridiagonal Cholesky on RTX 5090 using NVIDIA's Warp library + cuBLASDx/cuSolverDx:

**Key results on RTX 5090:**
- 500x speedup over QDLDL sparse solver
- 25x speedup over optimized CPU (BLASFEO)
- 2x speedup over NVIDIA cuDSS
- ~2.5x speedup over RTX 3080 (correlates with SM count ratio: 170/68)

**Implementation details:**
- **Fused kernel:** All operations in single kernel for small block sizes (n<=16)
- **Blocked kernel:** Tiles operations through shared memory for larger blocks
- Uses cuBLASDx for GEMM and cuSolverDx for potrf/trsm device-side
- Odd block sizes significantly slower (128-bit alignment requirement)
- Optimal when block size divisible by 16 (for single precision)

**Relevance:** This validates the cuBLASDx+cuSolverDx composition approach
for Cholesky on RTX 5090. Their fused kernel architecture is similar to what
we need for batched small Cholesky. However, they target block tridiagonal
structure (limited fill-in), not dense batched potrf.

## New Finding 5: BlockDSS - Batched Block Tridiagonal Solver (arXiv 2509.03015)

A separate paper presents BlockDSS, using recursive Schur-complement reduction
with three batched operations: potrf, trsm, gemm. Cross-platform CUDA/ROCm.
Competitive with cuDSS.

**Relevance:** Less directly relevant since it targets structured (tridiagonal)
systems, but the batched potrf+trsm+gemm composition pattern is exactly what
we need for batched dense Cholesky.

## New Finding 6: BaSpaCho - Facebook's Sparse Batched Cholesky

Facebook Research's BaSpaCho implements supernodal Cholesky with:
- Pure CUDA mode with batching support
- Parallel elimination of small independent supernodes
- Performance models to select optimal supernode merges

**Relevance:** Limited for dense batched Cholesky. BaSpaCho targets sparse
SPD matrices from nonlinear optimization (Levenberg-Marquardt). However, their
"parallel elimination of small nodes" concept could inform how we batch many
independent small dense factorizations.

## Updated Competitive Landscape (March 2026)

| Implementation | N=32 Batched | N=64 Batched | Notes |
|---------------|-------------|-------------|-------|
| cuSOLVER potrfBatched | Unknown (benchmark needed) | Unknown | PyTorch now prefers over MAGMA |
| MAGMA 2.9 potrf_batched | Improved (sm_120 support) | Improved | New Jan 2025 release |
| Custom register-resident | Target: 5-10x cuSOLVER | Target: 2-5x | Our opportunity |

## Recommendation

**Before writing code, benchmark the competition:**
1. `cusolverDnSpotrfBatched` at N=32,64 batch=1000,10000 on RTX 5090
2. `magma_spotrf_batched` at same sizes (requires MAGMA 2.9 install)
3. If neither exceeds ~50 GFLOPS effective throughput, there's room for our
   register-resident kernel. If they're already efficient, focus elsewhere.

The register-resident approach (N=32, one warp per matrix, all data in
registers, communication via `__shfl_sync`) remains the strongest path to
beating both libraries. For N=32, this avoids shared memory entirely and
eliminates all synchronization overhead except warp shuffles.
