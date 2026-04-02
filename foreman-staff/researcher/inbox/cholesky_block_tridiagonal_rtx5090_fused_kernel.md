# Block Tridiagonal Cholesky on RTX 5090: Fused Kernel Achieves 500x Speedup

**Source:** https://arxiv.org/html/2601.03754v1
**Relevant to:** numerical worker (Cholesky monolithic kernel)
**Worker's current problem:** Building a monolithic Cholesky kernel on sm_120 (RTX 5090) using cuBLASDx/cuSolverDx. Needs confirmation that this approach actually works on sm_120 hardware.

## What This Is

A January 2026 paper demonstrating GPU-accelerated Cholesky factorization of block tridiagonal matrices using NVIDIA's MathDx libraries (cuBLASDx + cuSolverDx) via the Warp framework. The paper benchmarks on RTX 3080 AND **RTX 5090**, confirming the approach works on sm_120.

## Why It Matters for Us

This is **direct proof** that the cuBLASDx + cuSolverDx fused kernel approach works on RTX 5090 (sm_120). The paper's fused kernel combines POTRF + TRSM + SYRK + GEMM into a single kernel launch, loading data into shared memory and processing all operations before writing back -- exactly what our worker needs.

## Key Technique: Fused Kernel Architecture

### What Gets Fused

Each "iteration" of the block Cholesky algorithm combines four operations into a single kernel launch:
1. **potrf** -- Cholesky factorization of diagonal block (cuSolverDx)
2. **trsm** -- Triangular solve for off-diagonal block (cuSolverDx)
3. **syrk** -- Symmetric rank-k update of next diagonal block (via GEMM from cuBLASDx)
4. **gemm** -- General matrix update (cuBLASDx)

The kernel loads all needed data into shared memory at the start, executes all four operations sequentially within shared memory, then writes results back to global memory. This eliminates all kernel launch overhead between operations.

### Thread Block Configuration

The paper uses **64 threads per block** for the experiments shown. Each thread block processes one batch of the factorization. Data movement between global and shared memory is handled explicitly.

### Shared Memory Constraint

When the block size NB is too large, the fused kernel runs out of shared memory (128KB per SM limit). For larger block sizes, a "blocked variant" uses tiling within the kernel to work around the shared memory limit -- similar to what our worker needs for N=4096 with NB=64.

### Performance on RTX 5090 (sm_120)

| Comparison | Speedup on RTX 5090 |
|-----------|---------------------|
| vs QDLDL (sparse solver) | **500x** |
| vs BLASFEO (CPU) | **25-40x** |
| vs NVIDIA CUDSS | **~2x** |

The paper shows that the fused approach beats even NVIDIA's own CUDSS library by ~2x on RTX 5090, confirming that custom fused kernels outperform library approaches on this hardware.

### Scaling Behavior

Performance scales logarithmically with problem size N up to saturation (~N=250 for their block tridiagonal structure). Beyond that, SMs become resource-constrained and scaling becomes linear.

## Adaptation for Dense Cholesky

The paper targets block tridiagonal matrices (specialized structure), but the kernel architecture directly transfers to dense Cholesky:

1. **Same building blocks:** potrf (cuSolverDx) + trsm (cuSolverDx) + gemm (cuBLASDx)
2. **Same fused kernel pattern:** load data to shared memory, run all operations, write back
3. **Same hardware:** RTX 5090, sm_120

For dense N=4096 Cholesky, the main difference is:
- Block tridiagonal: only 2 sub-diagonal blocks per column
- Dense: up to N/NB - 1 sub-diagonal blocks per column (more GEMM work)
- Dense requires iterating over more blocks, but the kernel fusion pattern is identical

### cuSOLVERDx Blocked POTRF Example

The cuSOLVERDx documentation includes a `blocked_potrf.cu` example that implements exactly this: dense blocked Cholesky using cuSolverDx for potrf/trsm and cuBLASDx for GEMM, all within a single kernel launch using one thread block per matrix.

**Key reference:** https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html

## Caveats

1. **Block tridiagonal != dense.** The paper's structure has O(N) blocks vs dense Cholesky's O(N^2) blocks. The per-panel GEMM/SYRK work in dense Cholesky is much larger, meaning the trailing update dominates more.

2. **64 threads may not be enough for dense.** The paper uses 64 threads per block, which is fine for small block-structured operations. cuSOLVER's monolithic kernel uses 256 threads. For dense N=4096, 256 threads gives better thread-level parallelism for the GEMM updates.

3. **Warp framework overhead.** The paper uses Warp (Python), which adds compilation overhead. Our worker writes raw CUDA, which should be more efficient.

4. **The 2x over CUDSS is for block tridiagonal.** Dense Cholesky against cuSOLVER's monolithic kernel is a harder target. cuSOLVER's `getrf_wo_pivot_params` achieves ~15 TFLOPS on a single SM -- that's what we need to match.

## Sources

- [GPU-Accelerated Cholesky Factorization of Block Tridiagonal Matrices (arXiv:2601.03754)](https://arxiv.org/html/2601.03754v1)
- [cuSOLVERDx Blocked Potrf Example](https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html)
- [NVIDIA Warp Documentation](https://docs.nvidia.com/warp/)
