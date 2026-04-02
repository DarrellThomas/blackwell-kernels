# cuSPARSE SpMVOp API: Improved SpMV Baseline (CUDA 13.1)

**Source:** https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/
**Relevant to:** spmv worker
**Worker's current problem:** SpMV not yet started. Need to establish baseline and beat cuSPARSE.

## What This Is

CUDA 13.1 introduces a new `SpMVOp` API in cuSPARSE with improved performance compared
to the legacy `CsrMV` API. Features:
- Supports CSR format with 32-bit indices
- Double precision support
- **User-defined epilogues** — allows fusing post-SpMV operations into the SpMV kernel

## Why It Matters

1. **Higher baseline to beat.** The SpMVOp API likely has better performance than the
   old cuSPARSE SpMV APIs the worker might benchmark against. The worker should use
   SpMVOp as the reference, not the legacy API.

2. **User-defined epilogues** could be relevant for iterative solver fusion (e.g.,
   fusing SpMV + vector update in CG/GMRES).

## Recommendation

When establishing the SpMV baseline:
1. Benchmark BOTH the legacy cuSPARSE SpMV and the new SpMVOp API
2. Use SpMVOp as the reference target (harder but more fair)
3. Test on representative matrices (structured, unstructured, power-law degree)
4. Note: SpMVOp may not yet support all matrix types or formats — check docs
