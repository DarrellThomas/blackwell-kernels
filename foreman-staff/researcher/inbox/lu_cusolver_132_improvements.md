# cuSOLVER 13.2 and cuBLAS 13.2 Improvements Relevant to Numerical Workers

**Sources:**
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [cuSOLVER 13.2 Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
- [GPU-Accelerated Cholesky of Block Tridiagonal Matrices (arXiv:2601.03754)](https://arxiv.org/abs/2601.03754)
**Relevant to:** LU worker, Cholesky (numerical) worker, QR worker
**Date:** 2026-03-15

---

## What This Is

CUDA 13.2 (released March 2026) brings improvements to cuSOLVER and cuBLAS that directly
affect our numerical factorization baselines.

---

## cuSOLVER 13.2 Changes

### FP64 Fixed-Point Emulation
cuSOLVER 13.2 adds FP64 fixed-point emulation support with new control APIs. This allows
cuSOLVER to perform "FP64-emulated calculations" that deliver "significant performance
gains, particularly for compute-intensive workloads."

**What this means:** cuSOLVER can now use FP32 tensor cores (TF32 MMA) to emulate FP64
precision via multi-precision algorithms (e.g., error-free transformations, compensated
summation). This could significantly speed up FP64 factorizations (LU, QR, Cholesky)
on Blackwell GPUs that have fast TF32 tensor cores.

**Impact on LU worker:** The cuSOLVER baseline for LU (sgetrf, 9.4 ms at N=4096) may
change if cuSOLVER 13.2 has better internal kernels. **Re-benchmark the baseline after
any CUDA upgrade.**

### New cusolverDnXsygvd API
Supports larger problem sizes for symmetric generalized eigenvalue decomposition.
Not directly relevant to our LU/Cholesky/QR work, but indicates active development
in dense linear algebra.

## cuBLAS 13.2 Changes

### Performance Improvements
- "Up to 20% performance speedup on RTX PRO 6000 for FP8, FP16/BF16, TF32, and INT8"
- RTX PRO 6000 is sm_120 -- these improvements apply to RTX 5090 too
- "Up to 3x performance improvement for MXFP8/NVFP4 on DGX Spark systems"

**Impact on GEMM worker:** Our 1.29x vs cuBLAS FP8 advantage may have narrowed. The
cuBLAS 13.2 FP8 path on sm_120 is likely 10-20% faster than cuBLAS 13.0.

**Impact on numerical workers:** cuBLAS TRSM, SYRK, and GEMM calls used by our
Cholesky and QR kernels may also be 10-20% faster, improving our CholQR2 and
blocked factorization timings proportionally.

### FP64 Fixed-Point Emulation for SYRK/HERK
"FP64 fixed-point emulation for SYRK/HERK routines" -- this enables FP64-precision
SYRK using FP32 tensor cores. For our QR worker's CholQR2 algorithm (which uses
SYRK for G = A^T @ A), this could speed up the SYRK component if precision allows.

### Bug Fixes
- "Resolved concurrent kernel execution issues with Tensor Memory"
- "Fixed large leading dimension problems causing incorrect results"
- "Corrected FP8 kernel hangs on Compute Capability 9.0"
- "Fixed C matrix broadcasting with LDC = 0"

---

## GPU-Accelerated Cholesky of Block Tridiagonal Matrices (arXiv:2601.03754)

A January 2026 paper from EPFL achieves:
- 25x improvement over optimized CPU implementation
- >2x acceleration vs NVIDIA cuDSS library
- Uses nested dissection + multi-stage permutation

**Not directly applicable** to our batched dense Cholesky (different problem structure),
but demonstrates that innovative parallelization strategies can beat NVIDIA's libraries
significantly -- even for factorizations that are traditionally hard to parallelize.

---

## Recommendations

### For LU Worker (just starting)
- **Do not upgrade to CUDA 13.2 yet** -- wait until v1 blocked LU is working and
  benchmarked against current cuSOLVER baseline
- Be aware that cuSOLVER 13.2 may improve the baseline if/when we upgrade
- The cuBLAS 13.2 TRSM/GEMM improvements will benefit the blocked LU algorithm's
  trailing update calls

### For QR Worker
- cuBLAS 13.2 SYRK improvements directly benefit CholQR2's G = A^T @ A step
- FP64 emulation for SYRK could enable higher-precision CholQR2 at near-FP32 speed
- Re-benchmark CholQR2 if/when CUDA is upgraded to 13.2

### For Cholesky Worker
- The monolithic single-block approach used by cuSOLVER is likely unchanged in 13.2
  (the FP64 emulation is for the BLAS kernels, not the monolithic factorization)
- The 0.55x gap is architectural, not toolchain-related

### For GEMM Worker
- **Re-benchmark cuBLAS FP8** after any CUDA upgrade -- the 1.29x advantage is
  against cuBLAS 13.0 and may narrow with cuBLAS 13.2's 20% improvement
