# cuBLAS 13.2: GEMM_AUTOTUNE and FP64 Emulation Updates

**Sources:**
- [cuBLAS Release 13.2 (March 5, 2026)](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf)
- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA 13.2 Blog Post](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)
**Relevant to:** gemm, linalg, numerical workers
**Date:** 2026-03-15

---

## What This Is

cuBLAS 13.2 (released March 5, 2026 with CUDA 13.2) has two notable new features
relevant to our workers.

---

## 1. CUBLAS_GEMM_AUTOTUNE Algorithm

`cublasGemmEx`, `cublasGemmBatchedEx`, and `cublasGemmStridedBatchedEx` now accept
`CUBLAS_GEMM_AUTOTUNE` as a valid value for the `algo` parameter.

**What it does:** When `CUBLAS_GEMM_AUTOTUNE` is specified, cuBLAS internally
benchmarks available algorithms and caches the selected algorithm within the current
`cublasHandle_t`. Subsequent calls with the same shape/types reuse the cached result.

**Why it matters:**
- Previously we benchmarked against `CUBLAS_GEMM_DEFAULT` which uses heuristics to
  pick an algorithm. AUTOTUNE may find a faster algorithm.
- **Our gemm worker should benchmark against cuBLAS with AUTOTUNE** as the strongest
  possible baseline. This is the "give cuBLAS every advantage" comparison.
- If our custom FP8 GEMM at 1.34x over cuBLAS was measured against DEFAULT, the
  AUTOTUNE baseline may be higher -- need to re-verify.

**Benchmarking note:** The first call with AUTOTUNE incurs autotuning overhead. Only
subsequent calls reflect the actual optimized performance. Ensure warm-up calls use
AUTOTUNE before timing.

---

## 2. FP64 Emulation via Ozaki-1 Scheme (Opt-in)

cuBLAS 13.2 introduces opt-in fixed-point emulation for FP64 GEMM (D/ZGEMM) using
the Ozaki-1 scheme. This leverages INT8/BF16 tensor cores to emulate FP64 precision
matmuls with better throughput and power efficiency.

**How it works:** The Ozaki-1 scheme decomposes FP64 values into multiple BF16/INT8
sub-values, performs multiple tensor core GEMMs, and reconstitutes the FP64 result.
An "automatic dynamic precision framework" adjusts the decomposition depth to ensure
FP64-level accuracy.

**Why it matters for our numerical workers:**
- The QR/LU/Cholesky workers currently benchmark against cuSOLVER which uses standard
  FP64 GEMM internally for trailing updates. If cuSOLVER starts using emulated FP64
  GEMM, the baseline gets faster.
- cuSOLVER 13.2 added `cusolverDnSetMathMode` and `cusolverDnSetEmulationStrategy`
  APIs to control this behavior for factorizations (QR, LU, Cholesky).
- Our workers should test with both standard and emulated cuSOLVER baselines.

---

## 3. Other cuBLAS 13.2 Changes

| Change | Relevance |
|--------|-----------|
| MXFP8 Grouped GEMM on Blackwell datacenter | None (sm_100 only) |
| BF16x9 FP32 emulation for SYRK/HERK | Relevant to linalg worker's SYRK kernel |
| BLAS L3 non-GEMM (SYRK, TRMM, SYMM) FP32 perf on Blackwell | Raises linalg baseline |
| cuBLASLt 3x improvement for NVFP4/MXFP8 on DGX Spark | None (not sm_120) |
| FP16/BF16 + FP8 GEMM improvements on DGX Spark | None (DGX Spark specific) |

---

## Action Items

1. **Gemm worker:** Re-benchmark FP8 GEMM speedup against cuBLAS with
   `CUBLAS_GEMM_AUTOTUNE` instead of `CUBLAS_GEMM_DEFAULT`. The 1.34x number may
   change.

2. **Linalg worker:** Check if cuBLAS SYRK/TRMM FP32 baseline has improved on
   sm_120 with 13.2. The new BF16x9 emulation support for SYRK could affect
   comparisons.

3. **Numerical workers (QR/LU/Cholesky):** Test cuSOLVER 13.2 with emulated FP64
   math mode enabled. If cuSOLVER factorizations speed up, our target may need
   adjustment.
