# cuBLAS BLAS Level 3 Improvements for Blackwell (CUDA 13)

**Source:** https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/
**Relevant to:** linalg worker
**Worker's current problem:** SYRK at 0.96x reference, TRMM at 1.02x reference. Both use cuBLAS/PyTorch delegation.

## What This Is

CUDA 13.0 includes "improved performance for BLAS L3 (non-GEMM) kernels (SYRK, HERK,
TRMM, and SYMM) with FP32 and CF32 precisions on NVIDIA Blackwell GPUs."

## Why It Matters

The linalg worker uses cuBLAS for SYRK and TRMM (via torch.mm delegation):
- **SYRK** at 0.96x — uses `torch.mm(A, A.t())` which calls cuBLAS GEMM, not SYRK
- **TRMM** at 1.02x — uses `torch.mm` with assume_triangular fast path

If the worker is benchmarking against cuBLAS SYRK/TRMM directly as the reference,
the reference may now be faster with CUDA 13 Blackwell optimizations. The worker
should **re-run baselines** to verify current reference performance hasn't improved.

Conversely, if the worker's delegation uses cuBLAS GEMM (not specialized SYRK/TRMM),
these improvements don't directly affect the worker's kernels — but they could
affect the "vs reference" column if the reference uses the specialized routines.

## Also Notable

cuBLAS now supports `CUBLAS_GEMM_AUTOTUNE` as an algorithm parameter for GemmEx,
GemmBatchedEx, and GemmStridedBatchedEx. This auto-tunes the GEMM algorithm and
caches the result. May improve batch GEMM performance.

## Recommendation

Re-benchmark SYRK and TRMM references with current CUDA 13 toolkit to ensure
the vs_ref numbers are still accurate.
