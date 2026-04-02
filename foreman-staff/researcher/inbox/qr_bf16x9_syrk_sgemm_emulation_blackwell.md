# BF16x9 FP32 Emulation for Custom SYRK on Blackwell

**Source:** NVIDIA Blog: "Unlocking Tensor Core Performance with Floating Point Emulation in cuBLAS" (2026)
**URL:** https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
**Also:** cuBLAS 13.2 documentation, https://docs.nvidia.com/cuda/cublas/
**Relevant to:** QR worker (custom SYRK kernel for CholQR2)
**Worker's current problem:** SYRK component of CholQR2 uses torch.mm(A.T, A) which dispatches to cuBLAS FP32 GEMM. Worker plans custom CUDA SYRK. Question: should the custom SYRK use BF16 MMA with FP32 accumulator, or full FP32?

---

## What This Is

NVIDIA has shipped BF16x9 FP32 emulation in cuBLAS 13.0 Update 2+. This algorithm decomposes each FP32 value into three BF16 values and performs 9 BF16 tensor core matrix multiplications to recover **full FP32 accuracy** while achieving **3-4x native FP32 throughput** on Blackwell.

The key question for the QR worker: should a custom SYRK kernel use:
- (A) Single BF16 MMA with FP32 accumulator (fast but ~1e-3 relative error per element)
- (B) BF16x9 emulation (full FP32 accuracy, 3-4x FP32 throughput)
- (C) Native FP32 CUDA cores

---

## Why It Matters for CholQR2 SYRK

The SYRK computes the Gram matrix G = A^T * A. This is used for Cholesky factorization, and the condition number of G = kappa(A)^2. Even small errors in G can cause Cholesky to fail or produce poor orthogonality.

- **Option A (BF16 MMA):** 1e-3 relative error per element. For well-conditioned matrices (kappa < 100), this is fine -- CholQR2's second iteration corrects. For kappa > 1000, dangerous.
- **Option B (BF16x9):** Full FP32 accuracy. 3-4x faster than option C. Best of both worlds.
- **Option C (FP32):** Baseline accuracy. Slowest.

**Recommendation: Use cuBLAS BF16x9 SGEMM for the SYRK step.** Don't write a custom SYRK kernel at all -- just call cuBLAS SGEMM with `CUBLAS_COMPUTE_32F` and let it automatically use BF16x9 on Blackwell. This gives:
1. Full FP32 accuracy (no stability risk)
2. 3-4x speedup over native FP32 SGEMM
3. Zero custom kernel development effort

---

## How BF16x9 Works (Brief)

Any FP32 value a = a0 + a1 + a2 where a0, a1, a2 are BF16 values (7-bit mantissa each, covering FP32's 23-bit mantissa). Matrix product C = A * B becomes:

```
C = sum over i,j in {0,1,2}: Ai * Bj
```

9 BF16 tensor core GEMMs, accumulated in FP32. The result is bit-exact with FP32 for most practical inputs.

---

## Automatic Dynamic Precision (ADP)

cuBLAS 13.2 includes ADP: the library automatically analyzes input matrices and determines if BF16x9 emulation is safe. For well-conditioned SYRK inputs, it will use BF16x9 automatically when you request `CUBLAS_COMPUTE_32F`.

To force BF16x9: use `CUBLAS_COMPUTE_32F_EMULATED_16BFX9`.

---

## Practical Impact on CholQR2 Performance

Current SYRK cost estimate: ~0.72ms per iteration (cuBLAS FP32 GEMM at 4096^3).

With BF16x9 (3-4x speedup): ~0.18-0.24ms per iteration.

Savings per CholQR2 (2 iterations): ~1.0ms. Not huge compared to 14ms total, but free performance with zero code changes.

---

## For Custom SYRK with Triangle-Awareness

If the worker still wants to build a custom SYRK that skips upper-triangle tiles (50% compute savings), the BF16x9 decomposition can be applied within the custom kernel:

1. Decompose A tiles into 3 BF16 components
2. For each lower-triangle output tile, compute 9 BF16 MMA products
3. Accumulate in FP32 registers

This combines the triangle-aware scheduling (50% fewer tiles) with BF16x9 accuracy, for a theoretical ~6-8x speedup over naive FP32 SYRK.

However, the implementation complexity is high. The pragmatic path: just call cuBLAS SGEMM with BF16x9 compute type and accept computing the full matrix. SYRK is only ~10% of CholQR2 time.

---

## Caveats

1. **BF16x9 requires Blackwell** -- the BF16 tensor core throughput ratio makes this worthwhile only on sm_120/sm_100+. On older GPUs, FP16x3 variants exist but with different trade-offs.
2. **cuBLAS may already be doing this** -- when torch.mm dispatches to cuBLAS with CUBLAS_COMPUTE_32F on Blackwell, cuBLAS might already use BF16x9 via ADP. Check with `CUBLAS_WORKSPACE_CONFIG` or nsight profiling.
3. **9 tensor core GEMMs is still 9x more MMA operations** -- throughput gain comes from tensor core's higher throughput vs FP32 CUDA cores, not from fewer operations.
