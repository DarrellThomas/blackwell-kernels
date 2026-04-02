# cuBLAS BF16x9 FP32 Emulation on Blackwell — Direct Path to Fast LU Trailing GEMM

**Source:** https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
**Source:** https://docs.nvidia.com/cuda/cublas/ (cuBLAS 13.2 documentation)
**Relevant to:** numerical/ worker (LU factorization trailing GEMM update)
**Worker's current problem:** LU trailing GEMM is 80-85% of total compute. Need tensor-core acceleration while maintaining FP32 accuracy. TF32 MMA is broken on sm_120 (B fragment diagonal broadcasting). BF16 MMA loses too much precision over 64 iterations.

---

## What This Is

NVIDIA cuBLAS 13.0 Update 2 introduced **BF16x9 FP32 emulation** -- a new algorithm that decomposes FP32 matrix multiplication into 9 BF16 tensor core matrix multiplications, recovering **full FP32 accuracy** while achieving **3-4x native FP32 throughput** on Blackwell.

This is a game-changer for the LU trailing GEMM update: it eliminates the precision vs performance tradeoff entirely.

---

## How BF16x9 Works

### The Decomposition

Any FP32 value can be **exactly** represented as three BF16 values:

```
a = a0 + 2^(-8) * a1 + 2^(-16) * a2
```

where a0, a1, a2 are BF16 values. FP32 has a 23-bit mantissa; BF16 has 7 bits. Three BF16 values cover 3 * 8 = 24 bits (with the implicit leading 1), which is enough to represent all FP32 mantissa bits.

### Matrix Decomposition

For matrices A and B:
```
A = A1 + 2^(-8) * A2 + 2^(-16) * A3   (each Ai is BF16)
B = B1 + 2^(-8) * B2 + 2^(-16) * B3   (each Bi is BF16)
```

The product A*B becomes:
```
A*B = A1*B1
    + 2^(-8)  * (A1*B2 + A2*B1)
    + 2^(-16) * (A1*B3 + A2*B2 + A3*B1)
    + 2^(-24) * (A2*B3 + A3*B2)         // these terms may be dropped
    + 2^(-32) * (A3*B3)                  // negligible
```

The first 7 terms (or a subset of 9 depending on accuracy requirements) are computed as separate BF16 GEMMs with FP32 accumulation, then scaled and summed.

### Why "x9"

The full expansion has 9 cross-product terms (3x3). All 9 are computed for guaranteed FP32 accuracy. The scaling factors (2^-8, 2^-16, etc.) are applied using **hardware-level scaling** on Blackwell -- a special feature that makes this efficient. On older architectures, the scaling overhead would negate the speedup.

---

## Performance

### Blackwell-Specific Results

| Hardware | FP32 Native TFLOPS | BF16x9 Emulated TFLOPS | Speedup |
|----------|-------------------|----------------------|---------|
| B200 | ~X | ~3-4X | **3-4x** |
| RTX PRO 6000 | ~X | ~3X | **~3x** |
| GB200 NVL72 | ~X | ~3-4X | **3-4x** |

At M=N=K=32768, emulation achieves 3-4x more TFLOPS than native FP32.

### Accuracy

The BF16x9 decomposition provides accuracy **as good as or better than** native FP32 SGEMM. Error distribution studies confirm this -- the extra accumulation precision from BF16 tensor cores with FP32 accumulators can actually be slightly better than native FP32 FMA.

### Automatic Dispatch

cuBLAS automatically uses BF16x9 when:
- `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` compute type is specified
- Problem size is large enough to benefit
- Architecture supports it (Blackwell)

For small problems, cuBLAS falls back to native FP32 -- no performance penalty.

---

## Why This Matters for LU on sm_120

### The Trailing GEMM Problem

The LU trailing update is: `A_trailing -= L_column * U_row` (standard GEMM).

For N=4096, NB=64:
- Total trailing GEMM FLOPs: ~43 GFLOP (94% of total LU compute)
- With BF16x9 at ~3x native FP32: trailing GEMM finishes in ~1/3 the time
- **With zero precision loss** -- no iterative refinement needed

### Comparison to Other Mixed-Precision Approaches

| Approach | Accuracy | Speedup vs FP32 | Complexity |
|----------|----------|-----------------|------------|
| **BF16x9 emulation** | **Full FP32** | **3-4x** | **Zero (cuBLAS API call)** |
| BF16 MMA + FP32 accum | ~2 decimal digits | 8-16x | Custom kernel |
| 3xTF32 (Ootomo/Yokota) | Full FP32 | 1.7x (A100) | Custom CUTLASS kernel |
| FP16 + iterative refine | Full FP64 via refinement | 4-6x (total solve) | Significant |

**BF16x9 wins for our use case.** It gives full FP32 accuracy with 3-4x speedup and zero code changes beyond a cuBLAS compute type flag. For the multi-kernel approach (v1), just change the cuBLAS SGEMM call's compute type.

### For the Monolithic Kernel

In the monolithic kernel's trailing GEMM phase, you can either:
1. Call cuBLAS with BF16x9 (multi-kernel approach)
2. Implement the BF16x9 decomposition manually using our BF16 MMA primitives:
   - Split each FP32 tile into 3 BF16 components
   - Run 9 (or 7) mma.sync m16n8k16 accumulations
   - Sum with appropriate scaling

The manual approach gives full control and avoids kernel launch overhead.

### cuBLAS 13.2 Extensions

cuBLAS 13.2 added BF16x9 support to `cublas[SC]syr[2]k` and `cublasCher[2]k` -- meaning SYRK (used in Cholesky) also benefits. This is relevant if the worker needs to share findings with the Cholesky worker.

---

## How to Use (v1 Multi-Kernel Approach)

```cpp
// Replace CUBLAS_COMPUTE_32F with CUBLAS_COMPUTE_32F_EMULATED_16BFX9
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    A, CUDA_R_32F, lda,
    B, CUDA_R_32F, ldb,
    &beta,
    C, CUDA_R_32F, ldc,
    CUBLAS_COMPUTE_32F_EMULATED_16BFX9,  // <-- THIS IS THE CHANGE
    CUBLAS_GEMM_DEFAULT);
```

That's it. No data format changes, no extra buffers, no iterative refinement.

---

## Caveats

1. **Blackwell-only performance benefit.** BF16x9 can run on other architectures but only provides speedup when BF16 TC throughput > 9x FP32 throughput. RTX 5090 (sm_120) qualifies.

2. **Hardware scaling required.** The 2^-8 and 2^-16 scaling factors use Blackwell-specific hardware. On sm_120 specifically, verify this works -- the blog mentions "select architectures" for BF16x9.

3. **9 GEMMs means 9x the memory traffic for A and B operands.** For bandwidth-bound problems (small K), BF16x9 won't help. For compute-bound trailing GEMMs in LU (large matrices), this is fine.

4. **Test empirically on sm_120.** The blog primarily discusses B200/GB200. Consumer Blackwell (sm_120) may have different characteristics.

---

## Sources

- [cuBLAS FP Emulation Blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
- [cuBLAS 13.2 Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Ootomo & Yokota, 2022 — Recovering Single Precision from Tensor Cores](https://arxiv.org/abs/2203.03341)
- [ORNL TF32/TF64 Frameworks (SC'23)](https://dl.acm.org/doi/10.1145/3624062.3624084)
