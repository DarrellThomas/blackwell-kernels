# HPL-MxP Mixed-Precision LU Scheme — Panel FP32 / Trailing FP16 / GMRES Refinement

**Source:** https://arxiv.org/abs/2509.19618 (Dongarra & Luszczek, 2025)
**Source:** https://hpl-mxp.org/
**Source:** https://arxiv.org/html/2412.19322v2 (Mixed-Precision Numerics Survey, Dec 2024)
**Relevant to:** numerical/ worker (LU factorization / getrf)
**Worker's current problem:** cuSOLVER does N=4096 in 9.4ms. Worker is building blocked LU with cuBLAS as v1 stepping stone. Need to know: what is the optimal mixed-precision scheme if we go beyond BF16x9 (which gives full FP32 accuracy but only 3-4x speedup)?

---

## What HPL-MxP Is

HPL-MxP is the mixed-precision successor to the High Performance Linpack benchmark. It solves Ax=b by:
1. LU factorization in **low precision** (FP16 or lower)
2. Iterative refinement via **GMRES** in high precision (FP64) to recover accuracy

The algorithm achieves FP64 accuracy while doing the expensive O(N^3) factorization in FP16.

---

## The Mixed-Precision LU Scheme

### Three-Precision Strategy

| Phase | Precision | Rationale |
|-------|-----------|-----------|
| Panel factorization (GETF2) | **FP32** | Pivoting needs accuracy for stability |
| Triangular solves (TRSM) | **FP32** | Small relative to GEMM, needs stability |
| Schur complement update (GEMM) | **FP16 with FP32 accumulation** | 80-85% of FLOPs, tensor core sweet spot |

This is the key insight: **only the trailing GEMM uses low precision**, and it uses FP32 accumulation within tensor cores to limit error propagation.

### Why GMRES Instead of Stationary Refinement

Traditional iterative refinement (IR) works for well-conditioned matrices:
```
x_{i+1} = x_i + (PA)^{-1} (b - Ax_i)
```

But it converges only when kappa_inf(A) * eps_low < 1 (where eps_low = 2^-11 for FP16). For condition numbers above ~10^3, standard IR stalls.

HPL-MxP uses **GMRES-based refinement** instead:
- Uses the low-precision LU factors as a **left preconditioner**
- GMRES operates in a Krylov subspace, handling ill-conditioning better
- Typically converges in **5-15 iterations** (benchmark allows up to 50)
- Each iteration: 1 GEMV + 1 triangular solve + 1 dot product (cheap relative to the O(N^3) factorization)

### Convergence Criterion

||Ax - b||_inf / (||A||_inf * ||x||_inf + ||b||_inf) / (n * eps_64) < 16

### Performance Results at Scale

| System | Speedup vs FP64 HPL | Notes |
|--------|---------------------|-------|
| Summit (V100) | 9.50x | FP16 TC / FP64 ratio is 16x theoretical |
| Frontier (MI250X) | 8.31x | FP16 / FP64 ratio is 8x theoretical |
| Total achieved | 339.86 PFLOP/s (FP8) | Exascale-level on large systems |

Speedups fall short of theoretical hardware FP ratios due to FP32 operations during panel factorization and refinement overhead.

---

## Application to Our LU at N=4096 on sm_120

### Option 1: BF16 Trailing GEMM + GMRES Refinement (Maximum Speed)

```
Step 1: LU factorize with BF16 MMA trailing GEMM, FP32 panel
  - Panel: FP32 GETF2 (argmax, swap, scale, rank-1 — all scalar FP32)
  - TRSM: FP32 (small relative to GEMM)
  - Trailing GEMM: BF16 mma.sync m16n8k16 with FP32 accumulation
  - Store L,U factors in FP32

Step 2: Solve Ax=b using L,U factors in FP32
Step 3: GMRES refinement (3-5 iterations) in FP32

Expected performance:
  - BF16 MMA throughput: ~330 TFLOPS
  - Trailing GEMM: 43 GFLOP / 330 TFLOPS * (1/efficiency) ≈ 0.3-0.5ms
  - Panel + TRSM + LASWP: ~1-2ms
  - GMRES refinement: ~0.2ms (5 iterations of GEMV + trsv)
  - Total: ~1.5-2.5ms (vs cuSOLVER's 9.4ms)
```

BUT: BF16 has only 7-bit mantissa. Over 64 iterations of rank-64 updates, accumulated error = O(4096 * 2^-8) = O(16) per element. **This will NOT produce valid FP32 LU factors.** Refinement can recover the solve accuracy, but the factors themselves are inaccurate.

For **solving Ax=b**, this works (GMRES recovers accuracy). For **just factorization** (returning L and U), it does NOT give FP32 quality factors.

### Option 2: BF16x9 Trailing GEMM (Full FP32 Accuracy, Good Speed)

```
Step 1: LU factorize with BF16x9 emulated SGEMM for trailing GEMM
  - All operations produce true FP32 results
  - Trailing GEMM: ~3-4x native FP32 throughput
  - No refinement needed

Expected performance:
  - Trailing GEMM at 3x: ~43 GFLOP / (3 * FP32_peak) ≈ 1-2ms
  - Panel + TRSM + LASWP: ~1-2ms
  - Total: ~2-4ms (vs cuSOLVER's 9.4ms)
```

This is simpler and gives true FP32 factors.

### Option 3: FP8 Trailing GEMM + Refinement (Maximum Theoretical Speed)

RTX 5090 has FP8 (e4m3) MMA at ~660 TFLOPS (2x BF16). Using FP8 for the trailing GEMM:

```
Step 1: LU factorize with FP8 trailing GEMM
  - Convert FP32 L,U tiles to FP8 for MMA
  - Accumulate in FP32
  - Massive speedup on trailing GEMM

Step 2: GMRES refinement

Problem: FP8 e4m3 has only 3-bit mantissa. Accumulated error over 64 iterations
would be catastrophic. This is NOT viable for LU factorization even with refinement
— the factors would be so inaccurate that GMRES would not converge.
```

**FP8 is NOT viable for LU trailing updates.** The precision is too low for the accumulated update pattern.

### Recommendation

**Use BF16x9 emulation (Option 2) for v1.** It gives full FP32 accuracy with 3-4x speedup and requires only a cuBLAS compute type change. If the worker later builds a monolithic kernel, implement the BF16x9 decomposition manually using existing BF16 MMA primitives.

If the goal is **solving Ax=b** (not just factorization), Option 1 (BF16 + GMRES) is viable and faster, but more complex to implement.

---

## Key Finding from Mixed-Precision Survey (Dec 2024)

The comprehensive survey (arxiv:2412.19322) notes:

- **Well-conditioned matrices** (positive eigenvalues, bounded away from zero): FP16 + IR converges in 1-3 iterations. 4x speedup on V100.
- **Moderately ill-conditioned** (kappa ~10^5): Achievable 3x speedup, requires more iterations.
- **Indefinite matrices / mixed eigenvalues**: IR may stall. GMRES-based refinement required.
- **Tensor core accumulation in FP32** is numerically superior to FP16 accumulation. This is important: our BF16 mma.sync already accumulates in FP32, which is the better variant.
- **FP8 for factorization**: The survey does NOT describe any viable FP8 dense factorization. FP8 is used only for training (gradient accumulation), not for numerical linear algebra.

---

## Sources

- [HPL-MxP Benchmark Paper (Dongarra & Luszczek, 2025)](https://arxiv.org/abs/2509.19618)
- [HPL-MxP Website](https://hpl-mxp.org/)
- [Mixed-Precision Numerics Survey (Dec 2024)](https://arxiv.org/abs/2412.19322)
- [Haidar et al., Tensor Core FP16 IR (SC18)](https://dl.acm.org/doi/10.1109/SC.2018.00050)
- [Mixed-Precision IR on GPUs (Royal Society, 2020)](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0110)
