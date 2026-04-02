# CholQR2 Advanced Optimization Research: Mixed Precision, Shifted CholQR3, Randomized Sketching, and More

**Sources:** Multiple (see per-finding URLs below)
**Relevant to:** QR worker
**Worker's current problem:** CholQR2 at 1.58x cuSOLVER (14.08ms vs 22.26ms at 4096x4096). Next steps include profiling components, exploring single-iteration CholQR, custom CUDA SYRK/TRSM, and fused SYRK+Cholesky.

---

## Finding 1: Mixed-Precision CholQR — BF16 SYRK with FP32 Accumulation

**Source:** Existing cross-pollination analysis + Zhang & Wu (arXiv:1912.05508)
**URL:** https://arxiv.org/abs/1912.05508

Zhang & Wu (2019) demonstrated mixed-precision QR on GPU using FP16 tensor cores, achieving 14x faster than single-precision cuSOLVER QR at large scale with slightly lower accuracy. Their approach uses half-precision tensor cores for the compute-heavy BLAS-3 operations while maintaining accuracy through algorithmic refinement.

**For CholQR2:** The SYRK step (A^T @ A) is a GEMM and can be computed with BF16 MMA (m16n8k16) + FP32 accumulators. This introduces ~1e-3 relative error per element in the Gram matrix, but this is acceptable because: (a) CholQR2's second iteration corrects orthogonality errors, and (b) the Cholesky step operates in FP32. On RTX 5090, BF16 tensor core throughput is ~1.5-2x FP32 throughput, so even at 0.97x cuBLAS BF16, a BF16 SYRK should be faster than FP32 cuBLAS SYRK.

**Caveat:** If SYRK is only ~1.4ms of 14ms total (see cross-pollination brief), even a 2x speedup on SYRK saves only ~0.7ms. Profile first to confirm the bottleneck.

---

## Finding 2: Shifted CholQR3 (sCQR3) — Fukaya et al. + Improved Column-Based Shift

### 2a. Original sCQR3

**Source:** Fukaya et al. (2018), arXiv:1809.11085
**URL:** https://arxiv.org/abs/1809.11085

sCQR3 extends CholQR2's applicability from kappa_2(X) < u^{-1/2} (about 10^4 for FP32) to kappa_2(X) = O(u^{-1}) (about 10^7 for FP32) by adding a diagonal shift: G = A^T @ A + s*I. The shifted Gram matrix is guaranteed positive-definite, so Cholesky always succeeds. The first shifted CholQR reduces the condition number to below u^{-1/2}, after which CholQR2 (two more iterations) produces O(u) orthogonality. Total: 3 CholQR iterations, hence "sCQR3."

**Cost:** 50% more FLOPs than CholQR2 (3 iterations vs 2). For the current test matrices (random Gaussian, kappa ~ 200-400), CholQR2 is sufficient and sCQR3 is unnecessary overhead.

### 2b. Improved sCQR3 with Column-Based Shift

**Source:** Fan, Guan, & Qiao (2024), arXiv:2408.06311
**URL:** https://arxiv.org/abs/2408.06311

This 2024 paper proposes a column-based matrix definition [X]_g that yields a smaller shift parameter s compared to the standard norm-based approach. A smaller shift means: (a) the shifted Gram matrix is closer to the true Gram matrix, improving accuracy; (b) the condition number reduction in the first iteration is more effective. The authors provide rigorous orthogonality and residual bounds showing enhanced numerical stability.

**For us:** Only relevant if the worker encounters ill-conditioned test matrices. For well-conditioned inputs, CholQR2 (2 iterations) remains optimal.

---

## Finding 3: Randomized Householder-Cholesky QR (rand_cholQR)

**Source:** Higgins, Szyld, Boman, Yamazaki (2023/2025), arXiv:2309.05868
**URL:** https://arxiv.org/abs/2309.05868
**Published:** Numerische Mathematik, 2025

rand_cholQR uses randomized sketching as a preconditioner before CholQR, achieving sCQR3-level stability with CholQR2-level cost. The algorithm:

1. Apply a sparse CountSketch S1 to V (cost: O(nm) FLOPs — just row sampling)
2. Apply a dense sketch S2 to S1*V (produces a small p2 x m matrix, p2 ~ 2m)
3. Compute QR of the small sketched matrix (cheap)
4. Use the R factor as a preconditioner: V_preconditioned = V * R^{-1}
5. Apply CholQR once to V_preconditioned

Total sketching cost: O(nm + m^4) FLOPs, which is less than the O(nm^2) cost of one CholQR iteration. The method is always stable (no condition number restriction) without the 50% overhead of sCQR3.

**GPU results (A100):** Up to 4% faster than CholQR2, and 56.6% faster than sCQR3. The speedup comes from replacing the third CholQR iteration (in sCQR3) with a cheap sketching step.

**For us:** This is interesting if the worker wants to handle arbitrary condition numbers without the cost of a third CholQR iteration. For well-conditioned matrices, the sketching overhead makes it slightly slower than CholQR2 (the 4% improvement may be within noise). The real value is robustness — it works on any input without needing to check condition numbers.

---

## Finding 4: Fused SYRK+Cholesky Kernels

**No papers or implementations found** that fuse A^T@A with potrf in a single GPU kernel. This is unsurprising because:

1. SYRK (A^T@A) is a massive parallel GEMM that saturates the GPU
2. Cholesky (potrf) is sequential along the diagonal with limited parallelism
3. The two operations have fundamentally different parallelism patterns

The closest related work is cuBLASDx's `simple_gemm_aat` example, which shows device-side A^T@A computation (SYRK as device-callable GEMM). This could be composed with a device-side Cholesky from cuSOLVERDx to avoid kernel launch overhead between the two steps, but it is not a truly fused kernel.

**Recommendation:** Do not pursue fusion. The SYRK and Cholesky steps have different enough parallelism that fusing them would likely hurt performance. Instead, minimize launch overhead by: (a) using CUDA graphs to capture the SYRK->Cholesky->TRSM sequence, or (b) using cuBLASDx/cuSOLVERDx device-side calls within a single kernel for the small-N case.

---

## Finding 5: CholQR for Tall-Skinny on GPU — TSQR Comparison

### 5a. Distributed CA-CholQR2

**Source:** Mijic, Kaushik, Davidovic (2024), arXiv:2405.04237
**URL:** https://arxiv.org/abs/2405.04237
**Also published:** Mathematics 13(22):3608, Nov 2025

This paper presents communication-avoiding CholQR2 with block Gram-Schmidt stabilization for distributed GPU systems. Key results: up to 80x faster than ScaLAPACK Householder QR on multi-GPU systems, and 6x faster on CPU-only systems. Handles condition numbers up to 10^15 by interleaving CholQR steps with Gram-Schmidt orthogonalization.

**For us:** The distributed aspects are irrelevant (single GPU). But the technique of interleaving CholQR with Gram-Schmidt ("mCQRGSI+") is interesting for robustness — it provides sCQR3-level stability without the third iteration, similar to rand_cholQR but deterministic.

**Open-source:** The HybridScale/CholeskyQR2-IM repository (https://github.com/HybridScale/CholeskyQR2-IM) has a CUDA+ROCm implementation.

### 5b. TSQR vs CholQR

No new 2024-2026 papers directly comparing TSQR and CholQR on single-GPU. The consensus from existing literature: CholQR2 dominates TSQR for tall-skinny matrices on single GPU because CholQR2 uses only BLAS-3 operations (SYRK + Cholesky + TRSM) while TSQR requires a reduction tree of small Householder QRs. TSQR's advantage is in distributed settings (less communication) and for extremely ill-conditioned matrices (inherently stable).

**For our 4096x4096 and 4096x1024:** CholQR2 is the right choice. TSQR would only matter for distributed multi-GPU or if we needed column-pivoted QR.

---

## Finding 6: Condition Number Estimation — Can We Skip the Second Iteration?

No new papers specifically address cheap condition number estimation for deciding between CholQR1 and CholQR2. However, from the existing literature:

### Known Bounds

- **CholQR1 is sufficient when:** kappa_2(A) < u^{-1/2} ~ 4096 (FP32) or ~724 (BF16 accum)
- **CholQR2 is needed when:** kappa_2(A) >= u^{-1/2} but < u^{-1} ~ 10^7

### Cheap Estimation Tricks

1. **Diagonal of R:** After the first CholQR iteration, R = chol(A^T@A). The ratio max(diag(R)) / min(diag(R)) is a cheap estimate of kappa(A). If this ratio < ~100, CholQR1 is sufficient. Cost: O(N) — just scan the diagonal of R you already computed.

2. **Cholesky residual check:** After the first CholQR, compute ||Q^T@Q - I||_F (or a cheap proxy: sample a few columns of Q^T@Q). If the orthogonality error is already < 10*u, skip the second iteration. Cost: sampling k columns costs O(m*k) for the GEMV.

3. **rand_cholQR approach:** The Higgins et al. paper's sketching step implicitly handles this — the sketch-based preconditioner ensures one CholQR iteration always suffices, without explicitly estimating the condition number.

**Recommendation:** For the worker's current test matrices (random Gaussian, kappa ~200-400), the cheapest approach is to just check diag(R) ratio after the first CholQR. If max/min < 100, skip iteration 2. This saves ~40% of total time when applicable, with negligible estimation cost.

---

## Finding 7: 3xTF32 and BF16x9 for SYRK

### 7a. 3xTF32 — NOT Applicable on sm_120

**Source:** Ootomo & Yokota (2022), arXiv:2203.03341
**URL:** https://arxiv.org/abs/2203.03341

The 3xTF32 algorithm decomposes FP32 GEMM into 3 TF32 tensor core passes, achieving 1.7x over native FP32 on A100. However, TF32 MMA (m16n8k8) has a known B-fragment diagonal broadcast defect on sm_120, making 3xTF32 unusable on RTX 5090.

### 7b. BF16x9 — Available in cuBLAS 13.2, Applies to SYRK

**Source:** cuBLAS 13.2 docs + NVIDIA blog
**URL:** https://docs.nvidia.com/cuda/cublas/

cuBLAS 13.2 added `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` which decomposes FP32 into 3 BF16 values (a = a0 + 2^{-8}*a1 + 2^{-16}*a2) and computes 9 BF16 tensor core GEMMs to recover full FP32 accuracy. Performance: 3-4x native FP32 on Blackwell.

**Critical for SYRK:** cuBLAS 13.2 explicitly added BF16x9 support to `cublas[SC]syr[2]k`. This means the SYRK step in CholQR2 can use tensor-core-accelerated FP32-accurate SYRK by simply setting the compute type.

**However:** BF16x9 requires "special hardware features" on compute capabilities 10.0 and 10.3 only. sm_120 (RTX 5090) is compute capability 12.0 — it is unclear whether BF16x9 is supported. Must test empirically. If it works, it is the easiest path to faster SYRK with zero precision loss.

### 7c. Ozaki Scheme — Emerging Alternative

**Source:** Schwarz et al. (2025), arXiv:2511.13778 — "Automatic Dynamic Precision"
**URL:** https://arxiv.org/abs/2511.13778

NVIDIA's ADP framework extends the Ozaki scheme for GPU-resident FP64 emulation using FP16/FP8/FP4 tensor cores. Achieves 2.3x over native FP64 on GB200, 13.2x on RTX Pro 6000. While targeted at FP64 emulation, the same decomposition principle applies to FP32 emulation using BF16 tensor cores.

Also: Mukunoki (2025, arXiv:2508.00441) explored FP8 tensor cores for the Ozaki scheme on "RTX Blackwell architecture GPU" — this may include sm_120 testing.

And: Ozaki Scheme II (arXiv:2504.08009) uses Chinese Remainder Theorem for integer-based decomposition, achieving 7.4-9.8 TFLOPS FP64 emulation on RTX 4090 (vs native FP64 which is negligible on consumer GPUs).

**For CholQR2 SYRK:** The BF16x9 cuBLAS path is simpler. Only pursue custom Ozaki if BF16x9 is not supported on sm_120.

---

## Finding 8: Communication-Avoiding QR (CAQR) with CholQR Building Blocks

**Source:** Mijic et al. (2024), arXiv:2405.04237
**URL:** https://arxiv.org/abs/2405.04237

The primary recent work combining CAQR with CholQR is the Mijic et al. paper (Finding 5a above), which uses CA-CholQR2 as the panel factorization within a blocked QR. For single-GPU, CAQR's communication-avoidance is less relevant (no inter-node communication). The key CAQR insight for single-GPU is: using CholQR instead of Householder for panel factorization within blocked QR can eliminate the sequential panel bottleneck.

**However:** The worker's current CholQR2 approach is already fundamentally different from blocked QR — it does not have panels at all. CholQR2 treats the entire matrix as one "panel" and does SYRK + Cholesky + TRSM. CAQR/blocked approaches would only matter if the worker switches to a different algorithm, which is not warranted given the current 1.58x lead over cuSOLVER.

---

## Summary: Priority-Ranked Optimization Paths

| Priority | Technique | Expected Gain | Effort | Notes |
|----------|-----------|---------------|--------|-------|
| 1 | Profile components (SYRK/Chol/TRSM breakdown) | Identifies bottleneck | Low | Must do first |
| 2 | Single-iteration CholQR1 (check diag(R) ratio) | ~40% if applicable | Low | Check max/min diag(R) < 100 |
| 3 | BF16x9 SYRK via cuBLAS compute type | ~2-3x on SYRK step | Low | Test if sm_120 supports it |
| 4 | Custom triangle-aware SYRK (from linalg brief) | ~1.5x on SYRK step | Medium | Skip upper-triangle tiles |
| 5 | Recursive TRSM with custom GEMM | ~1.5x on TRSM step | Medium | See cross-pollination brief |
| 6 | CUDA graphs for launch overhead | ~0.5-1ms | Low | Capture 2-iteration loop |
| 7 | rand_cholQR for robustness | ~equal speed, any kappa | Medium | Only if ill-conditioned inputs needed |

---

## Sources

- Zhang & Wu (2019): https://arxiv.org/abs/1912.05508
- Fukaya et al. (2018): https://arxiv.org/abs/1809.11085
- Fan, Guan, Qiao (2024): https://arxiv.org/abs/2408.06311
- Higgins et al. (2023/2025): https://arxiv.org/abs/2309.05868
- Mijic et al. (2024): https://arxiv.org/abs/2405.04237
- Ootomo & Yokota (2022): https://arxiv.org/abs/2203.03341
- Schwarz et al. (2025): https://arxiv.org/abs/2511.13778
- Mukunoki (2025): https://arxiv.org/abs/2508.00441
- Ozaki, Uchino, Imamura (2025): https://arxiv.org/abs/2504.08009
- HybridScale/CholeskyQR2-IM: https://github.com/HybridScale/CholeskyQR2-IM
- cuBLAS 13.2 BF16x9 docs: https://docs.nvidia.com/cuda/cublas/
