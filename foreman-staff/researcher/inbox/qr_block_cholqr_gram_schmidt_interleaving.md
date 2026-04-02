# Block CholQR-Gram-Schmidt Interleaving: 80x Speedup for Ill-Conditioned QR on GPU

**Source:** Mijic, Kaushik, Davidovic, "QR factorization of ill-conditioned tall-and-skinny matrices on distributed-memory systems," arXiv:2405.04237 (May 2024)
**Also:** Mijic et al., "Scalable QR Factorisation of Ill-Conditioned Tall-and-Skinny Matrices on Distributed GPU Systems." Mathematics 13(22):3608, Nov 2025.
**URL:** https://arxiv.org/abs/2405.04237
**Relevant to:** QR worker
**Worker's current problem:** CholQR2 beating cuSOLVER 1.58x at 4096x4096. Worker may encounter stability issues with ill-conditioned matrices. This provides a drop-in alternative that handles all condition numbers.

---

## What This Is

A modified CholQR2 algorithm that interleaves CholeskyQR steps with Gram-Schmidt orthogonalization across column panels. This achieves up to **80x speedup over Householder QR (ScaLAPACK) on GPU systems** while handling condition numbers up to 10^15 -- far beyond standard CholQR2's limit of ~10^7 (FP32).

---

## Why It Matters for Us

The QR worker's CholQR2 works well for well-conditioned random matrices. But:
1. If we add ill-conditioned test cases, CholQR2 will break (Cholesky of the Gram matrix fails)
2. Shifted CholQR3 handles this but costs 50% more (3 iterations vs 2)
3. This interleaving approach handles all condition numbers while staying at the same cost as CholQR2 for well-conditioned inputs

It's a strictly better algorithm: same speed when matrices are easy, works when matrices are hard.

---

## Key Technique: Panel-Based Interleaving

### Standard CholQR2 (the worker's current approach):
```
[Q1, R1] = CholQR(A)         // Full matrix CholQR
[Q2, R2] = CholQR(Q1)        // Second pass for orthogonality
R = R2 * R1                  // Combine
```
Fails when kappa(A) > ~10^7 (FP32) because A^T*A is numerically singular.

### Block CholQR-GS Interleaved:
```
Split A into b column panels: A = [A_1, A_2, ..., A_b]
Each panel has nb columns (e.g., nb = n/3 for 3 panels)

For panel j = 1..b:
    // Step 1: CholQR on panel j (first pass)
    [Q_j, R_j] = CholQR(A_j)

    // Step 2: Gram-Schmidt against all PREVIOUS panels
    // (orthogonalize Q_j against Q_1, ..., Q_{j-1})
    For i = 1..j-1:
        S_ij = Q_i^T * Q_j    // GEMM: small-ish
        Q_j = Q_j - Q_i * S_ij // GEMM: tall-skinny

    // Step 3: CholQR again (second pass for quality)
    [Q_j, R2_j] = CholQR(Q_j)
```

### Why It Works for Ill-Conditioned Matrices

The key insight: by dividing into b panels, each individual panel's condition number is much smaller than the full matrix. A matrix with kappa(A) = 10^15 might have individual panels with kappa ~ 10^5. CholQR easily handles 10^5.

The Gram-Schmidt step between panels removes cross-panel ill-conditioning. The second CholQR pass cleans up within-panel orthogonality.

### Communication Cost

Total communication: n*(n + b)*log2(P) -- lower than standard CholQR2's 2*n^2*log2(P) for single-GPU (P=1, this is just data movement through shared memory/registers).

---

## Performance

- **vs Householder (ScaLAPACK):** 6x faster on CPU, up to 80x faster on GPU
- **vs standard CholQR2:** Same cost for well-conditioned matrices (b=1 panel degenerates to standard CholQR2)
- **vs Shifted CholQR3:** Avoids the diagonal shift overhead, no need to compute shift parameter

---

## Application to Our sm_120 QR Kernel

### For CholQR2 (current approach):
If the current CholQR2 is working fine with well-conditioned test matrices, this is a future robustness upgrade rather than an immediate optimization.

### Implementation:
1. Add a condition number estimator (cheap: compare largest/smallest diagonal of R from first CholQR)
2. If kappa < threshold, use standard CholQR2 (current code, fastest path)
3. If kappa > threshold, switch to interleaved block CholQR-GS with b=3 panels
4. Total cost overhead for the estimator: negligible (just reading diagonal of R)

### Component reuse:
All sub-operations (SYRK, Cholesky, TRSM, GEMM for GS step) are the same primitives the worker already has. No new kernels needed -- just different orchestration.

---

## Caveats

1. **Well-conditioned matrices don't benefit** -- this is purely a robustness improvement. If all test matrices are random Gaussian (kappa ~ sqrt(m/n)), standard CholQR2 is fine.
2. **The Gram-Schmidt step adds FLOPs** -- for b=3 panels, the GS inter-panel orthogonalization is 3 extra GEMMs per panel (small but nonzero).
3. **The paper targets distributed GPU systems** -- on single-GPU (our case), the communication savings don't apply. The win is purely algorithmic stability.
4. **Reference implementation available:** https://github.com/HybridScale/CholeskyQR2-IM (CUDA + ROCm, open source)
