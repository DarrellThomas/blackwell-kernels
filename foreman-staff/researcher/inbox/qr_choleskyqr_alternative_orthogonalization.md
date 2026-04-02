# CholeskyQR2: Fast Alternative QR for Tall-Skinny Matrices on GPU

**Sources:**
- HybridScale/CholeskyQR2-IM (https://github.com/HybridScale/CholeskyQR2-IM) -- GPU implementation with Gram-Schmidt stabilization
- "Scalable QR Factorisation of Ill-Conditioned Tall-and-Skinny Matrices on Distributed GPU Systems." Mathematics 13(22):3608, Nov 2025. (https://www.mdpi.com/2227-7390/13/22/3608)
- "Analysis of Randomized Householder-Cholesky QR Factorization with Multisketching." Numerische Mathematik, 2025. (https://link.springer.com/article/10.1007/s00211-025-01492-5)
**Relevant to:** QR worker (for tall-skinny sub-problems within blocked QR)
**Worker's current problem:** Panel factorization in blocked QR produces tall-skinny sub-problems. CholeskyQR is a fast alternative for these.

## What This Is

CholeskyQR computes QR factorization by forming the Gram matrix A^T * A (a GEMM), Cholesky-factoring it to get R, then computing Q = A * R^{-1}. It's extremely fast on GPU because both steps are BLAS-3, but numerically unstable for ill-conditioned matrices. CholeskyQR2 runs CholeskyQR twice to recover orthogonality. Shifted variants (sCQR3) add diagonal shifts for ill-conditioned problems.

## Why It Matters for Us

For tall-skinny sub-problems (e.g., m x nb panel tiles in CAQR, or the recursive panel base case), CholeskyQR can be significantly faster than Householder QR because:
1. **A^T * A is a GEMM**: (nb x m) * (m x nb) = nb x nb. This uses tensor cores perfectly.
2. **Cholesky on nb x nb**: We already have Cholesky at 1.0x cuSOLVER. For nb=32-64, this is trivial.
3. **Q = A * R^{-1}**: TRSM on m x nb. We have TRSM from linalg/.

Total: 1 GEMM + 1 potrf + 1 TRSM. All BLAS-3. Zero BLAS-2.

Compare to Householder GEQR2: nb sequential columns of GEMV + rank-1 update (BLAS-2).

## Key Technique

### CholeskyQR:
```
R^T * R = A^T * A    // Form Gram matrix (GEMM), then Cholesky factor
Q = A * R^{-1}       // Backward solve (TRSM)
```

### CholeskyQR2 (for stability):
```
[Q1, R1] = CholeskyQR(A)     // First pass -- Q1 may not be orthogonal
[Q2, R2] = CholeskyQR(Q1)    // Second pass -- Q2 is orthogonal to machine eps
R = R2 * R1                  // Combine R factors (TRMM)
```

### Condition number requirement:
- CholeskyQR: requires kappa(A) < 1/sqrt(eps) ~ 10^4 for FP32
- CholeskyQR2: requires kappa(A) < 1/eps ~ 10^7 for FP32
- Shifted CholeskyQR3: works for any condition number (adds diagonal shift)

### Recent advances (2025):
- **mCQRGSI+**: Combines CholeskyQR speed with Gram-Schmidt stabilization. Handles ill-conditioned matrices without shifting.
- **rand_cholQR**: Uses randomized sketching to estimate condition number before choosing CholeskyQR vs CholeskyQR2. Nearly as fast as CholeskyQR2, handles all condition numbers.
- **CholeskyQR2-IM**: Open-source GPU library (CUDA + ROCm) for distributed tall-skinny QR using CholeskyQR2 with Gram-Schmidt stabilization.

## Application to sm_120

### Where to use CholeskyQR in our QR implementation:

1. **CAQR panel tiles**: If using Communication-Avoiding QR for the panel, each tile (e.g., 256 x 32) can be factored with CholeskyQR instead of Householder. The 32x32 Gram matrix GEMM + Cholesky is fast.

2. **Recursive QR base case**: When the recursion reaches a small enough sub-problem, CholeskyQR can replace GEQR2. Threshold: when m/nb > ~4 (tall enough that BLAS-3 dominates BLAS-2).

3. **NOT for the main panel**: The main panel factorization needs Householder reflectors (for the compact WY representation used in trailing updates). CholeskyQR produces Q explicitly, not as Householder vectors. Would need to convert Q to Householder form, which adds cost.

### Performance estimate:
For a 256 x 32 tile:
- CholeskyQR: 1 GEMM (32x256)*(256x32)=32x32 + potrf(32) + TRSM(256x32) ~ 3 kernel ops
- Householder GEQR2: 32 sequential columns of GEMV(256) + rank-1(256x31) ~ 32 sequential steps
- CholeskyQR should be ~5-10x faster for this sub-problem on GPU

### Building blocks we already have:
- BF16 GEMM for A^T * A (0.97x cuBLAS)
- Cholesky (potrf) for the Gram matrix (cuSOLVERDx device-side)
- TRSM for Q = A * R^{-1} (from linalg/)

## Caveats

1. **Output format**: CholeskyQR produces explicit Q, not Householder vectors + tau. For blocked QR that needs the compact WY representation (V, T) for trailing updates, Householder is required. CholeskyQR is only suitable for sub-problems where we need Q explicitly or where we can work with Q directly.

2. **Numerical stability**: CholeskyQR squares the condition number (kappa(A^T*A) = kappa(A)^2). For the panel tiles in our blocked QR, the condition number of each tile depends on the matrix -- may not always be well-conditioned. CholeskyQR2 (two passes) is the safe default.

3. **Not a replacement for Householder QR**: CholeskyQR is a complement for specific sub-problems, not a replacement for the full blocked Householder algorithm. The main QR factorization must produce Householder vectors for the trailing update.

4. **Cholesky failure**: If A^T * A is not positive definite (due to near-rank-deficiency or numerical issues), Cholesky factorization fails. Need a fallback to Householder for such cases.
