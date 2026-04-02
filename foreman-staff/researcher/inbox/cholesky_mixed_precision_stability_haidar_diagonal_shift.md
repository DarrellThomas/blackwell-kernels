# Mixed-Precision Cholesky Stability: Diagonal Shift and GMRES-IR Convergence

**Source:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7302814/ | https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf
**Relevant to:** numerical worker (Cholesky monolithic kernel)
**Worker's current problem:** BF16 MMA for SYRK loses ~1e-3 precision. Will this cause the Cholesky factorization to fail (loss of positive definiteness)? If so, what preprocessing or refinement can recover accuracy?

## What This Is

Haidar et al. (SC18, ICCS2020) published the definitive work on mixed-precision Cholesky with FP16/BF16 tensor core SYRK updates. Their key findings:
1. Loss of positive definiteness CAN occur when converting to low precision
2. A diagonal preprocessing step (Higham's technique) prevents this
3. GMRES-based iterative refinement recovers full FP64 accuracy
4. The combined solver works for condition numbers up to 10^9

## Why It Matters for Us

The worker is considering BF16 MMA (m16n8k16) for the trailing SYRK in a monolithic Cholesky kernel. This is the EXACT approach Haidar validated. The key question is: will the reduced precision break the factorization? The answer is: it depends on the matrix condition number, and there is a systematic fix.

## Key Technique: Three-Part Preprocessing

Before factorizing A in low precision, apply three preprocessing steps:

### 1. Two-Sided Diagonal Scaling

Normalize the matrix so diagonal elements are in [0, 1]:

```
D = diag(1/sqrt(diag(A)))
H = D * A * D
```

This ensures all diagonal elements of H are exactly 1.0. The scaling preserves positive definiteness and improves numerical stability.

### 2. Diagonal Shift (Higham's Technique)

If the matrix is near-singular relative to low precision, add a small perturbation:

```
H' = H + c * u_lp * I
```

where:
- u_lp = machine epsilon of the low-precision format (BF16: ~3.9e-3; FP16: ~4.9e-4)
- c = tunable constant (Haidar uses fractional values c < 1, not just integers)
- I = identity matrix

The shift pushes eigenvalues away from zero in the low-precision regime, preventing loss of definiteness. The perturbation is small enough that iterative refinement can correct for it.

**For BF16 (our case):** u_bf16 ≈ 2^(-8) ≈ 3.9e-3. A shift of c=0.5 gives alpha ≈ 2e-3. This is comparable to the BF16 conversion error itself, so the perturbation is well-matched.

### 3. Matrix Scaling

Scale the entire matrix to avoid overflow/underflow in low precision:

```
H'' = beta * H'
```

where beta_max ≈ 0.1 for FP16. For BF16, this is less critical since BF16 shares FP32's exponent range (no exponent-related overflow). However, scaling by 0.5 or 0.25 can still reduce underflow probability in the factored values.

## Key Technique: GMRES-Based Iterative Refinement

After computing the approximate Cholesky factor L_lp in low precision:

```
Solve A*x = b using:
1. x_0 = L_lp^{-T} * L_lp^{-1} * b    (forward/back substitution)
2. For k = 1, 2, ...:
   r_k = b - A * x_k                    (residual in FP32/FP64)
   d_k = GMRES(L_lp^{-T} * A * L_lp^{-1}, L_lp^{-T} * r_k)  (preconditioned solve)
   x_{k+1} = x_k + L_lp^{-1} * d_k
   if ||r_{k+1}|| < tol: break
```

The GMRES inner solver uses the low-precision Cholesky factors as a preconditioner for the original FP32/FP64 matrix A. This is much more robust than classic iterative refinement.

### Convergence Iteration Counts

| Matrix Property | Classic IR | GMRES-IR |
|----------------|-----------|----------|
| Well-conditioned, arithmetic eigenvalues | 2-3 | 3-4 |
| Moderate condition (kappa ~10^4) | DIVERGES | 5-8 |
| Clustered eigenvalues | DIVERGES | 27-32 |
| Maximum condition (kappa ~10^9) | DIVERGES | converges |

### Condition Number Limits

| Method | Max Condition Number |
|--------|---------------------|
| Low-precision Cholesky alone (no refinement) | kappa < 1/u_lp ≈ 256 for BF16 |
| Classic iterative refinement | kappa < 1/u_fp32 ≈ 10^7 |
| GMRES-IR with preprocessing | kappa < 10^9 |

## Application to Our Monolithic Kernel

For the worker's monolithic Cholesky on N=4096:

### When BF16 SYRK is SAFE (no preprocessing needed):
- Well-conditioned SPD matrices (kappa < 100)
- Most ML covariance matrices, kernel matrices, Gram matrices
- Matrices from physics simulations (stiffness matrices)

### When Preprocessing is Needed:
- Ill-conditioned matrices (kappa > 256 for BF16)
- Near-singular matrices
- Matrices with clustered small eigenvalues

### Practical Recommendation:

1. **For the monolithic kernel MVP:** Skip preprocessing. Use BF16 SYRK directly. Test accuracy against cuSOLVER. Most practical SPD matrices will work fine.

2. **If accuracy fails for some matrices:** Add diagonal scaling (step 1 above) as a preprocessing kernel before the monolithic factorization. This costs one global memory pass to compute and apply D.

3. **For FP32-quality results:** Use BF16x9 SYRK inside the monolithic kernel (9 MMA calls). This eliminates ALL precision concerns at the cost of 9x SYRK compute.

4. **For FP64-quality results:** Run monolithic Cholesky with BF16 SYRK, then apply 3-8 GMRES-IR iterations using the factors as a preconditioner. The GMRES solve is a separate kernel.

## Performance Impact

The Haidar paper reports:
- Mixed-precision Cholesky with FP16 SYRK: **4.7x speedup** over FP64 direct solve on V100
- With GMRES-IR (3-4 iterations): **2.3-2.7x speedup** including refinement cost
- Preprocessing overhead: negligible (one matrix scaling pass)

For our monolithic kernel: the SYRK is not the dominant cost (launch overhead is). So whether we use 1xBF16 or 9xBF16 for SYRK, the total kernel time difference is small relative to the 0.38ms launch overhead being eliminated.

## Caveats

1. **BF16 has wider exponent than FP16.** Haidar's preprocessing was designed for FP16 (5-bit exponent, overflow-prone). BF16 shares FP32's 8-bit exponent, so the matrix scaling step (step 3) is largely unnecessary for BF16. The diagonal shift (step 2) is still relevant.

2. **GMRES-IR is a separate solve step.** It requires TRSM with the factored L, plus SPMV with the original A. This cannot be fused into the monolithic factorization kernel. If iterative refinement is needed, it must be a post-processing step.

3. **For factorization-only (no solve):** If the worker only produces L without solving Ax=b, iterative refinement doesn't apply. In that case, the accuracy of L itself depends on the SYRK precision. Use BF16x9 for guaranteed FP32-quality L.

## Sources

- [Haidar et al. "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic" (SC18)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf) -- The definitive mixed-precision Cholesky paper
- [Haidar et al. "Investigating FP16-Enabled Mixed-Precision Solvers for SPD Matrices" (ICCS2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7302814/) -- Extended study with diagonal shift details
- [Higham et al. -- Squeezing a Matrix into Half Precision](https://doi.org/10.1137/18M1229511) -- Original diagonal shift technique
- [ORNL -- Mixed-Precision Iterative Refinement Using Tensor Cores](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0110) -- GMRES-IR theory and convergence analysis
