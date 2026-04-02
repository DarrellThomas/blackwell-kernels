# CholQR2 Optimization: Cross-Pollination from Other Workers

**Source:** Cross-project analysis (linalg, numerical/cholesky, gemm worker results)
**Relevant to:** qr worker
**Worker's current problem:** CholQR2 at 1.58x cuSOLVER (14.08ms vs 22.26ms at 4096x4096). Next step: profile components and optimize bottleneck.

## What This Is

The QR worker's CholQR2 is built from three components: SYRK (A^T@A), Cholesky,
and TRSM. Other workers in this project have optimized all three independently.
This brief connects those results to help the QR worker identify where to focus.

## Component Cost Estimates (4096x4096, FP32, 2 iterations)

CholQR2 total = 2 × (SYRK + Cholesky + TRSM) = 14.08ms

Estimated breakdown (verify with profiling):
- **SYRK (torch.mm(A.T, A)):** cuBLAS FP32 GEMM at 4096^3 ≈ 0.72ms × 2 = ~1.4ms
- **Cholesky (torch.linalg.cholesky):** cuSOLVER at N=4096 ≈ 1.8ms × 2 = ~3.6ms
- **TRSM (torch.linalg.solve_triangular):** likely ~4.5ms × 2 = ~9.0ms
- **Total estimated:** ~14.0ms (matches observation)

**TRSM is likely the dominant cost (~64% of total).**

## Cross-Pollination: What Our Workers Know

### 1. TRSM (linalg worker)
The linalg worker's TRSM is at 0.82x cuBLAS (using torch.linalg.solve_triangular
in F32). That IS the same call the QR worker uses. The linalg worker identified:
- Recursive decomposition (KBLAS-style) can reach 1.2-1.5x cuBLAS
- Mixed-precision: use our BF16 GEMM (0.97x cuBLAS) for off-diagonal blocks
- Custom 32x32 base-case with warp-shuffle forward substitution

**For QR:** The TRSM in CholQR2 is: Q = A @ R^{-1}, where A is M×N and R is
N×N upper triangular. This is a right-side TRSM. At 4096×4096, the GEMM calls
in recursive TRSM dominate — so our fast GEMM directly helps.

### 2. Cholesky (numerical worker)
The numerical worker's Cholesky is at 0.55x cuSOLVER for N=4096. That means
cuSOLVER's monolithic kernel is hard to beat. But for CholQR2, the Cholesky
step is only ~13% of total time — not the bottleneck.

**Key insight:** For smaller matrices (N=1024, tall-skinny QR), the Gram
matrix G=A^T@A is only 1024×1024. Cholesky at N=1024 is ~0.4ms from cuSOLVER
— negligible. TRSM dominates even more at this size.

### 3. SYRK → GEMM (gemm worker)
The SYRK A^T@A in torch.mm calls cuBLAS SGEMM. Our custom GEMM is:
- BF16: 0.97x cuBLAS (slower in FP32 equivalent)
- FP8: 1.29x cuBLAS (but precision might be insufficient for the Gram matrix)

**For QR:** The SYRK is already using cuBLAS's fastest GEMM. Custom SYRK with
BF16 MMA + FP32 accumulation could work if precision is acceptable (the Gram
matrix A^T@A just needs enough precision for Cholesky to succeed — BF16
accumulation with FP32 accumulators should be fine).

## Optimization Priority (by impact)

### Priority 1: TRSM optimization (~9ms of 14ms total)

The biggest win. Three approaches:

**a) Recursive TRSM with custom GEMM (modest effort):**
```python
def recursive_trsm(A, R, base_size=128):
    if R.shape[0] <= base_size:
        return torch.linalg.solve_triangular(R, A, upper=True, left=False)
    n = R.shape[0] // 2
    R11, R12, R22 = R[:n,:n], R[:n,n:], R[n:,n:]
    Q_right = recursive_trsm(A[:, n:], R22, base_size)
    A[:, :n] -= Q_right @ R12.T  # GEMM — use our custom kernel here
    Q_left = recursive_trsm(A[:, :n], R11, base_size)
    return torch.cat([Q_left, Q_right], dim=1)
```
This converts TRSM into GEMMs (which we're good at) + small base-case TRSMs.

**b) Use linalg's shipped TRSM primitive (if available):**
Check `/data/src/bwk/common/csrc/primitives/` for a shipped TRSM. If the linalg
worker ships a faster TRSM, the QR worker gets it for free.

**c) cuBLASDx device-side TRSM (if doing monolithic kernel):**
Not applicable at the Python level, but for a future CUDA CholQR kernel.

### Priority 2: Mixed-precision SYRK (~1.4ms, small but free speedup)

Replace `torch.mm(A.T, A)` with a custom BF16 GEMM that computes A^T@A with
FP32 accumulation. Our BF16 GEMM at 0.97x cuBLAS uses tensor cores (128
TFLOPS BF16 vs 83 TFLOPS FP32). Even at 0.97x cuBLAS BF16, the BF16 SYRK
should be faster than FP32 cuBLAS SYRK because of the ~1.5x throughput advantage.

**Precision check:** BF16 SYRK with FP32 accumulators introduces ~1e-3 relative
error per element. For the Gram matrix, this is acceptable — the Cholesky step
has FP32 precision, and CholQR2's second iteration corrects orthogonality.

### Priority 3: Reduce allocations & launches (~0.5-1ms)

- Pre-allocate workspace for G (N×N), R (N×N)
- Use CUDA graphs to capture the 2-iteration loop
- In-place operations where possible (A → Q in-place)

### Priority 4: Single iteration CholQR1 (saves ~40%)

If the input matrices are well-conditioned (kappa < sqrt(1/eps) ≈ 4096 for FP32),
single-iteration CholQR may suffice. Test orthogonality error |Q^T@Q - I|_F
on your test matrices. Random Gaussian matrices typically have kappa ≈ 200-400.

**Risk:** Some test matrices may be ill-conditioned. Need to verify test suite
coverage before dropping the second iteration.

## Recent Research: Shifted CholQR3 (for ill-conditioned matrices)

Fukaya et al. (2020) "Shifted CholeskyQR3: A Backward Stable Algorithm for
Solving Linear Systems" adds a diagonal shift σI before Cholesky to guarantee
positive-definiteness even for ill-conditioned A:

```
G = A^T @ A + σ * I     (shifted Gram matrix)
R = chol(G)^T
Q = A @ R^{-1}
```

This requires 3 iterations instead of 2, but handles any condition number.
For the current test matrices (random Gaussian), CholQR2 is sufficient.

## Caveats

1. **Profile first** before optimizing. The cost breakdown above is estimated.
   Run `torch.cuda.Event` timing around each component to get ground truth.

2. **Right-side TRSM** (Q = A @ R^{-1}) is different from left-side TRSM
   (L @ X = B). The linalg worker's TRSM research covers both sides, but
   make sure you're calling the right variant.

3. **CholQR2 already beats cuSOLVER by 1.58x.** The Householder approach
   (0.43x cuSOLVER) was correctly rejected. The question is how much further
   CholQR2 can be pushed, not whether it's the right algorithm.
