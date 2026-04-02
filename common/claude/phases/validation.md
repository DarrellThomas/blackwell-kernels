# Phase: Validation

You are in the **validation phase**. Your kernel must pass the BLAS compliance
checklist BEFORE it can ship. Speed is irrelevant if the kernel gives wrong
answers or crashes on valid input.

## First: Read Your Job Spec

```bash
fb job-show <your-job-id>    # shows spec with exact acceptance criteria
```

**Priority order: CORRECT → COMPLETE → FAST**

## Compliance Test

Run this BEFORE submitting. It tests all sizes, shapes, transpose cases,
alpha/beta, accuracy, and edge cases:

```bash
CUDA_VISIBLE_DEVICES=1 python3 /data/src/bwk/common/scripts/blas_compliance.py \
    python/blackwell_kernels/<your_module>.py <your_function> \
    --op <gemm|trsm|syrk|trmm|chol|lu|qr|gemv|dot|all>
```

Fix EVERY failure before moving on. The watchdog runs this same test
automatically — if it fails, your job goes back to rework.

---

## 1. Correctness (non-negotiable)

### Matrix Sizes — ALL must work
- [ ] Square: 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65
- [ ] Square large: 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025
- [ ] Square very large: 2048, 4096, 8192
- [ ] Rectangular tall: Mx1, Mx7, Mx15, Mx63, Mx64, Mx65, Mx128, MxN where M >> N
- [ ] Rectangular wide: 1xN, 7xN, 15xN, 63xN, 64xN, 65xN, 128xN, MxN where N >> M
- [ ] Degenerate: 0x0, 0xN, Mx0 (must not crash — return empty or error gracefully)
- [ ] Non-tile-aligned: sizes that are NOT multiples of your tile size
- [ ] Prime sizes: 7, 13, 17, 31, 61, 127, 251, 509, 1021

### Numerical Accuracy — ALL must pass
- [ ] Relative error vs reference < N * machine_epsilon
- [ ] Well-conditioned random matrices (condition number ~N)
- [ ] Ill-conditioned matrices (condition number 1e6, 1e10, 1e14)
- [ ] Matrices with zeros, ones, identity, near-zero values
- [ ] Large values (1e15) and small values (1e-15)
- [ ] Mixed scales (some elements 1e10, others 1e-10)
- [ ] NaN/Inf: must not produce NaN/Inf from finite inputs
- [ ] Compare against OpenBLAS reference at EVERY size

### Edge Cases — ALL must not crash
- [ ] Singular matrices (for factorizations: return error code)
- [ ] Non-positive-definite input to Cholesky (return error code)
- [ ] Rank-deficient input to QR (handle gracefully)
- [ ] Zero matrix input
- [ ] Identity matrix input (results should be exact)

## 2. BLAS Interface Completeness

### For GEMM-class operations
- [ ] Alpha/beta scaling: C = alpha * op(A) * op(B) + beta * C
- [ ] Alpha=0: must not read A or B (beta*C only)
- [ ] Beta=0: must not read C (alpha*A*B only)
- [ ] Alpha=1, beta=0: standard (C = A*B)
- [ ] Alpha=1, beta=1: accumulate (C += A*B)
- [ ] Alpha=-1, beta=1: subtract (C -= A*B) — used by Cholesky
- [ ] Arbitrary alpha/beta: alpha=2.7, beta=-0.3

### Transpose flags
- [ ] NN (no transpose, no transpose)
- [ ] TN (transpose A, no transpose B)
- [ ] NT (no transpose A, transpose B)
- [ ] TT (transpose both)

### Leading dimensions (stride support)
- [ ] lda = M (contiguous, standard case)
- [ ] lda > M (sub-matrix of a larger matrix — THIS IS HOW LAPACK CALLS YOU)
- [ ] lda with different values for A, B, C
- [ ] Non-contiguous memory access patterns

### For triangular operations (TRMM, TRSM, Cholesky)
- [ ] Upper triangular (uplo = 'U')
- [ ] Lower triangular (uplo = 'L')
- [ ] Unit diagonal (diag = 'U')
- [ ] Non-unit diagonal (diag = 'N')
- [ ] Left side (side = 'L'): op(A) * B
- [ ] Right side (side = 'R'): B * op(A)

### For vector operations (GEMV, DOT, NRM2, AXPY, etc.)
- [ ] incx = 1 (contiguous, standard)
- [ ] incx > 1 (strided access)
- [ ] incx < 0 (reverse order — yes, BLAS supports this)
- [ ] incy = 1, incy > 1, incy < 0

### Column-major / Row-major
- [ ] Column-major layout (Fortran/LAPACK/Octave default)
- [ ] Row-major layout (C default)

## 3. Performance (only after 1 and 2 pass)

- [ ] Benchmark at tile-aligned sizes
- [ ] Benchmark at non-tile-aligned sizes (must not be catastrophically slow)
- [ ] vs_ref >= 1.0 at primary benchmark size
- [ ] No performance cliff at size boundaries (N=64 fast, N=65 10x slower = bug)

## The Rule

**If a kernel crashes or gives wrong answers for ANY valid input, it is broken.**
It doesn't matter if it's 10x faster than cuBLAS on 4096x4096. A user will call
it with a 73x129 matrix and it will crash and they will never trust it again.

## When You Pass

When all applicable checklist items pass:
1. Set job state: `fb job-update <id> --state testing_pass --by <kernel> --reason "compliance passed"`
2. The watchdog will automatically run edge tests and advance your job.

When something fails:
1. Fix the failure
2. Re-run the compliance test
3. Do NOT move to testing_pass until everything passes
