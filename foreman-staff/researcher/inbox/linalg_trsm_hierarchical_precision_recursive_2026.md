# TRSM: Hierarchical Precision and Recursive Acceleration (January 2026)

**Source:** [Hierarchical Precision and Recursion for Accelerating Symmetric Linear Solves on MXUs (arxiv 2601.08082, January 2026)](https://arxiv.org/abs/2601.08082)
**Relevant to:** linalg worker
**Worker's current problem:** TRSM at 0.82x cuBLAS F32. Planning a native BF16 TRSM. The key question: can mixed-precision (BF16 off-diagonal + FP32 diagonal) beat full-precision cuBLAS?
**Supplements:** `linalg_recursive_trsm_to_gemm.md`, `linalg_trsm_julia_recursive_portable_2025.md`

---

## What This Is

A January 2026 paper that recursively decomposes POTRF (Cholesky), TRSM, and SYRK operations, then applies **hierarchical mixed precision**: FP16 for large off-diagonal GEMM blocks, FP32/FP64 for small diagonal blocks. Results on H200: 5x TRSM speedup over FP64 cuBLAS, 14x SYRK speedup.

---

## Why It Matters for Us

The key insight transfers directly to our BF16 TRSM work:

**Recursive TRSM exposes a precision hierarchy.** At each recursion level:
- The **off-diagonal update** (GEMM call) is tolerant of reduced precision -- errors are bounded per-call and don't accumulate along the recursion
- The **diagonal block** (base-case TRSM) is precision-sensitive -- errors propagate through forward/back substitution

This means you can use:
- **BF16 GEMM** (0.97x cuBLAS, or FP8 at 1.34x) for the off-diagonal updates
- **FP32** for the base-case 32x32 diagonal TRSM

The paper proves this maintains 100x better accuracy than pure FP16 while retaining 88% of FP16's peak speedup.

---

## Key Technique: Mixed-Precision Recursive TRSM

```
recursive_trsm_mixed(L, B, n):
    if n <= NB_BASE:
        return trsm_fp32(L, B)  // FP32 base case for accuracy

    n_half = n / 2
    recursive_trsm_mixed(L_top, B_top, n_half)

    // Off-diagonal update: BF16 GEMM (tolerant of lower precision)
    gemm_bf16(B_bottom -= L_offdiag * B_top)

    recursive_trsm_mixed(L_bottom, B_bottom, n_half)
```

**Why the off-diagonal GEMM can be low precision:**
- The GEMM computes `B -= L * X`, a single matrix multiply
- Its error is bounded by O(eps * ||L|| * ||X|| * sqrt(K)) where K is the inner dimension
- This error is additive and gets "fixed" by the subsequent base-case TRSM
- In contrast, the base-case forward substitution has error that grows with the number of steps

---

## Concrete Numbers (from the paper, on H200)

| Operation | FP64 cuBLAS | Recursive FP16+FP64 | Speedup |
|-----------|-------------|---------------------|---------|
| TRSM      | baseline    | 5x                  | 5x      |
| SYRK      | baseline    | 14x                 | 14x     |
| Cholesky  | baseline    | 5x                  | 5x      |

Accuracy: 100x better than pure FP16, while retaining 88% of FP16 throughput.

---

## What This Means for the Worker

The worker is planning a BF16 TRSM. This paper provides justification for the mixed-precision approach:

1. **Use BF16 GEMM for off-diagonal updates** -- this is where most of the compute is, and it's precision-tolerant
2. **Use FP32 for the base-case diagonal solve** -- this is small and precision-sensitive
3. **The speedup comes from the GEMM dominance** -- as N grows, the fraction of work in GEMM grows quadratically. For N=1024 with NB=32, roughly 95% of FLOPs are in GEMM calls.
4. **Our FP8 GEMM (1.34x cuBLAS) could be used** for even more throughput, if the accuracy is sufficient. The paper's analysis suggests this should work for NRHS >= N (rectangular case where GEMM dominates even more).

---

## Caveats

- The paper targets H200 with FP16 tensor cores and FP64 scalar. Our sm_120 has BF16 tensor cores (m16n8k16) and FP32 scalar. The principle is the same but the precision "levels" differ: BF16 off-diagonal + FP32 diagonal (us) vs FP16 off-diagonal + FP64 diagonal (them).
- BF16 has fewer mantissa bits (8) than FP16 (11). The paper's accuracy analysis uses FP16 error bounds. BF16 will have ~8x larger per-element error. This may require a slightly larger base case (NB=64 instead of 32) to keep the recursion depth (and thus error accumulation) manageable.
- The paper's implementation is in Julia. The algorithmic insight (mixed precision recursive decomposition) transfers directly to our CUDA implementation.
