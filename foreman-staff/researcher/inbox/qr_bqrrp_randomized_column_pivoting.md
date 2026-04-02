# BQRRP: Randomized Column-Pivoted QR on GPU

**Source:** "Anatomy of High-Performance Column-Pivoted QR Decomposition." arXiv:2507.00976, July 2025. (https://arxiv.org/html/2507.00976v1)
**Also:** RandLAPACK library (https://github.com/BallisticLA/RandLAPACK)
**Relevant to:** QR worker (future consideration, not immediate priority)
**Worker's current problem:** Building standard geqrf (unpivoted QR). This brief covers pivoted QR for potential future extension.

## What This Is

BQRRP (Block QR with Randomized Pivoting) is a GPU-implementable algorithm for QR factorization with column pivoting (QRCP). Traditional QRCP (LAPACK's GEQP3) is extremely slow on GPU because it requires column norm updates and pivot selection that are inherently sequential. BQRRP uses randomized sampling to select pivots in blocks, enabling GPU-friendly BLAS-3 operations.

## Why It Matters for Us

Column-pivoted QR is needed for rank-revealing factorizations (used in least squares, low-rank approximation, SVD preprocessing). If the QR worker ever needs QRCP, this is the GPU-friendly approach. Standard GEQP3 is up to **100x slower** than unpivoted geqrf on GPU.

## Key Technique

### Algorithm Overview:
```
1. Generate random Gaussian matrix S (d x m), d = ceil(gamma * b)
2. Compute sketch: SA = S * A  (small, d x n matrix)
3. Main loop (iterate over blocks of width b):
   a. QRCP on sketch SA to select b pivot columns  (small problem, O(d*b^2))
   b. Permute columns of A according to pivots
   c. QR factorize the b selected columns (standard geqr2)
   d. Apply reflectors to remaining columns (LARFB -- BLAS-3!)
   e. Update sketch SA deterministically (no new randomness needed)
```

### Why it's fast on GPU:
- Step (a) is tiny (d x n sketch, d ~ 128-256)
- Step (b) is column permutation (memory-bound but unavoidable)
- Steps (c) and (d) are standard blocked QR operations -- BLAS-3
- No per-column norm updates (the GPU killer in standard GEQP3)

### Performance on H100:
- **65% of cuSOLVER's unpivoted geqrf** throughput
- This is remarkable for pivoted QR -- standard GEQP3 achieves maybe 5-10% of geqrf
- CPU version: up to **100x faster** than LAPACK's GEQP3

### Column permutation overhead:
On GPU, permuting columns is expensive (memory-bound, non-coalesced access). The paper explores "parallel pivots" strategies to reduce this overhead.

## Application to sm_120

### Not an immediate priority:
The QR worker is building standard geqrf first. BQRRP is relevant only if:
- Downstream applications need rank-revealing QR (least squares with rank-deficient matrices)
- The QR project scope expands to include QRCP

### If implementing later:
- Use our BF16 GEMM for the trailing update (same as standard QR)
- The sketch computation S*A is a GEMM (small m dimension, can use cuBLAS)
- Column permutation is the unique challenge -- needs a custom kernel for non-coalesced column swaps
- The RandLAPACK library (open source) has both CPU and GPU implementations

## Caveats

1. **Only 65% of unpivoted QR throughput**: For problems that don't need pivoting, standard geqrf is better. BQRRP is for when you need rank-revealing properties.

2. **Tested on H100 only**: H100 has TMA and different memory hierarchy than sm_120. Column permutation costs will differ on RTX 5090.

3. **Randomized**: The pivot quality depends on the sampling factor gamma. Higher gamma = better pivots but slower. The paper recommends gamma = 1.0-1.5 for most applications.

4. **Not useful for our primary QR target**: Our primary goal is beating cuSOLVER geqrf for square matrices. BQRRP addresses a different problem (pivoted QR). Include in docs for completeness but don't prioritize.
