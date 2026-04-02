# PAQR: Pivoting-Avoiding QR Factorization on GPU

**Source:** "PAQR: Pivoting Avoiding QR factorization." IEEE IPDPS 2023. (https://ieeexplore.ieee.org/document/10177407/)
**Also:** https://www.netlib.org/utk/people/JackDongarra/PAPERS/PAQR.pdf
**Relevant to:** QR worker (future consideration)
**Worker's current problem:** Building standard geqrf first. PAQR is relevant if rank-deficient matrices are encountered.

## What This Is

PAQR detects and removes linearly dependent columns on-the-fly during QR factorization, without the expensive column pivoting of standard QRCP (GEQP3). It achieves the accuracy of column-pivoted QR with the speed of unpivoted QR.

## Why It Matters for Us

Standard geqrf wastes computation on rank-deficient columns. Standard GEQP3 (column-pivoted) is extremely slow on GPU. PAQR is the middle ground: fast like geqrf, accurate like GEQP3 for rank-deficient problems. If the QR worker needs to handle rank-deficient matrices, PAQR is the approach.

## Key Technique

- During panel factorization, PAQR monitors column norms
- If a column's norm drops below a threshold, it's flagged as linearly dependent
- The flagged column is skipped (not processed), reducing computation
- For full-rank matrices: PAQR has the same cost as standard QR
- For rank-deficient matrices: PAQR is faster than QR (skips dependent columns) and much faster than GEQP3

### GPU Performance:
- paqr_gpu kernel is **superior to reference kernels in every test case**
- Tested on AMD Instinct MI100 in double precision
- Compared against cuBLAS and hipBLAS batch QR implementations

## Application to sm_120

### Not immediate priority:
- Our primary target is standard geqrf for square matrices
- PAQR is relevant only for rank-deficient problems (least squares, regularization)
- When/if needed, the algorithm is a straightforward modification of blocked QR

### Implementation approach:
- Add norm monitoring to the panel factorization kernel
- Skip columns below threshold
- Everything else (trailing GEMM, LARFT) unchanged

## Caveats

1. **Tested on MI100, not NVIDIA**: GPU kernel details may differ for CUDA implementation
2. **Full-rank overhead**: For full-rank matrices, PAQR has slight overhead from norm monitoring (~1-2%)
3. **Not a replacement for geqrf**: Our primary target remains standard QR. PAQR is an extension for a specific use case.
