# cuSolverDx: Device-Side QR Factorization for sm_120

**Source:** https://docs.nvidia.com/cuda/cusolverdx/ | https://docs.nvidia.com/cuda/cusolverdx/release_notes.html
**Relevant to:** qr worker (when it starts)
**Worker's current problem:** QR not yet started. Will face same monolithic kernel challenge as LU and Cholesky.

## What This Is

cuSolverDx v0.3.0 provides device-side QR factorization (`geqrf`) and related
operations (`unmqr`, `ungqr`, `gels`) that can be called from within a CUDA kernel
on sm_120. Also provides LQ factorization (`gelqf`, `unmlq`, `unglq`).

## Why It Matters

The Cholesky project proved that multi-kernel approaches cannot beat cuSOLVER's
monolithic kernels due to launch overhead. cuSolverDx provides the building blocks
to compose a monolithic QR factorization kernel without writing the panel factorization
from scratch.

## Available Device-Side Functions

- `geqrf` — QR factorization (compute Q and R)
- `unmqr` — Apply Q (from geqrf) to a matrix
- `ungqr` — Generate explicit Q matrix
- `gelqf` — LQ factorization
- `gels` — Least squares solve via QR
- `trsm` — Triangular solve (for back-substitution)

## Composition Strategy for Blocked QR

```
Single kernel:
  for each panel column block:
    1. cuSolverDx::geqrf on panel        (device-side, no launch)
    2. cuSolverDx::unmqr to update trailing matrix  (device-side)
    3. __syncthreads()
```

## Architecture Support

sm_120 confirmed supported in v0.3.0 release notes. Performance improvements
noted for Hopper in release notes; sm_120 performance characteristics unknown.

## Recommendation

When the QR worker starts, this should be the FIRST approach investigated —
before attempting to write Householder reflections from scratch. The existing
QR research briefs on recursive tensor core QR should be used if cuSolverDx
panel sizes are too limiting.
