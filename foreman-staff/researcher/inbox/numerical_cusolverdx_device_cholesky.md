# cuSolverDx: Device-Side Cholesky for sm_120

**Source:** https://docs.nvidia.com/cuda/cusolverdx/get_started/posv.html | https://docs.nvidia.com/cuda/cusolverdx/release_notes.html
**Relevant to:** numerical worker (cholesky)
**Worker's current problem:** Cholesky at 0.55x cuSOLVER. The gap is entirely from kernel launch overhead in the multi-kernel blocked approach. cuSOLVER uses a single monolithic kernel. TF32 MMA B fragment broadcasting bug blocks custom device-side GEMM.

## What This Is

cuSolverDx v0.3.0 provides device-side Cholesky factorization (`potrf`, `potrs`,
`posv`) that can be called FROM WITHIN a CUDA kernel on sm_120. This could bypass
both blockers: kernel launch overhead AND the TF32 MMA bug.

## Why It Matters for Us

The Cholesky worker identified two insurmountable barriers:
1. **Launch overhead**: 190 CUDA Graph nodes add ~0.38ms unavoidable overhead
2. **TF32 MMA defect**: B fragment broadcasting makes custom device-side GEMM impossible
   with TF32 precision, and BF16 MMA has lower precision

cuSolverDx potentially solves BOTH:
- Device-side `potrf` eliminates all kernel launches (runs in YOUR kernel)
- cuSolverDx handles its own internal MMA/GEMM — it presumably uses whatever
  proprietary approach cuSOLVER itself uses (avoiding the TF32 B fragment bug)

## Composition Strategy

```
Single kernel (1 block, 256 threads):
  for each panel column block (NB=64):
    1. cuSolverDx::potrf on diagonal panel  (device-side)
    2. cuSolverDx::trsm on off-diagonal     (device-side)
    3. SYRK trailing update                  (device-side, possibly cuSolverDx or custom)
    4. __syncthreads()
```

This mirrors exactly what cuSOLVER's monolithic kernel does, but using the
public API building blocks.

## Key Details

- **sm_120 support:** Confirmed in v0.3.0 release notes
- **Known bug:** CUDA 12.8-13.0 miscompile for gesv_no_pivot on SM120 with
  real types. potrf/posv may or may not be affected — test empirically.
- **Available functions:** potrf (factorize), potrs (solve), posv (factorize+solve)
- **TRSM also available:** device-side triangular solve for the off-diagonal update

## Caveats

Same as LU brief: matrix size limits, thread block constraints, shared memory
usage, and performance characteristics are all unknown. Test with a minimal
64x64 panel first.

## Recommendation

The Cholesky worker should try cuSolverDx BEFORE attempting the monolithic
BF16 MMA device GEMM (which is listed as "major engineering effort" in next
directions). cuSolverDx could achieve the same goal with much less effort.
