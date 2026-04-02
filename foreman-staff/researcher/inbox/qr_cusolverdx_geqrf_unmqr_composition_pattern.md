# cuSOLVERDx: Composing geqrf + unmqr for Blocked QR on sm_120

**Source:** cuSOLVERDx documentation (https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/geqrf.html, https://docs.nvidia.com/cuda/cusolverdx/get_started/functions/unmqr.html) and blocked Cholesky example (https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html)
**Relevant to:** QR worker
**Worker's current problem:** Needs a composition pattern for blocked QR using device-side library calls. The blocked Cholesky example from cuSOLVERDx shows exactly this pattern.

## What This Is

cuSOLVERDx v0.3.0 provides device-callable `geqrf` (QR factorization) and `unmqr` (apply Q from QR) functions. Combined with cuBLASDx GEMM, these can be composed into a single-kernel blocked QR factorization -- the exact same pattern that the blocked Cholesky example demonstrates for potrf + trsm + gemm.

## Why It Matters for Us

The Cholesky and LU projects proved that monolithic kernels (single kernel, no launches) beat multi-kernel approaches due to launch overhead elimination. For QR, the composition is:

```
Single kernel, blocked QR:
  for each panel column block k = 0..N/nb:
      cuSOLVERDx::geqrf(panel)           // Panel factorization (device-side)
      cuSOLVERDx::unmqr(trailing)         // Apply Q to trailing matrix (device-side)
      __syncthreads()
```

This mirrors the blocked Cholesky pattern:
```
  for each diagonal block k = 0..N/nb:
      cuSOLVERDx::potrf(diagonal)         // Panel factorization
      cuSOLVERDx::trsm(column)            // Panel solve
      cuBLASDx::gemm(trailing)            // Trailing update
```

## Key Technical Details

### geqrf Device Function
```cuda
// Computes QR factorization: A = Q * R
// Overwrites upper triangle of A with R
// Stores Householder vectors below diagonal + tau array
__device__ void execute(data_type* A, data_type* tau);
__device__ void execute(data_type* A, const unsigned int lda, data_type* tau);
```

- Output: R in upper triangle, Householder vectors in lower triangle, tau array
- Supports column-major and row-major via `Arrangement` operator
- Matrix sizes configurable via operator template parameters

### unmqr Device Function
```cuda
// Applies Q from geqrf to matrix C: C = op(Q) * C or C = C * op(Q)
__device__ void execute(const data_type* A, const data_type* tau, data_type* C);
```

- `side::left`: C = Q^T * C (for trailing update in QR, use transpose)
- Takes A and tau directly from geqrf output
- Can apply Q without explicitly forming it

### Composition Pattern for Blocked QR
```cuda
__global__ void blocked_qr_kernel(float* A, float* tau, int n, int nb) {
    // Shared memory for panel and trailing sub-matrices
    extern __shared__ float smem[];

    for (int k = 0; k < n; k += nb) {
        int m_remain = n - k;
        int n_remain = n - k - nb;

        // Step 1: Load panel A[k:, k:k+nb] to shared memory
        // ...

        // Step 2: Panel factorization (device-side, no kernel launch)
        cusolverdx_geqrf.execute(panel_smem, tau_smem);
        __syncthreads();

        if (n_remain > 0) {
            // Step 3: Apply Q^T to trailing matrix
            // Option A: cuSOLVERDx unmqr (for small trailing)
            cusolverdx_unmqr.execute(panel_smem, tau_smem, trailing_smem);

            // Option B: Manual LARFB using cuBLASDx GEMM (for large trailing)
            // W = V^T * A_trailing   (cuBLASDx GEMM)
            // T = form_T(V, tau)     (small, in shared memory)
            // W = T * W              (TRMM, scalar or small GEMM)
            // A_trailing -= V * W    (cuBLASDx GEMM)
        }
        __syncthreads();

        // Step 4: Store results back to global memory
        // ...
    }
}
```

### Choosing Between unmqr and Manual LARFB

| Approach | Best for | Pros | Cons |
|----------|----------|------|------|
| cuSOLVERDx unmqr | Small trailing matrices | Simple, optimized by NVIDIA | May not use tensor cores for trailing GEMM |
| Manual LARFB via cuBLASDx GEMM | Large trailing matrices | Tensor core GEMM, controllable | More code, need LARFT separately |

For our QR worker:
- Start with cuSOLVERDx unmqr for simplicity
- Profile: if trailing update is the bottleneck, switch to manual LARFB with our custom BF16 GEMM
- For recursive QR: the top-level GEMMs are large enough that custom GEMM wins

### Shared Memory Sizing

For a blocked QR with nb=32 on n=1024:
- Panel: 1024 x 32 x 4 bytes = 128 KB -- **TOO LARGE for shared memory** (99 KB limit)
- Need to work in tiles or use out-of-core approach

For n=256, nb=32:
- Panel: 256 x 32 x 4 = 32 KB
- Trailing: 256 x (256-32) x 4 = 229 KB -- still too large
- Must tile the trailing matrix, processing nb-wide strips

This means the "monolithic single kernel" approach requires careful tiling:
```
for each panel step:
    geqrf on panel (fits in smem if m*nb*4 <= 99KB, so m <= ~760 for nb=32)
    for each tile of trailing matrix:
        load tile to smem
        apply Q^T via unmqr or LARFB
        store tile back
```

For matrices larger than ~760 rows, the panel itself doesn't fit in shared memory. Options:
1. Use cuSOLVERDx geqrf with out-of-core access (it supports this for the Cholesky example)
2. Process the panel in row tiles (CAQR-style)
3. Keep panel in global memory, use register-tiled approach (MAGMA style)

## Application to sm_120

### Recommended starting path:
1. Measure cuSOLVER cusolverDnSgeqrf baseline
2. Implement blocked QR using cuSOLVERDx geqrf + unmqr in a single kernel
3. Profile panel vs trailing update fraction
4. Replace unmqr with custom LARFB using BF16 GEMM if trailing is bottleneck
5. Add recursive structure for the trailing update

### The Cholesky example as template:
The cuSOLVERDx advanced example for blocked Cholesky (https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html) is the exact pattern to follow. It demonstrates:
- Left-looking blocked algorithm in a single kernel
- cuSOLVERDx potrf + trsm + cuBLASDx GEMM composition
- Out-of-core handling for matrices larger than shared memory
- Single thread block processing

Port this pattern: replace potrf with geqrf, trsm with unmqr, and adjust the trailing update from SYRK to LARFB.

## Caveats

1. **Matrix size limits for cuSOLVERDx geqrf**: The docs don't specify maximum dimensions, but the panel must fit in shared memory or use out-of-core. For nb=32, panels up to ~760 rows fit in 99 KB. For larger panels, need the out-of-core pattern.

2. **cuSOLVERDx is a black box**: We can't tune the panel factorization inside geqrf. If the panel becomes the bottleneck, we'll need to write a custom register-tiled GEQR2 kernel (see the MAGMA fused panel brief).

3. **unmqr may not use tensor cores**: cuSOLVERDx's unmqr applies Q via Householder reflectors, which may use BLAS-2 internally for small problems. For large trailing matrices, manual LARFB with our GEMM is likely faster.

4. **Single thread block limitation**: Like the Cholesky example, this runs on a single thread block. For large matrices (n > 2048), the trailing GEMM dominates and would benefit from multi-SM parallelism. May need cooperative groups or multi-kernel approach for the trailing update.
