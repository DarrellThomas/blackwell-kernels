# cuBLASDx: Device-Side TF32 GEMM for Monolithic Cholesky on sm_120

**Source:** https://docs.nvidia.com/cuda/cublasdx/requirements_func.html | https://docs.nvidia.com/cuda/cublasdx/examples.html | https://docs.nvidia.com/cuda/cusolverdx/examples/advanced_example.html
**Relevant to:** cholesky worker (numerical/)
**Worker's current problem:** 0.55x cuSOLVER because cuSOLVER uses a monolithic kernel with inlined CUTLASS-style TF32 GEMM device functions. Worker can't replicate this with cuBLAS (host-side only). Needs device-callable GEMM.

## What This Is

cuBLASDx (CUDA BLAS Device Extensions) is NVIDIA's official device-side BLAS API. It provides GEMM operations **callable from within your own CUDA kernel** — not as a separate kernel launch, but as a device function. It supports **TF32 on sm_120** and is designed to be composed with cuSOLVERDx for exactly the fused factorization pattern the worker needs.

## Why It Matters for Us

This is the **exact tool** needed to close the gap with cuSOLVER. The worker's analysis identified that cuSOLVER achieves ~15 TFLOPS on 1 SM by having inlined tensor-core GEMM device functions. cuBLASDx provides this exact capability:

1. **Device-side TF32 GEMM** — no kernel launch overhead between POTRF, TRSM, SYRK steps
2. **sm_120 is in the supported architecture list** (SM: 70, 72, 75, 80, 86, 87, 89, **120**, 121, 90, 100, 101, 103, 110)
3. **TF32 precision supported** (`cublasdx::tfloat32_t`) — matches cuSOLVER's approach
4. **Also supports BF16, FP8 e4m3, FP16, FP32, FP64**
5. **Designed to compose with cuSOLVERDx** — the blocked_potrf example shows exactly this

The worker estimated "2-3 days of focused kernel engineering" to write a CUTLASS-like GEMM from scratch. cuBLASDx eliminates this — the device GEMM is already optimized for tensor cores.

## Key Technique

### Architecture: cuSOLVERDx POTRF + cuBLASDx GEMM in a single kernel

```cpp
// Define the device-side GEMM type at compile time
using GEMM = decltype(
    cublasdx::Size<M, N, K>()  // tile dimensions
    + cublasdx::Precision<cublasdx::tfloat32_t>()  // TF32 precision
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::SM<120>()  // target sm_120
    + cublasdx::Block()  // thread-block level
);

__global__ void monolithic_cholesky(float* A, int N, int nb) {
    __shared__ float smem[...];  // sized for cuBLASDx + panel

    for (int k = 0; k < N; k += nb) {
        // 1. Panel factorization (cuSOLVERDx POTRF or custom potf2)
        // Uses FP32 scalar on diagonal tile
        device_potf2(A + k*lda + k, nb, smem);

        // 2. TRSM (cuSOLVERDx TRSM or recursive → GEMM)
        // Solves L * X = B for column tiles below diagonal

        // 3. Trailing update: SYRK + GEMM (cuBLASDx device GEMM)
        // A[m][m] -= A[m][k] * A[m][k]^T  — uses TF32 tensor cores
        GEMM().execute(alpha, smem_a, smem_b, beta, smem_c);
    }
}
```

### cuSOLVERDx provides device-side:
- **POTRF** (Cholesky factorization of small tiles)
- **TRSM** (triangular solve)
- **GETRF** (LU factorization)
- All callable within a kernel, no launch overhead

### cuBLASDx provides device-side:
- **GEMM** with TF32/BF16/FP8/FP32 precision
- Tensor core utilization via mma.sync
- Shared memory tiling built-in
- Available as `execute()` method callable within a kernel

### The cuSOLVERDx blocked_potrf example does exactly this:
The advanced example (`blocked_potrf.cu`) shows a **left-looking blocked Cholesky** in a single kernel using cuSOLVERDx POTRF + cuBLASDx GEMM. This is the reference implementation to study.

### How to get it:
cuBLASDx and cuSOLVERDx are part of the **MathDx** package. Download from: https://developer.nvidia.com/mathdx
- Requires CUDA Toolkit 12.0+
- Header-only library (compile-time optimization)

## Caveats

- **Single thread block per matrix.** The cuSOLVERDx blocked_potrf example uses "a single thread block to process each batch." For large matrices (N > 256), this limits you to one SM. cuSOLVER's monolithic kernel does the same — it runs on 1 SM with 256 threads. This is acceptable because for Cholesky the panel is sequential anyway.
- **Tile size must fit in shared memory.** cuBLASDx requires matrices A and B to fit in shared memory. On sm_120 with 99KB/block, this limits the effective tile size. The nb=64 tile with TF32 (4 bytes) needs 64×64×4 = 16KB per matrix — feasible for double-buffered configurations.
- **cuBLASDx GEMM is limited to GEMM.** No native SYRK. But SYRK(A) = GEMM(A, A^T) with only the lower triangle stored — use a GEMM call and ignore the upper triangle output.
- **The blocked_potrf example is designed for batched workloads** (many small matrices). For a single large matrix, the algorithm is the same but you might want multiple thread blocks working on different trailing update tiles. This hybrid approach (one block for panel + many blocks for trailing GEMM) may need manual coordination.
- **Compile-time tile sizes.** cuBLASDx defines GEMM sizes at compile time via template parameters. You may need multiple specializations for different tile sizes, or a fixed nb that works well across matrix sizes.
- **Not yet verified in our build system.** The MathDx headers need to be integrated with our CUDA 13 build. Verify compilation on sm_120 before committing to this approach.
