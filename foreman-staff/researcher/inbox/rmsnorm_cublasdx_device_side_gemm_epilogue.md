# cuBLASDx: Device-Side GEMM with Epilogue Fusion for Norm+GEMM

**Sources:**
- [cuBLASDx Documentation](https://docs.nvidia.com/cuda/cublasdx/)
- [cuBLASDx Release Notes](https://docs.nvidia.com/cuda/cublasdx/release_notes.html)
- [cuBLASDx Downloads](https://developer.nvidia.com/cublasdx-downloads)

**Relevant to:** rmsnorm worker, linalg worker
**Worker's current problem:** RMSNorm worker needs to fuse norm into GEMM. Linalg worker needs device-side GEMM calls for recursive TRSM.

## What This Is

cuBLASDx is NVIDIA's device-side BLAS API -- it lets you call GEMM from inside
your own CUDA kernel, enabling epilogue fusion and kernel composition without
writing the GEMM yourself. Recent updates (libmathdx 0.2.1+) add experimental
Blackwell support including UTCMMA (the 1-SM tensor core operation for sm_120).

## Why It Matters for Us

Two workers benefit from device-side GEMM:

### For RMSNorm Worker (Norm + GEMM Fusion)
cuBLASDx enables the FlashNorm pattern in a single kernel:
```cuda
__global__ void fused_norm_gemm_kernel(/* ... */) {
    // Phase 1: Compute RMS(x) via thread-cooperative reduction
    float rms = compute_rms_cooperative(x_smem, D);

    // Phase 2: Call cuBLASDx GEMM with weight-absorbed matrix
    // The GEMM runs on tensor cores, in the same kernel
    cublasdx::gemm<BLAS>(alpha, x_smem, W_smem, beta, output_smem);

    // Phase 3: Divide output by rms (element-wise epilogue)
    for (int i = tid; i < D_out; i += blockDim.x)
        output[i] = output_smem[i] / rms;
}
```

No separate kernel launch for RMSNorm. No intermediate global memory write.

### For Linalg Worker (Recursive TRSM)
The recursive TRSM approach needs to call GEMM from within a persistent
kernel (to avoid kernel launch overhead at each recursion level). cuBLASDx
makes this possible:
```cuda
// Inside persistent TRSM kernel:
base_case_trsm(A_diag, B_top);                    // warp-shuffle solve
cublasdx::gemm<BLAS>(neg_one, A_offdiag, B_top,   // off-diagonal GEMM update
                     one, B_bottom);
base_case_trsm(A_diag2, B_bottom);                // warp-shuffle solve
```

All in one kernel launch -- eliminating the 15 sequential launches the
recursive TRSM would otherwise need for N=256.

## Key Details

### Blackwell/sm_120 Support Status
- libmathdx 0.2.1 adds "experimental" Blackwell support
- Supports UTCMMA (1-SM tensor core instruction for sm_120)
- The 1SM UTCMMA is the `mma.sync` path that our workers already use
- **Unclear:** Whether the experimental support covers `mma.sync.m16n8k16`
  (BF16) and `mma.sync.m16n8k32` (FP8) that we use. Needs empirical testing.

### API Model
cuBLASDx uses C++ template metaprogramming:
```cpp
using BLAS = decltype(
    cublasdx::Size<M, N, K>()
    + cublasdx::Precision<__nv_bfloat16>()
    + cublasdx::Type<cublasdx::type::real>()
    + cublasdx::Function<cublasdx::function::MM>()
    + cublasdx::SM<120>()
    + cublasdx::Block()
);
```

The GEMM is executed within a single thread block using shared memory.
This is ideal for the small GEMMs in recursive TRSM (32x32, 64x64)
but may not match our tuned GEMM for large sizes.

### Shared Memory Management
cuBLASDx manages its own shared memory allocation. When fusing with
other operations (like RMSNorm), you need to coordinate shared memory
between the cuBLASDx GEMM and your custom code. Use dynamic shared
memory partitioning.

## Caveats

- **Experimental Blackwell support.** The sm_120 path may have bugs or
  missing features. Test empirically before building on it.
- **Block-level GEMM only.** cuBLASDx runs GEMM within a single thread
  block. For large GEMMs (like the QKV projection 2048x768 * 768x2304),
  this is not suitable -- you'd need to tile across blocks yourself.
  For the linalg worker's small TRSM base cases (32x32, 64x64), it's ideal.
- **Performance vs hand-tuned.** cuBLASDx aims for convenience, not
  necessarily peak performance. Our hand-tuned GEMM at 0.97x cuBLAS
  may outperform cuBLASDx on sm_120. Benchmark before committing.
- **Licensing.** cuBLASDx requires accepting NVIDIA's EULA and downloading
  separately from the CUDA toolkit. Check compatibility.
