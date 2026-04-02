# NRM2: Fused Single-Pass Vector Norm Kernel

**Source:** https://enccs.github.io/cuda/3.01_ParallelReduction/ | https://ar5iv.labs.arxiv.org/html/1305.1183
**Relevant to:** linalg worker (linalg/)
**Worker's current problem:** NRM2 at 0.79x reference. Next direction: "NRM2 fused kernel — single-pass reduce+sqrt to avoid two-kernel overhead."

## What This Is

A fused CUDA kernel that computes `||x||_2 = sqrt(sum(x_i^2))` in a single pass: vectorized loads, FMA accumulation of squares, parallel reduction, and final sqrt — all without materializing intermediate results to global memory or launching multiple kernels.

## Why It Matters for Us

The worker's NRM2 is 0.79x reference. Two likely issues:
1. **Two-kernel approach:** First kernel reduces sum-of-squares, second computes sqrt. The second launch adds 2-5 us overhead.
2. **Missing vectorization or occupancy tuning** — the dotproduct worker showed that float4 loads + grid-stride + FMA + atomicAdd can hit 89.4% of memory bandwidth.

The dotproduct kernel (v6) already achieves 1.48x cuBLAS. NRM2 is essentially the same pattern with `x*x` instead of `x*y`, plus a final `sqrtf`. The worker should be able to directly adapt the dotproduct kernel.

## Key Technique

### Single-pass NRM2 kernel:
```cuda
__global__ void nrm2_kernel(const float* x, float* result, int n) {
    float sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Vectorized grid-stride loop with FMA
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n4 = n / 4;
    for (int i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        sum = fmaf(v.x, v.x, sum);
        sum = fmaf(v.y, v.y, sum);
        sum = fmaf(v.z, v.z, sum);
        sum = fmaf(v.w, v.w, sum);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    // Block reduction via shared memory
    __shared__ float shared[32];
    if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = shared[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // atomicAdd to global result (single kernel, no second launch)
    if (threadIdx.x == 0) atomicAdd(result, sum);
}

// Host-side: single kernel + sqrtf on the scalar result
nrm2_kernel<<<nb, bs>>>(x, d_result, n);
// sqrtf applied on host after reading back, or:
// fuse into an epilogue kernel that's negligible cost
```

### Key optimizations (from dotproduct v6):
1. **float4 vectorized loads** — 4 elements per memory transaction
2. **FMA** (`fmaf(x, x, sum)`) — fused multiply-add, single instruction for x²+sum
3. **Grid-stride loop** — full occupancy across all SMs
4. **atomicAdd** — avoids second kernel launch (contention negligible with hundreds of blocks)
5. **Streaming loads** (`ld.global.cs`) for N > L2 size — bypasses L2, prevents thrashing
6. **Auto-tuned block/grid sizes** — bs=256 nb=680 for medium N

### Where does the sqrt go?
- **Option A:** Host-side `sqrtf` on the single float result (negligible latency)
- **Option B:** Epilogue block — have block 0 wait for all atomics, then compute sqrt
- **Option C:** Use `rsqrtf` if the reciprocal norm is what's really needed (common in normalization)

## Caveats

- **This is essentially the dotproduct kernel with x*x instead of x*y.** The worker should start from the dotproduct v6 code and modify the accumulation.
- **Numerical stability:** For very large vectors, sum-of-squares can overflow float. The standard fix is "Kahan-like" accumulation or using a maximum-scaling approach: find max(|x|), scale all squares by 1/max². cuBLAS does this. For typical ML tensor sizes this isn't needed.
- **BF16 input variant:** If inputs are BF16, load as `__nv_bfloat162` pairs, convert to FP32 for accumulation. The conversion is cheap (widening cast, single instruction).
- **The 0.79x gap may also come from PyTorch overhead.** The dotproduct worker proved 0 Python overhead with a C++ benchmark loop. The linalg worker should verify the same for NRM2 before optimizing the kernel itself.
