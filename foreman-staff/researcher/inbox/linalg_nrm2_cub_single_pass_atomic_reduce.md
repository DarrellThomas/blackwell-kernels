# NRM2: CUB Single-Pass Atomic Reduction — A Ready-Made Architecture for Fused Norm

**Sources:**
- [CCCL 3.1 Release Notes](https://github.com/NVIDIA/cccl/releases/tag/v3.1.0)
- [NVIDIA Blog: Controlling Floating-Point Determinism in CCCL](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/)
- [CUB DeviceReduce API](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html)
- [CUB TransformInputIterator](https://nvidia.github.io/cccl/cub/api/classcub_1_1TransformInputIterator.html)
- [Streamlining CUB with a Single-Call API (NVIDIA Blog)](https://developer.nvidia.com/blog/streamlining-cub-with-a-single-call-api)

**Relevant to:** linalg worker
**Worker's current problem:** NRM2 at 0.79x cuBLAS (8 us, N=4096 BF16). Next direction: single-pass fused reduce+sqrt kernel.

---

## What This Is

CCCL 3.1 (shipped with CUDA 13.1, December 2025) introduced a new single-pass atomic reduction in CUB that eliminates the two-kernel launch overhead of the traditional DeviceReduce. Combined with CUB's `TransformInputIterator`, this provides a composable architecture for building a fused norm kernel: transform (square) + reduce (sum) in a single kernel launch, with the sqrt applied to the final scalar.

---

## Why It Matters for Us

The worker's NRM2 is at 0.79x cuBLAS. The primary hypothesis for the gap is two-kernel overhead in the worker's current approach (or cuBLAS using safe-scaling that we could skip). CUB's new single-pass atomic reduce confirms that NVIDIA themselves consider single-kernel reductions worthwhile for small arrays. The key insight: **CUB's single-pass mode is "typically faster than the run-to-run deterministic version — particularly for smaller input arrays, where performing the reduction in a single kernel reduces latency from multiple kernel launches."** At N=4096 (very small), launch latency is the dominant cost.

---

## Key Technique: CUB Composable Fused Norm

### Architecture 1: CUB TransformInputIterator + DeviceReduce::Sum (Single-Pass)

```cpp
// Square functor
struct SquareBF16 {
    __host__ __device__ float operator()(const __nv_bfloat16& x) const {
        float v = __bfloat162float(x);
        return v * v;
    }
};

// Wrap input with transform iterator
cub::TransformInputIterator<float, SquareBF16, const __nv_bfloat16*>
    squared_iter(d_input, SquareBF16());

// Single-pass atomic reduce with CCCL 3.1 determinism API
auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);
cub::DeviceReduce::Sum(d_temp, temp_bytes, squared_iter, d_sum, N, stream, env);

// Then sqrtf on the scalar result (host-side or tiny kernel)
```

This fuses the square+sum into one kernel launch. The `not_guaranteed` determinism level activates the single-pass atomic path. The `TransformInputIterator` applies the square operation during the load phase with zero extra memory traffic.

### Architecture 2: Custom Kernel (What the Worker Should Build)

The CUB approach above is a baseline to benchmark against. But for maximum performance at N=4096, a custom kernel will win because:

1. **N=4096 is 32 warps of work** (4096 elements / 8 elements per uint4 load / 16 threads = ~32 loads total). This fits in a SINGLE thread block.
2. **Single-block kernel eliminates atomicAdd entirely** — no inter-block reduction needed.
3. **Fuse the sqrt** — last thread in the warp reduction applies sqrtf before writing the result.

```cuda
// Single-block fused NRM2 for small N (up to ~32K elements)
__global__ __launch_bounds__(256, 1)
void nrm2_fused_small(const __nv_bfloat16* __restrict__ x, float* result, int N) {
    const int tid = threadIdx.x;
    const int n_vecs = N / 8;  // 8 BF16 per uint4 load

    float acc = 0.0f;
    // Grid-stride within single block (256 threads, each handles multiple uint4 loads)
    for (int v = tid; v < n_vecs; v += blockDim.x) {
        uint4 xv = __ldg(reinterpret_cast<const uint4*>(&x[v * 8]));
        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float val = __bfloat162float(xp[i]);
            acc = fmaf(val, val, acc);
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    // Cross-warp reduction in shared memory
    __shared__ float warp_sums[8];  // 256 threads / 32 = 8 warps
    int warp = tid / 32, lane = tid % 32;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();

    // Final reduction + sqrt (thread 0 only)
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < 8; w++) total += warp_sums[w];
        *result = sqrtf(total);
    }
}
```

This kernel has:
- **1 kernel launch** (vs 2 for cuBLAS safe-scaling NRM2)
- **Zero atomicAdd** (single block)
- **Zero __threadfence** (no inter-block communication)
- **Fused sqrt** (no second kernel)
- **No cudaMemsetAsync** (result written directly)

For N=4096: 4096/8 = 512 vector loads / 256 threads = 2 loads per thread. The kernel is pure register work after the loads complete.

---

## How This Beats cuBLAS

cuBLAS NRM2 has three structural overheads the custom kernel avoids:

1. **Two kernel launches** — cuBLAS uses a grid reduction + separate finalization kernel. At N=4096, each launch costs ~2-5 us, so launch overhead alone is 4-10 us (the ENTIRE kernel runtime).

2. **Safe scaling (Anderson algorithm)** — cuBLAS adds 4-6 extra instructions per element (abs, two comparisons, conditional multiply, three-accumulator combine) to handle overflow/underflow. For BF16 ML data in [-10, 10], this is unnecessary.

3. **cudaMemset for result buffer** — cuBLAS zeroes the result buffer before the reduction, adding another mini-kernel launch.

The custom single-block kernel skips all three. DOT already beats cuBLAS by 1.48x using essentially the same architecture (vectorized loads, warp reduction, atomicAdd). NRM2 at 0.79x is anomalous — the gap is almost certainly cuBLAS's safe scaling overhead making cuBLAS NRM2 slower than cuBLAS DOT, not our kernel being slow.

---

## Diagnostic Step: Profile cuBLAS NRM2

Before building the fused kernel, run this diagnostic to confirm the hypothesis:

```bash
ncu --target-processes all --kernel-name-base demangled \
    --metrics sm__warps_active.avg.per_cycle_active,dram__bytes.sum \
    python3 -c "
import torch
x = torch.randn(4096, dtype=torch.bfloat16, device='cuda:1')
for _ in range(100): torch.linalg.vector_norm(x)
"
```

Count the kernel launches. If cuBLAS uses 2 kernels for NRM2, the fused kernel will win by eliminating one launch. If cuBLAS uses 1 kernel, the win comes from skipping safe scaling.

---

## Caveats

1. **N=4096 is tiny.** The entire input is 8 KB (4096 * 2 bytes). It fits in L1 cache. Kernel launch overhead dominates everything. A CUDA graph wrapping the custom kernel could further reduce launch overhead from ~3 us to ~1 us.

2. **The single-block approach only works for small N.** For N > ~32K, you need multiple blocks and atomicAdd (the architecture the worker already has for DOT). But the benchmark is N=4096 (64^2), so single-block is ideal.

3. **CUB single-pass atomic reduce is worth benchmarking as a comparison point.** If CUB's `TransformInputIterator + DeviceReduce::Sum` with `not_guaranteed` determinism already beats cuBLAS NRM2 at N=4096, the worker could use it directly and avoid writing a custom kernel.

4. **The fmaf trick matters.** `fmaf(val, val, acc)` is a single FMA instruction vs separate FMUL + FADD. The compiler should do this automatically with `-O2`, but making it explicit guarantees it.
