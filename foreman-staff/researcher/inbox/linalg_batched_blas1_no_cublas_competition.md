# Batched BLAS1: cuBLAS Has NO Batched DOT — Zero Competition

**Source:** cuBLAS 13.2 documentation, NVIDIA forums, Ginkgo ISC 2024
**Relevant to:** linalg worker
**Worker's current problem:** Next direction #3 is batched BLAS1 (batched DOT/NRM2 for iterative solver convergence). This brief confirms the opportunity.

## What This Is

Confirmed in cuBLAS 13.2 documentation: there is NO `cublasSdotBatched` or
`cublasSdotStridedBatched` API. cuBLAS provides batched GEMM, batched TRSM,
batched GETRF — but NOT batched Level 1 operations. This means a custom
batched DOT kernel has **zero library competition**.

## Why It Matters for Us

Our individual DOT is already at 1.48x cuBLAS. A batched DOT kernel would:
1. Amortize kernel launch overhead across many small reductions
2. Enable device-side convergence checks without host round-trips
3. Be the ONLY batched DOT available on sm_120

The use case: iterative solvers (CG, GMRES, BiCGSTAB) need multiple DOT/NRM2
per iteration for convergence testing. Currently each requires a separate
kernel launch (~2-3 us overhead each) + host copy for the scalar result.

## Key Implementation Pattern: One Block Per Vector

From NVIDIA forums and Ginkgo ISC 2024 paper:

```cuda
__global__ void batched_dot(
    const float* __restrict__ x,  // [batch, N] contiguous
    const float* __restrict__ y,  // [batch, N] contiguous
    float* results,               // [batch]
    int N, int batch_count
) {
    int batch_id = blockIdx.y;    // Y-dimension = batch index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const float* x_vec = x + batch_id * N;
    const float* y_vec = y + batch_id * N;

    // Vectorized grid-stride dot product for this batch
    float sum = 0.0f;
    const float4* x4 = (const float4*)(x_vec);
    const float4* y4 = (const float4*)(y_vec);
    for (int i = tid; i < N/4; i += blockDim.x * gridDim.x) {
        float4 xv = x4[i], yv = y4[i];
        sum += xv.x*yv.x + xv.y*yv.y + xv.z*yv.z + xv.w*yv.w;
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Block reduction via smem + atomicAdd to results[batch_id]
    // ...
}
```

Grid: `(blocks_per_vector, batch_count, 1)` — single launch for all batches.

## New Finding: ReSolve DCGS2 for GMRES

ReSolve library (ORNL, 2024) implements Delayed CGS2 (DCGS2) which reduces
GMRES orthogonalization from 3 global reductions per iteration to 1 — the
theoretical minimum. If/when the linalg worker builds GMRES, this algorithm
cuts synchronization overhead by 3x.

Source: https://arxiv.org/abs/2401.13926 | https://github.com/ORNL/ReSolve

## New Finding: CPU-Free Persistent CG

ParCoreLab (ICS 2023) demonstrates fully GPU-resident CG using persistent
kernels + cooperative groups grid_sync. Zero host-device synchronization
during the solve. The convergence check happens on-device.

Source: https://dl.acm.org/doi/10.1145/3577193.3593713
GitHub: https://github.com/ParCoreLab/CPU-Free-model

## Caveats

1. **Strided layout required:** Vectors must be packed contiguously with
   known stride. If the solver allocates vectors independently, they won't
   be strided. Pre-allocate a single buffer and slice.

2. **For BF16 inputs:** Use int4 loads (8 BF16 values per load) and convert
   to FP32 for accumulation. Same pattern as the existing DOT kernel.

3. **The primary value is NOT raw throughput** — individual DOTs are already
   fast. The value is **reduced launch overhead** (1 launch vs N launches)
   and **eliminated host sync** (device-side convergence flag).
