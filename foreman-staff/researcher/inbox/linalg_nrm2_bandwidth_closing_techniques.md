# NRM2 Bandwidth Gap: Techniques to Close 0.79x to 1.0x+ cuBLAS

**Sources:**
- [Crushing CPUs with 879 GB/s Reductions in CUDA (Ash Vardanian)](https://ashvardanian.com/posts/cuda-parallel-reductions/)
- [Faster Parallel Reductions on Kepler (NVIDIA Blog)](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [Modern Parallel Reduction for CC 12.0 (GitHub Gist)](https://gist.github.com/troelsy/fff6aac2226e080dcebf05531a11d44e)
- [NVIDIA CUDA Pro Tip: Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access)
- [Optimizing CUDA Code By Kernel Fusion: Application on BLAS (arXiv 1305.1183)](https://ar5iv.labs.arxiv.org/html/1305.1183)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [CUB DeviceReduce API](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html)
- [NVIDIA cuda-samples reductionMultiBlockCG](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu)
- [Cooperative Groups CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [NVIDIA Shared Memory Register Spilling Blog](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)
- [GTC 2025: CUDA Techniques to Maximize Memory Bandwidth and Hide Latency](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)

**Relevant to:** linalg worker
**Worker's current problem:** NRM2 at 0.79x cuBLAS (8 us, N=1M BF16). DOT already at 1.48x using the same architecture. Need to close the gap.

---

## Supplementary Brief

The previous briefs (`linalg_nrm2_optimization_techniques.md` and `linalg_nrm2_fused_single_pass.md`) covered single-pass architecture, grid tuning, fmaf, streaming loads, and cuBLAS internals. This brief covers additional techniques from new research that may help close the remaining gap.

---

## 1. CUB's Approach: Why It Hits 94% Bandwidth Saturation

CUB's `DeviceReduce` achieves 879 GB/s on RTX 3090 (94% of 936 GB/s peak). The key implementation details:

- **LOAD_LDG cache operator**: CUB uses `cub::LOAD_LDG` which routes through the read-only texture cache path. This gave a reported 2x speedup over default loads for large datasets. On sm_120, `__ldg` is the equivalent intrinsic. The worker's kernel already uses `__ldg`; confirm via ncu that loads actually go through `LDG.E.128` instructions in the SASS.

- **Items-per-thread tuning**: CUB tunes `ITEMS_PER_THREAD` and `BLOCK_THREADS` per architecture. Typical high-bandwidth configs use 8-16 items per thread with 128-256 threads per block. More items per thread = fewer blocks = less atomicAdd contention, but also less parallelism. The sweet spot for bandwidth-bound reductions on modern GPUs is typically:
  - 256 threads/block, 8-16 items per thread (before the grid-stride loop kicks in)
  - Grid size = 2-4x the number of SMs

- **Algorithm selection**: CUB picks between `BLOCK_REDUCE_RAKING` (shared memory tree) and `BLOCK_REDUCE_WARP_REDUCTIONS` (warp shuffle + shared memory for cross-warp). For sm_120, warp reductions should be preferred -- the worker is already doing this.

**Actionable**: Compare the worker's achieved DRAM bandwidth (from ncu `dram__bytes.sum` / kernel duration) against CUB's `DeviceReduce::Sum`. If CUB hits >90% and the custom kernel is at 70-80%, the bottleneck is in the memory access pattern, not the reduction tree.

---

## 2. Interleaved Multi-Reduction for ILP

The NVIDIA parallel reduction blog demonstrates interleaving multiple independent reductions to expose instruction-level parallelism (ILP):

```cuda
// Instead of:
float sum = 0.0f;
for (int i = tid; i < n; i += stride) {
    float val = x[i];
    sum = fmaf(val, val, sum);
}

// Interleave multiple accumulators:
float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
for (int i = tid; i < n; i += stride * 4) {
    float v0 = x[i];
    float v1 = x[i + stride];
    float v2 = x[i + stride * 2];
    float v3 = x[i + stride * 3];
    sum0 = fmaf(v0, v0, sum0);
    sum1 = fmaf(v1, v1, sum1);
    sum2 = fmaf(v2, v2, sum2);
    sum3 = fmaf(v3, v3, sum3);
}
float sum = sum0 + sum1 + sum2 + sum3;
```

**Why this helps**: Each `fmaf` has a latency of ~4 cycles on sm_120. With a single accumulator, the next FMA cannot issue until the previous one completes (RAW dependency on `sum`). With 4 accumulators, 4 independent FMAs can be in-flight simultaneously, hiding the FMA latency.

**Combined with vectorized loads**: The worker already loads 8 BF16 elements per uint4 load. After converting to float, accumulate into 4 separate sums (2 elements each), then combine after the loop. This adds zero memory traffic and may improve ALU utilization from the current 3.1%.

**Caveat**: For small N (1M elements, ~2 MB), the kernel is so short that ILP may not matter -- the limiting factor is likely memory latency to DRAM, not ALU throughput. Still worth trying since it's a trivial code change.

---

## 3. Warp-Atomic Single-Pass (Minimum Shared Memory)

The NVIDIA blog on Kepler reductions shows a "warp-atomic" approach that uses NO shared memory at all:

```cuda
__global__ void nrm2_warp_atomic(const __nv_bfloat16* x, float* result, int n) {
    float sum = 0.0f;
    // Grid-stride vectorized accumulation...

    // Warp reduction only (no shared memory)
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    // Every warp lane 0 does atomicAdd (no block-level reduction)
    if ((threadIdx.x & 31) == 0)
        atomicAdd(result, sum);
}
```

**Trade-off**: More atomicAdd calls (1 per warp instead of 1 per block), but eliminates shared memory allocation and `__syncthreads()`. With 256 threads/block and 256 blocks, this is 256*8 = 2048 atomicAdds total -- all serializing at the L2 controller on the same cache line.

**When this wins**: For very small kernels where the `__syncthreads` and shared memory allocation overhead is a meaningful fraction of runtime. With 8 us total kernel time, even 0.5 us saved on synchronization is 6%.

**When this loses**: For large grids where atomicAdd contention dominates. The L2 controller can handle ~1 atomicAdd per cycle; 2048 atomicAdds at ~1 GHz L2 clock = ~2 us of serialization. That's 25% of the 8 us runtime.

**Recommendation**: Try this approach with a SMALL grid (64-128 blocks). With 128 blocks * 8 warps = 1024 atomicAdds, serialization is ~1 us. Combined with zero shared memory overhead, this may be faster than the current block-reduction approach. The DOT kernel at 1.48x likely uses a similar pattern -- cross-check.

---

## 4. Cooperative Groups Single-Pass Reduction (Alternative Architecture)

NVIDIA's `reductionMultiBlockCG` sample demonstrates a true single-pass reduction using cooperative kernel launch:

```cuda
// Launch with cudaLaunchCooperativeKernel
__global__ void reduceSinglePass(float* in, float* out, int n) {
    cg::grid_group grid = cg::this_grid();

    // Phase 1: Grid-stride accumulate into shared memory
    // Phase 2: Block-level reduction
    // Phase 3: grid.sync() -- global barrier across ALL blocks
    // Phase 4: Block 0 reduces the per-block results

    grid.sync();  // Synchronize entire grid
    // Only block 0 does final reduction + sqrt
}
```

**Advantage**: No atomicAdd at all. The grid-wide sync guarantees all blocks have written their partial sums before the final reduction. The `sqrtf` happens inside the kernel with no race condition.

**Disadvantage**: Requires `cudaLaunchCooperativeKernel`, which limits the grid size to the maximum number of concurrent blocks the device can run. On RTX 5090 with 170 SMs and up to 32 blocks/SM, this allows up to 5440 blocks -- more than enough.

**Performance consideration**: The cooperative launch has slightly higher overhead than a regular launch (~1-2 us extra). For an 8 us kernel this matters. But eliminating atomicAdd contention entirely may compensate.

**Requires sm_60+**: sm_120 qualifies. Available since CUDA 9.

---

## 5. The DOT-NRM2 Gap Analysis

The worker's DOT is at 1.48x cuBLAS and NRM2 is at 0.79x. These are architecturally identical operations (grid-stride vectorized load, FMA accumulate, reduce, atomicAdd). The difference MUST come from one of:

1. **cuBLAS NRM2 is faster than cuBLAS DOT**: cuBLAS may use a more optimized kernel for NRM2 than DOT, or NRM2 may use a single fused kernel while DOT uses two. Profile both cuBLAS kernels with `nsys` to check kernel launch counts and durations.

2. **The custom NRM2 kernel is slower than the custom DOT kernel**: The NRM2 kernel may have extra instructions (BF16 conversion, FMA vs multiply+add), different grid/block sizes, or the sqrt adds overhead. Compare the two custom kernels' SASS and ncu metrics side-by-side.

3. **The sqrt adds measurable overhead**: The last-block pattern for in-kernel sqrt adds a `__threadfence()` and an extra `atomicAdd` (for the counter). If the sqrt is done on-host instead (read the single float back, call sqrtf), the kernel itself runs the same as DOT.

**Concrete experiment**: Run the NRM2 kernel WITHOUT the sqrt (just output sum-of-squares) and compare against DOT's timing. If they match, the gap is entirely in the sqrt epilogue and cuBLAS's relative NRM2 speed. If NRM2-without-sqrt is still slower than DOT, there's a kernel-level issue.

---

## 6. BF16 Conversion: Vectorized Approach

The current kernel loads 8 BF16 values via a uint4 load and converts them individually with `__bfloat162float`. CUDA provides `__bfloat1622float2` that converts a pair of BF16 values to float2 in a single operation:

```cuda
// Current (8 individual conversions):
__nv_bfloat16 vals[8];
memcpy(vals, &xv, sizeof(uint4));
float v0 = __bfloat162float(vals[0]);
// ... 7 more

// Vectorized (4 pair conversions):
__nv_bfloat162 pairs[4];
memcpy(pairs, &xv, sizeof(uint4));
float2 f0 = __bfloat1622float2(pairs[0]);
float2 f1 = __bfloat1622float2(pairs[1]);
float2 f2 = __bfloat1622float2(pairs[2]);
float2 f3 = __bfloat1622float2(pairs[3]);
// Then: sum = fmaf(f0.x, f0.x, sum); sum = fmaf(f0.y, f0.y, sum); ...
```

On sm_120, the BF16-to-float conversion is a simple bit shift (BF16 is the upper 16 bits of FP32), so `__bfloat162float` may already compile to a single instruction. Check the SASS output -- if it's already a single `PRMT` or `SHF` instruction per conversion, vectorizing won't help. But if the compiler generates multiple instructions, the paired conversion may reduce instruction count.

---

## 7. Launch Bounds and Occupancy Control

The worker should explicitly set `__launch_bounds__` on the NRM2 kernel:

```cuda
__global__ __launch_bounds__(256, 4)  // 256 threads, 4 blocks/SM
void nrm2_kernel(...)
```

This tells the compiler to optimize register allocation for 256 threads with 4 blocks/SM occupancy target. Without it, the compiler may allocate extra registers, reducing occupancy. With 64K registers/SM and 4 blocks * 256 threads = 1024 threads, each thread gets 64 registers -- plenty for a simple reduction kernel.

For comparison, the maximum occupancy configuration (48 warps = 1536 threads) would give 42 registers/thread, which is also sufficient but constrains the compiler more.

**Try both**: `__launch_bounds__(256, 4)` and `__launch_bounds__(256, 6)`. Compare ncu register counts and achieved occupancy.

---

## 8. L2 Cache Residency for Small Vectors

The RTX 5090 has 96 MB of L2 cache. For N=1M BF16 (2 MB), the entire input vector fits in L2 with room to spare. If the vector is already in L2 (from a prior kernel writing it), the NRM2 kernel should hit L2 bandwidth, not DRAM bandwidth.

**L2 bandwidth >> DRAM bandwidth**: L2 bandwidth on Blackwell is typically 3-4x DRAM bandwidth. If the vector is L2-resident, the kernel should be much faster.

**L2 persistence control**: Use `cudaAccessPropertyPersisting` to pin the input data in L2:

```cuda
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = (void*)x;
attr.accessPolicyWindow.num_bytes = n * sizeof(__nv_bfloat16);
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

**This may explain the cuBLAS gap**: cuBLAS may use L2 persistence hints internally, keeping the reduction input in L2. If our kernel doesn't, we may be paying DRAM latency while cuBLAS pays L2 latency.

---

## 9. Profile cuBLAS NRM2 to Understand the Target

Before optimizing further, run ncu on the cuBLAS NRM2 call to understand exactly what we're competing against:

```bash
ncu --target-processes all --kernel-name regex:nrm2 \
    --metrics dram__bytes.sum,l1tex__t_bytes.sum,sm__throughput.avg_pct_of_peak_sustained_elapsed \
    python -c "import torch; x = torch.randn(1000000, dtype=torch.bfloat16, device='cuda:1'); [torch.linalg.vector_norm(x) for _ in range(100)]"
```

Key metrics to capture:
- **Kernel name**: What function name does cuBLAS use? (reveals if it's a generic reduction or specialized nrm2)
- **Number of kernel launches**: 1 or 2? (confirms single-pass vs two-pass)
- **DRAM throughput**: What percentage of peak does cuBLAS achieve?
- **Block size and grid size**: Reveals cuBLAS's tuning choices
- **Register count**: How many registers does the cuBLAS kernel use?

This diagnostic is the single highest-value action. It tells the worker exactly what to match.

---

## Summary: Priority-Ordered Action Items

| # | Technique | Expected Impact | Effort | Already Covered? |
|---|-----------|----------------|--------|-----------------|
| 1 | Profile cuBLAS NRM2 with ncu | Diagnostic -- understand the target | Low | No |
| 2 | DOT-NRM2 gap analysis (remove sqrt, compare) | Diagnostic -- isolate the gap | Low | No |
| 3 | Warp-atomic approach (no shared memory) with small grid | 5-15% if sync overhead is the issue | Low | No |
| 4 | Multiple accumulators for ILP | 0-10% (helps if ALU-latency-limited) | Trivial | No |
| 5 | `__launch_bounds__` for occupancy control | 0-10% (compiler may over-allocate regs) | Trivial | No |
| 6 | L2 persistence hints for small vectors | 0-20% if L2 vs DRAM is the gap | Low | No |
| 7 | Vectorized BF16 pair conversion | 0-5% (check SASS first) | Trivial | No |
| 8 | Cooperative groups single-pass (no atomicAdd) | Alternative architecture worth trying | Medium | No |

Items 1-2 are diagnostics that should be done FIRST. The results determine which of items 3-8 will actually help.
