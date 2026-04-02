# Cooperative Groups Pivot Search: Grid-Wide Argmax for Monolithic LU

**Source:** https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
**Source:** https://dl.acm.org/doi/10.1145/3712285.3759875 (SC'24 HPL Panel Factorization)
**Source:** https://discourse.julialang.org/t/improving-performance-of-cuda-gpu-kernel-lu-factorization/132971
**Source:** https://arxiv.org/pdf/2004.05371 (GPU Synchronization Methods Study)
**Relevant to:** LU worker
**Worker's current problem:** Monolithic LU kernel needs device-side pivot selection (argmax over a column). For single-block panel, this is intra-block reduction. For multi-block cooperative kernel, need grid-wide reduction.

## What This Is

Survey of techniques for implementing the IDAMAX (find maximum absolute value
and its index) operation inside a monolithic LU kernel, covering both
single-block and cooperative grid approaches.

## Technique 1: Single-Block Warp-Shuffle Argmax (Recommended for Panel)

For single-block panel factorization with 256 threads, the column to search
has up to N=4096 elements. Each thread handles ceil(4096/256) = 16 elements.

```
Phase 1: Thread-local max (each thread scans its 16 elements)
  local_max = -inf, local_idx = -1
  for i in range(16):
      val = abs(A[thread_offset + i])
      if val > local_max: local_max = val; local_idx = thread_offset + i

Phase 2: Warp-level reduction (32 threads -> 1 max per warp)
  for offset = 16; offset >= 1; offset >>= 1:
      other_max = __shfl_xor_sync(0xFFFFFFFF, local_max, offset)
      other_idx = __shfl_xor_sync(0xFFFFFFFF, local_idx, offset)
      if other_max > local_max: local_max = other_max; local_idx = other_idx

Phase 3: Block-level reduction (8 warps -> 1 max per block)
  // Warp leaders write to shared memory
  if (lane_id == 0) { smem_max[warp_id] = local_max; smem_idx[warp_id] = local_idx; }
  __syncthreads()
  // First warp reduces across warp results
  if (warp_id == 0 && lane_id < 8):
      warp-level reduce over smem_max[0:8]
```

**Cost:** ~5 warp shuffles + 1 syncthreads + 8 shared memory accesses per column.
For 64 columns (NB=64): ~64 * (5 shuffles + sync) = negligible vs trailing GEMM.

## Technique 2: Cooperative Grid Argmax (For Multi-Block Approaches)

If using cooperative groups with `grid.sync()`, a grid-wide argmax requires:

```
// Each block finds its local maximum
block_argmax(column, &block_max, &block_idx);

// Write to global memory
atomicMax_with_index(global_max, global_idx, block_max, block_idx);

grid.sync();  // All blocks see the result
```

**The problem:** `atomicMax` for float requires a CAS loop (no native float atomicMax).
CUDA provides `atomicMax` for int/uint but not float. Workaround: reinterpret float
as int (works because IEEE 754 floats have the same ordering as ints for positive
values, and all absolute values are positive).

```cpp
// Trick: abs(float) -> unsigned int preserves ordering
__device__ void atomicArgmax(float* addr, int* idx_addr, float val, int idx) {
    unsigned int* uaddr = (unsigned int*)addr;
    unsigned int uval = __float_as_uint(val);
    unsigned int old = atomicMax(uaddr, uval);
    if (__uint_as_float(old) < val) {
        atomicExch(idx_addr, idx);  // Race condition possible!
    }
}
```

**Race condition warning:** The max value and index updates are not atomic together.
Solutions:
1. Pack (value, index) into a uint64 and use atomicMax on uint64
2. Use a two-phase approach: first find max value, then find its index
3. Use MAGMA's spin-wait atomic flag pattern (more complex)

### Two-Phase Grid-Wide Argmax (Cleanest Approach)

```
Phase 1: Block-local max, write to global array
  block_maxes[blockIdx.x] = local_max
  block_indices[blockIdx.x] = local_idx
  grid.sync()

Phase 2: Block 0 reduces across block results
  if (blockIdx.x == 0):
      // 340 block results -> one final max
      // 340 values can be reduced by 256 threads in 2 steps
      global_max = reduce(block_maxes, gridDim.x)
  grid.sync()

  // All blocks read the pivot from global memory
```

**Cost:** 2 `grid.sync()` calls per column. At ~1-5 us per sync, for 64 columns:
~128-640 us of sync overhead. Non-trivial but acceptable.

## Technique 3: HPL GPUPDFACT Pattern (SC'24)

The rocHPL GPUPDFACT kernel (SC'24) implements the full panel factorization
on GPU using cooperative groups:

```
for each column k in the panel:
    1. IDAMAX: cooperative grid reduction (block-local max + grid reduce)
    2. MAXSWAP: broadcast pivot info + row swap
    3. Rank-1 update: all blocks cooperate
    grid.sync()
```

Key implementation details from the paper:
- Uses HIP `cooperative_groups::grid_group::sync()` (equivalent to CUDA)
- Panel factorization kept entirely on GPU (no CPU involvement)
- Outperformed both CPU-based panel and dedicated-thread variants on MI250X
- Eliminates PCIe transfer overhead for panel data

## Recommendation for Our Monolithic LU

**Use single-block panel factorization (Technique 1):**

Rationale:
1. Panel factorization is ~5% of total compute at N=4096 with NB=64
2. Single-block avoids grid.sync() overhead during panel
3. 256 threads is sufficient for argmax over 4096 elements (16 elements/thread)
4. All other SMs are idle during panel anyway (cooperative approach helps)
5. Simple implementation, no race conditions

**Reserve cooperative grid approach for the trailing GEMM only:**
- All 340 blocks participate in the trailing GEMM (the bottleneck)
- grid.sync() between panel and trailing phases
- This matches the HPL GPUPDFACT architecture

## Caveats

1. **Cooperative launch on sm_120:** cudaLaunchCooperativeKernel requires all
   blocks to be resident. With 256 threads and ~48KB smem per block, expect
   ~2 blocks/SM = 340 blocks. Verify with cudaOccupancyMaxActiveBlocksPerMultiprocessor.

2. **grid.sync() cost:** Measured at 1-5 us on modern GPUs. With 64 panel
   iterations and 2-3 syncs per iteration, total sync overhead is ~200-1000 us.
   This is small vs the total factorization time (~5-10 ms) but not negligible.

3. **No new research on cooperative pivot algorithms:** The search found no
   papers since the HPL GPUPDFACT work (SC'24) that introduce novel pivot
   search techniques for GPU LU. The warp-shuffle + block-reduce + grid-sync
   pattern appears to be the state of the art.
