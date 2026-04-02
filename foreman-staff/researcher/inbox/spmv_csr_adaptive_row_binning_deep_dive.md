# CSR-Adaptive Row Binning: Implementation Deep Dive

**Sources:**
- [CSR-Adaptive: SC 2014 (Greathouse & Daga)](https://www.computermachines.org/joe/publications/pdfs/sc2014_csr-adaptive.pdf)
- [Structural Agnostic SpMV: HiPC 2015](https://www.computermachines.org/joe/publications/pdfs/hipc2015_spmv.pdf)
- [rocSPARSE CSR-Adaptive Implementation](https://github.com/ROCm/rocSPARSE)
- [Logarithmic Radix Binning: HPEC 2019 (Green & Fox)](https://ieeexplore.ieee.org/document/8916333/)
- [GPU Work Graphs SpMV: ISCA 2025 (Wildgrube et al.)](https://dl.acm.org/doi/10.1145/3695053.3731060)
- [Ginkgo Load-Balanced SpMV](https://dl.acm.org/doi/fullHtml/10.1145/3380930)
**Relevant to:** spmv worker
**Worker's current problem:** Building CSR SpMV kernel to beat cuSPARSE on RTX 5090

---

## What This Is

A detailed implementation guide for the CSR-Adaptive (row-binned) SpMV algorithm,
covering the exact row classification strategy, bin thresholds, and per-bin kernel
implementations. This goes deeper than the existing docs which describe the concept
but lack implementation-level specifics.

## Why It Matters for Us

cuSPARSE uses merge-based SpMV internally (confirmed by NVIDIA engineer fbusato).
Merge-based is robust but not optimal for any specific row-length distribution.
CSR-Adaptive with proper binning beats merge-based by 1.2-1.5x on irregular matrices
because each bin gets a kernel tuned for its specific row-length range.

---

## 1. The Row Classification Algorithm

### Analysis Pass (Preprocessing, O(M) time)

Single pass over `row_ptr[]` to classify rows. This is the key preprocessing step
that cuSPARSE's merge-based approach does NOT do.

```
Input: row_ptr[0..M], block_size (e.g., 256 threads = 8 warps)
Output: row_blocks[] array mapping thread blocks to row ranges

Algorithm:
  total_nnz_in_block = 0
  row_block_start = 0

  for i = 0 to M-1:
    row_len = row_ptr[i+1] - row_ptr[i]
    total_nnz_in_block += row_len

    if total_nnz_in_block > block_size:  // Can't fit more rows in this block
      if row_block_start == i:
        // Single row too long for one block -> CSR-VectorL
        // May span multiple blocks
        num_blocks_needed = ceil(row_len / block_size)
        for b in range(num_blocks_needed):
          row_blocks.append((i, i+1, VECTORL))
        row_block_start = i + 1
        total_nnz_in_block = 0
      else:
        // Previous rows fit as CSR-Stream
        row_blocks.append((row_block_start, i, STREAM))
        row_block_start = i
        total_nnz_in_block = row_len

  // Handle remaining rows
  if row_block_start < M:
    row_blocks.append((row_block_start, M, STREAM))
```

### Bin Categories (from SC 2014 paper)

The original CSR-Adaptive uses TWO strategies, not three:

| Strategy | Condition | Description |
|----------|-----------|-------------|
| **CSR-Stream** | Total nnz in row group fits in shared memory | Multiple rows packed into one block, processed via shared memory |
| **CSR-Vector** | Single row with nnz <= block_size | One warp (or block) processes one row |
| **CSR-VectorL** | Single row with nnz > block_size | Multiple blocks process one row, atomic reduction |

The threshold is whether the total nonzeros for a group of rows fits within
`block_size` (typically 256). If a single row exceeds this, it gets CSR-Vector
or CSR-VectorL treatment.

---

## 2. CSR-Stream: Short Rows Packed into Shared Memory

This is the key innovation. Multiple short rows are packed into a single thread
block, processed entirely in shared memory.

### How It Works

```
Given: block processes rows [row_start, row_end)
       total_nnz = sum of all nonzeros in these rows
       total_nnz <= block_size (fits in shared memory)

__shared__ float shared_vals[BLOCK_SIZE];   // nonzero values
__shared__ int   shared_cols[BLOCK_SIZE];   // column indices

Step 1: Collaborative load into shared memory
  Each thread loads one val/col pair from global memory (coalesced)
  tid = threadIdx.x
  if (tid < total_nnz):
    shared_vals[tid] = vals[row_ptr[row_start] + tid]
    shared_cols[tid] = col_idx[row_ptr[row_start] + tid]
  __syncthreads()

Step 2: Each thread processes assigned rows
  // Assign rows to threads round-robin or by row index
  for each row assigned to this thread:
    sum = 0
    for j = local_row_start to local_row_end:
      sum += shared_vals[j] * x[shared_cols[j]]
    y[row] = sum
```

### Why This Is Efficient

1. **Coalesced global loads**: All threads load consecutive val/col pairs
2. **Shared memory for reduction**: Random x[] accesses happen from L1/L2, not
   through the complex merge-path binary search
3. **No warp divergence within the block**: All threads participate in the load,
   then all participate in the computation
4. **Works great for many short rows**: FEM matrices with avg_nnz < 32 benefit
   enormously

### Optimal Configuration

- `BLOCK_SIZE = 256` (8 warps): Good balance of shared memory capacity and occupancy
- Shared memory per block: `256 * (4 + 4) = 2048 bytes` for val+col
- This leaves 95+ KB of shared memory for L1 cache carveout on sm_120

---

## 3. CSR-Vector: Warp-Per-Row for Medium Rows

Standard approach, well-documented in existing docs. Key implementation detail:

```cuda
// One warp per row
int row = blockIdx.x * warps_per_block + (threadIdx.x / 32);
int lane = threadIdx.x % 32;

if (row < M) {
    float sum = 0.0f;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];

    for (int j = start + lane; j < end; j += 32)
        sum += val[j] * x[col_idx[j]];

    // Warp shuffle reduction (5 steps for warp size 32)
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane == 0)
        y[row] = sum;
}
```

---

## 4. Sub-Warp Assignment: Matching Thread Count to Row Length

This is a refinement the existing docs mention but don't detail. Instead of always
using a full 32-thread warp per row, use cooperative groups or manual masking to
assign fewer threads to short rows:

```cuda
// Using cooperative_groups::thread_block_tile
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <int TILE_SIZE>  // 2, 4, 8, 16, or 32
__device__ void process_row_subwarp(
    int row, const int* row_ptr, const int* col_idx,
    const float* val, const float* x, float* y)
{
    auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
    int lane = tile.thread_rank();

    float sum = 0.0f;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];

    for (int j = start + lane; j < end; j += TILE_SIZE)
        sum += val[j] * x[col_idx[j]];

    // Reduction within the tile
    for (int offset = TILE_SIZE / 2; offset > 0; offset >>= 1)
        sum += tile.shfl_down(sum, offset);

    if (lane == 0)
        y[row] = sum;
}
```

### Optimal Sub-Warp Sizes

| Row Length (nnz) | Threads/Row | Rows/Warp | Efficiency |
|-----------------|-------------|-----------|------------|
| 1-2             | 2           | 16        | High: 16 rows per warp |
| 3-4             | 4           | 8         | 8 rows per warp |
| 5-8             | 8           | 4         | 4 rows per warp |
| 9-16            | 16          | 2         | 2 rows per warp |
| 17-32           | 32          | 1         | Standard CSR-Vector |
| 33+             | 32+ (multi-warp) | <1   | CSR-VectorL |

Using sub-warp tiles instead of full warps for short rows can improve throughput
by 2-8x for matrices dominated by short rows.

**sm_120 note**: `cooperative_groups::thread_block_tile` with compile-time tile
sizes (2, 4, 8, 16, 32) compiles to efficient warp shuffle instructions. No
shared memory or synchronization overhead.

---

## 5. Logarithmic Radix Binning (LRB) Alternative

LRB (Green & Fox, HPEC 2019) is a simpler alternative to CSR-Adaptive's
analysis pass:

### How It Works

1. Compute row lengths: `len[i] = row_ptr[i+1] - row_ptr[i]`
2. Compute bin index: `bin[i] = floor(log2(len[i]))` (or 0 for empty rows)
3. Sort rows into power-of-two bins: bin 0 (len 0-1), bin 1 (len 2-3),
   bin 2 (len 4-7), bin 3 (len 8-15), ...
4. Launch separate kernels for each non-empty bin

### Advantages Over CSR-Adaptive

- **Simpler preprocessing**: No sequential scan of row_ptr needed, fully parallel
- **Preprocessing runs on GPU**: 20x faster than CSR-Adaptive's host-side analysis
- **Within each bin, row lengths differ by at most 2x**: Minimal warp divergence
- **Each bin's kernel is simple**: Compile-time tile size matching the bin

### Disadvantages

- **Multiple kernel launches**: One per non-empty bin (typically 10-15 bins)
- **Less coalesced than CSR-Stream**: Rows are reordered, breaking spatial locality
- **Per-SpMV performance slightly lower**: LRB's reordering overhead shows up

### When to Use LRB vs CSR-Adaptive

- **LRB**: When preprocessing must be fast (few SpMV iterations per matrix)
- **CSR-Adaptive**: When preprocessing is amortized (iterative solvers, 100+ SpMVs)

### GPU Work Graphs SpMV (ISCA 2025) Connection

The Work Graphs paper uses a variant of LRB where the binning node on the GPU
directly spawns compute nodes for each bin, eliminating host-device round trips.
On CUDA, we can approximate this with:
- Persistent kernel with a global work queue
- Or separate kernel launches per bin (simpler, small overhead on sm_120)

---

## 6. Recommended Implementation for RTX 5090

### Two-Level Binning with Sub-Warp Assignment

```
Preprocessing (GPU, one-time):
  1. Compute row lengths in parallel
  2. Classify into 6 bins:
     - Bin 0: empty rows (len = 0)      -> skip
     - Bin 1: tiny rows (len 1-4)       -> tile_size=4, 8 rows/warp
     - Bin 2: short rows (len 5-32)     -> tile_size=32, 1 row/warp
     - Bin 3: medium rows (len 33-256)  -> tile_size=32, 1 row/warp (multi-pass)
     - Bin 4: long rows (len 257-4096)  -> full block per row
     - Bin 5: very long rows (len 4097+)-> multiple blocks per row + atomicAdd
  3. Count rows per bin, compute offsets
  4. Scatter row indices into bin-sorted array

Execution (per bin):
  - Bin 1: Launch with 4-thread tiles, 8 rows/warp
  - Bin 2: Launch with warp-per-row
  - Bin 3: Launch with warp-per-row, vectorized loads (float4)
  - Bin 4: Launch with block-per-row, shared memory reduction
  - Bin 5: Launch with multi-block-per-row, atomicAdd on y[]
```

### Performance Expectations

Based on LightSpMV, DASP, and Work Graphs results scaled to RTX 5090:
- 1.2-1.5x over cuSPARSE on irregular matrices (power-law, social graphs)
- 1.0-1.1x on regular matrices (cuSPARSE's merge-based is already good here)
- Combined with BF16 values: 1.5-2.0x over cuSPARSE FP32

---

## Caveats

1. **Preprocessing cost**: The row classification and bin-sorting takes O(M) GPU
   work. For a 1M-row matrix, this is ~0.1ms on RTX 5090. Amortized over even
   10 SpMV iterations, this is negligible.

2. **Row index indirection**: Bin-sorted row indices add one level of indirection
   for accessing row_ptr. This is a single random read per row -- negligible
   compared to the x-vector accesses.

3. **atomicAdd for very long rows**: On sm_120, FP32 atomicAdd is fast for low
   contention. Very long rows are rare, so contention is low.

4. **Multiple kernel launches**: On sm_120, kernel launch overhead is ~5-10us.
   With 5 bins, that's 25-50us of overhead. For matrices with nnz > 1M, SpMV
   takes 50+ us, so launch overhead is <50%. For small matrices, use a single
   kernel with if/else branching instead of separate launches.
