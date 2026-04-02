# HPL GPU Panel Factorization (GPUPDFACT) via Cooperative Groups

**Source:** https://dl.acm.org/doi/10.1145/3712285.3759875 (SC'24 — Insights from Optimizing HPL on Exascale Systems)
**Source:** https://arxiv.org/abs/2304.10397 (Optimizing HPL for Exascale, 2023)
**Source:** https://github.com/ROCm/rocHPL
**Relevant to:** numerical/ worker (LU factorization panel factorization strategy)
**Worker's current problem:** Panel factorization is the serial bottleneck in GPU LU. MAGMA uses spin-wait inter-block sync (wasteful). cuSOLVER uses single-block (limits parallelism). Need a clean approach for cooperative multi-block panel factorization.

---

## What This Is

The rocHPL team (SC'24) introduced GPUPDFACT -- a **GPU-based panel factorization** that uses **cooperative groups** (HIP equivalent of CUDA cooperative groups) to perform the entire panel factorization on GPU without CPU involvement. This outperformed both CPU-based panel factorization and the traditional dedicated-thread approach on Frontier (AMD MI250X).

---

## Key Architecture

### Traditional HPL Panel Factorization (CPU-Based)

Standard HPL moves the panel to CPU for factorization:
1. GPU does trailing GEMM update
2. Panel columns copied to CPU
3. CPU does GETF2 (IDAMAX + swap + scale + rank-1 update) per column
4. Panel copied back to GPU
5. GPU applies pivots and does next trailing update

**Problem:** CPU panel factorization is latency-sensitive and doesn't overlap well with GPU trailing GEMM. PCIe transfers add overhead.

### GPUPDFACT: All-GPU Panel Factorization

The GPUPDFACT approach keeps everything on GPU:

```
For each column k in the panel:
  1. IDAMAX: Find local max in column k using cooperative grid reduction
  2. MAXSWAP: Exchange pivot info (within cooperative grid)
  3. Rank-1 update: All blocks cooperate on trailing submatrix update

  All synchronized via cooperative group grid.sync()
```

### Why Cooperative Groups Work Here

- **Clean barrier semantics**: `grid.sync()` after each column's pivoting ensures all blocks see the updated pivot before proceeding
- **No spin-wait waste**: Unlike MAGMA's atomic flag pattern, blocks sleep at the barrier rather than burning cycles
- **No deadlock risk**: Cooperative launch guarantees all blocks are resident simultaneously
- **Full SM utilization**: During the rank-1 update phase, all SMs participate

### Performance Results

GPUPDFACT outperformed:
- CPU-based PDFACT (traditional HPL approach)
- Dedicated-thread variant (multi-threaded CPU factorization)

On Frontier's MI250X GPUs. The specific advantage comes from eliminating CPU-GPU data transfers for the panel and keeping the GPU busy during factorization.

---

## Look-Ahead Integration

### Depth-1 Look-Ahead in HPL

The standard HPL look-ahead technique:

```
Iteration k:
  1. Panel k factorization completes
  2. UPDATE NEXT PANEL FIRST: Apply pivots + TRSM + GEMM to columns (k+1)*NB : (k+2)*NB
  3. Panel k+1 can start factorizing IMMEDIATELY
  4. Meanwhile, bulk trailing GEMM runs on remaining columns (k+2)*NB : N
```

This hides panel factorization latency behind the bulk trailing GEMM.

### In a Cooperative Groups Kernel

```cpp
__global__ void lu_cooperative(float* A, int N, int NB) {
    cg::grid_group grid = cg::this_grid();

    for (int k = 0; k < N/NB; k++) {
        // Panel factorization: subset of blocks
        if (blockIdx.x < panel_blocks) {
            panel_factorize(A, k, NB);
        }
        grid.sync();  // Panel k done

        // LOOK-AHEAD: All blocks update ONLY next panel's columns
        update_next_panel(A, k, NB);  // TRSM + GEMM on NB columns
        grid.sync();  // Next panel ready for factorization

        // BULK: All blocks update remaining trailing matrix
        // Panel k+1 factorization could overlap here if we had async panels
        bulk_trailing_gemm(A, k, NB);
        grid.sync();  // Trailing done
    }
}
```

The limitation of cooperative groups: `grid.sync()` is a **full barrier** -- you can't have some blocks doing panel k+1 while others do the bulk GEMM. True look-ahead overlap requires either:
- Two cooperative launches (complex)
- Atomic flag approach for finer-grained sync (MAGMA pattern)
- Accept the full-barrier approach (simplest, still fast)

---

## Application to Our LU at N=4096 on sm_120

### Recommended Architecture

```
Grid: ~340 blocks (2 blocks/SM * 170 SMs)
Threads: 256 per block (8 warps)
Shared memory: ~48KB per block

for k = 0 to 63:  // 4096/64 = 64 iterations
    // Phase 1: Panel factorization (block 0 only, or first few blocks)
    if (blockIdx.x == 0):
        load_panel_to_smem(A, k)
        for j = 0 to NB-1:
            argmax_reduction(column j)    // warp shuffle + shared mem reduction
            swap_rows(pivot_row, j)       // cooperative within block
            scale_column(j)               // parallel across threads
            rank1_update(j, NB)           // parallel across threads
        store_panel_to_gmem(A, k)

    grid.sync();  // Panel done

    // Phase 2: Distributed LASWP + TRSM
    // Each block handles a chunk of columns for row swaps
    distributed_laswp(A, ipiv, k);
    // TRSM distributed across blocks
    distributed_trsm(A, k, NB);

    grid.sync();  // LASWP + TRSM done

    // Phase 3: Trailing GEMM
    // Grid-stride loop: each block handles tiles of the trailing matrix
    int trailing_size = N - (k+1)*NB;
    int num_tiles = ceil(trailing_size / TILE_M) * ceil(trailing_size / TILE_N);
    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x):
        load_L_and_U_tiles(A, k, tile);
        bf16_mma_gemm(L_tile, U_tile, C_tile);  // Our proven BF16 MMA
        store_C_tile(A, k, tile);

    grid.sync();  // Trailing done
```

### Panel Block Count Decision

For N=4096, NB=64, the panel is 4096 rows x 64 columns.

**Single-block panel (block 0 only):**
- 256 threads doing argmax over 4096 elements: ~16 rows per thread (NPAGES=16)
- Simple, no inter-block sync within panel
- Panel time: ~64 columns * (argmax + swap + scale + rank1) ≈ ~100us
- 169 SMs idle during panel

**Multi-block panel (MAGMA style):**
- 64 blocks (one per column), spin-wait on atomic flags
- More parallelism but wasted cycles spinning
- Complexity overhead for marginal panel speedup

**Recommendation: Single-block panel.** The panel is only ~5-10% of total time. The trailing GEMM dominates. Focus optimization effort on the trailing GEMM phase where all 340 blocks participate.

---

## Caveats

1. **Cooperative launch limits grid size.** All blocks must be resident. On sm_120 with 256 threads and ~48KB shared memory per block, expect ~2 blocks/SM = 340 blocks total. This is sufficient for N=4096 trailing GEMM.

2. **Grid sync cost.** Each `grid.sync()` takes ~1-5 microseconds. With 64 iterations and 3 syncs per iteration = 192 syncs = ~0.2-1ms overhead. Non-trivial but acceptable.

3. **Panel leaves SMs idle.** Single-block panel means 169 of 170 SMs are idle during factorization. For N=4096 with NB=64, panel is ~5% of total compute, so this is acceptable. For larger N or larger NB, consider the look-ahead pattern to overlap.

---

## Sources

- [Insights from Optimizing HPL on Exascale (SC'24)](https://dl.acm.org/doi/10.1145/3712285.3759875)
- [Optimizing HPL for Exascale (arXiv, 2023)](https://arxiv.org/abs/2304.10397)
- [rocHPL GitHub](https://github.com/ROCm/rocHPL)
- [CUDA Cooperative Groups Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
