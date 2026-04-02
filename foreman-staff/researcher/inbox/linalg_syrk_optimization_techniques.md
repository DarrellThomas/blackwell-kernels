# SYRK Optimization: Techniques to Beat cuBLAS on sm_120

**Sources:** See references at bottom
**Relevant to:** linalg worker
**Worker's current problem:** SYRK at 0.96x cuBLAS using `torch.mm(A, A.t())`. This delegates to cuBLAS GEMM, which computes the full N*N output even though only the lower (or upper) triangle is needed. The worker needs to either close the 4% gap or find a custom kernel approach that exploits symmetry.

---

## 1. Why torch.mm(A, A.t()) Leaves Performance on the Table

`torch.mm(A, A.t())` dispatches to cuBLAS GEMM (not cuBLAS SYRK). It computes ALL N*N output elements even though the result is symmetric, wasting ~50% of compute. The 0.96x result means our GEMM path is already very close to cuBLAS GEMM -- the question is whether a symmetry-aware kernel can do better.

**Key fact from NVIDIA (Feb 2025):** No algorithm in cuBLAS currently implements `CUBLASLT_ALGO_CAP_UPLO_SUPPORT`. The attribute exists in the API but is unimplemented. This means cuBLAS's own `cublasSsyrk`/`cublasHsyrk` likely runs a full GEMM internally and masks the output, rather than actually skipping upper-triangle computation. This is confirmed by the PyTorch forums, where NVIDIA staff noted that SYRK performance is typically slightly LOWER than GEMM due to the masking overhead.

**Implication:** cuBLAS SYRK is not a silver bullet. A custom kernel that genuinely skips upper-triangle tiles could beat both cuBLAS GEMM and cuBLAS SYRK.

---

## 2. Triangle-Aware Tile Scheduling (The Main Opportunity)

The core optimization: in a tiled GEMM producing an N*N output, tiles that fall entirely above the diagonal can be skipped. For an N*N output with TILE_M * TILE_N tiles, the lower triangle contains approximately `(N/TILE_M) * (N/TILE_N + 1) / 2` tiles instead of `(N/TILE_M) * (N/TILE_N)` -- roughly half.

### CUTLASS Rank2K Scheduler Formula

CUTLASS has a production implementation of this in their grouped kernel schedulers. The mapping from a linear threadblock ID `t` (zero-indexed) to lower-triangular tile coordinates `(i, j)` is:

```
i = floor(sqrt(2*t + 2.25) - 0.5)    // row (zero-indexed)
j = t - i*(i+1)/2                      // column within that row
```

This maps threadblock IDs densely onto only the lower-triangular tiles, with no wasted CTAs. The inverse (for upper-triangular) swaps i and j.

**Why this matters for us:** Our BF16 GEMM uses 64x64 tiles. For a 4096x4096 SYRK output, that's 64x64 = 4096 tiles in full GEMM, but only ~2080 tiles in the lower triangle. We could launch ~half the CTAs and still cover the entire output.

### Implementation Sketch for Our Kernel

Starting from the existing 64x64 BF16 GEMM kernel:

1. **Compute total lower-triangle tiles:** `num_tiles = grid_m * (grid_m + 1) / 2` (for square output where grid_m = grid_n = N/64)
2. **Launch only num_tiles CTAs** instead of grid_m * grid_n
3. **In the kernel prologue**, convert linear CTA index to (tile_row, tile_col) using the formula above
4. **Diagonal tiles** need special handling: only compute and store the lower-triangle portion of the tile output
5. **Off-diagonal tiles** below the diagonal: compute normally, also store the transposed result at (tile_col, tile_row) to fill both halves of the symmetric output

### Diagonal Tile Handling

For tiles on the diagonal (tile_row == tile_col), the MMA output contains both lower and upper triangle elements. After the GEMM accumulation, mask the output: only store elements where `row >= col` within the tile. This is a simple conditional in the epilogue, costing negligible overhead.

### Mirror-Write Optimization

For off-diagonal tiles (tile_row > tile_col), you compute C[i_block, j_block] = A_rows * A_cols^T. The symmetric counterpart C[j_block, i_block] is just the transpose of this result. You can write BOTH blocks from a single CTA:
- Store the computed tile at its natural position
- Store the transpose at the mirror position

This means you compute each off-diagonal tile once and write it twice, halving total compute while doubling epilogue writes. Since SYRK is compute-bound (not memory-bound) at reasonable matrix sizes, this is a net win.

---

## 3. Shared Memory Optimization: A and A^T Share Storage

In standard GEMM, we load tiles of A and B into separate shared memory buffers. For SYRK (C = A * A^T), B is just A^T -- they reference the same global memory. This enables:

**Option A: Single shared memory buffer.** Load the A tile once, use it for both the A and B operands of the MMA. The A operand reads rows and the B operand reads columns of the same shared memory tile. This halves shared memory usage for the input tiles.

cuBLASDx has a `simple_gemm_aat` example that demonstrates exactly this pattern -- A and A^T share the same shared memory allocation, with different access patterns (row-major vs column-major) into the same data.

**Option B: Transpose in shared memory.** Load A tile, then create the transposed B tile in shared memory from the same data. More shared memory but simpler MMA feeding. Given our 99KB budget and 6 blocks/SM target, this may be tight.

**Recommendation:** Option A is better for our architecture. With 64x64 tiles at BF16, one buffer = 64*32*2 = 4KB per double-buffer slot. Shared memory savings let us either increase occupancy or widen tiles.

---

## 4. What the Research Literature Says

### FLAME/UT Austin (FLAWN37): Level-3 BLAS on GPU

The FLAME project demonstrated that SYRK can be decomposed into a family of algorithmic variants. The key insight: "only the part that lies at or below the diagonal needs to be updated, where a special kernel updates the lower triangular block on the diagonal, and the right part is not updated at all." Their implementation achieved ~15% gains over naive approaches by choosing the right algorithmic variant per matrix shape.

### Popcorn (arXiv:2501.05587): SYRK vs GEMM Trade-offs

This paper directly benchmarks cuBLAS SYRK vs cuBLAS GEMM for computing symmetric outputs. Key finding: "SYRK requires only O(n^2*d/2) FLOPS while GEMM requires O(n^2*d)." However, GEMM outperforms SYRK when `n/d > 100` because cuBLAS SYRK has higher per-tile overhead. For roughly square matrices (n ~ d), SYRK wins. This suggests that for typical SYRK workloads (4096x4096 output from 4096xK input), a custom triangle-scheduled kernel should win.

### OSTI/IEEE (2018): SYRK Edge Cases

Research from Oak Ridge found that "a redesign of the xSYRK routine to introduce data persistence, dynamic scheduling and limit the scheduling to the lower diagonal via block index reordering resulted in a speedup of 54x for single precision" on edge cases (very thin/fat matrices). While the 54x is for extreme edge cases, the technique of reordering block indices to cover only the lower triangle is directly applicable.

### Recursive SYRK (arXiv:2601.08082): GPU Acceleration

This 2026 paper shows that recursively decomposing SYRK into GEMM sub-problems achieves 14x over cuBLAS SYRK on H200 (FP64). The recursive approach exposes larger GEMM sub-problems that better saturate GPU compute. However, this is most relevant for FP64 (where SYRK is severely underoptimized); for BF16 on sm_120, the simpler triangle-scheduling approach is likely sufficient.

---

## 5. Practical Approach for the Linalg Worker

Given that the BF16 GEMM kernel is already at 0.97x cuBLAS, the fastest path to beating cuBLAS on SYRK is:

### Option A: Modify the Existing GEMM Kernel (Recommended)

1. **Fork the 64x64 BF16 GEMM kernel** into a SYRK variant
2. **Add triangle-aware CTA scheduling** using the sqrt formula above
3. **Add mirror-write** in the epilogue for off-diagonal tiles
4. **Add diagonal masking** for tiles on the diagonal
5. **Exploit shared A/A^T** to reduce shared memory pressure

Expected speedup: ~1.5-1.8x over full GEMM (computing ~half the tiles with some mirror-write overhead). Since we're at 0.96x cuBLAS GEMM, this could reach ~1.4-1.7x cuBLAS GEMM for SYRK, which would be a significant win.

### Option B: Use cuBLAS SYRK Directly

Call `cublasSsyrk` or the BF16 equivalent instead of `torch.mm`. This is the simplest change but unlikely to beat GEMM significantly, since NVIDIA confirmed no algorithm currently exploits `UPLO_SUPPORT` in cublasLt.

### Option C: cuBLASDx Device-Side (For Cholesky Integration)

Use cuBLASDx's `simple_gemm_aat` pattern for device-side SYRK within larger factorization kernels. This is more relevant for the numerical/Cholesky worker than for standalone SYRK benchmarks.

---

## 6. Caveats

- **The sqrt formula has floating-point precision issues** for very large tile counts. Use integer arithmetic or a correction step for production code. CUTLASS uses careful integer math.
- **Mirror-write doubles epilogue memory traffic.** For memory-bound regimes (very large K), this could hurt. Profile to confirm compute-boundedness.
- **cuBLAS SYRK might improve.** The `CUBLASLT_ALGO_CAP_UPLO_SUPPORT` attribute exists but is unimplemented as of Feb 2025. NVIDIA could ship a real triangle-aware algorithm in a future cuBLAS update, raising the bar.
- **Our GEMM is at 0.97x cuBLAS, not 1.0x.** The SYRK kernel inherits whatever inefficiency the GEMM has. Improving the base GEMM also improves SYRK.

---

## References

- [CUTLASS Grouped Kernel Schedulers (Rank2K triangle scheduling)](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)
- [CUTLASS Example 31: Basic SYRK](https://github.com/NVIDIA/cutlass/tree/main/examples/31_basic_syrk)
- [cublasLt SYRK Issue #166 -- NVIDIA confirms UPLO_SUPPORT unimplemented](https://github.com/NVIDIA/CUDALibrarySamples/issues/166)
- [FLAME FLAWN37: Level-3 BLAS on GPU](https://www.cs.utexas.edu/~flame/pubs/FLAWN37.pdf)
- [Popcorn: SYRK vs GEMM on GPU (arXiv:2501.05587)](https://arxiv.org/abs/2501.05587)
- [Performance Issues of SYRK (IEEE/OSTI 2018)](https://www.osti.gov/biblio/1557469)
- [Hierarchical Recursive SYRK (arXiv:2601.08082)](https://arxiv.org/html/2601.08082v1)
- [KBLAS-GPU: Optimized BLAS for NVIDIA GPUs](https://github.com/ecrc/kblas-gpu)
- [PyTorch Forums: Symmetric output from B.mm(B.t())](https://discuss.pytorch.org/t/i-wanna-do-a-b-mm-b-t-to-get-the-upper-triangular-matrix-since-the-result-is-symmetric/110457)
- [cuBLAS SYRK performance discussion (NVIDIA Forums)](https://forums.developer.nvidia.com/t/using-cuda-streams-and-cublasdsyrk-to-replace-cublasdgemmbatched-call-results-in-very-low-performance/67300)
- [Optimizing Symmetric Dense Matrix-Vector Multiplication on GPUs (Nath et al.)](https://cseweb.ucsd.edu/~rknath/sc11.pdf)
