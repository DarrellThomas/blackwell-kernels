# Bank Conflict Reduction for Wide FP8 Tiles -- Addendum

**Source:** Multiple (see Sources section)
**Relevant to:** GEMM worker
**Worker's current problem:** 86K bank conflicts from B loads with N=128 stride; supplementary findings to the existing brief.

## What This Is

Additional research findings that supplement the existing brief (`gemm_fp8_bank_conflicts_wide_n_tile_swizzle.md`). New data points from DeepGEMM internals, gau-nernst's RTX 5090 flash attention, ldmatrix wavefront mechanics, and the Blackwell microbenchmark paper.

---

## 1. Confirmed: sm_120 Has 32 Banks / 4 Bytes Each (Same as Ampere/Hopper)

gau-nernst's flash attention blog for the RTX 5090 states explicitly: "GPU's shared memory is backed by 32 memory banks, with consecutive 4-byte memory addresses assigned to consecutive memory banks." The NVIDIA developer forum thread on Blackwell shared memory confirms: "32 bits per bank per clock cycle (per SM)" from compute capability 2.x through 12.x.

There is no 128-bit bank mode or 16-bank configuration on sm_120. The standard 32 banks / 4 bytes model applies. This means all swizzle math from Ampere/Ada carries over unchanged.

**Source:** [gau-nernst Flash Attention 5090](https://gau-nernst.github.io/fa-5090/), [NVIDIA Forum: Blackwell Shared Memory](https://forums.developer.nvidia.com/t/lsu-wavefront-scheduling-and-shared-memory-bank-utilization-on-blackwell/359791)

## 2. gau-nernst's Working Swizzle Template for RTX 5090

gau-nernst's flash attention implementation for the RTX 5090 includes a concrete, working swizzle template that was verified to eliminate bank conflicts on sm_120:

```cpp
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
    uint32_t row_idx = (index / STRIDE) % 8;
    uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return index ^ (bits_to_xor << 4);
}
```

This XORs bits 4-6 of the byte address with bits 0-2 of the row index. The key property: `swizzle(addr + offset) = swizzle(addr) XOR offset` when addresses are 16-byte aligned. This lets the kernel pre-compute the swizzled base address once per row and then XOR constant offsets for column stepping -- avoiding repeated swizzle computation inside the hot loop.

With this swizzle, gau-nernst went from 68% to 86% of speed-of-light on the RTX 5090. Bank conflicts dropped to zero (verified via ncu L1 wavefronts metric, actual/ideal ratio dropped from 8x to 1x).

**Relevance to GEMM worker:** This is a proven swizzle implementation for sm_120. The worker's current swizzle operates on element indices rather than byte addresses. Switching to byte-address-based swizzle (matching gau-nernst's approach) may simplify the 4-bit extension needed for N=128.

**Source:** [gau-nernst Flash Attention 5090](https://gau-nernst.github.io/fa-5090/)

## 3. ldmatrix Hardware Wavefront Splitting -- Per-Phase Conflict Analysis

When `ldmatrix.x2` or `ldmatrix.x4` executes, the hardware splits the 32-thread warp into 4 groups of 8 threads (phase 0: T0-T7, phase 1: T8-T15, phase 2: T16-T23, phase 3: T24-T31). Each phase is a separate 128-byte transaction. Bank conflicts are evaluated **per phase**, not across the full warp.

This means: for bank-conflict-free ldmatrix access, you only need to ensure that within each 8-thread group, no two threads hit the same bank. This is a weaker requirement than full-warp conflict freedom.

For the GEMM worker's B loads with `ldmatrix_x2_trans`:
- Each phase has 8 threads, each providing a 16-byte (128-bit) address
- With `.trans`, threads within a phase access 8 different rows at the same column offset
- The row stride is 256 bytes (128 BF16 elements), which is 2x the bank cycle (128 bytes)
- Per-phase: 8 threads hit rows 0,1,2,...,7 offset by 256 bytes each. Bank for thread t: `(base + t*256) / 4 mod 32 = (base/4 + t*64) mod 32 = (base/4) mod 32` for all t (since 64 mod 32 = 0). This is 8-way conflict per phase.

The 4-bit swizzle fix proposed in the existing brief addresses this by making the row index XOR into the column index with enough bits to cover the full 128-wide N dimension.

**NVIDIA Forum note:** Nsight Compute reports different bank conflict metrics depending on GPU generation. On sm_86+ the metrics are accurate; on sm_75 the LDSM instruction was not instrumented correctly. sm_120 should report accurate metrics since it's post-sm_86.

**Source:** [ldmatrix behavior (NVIDIA Forum)](https://forums.developer.nvidia.com/t/understanding-the-behaivor-of-ldmatrix-in-terms-of-shared-memory-access/278716), [Flash Attention Part 4 (lubits.ch)](https://lubits.ch/flash/Part-4)

## 4. DeepGEMM's Swizzle Approach (Hopper, but Technique Transfers)

DeepGEMM (DeepSeek) uses 128B swizzle for both A and B matrices. Their approach:

```cpp
col ^= row % (kSwizzleDMode / 16);
```

Where `kSwizzleDMode` = 128 for 128B swizzle mode. So `col ^= row % 8`, operating on 16-byte units. The XOR is applied to the 16-byte-granularity column index.

Key details:
- All shared memory buffers aligned to **1024 bytes** (`__align__(1024)`). The 128B swizzle pattern repeats every 1024 bytes, so 1024-byte alignment ensures the base offset is 0 for the swizzle.
- Bank group unit = 16 bytes (4 shared memory banks per group), matching the ldmatrix/stmatrix access granularity.
- The swizzle is bijective within domains of size `k * 2^(BBits + MBase)` = `k * 128` bytes.

**Relevance to GEMM worker:** The 1024-byte alignment requirement is worth noting. If the B shared memory buffer is not 1024-byte aligned, the swizzle pattern doesn't start at offset 0 and the bank conflict avoidance degrades. The current kernel uses `extern __shared__ char smem_raw[]` with B starting at `smem_A + 2 * A_SMEM_ELEMS`. This offset may or may not be 1024-byte aligned depending on tile sizes. For the 64x128 path: A_SMEM_ELEMS = 64*64 = 4096 BF16 = 8192 bytes, so B starts at offset 2*8192 = 16384 -- this IS 1024-byte aligned. Good.

For the 128x128 path: A_SMEM_ELEMS = 128*32 = 4096 BF16 = 8192 bytes, B starts at 16384 -- also 1024-byte aligned. Good.

**Source:** [DeepGEMM source (GitHub)](https://github.com/deepseek-ai/DeepGEMM), [DeepGEMM analysis (kingsleykim.dev)](https://kingsleykim.dev/blog/deepgemm/)

## 5. CUTLASS "Two Atoms" Approach for Rows Wider Than 128 Bytes

For BF16 with BLOCK_N=128: each row is 128 * 2 = 256 bytes. The maximum canonical CUTLASS swizzle atom is 128 bytes (`Swizzle<3,4,3>`). When the row exceeds 128 bytes, CUTLASS treats each row as containing multiple swizzle atoms side-by-side.

For a 256-byte row: two 128-byte atoms. Each atom is swizzled independently with `Swizzle<3,4,3>`. The tile_to_shape function replicates the swizzle atom across the wider dimension.

The proposed 4-bit XOR fix in the existing brief (`SWIZZLE_BITS=4, SWIZZLE_MASK=15`) is mathematically equivalent to two side-by-side 3-bit atoms. Here's why: with 4 XOR bits, the swizzle covers 16 chunks (128 elements). But because the XOR is `col ^ ((row & 15) << 3)`, and bits [5:3] vs bit [6] are swizzled differently, the left half (chunks 0-7) and right half (chunks 8-15) end up with independent 3-bit swizzle patterns -- exactly the two-atom model.

The CuTeDSL blog confirms the selection logic:
- `major_mode_size * element_width >= 1024 bits` --> SW128 (`Swizzle<3,4,3>`)
- For BF16, N=128: 128 * 16 = 2048 bits --> SW128 selected, then tiled 2x across N

**Source:** [CuTeDSL Swizzle Usage (veitner.bearblog.dev)](https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/), [CuTe Swizzle Math (Lei Mao)](https://leimao.github.io/blog/CuTe-Swizzle/)

## 6. Flash Attention Part 4: Concrete Bank Conflict Numbers and Swizzle Formula

The lubits.ch flash attention tutorial provides concrete before/after numbers that closely match our scenario:

**Before swizzle:**
- 8-way bank conflicts on ldmatrix (SMEM -> RF)
- 8-way bank conflicts on RF -> SMEM stores
- Kernel at 33 TFLOPS
- SMEM bandwidth utilization: 93.6% (heavily oversubscribed)

**After swizzle:**
- Zero bank conflicts
- Kernel at 66 TFLOPS (2x improvement)
- SMEM bandwidth utilization: 23.5%

The formula used:
```c
int get_swizzled_col(const int &row, const int &col) {
    const int region_row = row % BANKS_PER_VEC4_ACCESS;  // 8
    const int bank_col = col / ELEMS_PER_BANK;            // 16B chunks
    const int bank_offset = col % ELEMS_PER_BANK;
    return ((region_row ^ bank_col) * ELEMS_PER_BANK) + bank_offset;
}
```

This is the "row XOR col at 16-byte granularity" pattern. For their kernel with 16-byte vectorized access (4 banks at a time), they model shared memory as having "8 banks of 16B each" during vector loads. The hardware splits each 128-bit (16B) load into 4 phases of 8 threads; within each phase, 8 threads access 4 consecutive 4-byte banks simultaneously.

**Key insight for GEMM worker:** The 2x performance improvement from eliminating bank conflicts in a memory-bound attention kernel is the upper bound. The GEMM worker's kernel is more compute-balanced (math_throttle 27%), so the expected improvement is smaller but still meaningful (the existing brief estimates 3-8%).

**Source:** [Flash Attention Part 4 (lubits.ch)](https://lubits.ch/flash/Part-4)

## 7. H100 GEMM Worklog: Padding Naturally Resolves Some Warp Tiling Conflicts

The H100 GEMM worklog (Hamza El Shafie) found an interesting progression:

1. **Kernel 5 (scalar tiling):** 5-way bank conflicts on B loads. Padding with stride+4 fixes store conflicts but not load conflicts.
2. **Kernel 6 (warp tiling):** Bank conflicts disappear naturally because warp-local sub-tile access patterns naturally spread across more bank groups. "COLS_PER_THREAD = 4 spreads sharedB lanes across more bank groups."
3. **Kernel 7 (TMA):** Hardware swizzle handles everything.

The lesson: warp tiling geometry affects bank conflict severity. The GEMM worker uses 2x2 warp tiling (WARPS_M=2, WARPS_N=2). Each warp processes WARP_N = BLOCK_N / WARPS_N = 128 / 2 = 64 columns. Within a warp, MMA_N_TILES = 64 / 8 = 8 ldmatrix loads across the N dimension.

With 4-bit swizzle, each warp's 8 ldmatrix column positions (at n=0,8,16,...,56) would land on different swizzled banks. This should be conflict-free.

**Source:** [H100 GEMM Worklog (Hamza El Shafie)](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/)

## 8. Warning: "4-bit XOR swizzle for N=128" Was Already Tried (Experiment 113)

The agent state lists under "FP8 dead ends":
> "4-bit XOR swizzle for N=128 -- bank conflicts unchanged"

This is concerning. If the worker already tried a 4-bit XOR and it didn't help, the root cause may not be the swizzle bit width at all. Possible explanations:

1. **The swizzle was applied correctly to stores but not loads.** Both cp.async stores AND ldmatrix loads must use the same swizzle function. If only one side was updated, the data is swizzled in memory but read at un-swizzled addresses (or vice versa), producing wrong results or no conflict improvement.

2. **The mask was applied to the wrong bits.** The current swizzle XORs `col ^ ((row & MASK) << 3)`. With MASK=15, this XORs bits [6:3] of col with bits [3:0] of row. But if the B matrix is stored row-major as B[K][N], the "row" for swizzle purposes is the K index, not the output row. Verify that the row parameter passed to swizzle_idx for B loads matches the K row index used during B stores.

3. **The conflict source is not the column access pattern.** The 86K conflicts might come from a different access -- e.g., the conversion phase (bf16x2_pair_to_e4m3x4) or the store to C in the epilogue. Nsight Compute can break down bank conflicts by instruction. The worker should check `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (loads) vs `...op_st.sum` (stores) separately, and correlate with specific SASS instructions.

4. **ldmatrix_x2_trans has a fundamentally different address pattern than ldmatrix_x4.** For ldmatrix_x4 (A operand), each thread provides the address of an 8x8 sub-matrix and the hardware distributes across the warp. For ldmatrix_x2_trans (B operand), the `.trans` modifier changes the memory access pattern -- the hardware reads column data from row-strided addresses. The XOR swizzle must decorrelate the *row-strided* pattern specifically. A 4-bit XOR on the column index doesn't help if the conflict arises from the row stride.

**Recommended diagnostic:**
```bash
ncu --set full --section SourceCounters \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
              l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    --kernel-name fp8_gemm_kernel_64 ./benchmark
```

Then in the ncu GUI, correlate bank conflicts with specific source lines (the ldmatrix_x2_trans calls vs ldmatrix_x4_mma calls vs cp.async stores).

---

## Revised Recommendation

Given that experiment 113 ("4-bit XOR swizzle for N=128") was already tried and failed:

1. **Diagnose first:** Use ncu source-level bank conflict attribution to determine which specific instruction(s) produce the 86K conflicts. Is it B loads (ldmatrix_x2_trans), A loads (ldmatrix_x4_mma), B stores (cp.async), or the epilogue?

2. **If B loads via ldmatrix_x2_trans:** The XOR swizzle must operate on the physical byte address that ldmatrix sees, not on the logical element coordinate. Consider switching to gau-nernst's byte-address swizzle template (Section 2 above) instead of the element-coordinate swizzle.

3. **If conflicts survive all swizzle attempts:** Try padding the B tile (`BLOCK_N + 8` elements per row). Padding changes the row stride from 256 bytes to 272 bytes, which is not a multiple of 128 (the bank cycle), breaking the systematic aliasing. This is guaranteed to work regardless of access pattern.

4. **If the source is the epilogue (C stores):** This is a different problem entirely. The epilogue stores FP32->BF16 packed pairs to global memory. If smem is used as a staging area, a separate swizzle may be needed there.

---

## Sources

- [gau-nernst Flash Attention for RTX 5090](https://gau-nernst.github.io/fa-5090/) -- Working swizzle template for sm_120, confirmed bank conflict elimination
- [DeepGEMM Source (GitHub)](https://github.com/deepseek-ai/DeepGEMM) -- 128B swizzle with 1024-byte alignment, FP8 GEMM
- [DeepGEMM Analysis (kingsleykim.dev)](https://kingsleykim.dev/blog/deepgemm/) -- XOR swizzle internals, bank group units
- [Blackwell Shared Memory Banks (NVIDIA Forum)](https://forums.developer.nvidia.com/t/lsu-wavefront-scheduling-and-shared-memory-bank-utilization-on-blackwell/359791) -- Confirmed 32 banks, 4 bytes/bank on Blackwell
- [ldmatrix Behavior (NVIDIA Forum)](https://forums.developer.nvidia.com/t/understanding-the-behaivor-of-ldmatrix-in-terms-of-shared-memory-access/278716) -- Wavefront splitting into 4 phases of 8 threads
- [Flash Attention Part 4 (lubits.ch)](https://lubits.ch/flash/Part-4) -- 2x speedup from swizzle, per-phase conflict analysis
- [CuTeDSL Swizzle Usage (veitner.bearblog.dev)](https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/) -- Swizzle kind selection logic, atom tiling
- [CuTe Swizzle (Lei Mao)](https://leimao.github.io/blog/CuTe-Swizzle/) -- Parameter derivation formulas
- [Understanding CuTe Swizzling (veitner.bearblog.dev)](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) -- 32B/64B/128B pattern details
- [H100 GEMM Worklog (Hamza El Shafie)](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/) -- Padding resolves conflicts, warp tiling effects
- [CUTLASS Permuted Layout (NVIDIA Forum)](https://forums.developer.nvidia.com/t/understanding-cutlass-permuted-shared-memory-layout/303697) -- Store vs load permutation coordination
- [CUTLASS Discussion #1130 (GitHub)](https://github.com/NVIDIA/cutlass/discussions/1130) -- GTC 2020 conflict-free ldmatrix access
- [Shared Memory Microbenchmarks (Axel Feldmann)](https://feldmann.nyc/blog/smem-microbenchmarks) -- A100 bank conflict measurements, vectorized access behavior
- [ThunderKittens FP8 (Hazy Research)](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8) -- FP8 tile width requirements, 128B swizzle mode
- [Blackwell Tuning Guide (NVIDIA)](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) -- sm_120: 128KB smem/SM, 99KB/block
- [SW128 Swizzle Discussion (NVIDIA Forum)](https://forums.developer.nvidia.com/t/why-does-sw128-swizzle-3-4-3-produce-identical-bank-patterns-across-all-8-rows/360775) -- Correct smem_ptr_flag inclusion for swizzle
