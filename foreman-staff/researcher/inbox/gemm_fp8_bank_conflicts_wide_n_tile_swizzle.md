# Bank Conflict Elimination for FP8 GEMM B Operand with N=128 Tiles

**Source:** Multiple (see Sources section below)
**Relevant to:** GEMM worker
**Worker's current problem:** 86K bank conflicts from B loads with N=128 stride; current 3-bit XOR swizzle does not fully eliminate conflicts for wide B tile.

## What This Is

A synthesis of research on shared memory swizzle patterns, ldmatrix access mechanics, and bank conflict elimination techniques specifically relevant to the GEMM worker's FP8 kernel with BLOCK_N=128.

---

## 1. Root Cause Analysis: Why the Current Swizzle Leaves 86K Conflicts

The current `swizzle_idx<128>` computes:

```
NUM_CHUNKS = 128 / 8 = 16
SWIZZLE_BITS = 3  (since 16 >= 8)
SWIZZLE_MASK = 7
swizzled_col = col ^ ((row & 7) << 3)
```

This XORs bits [5:3] of the column index with bits [2:0] of the row index. The swizzle pattern repeats every 8 rows.

**The problem:** With BLOCK_N=128 (16 chunks of 8 BF16 elements), the column index has 4 significant chunk bits [6:3], but only 3 bits [5:3] are being swizzled. Chunk bit [6] (the highest chunk bit, distinguishing the left half vs right half of the 128-wide row) is never touched by the XOR. This means addresses in the left half (columns 0-63) and right half (columns 64-127) of the same row map to the same physical bank pattern.

When `ldmatrix_x2_trans` loads B fragments, threads within a group access addresses that span the full N=128 width. Two accesses that differ only in bit [6] of the column will hit the same bank -- the swizzle cannot disambiguate them.

**Contrast with BLOCK_K=64 (A operand):** `swizzle_idx<64>` has `NUM_CHUNKS = 64/8 = 8`, so 3 XOR bits fully cover all 3 chunk bits [5:3]. This is why A loads show zero bank conflicts -- the swizzle period matches the row width exactly.

## 2. The Fix: Match Swizzle Period to Row Width

The CuTe/CUTLASS framework defines swizzle configurations as `Swizzle<B, M, S>` where:
- **B** (BBits): number of bits in the XOR mask
- **M** (MBase): number of least-significant bits to keep constant
- **S** (SShift): shift distance between source bits and target bits

The formula is: `result = offset ^ ((offset >> S) & mask)` where `mask = ((1 << B) - 1) << M`.

For BF16 elements (2 bytes each) with 128-bit (16-byte) vector access:
- 8 BF16 elements per 128-bit vector, so MBase = 3 (preserve 3 bits = 8-element alignment)
- Row width = 128 elements = 16 chunks, requiring 4 bits to address all chunks
- BBits should be **4** (not 3) to fully cover the column space
- SShift = log2(128) - MBase = 7 - 3 = 4

This gives `Swizzle<4, 3, 4>`: XOR bits [6:3] of the column with bits [10:7] of the address (which encode the row).

**Translated to the current codebase:**

```c
// Current (3-bit, leaves bit [6] unswizzled):
constexpr int SWIZZLE_BITS = 3;
constexpr int SWIZZLE_MASK = 7;
int swizzled_col = col ^ ((row & SWIZZLE_MASK) << 3);

// Proposed (4-bit, covers all chunk bits for N=128):
constexpr int SWIZZLE_BITS = 4;
constexpr int SWIZZLE_MASK = 15;
int swizzled_col = col ^ ((row & SWIZZLE_MASK) << 3);
```

The swizzle now repeats every 16 rows (instead of 8). With BLOCK_K=64, there are 64 rows in B's shared memory tile. 64 / 16 = 4 full swizzle periods -- clean coverage.

**Critical constraint:** The swizzle_idx function currently caps at 3 bits (`NUM_CHUNKS >= 8 => 3`). For N=128 (NUM_CHUNKS=16), the formula should use 4 bits. The general rule: `SWIZZLE_BITS = ceil(log2(NUM_CHUNKS))` to fully cover the column address space.

## 3. The ldmatrix_x2_trans Access Pattern (Why It Creates Conflicts)

`ldmatrix.sync.aligned.x2.m8n8.shared.b16` with `.trans` loads two 8x8 matrices in transposed order. The warp's 32 threads are split into 4 groups of 8 consecutive threads. Within each group of 8 threads:

- Each thread provides a shared memory address pointing to 8 consecutive 16-bit elements (16 bytes)
- The hardware performs the load as a 128-byte transaction per group (8 threads x 16 bytes)
- With `.trans`, the thread-to-value mapping is transposed: threads that would normally get row data instead get column data

For B operand loaded column-major (K rows x N columns in shared memory):
- Thread group accesses 8 rows at stride = BLOCK_N elements = 128 x 2 bytes = 256 bytes per row step
- Row stride of 256 bytes means consecutive rows land on banks: bank = (base + n*256) / 4 mod 32
- 256 / 4 = 64 = 2 * 32 -- so every row step advances by exactly 2 full bank cycles, landing back on the **same bank**
- Without swizzle, this is a guaranteed 8-way bank conflict

The 3-bit XOR swizzle resolves this partially: it ensures rows 0-7 access different banks for bits [5:3], but rows 8-15 repeat the pattern identically. If any ldmatrix group spans rows from two different 8-row periods with matching low bits, conflicts remain.

With a 4-bit XOR, the pattern repeats every 16 rows, which is sufficient for the 16-row groups that ldmatrix_x2_trans accesses within a K tile.

## 4. Spatters.ca Reference Implementation (Ada, mma.sync)

The spatters.ca GEMM (already in worker's docs as `reference_spatters_mma_matmul.md`) uses a different swizzle approach for a 64x64 output tile with N=64:

```c
// Store: (laneID % 8) ^ (laneID / 8)  -- 3-bit XOR
// Load B: (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4)  -- 2-bit XOR
```

This works for N=64 because 8 chunks fully covered by 3 bits. For N=128, one needs to extend the approach. Spatters does not demonstrate N=128 tiles.

**Key insight from spatters:** The store-side and load-side XOR patterns are different because cp.async and ldmatrix have different thread-to-address mappings. Both must be consistent (store swizzle must match what ldmatrix expects after the swizzle is applied).

## 5. CUTLASS Permuted Layout for B Operand

CUTLASS implements the swizzled B layout in `include/cutlass/layout/tensor_op_multiplicand_sm75.h`. The key concepts:

- **Crosswise layout**: K is the contiguous dimension (matching B's column-major storage where K varies fastest)
- The XOR permutation formula: `store_column = (lane_id % 8) ^ (lane_id / 8)`
- For tiles wider than 64 columns, CUTLASS uses the 128B swizzle mode (`Swizzle<3,4,3>` in CuTe notation) which handles up to 128-byte row widths

For BF16 with N=128: row width = 128 x 2 = 256 bytes. This exceeds the 128B swizzle atom. CUTLASS handles this by treating the 256-byte row as **two 128-byte swizzle atoms** side by side. Each atom is swizzled independently. This is equivalent to using a 4-bit XOR mask (our proposed fix).

## 6. Alternative: Padding Instead of Swizzle Fix

Padding adds extra elements per row to break the bank alignment:

```c
// Current: B_smem[BLOCK_K][BLOCK_N] -- 64 x 128 BF16 = 256 bytes/row
// Padded:  B_smem[BLOCK_K][BLOCK_N + 8] -- 64 x 136 BF16 = 272 bytes/row
```

Padding by 8 BF16 elements (16 bytes = 4 banks) shifts the bank alignment by 4 per row. After 8 rows, all 32 banks are hit. This eliminates conflicts completely.

**Trade-offs:**
- Padding wastes shared memory: 64 x 8 x 2 = 1024 bytes per buffer, x2 for double-buffer = 2 KB
- Current B smem: 64 x 128 x 2 = 16 KB per buffer; padded: 64 x 136 x 2 = 17.4 KB
- Total smem increase: ~2.8 KB (from 48 KB to ~50.8 KB) -- still under 99 KB limit
- Padding changes all address calculations (cp.async store offsets, ldmatrix load offsets)
- Padding is simpler to implement but less elegant than fixing the swizzle

**Performance expectation:** The salykova SGEMM blog showed padding and swizzle achieve identical performance for bank conflict elimination (~20% faster than conflicted access).

## 7. FP8-Specific Considerations

With FP8 (1-byte elements), the bank conflict arithmetic changes:
- 32 banks x 4 bytes/bank = 128 bytes per bank cycle
- One FP8 element = 1 byte, so 4 elements per bank
- A row of 128 FP8 elements = 128 bytes = exactly one bank cycle

However, the worker's kernel stores BF16 in shared memory and converts to FP8 in registers. So the bank conflict analysis is based on BF16 (2-byte) elements, not FP8.

If the kernel were to store FP8 directly in shared memory (future optimization path with native FP8 inputs):
- Row of 128 FP8 elements = 128 bytes = one bank cycle -- trivially conflict-free for row access
- Column access with stride 128 bytes = 4 full bank cycles per step -- same bank every time
- Would still need swizzle, but with different parameters (MBase changes for 1-byte elements)

**Note on ldmatrix with FP8:** PTX ISA ldmatrix does not support 8-bit element loads directly. The forum discussion confirms that for FP8, you either: (a) load as 16-bit and reinterpret, or (b) use direct element loads and shuffle manually. The current approach of loading BF16 and converting is actually the cleanest path on sm_120.

## 8. ThunderKittens FP8 GEMM: Tile Width Insights

The ThunderKittens framework (Stanford Hazy Research) found that for FP8, input tile widths must be doubled vs BF16 to maintain bank-conflict-free access. Their FP8 GEMM achieves 1500 TFLOPS using 128B swizzle mode with minimum tile width of 32 elements (vs 16 for BF16).

Key finding: "Hardware utilization increases with tile width/swizzle" -- wider tiles with matching swizzle modes achieve higher utilization.

## 9. Complete PTX GEMM References (for Full Inner Loop Path)

If the worker considers the full PTX inner loop (~500 lines, listed as a remaining path):

**LeetCUDA / CUDA-Learn-Notes:** Contains multiple HGEMM implementations with mma.sync m16n8k16, including:
- Basic MMA kernel with padding for bank conflict avoidance (A_PAD=8, B_PAD=8)
- Swizzle variant (`hgemm_mma_stage_tn_swizzle_x4.cu`) that uses CuTe-style swizzle
- Claims 98-100% of cuBLAS on RTX 4090

**salykova SGEMM:** FP32 GEMM with complete PTX inner loop structure:
- Double-buffered shared memory with XOR-based buffer switching: `lds_a_addr ^= 8192`
- A operand padded to leading dim 132 (128 + 4 floats) for bank conflict avoidance
- B operand naturally conflict-free due to 32-thread row access
- Register double-buffering for load/compute overlap

**spatters/mma-matmul:** FP16 GEMM with:
- XOR-permuted shared memory (zero bank conflicts verified via ncu)
- 3-stage cp.async pipeline with ldmatrix_x4/x2
- 4x4 tiling reaching 93% of RTX 4090 peak
- MIT licensed, 3 files total -- the most readable reference

None of these have BF16 or FP8 variants. A BF16 port of spatters' kernel_3.cu would be the closest starting point for a full PTX GEMM inner loop on sm_120.

## Recommendation

**Immediate action (low effort, likely fixes most of the 86K conflicts):** Change `swizzle_idx` to use 4-bit XOR for `NUM_CHUNKS >= 16`:

```c
constexpr int SWIZZLE_BITS = (NUM_CHUNKS >= 16) ? 4 :
                              (NUM_CHUNKS >= 8) ? 3 :
                              (NUM_CHUNKS >= 4) ? 2 : 1;
```

This is a one-line change. The worker should verify with ncu that bank conflicts drop to near-zero.

**If swizzle fix is insufficient:** Padding B smem by 8 elements per row as a fallback. Costs ~2.8 KB extra shared memory but guaranteed to eliminate all conflicts.

**Performance estimate:** Eliminating 86K bank conflicts should reduce long_scoreboard stalls (currently 17%) by a significant fraction, potentially yielding a 3-8% speedup. The exact impact depends on how much the bank conflicts contribute to the overall long_scoreboard vs global memory latency.

---

## Sources

- [CuTe Swizzle Math (Lei Mao)](https://leimao.github.io/blog/CuTe-Swizzle/) -- Swizzle<B,M,S> parameter derivation
- [Understanding CuTe Swizzling - 32B, 64B, 128B Patterns (Simon Veitner)](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) -- Swizzle<3,4,3> for 128B patterns
- [CUDA Shared Memory Swizzling (Lei Mao)](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/) -- XOR swizzle formula and code
- [Flash Attention Part 4: Bank Conflicts & Swizzling (lubits.ch)](https://lubits.ch/flash/Part-4) -- ldmatrix bank conflict mechanics, swizzle formula derivation
- [Tensor Core MMA Swizzle Layout (Yifan Yang)](https://yang-yifan.github.io/blogs/mma_swizzle/mma_swizzle.html) -- Legal swizzle layouts, atom sizes, and how ldmatrix interacts with swizzled smem
- [CUTLASS Permuted Shared Memory (NVIDIA Forums)](https://forums.developer.nvidia.com/t/understanding-cutlass-permuted-shared-memory-layout/303697) -- Store/load permutation XOR formulas
- [CUTLASS XOR Layout Implementation (GitHub Discussion #510)](https://github.com/NVIDIA/cutlass/discussions/510) -- Source file locations: tensor_op_multiplicand_sm75.h
- [GTC 2020 CUTLASS Talk (GitHub Discussion #1130)](https://github.com/NVIDIA/cutlass/discussions/1130) -- Slides 45-48 on bank-conflict-free ldmatrix loads
- [spatters.ca MMA MatMul (Ada)](https://www.spatters.ca/mma-matmul) -- Complete mma.sync GEMM with XOR-permuted smem, 93% peak
- [salykova SGEMM](https://salykova.github.io/sgemm-gpu) -- Full PTX inner loop, padding-based bank conflict avoidance
- [LeetCUDA HGEMM (GitHub)](https://github.com/xlite-dev/LeetCUDA) -- Multiple HGEMM variants with swizzle, 98-100% cuBLAS
- [ThunderKittens FP8 (Hazy Research)](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8) -- FP8 tile width constraints, 128B swizzle for FP8
- [ldmatrix FP8 on sm120 (NVIDIA Forums)](https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254) -- ldmatrix does not support 8-bit directly
- [Bank Conflicts in Shared Memory (Ian Barber)](https://ianbarber.blog/2025/03/29/bank-conflicts-in-shared-memory/) -- Swizzle vs padding comparison
- [GEMM Bank Conflict Free Access (NVIDIA Forums)](https://forums.developer.nvidia.com/t/gemm-optimization-achieving-coalesced-and-bank-conflict-free-shared-memory-access/319329) -- B operand natural conflict-freedom analysis
- [H100 GEMM Optimization Worklog (Hamza El Shafie)](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/) -- Padding strategy, warp tiling for conflict reduction
