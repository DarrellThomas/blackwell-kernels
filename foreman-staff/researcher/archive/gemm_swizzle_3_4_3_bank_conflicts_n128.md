# Swizzle<3,4,3> Pattern for N=128 Bank Conflict Elimination

**Source:** https://leimao.github.io/blog/CuTe-Swizzle/ | https://forums.developer.nvidia.com/t/why-does-sw128-swizzle-3-4-3-produce-identical-bank-patterns-across-all-8-rows/360775 | https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/
**Relevant to:** GEMM worker (gemm/)
**Worker's current problem:** FP8 GEMM 64x128 tile has 86K bank conflicts from B loads with N=128 stride. Profile shows long_scoreboard 17% — bank conflicts during B matrix loads contribute to this.

## What This Is

The CuTe/CUTLASS swizzle system uses three parameters `Swizzle<BBits, MBase, SShift>` to eliminate shared memory bank conflicts via XOR-based address remapping. For the GEMM worker's N=128 tile with BF16 (2-byte elements), the correct configuration is `Swizzle<3,4,3>`.

## Why It Matters for Us

The GEMM worker's FP8 kernel at 64x128 tiles has 86K bank conflicts on B loads. The current XOR swizzle was designed for 64x64 tiles (N=64). With N=128, the B matrix occupies 256 bytes per row (128 BF16 elements x 2 bytes), which requires a different swizzle pattern than the 128-byte rows used in 64x64 tiling.

The bank conflict count directly contributes to the long_scoreboard stalls (17%) and limits SM throughput. Correct swizzling for N=128 could reduce or eliminate these conflicts.

## Key Technique

### Swizzle parameters for N=128:

**BF16 (2 bytes/element, 256 bytes/row):**
- MBase = log2(vector_elements) = log2(8) = 3 (128-bit vector / 16-bit element = 8 elements)
  - Actually for ldmatrix: MBase = 4 (128-bit access = 16 bytes, log2(16) = 4)
- BBits = log2(128_bytes_per_bank_cycle) - MBase = 7 - 4 = 3
- SShift = log2(row_bytes) - MBase ... but with 256 bytes/row: log2(256) - 4 = 4
- **Configuration: Swizzle<3,4,4>** for 256-byte rows, or **Swizzle<3,4,3>** for 128-byte sub-rows

**FP8 (1 byte/element, 128 bytes/row):**
- MBase = log2(16) = 4 (128-bit vector / 8-bit element = 16 elements)
- BBits = 7 - 4 = 3
- SShift = log2(128) - 4 = 3
- **Configuration: Swizzle<3,4,3>**

### XOR formula:
```
swizzled_col = col ^ (row_within_swizzle_group << SShift)
// Bits [MBase+SShift-1 : MBase] are XORed with bits [MBase+BBits-1 : MBase] shifted
```

### Implementation pattern:
```cpp
// For N=128 with FP8 (128 bytes/row):
// swizzle_group = 8 rows (2^BBits = 8)
// Within each group of 8 rows, XOR bits 4-6 of column index with row index bits 0-2
int swizzled_offset = offset ^ ((row & 0x7) << 4);  // XOR bits 4-6
```

### Important subtlety (from NVIDIA forum):
The SW128 swizzle (Swizzle<3,4,3>) XORs bits 4-6 using only the column position. Row indices affect bit 7+ only. This means the swizzle repeats every 8 rows — which is correct for ldmatrix (which accesses 8 rows at a time).

## Caveats

- **The worker's current XOR swizzle may already be Swizzle<3,4,3>** for the 64x64 case (where N=64 means 128 bytes/row with BF16). If so, the same swizzle may not fully cover N=128 (256 bytes/row with BF16). Need to verify the actual swizzle bits used.
- **ldmatrix accesses 8 rows of 16 bytes each.** With N=128 BF16, each row is 256 bytes = 16 ldmatrix-width chunks. The swizzle must be correct for each 128-byte sub-section.
- **For FP8 data:** N=128 with FP8 = 128 bytes/row, which is exactly one swizzle period. `Swizzle<3,4,3>` should work directly.
- **The 86K bank conflicts may have multiple sources.** Some may come from ldmatrix_x2_trans access patterns for B, not just raw shared memory layout. Profile with `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared` to isolate store vs load conflicts.
- **Changing the swizzle pattern requires updating ALL shared memory indexing** — both store (cp.async destination) and load (ldmatrix source) addresses must use the same swizzle.
