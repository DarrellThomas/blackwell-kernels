# Bank Conflict Elimination: Warp-Tile-Level Swizzle and Alternative B Layout

**Source:** Multiple (see Sources section)
**Relevant to:** GEMM worker
**Worker's current problem:** 86K bank conflicts from B loads with N=128 stride. Experiment 113 (4-bit XOR swizzle) already failed. Need a fundamentally different approach.

---

## Why Experiment 113 Failed: The Block-Level vs Warp-Tile-Level Swizzle Distinction

The existing briefs proposed extending the XOR swizzle from 3 bits to 4 bits, and the worker tried it (experiment 113: "4-bit XOR swizzle for N=128 -- bank conflicts unchanged"). The most likely reason this failed is a subtle but critical distinction identified in Alex Armbruster's tensor core GEMM tutorial:

**The swizzle must spread MMA tile rows across the ENTIRE warp tile width, not just within the MMA tile itself.**

The bank conflict arises because ldmatrix loads an 8x8 MMA tile where all 8 rows span just 4 banks (each row of 8 BF16 = 16 bytes = 4 banks). Eight rows at stride BLOCK_N=128 elements (256 bytes = 8 bank cycles) all land on the same 4 banks. The XOR swizzle (whether 3-bit or 4-bit) shuffles columns within each MMA tile, but if the shuffled columns still land on the same physical bank group, conflicts persist.

The fix from Armbruster: "the 8 rows of each MMA tile must be spread horizontally across the entire warp tile." For a warp tile of WN=64 (warp handles 64 of the 128 N columns), the MMA tile's 8 rows should be smeared across all 64 columns so they span 64*2=128 bytes = all 32 banks, instead of staying within 16 bytes = 4 banks.

### What This Means Concretely

The current approach:
```
// Store: swizzle within the 128-wide row
smem[k][swizzle(n)] = value;
// Load: ldmatrix_x2_trans at swizzle(n_start), reading 8 consecutive k-rows
```

The 4-bit XOR swizzle shuffles which column chunk (0-15) a value lands in, but within each ldmatrix group of 8 threads, all threads still access addresses at the same column offset modulo the bank width. The XOR changes the chunk index but not the sub-chunk alignment.

The Armbruster approach:
```
// Store: interleave MMA tile rows across the WARP tile width
// Row i of MMA tile t is stored at column (t * MMA_N + interleave(i, t))
// where interleave scatters the 8 rows of each tile across different 128-byte bank regions
```

This is a different permutation than XOR swizzle. It interleaves the K-dimension rows of B across the N-dimension columns at a warp-tile granularity. This breaks the pattern where all 8 rows accessed by ldmatrix_x2_trans hit the same bank.

## Technique 1: Byte-Address XOR Instead of Element-Index XOR

gau-nernst's flash attention for RTX 5090 applies the swizzle to physical byte addresses rather than element indices. The key difference from the worker's current approach:

```cpp
// gau-nernst's approach (confirmed bank-conflict-free on sm_120):
template <int STRIDE>
__device__ uint32_t swizzle(uint32_t byte_addr) {
    uint32_t row_idx = (byte_addr / STRIDE) % 8;
    uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return byte_addr ^ (bits_to_xor << 4);
}

// For B with STRIDE = BLOCK_N * sizeof(bf16) = 128 * 2 = 256:
// row_idx = (byte_addr / 256) % 8  -- which K-row (mod 8)
// bits_to_xor = row_idx / (64 / 256) = row_idx / 0 --> clamp via max(1) = row_idx
// return byte_addr ^ (row_idx << 4)
// This XORs bits [6:4] of the byte address with the K-row index
```

**Why this differs from experiment 113:**
- The worker's `swizzle_idx<128>` operates on element column index: `col ^ ((row & MASK) << 3)`
- gau-nernst's operates on byte address: `byte_addr ^ (bits_to_xor << 4)`
- The bit positions being XORed are DIFFERENT. gau-nernst XORs byte address bits [6:4] (which correspond to the 16-byte bank group). The worker's element-index XOR on bits [6:3] maps to byte address bits [7:4] for 2-byte elements -- one bit higher.
- This one-bit offset could mean the worker's swizzle is XORing the wrong bits to affect the bank assignment.

**The critical insight from gau-nernst:** The swizzle must flip bits in the byte address that correspond to the **bank index** (bits [6:2] for 32 banks of 4 bytes each). The relevant bits are [6:4] for 16-byte-granularity ldmatrix access. Operating at element-coordinate level with `<< 3` shifts produces `<< 4` in byte address space for 2-byte elements, which is correct. But the MASK application might be off.

**Concrete debugging step:** Print the actual byte addresses fed to ldmatrix_x2_trans for the B operand, with and without swizzle. Check whether threads 0-7 in a phase actually hit different banks (bank = byte_addr / 4 mod 32). If they don't, the swizzle is targeting the wrong bits.

## Technique 2: Column-Offset XOR for ldmatrix_x2_trans

In gau-nernst's V matrix load (which uses ldmatrix_x2_trans, same as the GEMM worker's B load), the address stepping across columns uses XOR instead of addition:

```cpp
// Instead of:
addr += mma_id_d * MMA_K * sizeof(nv_bfloat16);   // step across columns

// Use:
addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);   // XOR step across columns
```

This works because `swizzle(addr + offset) = swizzle(addr) XOR offset` when the offset is aligned to the swizzle granularity (16 bytes). Pre-computing the swizzled base address once per K-row and then XOR-stepping across columns avoids repeated swizzle function calls in the hot loop.

**Relevance:** If the worker is computing `swizzle(base + offset)` inside the loop, there may be an integer-arithmetic error. Switching to `swizzle(base) ^ offset` eliminates that class of bugs.

## Technique 3: Transpose B in Shared Memory (K-Major Instead of N-Major)

If swizzle fixes continue to fail, consider storing B in K-major order (transposed) in shared memory:

```
Current:   B_smem[K][N]  -- K-rows, N-columns, stride = N
Proposed:  B_smem[N][K]  -- N-rows, K-columns, stride = K
```

With BLOCK_K=64 and BLOCK_N=128:
- Current: stride = 128 elements = 256 bytes (2x bank cycle -- systematic conflict)
- Proposed: stride = 64 elements = 128 bytes (1x bank cycle)

With K-major B and stride=64 BF16=128 bytes:
- Row 0, col c: bank = (c * 2 / 4) % 32 = (c / 2) % 32
- Row 1, col c: bank = (128/4 + c/2) % 32 = (32 + c/2) % 32 = (c/2) % 32

Still the same bank! But now the dimension is K=64, and a standard 3-bit XOR swizzle fully covers 8 chunks of 8 elements = 64 elements. This is the regime where the existing swizzle_idx<64> works perfectly (zero bank conflicts on A operand).

**Trade-off:** The B transpose requires changing from `ldmatrix_x2_trans` to `ldmatrix_x4` (since B is already in the orientation that mma.sync expects after the transpose). This also changes the cp.async store pattern -- instead of storing B row-major as received from global memory, you'd need to transpose during the store.

**Transposing during cp.async is NOT possible** (cp.async is a direct copy). So you'd need either:
1. Store B normally, then transpose in smem using a separate step (wastes time)
2. Store B with permuted thread-to-address mapping during cp.async (threads write to transposed positions)

Option 2 is feasible: each thread computes its transposed destination address during cp.async. The global memory source is still coalesced (threads read consecutive B elements), but the smem destination is at a transposed offset. This is exactly what CUTLASS calls the "crosswise" layout.

## Technique 4: Padding (The Guaranteed Fallback)

If all swizzle approaches fail, padding is guaranteed to work:

```cpp
// Change B smem allocation from:
__shared__ half B_smem[2][BLOCK_K * BLOCK_N];        // 2 buffers * 64 * 128 = 16384 elements

// To:
__shared__ half B_smem[2][BLOCK_K * (BLOCK_N + PAD)]; // 2 buffers * 64 * 136 = 17408 elements

#define B_PAD 8
#define B_STRIDE (BLOCK_N + B_PAD)
// Access: B_smem[buf][k * B_STRIDE + n]
```

With PAD=8 BF16 elements (16 bytes = 4 banks):
- Row stride = 136 * 2 = 272 bytes = 68 banks = 2 full cycles + 4 banks
- Consecutive rows offset by 4 banks, guaranteeing that 8 consecutive rows span all 32 banks
- Zero bank conflicts for any access pattern

**Shared memory cost:**
- Current: 2 * 64 * 128 * 2 = 32768 bytes = 32 KB
- Padded: 2 * 64 * 136 * 2 = 34816 bytes = 34 KB
- Delta: 2 KB extra (well within budget; kernel currently uses 48 KB total smem)

**Code changes needed:**
1. Change B smem allocation size
2. Update all B store addresses: `B_smem[buf][k * B_STRIDE + n]` instead of `B_smem[buf][k * BLOCK_N + n]`
3. Update all B load addresses (ldmatrix_x2_trans pointer calculation)
4. Remove the existing B swizzle (no longer needed with padding)

This is the simplest fix and requires no understanding of swizzle mechanics. The salykova SGEMM uses exactly this approach (pad A by 4 floats = 16 bytes) and achieves bank-conflict-free access.

## Recommendation

Given that experiment 113 (4-bit XOR) already failed:

1. **First, diagnose** which specific instructions cause the 86K conflicts. Run:
   ```bash
   ncu --set full --section SourceCounters --kernel-name fp8_gemm_kernel_64 ./benchmark
   ```
   Check if conflicts are from ldmatrix_x2_trans (B loads), ldmatrix_x4_mma (A loads), cp.async stores, or the epilogue.

2. **If B loads via ldmatrix_x2_trans:** Try switching to gau-nernst's byte-address swizzle (Technique 1) to fix the bit-position mismatch.

3. **If swizzle still fails:** Try padding (Technique 4). It is guaranteed to work, costs 2 KB smem, and requires only 3-4 lines of code changes.

4. **If pursuing maximum performance:** Consider the K-major B transpose (Technique 3), which would let the existing conflict-free swizzle_idx<64> work for B loads. This is a larger change but more principled.

---

## Sources

- [Alex Armbruster: Fast MatMul with Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) -- Warp-tile-level swizzle explanation, MMA row spreading across warp tile width
- [gau-nernst Flash Attention RTX 5090](https://gau-nernst.github.io/fa-5090/) -- Working byte-address swizzle template, XOR column stepping for ldmatrix_x2_trans, confirmed zero bank conflicts on sm_120
- [CuTe Swizzle Math (Lei Mao)](https://leimao.github.io/blog/CuTe-Swizzle/) -- Swizzle<B,M,S> parameter derivation for different element types
- [Understanding CuTe Swizzling 32B/64B/128B (Simon Veitner)](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) -- Swizzle<3,4,3> bit manipulation details
- [CuTeDSL Swizzle Usage (Simon Veitner)](https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/) -- Swizzle selection based on tile shape and data type
- [CUDA Shared Memory Swizzling (Lei Mao)](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/) -- XOR formula and code examples
- [Flash Attention Part 4: Bank Conflicts (lubits.ch)](https://lubits.ch/flash/Part-4) -- Per-phase conflict analysis, 2x speedup from swizzle
- [CUTLASS Discussion #1130 (GitHub)](https://github.com/NVIDIA/cutlass/discussions/1130) -- GTC 2020 conflict-free ldmatrix access, 8x8 XOR block
- [CUTLASS Discussion #510 (GitHub)](https://github.com/NVIDIA/cutlass/discussions/510) -- XOR-permuted layout source files (tensor_op_multiplicand_sm75.h)
- [CUTLASS Permuted Layout (NVIDIA Forum)](https://forums.developer.nvidia.com/t/understanding-cutlass-permuted-shared-memory-layout/303697) -- Store vs load permutation: `store_column = (lane_id % 8) ^ (lane_id / 8)`
- [SW128 Swizzle Discussion (NVIDIA Forum)](https://forums.developer.nvidia.com/t/why-does-sw128-swizzle-3-4-3-produce-identical-bank-patterns-across-all-8-rows/360775) -- 128B swizzle row pattern repetition analysis
- [salykova SGEMM](https://salykova.github.io/sgemm-gpu) -- Padding-based bank conflict avoidance (A_PAD=4 floats)
- [ThunderKittens FP8 (Hazy Research)](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8) -- FP8 128B swizzle mode, minimum tile widths
