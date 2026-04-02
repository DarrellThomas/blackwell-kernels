# Bank Conflict Reduction for Wide N=128 Tiles

**Source:** CUTLASS documentation + CuTe swizzle analysis (https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) + Lei Mao swizzle blog (https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/) + CUTLASS discussions
**Relevant to:** GEMM worker (FP8 kernel)
**Worker's current problem:** 86K bank conflicts from B loads with 64×128 tile. Current XOR swizzle doesn't fully eliminate conflicts for N=128 stride.

## What This Is

Analysis of why the current XOR swizzle doesn't eliminate bank conflicts for the
wider 64×128 FP8 GEMM tile, and CUTLASS's canonical swizzle patterns that solve this.

## Why It Matters for Us

The FP8 GEMM at 1.29x cuBLAS has a balanced profile (math_throttle 27%, long_scoreboard
17%). The 86K bank conflicts (from B loads) are one of the remaining optimization
vectors. Eliminating them could push toward ~1.35x by reducing shared memory latency.

## Key Technique: CUTLASS Swizzle Patterns

### The Problem

With N=128 BF16 (256 bytes per row), the row stride is exactly 2× the bank pitch
(32 banks × 4 bytes = 128 bytes). This creates systematic bank conflicts:

```
Bank = (byte_address / 4) % 32

Row 0, col c: bank = (c * 2 / 4) % 32 = (c / 2) % 32
Row 1, col c: bank = (256/4 + c/2) % 32 = (64 + c/2) % 32 = (c/2) % 32  ← SAME BANK!
```

When ldmatrix_x2_trans loads column-major data, threads access the same column
across different rows → same bank → conflict.

For FP8 with N=128 (128 bytes per row = exact bank pitch):
```
Row 0, col c: bank = (c / 4) % 32
Row 1, col c: bank = (128/4 + c/4) % 32 = (32 + c/4) % 32 = (c/4) % 32  ← SAME!
```

The pattern is even worse for FP8 because the row stride equals the bank pitch exactly.

### CUTLASS Canonical Swizzle Patterns

CUTLASS defines swizzles as `Swizzle<BBits, MBase, SShift>`:

```
bit_msk = (1 << BBits) - 1
yyy_msk = bit_msk << (MBase + max(0, SShift))
zzz_msk = bit_msk << (MBase - min(0, SShift))

swizzled_offset = offset ^ ((offset & yyy_msk) >> SShift)
```

The canonical patterns (all with MBase=4, SShift=3):

| Pattern | BBits | Bytes | Effect | Use case |
|---------|-------|-------|--------|----------|
| `Swizzle<0,4,3>` | 0 | — | No swizzle | — |
| `Swizzle<1,4,3>` | 1 | 32B | XOR bit 7 with bit 4 | Narrow tiles |
| `Swizzle<2,4,3>` | 2 | 64B | XOR bits 7-8 with bits 4-5 | Medium tiles |
| `Swizzle<3,4,3>` | 3 | 128B | XOR bits 7-9 with bits 4-6 | **Wide tiles (N≥128)** |

For N=128 BF16 (256 bytes/row), you need the **128B swizzle** (`Swizzle<3,4,3>`)
to fully decorrelate row addresses from bank indices.

### Applying to Our Kernel

The current XOR swizzle in our kernels uses something like:
```cpp
int swizzled_col = col ^ (row & MASK);
```

For the 128B pattern, the equivalent is:
```cpp
// For 16-bit (BF16) elements, 128 elements per row:
// byte_offset = row * ROW_STRIDE + col * 2
// swizzled = byte_offset ^ ((byte_offset >> 3) & 0x70)
// Which in element coordinates:
int swizzled_col = col ^ ((row * (ROW_STRIDE/2) >> 3) & 0x38);  // 3-bit mask
```

Or equivalently, treating the shared memory as a 2D array:
```cpp
// For B[BLOCK_K][BLOCK_N] in BF16:
// Element (k, n) maps to swizzled address:
int n_swizzled = n ^ ((k & 0x7) << 3);  // XOR lower 3 bits of k into bits 3-5 of n
```

### Verifying Bank Conflict Freedom

After swizzle, verify with ncu:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./kernel
```

Target: < 1K conflicts (from 86K currently).

### Alternative: Padding

Instead of swizzle, add padding to each row to break the stride alignment:
```cpp
__shared__ __align__(128) half B_smem[BLOCK_K][BLOCK_N + PAD];
// PAD = 8 for BF16 (16 bytes) shifts each row by 1 bank
```

**Pros:** Simpler, no address computation overhead
**Cons:** Wastes shared memory (BLOCK_K × PAD × 2 bytes), may reduce occupancy

For BLOCK_K=64, PAD=8: overhead = 64 × 8 × 2 = 1024 bytes = 1 KB. Negligible.

**Try padding first** — it's the simplest fix and may fully resolve the 86K conflicts.

### B Operand Storage Order

The B operand can be stored in two ways:
1. **Row-major B[K][N]:** Standard for GEMM. ldmatrix loads along N dimension.
2. **Column-major B[N][K]:** Better for ldmatrix_x2_trans since it loads K-columns.

If B is stored row-major and loaded with ldmatrix_x2_trans, the transpose happens
at the instruction level. The swizzle must account for the actual access pattern:
- ldmatrix_x2_trans with row-major B: threads access (k, n) where k varies per
  thread within a group of 8 — this is a column access pattern hitting same bank.

Consider whether switching B to column-major storage in smem (transpose during
cp.async or after) could simplify the bank conflict pattern.

## Caveats

1. **The 128B swizzle may interfere with cp.async addressing.** cp.async loads
   128-bit (16-byte) chunks from global to shared. The swizzled smem destination
   addresses must still be 16-byte aligned for cp.async.cg to work.

2. **ldmatrix addressing with swizzle.** ldmatrix requires 128-bit aligned
   addresses. The swizzle must preserve this alignment. The CUTLASS patterns
   (MBase=4 = 16 bytes) are designed to maintain 16-byte alignment.

3. **Interaction with CTA swizzle.** The current kernel uses CTA swizzle=4 for
   L2 reuse of B columns. The smem swizzle is orthogonal to the CTA swizzle
   and they don't interfere.

4. **Test with both BF16 and FP8 paths.** The FP8 path has different element
   sizes (1 byte vs 2 bytes), so the bank conflict pattern differs. May need
   different swizzle parameters.
