# FP8 B Operand: Shared Memory Transpose for Column-Major Fragment Loading

**Source:** Cross-pollinated from attention worker research + CUDA transpose best practices
**Relevant to:** fused-mlp worker (FP8 GEMM1)
**Worker's current problem:** Native FP8 B with row-major smem is 4% slower than ldmatrix+CVT because scalar byte loads at stride-64 cause bank conflicts. Worker proposes column-major transpose in smem.

## What This Is

Analysis of the proposed smem transpose approach for FP8 B loading, plus an alternative
that avoids the transpose entirely by using ldmatrix.b16 + 2 PRMT instructions.

## Alternative to Transpose: ldmatrix.b16 + PRMT

Instead of transposing B to column-major in smem, keep B in its current BF16 format
in smem and use ldmatrix.b16 to load it, then merge the two K-half registers with
PRMT to match the interleaved B-fragment layout.

### How It Works

1. **Keep B as BF16 in smem** (same cp.async path as current BF16 kernel)
2. **ldmatrix_x2_trans** loads B operand into r0, r1 (same as BF16 path)
   - r0 = K-first-half data (FP8 K=0-15 as "16-bit" pairs)
   - r1 = K-second-half data (FP8 K=16-31 as "16-bit" pairs)
3. **2 PRMT instructions** merge bytes to match the interleaved layout:
   ```ptx
   prmt.b32 b0, r0, r1, 0x5410;  // {r0[0], r0[1], r1[0], r1[1]}
   prmt.b32 b1, r0, r1, 0x7632;  // {r0[2], r0[3], r1[2], r1[3]}
   ```
4. **Feed b0, b1 directly to m16n8k32 MMA** — no CVT needed

### Why This Is Better Than the Transpose Approach

| Factor | Transpose approach | ldmatrix+PRMT approach |
|--------|-------------------|------------------------|
| Extra smem | 2× B buffer needed | None (same BF16 buffer) |
| Extra syncs | 2 __syncthreads | None |
| Extra compute | N×K scalar loads+stores | 2 PRMT per B load |
| Double-buffer | Can't overlap prefetch with transpose | Standard double-buffer works |
| Complexity | Medium-high | Low (2 lines of PTX) |
| B data in smem | FP8 (half size → occupancy?) | BF16 (same as now) |

**The only advantage of the transpose approach** is halved B smem (FP8 instead of BF16),
which could increase occupancy. But if occupancy is already at 6 blocks/SM, this doesn't
help.

### With Pre-Quantized FP8 Inputs

If B arrives as FP8 from the host (pre-quantized weights):
1. **cp.async FP8 data** to smem (half the bandwidth)
2. **ldmatrix.b16** loads pairs of FP8 bytes as "16-bit elements"
3. **2 PRMT** to get interleaved layout
4. **MMA** directly — zero conversion

This eliminates ALL CVT instructions AND halves smem for B.

## If You Still Want the Transpose Approach

### Efficient Smem Transpose Pattern

The standard pattern for transposing during smem loading:

```
// Phase 1: Load B row-major from global to smem buffer A via cp.async
cp_async_group_commit();
__syncthreads();

// Phase 2: Transpose from buffer A (row-major) to buffer B (column-major)
// Each thread transposes one element
int src_row = threadIdx.x / BLOCK_N;
int src_col = threadIdx.x % BLOCK_N;
int dst_row = src_col;  // transposed
int dst_col = src_row;
// Use XOR swizzle on destination to avoid bank conflicts
int dst_swizzled = dst_col ^ (dst_row & 0x7);
smem_B_colmajor[dst_row][dst_swizzled] = smem_B_rowmajor[src_row][src_col];
__syncthreads();
```

### Bank-Conflict-Free Column-Major Access

For column-major FP8 B with N=8 (per MMA tile):
- Each column has K=32 bytes = 32 FP8 values
- 32 bytes / 4 bytes per bank = 8 elements per bank
- With K=32, column stride = 32 bytes → adjacent columns start at same bank

**Fix:** Pad each column to 36 bytes (4 bytes padding), or use XOR swizzle on column index.

## Caveats

1. **The ldmatrix+PRMT approach requires that B data in smem has the same layout as
   BF16.** If B is already FP8 in smem (from native FP8 inputs), the ldmatrix.b16
   interpretation of FP8 byte pairs as "16-bit elements" needs careful validation.

2. **PRMT selector values (0x5410, 0x7632) need empirical verification.** The principle
   is correct but the exact byte mapping from ldmatrix_x2_trans output to PRMT input
   needs to be tested with a minimal kernel.

3. **For the fused-mlp kernel specifically,** the A operand (input X) is loaded via
   ldmatrix_x4, and the existing BF16→FP8 conversion only applies to A (not B).
   If B weights are pre-quantized to FP8, the ldmatrix+PRMT approach eliminates
   the conversion entirely for B, and you'd only convert A.
