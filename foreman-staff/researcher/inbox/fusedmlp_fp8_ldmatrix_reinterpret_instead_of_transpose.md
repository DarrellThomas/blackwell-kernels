# FP8 B Loading: ldmatrix.b16 Reinterpret Instead of Column-Major Transpose

**Source:** CUTLASS mma_traits_sm120.hpp, gau-nernst NVRTC matmul blog, attention worker exp 66
**Relevant to:** fused-mlp worker
**Worker's current problem:** Native FP8 B with row-major smem is 4% slower than ldmatrix+CVT due to scalar byte loads with bank conflicts. Worker proposes column-major transpose in smem (2 extra syncs + temp buffer). There's a better approach.

## What This Is

Instead of transposing FP8 B from row-major to column-major in smem (which costs
2 extra __syncthreads and a temporary buffer), the worker should store FP8 data
directly in smem and use `ldmatrix.b16` to load it, reinterpreting 16-bit loads
as pairs of FP8 bytes. This is the standard technique used by every INT8/FP8 GEMM
since Ampere — CUTLASS confirms the fragment layout is identical.

## Why It Matters for Us

Your proposed column-major transpose approach has costs:
1. Extra smem region for temp buffer (row-major → column-major copy)
2. Two extra __syncthreads per K-tile (between cp.async, transpose, and compute)
3. Register-mediated transpose adds ~64 smem ops per thread per K-tile
4. Can't overlap prefetch with transpose (breaks double-buffer pipeline)

The ldmatrix.b16 approach has NONE of these costs — it uses the existing smem
layout and existing ldmatrix infrastructure, just with byte-level addressing.

## Key Technique

**The insight:** `ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16` loads 16-bit
values from smem. When smem contains FP8 (1-byte) data, each 16-bit load picks
up 2 adjacent FP8 bytes. If the FP8 data is laid out correctly, these pairs are
exactly the `{B[k, n], B[k+1, n]}` needed for the m16n8k32 B fragment.

**What "laid out correctly" means:**
- FP8 B stored in smem with K as the fast-varying dimension (column-major for B^T)
- Consecutive k-values for the same n-column are adjacent in memory
- B_smem[k, n] at address `base + n * K_stride + k` (k-major)

**Then ldmatrix.b16 on this layout:**
- Loads 16-bit value from address A → gets `{fp8[k], fp8[k+1]}`
- ldmatrix distributes across threads following the standard m8n8 pattern
- Result: uint32 registers contain `{B[k,n], B[k+1,n], B[k+16,n], B[k+17,n]}`
- This IS the exact interleaved layout you empirically verified!

**Implementation change is minimal:**
```cuda
// CURRENT (BF16): ldmatrix loads 16-bit BF16 values
// B smem layout: B_smem[n][k] where each element is 2 bytes (bf16)

// NEW (FP8): ldmatrix loads 16-bit values = 2 FP8 bytes each
// B smem layout: B_smem[n][k] where each element is 1 byte (fp8)
// ldmatrix address calculation: adjust for sizeof(fp8)=1 instead of sizeof(bf16)=2
// Same ldmatrix_x2_trans call, same register output format
```

**XOR swizzle:** Apply the same swizzle pattern but with byte-level addressing.
The swizzle index is computed from the address, so just make sure the base address
and stride account for 1-byte elements instead of 2-byte.

## Evidence This Works

1. **CUTLASS confirms:** SM120 FP8 MMA trait inherits SM80 INT8 fragment layout.
   INT8 m16n8k32 has used ldmatrix.b16 since Ampere (4+ years, production-proven).

2. **gau-nernst achieves 692 TFLOPS** FP8 on RTX 5090 using identical ldmatrix code
   for both BF16 and FP8 — only sizeof(TypeAB) changes in address calculations.

3. **Your own verified mapping** `{B[k, n], B[k+1, n], B[k+16, n], B[k+17, n]}`
   is exactly what ldmatrix.b16 produces when loading from column-major (k-major)
   FP8 smem. The k+16 stride comes from ldmatrix's cross-warp distribution.

## Concrete Recommendation

1. **Don't implement the transpose.** It adds complexity and overhead.

2. **Instead:** When loading FP8 B weights from global memory:
   - If weights are already stored column-major (k-major): cp.async directly
   - If weights are row-major: transpose ONCE at model load time (host-side),
     not per-forward-pass in the kernel

3. **For GEMM1 in fused-mlp:** The A matrix (input X) is row-major and needs
   BF16→FP8 conversion. This still uses the existing CVT path (convert in registers
   after ldmatrix for A). The B matrix (weight W1) can be pre-transposed to k-major
   FP8 and loaded via ldmatrix.b16 directly.

4. **Smem savings remain:** FP8 B in smem is half the bytes of BF16 B, regardless
   of whether you use transpose or ldmatrix.b16. The occupancy benefit is preserved.

## Caveats

1. **Test with a minimal kernel first.** Load known FP8 data via ldmatrix.b16,
   print per-thread register contents, verify against the mapping you already have.
   This should take <30 minutes.

2. **cp.async for FP8:** The cp.async instruction copies raw bytes, so it works
   for FP8 data without modification. Just ensure the smem allocation is aligned
   and the copy size is correct (half the bytes of BF16 for same tile).

3. **The existing ldmatrix+CVT path** (load BF16 via ldmatrix, convert to FP8
   in registers) is already working. The ldmatrix.b16-direct path eliminates
   the CVT step entirely — ~256 cycles saved per K-tile.
