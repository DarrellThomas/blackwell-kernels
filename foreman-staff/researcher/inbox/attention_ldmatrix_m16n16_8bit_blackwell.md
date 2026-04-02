# ldmatrix.m16n16 for 8-bit Elements on Blackwell

**Source:** https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (PTX ISA 9.2)
**Also:** https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
**Also:** https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
**Relevant to:** attention worker (FP8 kernel)
**Worker's current problem:** ldmatrix.m8n8.x4.b16 is the only efficient warp-collective smem load on older architectures. Need to know if sm_120 has a native 8-bit ldmatrix variant.

## What This Is

PTX ISA 9.2 introduces `ldmatrix.m16n16` shape that supports 8-bit, 6-bit, and 4-bit
element sizes. This is a NEW ldmatrix shape available on sm_100+ (Blackwell), which
includes sm_120 (RTX 5090). It directly loads 16x16 matrices of 8-bit elements,
eliminating the need for the "load as b16, reinterpret as 2xFP8" workaround.

## Why It Matters for Us

If ldmatrix.m16n16.b8 works on sm_120, it provides a NATIVE path for loading FP8
data from shared memory into MMA-ready registers, without any reinterpretation
tricks. This would be cleaner and potentially faster than the ldmatrix.m8n8.x4.b16
reinterpret approach.

## Key Technique

### New ldmatrix Shapes (PTX ISA 9.2)

| Shape | Element size | Matrix dimensions | Availability |
|-------|-------------|-------------------|--------------|
| .m8n8 | 16-bit (.b16) | 8x8 of 16-bit | sm_75+ (all) |
| .m16n16 | 8-bit (.b8x16) | 16x16 of 8-bit | sm_100+ (Blackwell) |
| .m16n16 | 6-bit (.b6x16_p32) | 16x16 of 6-bit | sm_100+ |
| .m8n16 | 6-bit, 4-bit | 8x16 | sm_100+ |

The .m16n16.b8x16 shape loads a 16x16 matrix of 8-bit elements, which is exactly
what's needed for FP8 fragment loading. A 16x16 of 8-bit = 256 bytes = 64 bytes per
8-thread group, matching the ldmatrix warp-collective pattern.

### CUTLASS Support

CUTLASS has added corresponding copy atoms:
- `LdMatrix16x16x8bOp` - loads 16x16 of 8-bit elements
- `StMatrix16x8x8bOp` - stores 16x8 of 8-bit elements
- `LdMatrix16x8x8bOp` - permuted variant for specific layouts

These now require explicit `transpose=True` when calling init, with copy traits
updated to be faithful to PTX without implicit permutations.

### How This Maps to m16n8k32 MMA

The m16n8k32 FP8 MMA A operand is a 16x32 matrix of 8-bit elements. This can be
loaded as two ldmatrix.m16n16 calls:
- First: 16x16 of FP8 (K columns 0-15)
- Second: 16x16 of FP8 (K columns 16-31)

Alternatively, the .m8n8.x4.b16 reinterpret approach loads the same data in one
call by treating pairs of FP8 as 16-bit elements.

### Architecture Verification Needed

The PTX ISA says .m16n16 shapes are "currently only available on sm_100 and higher."
sm_120 IS sm_100+, but sm_120 is consumer Blackwell (different from datacenter sm_100).
**Must verify empirically that ldmatrix.m16n16.b8x16 compiles and works on sm_120.**

The NVIDIA forum thread (linked above) confirms that on sm_120a a user was asking
about this exact problem but was told they could "also directly load the elements"
or "use [ldmatrix] and then reshuffle." The reply didn't explicitly confirm .m16n16
support on sm_120, which is concerning.

## Implementation Steps

1. **Test ldmatrix.m16n16.b8x16 on sm_120:**
   ```
   asm volatile(
       "ldmatrix.sync.aligned.m16n16.shared.b8x16 {%0}, [%1];"
       : "=r"(r0) : "r"(smem_addr));
   ```
   If this compiles and produces correct output on sm_120, use it.

2. **If .m16n16 is NOT available on sm_120:**
   Fall back to the ldmatrix.m8n8.x4.b16 reinterpret approach (existing brief).
   This is the safer path and is guaranteed to work.

3. **If .m16n16 IS available:**
   Determine register output format (how many registers, element mapping).
   May need 2 calls to cover the 16x32 A operand, vs 1 call with .m8n8.x4.b16.
   Benchmark both to determine which is faster.

## Caveats

1. **sm_120 vs sm_100 feature parity is uncertain.** Many sm_100 features (tcgen05,
   TMEM, TMA) are NOT available on sm_120. The .m16n16 ldmatrix shape may fall into
   this category. Must test before building on this.

2. **Even if available, .m8n8.x4.b16 reinterpret may be equivalent.** The .m16n16
   shape loads the same amount of data (256 bytes per warp). The difference is in how
   registers are organized. If .m8n8.x4.b16 already puts bytes in the right positions
   for m16n8k32 MMA (which the structural analysis suggests), the native 8-bit variant
   may offer no practical advantage.

3. **The .m16n16 shape loads a SQUARE matrix (16x16).** Our m16n8k32 MMA needs a
   16x32 A matrix. Two .m16n16 loads are needed, vs one .m8n8.x4.b16 load that covers
   8x16 of "16-bit pairs" = 8x32 of FP8. The x4 variant loads 4 sub-tiles at once,
   which may be more efficient than two separate .m16n16 calls.

4. **Recommendation: Try .m16n16 for curiosity, but the .m8n8.x4.b16 reinterpret
   approach is the primary path.** It works on sm_75+ (proven), needs no new
   instructions, and the structural analysis is sound.
