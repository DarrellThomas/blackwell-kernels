# FP8 Fragment Layout Conformance: byte_perm + shfl_sync Technique

**Source:** https://research.colfax-intl.com/adding-fp8-to-flashattention/
**Relevant to:** attention worker (FP8 kernel)
**Worker's current problem:** FP8 kernel has 14:1 conversion:MMA ratio (~448 ALU/KV block). The worker needs a way to feed FP32 softmax output (P) directly into FP8 MMA without an expensive smem round-trip.

## What This Is

Colfax Research's FP8 FlashAttention-2 implementation on Hopper documented a critical
insight: **FP32 accumulator fragment layout and FP8 operand A fragment layout are
structurally different**, requiring explicit register reorganization. They solved this
with a two-step in-register technique: `__byte_perm` (intra-thread byte rearrangement)
followed by `__shfl_sync` (inter-thread data exchange). This achieved >1 PFLOP/s on H100.

## Why It Matters for Us

Our P->FP8 conversion path (FP32 softmax output -> FP8 A operand for PV MMA) currently
uses `cvt.rn.satfinite.e4m3x2.f32` to downcast, then packs bytes. Even if we eliminate
K and V conversion via the ldmatrix reinterpret technique (see companion brief), the
P->FP8 path remains (~112 instructions/block). The Colfax technique could make this
more efficient by reorganizing the FP32 accumulator registers directly into the FP8
fragment layout expected by m16n8k32 MMA, potentially combining the downcast and
layout permutation.

**Critical caveat:** Colfax uses WGMMA (128-thread warpgroup, Hopper-only), which has
a DIFFERENT fragment layout than mma.sync (32-thread warp, sm_120). The specific
byte_perm selectors and shfl_sync maps are NOT directly portable. But the TECHNIQUE
(in-register reorganization using byte_perm + shfl_sync to match fragment layout) IS
applicable. We need to derive the correct selectors for mma.sync.m16n8k32 fragment layout.

## Key Technique

### The Layout Mismatch Problem

For WGMMA on Hopper (and likely analogous for mma.sync on sm_120):

- **FP32 accumulator (C/D fragment):** Each thread holds FP32 values indexed by
  dimension `d` in a contiguous pattern. The PTX ISA shows this follows a specific
  thread-to-element mapping (Figure 118 in PTX ISA 8.5).

- **FP8 operand A fragment:** The elements are arrayed in 4-element groups packed into
  32-bit registers, with a DIFFERENT thread-to-element mapping (Figure 122 in PTX ISA 8.5).

For BF16 MMA, the accumulator layout and operand A layout happen to be identical, so
P->BF16 conversion is trivial (just downcast in place with `cvt.rn.bf16x2.f32`). For
FP8 MMA, they are NOT identical, requiring data movement between threads.

### The Two-Step Solution

1. **`__byte_perm(a, b, selector)`** - Rearranges bytes within a thread's registers.
   Selects 4 bytes from the concatenation of two 32-bit values.
   - Selector `0x7654` picks the upper 4 bytes
   - Selector `0x3210` picks the lower 4 bytes
   - Custom selectors can extract arbitrary byte permutations

2. **`__shfl_sync(mask, val, srcLane)`** - Moves data between threads in a warp.
   - Uses lane-dependent source maps:
     ```
     upper_map[4] = {0, 3, 1, 2}   // for (threadIdx.x % 4) == {0,1,2,3}
     lower_map[4] = {1, 2, 0, 3}
     ```
   - Each thread sends/receives based on its position within the 4-thread group

### Applied to mma.sync on sm_120

For mma.sync.m16n8k32 on sm_120, the fragment layout is:
- A operand: 4 x uint32, each holding 4 FP8 bytes
- C/D accumulator: 4 x float32

The mapping between accumulator thread positions and operand A thread positions needs
to be derived empirically. The worker should:

1. Write a test that fills FP32 accumulators with known per-thread values
2. Apply the identity conversion (FP32 -> FP8 -> feed as A operand -> MMA with identity B)
3. Check which output positions correspond to which input threads
4. Derive the byte_perm selectors and shfl_sync maps from this

### What This Could Replace

Current P->FP8 path per nc-pair (approximate):
```
4x cvt.rn.satfinite.e4m3x2.f32  (downcast 8 FP32 -> 4 FP8 pairs)
2x SHL + PRMT                    (pack into 2 uint32)
total: ~7 instructions per nc-pair, ~112 per KV block
```

Optimized path with layout-aware conversion:
```
4x cvt.rn.satfinite.e4m3x2.f32  (downcast, same)
1-2x __byte_perm                 (rearrange bytes within thread)
1-2x __shfl_sync                 (move bytes between threads)
total: ~6-8 instructions, but potentially FEWER if the downcast
       and byte_perm can be fused
```

The savings may be modest for P conversion alone. The bigger win is that this
technique validates the approach of using in-register layout manipulation rather
than smem round-trips -- confirming that FP8 fragment layout differences can be
handled efficiently.

## Caveats

1. **WGMMA vs mma.sync fragment layouts are different.** The Colfax byte_perm
   selectors and shfl_sync maps are for 128-thread warpgroups, not 32-thread warps.
   Direct copy-paste will produce wrong results. Must derive sm_120 maps empirically.

2. **The P->FP8 conversion is NOT the main bottleneck.** At ~112 instructions/block
   vs ~448 for K+V conversion, P conversion is 25% of the total overhead. The
   ldmatrix reinterpret technique (eliminating K+V conversion) is the bigger win.

3. **Register pressure.** The shfl_sync approach requires temporary registers for
   the shuffled values. With only 5 spare registers (165/170.7 threshold), this
   needs careful management.

4. **Colfax achieved >1 PFLOP/s with this.** Their implementation validates that
   the FP32->FP8 layout conformance overhead is negligible when amortized over
   the MMA throughput gain.
