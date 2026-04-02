# FP8 P→A Conversion for mma.sync: Simpler Than WGMMA

**Source:** Derived from PTX ISA 9.2 Figures 22-24, CUTLASS mma_traits_sm89.hpp, Colfax FP8 FA2
**Relevant to:** attention worker (FP8 kernel)
**Worker's current problem:** The consolidated ldmatrix strategy (Step 4) requires converting FP32 P accumulator to FP8 A operand for PV MMA. Colfax's `ReorgCFp8toAFp8` uses byte_perm + shfl_sync, but that targets WGMMA (Hopper), not mma.sync (sm_120). The worker needs the mma.sync version.

## What This Is

An analysis showing that the P→FP8 A conversion for mma.sync m16n8k32 may NOT need cross-thread shuffles — unlike the WGMMA case documented by Colfax. The mma.sync register layouts are simpler because there's no TMEM/register file boundary.

## Why It Matters for Us

The worker's agent_state.md lists "FP8 native inputs via ldmatrix reinterpret" as direction #1. Step 4 of the consolidated strategy requires P→FP8 conversion. If this conversion needs shuffles (like WGMMA), it adds ~8-12 instructions per conversion block. If it works WITHOUT shuffles (just byte_perm packing), it's ~4 instructions. This difference matters because register pressure is already at 165/170.

## Key Analysis

### Why WGMMA needs shuffles but mma.sync might not

**WGMMA (Hopper):** Operates on 128-thread warpgroups (4 warps). The accumulator lives in registers distributed across 128 threads. The A operand can come from either registers or TMEM. The register-to-register path requires reshuffling because the 128-thread accumulator layout and the 128-thread A operand layout distribute data across threads differently. Colfax's `ReorgCFp8toAFp8` shuffles within 4-thread groups to fix this.

**mma.sync (sm_89/sm_120):** Operates on 32-thread warps. Both accumulator AND A operand are in the same warp's registers. The thread-to-data mapping is simpler:

- **Accumulator (SM80_16x8_Row):** Thread t holds d[0..3] covering rows {t%4*2, t%4*2+1} × cols {(t/4)%2*4+0..3} (from PTX ISA Figure 22)
- **FP8 A operand:** Thread t holds a[0..3] = 16 FP8 values across 4 registers

### The critical question: register packing order

For BF16 m16n8k16, the worker's existing `pack_bf16x2` does:
```
a[0] = pack(d[0], d[1])  // two FP32 → one BF16x2 register
a[1] = pack(d[2], d[3])
```
This works because the BF16 A operand's value layout matches the accumulator's — they're designed to be compatible for back-to-back MMAs.

For FP8 m16n8k32, each A register holds 4 FP8 values (not 2 BF16). The k-dimension doubles from 16 to 32, meaning each register covers 4 k-elements instead of 2. The question is whether the accumulator values from TWO consecutive m16n8k16-sized output tiles can be packed into one FP8 A register in the right order.

### Proposed approach: test simple packing first

**Hypothesis:** The FP8 m16n8k32 A operand layout is a natural extension of the BF16 m16n8k16 layout — each pair of BF16 values (2 bytes each = 4 bytes) in one register becomes 4 FP8 values (1 byte each = 4 bytes) in one register. The packing from two consecutive accumulator pairs:

```cuda
// Pack 4 FP32 accumulator values into 1 FP8 A register
// d[0], d[1] from P column c
// d[2], d[3] from P column c+1 (or c+k_stride)
uint32_t a_reg;
asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=r"(lo) : "f"(d[0]), "f"(d[1]));
asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=r"(hi) : "f"(d[2]), "f"(d[3]));
a_reg = (hi << 16) | (lo & 0xFFFF);
// Or use __byte_perm to combine: __byte_perm(lo, hi, 0x5410)
```

**This is a 5-minute test.** Write a small test kernel:
1. Set up known P values in FP32 accumulators
2. Pack them into FP8 A operands using simple conversion + byte_perm (NO shfl_sync)
3. Feed into m16n8k32 MMA with a known B operand
4. Compare output to expected result

If correct → no shuffles needed (save ~8 instructions per conversion)
If wrong → try byte_perm with different selector masks before adding shuffles

### Why this might work for mma.sync

The PTX ISA defines both the accumulator and A operand layouts for the SAME warp. In the m16n8k16 BF16 case, NVIDIA deliberately made them compatible (pack_bf16x2 works directly). For m16n8k32 FP8, the same design principle likely applies — NVIDIA wants back-to-back MMAs to be efficient. The 4-byte register holds 4 FP8 values mapped to the same thread/row positions, just with 2x the k-coverage.

The WGMMA case is different because warpgroups (128 threads) have a fundamentally different thread↔data distribution than warps (32 threads). The shuffle is fixing the warpgroup layout mismatch, not an inherent FP8 issue.

## Implementation Recommendation

1. **Try simple packing FIRST** (no shuffles) — 5 minute test
2. If wrong, try different byte_perm selectors (there are only 24 permutations of 4 bytes)
3. Only add shfl_sync if no byte_perm alone fixes it
4. The CUTLASS mma_traits_sm89.hpp layout strides are the ground truth — if byte orders are wrong, the stride values tell you exactly which bytes to swap

## Cost Comparison

| Approach | Instructions per 4-register conversion | Register overhead |
|----------|---------------------------------------|-------------------|
| Simple packing (no shuffle) | 4 CVT + 2 PRMT = 6 | 0 extra |
| With shfl_sync (Colfax pattern) | 4 CVT + 4 PRMT + 4 SHFL = 12 | 2 temp regs |
| Current BF16→FP8 conversion | ~448 ALU per KV block | ~10 temp regs |

Even the worst case (12 instructions with shuffles) is 37x fewer instructions than the current conversion path.

## Caveats

- The CUTLASS ALayout strides `Stride<Stride<_64,_1>, Stride<_16,_8,_256>>` suggest a non-trivial value packing order. The "simple packing" hypothesis needs empirical verification.
- If the m16n8k32 A layout interleaves k-values from non-adjacent accumulator positions, shuffles ARE needed. But this costs 12 instructions vs 448, so it's still a massive win.
- All of this assumes FP8 native inputs (from ldmatrix reinterpret). Without native inputs, the conversion overhead is on K/V loading, not P→A.
