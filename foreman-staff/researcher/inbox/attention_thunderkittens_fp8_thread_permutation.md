# ThunderKittens FP8 Thread Permutation for Fragment Layout

**Source:** https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8
**Also:** https://github.com/HazyResearch/ThunderKittens/pull/140
**Relevant to:** attention worker (FP8 kernel)
**Worker's current problem:** Need to understand FP8 fragment layout differences vs BF16 for the ldmatrix reinterpret approach and P->FP8 conversion.

## What This Is

ThunderKittens (Hazy Research) implemented FP8 support and documented the fundamental
layout difference between FP8 and BF16/FP16 fragments. Their key insight: FP8 fragments
have a **logically distinct thread-to-element mapping** from BF16/FP16, requiring
inter-thread data shuffling when converting between precisions. They achieved 1500 TFLOPS
in 95 lines of code on H100.

## Why It Matters for Us

Two direct applications for our attention kernel:

1. **P->FP8 conversion:** The softmax output P is in FP32 accumulator registers with
   BF16-compatible layout. Converting to FP8 A operand for PV MMA requires understanding
   exactly how the FP8 thread layout differs, because data must move between threads.

2. **Validating the ldmatrix reinterpret approach:** When we load FP8 data via
   ldmatrix.b16, each thread gets 2 FP8 bytes packed as one "16-bit element." The
   ThunderKittens work confirms this is the correct approach ("let the instruction load
   in 16-bits and use this to fill 2 fp8 values per load") and documents what shuffling
   is needed afterward.

## Key Technique

### The Fundamental Layout Difference

| Precision | Elements per thread per register | Packing |
|-----------|--------------------------------|---------|
| BF16/FP16 | 2 elements (16-bit each) | `{a0, a1}` in uint32 |
| FP8       | 4 elements (8-bit each)  | `{a0, a1, a2, a3}` in uint32 |

The thread-to-element mapping changes with this packing difference:
- In BF16 m16n8k16: thread T holds elements at positions determined by a 2-element stride
- In FP8 m16n8k32: thread T holds elements at positions determined by a 4-element stride

This means that when converting from BF16/FP32 layout to FP8 layout:
- Each FP8 thread needs data from **two** BF16 source threads
- Each BF16 source thread must send values to **two** different FP8 threads

### The ldmatrix.b16 Hack for FP8

ThunderKittens confirms: "let the instruction load in 16-bits and use this to fill
2 fp8 values per load." This is exactly our ldmatrix reinterpret approach.

The key detail: ldmatrix.b16 loads a 128-bit row per thread, interpreting it as 8
16-bit elements. When FP8 data is stored contiguously, each "16-bit element" naturally
holds 2 adjacent FP8 values. The hardware doesn't care about the data type -- it moves
bits. The FP8 values end up in the correct register positions if the smem layout
matches the expected fragment layout.

### Thread Permutation for Precision Conversion

When converting between BF16/FP32 tiles and FP8 tiles (as needed for P->FP8):

1. **Identify source/destination lane mapping:**
   - Compute which FP8 lane needs data from which BF16 lane
   - The PR #140 simplified this to a concise formula

2. **Execute shuffles:**
   - Use `__shfl_sync` to move data between lanes
   - Each thread does 1 shuffle to get the value it needs from another thread
   - But each FP8 thread needs values from 2 BF16 threads, so 2 shuffles per thread

3. **Pack the FP8 bytes:**
   - After shuffling, each thread has the correct FP32/BF16 values
   - Apply `cvt.rn.satfinite.e4m3x2.f32` to downcast pairs
   - The result is already in the correct register position for FP8 MMA

### Shared Memory Considerations for FP8

ThunderKittens enforces minimum FP8 tile widths of 32 elements (vs 16 for BF16) to
support their swizzle modes (32-byte, 64-byte, 128-byte). This aligns with our
observation that FP8 K tiles are 32 columns wide (32 FP8 values = 32 bytes per row
= 256 bits), matching the 128-bit ldmatrix row load with 2 loads per row.

The core 8x8 FP8 matrix in smem is 8x16 elements (compared to 8x8 for FP16),
because each "16-bit position" holds 2 FP8 values.

## Implementation Guidance for Our Kernel

### For ldmatrix FP8 reinterpret (K and V loading):
- Store FP8 data contiguously in smem, row-major
- Use ldmatrix_x4.b16 unchanged -- it loads "16-bit pairs" that are actually FP8 pairs
- The a1/a2 register swap (`ldmatrix_x4_mma`) applies identically
- XOR swizzle operates on 16-bit element indices, unchanged from BF16

### For P->FP8 conversion:
- FP32 accumulator registers are in "BF16-compatible" layout
- Must shuffle data between threads before packing to FP8
- Need 2 `__shfl_sync` calls per conversion group
- Then `cvt.rn.satfinite.e4m3x2.f32` to downcast
- Total cost: ~2 shuffles + 4 CVT instructions per 8 FP8 values

## Caveats

1. **ThunderKittens uses WGMMA (Hopper).** The specific lane mapping formulas are
   for 128-thread warpgroups, not 32-thread warps. The CONCEPT (FP8 needs inter-thread
   shuffling) transfers, but the exact lane IDs do not.

2. **The 1500 TFLOPS result is H100 WGMMA, not sm_120 mma.sync.** Our theoretical
   peak is different. The validation is that the approach works, not the specific number.

3. **For our ldmatrix loading path (K, V), NO shuffling is needed.** If FP8 data is
   pre-arranged in smem correctly, ldmatrix places bytes directly into the right
   register positions. Shuffling is only needed for the P->FP8 path where we're
   converting from FP32 accumulator layout.

4. **The PR #140 has a simplified lane ID formula.** Worth fetching the actual diff
   to get the concise computation, even though it's for WGMMA layout.
