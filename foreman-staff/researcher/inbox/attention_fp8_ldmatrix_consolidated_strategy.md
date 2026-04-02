# FP8 Warp-Collective Load Strategy: Consolidated Research Brief

**Sources:**
- https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
- https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8
- https://research.colfax-intl.com/adding-fp8-to-flashattention/
- https://gau-nernst.github.io/fa-5090/
- https://github.com/NVIDIA/cutlass (copy_sm100.hpp, copy_traits_sm100.hpp, mma_sm89.hpp, mma_traits_sm89.hpp, config.hpp)
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (PTX ISA 9.2)
- https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
- https://github.com/NVIDIA/cutlass/issues/2867
- https://forums.developer.nvidia.com/t/run-ptx-mma-sync-aligned-kind-mxf8f6f4-block-scale-scale-vec-1x-m16n8k32-on-sm-120a/329702
- https://github.com/HazyResearch/ThunderKittens/pull/140
**Relevant to:** attention worker (FP8 kernel)
**Worker's current problem:** Experiment 66 showed that replacing ldmatrix with scalar uint16 loads for FP8 native inputs causes a 19% regression (62 us vs 52 us baseline). ldmatrix loads 128 bits per thread per instruction; scalar loads move 16 bits. The throughput gap is fundamental. The worker needs a way to load FP8 data from shared memory at ldmatrix-class throughput.
**Status:** Consolidation of 5 previous briefs with new concrete findings from CUTLASS source code.

## EXECUTIVE SUMMARY

There are exactly TWO viable paths for loading FP8 data from shared memory into MMA registers at warp-collective throughput on sm_120. Both are confirmed by production code.

| Path | Instruction | Availability | Post-load fixup | Status |
|------|------------|-------------|-----------------|--------|
| A: b16 reinterpret | `ldmatrix.sync.aligned.m8n8.x4.b16` | sm_75+ (all) | Possibly PRMT | **Primary (recommended)** |
| B: native b8 | `ldmatrix.sync.aligned.m16n16.x1/x2.trans.shared.b8` | sm_100a+ / sm_120a+ | PRMT byte interleave (confirmed in CUTLASS) | **Backup** |

Path A is recommended because it is simpler, works on the existing code path (same instruction the BF16 kernel uses), has both transposed and non-transposed variants, and is externally validated by ThunderKittens.

---

## FINDING 1: ldmatrix.m8n8.x4.b16 Reinterpret is the Standard FP8 Loading Technique

ThunderKittens (Hazy Research, production FP8 GEMM achieving 1500 TFLOPS on H100) explicitly states:

> "ldmatrix assumes each matrix element holds 16-bits of data... let the instruction load in 16-bits and use this to fill 2 fp8 values per load."

This is how FP8 data is loaded in every production FP8 kernel that uses mma.sync:
1. FP8 data is stored in shared memory as contiguous bytes
2. `ldmatrix.m8n8.x4.b16` loads 128 bits per thread, treating each 16-bit pair as one "element"
3. The hardware does not care what the bits represent -- it moves 128 bits per thread
4. Each 32-bit output register contains 4 FP8 values (two "16-bit pairs")
5. The register content is directly usable as m16n8k32 MMA operands

**Why this works without reshuffling (structural argument):**

For m16n8k16 BF16 MMA:
- A operand = 4 x uint32_t (each holds 2 BF16 = 4 bytes)
- ldmatrix.x4.b16 produces exactly 4 x uint32_t

For m16n8k32 FP8 MMA:
- A operand = 4 x uint32_t (each holds 4 FP8 = 4 bytes)
- ldmatrix.x4.b16 produces exactly 4 x uint32_t

The register count is IDENTICAL. The thread-to-register mapping of ldmatrix.x4.b16 distributes data such that each thread's 4 registers cover 4 consecutive 8x8 sub-tiles along the k-dimension. For BF16 (k=16), this means 4 tiles of 8x8 = 8x32 in 16-bit units. When reinterpreted as FP8, the same 4 tiles are 8x32 in 8-bit units, which packs twice the k-values per sub-tile. The m16n8k32 MMA expects exactly this: 32 FP8 values along k, packed 4 per register.

**The one uncertainty:** Whether the byte ORDER within each register matches what m16n8k32 expects. If not, a single PRMT (byte permute) instruction per register fixes it. This costs 4 PRMT instructions total -- trivial compared to the 448 conversion instructions being eliminated.

---

## FINDING 2: ldmatrix.m16n16.b8 EXISTS on sm_120 but Has Limitations

### Confirmed available on sm_120a

CUTLASS `include/cute/arch/config.hpp` explicitly enables the b8 ldmatrix for sm_120a:

```cpp
#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || ... ||
     defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) || ...)
#  define CUTE_ARCH_LDSM_SM100A_ENABLED   // gates ldmatrix.m16n16.b8
#endif
```

### Only TRANSPOSED variant exists

CUTLASS defines only:
- `ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8` (2 output registers)
- `ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8` (4 output registers)

There is NO non-transposed `ldmatrix.m16n16.b8`. This means:
- Column-major FP8 data in smem: can use `.trans` directly
- Row-major FP8 data in smem: cannot use this instruction

### Requires post-load byte interleave

CUTLASS applies a byte rearrangement after every ldmatrix.m16n16.b8 call:

```cpp
// After ldmatrix.m16n16.x1.trans.b8 produces {tmp0, tmp1}:
uchar4& tmp0_ = reinterpret_cast<uchar4&>(tmp0);
uchar4& tmp1_ = reinterpret_cast<uchar4&>(tmp1);
uchar4 dst0_{tmp0_.x, tmp0_.y, tmp1_.x, tmp1_.y};
uchar4 dst1_{tmp0_.z, tmp0_.w, tmp1_.z, tmp1_.w};
```

This interleaves bytes across the two output registers. The pattern takes the first 2 bytes from each register and groups them, then the last 2 bytes. This compiles to PRMT instructions (very cheap, single-cycle).

CUTLASS comments note: "RefLayout of ldmatrix.m16n16.x1.trans won't match stmatrix.m16n8.x2.trans without additional transformations."

### Why Path A is still preferred over Path B

| Criterion | Path A (m8n8.x4.b16) | Path B (m16n16.b8) |
|-----------|----------------------|---------------------|
| Architecture | sm_75+ (proven everywhere) | sm_100a+/sm_120a+ only |
| Transpose modes | Both .trans and non-.trans | .trans ONLY |
| Registers per call | 4 (one call for full A operand) | 2 per call, need 2 calls for A (16x32) |
| Post-load fixup | Possibly PRMT (needs testing) | Definitely PRMT (confirmed by CUTLASS) |
| Shared mem layout | Same as existing BF16 (with byte pairs) | Must be column-major |
| Production validation | ThunderKittens (1500 TFLOPS) | CUTLASS only (datacenter-focused) |

---

## FINDING 3: m16n8k32 FP8 MMA Fragment Layout (from CUTLASS mma_traits_sm89.hpp)

The authoritative operand layouts for `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`:

**A operand (16x32, row-major):** 4 x uint32_t per thread
```
Layout: Shape<Shape<_4,_8>, Shape<_4,_2,_2>>
        Stride<Stride<_64,_1>, Stride<_16,_8,_256>>
```
- 32 threads (warp), each holds 4*2*2 = 16 FP8 values = 4 registers
- Thread dimension: groups of 4 within groups of 8 (standard MMA thread layout)
- Value dimension: 3D packing of 4x2x2 FP8 values across 4 registers

**B operand (8x32, column-major):** 2 x uint32_t per thread
```
Layout: Shape<Shape<_4,_8>, Shape<_4,_2>>
        Stride<Stride<_32,_1>, Stride<_8,_128>>
```
- Each thread holds 4*2 = 8 FP8 values = 2 registers

**C/D accumulator (16x8):** 4 x float32 per thread (same as BF16 m16n8k16)
```
Layout: SM80_16x8_Row
```

Key insight: The C/D accumulator layout is SHARED between BF16 m16n8k16 and FP8 m16n8k32. This means the output of QK^T MMA is in the same register format regardless of precision. However, converting this accumulator to an FP8 A operand for PV MMA requires byte_perm + shfl_sync because the A operand layout DIFFERS from the accumulator layout for FP8 (unlike BF16 where they are identical).

---

## FINDING 4: P-to-FP8 Conversion Needs Cross-Thread Shuffle

Colfax's FP8 FlashAttention-2 implementation documents that for FP8 MMA, the accumulator layout (FP32) and the A operand layout (FP8) differ in how bytes are distributed across threads. Their `ReorgCFp8toAFp8` function shows:

**Step 1: Within-thread byte permute**
```cuda
auto upper0 = __byte_perm(upper, lower, 0x7654);  // gather high bytes
auto lower0 = __byte_perm(upper, lower, 0x3210);  // gather low bytes
```

**Step 2: Cross-thread shuffle (within groups of 4 threads)**
```cuda
int upper_map[4] = {0, 3, 1, 2};
int lower_map[4] = {1, 2, 0, 3};
upper0 = __shfl_sync(0xFFFFFFFF, upper0, upper_map[threadIdx.x % 4], 4);
lower0 = __shfl_sync(0xFFFFFFFF, lower0, lower_map[threadIdx.x % 4], 4);
```

**Caveat:** Colfax targets Hopper WGMMA. For mma.sync on sm_89/sm_120, the shuffle pattern may differ. The CUTLASS ALayout from mma_traits_sm89.hpp is the authoritative reference. The PRINCIPLE (byte_perm + shfl_sync) is correct; the exact permutation maps need derivation from the CuTe layout strides.

**Cost estimate:** 2 PRMT + 2 SHFL per register pair = approximately 8-12 instructions per P-to-FP8 conversion block. Compare to the current 448 ALU instructions for BF16-to-FP8 conversion of K+V.

---

## FINDING 5: Additional ldmatrix Variants for Sub-Byte Types (Not Directly Useful But Informative)

CUTLASS defines format-converting ldmatrix variants for sub-byte types:

```
ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64   (4-bit -> 8-bit, padded)
ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32   (6-bit -> 8-bit, padded)
```

These load 4-bit or 6-bit data from smem and expand it to 8-bit in registers. They exist for NVFP4 and MXFP6 support. Not directly useful for FP8 but demonstrate that the sm_100+/sm_120+ hardware has flexible bit-width conversion in the load path.

---

## FINDING 6: The FP8 Swizzle Pattern Differs from BF16

ThunderKittens notes that FP8 core matrices are 8x16 elements (vs 8x8 for BF16), requiring different swizzle modes:

- BF16: 8x8 tile = 128 bytes/row, 16-byte alignment naturally fits 32-byte swizzle
- FP8: 8x16 tile = 128 bytes/row (same bytes but twice as many elements), minimum tile width = 32 elements

For the ldmatrix.m8n8.x4.b16 reinterpret approach, the swizzle pattern should work identically to BF16 because the instruction operates on the same byte granularity. The shared memory layout is byte-identical whether interpreted as BF16 or FP8 pairs.

---

## FINDING 7: No Other FP8 Warp-Collective Load Primitives Exist

Exhaustive search of PTX ISA 9.2 (CUDA 13.2) confirms:

| Instruction | FP8 support | Notes |
|------------|-------------|-------|
| `ldmatrix.m8n8.b16` | Via reinterpret (b16 = 2xFP8) | sm_75+, primary approach |
| `ldmatrix.m16n16.b8` | Native 8-bit | sm_100a+/sm_120a+, .trans only |
| `ldmatrix.m8n16.b8x16` | Format-converting (4/6-bit src) | Not for plain FP8 |
| `cp.async` | Type-agnostic byte copy | Global->smem only, not smem->regs |
| `ld.shared.b32/b64/b128` | Scalar loads | Not warp-collective, no fragment layout |
| `lop3` / `prmt` | Byte manipulation | Post-load only, not load instructions |

There is no hidden instruction. The two paths identified above are the only options.

---

## CONCRETE IMPLEMENTATION PLAN FOR THE ATTENTION WORKER

### Step 0: Minimal verification test (CRITICAL -- do before any kernel changes)

```cuda
// Test: load FP8 data from smem using ldmatrix.x4.b16, feed to m16n8k32 MMA
// 1. Write known FP8 values to smem (e.g., identity matrix as e4m3 bytes)
// 2. Use ldmatrix_x4 to load (treating FP8 pairs as b16)
// 3. Feed loaded registers directly to mma.sync.m16n8k32.f32.e4m3.e4m3.f32
// 4. Compare output to expected result (computed from the FP8 inputs)
//
// If correct: ldmatrix.b16 reinterpret works, no PRMT needed
// If wrong but close: need PRMT to fix byte order (try different selectors)
// If completely wrong: need to understand the mismatch before proceeding
```

### Step 1: Modify smem layout for FP8 K and V

Store FP8 K and V directly in shared memory (half the size of BF16):
- K: 64 x 64 x 1 byte = 4 KB per buffer (was 8 KB for BF16)
- V: 64 x 64 x 1 byte = 4 KB per buffer (was 8 KB for BF16)
- Total: 16 KB double-buffered (was 32 KB)
- Potential for 4 blocks/SM or more smem headroom

Use the same XOR swizzle pattern (byte-level swizzle is unchanged).

### Step 2: Load K using ldmatrix_x4.b16 (reinterpret)

For K as B operand in QK^T MMA (m16n8k32):
- B needs 2 x uint32_t per thread
- Use `ldmatrix_x2.b16` or `ldmatrix_x2_trans.b16` (same as BF16 path but data is FP8 pairs)
- Each 32-bit register holds 4 FP8 values

For K as A operand (if needed for different MMA orientation):
- A needs 4 x uint32_t per thread
- Use `ldmatrix_x4.b16` (same instruction as BF16 Q loading)

### Step 3: Load V using ldmatrix_x2_trans.b16 (reinterpret)

V as B operand in PV MMA:
- Same as K: `ldmatrix_x2_trans.b16` produces 2 x uint32_t
- Transposed variant handles the row/column orientation

### Step 4: Convert P (FP32 accumulator) to FP8 A operand

This is the ONLY part that needs new code beyond what exists:
1. Convert FP32 -> FP8 using existing `cvt.rn.satfinite.e4m3x2.f32` (already working)
2. Apply byte_perm + shfl_sync to rearrange from accumulator layout to A operand layout
3. The exact permutation pattern needs derivation from the CuTe layout strides (see Finding 3)
4. Alternative: skip the shuffle and test if it works without -- the mma.sync fragment layout may be more forgiving than WGMMA

### Expected Performance

If successful, the FP8 native input path eliminates:
- ~448 BF16-to-FP8 conversion ALU instructions per KV block
- ~7.5 us of conversion overhead (from agent_state.md analysis)

And adds:
- Possibly 4-8 PRMT instructions for byte fixup (if needed)
- ~8-12 instructions for P-to-FP8 layout conversion (byte_perm + shfl_sync)

Net savings: ~420+ instructions per KV block. Target: 44-45 us (from current 52 us).

---

## RISKS AND UNKNOWNS

1. **Byte ordering uncertainty.** The structural argument that ldmatrix.b16 reinterpret produces correct FP8 fragments is sound but unverified. The test in Step 0 resolves this within minutes.

2. **P-to-FP8 shuffle pattern for mma.sync.** Colfax's pattern is for WGMMA. The mma.sync version may be simpler (mma.sync uses the same register file, not TMEM). It could also be unnecessary if the accumulator-to-A-operand mapping happens to match for sm_89/sm_120 FP8.

3. **Register pressure.** Current FP8 kernel uses 165 regs (5 spare before hitting 3 blocks/SM). The native FP8 path should REDUCE register pressure by eliminating conversion temporaries. But the P-to-FP8 shuffle adds temporary registers. Net effect uncertain.

4. **ldmatrix_x2_trans for FP8 V loading.** The transposed ldmatrix for B operand needs testing. If the byte ordering is wrong for the transposed case (even if non-transposed works), an alternative is to pre-transpose V in smem or use a different loading strategy for V.

5. **sm_120a flag requirement.** The ldmatrix.m16n16.b8 native instruction requires compiling with `-arch=sm_120a` (architecture-accelerated features). This is NOT needed for Path A (ldmatrix.m8n8.b16 reinterpret), which works with plain `-arch=sm_120`.
