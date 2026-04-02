# FP8 ldmatrix: Concrete Implementations from CUTLASS Source + Colfax Code

**Sources:**
- https://github.com/NVIDIA/cutlass (copy_sm100.hpp, copy_traits_sm100.hpp, mma_sm89.hpp, mma_traits_sm89.hpp, config.hpp)
- https://research.colfax-intl.com/adding-fp8-to-flashattention/
- https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8
- https://gau-nernst.github.io/fa-5090/
- https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
**Relevant to:** attention worker (FP8 kernel), GEMM worker
**Worker's current problem:** FP8 kernel at 52 us uses BF16 inputs with in-register conversion (~448 ALU/KV block, 14:1 conversion:MMA ratio). Native FP8 inputs via scalar loads regressed 19% (exp 66). Need ldmatrix-based FP8 loading.

## CRITICAL FINDING 1: sm_120 DOES Have ldmatrix.m16n16.b8 (Confirmed in CUTLASS Source)

Previous brief noted uncertainty about whether ldmatrix.m16n16.b8 is available on sm_120.
**The answer is YES.** CUTLASS source code (`include/cute/arch/config.hpp`) shows:

```cpp
#if (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED))
#  define CUTE_ARCH_LDSM_SM100A_ENABLED
#  define CUTE_ARCH_STSM_SM100A_ENABLED
#endif
```

SM120A explicitly enables `CUTE_ARCH_LDSM_SM100A_ENABLED`, which gates the new b8 ldmatrix variants.

## CRITICAL FINDING 2: Only TRANSPOSED ldmatrix.m16n16.b8 Exists

CUTLASS `copy_sm100.hpp` defines ONLY transposed 8-bit ldmatrix variants:

```
ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8   (SM100_U8x8_LDSM_T)
ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8   (SM100_U8x16_LDSM_T)
```

There is NO non-transposed `ldmatrix.m16n16.b8` variant in the CUTLASS codebase. This means:
- **For column-major data (like K^T for QK^T MMA): use the .trans variant directly**
- **For row-major data: you cannot use ldmatrix.m16n16.b8 without pre-transposing in smem**

This is a significant constraint. The m8n8.x4.b16 reinterpret approach has BOTH .trans and non-.trans variants on all sm_75+ architectures.

## CRITICAL FINDING 3: ldmatrix.m16n16.b8 Needs Post-Load Byte Shuffle

CUTLASS applies a byte shuffle AFTER ldmatrix.m16n16.b8 to match stmatrix layout. From `copy_sm100.hpp`:

```cpp
// SM100_U8x8_LDSM_T (x1 variant, loads 16x16 of 8-bit, transposed)
asm volatile ("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];\n"
    : "=r"(tmp0), "=r"(tmp1) : "r"(smem_int_ptr));

// Comment: "RefLayout of ldmatrix.m16n16.x1.trans won't match
//  stmatrix.m16n8.x2.trans without additional transformations"

// Post-load byte rearrangement:
uchar4& tmp0_ = reinterpret_cast<uchar4&>(tmp0);
uchar4& tmp1_ = reinterpret_cast<uchar4&>(tmp1);
uchar4 dst0_{tmp0_.x, tmp0_.y, tmp1_.x, tmp1_.y};  // interleave bytes
uchar4 dst1_{tmp0_.z, tmp0_.w, tmp1_.z, tmp1_.w};
```

And for x2 variant (SM100_U8x16_LDSM_T, loads two 16x16 tiles):

```cpp
asm volatile ("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3) : "r"(smem_int_ptr));

uchar4 dst0_{tmp0_.x, tmp0_.y, tmp1_.x, tmp1_.y};
uchar4 dst1_{tmp0_.z, tmp0_.w, tmp1_.z, tmp1_.w};
uchar4 dst2_{tmp2_.x, tmp2_.y, tmp3_.x, tmp3_.y};
uchar4 dst3_{tmp2_.z, tmp2_.w, tmp3_.z, tmp3_.w};
```

**This is the byte permutation pattern.** ldmatrix.m16n16.b8 produces registers where
bytes from two "sub-tiles" are interleaved incorrectly. The fix is to take bytes
{a,b,c,d} from reg0 and {e,f,g,h} from reg1, producing {a,b,e,f} and {c,d,g,h}.

This is a simple PRMT instruction (byte permute), not the more expensive shfl_sync.
It operates within a single thread's registers, no cross-thread communication needed.

## FINDING 4: CUTLASS FP8 MMA Traits for SM89 (Confirmed Fragment Register Counts)

From `mma_sm89.hpp` and `mma_traits_sm89.hpp`:

```cpp
struct SM89_16x8x32_F32E4M3E4M3F32_TN {
  using ARegisters = uint32_t[4];   // 4 x 32-bit = 16 FP8 bytes = 16 e4m3 values
  using BRegisters = uint32_t[2];   // 2 x 32-bit = 8 FP8 bytes = 8 e4m3 values
  using CRegisters = float[4];      // 4 x 32-bit FP32 accumulator
  using DRegisters = float[4];

  // PTX: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
};
```

**A operand layout** (from mma_traits_sm89.hpp):
```cpp
using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                       Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
```

Decoding: This maps (thread_id, value_id) -> element_index for a 16x32 matrix.
- Thread dimension: Shape<_4,_8> = groups of 4 threads within groups of 8 = 32 threads
- Value dimension: Shape<_4,_2,_2> = 4*2*2 = 16 values per thread = 16 FP8 bytes = 4 registers

**B operand layout:**
```cpp
using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_4,  _2>>,
                       Stride<Stride<_32,_1>,Stride<_8,_128>>>;
```
- 4*2 = 8 values per thread = 8 FP8 bytes = 2 registers

**Accumulator layout** (shared with BF16 m16n8k16):
```cpp
using CLayout = SM80_16x8_Row;
// = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
//          Stride<Stride<_32,_1>,Stride<_16,_8>>>;
```

## FINDING 5: Colfax byte_perm + shfl_sync Algorithm (P -> FP8 A Fragment Conversion)

Colfax's FP8 FlashAttention-2 provides the exact algorithm for converting FP32
accumulator fragments to FP8 operand A fragments. This applies to our P->FP8 path
(converting softmax output P to FP8 A operand for PV MMA).

**Why this is needed:** FP32 accumulator layout != FP8 operand A layout (unlike BF16
where they are identical). After downcasting FP32 to FP8, bytes must be rearranged.

**Step 1: byte_perm (within-thread rearrangement):**
```cuda
// Given: upper = register with rows 0-1 data, lower = register with rows 2-3 data
auto upper0 = __byte_perm(upper, lower, 0x7654);
auto lower0 = __byte_perm(upper, lower, 0x3210);
```

**Step 2: shfl_sync (cross-thread exchange):**
```cuda
int upper_map[4] = {0, 3, 1, 2};
int lower_map[4] = {1, 2, 0, 3};

upper0 = __shfl_sync(0xFFFFFFFF, upper0, upper_map[threadIdx.x % 4], 4);
lower0 = __shfl_sync(0xFFFFFFFF, lower0, lower_map[threadIdx.x % 4], 4);
```

**Context:** This is called `ReorgCFp8toAFp8` in Colfax's code, invoked immediately
before the second GEMM (PV MMA). Note: Colfax targets WGMMA on Hopper, so the exact
shuffle pattern may differ for mma.sync on sm_89/sm_120. But the PRINCIPLE applies:
FP8 operand A has a different byte distribution across threads than FP32 accumulator.

**Impact estimate:** 2 PRMT + 2 SHFL per register pair = ~8 instructions per P->FP8
conversion (on top of the CVT instructions). This is much cheaper than the current
448-instruction BF16->FP8 conversion path.

## FINDING 6: ThunderKittens FP8 Details

ThunderKittens (Hazy Research) provides these additional implementation details:

1. **Minimum tile width = 32 for FP8** (vs 16 for BF16/FP16/FP32). This is because
   FP8 core matrices are 8x16 elements (not 8x8), so swizzle modes (32/64/128-byte)
   require wider tiles.

2. **Register shuffle between bf16 and fp8 thread ownership:** FP8 uses 4 elements/thread
   packing vs BF16's 2 elements/thread. When converting between precisions (like P from
   FP32 -> FP8), "fp8-threads need to obtain values from two threads. bf16-threads need
   to send values to two different fp8-threads." This is exactly the shfl_sync from Colfax.

3. **Performance:** "1500 TFLOPS in 95 lines of code" on H100. Their FP8 kernels achieve
   near-peak utilization using the ldmatrix.b16 reinterpret + warp shuffle approach.

## FINDING 7: No New FP8-Specific ldmatrix in CUDA 13.1/13.2

Searched PTX ISA 9.2 (CUDA 13.2) for any new FP8-specific ldmatrix instructions.
Results:
- **No `ldmatrix.m8n8.b8` variant exists** (cannot load 8-bit in the m8n8 shape)
- **ldmatrix.m16n16.b8` on sm_100+** is the only native 8-bit load (transposed only)
- **ldmatrix.m8n16 with .b8x16.b4x16_p64 and .b8x16.b6x16_p32** exist for sub-byte
  types with bit-width conversion, but these are for 4-bit and 6-bit data, not plain FP8
- **No `ldmatrix.m8n8.x4.b8` exists** - the .b8 type is only on .m16n16 shape

The PTX ISA 9.2 did add `mma.sync.aligned.kind::mxf8f6f4.block_scale` with UE8M0
scale factors for sm_120, which is the MXFP8 variant. But this doesn't change the
ldmatrix situation.

## SYNTHESIS: Two Viable FP8 Loading Paths for sm_120

### Path A: ldmatrix.m8n8.x4.b16 Reinterpret (RECOMMENDED - works on sm_75+)

1. Store FP8 data in smem with pairs of FP8 bytes treated as 16-bit elements
2. Use `ldmatrix.sync.aligned.m8n8.x4.b16` to load 4 registers
3. Each 32-bit register contains 4 FP8 values (2 "16-bit pairs")
4. Byte ordering within each register matches what m16n8k32 MMA expects
   (4 consecutive FP8 values in k-dimension)
5. **No post-load shuffle needed for K/V loading** (structural analysis, pending
   empirical verification)
6. **P->FP8 conversion DOES need shfl_sync** (Colfax confirmed, ~8 extra instructions)

**Advantages:** Works on all architectures (sm_75+), both .trans and non-.trans variants
available, well-understood from BF16 usage.

### Path B: ldmatrix.m16n16.x1/x2.trans.b8 Native (sm_100+/sm_120+ only)

1. Store FP8 data in smem in column-major order (for transposed load)
2. Use `ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8` (2 output regs for 16x16)
   or `ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8` (4 output regs for 16x32)
3. **MUST apply post-load byte interleave** (CUTLASS does this - see Finding 3)
4. Only transposed variant exists - requires column-major smem layout

**Advantages:** Native 8-bit instruction, no reinterpretation needed.
**Disadvantages:** Transposed only, needs byte shuffle, two calls for 16x32 A operand
(vs one ldmatrix.x4.b16 call), only works on sm_100+.

### Recommendation

**Path A (ldmatrix.m8n8.x4.b16 reinterpret) is the primary approach.** It is simpler,
works everywhere, and avoids the post-load byte shuffle that Path B requires. The
worker should:

1. Build a minimal test: store known FP8 values in smem as b16 pairs, ldmatrix.x4.b16,
   then feed to m16n8k32 MMA and verify output.
2. If byte ordering is wrong, add a PRMT (byte permute) to fix it - still cheaper than
   448 conversion ALU instructions.
3. For the P->FP8 path (accumulator to A operand), implement the Colfax byte_perm +
   shfl_sync pattern adapted for mma.sync (not wgmma).

## Caveats

1. **Colfax targets Hopper WGMMA, not sm_89/sm_120 mma.sync.** The byte_perm + shfl_sync
   pattern is conceptually correct but the exact shuffle maps may differ. The CUTLASS
   ALayout from mma_traits_sm89.hpp is the authoritative reference for sm_89/sm_120.

2. **The "no post-load shuffle needed" claim for Path A is structural analysis only.**
   It is based on the observation that ldmatrix.x4.b16 outputs 4 registers with the same
   thread-to-register mapping that m16n8k32 expects (since register counts are identical
   to m16n8k16 BF16). This MUST be verified empirically.

3. **ThunderKittens targets H100 (sm_90) with WGMMA.** Their ldmatrix.b16 reinterpret
   technique is the same principle but the specific fragment layouts differ from mma.sync.

4. **gau-nernst's FA on RTX 5090** achieves 94.39% peak (197.74 TFLOPS) for BF16 but
   has NO FP8 implementation. There is no public FP8 flash attention for sm_120 with
   mma.sync to reference.
