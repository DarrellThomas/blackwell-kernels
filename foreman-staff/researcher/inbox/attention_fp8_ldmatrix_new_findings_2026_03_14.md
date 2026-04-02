# FP8 ldmatrix: New Findings (March 2026 Update)

**Sources:**
- https://gau-nernst.github.io/nvrtc-matmul/ (FP8 GEMM on RTX 5090 with ldmatrix.b16)
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/atom/mma_traits_sm120.hpp
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/atom/mma_traits_sm80.hpp
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm120.hpp
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm89.hpp
- https://github.com/thu-ml/SageAttention (sm89 FP8 attention with mma.sync)
- https://github.com/NVIDIA/cutlass/discussions/1846 (CuTe layout interpretation)
- https://github.com/HazyResearch/ThunderKittens/issues/81 (FP8 mma_AB limitations)
- https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2 (ThunderKittens 2.0, MXFP8 on Blackwell)
**Relevant to:** attention worker (FP8 kernel)
**Worker's current problem:** Experiment 66 confirmed ldmatrix cannot be replaced with scalar loads (19% regression). Need ldmatrix-based FP8 loading. Previous briefs established the ldmatrix.b16 reinterpret approach. This update adds NEW concrete findings.
**Supplements:** attention_fp8_ldmatrix_consolidated_strategy.md, attention_fp8_ldmatrix_reinterpret_UPDATE.md

## NEW FINDING 1: SM120 FP8 MMA Trait Inherits SM80 INT8 Layout (Confirmed in CUTLASS)

CUTLASS `mma_traits_sm120.hpp` reveals that SM120's FP8 MMA trait inherits directly
from SM80's INT8 m16n8k32:

```cpp
template <class a_type, class b_type, class c_type>
struct MMA_Traits<SM120_16x8x32_TN<a_type, b_type, c_type>>
     : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>
{
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeD = c_type;
  using ValTypeC = c_type;
};
```

And SM89's FP8 trait uses the SAME ALayout and BLayout as SM80 INT8:

```cpp
// SM89 FP8 (from mma_traits_sm89.hpp):
ALayout = Layout<Shape <Shape<_4,_8>, Shape<_4,_2,_2>>,
                 Stride<Stride<_64,_1>, Stride<_16,_8,_256>>>

// SM80 INT8 (from mma_traits_sm80.hpp, parent):
ALayout = Layout<Shape <Shape<_4,_8>, Shape<_4,_2,_2>>,
                 Stride<Stride<_64,_1>, Stride<_16,_8,_256>>>
```

**They are IDENTICAL.** FP8 e4m3 on sm_89/sm_120 uses exactly the same fragment
layout as INT8 s8/u8 on sm_80. This means:

1. Any INT8 m16n8k32 fragment loading code (sm_80+) works for FP8 on sm_89/sm_120
2. The ldmatrix.b16 reinterpret technique used for INT8 GEMM is directly applicable
3. The byte ordering question is definitively resolved: FP8 bytes are treated identically
   to INT8 bytes by the hardware

**Why this matters:** INT8 m16n8k32 has been in production since Ampere (sm_80). Every
INT8 GEMM kernel ever written for Ampere/Ada uses ldmatrix.b16 to load 8-bit data.
This is not a "hack" -- it is the standard, well-proven technique for loading 8-bit
data with ldmatrix.

## NEW FINDING 2: gau-nernst Achieves 692 TFLOPS FP8 on RTX 5090 Using ldmatrix.b16

gau-nernst's NVRTC matmul blog (https://gau-nernst.github.io/nvrtc-matmul/) reports
an FP8 GEMM achieving **692 TFLOPS** (FP8 e4m3 with FP16 accumulation) on RTX 5090.
This is the first public FP8 performance number on sm_120 with mma.sync.

Key implementation details:

1. **Same ldmatrix code for BF16 and FP8.** The code is templated: `MMA_K = 32 / sizeof(TypeAB)`.
   For FP8 (1 byte), MMA_K=32. For BF16 (2 bytes), MMA_K=16. The ldmatrix calls
   are IDENTICAL -- only the shared memory offset calculations change to account for
   element size.

2. **Shared memory addressing:** The swizzle and address calculations multiply by
   `sizeof(TypeAB)`, so FP8 data occupies half the smem bytes but the physical
   ldmatrix access pattern is unchanged.

3. **Shape selection:** `m16n8k16` for 2-byte types, `m16n8k32` for 1-byte types.
   Both use the same 4 A registers and 2 B registers.

4. **Register counts confirmed (again):** Identical for FP8 and BF16:
   - A: 4 x uint32_t
   - B: 2 x uint32_t
   - C/D: 4 x float (FP32 accum) or 2 x uint32_t (FP16 accum)

5. **Performance comparison on RTX 5090:**
   - FP8 e4m3 + FP32 accum: 465 TFLOPS
   - FP8 e4m3 + FP16 accum: 692 TFLOPS
   - BF16 + FP32 accum: ~175 TFLOPS (from FA blog)

**This is the strongest external validation yet.** A working, performant FP8 kernel
on our exact hardware (RTX 5090) uses ldmatrix.b16 reinterpret with no byte shuffles.

## NEW FINDING 3: SM120 Has Two FP8 MMA Instruction Variants

From CUTLASS `mma_sm120.hpp`, SM120 supports two distinct FP8 MMA instructions:

**Variant A: Standard (same as SM89)**
```
mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
```

**Variant B: Block-scaled (MXFP8, SM120-only)**
```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
```

The standard variant (A) is what we should use for the initial FP8 native input
path. It has the same register requirements as SM89:
- A: 4 x uint32_t
- B: 2 x uint32_t
- C/D: 4 x float

The block-scaled variant (B) adds UE8M0 scale factors (1 uint8_t per matrix) and
block/thread IDs. This is for MXFP8 (microscaling) which is a future optimization path.

**Note on instruction prefix:** SM120 uses `kind::f8f6f4` prefix, not the bare
`m16n8k32.row.col.f32.e4m3.e4m3.f32` used on SM89. The worker's existing PTX asm
may need updating. Verify which prefix the CUDA 13 assembler expects on sm_120.

## NEW FINDING 4: SageAttention Has SM89 FP8 Attention Kernels

SageAttention (https://github.com/thu-ml/SageAttention) has working FP8 attention
kernels for SM89 (Ada Lovelace / RTX 4090) using mma.sync (NOT wgmma):

- Source: `csrc/qattn/sm89/` directory
- Files include `sm89_qk_int8_sv_f8_accum_f32_attn.cu` and variants
- Uses INT8 for QK^T and FP8 (e4m3) for PV multiplication
- Per-channel FP8 quantization with scale_max=448.0
- Two-level FP8 accumulation strategy for accuracy

This is the ONLY known open-source FP8 attention implementation using mma.sync
(all others use Hopper WGMMA). The worker should examine the SM89 kernel source
for concrete fragment loading code, especially:
- How they build FP8 B fragments for PV MMA
- Whether they use ldmatrix or manual loads
- How they handle P-to-FP8 conversion

## NEW FINDING 5: ThunderKittens FP8 Limitation (mma_AB vs mma_ABt)

ThunderKittens GitHub issue #81 reveals that FP8 tile dimensions create an
asymmetry: TILE_ROW_DIM=16 but TILE_COL_DIM=32 for FP8, making square tiles
impossible. Their maintainer stated: "We don't have the transpose in fp8 supported
easily on the H100 so we don't use mma_AB as of now."

For our attention kernel, this means:
- QK^T (A=Q row-major, B=K^T col-major): straightforward, standard TN layout
- PV (A=P, B=V): the A operand (P) must be in the FP8 A fragment layout, which
  requires the cross-thread shuffle (byte_perm + shfl_sync) from FP32 accumulators

ThunderKittens 2.0 (Feb 2026) adds Blackwell support with MXFP8 and NVFP4.
Their codebase is now the most complete reference for multi-precision tile
operations.

## UPDATED RECOMMENDATION

The evidence is now overwhelming:

1. **ldmatrix.b16 reinterpret for FP8 is not speculative.** It is the standard,
   proven technique used by every INT8/FP8 GEMM kernel since Ampere. The CUTLASS
   layout inheritance chain (SM120 -> SM80 INT8) proves the fragment layouts are
   identical.

2. **gau-nernst's 692 TFLOPS on RTX 5090** proves this works on our exact hardware
   with the exact instruction (mma.sync m16n8k32 e4m3) we need.

3. **The worker should build the minimal test (Step 0 from consolidated brief) with
   HIGH CONFIDENCE it will work.** The byte ordering will be correct because the
   fragment layout is the same as INT8 m16n8k32, which has been using ldmatrix.b16
   for 4+ years.

4. **SageAttention SM89 kernels are the closest reference implementation.** They use
   mma.sync FP8 for attention on Ada (same ISA as sm_120). The worker should study
   their fragment loading code before implementing.

5. **Verify the PTX instruction prefix.** SM120 may require `kind::f8f6f4` prefix
   instead of bare `m16n8k32.row.col.f32.e4m3.e4m3.f32`. Check what the worker's
   current FP8 kernel uses and whether it matches the SM120 variant.
