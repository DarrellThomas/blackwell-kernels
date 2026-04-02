# CUTLASS FP8/INT8 Shared Memory Loading: Implementation-Level Details

**Sources:**
- https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm89.h (SM89 FP8 MMA PTX assembly, fragment types)
- https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/mma_sm80.h (SM80 INT8 MMA PTX assembly, fragment types)
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm89.hpp (SM89 FP8 ALayout/BLayout definitions)
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm80.hpp (SM80 INT8 ALayout/BLayout definitions)
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm120.hpp (SM120 MMA atoms, block-scaled FP8)
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm100.hpp (SM100 ldmatrix.m16n16.b8 PTX)
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm75.hpp (SM75 ldmatrix.m8n8.b16 -- NO 8-bit variants)
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm75.hpp (Copy traits -- only U32 and U16 variants, no U8)
- https://github.com/NVIDIA/cutlass/discussions/647 (NVIDIA maintainer: "ldmatrix can only transpose 16-bit data")
- https://github.com/NVIDIA/cutlass/discussions/911 (INT8 interleaved layout, row permutation for ldmatrix)
- https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254 (SM120 FP8 ldmatrix forum thread)
- https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/layout/tensor_op_multiplicand_sm75.h (8-bit shared memory layout, XOR swizzle)
- https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h (Generic ldsm invocation)
- https://github.com/NVIDIA/cutlass/blob/main/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu (Example 58: Ada FP8 tile config)
- https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu (Example 79c: SM120 MXFP8)
- https://github.com/vllm-project/vllm/pull/17280 (vLLM SM120 FP8 CUTLASS kernel)

**Relevant to:** gemm worker, attention worker, fused-mlp worker (all FP8)
**Worker's current problem:** Workers need to understand how CUTLASS loads FP8/INT8 data from shared memory into MMA fragments. Specifically: does it use ldmatrix.b16 and reinterpret, or something else? What is the exact smem layout?

---

## FINDING 1: SM89 FP8 and SM80 INT8 Have IDENTICAL Fragment Layouts

This is the most important discovery. CUTLASS defines the exact same ALayout and BLayout
for SM89 FP8 m16n8k32 and SM80 INT8 m16n8k32:

**SM89 FP8 (e4m3 x e4m3 -> f32):**
```cpp
// From cute/atom/mma_traits_sm89.hpp
struct MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN> {
  using Shape_MNK = Shape<_16,_8,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_4,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_8,_128>>>;
  using CLayout = SM80_16x8_Row;
};
```

**SM80 INT8 (s8 x s8 -> s32):**
```cpp
// From cute/atom/mma_traits_sm80.hpp
struct MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> {
  using Shape_MNK = Shape<_16,_8,_32>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _4,_2,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_16,_8,_256>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_4,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_8,_128>>>;
  // CLayout differs (int32 vs float32 accumulator), but A/B are identical
};
```

**Why this matters:** FP8 is NOT special. It uses the exact same fragment-to-register
mapping as INT8. Whatever loading technique works for INT8 m16n8k32 will work
identically for FP8 m16n8k32. INT8 has been supported since Turing (sm_75) with
extensive optimization in CUTLASS. All that knowledge transfers directly.

---

## FINDING 2: CUTLASS Uses ldmatrix.b16 for ALL 8-bit Types (SM75-SM89)

CUTLASS's warp-level tile iterator (`mma_tensor_op_tile_iterator.h`) uses a single
generic `ldsm` (load shared matrix) function that always invokes `ldmatrix.m8n8.b16`:

```cpp
// From include/cutlass/arch/memory_sm75.h
// RowMajor, x4 variant:
asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
  : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
```

There is **no 8-bit specialization** in the SM75 copy atoms. The file
`copy_sm75.hpp` defines ONLY these ldmatrix variants:
- `SM75_U32x1_LDSM_N` (non-transposed, 1 register)
- `SM75_U32x2_LDSM_N` (non-transposed, 2 registers)
- `SM75_U32x4_LDSM_N` (non-transposed, 4 registers)
- `SM75_U16x2_LDSM_T` (transposed, 1 register via 2x16-bit)
- `SM75_U16x4_LDSM_T` (transposed, 2 registers)
- `SM75_U16x8_LDSM_T` (transposed, 4 registers)

All use `ldmatrix.m8n8.shared.b16`. No `b8` variants exist at the SM75 level.

**Implication:** For non-block-scaled FP8 on sm_89/sm_120, CUTLASS uses
ldmatrix.m8n8.x4.b16 to load 8-bit data by treating pairs of 8-bit elements
as single 16-bit values. The hardware doesn't care about the actual data type --
ldmatrix moves bits, not typed values.

---

## FINDING 3: The Exact PTX Register Mapping for FP8 m16n8k32

From `mma_sm89.h`, the FP8 MMA PTX assembly:

```ptx
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
  {%0, %1, %2, %3},         // D output: 4 float registers
  {%4, %5, %6, %7},         // A input: 4 uint32_t registers (16 FP8 values)
  {%8, %9},                 // B input: 2 uint32_t registers (8 FP8 values)
  {%10, %11, %12, %13};     // C accumulator: 4 float registers
```

Fragment sizes from the struct:
- **FragmentA** = `Array<float_e4m3_t, 16>` = 16 bytes = 4 x uint32_t registers
- **FragmentB** = `Array<float_e4m3_t, 8>` = 8 bytes = 2 x uint32_t registers
- **FragmentC** = `Array<float, 4>` = 16 bytes = 4 x float registers

Compare with BF16 m16n8k16:
- **FragmentA** = `Array<bfloat16_t, 8>` = 16 bytes = 4 x uint32_t registers
- **FragmentB** = `Array<bfloat16_t, 4>` = 8 bytes = 2 x uint32_t registers

**Same number of registers, same byte count.** The only difference is the
interpretation: BF16 packs 2 values per 32-bit register, FP8 packs 4 values
per 32-bit register. ldmatrix.x4.b16 loads exactly 4 x 32 bits = 16 bytes
of A-operand data, which is 8 BF16 values or 16 FP8 values.

---

## FINDING 4: SM120 Block-Scaled FP8 MMA Atoms

SM120 adds block-scaled FP8 MMA, which is a different instruction from the
"plain" FP8 MMA. From `mma_sm120.hpp`:

```cpp
// SM120 supports BOTH plain FP8 and block-scaled FP8:

// Plain (same as SM89):
struct SM120_16x8x32_TN<float, float_e4m3_t, float_e4m3_t, float> {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  // PTX: mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
};

// Block-scaled (SM120 exclusive):
namespace SM120::BLOCKSCALED {
  struct SM120_16x8x32_TN<float, float_e4m3_t, float_e4m3_t, float> {
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];
    // ADDITIONAL: sfa0 (uint8_t scale), sfb0 (uint8_t scale)
    // ADDITIONAL: bidA, tidA, bidB, tidB (block/thread IDs for scaling)
    // PTX: mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32...
  };
}
```

**Key point:** The A/B register counts are IDENTICAL (4 + 2 uint32_t) for both
plain and block-scaled FP8. The loading mechanism is the same. Block-scaling adds
scale factor registers but doesn't change how operand data is loaded from smem.

---

## FINDING 5: CUTLASS Shared Memory Layout for 8-bit Types

From `tensor_op_multiplicand_sm75.h`, the 8-bit smem layout uses:

- **kAccessSize = 128 bits** (same as all types -- this is the cache line width)
- **kElementsPerAccess = 16** (128 bits / 8 bits = 16 FP8 elements per 128-bit load)
- **kTileShapeContiguous = 8** (8 accesses form the contiguous tile dimension)
- **XOR swizzle:** `new_col = original_row XOR original_col` applied to an 8x8 basic
  block of 128-bit accesses

The row pitch depends on kCrosswise (set by the template instantiation, typically
16 or 32 elements for 8-bit). For k=32 (as in m16n8k32):

- Each 128-bit access holds 16 FP8 elements
- 2 accesses span 32 elements (one k-step)
- With XOR swizzle on the column index, bank conflicts are avoided

The TensorOpMultiplicandCongruous<8, kCrosswise> layout stores data such that
ldmatrix can load it directly. The 128-bit alignment guarantees the 16-byte
alignment requirement of ldmatrix is satisfied.

---

## FINDING 6: The NVIDIA Maintainer's Definitive Statement on ldmatrix and 8-bit

From CUTLASS Discussion #647, an NVIDIA engineer states:

> "ldmatrix can only transpose 16bit data."

This is a fundamental hardware constraint. For NN-layout GEMM with 8-bit:
- **ldmatrix.b16 without .trans** works fine for loading row-major A and col-major B
  (the standard TN layout used by mma.sync)
- **ldmatrix.b16 with .trans** transposes at 16-bit granularity, which swaps pairs
  of 8-bit values rather than individual values -- this is wrong for 8-bit transpose
- **Workaround for NN layout:** Use non-tensor-core path, or pre-transpose in smem

For our kernels using TN layout (row-major A, column-major B), this constraint
does NOT apply. We use ldmatrix.x4.b16 without .trans for A, and ldmatrix.x2.b16
with .trans (or without, depending on layout) for B.

---

## FINDING 7: CUTLASS Example 58 (Ada FP8) Tile Configuration

The Ada FP8 GEMM example uses:
```cpp
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;     // A is row-major
using LayoutB = cutlass::layout::ColumnMajor;  // B is column-major (TN layout)

cutlass::gemm::GemmShape<128, 64, 128>    // ThreadBlock tile
cutlass::gemm::GemmShape<64, 32, 128>     // Warp tile
cutlass::gemm::GemmShape<16, 8, 32>       // Instruction (m16n8k32)
```

**Note the k-dimension:** ThreadBlock k=128, Warp k=128, Instruction k=32.
This means 4 MMA instructions per k-iteration (128/32 = 4). Each MMA loads
16 FP8 values for A (4 registers) and 8 for B (2 registers).

---

## FINDING 8: SM120 ldmatrix.m16n16.b8 Exists But Is Limited

From `copy_sm100.hpp`, SM120 (via SM100 ISA inheritance) has native 8-bit ldmatrix:

```ptx
ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8   // 2 output registers, 16x16 tile
ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8   // 4 output registers, 16x32 tile
```

**Limitations:**
1. **Transposed only** -- no non-transposed variant. Data must be column-major in smem.
2. **Requires post-load byte interleave** -- CUTLASS applies uchar4 byte rearrangement
   after loading (see previous brief for the exact PRMT pattern).
3. **16x16 tile shape** -- different from the m8n8 shape used by ldmatrix.b16.

**For our kernels:** ldmatrix.m8n8.x4.b16 is strictly superior for non-block-scaled
FP8 because it has both trans and non-trans variants, requires no post-load shuffle,
and is the same instruction we already use for BF16. The m16n16.b8 variant is primarily
useful for CUTLASS's block-scaled MMA pipeline where the smem layout is already
column-major for the scaling factor application.

---

## SYNTHESIS: How to Load FP8 from Shared Memory for mma.sync m16n8k32

### Step-by-step recipe (confirmed from CUTLASS source):

1. **Store FP8 data in smem** using TensorOpMultiplicandCongruous<8, 32> layout
   (row-major with XOR swizzle). Each 128-bit cache line holds 16 FP8 values.

2. **Use ldmatrix.sync.aligned.m8n8.x4.shared.b16** to load A operand:
   - Each thread provides a 16-byte-aligned smem address
   - Returns 4 x uint32_t registers = 16 FP8 bytes per thread
   - The warp collectively loads a 16x32 tile of FP8 values

3. **Use ldmatrix.sync.aligned.m8n8.x2.shared.b16** to load B operand:
   - Returns 2 x uint32_t registers = 8 FP8 bytes per thread
   - The warp collectively loads an 8x32 tile of FP8 values

4. **Feed directly to MMA:**
   ```ptx
   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
     {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
   ```

5. **No post-load shuffle needed** for TN layout (row-major A, col-major B).
   The ldmatrix.b16 register layout naturally matches the MMA expectation because
   both FP8 and BF16 use the same number of 32-bit registers for their fragments.

### What this means for our workers:

- **For K/V loading in attention:** Store KV in smem with same XOR swizzle as BF16,
  ldmatrix.x4.b16 into registers, feed to m16n8k32. No conversion overhead.

- **For P->FP8 conversion in attention:** After softmax produces FP32 accumulator,
  convert FP32->FP8 in registers (CVT instructions), then apply byte_perm + shfl_sync
  to rearrange from CLayout to ALayout format.

- **For GEMM:** Identical to BF16 loading pipeline, just use m16n8k32 instead of
  m16n8k16. Shared memory capacity doubles (same tiles, half the element size).

## Caveats

1. **The "no post-load shuffle" claim assumes TN layout.** If either operand needs
   to be transposed, ldmatrix.b16.trans will swap pairs of FP8 values which is wrong.
   Use pre-transposed data in smem instead.

2. **The shared memory layout is different from BF16.** Even though ldmatrix.b16 is
   used for both, the kElementsPerAccess is 16 (vs 8 for BF16). The swizzle works at
   128-bit granularity regardless, but the logical row pitch differs.

3. **CUTLASS's "no specialization" approach means the generic tile iterator handles
   FP8 automatically** via sizeof_bits<Element> computations. Workers implementing
   custom kernels should follow the same pattern: compute all pointer arithmetic in
   terms of bytes/bits, not element counts.
