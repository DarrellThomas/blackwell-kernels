# FP8 ldmatrix: Verified Source Code from ThunderKittens + CUTLASS CuTe Layouts

**Sources:**
- ThunderKittens source: https://github.com/HazyResearch/ThunderKittens (cloned and read directly)
- CUTLASS CuTe: https://github.com/NVIDIA/cutlass include/cute/atom/mma_traits_sm89.hpp, include/cute/arch/mma_sm89.hpp, include/cute/arch/copy_sm100.hpp
- ThunderKittens blog: https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8
- NVIDIA forums: https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
- Lei Mao ldmatrix blog: https://leimao.github.io/blog/CuTe-ldmatrix/
- Colfax FP8 FA-2: https://research.colfax-intl.com/adding-fp8-to-flashattention/

**Relevant to:** attention worker (FP8 kernel), GEMM worker
**Worker's current problem:** Exp 66 used scalar uint16 loads for FP8 inputs from smem and was 19% slower (62 us vs 52 us). Need ldmatrix-class throughput for FP8 data.
**Status:** Supplements prior briefs with VERIFIED SOURCE CODE findings.

## What This Brief Adds

Previous briefs established the strategy (ldmatrix.m8n8.x4.b16 reinterpret approach).
This brief provides the **verified, line-by-line source code** from ThunderKittens showing
exactly how it works. I cloned the ThunderKittens repo and read the actual implementation.

---

## VERIFIED FINDING 1: ThunderKittens Uses Identical ldmatrix PTX for FP8 and BF16

File: `include/ops/thread/util/util.cuh`, lines 170-190

ThunderKittens defines FP8 ldmatrix as:

```cpp
template<> struct move<fp8e4m3_4> {
    __device__ static inline void ldsm4(
        fp8e4m3_4& dst1, fp8e4m3_4& dst2,
        fp8e4m3_4& dst3, fp8e4m3_4& dst4, uint32_t src) {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2),
              "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4)
            : "r"(src));
    }
};
```

Compare to the BF16 version (lines 94-97):

```cpp
template<> struct move<bf16_2> {
    __device__ static inline void ldsm4(
        bf16_2& dst1, bf16_2& dst2,
        bf16_2& dst3, bf16_2& dst4, uint32_t src) {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2),
              "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4)
            : "r"(src));
    }
};
```

**The PTX assembly is BYTE-FOR-BYTE IDENTICAL.** The only difference is the C++ type
of the destination variables. Both use `ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16`.
Both load 4 x uint32_t registers. The hardware does not distinguish between the types --
it just moves 128 bits.

**CRITICAL:** There is NO post-load byte shuffle, NO PRMT, NO reordering in the FP8
ldmatrix path in ThunderKittens. The loaded registers go directly to MMA.

---

## VERIFIED FINDING 2: ThunderKittens FP8 MMA Uses Same Register Pattern as BF16

File: `include/ops/group/mma/warp.cuh`, lines 159-186

The FP8 MMA function:

```cpp
__device__ static inline void hmma16816(
    float2 &d0, float2 &d1,
    const fp8e4m3_4 &a0, const fp8e4m3_4 &a1,
    const fp8e4m3_4 &a2, const fp8e4m3_4 &a3,
    const fp8e4m3_4 &b0, const fp8e4m3_4 &b1,
    const float2 &c0, const float2 &c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "+f"(d0.x), "+f"(d0.y), "+f"(d1.x), "+f"(d1.y)
        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
          "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
          "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
          "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y));
}
```

**Register signature is identical to BF16 m16n8k16:**
- A: 4 x uint32_t (a0..a3) -- same as BF16
- B: 2 x uint32_t (b0, b1) -- same as BF16
- C/D: 4 x float32 -- same as BF16

And in `mma_AB_base` for FP8 (lines 258-274), the operand mapping is:

```
A operand: a.data[0], a.data[1], a.data[2], a.data[3]
B operand for 1st MMA: b.data[0], b.data[2]
B operand for 2nd MMA: b.data[1], b.data[3]
```

This is EXACTLY the same pattern as BF16. The rt_base for FP8 has `data[4]` entries
(4 x fp8e4m3_4 = 4 x uint32_t), identical register layout as BF16's `data[4]`
(4 x bf16_2 = 4 x uint32_t).

---

## VERIFIED FINDING 3: FP8 Shared-to-Register Load Thread Mapping

File: `include/ops/group/memory/tile/shared_to_register.cuh`, lines 52-69

For 8-bit types (sizeof(ST::dtype) == 1), row-major layout:

```cpp
int warp_group_16 = (warp_laneid / 16);   // 0 or 1 (split warp into two halves)
int lane_in_16 = warp_laneid % 16;        // 0..15 within each half
int row = base_row + (lane_in_16 % 16);   // each thread covers one row (0..15)
int col = base_col + warp_group_16 * 16;  // first half: cols 0..15, second half: cols 16..31
```

Then loads using ldmatrix.x4.b16 (via `move<U2>::ldsm4`):

```cpp
move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row, col}));
dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
```

**Key details:**
- The FP8 base tile is 16x32 elements (TILE_ROW_DIM=16, TILE_COL_DIM=32 when sizeof(T)==1)
- The warp is split into two groups of 16 threads
- Each group of 16 threads covers one 16x16 half of the 16x32 tile
- Each thread addresses a unique row (0..15), contributing its address for ldmatrix
- ldmatrix.x4.b16 loads 4 x m8n8 sub-matrices per call from that group
- The convertor between fp8e4m3_4 types is a no-op (identity) -- just reinterpret_cast

**This means the shared memory layout for FP8 is:**
- Row-major FP8 bytes
- Each row has 32 consecutive FP8 bytes
- Swizzled with the same XOR pattern as BF16 (but using FP8 swizzle_bytes calculation)
- ldmatrix loads treat pairs of consecutive FP8 bytes as 16-bit "elements"

---

## VERIFIED FINDING 4: FP8 Tile Dimensions and Swizzle

File: `include/common/util.cuh`, line 62:
```cpp
template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? BASE_TILE_DIM * 2 : BASE_TILE_DIM;
```

So for FP8: TILE_COL_DIM = 32 (vs 16 for BF16).
TILE_ROW_DIM is always 16.

File: `include/types/shared/st.cuh`, lines 91-94 (swizzle_bytes for 8-bit types):
```cpp
sizeof(dtype) == 1 ? (
    (cols/kittens::TILE_COL_DIM<T>)%4 == 0 ? 128 :
    (cols/kittens::TILE_COL_DIM<T>)%2 == 0 ?  64 : 32
)
```

For a single 16x32 FP8 tile: cols/TILE_COL_DIM = 32/32 = 1, so swizzle_bytes = 32.
For a 16x64 FP8 tile: 64/32 = 2, so swizzle_bytes = 64.
For a 16x128 FP8 tile: 128/32 = 4, so swizzle_bytes = 128 (zero bank conflicts).

The swizzle pattern in idx() is the same XOR-based pattern as BF16:
```cpp
const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
return *(T*)((uint64_t)(addr) ^ swizzle);
```

**Implication for our kernel:** If our FP8 K tiles are 16x64 (d_head=64), we get
64-byte swizzle (2-way bank conflicts). For 16x128 we get 128-byte swizzle (zero conflicts).
With our 64x64 KV blocks that are loaded as multiple 16x32 FP8 tiles, 32-byte swizzle
applies unless we widen the tile layout.

---

## VERIFIED FINDING 5: Register Tile Element Counts for FP8

File: `include/types/register/rt_base.cuh`, lines 70-78:

```cpp
static constexpr int tile_size_row        = TILE_ROW_DIM<T>;    // 16
static constexpr int tile_size_col        = TILE_COL_DIM<T>;    // 32 for FP8
static constexpr int num_elements         = rows*cols;           // 512 for FP8
static constexpr int elements_per_thread  = num_elements / 32;  // 16 for FP8
static constexpr int packed_per_thread    = elements_per_thread / packing::num(); // 16/4 = 4
static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4; // 4*4/4 = 4
```

Each thread holds:
- 16 FP8 values
- Packed into 4 x fp8e4m3_4 (4 x uint32_t)
- = 4 registers

**This is the exact same register count as BF16** (4 x bf16_2 = 4 x uint32_t = 4 registers),
each holding the same number of bytes (16 bytes), just with different element counts
(8 BF16 values vs 16 FP8 values).

---

## SYNTHESIS: What the Worker Needs to Do

The ThunderKittens source code PROVES that the ldmatrix.b16 reinterpret approach works
with NO post-load fixup. The complete data path is:

1. **Store FP8 data in smem** as contiguous bytes, row-major, with XOR swizzle
   - Same swizzle pattern as BF16 but byte-level (swizzle_bytes may differ)
   - Each row of the 16x32 FP8 tile is 32 bytes

2. **Compute smem address** for ldmatrix:
   - Split warp into two groups of 16 threads (lanes 0-15, lanes 16-31)
   - Each thread addresses row = lane_in_16, column = group_16 * 16
   - Apply swizzle to the address (same XOR pattern as BF16)

3. **Issue ldmatrix.sync.aligned.m8n8.x4.b16**:
   - Same PTX instruction as BF16
   - Produces 4 x uint32_t per thread
   - Each uint32_t holds 4 FP8 values

4. **Feed registers directly to mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32**:
   - A operand: 4 registers (a0..a3) -- from ldmatrix for Q or from P-to-FP8 conversion
   - B operand: 2 registers (b0, b1) -- from ldmatrix for K or V
   - No byte reordering between ldmatrix and MMA

5. **The only tricky part is P-to-FP8 conversion** (softmax output to A operand):
   - FP32 accumulator layout differs from FP8 A operand layout
   - Requires byte_perm + shfl_sync (see Colfax pattern in prior briefs)
   - Exact shuffle maps for mma.sync need derivation

**Bottom line:** Replace the scalar FP8 loads in exp 66 with
`ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16` using the same PTX you already
use for BF16. The only change needed is the shared memory layout (FP8 bytes instead
of BF16 words) and the address calculation (thread mapping matches the 16x32 FP8 tile).

---

## CAVEATS

1. **ThunderKittens targets H100 (sm_90) with KITTENS_HOPPER.** The warp.cuh MMA
   functions (hmma16816 for FP8) are gated behind `#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)`.
   sm_120 (RTX 5090) qualifies as KITTENS_BLACKWELL. The PTX instructions are the same
   across sm_89/sm_90/sm_120 for mma.sync (not wgmma). Our kernel already uses these
   PTX instructions successfully.

2. **ThunderKittens uses a warpgroup-level abstraction** that distributes tiles across
   multiple warps. Our kernel uses single-warp loads. The ldmatrix call itself is
   warp-level (32 threads) and identical in both cases.

3. **The convertor<fp8e4m3_4, fp8e4m3_4> is identity.** When the source and destination
   types match (loading FP8 from FP8 smem into FP8 registers), no conversion happens.
   This is the zero-overhead path we want.

4. **FP8 base tile is 16x32 (not 16x16 like BF16).** The MMA instruction m16n8k32
   processes 32 values along k, so each base tile naturally has k=32. This doubles the
   k-dimension coverage per tile compared to BF16 m16n8k16.
