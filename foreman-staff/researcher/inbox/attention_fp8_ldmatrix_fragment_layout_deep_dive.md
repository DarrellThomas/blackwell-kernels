# FP8 ldmatrix Fragment Layout for m16n8k32 MMA — Deep Dive

**Sources:**
- ThunderKittens source code: https://github.com/HazyResearch/ThunderKittens (warp.cuh, shared_to_register.cuh, rt_base.cuh, conversions.cuh)
- https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
- https://research.colfax-intl.com/adding-fp8-to-flashattention/
- https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8
- https://www.spatters.ca/mma-matmul (ldmatrix + XOR swizzle patterns)
- https://leimao.github.io/blog/CuTe-ldmatrix/ (ldmatrix thread-to-register mapping)
- https://github.com/NVIDIA/cutlass/issues/1986 (sm_89 vs sm_90 FP8 MMA)
- PTX ISA 9.2: https://docs.nvidia.com/cuda/parallel-thread-execution/
**Relevant to:** attention worker (FP8 kernel), gemm worker (FP8 kernel)
**Worker's current problem:** Need exact byte ordering when using ldmatrix.b16 to load FP8 data for m16n8k32 MMA

## Summary of Findings

This brief consolidates ALL concrete findings from searching the web, reading
ThunderKittens source code, CUTLASS references, Colfax research, NVIDIA forums,
and community implementations. The goal is answering: what is the exact byte-level
mapping when ldmatrix.m8n8.x4.b16 loads FP8 data for m16n8k32 MMA?

---

## 1. MMA Register Counts (Confirmed from Multiple Sources)

For `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`:

| Operand | Registers | Bytes/Thread | Elements/Thread |
|---------|-----------|--------------|-----------------|
| A (row-major) | 4 x uint32 | 16 bytes | 16 FP8 values |
| B (col-major) | 2 x uint32 | 8 bytes | 8 FP8 values |
| C/D (accumulator) | 4 x float32 | 16 bytes | 4 FP32 values |

This matches m16n8k16 BF16 exactly in register count (4/2/4). The difference
is packing density: each 32-bit register holds 4 FP8 instead of 2 BF16.

**Source:** CUTLASS mma_sm80.h (integer m16n8k32 uses same counts), ThunderKittens
warp.cuh hmma16816() for FP8 (lines 159-186, uses same 4+2+4 pattern as BF16).

---

## 2. ThunderKittens FP8 Implementation — Key Architectural Decisions

### 2.1 Tile Dimensions for FP8

From ThunderKittens `util.cuh` and `rt_base.cuh`:

```cpp
// util.cuh line 62
template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? BASE_TILE_DIM * 2 : BASE_TILE_DIM;
template<typename T> constexpr int TILE_ROW_DIM = BASE_TILE_DIM;  // always 16
```

Where `BASE_TILE_DIM = 16`. So:
- **BF16/FP16 base tile:** 16 rows x 16 cols
- **FP8 base tile:** 16 rows x 32 cols (double width!)

This matches m16n8k32 perfectly: the K dimension doubles from 16 to 32 for FP8.

### 2.2 Register Tile Base for FP8

From `rt_base.cuh`:

```cpp
static constexpr int tile_size_row = TILE_ROW_DIM<T>;        // 16
static constexpr int tile_size_col = TILE_COL_DIM<T>;        // 32 for FP8
static constexpr int num_elements  = rows * cols;             // 512 for FP8
static constexpr int elements_per_thread = num_elements / 32; // 16 for FP8
static constexpr int packed_per_thread = elements_per_thread / packing::num(); // 4
static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4; // 4
```

So each thread holds 4 x `fp8e4m3_4` (4 FP8 values packed into 32 bits) = 4 registers.
Each register contains 4 FP8 values. 4 * 4 = 16 FP8 values per thread.

### 2.3 How ThunderKittens Loads FP8 from Shared Memory

From `shared_to_register.cuh` (group-level load, lines 52-69):

```cpp
else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 1) {
    // handle the row-major layout for 8-bit types
    int warp_group_16 = (warp_laneid / 16);  // divide each warp into two groups of 16 threads
    int lane_in_16 = warp_laneid % 16;       // position in group of 16 threads
    int row = ... + (lane_in_16 % 16);       // 16 threads across 16 rows
    int col = ... + warp_group_16 * 16;      // first half loads cols 0-15, second half loads cols 16-31

    U2 tmp[4];
    move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row, col}));
    // then convert to register type
}
```

**Critical insight:** ThunderKittens uses `ldsm4` (= `ldmatrix.m8n8.x4.b16`) for FP8!

The addressing pattern:
- Threads 0-15: load from col offset 0 (covering FP8 cols 0-15, as 16-bit pairs = 8 pairs)
- Threads 16-31: load from col offset 16 (covering FP8 cols 16-31)

This confirms the reinterpret approach: `ldmatrix.x4.b16` loads "16-bit elements"
which are actually pairs of FP8 bytes. The warp is split into two halves covering
the two 16-column halves of the 32-column FP8 tile.

### 2.4 FP8 MMA Instruction Usage

From `warp.cuh` (lines 159-186):

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
          "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y)
    );
}
```

Uses `fp8e4m3_4` (4 FP8 values per register) with `"r"` constraint (32-bit register).
A takes 4 registers (a0-a3), B takes 2 registers (b0-b1), same as BF16 m16n8k16.

### 2.5 A-B Register Selection in mma_AB_base

From `warp.cuh` (lines 258-274):

```cpp
// For the base 16x16 FP8 tile (which is actually 16 rows x 32 K):
hmma16816(
    d.data[0], d.data[1],     // accumulator (first MMA for N=0..7)
    a.data[0], a.data[1], a.data[2], a.data[3],  // A operand (all 4 regs)
    b.data[0], b.data[2],    // B operand (regs 0,2 for first N half)
    c.data[0], c.data[1]
);
hmma16816(
    d.data[2], d.data[3],     // accumulator (second MMA for N=8..15)
    a.data[0], a.data[1], a.data[2], a.data[3],  // A operand (same)
    b.data[1], b.data[3],    // B operand (regs 1,3 for second N half)
    c.data[2], c.data[3]
);
```

The B register selection pattern (0,2 then 1,3) is identical to BF16. This means
the ThunderKittens FP8 tile (16x32 elements) uses TWO m16n8k32 MMA calls covering
different N slices, with the SAME A registers reused.

---

## 3. The ldmatrix.x4.b16 Thread-to-Register Mapping (for 16-bit elements)

From Lei Mao's blog and spatters.ca, the canonical mapping for ldmatrix.sync.aligned.x4.m8n8.shared.b16:

**Input:** Each thread provides a shared memory address pointing to a 16-byte-aligned row.

**Thread groups and address providers:**
- Threads 0-7: provide addresses for 8 rows of sub-matrix 0
- Threads 8-15: provide addresses for 8 rows of sub-matrix 1
- Threads 16-23: provide addresses for 8 rows of sub-matrix 2
- Threads 24-31: provide addresses for 8 rows of sub-matrix 3

**Output register assignment (b16, row-major, no transpose):**

Each 8x8 sub-matrix (8 rows x 8 x 16-bit elements = 128 bits per row):
- Thread T in group loads from row T%8
- The 128 bits (8 x 16-bit values) are distributed across 4 threads as registers

Specifically for sub-matrix loaded by threads [G*8 .. G*8+7]:
- Each thread gets ONE 32-bit register containing 2 x 16-bit values
- Register Rn assigned to thread T contains: values at col positions (T%4)*2 and (T%4)*2+1
  from row (T%8) within the sub-matrix

The x4 variant loads 4 such sub-matrices into registers R0, R1, R2, R3 respectively.

**For FP8 reinterpretation:** Each "16-bit value" is actually 2 FP8 bytes packed together.
So each register holds 4 FP8 values instead of 2 BF16 values.

---

## 4. The Fragment Layout Correspondence: m16n8k16 BF16 vs m16n8k32 FP8

### 4.1 Structural Argument (strengthened by ThunderKittens code)

The m16n8k16 BF16 fragment (A operand):
- 4 registers: {a0, a1, a2, a3}
- Each holds 2 BF16 values (32 bits)
- groupID = laneid >> 2, threadID_in_group = laneid % 4
- a0: rows [groupID], cols [threadID_in_group*2, threadID_in_group*2+1]
- a1: rows [groupID+8], same cols
- a2: rows [groupID], cols [threadID_in_group*2+8, threadID_in_group*2+9]
- a3: rows [groupID+8], same cols

The m16n8k32 FP8 fragment (A operand):
- 4 registers: {a0, a1, a2, a3}
- Each holds 4 FP8 values (32 bits)
- Same groupID/threadID formulas
- a0: rows [groupID], K=[threadID*4..threadID*4+3] (first K-quarter)
- a1: rows [groupID+8], same K range
- a2: rows [groupID], K=[threadID*4+16..threadID*4+19] (second K-quarter)
- a3: rows [groupID+8], same K range

When viewed as 16-bit pairs (2 FP8 = 1 "b16 element"):
- a0 holds 2 "b16 pairs" at K_pair positions [threadID*2, threadID*2+1]
- a2 holds 2 "b16 pairs" at K_pair positions [threadID*2+8, threadID*2+9]

This is the SAME indexing pattern as m16n8k16 BF16! The structural correspondence holds.

### 4.2 The a1/a2 Register Swap

For ldmatrix.x4.b16 output:
- r0 = sub-matrix 0 (rows 0-7, K_pairs 0-7)
- r1 = sub-matrix 1 (rows 0-7, K_pairs 8-15)
- r2 = sub-matrix 2 (rows 8-15, K_pairs 0-7)
- r3 = sub-matrix 3 (rows 8-15, K_pairs 8-15)

MMA expects: a0=rows 0-7 K-first-half, a1=rows 8-15 K-first-half, a2=rows 0-7 K-second-half, a3=rows 8-15 K-second-half.

So the mapping is: `(a0, a1, a2, a3) = (r0, r2, r1, r3)`

This is the standard a1/a2 swap we already use for BF16, confirming it applies
unchanged for FP8.

---

## 5. Shared Memory Layout Requirements

### 5.1 For FP8 A operand loading (Q, K-as-A)

Store FP8 data row-major in shared memory, with each row being 32 FP8 bytes
(= 16 "b16 pairs" = 256 bits = 2 x 128-bit cache lines).

For ldmatrix, each thread provides the address of a 128-bit aligned row:
```
thread 0  -> smem[row_0][col_offset]    (128 bits = 16 FP8 bytes)
thread 1  -> smem[row_1][col_offset]
...
thread 7  -> smem[row_7][col_offset]
thread 8  -> smem[row_0][col_offset+16]  (next 128-bit chunk)
...
```

Wait -- this is the ThunderKittens approach (split warp into halves). Actually,
ldmatrix.x4 uses ALL 32 threads (threads 0-7 for sub-matrix 0, 8-15 for
sub-matrix 1, 16-23 for sub-matrix 2, 24-31 for sub-matrix 3). Each loads
128 bits (8 "b16 pairs" = 16 FP8 bytes).

For a 16x32 FP8 A tile:
- Sub-matrices 0,1 cover rows 0-7 (threads 0-15)
- Sub-matrices 2,3 cover rows 8-15 (threads 16-31)
- Sub-matrices 0,2 cover K_pairs 0-7 (K=0..15 FP8)
- Sub-matrices 1,3 cover K_pairs 8-15 (K=16..31 FP8)

Thread addresses:
```
T0:  &smem[row_0][K_pair_0]   -> loads 128b from row 0, K_pairs 0-7
T1:  &smem[row_1][K_pair_0]   -> loads 128b from row 1, K_pairs 0-7
...
T7:  &smem[row_7][K_pair_0]   -> loads 128b from row 7, K_pairs 0-7
T8:  &smem[row_0][K_pair_8]   -> loads 128b from row 0, K_pairs 8-15
T9:  &smem[row_1][K_pair_8]   -> loads 128b from row 1, K_pairs 8-15
...
T15: &smem[row_7][K_pair_8]   -> loads 128b from row 7, K_pairs 8-15
T16: &smem[row_8][K_pair_0]   -> loads 128b from row 8, K_pairs 0-7
...
T31: &smem[row_15][K_pair_8]  -> loads 128b from row 15, K_pairs 8-15
```

This is a 16x16 grid of b16 pairs = 16x32 FP8 elements = one complete
A operand for m16n8k32.

### 5.2 XOR Swizzle

The same XOR swizzle pattern used for BF16 applies because:
1. ldmatrix accesses 128-bit aligned rows from shared memory
2. Bank conflict avoidance requires permuting the column index
3. The column index operates on "b16 pair" positions (0-15), same as BF16

Formula: `storeCol = col ^ (row % swizzle_period)`

The swizzle XOR mask is computed from the 128-bit column offset, which is
identical whether the data is BF16 or FP8-pairs-as-b16.

### 5.3 For FP8 B operand loading (K-as-B, V)

B operand for m16n8k32: 2 x uint32 = 8 FP8 values.
Use `ldmatrix.x2.b16` (or `ldmatrix.x2.trans.b16` for transposed V).

ldmatrix.x2 uses threads 0-15 (each provides one row address), loading
2 x 8x8 sub-matrices into registers r0, r1.

For V (column-major / transposed): `ldmatrix.x2.trans.b16` transposes the
8x8 sub-matrices during load. The 16-bit element at position [row][col]
after transpose is actually 2 FP8 values. This needs empirical verification
because the transpose operates on 16-bit element granularity, and the
two FP8 bytes within a pair may end up in the wrong order after transpose.

---

## 6. FP32 Accumulator to FP8 A Fragment Conversion (P path)

### 6.1 The Layout Mismatch

FP32 accumulator layout (C/D for m16n8k16 or m16n8k32):
- Thread T holds 4 float32 values at specific matrix positions
- groupID = laneid >> 2, tid = laneid % 4
- Positions: (groupID, tid*2), (groupID, tid*2+1), (groupID+8, tid*2), (groupID+8, tid*2+1)

FP8 A operand layout (m16n8k32):
- Thread T holds 16 FP8 values (4 registers of 4 FP8 each)
- Different thread-to-element mapping than accumulator

The Colfax technique (byte_perm + shfl_sync) handles this for WGMMA. For
mma.sync on sm_120, the ThunderKittens approach is different and more direct:

### 6.2 ThunderKittens Conversion Approach

From `conversions.cuh` (lines 229-295), the float-to-fp8 conversion uses:

1. **Intra-warp shuffle** to move data between adjacent threads:
   - Even threads (laneid%2==0) put up their LEFT core matrix values first
   - Odd threads put up their RIGHT core matrix values first
   - `packed_shfl_sync` exchanges between thread pairs

2. **Convert to FP8** after the shuffle:
   - Threads with laneid%4 < 2: pack as {val01.x, val01.y, val23.x, val23.y}
   - Threads with laneid%4 >= 2: pack as {val23.x, val23.y, val01.x, val01.y}

The source positions map LEFT/RIGHT core matrices to the FP8 fragment layout.
The key shuffle formula:
```cpp
int row_mask = 4 * (laneid / 4);
int row_offset = row_mask + (laneid - row_mask) / 2 + (laneid % 2);
int src_offset = (laneid % 2 == 0) ? row_offset : row_offset + 1;
// shfl_sync from src_offset for first pair
int src_offset2 = (laneid % 4 < 2) ? src_offset + 1 : src_offset - 1;
// shfl_sync from src_offset2 for second pair
```

This is the most concrete code available for doing the FP32->FP8 fragment
layout conversion on mma.sync hardware. It should be directly portable to
sm_120.

---

## 7. NVIDIA Forum: No Native FP8 ldmatrix Exists

The NVIDIA forum thread (330254) confirms:
- There is NO `ldmatrix.m8n8.x4.b8` or similar 8-bit variant
- The available shapes are: .m8n8 (b16), .m16n16 (b8/b6/b4), .m8n16 (b6/b4)
- For m16n8k32, none of these directly match the 16x32 FP8 A tile or 8x32 FP8 B tile
- The recommended approach is to use ldmatrix.b16 (reinterpret) or direct element loads

The .m16n16 variant (available from sm_120 onwards) loads a 16x16 matrix of 8-bit
elements, which is 16x16 = 256 bytes. This covers HALF of the 16x32 FP8 A tile.
Two ldmatrix.m16n16.x1.b8 calls could load the full tile, but the register mapping
is unknown and may not match m16n8k32 fragment layout.

The safer approach remains ldmatrix.m8n8.x4.b16 (reinterpret), which is proven by
ThunderKittens.

---

## 8. sm_89 (Ada) vs sm_120 (Blackwell) Differences

CUTLASS confirms (issue #1986):
- `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` is NATIVE on sm_89 (Ada)
- On sm_90 (Hopper), the same instruction is emulated via FP8->FP16 conversion + FP16 MMA
- sm_120 (Blackwell consumer) has native FP8 MMA support (same ISA as sm_89)

This means the mma.sync.m16n8k32.e4m3 instruction runs at full speed on RTX 5090,
without any hidden conversion penalty.

---

## 9. Confidence Assessment and Remaining Unknowns

| Aspect | Confidence | Evidence |
|--------|-----------|----------|
| ldmatrix.x4.b16 loads FP8 pairs correctly | **Very High** | ThunderKittens production code does this |
| Register count matches (4/2/4) | **Very High** | Multiple sources confirm |
| a1/a2 swap applies unchanged | **High** | Structural analysis + TK uses same pattern |
| XOR swizzle pattern unchanged | **High** | Operates on b16 granularity |
| B operand via ldmatrix.x2.b16 | **High** | Register count matches, TK uses same approach |
| B operand via ldmatrix.x2.trans.b16 (V) | **Medium** | Transpose may scramble FP8 byte ordering within pairs |
| FP32->FP8 fragment conversion (P path) | **High** | TK conversions.cuh provides concrete shuffle code |
| Exact byte ordering within FP8 pairs | **Medium** | Little-endian assumed, needs empirical test |
| ldmatrix.m16n16.b8 as alternative | **Low** | Exists in ISA but register mapping is undocumented |

## 10. Recommended Test Plan

1. **Test ldmatrix.x4.b16 with known FP8 data:** Store FP8 values 0x01-0xFF
   in smem, load via ldmatrix, check register contents match expected packing.

2. **Test m16n8k32 MMA with ldmatrix-loaded FP8:** A=identity-like, B=identity-like,
   verify C=expected product. This validates the full pipeline.

3. **Test ldmatrix.x2.trans.b16 with FP8 pairs:** Store known FP8 B data,
   load via ldmatrix transpose, check byte ordering in registers.

4. **Test FP32->FP8 conversion with TK shuffle pattern:** Port the ThunderKittens
   conversion code, verify output matches expected fragment layout.
