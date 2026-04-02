# FP8 P->A Fragment Layout Conversion for m16n8k32 MMA

**Sources:**
- PTX ISA 8.5 Documentation, Sections 9.7.13.4.8-9.7.13.4.10 (Figures 54, 58, 59)
- [Colfax: Delivering 1 PFLOP/s with FP8 FlashAttention-2](https://research.colfax-intl.com/adding-fp8-to-flashattention/)
- [ThunderKittens: Bringing FP8 to theaters near you](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8)
- ThunderKittens source: `include/ops/group/register/tile/conversions.cuh`
- [NVIDIA Forum: How to load FP8 using ldmatrix on sm120](https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254)

**Relevant to:** attention worker (FP8 kernel)

**Worker's current problem:** Need to convert FP32 accumulators (P = softmax(QK^T)) to FP8 A operand fragments for PV MMA. The current BF16 path uses pack_bf16x2 for register-only P->A conversion. For FP8 m16n8k32, the fragment layouts are structurally different -- the A operand packs 4 bytes (4 FP8 values) per register instead of 2 BF16 values, and the column mapping changes from stride-2 to stride-4. This requires both byte packing AND cross-thread data movement.

## What This Is

A detailed analysis of the exact fragment layout mismatch between the m16n8k16 FP32 C/D accumulator (output of QK^T MMA) and the m16n8k32 FP8 A operand (input to PV MMA), with a concrete conversion strategy using CVT + byte_perm + shfl_sync.

---

## Fragment Layout Details

### Common Definitions

For all layouts below:
```
groupID         = %laneid >> 2        // 0-7 (which group of 4 threads)
threadID_in_group = %laneid % 4       // 0-3 (position within group)
```

### m16n8k16 BF16 C/D Accumulator Layout (QK^T output)

**PTX ISA Section 9.7.13.4.8, Figure 58**

The QK^T MMA (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`) produces 4 FP32 values per thread in registers c0, c1, c2, c3:

```
C/D fragment (16x8, .f32):
  row = groupID           for c0 and c1
        groupID + 8       for c2 and c3
  col = (threadID_in_group * 2) + (i & 0x1)    for ci where i = {0,...,3}
```

Concrete mapping for the 16x8 output matrix P:
```
         col 0    col 1    col 2    col 3    col 4    col 5    col 6    col 7
row 0:   T0:c0    T0:c1    T1:c0    T1:c1    T2:c0    T2:c1    T3:c0    T3:c1
row 1:   T4:c0    T4:c1    T5:c0    T5:c1    T6:c0    T6:c1    T7:c0    T7:c1
...
row 7:   T28:c0   T28:c1   T29:c0   T29:c1   T30:c0   T30:c1   T31:c0  T31:c1
row 8:   T0:c2    T0:c3    T1:c2    T1:c3    T2:c2    T2:c3    T3:c2    T3:c3
row 9:   T4:c2    T4:c3    T5:c2    T5:c3    T6:c2    T6:c3    T7:c2    T7:c3
...
row 15:  T28:c2   T28:c3   T29:c2   T29:c3   T30:c2   T30:c3   T31:c2  T31:c3
```

**Key property:** Each thread holds 2 adjacent columns in the same row. Thread T holds:
- c0 -> P[groupID, threadID_in_group*2]
- c1 -> P[groupID, threadID_in_group*2 + 1]
- c2 -> P[groupID+8, threadID_in_group*2]
- c3 -> P[groupID+8, threadID_in_group*2 + 1]

### m16n8k16 BF16 A Operand Layout (what BF16 PV uses)

**PTX ISA Section 9.7.13.4.8, Figure 54**

For `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, the A operand uses 4 registers (a0-a3), each holding a bf16x2 pair:

```
A fragment (16x16, .f16/.bf16, 4 regs = a0,a1,a2,a3):
  row = groupID           for a0 and a1 (also a4,a5 in 2nd half)
        groupID + 8       for a2 and a3 (also a6,a7 in 2nd half)
  col = (threadID_in_group * 2) + (i & 0x1)       for ai where i < 4
        (threadID_in_group * 2) + (i & 0x1) + 8   for ai where i >= 4
```

**Why BF16 P->A works with just CVT:** The C/D layout and BF16 A layout use the SAME row/column mapping for the first 8 columns (i < 4):
- c0 -> P[groupID, threadID_in_group*2]       matches a0's mapping
- c1 -> P[groupID, threadID_in_group*2 + 1]   matches a1's mapping
- c2 -> P[groupID+8, threadID_in_group*2]     matches a2's mapping
- c3 -> P[groupID+8, threadID_in_group*2 + 1] matches a3's mapping

So for BF16: `pack_bf16x2(c0, c1) -> a0` and `pack_bf16x2(c2, c3) -> a2`. No cross-thread movement needed. This is why our current register-only P->A conversion is so efficient.

### m16n8k32 FP8 A Operand Layout (what FP8 PV needs)

**PTX ISA Section 9.7.13.4.9 (integer type = same layout as FP8), Figure 59**

For `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, the A operand uses 4 registers (a0-a3), but each register now holds 4 FP8 values (4 bytes per uint32):

```
A fragment (16x32, .u8/.s8/.e4m3, 4 regs total but 2 regs for first 16 cols):
  Elements per thread: a0, a1, a2, a3, a4, a5, a6, a7  (8 elements per half-K)

  row = groupID           for ai where i < 4
        groupID + 8       for ai where i >= 4
  col = (threadID_in_group * 4) + (i & 0x3)    for ai where i = {0,...,7}
```

Concrete mapping for the first 16 columns of the 16x32 A matrix:
```
         col 0      col 1      col 2      col 3   |  col 4      col 5      col 6      col 7   | ...
row 0:   T0:a0      T0:a1      T0:a2      T0:a3   |  T1:a0      T1:a1      T1:a2      T1:a3   | T2:... T3:...
row 1:   T4:a0      T4:a1      T4:a2      T4:a3   |  T5:a0      T5:a1      T5:a2      T5:a3   | T6:... T7:...
...
row 7:   T28:a0     T28:a1     T28:a2     T28:a3  |  T29:a0     T29:a1     T29:a2     T29:a3  | T30:.. T31:..
row 8:   T0:a4      T0:a5      T0:a6      T0:a7   |  T1:a4      T1:a5      T1:a6      T1:a7   | T2:... T3:...
...
row 15:  T28:a4     T28:a5     T28:a6     T28:a7  |  T29:a4     T29:a5     T29:a6     T29:a7  | T30:.. T31:..
```

**Register packing:** Each uint32 register holds 4 FP8 bytes:
- reg0 = {a0, a1, a2, a3} = bytes at cols [threadID_in_group*4 .. threadID_in_group*4+3]
- reg1 = {a4, a5, a6, a7} = bytes at cols [threadID_in_group*4 .. threadID_in_group*4+3] in rows+8

### The Mismatch

The core problem: **column stride changes from 2 to 4**.

In the C/D accumulator, each thread covers 2 adjacent columns:
```
Thread T covers columns: [threadID_in_group*2, threadID_in_group*2 + 1]
  T0: cols 0,1     T1: cols 2,3     T2: cols 4,5     T3: cols 6,7
```

In the FP8 A operand, each thread covers 4 adjacent columns:
```
Thread T covers columns: [threadID_in_group*4 .. threadID_in_group*4 + 3]
  T0: cols 0,1,2,3     T1: cols 4,5,6,7     T2: cols 8,9,10,11     T3: cols 12,13,14,15
```

**What this means:** To build one FP8 A register for thread T0 (which needs P values at cols 0,1,2,3), we need:
- cols 0,1 from T0's c0,c1 (T0 already has these)
- cols 2,3 from T1's c0,c1 (T1 has these -- need cross-thread movement!)

Similarly, thread T1 needs:
- cols 4,5 from T2's c0,c1
- cols 6,7 from T3's c0,c1

And thread T2 needs:
- cols 0,1 from T0's c0,c1 (in the 2nd 8-column block, cols 8,9)
- cols 2,3 from T1's c0,c1 (cols 10,11)

Wait -- this is wrong. Let me reconsider. The P matrix is 16x8 (N=8 columns from QK^T output), but the A operand for PV is 16xD where D is the head dimension. The PV MMA tiles over the K dimension (which corresponds to the N=8 columns of P). So we are converting P[16x8] C/D fragments into P[16xk_tile] A fragments, iterating k_tile over the 8 columns in chunks.

**For m16n8k32 with FP8:** K=32, so each MMA consumes 32 columns of A. But P only has 8 columns (BKV=64 means 64/8=8 columns per nc pair, actually the N dimension of QK^T is BKV, and we tile it in chunks). Let me reconsider the actual kernel context.

In the attention kernel:
- QK^T MMA: m16n8k16 BF16, output is S[16x8] per MMA tile (nc dimension)
- After softmax: P[16x8] (same shape, same fragment layout)
- PV MMA: needs P as A operand. For m16n8k32 FP8, we need 32 FP8 elements per K-slice
- The K dimension of PV corresponds to the N dimension of QK^T (the KV sequence positions)
- With BKV=64, we have 64 KV positions = 8 nc-tiles of 8 columns each

So the P->A conversion is per-nc-tile: we take one 16x8 C/D fragment and convert it to form part of the A fragment for one m16n8k32 MMA.

**The actual conversion per nc-tile:**

For one 16x8 P tile (one nc pair from QK^T), each thread has 4 FP32 values mapping to 2 columns. In the FP8 A operand for a 16x8 sub-tile (8 of 32 K-columns), we need to pack these 8 columns with 4 FP8 values per register.

The C/D fragment has: thread T with `threadID_in_group = T % 4` holding cols `[T%4*2, T%4*2+1]`
The FP8 A fragment has: thread T with `threadID_in_group = T % 4` holding cols `[T%4*4, T%4*4+3]`

For a single 8-column sub-tile of the FP8 A (mapping to one QK^T nc-tile), the thread-to-column assignment changes:

**C/D (source):** 4 threads cover 8 columns as 2+2+2+2
```
T%4=0: cols 0,1    T%4=1: cols 2,3    T%4=2: cols 4,5    T%4=3: cols 6,7
```

**FP8 A (target):** 4 threads cover 8 columns as ... wait. With threadID_in_group*4, for 8 columns:
- T%4=0: cols 0,1,2,3    (needs data from T%4=0 AND T%4=1)
- T%4=1: cols 4,5,6,7    (needs data from T%4=2 AND T%4=3)
- T%4=2: cols 8,9,10,11  (out of range for 8-col tile -- belongs to next nc)
- T%4=3: cols 12,13,14,15 (out of range)

This means for a single 8-column P tile mapped into a 32-column FP8 A:
- The 8 P columns occupy columns [nc*8 .. nc*8+7] within the 32-wide A fragment
- Threads 0,1 in each group contribute to the first FP8 register
- Threads 2,3 in each group contribute to a different FP8 register

**The precise mismatch (per 4-thread group, per row-half):**

Source (c0,c1 from C/D, each fp32):
```
T%4=0 has: P[row, 0], P[row, 1]    (c0, c1)
T%4=1 has: P[row, 2], P[row, 3]    (c0, c1)
T%4=2 has: P[row, 4], P[row, 5]    (c0, c1)
T%4=3 has: P[row, 6], P[row, 7]    (c0, c1)
```

Target (FP8 A register for sub-tile at offset nc*8, packed as 4 bytes per uint32):
```
T%4=0 needs: P[row, 0], P[row, 1], P[row, 2], P[row, 3]  -> one uint32
T%4=1 needs: P[row, 4], P[row, 5], P[row, 6], P[row, 7]  -> one uint32
T%4=2 needs: (next sub-tile or unused depending on nc tiling)
T%4=3 needs: (next sub-tile or unused depending on nc tiling)
```

**Summary of required data movement:**
1. Convert each FP32 to FP8 (CVT instruction -- already working)
2. Thread T%4=0 needs its own 2 FP8 values PLUS T%4=1's 2 FP8 values -> shfl_sync needed
3. Thread T%4=1 needs T%4=2's 2 FP8 values PLUS T%4=3's 2 FP8 values -> shfl_sync needed
4. Pack 4 FP8 bytes into one uint32 -> byte_perm/PRMT

---

## Colfax's Solution: byte_perm + shfl_sync

Colfax's ReorgCFp8toAFp8 (for WGMMA on Hopper) addresses the same conceptual problem but for warpgroup MMA. Their approach uses two intrinsics:

### Step 1: Convert FP32 to FP8 pairs
Use `cvt.rn.satfinite.e4m3x2.f32` to convert pairs of FP32 values to packed FP8x2 (16-bit).

### Step 2: Byte permutation (within-thread rearrangement)
`__byte_perm(upper, lower, selector)` rearranges bytes. The selector is a 16-bit value where each nibble (4 bits = index 0-7) selects one byte from the 8 input bytes ({lower[0..3], upper[4..7]}).

Colfax uses selectors like:
```cpp
auto upper0 = __byte_perm(upper, lower, 0x7654);  // Take bytes 4,5,6,7 (all from upper)
auto lower0 = __byte_perm(upper, lower, 0x3210);  // Take bytes 0,1,2,3 (all from lower)
```

### Step 3: Cross-thread shuffle
```cpp
int upper_map[4] = {0, 3, 1, 2};
int lower_map[4] = {1, 2, 0, 3};

upper0 = __shfl_sync(0xffffffff, upper0, upper_map[threadIdx.x % 4], 4);
lower0 = __shfl_sync(0xffffffff, lower0, lower_map[threadIdx.x % 4], 4);
```

The `width=4` parameter in shfl_sync means the shuffle operates within 4-thread sub-warps, which is exactly the threadID_in_group boundary.

**Note:** Colfax's exact shuffle maps are for WGMMA (Hopper), not mma.sync. The principle is the same but the specific lane mappings differ because WGMMA has a different thread-to-element assignment than mma.sync.

---

## ThunderKittens' Solution (mma.sync compatible)

ThunderKittens handles the same FP32->FP8 conversion for mma.sync in `conversions.cuh`. Their approach (verified in source code):

```cpp
// FLOAT (SRC -- 1H x 2W) to FP8 (DST -- 1H x 1W)
int laneid = threadIdx.x % 32;

// Step 1: Prepare values for cross-thread exchange
// Even threads (laneid%2==0) put left core matrix first
// Odd threads (laneid%2==1) put right core matrix first
if (laneid % 2 == 0) {
    val1 = src.tiles[i][2*j + k/2].data[(k%2)+0];  // left tile
    val2 = src.tiles[i][2*j + k/2].data[(k%2)+2];  // right tile
} else {
    val1 = src.tiles[i][2*j + k/2].data[(k%2)+2];  // right tile first
    val2 = src.tiles[i][2*j + k/2].data[(k%2)+0];  // left tile first
}

// Step 2: Shuffle to gather data from paired thread
int row_mask = 4 * (laneid / 4);
int row_offset = row_mask + (laneid - row_mask) / 2 + (laneid % 2);
int src_offset = (laneid % 2 == 0) ? row_offset : row_offset + 1;
float2 val01 = packed_shfl_sync(MASK_ALL, val1, src_offset);

int src_offset2 = (laneid % 4 < 2) ? src_offset + 1 : src_offset - 1;
float2 val23 = packed_shfl_sync(MASK_ALL, val2, src_offset2);

// Step 3: Pack 4 values into fp8x4 with correct ordering
float4 f4;
if (laneid % 4 < 2) {
    f4 = {val01.x, val01.y, val23.x, val23.y};  // T_2N's pair, then T_2N+1's pair
} else {
    f4 = {val23.x, val23.y, val01.x, val01.y};  // reversed for upper threads
}
fp8_4_t f4_fp8 = convert<fp8_4_t>(f4);
```

**Key insight:** ThunderKittens uses FP32-domain shuffles (before FP8 conversion) rather than FP8-domain byte shuffles. This avoids needing PRMT instructions at the cost of moving 32-bit values across threads.

---

## Implementation for Our Kernel

For our attention kernel on sm_120 with `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`:

### Approach: CVT first, then pack with shuffle

This minimizes the number of shuffle operations by converting to FP8 first (compact) and then rearranging bytes.

```cpp
// Input: c0, c1, c2, c3 (FP32 from softmax, in C/D accumulator layout)
// These are P values for one nc-tile: P[row, col] where
//   c0 = P[groupID, tid_in_grp*2]
//   c1 = P[groupID, tid_in_grp*2+1]
//   c2 = P[groupID+8, tid_in_grp*2]
//   c3 = P[groupID+8, tid_in_grp*2+1]

// Step 1: Convert FP32 pairs to packed FP8x2 (2 FP8 values per 16-bit result)
uint32_t fp8_pair_upper, fp8_pair_lower;
asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n"  // NOTE: reversed operand order!
    : "=r"(fp8_pair_lower) : "f"(c0), "f"(c1));   // lower 16 bits: fp8(c0), fp8(c1)
asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n"
    : "=r"(fp8_pair_upper) : "f"(c2), "f"(c3));   // fp8(c2), fp8(c3)

// After CVT, each thread has:
//   fp8_pair_lower[7:0] = fp8(c0) = fp8(P[row, tid*2])
//   fp8_pair_lower[15:8] = fp8(c1) = fp8(P[row, tid*2+1])
//   fp8_pair_upper[7:0] = fp8(c2) = fp8(P[row+8, tid*2])
//   fp8_pair_upper[15:8] = fp8(c3) = fp8(P[row+8, tid*2+1])

// Step 2: Exchange FP8 pairs with neighbor thread
// Thread T%4=0 has cols 0,1; needs cols 0,1,2,3 -> needs T%4=1's cols 2,3
// Thread T%4=1 has cols 2,3; needs cols 0,1,2,3 -> needs T%4=0's cols 0,1
// (for the FIRST sub-tile)
// Thread T%4=2 has cols 4,5; needs cols 4,5,6,7 -> needs T%4=3's cols 6,7
// Thread T%4=3 has cols 6,7; needs cols 4,5,6,7 -> needs T%4=2's cols 4,5
// (for the SECOND sub-tile)

int lane = threadIdx.x % 32;
int tid_in_grp = lane % 4;

// Neighbor's fp8 pair (within the 4-thread group)
// Even threads (0,2) get from odd neighbor (1,3)
// Odd threads (1,3) get from even neighbor (0,2)
int src_lane = (tid_in_grp & 1) ? (lane - 1) : (lane + 1);
uint32_t neighbor_lower = __shfl_sync(0xffffffff, fp8_pair_lower, src_lane);
uint32_t neighbor_upper = __shfl_sync(0xffffffff, fp8_pair_upper, src_lane);

// Step 3: Pack 4 FP8 bytes into one uint32
// For even threads (tid_in_grp = 0 or 2): own pair is cols 2k, 2k+1; neighbor has 2k+2, 2k+3
// For odd threads (tid_in_grp = 1 or 3): neighbor has 2k, 2k+1; own pair is 2k+2, 2k+3
uint32_t fp8x4_row0, fp8x4_row8;

if (tid_in_grp & 1) {
    // Odd thread: neighbor's pair goes in low bytes, own pair in high bytes
    // Result: {neighbor[0], neighbor[1], own[0], own[1]}
    fp8x4_row0 = __byte_perm(neighbor_lower, fp8_pair_lower, 0x5410);
    fp8x4_row8 = __byte_perm(neighbor_upper, fp8_pair_upper, 0x5410);
} else {
    // Even thread: own pair goes in low bytes, neighbor's pair in high bytes
    // Result: {own[0], own[1], neighbor[0], neighbor[1]}
    fp8x4_row0 = __byte_perm(fp8_pair_lower, neighbor_lower, 0x5410);
    fp8x4_row8 = __byte_perm(fp8_pair_upper, neighbor_upper, 0x5410);
}

// Now fp8x4_row0 and fp8x4_row8 are packed uint32 with 4 FP8 values each
// BUT: only threads 0,1 (and 2,3) in each group have valid sub-tiles
// Thread pair (0,1) both have the same 4-column sub-tile for cols 0-3
// Thread pair (2,3) both have the same 4-column sub-tile for cols 4-7
//
// For the FP8 A operand register assignment:
//   Thread 0 needs cols 0-3 (sub-tile 0) -> fp8x4_row0 is correct
//   Thread 1 needs cols 4-7 (sub-tile 1) -> needs (T2,T3)'s data
//   Thread 2 needs cols 8-11 -> from next nc tile
//   Thread 3 needs cols 12-15 -> from next nc tile

// This is where it gets kernel-specific: how the 8 columns from one QK^T
// nc-tile map into the 32-column FP8 A depends on which nc position we're at.
// The A fragment's K=32 dimension spans 4 nc-tiles of 8 columns each.
```

### Alternative: Simplified approach for our kernel

Given our kernel tiles PV with nc-pairs (8 columns per pair), and m16n8k32 needs 32 K-columns, we accumulate 4 nc-pairs before issuing one FP8 MMA. A simpler approach:

```cpp
// For each nc-pair, convert 4 FP32 values to FP8 and store to a register array
// After 4 nc-pairs, we have 16 FP8 values per thread-row = 4 uint32 registers

// Per nc-pair (nc = 0,1,2,3):
uint32_t fp8_pair;
asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n"
    : "=r"(fp8_pair) : "f"(c0), "f"(c1));  // 2 FP8 values

// Collect into the FP8 A register via shfl + byte_perm
// Thread T%4=k has cols 2k, 2k+1 from nc-pair
// FP8 A register needs cols [T%4*4 .. T%4*4+3]
// So thread T%4=0 needs nc0's (T0 cols 0,1) + nc0's (T1 cols 2,3)
//    thread T%4=1 needs nc0's (T2 cols 4,5) + nc0's (T3 cols 6,7)
//    thread T%4=2 needs nc1's (T0 cols 0,1) + nc1's (T1 cols 2,3) [offset by 8]
//    thread T%4=3 needs nc1's (T2 cols 4,5) + nc1's (T3 cols 6,7) [offset by 8]

// This maps to:
// FP8 A reg for K-cols [0..3]:  gather from T%4=0 and T%4=1 at nc=0
// FP8 A reg for K-cols [4..7]:  gather from T%4=2 and T%4=3 at nc=0
// FP8 A reg for K-cols [8..11]: gather from T%4=0 and T%4=1 at nc=1
// etc.
```

### Practical Implementation: 2 shfl + 2 byte_perm per nc-pair

The most efficient approach for our kernel, processing one nc-pair at a time:

```cpp
__device__ inline void convert_p_fp32_to_fp8_a_fragment(
    float c0, float c1, float c2, float c3,  // P fragment (one nc-pair)
    int nc_within_k32,                         // which of the 4 nc-pairs (0-3)
    uint32_t &a_reg_row0,                      // output A register for rows 0-7
    uint32_t &a_reg_row8                       // output A register for rows 8-15
) {
    int lane = threadIdx.x % 32;
    int tid_in_grp = lane % 4;

    // Step 1: Convert to FP8 pairs
    uint32_t fp8_lo, fp8_hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n"
        : "=r"(fp8_lo) : "f"(c0), "f"(c1));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;\n"
        : "=r"(fp8_hi) : "f"(c2), "f"(c3));

    // Step 2: Get neighbor's FP8 pair
    int neighbor = (tid_in_grp & 1) ? (lane - 1) : (lane + 1);
    uint32_t nb_lo = __shfl_sync(0xffffffff, fp8_lo, neighbor);
    uint32_t nb_hi = __shfl_sync(0xffffffff, fp8_hi, neighbor);

    // Step 3: Pack 4 bytes
    // Even thread: {own_byte0, own_byte1, neighbor_byte0, neighbor_byte1}
    // Odd thread:  {neighbor_byte0, neighbor_byte1, own_byte0, own_byte1}
    if (tid_in_grp & 1) {
        a_reg_row0 = __byte_perm(nb_lo, fp8_lo, 0x5410);
        a_reg_row8 = __byte_perm(nb_hi, fp8_hi, 0x5410);
    } else {
        a_reg_row0 = __byte_perm(fp8_lo, nb_lo, 0x5410);
        a_reg_row8 = __byte_perm(fp8_hi, nb_hi, 0x5410);
    }

    // Now threads 0,1 both have the uint32 for cols 0-3
    // and threads 2,3 both have the uint32 for cols 4-7
    //
    // For the FP8 A fragment, thread T%4=k needs the uint32 for cols [k*4..k*4+3]
    // So we need one more shuffle to route the right sub-tile to the right thread:
    //   T%4=0 keeps its own (cols 0-3)
    //   T%4=1 needs (cols 4-7) which threads 2,3 have
    //   T%4=2 needs (cols 8-11) which is from a DIFFERENT nc-pair
    //   T%4=3 needs (cols 12-15) which is from a DIFFERENT nc-pair

    // The routing depends on nc_within_k32:
    // nc=0 -> fills cols 0-7 -> threads 0,1 get the data directly
    // nc=1 -> fills cols 8-15 -> threads 2,3 get it
    // nc=2 -> fills cols 16-23 -> (next register pair)
    // nc=3 -> fills cols 24-31 -> (next register pair)

    // For nc=0: T0 wants cols 0-3 (from T0,T1), T1 wants cols 4-7 (from T2,T3)
    // Need one more shfl to get T2,T3's result to T1
    if (nc_within_k32 == 0 || nc_within_k32 == 2) {
        // Even nc: this fills the low sub-tile (first 2 threads get valid data)
        // T0 already has cols 0-3, T2 already has cols 4-7
        // T1 needs cols 4-7 from T2, T3 needs cols 0-3 from T0(doesn't actually -- T3 unused here)
        int src = (tid_in_grp < 2) ? lane : lane - 2;
        a_reg_row0 = __shfl_sync(0xffffffff, a_reg_row0, src);
        a_reg_row8 = __shfl_sync(0xffffffff, a_reg_row8, src);
    } else {
        // Odd nc: fills the high sub-tile
        int src = (tid_in_grp >= 2) ? lane : lane + 2;
        a_reg_row0 = __shfl_sync(0xffffffff, a_reg_row0, src);
        a_reg_row8 = __shfl_sync(0xffffffff, a_reg_row8, src);
    }
}
```

**WARNING:** The above code is a starting point that needs empirical verification on sm_120. The exact byte ordering within registers (which byte is "low" vs "high" in the uint32) and the CVT operand reversal (verified in hard-won lessons: first source goes to HIGH bits) must be tested carefully. The shfl routing logic for mapping nc-pairs to K-columns within the 32-wide FP8 A fragment is the most subtle part and may need adjustment based on testing.

---

## Register Cost

Per nc-pair conversion:
- 2x CVT instructions (fp32 pair -> fp8x2): 0 extra registers (in-place)
- 2x shfl_sync: 2 temporary registers (neighbor values)
- 2x byte_perm: 0 extra registers (in-place)
- 2x shfl_sync (routing): 0 extra registers (in-place)

**Total per nc-pair: ~2 temporary registers + 6 instructions**

For 4 nc-pairs to fill one m16n8k32 A fragment (4 uint32 registers):
- 24 instructions total (8 CVT + 8 shfl + 4 byte_perm + 4 shfl for routing)
- 2 temporary registers

Compare to current BF16 path: 2 CVT (pack_bf16x2) per nc-pair, 8 total for 4 nc-pairs. The FP8 path adds ~16 instructions (8 shfl + 4 byte_perm + 4 shfl) on top of the 8 CVT. This is significantly less than the current ~448 ALU instructions for K+V BF16->FP8 conversion.

---

## Caveats

1. **sm_120 specifics:** All code above uses `mma.sync` (not WGMMA). Colfax's exact shuffle maps are for WGMMA on Hopper and will NOT work directly. The fragment layouts are different. Use the PTX ISA formulas from section 9.7.13.4.8-9.7.13.4.10 as ground truth.

2. **CVT operand order:** Verified in hard-won lessons: `cvt.rn.satfinite.e4m3x2.f32` puts first source in HIGH byte, second in LOW byte. The code above accounts for this with the operand swap.

3. **Byte packing order:** The selector `0x5410` in __byte_perm means: output byte 0 = input byte 0, byte 1 = input byte 1, byte 2 = input byte 4, byte 3 = input byte 5. This concatenates the low 2 bytes of the first argument with the low 2 bytes of the second argument. Verify empirically that this matches the FP8 A fragment's expected byte order within the uint32.

4. **nc-to-K mapping:** The exact mapping of nc-pair indices to K-columns in the 32-wide FP8 A fragment needs empirical verification. The code assumes nc=0 maps to K-cols 0-7, nc=1 to 8-15, etc., but the actual mapping depends on how the kernel tiles the PV computation.

5. **This eliminates BF16->FP8 conversion entirely for P.** The conversion happens directly from FP32 accumulators to FP8 A fragments. However, V still needs conversion (or native FP8 input).

6. **Register pressure:** The current FP8 kernel uses 165 regs with only 5 spare. Adding 2 temporary registers may be tight. Monitor with `--ptxas-options=-v`.

---

## __byte_perm Reference

`__byte_perm(x, y, selector)` maps to PTX `prmt` instruction.

Input bytes are numbered 0-7:
- Bytes 0-3: from parameter `x` (byte 0 = x[7:0], byte 1 = x[15:8], byte 2 = x[23:16], byte 3 = x[31:24])
- Bytes 4-7: from parameter `y` (byte 4 = y[7:0], byte 5 = y[15:8], byte 6 = y[23:16], byte 7 = y[31:24])

Selector is 16 bits, 4 nibbles, each nibble (3 bits used) selects one output byte:
- Bits [2:0]: output byte 0
- Bits [6:4]: output byte 1
- Bits [10:8]: output byte 2
- Bits [14:12]: output byte 3

Common selectors:
```
0x3210 = identity (copy x as-is)
0x7654 = identity (copy y as-is)
0x5410 = {x[0], x[1], y[0], y[1]}  -- concatenate low halves
0x7632 = {x[2], x[3], y[2], y[3]}  -- concatenate high halves
0x1054 = {y[0], y[1], x[0], x[1]}  -- swap and concatenate low halves
```
