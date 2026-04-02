# Bank Conflict Patterns for ldmatrix — Swizzle Reference

**Purpose:** Shared memory layouts that achieve zero bank conflicts for ldmatrix.x4 and ldmatrix.x2.trans, with XOR swizzle formulas.
**Last updated:** 2026-03-13

---

## 1. The Problem: ldmatrix Bank Conflicts

### Shared Memory Bank Structure

- 32 banks, 4 bytes per bank, 128 bytes per bank cycle
- BF16 elements: 2 bytes each, so 2 elements per bank
- A row of 16 BF16 elements = 32 bytes = 8 banks
- A row of 8 × uint4 (128 bits each) = 128 bytes = all 32 banks

### ldmatrix Access Pattern

`ldmatrix.sync.aligned.x4.m8n8.shared.b16` loads 4 × 8×8 sub-matrices (16 bytes per thread):
- The warp's 32 threads provide 32 addresses
- Each thread's address points to 16 bytes (8 BF16 elements = one row of an 8×8 tile)
- The hardware splits the 32-thread request into **4 phases of 8 threads** each
- Each phase transfers 8 × 16 bytes = 128 bytes

**Conflict occurs when** two threads within the same 8-thread phase access addresses that map to the same bank but at different addresses.

### Without Swizzle: 8-Way Bank Conflicts

If BF16 data is stored in row-major order:
```
Row 0: elements [0..15] → banks [0,0,1,1,2,2,...,7,7]
Row 1: elements [0..15] → banks [0,0,1,1,2,2,...,7,7]  ← SAME banks as row 0!
...
Row 7: elements [0..15] → banks [0,0,1,1,2,2,...,7,7]  ← 8-way conflict
```

All 8 threads in a phase reading from the same column hit the same bank. This serializes what should be parallel, requiring 8x more wavefronts.

**Measured impact:** lubits.ch flash attention tutorial showed 2x speedup from eliminating bank conflicts (33.28 → 66.12 TFLOPS on RTX 3090).

---

## 2. XOR Swizzle Formula

### Core Idea

XOR the column index with a function of the row index. Different rows access different columns for the same logical position, spreading accesses across banks.

### spatters.ca Formula (Ada GEMM, uint4 storage)

**Source:** https://www.spatters.ca/mma-matmul

Storage unit: `uint4` (128 bits = 8 FP16/BF16 values). Shared memory declared as `uint4 As[ROWS][8]` — each row has 8 uint4 elements = 128 bytes = all 32 banks.

**Store phase (global → shared):**
```c
int storeRow = warpID * 4 + laneID / 8;
int storeCol = (laneID % 8) ^ (laneID / 8);
//                              ^^^^^^^^^ XOR with row-within-group
```

**Load phase A (shared → register via ldmatrix.x4):**
```c
int loadRowA = (laneID % 16) / 2;
int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
```

**Load phase B (shared → register via ldmatrix.x2):**
```c
int loadRowB = (laneID % 8) / 2;
int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);
```

**For k=2..3 subtiles:** XOR column with 2: `loadColA ^ 2`, `loadColB ^ 2`

### gau-nernst Formula (sm_120 Flash Attention)

**Source:** https://gau-nernst.github.io/fa-5090/

Generic swizzle function for any stride:

```c
// For a buffer with stride STRIDE (in bytes):
int row_idx = (byte_index / STRIDE) % 8;
int bits_to_xor = row_idx / max(64 / STRIDE, 1);
int swizzled = byte_index ^ (bits_to_xor << 4);
```

This shifts bits 4-6 of the address by XORing with row index bits 0-2. The division by `max(64/STRIDE, 1)` adapts the swizzle pattern to different row widths.

### lubits.ch Formula (Flash Attention, BF16)

**Source:** https://lubits.ch/flash/Part-4

```c
const int BANKS_PER_VEC4_ACCESS = 8;  // 16B access spans 4 banks, but 8 threads/phase
const int ELEMS_PER_BANK = 8;         // 8 BF16 elements per bank group

int region_row = row % BANKS_PER_VEC4_ACCESS;
int bank_col = col / ELEMS_PER_BANK;
int bank_offset = col % ELEMS_PER_BANK;
int swizzled_col = ((region_row ^ bank_col) * ELEMS_PER_BANK) + bank_offset;
```

### CUTLASS Formula

**Source:** https://github.com/NVIDIA/cutlass/discussions/1130

CUTLASS uses a parameterized swizzle template `Swizzle<B,M,S>`:

```
new_col = original_row XOR original_col
```

Applied within 8×8 partition blocks. The XOR distributes column accesses across banks based on the row index, ensuring no two threads in the same ldmatrix phase access the same bank.

---

## 3. Making One Layout Work for Both ldmatrix.x4 and ldmatrix.x2.trans

### The Challenge

In attention kernels, the SAME shared memory region may be loaded as:
- **ldmatrix.x4** (row-major, for Q and K fragments in QK^T)
- **ldmatrix.x2.trans** (transposed, for V fragments in PV)

Both access patterns must be bank-conflict-free with the same swizzle.

### Why XOR Swizzle Works for Both

The XOR swizzle `new_col = row ^ col` creates a **Latin square** — each row has a unique permutation of columns, and each column has a unique permutation across rows.

For row-major loads (ldmatrix.x4): threads load consecutive elements within a row. The swizzle ensures different rows access different bank groups.

For column-major loads (ldmatrix.x2.trans): threads load elements from different rows in the same column. The swizzle ensures different rows in the same column map to different banks.

**The Latin square property guarantees both access patterns are conflict-free.** This is why XOR swizzle is universally used — it's the only simple pattern that handles both orientations.

### gau-nernst's Approach

Q, K, V share swizzled shared memory:
- Q_smem overlaps with K_smem + V_smem (Q loaded once to registers, then smem reused)
- Same XOR swizzle applied to all three
- K loaded via ldmatrix.x4 (row-major)
- V loaded via ldmatrix.x4.trans (column-major, for PV = P × V)

### Our Kernel's Approach

We already use XOR swizzle. K is loaded via ldmatrix.x4, V via ldmatrix.x2.trans. Both use the same swizzled shared memory layout. This is confirmed correct.

---

## 4. Swizzle for RF→SMEM Operations (P Conversion)

### The Unique Challenge

When converting P (FP32 accumulators) back to shared memory for the next MMA phase, the MMA output fragment layout doesn't match a natural row-major store pattern.

The MMA D-fragment for m16n8k16:
- Thread T stores: `D[T/4, (T%4)*2]` and `D[T/4, (T%4)*2+1]` (consecutive pair)
- Plus: `D[T/4+8, (T%4)*2]` and `D[T/4+8, (T%4)*2+1]` (8 rows down)

When threads write these 4-byte FP32 values to shared memory:
- Threads spaced 4 apart (0, 4, 8, 12, ...) map to the same bank (they write to the same column)
- This creates 8-way bank conflicts on the RF→SMEM path

### Solution: Swizzle the RF→SMEM Store Too

Apply XOR swizzle to the shared base address for all threads in a row, then add each thread's individual offset:

```c
int swizzled_base = base_col ^ (row % 8);
int store_addr = row * stride + swizzled_base + thread_offset;
```

### Our Solution: Skip SMEM Entirely

Our kernel (v7) does P→A conversion entirely in registers via warp shuffles. This eliminates the RF→SMEM bank conflict problem entirely. The register-only path was a major optimization (documented in 04_HARD_WON_LESSONS.md).

gau-nernst confirmed that P→A packing needs no inter-thread shuffle because the MMA accumulator layout matches the A-multiplicand layout within each 8×8 tile. Just convert FP32 → BF16 in-place.

---

## 5. Practical Swizzle Implementation for Our GEMM

### Current Shared Memory Layout

```c
// Our GEMM kernel uses:
__shared__ __align__(16) __nv_bfloat16 smem_a[2][BK][BM];  // Double-buffered
__shared__ __align__(16) __nv_bfloat16 smem_b[2][BK][BN];

// Swizzled index computation:
int swizzled_col = col ^ ((row % 8) * (128 / (BM * sizeof(__nv_bfloat16))));
```

### Recommended Swizzle Pattern for BF16 GEMM with ldmatrix.x4

For a tile stored as `bf16 smem[ROWS][COLS]` where COLS is the contiguous dimension:

```c
// Store: global → shared (during cp.async or register-mediated copy)
int store_col = (tid % 8) ^ (tid / 8);  // XOR within 8-element groups

// Load A: shared → register (ldmatrix.x4, 16×16 tile)
int load_row = (lane_id % 16);
int load_col = ((lane_id / 16) * 8) ^ (load_row % 8);
// Each thread provides address: smem[load_row][load_col]

// Load B: shared → register (ldmatrix.x4.trans, 16×16 tile transposed)
// Same swizzle formula — Latin square property ensures both patterns are conflict-free
int load_row_b = (lane_id % 16);
int load_col_b = ((lane_id / 16) * 8) ^ (load_row_b % 8);
```

### Verification

Always verify zero bank conflicts via Nsight Compute:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./kernel
```

Target: 0 conflicts in the inner loop. Any non-zero value indicates the swizzle formula is wrong or incomplete.

---

## References

- spatters.ca MMA matmul (XOR swizzle derivation): https://www.spatters.ca/mma-matmul
- gau-nernst flash attention (sm_120 swizzle): https://gau-nernst.github.io/fa-5090/
- lubits.ch flash attention Part 4 (bank conflict tutorial): https://lubits.ch/flash/Part-4
- CUTLASS discussion #1130 (permuted layout): https://github.com/NVIDIA/cutlass/discussions/1130
- Lei Mao shared memory swizzling: https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/
- NVIDIA forum — ldmatrix behavior: https://forums.developer.nvidia.com/t/understanding-the-behaivor-of-ldmatrix-in-terms-of-shared-memory-access/278716
- NVIDIA forum — CUTLASS permuted layout: https://forums.developer.nvidia.com/t/understanding-cutlass-permuted-shared-memory-layout/303697
