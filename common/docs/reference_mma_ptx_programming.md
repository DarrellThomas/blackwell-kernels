# MMA PTX Programming Reference

**Sources:** spatters.ca MMA matmul tutorial, Bruce-Lee-LY cuda_hgemm, empirical verification on sm_120
**Scope:** Practical guide to writing mma.sync kernels with inline PTX assembly

---

## 1. mma.sync.aligned.m16n8k16 Overview

The fundamental tensor core operation on sm_120 (consumer Blackwell):

```
D[16x8] = A[16x16] * B[16x8] + C[16x8]
```

- A is 16x16 in row-major layout
- B is 16x8 in column-major layout (instruction computes A * B, with B stored col-major)
- C and D are 16x8 in row-major layout, f32 accumulators
- All 32 threads in the warp cooperate

**PTX instruction:**
```ptx
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {d0, d1, d2, d3},
    {a0, a1, a2, a3},
    {b0, b1},
    {c0, c1, c2, c3};
```

---

## 2. Fragment Register Layout

### Per-Thread Register Allocation

| Fragment | Shape | Registers per Thread | Constraint | Total per Warp |
|----------|-------|---------------------|------------|---------------|
| A | 16x16 | 4x uint32 (`"r"`) | Each holds 2 bf16 | 128 regs |
| B | 16x8 | 2x uint32 (`"r"`) | Each holds 2 bf16 | 64 regs |
| C (accum) | 16x8 | 4x float (`"f"`) | f32 values | 128 regs |
| D (output) | 16x8 | 4x float (`"f"`) | f32 values | 128 regs |

### D-Fragment: Thread-to-Matrix-Element Mapping

For lane `T` (0-31) within the warp, the 4 output registers map to:

```
d0 -> D[T/4,     (T%4)*2    ]
d1 -> D[T/4,     (T%4)*2 + 1]
d2 -> D[T/4 + 8, (T%4)*2    ]
d3 -> D[T/4 + 8, (T%4)*2 + 1]
```

**Concrete example:**
```
Lane 0:  d0=D[0,0]  d1=D[0,1]  d2=D[8,0]  d3=D[8,1]
Lane 1:  d0=D[0,2]  d1=D[0,3]  d2=D[8,2]  d3=D[8,3]
Lane 2:  d0=D[0,4]  d1=D[0,5]  d2=D[8,4]  d3=D[8,5]
Lane 3:  d0=D[0,6]  d1=D[0,7]  d2=D[8,6]  d3=D[8,7]
Lane 4:  d0=D[1,0]  d1=D[1,1]  d2=D[9,0]  d3=D[9,1]
Lane 5:  d0=D[1,2]  d1=D[1,3]  d2=D[9,2]  d3=D[9,3]
...
Lane 28: d0=D[7,0]  d1=D[7,1]  d2=D[15,0] d3=D[15,1]
Lane 31: d0=D[7,6]  d1=D[7,7]  d2=D[15,6] d3=D[15,7]
```

The C-fragment has the identical mapping (same layout, just input vs output).

### A-Fragment: Thread-to-Matrix-Element Mapping (ALT Mapping)

A is 16x16, split into four 8x8 sub-matrices. Each register holds a bf16x2 pair.

```
sub = lane_id / 8;                    // 0, 1, 2, 3
row = (sub/2)*8 + lane_id % 8;        // 0-7 for sub 0,1; 8-15 for sub 2,3
col = (sub%2)*8;                       // 0 for sub 0,2; 8 for sub 1,3
```

Each of the 4 registers holds elements at `(row, col)` and `(row, col+1)` as bf16x2.

**ldmatrix_x4 output order vs MMA input order:**
```
ldmatrix output:  r0 = sub0(rows 0-7,  cols 0-7)
                  r1 = sub1(rows 0-7,  cols 8-15)
                  r2 = sub2(rows 8-15, cols 0-7)
                  r3 = sub3(rows 8-15, cols 8-15)

MMA expects:      a0 = sub0(rows 0-7,  cols 0-7)   = r0
                  a1 = sub2(rows 8-15, cols 0-7)    = r2  <-- SWAPPED
                  a2 = sub1(rows 0-7,  cols 8-15)   = r1  <-- SWAPPED
                  a3 = sub3(rows 8-15, cols 8-15)   = r3
```

### B-Fragment: Thread-to-Matrix-Element Mapping

B is 16x8 in col-major (equivalently, B^T is 8x16 in row-major).

For A*B^T computation (e.g., Q*K^T): load K rows directly:
```
b0 = {B[T/4, (T%4)*2], B[T/4, (T%4)*2+1]}           // bf16x2
b1 = {B[T/4, (T%4)*2+8], B[T/4, (T%4)*2+9]}          // bf16x2
```

For A*B computation (e.g., P*V): use `ldmatrix_x2_trans` which transposes during load, giving col-major layout suitable for direct A*B.

---

## 3. ldmatrix: Loading Fragments from Shared Memory

### How ldmatrix Works

`ldmatrix` is a warp-cooperative instruction. Each thread provides a shared memory address pointing to a 128-bit (16-byte, 8 bf16 elements) row. The instruction loads the data and distributes it across warp lanes in the fragment layout expected by `mma.sync`.

### Addressing Pattern

For `.x4` (loads a 16x16 tile as four 8x8 sub-matrices):
- Threads 0-7: provide addresses for rows 0-7 of sub-matrix 0
- Threads 8-15: provide addresses for rows 0-7 of sub-matrix 1
- Threads 16-23: provide addresses for rows 0-7 of sub-matrix 2
- Threads 24-31: provide addresses for rows 0-7 of sub-matrix 3

For `.x2` (loads an 8x16 tile as two 8x8 sub-matrices):
- Threads 0-7: rows 0-7 of first sub-matrix
- Threads 8-15: rows 0-7 of second sub-matrix
- Threads 16-31: addresses ignored (but all threads must participate)

### Inline PTX Wrappers

```cpp
// Standard x4 load
__device__ void ldmatrix_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
                            const void *smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// x4 with baked-in a1/a2 swap (preferred for A-fragment)
__device__ void ldmatrix_x4_mma(uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
                                const void *smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %2, %1, %3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(addr));
}

// x2 transposed (for V loading in P*V)
__device__ void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1,
                                  const void *smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}
```

### Key Insight: ldmatrix_x4_mma

The standard `ldmatrix_x4` outputs registers as: `r0=sub0, r1=sub1, r2=sub2, r3=sub3`

But `mma.sync` expects: `a0=sub0, a1=sub2, a2=sub1, a3=sub3` (r1 and r2 swapped)

By reordering the output operands in the asm template as `{%0, %2, %1, %3}`, the hardware loads land directly in MMA order. This eliminates MOV instructions that would otherwise be needed for the swap.

---

## 4. Shared Memory Layout with Swizzle

### Bank Conflict Problem

Shared memory has 32 banks, 4 bytes each. When loading a column from a row-major matrix, all threads in a group of 8 may hit the same bank.

### XOR Swizzle Solution

Apply XOR permutation when storing to shared memory:
```cpp
storeCol = (laneID % 8) ^ (laneID / 8);
```

When loading with ldmatrix, compute the inverse permutation:
```cpp
loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);
```

This ensures threads within each 8-thread memory request access different banks.

### This Project's Swizzle (from swizzle.cuh)

```cpp
template <int COLS>
__device__ int swizzle_idx(int row, int col) {
    constexpr int NUM_CHUNKS = COLS / 8;
    constexpr int SWIZZLE_BITS = (NUM_CHUNKS >= 8) ? 3 : (NUM_CHUNKS >= 4) ? 2 : 1;
    constexpr int SWIZZLE_MASK = (1 << SWIZZLE_BITS) - 1;
    int swizzled_col = col ^ ((row & SWIZZLE_MASK) << 3);
    return row * COLS + swizzled_col;
}
```

---

## 5. cp.async: Asynchronous Global-to-Shared Copies

### Macro Definitions (Bruce-Lee-LY style)

```cpp
#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" \
        ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" \
        ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N)  asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
#define CP_ASYNC_WAIT_ALL()     asm volatile("cp.async.wait_all;\n" ::)
```

### Pipeline Pattern (Double-Buffer)

```
Iteration 0:
  cp.async [buf0] <- global[tile0]    // Load tile 0 into buffer 0
  commit_group
  wait_all                             // Wait for tile 0
  __syncthreads()
  compute(buf0)                        // Use tile 0

Iteration n (steady state):
  cp.async [buf_next] <- global[tile_n+1]  // Prefetch next tile
  commit_group
  wait_group(1)                        // Wait for current tile (allow 1 in flight)
  __syncthreads()
  compute(buf_current)                 // Use current tile
```

---

## 6. Complete MMA Inner Loop Pattern

### BF16 GEMM Inner Loop

```cpp
// Registers for one m16n8 tile
float d0 = 0, d1 = 0, d2 = 0, d3 = 0;  // Accumulator

for (int k = 0; k < K; k += 16) {
    // Load A fragment from shared memory
    uint32_t a0, a1, a2, a3;
    ldmatrix_x4_mma(a0, a1, a2, a3, &smem_A[row_offset + k]);

    // Load B fragment from shared memory (transposed)
    uint32_t b0, b1;
    ldmatrix_x2_trans(b0, b1, &smem_B[k * N_tile + col_offset]);

    // Tensor core MMA
    mma_m16n8k16_bf16_nv(d0, d1, d2, d3,
                          a0, a1, a2, a3,
                          b0, b1,
                          d0, d1, d2, d3);  // Accumulate in-place
}
```

### Attention QK^T + Softmax + PV Pattern

```cpp
// Phase 1: QK^T (A * B^T)
for (int kc = 0; kc < D; kc += 16) {
    // Load Q (A-fragment) via ldmatrix_x4_mma
    // Load K (B-fragment) manually for B^T layout
    // MMA: S += Q * K^T
}

// Phase 2: Softmax on S (scalar FP32)
// max, exp2f, sum, normalize

// Phase 3: P*V (A * B)
for (int dc = 0; dc < D; dc += 8) {
    // Convert P from FP32 to BF16 (register-only)
    // Load V via ldmatrix_x2_trans
    // MMA: O += P * V
}
```

---

## 7. Register-Only P Conversion (FP32 -> BF16 for MMA)

Instead of writing P to shared memory and reloading via ldmatrix, convert in registers using warp shuffles:

```cpp
// Pack FP32 pairs into BF16x2 for MMA A-fragment
__device__ uint32_t pack_bf16x2(float a, float b) {
    uint32_t result;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n"  // NOTE: operand swap!
        : "=r"(result) : "f"(a), "f"(b));
    return result;
}

// After softmax, P values are in d0-d3 (same layout as D-fragment)
// Use __shfl_xor_sync to redistribute into A-fragment layout
```

This eliminates the shared memory round-trip that was the dominant source of bank conflicts in the attention kernel.

---

## 8. Performance Results and Reference Points

### Ada (RTX 4090) — spatters.ca

- m16n8k16 instruction latency: ~32 cycles
- cuBLAS 4096^3 GEMM: 153.6 TFLOPS (895 us)
- Custom kernel matched cuBLAS via interleaved MMA + load scheduling
- Key: reduced per-MMA overhead from 179.9 to 34.2 cycles (93% peak)

### sm_120 (RTX 5090) — This Project

- BF16 GEMM: 0.98x cuBLAS at 4096^3 (64x64 tiles, 6 blocks/SM)
- Flash Attention v2: 1.76x cuDNN SDPA at N=2048 D=64 causal
- 145 registers, 3 blocks/SM, 12 warps
- Dominant stall: math_pipe_throttle (48%)

### Occupancy Sweet Spots on sm_120

| Regs/Thread | Max Warps/SM | Blocks (4 warp) | Notes |
|-------------|-------------|-----------------|-------|
| 80 | 24 | 6 | Best for GEMM |
| 128 | 15 | 3-4 | Good general purpose |
| 145 | 13 | 3 | Current attention kernel |
| 255 | 7 | 1-2 | Very low occupancy |

---

## 9. Critical ISA Facts for sm_120

1. **sm_120 uses `mma.sync`, NOT `tcgen05`.** Do not use datacenter Blackwell examples.

2. **The a1/a2 register swap is required.** ldmatrix_x4 and mma.sync disagree on register order for the A-fragment. Use `ldmatrix_x4_mma()` to handle this.

3. **`ldmatrix_x2_trans` computes A*B, not A*B^T.** Correct for P*V. Do not add extra transpose for V.

4. **`cvt.rn.bf16x2.f32` has reversed operand order vs C++.** First source goes to HIGH bits in PTX.

5. **`shfl.sync.bfly` c parameter = 31.** Not 0x1f1f.

6. **Non-volatile MMA is safe and preferred.** Outputs serve as compiler anchors. ldmatrix and cp.async must stay volatile.

7. **Static shared memory limit is 48 KB.** Above 48 KB requires `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`.

8. **Max shared memory per block is 99 KB** (128 KB per SM minus 1 KB CUDA overhead per block).
