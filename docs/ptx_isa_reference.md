# PTX ISA Reference for sm_120 Kernel Development

**Source:** NVIDIA PTX ISA 9.2 Documentation + empirical verification on RTX 5090
**Scope:** Instructions relevant to consumer Blackwell (sm_120) using `mma.sync` ISA. Excludes `tcgen05` (datacenter sm_100).

---

## 1. Register Types and Declaration

### PTX Register Types

| Type | Width | Description | Inline ASM Constraint |
|------|-------|-------------|----------------------|
| `.pred` | 1-bit | Predicate register | (no direct constraint) |
| `.b16` / `.u16` | 16-bit | Half-precision, unsigned 16 | `"h"` |
| `.b32` / `.u32` / `.s32` | 32-bit | General purpose integer | `"r"` |
| `.f32` | 32-bit | Single-precision float | `"f"` |
| `.b64` / `.u64` / `.s64` | 64-bit | 64-bit integer / pointer | `"l"` |
| `.f64` | 64-bit | Double-precision float | `"d"` |

### Declaration Syntax (inside PTX asm blocks)

```ptx
.reg .b32 r0, r1, r2;          // 32-bit general purpose
.reg .f32 f0, f1;              // Single-precision float
.reg .b16 h0, h1;              // 16-bit registers
.reg .pred p0, p1;             // Predicate registers
.reg .u64 addr;                // 64-bit address
```

### Special Registers

```ptx
mov.u32 r0, %tid.x;           // Thread ID (x, y, z)
mov.u32 r1, %ctaid.x;         // CTA (block) ID
mov.u32 r2, %laneid;          // Warp lane (0-31)
mov.u32 r3, %warpid;          // Warp ID within CTA
mov.u32 r4, %smid;            // SM ID
mov.u64 r5, %clock64;         // 64-bit cycle counter
mov.u32 r6, %nctaid.x;        // Grid dimension
mov.u32 r7, %ntid.x;          // Block dimension
```

---

## 2. mma.sync Instructions

### Overview

sm_120 uses `mma.sync.aligned` for tensor core operations. All 32 threads in a warp cooperate on one matrix multiply-accumulate: `D = A * B + C`.

The instruction is warp-synchronous -- all threads must participate.

### m16n8k16 — BF16/F16 (Primary for attention/GEMM)

**Instruction format:**
```ptx
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {d0, d1, d2, d3},           // D: 4x f32 output
    {a0, a1, a2, a3},           // A: 4x b32 (8 bf16 values)
    {b0, b1},                   // B: 2x b32 (4 bf16 values)
    {c0, c1, c2, c3};           // C: 4x f32 accumulator
```

**Register counts per thread:**
| Fragment | Registers | Type | Elements per thread |
|----------|-----------|------|-------------------|
| A (16x16) | 4x uint32 | `.b32` | 8 bf16 packed in pairs |
| B (16x8) | 2x uint32 | `.b32` | 4 bf16 packed in pairs |
| C (16x8) | 4x float | `.f32` | 4 f32 values |
| D (16x8) | 4x float | `.f32` | 4 f32 values |

**Supported type combinations for m16n8k16:**
```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32     // F16 in, F32 accum
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32   // BF16 in, F32 accum
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16     // F16 in, F16 accum (2 output regs)
```

**F16 accumulator variant (2 output registers):**
```ptx
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
    {d0, d1},                   // D: 2x b32 (4 f16 packed)
    {a0, a1, a2, a3},           // A: 4x b32
    {b0, b1},                   // B: 2x b32
    {c0, c1};                   // C: 2x b32 (4 f16 packed)
```

**Inline asm wrapper (from this project):**
```cpp
asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11, %12, %13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));
```

### m16n8k32 — FP8 (2x throughput vs BF16)

**Instruction format:**
```ptx
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    {d0, d1, d2, d3},           // D: 4x f32 output
    {a0, a1, a2, a3},           // A: 4x b32 (16 e4m3 values)
    {b0, b1},                   // B: 2x b32 (8 e4m3 values)
    {c0, c1, c2, c3};           // C: 4x f32 accumulator
```

**Register counts (same shape as BF16, but 2x K dimension):**
| Fragment | Registers | Type | Elements per thread |
|----------|-----------|------|-------------------|
| A (16x32) | 4x uint32 | `.b32` | 16 fp8 packed 4-per-register |
| B (32x8) | 2x uint32 | `.b32` | 8 fp8 packed 4-per-register |
| C (16x8) | 4x float | `.f32` | 4 f32 values |
| D (16x8) | 4x float | `.f32` | 4 f32 values |

**FP8 type combinations:**
```
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32   // E4M3 x E4M3
mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32   // E5M2 x E5M2
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32   // Mixed: E4M3 x E5M2
mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32   // Mixed: E5M2 x E4M3
```

### m16n8k16 Fragment Layouts (Empirically Verified on sm_120)

**Thread (lane) to matrix element mapping:**

Let `T = threadIdx.x % 32` (lane ID within warp).

**D-fragment output (16x8 matrix, f32):**
```
d0 -> D[T/4, (T%4)*2]         // row = T/4, col = (T%4)*2
d1 -> D[T/4, (T%4)*2 + 1]     // same row, next column
d2 -> D[T/4 + 8, (T%4)*2]     // 8 rows down
d3 -> D[T/4 + 8, (T%4)*2 + 1] // 8 rows down, next column
```

Lanes 0-3 cover rows 0,8 cols 0-1; lanes 4-7 cover rows 1,9 cols 0-1; etc.

**A-fragment input (16x16 matrix, bf16 packed in uint32):**

ldmatrix_x4 outputs: `r0=m0k0, r1=m0k1, r2=m1k0, r3=m1k1`
MMA expects:         `a0=m0k0, a1=m1k0, a2=m0k1, a3=m1k1`

**THE a1/a2 SWAP IS REQUIRED.** Pass `(r0, r2, r1, r3)` to MMA, or use `ldmatrix_x4_mma()` which bakes the swap into the operand order `{%0, %2, %1, %3}`.

**A-fragment addressing (ALT mapping, verified on sm_120):**
```
sub = lane_id / 8;                        // 0, 1, 2, 3
row = (sub/2)*8 + lane_id%8;              // 0-7 for sub 0,1; 8-15 for sub 2,3
col = (sub%2)*8;                          // 0 for sub 0,2; 8 for sub 1,3
// Each register holds bf16x2: elements at (row, col) and (row, col+1)
```

**B-fragment for A*B^T (e.g., Q*K^T):**
Load consecutive elements from same row:
```
b0 = {B[T/4, (T%4)*2], B[T/4, (T%4)*2+1]}     // bf16x2
b1 = {B[T/4, (T%4)*2+8], B[T/4, (T%4)*2+9]}   // bf16x2
```

**B-fragment for A*B (loaded via ldmatrix_x2_trans):**
`ldmatrix_x2_trans` gives B_col[k,n] = Bsrc[k,n], so the MMA computes A*B (not A*B^T). This is correct for P*V multiplication where V is already in the right layout.

### .satfinite Modifier

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.satfinite ...
```
Clamps output to finite range (no NaN/Inf propagation). Generally not used in attention kernels where we need to detect overflow.

---

## 3. ldmatrix Instructions

### Overview

`ldmatrix` is a warp-cooperative instruction that loads 8x8 matrix tiles from shared memory directly into registers in the layout expected by `mma.sync`. Each thread provides a shared memory address, and the instruction distributes the loaded data across all warp lanes.

### Syntax

```ptx
ldmatrix.sync.aligned.m8n8.x1.shared.b16 {r0}, [addr];
ldmatrix.sync.aligned.m8n8.x2.shared.b16 {r0, r1}, [addr];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0, r1, r2, r3}, [addr];

// Transposed variants (for column-major loads):
ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {r0}, [addr];
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {r0, r1}, [addr];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {r0, r1, r2, r3}, [addr];
```

### Variants

| Variant | Output Regs | Matrix Tiles | Use Case |
|---------|-------------|-------------|----------|
| `.x1` | 1 uint32 | One 8x8 | Small fragments |
| `.x2` | 2 uint32 | Two 8x8 | B-fragment for mma (2 regs) |
| `.x4` | 4 uint32 | Four 8x8 | A-fragment for mma (4 regs) |
| `.x2.trans` | 2 uint32 | Two 8x8, transposed | V loading for P*V |
| `.x4.trans` | 4 uint32 | Four 8x8, transposed | Column-major A loading |

### Addressing

- **Address:** Each thread provides a 32-bit shared memory address (use `__cvta_generic_to_shared()` to convert)
- **Alignment:** Source address must be 16-byte (128-bit) aligned
- **Thread mapping:** Threads 0-7 load row 0-7 of first 8x8 tile, threads 8-15 load row 0-7 of second tile, etc.
- Each thread's address points to a 128-bit (8 bf16 element) row

### ldmatrix_x4 Output Register Order vs MMA Input Order

This is the critical a1/a2 swap issue:

```
ldmatrix_x4 outputs:  r0 = sub0(m0,k0)  r1 = sub1(m0,k1)  r2 = sub2(m1,k0)  r3 = sub3(m1,k1)
mma.sync expects:     a0 = sub0(m0,k0)  a1 = sub2(m1,k0)  a2 = sub1(m0,k1)  a3 = sub3(m1,k1)
```

**Solution 1 (legacy):** Swap after load: `mma(..., r0, r2, r1, r3, ...)`
**Solution 2 (preferred):** Use `ldmatrix_x4_mma()` with operand reorder `{%0, %2, %1, %3}`

### Inline ASM Wrappers

```cpp
// Standard ldmatrix_x4
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    : "r"(smem_addr));

// ldmatrix_x4 with baked-in a1/a2 swap (preferred)
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %2, %1, %3}, [%4];\n"
    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
    : "r"(smem_addr));

// ldmatrix_x2_trans (for V loading)
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
    : "=r"(r0), "=r"(r1)
    : "r"(smem_addr));
```

---

## 4. cp.async Instructions

### Overview

`cp.async` copies data from global memory to shared memory bypassing registers. This frees register pressure and allows overlapping data movement with computation.

### Syntax

```ptx
// Cache-global policy (write to L2, invalidate L1 — preferred for streaming)
cp.async.cg.shared.global [dst_shared], [src_global], 16;       // 16 bytes
cp.async.cg.shared.global [dst_shared], [src_global], 8;        // 8 bytes
cp.async.cg.shared.global [dst_shared], [src_global], 4;        // 4 bytes

// Cache-all policy (fill both L1 and L2 — for data reused soon)
cp.async.ca.shared.global [dst_shared], [src_global], 16;
cp.async.ca.shared.global [dst_shared], [src_global], 8;
cp.async.ca.shared.global [dst_shared], [src_global], 4;

// With L2 cache hint (CUDA 11.4+)
cp.async.cg.shared.global.L2::128B [dst_shared], [src_global], 16;
cp.async.ca.shared.global.L2::128B [dst_shared], [src_global], 16;
```

### Size Variants

| Size | Bytes | Elements (BF16) | Elements (FP8) | Alignment |
|------|-------|-----------------|----------------|-----------|
| 4 | 4B | 2 bf16 | 4 fp8 | 4-byte |
| 8 | 8B | 4 bf16 | 8 fp8 | 8-byte |
| 16 | 16B | 8 bf16 | 16 fp8 | 16-byte |

### Partial Copy with Zero-Fill

```ptx
// Copy src-size bytes from global, zero-fill remainder up to cp-size
cp.async.cg.shared.global [dst_shared], [src_global], cp-size, src-size;
```

When `src-size < cp-size`, the remaining bytes in the destination are zero-filled. This is useful for boundary handling without separate conditional logic.

### Commit and Wait Groups

```ptx
cp.async.commit_group;         // Commit all pending cp.async ops into a group
cp.async.wait_group N;         // Wait until N or fewer groups remain in flight
cp.async.wait_all;             // Wait for all groups to complete (= wait_group 0)
```

**Pipelining pattern (double-buffer):**
```ptx
// Stage 0: issue first loads
cp.async.cg.shared.global [buf0], [src0], 16;
cp.async.commit_group;

// Stage 1: issue second loads while first completes
cp.async.cg.shared.global [buf1], [src1], 16;
cp.async.commit_group;
cp.async.wait_group 1;         // Wait for stage 0 to complete
// ... use buf0 ...

// Next iteration:
cp.async.cg.shared.global [buf0], [src2], 16;
cp.async.commit_group;
cp.async.wait_group 1;         // Wait for stage 1 to complete
// ... use buf1 ...
```

### Cache Policies

| Policy | Behavior | Use Case |
|--------|----------|----------|
| `.cg` | Cache in L2 only, bypass L1 | Streaming loads (tiles used once per block) |
| `.ca` | Cache in both L1 and L2 | Data reused multiple times by same warp |

### Inline ASM Wrappers

```cpp
// 16-byte copy with .cg policy
uint32_t dst = __cvta_generic_to_shared(dst_smem);
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
    :: "r"(dst), "l"(src_gmem));

// Commit and wait
asm volatile("cp.async.commit_group;\n");
asm volatile("cp.async.wait_group %0;\n" :: "n"(0));
```

---

## 5. Shuffle Instructions

### Syntax

```ptx
shfl.sync.idx.b32   dst, src, srcLane, clamp;     // Read from specific lane
shfl.sync.up.b32    dst, src, delta, clamp;        // Read from lane - delta
shfl.sync.down.b32  dst, src, delta, clamp;        // Read from lane + delta
shfl.sync.bfly.b32  dst, src, laneMask, clamp;     // Read from lane XOR mask
```

### Parameters

- **Membership mask:** Always `0xffffffff` for full-warp participation (first operand in C++ intrinsic, implicit in PTX `.sync` form)
- **clamp (c parameter):** For `.bfly` mode on sm_120, use `c = 31` (0x1f). The packed format `0x1f1f` does NOT work -- only `c[4:0]` matters for `.bfly`. Verified empirically.
- **dst/src:** `.b32` (32-bit) registers. For f32 data, use same register.

### C++ Intrinsics (preferred)

```cpp
float val = __shfl_xor_sync(0xffffffff, val, mask);   // Butterfly
float val = __shfl_sync(0xffffffff, val, lane);        // Direct read
float val = __shfl_down_sync(0xffffffff, val, delta);  // Down
float val = __shfl_up_sync(0xffffffff, val, delta);    // Up
```

### PTX Shuffle (for inline asm blocks)

```ptx
// Butterfly shuffle for warp reduction (XOR mask = 1, 2, 4, 8, 16)
shfl.sync.bfly.b32 f1, f0, 1, 31;     // Exchange with lane XOR 1
shfl.sync.bfly.b32 f1, f0, 2, 31;     // Exchange with lane XOR 2

// IMPORTANT: c parameter = 31 (NOT 0x1f1f)
// Empirically verified on sm_120: c=31 works, c=0x1f1f produces identity
```

### Predicated Shuffle

```ptx
shfl.sync.bfly.b32 dst|pred, src, laneMask, clamp;
// pred is set if source lane was valid (within clamp range)
```

---

## 6. Conversion Instructions

### bf16 Pack/Unpack

```ptx
// Pack two f32 values into one bf16x2 register
cvt.rn.bf16x2.f32 r_bf16x2, f_high, f_low;
// WARNING: Operand order is REVERSED vs C++!
//   PTX: first source (f_high) -> HIGH bits [31:16]
//        second source (f_low) -> LOW bits [15:0]
//   C++: __floats2bfloat162_rn(a, b) puts a in LOW, b in HIGH
// To match C++ order: cvt.rn.bf16x2.f32 d, b, a  (swap a and b)
```

This operand reversal was empirically verified -- without the swap, MMA gets garbage P fragments.

### f32 <-> bf16

```ptx
cvt.rn.bf16.f32 h0, f0;       // f32 -> bf16 (round to nearest)
cvt.rz.bf16.f32 h1, f1;       // f32 -> bf16 (round toward zero)
cvt.f32.bf16 f2, h2;           // bf16 -> f32
```

### f32 <-> f16

```ptx
cvt.rn.f16.f32 h0, f0;        // f32 -> f16
cvt.f32.f16 f0, h0;            // f16 -> f32
```

### FP8 Conversions

```ptx
cvt.rn.satfinite.e4m3x2.f32 r0, f_high, f_low;   // 2x f32 -> packed e4m3x2
cvt.rn.satfinite.e5m2x2.f32 r0, f_high, f_low;   // 2x f32 -> packed e5m2x2
cvt.f32.e4m3 f0, r0;                               // e4m3 -> f32
cvt.f32.e5m2 f0, r0;                               // e5m2 -> f32
```

### Rounding Modes

| Mode | Suffix | Description |
|------|--------|-------------|
| Round to nearest even | `.rn` | Default, IEEE 754 |
| Round toward zero | `.rz` | Truncation |
| Round toward +inf | `.rp` | Ceiling |
| Round toward -inf | `.rm` | Floor |

---

## 7. Arithmetic and Math Instructions

### Basic Arithmetic

```ptx
add.f32 d, a, b;              // d = a + b
sub.f32 d, a, b;              // d = a - b
mul.f32 d, a, b;              // d = a * b
fma.rn.f32 d, a, b, c;        // d = a * b + c (fused)
neg.f32 d, a;                  // d = -a
abs.f32 d, a;                  // d = |a|
max.f32 d, a, b;              // d = max(a, b)
min.f32 d, a, b;              // d = min(a, b)
```

### Special Math (MUFU Instructions)

These map to the Multi-Function Unit and execute in 1-2 cycles:

```ptx
ex2.approx.f32 d, a;          // d = 2^a (fast approximation, maps to MUFU.EX2)
ex2.approx.ftz.f32 d, a;      // Same with flush-to-zero
lg2.approx.f32 d, a;          // d = log2(a) (maps to MUFU.LG2)
rcp.approx.f32 d, a;          // d = 1/a (maps to MUFU.RCP)
rcp.approx.ftz.f32 d, a;      // Same with flush-to-zero
rsqrt.approx.f32 d, a;        // d = 1/sqrt(a) (maps to MUFU.RSQ)
sin.approx.f32 d, a;          // d = sin(a) (limited range)
cos.approx.f32 d, a;          // d = cos(a) (limited range)
```

**Key fact:** `ex2.approx.f32` in PTX matches `exp2f()` in C++ under `--use_fast_math`. Both map to `MUFU.EX2` in SASS. Safe to use interchangeably. Verified on sm_120.

### Integer Arithmetic

```ptx
add.u32 d, a, b;              // Unsigned add
add.s32 d, a, b;              // Signed add
mul.lo.u32 d, a, b;           // Low 32 bits of multiply
mul.hi.u32 d, a, b;           // High 32 bits
mad.lo.u32 d, a, b, c;        // d = a*b + c (low)
shl.b32 d, a, bits;           // Left shift
shr.u32 d, a, bits;           // Right shift (unsigned)
and.b32 d, a, b;              // Bitwise AND
or.b32 d, a, b;               // Bitwise OR
xor.b32 d, a, b;              // Bitwise XOR
```

---

## 8. Control Flow

### Predicate Set (Comparison)

```ptx
setp.lt.f32 p, a, b;          // p = (a < b)
setp.le.f32 p, a, b;          // p = (a <= b)
setp.gt.f32 p, a, b;          // p = (a > b)
setp.ge.f32 p, a, b;          // p = (a >= b)
setp.eq.f32 p, a, b;          // p = (a == b)
setp.ne.f32 p, a, b;          // p = (a != b)
setp.lt.s32 p, a, b;          // Integer comparison (signed)
setp.lt.u32 p, a, b;          // Integer comparison (unsigned)
setp.ne.b32 p, a, 0;          // Boolean test
```

### Predicated Execution

```ptx
@p  mov.b32 r0, r1;           // Execute only if p is true
@!p add.f32 f0, f1, f2;       // Execute only if p is false
@p  cp.async.cg.shared.global [dst], [src], 16;  // Conditional copy
```

### Branch

```ptx
bra target_label;             // Unconditional branch
bra.uni target_label;         // Uniform branch (all threads agree)
@p bra target_label;          // Conditional branch
```

### Labels

```ptx
loop_start:
    // ... loop body ...
    @p bra loop_start;
```

---

## 9. Synchronization and Memory Ordering

### Barrier (Intra-CTA)

```ptx
bar.sync 0;                   // All threads in CTA wait at barrier 0
bar.sync 0, 128;              // Wait for 128 threads (partial barrier)
bar.warp.sync 0xffffffff;     // Warp-level sync (all lanes)
```

### Memory Fence

```ptx
membar.cta;                   // CTA-scope fence (shared memory ordering)
membar.gl;                    // Global scope fence
membar.sys;                   // System scope fence
fence.acq_rel.cta;            // Acquire-release, CTA scope
fence.acq_rel.gpu;            // Acquire-release, GPU scope
```

---

## 10. Memory Operations

### Global Memory

```ptx
ld.global.b32 r0, [addr];             // Load 32-bit
ld.global.v4.b32 {r0,r1,r2,r3}, [addr]; // Load 128-bit (vectorized)
st.global.b32 [addr], r0;             // Store 32-bit
st.global.v4.b32 [addr], {r0,r1,r2,r3}; // Store 128-bit
```

### Shared Memory

```ptx
ld.shared.b32 r0, [addr];             // Load from shared
ld.shared.v4.u32 {r0,r1,r2,r3}, [addr]; // Vectorized load
st.shared.b32 [addr], r0;             // Store to shared
st.shared.v4.u32 [addr], {r0,r1,r2,r3}; // Vectorized store
```

### Cache Hints

```ptx
ld.global.cg.b32 r0, [addr];          // Cache global (L2 only)
ld.global.ca.b32 r0, [addr];          // Cache all (L1 + L2)
ld.global.cs.b32 r0, [addr];          // Cache streaming
```

---

## 11. Inline ASM Conventions Summary

### Constraint Quick Reference

| Constraint | PTX Type | C++ Type | Width |
|-----------|----------|----------|-------|
| `"r"` | `.u32` | `uint32_t`, `int` | 32-bit |
| `"f"` | `.f32` | `float` | 32-bit |
| `"d"` | `.f64` | `double` | 64-bit |
| `"l"` | `.u64` | `uint64_t`, pointer | 64-bit |
| `"h"` | `.u16` | `uint16_t` | 16-bit |
| `"n"` | immediate | compile-time constant | N/A |
| `"C"` | string | `constexpr char[]` | N/A |

### Output/Input Modifiers

| Modifier | Meaning |
|----------|---------|
| `"=r"` | Write-only output register |
| `"+r"` | Read-write register (for conditional updates) |
| `"r"` | Read-only input register |

### volatile vs non-volatile

- **`asm volatile`**: Compiler cannot delete or reorder. Required for: ldmatrix, cp.async, barriers, memory operations
- **`asm`** (non-volatile): Compiler may reorder relative to other non-volatile asm and C++ code. Use for: mma.sync in compute-bound kernels (allows better interleaving with loads)

### Scoping temp registers

Always wrap temp register declarations in braces to avoid namespace conflicts when inlined:
```cpp
asm("{\n\t"
    " .reg .u32 tmp;\n\t"
    " mul.lo.u32 tmp, %1, %1;\n\t"
    " mul.lo.u32 %0, tmp, %1;\n\t"
    "}"
    : "=r"(y) : "r"(x));
```

### Operand numbering

`%0` = first output, `%1` = second output, ..., then inputs continue the numbering.

### Escaping

Use `%%` for PTX special registers: `%%clock`, `%%laneid`, `%%tid.x`.

---

## 12. PTX ISA Gotchas for sm_120 (Empirically Verified)

1. **cvt.rn.bf16x2.f32 operand order is REVERSED vs C++.** PTX puts first source in HIGH bits. See section 6.

2. **shfl.sync.bfly c parameter = 31, NOT 0x1f1f.** Only c[4:0] matters for .bfly mode. See section 5.

3. **Monolithic asm blocks with >50 "+f" operands produce wrong results.** Threshold is between 34 (works) and 66 (broken). ptxas register allocator limitation. See `04_HARD_WON_LESSONS.md`.

4. **ptxas reorders instructions within asm blocks aggressively.** For blocks under ~50 operands, the programmer's instruction order has no effect on final SASS. The compiler chooses its preferred schedule regardless.

5. **Non-volatile MMA is always correct.** The compiler will not delete MMA instructions because they have output operands. Using `asm` instead of `asm volatile` for MMA allows beneficial reordering.

6. **ldmatrix and cp.async MUST be volatile.** They have side effects (shared memory writes/loads) that the compiler cannot track through operands alone.
