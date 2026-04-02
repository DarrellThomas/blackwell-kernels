# MXFP8 Native MMA Implementation Guide for sm_120

**Sources:**
- [Triton PR #7918: Native MXFP FP8 scaled_dot for SM120](https://github.com/triton-lang/triton/pull/7918)
- [NVIDIA Forum: mma.sync block_scale on sm_120a](https://forums.developer.nvidia.com/t/run-ptx-mma-sync-aligned-kind-mxf8f6f4-block-scale-scale-vec-1x-m16n8k32-on-sm-120a/329702)
- [CUTLASS mma_sm120.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm120.hpp)
- [Triton Block-Scaled Matmul Tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
- [OCP MX Scaling Formats](https://fprox.substack.com/p/ocp-mx-scaling-formats)
- [Fal.ai MXFP8 Quantizer on Blackwell](https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/)
- [Cursor MXFP8 Kernels Blog](https://cursor.com/blog/kernels)
- [Hyunsung Lee: Adding Scaled Dot Product to Triton](https://ita9naiwa.github.io/mlsys/2025/08/29/triton-first-pr.html)
- [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html)

**Relevant to:** attention, gemm, fused-mlp workers
**Worker's current problem:** Standard FP8 uses per-tensor scaling (limited precision) and software scale multiply (adds ALU overhead). Attention FP8 has 14:1 conversion-to-MMA ratio with 448 ALU instructions per KV block for BF16-to-FP8 conversion.

---

## 1. MXFP8 Instruction Details

### 1.1 Full PTX Instruction

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
```

Breakdown of qualifiers:
- `kind::mxf8f6f4` -- microscaling format family (FP8, FP6, FP4)
- `block_scale` -- hardware applies per-block scale factors
- `scale_vec::1X` -- one scale factor per 32-element K-block (the only option for mxf8f6f4, since atom_K=32)
- `m16n8k32` -- same tile dimensions as standard FP8 MMA
- `f32.e4m3.e4m3.f32` -- D=f32, A=e4m3, B=e4m3, C=f32
- `ue8m0` -- unsigned 8-bit exponent scale factor type

### 1.2 Supported Type Combinations

From CUTLASS `mma_sm120.hpp`, all these input type combinations are supported:
- `e4m3 x e4m3` (our primary target)
- `e5m2 x e5m2`
- `e4m3 x e5m2` and `e5m2 x e4m3`
- `e4m3 x e3m2`, `e4m3 x e2m3`, `e4m3 x e2m1` (and all reverse combos)
- `e5m2 x e3m2`, `e5m2 x e2m3`, `e5m2 x e2m1` (and all reverse combos)

All variants use `f32` output and `ue8m0` scale factors.

### 1.3 Build Requirement

**CRITICAL: Must compile with `-arch=sm_120a` (with the "a" suffix).**

Using plain `sm_120` causes ptxas error: `"Feature '.block_scale' not supported on .target 'sm_120'"`. The RTX 5090 supports sm_120a. The "a" suffix enables architecture-accelerated features including block_scale MMA.

### 1.4 PTX ISA Version

The instruction requires PTX ISA 8.7+ (introduced with CUDA 12.8). CUDA 13.0 includes PTX ISA 9.0 which fully supports this. Our toolchain is compatible.

---

## 2. Inline PTX Assembly Template

### 2.1 Complete asm() Template

From CUTLASS `mma_sm120.hpp` (SM120::BLOCKSCALED namespace):

```c
asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
    ".m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"   // D: 4x f32 output registers
    "{%4,  %5,  %6,  %7},"   // A: 4x uint32 (16 rows x 32 cols, packed FP8)
    "{%8,  %9},"              // B: 2x uint32 (8 rows x 32 cols, packed FP8)
    "{%10, %11, %12, %13},"   // C: 4x f32 accumulator input
    "{%14},"                  // SFA: scale factor for A (uint32, 4 packed e8m0 bytes)
    "{%15, %16},"             // bidA, tidA: byte-selector and thread-selector for A
    "{%17},"                  // SFB: scale factor for B (uint32, 4 packed e8m0 bytes)
    "{%18, %19};\n"           // bidB, tidB: byte-selector and thread-selector for B
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                         // outputs
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),                             // A operands
      "r"(b0), "r"(b1),                                                // B operands
      "f"(c0), "f"(c1), "f"(c2), "f"(c3),                             // C operands
      "r"(uint32_t(sfa)), "h"(bidA), "h"(tidA),                       // A scale + selectors
      "r"(uint32_t(sfb)), "h"(bidB), "h"(tidB)                        // B scale + selectors
);
```

### 2.2 Operand Summary

| Operand | Count | Type | Constraint | Description |
|---------|-------|------|------------|-------------|
| D (output) | 4 | float | `"=f"` | Accumulated result |
| A (matrix) | 4 | uint32_t | `"r"` | 16x32 FP8 tile, packed 4 per uint32 |
| B (matrix) | 2 | uint32_t | `"r"` | 8x32 FP8 tile, packed 4 per uint32 |
| C (accum)  | 4 | float | `"f"` | Accumulator input |
| SFA (scale A) | 1 | uint32_t | `"r"` | 4 packed e8m0 bytes for A rows |
| bidA | 1 | uint16_t | `"h"` | Byte selector for A scale |
| tidA | 1 | uint16_t | `"h"` | Thread selector for A scale |
| SFB (scale B) | 1 | uint32_t | `"r"` | 4 packed e8m0 bytes for B columns |
| bidB | 1 | uint16_t | `"h"` | Byte selector for B scale |
| tidB | 1 | uint16_t | `"h"` | Thread selector for B scale |

**Total operands: 20** (4 outputs + 16 inputs) vs standard FP8 MMA's 10 (4 outputs + 6 inputs).

### 2.3 Comparison with Standard FP8 MMA

Our current standard FP8 MMA:
```c
asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3)
);
```

The MXFP8 variant adds 6 extra operands: SFA, bidA, tidA, SFB, bidB, tidB.

---

## 3. Scale Factor Mechanism: byte-selector and thread-selector

### 3.1 How Scale Factors Are Packed

Each scale factor register (SFA or SFB) is a **32-bit register containing 4 packed e8m0 bytes**. The 4 bytes correspond to 4 different MMA tiles (different M-rows for A, different N-columns for B).

```
SFA (uint32_t): [byte3 | byte2 | byte1 | byte0]
                  ^        ^       ^       ^
                  row 3    row 2   row 1   row 0
                  (of 16-row groups within this thread's scale data)
```

### 3.2 byte-selector (bidA, bidB)

The byte-selector is a 2-bit value (0-3) that selects which of the 4 bytes in the packed 32-bit scale register to use for THIS particular MMA invocation.

For the A scale factor:
- `bidA = 0` selects byte 0 (bits 7:0) of the SFA register
- `bidA = 1` selects byte 1 (bits 15:8)
- `bidA = 2` selects byte 2 (bits 23:16)
- `bidA = 3` selects byte 3 (bits 31:24)

This allows packing 4 scale factors for 4 different M-tile positions into a single 32-bit register, then selecting the right one per MMA call.

### 3.3 thread-selector (tidA, tidB)

The thread-selector determines which thread-pair within a quad provides the scale factor values:
- `tidA = 0` -- the thread-pair with `%laneid % 4 == 0 or 1` provides SFA values
- `tidA = 1` -- the thread-pair with `%laneid % 4 == 2 or 3` provides SFA values

### 3.4 CUTLASS Default Values

CUTLASS sets all selectors to 0:
```c
static constexpr uint16_t tidA = 0;
static constexpr uint16_t bidA = 0;
static constexpr uint16_t tidB = 0;
static constexpr uint16_t bidB = 0;
```

This means: use byte 0 of the SFA/SFB register, from the lower thread-pair in each quad.

### 3.5 Triton's Scale Factor Selection Logic

From the Triton PR #7918 `callMmaScaled()` function, the byte-selector is computed dynamically:

```c
unsigned aByte = (m / 2) & 0x3;  // m = M-tile index within the warp
unsigned bByte = n & 0x3;         // n = N-tile index within the warp
appendScale(aScaleValue, aByte, /*threadId*/ 0);
appendScale(bScaleValue, bByte, /*threadId*/ 0);
```

This means:
- For **A scales**: the byte-selector cycles through 0,0,1,1,2,2,3,3 as the M-tile index increases (each m16 MMA covers 2 "m groups", hence `m/2`)
- For **B scales**: the byte-selector cycles through 0,1,2,3 as the N-tile index increases
- Thread-selector is always 0

### 3.6 Scale Factor Packing in Triton

Triton packs 4 scale bytes into a uint32 per scale register:

```c
auto pack4BytesToI32 = [&](ArrayRef<Value> bytes) -> Value {
    Value acc = i32_val(0);
    for (int i = 0; i < 4; ++i) {
        Value bv = (i < bytes.size()) ? toI32(bytes[i]) : i32_val(0);
        acc = or_(acc, shl(bv, i32_val(8 * i)));
    }
    return acc;
};
```

This packs up to 4 e8m0 scale values into consecutive bytes of a uint32.

---

## 4. Scale Factor Layout Across Warp Threads

### 4.1 Scale Layout for A (per-row)

For an m16n8k32 MMA with `scale_vec::1X`:
- A is 16x32 (16 rows, 32 K-elements)
- With block size = 32, each of the 16 rows needs exactly 1 scale factor
- Total: 16 e8m0 scale values for A

The scale factor is expected in a "sub-column" of a 32-lane x 4-column tile. The 32 warp threads each contribute to the scale distribution.

From Triton's `getSM120DotScaledScaleLayout` (LinearLayoutConversions.cpp), the A-scale layout for basic config (128x32, warpsPerCTA=[4,1]):

```
register: {0,1}, {0,2}, {0,4}, {0,8}, {0,16}, {64,0}
lane:     {8,0}, {0,0}, {1,0}, {2,0}, {4,0}
warp:     {16,0}, {32,0}
```

This tells us:
- Lane 0 contributes scale for row 8, lanes 2-4 contribute rows 1-4
- Lanes 1 is unused (zero contribution to row dimension)
- The scale factors are distributed across specific lanes, with lane 0 at row offset 8

### 4.2 Scale Layout for B (per-column)

For m16n8k32 MMA:
- B is 8x32 (8 columns, 32 K-elements)
- With block size = 32, each of the 8 columns needs 1 scale factor
- Total: 8 e8m0 scale values for B

From Triton's layout (128x32, warpsPerCTA=[1,4]):

```
register: {0,1}, {0,2}, {0,4}, {0,8}, {0,16}, {0,32}, {0,64}
lane:     {0,0}, {0,0}, {1,0}, {2,0}, {4,0}
warp:     {8,0}, {16,0}
```

B scales are spread across the N (column) dimension through registers.

### 4.3 Practical Implication for Hand-Written Kernels

The key insight: **scales are packed 4 per uint32 register, with byte-selectors choosing which byte per MMA call**. When iterating over M-tiles:
- Pack 4 row-scales into one uint32 for SFA
- Use bidA = 0,1,2,3 for successive M-tile MMAs

When iterating over N-tiles:
- Pack 4 column-scales into one uint32 for SFB
- Use bidB = 0,1,2,3 for successive N-tile MMAs

For a GEMM with 4 M-tiles and 4 N-tiles per warp:
- 1 SFA register holds all 4 M-scales
- 1 SFB register holds all 4 N-scales
- Each MMA call selects the right byte via bidA/bidB

---

## 5. Scale Format: e8m0 (UE8M0)

### 5.1 Encoding

| Field | Bits | Description |
|-------|------|-------------|
| Exponent | 8 (all bits) | Unsigned biased exponent |
| Bias | 127 | Same as IEEE 754 single-precision exponent |
| Mantissa | 0 | Implicit 1.0 (no mantissa bits) |

### 5.2 Value Interpretation

```
value = 2^(encoded_byte - 127)
```

Examples:
| Encoded (uint8) | Binary | Value |
|----------------|--------|-------|
| 0 | 0x00 | 2^(-127) ~ 5.88e-39 |
| 1 | 0x01 | 2^(-126) ~ 1.18e-38 |
| 127 | 0x7F | 2^0 = 1.0 |
| 128 | 0x80 | 2^1 = 2.0 |
| 200 | 0xC8 | 2^73 ~ 9.44e21 |
| 254 | 0xFE | 2^127 ~ 1.70e38 |
| 255 | 0xFF | **NaN** (reserved) |

### 5.3 Dynamic Range

- Minimum positive: 2^(-127) ~ 5.88e-39
- Maximum finite: 2^127 ~ 1.70e38
- Resolution: only powers of 2 (no intermediate values)

### 5.4 Rounding to e8m0

The OCP MX spec uses **round towards zero** (floor of log2):
```c
// Given: amax = max absolute value in 32-element block
// FP8 E4M3 max finite = 448.0
uint8_t scale_e8m0 = (uint8_t)floorf(log2f(amax / 448.0f)) + 127;
```

Alternative (round up to avoid overflow, used in some implementations):
```c
uint8_t scale_e8m0 = (uint8_t)ceilf(log2f(amax / 448.0f)) + 127;
```

Round-up is safer (prevents saturation) but wastes ~0.5 bits of precision on average. The Fal.ai blog uses round-up. The MX spec uses round-towards-zero.

---

## 6. In-Kernel MXFP8 Quantization

### 6.1 Quantization Algorithm

For each 32-element block of BF16/FP32 input:

```c
// 1. Find block maximum
float amax = 0.0f;
for (int i = 0; i < 32; i++) {
    amax = fmaxf(amax, fabsf(input[i]));
}

// 2. Compute e8m0 scale (round up to avoid overflow)
// 448.0 = max finite value of FP8 E4M3
float scale_float = amax / 448.0f;
int exponent = (scale_float > 0) ? (int)ceilf(log2f(scale_float)) : -127;
exponent = max(-127, min(127, exponent));
uint8_t scale_e8m0 = (uint8_t)(exponent + 127);

// 3. Quantize each element
float inv_scale = 1.0f / (float)(1 << (exponent));  // = 2^(-exponent)
for (int i = 0; i < 32; i++) {
    output_fp8[i] = __nv_cvt_float_to_e4m3(input[i] * inv_scale);
}
```

### 6.2 Cost Analysis for In-Kernel Quantization

Per 32-element block:
- Block max: ~32 fabs + 31 fmax = ~63 instructions
- Scale computation: ~5 instructions (log2f, ceilf, clamp, add, shift)
- Element quantization: ~32 multiplies + 32 CVT instructions = ~64 instructions
- **Total: ~132 instructions per 32 elements = ~4.1 instructions per element**

Compare to current BF16-to-FP8 conversion (no scaling): ~7 SASS per 4 elements = 1.75 per element.

**MXFP8 in-kernel quantization is ~2.3x more expensive than simple conversion.** However, it eliminates the post-MMA software scale multiply, and the per-block precision is much better.

### 6.3 Optimization: Warp-Cooperative Block Max

The block max can be computed cooperatively:
```c
// Each thread holds some elements, reduce within warp
float thread_max = ...; // max of this thread's elements
// Warp shuffle reduction for block max
thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, 1));
thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, 2));
// etc.
```

For K=32 with 32 threads, each thread holds exactly 1 element per row-block. The warp shuffle reduction for 32 elements across 32 lanes is efficient.

### 6.4 Pre-Computed Scales (Preferred for GEMM)

For GEMM, scales can be pre-computed on the host/in a separate kernel:
```python
# Host-side (PyTorch)
blocks = tensor.reshape(M, K // 32, 32)
block_max = blocks.abs().max(dim=-1).values  # (M, K//32)
scale_e8m0 = torch.clamp(torch.ceil(torch.log2(block_max / 448.0)) + 127, 0, 254).to(torch.uint8)
tensor_fp8 = (tensor / (2.0 ** (scale_e8m0.unsqueeze(-1).float() - 127))).to(torch.float8_e4m3fn)
```

Scale tensor shape: `(M, K/32)` for A, `(N, K/32)` for B.

---

## 7. Shared Memory Layout for MXFP8

### 7.1 Data + Scale Organization

The FP8 element data and e8m0 scales should be stored in **separate smem regions**:

```
smem layout:
[A_fp8_data: M_tile * K_tile bytes]        // e.g., 64*64 = 4096 bytes
[B_fp8_data: N_tile * K_tile bytes]        // e.g., 64*64 = 4096 bytes
[A_scales: M_tile * (K_tile/32) bytes]     // e.g., 64*2 = 128 bytes
[B_scales: N_tile * (K_tile/32) bytes]     // e.g., 64*2 = 128 bytes
```

Scale overhead is tiny: 1 byte per 32 data bytes = **3.1% overhead**.

### 7.2 cp.async for Data Loading

FP8 data loads work identically to standard FP8:
```c
// Load FP8 data from global to smem
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;  // 16 bytes = 16 FP8 values
```

### 7.3 Scale Factor Loading

Scales are much smaller (1/32 the size of data). Options:
1. **cp.async alongside data** -- load scale tiles at the same time as data tiles
2. **Scalar loads** -- scales are small enough to load with regular LD.SHARED
3. **Register broadcast** -- for K=32 (single K-step), one scale per row/col can be loaded once

For a 64x64 tile with K_tile=64:
- A_scales: 64 rows x 2 K-blocks = 128 bytes (trivial)
- B_scales: 64 cols x 2 K-blocks = 128 bytes (trivial)

### 7.4 Double Buffering

Double buffer both data and scales together:
```
[A_data_buf0] [A_data_buf1] [B_data_buf0] [B_data_buf1]
[A_scale_buf0] [A_scale_buf1] [B_scale_buf0] [B_scale_buf1]
```

Scale buffers add negligible smem: ~256 bytes per buffer pair (vs ~8KB for data).

---

## 8. Register Pressure Analysis

### 8.1 Per-MMA Register Usage

Standard FP8 MMA: 4(D) + 4(A) + 2(B) + 4(C) = **14 registers**
MXFP8 MMA: 4(D) + 4(A) + 2(B) + 4(C) + 1(SFA) + 1(bidA) + 1(tidA) + 1(SFB) + 1(bidB) + 1(tidB) = **20 registers**

However, bidA, tidA, bidB, tidB are **compile-time constants** (uint16 immediates), so they don't consume physical registers. The real extra cost is:
- **SFA: 1 uint32 register** (4 packed e8m0 scales)
- **SFB: 1 uint32 register** (4 packed e8m0 scales)

**Net increase: +2 registers per active MMA pair for scales.**

### 8.2 Scale Packing Amortization

Since 4 scale bytes pack into 1 uint32, and bidA/bidB select which byte per MMA:
- For 4 M-tiles (repM=4): 1 SFA register serves all 4 MMAs
- For 4 N-tiles (repN=4): 1 SFB register serves all 4 MMAs

With K-tiling of 64 (2 K-steps of 32): need 2 SFA and 2 SFB registers for double-buffering scales across K-steps.

**Worst case: +4 registers total** (2 SFA + 2 SFB for double-buffered K).

### 8.3 Impact on Our Kernels

- **Attention** (165 regs, 5 spare before hitting 3-block occupancy threshold): +4 registers is tight but feasible. We'd be at 169 regs vs 170.7 threshold.
- **GEMM** (occupancy-first, 6 blocks/SM): +4 registers is negligible relative to the ~128 register budget.

---

## 9. Integration with Existing Kernels

### 9.1 GEMM Kernel Changes

The GEMM kernel currently uses standard FP8 MMA with pre-quantized FP8 inputs and a per-tensor scale applied after the GEMM:

```
D = (A_fp8 @ B_fp8) * scale_A * scale_B
```

With MXFP8:

```
D = (A_fp8 * SFA) @ (B_fp8 * SFB)   // hardware applies per-block scales
```

Required changes:
1. **Input format**: Accept MXFP8 data + separate scale tensors from host
   - A_data: `(M, K)` as FP8 e4m3 (same as now)
   - A_scales: `(M, K/32)` as uint8 (e8m0) -- NEW
   - B_data: `(N, K)` as FP8 e4m3 (same as now, but note B scales are `(N, K/32)` not transposed)
   - B_scales: `(N, K/32)` as uint8 (e8m0) -- NEW

2. **smem allocation**: Add scale regions (~256 bytes per double-buffer stage, negligible)

3. **MMA call**: Replace standard FP8 MMA with MXFP8 MMA (add scale operands)

4. **Epilogue**: Remove the per-tensor scale multiply (hardware already applied per-block scales)

5. **Build flag**: Change `-arch=sm_120` to `-arch=sm_120a`

### 9.2 Attention Kernel Changes

The attention kernel's FP8 path currently does BF16-to-FP8 conversion in registers:

```
BF16 Q,K,V -> in-register cvt -> FP8 fragments -> standard FP8 MMA
```

With MXFP8, two approaches:

**Approach A: Pre-quantized MXFP8 inputs (simpler, for inference)**
- Accept Q, K, V as pre-quantized MXFP8 (data + scales from host)
- Load FP8 data via cp.async/ldmatrix as now
- Load scale factors separately (tiny, 128 bytes per 64x64 tile)
- Use MXFP8 MMA with hardware scaling
- Eliminates ALL in-kernel conversion (the 448 ALU instructions per KV block)
- **Expected speedup: significant** -- removes the 14:1 conversion bottleneck

**Approach B: In-kernel MXFP8 quantization (for training or BF16 inputs)**
- Load BF16 K/V to smem
- Convert BF16 -> MXFP8 in registers (compute block max, scale, quantize)
- Use MXFP8 MMA
- More complex quantization than current simple BF16-to-FP8 cvt
- May not be faster than current path due to increased quantization cost

**Recommendation: Start with Approach A.** This eliminates conversion entirely and is the cleanest path to measuring raw MXFP8 MMA throughput.

### 9.3 Critical Detail: ldmatrix Compatibility

The MXFP8 MMA uses the **same A/B fragment layout as standard FP8 MMA** (m16n8k32). This means:
- ldmatrix works identically for loading FP8 data fragments
- The existing fragment building code (ldmatrix_x4, a1/a2 swap) is unchanged
- Only the MMA call itself changes (additional scale operands)

This is the key advantage: **MXFP8 is a drop-in replacement at the MMA level**, not a memory layout change.

---

## 10. Performance Expectations

### 10.1 Triton PR #7918 Benchmarks

Llama3-8B-Instruct inference (in_len=1024, out_len=1024, batch=128, RTX 5090):
| Configuration | Time | Relative |
|-------------|------|----------|
| Standard FP8 x FP8 | 42.83s | 1.00x |
| MXFP8 native (this instruction) | 44.45s | 0.96x (4% slower) |
| MXFP8 emulated | 76.44s | 0.56x (44% slower) |

**Key insight: Native MXFP8 is ~4% slower than standard FP8 in end-to-end inference.** This suggests the MMA throughput is comparable but there's a small overhead from scale management (loading scales, packing, passing to instruction).

### 10.2 Expected Impact on Our Kernels

**GEMM:**
- Currently at 1.34x cuBLAS with standard FP8
- MXFP8 adds per-block precision at near-identical throughput
- Main benefit: better accuracy (precision improvement), not speed
- Small perf overhead from scale loading (~1-3%)

**Attention:**
- Currently at 2.33x SDPA with FP8 (52 us, SM 43.8%)
- **With pre-quantized MXFP8 inputs**: eliminates 448 ALU conversion instructions
  - Removes ~7.5 us of conversion overhead per KV block
  - Could bring kernel from 52 us toward ~45 us
  - SM utilization should increase (less ALU bottleneck, more MMA throughput)
- **With in-kernel MXFP8 quantization**: likely similar or worse than current path
  - Block-max computation + scale derivation adds ~2.3x more ALU than simple cvt
  - Only useful if precision gain justifies the cost

### 10.3 Precision Improvement

Per-block scaling (32-element blocks) vs per-tensor scaling:
- Per-tensor: one scale for the entire tensor -- outliers dominate, compressing the range for all elements
- Per-block: one scale per 32 elements -- each block adapts to local magnitude
- FlashAttention-3 reports **2.6x smaller error** with block quantization vs per-tensor
- GEMM relative error should drop from ~3.7% significantly

---

## 11. Complete Working Example

### 11.1 Minimal MXFP8 MMA Wrapper

```c
__device__ void mma_mxfp8_m16n8k32(
    float &d0, float &d1, float &d2, float &d3,       // output
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, // A fragments (FP8)
    uint32_t b0, uint32_t b1,                           // B fragments (FP8)
    float c0, float c1, float c2, float c3,             // accumulator
    uint32_t sfa_packed,                                 // 4 e8m0 scales for A rows
    uint16_t bidA,                                       // byte selector (0-3)
    uint32_t sfb_packed,                                 // 4 e8m0 scales for B cols
    uint16_t bidB                                        // byte selector (0-3)
) {
    uint16_t tidA = 0;  // always 0 (lower thread-pair in quad)
    uint16_t tidB = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa_packed), "h"(bidA), "h"(tidA),
          "r"(sfb_packed), "h"(bidB), "h"(tidB)
    );
}
```

### 11.2 Scale Factor Packing Helper

```c
// Pack 4 e8m0 scale bytes into a uint32
__device__ uint32_t pack_scales_4x(uint8_t s0, uint8_t s1, uint8_t s2, uint8_t s3) {
    return (uint32_t)s0
         | ((uint32_t)s1 << 8)
         | ((uint32_t)s2 << 16)
         | ((uint32_t)s3 << 24);
}
```

### 11.3 Usage Pattern in K-Loop

```c
// For each K-step of 32 elements:
for (int k = 0; k < K_STEPS; k++) {
    // Load A and B fragments (same as standard FP8)
    // ... ldmatrix loads ...

    // Load scale factors for this K-block
    // A scales: 1 per row, pack 4 rows into uint32
    uint32_t sfa = pack_scales_4x(
        a_scale[row_base + 0][k],  // scale for rows 0-15 of first m16 tile
        a_scale[row_base + 1][k],  // scale for rows 16-31 (second m16 tile)
        a_scale[row_base + 2][k],  // etc.
        a_scale[row_base + 3][k]
    );

    uint32_t sfb = pack_scales_4x(
        b_scale[col_base + 0][k],
        b_scale[col_base + 1][k],
        b_scale[col_base + 2][k],
        b_scale[col_base + 3][k]
    );

    // Issue MMAs with byte selectors
    for (int m = 0; m < REP_M; m++) {
        for (int n = 0; n < REP_N; n++) {
            mma_mxfp8_m16n8k32(
                d[m][n][0], d[m][n][1], d[m][n][2], d[m][n][3],
                a[m][0], a[m][1], a[m][2], a[m][3],
                b[n][0], b[n][1],
                d[m][n][0], d[m][n][1], d[m][n][2], d[m][n][3],
                sfa, (uint16_t)(m & 0x3),   // byte-selector for A
                sfb, (uint16_t)(n & 0x3)    // byte-selector for B
            );
        }
    }
}
```

---

## 12. Caveats and Unknowns

### 12.1 Confirmed Facts
- The instruction works on sm_120a (confirmed by Triton PR #7918, CUTLASS, NVIDIA forum)
- Fragment layout for A/B data is identical to standard FP8 MMA m16n8k32
- Scale factors are packed 4-per-uint32 with byte selectors
- CUDA 13 / PTX ISA 9.0 supports this instruction
- RTX 5090 supports sm_120a

### 12.2 Unknowns Requiring Verification
- **Exact thread-to-scale mapping**: The Triton LinearLayout code and CUTLASS use specific scale distribution patterns. Our hand-written kernels need to match this exactly. The safest approach: start with CUTLASS's default (bidA=0, tidA=0, pack only the needed scale in byte 0) and expand.
- **MMA throughput parity**: Is the MXFP8 MMA exactly the same throughput as standard FP8 MMA, or is there a small throughput penalty from the extra scale operands?
- **Scale loading overhead**: In a pipelined kernel, loading the small scale tensors adds some memory traffic and instruction overhead. Need to measure empirically.
- **Interaction with non-volatile MMA**: Our kernels use non-volatile MMA for compiler scheduling freedom. Verify this works with the MXFP8 variant.
- **ptxas register allocation**: The extra operands may affect ptxas register allocation and scheduling. Need to check register count doesn't cross occupancy thresholds.

### 12.3 Risks
- **Triton PR was reverted (PR #8029)**: The revert was due to functional and performance regressions in Triton's internal workloads, NOT because the instruction doesn't work. The instruction itself is valid; the Triton lowering had bugs in scale layout handling. Our hand-written PTX is not affected by Triton's lowering bugs.
- **Scale layout mismatch**: If the scale factors are not in the exact layout the hardware expects, results will be silently wrong. Must validate against known-good reference (e.g., CUTLASS or Triton's test cases).
- **MXFP4 temptation**: sm_120a does NOT support MXFP4 (`.kind::mxf4`). Only sm_120f does. Do not attempt MXFP4.

### 12.4 Verification Strategy
1. Write a standalone test kernel that does a single 16x8x32 MXFP8 MMA
2. Compute reference in FP32: `D = C + sum_k((A[i][k] * 2^(sfa[i][k/32]-127)) * (B[j][k] * 2^(sfb[j][k/32]-127)))`
3. Compare hardware output to reference
4. If correct, integrate into GEMM kernel first (simpler than attention)
5. Benchmark MXFP8 GEMM vs standard FP8 GEMM
6. If performance is acceptable, integrate into attention kernel

---

## 13. Summary: Action Items for Workers

### GEMM Worker
1. Add `-arch=sm_120a` build flag
2. Create `mma_mxfp8_m16n8k32()` wrapper function
3. Accept MXFP8 inputs (data + scales) from host
4. Load scale factors to registers alongside data tiles
5. Replace standard FP8 MMA call with MXFP8 variant
6. Remove post-GEMM per-tensor scale multiply
7. Validate correctness against FP32 reference
8. Benchmark vs standard FP8 and cuBLAS

### Attention Worker
1. Start with pre-quantized MXFP8 Q/K/V inputs (Approach A)
2. Load scale factors alongside data tiles (tiny overhead)
3. Replace FP8 MMA with MXFP8 MMA in QK^T and PV phases
4. Eliminate ALL BF16-to-FP8 conversion code (~448 ALU per KV block)
5. Measure: does removing conversion overhead reduce the 18% scoreboard stalls?
6. Validate correctness with reference attention computation

### Fused MLP Worker
1. Same as GEMM: MXFP8 inputs with hardware scaling
2. Eliminates per-tensor scale handling between fused GEMM stages
