# ldmatrix.m16n16 on sm_120: Confirmed Available, With Caveats

**Sources:**
- https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm100.hpp (CUTLASS source: actual PTX inline asm)
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/config.hpp (architecture capability macros)
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm100.hpp (copy traits with layouts)
- https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/ (ldmatrix shape reference)
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (PTX ISA 9.2)
- https://research.colfax-intl.com/adding-fp8-to-flashattention/ (byte_perm + shfl_sync for FP8 fragment conformance)
- https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8 (ThunderKittens FP8: ldmatrix.b16 reinterpret approach)
- https://github.com/nvidia/cutlass/issues/2901 (LdMatrix16x16x8bOp layout issue)
- https://docs.nvidia.com/cutlass/latest/CHANGELOG.html (CUTLASS 4.4.0 changelog)
- https://arxiv.org/html/2507.10789v1 (Blackwell microbenchmarks: QMMA confirmed for FP8 on sm_120)
- https://gau-nernst.github.io/nvrtc-matmul/ (FP8 MMA with FP16 accumulation on sm_120: 692 TFLOPS)

**Relevant to:** fused-mlp worker (FP8 GEMM1 path)
**Worker's current problem:** FP8 GEMM1 bottleneck is `long_scoreboard 40%` from BF16-to-FP8 conversion doubling A loads. Native FP8 inputs would eliminate conversion overhead. The worker needs to load FP8 data efficiently from shared memory into MMA registers for `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.

---

## ANSWER: ldmatrix.m16n16.b8 IS available on sm_120

**Confirmed via CUTLASS source code (copy_sm100.hpp and config.hpp).**

The architecture capability macro `CUTE_ARCH_LDSM_SM100A_ENABLED` -- which gates all ldmatrix.m16n16.b8 operations -- is explicitly enabled for sm_120a:

```
config.hpp defines CUTE_ARCH_LDSM_SM100A_ENABLED when any of:
  CUTLASS_ARCH_MMA_SM100A_ENABLED
  CUTLASS_ARCH_MMA_SM120A_ENABLED
  CUTLASS_ARCH_MMA_SM121A_ENABLED
  (and SM100F, SM110A, SM110F, SM120F, SM121F variants)
```

This means the RTX 5090 (sm_120a) can execute ldmatrix.m16n16.b8 instructions. The `a` suffix is required -- compile with `-arch=sm_120a`.

---

## FINDING 1: Exact PTX Instructions Available on sm_120

From CUTLASS `include/cute/arch/copy_sm100.hpp`, two structs emit ldmatrix.m16n16 inline assembly:

### SM100_U8x8_LDSM_T (ldmatrix.m16n16.x1.trans)
```
PTX: "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];"
Input:  1 x uint128_t (smem address, 128-bit aligned)
Output: 2 x uint32_t (dst0, dst1)
```
Loads one 16x16 matrix of 8-bit elements (transposed) into 2 registers per thread.

### SM100_U8x16_LDSM_T (ldmatrix.m16n16.x2.trans)
```
PTX: "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];"
Input:  1 x uint128_t (smem address, 128-bit aligned)
Output: 4 x uint32_t (dst0, dst1, dst2, dst3)
```
Loads two 16x16 matrices of 8-bit elements (transposed) into 4 registers per thread.

### Important: Type is `.b8`, NOT `.b8x16`

The PTX ISA documentation lists the destination format as `.b8x16` in its specification syntax, but the CUTLASS-emitted PTX uses `.b8`. These appear to be the same instruction. The earlier brief's reference to `.b8x16` as the format specifier was from the PTX ISA spec syntax; CUTLASS uses the shorter `.b8` form.

---

## FINDING 2: Only `.trans` Variant Exists

CUTLASS defines only transposed ldmatrix.m16n16.b8 operations. There is NO non-transposed version:

| Instruction | Exists? | Purpose |
|-------------|---------|---------|
| `ldmatrix.m16n16.x1.trans.b8` | YES | Column-major 8-bit data |
| `ldmatrix.m16n16.x2.trans.b8` | YES | Two column-major 8-bit matrices |
| `ldmatrix.m16n16.x1.b8` (no trans) | NO | Would need row-major data |
| `ldmatrix.m16n16.x2.b8` (no trans) | NO | Would need row-major data |

This limits usability: your FP8 data must be in column-major layout in shared memory to use this instruction. Row-major data (the common case for A operands) cannot use ldmatrix.m16n16.b8 directly.

---

## FINDING 3: Post-Load Byte Interleave is MANDATORY

CUTLASS applies a byte rearrangement after every ldmatrix.m16n16.b8 load. This is not optional -- the raw register output does not match MMA fragment layout.

### For .x1 (SM100_U8x8_LDSM_T):
```cpp
// ldmatrix produces {tmp0, tmp1}
uchar4& tmp0_ = reinterpret_cast<uchar4&>(tmp0);
uchar4& tmp1_ = reinterpret_cast<uchar4&>(tmp1);
// Interleave bytes across registers:
uchar4 dst0_{tmp0_.x, tmp0_.y, tmp1_.x, tmp1_.y};  // bytes 0,1 from each
uchar4 dst1_{tmp0_.z, tmp0_.w, tmp1_.z, tmp1_.w};  // bytes 2,3 from each
```

### For .x2 (SM100_U8x16_LDSM_T):
```cpp
// ldmatrix produces {tmp0, tmp1, tmp2, tmp3}
// Same pattern applied to both pairs:
dst0 = {tmp0.x, tmp0.y, tmp1.x, tmp1.y}
dst1 = {tmp0.z, tmp0.w, tmp1.z, tmp1.w}
dst2 = {tmp2.x, tmp2.y, tmp3.x, tmp3.y}
dst3 = {tmp2.z, tmp2.w, tmp3.z, tmp3.w}
```

CUTLASS comments explain: the byte interleave is done here "so we don't need to add an additional register-to-register copy at the collective layer." The uchar4 assignments compile to PRMT instructions (single cycle each, zero latency impact).

---

## FINDING 4: ldmatrix.m8n8.x4.b16 Reinterpret Remains the Better Path for Row-Major FP8

Given that ldmatrix.m16n16.b8 is transposed-only and requires post-load byte fixup, the existing `ldmatrix.m8n8.x4.b16` reinterpret approach is still recommended for the fused-mlp worker's use case:

| Criterion | ldmatrix.m8n8.x4.b16 (reinterpret) | ldmatrix.m16n16.b8 (native) |
|-----------|-------------------------------------|------------------------------|
| Architecture | sm_75+ (all modern GPUs) | sm_120a+ only (needs `-arch=sm_120a`) |
| Data layout | Row-major OR column-major | Column-major ONLY (.trans only) |
| Transpose variants | Both .trans and non-.trans | .trans ONLY |
| Registers per call | 4 (covers 16x32 FP8 in one call) | 2 per .x1, 4 per .x2 |
| Post-load fixup | Maybe PRMT (needs testing) | Definitely PRMT (confirmed) |
| Production use | ThunderKittens (1500 TFLOPS on H100) | CUTLASS datacenter kernels |
| Compile flag | `-arch=sm_120` (standard) | `-arch=sm_120a` (accelerated) |

**Recommendation for fused-mlp worker:** Use ldmatrix.m8n8.x4.b16 for loading FP8 data. This is the same instruction already used for BF16 loading. Store FP8 values as contiguous bytes in shared memory, load with ldmatrix treating pairs of FP8 as b16 elements, and feed registers directly to `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`.

---

## FINDING 5: PRMT (Byte Permute) for Fragment Layout Conformance

### What PRMT Does
`prmt.b32 d, a, b, selector` selects 4 bytes from the 8-byte concatenation of {a, b} using the selector as a byte index map. It is a single-cycle ALU instruction.

CUDA intrinsic: `__byte_perm(a, b, selector)` maps to PRMT.

### When PRMT is Needed
There are two scenarios where PRMT is needed for FP8:

**Scenario A: After ldmatrix.m16n16.b8 (confirmed)**
The byte interleave shown in Finding 3 is implemented via PRMT-equivalent operations. Cost: 2 PRMT per 2-register load, or 4 PRMT per 4-register load.

**Scenario B: After ldmatrix.m8n8.x4.b16 reinterpret (uncertain)**
If the byte order within each 32-bit register does not match what m16n8k32 MMA expects, PRMT can fix it. Whether this is needed is unknown and must be tested empirically. The structural argument (same register count, same byte granularity) suggests it might work without PRMT.

### Colfax's FP8 Layout Conformance Technique
For converting FP32 accumulator fragments to FP8 A operands (e.g., P -> FP8_A for PV multiplication):

```cuda
// Step 1: Within-thread byte rearrangement
auto upper0 = __byte_perm(upper, lower, 0x7654);  // gather high bytes
auto lower0 = __byte_perm(upper, lower, 0x3210);  // gather low bytes

// Step 2: Cross-thread data exchange
int upper_map[4] = {0, 3, 1, 2};
int lower_map[4] = {1, 2, 0, 3};
upper0 = __shfl_sync(0xFFFFFFFF, upper0, upper_map[threadIdx.x % 4], 4);
lower0 = __shfl_sync(0xFFFFFFFF, lower0, lower_map[threadIdx.x % 4], 4);
```

**Caveat:** This exact pattern is from Colfax's Hopper WGMMA code. For mma.sync on sm_120, the shuffle pattern may differ or may not be needed at all. The principle (byte_perm for intra-thread, shfl_sync for inter-thread) is sound; the exact selector values need derivation from sm_89/sm_120 MMA fragment layouts.

---

## FINDING 6: FP8 MMA Performance on sm_120

From gau-nernst's experiments on RTX 5090:
- FP8 MMA (m16n8k32) with FP16 accumulation: ~692 TFLOPS
- FP8 MMA with FP32 accumulation: ~465 TFLOPS
- BF16 MMA (m16n8k16): ~209.5 TFLOPS peak

FP8 with FP16 accumulation is 1.5x faster than FP8 with FP32 accumulation and 3.3x faster than BF16. However, FP16 accumulation limits precision. For the fused-mlp GEMM1 (where outputs feed into activation), FP32 accumulation is likely required.

The Blackwell microbenchmark paper confirms sm_120 uses "QMMA" (extended MMA) instructions for FP8, which are distinct from datacenter Blackwell's tcgen05.

---

## FINDING 7: Additional ldmatrix Variants for Sub-Byte Types

CUTLASS defines format-converting ldmatrix operations that load narrow data and expand to 8-bit:

```
ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64   (4-bit -> 8-bit, padded)
ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64
ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32   (6-bit -> 8-bit, padded)
ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32
```

These use the .m8n16 shape (not .m16n16) and are for NVFP4 and MXFP6 support. Not directly applicable to FP8 loading but confirm the sm_100+/sm_120+ hardware has flexible bit-width handling in the load path.

---

## SUMMARY FOR THE FUSED-MLP WORKER

1. **ldmatrix.m16n16.b8 IS available on sm_120a** -- confirmed by CUTLASS source code and architecture capability macros. Requires `-arch=sm_120a` compile flag.

2. **But it is NOT the recommended path** for your use case because:
   - Only transposed variant exists (your A matrix data is likely row-major)
   - Requires mandatory post-load byte interleave (PRMT)
   - Requires the `a` suffix compile flag

3. **Use ldmatrix.m8n8.x4.b16 reinterpret instead.** This is the proven production technique (ThunderKittens uses it). Store FP8 bytes in shared memory, load them as b16 pairs. Same instruction you already use for BF16.

4. **Empirical test required:** Load FP8 data with ldmatrix.x4.b16, feed to m16n8k32 MMA, verify correctness. If byte order is wrong, add PRMT fixup (trivial: 4 instructions).

5. **For P-to-FP8 conversion** (accumulator to A operand): Use __byte_perm + __shfl_sync. The exact selector values need derivation from sm_89/sm_120 MMA fragment layout strides, but the technique is validated by Colfax's FP8 FlashAttention.
