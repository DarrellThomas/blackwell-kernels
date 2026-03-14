// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

#pragma once

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// FP8 e4m3 conversion helpers for sm_120
//
// sm_120 ISA support (empirically verified 2026-03-13):
//   cvt.rn.satfinite.e4m3x2.f32     — WORKS (2 floats → 2 packed FP8, one instruction)
//   cvt.rn.satfinite.e4m3x2.bf16x2  — NOT SUPPORTED (ptxas error: "not supported on .target sm_120")
//
// HISTORY: The original version of this file used scalar __nv_fp8_e4m3 constructors
// + manual byte packing (~11 instructions per uint32). A header comment incorrectly
// claimed BOTH cvt variants were unsupported on sm_120. Only the bf16x2 variant
// fails; the f32 variant compiles and runs correctly. The PTX ISA docs list it as
// sm_89+, and sm_120 qualifies — but sm_120 is a different microarchitecture with
// selective instruction support, so you can't assume. We wrote a smoke test, and
// it worked. Switching to vectorized CVT took FP8 attention from 0.080ms (SLOWER
// than BF16 v2) to 0.056ms (1.24x FASTER). One wrong comment blocked the entire
// FP8 optimization path. Always test PTX instructions empirically on sm_120.
//
// The f32 variant has REVERSED operand order (same pattern as cvt.bf16x2.f32):
//   cvt.rn.satfinite.e4m3x2.f32 d, hi, lo  →  d = {lo_fp8, hi_fp8}
// All packed CVT instructions on sm_120 exhibit this reversal. The wrappers below
// handle it so callers don't need to think about it.
//
// Fragment packing: m16n8k32 uses 4 FP8 per uint32_t (vs 2 BF16 per uint32_t).
// To build an FP8 A-fragment from two BF16 A-fragments (k=16 each → k=32):
//   BF16 dc0: Q_bf16[dc0][i] holds 2 BF16 values for k[0:15]
//   BF16 dc1: Q_bf16[dc1][i] holds 2 BF16 values for k[16:31]
//   FP8:      Q_fp8[i] = pack(cvt(bf16_dc0_lo), cvt(bf16_dc0_hi),
//                              cvt(bf16_dc1_lo), cvt(bf16_dc1_hi))

namespace bk {

// ============================================================
// Vectorized conversion: 2 floats → 2 packed FP8 e4m3 (one PTX instruction)
// ============================================================

// Convert 2 floats to packed e4m3x2 (uint16_t with 2 FP8 bytes)
// Result: low byte = fp8(a), high byte = fp8(b)
__device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float a, float b)
{
    uint32_t result;
    // CVT has reversed operand order: first source → HIGH byte
    // We pass (b, a) so that a→low, b→high
    asm("{ .reg .b16 p;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 p, %1, %2;\n"
        "  cvt.u32.u16 %0, p;\n"
        "}\n"
        : "=r"(result) : "f"(b), "f"(a));
    return static_cast<uint16_t>(result);
}

// Convert a BF16x2 (uint32_t with 2 BF16) to 2 FP8 e4m3 bytes packed in low 16 bits
// Uses vectorized CVT: unpack bf16x2 → 2 floats → cvt.e4m3x2.f32
__device__ __forceinline__ uint16_t bf16x2_to_e4m3x2(uint32_t bf16x2)
{
    __nv_bfloat162 bf = *reinterpret_cast<__nv_bfloat162*>(&bf16x2);
    float lo = __bfloat162float(bf.x);  // low 16 bits of bf16x2
    float hi = __bfloat162float(bf.y);  // high 16 bits
    return f32x2_to_e4m3x2(lo, hi);
}

// Convert two BF16x2 registers (covering k=16 each) into one FP8x4 register (k=32)
// bf16_k0 holds 2 BF16 for k[0:15], bf16_k1 holds 2 BF16 for k[16:31]
// Returns uint32_t with 4 FP8 values: {k0_lo, k0_hi, k1_lo, k1_hi}
__device__ __forceinline__ uint32_t bf16x2_pair_to_e4m3x4(uint32_t bf16_k0, uint32_t bf16_k1)
{
    uint16_t fp8_k0 = bf16x2_to_e4m3x2(bf16_k0);
    uint16_t fp8_k1 = bf16x2_to_e4m3x2(bf16_k1);
    return uint32_t(fp8_k0) | (uint32_t(fp8_k1) << 16);
}

// Pack FP32 softmax output into FP8 A-fragment for m16n8k32 PV MMA.
// Takes 4 FP32 values (2 from nc0 covering k[0:7], 2 from nc2 covering k[16:23])
// and produces one uint32_t with 4 packed FP8 values.
__device__ __forceinline__ uint32_t pack_p_fp8(float s0, float s1, float s2, float s3)
{
    uint16_t lo = f32x2_to_e4m3x2(s0, s1);
    uint16_t hi = f32x2_to_e4m3x2(s2, s3);
    return uint32_t(lo) | (uint32_t(hi) << 16);
}

} // namespace bk
