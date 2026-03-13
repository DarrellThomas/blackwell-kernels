// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

#pragma once

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// FP8 e4m3 conversion helpers for sm_120
//
// sm_120 does NOT support PTX cvt.e4m3x2.bf16x2 or cvt.e4m3x2.f32.
// All conversions use C++ intrinsics (__nv_fp8_e4m3 constructor).
//
// Fragment packing: m16n8k32 uses 4 FP8 per uint32_t (vs 2 BF16 per uint32_t).
// To build an FP8 A-fragment from two BF16 A-fragments (k=16 each → k=32):
//   BF16 dc0: Q_bf16[dc0][i] holds 2 BF16 values for k[0:15]
//   BF16 dc1: Q_bf16[dc1][i] holds 2 BF16 values for k[16:31]
//   FP8:      Q_fp8[i] = pack(cvt(bf16_dc0_lo), cvt(bf16_dc0_hi),
//                              cvt(bf16_dc1_lo), cvt(bf16_dc1_hi))

namespace bk {

// Convert a single float to FP8 e4m3 byte
__device__ __forceinline__ uint8_t f32_to_e4m3_byte(float v)
{
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(v);
    return *reinterpret_cast<uint8_t*>(&fp8);
}

// Pack 4 FP8 bytes into a uint32_t
__device__ __forceinline__ uint32_t pack_e4m3x4(uint8_t b0, uint8_t b1,
                                                 uint8_t b2, uint8_t b3)
{
    return uint32_t(b0) | (uint32_t(b1) << 8) |
           (uint32_t(b2) << 16) | (uint32_t(b3) << 24);
}

// Convert a BF16x2 (uint32_t with 2 BF16) to 2 FP8 e4m3 bytes packed in low 16 bits
__device__ __forceinline__ uint16_t bf16x2_to_e4m3x2(uint32_t bf16x2)
{
    __nv_bfloat162 bf = *reinterpret_cast<__nv_bfloat162*>(&bf16x2);
    uint8_t lo = f32_to_e4m3_byte(__bfloat162float(bf.x));  // low 16 bits of bf16x2
    uint8_t hi = f32_to_e4m3_byte(__bfloat162float(bf.y));  // high 16 bits
    return uint16_t(lo) | (uint16_t(hi) << 8);
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

// Convert two FP32 values to 2 FP8 e4m3 bytes packed in a uint16_t
// (for P→A conversion in PV MMA)
__device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float a, float b)
{
    uint8_t ba = f32_to_e4m3_byte(a);
    uint8_t bb = f32_to_e4m3_byte(b);
    return uint16_t(ba) | (uint16_t(bb) << 8);
}

// Pack FP32 softmax output into FP8 A-fragment for m16n8k32 PV MMA.
// Takes 4 S_rmem nc-pairs (covering k=32) and produces 4 uint32_t A-fragment.
// S[nc][0] = row0_col0, S[nc][1] = row0_col1 (from D-fragment layout)
// S[nc][2] = row1_col0, S[nc][3] = row1_col1
//
// For BF16 m16n8k16: P_a[0] = pack_bf16x2(S[nc0][0], S[nc0][1])
// For FP8 m16n8k32: P_a[0] = pack_e4m3x4(S[nc0][0], S[nc0][1], S[nc2][0], S[nc2][1])
// where nc0 covers k[0:7] and nc2 covers k[16:23]
__device__ __forceinline__ uint32_t pack_p_fp8(float s0, float s1, float s2, float s3)
{
    return pack_e4m3x4(f32_to_e4m3_byte(s0), f32_to_e4m3_byte(s1),
                       f32_to_e4m3_byte(s2), f32_to_e4m3_byte(s3));
}

} // namespace bk
