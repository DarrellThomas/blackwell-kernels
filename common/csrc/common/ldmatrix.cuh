// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

#pragma once

#include <cuda_runtime.h>

// ldmatrix: shared memory -> register load helpers for sm_120
//
// ldmatrix loads data from shared memory directly into the register
// layout expected by mma.sync, avoiding manual data movement.
// Each thread loads one 8x8 matrix fragment.

namespace bk {

// ============================================================
// ldmatrix: load matrix fragments from shared memory to registers
// ============================================================

// Load 1 matrix (m8n8, 1x uint32_t per thread)
__device__ __forceinline__ uint32_t ldmatrix_x1(const void *smem_ptr)
{
    uint32_t reg;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
        : "=r"(reg)
        : "r"(addr));
    return reg;
}

// Load 2 matrices (m8n8 x2, 2x uint32_t per thread)
__device__ __forceinline__ void ldmatrix_x2(uint32_t &r0, uint32_t &r1, const void *smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

// Load 4 matrices (m8n8 x4, 4x uint32_t per thread)
// This is the most common variant for BF16 mma.sync (loads A fragment)
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
    const void *smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// Load 4 matrices transposed (for B matrix in col-major layout)
__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
    const void *smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// Load 2 matrices transposed
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1, const void *smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

// Load 4 matrices with a1/a2 swap baked into output operand order.
// ldmatrix outputs: r0=m0k0, r1=m0k1, r2=m1k0, r3=m1k1
// MMA expects:      a0=m0k0, a1=m1k0, a2=m0k1, a3=m1k1
// By swapping operands {%0, %2, %1, %3} in the instruction, the outputs
// land directly in MMA order — no MOV instructions needed for the swap.
__device__ __forceinline__ void ldmatrix_x4_mma(
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3,
    const void *smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %2, %1, %3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(addr));
}

} // namespace bk
