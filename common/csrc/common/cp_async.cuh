// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

#pragma once

#include <cuda_runtime.h>

// Async memory copy helpers for sm_120
//
// cp.async copies data from global memory to shared memory without
// going through registers. This frees up registers and overlaps
// memory transfers with computation.

namespace bk {

// ============================================================
// cp.async: global -> shared memory (bypasses registers)
// ============================================================

// Copy 16 bytes (128 bits) asynchronously
__device__ __forceinline__ void cp_async_128(void *dst_smem, const void *src_gmem)
{
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(dst), "l"(src_gmem));
}

// Copy 8 bytes (64 bits) asynchronously
__device__ __forceinline__ void cp_async_64(void *dst_smem, const void *src_gmem)
{
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n"
        :
        : "r"(dst), "l"(src_gmem));
}

// Copy 16 bytes with predicate (for boundary handling)
__device__ __forceinline__ void cp_async_128_pred(void *dst_smem, const void *src_gmem, bool pred)
{
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "}\n"
        :
        : "r"(dst), "l"(src_gmem), "r"((int)pred));
}

// Copy 16 bytes with predicate; zero-fills destination when pred is false
__device__ __forceinline__ void cp_async_128_zfill(void *dst_smem, const void *src_gmem, bool pred)
{
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  .reg .u32 z;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  mov.u32 z, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {z, z, z, z};\n"
        "}\n"
        :
        : "r"(dst), "l"(src_gmem), "r"((int)pred));
}

// ============================================================
// cp.async barriers
// ============================================================

// Commit all pending cp.async operations into a group
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}

// Wait for N or fewer groups to remain in flight
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

// Wait for all cp.async operations to complete
__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;\n");
}

} // namespace bk
