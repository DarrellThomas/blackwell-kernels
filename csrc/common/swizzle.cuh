// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

#pragma once

#include <cuda_runtime.h>

// Shared memory swizzling for bank conflict resolution on sm_120
//
// Shared memory has 32 banks, each 4 bytes wide. When multiple threads
// in a warp access the same bank, accesses are serialized (bank conflict).
//
// XOR swizzling remaps addresses so that consecutive threads access
// different banks, even for strided access patterns like column reads.
//
// The 128 KB shared memory per SM on RTX 5090 gives us room for
// double-buffered tiles with padding.

namespace bk {

// ============================================================
// XOR swizzle: remap shared memory address to avoid bank conflicts
// ============================================================

// Swizzle for 128-bit (16-byte) access pattern
// row, col are in units of elements (e.g., BF16 values)
// Returns byte offset into shared memory
template <int COLS>
__device__ __forceinline__ int swizzle_offset(int row, int col)
{
    // Each bank is 4 bytes. 32 banks = 128 bytes per bank cycle.
    // For BF16 (2 bytes/element), 8 elements per bank cycle row.
    // XOR the row with (col / 8) to permute bank assignment.
    constexpr int ELEMS_PER_BANK = 8; // 16 bytes / 2 bytes per BF16
    int swizzled_col = col ^ ((row % ELEMS_PER_BANK) * (COLS / ELEMS_PER_BANK));
    // Clamp to valid range
    swizzled_col = swizzled_col % COLS;
    return (row * COLS + swizzled_col) * sizeof(__nv_bfloat16);
}

// Simple padding-based conflict avoidance
// Add PAD elements to each row to shift bank alignment
// For BF16 with 16-byte loads: PAD=8 (16 bytes) is usually enough
template <int COLS, int PAD = 8>
__device__ __forceinline__ int padded_offset(int row, int col)
{
    return (row * (COLS + PAD) + col) * sizeof(__nv_bfloat16);
}

// Byte offset for padded shared memory layout
template <int COLS, int PAD = 8>
__device__ __forceinline__ int padded_smem_idx(int row, int col)
{
    return row * (COLS + PAD) + col;
}

// ============================================================
// Shared memory size calculations
// ============================================================

// Size of a padded tile in bytes
template <int ROWS, int COLS, int PAD = 8>
constexpr int padded_tile_bytes()
{
    return ROWS * (COLS + PAD) * sizeof(__nv_bfloat16);
}

// Size of a padded tile in elements
template <int ROWS, int COLS, int PAD = 8>
constexpr int padded_tile_elems()
{
    return ROWS * (COLS + PAD);
}

} // namespace bk
