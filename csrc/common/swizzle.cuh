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

// Returns element index into unpadded BF16 shared memory array.
// XOR-swizzles 8-element (16-byte) chunks to eliminate bank conflicts.
// COLS must be a power of 2 and a multiple of 8.
//
// The swizzle operates on chunk indices (col >> 3). For a given row,
// chunks are permuted by XOR with (row & SWIZZLE_MASK). This ensures
// that the same logical column maps to different physical banks across
// rows: bank_group = ((col>>3) ^ (row & MASK)) % 8.
//
// With MASK=7 (COLS >= 64), 8 consecutive rows produce 8 unique bank
// groups — zero conflicts for any column access pattern.
template <int COLS>
__device__ __forceinline__ int swizzle_idx(int row, int col)
{
    constexpr int NUM_CHUNKS = COLS / 8;
    constexpr int SWIZZLE_BITS = (NUM_CHUNKS >= 8) ? 3 :
                                  (NUM_CHUNKS >= 4) ? 2 : 1;
    constexpr int SWIZZLE_MASK = (1 << SWIZZLE_BITS) - 1;
    int swizzled_col = col ^ ((row & SWIZZLE_MASK) << 3);
    return row * COLS + swizzled_col;
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
