// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Forward declarations for flash attention on sm_120
//
// Architecture (from plan):
//   Global Memory --[cp.async]--> Shared Memory --[ldmatrix]--> Registers --[mma.sync]--> Registers
//                                                                                            |
//                                                                   Online softmax <----------+
//                                                                         |
//                                                                         v
//                                                                   Accumulator
//                                                                         |
//                                                                   --[store]--> Global Memory

namespace bk {

// Tile dimensions for BF16 attention
// Q tile: BLOCK_M x HEAD_DIM
// K tile: BLOCK_KV x HEAD_DIM
// V tile: BLOCK_KV x HEAD_DIM
// Output: BLOCK_M x HEAD_DIM
constexpr int BLOCK_M = 64;       // Rows of Q processed per block
constexpr int BLOCK_KV = 32;      // Rows of K/V processed per iteration (v1: 32, v3: 64)
constexpr int WARP_SIZE = 32;

// Launch the forward attention kernel
void flash_attn_fwd(
    const __nv_bfloat16 *Q,  // [batch, heads, seq_len, head_dim]
    const __nv_bfloat16 *K,  // [batch, heads, seq_len, head_dim]
    const __nv_bfloat16 *V,  // [batch, heads, seq_len, head_dim]
    __nv_bfloat16 *O,        // [batch, heads, seq_len, head_dim]
    float *L,                // [batch, heads, seq_len] logsumexp for backward
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,             // 1/sqrt(head_dim)
    bool causal,
    cudaStream_t stream);

} // namespace bk
