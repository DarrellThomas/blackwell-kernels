// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Flash Attention v2 for sm_120 (RTX 5090)
// Uses mma.sync.aligned.m16n8k16 tensor core instructions for Q*K^T and P*V.
//
// Architecture:
//   Global Memory --> Shared Memory --[ldmatrix]--> Registers --[mma.sync]--> Registers
//                                                                                |
//                                                         Online softmax <-------+
//                                                               |
//                                                         Accumulator (FP32)
//                                                               |
//                                                         --[store]--> Global Memory
//
// Tile config: BLOCK_Q=64/128, BLOCK_KV=32/64, 4 warps (128 threads)
// Each warp handles BLOCK_Q/4 Q rows (one or two m16 MMA tiles).
// Q loaded once to registers, reused across all KV blocks.
// S→P conversion via warp shuffles (register-only, no shared memory).
// Q overlaps KV buffer space during init (freed before KV loop).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#include "mma_sm120.cuh"
#include "ldmatrix.cuh"
#include "cp_async.cuh"
#include "swizzle.cuh"

// ============================================================
// Tile and thread configuration
// ============================================================

constexpr int V2_NUM_WARPS = 4;
constexpr int V2_WARP_SIZE = 32;
constexpr int V2_THREADS = V2_NUM_WARPS * V2_WARP_SIZE;  // 128

// Pack two float values into a uint32_t as bf16x2 (for MMA A-fragment)
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b)
{
    __nv_bfloat162 packed = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&packed);
}

// ============================================================
// v2 kernel — templated on HEAD_DIM, BLOCK_KV, BLOCK_Q
// ============================================================

template <int HEAD_DIM, int BLOCK_KV, int BLOCK_Q>
__global__ void __launch_bounds__(V2_THREADS, (BLOCK_Q <= 64) ? 3 : 2)
flash_attn_v2_kernel(
    const __nv_bfloat16 *__restrict__ Q,   // [B*H, N, D]
    const __nv_bfloat16 *__restrict__ K,   // [B*H, N, D]
    const __nv_bfloat16 *__restrict__ V,   // [B*H, N, D]
    __nv_bfloat16 *__restrict__ O,         // [B*H, N, D]
    float *__restrict__ L,                 // [B*H, N]
    int seq_len,
    float scale,
    bool causal)
{
    // Derived constants
    constexpr int WARP_Q = BLOCK_Q / V2_NUM_WARPS;       // 16 or 32
    constexpr int WARP_Q_TILES = WARP_Q / 16;            // 1 or 2
    constexpr int D_CHUNKS = HEAD_DIM / 16;
    constexpr int O_N_CHUNKS = HEAD_DIM / 8;
    constexpr int S_N_CHUNKS = BLOCK_KV / 8;
    constexpr int P_K_CHUNKS = BLOCK_KV / 16;
    constexpr int KV_SMEM_ELEMS = BLOCK_KV * HEAD_DIM;

    const int bh_idx = blockIdx.y;
    const int q_block = blockIdx.x;
    const int q_start = q_block * BLOCK_Q;

    const int tid = threadIdx.x;
    const int warp_id = tid / V2_WARP_SIZE;
    const int lane_id = tid % V2_WARP_SIZE;

    const __nv_bfloat16 *Q_bh = Q + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *K_bh = K + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *V_bh = V + bh_idx * seq_len * HEAD_DIM;
    __nv_bfloat16 *O_bh = O + bh_idx * seq_len * HEAD_DIM;
    float *L_bh = L + bh_idx * seq_len;

    // ---- Shared memory layout (double-buffered K/V, XOR swizzled) ----
    // Q overlaps K0+K1 during Phase A/B, then freed for KV use.
    // Layout: [K0: KV] [K1: KV] [V0: KV] [V1: KV]
    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_base = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *smem_Q = smem_base;  // Q overlaps K0+(K1) during init
    __nv_bfloat16 *smem_K_base = smem_base;
    __nv_bfloat16 *smem_V_base = smem_base + 2 * KV_SMEM_ELEMS;

    // ================================================================
    // Phase A: Load Q tile global → shared memory via cp.async
    // ================================================================
    {
        constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;  // 16B = 8 bf16
        constexpr int TOTAL_CHUNKS = BLOCK_Q * CHUNKS_PER_ROW;
        for (int i = tid; i < TOTAL_CHUNKS; i += V2_THREADS) {
            int row = i / CHUNKS_PER_ROW;
            int col = (i % CHUNKS_PER_ROW) * 8;
            int global_row = q_start + row;
            bk::cp_async_128_zfill(
                &smem_Q[bk::swizzle_idx<HEAD_DIM>(row, col)],
                &Q_bh[global_row * HEAD_DIM + col],
                global_row < seq_len);
        }
    }
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    // ================================================================
    // Phase B: Load Q shared → registers via ldmatrix_x4
    // ================================================================
    uint32_t Q_rmem[WARP_Q_TILES][D_CHUNKS][4];

    {
        int warp_q_off = warp_id * WARP_Q;
        int sub = lane_id / 8;
        int t_in_sub = lane_id % 8;
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            int tile_off = warp_q_off + t * 16;
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS; dc++) {
                int smem_row = tile_off + (sub / 2) * 8 + t_in_sub;
                int smem_col = dc * 16 + (sub % 2) * 8;
                const void *addr = &smem_Q[bk::swizzle_idx<HEAD_DIM>(smem_row, smem_col)];
                bk::ldmatrix_x4(Q_rmem[t][dc][0], Q_rmem[t][dc][1],
                                Q_rmem[t][dc][2], Q_rmem[t][dc][3], addr);
            }
        }
    }
    // Pre-scale Q by scale*LOG2E — puts S in log2 space so softmax uses exp2
    {
        __nv_bfloat162 scale_vec = __float2bfloat162_rn(scale * 1.4426950408889634f);
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS; dc++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat162 q_val = *reinterpret_cast<__nv_bfloat162*>(&Q_rmem[t][dc][i]);
                    q_val = __hmul2(q_val, scale_vec);
                    Q_rmem[t][dc][i] = *reinterpret_cast<uint32_t*>(&q_val);
                }
            }
        }
    }
    __syncthreads();
    // Q now in registers (pre-scaled). K0+K1 space is free for KV double-buffering.

    // ================================================================
    // Phase C: Initialize O accumulators and softmax state
    // ================================================================
    float O_rmem[WARP_Q_TILES][O_N_CHUNKS][4];
    #pragma unroll
    for (int t = 0; t < WARP_Q_TILES; t++) {
        #pragma unroll
        for (int n = 0; n < O_N_CHUNKS; n++) {
            O_rmem[t][n][0] = 0.0f; O_rmem[t][n][1] = 0.0f;
            O_rmem[t][n][2] = 0.0f; O_rmem[t][n][3] = 0.0f;
        }
    }

    // Per-tile softmax state: 2 rows per tile (row0 = lane/4, row1 = lane/4+8)
    float row_max[2 * WARP_Q_TILES];
    float row_sum[2 * WARP_Q_TILES];
    int global_rows[2 * WARP_Q_TILES];
    #pragma unroll
    for (int t = 0; t < WARP_Q_TILES; t++) {
        row_max[2*t]   = -FLT_MAX;
        row_max[2*t+1] = -FLT_MAX;
        row_sum[2*t]   = 0.0f;
        row_sum[2*t+1] = 0.0f;
        global_rows[2*t]   = q_start + warp_id * WARP_Q + t * 16 + (lane_id / 4);
        global_rows[2*t+1] = global_rows[2*t] + 8;
    }

    // ================================================================
    // Phase D: KV loop
    // ================================================================
    int kv_end = causal ? min(seq_len, q_start + BLOCK_Q) : seq_len;
    int num_kv_blocks = (kv_end + BLOCK_KV - 1) / BLOCK_KV;

    // ================================================================
    // Prologue: Load first K/V tile into double-buffer slot 0
    // ================================================================
    {
        constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
        constexpr int TOTAL_CHUNKS = BLOCK_KV * CHUNKS_PER_ROW;
        for (int i = tid; i < TOTAL_CHUNKS; i += V2_THREADS) {
            int row = i / CHUNKS_PER_ROW;
            int col = (i % CHUNKS_PER_ROW) * 8;
            int gkv = row;
            bk::cp_async_128_zfill(
                &smem_K_base[bk::swizzle_idx<HEAD_DIM>(row, col)],
                &K_bh[gkv * HEAD_DIM + col],
                gkv < seq_len);
            bk::cp_async_128_zfill(
                &smem_V_base[bk::swizzle_idx<HEAD_DIM>(row, col)],
                &V_bh[gkv * HEAD_DIM + col],
                gkv < seq_len);
        }
    }
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_KV;
        int cur = kv_block & 1;
        __nv_bfloat16 *smem_K_cur = smem_K_base + cur * KV_SMEM_ELEMS;
        __nv_bfloat16 *smem_V_cur = smem_V_base + cur * KV_SMEM_ELEMS;

        // ============================================================
        // D.2: Compute S = Q * K^T using MMA
        // ============================================================
        float S_rmem[WARP_Q_TILES][S_N_CHUNKS][4];
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            #pragma unroll
            for (int n = 0; n < S_N_CHUNKS; n++) {
                S_rmem[t][n][0] = 0.0f; S_rmem[t][n][1] = 0.0f;
                S_rmem[t][n][2] = 0.0f; S_rmem[t][n][3] = 0.0f;
            }
        }

        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; dc++) {
            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc += 2) {
                // ldmatrix.x4: load 2 consecutive K B-fragments at once
                int sub = lane_id / 8;
                int t_in_sub = lane_id % 8;
                int k_row = (nc + sub / 2) * 8 + t_in_sub;
                int k_col = dc * 16 + (sub % 2) * 8;
                const void *addr_k = &smem_K_cur[bk::swizzle_idx<HEAD_DIM>(k_row, k_col)];

                uint32_t K_r0, K_r1, K_r2, K_r3;
                bk::ldmatrix_x4(K_r0, K_r1, K_r2, K_r3, addr_k);

                // MMA for each Q tile (K loaded once, reused across tiles)
                #pragma unroll
                for (int t = 0; t < WARP_Q_TILES; t++) {
                    bk::mma_m16n8k16_bf16(
                        S_rmem[t][nc][0], S_rmem[t][nc][1],
                        S_rmem[t][nc][2], S_rmem[t][nc][3],
                        Q_rmem[t][dc][0], Q_rmem[t][dc][2],
                        Q_rmem[t][dc][1], Q_rmem[t][dc][3],
                        K_r0, K_r1,
                        S_rmem[t][nc][0], S_rmem[t][nc][1],
                        S_rmem[t][nc][2], S_rmem[t][nc][3]);

                    bk::mma_m16n8k16_bf16(
                        S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                        S_rmem[t][nc+1][2], S_rmem[t][nc+1][3],
                        Q_rmem[t][dc][0], Q_rmem[t][dc][2],
                        Q_rmem[t][dc][1], Q_rmem[t][dc][3],
                        K_r2, K_r3,
                        S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                        S_rmem[t][nc+1][2], S_rmem[t][nc+1][3]);
                }
            }
        }

        // ============================================================
        // Prefetch next K/V tile (after QK^T, overlaps softmax + P*V)
        // ============================================================
        if (kv_block + 1 < num_kv_blocks) {
            int nxt = 1 - cur;
            int kv_start_nxt = (kv_block + 1) * BLOCK_KV;
            __nv_bfloat16 *smem_K_nxt = smem_K_base + nxt * KV_SMEM_ELEMS;
            __nv_bfloat16 *smem_V_nxt = smem_V_base + nxt * KV_SMEM_ELEMS;
            constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
            constexpr int TOTAL_CHUNKS = BLOCK_KV * CHUNKS_PER_ROW;
            for (int i = tid; i < TOTAL_CHUNKS; i += V2_THREADS) {
                int row = i / CHUNKS_PER_ROW;
                int col = (i % CHUNKS_PER_ROW) * 8;
                int gkv = kv_start_nxt + row;
                bk::cp_async_128_zfill(
                    &smem_K_nxt[bk::swizzle_idx<HEAD_DIM>(row, col)],
                    &K_bh[gkv * HEAD_DIM + col],
                    gkv < seq_len);
                bk::cp_async_128_zfill(
                    &smem_V_nxt[bk::swizzle_idx<HEAD_DIM>(row, col)],
                    &V_bh[gkv * HEAD_DIM + col],
                    gkv < seq_len);
            }
        }
        bk::cp_async_commit();

        // ============================================================
        // D.3: Apply causal mask (per tile)
        // ============================================================
        if ((causal && kv_start + BLOCK_KV > q_start) ||
            kv_start + BLOCK_KV > seq_len) {
            #pragma unroll
            for (int t = 0; t < WARP_Q_TILES; t++) {
                int gr0 = global_rows[2*t];
                int gr1 = global_rows[2*t+1];
                #pragma unroll
                for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                    int col0 = kv_start + nc * 8 + (lane_id % 4) * 2;
                    int col1 = col0 + 1;

                    if (causal) {
                        if (col0 > gr0) S_rmem[t][nc][0] = -FLT_MAX;
                        if (col1 > gr0) S_rmem[t][nc][1] = -FLT_MAX;
                        if (col0 > gr1) S_rmem[t][nc][2] = -FLT_MAX;
                        if (col1 > gr1) S_rmem[t][nc][3] = -FLT_MAX;
                    }
                    if (col0 >= seq_len) { S_rmem[t][nc][0] = -FLT_MAX; S_rmem[t][nc][2] = -FLT_MAX; }
                    if (col1 >= seq_len) { S_rmem[t][nc][1] = -FLT_MAX; S_rmem[t][nc][3] = -FLT_MAX; }
                }
            }
        }

        // ============================================================
        // D.4: Online softmax in registers (per tile)
        // ============================================================
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            float this_max[2] = {-FLT_MAX, -FLT_MAX};
            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                this_max[0] = fmaxf(this_max[0], fmaxf(S_rmem[t][nc][0], S_rmem[t][nc][1]));
                this_max[1] = fmaxf(this_max[1], fmaxf(S_rmem[t][nc][2], S_rmem[t][nc][3]));
            }

            this_max[0] = fmaxf(this_max[0], __shfl_xor_sync(0xffffffff, this_max[0], 1));
            this_max[0] = fmaxf(this_max[0], __shfl_xor_sync(0xffffffff, this_max[0], 2));
            this_max[1] = fmaxf(this_max[1], __shfl_xor_sync(0xffffffff, this_max[1], 1));
            this_max[1] = fmaxf(this_max[1], __shfl_xor_sync(0xffffffff, this_max[1], 2));

            float new_max[2] = {fmaxf(row_max[2*t], this_max[0]),
                                fmaxf(row_max[2*t+1], this_max[1])};

            float rescale[2];
            rescale[0] = exp2f(row_max[2*t] - new_max[0]);
            rescale[1] = exp2f(row_max[2*t+1] - new_max[1]);

            #pragma unroll
            for (int n = 0; n < O_N_CHUNKS; n++) {
                O_rmem[t][n][0] *= rescale[0]; O_rmem[t][n][1] *= rescale[0];
                O_rmem[t][n][2] *= rescale[1]; O_rmem[t][n][3] *= rescale[1];
            }
            row_sum[2*t]   *= rescale[0];
            row_sum[2*t+1] *= rescale[1];

            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                S_rmem[t][nc][0] = exp2f(S_rmem[t][nc][0] - new_max[0]);
                S_rmem[t][nc][1] = exp2f(S_rmem[t][nc][1] - new_max[0]);
                S_rmem[t][nc][2] = exp2f(S_rmem[t][nc][2] - new_max[1]);
                S_rmem[t][nc][3] = exp2f(S_rmem[t][nc][3] - new_max[1]);
            }

            float local_sum[2] = {0.0f, 0.0f};
            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                local_sum[0] += S_rmem[t][nc][0] + S_rmem[t][nc][1];
                local_sum[1] += S_rmem[t][nc][2] + S_rmem[t][nc][3];
            }
            local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 1);
            local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 2);
            local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 1);
            local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 2);

            row_sum[2*t]   += local_sum[0];
            row_sum[2*t+1] += local_sum[1];
            row_max[2*t]   = new_max[0];
            row_max[2*t+1] = new_max[1];
        }

        // ============================================================
        // D.5-6: Register-only P→A conversion + P*V MMA (per tile)
        // ============================================================
        {
            #pragma unroll
            for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                int nc0 = 2 * kc;
                int nc1 = 2 * kc + 1;

                // Pack P for each tile
                uint32_t P_a[WARP_Q_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_Q_TILES; t++) {
                    P_a[t][0] = pack_bf16x2(S_rmem[t][nc0][0], S_rmem[t][nc0][1]);
                    P_a[t][1] = pack_bf16x2(S_rmem[t][nc0][2], S_rmem[t][nc0][3]);
                    P_a[t][2] = pack_bf16x2(S_rmem[t][nc1][0], S_rmem[t][nc1][1]);
                    P_a[t][3] = pack_bf16x2(S_rmem[t][nc1][2], S_rmem[t][nc1][3]);
                }

                // Preload ALL V fragments for this kc before any MMA
                // This separates loads from compute, allowing the compiler
                // to hoist V loads into the softmax gap
                uint32_t V_all[O_N_CHUNKS / 2][4];
                {
                    int sub = lane_id / 8;
                    int t_in_sub = lane_id % 8;
                    #pragma unroll
                    for (int nc = 0; nc < O_N_CHUNKS; nc += 2) {
                        int v_row = kc * 16 + (sub % 2) * 8 + t_in_sub;
                        int v_col = (nc + sub / 2) * 8;
                        const void *addr_v = &smem_V_cur[bk::swizzle_idx<HEAD_DIM>(v_row, v_col)];
                        bk::ldmatrix_x4_trans(V_all[nc/2][0], V_all[nc/2][1],
                                              V_all[nc/2][2], V_all[nc/2][3], addr_v);
                    }
                }

                // MMA with preloaded V
                #pragma unroll
                for (int nc = 0; nc < O_N_CHUNKS; nc += 2) {
                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        bk::mma_m16n8k16_bf16(
                            O_rmem[t][nc][0], O_rmem[t][nc][1],
                            O_rmem[t][nc][2], O_rmem[t][nc][3],
                            P_a[t][0], P_a[t][1], P_a[t][2], P_a[t][3],
                            V_all[nc/2][0], V_all[nc/2][1],
                            O_rmem[t][nc][0], O_rmem[t][nc][1],
                            O_rmem[t][nc][2], O_rmem[t][nc][3]);

                        bk::mma_m16n8k16_bf16(
                            O_rmem[t][nc+1][0], O_rmem[t][nc+1][1],
                            O_rmem[t][nc+1][2], O_rmem[t][nc+1][3],
                            P_a[t][0], P_a[t][1], P_a[t][2], P_a[t][3],
                            V_all[nc/2][2], V_all[nc/2][3],
                            O_rmem[t][nc+1][0], O_rmem[t][nc+1][1],
                            O_rmem[t][nc+1][2], O_rmem[t][nc+1][3]);
                    }
                }
            }
        }

        // Sync: prefetch data visible for next iteration
        bk::cp_async_wait<0>();
        __syncthreads();
    } // end KV loop

    // ================================================================
    // Phase E: Final normalization and output store
    // ================================================================
    #pragma unroll
    for (int t = 0; t < WARP_Q_TILES; t++) {
        float inv_sum[2];
        inv_sum[0] = (row_sum[2*t] > 0.0f) ? 1.0f / row_sum[2*t] : 0.0f;
        inv_sum[1] = (row_sum[2*t+1] > 0.0f) ? 1.0f / row_sum[2*t+1] : 0.0f;

        #pragma unroll
        for (int nc = 0; nc < O_N_CHUNKS; nc++) {
            O_rmem[t][nc][0] *= inv_sum[0]; O_rmem[t][nc][1] *= inv_sum[0];
            O_rmem[t][nc][2] *= inv_sum[1]; O_rmem[t][nc][3] *= inv_sum[1];
        }

        int gr0 = global_rows[2*t];
        int gr1 = global_rows[2*t+1];

        // Store O: pack adjacent bf16 pairs into uint32 for coalesced 4-byte stores
        #pragma unroll
        for (int nc = 0; nc < O_N_CHUNKS; nc++) {
            int col0 = nc * 8 + (lane_id % 4) * 2;

            if (gr0 < seq_len) {
                *reinterpret_cast<uint32_t*>(&O_bh[gr0 * HEAD_DIM + col0]) =
                    pack_bf16x2(O_rmem[t][nc][0], O_rmem[t][nc][1]);
            }
            if (gr1 < seq_len) {
                *reinterpret_cast<uint32_t*>(&O_bh[gr1 * HEAD_DIM + col0]) =
                    pack_bf16x2(O_rmem[t][nc][2], O_rmem[t][nc][3]);
            }
        }

        // Store logsumexp — row_max is in log2 space, convert back: max/LOG2E + log(sum)
        if (lane_id % 4 == 0) {
            if (gr0 < seq_len)
                L_bh[gr0] = row_max[2*t] * 0.6931471805599453f + __logf(row_sum[2*t]);
            if (gr1 < seq_len)
                L_bh[gr1] = row_max[2*t+1] * 0.6931471805599453f + __logf(row_sum[2*t+1]);
        }
    }
}

// ============================================================
// Host launch
// ============================================================

namespace bk {

void flash_attn_v2_fwd(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    float *L,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream)
{
    int bh = batch_size * num_heads;

    // Shared memory: [K0] [K1] [V0] [V1] — Q overlaps K during init
    auto compute_smem = [](int hd, int bkv) {
        int kv_elems = bkv * hd;
        return 4 * kv_elems * (int)sizeof(__nv_bfloat16);
    };

    auto launch = [&](auto kernel_fn, int smem_bytes, int block_q) {
        int num_q_blocks = (seq_len + block_q - 1) / block_q;
        dim3 grid(num_q_blocks, bh);
        dim3 block(V2_THREADS);
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel_fn,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }
        kernel_fn<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    };

    switch (head_dim) {
        case 32:  launch(flash_attn_v2_kernel<32, 64, 64>,   compute_smem(32, 64),  64);  break;
        case 64: {
            // BQ=128 doubles MMA reuse per K/V load but needs enough blocks to fill GPU.
            // With launch_bounds occupancy=2 and 170 SMs, need >=340 blocks.
            int blocks_bq128 = ((seq_len + 127) / 128) * bh;
            if (blocks_bq128 >= 340) {
                launch(flash_attn_v2_kernel<64, 64, 128>, compute_smem(64, 64), 128);
            } else {
                launch(flash_attn_v2_kernel<64, 64, 64>,  compute_smem(64, 64), 64);
            }
            break;
        }
        case 128: launch(flash_attn_v2_kernel<128, 32, 64>,  compute_smem(128, 32), 64);  break;
        default: break;
    }
}

} // namespace bk

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor flash_attn_v2_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
    bool causal)
{
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be BF16");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_cuda() && K.dtype() == torch::kBFloat16 && K.is_contiguous());
    TORCH_CHECK(V.is_cuda() && V.dtype() == torch::kBFloat16 && V.is_contiguous());

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    TORCH_CHECK(D == 32 || D == 64 || D == 128, "head_dim must be 32, 64, or 128");

    Q = Q.reshape({B * H, N, D});
    K = K.reshape({B * H, N, D});
    V = V.reshape({B * H, N, D});

    auto O_out = torch::empty_like(Q);
    auto L_out = torch::empty({B * H, N}, Q.options().dtype(torch::kFloat32));

    bk::flash_attn_v2_fwd(
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(O_out.data_ptr()),
        L_out.data_ptr<float>(),
        B, H, N, D, scale, causal,
        at::cuda::getCurrentCUDAStream());

    return O_out.reshape({B, H, N, D});
}
