// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Flash Attention FP8 for sm_120 (RTX 5090)
// Uses mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 tensor core instructions.
// 2x tensor throughput vs BF16 m16n8k16 (k doubles from 16→32).
//
// Architecture:
//   Global Memory --> Shared Memory (BF16) --[ldmatrix]--> Registers (BF16)
//                                                             |
//                                                   Convert BF16 → FP8 (in registers)
//                                                             |
//                                                  --[mma.sync e4m3]--> Registers (FP32)
//                                                                            |
//                                                     Online softmax (FP32) <+
//                                                               |
//                                                     Convert P (FP32→FP8)
//                                                               |
//                                                  --[mma.sync e4m3]--> Accumulator (FP32)
//                                                               |
//                                                  --[store]--> Global Memory
//
// Tile config: BLOCK_Q=64, BLOCK_KV=64, 4 warps (128 threads)
// Data stays as BF16 in shared memory; converted to FP8 in registers before MMA.
// Softmax remains entirely in FP32.
// Q overlaps KV buffer space during init (freed before KV loop).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#include "mma_sm120.cuh"
#include "ldmatrix.cuh"
#include "cp_async.cuh"
#include "swizzle.cuh"
#include "fp8_convert.cuh"

// ============================================================
// Tile and thread configuration
// ============================================================

constexpr int FP8_NUM_WARPS = 4;
constexpr int FP8_WARP_SIZE = 32;
constexpr int FP8_THREADS = FP8_NUM_WARPS * FP8_WARP_SIZE;  // 128

// Pack two float values into a uint32_t as bf16x2 (for output store)
__device__ __forceinline__ uint32_t pack_bf16x2_fp8(float a, float b)
{
    __nv_bfloat162 packed = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&packed);
}

// ============================================================
// FP8 kernel — templated on HEAD_DIM, BLOCK_KV, BLOCK_Q
// ============================================================

template <int HEAD_DIM, int BLOCK_KV, int BLOCK_Q>
__global__ void __launch_bounds__(FP8_THREADS, (BLOCK_Q <= 64) ? 3 : 2)
flash_attn_fp8_kernel(
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
    constexpr int WARP_Q = BLOCK_Q / FP8_NUM_WARPS;       // 16 or 32
    constexpr int WARP_Q_TILES = WARP_Q / 16;             // 1 or 2

    // BF16 loading constants (data is BF16 in shared memory)
    constexpr int D_CHUNKS_BF16 = HEAD_DIM / 16;          // 4 for D=64

    // FP8 MMA constants (k=32 doubles reduction dimension)
    constexpr int D_CHUNKS_FP8 = HEAD_DIM / 32;           // 2 for D=64
    constexpr int P_K_CHUNKS_FP8 = BLOCK_KV / 32;         // 2 for BKV=64

    // These stay the same as v2 (softmax and output are FP32)
    constexpr int O_N_CHUNKS = HEAD_DIM / 8;              // 8
    constexpr int S_N_CHUNKS = BLOCK_KV / 8;              // 8
    constexpr int KV_SMEM_ELEMS = BLOCK_KV * HEAD_DIM;

    const int bh_idx = blockIdx.y;
    const int q_block = blockIdx.x;
    const int q_start = q_block * BLOCK_Q;

    const int tid = threadIdx.x;
    const int warp_id = tid / FP8_WARP_SIZE;
    const int lane_id = tid % FP8_WARP_SIZE;

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
        for (int i = tid; i < TOTAL_CHUNKS; i += FP8_THREADS) {
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
    // Phase B: Load Q shared → registers via ldmatrix_x4_mma (BF16)
    // Then pre-scale and convert to FP8.
    // ================================================================

    // First load Q as BF16 (same as v2)
    uint32_t Q_bf16[WARP_Q_TILES][D_CHUNKS_BF16][4];
    {
        int warp_q_off = warp_id * WARP_Q;
        int sub = lane_id / 8;
        int t_in_sub = lane_id % 8;
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            int tile_off = warp_q_off + t * 16;
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS_BF16; dc++) {
                int smem_row = tile_off + (sub / 2) * 8 + t_in_sub;
                int smem_col = dc * 16 + (sub % 2) * 8;
                const void *addr = &smem_Q[bk::swizzle_idx<HEAD_DIM>(smem_row, smem_col)];
                bk::ldmatrix_x4_mma(Q_bf16[t][dc][0], Q_bf16[t][dc][1],
                                    Q_bf16[t][dc][2], Q_bf16[t][dc][3], addr);
            }
        }
    }

    // Pre-scale Q by scale*LOG2E in BF16 (puts S in log2 space for exp2f softmax)
    {
        __nv_bfloat162 scale_vec = __float2bfloat162_rn(scale * 1.4426950408889634f);
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS_BF16; dc++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat162 q_val = *reinterpret_cast<__nv_bfloat162*>(&Q_bf16[t][dc][i]);
                    q_val = __hmul2(q_val, scale_vec);
                    Q_bf16[t][dc][i] = *reinterpret_cast<uint32_t*>(&q_val);
                }
            }
        }
    }

    // Convert pre-scaled BF16 Q to FP8 A-fragments for m16n8k32.
    // Each FP8 dc combines TWO consecutive BF16 dc pairs (k=16+16 → k=32).
    // A-fragment for m16n8k32: 4x uint32_t, each holding 4 FP8 values.
    // BF16 dc0 provides k[0:15], BF16 dc1 provides k[16:31].
    uint32_t Q_fp8[WARP_Q_TILES][D_CHUNKS_FP8][4];
    {
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            #pragma unroll
            for (int dc_fp8 = 0; dc_fp8 < D_CHUNKS_FP8; dc_fp8++) {
                int dc_bf16_0 = dc_fp8 * 2;      // first BF16 k=16 chunk
                int dc_bf16_1 = dc_fp8 * 2 + 1;  // second BF16 k=16 chunk
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    Q_fp8[t][dc_fp8][i] = bk::bf16x2_pair_to_e4m3x4(
                        Q_bf16[t][dc_bf16_0][i], Q_bf16[t][dc_bf16_1][i]);
                }
            }
        }
    }
    __syncthreads();
    // Q now in FP8 registers (pre-scaled). K0+K1 space is free for KV double-buffering.

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
        for (int i = tid; i < TOTAL_CHUNKS; i += FP8_THREADS) {
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
        // D.2: Compute S = Q * K^T using FP8 MMA (m16n8k32)
        // ============================================================
        // S is in FP32, same layout as v2 (S_N_CHUNKS x 4 per tile).
        // Each FP8 dc iteration covers k=32 (two BF16 k=16 chunks).
        // K is loaded as BF16 via ldmatrix_x4, two dc-pairs combined to FP8 B-fragments.

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
        for (int dc_fp8 = 0; dc_fp8 < D_CHUNKS_FP8; dc_fp8++) {
            // Load TWO BF16 K dc-chunks and convert to FP8 B-fragments.
            // For Q*K^T: B-fragment comes from K rows (transposed).
            // ldmatrix_x4 loads 2 consecutive n8k16 B-tiles at once.
            // We load dc_bf16_0 and dc_bf16_1 separately, then combine into FP8.

            int dc_bf16_0 = dc_fp8 * 2;
            int dc_bf16_1 = dc_fp8 * 2 + 1;

            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc += 2) {
                // Load K B-fragments for dc_bf16_0 (k[0:15] of this fp8 chunk)
                int sub = lane_id / 8;
                int t_in_sub = lane_id % 8;
                int k_row = (nc + sub / 2) * 8 + t_in_sub;
                int k_col_0 = dc_bf16_0 * 16 + (sub % 2) * 8;
                const void *addr_k0 = &smem_K_cur[bk::swizzle_idx<HEAD_DIM>(k_row, k_col_0)];

                uint32_t K_bf16_0_r0, K_bf16_0_r1, K_bf16_0_r2, K_bf16_0_r3;
                bk::ldmatrix_x4(K_bf16_0_r0, K_bf16_0_r1, K_bf16_0_r2, K_bf16_0_r3, addr_k0);

                // Load K B-fragments for dc_bf16_1 (k[16:31] of this fp8 chunk)
                int k_col_1 = dc_bf16_1 * 16 + (sub % 2) * 8;
                const void *addr_k1 = &smem_K_cur[bk::swizzle_idx<HEAD_DIM>(k_row, k_col_1)];

                uint32_t K_bf16_1_r0, K_bf16_1_r1, K_bf16_1_r2, K_bf16_1_r3;
                bk::ldmatrix_x4(K_bf16_1_r0, K_bf16_1_r1, K_bf16_1_r2, K_bf16_1_r3, addr_k1);

                // Convert BF16 B-fragment pairs to FP8 B-fragments for m16n8k32.
                // BF16 m16n8k16 B-fragment: 2 uint32_t (b0, b1) each holding 2 BF16 = 4 BF16 total.
                // FP8 m16n8k32 B-fragment: 2 uint32_t (b0, b1) each holding 4 FP8 = 8 FP8 total.
                //
                // ldmatrix_x4 gives (r0,r1,r2,r3) for 2 n8-tiles:
                //   Tile nc+0: b0=r0, b1=r1 (BF16 B-fragment for first n8)
                //   Tile nc+1: b0=r2, b1=r3 (BF16 B-fragment for second n8)
                //
                // For FP8 m16n8k32: combine b0_k0+b0_k1 → fp8_b0, b1_k0+b1_k1 → fp8_b1

                // First n8 tile (nc+0):
                uint32_t K_fp8_nc0_b0 = bk::bf16x2_pair_to_e4m3x4(K_bf16_0_r0, K_bf16_1_r0);
                uint32_t K_fp8_nc0_b1 = bk::bf16x2_pair_to_e4m3x4(K_bf16_0_r1, K_bf16_1_r1);

                // Second n8 tile (nc+1):
                uint32_t K_fp8_nc1_b0 = bk::bf16x2_pair_to_e4m3x4(K_bf16_0_r2, K_bf16_1_r2);
                uint32_t K_fp8_nc1_b1 = bk::bf16x2_pair_to_e4m3x4(K_bf16_0_r3, K_bf16_1_r3);

                // MMA for each Q tile (K loaded once, reused across tiles)
                #pragma unroll
                for (int t = 0; t < WARP_Q_TILES; t++) {
                    bk::mma_m16n8k32_e4m3_nv(
                        S_rmem[t][nc][0], S_rmem[t][nc][1],
                        S_rmem[t][nc][2], S_rmem[t][nc][3],
                        Q_fp8[t][dc_fp8][0], Q_fp8[t][dc_fp8][1],
                        Q_fp8[t][dc_fp8][2], Q_fp8[t][dc_fp8][3],
                        K_fp8_nc0_b0, K_fp8_nc0_b1,
                        S_rmem[t][nc][0], S_rmem[t][nc][1],
                        S_rmem[t][nc][2], S_rmem[t][nc][3]);

                    bk::mma_m16n8k32_e4m3_nv(
                        S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                        S_rmem[t][nc+1][2], S_rmem[t][nc+1][3],
                        Q_fp8[t][dc_fp8][0], Q_fp8[t][dc_fp8][1],
                        Q_fp8[t][dc_fp8][2], Q_fp8[t][dc_fp8][3],
                        K_fp8_nc1_b0, K_fp8_nc1_b1,
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
            for (int i = tid; i < TOTAL_CHUNKS; i += FP8_THREADS) {
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
        // D.3: Apply causal mask (per tile) — identical to v2
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
        // D.4: Online softmax in registers (per tile) — identical to v2
        // Entirely FP32: row_max, exp2f, row_sum
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
        // D.5-6: Register-only P→A conversion (FP8) + P*V MMA (per tile)
        // ============================================================
        // For FP8 m16n8k32: each kc_fp8 iteration covers k=32 of the PV product.
        // That means consuming 4 S nc-pairs (4 x n8 = 32 columns of P).
        // BF16 v2 consumed 2 nc-pairs per kc (2 x n8k16 = 16 columns).
        //
        // P A-fragment packing (FP8):
        //   BF16: P_a[0] = pack_bf16x2(S[nc0][0], S[nc0][1])  — 2 values, k=16
        //   FP8:  P_a[0] = pack_p_fp8(S[nc0][0], S[nc0][1], S[nc2][0], S[nc2][1])
        //         where nc0 covers k[0:7] and nc2 covers k[16:23] of the k=32 chunk.
        //
        // Mapping: For kc_fp8, the 4 nc-pairs are at indices:
        //   nc0 = 4*kc_fp8, nc1 = 4*kc_fp8+1, nc2 = 4*kc_fp8+2, nc3 = 4*kc_fp8+3
        // nc0,nc1 provide k[0:15], nc2,nc3 provide k[16:31].
        {
            #pragma unroll
            for (int kc_fp8 = 0; kc_fp8 < P_K_CHUNKS_FP8; kc_fp8++) {
                int nc0 = 4 * kc_fp8;        // k[0:7]
                int nc1 = 4 * kc_fp8 + 1;    // k[8:15]
                int nc2 = 4 * kc_fp8 + 2;    // k[16:23]
                int nc3 = 4 * kc_fp8 + 3;    // k[24:31]

                // Pack P into FP8 A-fragments for each tile
                uint32_t P_a[WARP_Q_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_Q_TILES; t++) {
                    // m16n8k32 A-fragment: 4 x uint32_t, each 4 FP8 values.
                    // Same a1/a2 swap pattern as BF16 but doubled across k halves:
                    //   a0 = row0, {k_first_half, k_second_half} pair 0
                    //   a1 = row1, {k_first_half, k_second_half} pair 0
                    //   a2 = row0, {k_first_half, k_second_half} pair 1
                    //   a3 = row1, {k_first_half, k_second_half} pair 1
                    P_a[t][0] = bk::pack_p_fp8(S_rmem[t][nc0][0], S_rmem[t][nc0][1],
                                                S_rmem[t][nc2][0], S_rmem[t][nc2][1]);
                    P_a[t][1] = bk::pack_p_fp8(S_rmem[t][nc0][2], S_rmem[t][nc0][3],
                                                S_rmem[t][nc2][2], S_rmem[t][nc2][3]);
                    P_a[t][2] = bk::pack_p_fp8(S_rmem[t][nc1][0], S_rmem[t][nc1][1],
                                                S_rmem[t][nc3][0], S_rmem[t][nc3][1]);
                    P_a[t][3] = bk::pack_p_fp8(S_rmem[t][nc1][2], S_rmem[t][nc1][3],
                                                S_rmem[t][nc3][2], S_rmem[t][nc3][3]);
                }

                // Preload ALL V fragments for this kc_fp8 before any MMA.
                // Load TWO k=16 V chunks via ldmatrix_x4_trans, convert to FP8.
                // kc_fp8 covers k=32: V rows [kc_fp8*32 .. kc_fp8*32+31].
                int k_off_0 = kc_fp8 * 32;
                int k_off_1 = kc_fp8 * 32 + 16;
                uint32_t V_fp8[O_N_CHUNKS / 2][4];  // [nc_pair][b0_nc0, b1_nc0, b0_nc1, b1_nc1]
                {
                    int sub = lane_id / 8;
                    int t_in_sub = lane_id % 8;
                    #pragma unroll
                    for (int nc = 0; nc < O_N_CHUNKS; nc += 2) {
                        // Load V BF16 chunk 0 (k[0:15])
                        int v_row_0 = k_off_0 + (sub % 2) * 8 + t_in_sub;
                        int v_col = (nc + sub / 2) * 8;
                        const void *addr_v0 = &smem_V_cur[bk::swizzle_idx<HEAD_DIM>(v_row_0, v_col)];
                        uint32_t V0_r0, V0_r1, V0_r2, V0_r3;
                        bk::ldmatrix_x4_trans(V0_r0, V0_r1, V0_r2, V0_r3, addr_v0);

                        // Load V BF16 chunk 1 (k[16:31])
                        int v_row_1 = k_off_1 + (sub % 2) * 8 + t_in_sub;
                        const void *addr_v1 = &smem_V_cur[bk::swizzle_idx<HEAD_DIM>(v_row_1, v_col)];
                        uint32_t V1_r0, V1_r1, V1_r2, V1_r3;
                        bk::ldmatrix_x4_trans(V1_r0, V1_r1, V1_r2, V1_r3, addr_v1);

                        // First n8 tile (nc+0): combine k0+k1 halves
                        V_fp8[nc/2][0] = bk::bf16x2_pair_to_e4m3x4(V0_r0, V1_r0);
                        V_fp8[nc/2][1] = bk::bf16x2_pair_to_e4m3x4(V0_r1, V1_r1);
                        // Second n8 tile (nc+1): combine k0+k1 halves
                        V_fp8[nc/2][2] = bk::bf16x2_pair_to_e4m3x4(V0_r2, V1_r2);
                        V_fp8[nc/2][3] = bk::bf16x2_pair_to_e4m3x4(V0_r3, V1_r3);
                    }
                }

                // MMA with preloaded FP8 V
                #pragma unroll
                for (int nc = 0; nc < O_N_CHUNKS; nc += 2) {
                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        // First n8 tile (nc+0)
                        bk::mma_m16n8k32_e4m3_nv(
                            O_rmem[t][nc][0], O_rmem[t][nc][1],
                            O_rmem[t][nc][2], O_rmem[t][nc][3],
                            P_a[t][0], P_a[t][1], P_a[t][2], P_a[t][3],
                            V_fp8[nc/2][0], V_fp8[nc/2][1],
                            O_rmem[t][nc][0], O_rmem[t][nc][1],
                            O_rmem[t][nc][2], O_rmem[t][nc][3]);

                        // Second n8 tile (nc+1)
                        bk::mma_m16n8k32_e4m3_nv(
                            O_rmem[t][nc+1][0], O_rmem[t][nc+1][1],
                            O_rmem[t][nc+1][2], O_rmem[t][nc+1][3],
                            P_a[t][0], P_a[t][1], P_a[t][2], P_a[t][3],
                            V_fp8[nc/2][2], V_fp8[nc/2][3],
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
                    pack_bf16x2_fp8(O_rmem[t][nc][0], O_rmem[t][nc][1]);
            }
            if (gr1 < seq_len) {
                *reinterpret_cast<uint32_t*>(&O_bh[gr1 * HEAD_DIM + col0]) =
                    pack_bf16x2_fp8(O_rmem[t][nc][2], O_rmem[t][nc][3]);
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

void flash_attn_fp8_fwd(
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
    // Same layout as v2: all BF16 in shared memory, converted to FP8 in registers.
    auto compute_smem = [](int hd, int bkv) {
        int kv_elems = bkv * hd;
        return 4 * kv_elems * (int)sizeof(__nv_bfloat16);
    };

    auto launch = [&](auto kernel_fn, int smem_bytes, int block_q) {
        int num_q_blocks = (seq_len + block_q - 1) / block_q;
        dim3 grid(num_q_blocks, bh);
        dim3 block(FP8_THREADS);
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel_fn,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }
        kernel_fn<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    };

    // Initially only D=64 BKV=64 is supported
    switch (head_dim) {
        case 64:
            launch(flash_attn_fp8_kernel<64, 64, 64>, compute_smem(64, 64), 64);
            break;
        default: break;
    }
}

} // namespace bk

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor flash_attn_fp8_forward(
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

    TORCH_CHECK(D == 64, "FP8 attention currently only supports head_dim=64");

    Q = Q.reshape({B * H, N, D});
    K = K.reshape({B * H, N, D});
    V = V.reshape({B * H, N, D});

    auto O_out = torch::empty_like(Q);
    auto L_out = torch::empty({B * H, N}, Q.options().dtype(torch::kFloat32));

    bk::flash_attn_fp8_fwd(
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(O_out.data_ptr()),
        L_out.data_ptr<float>(),
        B, H, N, D, scale, causal,
        at::cuda::getCurrentCUDAStream());

    return O_out.reshape({B, H, N, D});
}
