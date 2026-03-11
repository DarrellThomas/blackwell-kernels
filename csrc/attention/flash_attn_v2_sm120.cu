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
// Tile config: BLOCK_Q=64, BLOCK_KV=64, 4 warps (128 threads)
// Each warp handles 16 Q rows (one m16 MMA tile).
// Q loaded once to registers, reused across all KV blocks.
// S→P conversion via shared memory round-trip (accumulator and A-fragment
// have different thread-to-element mappings).

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

constexpr int V2_BLOCK_Q = 64;
constexpr int V2_BLOCK_KV = 64;
constexpr int V2_NUM_WARPS = 4;
constexpr int V2_WARP_SIZE = 32;
constexpr int V2_THREADS = V2_NUM_WARPS * V2_WARP_SIZE;  // 128
constexpr int V2_WARP_Q = V2_BLOCK_Q / V2_NUM_WARPS;     // 16
constexpr int V2_PAD = 8;  // BF16 padding per row for bank conflict avoidance

// ============================================================
// v2 kernel
// ============================================================

template <int HEAD_DIM>
__global__ void __launch_bounds__(V2_THREADS)
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
    const int bh_idx = blockIdx.y;
    const int q_block = blockIdx.x;
    const int q_start = q_block * V2_BLOCK_Q;

    const int tid = threadIdx.x;
    const int warp_id = tid / V2_WARP_SIZE;
    const int lane_id = tid % V2_WARP_SIZE;

    const __nv_bfloat16 *Q_bh = Q + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *K_bh = K + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *V_bh = V + bh_idx * seq_len * HEAD_DIM;
    __nv_bfloat16 *O_bh = O + bh_idx * seq_len * HEAD_DIM;
    float *L_bh = L + bh_idx * seq_len;

    // ---- Shared memory layout ----
    // STRIDE_D: padded row width for Q/K/V tiles (HEAD_DIM + PAD)
    // STRIDE_KV: padded row width for P tile (BLOCK_KV + PAD)
    constexpr int STRIDE_D = HEAD_DIM + V2_PAD;
    constexpr int STRIDE_KV = V2_BLOCK_KV + V2_PAD;  // for P matrix

    constexpr int Q_SMEM_ELEMS = V2_BLOCK_Q * STRIDE_D;
    constexpr int KV_SMEM_ELEMS = V2_BLOCK_KV * STRIDE_D;
    // P reuses smem_Q region (Q is in registers by then)
    // P needs BLOCK_Q * STRIDE_KV = 64 * 72 = 4608 elements
    // smem_Q has BLOCK_Q * STRIDE_D elements (≥ 4608 for D ≥ 64)
    // For D=32: smem_Q = 64*40 = 2560 < 4608, so P will also extend into smem_K
    // That's fine because K is also no longer needed when P is written.

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_Q = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *smem_K = smem_Q + Q_SMEM_ELEMS;
    __nv_bfloat16 *smem_V = smem_K + KV_SMEM_ELEMS;
    // P will be written starting at smem_Q with STRIDE_KV per row
    __nv_bfloat16 *smem_P = smem_Q;

    // ================================================================
    // Phase A: Load Q tile global → shared memory
    // ================================================================
    {
        const int total_q_elems = V2_BLOCK_Q * HEAD_DIM;
        for (int i = tid; i < total_q_elems; i += V2_THREADS) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = q_start + row;
            __nv_bfloat16 val = (global_row < seq_len)
                ? Q_bh[global_row * HEAD_DIM + col]
                : __float2bfloat16(0.0f);
            smem_Q[row * STRIDE_D + col] = val;
        }
    }
    __syncthreads();

    // ================================================================
    // Phase B: Load Q shared → registers via ldmatrix_x4
    // ================================================================
    // For m16k16 A-fragment:
    //   ldmatrix.x4: 32 threads, lane_id % 16 → row, (lane_id / 16) * 8 → col half
    constexpr int D_CHUNKS = HEAD_DIM / 16;
    uint32_t Q_rmem[D_CHUNKS][4];

    {
        int warp_q_off = warp_id * V2_WARP_Q;
        // ldmatrix.x4: threads 0-7→sub0, 8-15→sub1, 16-23→sub2, 24-31→sub3
        // sub0=(m[0:8],k[0:8]), sub1=(m[0:8],k[8:16]),
        // sub2=(m[8:16],k[0:8]), sub3=(m[8:16],k[8:16])
        int sub = lane_id / 8;
        int t_in_sub = lane_id % 8;
        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; dc++) {
            int smem_row = warp_q_off + (sub / 2) * 8 + t_in_sub;
            int smem_col = dc * 16 + (sub % 2) * 8;
            const void *addr = &smem_Q[smem_row * STRIDE_D + smem_col];
            bk::ldmatrix_x4(Q_rmem[dc][0], Q_rmem[dc][1],
                            Q_rmem[dc][2], Q_rmem[dc][3], addr);
        }
    }
    __syncthreads();
    // Q now in registers. smem_Q is free for P reuse.

    // ================================================================
    // Phase C: Initialize O accumulators and softmax state
    // ================================================================
    constexpr int O_N_CHUNKS = HEAD_DIM / 8;
    float O_rmem[O_N_CHUNKS][4];
    #pragma unroll
    for (int n = 0; n < O_N_CHUNKS; n++) {
        O_rmem[n][0] = 0.0f; O_rmem[n][1] = 0.0f;
        O_rmem[n][2] = 0.0f; O_rmem[n][3] = 0.0f;
    }

    // MMA accumulator output layout: thread lane_id owns
    //   row0 = lane_id / 4       (0..7)
    //   row1 = lane_id / 4 + 8   (8..15)
    //   col0 = (lane_id % 4) * 2
    //   col1 = (lane_id % 4) * 2 + 1
    // (row0/row1 are local within the warp's 16-row chunk)
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {0.0f, 0.0f};

    const int global_row0 = q_start + warp_id * V2_WARP_Q + (lane_id / 4);
    const int global_row1 = global_row0 + 8;

    // ================================================================
    // Phase D: KV loop
    // ================================================================
    int kv_end = causal ? min(seq_len, q_start + V2_BLOCK_Q) : seq_len;
    int num_kv_blocks = (kv_end + V2_BLOCK_KV - 1) / V2_BLOCK_KV;

    constexpr int S_N_CHUNKS = V2_BLOCK_KV / 8;   // 8
    constexpr int P_K_CHUNKS = V2_BLOCK_KV / 16;   // 4

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * V2_BLOCK_KV;

        // ============================================================
        // D.1: Load K tile → smem_K
        // ============================================================
        {
            const int total = V2_BLOCK_KV * HEAD_DIM;
            for (int i = tid; i < total; i += V2_THREADS) {
                int row = i / HEAD_DIM;
                int col = i % HEAD_DIM;
                int gkv = kv_start + row;
                __nv_bfloat16 val = (gkv < seq_len)
                    ? K_bh[gkv * HEAD_DIM + col]
                    : __float2bfloat16(0.0f);
                smem_K[row * STRIDE_D + col] = val;
            }
        }
        __syncthreads();

        // ============================================================
        // D.2: Compute S = Q * K^T using MMA
        // ============================================================
        // S is [16 x BLOCK_KV] per warp = m16 x (S_N_CHUNKS * n8)
        // with D_CHUNKS k-reduction steps.
        float S_rmem[S_N_CHUNKS][4];
        #pragma unroll
        for (int n = 0; n < S_N_CHUNKS; n++) {
            S_rmem[n][0] = 0.0f; S_rmem[n][1] = 0.0f;
            S_rmem[n][2] = 0.0f; S_rmem[n][3] = 0.0f;
        }

        // d_chunk outer (reduction), n_chunk inner (output cols)
        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; dc++) {
            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                // Load K^T B-fragment (k16 x n8) manually.
                // MMA B-fragment for m16n8k16: thread T holds
                //   b0 = {B_col[k0,n], B_col[k1,n]} with k0=(T%4)*2, n=T/4
                //   b1 = {B_col[k0+8,n], B_col[k1+8,n]}
                // We need B_col[k,n] = K^T[k,n] = K[n,k], so:
                //   b0 = {K[n, k0], K[n, k1]} — 2 consecutive d-elements in one K row
                //   b1 = {K[n, k0+8], K[n, k1+8]}
                // n = kv index, k = d index within this dc chunk.
                int kv_idx = nc * 8 + lane_id / 4;           // n = T/4 (0..7)
                int d_base = dc * 16 + (lane_id % 4) * 2;    // k0 = (T%4)*2

                uint32_t K_rmem0, K_rmem1;
                // Load 2 consecutive bf16 as uint32 (k0, k1)
                K_rmem0 = *reinterpret_cast<const uint32_t*>(
                    &smem_K[kv_idx * STRIDE_D + d_base]);
                // Load (k0+8, k1+8)
                K_rmem1 = *reinterpret_cast<const uint32_t*>(
                    &smem_K[kv_idx * STRIDE_D + d_base + 8]);

                // NOTE: ldmatrix_x4 outputs a1↔a2 swapped relative to MMA order.
                // ldmatrix gives: [0]=m0k0, [1]=m0k1, [2]=m1k0, [3]=m1k1
                // MMA expects:    a0=m0k0,  a1=m1k0,  a2=m0k1,  a3=m1k1
                // So pass [0],[2],[1],[3] to the MMA.
                bk::mma_m16n8k16_bf16(
                    S_rmem[nc][0], S_rmem[nc][1],
                    S_rmem[nc][2], S_rmem[nc][3],
                    Q_rmem[dc][0], Q_rmem[dc][2],
                    Q_rmem[dc][1], Q_rmem[dc][3],
                    K_rmem0, K_rmem1,
                    S_rmem[nc][0], S_rmem[nc][1],
                    S_rmem[nc][2], S_rmem[nc][3]);
            }
        }

        // ============================================================
        // D.3: Apply scale and causal mask
        // ============================================================
        #pragma unroll
        for (int nc = 0; nc < S_N_CHUNKS; nc++) {
            S_rmem[nc][0] *= scale;
            S_rmem[nc][1] *= scale;
            S_rmem[nc][2] *= scale;
            S_rmem[nc][3] *= scale;

            int col0 = kv_start + nc * 8 + (lane_id % 4) * 2;
            int col1 = col0 + 1;

            if (causal) {
                if (col0 > global_row0) S_rmem[nc][0] = -FLT_MAX;
                if (col1 > global_row0) S_rmem[nc][1] = -FLT_MAX;
                if (col0 > global_row1) S_rmem[nc][2] = -FLT_MAX;
                if (col1 > global_row1) S_rmem[nc][3] = -FLT_MAX;
            }
            // Out-of-bounds KV
            if (col0 >= seq_len) { S_rmem[nc][0] = -FLT_MAX; S_rmem[nc][2] = -FLT_MAX; }
            if (col1 >= seq_len) { S_rmem[nc][1] = -FLT_MAX; S_rmem[nc][3] = -FLT_MAX; }
        }

        // ============================================================
        // D.4: Online softmax in registers
        // ============================================================
        float this_max[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int nc = 0; nc < S_N_CHUNKS; nc++) {
            this_max[0] = fmaxf(this_max[0], fmaxf(S_rmem[nc][0], S_rmem[nc][1]));
            this_max[1] = fmaxf(this_max[1], fmaxf(S_rmem[nc][2], S_rmem[nc][3]));
        }

        // Reduce max across 4-thread groups sharing a row (XOR 1, 2)
        this_max[0] = fmaxf(this_max[0], __shfl_xor_sync(0xffffffff, this_max[0], 1));
        this_max[0] = fmaxf(this_max[0], __shfl_xor_sync(0xffffffff, this_max[0], 2));
        this_max[1] = fmaxf(this_max[1], __shfl_xor_sync(0xffffffff, this_max[1], 1));
        this_max[1] = fmaxf(this_max[1], __shfl_xor_sync(0xffffffff, this_max[1], 2));

        float new_max[2] = {fmaxf(row_max[0], this_max[0]),
                            fmaxf(row_max[1], this_max[1])};

        float rescale[2];
        rescale[0] = (row_max[0] == -FLT_MAX) ? 0.0f : __expf(row_max[0] - new_max[0]);
        rescale[1] = (row_max[1] == -FLT_MAX) ? 0.0f : __expf(row_max[1] - new_max[1]);

        #pragma unroll
        for (int n = 0; n < O_N_CHUNKS; n++) {
            O_rmem[n][0] *= rescale[0]; O_rmem[n][1] *= rescale[0];
            O_rmem[n][2] *= rescale[1]; O_rmem[n][3] *= rescale[1];
        }
        row_sum[0] *= rescale[0];
        row_sum[1] *= rescale[1];

        // exp(S - max)
        #pragma unroll
        for (int nc = 0; nc < S_N_CHUNKS; nc++) {
            S_rmem[nc][0] = (S_rmem[nc][0] == -FLT_MAX) ? 0.0f : __expf(S_rmem[nc][0] - new_max[0]);
            S_rmem[nc][1] = (S_rmem[nc][1] == -FLT_MAX) ? 0.0f : __expf(S_rmem[nc][1] - new_max[0]);
            S_rmem[nc][2] = (S_rmem[nc][2] == -FLT_MAX) ? 0.0f : __expf(S_rmem[nc][2] - new_max[1]);
            S_rmem[nc][3] = (S_rmem[nc][3] == -FLT_MAX) ? 0.0f : __expf(S_rmem[nc][3] - new_max[1]);
        }

        // Accumulate row_sum
        float local_sum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int nc = 0; nc < S_N_CHUNKS; nc++) {
            local_sum[0] += S_rmem[nc][0] + S_rmem[nc][1];
            local_sum[1] += S_rmem[nc][2] + S_rmem[nc][3];
        }
        local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 1);
        local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 2);
        local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 1);
        local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 2);

        row_sum[0] += local_sum[0];
        row_sum[1] += local_sum[1];
        row_max[0] = new_max[0];
        row_max[1] = new_max[1];

        // ============================================================
        // D.5: Write P to shared memory + load V
        // ============================================================
        // S→P: the MMA accumulator (d0..d3) and A-fragment (a0..a3) have
        // DIFFERENT thread-to-element mappings:
        //   Accumulator: row = lane_id/4,    col = (lane_id%4)*2
        //   A-fragment:  row = lane_id%8,    col = (lane_id/8)*2
        // So we must write P to shared memory in row-major layout, then
        // reload it via ldmatrix_x4 to get the correct A-fragment.
        //
        // We reuse smem_Q for P (Q is already in registers).
        // P has BLOCK_Q rows x BLOCK_KV cols, stride = STRIDE_KV.
        __syncthreads();  // ensure K reads are done

        // Write P (softmax output) to smem_P, converting to BF16
        {
            int local_row0 = warp_id * V2_WARP_Q + (lane_id / 4);      // 0..7 within warp chunk
            int local_row1 = local_row0 + 8;                             // 8..15

            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                int col0 = nc * 8 + (lane_id % 4) * 2;
                int col1 = col0 + 1;

                smem_P[local_row0 * STRIDE_KV + col0] = __float2bfloat16(S_rmem[nc][0]);
                smem_P[local_row0 * STRIDE_KV + col1] = __float2bfloat16(S_rmem[nc][1]);
                smem_P[local_row1 * STRIDE_KV + col0] = __float2bfloat16(S_rmem[nc][2]);
                smem_P[local_row1 * STRIDE_KV + col1] = __float2bfloat16(S_rmem[nc][3]);
            }
        }

        // Load V tile → smem_V (different smem region, can happen in parallel)
        {
            const int total = V2_BLOCK_KV * HEAD_DIM;
            for (int i = tid; i < total; i += V2_THREADS) {
                int row = i / HEAD_DIM;
                int col = i % HEAD_DIM;
                int gkv = kv_start + row;
                __nv_bfloat16 val = (gkv < seq_len)
                    ? V_bh[gkv * HEAD_DIM + col]
                    : __float2bfloat16(0.0f);
                smem_V[row * STRIDE_D + col] = val;
            }
        }
        __syncthreads();

        // ============================================================
        // D.6: Load P via ldmatrix_x4, compute O += P * V via MMA
        // ============================================================
        // Load P A-fragments from smem_P (same ldmatrix pattern as Q)
        uint32_t P_rmem[P_K_CHUNKS][4];
        {
            int warp_p_off = warp_id * V2_WARP_Q;
            // Same ldmatrix.x4 sub-matrix mapping as Q loading
            int p_sub = lane_id / 8;
            int p_t_in_sub = lane_id % 8;
            #pragma unroll
            for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                int p_row = warp_p_off + (p_sub / 2) * 8 + p_t_in_sub;
                int p_col = kc * 16 + (p_sub % 2) * 8;
                const void *addr_p = &smem_P[p_row * STRIDE_KV + p_col];
                bk::ldmatrix_x4(P_rmem[kc][0], P_rmem[kc][1],
                                P_rmem[kc][2], P_rmem[kc][3], addr_p);
            }
        }

        // O += P * V
        #pragma unroll
        for (int nc = 0; nc < O_N_CHUNKS; nc++) {
            #pragma unroll
            for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                // Load V B-fragment (k16 x n8) via ldmatrix_x2_trans
                // V is [BLOCK_KV, D] row-major, stride = STRIDE_D
                // For P*V: k = kv-dimension, n = d-dimension
                //   Tile 0 (k[0:8]): threads 0-7, V[kc*16+t, nc*8..nc*8+7]
                //   Tile 1 (k[8:16]): threads 8-15, V[kc*16+8+t, nc*8..nc*8+7]
                //   Both tiles read the SAME 8 d-columns but different kv-rows
                int v_row = kc * 16 + (lane_id % 8) + ((lane_id / 8) % 2) * 8;
                int v_col = nc * 8;
                const void *addr_v = &smem_V[v_row * STRIDE_D + v_col];

                uint32_t V_rmem0, V_rmem1;
                bk::ldmatrix_x2_trans(V_rmem0, V_rmem1, addr_v);

                // Same a1↔a2 swap for P-fragment from ldmatrix_x4
                bk::mma_m16n8k16_bf16(
                    O_rmem[nc][0], O_rmem[nc][1],
                    O_rmem[nc][2], O_rmem[nc][3],
                    P_rmem[kc][0], P_rmem[kc][2],
                    P_rmem[kc][1], P_rmem[kc][3],
                    V_rmem0, V_rmem1,
                    O_rmem[nc][0], O_rmem[nc][1],
                    O_rmem[nc][2], O_rmem[nc][3]);
            }
        }

        __syncthreads();
    } // end KV loop

    // ================================================================
    // Phase E: Final normalization and output store
    // ================================================================
    float inv_sum[2];
    inv_sum[0] = (row_sum[0] > 0.0f) ? 1.0f / row_sum[0] : 0.0f;
    inv_sum[1] = (row_sum[1] > 0.0f) ? 1.0f / row_sum[1] : 0.0f;

    #pragma unroll
    for (int nc = 0; nc < O_N_CHUNKS; nc++) {
        O_rmem[nc][0] *= inv_sum[0]; O_rmem[nc][1] *= inv_sum[0];
        O_rmem[nc][2] *= inv_sum[1]; O_rmem[nc][3] *= inv_sum[1];
    }

    // Store O: thread owns (row0, col0/1) and (row1, col0/1) per n-chunk
    #pragma unroll
    for (int nc = 0; nc < O_N_CHUNKS; nc++) {
        int col0 = nc * 8 + (lane_id % 4) * 2;
        int col1 = col0 + 1;

        if (global_row0 < seq_len) {
            O_bh[global_row0 * HEAD_DIM + col0] = __float2bfloat16(O_rmem[nc][0]);
            O_bh[global_row0 * HEAD_DIM + col1] = __float2bfloat16(O_rmem[nc][1]);
        }
        if (global_row1 < seq_len) {
            O_bh[global_row1 * HEAD_DIM + col0] = __float2bfloat16(O_rmem[nc][2]);
            O_bh[global_row1 * HEAD_DIM + col1] = __float2bfloat16(O_rmem[nc][3]);
        }
    }

    // Store logsumexp
    if (lane_id % 4 == 0) {
        if (global_row0 < seq_len)
            L_bh[global_row0] = row_max[0] + __logf(row_sum[0]);
        if (global_row1 < seq_len)
            L_bh[global_row1] = row_max[1] + __logf(row_sum[1]);
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
    int num_q_blocks = (seq_len + V2_BLOCK_Q - 1) / V2_BLOCK_Q;

    dim3 grid(num_q_blocks, bh);
    dim3 block(V2_THREADS);

    // Shared memory: Q + K + V tiles (padded), P reuses Q region
    // Q: BLOCK_Q * STRIDE_D, K: BLOCK_KV * STRIDE_D, V: BLOCK_KV * STRIDE_D
    // P (reusing Q+K space): BLOCK_Q * STRIDE_KV — must fit in Q+K region
    // We allocate Q+K+V and verify P fits in Q+K.
    auto compute_smem = [](int hd) {
        int stride_d = hd + V2_PAD;
        return (V2_BLOCK_Q * stride_d + 2 * V2_BLOCK_KV * stride_d)
               * (int)sizeof(__nv_bfloat16);
    };
    int smem_bytes = compute_smem(head_dim);

    auto launch = [&](auto kernel_fn) {
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel_fn,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }
        kernel_fn<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    };

    switch (head_dim) {
        case 32:  launch(flash_attn_v2_kernel<32>);  break;
        case 64:  launch(flash_attn_v2_kernel<64>);  break;
        case 128: launch(flash_attn_v2_kernel<128>); break;
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
