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
// Tile config: BLOCK_Q=64, BLOCK_KV=32/64, 4 warps (128 threads)
// Each warp handles 16 Q rows (one m16 MMA tile).
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

constexpr int V2_BLOCK_Q = 64;
constexpr int V2_NUM_WARPS = 4;
constexpr int V2_WARP_SIZE = 32;
constexpr int V2_THREADS = V2_NUM_WARPS * V2_WARP_SIZE;  // 128
constexpr int V2_WARP_Q = V2_BLOCK_Q / V2_NUM_WARPS;     // 16

// Pack two float values into a uint32_t as bf16x2 (for MMA A-fragment)
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b)
{
    __nv_bfloat162 packed = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&packed);
}

// ============================================================
// v2 kernel
// ============================================================

template <int HEAD_DIM, int BLOCK_KV>
__global__ void __launch_bounds__(V2_THREADS, 3)
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

    // ---- Shared memory layout (double-buffered K/V, XOR swizzled) ----
    // P conversion is register-only (shuffles), no shared memory needed.
    // Q overlaps K0+K1 during Phase A/B, then freed for KV use.
    // Layout: [K0: KV] [K1: KV] [V0: KV] [V1: KV]
    constexpr int KV_SMEM_ELEMS = BLOCK_KV * HEAD_DIM;

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
        constexpr int TOTAL_CHUNKS = V2_BLOCK_Q * CHUNKS_PER_ROW;
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
            const void *addr = &smem_Q[bk::swizzle_idx<HEAD_DIM>(smem_row, smem_col)];
            bk::ldmatrix_x4(Q_rmem[dc][0], Q_rmem[dc][1],
                            Q_rmem[dc][2], Q_rmem[dc][3], addr);
        }
    }
    // Pre-scale Q by scale*LOG2E — puts S in log2 space so softmax uses exp2
    {
        __nv_bfloat162 scale_vec = __float2bfloat162_rn(scale * 1.4426950408889634f);
        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; dc++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                __nv_bfloat162 q_val = *reinterpret_cast<__nv_bfloat162*>(&Q_rmem[dc][i]);
                q_val = __hmul2(q_val, scale_vec);
                Q_rmem[dc][i] = *reinterpret_cast<uint32_t*>(&q_val);
            }
        }
    }
    __syncthreads();
    // Q now in registers (pre-scaled). K0+K1 space is free for KV double-buffering.

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
    int num_kv_blocks = (kv_end + BLOCK_KV - 1) / BLOCK_KV;

    constexpr int S_N_CHUNKS = BLOCK_KV / 8;   // 8
    constexpr int P_K_CHUNKS = BLOCK_KV / 16;   // 4

    // ================================================================
    // Prologue: Load first K/V tile into double-buffer slot 0
    // ================================================================
    {
        __nv_bfloat16 *smem_K0 = smem_K_base;
        __nv_bfloat16 *smem_V0 = smem_V_base;
        constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
        constexpr int TOTAL_CHUNKS = BLOCK_KV * CHUNKS_PER_ROW;
        for (int i = tid; i < TOTAL_CHUNKS; i += V2_THREADS) {
            int row = i / CHUNKS_PER_ROW;
            int col = (i % CHUNKS_PER_ROW) * 8;
            int gkv = row;
            bk::cp_async_128_zfill(
                &smem_K0[bk::swizzle_idx<HEAD_DIM>(row, col)],
                &K_bh[gkv * HEAD_DIM + col],
                gkv < seq_len);
            bk::cp_async_128_zfill(
                &smem_V0[bk::swizzle_idx<HEAD_DIM>(row, col)],
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
        float S_rmem[S_N_CHUNKS][4];
        #pragma unroll
        for (int n = 0; n < S_N_CHUNKS; n++) {
            S_rmem[n][0] = 0.0f; S_rmem[n][1] = 0.0f;
            S_rmem[n][2] = 0.0f; S_rmem[n][3] = 0.0f;
        }

        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; dc++) {
            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                // ldmatrix_x2: warp-cooperative load of B fragment for Q*K^T
                // Threads 0-7→matrix0 (cols 0-7), 8-15→matrix1 (cols 8-15),
                // 16-23/24-31 mirror 0-7/8-15.
                int k_row = nc * 8 + (lane_id % 8);
                int k_col = dc * 16 + ((lane_id / 8) % 2) * 8;
                const void *addr_k = &smem_K_cur[bk::swizzle_idx<HEAD_DIM>(k_row, k_col)];

                uint32_t K_rmem0, K_rmem1;
                bk::ldmatrix_x2(K_rmem0, K_rmem1, addr_k);

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
        // D.3: Apply scale and causal mask
        // ============================================================
        // Scale already applied to Q registers (Phase B)
        #pragma unroll
        for (int nc = 0; nc < S_N_CHUNKS; nc++) {
            int col0 = kv_start + nc * 8 + (lane_id % 4) * 2;
            int col1 = col0 + 1;

            if (causal) {
                if (col0 > global_row0) S_rmem[nc][0] = -FLT_MAX;
                if (col1 > global_row0) S_rmem[nc][1] = -FLT_MAX;
                if (col0 > global_row1) S_rmem[nc][2] = -FLT_MAX;
                if (col1 > global_row1) S_rmem[nc][3] = -FLT_MAX;
            }
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

        this_max[0] = fmaxf(this_max[0], __shfl_xor_sync(0xffffffff, this_max[0], 1));
        this_max[0] = fmaxf(this_max[0], __shfl_xor_sync(0xffffffff, this_max[0], 2));
        this_max[1] = fmaxf(this_max[1], __shfl_xor_sync(0xffffffff, this_max[1], 1));
        this_max[1] = fmaxf(this_max[1], __shfl_xor_sync(0xffffffff, this_max[1], 2));

        float new_max[2] = {fmaxf(row_max[0], this_max[0]),
                            fmaxf(row_max[1], this_max[1])};

        float rescale[2];
        rescale[0] = exp2f(row_max[0] - new_max[0]);
        rescale[1] = exp2f(row_max[1] - new_max[1]);

        #pragma unroll
        for (int n = 0; n < O_N_CHUNKS; n++) {
            O_rmem[n][0] *= rescale[0]; O_rmem[n][1] *= rescale[0];
            O_rmem[n][2] *= rescale[1]; O_rmem[n][3] *= rescale[1];
        }
        row_sum[0] *= rescale[0];
        row_sum[1] *= rescale[1];

        #pragma unroll
        for (int nc = 0; nc < S_N_CHUNKS; nc++) {
            S_rmem[nc][0] = exp2f(S_rmem[nc][0] - new_max[0]);
            S_rmem[nc][1] = exp2f(S_rmem[nc][1] - new_max[0]);
            S_rmem[nc][2] = exp2f(S_rmem[nc][2] - new_max[1]);
            S_rmem[nc][3] = exp2f(S_rmem[nc][3] - new_max[1]);
        }

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
        // D.5-6: Register-only P→A conversion + P*V MMA
        // ============================================================
        // D-fragment: d0=P[T/4,(T%4)*2], d1=P[T/4,(T%4)*2+1],
        //             d2=P[T/4+8,(T%4)*2], d3=P[T/4+8,(T%4)*2+1]
        // A-fragment: a0={A[T/4,(T%4)*2..+1]}, a1={A[T/4+8,(T%4)*2..+1]},
        //             a2={A[T/4,(T%4)*2+8..+9]}, a3={A[T/4+8,(T%4)*2+8..+9]}
        // Same thread-to-element mapping! Just pack float pairs → bf16x2.
        // Each k-chunk (16 cols) spans 2 n-chunks (8 cols each).
        {
            #pragma unroll
            for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                int nc0 = 2 * kc;
                int nc1 = 2 * kc + 1;

                uint32_t P_a0 = pack_bf16x2(S_rmem[nc0][0], S_rmem[nc0][1]);
                uint32_t P_a1 = pack_bf16x2(S_rmem[nc0][2], S_rmem[nc0][3]);
                uint32_t P_a2 = pack_bf16x2(S_rmem[nc1][0], S_rmem[nc1][1]);
                uint32_t P_a3 = pack_bf16x2(S_rmem[nc1][2], S_rmem[nc1][3]);

                #pragma unroll
                for (int nc = 0; nc < O_N_CHUNKS; nc++) {
                    int v_row = kc * 16 + (lane_id % 8) + ((lane_id / 8) % 2) * 8;
                    int v_col = nc * 8;
                    const void *addr_v = &smem_V_cur[bk::swizzle_idx<HEAD_DIM>(v_row, v_col)];

                    uint32_t V_rmem0, V_rmem1;
                    bk::ldmatrix_x2_trans(V_rmem0, V_rmem1, addr_v);

                    bk::mma_m16n8k16_bf16(
                        O_rmem[nc][0], O_rmem[nc][1],
                        O_rmem[nc][2], O_rmem[nc][3],
                        P_a0, P_a1, P_a2, P_a3,
                        V_rmem0, V_rmem1,
                        O_rmem[nc][0], O_rmem[nc][1],
                        O_rmem[nc][2], O_rmem[nc][3]);
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
    float inv_sum[2];
    inv_sum[0] = (row_sum[0] > 0.0f) ? 1.0f / row_sum[0] : 0.0f;
    inv_sum[1] = (row_sum[1] > 0.0f) ? 1.0f / row_sum[1] : 0.0f;

    #pragma unroll
    for (int nc = 0; nc < O_N_CHUNKS; nc++) {
        O_rmem[nc][0] *= inv_sum[0]; O_rmem[nc][1] *= inv_sum[0];
        O_rmem[nc][2] *= inv_sum[1]; O_rmem[nc][3] *= inv_sum[1];
    }

    // Store O: pack adjacent bf16 pairs into uint32 for coalesced 4-byte stores
    #pragma unroll
    for (int nc = 0; nc < O_N_CHUNKS; nc++) {
        int col0 = nc * 8 + (lane_id % 4) * 2;

        if (global_row0 < seq_len) {
            *reinterpret_cast<uint32_t*>(&O_bh[global_row0 * HEAD_DIM + col0]) =
                pack_bf16x2(O_rmem[nc][0], O_rmem[nc][1]);
        }
        if (global_row1 < seq_len) {
            *reinterpret_cast<uint32_t*>(&O_bh[global_row1 * HEAD_DIM + col0]) =
                pack_bf16x2(O_rmem[nc][2], O_rmem[nc][3]);
        }
    }

    // Store logsumexp — row_max is in log2 space, convert back: max/LOG2E + log(sum)
    if (lane_id % 4 == 0) {
        if (global_row0 < seq_len)
            L_bh[global_row0] = row_max[0] * 0.6931471805599453f + __logf(row_sum[0]);
        if (global_row1 < seq_len)
            L_bh[global_row1] = row_max[1] * 0.6931471805599453f + __logf(row_sum[1]);
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

    // Shared memory: [K0] [K1] [V0] [V1] — Q overlaps K during init
    auto compute_smem = [](int hd, int bkv) {
        int kv_elems = bkv * hd;
        return 4 * kv_elems * (int)sizeof(__nv_bfloat16);
    };

    auto launch = [&](auto kernel_fn, int smem_bytes) {
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel_fn,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }
        kernel_fn<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    };

    // BLOCK_KV=64 gives best MMA:overhead ratio for D=32/64 (32KB smem, 4 blocks/SM)
    // D=128 uses BLOCK_KV=32 to keep smem at 16KB for higher occupancy
    switch (head_dim) {
        case 32:  launch(flash_attn_v2_kernel<32, 64>,  compute_smem(32, 64));  break;
        case 64:  launch(flash_attn_v2_kernel<64, 64>,  compute_smem(64, 64));  break;
        case 128: launch(flash_attn_v2_kernel<128, 32>, compute_smem(128, 32)); break;
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
