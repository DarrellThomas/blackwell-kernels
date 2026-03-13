// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Flash Attention v3 for sm_120 (RTX 5090)
// Hand-scheduled PTX inner loop for softmax + PV MMA.
//
// Key change from v2: the exp2f + sum + pack_P + V_load + PV_MMA section
// is a single hand-scheduled PTX block (~170 instructions) that interleaves:
//   - exp2f[kc+1] with MMA[kc] (cross-kc overlap via dual P register sets)
//   - V loads between MMA pairs (reduces math_pipe_throttle bursts)
//   - Sum shuffles deferred to kc=3 epilogue (removes sync barrier)
//
// QK^T, prefetch, causal mask, and output store remain in C++.

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
#include "ptx_softmax_pv.cuh"

constexpr int V3_NUM_WARPS = 4;
constexpr int V3_WARP_SIZE = 32;
constexpr int V3_THREADS = V3_NUM_WARPS * V3_WARP_SIZE;  // 128

__device__ __forceinline__ uint32_t v3_pack_bf16x2(float a, float b)
{
    __nv_bfloat162 packed = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&packed);
}

// ============================================================
// v3 kernel — fused exp2f+PV with deferred sum shuffle
// ============================================================

template <int HEAD_DIM, int BLOCK_KV, int BLOCK_Q>
__global__ void __launch_bounds__(V3_THREADS, (BLOCK_Q <= 64) ? 3 : 2)
flash_attn_v3_kernel(
    const __nv_bfloat16 *__restrict__ Q,
    const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V,
    __nv_bfloat16 *__restrict__ O_out,
    float *__restrict__ L,
    int seq_len,
    float scale,
    bool causal)
{
    constexpr int WARP_Q = BLOCK_Q / V3_NUM_WARPS;
    constexpr int WARP_Q_TILES = WARP_Q / 16;
    constexpr int D_CHUNKS = HEAD_DIM / 16;
    constexpr int O_N_CHUNKS = HEAD_DIM / 8;
    constexpr int S_N_CHUNKS = BLOCK_KV / 8;
    constexpr int P_K_CHUNKS = BLOCK_KV / 16;
    constexpr int KV_SMEM_ELEMS = BLOCK_KV * HEAD_DIM;

    const int bh_idx = blockIdx.y;
    const int q_block = blockIdx.x;
    const int q_start = q_block * BLOCK_Q;

    const int tid = threadIdx.x;
    const int warp_id = tid / V3_WARP_SIZE;
    const int lane_id = tid % V3_WARP_SIZE;

    const __nv_bfloat16 *Q_bh = Q + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *K_bh = K + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *V_bh = V + bh_idx * seq_len * HEAD_DIM;
    __nv_bfloat16 *O_bh = O_out + bh_idx * seq_len * HEAD_DIM;
    float *L_bh = L + bh_idx * seq_len;

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_base = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *smem_Q = smem_base;
    __nv_bfloat16 *smem_K_base = smem_base;
    __nv_bfloat16 *smem_V_base = smem_base + 2 * KV_SMEM_ELEMS;

    // ================================================================
    // Phase A: Load Q tile global → shared memory via cp.async
    // ================================================================
    {
        constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
        constexpr int TOTAL_CHUNKS = BLOCK_Q * CHUNKS_PER_ROW;
        for (int i = tid; i < TOTAL_CHUNKS; i += V3_THREADS) {
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
    // Phase B: Load Q shared → registers via ldmatrix_x4_mma
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
                bk::ldmatrix_x4_mma(Q_rmem[t][dc][0], Q_rmem[t][dc][1],
                                    Q_rmem[t][dc][2], Q_rmem[t][dc][3], addr);
            }
        }
    }
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

    // Prologue: Load first K/V tile
    {
        constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
        constexpr int TOTAL_CHUNKS = BLOCK_KV * CHUNKS_PER_ROW;
        for (int i = tid; i < TOTAL_CHUNKS; i += V3_THREADS) {
            int row = i / CHUNKS_PER_ROW;
            int col = (i % CHUNKS_PER_ROW) * 8;
            bk::cp_async_128_zfill(
                &smem_K_base[bk::swizzle_idx<HEAD_DIM>(row, col)],
                &K_bh[row * HEAD_DIM + col],
                row < seq_len);
            bk::cp_async_128_zfill(
                &smem_V_base[bk::swizzle_idx<HEAD_DIM>(row, col)],
                &V_bh[row * HEAD_DIM + col],
                row < seq_len);
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
        // D.2: Compute S = Q * K^T using MMA (identical to v2)
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
                int sub = lane_id / 8;
                int t_in_sub = lane_id % 8;
                int k_row = (nc + sub / 2) * 8 + t_in_sub;
                int k_col = dc * 16 + (sub % 2) * 8;
                const void *addr_k = &smem_K_cur[bk::swizzle_idx<HEAD_DIM>(k_row, k_col)];

                uint32_t K_r0, K_r1, K_r2, K_r3;
                bk::ldmatrix_x4(K_r0, K_r1, K_r2, K_r3, addr_k);

                #pragma unroll
                for (int t = 0; t < WARP_Q_TILES; t++) {
                    bk::mma_m16n8k16_bf16_nv(
                        S_rmem[t][nc][0], S_rmem[t][nc][1],
                        S_rmem[t][nc][2], S_rmem[t][nc][3],
                        Q_rmem[t][dc][0], Q_rmem[t][dc][1],
                        Q_rmem[t][dc][2], Q_rmem[t][dc][3],
                        K_r0, K_r1,
                        S_rmem[t][nc][0], S_rmem[t][nc][1],
                        S_rmem[t][nc][2], S_rmem[t][nc][3]);
                    bk::mma_m16n8k16_bf16_nv(
                        S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                        S_rmem[t][nc+1][2], S_rmem[t][nc+1][3],
                        Q_rmem[t][dc][0], Q_rmem[t][dc][1],
                        Q_rmem[t][dc][2], Q_rmem[t][dc][3],
                        K_r2, K_r3,
                        S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                        S_rmem[t][nc+1][2], S_rmem[t][nc+1][3]);
                }
            }
        }

        // ============================================================
        // Prefetch next K/V tile (identical to v2)
        // ============================================================
        if (kv_block + 1 < num_kv_blocks) {
            int nxt = 1 - cur;
            int kv_start_nxt = (kv_block + 1) * BLOCK_KV;
            __nv_bfloat16 *smem_K_nxt = smem_K_base + nxt * KV_SMEM_ELEMS;
            __nv_bfloat16 *smem_V_nxt = smem_V_base + nxt * KV_SMEM_ELEMS;
            constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
            constexpr int TOTAL_CHUNKS = BLOCK_KV * CHUNKS_PER_ROW;
            for (int i = tid; i < TOTAL_CHUNKS; i += V3_THREADS) {
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
        // D.3: Apply causal mask (identical to v2)
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
        // D.4-6: PTX-scheduled softmax + PV MMA
        //
        // Row max + rescale in C++ (compiler handles fine).
        // exp2f + sum + pack + V load + PV MMA in hand-scheduled PTX
        // with cross-kc interleaving (exp2f[kc+1] overlaps MMA[kc]).
        // ============================================================
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            // --- Row max reduction (C++, same as v2) ---
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

            // --- Rescale O accumulators (C++) ---
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
            row_max[2*t]   = new_max[0];
            row_max[2*t+1] = new_max[1];

            // --- Precompute V smem addresses (needed by PTX block) ---
            uint32_t V_addrs[P_K_CHUNKS][O_N_CHUNKS / 2];
            {
                int sub = lane_id / 8;
                int t_in_sub = lane_id % 8;
                #pragma unroll
                for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                    #pragma unroll
                    for (int nc = 0; nc < O_N_CHUNKS; nc += 2) {
                        int v_row = kc * 16 + (sub % 2) * 8 + t_in_sub;
                        int v_col = (nc + sub / 2) * 8;
                        V_addrs[kc][nc/2] = static_cast<uint32_t>(
                            __cvta_generic_to_shared(
                                &smem_V_cur[bk::swizzle_idx<HEAD_DIM>(v_row, v_col)]));
                    }
                }
            }

            // --- PTX or C++ path depending on config ---
            if constexpr (HEAD_DIM == 64 && BLOCK_KV == 64) {
                // 2-kc paired PTX blocks with cross-kc interleaving.
                // Each block: exp2f + sum + pack + V load + MMA for 2 kc.
                // S passed as read-only inputs; exp2f on .reg copies inside PTX.
                // 34 "+f" outputs (32 O + 2 sum), 26 inputs = 60 operands.
                float local_sum[2] = {0.0f, 0.0f};

                // Block A: kc=0+1 (S[0..3], V_addrs[0..1])
                asm volatile(
                "{\n"
                ".reg .f32 s00,s01,s02,s03,s10,s11,s12,s13;\n"
                ".reg .f32 s20,s21,s22,s23,s30,s31,s32,s33;\n"
                ".reg .b32 pa0,pa1,pa2,pa3,pb0,pb1,pb2,pb3,v0,v1,v2,v3;\n"
                // exp2f S[0..1] → .reg copies
                "sub.f32 s00, %34, %50;\n" "ex2.approx.f32 s00, s00;\n"
                "sub.f32 s01, %35, %50;\n" "ex2.approx.f32 s01, s01;\n"
                "sub.f32 s02, %36, %51;\n" "ex2.approx.f32 s02, s02;\n"
                "sub.f32 s03, %37, %51;\n" "ex2.approx.f32 s03, s03;\n"
                "sub.f32 s10, %38, %50;\n" "ex2.approx.f32 s10, s10;\n"
                "sub.f32 s11, %39, %50;\n" "ex2.approx.f32 s11, s11;\n"
                "sub.f32 s12, %40, %51;\n" "ex2.approx.f32 s12, s12;\n"
                "sub.f32 s13, %41, %51;\n" "ex2.approx.f32 s13, s13;\n"
                // sum kc=0
                "add.f32 %32, s00, s01;\n" "add.f32 %33, s02, s03;\n"
                "add.f32 %32, %32, s10;\n" "add.f32 %32, %32, s11;\n"
                "add.f32 %33, %33, s12;\n" "add.f32 %33, %33, s13;\n"
                // pack P_A from S[0..1]
                "cvt.rn.bf16x2.f32 pa0, s01, s00;\n"
                "cvt.rn.bf16x2.f32 pa1, s03, s02;\n"
                "cvt.rn.bf16x2.f32 pa2, s11, s10;\n"
                "cvt.rn.bf16x2.f32 pa3, s13, s12;\n"
                // V[kc=0][nc=0], MMA O[0..1]
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%52];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {pa0,pa1,pa2,pa3}, {v0,v1}, {%0,%1,%2,%3};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%4,%5,%6,%7}, {pa0,pa1,pa2,pa3}, {v2,v3}, {%4,%5,%6,%7};\n"
                // exp2f S[2..3] — overlaps with MMA pipeline
                "sub.f32 s20, %42, %50;\n" "ex2.approx.f32 s20, s20;\n"
                "sub.f32 s21, %43, %50;\n" "ex2.approx.f32 s21, s21;\n"
                "sub.f32 s22, %44, %51;\n" "ex2.approx.f32 s22, s22;\n"
                "sub.f32 s23, %45, %51;\n" "ex2.approx.f32 s23, s23;\n"
                // V[kc=0][nc=1], MMA O[2..3]
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%53];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%8,%9,%10,%11}, {pa0,pa1,pa2,pa3}, {v0,v1}, {%8,%9,%10,%11};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%12,%13,%14,%15}, {pa0,pa1,pa2,pa3}, {v2,v3}, {%12,%13,%14,%15};\n"
                // exp2f S[3]
                "sub.f32 s30, %46, %50;\n" "ex2.approx.f32 s30, s30;\n"
                "sub.f32 s31, %47, %50;\n" "ex2.approx.f32 s31, s31;\n"
                "sub.f32 s32, %48, %51;\n" "ex2.approx.f32 s32, s32;\n"
                "sub.f32 s33, %49, %51;\n" "ex2.approx.f32 s33, s33;\n"
                // sum kc=1
                "add.f32 %32, %32, s20;\n" "add.f32 %32, %32, s21;\n"
                "add.f32 %33, %33, s22;\n" "add.f32 %33, %33, s23;\n"
                "add.f32 %32, %32, s30;\n" "add.f32 %32, %32, s31;\n"
                "add.f32 %33, %33, s32;\n" "add.f32 %33, %33, s33;\n"
                // pack P_B from S[2..3]
                "cvt.rn.bf16x2.f32 pb0, s21, s20;\n"
                "cvt.rn.bf16x2.f32 pb1, s23, s22;\n"
                "cvt.rn.bf16x2.f32 pb2, s31, s30;\n"
                "cvt.rn.bf16x2.f32 pb3, s33, s32;\n"
                // V[kc=0][nc=2..3], MMA O[4..7] — finish kc=0
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%54];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%16,%17,%18,%19}, {pa0,pa1,pa2,pa3}, {v0,v1}, {%16,%17,%18,%19};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%20,%21,%22,%23}, {pa0,pa1,pa2,pa3}, {v2,v3}, {%20,%21,%22,%23};\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%55];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%24,%25,%26,%27}, {pa0,pa1,pa2,pa3}, {v0,v1}, {%24,%25,%26,%27};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%28,%29,%30,%31}, {pa0,pa1,pa2,pa3}, {v2,v3}, {%28,%29,%30,%31};\n"
                // kc=1 MMA with P_B
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%56];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {pb0,pb1,pb2,pb3}, {v0,v1}, {%0,%1,%2,%3};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%4,%5,%6,%7}, {pb0,pb1,pb2,pb3}, {v2,v3}, {%4,%5,%6,%7};\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%57];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%8,%9,%10,%11}, {pb0,pb1,pb2,pb3}, {v0,v1}, {%8,%9,%10,%11};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%12,%13,%14,%15}, {pb0,pb1,pb2,pb3}, {v2,v3}, {%12,%13,%14,%15};\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%58];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%16,%17,%18,%19}, {pb0,pb1,pb2,pb3}, {v0,v1}, {%16,%17,%18,%19};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%20,%21,%22,%23}, {pb0,pb1,pb2,pb3}, {v2,v3}, {%20,%21,%22,%23};\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%59];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%24,%25,%26,%27}, {pb0,pb1,pb2,pb3}, {v0,v1}, {%24,%25,%26,%27};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%28,%29,%30,%31}, {pb0,pb1,pb2,pb3}, {v2,v3}, {%28,%29,%30,%31};\n"
                "}\n"
                // Outputs: O[0..7] (%0-%31), sum0/sum1 (%32-%33)
                : "+f"(O_rmem[t][0][0]), "+f"(O_rmem[t][0][1]), "+f"(O_rmem[t][0][2]), "+f"(O_rmem[t][0][3]),
                  "+f"(O_rmem[t][1][0]), "+f"(O_rmem[t][1][1]), "+f"(O_rmem[t][1][2]), "+f"(O_rmem[t][1][3]),
                  "+f"(O_rmem[t][2][0]), "+f"(O_rmem[t][2][1]), "+f"(O_rmem[t][2][2]), "+f"(O_rmem[t][2][3]),
                  "+f"(O_rmem[t][3][0]), "+f"(O_rmem[t][3][1]), "+f"(O_rmem[t][3][2]), "+f"(O_rmem[t][3][3]),
                  "+f"(O_rmem[t][4][0]), "+f"(O_rmem[t][4][1]), "+f"(O_rmem[t][4][2]), "+f"(O_rmem[t][4][3]),
                  "+f"(O_rmem[t][5][0]), "+f"(O_rmem[t][5][1]), "+f"(O_rmem[t][5][2]), "+f"(O_rmem[t][5][3]),
                  "+f"(O_rmem[t][6][0]), "+f"(O_rmem[t][6][1]), "+f"(O_rmem[t][6][2]), "+f"(O_rmem[t][6][3]),
                  "+f"(O_rmem[t][7][0]), "+f"(O_rmem[t][7][1]), "+f"(O_rmem[t][7][2]), "+f"(O_rmem[t][7][3]),
                  "+f"(local_sum[0]), "+f"(local_sum[1])                                                        // %32-33
                // Inputs: S[0..3] (%34-%49), new_max (%50-%51), V_addrs[0..1] (%52-%59)
                : "f"(S_rmem[t][0][0]), "f"(S_rmem[t][0][1]), "f"(S_rmem[t][0][2]), "f"(S_rmem[t][0][3]),      // %34-37
                  "f"(S_rmem[t][1][0]), "f"(S_rmem[t][1][1]), "f"(S_rmem[t][1][2]), "f"(S_rmem[t][1][3]),      // %38-41
                  "f"(S_rmem[t][2][0]), "f"(S_rmem[t][2][1]), "f"(S_rmem[t][2][2]), "f"(S_rmem[t][2][3]),      // %42-45
                  "f"(S_rmem[t][3][0]), "f"(S_rmem[t][3][1]), "f"(S_rmem[t][3][2]), "f"(S_rmem[t][3][3]),      // %46-49
                  "f"(new_max[0]), "f"(new_max[1]),                                                              // %50-51
                  "r"(V_addrs[0][0]), "r"(V_addrs[0][1]), "r"(V_addrs[0][2]), "r"(V_addrs[0][3]),              // %52-55
                  "r"(V_addrs[1][0]), "r"(V_addrs[1][1]), "r"(V_addrs[1][2]), "r"(V_addrs[1][3])               // %56-59
                );

                // Block B: kc=2+3 (S[4..7], V_addrs[2..3]) — same structure
                asm volatile(
                "{\n"
                ".reg .f32 s40,s41,s42,s43,s50,s51,s52,s53;\n"
                ".reg .f32 s60,s61,s62,s63,s70,s71,s72,s73;\n"
                ".reg .b32 pa0,pa1,pa2,pa3,pb0,pb1,pb2,pb3,v0,v1,v2,v3;\n"
                // exp2f S[4..5]
                "sub.f32 s40, %34, %50;\n" "ex2.approx.f32 s40, s40;\n"
                "sub.f32 s41, %35, %50;\n" "ex2.approx.f32 s41, s41;\n"
                "sub.f32 s42, %36, %51;\n" "ex2.approx.f32 s42, s42;\n"
                "sub.f32 s43, %37, %51;\n" "ex2.approx.f32 s43, s43;\n"
                "sub.f32 s50, %38, %50;\n" "ex2.approx.f32 s50, s50;\n"
                "sub.f32 s51, %39, %50;\n" "ex2.approx.f32 s51, s51;\n"
                "sub.f32 s52, %40, %51;\n" "ex2.approx.f32 s52, s52;\n"
                "sub.f32 s53, %41, %51;\n" "ex2.approx.f32 s53, s53;\n"
                // sum kc=2
                "add.f32 %32, %32, s40;\n" "add.f32 %32, %32, s41;\n"
                "add.f32 %33, %33, s42;\n" "add.f32 %33, %33, s43;\n"
                "add.f32 %32, %32, s50;\n" "add.f32 %32, %32, s51;\n"
                "add.f32 %33, %33, s52;\n" "add.f32 %33, %33, s53;\n"
                // pack P_A from S[4..5]
                "cvt.rn.bf16x2.f32 pa0, s41, s40;\n"
                "cvt.rn.bf16x2.f32 pa1, s43, s42;\n"
                "cvt.rn.bf16x2.f32 pa2, s51, s50;\n"
                "cvt.rn.bf16x2.f32 pa3, s53, s52;\n"
                // V[kc=2][nc=0], MMA O[0..1]
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%52];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {pa0,pa1,pa2,pa3}, {v0,v1}, {%0,%1,%2,%3};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%4,%5,%6,%7}, {pa0,pa1,pa2,pa3}, {v2,v3}, {%4,%5,%6,%7};\n"
                // exp2f S[6..7] — overlaps with MMA
                "sub.f32 s60, %42, %50;\n" "ex2.approx.f32 s60, s60;\n"
                "sub.f32 s61, %43, %50;\n" "ex2.approx.f32 s61, s61;\n"
                "sub.f32 s62, %44, %51;\n" "ex2.approx.f32 s62, s62;\n"
                "sub.f32 s63, %45, %51;\n" "ex2.approx.f32 s63, s63;\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%53];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%8,%9,%10,%11}, {pa0,pa1,pa2,pa3}, {v0,v1}, {%8,%9,%10,%11};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%12,%13,%14,%15}, {pa0,pa1,pa2,pa3}, {v2,v3}, {%12,%13,%14,%15};\n"
                "sub.f32 s70, %46, %50;\n" "ex2.approx.f32 s70, s70;\n"
                "sub.f32 s71, %47, %50;\n" "ex2.approx.f32 s71, s71;\n"
                "sub.f32 s72, %48, %51;\n" "ex2.approx.f32 s72, s72;\n"
                "sub.f32 s73, %49, %51;\n" "ex2.approx.f32 s73, s73;\n"
                // sum kc=3
                "add.f32 %32, %32, s60;\n" "add.f32 %32, %32, s61;\n"
                "add.f32 %33, %33, s62;\n" "add.f32 %33, %33, s63;\n"
                "add.f32 %32, %32, s70;\n" "add.f32 %32, %32, s71;\n"
                "add.f32 %33, %33, s72;\n" "add.f32 %33, %33, s73;\n"
                // pack P_B from S[6..7]
                "cvt.rn.bf16x2.f32 pb0, s61, s60;\n"
                "cvt.rn.bf16x2.f32 pb1, s63, s62;\n"
                "cvt.rn.bf16x2.f32 pb2, s71, s70;\n"
                "cvt.rn.bf16x2.f32 pb3, s73, s72;\n"
                // V[kc=2][nc=2..3], MMA O[4..7]
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%54];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%16,%17,%18,%19}, {pa0,pa1,pa2,pa3}, {v0,v1}, {%16,%17,%18,%19};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%20,%21,%22,%23}, {pa0,pa1,pa2,pa3}, {v2,v3}, {%20,%21,%22,%23};\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%55];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%24,%25,%26,%27}, {pa0,pa1,pa2,pa3}, {v0,v1}, {%24,%25,%26,%27};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%28,%29,%30,%31}, {pa0,pa1,pa2,pa3}, {v2,v3}, {%28,%29,%30,%31};\n"
                // kc=3 MMA with P_B
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%56];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {pb0,pb1,pb2,pb3}, {v0,v1}, {%0,%1,%2,%3};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%4,%5,%6,%7}, {pb0,pb1,pb2,pb3}, {v2,v3}, {%4,%5,%6,%7};\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%57];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%8,%9,%10,%11}, {pb0,pb1,pb2,pb3}, {v0,v1}, {%8,%9,%10,%11};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%12,%13,%14,%15}, {pb0,pb1,pb2,pb3}, {v2,v3}, {%12,%13,%14,%15};\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%58];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%16,%17,%18,%19}, {pb0,pb1,pb2,pb3}, {v0,v1}, {%16,%17,%18,%19};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%20,%21,%22,%23}, {pb0,pb1,pb2,pb3}, {v2,v3}, {%20,%21,%22,%23};\n"
                "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%59];\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%24,%25,%26,%27}, {pb0,pb1,pb2,pb3}, {v0,v1}, {%24,%25,%26,%27};\n"
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%28,%29,%30,%31}, {pb0,pb1,pb2,pb3}, {v2,v3}, {%28,%29,%30,%31};\n"
                "}\n"
                : "+f"(O_rmem[t][0][0]), "+f"(O_rmem[t][0][1]), "+f"(O_rmem[t][0][2]), "+f"(O_rmem[t][0][3]),
                  "+f"(O_rmem[t][1][0]), "+f"(O_rmem[t][1][1]), "+f"(O_rmem[t][1][2]), "+f"(O_rmem[t][1][3]),
                  "+f"(O_rmem[t][2][0]), "+f"(O_rmem[t][2][1]), "+f"(O_rmem[t][2][2]), "+f"(O_rmem[t][2][3]),
                  "+f"(O_rmem[t][3][0]), "+f"(O_rmem[t][3][1]), "+f"(O_rmem[t][3][2]), "+f"(O_rmem[t][3][3]),
                  "+f"(O_rmem[t][4][0]), "+f"(O_rmem[t][4][1]), "+f"(O_rmem[t][4][2]), "+f"(O_rmem[t][4][3]),
                  "+f"(O_rmem[t][5][0]), "+f"(O_rmem[t][5][1]), "+f"(O_rmem[t][5][2]), "+f"(O_rmem[t][5][3]),
                  "+f"(O_rmem[t][6][0]), "+f"(O_rmem[t][6][1]), "+f"(O_rmem[t][6][2]), "+f"(O_rmem[t][6][3]),
                  "+f"(O_rmem[t][7][0]), "+f"(O_rmem[t][7][1]), "+f"(O_rmem[t][7][2]), "+f"(O_rmem[t][7][3]),
                  "+f"(local_sum[0]), "+f"(local_sum[1])
                : "f"(S_rmem[t][4][0]), "f"(S_rmem[t][4][1]), "f"(S_rmem[t][4][2]), "f"(S_rmem[t][4][3]),
                  "f"(S_rmem[t][5][0]), "f"(S_rmem[t][5][1]), "f"(S_rmem[t][5][2]), "f"(S_rmem[t][5][3]),
                  "f"(S_rmem[t][6][0]), "f"(S_rmem[t][6][1]), "f"(S_rmem[t][6][2]), "f"(S_rmem[t][6][3]),
                  "f"(S_rmem[t][7][0]), "f"(S_rmem[t][7][1]), "f"(S_rmem[t][7][2]), "f"(S_rmem[t][7][3]),
                  "f"(new_max[0]), "f"(new_max[1]),
                  "r"(V_addrs[2][0]), "r"(V_addrs[2][1]), "r"(V_addrs[2][2]), "r"(V_addrs[2][3]),
                  "r"(V_addrs[3][0]), "r"(V_addrs[3][1]), "r"(V_addrs[3][2]), "r"(V_addrs[3][3])
                );

                // Deferred sum shuffle (C++)
                local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 1);
                local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 2);
                local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 1);
                local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 2);
                row_sum[2*t] += local_sum[0];
                row_sum[2*t+1] += local_sum[1];
            } else {
                // C++ fallback for other configs (same as v2)
                float local_sum[2] = {0.0f, 0.0f};
                #pragma unroll
                for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                    S_rmem[t][nc][0] = exp2f(S_rmem[t][nc][0] - new_max[0]);
                    S_rmem[t][nc][1] = exp2f(S_rmem[t][nc][1] - new_max[0]);
                    S_rmem[t][nc][2] = exp2f(S_rmem[t][nc][2] - new_max[1]);
                    S_rmem[t][nc][3] = exp2f(S_rmem[t][nc][3] - new_max[1]);
                    local_sum[0] += S_rmem[t][nc][0] + S_rmem[t][nc][1];
                    local_sum[1] += S_rmem[t][nc][2] + S_rmem[t][nc][3];
                }
                local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 1);
                local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 2);
                local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 1);
                local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 2);
                row_sum[2*t] += local_sum[0];
                row_sum[2*t+1] += local_sum[1];
                #pragma unroll
                for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                    int nc0 = 2 * kc, nc1 = 2 * kc + 1;
                    uint32_t P_a[4];
                    P_a[0] = v3_pack_bf16x2(S_rmem[t][nc0][0], S_rmem[t][nc0][1]);
                    P_a[1] = v3_pack_bf16x2(S_rmem[t][nc0][2], S_rmem[t][nc0][3]);
                    P_a[2] = v3_pack_bf16x2(S_rmem[t][nc1][0], S_rmem[t][nc1][1]);
                    P_a[3] = v3_pack_bf16x2(S_rmem[t][nc1][2], S_rmem[t][nc1][3]);
                    int sub = lane_id / 8, t_in_sub = lane_id % 8;
                    uint32_t V_f[O_N_CHUNKS / 2][4];
                    #pragma unroll
                    for (int nc = 0; nc < O_N_CHUNKS; nc += 2) {
                        int v_row = kc * 16 + (sub % 2) * 8 + t_in_sub;
                        int v_col = (nc + sub / 2) * 8;
                        const void *av = &smem_V_cur[bk::swizzle_idx<HEAD_DIM>(v_row, v_col)];
                        bk::ldmatrix_x4_trans(V_f[nc/2][0], V_f[nc/2][1],
                                              V_f[nc/2][2], V_f[nc/2][3], av);
                    }
                    #pragma unroll
                    for (int nc = 0; nc < O_N_CHUNKS; nc += 2) {
                        bk::mma_m16n8k16_bf16_nv(
                            O_rmem[t][nc][0], O_rmem[t][nc][1],
                            O_rmem[t][nc][2], O_rmem[t][nc][3],
                            P_a[0], P_a[1], P_a[2], P_a[3],
                            V_f[nc/2][0], V_f[nc/2][1],
                            O_rmem[t][nc][0], O_rmem[t][nc][1],
                            O_rmem[t][nc][2], O_rmem[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            O_rmem[t][nc+1][0], O_rmem[t][nc+1][1],
                            O_rmem[t][nc+1][2], O_rmem[t][nc+1][3],
                            P_a[0], P_a[1], P_a[2], P_a[3],
                            V_f[nc/2][2], V_f[nc/2][3],
                            O_rmem[t][nc+1][0], O_rmem[t][nc+1][1],
                            O_rmem[t][nc+1][2], O_rmem[t][nc+1][3]);
                    }
                }
            }
        }

        bk::cp_async_wait<0>();
        __syncthreads();
    } // end KV loop

    // ================================================================
    // Phase E: Final normalization and output store (identical to v2)
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

        #pragma unroll
        for (int nc = 0; nc < O_N_CHUNKS; nc++) {
            int col0 = nc * 8 + (lane_id % 4) * 2;
            if (gr0 < seq_len) {
                *reinterpret_cast<uint32_t*>(&O_bh[gr0 * HEAD_DIM + col0]) =
                    v3_pack_bf16x2(O_rmem[t][nc][0], O_rmem[t][nc][1]);
            }
            if (gr1 < seq_len) {
                *reinterpret_cast<uint32_t*>(&O_bh[gr1 * HEAD_DIM + col0]) =
                    v3_pack_bf16x2(O_rmem[t][nc][2], O_rmem[t][nc][3]);
            }
        }

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

void flash_attn_v3_fwd(
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

    auto compute_smem = [](int hd, int bkv) {
        int kv_elems = bkv * hd;
        return 4 * kv_elems * (int)sizeof(__nv_bfloat16);
    };

    auto launch = [&](auto kernel_fn, int smem_bytes, int block_q) {
        int num_q_blocks = (seq_len + block_q - 1) / block_q;
        dim3 grid(num_q_blocks, bh);
        dim3 block(V3_THREADS);
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel_fn,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }
        kernel_fn<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    };

    switch (head_dim) {
        case 32:  launch(flash_attn_v3_kernel<32, 64, 64>,   compute_smem(32, 64),  64);  break;
        case 64: {
            int blocks_bq128 = ((seq_len + 127) / 128) * bh;
            if (blocks_bq128 >= 340) {
                launch(flash_attn_v3_kernel<64, 64, 128>, compute_smem(64, 64), 128);
            } else {
                launch(flash_attn_v3_kernel<64, 64, 64>,  compute_smem(64, 64), 64);
            }
            break;
        }
        case 128: launch(flash_attn_v3_kernel<128, 32, 64>,  compute_smem(128, 32), 64);  break;
        default: break;
    }
}

} // namespace bk

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor flash_attn_v3_forward(
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

    bk::flash_attn_v3_fwd(
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(O_out.data_ptr()),
        L_out.data_ptr<float>(),
        B, H, N, D, scale, causal,
        at::cuda::getCurrentCUDAStream());

    return O_out.reshape({B, H, N, D});
}
