// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// MLA (Multi-Latent Attention) Forward Kernel for sm_120 (RTX 5090)
//
// Split-head QK^T: score = scale * (Q_nope @ K_nope^T + Q_rope @ K_rope^T)
// Asymmetric D_V: O dimension may differ from D_QK = D_NOPE + D_ROPE
//
// Architecture (adapted from flash_attn_v2_sm120.cu):
//   - Two Q register sets: Q_nope_rmem, Q_rope_rmem (loaded once, reused)
//   - Three smem regions per double-buffer slot: K_nope, K_rope, V
//   - QK^T accumulates two sub-products into shared S_rmem accumulators
//   - PV phase identical to v2 but uses D_V dimension
//   - All proven patterns preserved: cp.async double-buffer, XOR swizzle,
//     register-only P conversion, non-volatile MMA, ldmatrix_x4_mma, exp2f

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
// Thread configuration (matches v2)
// ============================================================

constexpr int MLA_NUM_WARPS = 4;
constexpr int MLA_WARP_SIZE = 32;
constexpr int MLA_THREADS = MLA_NUM_WARPS * MLA_WARP_SIZE;  // 128

__device__ __forceinline__ uint32_t mla_pack_bf16x2(float a, float b)
{
    __nv_bfloat162 packed = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&packed);
}

// Safe XOR swizzle for non-power-of-2 column counts.
// bk::swizzle_idx only works when COLS is a power of 2. For COLS like 48
// (6 chunks) or 96 (12 chunks), the XOR can map columns out of bounds.
//
// Fix: SWIZZLE_BITS = largest k where NUM_CHUNKS % (1<<k) == 0.
//   COLS=16 (2 chunks): bits=1   COLS=32 (4 chunks): bits=2
//   COLS=48 (6 chunks): bits=1   COLS=64 (8 chunks): bits=3
//   COLS=96 (12 chunks): bits=2  COLS=128 (16 chunks): bits=3
template <int COLS>
__device__ __forceinline__ int mla_swizzle_idx(int row, int col)
{
    constexpr int NUM_CHUNKS = COLS / 8;
    // Largest k where NUM_CHUNKS % (1<<k) == 0, capped at 3
    constexpr int SWIZZLE_BITS =
        (NUM_CHUNKS % 8 == 0) ? 3 :
        (NUM_CHUNKS % 4 == 0) ? 2 :
        (NUM_CHUNKS % 2 == 0) ? 1 : 0;
    constexpr int SWIZZLE_MASK = (1 << SWIZZLE_BITS) - 1;
    int swizzled_col = col ^ ((row & SWIZZLE_MASK) << 3);
    return row * COLS + swizzled_col;
}

// ============================================================
// MLA forward kernel — templated on D_NOPE, D_ROPE, D_V, BLOCK_KV, BLOCK_Q
// ============================================================

template <int D_NOPE, int D_ROPE, int D_V, int BLOCK_KV, int BLOCK_Q>
__global__ void __launch_bounds__(MLA_THREADS, (BLOCK_Q <= 64) ? 3 : 2)
mla_attn_fwd_kernel(
    const __nv_bfloat16 *__restrict__ Q_nope,  // base ptr
    const __nv_bfloat16 *__restrict__ Q_rope,
    const __nv_bfloat16 *__restrict__ K_nope,
    const __nv_bfloat16 *__restrict__ K_rope,
    const __nv_bfloat16 *__restrict__ V_in,
    __nv_bfloat16 *__restrict__ O_out,
    float *__restrict__ L_out,                 // [BH, T]
    int seq_len,
    int kv_len,
    float scale,
    bool causal,
    // Strides (in elements): separate batch and head strides for BTHD support
    int64_t qn_stride_b, int64_t qn_stride_h, int64_t qn_stride_seq,
    int64_t qr_stride_b, int64_t qr_stride_h, int64_t qr_stride_seq,
    int64_t kn_stride_b, int64_t kn_stride_h, int64_t kn_stride_seq,
    int64_t kr_stride_b, int64_t kr_stride_h, int64_t kr_stride_seq,
    int64_t v_stride_b,  int64_t v_stride_h,  int64_t v_stride_seq,
    int64_t o_stride_b,  int64_t o_stride_h,  int64_t o_stride_seq,
    int H)
{
    // Derived compile-time constants
    constexpr int WARP_Q = BLOCK_Q / MLA_NUM_WARPS;
    constexpr int WARP_Q_TILES = WARP_Q / 16;

    // QK^T sub-product dimensions
    constexpr int D_NOPE_CHUNKS = D_NOPE / 16;
    constexpr int D_ROPE_CHUNKS = D_ROPE / 16;

    // Score tile (BLOCK_KV columns)
    constexpr int S_N_CHUNKS = BLOCK_KV / 8;
    constexpr int P_K_CHUNKS = BLOCK_KV / 16;

    // Output tile (D_V columns)
    constexpr int O_N_CHUNKS = D_V / 8;

    // Smem region sizes (in bf16 elements, using real dimensions)
    // mla_swizzle_idx handles non-power-of-2 COLS safely
    constexpr int KN_SMEM_ELEMS = BLOCK_KV * D_NOPE;
    constexpr int KR_SMEM_ELEMS = BLOCK_KV * D_ROPE;
    constexpr int V_SMEM_ELEMS  = BLOCK_KV * D_V;
    constexpr int SLOT_ELEMS = KN_SMEM_ELEMS + KR_SMEM_ELEMS + V_SMEM_ELEMS;

    // Block/thread indices
    const int bh_idx = blockIdx.y;
    const int q_block = blockIdx.x;
    const int q_start = q_block * BLOCK_Q;
    const int tid = threadIdx.x;
    const int warp_id = tid / MLA_WARP_SIZE;
    const int lane_id = tid % MLA_WARP_SIZE;

    // Pointers for this batch*head — stride-aware for BTHD/BHTD layout
    const int b_idx = bh_idx / H;
    const int h_idx = bh_idx % H;
    const __nv_bfloat16 *Qn_bh = Q_nope + b_idx * qn_stride_b + h_idx * qn_stride_h;
    const __nv_bfloat16 *Qr_bh = Q_rope + b_idx * qr_stride_b + h_idx * qr_stride_h;
    const __nv_bfloat16 *Kn_bh = K_nope + b_idx * kn_stride_b + h_idx * kn_stride_h;
    const __nv_bfloat16 *Kr_bh = K_rope + b_idx * kr_stride_b + h_idx * kr_stride_h;
    const __nv_bfloat16 *V_bh  = V_in   + b_idx * v_stride_b  + h_idx * v_stride_h;
    __nv_bfloat16 *O_bh = O_out + b_idx * o_stride_b + h_idx * o_stride_h;
    float *L_bh = L_out + bh_idx * seq_len;  // L is always [BH, T]

    // ---- Shared memory layout ----
    // Two double-buffer slots: [slot0: Kn|Kr|V] [slot1: Kn|Kr|V]
    // Q_nope and Q_rope loaded sequentially through a temp region that overlaps slot0.
    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_base = reinterpret_cast<__nv_bfloat16 *>(smem_raw);

    // Helper lambdas for buffer addressing
    auto smem_Kn = [&](int buf) -> __nv_bfloat16* {
        return smem_base + buf * SLOT_ELEMS;
    };
    auto smem_Kr = [&](int buf) -> __nv_bfloat16* {
        return smem_base + buf * SLOT_ELEMS + KN_SMEM_ELEMS;
    };
    auto smem_V = [&](int buf) -> __nv_bfloat16* {
        return smem_base + buf * SLOT_ELEMS + KN_SMEM_ELEMS + KR_SMEM_ELEMS;
    };

    // ================================================================
    // Phase A: Load Q_nope + Q_rope to registers (merged, single phase)
    // Q_nope → smem region [0, BLOCK_Q*D_NOPE)
    // Q_rope → smem region [BLOCK_Q*D_NOPE, BLOCK_Q*D_NOPE + BLOCK_Q*D_ROPE)
    // Both overlap slot 0 of KV double-buffer (freed before KV loop)
    // ================================================================
    constexpr int QN_SMEM_ELEMS = BLOCK_Q * D_NOPE;
    __nv_bfloat16 *smem_Qn = smem_base;
    __nv_bfloat16 *smem_Qr = smem_base + QN_SMEM_ELEMS;

    uint32_t Qn_rmem[WARP_Q_TILES][D_NOPE_CHUNKS][4];
    uint32_t Qr_rmem[WARP_Q_TILES][D_ROPE_CHUNKS][4];
    {
        // Load both Q regions simultaneously via cp.async
        constexpr int QN_CPR = D_NOPE / 8;
        constexpr int QN_TOTAL = BLOCK_Q * QN_CPR;
        for (int i = tid; i < QN_TOTAL; i += MLA_THREADS) {
            int row = i / QN_CPR;
            int col = (i % QN_CPR) * 8;
            int global_row = q_start + row;
            bk::cp_async_128_zfill(
                &smem_Qn[mla_swizzle_idx<D_NOPE>(row, col)],
                &Qn_bh[global_row * qn_stride_seq + col],
                global_row < seq_len);
        }
        constexpr int QR_CPR = D_ROPE / 8;
        constexpr int QR_TOTAL = BLOCK_Q * QR_CPR;
        for (int i = tid; i < QR_TOTAL; i += MLA_THREADS) {
            int row = i / QR_CPR;
            int col = (i % QR_CPR) * 8;
            int global_row = q_start + row;
            bk::cp_async_128_zfill(
                &smem_Qr[mla_swizzle_idx<D_ROPE>(row, col)],
                &Qr_bh[global_row * qr_stride_seq + col],
                global_row < seq_len);
        }
        bk::cp_async_commit();
        bk::cp_async_wait<0>();
        __syncthreads();

        // ldmatrix both Q_nope and Q_rope from smem → registers
        int warp_q_off = warp_id * WARP_Q;
        int sub = lane_id / 8;
        int t_in_sub = lane_id % 8;
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            int tile_off = warp_q_off + t * 16;
            #pragma unroll
            for (int dc = 0; dc < D_NOPE_CHUNKS; dc++) {
                int smem_row = tile_off + (sub / 2) * 8 + t_in_sub;
                int smem_col = dc * 16 + (sub % 2) * 8;
                const void *addr = &smem_Qn[mla_swizzle_idx<D_NOPE>(smem_row, smem_col)];
                bk::ldmatrix_x4_mma(Qn_rmem[t][dc][0], Qn_rmem[t][dc][1],
                                    Qn_rmem[t][dc][2], Qn_rmem[t][dc][3], addr);
            }
            #pragma unroll
            for (int dc = 0; dc < D_ROPE_CHUNKS; dc++) {
                int smem_row = tile_off + (sub / 2) * 8 + t_in_sub;
                int smem_col = dc * 16 + (sub % 2) * 8;
                const void *addr = &smem_Qr[mla_swizzle_idx<D_ROPE>(smem_row, smem_col)];
                bk::ldmatrix_x4_mma(Qr_rmem[t][dc][0], Qr_rmem[t][dc][1],
                                    Qr_rmem[t][dc][2], Qr_rmem[t][dc][3], addr);
            }
        }

        // Pre-scale both by scale * LOG2E
        __nv_bfloat162 scale_vec = __float2bfloat162_rn(scale * 1.4426950408889634f);
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            #pragma unroll
            for (int dc = 0; dc < D_NOPE_CHUNKS; dc++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat162 q_val = *reinterpret_cast<__nv_bfloat162*>(&Qn_rmem[t][dc][i]);
                    q_val = __hmul2(q_val, scale_vec);
                    Qn_rmem[t][dc][i] = *reinterpret_cast<uint32_t*>(&q_val);
                }
            }
            #pragma unroll
            for (int dc = 0; dc < D_ROPE_CHUNKS; dc++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat162 q_val = *reinterpret_cast<__nv_bfloat162*>(&Qr_rmem[t][dc][i]);
                    q_val = __hmul2(q_val, scale_vec);
                    Qr_rmem[t][dc][i] = *reinterpret_cast<uint32_t*>(&q_val);
                }
            }
        }
        __syncthreads();
    }
    // Both Q sets now in registers. Smem is free for KV double-buffering.

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
    int kv_end = causal ? min(kv_len, q_start + BLOCK_Q) : kv_len;
    int num_kv_blocks = (kv_end + BLOCK_KV - 1) / BLOCK_KV;

    // ================================================================
    // Prologue: Load first KV tile into buffer slot 0
    // ================================================================
    {
        // Load K_nope
        constexpr int KN_CPR = D_NOPE / 8;
        constexpr int KN_TOTAL = BLOCK_KV * KN_CPR;
        for (int i = tid; i < KN_TOTAL; i += MLA_THREADS) {
            int row = i / KN_CPR;
            int col = (i % KN_CPR) * 8;
            bk::cp_async_128_zfill(
                &smem_Kn(0)[mla_swizzle_idx<D_NOPE>(row, col)],
                &Kn_bh[row * kn_stride_seq + col],
                row < kv_len);
        }
        // Load K_rope
        constexpr int KR_CPR = D_ROPE / 8;
        constexpr int KR_TOTAL = BLOCK_KV * KR_CPR;
        for (int i = tid; i < KR_TOTAL; i += MLA_THREADS) {
            int row = i / KR_CPR;
            int col = (i % KR_CPR) * 8;
            bk::cp_async_128_zfill(
                &smem_Kr(0)[mla_swizzle_idx<D_ROPE>(row, col)],
                &Kr_bh[row * kr_stride_seq + col],
                row < kv_len);
        }
        // Load V
        constexpr int V_CPR = D_V / 8;
        constexpr int V_TOTAL = BLOCK_KV * V_CPR;
        for (int i = tid; i < V_TOTAL; i += MLA_THREADS) {
            int row = i / V_CPR;
            int col = (i % V_CPR) * 8;
            bk::cp_async_128_zfill(
                &smem_V(0)[mla_swizzle_idx<D_V>(row, col)],
                &V_bh[row * v_stride_seq + col],
                row < kv_len);
        }
    }
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    // ================================================================
    // Main KV loop
    // ================================================================
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_KV;
        int cur = kv_block & 1;

        __nv_bfloat16 *cur_Kn = smem_Kn(cur);
        __nv_bfloat16 *cur_Kr = smem_Kr(cur);
        __nv_bfloat16 *cur_V  = smem_V(cur);

        // ============================================================
        // D.2: QK^T — zero S, then two sub-products
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

        // QK^T: separate nope then rope loops (sequential).
        // Separate loops avoid tight RAW dependencies between sub-products
        // writing to the same S_rmem accumulators within a single dc iteration.
        {
            int sub = lane_id / 8;
            int t_in_sub = lane_id % 8;

            // Sub-product 1: Q_nope @ K_nope^T
            #pragma unroll
            for (int dc = 0; dc < D_NOPE_CHUNKS; dc++) {
                #pragma unroll
                for (int nc = 0; nc < S_N_CHUNKS; nc += 2) {
                    int k_row = (nc + sub / 2) * 8 + t_in_sub;
                    int k_col = dc * 16 + (sub % 2) * 8;
                    const void *addr_k = &cur_Kn[mla_swizzle_idx<D_NOPE>(k_row, k_col)];

                    uint32_t K_r0, K_r1, K_r2, K_r3;
                    bk::ldmatrix_x4(K_r0, K_r1, K_r2, K_r3, addr_k);

                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            S_rmem[t][nc][0], S_rmem[t][nc][1],
                            S_rmem[t][nc][2], S_rmem[t][nc][3],
                            Qn_rmem[t][dc][0], Qn_rmem[t][dc][1],
                            Qn_rmem[t][dc][2], Qn_rmem[t][dc][3],
                            K_r0, K_r1,
                            S_rmem[t][nc][0], S_rmem[t][nc][1],
                            S_rmem[t][nc][2], S_rmem[t][nc][3]);

                        bk::mma_m16n8k16_bf16_nv(
                            S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                            S_rmem[t][nc+1][2], S_rmem[t][nc+1][3],
                            Qn_rmem[t][dc][0], Qn_rmem[t][dc][1],
                            Qn_rmem[t][dc][2], Qn_rmem[t][dc][3],
                            K_r2, K_r3,
                            S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                            S_rmem[t][nc+1][2], S_rmem[t][nc+1][3]);
                    }
                }
            }

            // Sub-product 2: Q_rope @ K_rope^T (accumulate into same S_rmem)
            #pragma unroll
            for (int dc = 0; dc < D_ROPE_CHUNKS; dc++) {
                #pragma unroll
                for (int nc = 0; nc < S_N_CHUNKS; nc += 2) {
                    int k_row = (nc + sub / 2) * 8 + t_in_sub;
                    int k_col = dc * 16 + (sub % 2) * 8;
                    const void *addr_k = &cur_Kr[mla_swizzle_idx<D_ROPE>(k_row, k_col)];

                    uint32_t K_r0, K_r1, K_r2, K_r3;
                    bk::ldmatrix_x4(K_r0, K_r1, K_r2, K_r3, addr_k);

                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            S_rmem[t][nc][0], S_rmem[t][nc][1],
                            S_rmem[t][nc][2], S_rmem[t][nc][3],
                            Qr_rmem[t][dc][0], Qr_rmem[t][dc][1],
                            Qr_rmem[t][dc][2], Qr_rmem[t][dc][3],
                            K_r0, K_r1,
                            S_rmem[t][nc][0], S_rmem[t][nc][1],
                            S_rmem[t][nc][2], S_rmem[t][nc][3]);

                        bk::mma_m16n8k16_bf16_nv(
                            S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                            S_rmem[t][nc+1][2], S_rmem[t][nc+1][3],
                            Qr_rmem[t][dc][0], Qr_rmem[t][dc][1],
                            Qr_rmem[t][dc][2], Qr_rmem[t][dc][3],
                            K_r2, K_r3,
                            S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                            S_rmem[t][nc+1][2], S_rmem[t][nc+1][3]);
                    }
                }
            }
        }

        // ============================================================
        // Prefetch next KV tile (after QK^T, overlaps softmax + P*V)
        // ============================================================
        if (kv_block + 1 < num_kv_blocks) {
            int nxt = 1 - cur;
            int kv_start_nxt = (kv_block + 1) * BLOCK_KV;

            __nv_bfloat16 *nxt_Kn = smem_Kn(nxt);
            __nv_bfloat16 *nxt_Kr = smem_Kr(nxt);
            __nv_bfloat16 *nxt_V  = smem_V(nxt);

            constexpr int KN_CPR = D_NOPE / 8;
            constexpr int KN_TOTAL = BLOCK_KV * KN_CPR;
            for (int i = tid; i < KN_TOTAL; i += MLA_THREADS) {
                int row = i / KN_CPR;
                int col = (i % KN_CPR) * 8;
                int gkv = kv_start_nxt + row;
                bk::cp_async_128_zfill(
                    &nxt_Kn[mla_swizzle_idx<D_NOPE>(row, col)],
                    &Kn_bh[gkv * kn_stride_seq + col],
                    gkv < kv_len);
            }
            constexpr int KR_CPR = D_ROPE / 8;
            constexpr int KR_TOTAL = BLOCK_KV * KR_CPR;
            for (int i = tid; i < KR_TOTAL; i += MLA_THREADS) {
                int row = i / KR_CPR;
                int col = (i % KR_CPR) * 8;
                int gkv = kv_start_nxt + row;
                bk::cp_async_128_zfill(
                    &nxt_Kr[mla_swizzle_idx<D_ROPE>(row, col)],
                    &Kr_bh[gkv * kr_stride_seq + col],
                    gkv < kv_len);
            }
            constexpr int V_CPR = D_V / 8;
            constexpr int V_TOTAL = BLOCK_KV * V_CPR;
            for (int i = tid; i < V_TOTAL; i += MLA_THREADS) {
                int row = i / V_CPR;
                int col = (i % V_CPR) * 8;
                int gkv = kv_start_nxt + row;
                bk::cp_async_128_zfill(
                    &nxt_V[mla_swizzle_idx<D_V>(row, col)],
                    &V_bh[gkv * v_stride_seq + col],
                    gkv < kv_len);
            }
        }
        bk::cp_async_commit();

        // ============================================================
        // D.3: Causal mask
        // ============================================================
        if ((causal && kv_start + BLOCK_KV > q_start) ||
            kv_start + BLOCK_KV > kv_len) {
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
                    if (col0 >= kv_len) { S_rmem[t][nc][0] = -FLT_MAX; S_rmem[t][nc][2] = -FLT_MAX; }
                    if (col1 >= kv_len) { S_rmem[t][nc][1] = -FLT_MAX; S_rmem[t][nc][3] = -FLT_MAX; }
                }
            }
        }

        // ============================================================
        // D.4: Online softmax (identical to v2)
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
        // D.5-6: P→A conversion + P*V MMA (uses D_V dimension)
        // ============================================================
        {
            #pragma unroll
            for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                int nc0 = 2 * kc;
                int nc1 = 2 * kc + 1;

                // Pack P fragments
                uint32_t P_a[WARP_Q_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_Q_TILES; t++) {
                    P_a[t][0] = mla_pack_bf16x2(S_rmem[t][nc0][0], S_rmem[t][nc0][1]);
                    P_a[t][1] = mla_pack_bf16x2(S_rmem[t][nc0][2], S_rmem[t][nc0][3]);
                    P_a[t][2] = mla_pack_bf16x2(S_rmem[t][nc1][0], S_rmem[t][nc1][1]);
                    P_a[t][3] = mla_pack_bf16x2(S_rmem[t][nc1][2], S_rmem[t][nc1][3]);
                }

                // Preload V fragments via ldmatrix_x4_trans (D_V swizzle)
                uint32_t V_all[O_N_CHUNKS / 2][4];
                {
                    int sub = lane_id / 8;
                    int t_in_sub = lane_id % 8;
                    #pragma unroll
                    for (int nc = 0; nc < O_N_CHUNKS; nc += 2) {
                        int v_row = kc * 16 + (sub % 2) * 8 + t_in_sub;
                        int v_col = (nc + sub / 2) * 8;
                        const void *addr_v = &cur_V[mla_swizzle_idx<D_V>(v_row, v_col)];
                        bk::ldmatrix_x4_trans(V_all[nc/2][0], V_all[nc/2][1],
                                              V_all[nc/2][2], V_all[nc/2][3], addr_v);
                    }
                }

                // MMA: P * V
                #pragma unroll
                for (int nc = 0; nc < O_N_CHUNKS; nc += 2) {
                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            O_rmem[t][nc][0], O_rmem[t][nc][1],
                            O_rmem[t][nc][2], O_rmem[t][nc][3],
                            P_a[t][0], P_a[t][1], P_a[t][2], P_a[t][3],
                            V_all[nc/2][0], V_all[nc/2][1],
                            O_rmem[t][nc][0], O_rmem[t][nc][1],
                            O_rmem[t][nc][2], O_rmem[t][nc][3]);

                        bk::mma_m16n8k16_bf16_nv(
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

        // Wait for prefetch
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

        // Store O: pack bf16 pairs for coalesced stores (D_V columns)
        #pragma unroll
        for (int nc = 0; nc < O_N_CHUNKS; nc++) {
            int col0 = nc * 8 + (lane_id % 4) * 2;

            if (gr0 < seq_len) {
                *reinterpret_cast<uint32_t*>(&O_bh[gr0 * o_stride_seq + col0]) =
                    mla_pack_bf16x2(O_rmem[t][nc][0], O_rmem[t][nc][1]);
            }
            if (gr1 < seq_len) {
                *reinterpret_cast<uint32_t*>(&O_bh[gr1 * o_stride_seq + col0]) =
                    mla_pack_bf16x2(O_rmem[t][nc][2], O_rmem[t][nc][3]);
            }
        }

        // Store logsumexp
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

template <int D_NOPE, int D_ROPE, int D_V>
void mla_attn_fwd_launch(
    const __nv_bfloat16 *Q_nope,
    const __nv_bfloat16 *Q_rope,
    const __nv_bfloat16 *K_nope,
    const __nv_bfloat16 *K_rope,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    float *L,
    int batch_heads,
    int seq_len,
    int kv_len,
    float scale,
    bool causal,
    int64_t qn_stride_b, int64_t qn_stride_h, int64_t qn_stride_seq,
    int64_t qr_stride_b, int64_t qr_stride_h, int64_t qr_stride_seq,
    int64_t kn_stride_b, int64_t kn_stride_h, int64_t kn_stride_seq,
    int64_t kr_stride_b, int64_t kr_stride_h, int64_t kr_stride_seq,
    int64_t v_stride_b,  int64_t v_stride_h,  int64_t v_stride_seq,
    int64_t o_stride_b,  int64_t o_stride_h,  int64_t o_stride_seq,
    int H,
    cudaStream_t stream)
{
    constexpr int BLOCK_Q = 64;

    // Choose BLOCK_KV to maximize occupancy:
    // smem per block = 2 * BKV * (D_NOPE + D_ROPE + D_V) * 2 bytes
    // Prefer BKV=64 if 3+ blocks/SM; fall back to BKV=48 if it improves occupancy
    constexpr int D_SUM = D_NOPE + D_ROPE + D_V;
    constexpr int smem_bkv64 = 2 * 64 * D_SUM * (int)sizeof(__nv_bfloat16);
    constexpr int smem_bkv48 = 2 * 48 * D_SUM * (int)sizeof(__nv_bfloat16);
    constexpr int blocks_bkv64 = 128 * 1024 / smem_bkv64;
    constexpr int blocks_bkv48 = 128 * 1024 / smem_bkv48;
    // Use BKV=48 only when BKV=64 can't reach 3 blocks/SM (occupancy bottleneck).
    // For configs where BKV=64 already gives 3+ blocks, extra softmax passes hurt.
    constexpr bool use_bkv48 = (blocks_bkv64 < 3) && (blocks_bkv48 > blocks_bkv64);
    constexpr int BLOCK_KV = use_bkv48 ? 48 : 64;

    constexpr int slot_bytes = BLOCK_KV * (D_NOPE + D_ROPE + D_V) * (int)sizeof(__nv_bfloat16);
    constexpr int smem_bytes = 2 * slot_bytes;

    // Dynamic BQ dispatch: BQ=128 when grid is large enough (same threshold as v2)
    // BQ=128 doubles MMA reuse per KV load. Needs launch_bounds(128, 2).
    int blocks_bq128 = ((seq_len + 127) / 128) * batch_heads;
    // Q smem for BQ=128: needs slot_bytes >= 128 * (D_NOPE + D_ROPE) * sizeof(bf16)
    constexpr int q128_bytes = 128 * (D_NOPE + D_ROPE) * (int)sizeof(__nv_bfloat16);
    constexpr bool bq128_fits = (smem_bytes >= q128_bytes);  // Q overlaps total smem, not one slot

    auto launch = [&](auto kernel_fn, int bq, int smem) {
        int nqb = (seq_len + bq - 1) / bq;
        dim3 grid(nqb, batch_heads);
        dim3 block(MLA_THREADS);
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(kernel_fn,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kernel_fn<<<grid, block, smem, stream>>>(
            Q_nope, Q_rope, K_nope, K_rope, V, O, L,
            seq_len, kv_len, scale, causal,
            qn_stride_b, qn_stride_h, qn_stride_seq,
            qr_stride_b, qr_stride_h, qr_stride_seq,
            kn_stride_b, kn_stride_h, kn_stride_seq,
            kr_stride_b, kr_stride_h, kr_stride_seq,
            v_stride_b, v_stride_h, v_stride_seq,
            o_stride_b, o_stride_h, o_stride_seq, H);
    };

    // Higher threshold than v2: BQ=128 only helps when grid is very large (3+ waves at 2 blocks/SM)
    // For moderate grids (e.g., 3B T=2048), the occupancy drop from 3→2 blocks/SM hurts.
    if (bq128_fits && blocks_bq128 >= 1024) {
        launch(mla_attn_fwd_kernel<D_NOPE, D_ROPE, D_V, BLOCK_KV, 128>, 128, smem_bytes);
    } else {
        launch(mla_attn_fwd_kernel<D_NOPE, D_ROPE, D_V, BLOCK_KV, 64>, 64, smem_bytes);
    }
}

// Explicit instantiations for all MLA variants
template void mla_attn_fwd_launch<48, 16, 48>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*,
    int, int, int, float, bool,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, cudaStream_t);

template void mla_attn_fwd_launch<64, 32, 64>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*,
    int, int, int, float, bool,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, cudaStream_t);

template void mla_attn_fwd_launch<96, 32, 96>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*,
    int, int, int, float, bool,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, cudaStream_t);

template void mla_attn_fwd_launch<128, 64, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*,
    int, int, int, float, bool,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, cudaStream_t);

} // namespace bk

// ============================================================
// PyTorch binding
// ============================================================


std::vector<torch::Tensor> mla_attn_forward(
    torch::Tensor Q_nope,   // [B, H, T, D] or [B, T, H, D]
    torch::Tensor Q_rope,
    torch::Tensor K_nope,
    torch::Tensor K_rope,
    torch::Tensor V,
    float scale,
    bool causal,
    bool bthd_layout = false)  // true = [B,T,H,D], false = [B,H,T,D] (default)
{
    TORCH_CHECK(Q_nope.is_cuda(), "Q_nope must be CUDA");
    TORCH_CHECK(Q_nope.dtype() == torch::kBFloat16, "Q_nope must be BF16");
    TORCH_CHECK(Q_nope.dim() == 4, "Q_nope must be 4D");
    TORCH_CHECK(Q_rope.is_cuda() && Q_rope.dtype() == torch::kBFloat16);
    TORCH_CHECK(K_nope.is_cuda() && K_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(K_rope.is_cuda() && K_rope.dtype() == torch::kBFloat16);
    TORCH_CHECK(V.is_cuda() && V.dtype() == torch::kBFloat16);
    TORCH_CHECK(Q_nope.stride(3) == 1, "D dimension must be contiguous");

    int B = Q_nope.size(0);
    int d_nope = Q_nope.size(3);
    int d_rope = Q_rope.size(3);
    int d_v = V.size(3);

    int dim_h = bthd_layout ? 2 : 1;
    int dim_t = bthd_layout ? 1 : 2;
    int H = Q_nope.size(dim_h);
    int T = Q_nope.size(dim_t);
    int S = K_nope.size(dim_t);
    int BH = B * H;

    // Extract strides: [batch_stride, head_stride, seq_stride] for each tensor
    auto qs = Q_nope.strides();
    int64_t qn_sb = qs[0], qn_sh = qs[dim_h], qn_ss = qs[dim_t];
    auto qrs = Q_rope.strides();
    int64_t qr_sb = qrs[0], qr_sh = qrs[dim_h], qr_ss = qrs[dim_t];
    auto kns = K_nope.strides();
    int64_t kn_sb = kns[0], kn_sh = kns[dim_h], kn_ss = kns[dim_t];
    auto krs = K_rope.strides();
    int64_t kr_sb = krs[0], kr_sh = krs[dim_h], kr_ss = krs[dim_t];
    auto vs = V.strides();
    int64_t v_sb = vs[0], v_sh = vs[dim_h], v_ss = vs[dim_t];

    // Output is always [B,H,T,D_V] contiguous (internal layout)
    auto O_out = torch::empty({BH, T, d_v}, V.options());
    auto L_out = torch::empty({BH, T}, Q_nope.options().dtype(torch::kFloat32));
    int64_t o_sb = (int64_t)H * T * d_v;
    int64_t o_sh = (int64_t)T * d_v;
    int64_t o_ss = (int64_t)d_v;

    auto stream = at::cuda::getCurrentCUDAStream();

    #define MLA_DISPATCH(DN, DR, DV) \
        if (d_nope == DN && d_rope == DR && d_v == DV) { \
            bk::mla_attn_fwd_launch<DN, DR, DV>( \
                reinterpret_cast<const __nv_bfloat16*>(Q_nope.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(Q_rope.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(K_nope.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(K_rope.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()), \
                reinterpret_cast<__nv_bfloat16*>(O_out.data_ptr()), \
                L_out.data_ptr<float>(), \
                BH, T, S, scale, causal, \
                qn_sb, qn_sh, qn_ss, qr_sb, qr_sh, qr_ss, \
                kn_sb, kn_sh, kn_ss, kr_sb, kr_sh, kr_ss, \
                v_sb, v_sh, v_ss, o_sb, o_sh, o_ss, H, stream); \
        }

    MLA_DISPATCH(48, 16, 48)
    else MLA_DISPATCH(64, 32, 64)
    else MLA_DISPATCH(96, 32, 96)
    else MLA_DISPATCH(128, 64, 128)
    else {
        TORCH_CHECK(false, "Unsupported MLA dimensions: d_nope=", d_nope,
                    " d_rope=", d_rope, " d_v=", d_v);
    }

    #undef MLA_DISPATCH

    return {O_out.reshape({B, H, T, d_v}), L_out.reshape({B, H, T})};
}
