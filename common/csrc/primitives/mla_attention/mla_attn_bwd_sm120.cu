// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// MLA (Multi-Latent Attention) Backward — Fused CUDA MMA Kernel for sm_120
//
// FlashAttention-2 backward structure:
//   - Grid: (ceil(S/BKV), B*H) — each threadblock owns a BKV slice of K/V
//   - Inner loop over Q blocks: recompute S, softmax, compute gradients
//   - dK_nope, dK_rope, dV accumulated in registers (owned by threadblock)
//   - dQ_nope, dQ_rope written to FP32 global buffer (partitioned, no atomics needed
//     because we swap the loop order: outer Q, inner KV for the dQ pass)
//
// Two-pass approach to avoid atomicAdd on dQ:
//   Pass 1 (outer KV, inner Q): compute dV, dK — accumulated in registers per KV block
//   Pass 2 (outer Q, inner KV): compute dQ — accumulated in registers per Q block
//
// Uses same MMA pipeline as forward: mma_m16n8k16_bf16_nv, ldmatrix_x4_mma,
// cp.async, XOR swizzle (mla_bwd_swizzle_idx), register-only P conversion.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <algorithm>

#include "mma_sm120.cuh"
#include "ldmatrix.cuh"
#include "cp_async.cuh"
#include "swizzle.cuh"

// Safe swizzle for non-power-of-2 column counts (same as forward)
template <int COLS>
__device__ __forceinline__ int mla_bwd_swizzle(int row, int col)
{
    constexpr int NUM_CHUNKS = COLS / 8;
    constexpr int SWIZZLE_BITS =
        (NUM_CHUNKS % 8 == 0) ? 3 :
        (NUM_CHUNKS % 4 == 0) ? 2 :
        (NUM_CHUNKS % 2 == 0) ? 1 : 0;
    constexpr int SWIZZLE_MASK = (1 << SWIZZLE_BITS) - 1;
    int swizzled_col = col ^ ((row & SWIZZLE_MASK) << 3);
    return row * COLS + swizzled_col;
}

constexpr int BWD_WARPS = 4;
constexpr int BWD_WARP = 32;
constexpr int BWD_THREADS = BWD_WARPS * BWD_WARP;

__device__ __forceinline__ uint32_t bwd_pack_bf16x2(float a, float b)
{
    __nv_bfloat162 packed = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&packed);
}

// ============================================================
// Pre-compute Di = rowsum(dO * O) — launched separately
// ============================================================
template <int D_V>
__global__ void mla_compute_Di_kernel(
    const __nv_bfloat16 *__restrict__ dO,
    const __nv_bfloat16 *__restrict__ O,
    float *__restrict__ Di,
    int T)
{
    int bh = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= T) return;
    const __nv_bfloat16 *dO_row = dO + (bh * T + row) * D_V;
    const __nv_bfloat16 *O_row  = O  + (bh * T + row) * D_V;
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_V; d++)
        sum += __bfloat162float(dO_row[d]) * __bfloat162float(O_row[d]);
    Di[bh * T + row] = sum;
}

// ============================================================
// Pass 1: dV + dK kernel (outer KV, inner Q)
// Each threadblock owns a BKV slice. Accumulates dV, dKn, dKr in registers.
// ============================================================
template <int D_NOPE, int D_ROPE, int D_V, int BKV, int BQ>
__global__ void __launch_bounds__(BWD_THREADS, 2)
mla_bwd_dVdK_kernel(
    const __nv_bfloat16 *__restrict__ Q_nope,
    const __nv_bfloat16 *__restrict__ Q_rope,
    const __nv_bfloat16 *__restrict__ K_nope,
    const __nv_bfloat16 *__restrict__ K_rope,
    const __nv_bfloat16 *__restrict__ V_in,
    const __nv_bfloat16 *__restrict__ dO_in,
    const float *__restrict__ L,
    const float *__restrict__ Di,
    float *__restrict__ dV_out,     // [BH, S, D_V] FP32
    float *__restrict__ dKn_out,    // [BH, S, D_NOPE] FP32
    float *__restrict__ dKr_out,    // [BH, S, D_ROPE] FP32
    int T, int S, float scale, bool causal)
{
    constexpr int WARP_KV = BKV / BWD_WARPS;
    constexpr int WARP_KV_TILES = WARP_KV / 16;
    constexpr int D_NOPE_CHUNKS = D_NOPE / 16;
    constexpr int D_ROPE_CHUNKS = D_ROPE / 16;
    constexpr int DV_N_CHUNKS = D_V / 8;
    constexpr int DN_N_CHUNKS = D_NOPE / 8;
    constexpr int DR_N_CHUNKS = D_ROPE / 8;
    constexpr int S_N_CHUNKS = BQ / 8;
    constexpr int P_K_CHUNKS = BQ / 16;

    // Smem layout: [K_nope | K_rope | V | Q_nope | Q_rope | dO]
    // K/V stay for the entire kernel; Q/dO reloaded each inner iteration
    constexpr int KN_ELEMS = BKV * D_NOPE;
    constexpr int KR_ELEMS = BKV * D_ROPE;
    constexpr int V_ELEMS  = BKV * D_V;
    constexpr int QN_ELEMS = BQ * D_NOPE;
    constexpr int QR_ELEMS = BQ * D_ROPE;
    constexpr int DO_ELEMS = BQ * D_V;

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16 *smem_Kn = smem;
    __nv_bfloat16 *smem_Kr = smem_Kn + KN_ELEMS;
    __nv_bfloat16 *smem_V  = smem_Kr + KR_ELEMS;
    __nv_bfloat16 *smem_Qn = smem_V  + V_ELEMS;
    __nv_bfloat16 *smem_Qr = smem_Qn + QN_ELEMS;
    __nv_bfloat16 *smem_dO = smem_Qr + QR_ELEMS;

    const int bh = blockIdx.y;
    const int kv_block = blockIdx.x;
    const int kv_start = kv_block * BKV;
    const int tid = threadIdx.x;
    const int warp_id = tid / BWD_WARP;
    const int lane_id = tid % BWD_WARP;

    const __nv_bfloat16 *Qn_bh = Q_nope + bh * T * D_NOPE;
    const __nv_bfloat16 *Qr_bh = Q_rope + bh * T * D_ROPE;
    const __nv_bfloat16 *Kn_bh = K_nope + bh * S * D_NOPE;
    const __nv_bfloat16 *Kr_bh = K_rope + bh * S * D_ROPE;
    const __nv_bfloat16 *V_bh  = V_in   + bh * S * D_V;
    const __nv_bfloat16 *dO_bh = dO_in  + bh * T * D_V;
    const float *L_bh = L + bh * T;
    const float *Di_bh = Di + bh * T;

    // ---- Load K_nope, K_rope, V for this KV block (stays in smem) ----
    {
        constexpr int KN_CPR = D_NOPE / 8;
        for (int i = tid; i < BKV * KN_CPR; i += BWD_THREADS) {
            int r = i / KN_CPR, c = (i % KN_CPR) * 8;
            int gr = kv_start + r;
            bk::cp_async_128_zfill(&smem_Kn[mla_bwd_swizzle<D_NOPE>(r, c)],
                                   &Kn_bh[gr * D_NOPE + c], gr < S);
        }
        constexpr int KR_CPR = D_ROPE / 8;
        for (int i = tid; i < BKV * KR_CPR; i += BWD_THREADS) {
            int r = i / KR_CPR, c = (i % KR_CPR) * 8;
            int gr = kv_start + r;
            bk::cp_async_128_zfill(&smem_Kr[mla_bwd_swizzle<D_ROPE>(r, c)],
                                   &Kr_bh[gr * D_ROPE + c], gr < S);
        }
        constexpr int V_CPR = D_V / 8;
        for (int i = tid; i < BKV * V_CPR; i += BWD_THREADS) {
            int r = i / V_CPR, c = (i % V_CPR) * 8;
            int gr = kv_start + r;
            bk::cp_async_128_zfill(&smem_V[mla_bwd_swizzle<D_V>(r, c)],
                                   &V_bh[gr * D_V + c], gr < S);
        }
    }
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    // ---- Initialize dV, dKn, dKr accumulators (per warp, WARP_KV rows) ----
    float dV_acc[WARP_KV_TILES][DV_N_CHUNKS][4];
    float dKn_acc[WARP_KV_TILES][DN_N_CHUNKS][4];
    float dKr_acc[WARP_KV_TILES][DR_N_CHUNKS][4];
    #pragma unroll
    for (int t = 0; t < WARP_KV_TILES; t++) {
        for (int n = 0; n < DV_N_CHUNKS; n++)
            dV_acc[t][n][0] = dV_acc[t][n][1] = dV_acc[t][n][2] = dV_acc[t][n][3] = 0.f;
        for (int n = 0; n < DN_N_CHUNKS; n++)
            dKn_acc[t][n][0] = dKn_acc[t][n][1] = dKn_acc[t][n][2] = dKn_acc[t][n][3] = 0.f;
        for (int n = 0; n < DR_N_CHUNKS; n++)
            dKr_acc[t][n][0] = dKr_acc[t][n][1] = dKr_acc[t][n][2] = dKr_acc[t][n][3] = 0.f;
    }

    // ---- Inner loop over Q blocks ----
    int q_end_limit = causal ? min(T, kv_start + BKV) : T;
    int num_q_blocks = (q_end_limit + BQ - 1) / BQ;

    for (int qb = 0; qb < num_q_blocks; qb++) {
        int q_start = qb * BQ;

        // Skip if this Q block is fully before our KV block (causal: all masked)
        if (causal && q_start + BQ <= kv_start) {
            // All Q rows in this block have q_row < kv_start, so kv_col > q_row for all
            // Wait, causal means mask where kv_col > q_row. If q_start+BQ <= kv_start,
            // then all q_rows < kv_start <= kv_cols, so all scores are masked. Skip.
            // Actually no — if q_start < kv_start, some q_rows might still attend to kv_start.
            // Skip only if ALL kv_cols > ALL q_rows: kv_start > q_start + BQ - 1.
            // i.e., kv_start >= q_start + BQ. This is the condition above. Correct.
            continue;
        }

        // Load Q_nope, Q_rope, dO for this Q block
        {
            constexpr int QN_CPR = D_NOPE / 8;
            for (int i = tid; i < BQ * QN_CPR; i += BWD_THREADS) {
                int r = i / QN_CPR, c = (i % QN_CPR) * 8;
                int gr = q_start + r;
                bk::cp_async_128_zfill(&smem_Qn[mla_bwd_swizzle<D_NOPE>(r, c)],
                                       &Qn_bh[gr * D_NOPE + c], gr < T);
            }
            constexpr int QR_CPR = D_ROPE / 8;
            for (int i = tid; i < BQ * QR_CPR; i += BWD_THREADS) {
                int r = i / QR_CPR, c = (i % QR_CPR) * 8;
                int gr = q_start + r;
                bk::cp_async_128_zfill(&smem_Qr[mla_bwd_swizzle<D_ROPE>(r, c)],
                                       &Qr_bh[gr * D_ROPE + c], gr < T);
            }
            constexpr int DO_CPR = D_V / 8;
            for (int i = tid; i < BQ * DO_CPR; i += BWD_THREADS) {
                int r = i / DO_CPR, c = (i % DO_CPR) * 8;
                int gr = q_start + r;
                bk::cp_async_128_zfill(&smem_dO[mla_bwd_swizzle<D_V>(r, c)],
                                       &dO_bh[gr * D_V + c], gr < T);
            }
        }
        bk::cp_async_commit();
        bk::cp_async_wait<0>();
        __syncthreads();

        // ---- Recompute S = scale * (Qn @ Kn^T + Qr @ Kr^T) ----
        // Each warp computes WARP_KV rows of S^T (= BKV-partitioned).
        // S^T[kv, q] = scale * (Kn[kv,:] @ Qn[q,:]^T + Kr[kv,:] @ Qr[q,:]^T)
        // This is equivalent to computing S^T = K @ Q^T, which gives us the
        // transposed score matrix directly — needed for dV = S^T_softmax @ dO.
        //
        // MMA: S^T_tile[kv_tile, q_n] = sum_d K[kv_tile, d] * Q^T[d, q_n]
        // A = K fragments (row-major from K smem), B = Q^T (col-major = Q row-major via ldmatrix_x4)
        constexpr int ST_N_CHUNKS = BQ / 8;  // S^T has BQ columns

        float ST_rmem[WARP_KV_TILES][ST_N_CHUNKS][4];
        #pragma unroll
        for (int t = 0; t < WARP_KV_TILES; t++)
            for (int n = 0; n < ST_N_CHUNKS; n++)
                ST_rmem[t][n][0] = ST_rmem[t][n][1] = ST_rmem[t][n][2] = ST_rmem[t][n][3] = 0.f;

        {
            int sub = lane_id / 8;
            int t_in_sub = lane_id % 8;
            int warp_kv_off = warp_id * WARP_KV;

            // Nope: K_nope[kv, d] @ Q_nope^T[d, q]
            #pragma unroll
            for (int dc = 0; dc < D_NOPE_CHUNKS; dc++) {
                // Load K_nope A-fragments for this warp's KV rows
                uint32_t Ka[WARP_KV_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_KV_TILES; t++) {
                    int kr = warp_kv_off + t * 16 + (sub / 2) * 8 + t_in_sub;
                    int kc = dc * 16 + (sub % 2) * 8;
                    bk::ldmatrix_x4_mma(Ka[t][0], Ka[t][1], Ka[t][2], Ka[t][3],
                        &smem_Kn[mla_bwd_swizzle<D_NOPE>(kr, kc)]);
                }

                #pragma unroll
                for (int nc = 0; nc < ST_N_CHUNKS; nc += 2) {
                    // Load Q_nope B-fragments (Q as col-major = Q^T)
                    int qr = (nc + sub / 2) * 8 + t_in_sub;
                    int qc = dc * 16 + (sub % 2) * 8;
                    uint32_t Qb0, Qb1, Qb2, Qb3;
                    bk::ldmatrix_x4(Qb0, Qb1, Qb2, Qb3,
                        &smem_Qn[mla_bwd_swizzle<D_NOPE>(qr, qc)]);

                    #pragma unroll
                    for (int t = 0; t < WARP_KV_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            ST_rmem[t][nc][0], ST_rmem[t][nc][1],
                            ST_rmem[t][nc][2], ST_rmem[t][nc][3],
                            Ka[t][0], Ka[t][1], Ka[t][2], Ka[t][3],
                            Qb0, Qb1,
                            ST_rmem[t][nc][0], ST_rmem[t][nc][1],
                            ST_rmem[t][nc][2], ST_rmem[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            ST_rmem[t][nc+1][0], ST_rmem[t][nc+1][1],
                            ST_rmem[t][nc+1][2], ST_rmem[t][nc+1][3],
                            Ka[t][0], Ka[t][1], Ka[t][2], Ka[t][3],
                            Qb2, Qb3,
                            ST_rmem[t][nc+1][0], ST_rmem[t][nc+1][1],
                            ST_rmem[t][nc+1][2], ST_rmem[t][nc+1][3]);
                    }
                }
            }

            // Rope: K_rope[kv, d] @ Q_rope^T[d, q]
            #pragma unroll
            for (int dc = 0; dc < D_ROPE_CHUNKS; dc++) {
                uint32_t Ka[WARP_KV_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_KV_TILES; t++) {
                    int kr = warp_kv_off + t * 16 + (sub / 2) * 8 + t_in_sub;
                    int kc = dc * 16 + (sub % 2) * 8;
                    bk::ldmatrix_x4_mma(Ka[t][0], Ka[t][1], Ka[t][2], Ka[t][3],
                        &smem_Kr[mla_bwd_swizzle<D_ROPE>(kr, kc)]);
                }

                #pragma unroll
                for (int nc = 0; nc < ST_N_CHUNKS; nc += 2) {
                    int qr = (nc + sub / 2) * 8 + t_in_sub;
                    int qc = dc * 16 + (sub % 2) * 8;
                    uint32_t Qb0, Qb1, Qb2, Qb3;
                    bk::ldmatrix_x4(Qb0, Qb1, Qb2, Qb3,
                        &smem_Qr[mla_bwd_swizzle<D_ROPE>(qr, qc)]);

                    #pragma unroll
                    for (int t = 0; t < WARP_KV_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            ST_rmem[t][nc][0], ST_rmem[t][nc][1],
                            ST_rmem[t][nc][2], ST_rmem[t][nc][3],
                            Ka[t][0], Ka[t][1], Ka[t][2], Ka[t][3],
                            Qb0, Qb1,
                            ST_rmem[t][nc][0], ST_rmem[t][nc][1],
                            ST_rmem[t][nc][2], ST_rmem[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            ST_rmem[t][nc+1][0], ST_rmem[t][nc+1][1],
                            ST_rmem[t][nc+1][2], ST_rmem[t][nc+1][3],
                            Ka[t][0], Ka[t][1], Ka[t][2], Ka[t][3],
                            Qb2, Qb3,
                            ST_rmem[t][nc+1][0], ST_rmem[t][nc+1][1],
                            ST_rmem[t][nc+1][2], ST_rmem[t][nc+1][3]);
                    }
                }
            }
        }

        // Apply scale
        #pragma unroll
        for (int t = 0; t < WARP_KV_TILES; t++)
            for (int n = 0; n < ST_N_CHUNKS; n++) {
                ST_rmem[t][n][0] *= scale; ST_rmem[t][n][1] *= scale;
                ST_rmem[t][n][2] *= scale; ST_rmem[t][n][3] *= scale;
            }

        // ---- Causal mask + Softmax via saved L ----
        // ST_rmem[kv_tile][q_nc] holds S^T[kv, q]. Mask where kv > q.
        // P^T[kv, q] = exp(S^T[kv, q] - L[q])
        // We need L[q] for each q column. Load L for this Q block.
        // D-fragment layout: d0→(T/4, (T%4)*2), d1→(T/4, (T%4)*2+1), d2→(T/4+8, ...), d3→(T/4+8, ...)
        // For S^T: M=KV rows, N=Q cols. Each element [kv_row, q_col].
        // d0 → kv_row = T/4, q_col = (T%4)*2  where T = lane_id
        // d2 → kv_row = T/4+8
        {
            int warp_kv_off = warp_id * WARP_KV;
            #pragma unroll
            for (int t = 0; t < WARP_KV_TILES; t++) {
                int kv_r0 = kv_start + warp_kv_off + t * 16 + (lane_id / 4);
                int kv_r1 = kv_r0 + 8;
                #pragma unroll
                for (int nc = 0; nc < ST_N_CHUNKS; nc++) {
                    int q_c0 = q_start + nc * 8 + (lane_id % 4) * 2;
                    int q_c1 = q_c0 + 1;

                    // Causal: mask where kv > q
                    if (causal) {
                        if (kv_r0 > q_c0) ST_rmem[t][nc][0] = -FLT_MAX;
                        if (kv_r0 > q_c1) ST_rmem[t][nc][1] = -FLT_MAX;
                        if (kv_r1 > q_c0) ST_rmem[t][nc][2] = -FLT_MAX;
                        if (kv_r1 > q_c1) ST_rmem[t][nc][3] = -FLT_MAX;
                    }
                    // OOB mask
                    if (kv_r0 >= S) { ST_rmem[t][nc][0] = -FLT_MAX; ST_rmem[t][nc][2] = -FLT_MAX; }
                    if (kv_r1 >= S) { ST_rmem[t][nc][2] = -FLT_MAX; ST_rmem[t][nc][3] = -FLT_MAX; }
                    if (q_c0 >= T) { ST_rmem[t][nc][0] = -FLT_MAX; ST_rmem[t][nc][2] = -FLT_MAX; }
                    if (q_c1 >= T) { ST_rmem[t][nc][1] = -FLT_MAX; ST_rmem[t][nc][3] = -FLT_MAX; }

                    // P^T[kv, q] = exp(S^T[kv, q] - L[q])
                    float L_q0 = (q_c0 < T) ? L_bh[q_c0] : 0.f;
                    float L_q1 = (q_c1 < T) ? L_bh[q_c1] : 0.f;
                    ST_rmem[t][nc][0] = expf(ST_rmem[t][nc][0] - L_q0);
                    ST_rmem[t][nc][1] = expf(ST_rmem[t][nc][1] - L_q1);
                    ST_rmem[t][nc][2] = expf(ST_rmem[t][nc][2] - L_q0);
                    ST_rmem[t][nc][3] = expf(ST_rmem[t][nc][3] - L_q1);
                }
            }
        }
        // ST_rmem now holds P^T[kv, q] in FP32

        // ---- dV += P^T @ dO ----
        // P^T is [BKV x BQ], dO is [BQ x D_V]. Result: [BKV x D_V].
        // Each warp's P^T tile is [WARP_KV x BQ], already in ST_rmem.
        // Pack P^T to BF16 A-fragments and MMA with dO B-fragments.
        {
            #pragma unroll
            for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                int nc0 = 2 * kc, nc1 = 2 * kc + 1;
                uint32_t Pa[WARP_KV_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_KV_TILES; t++) {
                    Pa[t][0] = bwd_pack_bf16x2(ST_rmem[t][nc0][0], ST_rmem[t][nc0][1]);
                    Pa[t][1] = bwd_pack_bf16x2(ST_rmem[t][nc0][2], ST_rmem[t][nc0][3]);
                    Pa[t][2] = bwd_pack_bf16x2(ST_rmem[t][nc1][0], ST_rmem[t][nc1][1]);
                    Pa[t][3] = bwd_pack_bf16x2(ST_rmem[t][nc1][2], ST_rmem[t][nc1][3]);
                }

                // Load dO B-fragments via ldmatrix_x4_trans
                int sub = lane_id / 8;
                int t_in_sub = lane_id % 8;
                uint32_t dOb[DV_N_CHUNKS / 2][4];
                #pragma unroll
                for (int nc = 0; nc < DV_N_CHUNKS; nc += 2) {
                    int dr = kc * 16 + (sub % 2) * 8 + t_in_sub;
                    int dc = (nc + sub / 2) * 8;
                    bk::ldmatrix_x4_trans(dOb[nc/2][0], dOb[nc/2][1],
                                          dOb[nc/2][2], dOb[nc/2][3],
                        &smem_dO[mla_bwd_swizzle<D_V>(dr, dc)]);
                }

                #pragma unroll
                for (int nc = 0; nc < DV_N_CHUNKS; nc += 2) {
                    #pragma unroll
                    for (int t = 0; t < WARP_KV_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            dV_acc[t][nc][0], dV_acc[t][nc][1],
                            dV_acc[t][nc][2], dV_acc[t][nc][3],
                            Pa[t][0], Pa[t][1], Pa[t][2], Pa[t][3],
                            dOb[nc/2][0], dOb[nc/2][1],
                            dV_acc[t][nc][0], dV_acc[t][nc][1],
                            dV_acc[t][nc][2], dV_acc[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            dV_acc[t][nc+1][0], dV_acc[t][nc+1][1],
                            dV_acc[t][nc+1][2], dV_acc[t][nc+1][3],
                            Pa[t][0], Pa[t][1], Pa[t][2], Pa[t][3],
                            dOb[nc/2][2], dOb[nc/2][3],
                            dV_acc[t][nc+1][0], dV_acc[t][nc+1][1],
                            dV_acc[t][nc+1][2], dV_acc[t][nc+1][3]);
                    }
                }
            }
        }

        // ---- Compute dS^T = P^T * (dO @ V^T - Di)^T * scale ----
        // First: dP^T[kv, q] = V[kv,:] @ dO[q,:]^T (same structure as K@Q^T)
        // Then: dS^T = P^T * (dP^T - Di[q]) * scale
        float dST_rmem[WARP_KV_TILES][ST_N_CHUNKS][4];
        #pragma unroll
        for (int t = 0; t < WARP_KV_TILES; t++)
            for (int n = 0; n < ST_N_CHUNKS; n++)
                dST_rmem[t][n][0] = dST_rmem[t][n][1] = dST_rmem[t][n][2] = dST_rmem[t][n][3] = 0.f;

        // dP^T = V @ dO^T: A = V fragments, B = dO^T (= dO row-major via ldmatrix)
        {
            constexpr int DV_CHUNKS = D_V / 16;
            int sub = lane_id / 8;
            int t_in_sub = lane_id % 8;
            int warp_kv_off = warp_id * WARP_KV;

            #pragma unroll
            for (int dc = 0; dc < DV_CHUNKS; dc++) {
                uint32_t Va[WARP_KV_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_KV_TILES; t++) {
                    int vr = warp_kv_off + t * 16 + (sub / 2) * 8 + t_in_sub;
                    int vc = dc * 16 + (sub % 2) * 8;
                    bk::ldmatrix_x4_mma(Va[t][0], Va[t][1], Va[t][2], Va[t][3],
                        &smem_V[mla_bwd_swizzle<D_V>(vr, vc)]);
                }

                #pragma unroll
                for (int nc = 0; nc < ST_N_CHUNKS; nc += 2) {
                    int dr = (nc + sub / 2) * 8 + t_in_sub;
                    int dcc = dc * 16 + (sub % 2) * 8;
                    uint32_t dOb0, dOb1, dOb2, dOb3;
                    bk::ldmatrix_x4(dOb0, dOb1, dOb2, dOb3,
                        &smem_dO[mla_bwd_swizzle<D_V>(dr, dcc)]);

                    #pragma unroll
                    for (int t = 0; t < WARP_KV_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            dST_rmem[t][nc][0], dST_rmem[t][nc][1],
                            dST_rmem[t][nc][2], dST_rmem[t][nc][3],
                            Va[t][0], Va[t][1], Va[t][2], Va[t][3],
                            dOb0, dOb1,
                            dST_rmem[t][nc][0], dST_rmem[t][nc][1],
                            dST_rmem[t][nc][2], dST_rmem[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            dST_rmem[t][nc+1][0], dST_rmem[t][nc+1][1],
                            dST_rmem[t][nc+1][2], dST_rmem[t][nc+1][3],
                            Va[t][0], Va[t][1], Va[t][2], Va[t][3],
                            dOb2, dOb3,
                            dST_rmem[t][nc+1][0], dST_rmem[t][nc+1][1],
                            dST_rmem[t][nc+1][2], dST_rmem[t][nc+1][3]);
                    }
                }
            }
        }

        // dS^T = P^T * (dP^T - Di[q]) * scale
        {
            #pragma unroll
            for (int t = 0; t < WARP_KV_TILES; t++) {
                #pragma unroll
                for (int nc = 0; nc < ST_N_CHUNKS; nc++) {
                    int q_c0 = q_start + nc * 8 + (lane_id % 4) * 2;
                    int q_c1 = q_c0 + 1;
                    float Di_q0 = (q_c0 < T) ? Di_bh[q_c0] : 0.f;
                    float Di_q1 = (q_c1 < T) ? Di_bh[q_c1] : 0.f;

                    dST_rmem[t][nc][0] = ST_rmem[t][nc][0] * (dST_rmem[t][nc][0] - Di_q0) * scale;
                    dST_rmem[t][nc][1] = ST_rmem[t][nc][1] * (dST_rmem[t][nc][1] - Di_q1) * scale;
                    dST_rmem[t][nc][2] = ST_rmem[t][nc][2] * (dST_rmem[t][nc][2] - Di_q0) * scale;
                    dST_rmem[t][nc][3] = ST_rmem[t][nc][3] * (dST_rmem[t][nc][3] - Di_q1) * scale;
                }
            }
        }

        // ---- dK += dS^T @ Q ----
        // dS^T is [BKV x BQ], Q is [BQ x D]. Result: [BKV x D].
        // A = dS^T (in registers), B = Q (in smem).
        // Pack dS^T to BF16 A-fragments, MMA with Q B-fragments.
        {
            #pragma unroll
            for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                int nc0 = 2 * kc, nc1 = 2 * kc + 1;
                uint32_t dSa[WARP_KV_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_KV_TILES; t++) {
                    dSa[t][0] = bwd_pack_bf16x2(dST_rmem[t][nc0][0], dST_rmem[t][nc0][1]);
                    dSa[t][1] = bwd_pack_bf16x2(dST_rmem[t][nc0][2], dST_rmem[t][nc0][3]);
                    dSa[t][2] = bwd_pack_bf16x2(dST_rmem[t][nc1][0], dST_rmem[t][nc1][1]);
                    dSa[t][3] = bwd_pack_bf16x2(dST_rmem[t][nc1][2], dST_rmem[t][nc1][3]);
                }

                int sub = lane_id / 8;
                int t_in_sub = lane_id % 8;

                // dKn += dS^T @ Qn
                uint32_t Qb[DN_N_CHUNKS / 2][4];
                #pragma unroll
                for (int nc = 0; nc < DN_N_CHUNKS; nc += 2) {
                    int qr = kc * 16 + (sub % 2) * 8 + t_in_sub;
                    int qc = (nc + sub / 2) * 8;
                    bk::ldmatrix_x4_trans(Qb[nc/2][0], Qb[nc/2][1],
                                          Qb[nc/2][2], Qb[nc/2][3],
                        &smem_Qn[mla_bwd_swizzle<D_NOPE>(qr, qc)]);
                }
                #pragma unroll
                for (int nc = 0; nc < DN_N_CHUNKS; nc += 2) {
                    #pragma unroll
                    for (int t = 0; t < WARP_KV_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            dKn_acc[t][nc][0], dKn_acc[t][nc][1],
                            dKn_acc[t][nc][2], dKn_acc[t][nc][3],
                            dSa[t][0], dSa[t][1], dSa[t][2], dSa[t][3],
                            Qb[nc/2][0], Qb[nc/2][1],
                            dKn_acc[t][nc][0], dKn_acc[t][nc][1],
                            dKn_acc[t][nc][2], dKn_acc[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            dKn_acc[t][nc+1][0], dKn_acc[t][nc+1][1],
                            dKn_acc[t][nc+1][2], dKn_acc[t][nc+1][3],
                            dSa[t][0], dSa[t][1], dSa[t][2], dSa[t][3],
                            Qb[nc/2][2], Qb[nc/2][3],
                            dKn_acc[t][nc+1][0], dKn_acc[t][nc+1][1],
                            dKn_acc[t][nc+1][2], dKn_acc[t][nc+1][3]);
                    }
                }

                // dKr += dS^T @ Qr
                uint32_t Qrb[DR_N_CHUNKS / 2][4];
                #pragma unroll
                for (int nc = 0; nc < DR_N_CHUNKS; nc += 2) {
                    int qr = kc * 16 + (sub % 2) * 8 + t_in_sub;
                    int qc = (nc + sub / 2) * 8;
                    bk::ldmatrix_x4_trans(Qrb[nc/2][0], Qrb[nc/2][1],
                                          Qrb[nc/2][2], Qrb[nc/2][3],
                        &smem_Qr[mla_bwd_swizzle<D_ROPE>(qr, qc)]);
                }
                #pragma unroll
                for (int nc = 0; nc < DR_N_CHUNKS; nc += 2) {
                    #pragma unroll
                    for (int t = 0; t < WARP_KV_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            dKr_acc[t][nc][0], dKr_acc[t][nc][1],
                            dKr_acc[t][nc][2], dKr_acc[t][nc][3],
                            dSa[t][0], dSa[t][1], dSa[t][2], dSa[t][3],
                            Qrb[nc/2][0], Qrb[nc/2][1],
                            dKr_acc[t][nc][0], dKr_acc[t][nc][1],
                            dKr_acc[t][nc][2], dKr_acc[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            dKr_acc[t][nc+1][0], dKr_acc[t][nc+1][1],
                            dKr_acc[t][nc+1][2], dKr_acc[t][nc+1][3],
                            dSa[t][0], dSa[t][1], dSa[t][2], dSa[t][3],
                            Qrb[nc/2][2], Qrb[nc/2][3],
                            dKr_acc[t][nc+1][0], dKr_acc[t][nc+1][1],
                            dKr_acc[t][nc+1][2], dKr_acc[t][nc+1][3]);
                    }
                }
            }
        }

        __syncthreads();
    } // end inner Q loop

    // ---- Store dV, dKn, dKr to global FP32 ----
    {
        int warp_kv_off = warp_id * WARP_KV;
        #pragma unroll
        for (int t = 0; t < WARP_KV_TILES; t++) {
            int gr0 = kv_start + warp_kv_off + t * 16 + (lane_id / 4);
            int gr1 = gr0 + 8;

            // dV
            #pragma unroll
            for (int nc = 0; nc < DV_N_CHUNKS; nc++) {
                int col = nc * 8 + (lane_id % 4) * 2;
                if (gr0 < S) { dV_out[bh*S*D_V + gr0*D_V + col] = dV_acc[t][nc][0];
                               dV_out[bh*S*D_V + gr0*D_V + col+1] = dV_acc[t][nc][1]; }
                if (gr1 < S) { dV_out[bh*S*D_V + gr1*D_V + col] = dV_acc[t][nc][2];
                               dV_out[bh*S*D_V + gr1*D_V + col+1] = dV_acc[t][nc][3]; }
            }
            // dKn
            #pragma unroll
            for (int nc = 0; nc < DN_N_CHUNKS; nc++) {
                int col = nc * 8 + (lane_id % 4) * 2;
                if (gr0 < S) { dKn_out[bh*S*D_NOPE + gr0*D_NOPE + col] = dKn_acc[t][nc][0];
                               dKn_out[bh*S*D_NOPE + gr0*D_NOPE + col+1] = dKn_acc[t][nc][1]; }
                if (gr1 < S) { dKn_out[bh*S*D_NOPE + gr1*D_NOPE + col] = dKn_acc[t][nc][2];
                               dKn_out[bh*S*D_NOPE + gr1*D_NOPE + col+1] = dKn_acc[t][nc][3]; }
            }
            // dKr
            #pragma unroll
            for (int nc = 0; nc < DR_N_CHUNKS; nc++) {
                int col = nc * 8 + (lane_id % 4) * 2;
                if (gr0 < S) { dKr_out[bh*S*D_ROPE + gr0*D_ROPE + col] = dKr_acc[t][nc][0];
                               dKr_out[bh*S*D_ROPE + gr0*D_ROPE + col+1] = dKr_acc[t][nc][1]; }
                if (gr1 < S) { dKr_out[bh*S*D_ROPE + gr1*D_ROPE + col] = dKr_acc[t][nc][2];
                               dKr_out[bh*S*D_ROPE + gr1*D_ROPE + col+1] = dKr_acc[t][nc][3]; }
            }
        }
    }
}

// ============================================================
// Pass 2: dQ kernel (outer Q, inner KV) — mirrors forward structure
// Each threadblock owns a BQ slice. Loads Q/dO once.
// Inner loop over KV: recompute S→P→dS, accumulate dQ = dS @ K.
// ============================================================
template <int D_NOPE, int D_ROPE, int D_V, int BKV, int BQ>
__global__ void __launch_bounds__(BWD_THREADS)
mla_bwd_dQ_kernel(
    const __nv_bfloat16 *__restrict__ Q_nope,
    const __nv_bfloat16 *__restrict__ Q_rope,
    const __nv_bfloat16 *__restrict__ K_nope,
    const __nv_bfloat16 *__restrict__ K_rope,
    const __nv_bfloat16 *__restrict__ V_in,
    const __nv_bfloat16 *__restrict__ dO_in,
    const float *__restrict__ L,
    const float *__restrict__ Di,
    float *__restrict__ dQn_out,   // [BH, T, D_NOPE] FP32
    float *__restrict__ dQr_out,   // [BH, T, D_ROPE] FP32
    int T, int S, float scale, bool causal)
{
    // Mirrors forward: each warp handles WARP_Q = BQ/4 rows
    constexpr int WARP_Q = BQ / BWD_WARPS;
    constexpr int WARP_Q_TILES = WARP_Q / 16;
    constexpr int D_NOPE_CHUNKS = D_NOPE / 16;
    constexpr int D_ROPE_CHUNKS = D_ROPE / 16;
    constexpr int DV_CHUNKS = D_V / 16;
    constexpr int S_N_CHUNKS = BKV / 8;
    constexpr int P_K_CHUNKS = BKV / 16;
    constexpr int DN_N_CHUNKS = D_NOPE / 8;
    constexpr int DR_N_CHUNKS = D_ROPE / 8;

    // Smem: [K_nope | K_rope | V | dO]  — K/V loaded per KV block, dO stays
    // Q loaded to smem temporarily at start, then freed (overlaps KV region)
    constexpr int KN_ELEMS = BKV * D_NOPE;
    constexpr int KR_ELEMS = BKV * D_ROPE;
    constexpr int V_ELEMS  = BKV * D_V;
    constexpr int DO_ELEMS = BQ * D_V;
    constexpr int QN_ELEMS = BQ * D_NOPE;
    constexpr int QR_ELEMS = BQ * D_ROPE;

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16 *smem_Kn = smem;
    __nv_bfloat16 *smem_Kr = smem_Kn + KN_ELEMS;
    __nv_bfloat16 *smem_V  = smem_Kr + KR_ELEMS;
    __nv_bfloat16 *smem_dO = smem_V  + V_ELEMS;

    const int bh = blockIdx.y;
    const int q_block = blockIdx.x;
    const int q_start = q_block * BQ;
    const int tid = threadIdx.x;
    const int warp_id = tid / BWD_WARP;
    const int lane_id = tid % BWD_WARP;

    const __nv_bfloat16 *Kn_bh = K_nope + bh * S * D_NOPE;
    const __nv_bfloat16 *Kr_bh = K_rope + bh * S * D_ROPE;
    const __nv_bfloat16 *V_bh  = V_in   + bh * S * D_V;
    const __nv_bfloat16 *dO_bh = dO_in  + bh * T * D_V;
    const float *L_bh = L + bh * T;
    const float *Di_bh = Di + bh * T;

    // Load Q_nope + Q_rope to smem (overlaps KV region), then ldmatrix to registers
    // This mirrors the forward's Phase A.
    uint32_t Qn_rmem[WARP_Q_TILES][D_NOPE_CHUNKS][4];
    uint32_t Qr_rmem[WARP_Q_TILES][D_ROPE_CHUNKS][4];
    {
        __nv_bfloat16 *smem_Qtmp = smem;  // overlaps KV region (freed before KV loop)
        __nv_bfloat16 *smem_Qn_tmp = smem_Qtmp;
        __nv_bfloat16 *smem_Qr_tmp = smem_Qtmp + QN_ELEMS;

        constexpr int QN_CPR = D_NOPE / 8;
        for (int i = tid; i < BQ * QN_CPR; i += BWD_THREADS) {
            int r = i / QN_CPR, c = (i % QN_CPR) * 8;
            int gr = q_start + r;
            bk::cp_async_128_zfill(&smem_Qn_tmp[mla_bwd_swizzle<D_NOPE>(r, c)],
                                   &(Q_nope + bh * T * D_NOPE)[gr * D_NOPE + c], gr < T);
        }
        constexpr int QR_CPR = D_ROPE / 8;
        for (int i = tid; i < BQ * QR_CPR; i += BWD_THREADS) {
            int r = i / QR_CPR, c = (i % QR_CPR) * 8;
            int gr = q_start + r;
            bk::cp_async_128_zfill(&smem_Qr_tmp[mla_bwd_swizzle<D_ROPE>(r, c)],
                                   &(Q_rope + bh * T * D_ROPE)[gr * D_ROPE + c], gr < T);
        }
        bk::cp_async_commit();
        bk::cp_async_wait<0>();
        __syncthreads();

        int warp_q_off = warp_id * WARP_Q;
        int sub = lane_id / 8;
        int t_in_sub = lane_id % 8;
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            int tile_off = warp_q_off + t * 16;
            #pragma unroll
            for (int dc = 0; dc < D_NOPE_CHUNKS; dc++) {
                int sr = tile_off + (sub / 2) * 8 + t_in_sub;
                int sc = dc * 16 + (sub % 2) * 8;
                bk::ldmatrix_x4_mma(Qn_rmem[t][dc][0], Qn_rmem[t][dc][1],
                                    Qn_rmem[t][dc][2], Qn_rmem[t][dc][3],
                    &smem_Qn_tmp[mla_bwd_swizzle<D_NOPE>(sr, sc)]);
            }
            #pragma unroll
            for (int dc = 0; dc < D_ROPE_CHUNKS; dc++) {
                int sr = tile_off + (sub / 2) * 8 + t_in_sub;
                int sc = dc * 16 + (sub % 2) * 8;
                bk::ldmatrix_x4_mma(Qr_rmem[t][dc][0], Qr_rmem[t][dc][1],
                                    Qr_rmem[t][dc][2], Qr_rmem[t][dc][3],
                    &smem_Qr_tmp[mla_bwd_swizzle<D_ROPE>(sr, sc)]);
            }
        }
        __syncthreads();
    }

    // Load dO for this Q block (stays in smem for entire kernel)
    {
        constexpr int DO_CPR = D_V / 8;
        for (int i = tid; i < BQ * DO_CPR; i += BWD_THREADS) {
            int r = i / DO_CPR, c = (i % DO_CPR) * 8;
            int gr = q_start + r;
            bk::cp_async_128_zfill(&smem_dO[mla_bwd_swizzle<D_V>(r, c)],
                                   &dO_bh[gr * D_V + c], gr < T);
        }
    }
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    // Initialize dQn, dQr accumulators
    float dQn_acc[WARP_Q_TILES][DN_N_CHUNKS][4];
    float dQr_acc[WARP_Q_TILES][DR_N_CHUNKS][4];
    #pragma unroll
    for (int t = 0; t < WARP_Q_TILES; t++) {
        for (int n = 0; n < DN_N_CHUNKS; n++)
            dQn_acc[t][n][0] = dQn_acc[t][n][1] = dQn_acc[t][n][2] = dQn_acc[t][n][3] = 0.f;
        for (int n = 0; n < DR_N_CHUNKS; n++)
            dQr_acc[t][n][0] = dQr_acc[t][n][1] = dQr_acc[t][n][2] = dQr_acc[t][n][3] = 0.f;
    }

    // Softmax state for this Q block
    int global_rows[2 * WARP_Q_TILES];
    #pragma unroll
    for (int t = 0; t < WARP_Q_TILES; t++) {
        global_rows[2*t]   = q_start + warp_id * WARP_Q + t * 16 + (lane_id / 4);
        global_rows[2*t+1] = global_rows[2*t] + 8;
    }

    // Inner loop over KV blocks — with prologue + prefetch (mirrors forward pattern)
    int kv_end = causal ? min(S, q_start + BQ) : S;
    int num_kv_blocks = (kv_end + BKV - 1) / BKV;

    // Macro for loading KV to smem
    #define LOAD_KV_TO_SMEM(kv_start_val) \
    { \
        constexpr int KN_CPR = D_NOPE / 8; \
        for (int i = tid; i < BKV * KN_CPR; i += BWD_THREADS) { \
            int r = i / KN_CPR, c = (i % KN_CPR) * 8; \
            int gr = (kv_start_val) + r; \
            bk::cp_async_128_zfill(&smem_Kn[mla_bwd_swizzle<D_NOPE>(r, c)], \
                                   &Kn_bh[gr * D_NOPE + c], gr < S); \
        } \
        constexpr int KR_CPR = D_ROPE / 8; \
        for (int i = tid; i < BKV * KR_CPR; i += BWD_THREADS) { \
            int r = i / KR_CPR, c = (i % KR_CPR) * 8; \
            int gr = (kv_start_val) + r; \
            bk::cp_async_128_zfill(&smem_Kr[mla_bwd_swizzle<D_ROPE>(r, c)], \
                                   &Kr_bh[gr * D_ROPE + c], gr < S); \
        } \
        constexpr int V_CPR = D_V / 8; \
        for (int i = tid; i < BKV * V_CPR; i += BWD_THREADS) { \
            int r = i / V_CPR, c = (i % V_CPR) * 8; \
            int gr = (kv_start_val) + r; \
            bk::cp_async_128_zfill(&smem_V[mla_bwd_swizzle<D_V>(r, c)], \
                                   &V_bh[gr * D_V + c], gr < S); \
        } \
    }

    // Prologue: load first KV block
    if (num_kv_blocks > 0) {
        LOAD_KV_TO_SMEM(0)
        bk::cp_async_commit();
        bk::cp_async_wait<0>();
        __syncthreads();
    }

    for (int kvb = 0; kvb < num_kv_blocks; kvb++) {
        int kv_start = kvb * BKV;

        // Recompute S = scale * (Qn @ Kn^T + Qr @ Kr^T) via MMA
        // Q is in registers (loaded once at start), K from smem
        float S_rmem[WARP_Q_TILES][S_N_CHUNKS][4];
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++)
            for (int n = 0; n < S_N_CHUNKS; n++)
                S_rmem[t][n][0] = S_rmem[t][n][1] = S_rmem[t][n][2] = S_rmem[t][n][3] = 0.f;

        {
            int sub = lane_id / 8;
            int t_in_sub = lane_id % 8;

            // Nope sub-product
            #pragma unroll
            for (int dc = 0; dc < D_NOPE_CHUNKS; dc++) {
                #pragma unroll
                for (int nc = 0; nc < S_N_CHUNKS; nc += 2) {
                    int kr = (nc + sub / 2) * 8 + t_in_sub;
                    int kc = dc * 16 + (sub % 2) * 8;
                    uint32_t Kb0, Kb1, Kb2, Kb3;
                    bk::ldmatrix_x4(Kb0, Kb1, Kb2, Kb3,
                        &smem_Kn[mla_bwd_swizzle<D_NOPE>(kr, kc)]);

                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            S_rmem[t][nc][0], S_rmem[t][nc][1],
                            S_rmem[t][nc][2], S_rmem[t][nc][3],
                            Qn_rmem[t][dc][0], Qn_rmem[t][dc][1],
                            Qn_rmem[t][dc][2], Qn_rmem[t][dc][3],
                            Kb0, Kb1,
                            S_rmem[t][nc][0], S_rmem[t][nc][1],
                            S_rmem[t][nc][2], S_rmem[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                            S_rmem[t][nc+1][2], S_rmem[t][nc+1][3],
                            Qn_rmem[t][dc][0], Qn_rmem[t][dc][1],
                            Qn_rmem[t][dc][2], Qn_rmem[t][dc][3],
                            Kb2, Kb3,
                            S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                            S_rmem[t][nc+1][2], S_rmem[t][nc+1][3]);
                    }
                }
            }

            // Rope sub-product
            #pragma unroll
            for (int dc = 0; dc < D_ROPE_CHUNKS; dc++) {
                #pragma unroll
                for (int nc = 0; nc < S_N_CHUNKS; nc += 2) {
                    int kr = (nc + sub / 2) * 8 + t_in_sub;
                    int kc = dc * 16 + (sub % 2) * 8;
                    uint32_t Kb0, Kb1, Kb2, Kb3;
                    bk::ldmatrix_x4(Kb0, Kb1, Kb2, Kb3,
                        &smem_Kr[mla_bwd_swizzle<D_ROPE>(kr, kc)]);

                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            S_rmem[t][nc][0], S_rmem[t][nc][1],
                            S_rmem[t][nc][2], S_rmem[t][nc][3],
                            Qr_rmem[t][dc][0], Qr_rmem[t][dc][1],
                            Qr_rmem[t][dc][2], Qr_rmem[t][dc][3],
                            Kb0, Kb1,
                            S_rmem[t][nc][0], S_rmem[t][nc][1],
                            S_rmem[t][nc][2], S_rmem[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                            S_rmem[t][nc+1][2], S_rmem[t][nc+1][3],
                            Qr_rmem[t][dc][0], Qr_rmem[t][dc][1],
                            Qr_rmem[t][dc][2], Qr_rmem[t][dc][3],
                            Kb2, Kb3,
                            S_rmem[t][nc+1][0], S_rmem[t][nc+1][1],
                            S_rmem[t][nc+1][2], S_rmem[t][nc+1][3]);
                    }
                }
            }
        }

        // Scale, mask, softmax
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            for (int n = 0; n < S_N_CHUNKS; n++) {
                S_rmem[t][n][0] *= scale; S_rmem[t][n][1] *= scale;
                S_rmem[t][n][2] *= scale; S_rmem[t][n][3] *= scale;
            }
        }

        // Causal mask + P = exp(S - L)
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            int gr0 = global_rows[2*t], gr1 = global_rows[2*t+1];
            float L0 = (gr0 < T) ? L_bh[gr0] : 0.f;
            float L1 = (gr1 < T) ? L_bh[gr1] : 0.f;
            float Di0 = (gr0 < T) ? Di_bh[gr0] : 0.f;
            float Di1 = (gr1 < T) ? Di_bh[gr1] : 0.f;

            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                int c0 = kv_start + nc * 8 + (lane_id % 4) * 2;
                int c1 = c0 + 1;
                if (causal) {
                    if (c0 > gr0) S_rmem[t][nc][0] = -FLT_MAX;
                    if (c1 > gr0) S_rmem[t][nc][1] = -FLT_MAX;
                    if (c0 > gr1) S_rmem[t][nc][2] = -FLT_MAX;
                    if (c1 > gr1) S_rmem[t][nc][3] = -FLT_MAX;
                }
                if (c0 >= S) { S_rmem[t][nc][0] = -FLT_MAX; S_rmem[t][nc][2] = -FLT_MAX; }
                if (c1 >= S) { S_rmem[t][nc][1] = -FLT_MAX; S_rmem[t][nc][3] = -FLT_MAX; }

                float P0 = expf(S_rmem[t][nc][0] - L0);
                float P1 = expf(S_rmem[t][nc][1] - L0);
                float P2 = expf(S_rmem[t][nc][2] - L1);
                float P3 = expf(S_rmem[t][nc][3] - L1);

                // dP = dO @ V^T will be computed via MMA next
                // For now, store P in S_rmem for dS computation later
                S_rmem[t][nc][0] = P0;
                S_rmem[t][nc][1] = P1;
                S_rmem[t][nc][2] = P2;
                S_rmem[t][nc][3] = P3;
            }
        }

        // dP = dO @ V^T: [BQ x BKV], each warp computes WARP_Q rows
        float dP_rmem[WARP_Q_TILES][S_N_CHUNKS][4];
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++)
            for (int n = 0; n < S_N_CHUNKS; n++)
                dP_rmem[t][n][0] = dP_rmem[t][n][1] = dP_rmem[t][n][2] = dP_rmem[t][n][3] = 0.f;

        {
            int sub = lane_id / 8;
            int t_in_sub = lane_id % 8;
            int warp_q_off = warp_id * WARP_Q;

            #pragma unroll
            for (int dc = 0; dc < DV_CHUNKS; dc++) {
                // Load dO A-fragments from smem
                uint32_t dOa[WARP_Q_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_Q_TILES; t++) {
                    int dr = warp_q_off + t * 16 + (sub / 2) * 8 + t_in_sub;
                    int dcc = dc * 16 + (sub % 2) * 8;
                    bk::ldmatrix_x4_mma(dOa[t][0], dOa[t][1], dOa[t][2], dOa[t][3],
                        &smem_dO[mla_bwd_swizzle<D_V>(dr, dcc)]);
                }

                // Load V B-fragments (V row-major → V^T col-major → ldmatrix_x4)
                #pragma unroll
                for (int nc = 0; nc < S_N_CHUNKS; nc += 2) {
                    int vr = (nc + sub / 2) * 8 + t_in_sub;
                    int vc = dc * 16 + (sub % 2) * 8;
                    uint32_t Vb0, Vb1, Vb2, Vb3;
                    bk::ldmatrix_x4(Vb0, Vb1, Vb2, Vb3,
                        &smem_V[mla_bwd_swizzle<D_V>(vr, vc)]);

                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            dP_rmem[t][nc][0], dP_rmem[t][nc][1],
                            dP_rmem[t][nc][2], dP_rmem[t][nc][3],
                            dOa[t][0], dOa[t][1], dOa[t][2], dOa[t][3],
                            Vb0, Vb1,
                            dP_rmem[t][nc][0], dP_rmem[t][nc][1],
                            dP_rmem[t][nc][2], dP_rmem[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            dP_rmem[t][nc+1][0], dP_rmem[t][nc+1][1],
                            dP_rmem[t][nc+1][2], dP_rmem[t][nc+1][3],
                            dOa[t][0], dOa[t][1], dOa[t][2], dOa[t][3],
                            Vb2, Vb3,
                            dP_rmem[t][nc+1][0], dP_rmem[t][nc+1][1],
                            dP_rmem[t][nc+1][2], dP_rmem[t][nc+1][3]);
                    }
                }
            }
        }

        // dS = P * (dP - Di) * scale
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            int gr0 = global_rows[2*t], gr1 = global_rows[2*t+1];
            float Di0 = (gr0 < T) ? Di_bh[gr0] : 0.f;
            float Di1 = (gr1 < T) ? Di_bh[gr1] : 0.f;
            #pragma unroll
            for (int nc = 0; nc < S_N_CHUNKS; nc++) {
                dP_rmem[t][nc][0] = S_rmem[t][nc][0] * (dP_rmem[t][nc][0] - Di0) * scale;
                dP_rmem[t][nc][1] = S_rmem[t][nc][1] * (dP_rmem[t][nc][1] - Di0) * scale;
                dP_rmem[t][nc][2] = S_rmem[t][nc][2] * (dP_rmem[t][nc][2] - Di1) * scale;
                dP_rmem[t][nc][3] = S_rmem[t][nc][3] * (dP_rmem[t][nc][3] - Di1) * scale;
            }
        }
        // dP_rmem is now dS_rmem

        // dQ += dS @ K: [BQ x D], each warp computes WARP_Q rows
        // A = dS [BQ x BKV], B = K [BKV x D] (loaded via ldmatrix_x4_trans for A*B)
        {
            #pragma unroll
            for (int kc = 0; kc < P_K_CHUNKS; kc++) {
                int nc0 = 2 * kc, nc1 = 2 * kc + 1;
                uint32_t dSa[WARP_Q_TILES][4];
                #pragma unroll
                for (int t = 0; t < WARP_Q_TILES; t++) {
                    dSa[t][0] = bwd_pack_bf16x2(dP_rmem[t][nc0][0], dP_rmem[t][nc0][1]);
                    dSa[t][1] = bwd_pack_bf16x2(dP_rmem[t][nc0][2], dP_rmem[t][nc0][3]);
                    dSa[t][2] = bwd_pack_bf16x2(dP_rmem[t][nc1][0], dP_rmem[t][nc1][1]);
                    dSa[t][3] = bwd_pack_bf16x2(dP_rmem[t][nc1][2], dP_rmem[t][nc1][3]);
                }

                int sub = lane_id / 8;
                int t_in_sub = lane_id % 8;

                // dQn += dS @ Kn
                uint32_t Knb[DN_N_CHUNKS / 2][4];
                #pragma unroll
                for (int nc = 0; nc < DN_N_CHUNKS; nc += 2) {
                    int kr = kc * 16 + (sub % 2) * 8 + t_in_sub;
                    int kc2 = (nc + sub / 2) * 8;
                    bk::ldmatrix_x4_trans(Knb[nc/2][0], Knb[nc/2][1],
                                          Knb[nc/2][2], Knb[nc/2][3],
                        &smem_Kn[mla_bwd_swizzle<D_NOPE>(kr, kc2)]);
                }
                #pragma unroll
                for (int nc = 0; nc < DN_N_CHUNKS; nc += 2) {
                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            dQn_acc[t][nc][0], dQn_acc[t][nc][1],
                            dQn_acc[t][nc][2], dQn_acc[t][nc][3],
                            dSa[t][0], dSa[t][1], dSa[t][2], dSa[t][3],
                            Knb[nc/2][0], Knb[nc/2][1],
                            dQn_acc[t][nc][0], dQn_acc[t][nc][1],
                            dQn_acc[t][nc][2], dQn_acc[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            dQn_acc[t][nc+1][0], dQn_acc[t][nc+1][1],
                            dQn_acc[t][nc+1][2], dQn_acc[t][nc+1][3],
                            dSa[t][0], dSa[t][1], dSa[t][2], dSa[t][3],
                            Knb[nc/2][2], Knb[nc/2][3],
                            dQn_acc[t][nc+1][0], dQn_acc[t][nc+1][1],
                            dQn_acc[t][nc+1][2], dQn_acc[t][nc+1][3]);
                    }
                }

                // dQr += dS @ Kr
                uint32_t Krb[DR_N_CHUNKS / 2][4];
                #pragma unroll
                for (int nc = 0; nc < DR_N_CHUNKS; nc += 2) {
                    int kr = kc * 16 + (sub % 2) * 8 + t_in_sub;
                    int kc2 = (nc + sub / 2) * 8;
                    bk::ldmatrix_x4_trans(Krb[nc/2][0], Krb[nc/2][1],
                                          Krb[nc/2][2], Krb[nc/2][3],
                        &smem_Kr[mla_bwd_swizzle<D_ROPE>(kr, kc2)]);
                }
                #pragma unroll
                for (int nc = 0; nc < DR_N_CHUNKS; nc += 2) {
                    #pragma unroll
                    for (int t = 0; t < WARP_Q_TILES; t++) {
                        bk::mma_m16n8k16_bf16_nv(
                            dQr_acc[t][nc][0], dQr_acc[t][nc][1],
                            dQr_acc[t][nc][2], dQr_acc[t][nc][3],
                            dSa[t][0], dSa[t][1], dSa[t][2], dSa[t][3],
                            Krb[nc/2][0], Krb[nc/2][1],
                            dQr_acc[t][nc][0], dQr_acc[t][nc][1],
                            dQr_acc[t][nc][2], dQr_acc[t][nc][3]);
                        bk::mma_m16n8k16_bf16_nv(
                            dQr_acc[t][nc+1][0], dQr_acc[t][nc+1][1],
                            dQr_acc[t][nc+1][2], dQr_acc[t][nc+1][3],
                            dSa[t][0], dSa[t][1], dSa[t][2], dSa[t][3],
                            Krb[nc/2][2], Krb[nc/2][3],
                            dQr_acc[t][nc+1][0], dQr_acc[t][nc+1][1],
                            dQr_acc[t][nc+1][2], dQr_acc[t][nc+1][3]);
                    }
                }
            }
        }

        // Load next KV block (after all computation on current KV is done)
        if (kvb + 1 < num_kv_blocks) {
            __syncthreads();  // ensure all threads done reading current KV
            LOAD_KV_TO_SMEM((kvb + 1) * BKV)
            bk::cp_async_commit();
            bk::cp_async_wait<0>();
            __syncthreads();
        }
    } // end KV loop
    #undef LOAD_KV_TO_SMEM

    // Store dQn, dQr to global FP32
    {
        int warp_q_off = warp_id * WARP_Q;
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            int gr0 = global_rows[2*t], gr1 = global_rows[2*t+1];
            #pragma unroll
            for (int nc = 0; nc < DN_N_CHUNKS; nc++) {
                int col = nc * 8 + (lane_id % 4) * 2;
                if (gr0 < T) { dQn_out[bh*T*D_NOPE + gr0*D_NOPE + col] = dQn_acc[t][nc][0];
                               dQn_out[bh*T*D_NOPE + gr0*D_NOPE + col+1] = dQn_acc[t][nc][1]; }
                if (gr1 < T) { dQn_out[bh*T*D_NOPE + gr1*D_NOPE + col] = dQn_acc[t][nc][2];
                               dQn_out[bh*T*D_NOPE + gr1*D_NOPE + col+1] = dQn_acc[t][nc][3]; }
            }
            #pragma unroll
            for (int nc = 0; nc < DR_N_CHUNKS; nc++) {
                int col = nc * 8 + (lane_id % 4) * 2;
                if (gr0 < T) { dQr_out[bh*T*D_ROPE + gr0*D_ROPE + col] = dQr_acc[t][nc][0];
                               dQr_out[bh*T*D_ROPE + gr0*D_ROPE + col+1] = dQr_acc[t][nc][1]; }
                if (gr1 < T) { dQr_out[bh*T*D_ROPE + gr1*D_ROPE + col] = dQr_acc[t][nc][2];
                               dQr_out[bh*T*D_ROPE + gr1*D_ROPE + col+1] = dQr_acc[t][nc][3]; }
            }
        }
    }
}

// ============================================================
// Host launch + PyTorch binding
// ============================================================
namespace bk {

template <int D_NOPE, int D_ROPE, int D_V>
void mla_attn_bwd_launch(
    const __nv_bfloat16 *dO, const __nv_bfloat16 *Q_nope, const __nv_bfloat16 *Q_rope,
    const __nv_bfloat16 *K_nope, const __nv_bfloat16 *K_rope,
    const __nv_bfloat16 *V, const float *L, const float *Di,
    float *dV, float *dKn, float *dKr,
    int BH, int T, int S, float scale, bool causal, cudaStream_t stream)
{
    // dVdK needs BKV divisible by 64 for WARP_KV=16 MMA tiles with 4 warps
    constexpr int BKV = 64;
    constexpr int BQ = 64;
    constexpr int smem_bytes =
        (BKV * D_NOPE + BKV * D_ROPE + BKV * D_V +
         BQ * D_NOPE + BQ * D_ROPE + BQ * D_V) * (int)sizeof(__nv_bfloat16);

    int num_kv_blocks = (S + BKV - 1) / BKV;
    dim3 grid(num_kv_blocks, BH);
    dim3 block(BWD_THREADS);

    auto kernel = mla_bwd_dVdK_kernel<D_NOPE, D_ROPE, D_V, BKV, BQ>;
    if (smem_bytes > 48 * 1024)
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel<<<grid, block, smem_bytes, stream>>>(
        Q_nope, Q_rope, K_nope, K_rope, V, dO, L, Di,
        dV, dKn, dKr, T, S, scale, causal);
}

template void mla_attn_bwd_launch<48, 16, 48>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const float*, const float*, float*, float*, float*,
    int, int, int, float, bool, cudaStream_t);
template void mla_attn_bwd_launch<64, 32, 64>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const float*, const float*, float*, float*, float*,
    int, int, int, float, bool, cudaStream_t);
template void mla_attn_bwd_launch<96, 32, 96>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const float*, const float*, float*, float*, float*,
    int, int, int, float, bool, cudaStream_t);
template void mla_attn_bwd_launch<128, 64, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const float*, const float*, float*, float*, float*,
    int, int, int, float, bool, cudaStream_t);

} // namespace bk

// ============================================================
// PyTorch binding — hybrid: fused dV/dK kernel + ATen dQ
// ============================================================
std::vector<torch::Tensor> mla_attn_backward(
    torch::Tensor dO, torch::Tensor Q_nope, torch::Tensor Q_rope,
    torch::Tensor K_nope, torch::Tensor K_rope,
    torch::Tensor V, torch::Tensor O, torch::Tensor L,
    float scale, bool causal)
{
    int B = Q_nope.size(0), H = Q_nope.size(1), T = Q_nope.size(2), S = K_nope.size(2);
    int d_nope = Q_nope.size(3), d_rope = Q_rope.size(3), d_v = V.size(3);
    int BH = B * H;
    auto f32 = torch::kFloat32;
    auto bf16 = torch::kBFloat16;
    auto stream = at::cuda::getCurrentCUDAStream();

    // Flatten batch*heads
    auto Qn = Q_nope.reshape({BH, T, d_nope});
    auto Qr = Q_rope.reshape({BH, T, d_rope});
    auto Kn = K_nope.reshape({BH, S, d_nope});
    auto Kr = K_rope.reshape({BH, S, d_rope});
    auto Vr = V.reshape({BH, S, d_v});
    auto dO_flat = dO.reshape({BH, T, d_v});
    auto O_flat = O.reshape({BH, T, d_v});
    auto L_flat = L.reshape({BH, T});

    // Step 1: Compute Di = rowsum(dO * O)
    auto Di = torch::empty({BH, T}, L.options());
    {
        int threads = 256;
        dim3 grid((T + threads - 1) / threads, BH);
        #define LAUNCH_DI(DV) \
            if (d_v == DV) mla_compute_Di_kernel<DV><<<grid, threads, 0, stream>>>( \
                reinterpret_cast<const __nv_bfloat16*>(dO_flat.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(O_flat.data_ptr()), \
                Di.data_ptr<float>(), T);
        LAUNCH_DI(48) else LAUNCH_DI(64) else LAUNCH_DI(96) else LAUNCH_DI(128)
        #undef LAUNCH_DI
    }

    // Output tensors — FP32 (kernels accumulate in FP32, convert at store time)
    // We keep FP32 for accumulation precision, convert to BF16 at the very end
    auto dV_f32 = torch::empty({BH, S, d_v}, Qn.options().dtype(f32));
    auto dKn_f32 = torch::empty({BH, S, d_nope}, Qn.options().dtype(f32));
    auto dKr_f32 = torch::empty({BH, S, d_rope}, Qn.options().dtype(f32));

    // Step 2: Fused dV + dK kernel
    #define MLA_BWD_DISPATCH(DN, DR, DV) \
        if (d_nope == DN && d_rope == DR && d_v == DV) { \
            bk::mla_attn_bwd_launch<DN, DR, DV>( \
                reinterpret_cast<const __nv_bfloat16*>(dO_flat.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(Qn.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(Qr.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(Kn.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(Kr.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(Vr.data_ptr()), \
                L_flat.data_ptr<float>(), Di.data_ptr<float>(), \
                dV_f32.data_ptr<float>(), dKn_f32.data_ptr<float>(), dKr_f32.data_ptr<float>(), \
                BH, T, S, scale, causal, stream); \
        }
    MLA_BWD_DISPATCH(48, 16, 48)
    else MLA_BWD_DISPATCH(64, 32, 64)
    else MLA_BWD_DISPATCH(96, 32, 96)
    else MLA_BWD_DISPATCH(128, 64, 128)
    #undef MLA_BWD_DISPATCH

    // Step 3: Fused dQ kernel
    auto dQn_f32 = torch::empty({BH, T, d_nope}, Qn.options().dtype(f32));
    auto dQr_f32 = torch::empty({BH, T, d_rope}, Qn.options().dtype(f32));
    {
        constexpr int BKV = 48;  // Match dVdK tile size
        constexpr int BQ = 64;

        int num_q_blocks = (T + BQ - 1) / BQ;
        dim3 grid(num_q_blocks, BH);
        dim3 block(BWD_THREADS);

        #define MLA_DQ_DISPATCH(DN, DR, DV) \
            if (d_nope == DN && d_rope == DR && d_v == DV) { \
                constexpr int dq_smem = (BKV*(DN+DR+DV) + BQ*DV) * (int)sizeof(__nv_bfloat16); \
                auto kern = mla_bwd_dQ_kernel<DN, DR, DV, BKV, BQ>; \
                if (dq_smem > 48*1024) \
                    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, dq_smem); \
                kern<<<grid, block, dq_smem, stream>>>( \
                    reinterpret_cast<const __nv_bfloat16*>(Qn.data_ptr()), \
                    reinterpret_cast<const __nv_bfloat16*>(Qr.data_ptr()), \
                    reinterpret_cast<const __nv_bfloat16*>(Kn.data_ptr()), \
                    reinterpret_cast<const __nv_bfloat16*>(Kr.data_ptr()), \
                    reinterpret_cast<const __nv_bfloat16*>(Vr.data_ptr()), \
                    reinterpret_cast<const __nv_bfloat16*>(dO_flat.data_ptr()), \
                    L_flat.data_ptr<float>(), Di.data_ptr<float>(), \
                    dQn_f32.data_ptr<float>(), dQr_f32.data_ptr<float>(), \
                    T, S, scale, causal); \
            }
        MLA_DQ_DISPATCH(48, 16, 48)
        else MLA_DQ_DISPATCH(64, 32, 64)
        else MLA_DQ_DISPATCH(96, 32, 96)
        else MLA_DQ_DISPATCH(128, 64, 128)
        #undef MLA_DQ_DISPATCH
    }

    return {dQn_f32.to(bf16).reshape({B,H,T,d_nope}),
            dQr_f32.to(bf16).reshape({B,H,T,d_rope}),
            dKn_f32.to(bf16).reshape({B,H,S,d_nope}),
            dKr_f32.to(bf16).reshape({B,H,S,d_rope}),
            dV_f32.to(bf16).reshape({B,H,S,d_v})};
}
