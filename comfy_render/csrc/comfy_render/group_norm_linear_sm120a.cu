// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Fused GroupNorm + Linear for sm_120a (RTX 5090).
// Phase 1: GroupNorm statistics (mean/invstd per group per row).
// Phase 2: Tiled GEMM with on-the-fly normalization via MMA m16n8k16.
//
// Eliminates the global memory round-trip for the normalized intermediate:
// instead of GroupNorm → write → read → Linear, the normalized values
// flow directly from registers into the MMA pipeline.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cmath>

#include "mma_sm120.cuh"
#include "ldmatrix.cuh"
#include "cp_async.cuh"
#include "swizzle.cuh"

namespace {

constexpr int WARP_SIZE = 32;
constexpr float GN_EPS = 1e-5f;

// ════════════════════════════════════════════════════════════════════════
// Phase 1: GroupNorm statistics kernel
// One block per row. Computes mean and inv_std for each of `groups` groups.
// ════════════════════════════════════════════════════════════════════════

__global__ void __launch_bounds__(128)
group_norm_stats_kernel(
    const __nv_bfloat16* __restrict__ X,   // [M, C]
    float* __restrict__ mean_out,           // [M, groups]
    float* __restrict__ invstd_out,         // [M, groups]
    int M, int C, int groups
) {
    int row = blockIdx.x;
    if (row >= M) return;

    int group_size = C / groups;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    constexpr int NUM_WARPS = 4;

    __shared__ float warp_buf[2 * NUM_WARPS];

    const __nv_bfloat16* row_ptr = X + (size_t)row * C;

    for (int g = 0; g < groups; g++) {
        float sum = 0.0f, sum_sq = 0.0f;
        int base = g * group_size;

        for (int i = tid; i < group_size; i += blockDim.x) {
            float v = __bfloat162float(row_ptr[base + i]);
            sum += v;
            sum_sq += v * v;
        }

        // Warp-level reduce
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }

        // Cross-warp reduce via shared memory
        if (lane == 0) {
            warp_buf[warp_id] = sum;
            warp_buf[warp_id + NUM_WARPS] = sum_sq;
        }
        __syncthreads();

        if (tid == 0) {
            float s = 0.0f, sq = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                s += warp_buf[w];
                sq += warp_buf[w + NUM_WARPS];
            }
            float m = s / group_size;
            float var = sq / group_size - m * m;
            mean_out[row * groups + g] = m;
            invstd_out[row * groups + g] = rsqrtf(var + GN_EPS);
        }
        __syncthreads();
    }
}

// ════════════════════════════════════════════════════════════════════════
// Phase 2: Fused normalize + GEMM (v3: large tiles)
// Y[M, C_out] = normalize(X) @ W^T + linear_bias
//
// TILE_M=128, TILE_N=64, K_STEP=32 → AI=42.7 FLOPs/byte
// 4 warps, each handles 32M x 64N (2 MMA M-tiles x 8 N-tiles)
// Double-buffered B, vectorized A loads, non-volatile MMA
// ════════════════════════════════════════════════════════════════════════

constexpr int TILE_M = 128;
constexpr int TILE_N = 64;
constexpr int K_STEP = 32;
constexpr int N_MQ = 2;   // M-tiles per warp (32 rows / 16 per MMA)
constexpr int N_NT = 8;   // N-tiles per warp (64 cols / 8 per MMA)

__global__ void __launch_bounds__(128)
fused_norm_linear_kernel(
    const __nv_bfloat16* __restrict__ X,       // [M, C_in]
    const __nv_bfloat16* __restrict__ W,       // [C_out, C_in]
    const float* __restrict__ gn_mean,          // [M, groups]
    const float* __restrict__ gn_invstd,        // [M, groups]
    const __nv_bfloat16* __restrict__ gamma,   // [C_in] BF16
    const __nv_bfloat16* __restrict__ beta,    // [C_in] BF16
    const __nv_bfloat16* __restrict__ linear_bias, // [C_out] BF16
    __nv_bfloat16* __restrict__ Y,             // [M, C_out]
    int M, int C_in, int C_out, int groups
) {
    int bm = blockIdx.x;
    int bn = blockIdx.y;
    int m_start = bm * TILE_M;
    int n_start = bn * TILE_N;
    if (m_start >= M) return;

    constexpr int BLOCK = 128;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int group_size = C_in / groups;

    // Shared memory: A[128*32] + B0[64*32] + B1[64*32]
    extern __shared__ char smem_raw[];
    __nv_bfloat16* a_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* b_smem[2];
    b_smem[0] = a_smem + TILE_M * K_STEP;
    b_smem[1] = b_smem[0] + TILE_N * K_STEP;

    // Accumulators: 4 warps x 2 M-tiles x 8 N-tiles x 4 MMA outputs
    float acc[N_MQ * N_NT][4];
    #pragma unroll
    for (int i = 0; i < N_MQ * N_NT; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    int b_phase = 0;

    // ── Prologue: prefetch B[0] ──────────────────────────────────────
    {
        constexpr int B_CHUNKS = TILE_N * (K_STEP / 8);
        for (int idx = tid; idx < B_CHUNKS; idx += BLOCK) {
            int row = idx / (K_STEP / 8);
            int col8 = (idx % (K_STEP / 8)) * 8;
            int gn = n_start + row;
            bool valid = (gn < C_out);
            bk::cp_async_128_zfill(
                &b_smem[0][bk::swizzle_idx<K_STEP>(row, col8)],
                &W[(size_t)gn * C_in + col8], valid);
        }
        bk::cp_async_commit();
    }

    // ── K dimension loop ──────────────────────────────────────────────
    int num_k_steps = (C_in + K_STEP - 1) / K_STEP;
    for (int ki = 0; ki < num_k_steps; ki++) {
        int k = ki * K_STEP;
        bool has_next = (ki + 1 < num_k_steps);

        // Load A tile [128, 32]: vectorized 128-bit loads, normalize
        constexpr int A_VECS = TILE_M * (K_STEP / 8); // 128*4 = 512
        for (int idx = tid; idx < A_VECS; idx += BLOCK) {
            int row = idx / (K_STEP / 8);
            int col8 = (idx % (K_STEP / 8)) * 8;
            int gm = m_start + row;
            int gk = k + col8;

            if (gm < M) {
                uint4 chunk = *reinterpret_cast<const uint4*>(&X[(size_t)gm * C_in + gk]);
                __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&chunk);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float x = __bfloat162float(vals[i]);
                    int g = (gk + i) / group_size;
                    float m = gn_mean[gm * groups + g];
                    float s = gn_invstd[gm * groups + g];
                    float gam = __bfloat162float(gamma[gk + i]);
                    float bet = __bfloat162float(beta[gk + i]);
                    float normed = gam * (x - m) * s + bet;
                    a_smem[bk::swizzle_idx<K_STEP>(row, col8 + i)] = __float2bfloat16(normed);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    a_smem[bk::swizzle_idx<K_STEP>(row, col8 + i)] = __float2bfloat16(0.0f);
            }
        }

        // Wait for B[k]
        bk::cp_async_wait_all();
        __syncthreads();

        // Prefetch B[k+K_STEP]
        if (has_next) {
            int k_next = (ki + 1) * K_STEP;
            constexpr int B_CHUNKS = TILE_N * (K_STEP / 8);
            for (int idx = tid; idx < B_CHUNKS; idx += BLOCK) {
                int row = idx / (K_STEP / 8);
                int col8 = (idx % (K_STEP / 8)) * 8;
                int gn = n_start + row;
                bool valid = (gn < C_out);
                bk::cp_async_128_zfill(
                    &b_smem[1 - b_phase][bk::swizzle_idx<K_STEP>(row, col8)],
                    &W[(size_t)gn * C_in + k_next + col8], valid);
            }
            bk::cp_async_commit();
        }

        // ── MMA: 2 K-sub-tiles x 2 M-tiles x 8 N-tiles ─────────────
        __nv_bfloat16* b_cur = b_smem[b_phase];
        int warp_m_base = warp_id * 32; // 32 rows per warp

        #pragma unroll
        for (int ks = 0; ks < K_STEP / 16; ks++) {
            // Load A fragments for both M-tiles
            uint32_t a_frag[N_MQ][4];
            #pragma unroll
            for (int mq = 0; mq < N_MQ; mq++) {
                int sub = lane / 8, sub_row = lane % 8;
                int row = warp_m_base + mq * 16 + (sub < 2 ? sub_row : 8 + sub_row);
                int col = ks * 16 + (sub % 2) * 8;
                bk::ldmatrix_x4_mma(a_frag[mq][0], a_frag[mq][1],
                    a_frag[mq][2], a_frag[mq][3],
                    &a_smem[bk::swizzle_idx<K_STEP>(row, col)]);
            }

            #pragma unroll
            for (int nt = 0; nt < N_NT; nt++) {
                uint32_t b_frag[2];
                {
                    int sub_row = lane % 8;
                    int k_off = ks * 16 + ((lane / 8) % 2) * 8;
                    int row = nt * 8 + sub_row;
                    bk::ldmatrix_x2(b_frag[0], b_frag[1],
                        &b_cur[bk::swizzle_idx<K_STEP>(row, k_off)]);
                }

                #pragma unroll
                for (int mq = 0; mq < N_MQ; mq++) {
                    int ai = mq * N_NT + nt;
                    bk::mma_m16n8k16_bf16_nv(
                        acc[ai][0], acc[ai][1], acc[ai][2], acc[ai][3],
                        a_frag[mq][0], a_frag[mq][1],
                        a_frag[mq][2], a_frag[mq][3],
                        b_frag[0], b_frag[1],
                        acc[ai][0], acc[ai][1], acc[ai][2], acc[ai][3]);
                }
            }
        }

        __syncthreads();
        b_phase ^= 1;
    }

    // ── Store output with bias ────────────────────────────────────────
    int mma_row_offset = lane / 4;       // 0-7 within 16-row MMA tile
    int mma_col = (lane % 4) * 2;

    #pragma unroll
    for (int mq = 0; mq < N_MQ; mq++) {
        int gm_a = m_start + warp_id * 32 + mq * 16 + mma_row_offset;
        int gm_b = gm_a + 8;

        #pragma unroll
        for (int nt = 0; nt < N_NT; nt++) {
            int gc = n_start + nt * 8 + mma_col;
            if (gc + 1 >= C_out) continue;
            float bias0 = __bfloat162float(linear_bias[gc]);
            float bias1 = __bfloat162float(linear_bias[gc + 1]);
            int ai = mq * N_NT + nt;

            if (gm_a < M) {
                Y[(size_t)gm_a * C_out + gc]     = __float2bfloat16(acc[ai][0] + bias0);
                Y[(size_t)gm_a * C_out + gc + 1] = __float2bfloat16(acc[ai][1] + bias1);
            }
            if (gm_b < M) {
                Y[(size_t)gm_b * C_out + gc]     = __float2bfloat16(acc[ai][2] + bias0);
                Y[(size_t)gm_b * C_out + gc + 1] = __float2bfloat16(acc[ai][3] + bias1);
            }
        }
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════════
// Torch entry point
// ════════════════════════════════════════════════════════════════════════

torch::Tensor fused_group_norm_linear_forward(
    torch::Tensor X,           // [M, C_in] BF16
    torch::Tensor weight,      // [C_out, C_in] BF16
    torch::Tensor gamma,       // [C_in] any float
    torch::Tensor beta,        // [C_in] any float
    torch::Tensor linear_bias, // [C_out] any float
    int64_t groups
) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(), "X must be contiguous CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "weight must be contiguous CUDA");
    TORCH_CHECK(X.dtype() == torch::kBFloat16, "X must be BF16");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be BF16");
    TORCH_CHECK(X.dim() == 2 && weight.dim() == 2, "X and weight must be 2D");

    int M = X.size(0);
    int C_in = X.size(1);
    int C_out = weight.size(0);
    TORCH_CHECK(weight.size(1) == C_in, "weight K dim must match X channels");
    TORCH_CHECK(C_in % groups == 0, "C_in must be divisible by groups");

    // Dispatch: custom fused GEMM for large-M/small-C (bandwidth-limited),
    //           cuBLAS path for compute-heavy cases
    bool use_fused = (M >= 2048) && (C_in <= 320) && (C_out <= 320);

    if (use_fused) {
        // ── Fused path: custom stats + custom normalize-GEMM ──
        // Accept BF16 gamma/beta/bias directly (convert per-element in kernel)
        auto gamma_bf = gamma.to(X.dtype()).contiguous();
        auto beta_bf = beta.to(X.dtype()).contiguous();
        auto bias_bf = linear_bias.to(X.dtype()).contiguous();

        auto opts_f32 = X.options().dtype(torch::kFloat32);
        auto mean = torch::empty({M, groups}, opts_f32);
        auto invstd = torch::empty({M, groups}, opts_f32);

        group_norm_stats_kernel<<<M, 128>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
            mean.data_ptr<float>(),
            invstd.data_ptr<float>(),
            M, C_in, (int)groups);

        auto Y = torch::empty({M, C_out}, X.options());

        dim3 grid((M + TILE_M - 1) / TILE_M, (C_out + TILE_N - 1) / TILE_N);
        int smem_bytes = (TILE_M + 2 * TILE_N) * K_STEP * (int)sizeof(__nv_bfloat16);

        fused_norm_linear_kernel<<<grid, 128, smem_bytes>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            mean.data_ptr<float>(),
            invstd.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(gamma_bf.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(beta_bf.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(bias_bf.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
            M, C_in, C_out, (int)groups);

        return Y;
    } else {
        // ── cuBLAS path: C++ GroupNorm + torch::mm (saves Python dispatch) ──
        // Convert to X's dtype only if needed (avoid kernel launches for no-ops)
        auto gamma_x = (gamma.dtype() == X.dtype()) ? gamma : gamma.to(X.dtype());
        auto beta_x = (beta.dtype() == X.dtype()) ? beta : beta.to(X.dtype());
        auto bias_x = (linear_bias.dtype() == X.dtype()) ? linear_bias
                       : linear_bias.to(X.dtype());

        // GroupNorm (PyTorch's optimized kernel, zero Python dispatch)
        auto x_3d = X.unsqueeze(2);
        auto gn_out = torch::group_norm(x_3d, groups, gamma_x, beta_x).squeeze(2);

        // GEMM via cuBLAS
        auto Y = torch::addmm(bias_x.unsqueeze(0), gn_out, weight.t());
        return Y;
    }
}
