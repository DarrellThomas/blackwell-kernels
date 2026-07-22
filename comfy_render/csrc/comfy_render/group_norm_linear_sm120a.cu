// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// GroupNorm + Linear for sm_120a (RTX 5090).
//
// Strategy: fast custom GroupNorm kernel + cuBLAS for the GEMM.
// The prior approach (hand-rolled MMA GEMM with on-the-fly normalization)
// was 2-5x slower than cuBLAS for C>=640. The ~5us fusion saving from
// eliminating the intermediate write-read doesn't justify that regression.
//
// GroupNorm kernel: two variants with adaptive dispatch:
//   1. No-cache (M>=512): stats from global, normalize from L2. One __syncthreads.
//      Better for large M where L2 is warm from many concurrent blocks.
//   2. Cached (M<512): load row to smem, stats+normalize from smem. Two __syncthreads.
//      Better for small M where L2 is cold and the extra global read hurts.
//
// Linear: torch::linear (cuBLAS). Unbeatable for compute-bound GEMMs.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

// ════════════════════════════════════════════════════════════════════════
// Normalize + write helper (shared between both variants)
// ════════════════════════════════════════════════════════════════════════

template <int BLOCK_SIZE>
__device__ __forceinline__ void normalize_and_write(
    const int4* __restrict__ x_src,    // source for X data (smem or global)
    int4* __restrict__ y_row,
    const int4* __restrict__ gamma_v,
    const int4* __restrict__ beta_v,
    const float* __restrict__ stats,
    int Cv, int group_size, int groups
) {
    const int tid = threadIdx.x;
    for (int i = tid; i < Cv; i += BLOCK_SIZE) {
        int elem_base = i * 8;
        int g0 = elem_base / group_size;
        int boundary = (g0 + 1) * group_size - elem_base;

        float m0 = stats[g0 * 2],     s0 = stats[g0 * 2 + 1];
        float m1 = m0, s1 = s0;
        if (boundary < 8 && g0 + 1 < groups) {
            m1 = stats[(g0 + 1) * 2];
            s1 = stats[(g0 + 1) * 2 + 1];
        }

        int4 xv = x_src[i], gv = gamma_v[i], bv = beta_v[i];
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
        const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&bv);
        int4 yv;
        __nv_bfloat162* yp = reinterpret_cast<__nv_bfloat162*>(&yv);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int e = j * 2;
            float ml = (e     < boundary) ? m0 : m1;
            float sl = (e     < boundary) ? s0 : s1;
            float mh = (e + 1 < boundary) ? m0 : m1;
            float sh = (e + 1 < boundary) ? s0 : s1;

            float x_lo = __bfloat162float(xp[j].x);
            float x_hi = __bfloat162float(xp[j].y);
            float ga_lo = __bfloat162float(gp[j].x);
            float ga_hi = __bfloat162float(gp[j].y);
            float be_lo = __bfloat162float(bp[j].x);
            float be_hi = __bfloat162float(bp[j].y);

            yp[j] = __floats2bfloat162_rn(
                ga_lo * (x_lo - ml) * sl + be_lo,
                ga_hi * (x_hi - mh) * sh + be_hi);
        }

        y_row[i] = yv;
    }
}

// ════════════════════════════════════════════════════════════════════════
// Per-group reduction helper (shared between both variants)
// ════════════════════════════════════════════════════════════════════════

template <int NUM_WARPS>
__device__ __forceinline__ void compute_group_stats(
    const __nv_bfloat16* __restrict__ data,  // source (smem or global row ptr)
    float* __restrict__ stats,
    int group_size, int groups, float eps
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int gpw = (groups + NUM_WARPS - 1) / NUM_WARPS;

    for (int gw = 0; gw < gpw; gw++) {
        int g = warp_id * gpw + gw;
        if (g >= groups) break;

        int base = g * group_size;
        float sum = 0.0f, sum_sq = 0.0f;

        for (int i = lane; i < group_size; i += 32) {
            float v = __bfloat162float(data[base + i]);
            sum += v;
            sum_sq += v * v;
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, o);
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
        }

        if (lane == 0) {
            float mean = sum / (float)group_size;
            float var = sum_sq / (float)group_size - mean * mean;
            stats[g * 2]     = mean;
            stats[g * 2 + 1] = rsqrtf(var + eps);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Variant 1: No-cache (for M >= 512)
// Stats from global memory, normalize from global (L2 hit).
// Only one __syncthreads. Minimal smem. Higher occupancy.
// ════════════════════════════════════════════════════════════════════════

template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
group_norm_fwd_nocache(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ Y,
    const int C, const int groups, const float eps
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const int row = blockIdx.x;
    const int group_size = C / groups;

    extern __shared__ char smem[];
    float* stats = reinterpret_cast<float*>(smem);

    const __nv_bfloat16* row_ptr = X + (size_t)row * C;

    compute_group_stats<NUM_WARPS>(row_ptr, stats, group_size, groups, eps);
    __syncthreads();

    const int Cv = C / 8;
    normalize_and_write<BLOCK_SIZE>(
        reinterpret_cast<const int4*>(row_ptr),
        reinterpret_cast<int4*>(Y + (size_t)row * C),
        reinterpret_cast<const int4*>(gamma),
        reinterpret_cast<const int4*>(beta),
        stats, Cv, group_size, groups);
}

// ════════════════════════════════════════════════════════════════════════
// Variant 2: Cached (for M < 512)
// Load row to smem, stats+normalize from smem. Two __syncthreads.
// Better when L2 is cold (few concurrent blocks).
// ════════════════════════════════════════════════════════════════════════

template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
group_norm_fwd_cached(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ Y,
    const int C, const int groups, const float eps
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int group_size = C / groups;

    extern __shared__ char smem[];
    __nv_bfloat16* x_cache = reinterpret_cast<__nv_bfloat16*>(smem);
    float* stats = reinterpret_cast<float*>(smem + C * sizeof(__nv_bfloat16));

    // Load row to shared memory (128-bit vectorized)
    const int Cv = C / 8;
    const int4* x_row = reinterpret_cast<const int4*>(X + (size_t)row * C);
    int4* cache = reinterpret_cast<int4*>(x_cache);
    for (int i = tid; i < Cv; i += BLOCK_SIZE)
        cache[i] = x_row[i];
    __syncthreads();

    compute_group_stats<NUM_WARPS>(x_cache, stats, group_size, groups, eps);
    __syncthreads();

    normalize_and_write<BLOCK_SIZE>(
        cache,
        reinterpret_cast<int4*>(Y + (size_t)row * C),
        reinterpret_cast<const int4*>(gamma),
        reinterpret_cast<const int4*>(beta),
        stats, Cv, group_size, groups);
}

} // namespace

// ════════════════════════════════════════════════════════════════════════
// Torch entry points
// ════════════════════════════════════════════════════════════════════════

torch::Tensor group_norm_forward(
    torch::Tensor X,      // [M, C] BF16
    torch::Tensor gamma,  // [C]
    torch::Tensor beta,   // [C]
    int64_t groups,
    double eps
) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(), "X must be contiguous CUDA");
    TORCH_CHECK(X.dtype() == torch::kBFloat16, "X must be BF16");
    TORCH_CHECK(X.dim() == 2, "X must be 2D");

    int M = X.size(0), C = X.size(1);
    TORCH_CHECK(C % groups == 0, "C must be divisible by groups");
    TORCH_CHECK(C % 8 == 0, "C must be divisible by 8 for vectorized loads");

    auto gamma_bf = (gamma.dtype() == torch::kBFloat16) ? gamma.contiguous()
                    : gamma.to(torch::kBFloat16).contiguous();
    auto beta_bf = (beta.dtype() == torch::kBFloat16) ? beta.contiguous()
                   : beta.to(torch::kBFloat16).contiguous();
    TORCH_CHECK(gamma_bf.size(0) == C && beta_bf.size(0) == C, "gamma/beta size");

    auto Y = torch::empty_like(X);
    constexpr int BLOCK = 256;

    auto x_ptr = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>());
    auto g_ptr = reinterpret_cast<const __nv_bfloat16*>(gamma_bf.data_ptr<at::BFloat16>());
    auto b_ptr = reinterpret_cast<const __nv_bfloat16*>(beta_bf.data_ptr<at::BFloat16>());
    auto y_ptr = reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>());

    if (M >= 512) {
        // No-cache: stats from global, L2 for normalize. One sync, minimal smem.
        int smem = (int)groups * 2 * (int)sizeof(float);
        group_norm_fwd_nocache<BLOCK><<<M, BLOCK, smem>>>(
            x_ptr, g_ptr, b_ptr, y_ptr, C, (int)groups, (float)eps);
    } else {
        // Cached: row in smem. Two syncs, more smem, but avoids cold L2 re-reads.
        int smem = C * (int)sizeof(__nv_bfloat16) + (int)groups * 2 * (int)sizeof(float);
        group_norm_fwd_cached<BLOCK><<<M, BLOCK, smem>>>(
            x_ptr, g_ptr, b_ptr, y_ptr, C, (int)groups, (float)eps);
    }

    return Y;
}

torch::Tensor fused_group_norm_linear_forward(
    torch::Tensor X,            // [M, C_in] BF16
    torch::Tensor weight,       // [C_out, C_in] BF16
    torch::Tensor gamma,        // [C_in]
    torch::Tensor beta,         // [C_in]
    torch::Tensor linear_bias,  // [C_out]
    int64_t groups
) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous(), "X must be contiguous CUDA");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "weight contiguous CUDA");
    TORCH_CHECK(X.dtype() == torch::kBFloat16, "X must be BF16");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be BF16");
    TORCH_CHECK(X.dim() == 2 && weight.dim() == 2, "X/weight must be 2D");

    int M = X.size(0), C_in = X.size(1), C_out = weight.size(0);
    TORCH_CHECK(weight.size(1) == C_in, "weight K dim must match C_in");
    TORCH_CHECK(C_in % groups == 0, "C_in must be divisible by groups");
    TORCH_CHECK(C_in % 8 == 0, "C_in must be divisible by 8");

    // Step 1: Custom GroupNorm
    auto normed = group_norm_forward(X, gamma, beta, groups, 1e-5);

    // Step 2: Linear via cuBLAS (torch::linear uses the most optimized path)
    auto bias_bf = (linear_bias.dtype() == torch::kBFloat16) ? linear_bias.contiguous()
                   : linear_bias.to(torch::kBFloat16).contiguous();
    auto Y = torch::linear(normed, weight, bias_bf);
    return Y;
}
