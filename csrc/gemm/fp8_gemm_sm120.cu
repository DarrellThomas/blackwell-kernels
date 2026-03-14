// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// FP8 GEMM kernel for sm_120 (RTX 5090)
// Uses mma.sync.aligned.m16n8k32 tensor core instructions (2x throughput vs BF16).
//
// Architecture:
//   Global Memory --[cp.async]--> Shared Memory (BF16) --[ldmatrix_x4_mma]--> Registers (BF16)
//       --[cvt.e4m3x2.f32]--> FP8 Registers --[mma.sync m16n8k32]--> FP32 accumulators
//       --> BF16 output
//
// Computes C = A * B where A is [M, K], B is [K, N], C is [M, N].
// Inputs are BF16, converted to FP8 e4m3 on the fly for 2x MMA throughput.
// FP32 accumulators prevent precision loss during accumulation.
//
// Two tile configs dispatched by problem size:
//   Small: BLOCK_M=64,  BLOCK_N=64,  BLOCK_K=64, 4 warps — high occupancy for small grids
//   Large: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 8 warps — 2x data reuse for L2-bound sizes
//
// CTA swizzle groups concurrent blocks to share B columns for L2 reuse.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "mma_sm120.cuh"
#include "ldmatrix.cuh"
#include "cp_async.cuh"
#include "swizzle.cuh"
#include "fp8_convert.cuh"

// ============================================================
// FP8 compute tile: load BF16 from smem, convert to FP8, execute MMA
// (Shared between both tile configs — fully parametric)
// ============================================================

template <int M_TILES, int N_TILES, int K_TILES, int BLOCK_K_PARAM, int BLOCK_N_PARAM>
__device__ __forceinline__ void compute_fp8_tile(
    float accum[][N_TILES][4],
    const __nv_bfloat16 *A_smem,
    const __nv_bfloat16 *B_smem,
    int warp_m_off,
    int warp_n_off,
    int sub, int t_in_sub, int lane_id)
{
    int b_row_base = (lane_id % 8) + ((lane_id / 8) % 2) * 8;

    #pragma unroll
    for (int kc = 0; kc < K_TILES; kc++) {
        // Load A fragments (BF16) for both k-halves and convert to FP8
        uint32_t a_fp8[M_TILES][4];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; mt++) {
            int smem_row = warp_m_off + mt * 16 + (sub / 2) * 8 + t_in_sub;

            int smem_col_k0 = kc * 32 + (sub % 2) * 8;
            const void *addr_k0 = &A_smem[bk::swizzle_idx<BLOCK_K_PARAM>(smem_row, smem_col_k0)];
            uint32_t bf16_k0[4];
            bk::ldmatrix_x4_mma(bf16_k0[0], bf16_k0[1], bf16_k0[2], bf16_k0[3], addr_k0);

            int smem_col_k1 = kc * 32 + 16 + (sub % 2) * 8;
            const void *addr_k1 = &A_smem[bk::swizzle_idx<BLOCK_K_PARAM>(smem_row, smem_col_k1)];
            uint32_t bf16_k1[4];
            bk::ldmatrix_x4_mma(bf16_k1[0], bf16_k1[1], bf16_k1[2], bf16_k1[3], addr_k1);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_fp8[mt][i] = bk::bf16x2_pair_to_e4m3x4(bf16_k0[i], bf16_k1[i]);
            }
        }

        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            int b_col = warp_n_off + nt * 8;

            int b_row_k0 = b_row_base + kc * 32;
            const void *addr_b_k0 = &B_smem[bk::swizzle_idx<BLOCK_N_PARAM>(b_row_k0, b_col)];
            uint32_t b_bf16_k0_0, b_bf16_k0_1;
            bk::ldmatrix_x2_trans(b_bf16_k0_0, b_bf16_k0_1, addr_b_k0);

            int b_row_k1 = b_row_base + kc * 32 + 16;
            const void *addr_b_k1 = &B_smem[bk::swizzle_idx<BLOCK_N_PARAM>(b_row_k1, b_col)];
            uint32_t b_bf16_k1_0, b_bf16_k1_1;
            bk::ldmatrix_x2_trans(b_bf16_k1_0, b_bf16_k1_1, addr_b_k1);

            uint32_t b_fp8_0 = bk::bf16x2_pair_to_e4m3x4(b_bf16_k0_0, b_bf16_k1_0);
            uint32_t b_fp8_1 = bk::bf16x2_pair_to_e4m3x4(b_bf16_k0_1, b_bf16_k1_1);

            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                bk::mma_m16n8k32_e4m3_nv(
                    accum[mt][nt][0], accum[mt][nt][1],
                    accum[mt][nt][2], accum[mt][nt][3],
                    a_fp8[mt][0], a_fp8[mt][1], a_fp8[mt][2], a_fp8[mt][3],
                    b_fp8_0, b_fp8_1,
                    accum[mt][nt][0], accum[mt][nt][1],
                    accum[mt][nt][2], accum[mt][nt][3]);
            }
        }
    }
}

// ============================================================
// Generic FP8 GEMM kernel body — parametric on tile config
// ============================================================

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS, int WARPS_M, int WARPS_N>
__device__ void fp8_gemm_body(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    __nv_bfloat16 *__restrict__ C,
    int M, int N, int K,
    int grid_m, int grid_n)
{
    constexpr int WARP_SIZE = 32;
    constexpr int THREADS = NUM_WARPS * WARP_SIZE;
    constexpr int WARP_M = BLOCK_M / WARPS_M;
    constexpr int WARP_N = BLOCK_N / WARPS_N;
    constexpr int MMA_M_TILES = WARP_M / 16;
    constexpr int MMA_N_TILES = WARP_N / 8;
    constexpr int MMA_K_TILES = BLOCK_K / 32;
    constexpr int A_SMEM_ELEMS = BLOCK_M * BLOCK_K;
    constexpr int B_SMEM_ELEMS = BLOCK_K * BLOCK_N;

    // CTA swizzle for L2 locality
    int sw = min(8, grid_n);
    int linear_bid = blockIdx.x + blockIdx.y * gridDim.x;
    int tiles_per_super = sw * grid_m;
    int super_id = linear_bid / tiles_per_super;
    int within = linear_bid % tiles_per_super;
    int bn_idx = super_id * sw + within % sw;
    int bm_idx = within / sw;

    const int bm = bm_idx * BLOCK_M;
    const int bn = bn_idx * BLOCK_N;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_A = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *smem_B = smem_A + 2 * A_SMEM_ELEMS;

    float C_rmem[MMA_M_TILES][MMA_N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < MMA_M_TILES; mt++)
        #pragma unroll
        for (int nt = 0; nt < MMA_N_TILES; nt++) {
            C_rmem[mt][nt][0] = 0.0f; C_rmem[mt][nt][1] = 0.0f;
            C_rmem[mt][nt][2] = 0.0f; C_rmem[mt][nt][3] = 0.0f;
        }

    int num_k_blocks = K / BLOCK_K;

    auto load_A_tile = [&](int k_start, __nv_bfloat16 *dst) {
        constexpr int A_CHUNKS_PER_ROW = BLOCK_K / 8;
        constexpr int A_TOTAL_CHUNKS = BLOCK_M * A_CHUNKS_PER_ROW;
        #pragma unroll
        for (int i = tid; i < A_TOTAL_CHUNKS; i += THREADS) {
            int row = i / A_CHUNKS_PER_ROW;
            int col = (i % A_CHUNKS_PER_ROW) * 8;
            bk::cp_async_128(&dst[bk::swizzle_idx<BLOCK_K>(row, col)],
                             &A[(bm + row) * K + k_start + col]);
        }
    };

    auto load_B_tile = [&](int k_start, __nv_bfloat16 *dst) {
        constexpr int B_CHUNKS_PER_ROW = BLOCK_N / 8;
        constexpr int B_TOTAL_CHUNKS = BLOCK_K * B_CHUNKS_PER_ROW;
        #pragma unroll
        for (int i = tid; i < B_TOTAL_CHUNKS; i += THREADS) {
            int row = i / B_CHUNKS_PER_ROW;
            int col = (i % B_CHUNKS_PER_ROW) * 8;
            bk::cp_async_128(&dst[bk::swizzle_idx<BLOCK_N>(row, col)],
                             &B[(k_start + row) * N + bn + col]);
        }
    };

    // Prologue
    load_A_tile(0, smem_A);
    load_B_tile(0, smem_B);
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    int sub = lane_id / 8;
    int t_in_sub = lane_id % 8;
    int warp_m_id = warp_id / WARPS_N;
    int warp_n_id = warp_id % WARPS_N;
    int warp_m_off = warp_m_id * WARP_M;
    int warp_n_off = warp_n_id * WARP_N;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int cur = kb & 1;
        __nv_bfloat16 *A_cur = smem_A + cur * A_SMEM_ELEMS;
        __nv_bfloat16 *B_cur = smem_B + cur * B_SMEM_ELEMS;

        if (kb + 1 < num_k_blocks) {
            int nxt = 1 - cur;
            load_A_tile((kb + 1) * BLOCK_K, smem_A + nxt * A_SMEM_ELEMS);
            load_B_tile((kb + 1) * BLOCK_K, smem_B + nxt * B_SMEM_ELEMS);
        }
        bk::cp_async_commit();

        compute_fp8_tile<MMA_M_TILES, MMA_N_TILES, MMA_K_TILES,
                         BLOCK_K, BLOCK_N>(
            C_rmem, A_cur, B_cur, warp_m_off, warp_n_off,
            sub, t_in_sub, lane_id);

        bk::cp_async_wait<0>();
        __syncthreads();
    }

    // Epilogue: store C
    #pragma unroll
    for (int mt = 0; mt < MMA_M_TILES; mt++) {
        int row0 = bm + warp_m_off + mt * 16 + (lane_id / 4);
        int row1 = row0 + 8;
        #pragma unroll
        for (int nt = 0; nt < MMA_N_TILES; nt++) {
            int col0 = bn + warp_n_off + nt * 8 + (lane_id % 4) * 2;

            __nv_bfloat162 packed0 = __floats2bfloat162_rn(
                C_rmem[mt][nt][0], C_rmem[mt][nt][1]);
            *reinterpret_cast<uint32_t*>(&C[row0 * N + col0]) =
                *reinterpret_cast<uint32_t*>(&packed0);

            __nv_bfloat162 packed1 = __floats2bfloat162_rn(
                C_rmem[mt][nt][2], C_rmem[mt][nt][3]);
            *reinterpret_cast<uint32_t*>(&C[row1 * N + col0]) =
                *reinterpret_cast<uint32_t*>(&packed1);
        }
    }
}

// ============================================================
// Two kernel instantiations with appropriate launch_bounds
// ============================================================

// Small config: 64×64, 4 warps, 6 blocks/SM for high occupancy
__global__ void __launch_bounds__(128, 6)
fp8_gemm_kernel_64(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    __nv_bfloat16 *__restrict__ C,
    int M, int N, int K, int grid_m, int grid_n)
{
    fp8_gemm_body<64, 64, 64, 4, 2, 2>(A, B, C, M, N, K, grid_m, grid_n);
}

// Large config: 128×128, 8 warps, fewer blocks but better data reuse
__global__ void __launch_bounds__(256, 1)
fp8_gemm_kernel_128(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    __nv_bfloat16 *__restrict__ C,
    int M, int N, int K, int grid_m, int grid_n)
{
    fp8_gemm_body<128, 128, 32, 8, 4, 2>(A, B, C, M, N, K, grid_m, grid_n);
}

// ============================================================
// Host launch — dispatches based on problem size
// ============================================================

namespace bk {

void fp8_gemm_fwd(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    __nv_bfloat16 *C,
    int M, int N, int K,
    cudaStream_t stream)
{
    // Two conditions trigger the 128×128 path (either means L2 pressure):
    // 1. Total input data > 64MB — data doesn't fit comfortably in 96MB L2
    // 2. Grid blocks > 4096 — too many concurrent blocks thrash L2 even with
    //    smaller total data (e.g., 8192×2048×8192 is 64MB but 16384 blocks)
    long long total_bytes = (long long)(M) * K * 2 + (long long)(K) * N * 2;
    long long small_grid_blocks = (long long)(M / 64) * (N / 64);
    bool use_large = (total_bytes > 64LL * 1024 * 1024) || (small_grid_blocks > 4096);

    if (use_large) {
        constexpr int BM = 128, BN = 128, BK = 32;
        int grid_m = (M + BM - 1) / BM;
        int grid_n = (N + BN - 1) / BN;
        dim3 grid(grid_n, grid_m);
        dim3 block(256);

        constexpr int smem_bytes = 2 * (BM * BK + BK * BN) * sizeof(__nv_bfloat16);
        // 32KB — under 48KB limit, no attribute needed

        fp8_gemm_kernel_128<<<grid, block, smem_bytes, stream>>>(
            A, B, C, M, N, K, grid_m, grid_n);
    } else {
        constexpr int BM = 64, BN = 64, BK = 64;
        int grid_m = (M + BM - 1) / BM;
        int grid_n = (N + BN - 1) / BN;
        dim3 grid(grid_n, grid_m);
        dim3 block(128);

        constexpr int smem_bytes = 2 * (BM * BK + BK * BN) * sizeof(__nv_bfloat16);
        // 32KB — under 48KB limit, no attribute needed

        fp8_gemm_kernel_64<<<grid, block, smem_bytes, stream>>>(
            A, B, C, M, N, K, grid_m, grid_n);
    }
}

} // namespace bk

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor fp8_gemm(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16, "A must be CUDA BF16");
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kBFloat16, "B must be CUDA BF16");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Pad to largest tile multiples (128) — works for both configs
    constexpr int PAD_M = 128, PAD_N = 128, PAD_K = 64;
    int M_pad = ((M + PAD_M - 1) / PAD_M) * PAD_M;
    int K_pad = ((K + PAD_K - 1) / PAD_K) * PAD_K;
    int N_pad = ((N + PAD_N - 1) / PAD_N) * PAD_N;

    bool needs_pad = (M != M_pad || K != K_pad || N != N_pad);

    torch::Tensor A_padded, B_padded;
    if (needs_pad) {
        A_padded = torch::zeros({M_pad, K_pad}, A.options());
        A_padded.narrow(0, 0, M).narrow(1, 0, K).copy_(A);
        B_padded = torch::zeros({K_pad, N_pad}, B.options());
        B_padded.narrow(0, 0, K).narrow(1, 0, N).copy_(B);
    } else {
        A_padded = A;
        B_padded = B;
    }

    auto C_padded = torch::empty({M_pad, N_pad}, A.options());

    bk::fp8_gemm_fwd(
        reinterpret_cast<const __nv_bfloat16 *>(A_padded.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(B_padded.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(C_padded.data_ptr()),
        M_pad, N_pad, K_pad,
        at::cuda::getCurrentCUDAStream());

    if (needs_pad) {
        return C_padded.narrow(0, 0, M).narrow(1, 0, N).contiguous();
    }
    return C_padded;
}
