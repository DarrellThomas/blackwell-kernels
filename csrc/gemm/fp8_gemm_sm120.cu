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
// Tile config: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, 4 warps (128 threads)
// With m16n8k32, BLOCK_K=32 is exactly 1 MMA k-step (vs 2 for BF16 m16n8k16).
// This means each K-tile produces half the MMA calls, but each MMA does 2x work.
//
// Key insight: The BF16→FP8 conversion uses vectorized cvt.rn.satfinite.e4m3x2.f32
// which was empirically verified to work on sm_120 (the bf16x2 variant does NOT work).

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
// Tile and thread configuration
// ============================================================

constexpr int FP8_BLOCK_M = 64;
constexpr int FP8_BLOCK_N = 64;
constexpr int FP8_BLOCK_K = 64;   // Larger K tile: 2 MMA k-steps of k=32
constexpr int FP8_NUM_WARPS = 4;
constexpr int FP8_WARP_SIZE = 32;
constexpr int FP8_THREADS = FP8_NUM_WARPS * FP8_WARP_SIZE;  // 128

// Warp layout: 2 warps in M × 2 warps in N = 4 warps
constexpr int FP8_WARPS_M = 2;
constexpr int FP8_WARPS_N = 2;
constexpr int FP8_WARP_M = FP8_BLOCK_M / FP8_WARPS_M;      // 32
constexpr int FP8_WARP_N = FP8_BLOCK_N / FP8_WARPS_N;       // 32

// MMA tile: m16n8k32 (FP8)
// Per warp: 2 m16 tiles × 4 n8 tiles = 8 MMA per K-chunk
constexpr int FP8_MMA_M_TILES = FP8_WARP_M / 16;            // 2
constexpr int FP8_MMA_N_TILES = FP8_WARP_N / 8;             // 4
constexpr int FP8_MMA_K_TILES = FP8_BLOCK_K / 32;           // 2 (k=32 per MMA)

// ============================================================
// FP8 compute tile: load BF16 from smem, convert to FP8, execute MMA
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
    // For m16n8k32 FP8:
    // A fragment: 4 × uint32_t, each holding 4 FP8 values (16 FP8 total per thread)
    // B fragment: 2 × uint32_t, each holding 4 FP8 values (8 FP8 total per thread)
    //
    // We load BF16 data via ldmatrix (which gives us BF16x2 registers),
    // then convert pairs of BF16x2 registers into FP8x4 registers.
    //
    // ldmatrix_x4_mma for k=16 gives: a[0..3] as uint32_t (BF16x2 each)
    // For k=32, we need two ldmatrix_x4_mma calls (k_half=0 and k_half=1)
    // and merge them: FP8_a[i] = pack(cvt(bf16_k0[i]), cvt(bf16_k1[i]))

    int b_row_base = (lane_id % 8) + ((lane_id / 8) % 2) * 8;

    #pragma unroll
    for (int kc = 0; kc < K_TILES; kc++) {
        // Load A fragments (BF16) for both k-halves and convert to FP8
        uint32_t a_fp8[M_TILES][4];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; mt++) {
            int smem_row = warp_m_off + mt * 16 + (sub / 2) * 8 + t_in_sub;

            // Load BF16 for k_half=0 (columns [kc*32 .. kc*32+15])
            int smem_col_k0 = kc * 32 + (sub % 2) * 8;
            const void *addr_k0 = &A_smem[bk::swizzle_idx<BLOCK_K_PARAM>(smem_row, smem_col_k0)];
            uint32_t bf16_k0[4];
            bk::ldmatrix_x4_mma(bf16_k0[0], bf16_k0[1], bf16_k0[2], bf16_k0[3], addr_k0);

            // Load BF16 for k_half=1 (columns [kc*32+16 .. kc*32+31])
            int smem_col_k1 = kc * 32 + 16 + (sub % 2) * 8;
            const void *addr_k1 = &A_smem[bk::swizzle_idx<BLOCK_K_PARAM>(smem_row, smem_col_k1)];
            uint32_t bf16_k1[4];
            bk::ldmatrix_x4_mma(bf16_k1[0], bf16_k1[1], bf16_k1[2], bf16_k1[3], addr_k1);

            // Convert BF16x2 pairs → FP8x4
            // Each bf16_k0[i] holds 2 BF16 for k[0:15], bf16_k1[i] for k[16:31]
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_fp8[mt][i] = bk::bf16x2_pair_to_e4m3x4(bf16_k0[i], bf16_k1[i]);
            }
        }

        // Stream B across N-tiles, convert to FP8, execute MMA
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            int b_col = warp_n_off + nt * 8;

            // Load B BF16 for k_half=0
            int b_row_k0 = b_row_base + kc * 32;
            const void *addr_b_k0 = &B_smem[bk::swizzle_idx<BLOCK_N_PARAM>(b_row_k0, b_col)];
            uint32_t b_bf16_k0_0, b_bf16_k0_1;
            bk::ldmatrix_x2_trans(b_bf16_k0_0, b_bf16_k0_1, addr_b_k0);

            // Load B BF16 for k_half=1
            int b_row_k1 = b_row_base + kc * 32 + 16;
            const void *addr_b_k1 = &B_smem[bk::swizzle_idx<BLOCK_N_PARAM>(b_row_k1, b_col)];
            uint32_t b_bf16_k1_0, b_bf16_k1_1;
            bk::ldmatrix_x2_trans(b_bf16_k1_0, b_bf16_k1_1, addr_b_k1);

            // Convert B to FP8: merge k_half=0 and k_half=1
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
// FP8 GEMM kernel — ZERO boundary checks version
// ============================================================

__global__ void __launch_bounds__(FP8_THREADS, 6)
fp8_gemm_kernel(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    __nv_bfloat16 *__restrict__ C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * FP8_BLOCK_M;
    const int bn = blockIdx.x * FP8_BLOCK_N;

    const int tid = threadIdx.x;
    const int warp_id = tid / FP8_WARP_SIZE;
    const int lane_id = tid % FP8_WARP_SIZE;

    // Shared memory: double-buffered BF16 tiles
    constexpr int A_SMEM_ELEMS = FP8_BLOCK_M * FP8_BLOCK_K;
    constexpr int B_SMEM_ELEMS = FP8_BLOCK_K * FP8_BLOCK_N;

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_A = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *smem_B = smem_A + 2 * A_SMEM_ELEMS;

    // Initialize FP32 accumulators
    float C_rmem[FP8_MMA_M_TILES][FP8_MMA_N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < FP8_MMA_M_TILES; mt++) {
        #pragma unroll
        for (int nt = 0; nt < FP8_MMA_N_TILES; nt++) {
            C_rmem[mt][nt][0] = 0.0f;
            C_rmem[mt][nt][1] = 0.0f;
            C_rmem[mt][nt][2] = 0.0f;
            C_rmem[mt][nt][3] = 0.0f;
        }
    }

    int num_k_blocks = K / FP8_BLOCK_K;

    // Load helpers — NO boundary checks
    auto load_A_tile = [&](int k_start, __nv_bfloat16 *dst) {
        constexpr int A_CHUNKS_PER_ROW = FP8_BLOCK_K / 8;
        constexpr int A_TOTAL_CHUNKS = FP8_BLOCK_M * A_CHUNKS_PER_ROW;
        #pragma unroll
        for (int i = tid; i < A_TOTAL_CHUNKS; i += FP8_THREADS) {
            int row = i / A_CHUNKS_PER_ROW;
            int col = (i % A_CHUNKS_PER_ROW) * 8;
            int gm = bm + row;
            int gk = k_start + col;
            int dst_idx = bk::swizzle_idx<FP8_BLOCK_K>(row, col);
            bk::cp_async_128(&dst[dst_idx], &A[gm * K + gk]);
        }
    };

    auto load_B_tile = [&](int k_start, __nv_bfloat16 *dst) {
        constexpr int B_CHUNKS_PER_ROW = FP8_BLOCK_N / 8;
        constexpr int B_TOTAL_CHUNKS = FP8_BLOCK_K * B_CHUNKS_PER_ROW;
        #pragma unroll
        for (int i = tid; i < B_TOTAL_CHUNKS; i += FP8_THREADS) {
            int row = i / B_CHUNKS_PER_ROW;
            int col = (i % B_CHUNKS_PER_ROW) * 8;
            int gk = k_start + row;
            int gn = bn + col;
            int dst_idx = bk::swizzle_idx<FP8_BLOCK_N>(row, col);
            bk::cp_async_128(&dst[dst_idx], &B[gk * N + gn]);
        }
    };

    // Prologue
    load_A_tile(0, smem_A);
    load_B_tile(0, smem_B);
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    // Main K-loop
    int sub = lane_id / 8;
    int t_in_sub = lane_id % 8;
    int warp_m_id = warp_id / FP8_WARPS_N;
    int warp_n_id = warp_id % FP8_WARPS_N;
    int warp_m_off = warp_m_id * FP8_WARP_M;
    int warp_n_off = warp_n_id * FP8_WARP_N;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int cur = kb & 1;
        __nv_bfloat16 *A_cur = smem_A + cur * A_SMEM_ELEMS;
        __nv_bfloat16 *B_cur = smem_B + cur * B_SMEM_ELEMS;

        // Prefetch next K-tile
        if (kb + 1 < num_k_blocks) {
            int nxt = 1 - cur;
            int k_start_nxt = (kb + 1) * FP8_BLOCK_K;
            load_A_tile(k_start_nxt, smem_A + nxt * A_SMEM_ELEMS);
            load_B_tile(k_start_nxt, smem_B + nxt * B_SMEM_ELEMS);
        }
        bk::cp_async_commit();

        compute_fp8_tile<FP8_MMA_M_TILES, FP8_MMA_N_TILES, FP8_MMA_K_TILES,
                         FP8_BLOCK_K, FP8_BLOCK_N>(
            C_rmem, A_cur, B_cur, warp_m_off, warp_n_off,
            sub, t_in_sub, lane_id);

        bk::cp_async_wait<0>();
        __syncthreads();
    }

    // Epilogue: store C
    #pragma unroll
    for (int mt = 0; mt < FP8_MMA_M_TILES; mt++) {
        int row0 = bm + warp_m_off + mt * 16 + (lane_id / 4);
        int row1 = row0 + 8;

        #pragma unroll
        for (int nt = 0; nt < FP8_MMA_N_TILES; nt++) {
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
// Host launch
// ============================================================

namespace bk {

void fp8_gemm_fwd(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    __nv_bfloat16 *C,
    int M, int N, int K,
    cudaStream_t stream)
{
    int grid_m = (M + FP8_BLOCK_M - 1) / FP8_BLOCK_M;
    int grid_n = (N + FP8_BLOCK_N - 1) / FP8_BLOCK_N;
    dim3 grid(grid_n, grid_m);
    dim3 block(FP8_THREADS);

    // Shared memory: 2*(A_tile + B_tile) with BLOCK_K=64
    // A: 64*64*2 = 8KB, B: 64*64*2 = 8KB, double-buffered = 32KB
    constexpr int smem_bytes = 2 * (FP8_BLOCK_M * FP8_BLOCK_K +
                                     FP8_BLOCK_K * FP8_BLOCK_N) *
                                sizeof(__nv_bfloat16);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(fp8_gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    fp8_gemm_kernel<<<grid, block, smem_bytes, stream>>>(A, B, C, M, N, K);
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

    // Pad to exact tile multiples
    int M_pad = ((M + FP8_BLOCK_M - 1) / FP8_BLOCK_M) * FP8_BLOCK_M;
    int K_pad = ((K + FP8_BLOCK_K - 1) / FP8_BLOCK_K) * FP8_BLOCK_K;
    int N_pad = ((N + FP8_BLOCK_N - 1) / FP8_BLOCK_N) * FP8_BLOCK_N;

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
