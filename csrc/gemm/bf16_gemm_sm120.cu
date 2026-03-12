// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// BF16 GEMM kernel for sm_120 (RTX 5090)
// Uses mma.sync.aligned.m16n8k16 tensor core instructions.
//
// Architecture:
//   Global Memory --[cp.async]--> Shared Memory --[ldmatrix]--> Registers --[mma.sync]--> Registers
//                                                                                           |
//                                                                                     FP32 accumulators
//                                                                                           |
//                                                                                     BF16 output
//
// Computes C = A * B where A is [M, K], B is [K, N], C is [M, N].
// A is row-major, B is row-major. MMA does A(row) * B(col), so we load B transposed
// via ldmatrix_x2_trans to get col-major fragments directly.
//
// Tile config: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 8 warps (256 threads)
// Each warp computes a 16x128 sub-tile of C (1 m16 tile in M, all N tiles).
// Double-buffered K tiles for pipelining.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "mma_sm120.cuh"
#include "ldmatrix.cuh"
#include "cp_async.cuh"
#include "swizzle.cuh"

// ============================================================
// Tile and thread configuration
// ============================================================

constexpr int GEMM_BLOCK_M = 128;
constexpr int GEMM_BLOCK_N = 128;
constexpr int GEMM_BLOCK_K = 32;
constexpr int GEMM_NUM_WARPS = 8;
constexpr int GEMM_WARP_SIZE = 32;
constexpr int GEMM_THREADS = GEMM_NUM_WARPS * GEMM_WARP_SIZE;  // 256

// Each warp handles WARP_M rows x full BLOCK_N columns
constexpr int GEMM_WARP_M = GEMM_BLOCK_M / GEMM_NUM_WARPS;    // 16

// MMA tile: m16n8k16
// Per warp: 1 m16 tile in M dimension, BLOCK_N/8 n8 tiles in N dimension
constexpr int GEMM_MMA_M_TILES = GEMM_WARP_M / 16;             // 1
constexpr int GEMM_MMA_N_TILES = GEMM_BLOCK_N / 8;             // 16
constexpr int GEMM_MMA_K_TILES = GEMM_BLOCK_K / 16;            // 2

// ============================================================
// GEMM kernel
// ============================================================

__global__ void __launch_bounds__(GEMM_THREADS, 2)
bf16_gemm_kernel(
    const __nv_bfloat16 *__restrict__ A,   // [M, K] row-major
    const __nv_bfloat16 *__restrict__ B,   // [K, N] row-major
    __nv_bfloat16 *__restrict__ C,         // [M, N] row-major
    int M, int N, int K)
{
    const int bm = blockIdx.y * GEMM_BLOCK_M;  // M-tile start
    const int bn = blockIdx.x * GEMM_BLOCK_N;  // N-tile start

    const int tid = threadIdx.x;
    const int warp_id = tid / GEMM_WARP_SIZE;
    const int lane_id = tid % GEMM_WARP_SIZE;

    // ---- Shared memory layout (double-buffered A and B) ----
    // A: [BLOCK_M, BLOCK_K] x 2 buffers, swizzled
    // B: [BLOCK_K, BLOCK_N] x 2 buffers, swizzled
    constexpr int A_SMEM_ELEMS = GEMM_BLOCK_M * GEMM_BLOCK_K;
    constexpr int B_SMEM_ELEMS = GEMM_BLOCK_K * GEMM_BLOCK_N;

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_A = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *smem_B = smem_A + 2 * A_SMEM_ELEMS;

    // ============================================================
    // Initialize FP32 accumulators: each warp has WARP_M/16 * BLOCK_N/8 MMA tiles
    // Each MMA tile produces 4 floats per thread
    // ============================================================
    float C_rmem[GEMM_MMA_M_TILES][GEMM_MMA_N_TILES][4];
    #pragma unroll
    for (int mt = 0; mt < GEMM_MMA_M_TILES; mt++) {
        #pragma unroll
        for (int nt = 0; nt < GEMM_MMA_N_TILES; nt++) {
            C_rmem[mt][nt][0] = 0.0f;
            C_rmem[mt][nt][1] = 0.0f;
            C_rmem[mt][nt][2] = 0.0f;
            C_rmem[mt][nt][3] = 0.0f;
        }
    }

    int num_k_blocks = (K + GEMM_BLOCK_K - 1) / GEMM_BLOCK_K;

    // cp.async requires 16-byte aligned source and full 8-element chunks in bounds.
    // For partial boundary chunks, fall back to element-wise loads with zero padding.
    const bool A_aligned = (K % 8 == 0);
    const bool B_aligned = (N % 8 == 0);
    const __nv_bfloat16 ZERO_BF16 = __float2bfloat16_rn(0.0f);

    // ============================================================
    // Helper: load A tile [BLOCK_M, BLOCK_K] from global to shared
    // ============================================================
    auto load_A_tile = [&](int k_start, __nv_bfloat16 *dst) {
        constexpr int A_CHUNKS_PER_ROW = GEMM_BLOCK_K / 8;
        constexpr int A_TOTAL_CHUNKS = GEMM_BLOCK_M * A_CHUNKS_PER_ROW;
        for (int i = tid; i < A_TOTAL_CHUNKS; i += GEMM_THREADS) {
            int row = i / A_CHUNKS_PER_ROW;
            int col = (i % A_CHUNKS_PER_ROW) * 8;
            int gm = bm + row;
            int gk = k_start + col;
            int dst_idx = bk::swizzle_idx<GEMM_BLOCK_K>(row, col);
            bool full_chunk = gm < M && (gk + 7) < K;
            if (A_aligned && full_chunk) {
                bk::cp_async_128(&dst[dst_idx], &A[gm * K + gk]);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    dst[dst_idx + j] = (gm < M && (gk + j) < K)
                        ? A[gm * K + gk + j] : ZERO_BF16;
            }
        }
    };

    // ============================================================
    // Helper: load B tile [BLOCK_K, BLOCK_N] from global to shared
    // ============================================================
    auto load_B_tile = [&](int k_start, __nv_bfloat16 *dst) {
        constexpr int B_CHUNKS_PER_ROW = GEMM_BLOCK_N / 8;
        constexpr int B_TOTAL_CHUNKS = GEMM_BLOCK_K * B_CHUNKS_PER_ROW;
        for (int i = tid; i < B_TOTAL_CHUNKS; i += GEMM_THREADS) {
            int row = i / B_CHUNKS_PER_ROW;
            int col = (i % B_CHUNKS_PER_ROW) * 8;
            int gk = k_start + row;
            int gn = bn + col;
            int dst_idx = bk::swizzle_idx<GEMM_BLOCK_N>(row, col);
            bool full_chunk = gk < K && (gn + 7) < N;
            if (B_aligned && full_chunk) {
                bk::cp_async_128(&dst[dst_idx], &B[gk * N + gn]);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    dst[dst_idx + j] = (gk < K && (gn + j) < N)
                        ? B[gk * N + gn + j] : ZERO_BF16;
            }
        }
    };

    // ============================================================
    // Prologue: load first K-tile into buffer 0
    // ============================================================
    load_A_tile(0, smem_A);
    load_B_tile(0, smem_B);
    if (A_aligned || B_aligned) {
        bk::cp_async_commit();
        bk::cp_async_wait<0>();
    }
    __syncthreads();

    // ============================================================
    // Main K-loop
    // ============================================================
    for (int kb = 0; kb < num_k_blocks; kb++) {
        int cur = kb & 1;
        __nv_bfloat16 *A_cur = smem_A + cur * A_SMEM_ELEMS;
        __nv_bfloat16 *B_cur = smem_B + cur * B_SMEM_ELEMS;

        // ---- Prefetch next K-tile into alternate buffer ----
        if (kb + 1 < num_k_blocks) {
            int nxt = 1 - cur;
            int k_start_nxt = (kb + 1) * GEMM_BLOCK_K;
            load_A_tile(k_start_nxt, smem_A + nxt * A_SMEM_ELEMS);
            load_B_tile(k_start_nxt, smem_B + nxt * B_SMEM_ELEMS);
        }
        if (A_aligned || B_aligned)
            bk::cp_async_commit();

        // ---- Compute: iterate over K-chunks within the tile ----
        #pragma unroll
        for (int kc = 0; kc < GEMM_MMA_K_TILES; kc++) {

            // Load A fragments via ldmatrix_x4
            // Each warp loads its own WARP_M rows (2 m16 tiles)
            #pragma unroll
            for (int mt = 0; mt < GEMM_MMA_M_TILES; mt++) {
                int warp_m_off = warp_id * GEMM_WARP_M + mt * 16;
                int sub = lane_id / 8;
                int t_in_sub = lane_id % 8;
                int smem_row = warp_m_off + (sub / 2) * 8 + t_in_sub;
                int smem_col = kc * 16 + (sub % 2) * 8;
                const void *addr_a = &A_cur[bk::swizzle_idx<GEMM_BLOCK_K>(smem_row, smem_col)];

                uint32_t A_r0, A_r1, A_r2, A_r3;
                bk::ldmatrix_x4(A_r0, A_r1, A_r2, A_r3, addr_a);

                // Compute all N-tiles with this A fragment
                #pragma unroll
                for (int nt = 0; nt < GEMM_MMA_N_TILES; nt++) {
                    // Load B fragment via ldmatrix_x2_trans
                    // B is [K, N] row-major in shared memory.
                    // ldmatrix_x2_trans gives col-major fragment: computes A*B (not A*B^T)
                    int b_row = kc * 16 + (lane_id % 8) + ((lane_id / 8) % 2) * 8;
                    int b_col = nt * 8;
                    const void *addr_b = &B_cur[bk::swizzle_idx<GEMM_BLOCK_N>(b_row, b_col)];

                    uint32_t B_r0, B_r1;
                    bk::ldmatrix_x2_trans(B_r0, B_r1, addr_b);

                    // a1/a2 swap: hard-won lesson from attention kernel
                    bk::mma_m16n8k16_bf16(
                        C_rmem[mt][nt][0], C_rmem[mt][nt][1],
                        C_rmem[mt][nt][2], C_rmem[mt][nt][3],
                        A_r0, A_r2, A_r1, A_r3,  // swap r1<->r2
                        B_r0, B_r1,
                        C_rmem[mt][nt][0], C_rmem[mt][nt][1],
                        C_rmem[mt][nt][2], C_rmem[mt][nt][3]);
                }
            }
        }

        // Wait for prefetch to complete before next iteration
        if (A_aligned || B_aligned)
            bk::cp_async_wait<0>();
        __syncthreads();
    } // end K-loop

    // ============================================================
    // Epilogue: store C from FP32 accumulators to global memory as BF16
    // ============================================================
    // D-fragment layout: d0=C[T/4,(T%4)*2], d1=C[T/4,(T%4)*2+1],
    //                    d2=C[T/4+8,(T%4)*2], d3=C[T/4+8,(T%4)*2+1]
    #pragma unroll
    for (int mt = 0; mt < GEMM_MMA_M_TILES; mt++) {
        int row0 = bm + warp_id * GEMM_WARP_M + mt * 16 + (lane_id / 4);
        int row1 = row0 + 8;

        #pragma unroll
        for (int nt = 0; nt < GEMM_MMA_N_TILES; nt++) {
            int col0 = bn + nt * 8 + (lane_id % 4) * 2;

            int col1 = col0 + 1;

            // Packed 4-byte store when both columns are in bounds
            if (row0 < M && col1 < N) {
                __nv_bfloat162 packed = __floats2bfloat162_rn(
                    C_rmem[mt][nt][0], C_rmem[mt][nt][1]);
                *reinterpret_cast<uint32_t*>(&C[row0 * N + col0]) =
                    *reinterpret_cast<uint32_t*>(&packed);
            } else if (row0 < M && col0 < N) {
                C[row0 * N + col0] = __float2bfloat16_rn(C_rmem[mt][nt][0]);
            }
            if (row1 < M && col1 < N) {
                __nv_bfloat162 packed = __floats2bfloat162_rn(
                    C_rmem[mt][nt][2], C_rmem[mt][nt][3]);
                *reinterpret_cast<uint32_t*>(&C[row1 * N + col0]) =
                    *reinterpret_cast<uint32_t*>(&packed);
            } else if (row1 < M && col0 < N) {
                C[row1 * N + col0] = __float2bfloat16_rn(C_rmem[mt][nt][2]);
            }
        }
    }
}

// ============================================================
// Host launch
// ============================================================

namespace bk {

void bf16_gemm_fwd(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    __nv_bfloat16 *C,
    int M, int N, int K,
    cudaStream_t stream)
{
    int grid_m = (M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M;
    int grid_n = (N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N;
    dim3 grid(grid_n, grid_m);
    dim3 block(GEMM_THREADS);

    // Shared memory: 2*(A_tile + B_tile)
    // A tile: BLOCK_M * BLOCK_K * sizeof(bf16) = 128*32*2 = 8KB
    // B tile: BLOCK_K * BLOCK_N * sizeof(bf16) = 32*128*2 = 8KB
    // Double-buffered: 2*(8+8) = 32KB — fits in 48KB static limit
    constexpr int smem_bytes = 2 * (GEMM_BLOCK_M * GEMM_BLOCK_K +
                                     GEMM_BLOCK_K * GEMM_BLOCK_N) *
                                sizeof(__nv_bfloat16);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(bf16_gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    bf16_gemm_kernel<<<grid, block, smem_bytes, stream>>>(A, B, C, M, N, K);
}

} // namespace bk

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor bf16_gemm(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16, "A must be CUDA BF16");
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kBFloat16, "B must be CUDA BF16");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());  // BF16 output

    bk::bf16_gemm_fwd(
        reinterpret_cast<const __nv_bfloat16 *>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(C.data_ptr()),
        M, N, K,
        at::cuda::getCurrentCUDAStream());

    return C;
}
