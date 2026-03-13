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
// Tile config: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, 4 warps (128 threads)
// Warp layout: 2 in M × 2 in N. Each warp computes 32×32 of output
// (2 m16 tiles × 4 n8 tiles). 6 blocks/SM = 24 warps for high occupancy.
// Double-buffered K tiles for pipelining.
//
// Key optimization: Python wrapper pads inputs to exact tile multiples,
// so the kernel has ZERO boundary checks — all loads are 16-byte aligned,
// all stores are valid. No branches in the hot loop.

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

constexpr int GEMM_BLOCK_M = 64;
constexpr int GEMM_BLOCK_N = 64;
constexpr int GEMM_BLOCK_K = 32;
constexpr int GEMM_NUM_WARPS = 4;
constexpr int GEMM_WARP_SIZE = 32;
constexpr int GEMM_THREADS = GEMM_NUM_WARPS * GEMM_WARP_SIZE;  // 128

// Warp layout: 2 warps in M × 2 warps in N = 4 warps
// Each warp computes 32×32 of output (2 m16 tiles × 4 n8 tiles)
// Target: 6 blocks/SM (24 warps, 6/subpartition) — smaller blocks, more of them
constexpr int GEMM_WARPS_M = 2;
constexpr int GEMM_WARPS_N = 2;
constexpr int GEMM_WARP_M = GEMM_BLOCK_M / GEMM_WARPS_M;      // 32
constexpr int GEMM_WARP_N = GEMM_BLOCK_N / GEMM_WARPS_N;      // 32

// MMA tile: m16n8k16
// Per warp: 2 m16 tiles × 4 n8 tiles = 8 MMA per K-chunk
constexpr int GEMM_MMA_M_TILES = GEMM_WARP_M / 16;             // 2
constexpr int GEMM_MMA_N_TILES = GEMM_WARP_N / 8;              // 4
constexpr int GEMM_MMA_K_TILES = GEMM_BLOCK_K / 16;            // 2

// ============================================================
// Compute: preload A per kc, stream B across N-tiles.
// Non-volatile MMA gives compiler full scheduling freedom.
// ============================================================

template <int M_TILES, int N_TILES, int K_TILES, int BLOCK_K_PARAM, int BLOCK_N_PARAM>
__device__ __forceinline__ void compute_full_tile(
    float accum[][N_TILES][4],          // [M_TILES][N_TILES][4] accumulators
    const __nv_bfloat16 *A_smem,        // A shared memory (XOR swizzled)
    const __nv_bfloat16 *B_smem,        // B shared memory (XOR swizzled)
    int warp_m_off,                     // M offset for this warp
    int warp_n_off,                     // N offset for this warp
    int sub, int t_in_sub, int lane_id) // thread addressing
{
    int b_row_base = (lane_id % 8) + ((lane_id / 8) % 2) * 8;

    #pragma unroll
    for (int kc = 0; kc < K_TILES; kc++) {
        // Preload A fragments for both M-tiles in this K-chunk
        uint32_t a[M_TILES][4];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; mt++) {
            int smem_row = warp_m_off + mt * 16 + (sub / 2) * 8 + t_in_sub;
            int smem_col = kc * 16 + (sub % 2) * 8;
            const void *addr = &A_smem[bk::swizzle_idx<BLOCK_K_PARAM>(smem_row, smem_col)];
            bk::ldmatrix_x4_mma(a[mt][0], a[mt][1], a[mt][2], a[mt][3], addr);
        }

        // Stream B across N-tiles, execute MMA for both M-tiles
        int b_row = b_row_base + kc * 16;
        #pragma unroll
        for (int nt = 0; nt < N_TILES; nt++) {
            int b_col = warp_n_off + nt * 8;
            const void *addr_b = &B_smem[bk::swizzle_idx<BLOCK_N_PARAM>(b_row, b_col)];
            uint32_t b0, b1;
            bk::ldmatrix_x2_trans(b0, b1, addr_b);

            #pragma unroll
            for (int mt = 0; mt < M_TILES; mt++) {
                bk::mma_m16n8k16_bf16_nv(
                    accum[mt][nt][0], accum[mt][nt][1],
                    accum[mt][nt][2], accum[mt][nt][3],
                    a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                    b0, b1,
                    accum[mt][nt][0], accum[mt][nt][1],
                    accum[mt][nt][2], accum[mt][nt][3]);
            }
        }
    }
}

// ============================================================
// GEMM kernel — ZERO boundary checks version
// Requires: M % BLOCK_M == 0, N % BLOCK_N == 0, K % BLOCK_K == 0
// The Python wrapper pads inputs to guarantee this.
// ============================================================

__global__ void __launch_bounds__(GEMM_THREADS, 6)
bf16_gemm_kernel(
    const __nv_bfloat16 *__restrict__ A,   // [M_padded, K_padded] row-major
    const __nv_bfloat16 *__restrict__ B,   // [K_padded, N_padded] row-major
    __nv_bfloat16 *__restrict__ C,         // [M_padded, N_padded] row-major
    int M, int N, int K)
{
    const int bm = blockIdx.y * GEMM_BLOCK_M;
    const int bn = blockIdx.x * GEMM_BLOCK_N;

    const int tid = threadIdx.x;
    const int warp_id = tid / GEMM_WARP_SIZE;
    const int lane_id = tid % GEMM_WARP_SIZE;

    // ---- Shared memory layout (double-buffered A and B) ----
    constexpr int A_SMEM_ELEMS = GEMM_BLOCK_M * GEMM_BLOCK_K;
    constexpr int B_SMEM_ELEMS = GEMM_BLOCK_K * GEMM_BLOCK_N;

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_A = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *smem_B = smem_A + 2 * A_SMEM_ELEMS;

    // ============================================================
    // Initialize FP32 accumulators — [M_TILES][N_TILES][4]
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

    int num_k_blocks = K / GEMM_BLOCK_K;  // exact division guaranteed

    // ============================================================
    // Load helpers — NO boundary checks, ALL cp.async
    // ============================================================
    auto load_A_tile = [&](int k_start, __nv_bfloat16 *dst) {
        constexpr int A_CHUNKS_PER_ROW = GEMM_BLOCK_K / 8;
        constexpr int A_TOTAL_CHUNKS = GEMM_BLOCK_M * A_CHUNKS_PER_ROW;
        #pragma unroll
        for (int i = tid; i < A_TOTAL_CHUNKS; i += GEMM_THREADS) {
            int row = i / A_CHUNKS_PER_ROW;
            int col = (i % A_CHUNKS_PER_ROW) * 8;
            int gm = bm + row;
            int gk = k_start + col;
            int dst_idx = bk::swizzle_idx<GEMM_BLOCK_K>(row, col);
            bk::cp_async_128(&dst[dst_idx], &A[gm * K + gk]);
        }
    };

    auto load_B_tile = [&](int k_start, __nv_bfloat16 *dst) {
        constexpr int B_CHUNKS_PER_ROW = GEMM_BLOCK_N / 8;
        constexpr int B_TOTAL_CHUNKS = GEMM_BLOCK_K * B_CHUNKS_PER_ROW;
        #pragma unroll
        for (int i = tid; i < B_TOTAL_CHUNKS; i += GEMM_THREADS) {
            int row = i / B_CHUNKS_PER_ROW;
            int col = (i % B_CHUNKS_PER_ROW) * 8;
            int gk = k_start + row;
            int gn = bn + col;
            int dst_idx = bk::swizzle_idx<GEMM_BLOCK_N>(row, col);
            bk::cp_async_128(&dst[dst_idx], &B[gk * N + gn]);
        }
    };

    // ============================================================
    // Prologue: load first K-tile
    // ============================================================
    load_A_tile(0, smem_A);
    load_B_tile(0, smem_B);
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    // ============================================================
    // Main K-loop — tight, no branches except the loop itself
    // ============================================================
    int sub = lane_id / 8;
    int t_in_sub = lane_id % 8;
    int warp_m_id = warp_id / GEMM_WARPS_N;   // 0..3
    int warp_n_id = warp_id % GEMM_WARPS_N;   // 0..1
    int warp_m_off = warp_m_id * GEMM_WARP_M; // 0, 32, 64, 96
    int warp_n_off = warp_n_id * GEMM_WARP_N; // 0, 64

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int cur = kb & 1;
        __nv_bfloat16 *A_cur = smem_A + cur * A_SMEM_ELEMS;
        __nv_bfloat16 *B_cur = smem_B + cur * B_SMEM_ELEMS;

        // Prefetch next K-tile
        if (kb + 1 < num_k_blocks) {
            int nxt = 1 - cur;
            int k_start_nxt = (kb + 1) * GEMM_BLOCK_K;
            load_A_tile(k_start_nxt, smem_A + nxt * A_SMEM_ELEMS);
            load_B_tile(k_start_nxt, smem_B + nxt * B_SMEM_ELEMS);
        }
        bk::cp_async_commit();

        // Compute: preload ALL fragments for ALL K-chunks, then fire ALL MMAs
        compute_full_tile<GEMM_MMA_M_TILES, GEMM_MMA_N_TILES, GEMM_MMA_K_TILES,
                          GEMM_BLOCK_K, GEMM_BLOCK_N>(
            C_rmem, A_cur, B_cur, warp_m_off, warp_n_off,
            sub, t_in_sub, lane_id);

        bk::cp_async_wait<0>();
        __syncthreads();
    }

    // ============================================================
    // Epilogue: store C — NO boundary checks (padded output)
    // ============================================================
    #pragma unroll
    for (int mt = 0; mt < GEMM_MMA_M_TILES; mt++) {
        int row0 = bm + warp_m_off + mt * 16 + (lane_id / 4);
        int row1 = row0 + 8;

        #pragma unroll
        for (int nt = 0; nt < GEMM_MMA_N_TILES; nt++) {
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

    // Shared memory: 2*(A_tile + B_tile) = 32KB
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
// PyTorch bindings — handles padding to exact tile multiples
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

    // Pad to exact tile multiples — zero-padded math is still correct for GEMM
    int M_pad = ((M + GEMM_BLOCK_M - 1) / GEMM_BLOCK_M) * GEMM_BLOCK_M;
    int K_pad = ((K + GEMM_BLOCK_K - 1) / GEMM_BLOCK_K) * GEMM_BLOCK_K;
    int N_pad = ((N + GEMM_BLOCK_N - 1) / GEMM_BLOCK_N) * GEMM_BLOCK_N;

    bool needs_pad = (M != M_pad || K != K_pad || N != N_pad);

    torch::Tensor A_padded, B_padded;
    if (needs_pad) {
        // Pad with zeros — C_padded = A_padded * B_padded still gives correct
        // result in the [0:M, 0:N] submatrix because zeros don't contribute
        A_padded = torch::zeros({M_pad, K_pad}, A.options());
        A_padded.narrow(0, 0, M).narrow(1, 0, K).copy_(A);
        B_padded = torch::zeros({K_pad, N_pad}, B.options());
        B_padded.narrow(0, 0, K).narrow(1, 0, N).copy_(B);
    } else {
        A_padded = A;
        B_padded = B;
    }

    auto C_padded = torch::empty({M_pad, N_pad}, A.options());

    bk::bf16_gemm_fwd(
        reinterpret_cast<const __nv_bfloat16 *>(A_padded.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(B_padded.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(C_padded.data_ptr()),
        M_pad, N_pad, K_pad,
        at::cuda::getCurrentCUDAStream());

    // Extract the unpadded result
    if (needs_pad) {
        return C_padded.narrow(0, 0, M).narrow(1, 0, N).contiguous();
    }
    return C_padded;
}
