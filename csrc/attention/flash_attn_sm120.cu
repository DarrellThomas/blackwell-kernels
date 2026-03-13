// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// Flash Attention v1 for sm_120 (RTX 5090)
// v1 = correctness first. Pure FP32 math through shared memory.
// No MMA, no ldmatrix, no swizzle. Just tiled flash attention.
// v2 will add mma.sync, v3 will add pipelining + swizzle.

constexpr int BLOCK_Q = 64;
constexpr int BLOCK_KV = 32;
constexpr int WARP_SIZE = 32;

// Each thread handles a subset of the output elements.
// We use a 2D thread mapping: threads cover (q_row, d_col) positions.

template <int HEAD_DIM>
__global__ void __launch_bounds__(256)
flash_attn_kernel(
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
    const int q_start = q_block * BLOCK_Q;

    const int tid = threadIdx.x;

    // Each thread is responsible for one row of Q within the block
    // We have 256 threads, BLOCK_Q=64 rows, so 4 threads per row
    // Thread tid handles: row = tid / (256 / BLOCK_Q), col_group = tid % (256/BLOCK_Q)
    // 256/64 = 4 threads per row, each handles HEAD_DIM/4 columns
    constexpr int THREADS_PER_ROW = 4;
    constexpr int COLS_PER_THREAD = HEAD_DIM / THREADS_PER_ROW;

    const int local_row = tid / THREADS_PER_ROW;  // 0..63
    const int col_group = tid % THREADS_PER_ROW;  // 0..3
    const int col_start = col_group * COLS_PER_THREAD;

    // Bounds check
    const int global_row = q_start + local_row;
    if (global_row >= seq_len) return;

    // Pointers for this batch/head
    const __nv_bfloat16 *Q_bh = Q + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *K_bh = K + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *V_bh = V + bh_idx * seq_len * HEAD_DIM;
    __nv_bfloat16 *O_bh = O + bh_idx * seq_len * HEAD_DIM;
    float *L_bh = L + bh_idx * seq_len;

    // Shared memory for K and V tiles
    // K tile: BLOCK_KV x HEAD_DIM
    // V tile: BLOCK_KV x HEAD_DIM
    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_K = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *smem_V = smem_K + BLOCK_KV * HEAD_DIM;
    // Scratch for attention scores: BLOCK_Q x BLOCK_KV floats
    float *smem_S = reinterpret_cast<float *>(smem_V + BLOCK_KV * HEAD_DIM);

    // Load Q row for this thread into registers
    float Q_local[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) {
        Q_local[c] = __bfloat162float(Q_bh[global_row * HEAD_DIM + col_start + c]);
    }

    // Output accumulator
    float O_local[COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) O_local[c] = 0.0f;

    // Online softmax state (shared across col_groups via shuffle/smem)
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    // KV loop
    int kv_end = causal ? min(seq_len, q_start + BLOCK_Q) : seq_len;
    int num_kv_blocks = (kv_end + BLOCK_KV - 1) / BLOCK_KV;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_KV;
        int kv_count = min(BLOCK_KV, kv_end - kv_start);

        // Cooperatively load K and V tiles into shared memory
        // 256 threads, BLOCK_KV * HEAD_DIM elements per tile
        int tile_elems = BLOCK_KV * HEAD_DIM;
        for (int i = tid; i < tile_elems; i += 256) {
            int r = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            if (kv_start + r < seq_len) {
                smem_K[i] = K_bh[(kv_start + r) * HEAD_DIM + c];
                smem_V[i] = V_bh[(kv_start + r) * HEAD_DIM + c];
            } else {
                smem_K[i] = __float2bfloat16(0.0f);
                smem_V[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Compute S[local_row][kv] = dot(Q[global_row], K[kv]) * scale
        // Each thread computes partial dot products for its column group,
        // then we reduce across col_groups.

        // All THREADS_PER_ROW threads for the same row cooperate on the dot product
        for (int kv = 0; kv < kv_count; kv++) {
            float dot = 0.0f;
            #pragma unroll
            for (int c = 0; c < COLS_PER_THREAD; c++) {
                dot += Q_local[c] * __bfloat162float(smem_K[kv * HEAD_DIM + col_start + c]);
            }

            // Reduce across THREADS_PER_ROW threads (col_groups for same row)
            // These threads have consecutive thread IDs: local_row*4 + {0,1,2,3}
            #pragma unroll
            for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            }

            float s = dot * scale;

            // Causal mask
            if (causal && (kv_start + kv) > global_row) {
                s = -FLT_MAX;
            }

            // Store to shared memory so all col_groups can read the same S values
            if (col_group == 0) {
                smem_S[local_row * BLOCK_KV + kv] = s;
            }
        }
        __syncthreads();

        // Online softmax + accumulate O
        // Find max of this block's S values for this row
        float block_max = -FLT_MAX;
        for (int kv = 0; kv < kv_count; kv++) {
            float s = smem_S[local_row * BLOCK_KV + kv];
            block_max = fmaxf(block_max, s);
        }

        // Rescale previous accumulator
        float new_max = fmaxf(row_max, block_max);
        float rescale = (row_max == -FLT_MAX) ? 0.0f : __expf(row_max - new_max);

        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; c++) {
            O_local[c] *= rescale;
        }
        row_sum *= rescale;

        // Compute exp(S - max) and accumulate
        for (int kv = 0; kv < kv_count; kv++) {
            float s = smem_S[local_row * BLOCK_KV + kv];
            float p = __expf(s - new_max);
            row_sum += p;

            // O += p * V[kv]
            #pragma unroll
            for (int c = 0; c < COLS_PER_THREAD; c++) {
                O_local[c] += p * __bfloat162float(smem_V[kv * HEAD_DIM + col_start + c]);
            }
        }
        row_max = new_max;

        __syncthreads();
    }

    // Final normalization and store
    float inv_sum = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) {
        O_bh[global_row * HEAD_DIM + col_start + c] = __float2bfloat16(O_local[c] * inv_sum);
    }

    // Store logsumexp (one thread per row writes it)
    if (col_group == 0) {
        L_bh[global_row] = row_max + __logf(row_sum);
    }
}

// ============================================================
// Host launch function
// ============================================================

namespace bk {

void flash_attn_fwd(
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
    int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;

    dim3 grid(num_q_blocks, bh);
    dim3 block(256);

    // Shared memory: K tile + V tile + S scratch
    int smem_bytes = BLOCK_KV * head_dim * sizeof(__nv_bfloat16) * 2  // K + V
                   + BLOCK_Q * BLOCK_KV * sizeof(float);              // S scratch

    auto launch = [&](auto kernel_fn) {
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }
        kernel_fn<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    };

    switch (head_dim) {
        case 32:  launch(flash_attn_kernel<32>);  break;
        case 64:  launch(flash_attn_kernel<64>);  break;
        case 128: launch(flash_attn_kernel<128>); break;
        default: break;
    }
}

} // namespace bk

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor flash_attn_forward(
    torch::Tensor Q,   // [B, H, N, D]
    torch::Tensor K,   // [B, H, N, D]
    torch::Tensor V,   // [B, H, N, D]
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

    auto O = torch::empty_like(Q);
    auto L = torch::empty({B * H, N}, Q.options().dtype(torch::kFloat32));

    bk::flash_attn_fwd(
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(O.data_ptr()),
        L.data_ptr<float>(),
        B, H, N, D, scale, causal,
        at::cuda::getCurrentCUDAStream());

    return O.reshape({B, H, N, D});
}

// Declared in flash_attn_v2_sm120.cu
torch::Tensor flash_attn_v2_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    float scale, bool causal);

// Declared in flash_attn_v3_sm120.cu
torch::Tensor flash_attn_v3_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    float scale, bool causal);

// Declared in bf16_gemm_sm120.cu
torch::Tensor bf16_gemm(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attn_forward", &flash_attn_forward,
          "Flash Attention forward v1 (sm_120, scalar FP32)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("scale"), py::arg("causal") = false);
    m.def("flash_attn_v2_forward", &flash_attn_v2_forward,
          "Flash Attention forward v2 (sm_120, MMA tensor cores)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("scale"), py::arg("causal") = false);
    m.def("flash_attn_v3_forward", &flash_attn_v3_forward,
          "Flash Attention forward v3 (sm_120, fused exp2f+PV)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("scale"), py::arg("causal") = false);
    m.def("bf16_gemm", &bf16_gemm,
          "BF16 GEMM (sm_120, MMA tensor cores)",
          py::arg("A"), py::arg("B"));
}
