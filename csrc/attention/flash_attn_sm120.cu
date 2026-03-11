#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "mma_sm120.cuh"
#include "cp_async.cuh"
#include "ldmatrix.cuh"
#include "swizzle.cuh"
#include "flash_attn_sm120.cuh"

// Flash Attention v1 for sm_120 (RTX 5090)
// Phase 2 implementation - correctness first, then optimize.
//
// This is a stub that registers the PyTorch extension entry points.
// The actual kernel implementation will be built in Phase 2.

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
    // TODO: Phase 2 - implement fused attention kernel
    // v1: Single-stage loading, cp.async, ldmatrix, mma.sync, online softmax
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

    TORCH_CHECK(D <= 128, "head_dim must be <= 128");

    auto O = torch::empty_like(Q);
    auto L = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));

    bk::flash_attn_fwd(
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(O.data_ptr()),
        L.data_ptr<float>(),
        B, H, N, D, scale, causal,
        at::cuda::getCurrentCUDAStream());

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attn_forward", &flash_attn_forward,
          "Flash Attention forward (sm_120 optimized)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("scale"), py::arg("causal") = false);
}
