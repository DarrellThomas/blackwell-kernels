// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
// Benchmark bindings: custom MMA kernels + cuBLAS references

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

// Forward declarations from primitive .cu files
torch::Tensor syrk_f32(torch::Tensor A);
torch::Tensor trmm_f32(torch::Tensor L, torch::Tensor B, bool upper);

// cuBLAS TRMM reference: C = L @ B (left, lower/upper, non-unit diagonal)
// Uses column-major convention: row-major L@B = col-major B^T @ L^T
torch::Tensor cublas_trmm_ref(torch::Tensor L, torch::Tensor B, bool upper)
{
    TORCH_CHECK(L.is_cuda() && L.dtype() == torch::kFloat32);
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kFloat32);
    TORCH_CHECK(L.is_contiguous() && B.is_contiguous());

    int M = L.size(0);
    int N = B.size(1);

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto C = B.clone();
    float alpha = 1.0f;

    // Row-major C = L @ B is column-major C^T = B^T @ L^T
    // So: side=RIGHT, uplo=flip(upper), op=TRANSPOSE for L
    // But cublasStrmm overwrites B, so we clone first.
    //
    // Actually, simplest: compute in column-major by treating row-major as transposed.
    // Row-major L[M,M] @ B[M,N] => col-major: (L@B)^T = B^T @ L^T
    // B^T is [N,M] col-major (= B row-major viewed as col-major)
    // L^T is [M,M] col-major with flipped fill mode
    //
    // cublasStrmm(handle, RIGHT, flip_fill, NO_TRANS, NON_UNIT, N, M, alpha,
    //             L_ptr, M, B_ptr, N, C_ptr, N)
    cublasFillMode_t fill = upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

    cublasStrmm(handle,
        CUBLAS_SIDE_RIGHT, fill, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        N, M, &alpha,
        L.data_ptr<float>(), M,
        B.data_ptr<float>(), N,
        C.data_ptr<float>(), N);

    return C;
}

// cuBLAS SYRK reference: C = A @ A^T using cublasSsyrk
torch::Tensor cublas_syrk_ref(torch::Tensor A)
{
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32);
    TORCH_CHECK(A.is_contiguous());

    int N = A.size(0);
    int K = A.size(1);

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto C = torch::zeros({N, N}, A.options());
    float alpha = 1.0f, beta = 0.0f;

    // Row-major A[N,K] @ A^T[K,N] => col-major: (A@A^T)^T = A@A^T (symmetric)
    // Col-major view: A is [K,N] (transposed), so A^T in col-major is A in row-major.
    // cublasSsyrk(handle, LOWER, TRANSPOSE, N, K, alpha, A_ptr, K, beta, C_ptr, N)
    cublasSsyrk(handle,
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
        N, K, &alpha,
        A.data_ptr<float>(), K,
        &beta,
        C.data_ptr<float>(), N);

    // Symmetrize: cublasSsyrk only fills lower triangle
    auto C_lower = C.tril();
    C = C_lower + C_lower.tril(-1).t();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("syrk_f32", &syrk_f32, "FP32 SYRK with BF16 MMA (custom kernel)");
    m.def("trmm_f32", &trmm_f32, "FP32 TRMM with BF16 MMA (custom kernel)");
    m.def("cublas_trmm", &cublas_trmm_ref, "cuBLAS STRMM reference");
    m.def("cublas_syrk", &cublas_syrk_ref, "cuBLAS SSYRK reference");
}
