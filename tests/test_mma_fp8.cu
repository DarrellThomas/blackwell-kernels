// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// FP8 MMA smoke test: m16n8k32.row.col.f32.e4m3.e4m3.f32
// Verifies:
//   1. Fragment layout (A, B, D) for FP8 m16n8k32
//   2. Whether a1/a2 swap applies (it does for BF16 m16n8k16)
//   3. B-fragment layout for A*B^T pattern
//   4. BF16-to-FP8 conversion correctness
//
// D[16,8] = A[16,32] * B[8,32]^T  (A*B^T with k=32)

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// FP8 e4m3 MMA
__device__ __forceinline__ void mma_m16n8k32_e4m3(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// Pack 4 FP8 bytes into uint32_t
__device__ __forceinline__ uint32_t pack_e4m3x4(
    __nv_fp8_e4m3 v0, __nv_fp8_e4m3 v1,
    __nv_fp8_e4m3 v2, __nv_fp8_e4m3 v3)
{
    uint8_t b0 = *reinterpret_cast<uint8_t*>(&v0);
    uint8_t b1 = *reinterpret_cast<uint8_t*>(&v1);
    uint8_t b2 = *reinterpret_cast<uint8_t*>(&v2);
    uint8_t b3 = *reinterpret_cast<uint8_t*>(&v3);
    return uint32_t(b0) | (uint32_t(b1) << 8) | (uint32_t(b2) << 16) | (uint32_t(b3) << 24);
}

__host__ __device__ __nv_fp8_e4m3 float_to_e4m3(float f) {
    return __nv_fp8_e4m3(f);
}

__host__ __device__ float e4m3_to_float(__nv_fp8_e4m3 v) {
    return float(v);
}

// Test 1: A*B^T with a1/a2 swap (extrapolated from BF16 pattern)
// A[16,32] row-major, B[8,32] row-major, D[16,8] = A * B^T
//
// FP8 m16n8k32 fragment layout hypothesis (extending BF16 m16n8k16):
//   A-fragment: each uint32_t holds 4 FP8 values (was 2 BF16)
//     a0 = {A[T/4, (T%4)*4+0], A[T/4, (T%4)*4+1], A[T/4, (T%4)*4+2], A[T/4, (T%4)*4+3]}
//     a1 = A[T/4, (T%4)*4 + 16..19]       (k offset by 16)
//     a2 = A[T/4+8, (T%4)*4 + 0..3]       (m offset by 8)
//     a3 = A[T/4+8, (T%4)*4 + 16..19]     (both offsets)
//   B-fragment for B^T: each uint32_t holds 4 FP8 from same B row
//     b0 = {B[T/4, (T%4)*4+0..3]}         (n-row, k[0..15])
//     b1 = {B[T/4, (T%4)*4+16..19]}       (n-row, k[16..31])

__global__ void test_fp8_mma_abt(
    const float *__restrict__ A_f32, // [16, 32]
    const float *__restrict__ B_f32, // [8, 32]
    float *__restrict__ D_out,       // [16, 8]
    int swap_a1a2)
{
    int T = threadIdx.x;
    int m0 = T / 4;
    int k_base = (T % 4) * 4;

    // Build A-fragment: 4 FP8 per uint32_t, k=32
    uint32_t a0 = pack_e4m3x4(
        float_to_e4m3(A_f32[m0 * 32 + k_base + 0]),
        float_to_e4m3(A_f32[m0 * 32 + k_base + 1]),
        float_to_e4m3(A_f32[m0 * 32 + k_base + 2]),
        float_to_e4m3(A_f32[m0 * 32 + k_base + 3]));
    uint32_t a1 = pack_e4m3x4(
        float_to_e4m3(A_f32[m0 * 32 + k_base + 16]),
        float_to_e4m3(A_f32[m0 * 32 + k_base + 17]),
        float_to_e4m3(A_f32[m0 * 32 + k_base + 18]),
        float_to_e4m3(A_f32[m0 * 32 + k_base + 19]));
    uint32_t a2 = pack_e4m3x4(
        float_to_e4m3(A_f32[(m0+8) * 32 + k_base + 0]),
        float_to_e4m3(A_f32[(m0+8) * 32 + k_base + 1]),
        float_to_e4m3(A_f32[(m0+8) * 32 + k_base + 2]),
        float_to_e4m3(A_f32[(m0+8) * 32 + k_base + 3]));
    uint32_t a3 = pack_e4m3x4(
        float_to_e4m3(A_f32[(m0+8) * 32 + k_base + 16]),
        float_to_e4m3(A_f32[(m0+8) * 32 + k_base + 17]),
        float_to_e4m3(A_f32[(m0+8) * 32 + k_base + 18]),
        float_to_e4m3(A_f32[(m0+8) * 32 + k_base + 19]));

    // Build B-fragment for B^T: 4 FP8 per uint32_t from same row
    int n = T / 4;
    int bk_base = (T % 4) * 4;
    uint32_t b0 = pack_e4m3x4(
        float_to_e4m3(B_f32[n * 32 + bk_base + 0]),
        float_to_e4m3(B_f32[n * 32 + bk_base + 1]),
        float_to_e4m3(B_f32[n * 32 + bk_base + 2]),
        float_to_e4m3(B_f32[n * 32 + bk_base + 3]));
    uint32_t b1 = pack_e4m3x4(
        float_to_e4m3(B_f32[n * 32 + bk_base + 16]),
        float_to_e4m3(B_f32[n * 32 + bk_base + 17]),
        float_to_e4m3(B_f32[n * 32 + bk_base + 18]),
        float_to_e4m3(B_f32[n * 32 + bk_base + 19]));

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    if (swap_a1a2) {
        mma_m16n8k32_e4m3(d0, d1, d2, d3, a0, a2, a1, a3, b0, b1, 0, 0, 0, 0);
    } else {
        mma_m16n8k32_e4m3(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);
    }

    // D-fragment: same as BF16
    int dm = T / 4;
    int dn = (T % 4) * 2;
    D_out[dm*8 + dn]         = d0;
    D_out[dm*8 + dn + 1]     = d1;
    D_out[(dm+8)*8 + dn]     = d2;
    D_out[(dm+8)*8 + dn + 1] = d3;
}

int main() {
    const int M = 16, N = 8, K = 32;
    float h_A[M*K], h_B[N*K], h_D[M*N], ref_D[M*N];

    srand(42);
    // Use small values to stay in FP8 e4m3 range
    for (int i = 0; i < M*K; i++)
        h_A[i] = (rand() % 20 - 10) / 10.0f;  // [-1, 1]
    for (int i = 0; i < N*K; i++)
        h_B[i] = (rand() % 20 - 10) / 10.0f;

    // CPU ref: D[m][n] = sum_k A[m][k] * fp8(B[n][k])
    // (use FP8-quantized values for reference to match GPU)
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                float a_q = e4m3_to_float(float_to_e4m3(h_A[m*K+k]));
                float b_q = e4m3_to_float(float_to_e4m3(h_B[n*K+k]));
                sum += a_q * b_q;
            }
            ref_D[m*N+n] = sum;
        }

    float *d_A, *d_B, *d_D;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, N*K*sizeof(float));
    cudaMalloc(&d_D, M*N*sizeof(float));
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*K*sizeof(float), cudaMemcpyHostToDevice);

    // Test WITHOUT a1/a2 swap
    cudaMemset(d_D, 0, M*N*sizeof(float));
    test_fp8_mma_abt<<<1, 32>>>(d_A, d_B, d_D, 0);
    cudaMemcpy(h_D, d_D, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    float max_err_no_swap = 0;
    int mis_no = 0;
    for (int i = 0; i < M*N; i++) {
        float e = fabsf(h_D[i] - ref_D[i]);
        if (e > max_err_no_swap) max_err_no_swap = e;
        if (e > 0.2f) mis_no++;
    }
    printf("FP8 m16n8k32 A*B^T (NO swap):   max_err=%.4f mismatches=%d/%d %s\n",
           max_err_no_swap, mis_no, M*N, mis_no == 0 ? "PASS" : "FAIL");

    // Test WITH a1/a2 swap
    cudaMemset(d_D, 0, M*N*sizeof(float));
    test_fp8_mma_abt<<<1, 32>>>(d_A, d_B, d_D, 1);
    cudaMemcpy(h_D, d_D, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    float max_err_swap = 0;
    int mis_sw = 0;
    for (int i = 0; i < M*N; i++) {
        float e = fabsf(h_D[i] - ref_D[i]);
        if (e > max_err_swap) max_err_swap = e;
        if (e > 0.2f) mis_sw++;
    }
    printf("FP8 m16n8k32 A*B^T (WITH swap): max_err=%.4f mismatches=%d/%d %s\n",
           max_err_swap, mis_sw, M*N, mis_sw == 0 ? "PASS" : "FAIL");

    if (mis_no > 0 && mis_sw > 0) {
        printf("\nBOTH FAILED — fragment layout hypothesis is wrong. Dumping:\n");
        for (int m = 0; m < 4; m++) {
            printf("Row %d GPU(swap):", m);
            for (int n = 0; n < N; n++) printf(" %7.3f", h_D[m*N+n]);
            printf("\n         REF:   ");
            for (int n = 0; n < N; n++) printf(" %7.3f", ref_D[m*N+n]);
            printf("\n");
        }
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    return 0;
}
