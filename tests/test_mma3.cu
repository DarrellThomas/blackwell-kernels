// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
// Test MMA with manual B-fragment loading (no ldmatrix_x2_trans for B).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

constexpr int PAD = 8;

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// D = A * K^T where A is [16, 16] and K is [8, 16] (both row-major)
// A loaded via ldmatrix_x4, K loaded manually into B-fragment
__global__ void test_manual_b_kernel(
    const __nv_bfloat16 *__restrict__ A_in,
    const __nv_bfloat16 *__restrict__ K_in,
    float *__restrict__ D_out)
{
    constexpr int STRIDE_A = 16 + PAD;
    constexpr int STRIDE_K = 16 + PAD;

    __shared__ __nv_bfloat16 smem_A[16 * STRIDE_A];
    __shared__ __nv_bfloat16 smem_K[8 * STRIDE_K];

    int tid = threadIdx.x;

    for (int i = tid; i < 16 * 16; i += 32) {
        int r = i / 16, c = i % 16;
        smem_A[r * STRIDE_A + c] = A_in[i];
    }
    for (int i = tid; i < 8 * 16; i += 32) {
        int r = i / 16, c = i % 16;
        smem_K[r * STRIDE_K + c] = K_in[i];
    }
    __syncthreads();

    // Load A via ldmatrix_x4 (ALT mapping)
    int sub = tid / 8;
    int t_in_sub = tid % 8;
    int a_row = (sub / 2) * 8 + t_in_sub;
    int a_col = (sub % 2) * 8;
    uint32_t a0, a1, a2, a3;
    {
        uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_A[a_row * STRIDE_A + a_col]));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "r"(addr));
    }

    // Load K into B-fragment MANUALLY
    // B_col[k, n] = K^T[k, n] = K[n, k]
    // b0 = {K[n, k0], K[n, k1]} where n = T/4, k0 = (T%4)*2
    // b1 = {K[n, k0+8], K[n, k1+8]}
    int kv_idx = tid / 4;           // n (0..7)
    int d_base = (tid % 4) * 2;     // k0

    uint32_t b0, b1;
    b0 = *reinterpret_cast<const uint32_t*>(&smem_K[kv_idx * STRIDE_K + d_base]);
    b1 = *reinterpret_cast<const uint32_t*>(&smem_K[kv_idx * STRIDE_K + d_base + 8]);

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    int row0 = tid / 4;
    int col0 = (tid % 4) * 2;
    D_out[row0 * 8 + col0]         = d0;
    D_out[row0 * 8 + col0 + 1]     = d1;
    D_out[(row0 + 8) * 8 + col0]     = d2;
    D_out[(row0 + 8) * 8 + col0 + 1] = d3;
}

int main() {
    __nv_bfloat16 h_A[16 * 16], h_K[8 * 16];
    float h_D[16 * 8], ref_D[16 * 8];

    srand(42);
    for (int i = 0; i < 16 * 16; i++)
        h_A[i] = __float2bfloat16((rand() % 100 - 50) / 50.0f);
    for (int i = 0; i < 8 * 16; i++)
        h_K[i] = __float2bfloat16((rand() % 100 - 50) / 50.0f);

    // CPU ref: D[i][j] = sum_k A[i][k] * K[j][k] (K^T multiplication)
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float sum = 0;
            for (int k = 0; k < 16; k++) {
                float a = __bfloat162float(h_A[i * 16 + k]);
                float b = __bfloat162float(h_K[j * 16 + k]);
                sum += a * b;
            }
            ref_D[i * 8 + j] = sum;
        }
    }

    __nv_bfloat16 *d_A, *d_K;
    float *d_D;
    cudaMalloc(&d_A, 16 * 16 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K, 8 * 16 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_D, 16 * 8 * sizeof(float));
    cudaMemcpy(d_A, h_A, 16 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, 8 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    test_manual_b_kernel<<<1, 32>>>(d_A, d_K, d_D);
    cudaMemcpy(h_D, d_D, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0;
    int mismatches = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float err = fabsf(h_D[i * 8 + j] - ref_D[i * 8 + j]);
            if (err > max_err) max_err = err;
            if (err > 0.1f) mismatches++;
        }
    }
    printf("=== Manual B loading: D = A * K^T ===\n");
    printf("max_err = %f, mismatches = %d / %d\n", max_err, mismatches, 16 * 8);

    if (mismatches == 0 && max_err < 0.5f) {
        printf("PASS!\n");
    } else {
        for (int i = 0; i < 4; i++) {
            printf("Row %2d GPU: ", i);
            for (int j = 0; j < 8; j++) printf("%8.3f", h_D[i * 8 + j]);
            printf("\nRow %2d REF: ", i);
            for (int j = 0; j < 8; j++) printf("%8.3f", ref_D[i * 8 + j]);
            printf("\n");
        }
    }

    cudaFree(d_A); cudaFree(d_K); cudaFree(d_D);
    return 0;
}
