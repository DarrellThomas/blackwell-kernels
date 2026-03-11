// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
// Standalone MMA test: load A via ldmatrix_x4, B via ldmatrix_x2_trans,
// compute D = A * B^T + C, verify against CPU reference.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

constexpr int PAD = 8;

// MMA m16n8k16 BF16
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

// Test kernel: compute S = A * B^T for a single m16n8k16 tile
// A is [16, 16] row-major, B is [8, 16] row-major (B^T is [16, 8])
// The MMA computes D = A[m16,k16] * (B^T)[k16,n8] = A * B^T
// Since MMA B operand is col-major: B_col[k,n] = B_row[n,k]
// And we use ldmatrix.trans to load B row-major and transpose.
__global__ void test_mma_kernel(
    const __nv_bfloat16 *__restrict__ A_in,   // [16, 16] row-major
    const __nv_bfloat16 *__restrict__ B_in,   // [8, 16] row-major (represent K)
    float *__restrict__ D_out)                 // [16, 8] row-major
{
    constexpr int STRIDE_A = 16 + PAD;
    constexpr int STRIDE_B = 16 + PAD;

    __shared__ __nv_bfloat16 smem_A[16 * STRIDE_A];
    __shared__ __nv_bfloat16 smem_B[8 * STRIDE_B];

    int tid = threadIdx.x;

    // Load A to shared memory
    for (int i = tid; i < 16 * 16; i += 32) {
        int r = i / 16, c = i % 16;
        smem_A[r * STRIDE_A + c] = A_in[i];
    }
    // Load B to shared memory
    for (int i = tid; i < 8 * 16; i += 32) {
        int r = i / 16, c = i % 16;
        smem_B[r * STRIDE_B + c] = B_in[i];
    }
    __syncthreads();

    // Load A fragment via ldmatrix_x4 (ALT mapping)
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

    // Load B fragment via ldmatrix_x2_trans
    // B is [8, 16] row-major. We want K^T = B^T as the MMA B operand.
    // ldmatrix_x2_trans: threads 0-7 load matrix 0 (first k8 half),
    //                    threads 8-15 load matrix 1 (second k8 half)
    int b_row = tid % 8;                          // 0-7 (kv rows)
    int b_col = ((tid / 8) % 2) * 8;             // 0 or 8 (d halves)
    uint32_t b0, b1;
    {
        uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_B[b_row * STRIDE_B + b_col]));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1)
            : "r"(addr));
    }

    // MMA: D = A * B^T (C = 0)
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    // Store D[16, 8] in the D-fragment layout:
    // d0 = D[T/4, (T%4)*2], d1 = D[T/4, (T%4)*2+1]
    // d2 = D[T/4+8, (T%4)*2], d3 = D[T/4+8, (T%4)*2+1]
    int row0 = tid / 4;
    int col0 = (tid % 4) * 2;
    D_out[row0 * 8 + col0]     = d0;
    D_out[row0 * 8 + col0 + 1] = d1;
    D_out[(row0 + 8) * 8 + col0]     = d2;
    D_out[(row0 + 8) * 8 + col0 + 1] = d3;
}

int main() {
    // A = [16, 16], B = [8, 16]
    // D = A * B^T = [16, 8]
    __nv_bfloat16 h_A[16 * 16], h_B[8 * 16];
    float h_D[16 * 8];
    float ref_D[16 * 8];

    // Initialize with small values to avoid BF16 precision issues
    srand(42);
    for (int i = 0; i < 16 * 16; i++)
        h_A[i] = __float2bfloat16((rand() % 100 - 50) / 50.0f);
    for (int i = 0; i < 8 * 16; i++)
        h_B[i] = __float2bfloat16((rand() % 100 - 50) / 50.0f);

    // CPU reference: D[i][j] = sum_k A[i][k] * B[j][k]
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float sum = 0;
            for (int k = 0; k < 16; k++) {
                float a = __bfloat162float(h_A[i * 16 + k]);
                float b = __bfloat162float(h_B[j * 16 + k]);
                sum += a * b;
            }
            ref_D[i * 8 + j] = sum;
        }
    }

    __nv_bfloat16 *d_A, *d_B;
    float *d_D;
    cudaMalloc(&d_A, 16 * 16 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, 8 * 16 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_D, 16 * 8 * sizeof(float));
    cudaMemcpy(d_A, h_A, 16 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 8 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    test_mma_kernel<<<1, 32>>>(d_A, d_B, d_D);
    cudaMemcpy(h_D, d_D, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== MMA Test: D = A * B^T ===\n");
    float max_err = 0;
    int mismatches = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            float err = fabsf(h_D[i * 8 + j] - ref_D[i * 8 + j]);
            if (err > max_err) max_err = err;
            if (err > 0.1f) mismatches++;
        }
    }
    printf("max_err = %f, mismatches = %d / %d\n", max_err, mismatches, 16 * 8);

    if (mismatches > 0 || max_err > 0.5f) {
        printf("\nFirst 4 rows of GPU output vs reference:\n");
        for (int i = 0; i < 4; i++) {
            printf("Row %2d GPU: ", i);
            for (int j = 0; j < 8; j++) printf("%8.3f", h_D[i * 8 + j]);
            printf("\nRow %2d REF: ", i);
            for (int j = 0; j < 8; j++) printf("%8.3f", ref_D[i * 8 + j]);
            printf("\n");
        }
    } else {
        printf("PASS!\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    return 0;
}
