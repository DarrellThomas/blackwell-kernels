// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
// Definitive MMA register layout test: fully manual register loading.
// No ldmatrix, no shared memory. A = sequential, B = identity.
// If D = first 8 cols of A, our register layout understanding is correct.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

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

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    uint16_t lo_bits = *reinterpret_cast<uint16_t*>(&lo);
    uint16_t hi_bits = *reinterpret_cast<uint16_t*>(&hi);
    return lo_bits | (uint32_t(hi_bits) << 16);
}

// Test all 8 possible (row,col) interpretations of A and B fragments
__global__ void test_all_mappings(
    const __nv_bfloat16 *__restrict__ A,  // [16, 16] sequential
    const __nv_bfloat16 *__restrict__ B,  // [8, 16] identity
    float *__restrict__ D_out,            // [8 * 16 * 8] - 8 mapping variants
    int *__restrict__ mapping_info)        // debug info
{
    int tid = threadIdx.x;

    // Indices
    int g0 = tid / 4;       // 0..7
    int g1 = (tid % 4) * 2; // 0,2,4,6

    // We test different combinations of A-fragment and B-fragment layouts.
    // A-fragment has two possible row/col assignments:
    //   Layout A0: a0={A[g0, g1], A[g0, g1+1]}, a1={A[g0, g1+8], A[g0, g1+9]},
    //              a2={A[g0+8, g1], A[g0+8, g1+1]}, a3={A[g0+8, g1+8], A[g0+8, g1+9]}
    //   Layout A1: a0={A[g1, g0], A[g1+1, g0]}, a1={A[g1+8, g0], A[g1+9, g0]},
    //              a2={A[g1, g0+8], A[g1+1, g0+8]}, a3={A[g1+8, g0+8], A[g1+9, g0+8]}
    // B-fragment has two possible assignments:
    //   Layout B0: b0={B[g0, g1], B[g0, g1+1]}, b1={B[g0, g1+8], B[g0, g1+9]}
    //   Layout B1: b0={B[g1, g0], B[g1+1, g0]}, b1={B[g1, g0+8], B[g1+1, g0+8]}
    // D-fragment has two possible output mappings:
    //   Layout D0: d0->D[g0, g1], d1->D[g0, g1+1], d2->D[g0+8, g1], d3->D[g0+8, g1+1]
    //   Layout D1: d0->D[g1, g0], d1->D[g1+1, g0], d2->D[g1, g0+8], d3->D[g1+1, g0+8]

    // Test 1: Original order (a0, a1, a2, a3) with B0 layout
    {
        uint32_t a0 = pack_bf16(A[g0*16 + g1],       A[g0*16 + g1 + 1]);
        uint32_t a1 = pack_bf16(A[g0*16 + g1 + 8],   A[g0*16 + g1 + 9]);
        uint32_t a2 = pack_bf16(A[(g0+8)*16 + g1],   A[(g0+8)*16 + g1 + 1]);
        uint32_t a3 = pack_bf16(A[(g0+8)*16 + g1+8], A[(g0+8)*16 + g1 + 9]);
        uint32_t b0 = pack_bf16(B[g0*16 + g1],       B[g0*16 + g1 + 1]);
        uint32_t b1 = pack_bf16(B[g0*16 + g1 + 8],   B[g0*16 + g1 + 9]);

        float d0=0, d1=0, d2=0, d3=0;
        // ORIGINAL: a0, a1, a2, a3
        mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);
        float *out = D_out;
        out[g0*8 + g1] = d0; out[g0*8 + g1+1] = d1;
        out[(g0+8)*8 + g1] = d2; out[(g0+8)*8 + g1+1] = d3;

        d0=0; d1=0; d2=0; d3=0;
        // SWAPPED: a0, a2, a1, a3 (swap a1 and a2)
        mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a2, a1, a3, b0, b1, 0, 0, 0, 0);
        out = D_out + 16*8;
        out[g0*8 + g1] = d0; out[g0*8 + g1+1] = d1;
        out[(g0+8)*8 + g1] = d2; out[(g0+8)*8 + g1+1] = d3;
    }
}

int main() {
    // A = sequential: A[m][k] = m*16 + k
    __nv_bfloat16 h_A[16 * 16];
    for (int m = 0; m < 16; m++)
        for (int k = 0; k < 16; k++)
            h_A[m * 16 + k] = __float2bfloat16(float(m * 16 + k));

    // B = identity 8x16: B[n][k] = delta(n, k)
    __nv_bfloat16 h_B[8 * 16];
    for (int n = 0; n < 8; n++)
        for (int k = 0; k < 16; k++)
            h_B[n * 16 + k] = __float2bfloat16((n == k) ? 1.0f : 0.0f);

    // Expected: D = A * B^T = first 8 cols of A
    // D[m][n] = A[m][n] for n=0..7
    float ref_D[16 * 8];
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 8; n++)
            ref_D[m * 8 + n] = float(m * 16 + n);

    __nv_bfloat16 *d_A, *d_B;
    float *d_D;
    int *d_info;
    float h_D[2 * 16 * 8];  // 2 variants
    cudaMalloc(&d_A, 16 * 16 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, 8 * 16 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_D, 2 * 16 * 8 * sizeof(float));
    cudaMalloc(&d_info, 32 * sizeof(int));
    cudaMemcpy(d_A, h_A, 16 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 8 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_D, 0, 2 * 16 * 8 * sizeof(float));

    test_all_mappings<<<1, 32>>>(d_A, d_B, d_D, d_info);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_D, d_D, 2 * 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    const char *names[] = {"ORIGINAL (a0,a1,a2,a3)", "SWAPPED  (a0,a2,a1,a3)"};
    for (int v = 0; v < 2; v++) {
        float *out = h_D + v * 16 * 8;
        float max_err = 0;
        int mismatches = 0;
        for (int i = 0; i < 16 * 8; i++) {
            float e = fabsf(out[i] - ref_D[i]);
            if (e > max_err) max_err = e;
            if (e > 0.5f) mismatches++;
        }
        printf("%s: max_err=%.1f mismatches=%d/128", names[v], max_err, mismatches);
        if (mismatches == 0) printf("  *** PASS ***");
        printf("\n");
        for (int i = 0; i < 16; i++) {
            printf("  Row%2d:", i);
            for (int j = 0; j < 8; j++) printf("%6.0f", out[i*8+j]);
            printf("  exp:");
            for (int j = 0; j < 8; j++) printf("%4.0f", ref_D[i*8+j]);
            float row_err = 0;
            for (int j = 0; j < 8; j++) {
                float e = fabsf(out[i*8+j] - ref_D[i*8+j]);
                if (e > row_err) row_err = e;
            }
            if (row_err > 0.5f) printf(" *** ERR");
            printf("\n");
        }
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D); cudaFree(d_info);
    return 0;
}
