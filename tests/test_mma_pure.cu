// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
// Pure MMA test: set A/B fragments directly from registers (no ldmatrix).

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
    uint32_t result;
    uint16_t lo_bits = *reinterpret_cast<uint16_t*>(&lo);
    uint16_t hi_bits = *reinterpret_cast<uint16_t*>(&hi);
    result = lo_bits | (uint32_t(hi_bits) << 16);
    return result;
}

// D = A * B^T where A[16][16], B[8][16], all in global memory
// Each thread manually constructs its own A/B fragment from global mem.
__global__ void test_pure_mma(
    const __nv_bfloat16 *__restrict__ A,  // [16, 16]
    const __nv_bfloat16 *__restrict__ B,  // [8, 16]
    float *__restrict__ D_out)             // [16, 8]
{
    int tid = threadIdx.x;

    // A-fragment: thread T holds
    // a0 = {A[T/4, (T%4)*2], A[T/4, (T%4)*2+1]}         -- m[0:8], k[0:8]
    // a1 = {A[T/4, (T%4)*2+8], A[T/4, (T%4)*2+9]}       -- m[0:8], k[8:16]
    // a2 = {A[T/4+8, (T%4)*2], A[T/4+8, (T%4)*2+1]}     -- m[8:16], k[0:8]
    // a3 = {A[T/4+8, (T%4)*2+8], A[T/4+8, (T%4)*2+9]}   -- m[8:16], k[8:16]
    int a_row0 = tid / 4;
    int a_row1 = a_row0 + 8;
    int a_col0 = (tid % 4) * 2;

    uint32_t a0 = pack_bf16(A[a_row0 * 16 + a_col0],     A[a_row0 * 16 + a_col0 + 1]);
    uint32_t a1 = pack_bf16(A[a_row0 * 16 + a_col0 + 8], A[a_row0 * 16 + a_col0 + 9]);
    uint32_t a2 = pack_bf16(A[a_row1 * 16 + a_col0],     A[a_row1 * 16 + a_col0 + 1]);
    uint32_t a3 = pack_bf16(A[a_row1 * 16 + a_col0 + 8], A[a_row1 * 16 + a_col0 + 9]);

    // B-fragment (from empirical ldmatrix_x2_trans dump):
    // b0 = {B_src[(T%4)*2, T/4], B_src[(T%4)*2+1, T/4]}
    // b1 = {B_src[(T%4)*2, T/4+8], B_src[(T%4)*2+1, T/4+8]}
    // B_src rows = kv (n-dim), B_src cols = d (k-dim)
    // This means the MMA computes D[m,n] = sum_k A[m,k] * B_src[n,k]
    // i.e., D = A * B^T (which is what we want for attention)
    int b_row0 = (tid % 4) * 2;   // kv index (0,2,4,6)
    int b_col  = tid / 4;          // d index (0..7)

    uint32_t b0 = pack_bf16(B[b_row0 * 16 + b_col],       B[(b_row0+1) * 16 + b_col]);
    uint32_t b1 = pack_bf16(B[b_row0 * 16 + b_col + 8],   B[(b_row0+1) * 16 + b_col + 8]);

    // MMA
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    // Store D[16, 8]
    int row0 = tid / 4;
    int col0 = (tid % 4) * 2;
    D_out[row0 * 8 + col0]         = d0;
    D_out[row0 * 8 + col0 + 1]     = d1;
    D_out[(row0 + 8) * 8 + col0]     = d2;
    D_out[(row0 + 8) * 8 + col0 + 1] = d3;
}

int main() {
    __nv_bfloat16 h_A[16 * 16], h_B[8 * 16];
    float h_D[16 * 8], ref_D[16 * 8];

    srand(42);
    for (int i = 0; i < 16 * 16; i++)
        h_A[i] = __float2bfloat16((rand() % 100 - 50) / 50.0f);
    for (int i = 0; i < 8 * 16; i++)
        h_B[i] = __float2bfloat16((rand() % 100 - 50) / 50.0f);

    // D[i][j] = sum_k A[i][k] * B[j][k]
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++) {
            float sum = 0;
            for (int k = 0; k < 16; k++)
                sum += __bfloat162float(h_A[i*16+k]) * __bfloat162float(h_B[j*16+k]);
            ref_D[i*8+j] = sum;
        }

    __nv_bfloat16 *d_A, *d_B; float *d_D;
    cudaMalloc(&d_A, 256*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, 128*sizeof(__nv_bfloat16));
    cudaMalloc(&d_D, 128*sizeof(float));
    cudaMemcpy(d_A, h_A, 256*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 128*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    test_pure_mma<<<1, 32>>>(d_A, d_B, d_D);
    cudaMemcpy(h_D, d_D, 128*sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0;
    int mis = 0;
    for (int i = 0; i < 128; i++) {
        float e = fabsf(h_D[i] - ref_D[i]);
        if (e > max_err) max_err = e;
        if (e > 0.1f) mis++;
    }
    printf("Pure register MMA: max_err=%f, mismatches=%d/128\n", max_err, mis);
    if (mis == 0) { printf("PASS!\n"); return 0; }

    for (int i = 0; i < 4; i++) {
        printf("Row %d GPU:", i);
        for (int j = 0; j < 8; j++) printf(" %7.3f", h_D[i*8+j]);
        printf("\n     REF:");
        for (int j = 0; j < 8; j++) printf(" %7.3f", ref_D[i*8+j]);
        printf("\n");
    }
    return 0;
}
