// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
// Test ldmatrix_x2_trans B-fragment loading and verify MMA result.

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

// Dump the B-fragment contents from ldmatrix_x2_trans
__global__ void dump_b_fragment(
    const __nv_bfloat16 *__restrict__ B_in,  // [8, 16] row-major
    float *__restrict__ b_dump)               // [32, 4] - thread, (b0.lo, b0.hi, b1.lo, b1.hi)
{
    constexpr int STRIDE_B = 16 + PAD;
    __shared__ __nv_bfloat16 smem_B[8 * STRIDE_B];

    int tid = threadIdx.x;
    for (int i = tid; i < 8 * 16; i += 32) {
        int r = i / 16, c = i % 16;
        smem_B[r * STRIDE_B + c] = B_in[i];
    }
    __syncthreads();

    int b_row = tid % 8;
    int b_col = ((tid / 8) % 2) * 8;
    uint32_t b0, b1;
    {
        uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_B[b_row * STRIDE_B + b_col]));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1)
            : "r"(addr));
    }

    __nv_bfloat162 *p0 = reinterpret_cast<__nv_bfloat162*>(&b0);
    __nv_bfloat162 *p1 = reinterpret_cast<__nv_bfloat162*>(&b1);

    b_dump[tid * 4 + 0] = __bfloat162float(p0->x);
    b_dump[tid * 4 + 1] = __bfloat162float(p0->y);
    b_dump[tid * 4 + 2] = __bfloat162float(p1->x);
    b_dump[tid * 4 + 3] = __bfloat162float(p1->y);
}

// Test with identity B: B[j][k] = (j==k) ? 1 : 0, so D = A * B^T = first 8 cols of A
__global__ void test_identity_b(
    const __nv_bfloat16 *__restrict__ A_in,
    float *__restrict__ D_out)
{
    constexpr int STRIDE_A = 16 + PAD;
    constexpr int STRIDE_B = 16 + PAD;

    __shared__ __nv_bfloat16 smem_A[16 * STRIDE_A];
    __shared__ __nv_bfloat16 smem_B[8 * STRIDE_B];

    int tid = threadIdx.x;

    // Load A
    for (int i = tid; i < 16 * 16; i += 32) {
        int r = i / 16, c = i % 16;
        smem_A[r * STRIDE_A + c] = A_in[i];
    }
    // Set B = 8x16 identity-like: B[j][k] = (j==k) ? 1 : 0
    for (int i = tid; i < 8 * (16 + PAD); i += 32) {
        int r = i / (16 + PAD), c = i % (16 + PAD);
        if (c < 16)
            smem_B[i] = __float2bfloat16((r == c) ? 1.0f : 0.0f);
        else
            smem_B[i] = __float2bfloat16(0.0f);  // padding
    }
    __syncthreads();

    // Load A via ldmatrix_x4
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

    // Load B via ldmatrix_x2_trans
    int b_row = tid % 8;
    int b_col = ((tid / 8) % 2) * 8;
    uint32_t b0, b1;
    {
        uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem_B[b_row * STRIDE_B + b_col]));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1)
            : "r"(addr));
    }

    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

    int row0 = tid / 4;
    int col0 = (tid % 4) * 2;
    D_out[row0 * 8 + col0]     = d0;
    D_out[row0 * 8 + col0 + 1] = d1;
    D_out[(row0 + 8) * 8 + col0]     = d2;
    D_out[(row0 + 8) * 8 + col0 + 1] = d3;
}

int main() {
    // Test 1: Dump B-fragment from ldmatrix_x2_trans with known B
    // B[j][k] = j*16+k (as BF16), same encoding as the ldmatrix test
    printf("=== Test 1: ldmatrix_x2_trans B-fragment dump ===\n");
    {
        __nv_bfloat16 h_B[8 * 16];
        for (int j = 0; j < 8; j++)
            for (int k = 0; k < 16; k++)
                h_B[j * 16 + k] = __float2bfloat16(float(j * 16 + k));

        __nv_bfloat16 *d_B;
        float *d_dump;
        float h_dump[32 * 4];
        cudaMalloc(&d_B, 8 * 16 * sizeof(__nv_bfloat16));
        cudaMalloc(&d_dump, 32 * 4 * sizeof(float));
        cudaMemcpy(d_B, h_B, 8 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

        dump_b_fragment<<<1, 32>>>(d_B, d_dump);
        cudaMemcpy(h_dump, d_dump, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

        // Expected B-fragment layout for m16n8k16:
        // b0 = {B^T[(T%4)*2, T/4], B^T[(T%4)*2+1, T/4]} = {B[T/4, (T%4)*2], B[T/4, (T%4)*2+1]}
        // b1 = {B^T[(T%4)*2+8, T/4], B^T[(T%4)*2+9, T/4]} = {B[T/4, (T%4)*2+8], B[T/4, (T%4)*2+9]}

        printf("Thread | b0.lo b0.hi | b1.lo b1.hi | Expected b0           | Expected b1\n");
        printf("       | [kv,d]      | [kv,d]      | B[T/4,(T%%4)*2+0..1]  | B[T/4,(T%%4)*2+8..9]\n");
        printf("-------+-------------+-------------+-----------------------+---------------------\n");
        int errs = 0;
        for (int t = 0; t < 32; t++) {
            float b0lo = h_dump[t * 4 + 0];
            float b0hi = h_dump[t * 4 + 1];
            float b1lo = h_dump[t * 4 + 2];
            float b1hi = h_dump[t * 4 + 3];

            // Expected: B[T/4, (T%4)*2], B[T/4, (T%4)*2+1], B[T/4, (T%4)*2+8], B[T/4, (T%4)*2+9]
            int kv = t / 4;
            int d0 = (t % 4) * 2;
            float exp_b0lo = kv * 16 + d0;
            float exp_b0hi = kv * 16 + d0 + 1;
            float exp_b1lo = kv * 16 + d0 + 8;
            float exp_b1hi = kv * 16 + d0 + 9;

            char ok0 = (fabsf(b0lo - exp_b0lo) < 0.5f && fabsf(b0hi - exp_b0hi) < 0.5f) ? ' ' : '!';
            char ok1 = (fabsf(b1lo - exp_b1lo) < 0.5f && fabsf(b1hi - exp_b1hi) < 0.5f) ? ' ' : '!';
            if (ok0 == '!' || ok1 == '!') errs++;

            printf("T%2d    | %5.0f %5.0f %c| %5.0f %5.0f %c| %5.0f %5.0f             | %5.0f %5.0f\n",
                t, b0lo, b0hi, ok0, b1lo, b1hi, ok1,
                exp_b0lo, exp_b0hi, exp_b1lo, exp_b1hi);
        }
        printf("B-fragment errors: %d / 32\n\n", errs);

        cudaFree(d_B);
        cudaFree(d_dump);
    }

    // Test 2: Identity B MMA test
    printf("=== Test 2: D = A * I^T (should give first 8 cols of A) ===\n");
    {
        __nv_bfloat16 h_A[16 * 16];
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++)
                h_A[i * 16 + j] = __float2bfloat16(float(i * 16 + j));

        __nv_bfloat16 *d_A;
        float *d_D;
        float h_D[16 * 8];
        cudaMalloc(&d_A, 16 * 16 * sizeof(__nv_bfloat16));
        cudaMalloc(&d_D, 16 * 8 * sizeof(float));
        cudaMemcpy(d_A, h_A, 16 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

        test_identity_b<<<1, 32>>>(d_A, d_D);
        cudaMemcpy(h_D, d_D, 16 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

        // D[i][j] should equal A[i][j] for j=0..7 (first 8 cols)
        float max_err = 0;
        int mismatches = 0;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 8; j++) {
                float expected = float(i * 16 + j);
                float err = fabsf(h_D[i * 8 + j] - expected);
                if (err > max_err) max_err = err;
                if (err > 0.5f) mismatches++;
            }
        }
        printf("max_err = %f, mismatches = %d / %d\n", max_err, mismatches, 16 * 8);

        if (mismatches > 0) {
            printf("\nGPU output vs expected (first 4 rows):\n");
            for (int i = 0; i < 4; i++) {
                printf("Row %2d GPU: ", i);
                for (int j = 0; j < 8; j++) printf("%7.1f", h_D[i * 8 + j]);
                printf("\nRow %2d EXP: ", i);
                for (int j = 0; j < 8; j++) printf("%7.1f", float(i * 16 + j));
                printf("\n");
            }
        } else {
            printf("PASS!\n");
        }

        cudaFree(d_A);
        cudaFree(d_D);
    }

    return 0;
}
