// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
// Minimal test to verify ldmatrix_x4 thread-to-register mapping.
// Loads a known 16x16 BF16 matrix via ldmatrix_x4 and dumps register contents.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Pad stride for bank conflicts (same as v2 kernel)
constexpr int PAD = 8;
constexpr int DIM = 16;
constexpr int STRIDE = DIM + PAD;

__global__ void test_ldmatrix_x4_kernel(
    const __nv_bfloat16 *__restrict__ input,  // [16, 16] row-major
    float *__restrict__ output)                // [32, 4, 2] - thread, reg, elem
{
    __shared__ __nv_bfloat16 smem[16 * STRIDE];

    int tid = threadIdx.x;
    // Load input to shared memory with padding
    for (int i = tid; i < 16 * 16; i += 32) {
        int row = i / 16;
        int col = i % 16;
        smem[row * STRIDE + col] = input[i];
    }
    __syncthreads();

    // Test 1: ORIGINAL mapping (lane_id % 16 for row, lane_id / 16 for col)
    {
        int row = tid % 16;
        int col = (tid / 16) * 8;
        uint32_t r0, r1, r2, r3;
        const void *addr = &smem[row * STRIDE + col];
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(addr))));

        // Unpack each register (2 bf16 per uint32)
        __nv_bfloat162 *p0 = reinterpret_cast<__nv_bfloat162*>(&r0);
        __nv_bfloat162 *p1 = reinterpret_cast<__nv_bfloat162*>(&r1);
        __nv_bfloat162 *p2 = reinterpret_cast<__nv_bfloat162*>(&r2);
        __nv_bfloat162 *p3 = reinterpret_cast<__nv_bfloat162*>(&r3);

        int base = tid * 8;  // 8 floats per thread (4 regs × 2 elems)
        output[base + 0] = __bfloat162float(p0->x);
        output[base + 1] = __bfloat162float(p0->y);
        output[base + 2] = __bfloat162float(p1->x);
        output[base + 3] = __bfloat162float(p1->y);
        output[base + 4] = __bfloat162float(p2->x);
        output[base + 5] = __bfloat162float(p2->y);
        output[base + 6] = __bfloat162float(p3->x);
        output[base + 7] = __bfloat162float(p3->y);
    }
}

__global__ void test_ldmatrix_x4_alt_kernel(
    const __nv_bfloat16 *__restrict__ input,  // [16, 16] row-major
    float *__restrict__ output)                // [32, 4, 2]
{
    __shared__ __nv_bfloat16 smem[16 * STRIDE];

    int tid = threadIdx.x;
    for (int i = tid; i < 16 * 16; i += 32) {
        int row = i / 16;
        int col = i % 16;
        smem[row * STRIDE + col] = input[i];
    }
    __syncthreads();

    // Test 2: ALT mapping (lane_id / 8 sub-matrix, lane_id % 8 within sub)
    {
        int sub = tid / 8;
        int t_in_sub = tid % 8;
        int row = (sub / 2) * 8 + t_in_sub;
        int col = (sub % 2) * 8;
        uint32_t r0, r1, r2, r3;
        const void *addr = &smem[row * STRIDE + col];
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(addr))));

        __nv_bfloat162 *p0 = reinterpret_cast<__nv_bfloat162*>(&r0);
        __nv_bfloat162 *p1 = reinterpret_cast<__nv_bfloat162*>(&r1);
        __nv_bfloat162 *p2 = reinterpret_cast<__nv_bfloat162*>(&r2);
        __nv_bfloat162 *p3 = reinterpret_cast<__nv_bfloat162*>(&r3);

        int base = tid * 8;
        output[base + 0] = __bfloat162float(p0->x);
        output[base + 1] = __bfloat162float(p0->y);
        output[base + 2] = __bfloat162float(p1->x);
        output[base + 3] = __bfloat162float(p1->y);
        output[base + 4] = __bfloat162float(p2->x);
        output[base + 5] = __bfloat162float(p2->y);
        output[base + 6] = __bfloat162float(p3->x);
        output[base + 7] = __bfloat162float(p3->y);
    }
}

int main() {
    // Create a 16x16 matrix where element [r][c] = r*16 + c (as BF16)
    // This makes it trivial to identify which element ended up where.
    __nv_bfloat16 h_input[16 * 16];
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++)
            h_input[r * 16 + c] = __float2bfloat16(float(r * 16 + c));

    __nv_bfloat16 *d_input;
    float *d_output;
    float h_output[32 * 8];

    cudaMalloc(&d_input, 16 * 16 * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output, 32 * 8 * sizeof(float));
    cudaMemcpy(d_input, h_input, 16 * 16 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Test 1: Original mapping
    test_ldmatrix_x4_kernel<<<1, 32>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, 32 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== ORIGINAL mapping (lane%%16 row, lane/16 col) ===\n");
    printf("Thread | r0 (elem0, elem1) | r1 (elem0, elem1) | r2 (elem0, elem1) | r3 (elem0, elem1)\n");
    printf("       | [row, col]        | [row, col]        | [row, col]        | [row, col]\n");
    printf("-------+-------------------+-------------------+-------------------+-------------------\n");
    for (int t = 0; t < 32; t++) {
        int base = t * 8;
        printf("T%2d    |", t);
        for (int reg = 0; reg < 4; reg++) {
            float e0 = h_output[base + reg * 2];
            float e1 = h_output[base + reg * 2 + 1];
            int r0 = (int)e0 / 16, c0 = (int)e0 % 16;
            int r1 = (int)e1 / 16, c1 = (int)e1 % 16;
            printf(" [%2d,%2d],[%2d,%2d] |", r0, c0, r1, c1);
        }
        printf("\n");
    }

    // Expected A-fragment for m16n8k16: thread T holds
    // a0: A[T%8, (T/8)*2], A[T%8, (T/8)*2+1]  -- m[0:8] x k[0:8]
    // a1: A[T%8, (T/8)*2+8], A[T%8, (T/8)*2+9] -- m[0:8] x k[8:16]
    // a2: A[T%8+8, (T/8)*2], A[T%8+8, (T/8)*2+1] -- m[8:16] x k[0:8]
    // a3: A[T%8+8, (T/8)*2+8], A[T%8+8, (T/8)*2+9] -- m[8:16] x k[8:16]
    printf("\n=== Expected A-fragment (m16n8k16) ===\n");
    printf("Thread | a0 [row,col]      | a1 [row,col]      | a2 [row,col]      | a3 [row,col]\n");
    printf("-------+-------------------+-------------------+-------------------+-------------------\n");
    for (int t = 0; t < 32; t++) {
        int r_lo = t % 8;
        int c_base = (t / 8) * 2;
        printf("T%2d    | [%2d,%2d],[%2d,%2d] | [%2d,%2d],[%2d,%2d] | [%2d,%2d],[%2d,%2d] | [%2d,%2d],[%2d,%2d] |\n",
            t,
            r_lo, c_base, r_lo, c_base+1,
            r_lo, c_base+8, r_lo, c_base+9,
            r_lo+8, c_base, r_lo+8, c_base+1,
            r_lo+8, c_base+8, r_lo+8, c_base+9);
    }

    // Test 2: Alt mapping
    test_ldmatrix_x4_alt_kernel<<<1, 32>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, 32 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== ALT mapping (sub=lane/8, t_in=lane%%8) ===\n");
    printf("Thread | r0 (elem0, elem1) | r1 (elem0, elem1) | r2 (elem0, elem1) | r3 (elem0, elem1)\n");
    printf("-------+-------------------+-------------------+-------------------+-------------------\n");
    for (int t = 0; t < 32; t++) {
        int base = t * 8;
        printf("T%2d    |", t);
        for (int reg = 0; reg < 4; reg++) {
            float e0 = h_output[base + reg * 2];
            float e1 = h_output[base + reg * 2 + 1];
            int r0 = (int)e0 / 16, c0 = (int)e0 % 16;
            int r1 = (int)e1 / 16, c1 = (int)e1 % 16;
            printf(" [%2d,%2d],[%2d,%2d] |", r0, c0, r1, c1);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
