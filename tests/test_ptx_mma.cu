// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Proof of concept: PTX register declarations inside inline asm
// to eliminate MOV instructions in the MMA inner loop.
//
// The hypothesis: declaring .reg variables in PTX inside an asm block
// lets us load A fragments once and reuse them across multiple MMAs,
// bypassing ptxas's tendency to funnel everything through R12-R15.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Minimal shared memory: 2 tiles of BF16
// A: [16, 16] = 256 elements = 512 bytes
// B: [16, 8] × 2 = 128 elements × 2 = 512 bytes
constexpr int SMEM_A_ELEMS = 16 * 16;   // m16k16, one A tile
constexpr int SMEM_B_ELEMS = 16 * 8;    // k16n8, one B tile

// Reference kernel: separate C++ calls (generates MOVs)
__global__ void __launch_bounds__(32, 1)
mma_reference(const __nv_bfloat16 *A_global, const __nv_bfloat16 *B_global,
              float *C_global)
{
    __shared__ __nv_bfloat16 smem_A[SMEM_A_ELEMS];  // [16, 16]
    __shared__ __nv_bfloat16 smem_B[2 * SMEM_B_ELEMS];  // 2 × [16, 8]

    int lane = threadIdx.x;

    // Load A and B into shared memory
    for (int i = lane; i < SMEM_A_ELEMS; i += 32)
        smem_A[i] = A_global[i];
    for (int i = lane; i < 2 * SMEM_B_ELEMS; i += 32)
        smem_B[i] = B_global[i];
    __syncthreads();

    // Load A fragment via ldmatrix_x4
    int sub = lane / 8;
    int t_in_sub = lane % 8;
    int a_row = (sub / 2) * 8 + t_in_sub;
    int a_col = (sub % 2) * 8;
    uint32_t a_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem_A[a_row * 16 + a_col]));
    uint32_t A_r0, A_r1, A_r2, A_r3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(A_r0), "=r"(A_r1), "=r"(A_r2), "=r"(A_r3)
        : "r"(a_addr));

    // Load 2 B fragments via ldmatrix_x2_trans
    int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
    uint32_t b_addr0 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem_B[b_row * 8]));
    uint32_t b_addr1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem_B[SMEM_B_ELEMS + b_row * 8]));

    // Accumulators for 2 output tiles
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    float c4 = 0, c5 = 0, c6 = 0, c7 = 0;

    // REFERENCE: separate asm calls with manual swap
    uint32_t B0_r0, B0_r1;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(B0_r0), "=r"(B0_r1)
        : "r"(b_addr0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
        : "r"(A_r0), "r"(A_r2), "r"(A_r1), "r"(A_r3),  // swap r1<->r2
          "r"(B0_r0), "r"(B0_r1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));

    uint32_t B1_r0, B1_r1;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(B1_r0), "=r"(B1_r1)
        : "r"(b_addr1));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c4), "=f"(c5), "=f"(c6), "=f"(c7)
        : "r"(A_r0), "r"(A_r2), "r"(A_r1), "r"(A_r3),
          "r"(B1_r0), "r"(B1_r1),
          "f"(c4), "f"(c5), "f"(c6), "f"(c7));

    // Store results
    int t4 = lane / 4;
    int col = (lane % 4) * 2;
    if (t4 < 2 && col < 8) {
        C_global[t4 * 8 + col]     = c0;
        C_global[t4 * 8 + col + 1] = c1;
        C_global[(t4 + 8) * 8 + col]     = c2;
        C_global[(t4 + 8) * 8 + col + 1] = c3;
    }
    if (t4 < 2 && col < 8) {
        C_global[128 + t4 * 8 + col]     = c4;
        C_global[128 + t4 * 8 + col + 1] = c5;
        C_global[128 + (t4 + 8) * 8 + col]     = c6;
        C_global[128 + (t4 + 8) * 8 + col + 1] = c7;
    }
}

// PTX kernel: declare .reg inside asm block, load A once, reuse for both MMAs
__global__ void __launch_bounds__(32, 1)
mma_ptx_regs(const __nv_bfloat16 *A_global, const __nv_bfloat16 *B_global,
             float *C_global)
{
    __shared__ __nv_bfloat16 smem_A[SMEM_A_ELEMS];
    __shared__ __nv_bfloat16 smem_B[2 * SMEM_B_ELEMS];

    int lane = threadIdx.x;

    for (int i = lane; i < SMEM_A_ELEMS; i += 32)
        smem_A[i] = A_global[i];
    for (int i = lane; i < 2 * SMEM_B_ELEMS; i += 32)
        smem_B[i] = B_global[i];
    __syncthreads();

    // Compute addresses
    int sub = lane / 8;
    int t_in_sub = lane % 8;
    int a_row = (sub / 2) * 8 + t_in_sub;
    int a_col = (sub % 2) * 8;
    uint32_t a_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem_A[a_row * 16 + a_col]));
    int b_row = (lane % 8) + ((lane / 8) % 2) * 8;
    uint32_t b_addr0 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem_B[b_row * 8]));
    uint32_t b_addr1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem_B[SMEM_B_ELEMS + b_row * 8]));

    // Accumulators
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    float c4 = 0, c5 = 0, c6 = 0, c7 = 0;

    // PTX APPROACH: single asm block with .reg declarations
    // A loaded once into PTX-named registers, reused for both B tiles
    asm volatile(
        // Declare PTX registers for A (with swap baked in)
        ".reg .b32 pa<4>;\n"
        // Declare PTX registers for B
        ".reg .b32 pb<2>;\n"

        // Load A fragment (once) — swap operand order for a1/a2
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {pa0, pa2, pa1, pa3}, [%8];\n"

        // B tile 0: load + MMA
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {pb0, pb1}, [%9];\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "  {%0, %1, %2, %3}, {pa0, pa1, pa2, pa3}, {pb0, pb1}, {%0, %1, %2, %3};\n"

        // B tile 1: load + MMA (reusing A from pa0-pa3)
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {pb0, pb1}, [%10];\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "  {%4, %5, %6, %7}, {pa0, pa1, pa2, pa3}, {pb0, pb1}, {%4, %5, %6, %7};\n"

        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3),  // %0-3: tile 0 accum
          "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)    // %4-7: tile 1 accum
        : "r"(a_addr),   // %8: A shared address
          "r"(b_addr0),  // %9: B tile 0 address
          "r"(b_addr1)   // %10: B tile 1 address
    );

    // Store results
    int t4 = lane / 4;
    int col = (lane % 4) * 2;
    if (t4 < 2 && col < 8) {
        C_global[t4 * 8 + col]     = c0;
        C_global[t4 * 8 + col + 1] = c1;
        C_global[(t4 + 8) * 8 + col]     = c2;
        C_global[(t4 + 8) * 8 + col + 1] = c3;
    }
    if (t4 < 2 && col < 8) {
        C_global[128 + t4 * 8 + col]     = c4;
        C_global[128 + t4 * 8 + col + 1] = c5;
        C_global[128 + (t4 + 8) * 8 + col]     = c6;
        C_global[128 + (t4 + 8) * 8 + col + 1] = c7;
    }
}

int main()
{
    // Allocate and init data
    const int A_size = 16 * 16;
    const int B_size = 2 * 16 * 8;
    const int C_size = 2 * 16 * 8;

    __nv_bfloat16 *h_A = new __nv_bfloat16[A_size];
    __nv_bfloat16 *h_B = new __nv_bfloat16[B_size];
    float *h_C_ref = new float[C_size]();
    float *h_C_ptx = new float[C_size]();

    srand(42);
    for (int i = 0; i < A_size; i++)
        h_A[i] = __float2bfloat16_rn((float)(rand() % 10 - 5) / 5.0f);
    for (int i = 0; i < B_size; i++)
        h_B[i] = __float2bfloat16_rn((float)(rand() % 10 - 5) / 5.0f);

    __nv_bfloat16 *d_A, *d_B;
    float *d_C_ref, *d_C_ptx;
    cudaMalloc(&d_A, A_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, B_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_ref, C_size * sizeof(float));
    cudaMalloc(&d_C_ptx, C_size * sizeof(float));
    cudaMemcpy(d_A, h_A, A_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_C_ref, 0, C_size * sizeof(float));
    cudaMemset(d_C_ptx, 0, C_size * sizeof(float));

    // Run both kernels
    mma_reference<<<1, 32>>>(d_A, d_B, d_C_ref);
    mma_ptx_regs<<<1, 32>>>(d_A, d_B, d_C_ptx);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_ref, d_C_ref, C_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ptx, d_C_ptx, C_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float max_err = 0;
    for (int i = 0; i < C_size; i++) {
        float err = fabsf(h_C_ref[i] - h_C_ptx[i]);
        if (err > max_err) max_err = err;
    }

    printf("max_err between reference and PTX: %f\n", max_err);
    if (max_err < 0.01f)
        printf("PASS: PTX registers produce identical results\n");
    else
        printf("FAIL: results differ!\n");

    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C_ref; delete[] h_C_ptx;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_ref); cudaFree(d_C_ptx);
    return max_err < 0.01f ? 0 : 1;
}
