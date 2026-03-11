// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Smoke test: verify mma.sync.aligned.m16n8k16 compiles and runs on sm_120.
// This validates that the toolchain is correctly configured.
//
// Build: nvcc -O3 -gencode arch=compute_120,code=sm_120 -o test_mma_smoke tests/test_mma_smoke.cu
// Run:   ./test_mma_smoke

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Minimal BF16 mma.sync wrapper
__device__ void mma_m16n8k16_bf16(
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

__global__ void test_mma_kernel(float *output)
{
    // Initialize A fragment with 1.0 in BF16 (0x3F80 packed as pairs)
    // BF16 1.0 = 0x3F80, packed pair = 0x3F803F80
    uint32_t a0 = 0x3F803F80;
    uint32_t a1 = 0x3F803F80;
    uint32_t a2 = 0x3F803F80;
    uint32_t a3 = 0x3F803F80;

    // Initialize B fragment with 1.0
    uint32_t b0 = 0x3F803F80;
    uint32_t b1 = 0x3F803F80;

    // Accumulator starts at 0
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // Execute MMA: C = A * B + 0
    // A is m16k16 of 1.0, B is k16n8 of 1.0
    // Result should be 16.0 (sum of 16 products of 1.0 * 1.0)
    mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, 0.0f, 0.0f, 0.0f, 0.0f);

    // Thread 0 writes results
    if (threadIdx.x == 0) {
        output[0] = d0;
        output[1] = d1;
        output[2] = d2;
        output[3] = d3;
    }
}

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

int main()
{
    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs: %d, Shared mem/block: %zu KB, Registers/block: %d\n",
           prop.multiProcessorCount, prop.sharedMemPerBlock / 1024, prop.regsPerBlock);
    printf("L2 cache: %d MB, Global mem: %.1f GB\n",
           prop.l2CacheSize / (1024 * 1024), prop.totalGlobalMem / 1e9);

    // Allocate output
    float *d_output;
    float h_output[4];
    CHECK_CUDA(cudaMalloc(&d_output, 4 * sizeof(float)));

    // Launch with 1 warp (32 threads) - minimum for mma.sync
    test_mma_kernel<<<1, 32>>>(d_output);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nmma.sync.aligned.m16n8k16 result (thread 0 accumulators):\n");
    printf("  d0=%.1f  d1=%.1f  d2=%.1f  d3=%.1f\n",
           h_output[0], h_output[1], h_output[2], h_output[3]);

    // With all-ones A (m16k16) and all-ones B (k16n8), each output element
    // should be 16.0 (dot product of 16 ones)
    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (h_output[i] != 16.0f) {
            pass = false;
        }
    }

    printf("\n%s: mma.sync BF16 on sm_120\n", pass ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_output));
    return pass ? 0 : 1;
}
