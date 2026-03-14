// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Stage 1: PTX QK^T proof of concept
//
// One warp (32 threads) computes S[16,64] = Q[16,64] × K[64,64]^T
// using PTX inline asm for:
//   - Swizzled shared memory address computation
//   - ldmatrix_x4 with a1/a2 swap (ldmatrix_x4_mma pattern)
//   - mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Validates PTX address arithmetic and MMA against CPU reference.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>

constexpr int QM = 16;   // Q rows
constexpr int KN = 64;   // K rows (= S columns)
constexpr int HD = 64;   // head dim (shared k dimension)

// Swizzle matching swizzle.cuh for COLS=64
__device__ __forceinline__ int swizzle_idx(int row, int col)
{
    int swizzled_col = col ^ ((row & 7) << 3);
    return row * HD + swizzled_col;
}

// ============================================================
// PTX macros
// ============================================================

// S accumulator operand groups (8 nc_chunks × 4 floats)
#define S0 "%0,%1,%2,%3"
#define S1 "%4,%5,%6,%7"
#define S2 "%8,%9,%10,%11"
#define S3 "%12,%13,%14,%15"
#define S4 "%16,%17,%18,%19"
#define S5 "%20,%21,%22,%23"
#define S6 "%24,%25,%26,%27"
#define S7 "%28,%29,%30,%31"

// Q fragment register groups per dc chunk
#define QF0 "q0_0,q0_1,q0_2,q0_3"
#define QF1 "q1_0,q1_1,q1_2,q1_3"
#define QF2 "q2_0,q2_1,q2_2,q2_3"
#define QF3 "q3_0,q3_1,q3_2,q3_3"

// Load Q fragment for one dc chunk via ldmatrix_x4_mma (baked a1/a2 swap).
// q_row and rmask must already be set.
// dc_off: "0", "16", "32", "48"
#define PTX_Q_LOAD(dc_off, q0, q1, q2, q3)                 \
    "shl.b32 col, smod2, 3;\n"                             \
    "add.u32 col, col, " dc_off ";\n"                      \
    "xor.b32 sw_col, col, rmask;\n"                         \
    "mad.lo.u32 idx, q_row, 64, sw_col;\n"                  \
    "shl.b32 idx, idx, 1;\n"                               \
    "add.u32 addr, %32, idx;\n"                             \
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "            \
    "{" q0 "," q2 "," q1 "," q3 "}, [addr];\n"

// Load K fragment + two MMAs for one (dc, nc_pair).
// dc_off: dc*16, nc_off: nc_pair*16
#define PTX_K_MMA(dc_off, nc_off, q, s0, s1)               \
    "add.u32 k_row, q_row, " nc_off ";\n"                  \
    "shl.b32 col, smod2, 3;\n"                             \
    "add.u32 col, col, " dc_off ";\n"                      \
    "and.b32 k_rmask, k_row, 7;\n"                         \
    "shl.b32 k_rmask, k_rmask, 3;\n"                       \
    "xor.b32 sw_col, col, k_rmask;\n"                      \
    "mad.lo.u32 idx, k_row, 64, sw_col;\n"                 \
    "shl.b32 idx, idx, 1;\n"                               \
    "add.u32 addr, %33, idx;\n"                             \
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "            \
    "{k0,k1,k2,k3}, [addr];\n"                             \
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
    "{" s0 "}, {" q "}, {k0,k1}, {" s0 "};\n"              \
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
    "{" s1 "}, {" q "}, {k2,k3}, {" s1 "};\n"

// ============================================================
// Kernel: 1 block, 32 threads (1 warp)
// ============================================================

__global__ void test_ptx_qkt_kernel(
    const __nv_bfloat16 *__restrict__ Q_in,
    const __nv_bfloat16 *__restrict__ K_in,
    float *__restrict__ S_out)
{
    __shared__ __nv_bfloat16 smem_Q[QM * HD];
    __shared__ __nv_bfloat16 smem_K[KN * HD];

    int tid = threadIdx.x;

    // Load Q into swizzled smem
    for (int i = tid; i < QM * HD; i += 32) {
        int row = i / HD;
        int col = i % HD;
        smem_Q[swizzle_idx(row, col)] = Q_in[i];
    }

    // Load K into swizzled smem
    for (int i = tid; i < KN * HD; i += 32) {
        int row = i / HD;
        int col = i % HD;
        smem_K[swizzle_idx(row, col)] = K_in[i];
    }

    __syncwarp();

    uint32_t q_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_Q));
    uint32_t k_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_K));

    float S[8][4] = {};

    // ================================================================
    // PTX: S[16,64] = Q[16,64] * K[64,64]^T
    //
    // Operands (35 total, well under 50 threshold):
    //   Outputs %0-%31:  S[8][4] (32 "+f")
    //   Inputs  %32-%34: q_base, k_base, lane_id (3 "r")
    // ================================================================
    asm volatile(
        "{\n"
        ".reg .u32 sub, tsub, hsub, smod2;\n"
        ".reg .u32 q_row, rmask, col, sw_col, idx, addr;\n"
        ".reg .u32 k_row, k_rmask;\n"
        ".reg .b32 q0_0,q0_1,q0_2,q0_3;\n"
        ".reg .b32 q1_0,q1_1,q1_2,q1_3;\n"
        ".reg .b32 q2_0,q2_1,q2_2,q2_3;\n"
        ".reg .b32 q3_0,q3_1,q3_2,q3_3;\n"
        ".reg .b32 k0, k1, k2, k3;\n"
        "\n"

        // Lane decomposition
        "shr.b32 sub, %34, 3;\n"
        "and.b32 tsub, %34, 7;\n"
        "shr.b32 hsub, sub, 1;\n"
        "and.b32 smod2, sub, 1;\n"
        "\n"

        // Q row (shared across all dc): q_row = hsub*8 + tsub
        "shl.b32 q_row, hsub, 3;\n"
        "add.u32 q_row, q_row, tsub;\n"
        // Q row mask for swizzle: rmask = (q_row & 7) << 3
        "and.b32 rmask, q_row, 7;\n"
        "shl.b32 rmask, rmask, 3;\n"
        "\n"

        // Load Q fragments for all 4 dc chunks
        PTX_Q_LOAD("0",  "q0_0", "q0_1", "q0_2", "q0_3")
        PTX_Q_LOAD("16", "q1_0", "q1_1", "q1_2", "q1_3")
        PTX_Q_LOAD("32", "q2_0", "q2_1", "q2_2", "q2_3")
        PTX_Q_LOAD("48", "q3_0", "q3_1", "q3_2", "q3_3")
        "\n"

        // dc=0: K load + MMA for all 4 nc_pairs
        PTX_K_MMA("0", "0",  QF0, S0, S1)
        PTX_K_MMA("0", "16", QF0, S2, S3)
        PTX_K_MMA("0", "32", QF0, S4, S5)
        PTX_K_MMA("0", "48", QF0, S6, S7)

        // dc=1
        PTX_K_MMA("16", "0",  QF1, S0, S1)
        PTX_K_MMA("16", "16", QF1, S2, S3)
        PTX_K_MMA("16", "32", QF1, S4, S5)
        PTX_K_MMA("16", "48", QF1, S6, S7)

        // dc=2
        PTX_K_MMA("32", "0",  QF2, S0, S1)
        PTX_K_MMA("32", "16", QF2, S2, S3)
        PTX_K_MMA("32", "32", QF2, S4, S5)
        PTX_K_MMA("32", "48", QF2, S6, S7)

        // dc=3
        PTX_K_MMA("48", "0",  QF3, S0, S1)
        PTX_K_MMA("48", "16", QF3, S2, S3)
        PTX_K_MMA("48", "32", QF3, S4, S5)
        PTX_K_MMA("48", "48", QF3, S6, S7)

        "}\n"

        : "+f"(S[0][0]), "+f"(S[0][1]), "+f"(S[0][2]), "+f"(S[0][3]),
          "+f"(S[1][0]), "+f"(S[1][1]), "+f"(S[1][2]), "+f"(S[1][3]),
          "+f"(S[2][0]), "+f"(S[2][1]), "+f"(S[2][2]), "+f"(S[2][3]),
          "+f"(S[3][0]), "+f"(S[3][1]), "+f"(S[3][2]), "+f"(S[3][3]),
          "+f"(S[4][0]), "+f"(S[4][1]), "+f"(S[4][2]), "+f"(S[4][3]),
          "+f"(S[5][0]), "+f"(S[5][1]), "+f"(S[5][2]), "+f"(S[5][3]),
          "+f"(S[6][0]), "+f"(S[6][1]), "+f"(S[6][2]), "+f"(S[6][3]),
          "+f"(S[7][0]), "+f"(S[7][1]), "+f"(S[7][2]), "+f"(S[7][3])
        : "r"(q_base), "r"(k_base), "r"(tid)
    );

    // Write S to global memory using MMA D-fragment mapping:
    // d0→D[T/4,(T%4)*2], d1→D[T/4,(T%4)*2+1],
    // d2→D[T/4+8,(T%4)*2], d3→D[T/4+8,(T%4)*2+1]
    int m0 = tid / 4;
    int n_off = (tid % 4) * 2;
    for (int nc = 0; nc < 8; nc++) {
        int n0 = nc * 8 + n_off;
        S_out[m0 * KN + n0]           = S[nc][0];
        S_out[m0 * KN + n0 + 1]       = S[nc][1];
        S_out[(m0 + 8) * KN + n0]     = S[nc][2];
        S_out[(m0 + 8) * KN + n0 + 1] = S[nc][3];
    }
}

// ============================================================
// Host
// ============================================================

int main()
{
    __nv_bfloat16 h_Q[QM * HD], h_K[KN * HD];
    float h_S[QM * KN], h_ref[QM * KN];

    srand(42);
    for (int i = 0; i < QM * HD; i++)
        h_Q[i] = __float2bfloat16((rand() % 200 - 100) / 100.0f);
    for (int i = 0; i < KN * HD; i++)
        h_K[i] = __float2bfloat16((rand() % 200 - 100) / 100.0f);

    // CPU reference: S[m][n] = sum_k Q[m][k] * K[n][k]
    for (int m = 0; m < QM; m++)
        for (int n = 0; n < KN; n++) {
            float sum = 0.0f;
            for (int k = 0; k < HD; k++)
                sum += __bfloat162float(h_Q[m * HD + k]) *
                       __bfloat162float(h_K[n * HD + k]);
            h_ref[m * KN + n] = sum;
        }

    __nv_bfloat16 *d_Q, *d_K;
    float *d_S;
    cudaMalloc(&d_Q, QM * HD * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K, KN * HD * sizeof(__nv_bfloat16));
    cudaMalloc(&d_S, QM * KN * sizeof(float));
    cudaMemcpy(d_Q, h_Q, QM * HD * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, KN * HD * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_S, 0, QM * KN * sizeof(float));

    test_ptx_qkt_kernel<<<1, 32>>>(d_Q, d_K, d_S);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_S, d_S, QM * KN * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < QM * KN; i++) {
        float e = fabsf(h_S[i] - h_ref[i]);
        if (e > max_err) max_err = e;
        if (e > 0.5f) mismatches++;
    }

    printf("PTX QK^T (D=64, 16x64 output): max_err=%.4f, mismatches=%d/%d\n",
           max_err, mismatches, QM * KN);

    if (mismatches == 0 && max_err < 0.5f) {
        printf("PASS!\n");
    } else {
        printf("FAIL!\n");
        for (int m = 0; m < 4; m++) {
            printf("Row %d GPU:", m);
            for (int n = 0; n < 8; n++) printf(" %7.3f", h_S[m * KN + n]);
            printf("\n     REF:");
            for (int n = 0; n < 8; n++) printf(" %7.3f", h_ref[m * KN + n]);
            printf("\n");
        }
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_S);
    return mismatches > 0 ? 1 : 0;
}
