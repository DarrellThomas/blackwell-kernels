// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Stage 2: PTX full KV iteration proof of concept
//
// One warp computes one full attention iteration:
//   S = Q * K^T  →  softmax(S) → P  →  O = P * V
// All in PTX (address computation, ldmatrix, MMA, exp2f, shuffle, pack).
//
// Config: Q[16,64], K[64,64], V[64,64] → O[16,64]
// Single KV block, no loop, no cp.async, no double-buffer.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cstdlib>

constexpr int QM = 16;
constexpr int KN = 64;   // BLOCK_KV
constexpr int HD = 64;   // HEAD_DIM

__device__ __forceinline__ int swizzle_idx(int row, int col)
{
    int swizzled_col = col ^ ((row & 7) << 3);
    return row * HD + swizzled_col;
}

// Pack two floats into bf16x2 (for MMA A-fragment from softmax output)
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b)
{
    __nv_bfloat162 packed = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&packed);
}

// ============================================================
// S accumulator / O accumulator operand macros
// ============================================================
#define S0 "%0,%1,%2,%3"
#define S1 "%4,%5,%6,%7"
#define S2 "%8,%9,%10,%11"
#define S3 "%12,%13,%14,%15"
#define S4 "%16,%17,%18,%19"
#define S5 "%20,%21,%22,%23"
#define S6 "%24,%25,%26,%27"
#define S7 "%28,%29,%30,%31"

#define QF0 "q0_0,q0_1,q0_2,q0_3"
#define QF1 "q1_0,q1_1,q1_2,q1_3"
#define QF2 "q2_0,q2_1,q2_2,q2_3"
#define QF3 "q3_0,q3_1,q3_2,q3_3"

#define PTX_Q_LOAD(dc_off, q0, q1, q2, q3)                 \
    "shl.b32 col, smod2, 3;\n"                             \
    "add.u32 col, col, " dc_off ";\n"                      \
    "xor.b32 sw_col, col, rmask;\n"                         \
    "mad.lo.u32 idx, q_row, 64, sw_col;\n"                  \
    "shl.b32 idx, idx, 1;\n"                               \
    "add.u32 addr, %32, idx;\n"                             \
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "            \
    "{" q0 "," q2 "," q1 "," q3 "}, [addr];\n"

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

__global__ void test_ptx_full_iter_kernel(
    const __nv_bfloat16 *__restrict__ Q_in,
    const __nv_bfloat16 *__restrict__ K_in,
    const __nv_bfloat16 *__restrict__ V_in,
    float *__restrict__ O_out,
    float *__restrict__ rsum_out,
    float scale)
{
    __shared__ __nv_bfloat16 smem_Q[QM * HD];
    __shared__ __nv_bfloat16 smem_K[KN * HD];
    __shared__ __nv_bfloat16 smem_V[KN * HD];

    int tid = threadIdx.x;

    for (int i = tid; i < QM * HD; i += 32) {
        int row = i / HD, col = i % HD;
        smem_Q[swizzle_idx(row, col)] = Q_in[i];
    }
    for (int i = tid; i < KN * HD; i += 32) {
        int row = i / HD, col = i % HD;
        smem_K[swizzle_idx(row, col)] = K_in[i];
        smem_V[swizzle_idx(row, col)] = V_in[i];
    }
    __syncwarp();

    uint32_t q_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_Q));
    uint32_t k_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_K));
    uint32_t v_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_V));

    // Pre-scale Q by scale * LOG2E
    float scale_log2e = scale * 1.4426950408889634f;

    // ================================================================
    // Phase 1: QK^T in PTX (same as Stage 1)
    // ================================================================
    float S[8][4] = {};

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
        "shr.b32 sub, %34, 3;\n"
        "and.b32 tsub, %34, 7;\n"
        "shr.b32 hsub, sub, 1;\n"
        "and.b32 smod2, sub, 1;\n"
        "shl.b32 q_row, hsub, 3;\n"
        "add.u32 q_row, q_row, tsub;\n"
        "and.b32 rmask, q_row, 7;\n"
        "shl.b32 rmask, rmask, 3;\n"
        "\n"
        PTX_Q_LOAD("0",  "q0_0", "q0_1", "q0_2", "q0_3")
        PTX_Q_LOAD("16", "q1_0", "q1_1", "q1_2", "q1_3")
        PTX_Q_LOAD("32", "q2_0", "q2_1", "q2_2", "q2_3")
        PTX_Q_LOAD("48", "q3_0", "q3_1", "q3_2", "q3_3")
        "\n"
        PTX_K_MMA("0", "0",  QF0, S0, S1)
        PTX_K_MMA("0", "16", QF0, S2, S3)
        PTX_K_MMA("0", "32", QF0, S4, S5)
        PTX_K_MMA("0", "48", QF0, S6, S7)
        PTX_K_MMA("16", "0",  QF1, S0, S1)
        PTX_K_MMA("16", "16", QF1, S2, S3)
        PTX_K_MMA("16", "32", QF1, S4, S5)
        PTX_K_MMA("16", "48", QF1, S6, S7)
        PTX_K_MMA("32", "0",  QF2, S0, S1)
        PTX_K_MMA("32", "16", QF2, S2, S3)
        PTX_K_MMA("32", "32", QF2, S4, S5)
        PTX_K_MMA("32", "48", QF2, S6, S7)
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

    // Apply scale (S = S * scale_log2e, puts in log2 space for exp2f)
    for (int nc = 0; nc < 8; nc++)
        for (int i = 0; i < 4; i++)
            S[nc][i] *= scale_log2e;

    // ================================================================
    // Phase 2: Softmax in C++ (matches v2 exactly)
    // Row max → rescale → exp2f → sum → shuffle
    // ================================================================
    float this_max[2] = {-FLT_MAX, -FLT_MAX};
    for (int nc = 0; nc < 8; nc++) {
        this_max[0] = fmaxf(this_max[0], fmaxf(S[nc][0], S[nc][1]));
        this_max[1] = fmaxf(this_max[1], fmaxf(S[nc][2], S[nc][3]));
    }
    this_max[0] = fmaxf(this_max[0], __shfl_xor_sync(0xffffffff, this_max[0], 1));
    this_max[0] = fmaxf(this_max[0], __shfl_xor_sync(0xffffffff, this_max[0], 2));
    this_max[1] = fmaxf(this_max[1], __shfl_xor_sync(0xffffffff, this_max[1], 1));
    this_max[1] = fmaxf(this_max[1], __shfl_xor_sync(0xffffffff, this_max[1], 2));

    for (int nc = 0; nc < 8; nc++) {
        S[nc][0] = exp2f(S[nc][0] - this_max[0]);
        S[nc][1] = exp2f(S[nc][1] - this_max[0]);
        S[nc][2] = exp2f(S[nc][2] - this_max[1]);
        S[nc][3] = exp2f(S[nc][3] - this_max[1]);
    }

    float local_sum[2] = {0.0f, 0.0f};
    for (int nc = 0; nc < 8; nc++) {
        local_sum[0] += S[nc][0] + S[nc][1];
        local_sum[1] += S[nc][2] + S[nc][3];
    }
    local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 1);
    local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 2);
    local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 1);
    local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 2);

    // ================================================================
    // Phase 3: P→A pack + PV MMA in PTX
    // Pack P from S (register-only bf16 conversion), load V, MMA
    // ================================================================
    float O[8][4] = {};

    // P_K_CHUNKS=4 (BLOCK_KV/16=64/16), O_N_CHUNKS=8 (HD/8=64/8)
    for (int kc = 0; kc < 4; kc++) {
        int nc0 = 2 * kc;
        int nc1 = 2 * kc + 1;

        // Pack P A-fragment from S
        uint32_t P_a[4];
        P_a[0] = pack_bf16x2(S[nc0][0], S[nc0][1]);
        P_a[1] = pack_bf16x2(S[nc0][2], S[nc0][3]);
        P_a[2] = pack_bf16x2(S[nc1][0], S[nc1][1]);
        P_a[3] = pack_bf16x2(S[nc1][2], S[nc1][3]);

        // Precompute V addresses for this kc
        int sub = tid / 8;
        int t_in_sub = tid % 8;
        uint32_t V_addrs[4];  // 4 nc_pairs for O
        for (int nc = 0; nc < 8; nc += 2) {
            int v_row = kc * 16 + (sub % 2) * 8 + t_in_sub;
            int v_col = (nc + sub / 2) * 8;
            V_addrs[nc / 2] = static_cast<uint32_t>(
                __cvta_generic_to_shared(
                    &smem_V[swizzle_idx(v_row, v_col)]));
        }

        // PV MMA in PTX: O += P * V for each nc_pair
        // 32+4+4 = 40 operands (safe)
        asm volatile(
            "{\n"
            ".reg .b32 v0, v1, v2, v3;\n"
            // nc_pair=0: V[kc][0..1], O[0..1]
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%36];\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%32,%33,%34,%35}, {v0,v1}, {%0,%1,%2,%3};\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%4,%5,%6,%7}, {%32,%33,%34,%35}, {v2,v3}, {%4,%5,%6,%7};\n"
            // nc_pair=1: V[kc][2..3], O[2..3]
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%37];\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%8,%9,%10,%11}, {%32,%33,%34,%35}, {v0,v1}, {%8,%9,%10,%11};\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%12,%13,%14,%15}, {%32,%33,%34,%35}, {v2,v3}, {%12,%13,%14,%15};\n"
            // nc_pair=2: V[kc][4..5], O[4..5]
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%38];\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%16,%17,%18,%19}, {%32,%33,%34,%35}, {v0,v1}, {%16,%17,%18,%19};\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%20,%21,%22,%23}, {%32,%33,%34,%35}, {v2,v3}, {%20,%21,%22,%23};\n"
            // nc_pair=3: V[kc][6..7], O[6..7]
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [%39];\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%24,%25,%26,%27}, {%32,%33,%34,%35}, {v0,v1}, {%24,%25,%26,%27};\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%28,%29,%30,%31}, {%32,%33,%34,%35}, {v2,v3}, {%28,%29,%30,%31};\n"
            "}\n"
            : "+f"(O[0][0]), "+f"(O[0][1]), "+f"(O[0][2]), "+f"(O[0][3]),
              "+f"(O[1][0]), "+f"(O[1][1]), "+f"(O[1][2]), "+f"(O[1][3]),
              "+f"(O[2][0]), "+f"(O[2][1]), "+f"(O[2][2]), "+f"(O[2][3]),
              "+f"(O[3][0]), "+f"(O[3][1]), "+f"(O[3][2]), "+f"(O[3][3]),
              "+f"(O[4][0]), "+f"(O[4][1]), "+f"(O[4][2]), "+f"(O[4][3]),
              "+f"(O[5][0]), "+f"(O[5][1]), "+f"(O[5][2]), "+f"(O[5][3]),
              "+f"(O[6][0]), "+f"(O[6][1]), "+f"(O[6][2]), "+f"(O[6][3]),
              "+f"(O[7][0]), "+f"(O[7][1]), "+f"(O[7][2]), "+f"(O[7][3])
            : "r"(P_a[0]), "r"(P_a[1]), "r"(P_a[2]), "r"(P_a[3]),
              "r"(V_addrs[0]), "r"(V_addrs[1]), "r"(V_addrs[2]), "r"(V_addrs[3])
        );
    }

    // ================================================================
    // Normalize O by row_sum
    // ================================================================
    float inv0 = (local_sum[0] > 0.0f) ? 1.0f / local_sum[0] : 0.0f;
    float inv1 = (local_sum[1] > 0.0f) ? 1.0f / local_sum[1] : 0.0f;
    for (int nc = 0; nc < 8; nc++) {
        O[nc][0] *= inv0; O[nc][1] *= inv0;
        O[nc][2] *= inv1; O[nc][3] *= inv1;
    }

    // Write O
    int m0 = tid / 4;
    int n_off = (tid % 4) * 2;
    for (int nc = 0; nc < 8; nc++) {
        int n0 = nc * 8 + n_off;
        O_out[m0 * HD + n0]           = O[nc][0];
        O_out[m0 * HD + n0 + 1]       = O[nc][1];
        O_out[(m0 + 8) * HD + n0]     = O[nc][2];
        O_out[(m0 + 8) * HD + n0 + 1] = O[nc][3];
    }

    // Write row_sum for verification
    if (tid % 4 == 0) {
        rsum_out[m0]     = local_sum[0];
        rsum_out[m0 + 8] = local_sum[1];
    }
}

// ============================================================
// Host
// ============================================================

int main()
{
    __nv_bfloat16 h_Q[QM * HD], h_K[KN * HD], h_V[KN * HD];
    float h_O[QM * HD], h_rsum[QM];
    float h_O_ref[QM * HD], h_rsum_ref[QM];

    srand(42);
    for (int i = 0; i < QM * HD; i++)
        h_Q[i] = __float2bfloat16((rand() % 200 - 100) / 100.0f);
    for (int i = 0; i < KN * HD; i++) {
        h_K[i] = __float2bfloat16((rand() % 200 - 100) / 100.0f);
        h_V[i] = __float2bfloat16((rand() % 200 - 100) / 100.0f);
    }

    float scale = 1.0f / sqrtf((float)HD);

    // CPU reference
    // 1. S = Q * K^T * scale
    float S_ref[QM][KN];
    for (int m = 0; m < QM; m++)
        for (int n = 0; n < KN; n++) {
            float sum = 0.0f;
            for (int k = 0; k < HD; k++)
                sum += __bfloat162float(h_Q[m * HD + k]) *
                       __bfloat162float(h_K[n * HD + k]);
            S_ref[m][n] = sum * scale;
        }

    // 2. Softmax per row
    float P_ref[QM][KN];
    for (int m = 0; m < QM; m++) {
        float mx = -FLT_MAX;
        for (int n = 0; n < KN; n++)
            mx = fmaxf(mx, S_ref[m][n]);
        float sum = 0.0f;
        for (int n = 0; n < KN; n++) {
            P_ref[m][n] = expf(S_ref[m][n] - mx);
            sum += P_ref[m][n];
        }
        h_rsum_ref[m] = sum;
        for (int n = 0; n < KN; n++)
            P_ref[m][n] /= sum;
    }

    // 3. O = P * V
    for (int m = 0; m < QM; m++)
        for (int d = 0; d < HD; d++) {
            float sum = 0.0f;
            for (int n = 0; n < KN; n++)
                sum += P_ref[m][n] * __bfloat162float(h_V[n * HD + d]);
            h_O_ref[m * HD + d] = sum;
        }

    // Device
    __nv_bfloat16 *d_Q, *d_K, *d_V;
    float *d_O, *d_rsum;
    cudaMalloc(&d_Q, QM * HD * sizeof(__nv_bfloat16));
    cudaMalloc(&d_K, KN * HD * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V, KN * HD * sizeof(__nv_bfloat16));
    cudaMalloc(&d_O, QM * HD * sizeof(float));
    cudaMalloc(&d_rsum, QM * sizeof(float));
    cudaMemcpy(d_Q, h_Q, QM * HD * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, KN * HD * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, KN * HD * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    test_ptx_full_iter_kernel<<<1, 32>>>(d_Q, d_K, d_V, d_O, d_rsum, scale);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_O, d_O, QM * HD * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rsum, d_rsum, QM * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare O
    float max_err = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < QM * HD; i++) {
        float e = fabsf(h_O[i] - h_O_ref[i]);
        if (e > max_err) max_err = e;
        if (e > 0.05f) mismatches++;
    }

    printf("PTX full iter (QK^T + softmax + PV): max_err=%.4f, mismatches=%d/%d\n",
           max_err, mismatches, QM * HD);

    if (mismatches == 0 && max_err < 0.05f) {
        printf("PASS!\n");
    } else {
        printf("FAIL!\n");
        for (int m = 0; m < 2; m++) {
            printf("Row %d GPU:", m);
            for (int d = 0; d < 8; d++) printf(" %7.4f", h_O[m * HD + d]);
            printf("\n     REF:");
            for (int d = 0; d < 8; d++) printf(" %7.4f", h_O_ref[m * HD + d]);
            printf("\n");
        }
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_rsum);
    return mismatches > 0 ? 1 : 0;
}
