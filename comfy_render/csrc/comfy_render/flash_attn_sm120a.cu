// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Flash Attention forward for sm_120a (RTX 5090).
// D=64: Br=64 Bc=128 non-causal, Br=64 Bc=64 causal (K double-buffer).
// D=128: MMA kernel with K double-buffer pipeline.
// D=40: scalar fallback.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cmath>
#include <cfloat>

#include "mma_sm120.cuh"
#include "ldmatrix.cuh"
#include "cp_async.cuh"

namespace {

constexpr int WARP_SIZE = 32;
constexpr float LOG2E_F = 1.4426950408889634f;

template <int STRIDE>
__device__ __forceinline__ int swz(int row, int col) {
    return row * STRIDE + (col ^ ((row & 7) << 3));
}

__device__ __forceinline__ uint32_t pack_bf16_pair(float a, float b) {
    uint32_t r;
    asm("{ .reg .b16 lo, hi;\n"
        "  cvt.rn.bf16.f32 lo, %1;\n"
        "  cvt.rn.bf16.f32 hi, %2;\n"
        "  mov.b32 %0, {lo, hi}; }\n"
        : "=r"(r) : "f"(a), "f"(b));
    return r;
}

// ═════════════════════════════════════════��═════════════════════════════
// D=64 Br=64: pipelined K double-buffer + V async overlap
// Smem: Q/V[8KB] + K_A[8KB] + K_B[8KB] = 24KB
// Pipeline: V[t] load → QK^T → K[t+1] prefetch → wait V → softmax → P@V
// ════════════════════════════════════════════��══════════════════════════
template <bool CAUSAL>
__global__ void __launch_bounds__(128)
flash_attn_mma_d64(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const int N, const float scale
) {
    constexpr int Br = 64, Bc = 64, D = 64;
    constexpr int BLOCK = 128, WARP_ROWS = 16;
    constexpr int K_STEPS = 4, N_TILES = 8, ACC = 32;
    constexpr int TILE_CHUNKS = Bc * (D / 8);

    const int bh = blockIdx.x;
    const int q_start = blockIdx.y * Br;
    if (q_start >= N) return;

    const int tid = threadIdx.x, warp_id = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const __nv_bfloat16* Q_ptr = Q + (size_t)bh * N * D;
    const __nv_bfloat16* K_ptr = K + (size_t)bh * N * D;
    const __nv_bfloat16* V_ptr = V + (size_t)bh * N * D;
    __nv_bfloat16*       O_ptr = O + (size_t)bh * N * D;

    // Smem: [Q/V: Br*D = 8KB] [K_A: Bc*D = 8KB] [K_B: 8KB] = 24KB
    extern __shared__ char smem_raw[];
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_smem_a = q_smem + Br * D;
    __nv_bfloat16* k_smem_b = k_smem_a + Bc * D;

    // ─── Load Q to smem via cp.async ─────────────────────────────────
    constexpr int Q_CHUNKS = Br * (D / 8);
    for (int ci = tid; ci < Q_CHUNKS; ci += BLOCK) {
        int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
        int gr = q_start + row;
        bk::cp_async_128_zfill(&q_smem[swz<D>(row, col8)], &Q_ptr[gr * D + col8], gr < N);
    }
    bk::cp_async_commit();
    bk::cp_async_wait_all();
    __syncthreads();

    // ─── Q → registers ──────────────────────────────────────────────
    float o_acc[ACC];
    #pragma unroll
    for (int i = 0; i < ACC; i++) o_acc[i] = 0.0f;

    const int mma_row_a = lane / 4, mma_row_b = mma_row_a + 8;
    const int global_row_a = q_start + warp_id * WARP_ROWS + mma_row_a;
    const int global_row_b = q_start + warp_id * WARP_ROWS + mma_row_b;
    float m_a = -FLT_MAX, l_a = 0.0f, m_b = -FLT_MAX, l_b = 0.0f;

    uint32_t q_frag[K_STEPS * 4];
    #pragma unroll
    for (int k = 0; k < K_STEPS; k++) {
        int sub = lane / 8, sub_row = lane % 8;
        int row = warp_id * WARP_ROWS + (sub < 2 ? sub_row : 8 + sub_row);
        int col = k * 16 + (sub % 2) * 8;
        bk::ldmatrix_x4_mma(q_frag[k*4], q_frag[k*4+1], q_frag[k*4+2], q_frag[k*4+3],
            &q_smem[swz<D>(row, col)]);
    }

    // Pre-scale Q fragments by 1/sqrt(D) in BF16 (once, not per iteration)
    uint32_t scale_packed = pack_bf16_pair(scale, scale);
    #pragma unroll
    for (int i = 0; i < K_STEPS * 4; i++) {
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(q_frag[i]) : "r"(q_frag[i]), "r"(scale_packed));
    }
    __syncthreads();

    // V reuses Q smem (Q already in registers)
    __nv_bfloat16* v_smem = q_smem;

    const int kv_tiles = (N + Bc - 1) / Bc;
    const int kv_end = CAUSAL ? min(kv_tiles, (q_start + Br + Bc - 1) / Bc) : kv_tiles;
    int k_phase = 0;

    // ─── Prologue: prefetch K[0] ─────────────────────────────────────
    for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
        int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
        bk::cp_async_128_zfill(&k_smem_a[swz<D>(row, col8)], &K_ptr[row * D + col8], row < N);
    }
    bk::cp_async_commit();
    bk::cp_async_wait_all();
    __syncthreads();

    // ─── Main loop (pipelined) ───────────────────────────────────────
    for (int kv_t = 0; kv_t < kv_end; kv_t++) {
        const int kv_start = kv_t * Bc;
        __nv_bfloat16* k_cur = k_phase ? k_smem_b : k_smem_a;
        const bool has_next = (kv_t + 1 < kv_end);

        // ── V async load (overlaps with QK^T compute) ────────────────
        for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
            int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
            int gr = kv_start + row;
            bk::cp_async_128_zfill(&v_smem[swz<D>(row, col8)], &V_ptr[gr * D + col8], gr < N);
        }
        bk::cp_async_commit();

        // ── S = Q_regs @ K^T (K already in k_cur from double-buffer) ─
        float s_acc[ACC];
        #pragma unroll
        for (int i = 0; i < ACC; i++) s_acc[i] = 0.0f;

        #pragma unroll
        for (int k = 0; k < K_STEPS; k++) {
            uint32_t a0 = q_frag[k*4], a1 = q_frag[k*4+1], a2 = q_frag[k*4+2], a3 = q_frag[k*4+3];
            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                uint32_t b0, b1;
                { int mat=(lane>>3)&1; bk::ldmatrix_x2(b0,b1,&k_cur[swz<D>(n*8+(lane&7), k*16+mat*8)]); }
                bk::mma_m16n8k16_bf16_nv(s_acc[n*4],s_acc[n*4+1],s_acc[n*4+2],s_acc[n*4+3],
                    a0,a1,a2,a3,b0,b1, s_acc[n*4],s_acc[n*4+1],s_acc[n*4+2],s_acc[n*4+3]);
            }
        }

        // ── K[next] prefetch ─────────────────────────────��────────────
        if (has_next) {
            int nxt = (kv_t + 1) * Bc;
            __nv_bfloat16* k_nxt = k_phase ? k_smem_a : k_smem_b;
            for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
                int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
                int gr = nxt + row;
                bk::cp_async_128_zfill(&k_nxt[swz<D>(row, col8)], &K_ptr[gr * D + col8], gr < N);
            }
            bk::cp_async_commit();
        }

        // ── Wait for V ──────────────────────────────────���────────────
        if (has_next) bk::cp_async_wait<1>();
        else          bk::cp_async_wait<0>();
        __syncthreads();

        // ── Softmax ─────────────────────────────────��────────────────
        float rmax_a = -FLT_MAX, rmax_b = -FLT_MAX;
        #pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            int col0 = n*8+(lane%4)*2, kv0 = kv_start+col0, kv1 = kv0+1;
            if (kv0>=N||(CAUSAL&&kv0>global_row_a)) s_acc[n*4]=-FLT_MAX;
            if (kv1>=N||(CAUSAL&&kv1>global_row_a)) s_acc[n*4+1]=-FLT_MAX;
            if (kv0>=N||(CAUSAL&&kv0>global_row_b)) s_acc[n*4+2]=-FLT_MAX;
            if (kv1>=N||(CAUSAL&&kv1>global_row_b)) s_acc[n*4+3]=-FLT_MAX;
            if (global_row_a>=N){s_acc[n*4]=-FLT_MAX;s_acc[n*4+1]=-FLT_MAX;}
            if (global_row_b>=N){s_acc[n*4+2]=-FLT_MAX;s_acc[n*4+3]=-FLT_MAX;}
            rmax_a=fmaxf(rmax_a,fmaxf(s_acc[n*4],s_acc[n*4+1]));
            rmax_b=fmaxf(rmax_b,fmaxf(s_acc[n*4+2],s_acc[n*4+3]));
        }
        #pragma unroll
        for (int d=1;d<4;d<<=1){rmax_a=fmaxf(rmax_a,__shfl_xor_sync(0xffffffff,rmax_a,d));rmax_b=fmaxf(rmax_b,__shfl_xor_sync(0xffffffff,rmax_b,d));}

        float m_new_a=fmaxf(m_a,rmax_a), m_new_b=fmaxf(m_b,rmax_b);
        float sc_a=exp2f(m_a-m_new_a), sc_b=exp2f(m_b-m_new_b);
        #pragma unroll
        for(int i=0;i<ACC;i+=4){o_acc[i]*=sc_a;o_acc[i+1]*=sc_a;o_acc[i+2]*=sc_b;o_acc[i+3]*=sc_b;}
        l_a*=sc_a;l_b*=sc_b;m_a=m_new_a;m_b=m_new_b;

        float lt_a=0,lt_b=0;
        #pragma unroll
        for(int n=0;n<N_TILES;n++){
            float p0=exp2f(s_acc[n*4]-m_new_a);
            float p1=exp2f(s_acc[n*4+1]-m_new_a);
            float p2=exp2f(s_acc[n*4+2]-m_new_b);
            float p3=exp2f(s_acc[n*4+3]-m_new_b);
            lt_a+=p0+p1;lt_b+=p2+p3;s_acc[n*4]=p0;s_acc[n*4+1]=p1;s_acc[n*4+2]=p2;s_acc[n*4+3]=p3;
        }
        #pragma unroll
        for(int d=1;d<4;d<<=1){lt_a+=__shfl_xor_sync(0xffffffff,lt_a,d);lt_b+=__shfl_xor_sync(0xffffffff,lt_b,d);}
        l_a+=lt_a;l_b+=lt_b;

        // ── O += P @ V ───────────────────────────────────────────────
        #pragma unroll
        for(int k=0;k<K_STEPS;k++){
            int nt0=k*2,nt1=k*2+1;
            uint32_t pa0=pack_bf16_pair(s_acc[nt0*4],s_acc[nt0*4+1]),pa1=pack_bf16_pair(s_acc[nt0*4+2],s_acc[nt0*4+3]);
            uint32_t pa2=pack_bf16_pair(s_acc[nt1*4],s_acc[nt1*4+1]),pa3=pack_bf16_pair(s_acc[nt1*4+2],s_acc[nt1*4+3]);
            #pragma unroll
            for(int n=0;n<N_TILES;n++){
                uint32_t vb0,vb1;
                {int mat=(lane>>3)&1;bk::ldmatrix_x2_trans(vb0,vb1,&v_smem[swz<D>(k*16+mat*8+(lane&7),n*8)]);}
                bk::mma_m16n8k16_bf16_nv(o_acc[n*4],o_acc[n*4+1],o_acc[n*4+2],o_acc[n*4+3],
                    pa0,pa1,pa2,pa3,vb0,vb1, o_acc[n*4],o_acc[n*4+1],o_acc[n*4+2],o_acc[n*4+3]);
            }
        }

        // ── Wait K[next], ensure V consumed ──────────────────────────
        if (has_next) bk::cp_async_wait<0>();
        __syncthreads();
        k_phase ^= 1;
    }

    // ─── Final normalize + store ─────────────────────────────────────
    float inv_a=(l_a>0)?(1.0f/l_a):0.0f, inv_b=(l_b>0)?(1.0f/l_b):0.0f;
    #pragma unroll
    for(int n=0;n<N_TILES;n++){
        int col0=n*8+(lane%4)*2;
        if(global_row_a<N&&col0+1<D){
            uint32_t p=pack_bf16_pair(o_acc[n*4]*inv_a, o_acc[n*4+1]*inv_a);
            *reinterpret_cast<uint32_t*>(&O_ptr[global_row_a*D+col0])=p;
        }
        if(global_row_b<N&&col0+1<D){
            uint32_t p=pack_bf16_pair(o_acc[n*4+2]*inv_b, o_acc[n*4+3]*inv_b);
            *reinterpret_cast<uint32_t*>(&O_ptr[global_row_b*D+col0])=p;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// D=64 Br=64 Bc=128: pipelined single-buffer K + V async overlap
// V load overlaps QK^T; K[next] loads into same buffer during P@V.
// Smem: [Q/V: 16KB] [K: 16KB] = 32KB (no extra smem vs unpipelined)
// ═══════════════════════════════════════════════════════════════════════
template <bool CAUSAL>
__global__ void __launch_bounds__(128)
flash_attn_mma_d64_bc128(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const int N, const float scale
) {
    constexpr int Br = 64, Bc = 128, D = 64;
    constexpr int BLOCK = 128, WARP_ROWS = 16;
    constexpr int QK_K = D / 16;      // 4  — k-steps for Q@K^T
    constexpr int QK_N = Bc / 8;      // 16 — n-tiles for Q@K^T
    constexpr int S_ACC = QK_N * 4;   // 64
    constexpr int PV_K = Bc / 16;     // 8  — k-steps for P@V
    constexpr int PV_N = D / 8;       // 8  — n-tiles for P@V
    constexpr int O_ACC = PV_N * 4;   // 32
    constexpr int V_BUF = Bc * D;     // 8192 bf16 = 16KB
    constexpr int KV_CHUNKS = Bc * (D / 8); // 1024

    const int bh = blockIdx.x;
    const int q_start = blockIdx.y * Br;
    if (q_start >= N) return;

    const int tid = threadIdx.x, warp_id = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const __nv_bfloat16* Q_ptr = Q + (size_t)bh * N * D;
    const __nv_bfloat16* K_ptr = K + (size_t)bh * N * D;
    const __nv_bfloat16* V_ptr = V + (size_t)bh * N * D;
    __nv_bfloat16*       O_ptr = O + (size_t)bh * N * D;

    // Smem: [Q/V buffer: V_BUF = 16KB] [K buffer: Bc*D = 16KB] = 32KB
    extern __shared__ char smem_raw[];
    __nv_bfloat16* qv_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_smem  = qv_smem + V_BUF;

    // ─── Load Q to smem via cp.async ────────────────────────────────
    constexpr int Q_CHUNKS = Br * (D / 8);
    for (int ci = tid; ci < Q_CHUNKS; ci += BLOCK) {
        int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
        int gr = q_start + row;
        bk::cp_async_128_zfill(&qv_smem[swz<D>(row, col8)], &Q_ptr[gr * D + col8], gr < N);
    }
    bk::cp_async_commit();
    bk::cp_async_wait_all();
    __syncthreads();

    // ─── Q → registers ──────────────────────────────────────────────
    uint32_t q_frag[QK_K * 4];
    #pragma unroll
    for (int k = 0; k < QK_K; k++) {
        int sub = lane / 8, sub_row = lane % 8;
        int row = warp_id * WARP_ROWS + (sub < 2 ? sub_row : 8 + sub_row);
        int col = k * 16 + (sub % 2) * 8;
        bk::ldmatrix_x4_mma(q_frag[k*4], q_frag[k*4+1], q_frag[k*4+2], q_frag[k*4+3],
            &qv_smem[swz<D>(row, col)]);
    }

    // Pre-scale Q in BF16
    uint32_t scale_packed = pack_bf16_pair(scale, scale);
    #pragma unroll
    for (int i = 0; i < QK_K * 4; i++) {
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(q_frag[i]) : "r"(q_frag[i]), "r"(scale_packed));
    }
    __syncthreads();

    // V reuses qv_smem (Q already in registers)
    __nv_bfloat16* v_smem = qv_smem;

    float o_acc[O_ACC];
    #pragma unroll
    for (int i = 0; i < O_ACC; i++) o_acc[i] = 0.0f;

    const int mma_row_a = lane / 4, mma_row_b = mma_row_a + 8;
    const int global_row_a = q_start + warp_id * WARP_ROWS + mma_row_a;
    const int global_row_b = q_start + warp_id * WARP_ROWS + mma_row_b;
    float m_a = -FLT_MAX, l_a = 0.0f, m_b = -FLT_MAX, l_b = 0.0f;

    const int kv_tiles = (N + Bc - 1) / Bc;
    const int kv_end = CAUSAL ? min(kv_tiles, (q_start + Br + Bc - 1) / Bc) : kv_tiles;

    // ─── Prologue: prefetch K[0] into k_smem ────────────────────────
    for (int ci = tid; ci < KV_CHUNKS; ci += BLOCK) {
        int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
        bk::cp_async_128_zfill(&k_smem[swz<D>(row, col8)], &K_ptr[row * D + col8], row < N);
    }
    bk::cp_async_commit();
    bk::cp_async_wait_all();
    __syncthreads();

    // ─── Main loop (pipelined) ──────────────────────────────────────
    for (int kv_t = 0; kv_t < kv_end; kv_t++) {
        const int kv_start = kv_t * Bc;
        const bool has_next = (kv_t + 1 < kv_end);

        // ── V async load (overlaps with QK^T compute) ───────────────
        for (int ci = tid; ci < KV_CHUNKS; ci += BLOCK) {
            int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
            int gr = kv_start + row;
            bk::cp_async_128_zfill(&v_smem[swz<D>(row, col8)], &V_ptr[gr * D + col8], gr < N);
        }
        bk::cp_async_commit();

        // ── S = Q_regs @ K^T (K already in k_smem, V loads in bg) ──
        float s_acc[S_ACC];
        #pragma unroll
        for (int i = 0; i < S_ACC; i++) s_acc[i] = 0.0f;

        #pragma unroll
        for (int k = 0; k < QK_K; k++) {
            uint32_t a0 = q_frag[k*4], a1 = q_frag[k*4+1], a2 = q_frag[k*4+2], a3 = q_frag[k*4+3];
            #pragma unroll
            for (int n = 0; n < QK_N; n++) {
                uint32_t b0, b1;
                { int mat=(lane>>3)&1; bk::ldmatrix_x2(b0,b1,&k_smem[swz<D>(n*8+(lane&7), k*16+mat*8)]); }
                bk::mma_m16n8k16_bf16_nv(s_acc[n*4],s_acc[n*4+1],s_acc[n*4+2],s_acc[n*4+3],
                    a0,a1,a2,a3,b0,b1, s_acc[n*4],s_acc[n*4+1],s_acc[n*4+2],s_acc[n*4+3]);
            }
        }

        // ── Wait V, sync (QK^T done → k_smem is free) ──────────────
        bk::cp_async_wait<0>();
        __syncthreads();

        // ── K[next] prefetch into k_smem (reuse same buffer) ────────
        if (has_next) {
            int nxt = (kv_t + 1) * Bc;
            for (int ci = tid; ci < KV_CHUNKS; ci += BLOCK) {
                int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
                int gr = nxt + row;
                bk::cp_async_128_zfill(&k_smem[swz<D>(row, col8)], &K_ptr[gr * D + col8], gr < N);
            }
            bk::cp_async_commit();
        }

        // ── Softmax (K[next] loads in background) ───────────────────
        float rmax_a = -FLT_MAX, rmax_b = -FLT_MAX;
        #pragma unroll
        for (int n = 0; n < QK_N; n++) {
            int col0 = n*8+(lane%4)*2, kv0 = kv_start+col0, kv1 = kv0+1;
            if (kv0>=N||(CAUSAL&&kv0>global_row_a)) s_acc[n*4]=-FLT_MAX;
            if (kv1>=N||(CAUSAL&&kv1>global_row_a)) s_acc[n*4+1]=-FLT_MAX;
            if (kv0>=N||(CAUSAL&&kv0>global_row_b)) s_acc[n*4+2]=-FLT_MAX;
            if (kv1>=N||(CAUSAL&&kv1>global_row_b)) s_acc[n*4+3]=-FLT_MAX;
            if (global_row_a>=N){s_acc[n*4]=-FLT_MAX;s_acc[n*4+1]=-FLT_MAX;}
            if (global_row_b>=N){s_acc[n*4+2]=-FLT_MAX;s_acc[n*4+3]=-FLT_MAX;}
            rmax_a=fmaxf(rmax_a,fmaxf(s_acc[n*4],s_acc[n*4+1]));
            rmax_b=fmaxf(rmax_b,fmaxf(s_acc[n*4+2],s_acc[n*4+3]));
        }
        #pragma unroll
        for (int d=1;d<4;d<<=1){rmax_a=fmaxf(rmax_a,__shfl_xor_sync(0xffffffff,rmax_a,d));rmax_b=fmaxf(rmax_b,__shfl_xor_sync(0xffffffff,rmax_b,d));}

        float m_new_a=fmaxf(m_a,rmax_a), m_new_b=fmaxf(m_b,rmax_b);
        float sc_a=__expf(m_a-m_new_a), sc_b=__expf(m_b-m_new_b);
        #pragma unroll
        for(int i=0;i<O_ACC;i+=4){o_acc[i]*=sc_a;o_acc[i+1]*=sc_a;o_acc[i+2]*=sc_b;o_acc[i+3]*=sc_b;}
        l_a*=sc_a;l_b*=sc_b;m_a=m_new_a;m_b=m_new_b;

        float lt_a=0,lt_b=0;
        #pragma unroll
        for(int n=0;n<QK_N;n++){
            float p0=__expf(s_acc[n*4]-m_new_a);
            float p1=__expf(s_acc[n*4+1]-m_new_a);
            float p2=__expf(s_acc[n*4+2]-m_new_b);
            float p3=__expf(s_acc[n*4+3]-m_new_b);
            lt_a+=p0+p1;lt_b+=p2+p3;s_acc[n*4]=p0;s_acc[n*4+1]=p1;s_acc[n*4+2]=p2;s_acc[n*4+3]=p3;
        }
        #pragma unroll
        for(int d=1;d<4;d<<=1){lt_a+=__shfl_xor_sync(0xffffffff,lt_a,d);lt_b+=__shfl_xor_sync(0xffffffff,lt_b,d);}
        l_a+=lt_a;l_b+=lt_b;

        // ── O += P @ V (K[next] still loading in background) ────────
        #pragma unroll
        for(int pk=0;pk<PV_K;pk++){
            int nt0=pk*2,nt1=pk*2+1;
            uint32_t pa0=pack_bf16_pair(s_acc[nt0*4],s_acc[nt0*4+1]),pa1=pack_bf16_pair(s_acc[nt0*4+2],s_acc[nt0*4+3]);
            uint32_t pa2=pack_bf16_pair(s_acc[nt1*4],s_acc[nt1*4+1]),pa3=pack_bf16_pair(s_acc[nt1*4+2],s_acc[nt1*4+3]);
            #pragma unroll
            for(int pn=0;pn<PV_N;pn++){
                uint32_t vb0,vb1;
                {int mat=(lane>>3)&1;bk::ldmatrix_x2_trans(vb0,vb1,&v_smem[swz<D>(pk*16+mat*8+(lane&7),pn*8)]);}
                bk::mma_m16n8k16_bf16_nv(o_acc[pn*4],o_acc[pn*4+1],o_acc[pn*4+2],o_acc[pn*4+3],
                    pa0,pa1,pa2,pa3,vb0,vb1, o_acc[pn*4],o_acc[pn*4+1],o_acc[pn*4+2],o_acc[pn*4+3]);
            }
        }

        // ── Wait K[next], ensure V consumed ─────────────────────────
        if (has_next) bk::cp_async_wait<0>();
        __syncthreads();
    }

    // ─── Final normalize + store ────────────────────────────────────
    float inv_a=(l_a>0)?(1.0f/l_a):0.0f, inv_b=(l_b>0)?(1.0f/l_b):0.0f;
    #pragma unroll
    for(int n=0;n<PV_N;n++){
        int col0=n*8+(lane%4)*2;
        if(global_row_a<N&&col0+1<D){
            uint32_t p=pack_bf16_pair(o_acc[n*4]*inv_a, o_acc[n*4+1]*inv_a);
            *reinterpret_cast<uint32_t*>(&O_ptr[global_row_a*D+col0])=p;
        }
        if(global_row_b<N&&col0+1<D){
            uint32_t p=pack_bf16_pair(o_acc[n*4+2]*inv_b, o_acc[n*4+3]*inv_b);
            *reinterpret_cast<uint32_t*>(&O_ptr[global_row_b*D+col0])=p;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// D=64 Br=128: non-causal, pipelined K double-buffer
// Each warp handles 32 Q rows = 2 MMA-M tiles.
// K fragment loaded once, reused for both MMA-M tiles.
// Smem: Q/V[16KB] + K_A[8KB] + K_B[8KB] = 32KB
// ═══════════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(128)
flash_attn_mma_d64_br128(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const int N, const float scale
) {
    constexpr int Br = 128, Bc = 64, D = 64;
    constexpr int BLOCK = 128, WARP_ROWS = 32;
    constexpr int K_STEPS = 4, N_TILES = 8;
    constexpr int MQ = 2;  // MMA-M tiles per warp
    constexpr int TILE_CHUNKS = Bc * (D / 8); // 512

    const int bh = blockIdx.x;
    const int q_start = blockIdx.y * Br;
    if (q_start >= N) return;

    const int tid = threadIdx.x, warp_id = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const __nv_bfloat16* Q_ptr = Q + (size_t)bh * N * D;
    const __nv_bfloat16* K_ptr = K + (size_t)bh * N * D;
    const __nv_bfloat16* V_ptr = V + (size_t)bh * N * D;
    __nv_bfloat16*       O_ptr = O + (size_t)bh * N * D;

    // Smem: [Q/V: Br*D = 16KB] [K_A: Bc*D = 8KB] [K_B: 8KB] = 32KB
    extern __shared__ char smem_raw[];
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_smem_a = q_smem + Br * D;
    __nv_bfloat16* k_smem_b = k_smem_a + Bc * D;

    // ─── Load Q to smem via cp.async (bypass registers) ─────────────
    constexpr int Q_CHUNKS = Br * (D / 8);
    for (int ci = tid; ci < Q_CHUNKS; ci += BLOCK) {
        int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
        int gr = q_start + row;
        bk::cp_async_128_zfill(&q_smem[swz<D>(row, col8)], &Q_ptr[gr * D + col8], gr < N);
    }
    bk::cp_async_commit();
    bk::cp_async_wait_all();
    __syncthreads();

    // ─── Q → registers: 2 MMA-M tiles × 4 k-steps × 4 regs = 32 ──
    const int q_base = warp_id * WARP_ROWS;
    uint32_t q_frag[MQ * K_STEPS * 4]; // 32 uint32
    #pragma unroll
    for (int mq = 0; mq < MQ; mq++) {
        #pragma unroll
        for (int k = 0; k < K_STEPS; k++) {
            int sub = lane / 8, sub_row = lane % 8;
            int row = q_base + mq * 16 + (sub < 2 ? sub_row : 8 + sub_row);
            int col = k * 16 + (sub % 2) * 8;
            int fi = (mq * K_STEPS + k) * 4;
            bk::ldmatrix_x4_mma(q_frag[fi], q_frag[fi+1], q_frag[fi+2], q_frag[fi+3],
                &q_smem[swz<D>(row, col)]);
        }
    }

    // Pre-scale Q fragments by 1/sqrt(D) in BF16 (once, not per iteration)
    uint32_t scale_packed = pack_bf16_pair(scale, scale);
    #pragma unroll
    for (int i = 0; i < MQ * K_STEPS * 4; i++) {
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(q_frag[i]) : "r"(q_frag[i]), "r"(scale_packed));
    }
    __syncthreads();

    // V reuses first Bc*D of q_smem (only need 8KB of the 16KB)
    __nv_bfloat16* v_smem = q_smem;

    // ─── Accumulators: 2 MMA-M tiles × 8 n-tiles × 4 = 64 each ────
    float o_acc[MQ * N_TILES * 4];
    #pragma unroll
    for (int i = 0; i < MQ * N_TILES * 4; i++) o_acc[i] = 0.0f;

    const int mma_row_base = lane / 4; // 0..7 within a 16-row MMA tile
    // Per mq tile: row_a = q_start + warp_id*32 + mq*16 + mma_row_base
    //              row_b = row_a + 8

    float m_vals[MQ * 2], l_vals[MQ * 2]; // [mq0_a, mq0_b, mq1_a, mq1_b]
    #pragma unroll
    for (int i = 0; i < MQ * 2; i++) { m_vals[i] = -FLT_MAX; l_vals[i] = 0.0f; }

    const int kv_tiles = (N + Bc - 1) / Bc;
    int k_phase = 0;

    // ─── Prologue: prefetch K[0] ────────────────────────────────────
    for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
        int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
        bk::cp_async_128_zfill(&k_smem_a[swz<D>(row, col8)], &K_ptr[row * D + col8], row < N);
    }
    bk::cp_async_commit();
    bk::cp_async_wait_all();
    __syncthreads();

    // ─── Main loop ──────────────────────────────────────────────────
    for (int kv_t = 0; kv_t < kv_tiles; kv_t++) {
        const int kv_start = kv_t * Bc;
        __nv_bfloat16* k_cur = k_phase ? k_smem_b : k_smem_a;
        const bool has_next = (kv_t + 1 < kv_tiles);

        // ── V async load ────────────────────────────────────────────
        for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
            int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
            int gr = kv_start + row;
            bk::cp_async_128_zfill(&v_smem[swz<D>(row, col8)], &V_ptr[gr * D + col8], gr < N);
        }
        bk::cp_async_commit();

        // ── S = Q_regs @ K^T (64 MMAs: 2 mq × 4 k × 8 n) ─────────
        float s_acc[MQ * N_TILES * 4];
        #pragma unroll
        for (int i = 0; i < MQ * N_TILES * 4; i++) s_acc[i] = 0.0f;

        #pragma unroll
        for (int k = 0; k < K_STEPS; k++) {
            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                // Load K fragment — shared by both MMA-M tiles
                uint32_t b0, b1;
                { int mat=(lane>>3)&1; bk::ldmatrix_x2(b0,b1,&k_cur[swz<D>(n*8+(lane&7), k*16+mat*8)]); }

                #pragma unroll
                for (int mq = 0; mq < MQ; mq++) {
                    int qi = (mq * K_STEPS + k) * 4;
                    int si = (mq * N_TILES + n) * 4;
                    bk::mma_m16n8k16_bf16_nv(
                        s_acc[si], s_acc[si+1], s_acc[si+2], s_acc[si+3],
                        q_frag[qi], q_frag[qi+1], q_frag[qi+2], q_frag[qi+3],
                        b0, b1,
                        s_acc[si], s_acc[si+1], s_acc[si+2], s_acc[si+3]);
                }
            }
        }

        // ── K[next] prefetch ────────────────────────────────────────
        if (has_next) {
            int nxt = (kv_t + 1) * Bc;
            __nv_bfloat16* k_nxt = k_phase ? k_smem_a : k_smem_b;
            for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
                int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
                int gr = nxt + row;
                bk::cp_async_128_zfill(&k_nxt[swz<D>(row, col8)], &K_ptr[gr * D + col8], gr < N);
            }
            bk::cp_async_commit();
        }

        // ── Wait for V ──────────────────────────────────────────────
        if (has_next) bk::cp_async_wait<1>();
        else          bk::cp_async_wait<0>();
        __syncthreads();

        // ── Softmax (per MMA-M tile) ────────────────────────────────
        #pragma unroll
        for (int mq = 0; mq < MQ; mq++) {
            int gr_a = q_start + warp_id * WARP_ROWS + mq * 16 + mma_row_base;
            int gr_b = gr_a + 8;
            int mi = mq * 2;

            float rmax_a = -FLT_MAX, rmax_b = -FLT_MAX;
            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                int si = (mq * N_TILES + n) * 4;
                int col0 = n * 8 + (lane % 4) * 2;
                int kv0 = kv_start + col0, kv1 = kv0 + 1;
                if (kv0 >= N) s_acc[si] = -FLT_MAX;
                if (kv1 >= N) s_acc[si+1] = -FLT_MAX;
                if (kv0 >= N) s_acc[si+2] = -FLT_MAX;
                if (kv1 >= N) s_acc[si+3] = -FLT_MAX;
                if (gr_a >= N) { s_acc[si] = -FLT_MAX; s_acc[si+1] = -FLT_MAX; }
                if (gr_b >= N) { s_acc[si+2] = -FLT_MAX; s_acc[si+3] = -FLT_MAX; }
                rmax_a = fmaxf(rmax_a, fmaxf(s_acc[si], s_acc[si+1]));
                rmax_b = fmaxf(rmax_b, fmaxf(s_acc[si+2], s_acc[si+3]));
            }
            #pragma unroll
            for (int d = 1; d < 4; d <<= 1) {
                rmax_a = fmaxf(rmax_a, __shfl_xor_sync(0xffffffff, rmax_a, d));
                rmax_b = fmaxf(rmax_b, __shfl_xor_sync(0xffffffff, rmax_b, d));
            }

            float m_new_a = fmaxf(m_vals[mi], rmax_a);
            float m_new_b = fmaxf(m_vals[mi+1], rmax_b);
            float sc_a = __expf(m_vals[mi] - m_new_a);
            float sc_b = __expf(m_vals[mi+1] - m_new_b);

            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                int oi = (mq * N_TILES + n) * 4;
                o_acc[oi] *= sc_a; o_acc[oi+1] *= sc_a;
                o_acc[oi+2] *= sc_b; o_acc[oi+3] *= sc_b;
            }
            l_vals[mi] *= sc_a; l_vals[mi+1] *= sc_b;
            m_vals[mi] = m_new_a; m_vals[mi+1] = m_new_b;

            float lt_a = 0, lt_b = 0;
            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                int si = (mq * N_TILES + n) * 4;
                float p0 = __expf(s_acc[si]   - m_new_a);
                float p1 = __expf(s_acc[si+1] - m_new_a);
                float p2 = __expf(s_acc[si+2] - m_new_b);
                float p3 = __expf(s_acc[si+3] - m_new_b);
                lt_a += p0 + p1; lt_b += p2 + p3;
                s_acc[si] = p0; s_acc[si+1] = p1; s_acc[si+2] = p2; s_acc[si+3] = p3;
            }
            #pragma unroll
            for (int d = 1; d < 4; d <<= 1) {
                lt_a += __shfl_xor_sync(0xffffffff, lt_a, d);
                lt_b += __shfl_xor_sync(0xffffffff, lt_b, d);
            }
            l_vals[mi] += lt_a; l_vals[mi+1] += lt_b;
        }

        // ── O += P @ V (64 MMAs: 4 pk × 8 pn, V shared across mq) ─
        #pragma unroll
        for (int pk = 0; pk < K_STEPS; pk++) {
            // Pack P fragments for both mq tiles
            uint32_t pa[MQ][4];
            #pragma unroll
            for (int mq = 0; mq < MQ; mq++) {
                int nt0 = pk * 2, nt1 = pk * 2 + 1;
                int s0 = (mq * N_TILES + nt0) * 4, s1 = (mq * N_TILES + nt1) * 4;
                pa[mq][0] = pack_bf16_pair(s_acc[s0],   s_acc[s0+1]);
                pa[mq][1] = pack_bf16_pair(s_acc[s0+2], s_acc[s0+3]);
                pa[mq][2] = pack_bf16_pair(s_acc[s1],   s_acc[s1+1]);
                pa[mq][3] = pack_bf16_pair(s_acc[s1+2], s_acc[s1+3]);
            }

            #pragma unroll
            for (int pn = 0; pn < N_TILES; pn++) {
                uint32_t vb0, vb1;
                { int mat=(lane>>3)&1; bk::ldmatrix_x2_trans(vb0,vb1,&v_smem[swz<D>(pk*16+mat*8+(lane&7), pn*8)]); }
                #pragma unroll
                for (int mq = 0; mq < MQ; mq++) {
                    int oi = (mq * N_TILES + pn) * 4;
                    bk::mma_m16n8k16_bf16_nv(
                        o_acc[oi], o_acc[oi+1], o_acc[oi+2], o_acc[oi+3],
                        pa[mq][0], pa[mq][1], pa[mq][2], pa[mq][3],
                        vb0, vb1,
                        o_acc[oi], o_acc[oi+1], o_acc[oi+2], o_acc[oi+3]);
                }
            }
        }

        // ── Wait K[next], ensure V consumed ─────────────────────────
        if (has_next) bk::cp_async_wait<0>();
        __syncthreads();
        k_phase ^= 1;
    }

    // ─── Final normalize + store ────────────────────────────────────
    #pragma unroll
    for (int mq = 0; mq < MQ; mq++) {
        int gr_a = q_start + warp_id * WARP_ROWS + mq * 16 + mma_row_base;
        int gr_b = gr_a + 8;
        float inv_a = (l_vals[mq*2] > 0) ? (1.0f / l_vals[mq*2]) : 0.0f;
        float inv_b = (l_vals[mq*2+1] > 0) ? (1.0f / l_vals[mq*2+1]) : 0.0f;
        #pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            int oi = (mq * N_TILES + n) * 4;
            int col0 = n * 8 + (lane % 4) * 2;
            if (gr_a < N && col0 + 1 < D) {
                uint32_t p = pack_bf16_pair(o_acc[oi] * inv_a, o_acc[oi+1] * inv_a);
                *reinterpret_cast<uint32_t*>(&O_ptr[gr_a*D+col0]) = p;
            }
            if (gr_b < N && col0 + 1 < D) {
                uint32_t p = pack_bf16_pair(o_acc[oi+2] * inv_b, o_acc[oi+3] * inv_b);
                *reinterpret_cast<uint32_t*>(&O_ptr[gr_b*D+col0]) = p;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// D=128 MMA kernel: pipelined K double-buffer
// Smem: V[16KB] + K_A[16KB] + K_B[16KB] = 48KB
// ═══════════════════════════════════════════════════════════════════════
template <bool CAUSAL>
__global__ void __launch_bounds__(128)
flash_attn_mma_d128(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const int N, const float scale
) {
    constexpr int Br = 64, Bc = 64, D = 128;
    constexpr int BLOCK = 128, WARP_ROWS = 16;
    constexpr int QKT_K = 8, QKT_N = 8, S_ACC = 32;
    constexpr int PV_K = 4, PV_N = 16, O_ACC = 64;
    constexpr int TILE_CHUNKS = Bc * (D / 8);

    const int bh = blockIdx.x;
    const int q_start = blockIdx.y * Br;
    if (q_start >= N) return;

    const int tid = threadIdx.x, warp_id = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const __nv_bfloat16* Q_ptr = Q + (size_t)bh * N * D;
    const __nv_bfloat16* K_ptr = K + (size_t)bh * N * D;
    const __nv_bfloat16* V_ptr = V + (size_t)bh * N * D;
    __nv_bfloat16*       O_ptr = O + (size_t)bh * N * D;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_smem_a = q_smem + Br * D;
    __nv_bfloat16* k_smem_b = k_smem_a + Bc * D;

    // ─── Load Q to smem via cp.async (bypass registers) ─────────────
    constexpr int Q_CHUNKS = Br * (D / 8);
    for (int ci = tid; ci < Q_CHUNKS; ci += BLOCK) {
        int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
        int gr = q_start + row;
        bk::cp_async_128_zfill(&q_smem[swz<D>(row, col8)], &Q_ptr[gr * D + col8], gr < N);
    }
    bk::cp_async_commit();
    bk::cp_async_wait_all();
    __syncthreads();

    // ─── Q → registers + BF16 pre-scale ─────────────────────────────
    uint32_t q_frag[QKT_K * 4];
    #pragma unroll
    for (int k = 0; k < QKT_K; k++) {
        int sub = lane / 8, sub_row = lane % 8;
        int row = warp_id * WARP_ROWS + (sub < 2 ? sub_row : 8 + sub_row);
        int col = k * 16 + (sub % 2) * 8;
        bk::ldmatrix_x4_mma(q_frag[k*4], q_frag[k*4+1], q_frag[k*4+2], q_frag[k*4+3],
            &q_smem[swz<D>(row, col)]);
    }

    uint32_t scale_packed = pack_bf16_pair(scale, scale);
    #pragma unroll
    for (int i = 0; i < QKT_K * 4; i++) {
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(q_frag[i]) : "r"(q_frag[i]), "r"(scale_packed));
    }
    __syncthreads();

    __nv_bfloat16* v_smem = q_smem;

    float o_acc[O_ACC];
    #pragma unroll
    for (int i = 0; i < O_ACC; i++) o_acc[i] = 0.0f;

    const int mma_row_a = lane / 4, mma_row_b = mma_row_a + 8;
    const int global_row_a = q_start + warp_id * WARP_ROWS + mma_row_a;
    const int global_row_b = q_start + warp_id * WARP_ROWS + mma_row_b;
    float m_a = -FLT_MAX, l_a = 0, m_b = -FLT_MAX, l_b = 0;

    const int kv_tiles = (N + Bc - 1) / Bc;
    const int kv_end = CAUSAL ? min(kv_tiles, (q_start + Br + Bc - 1) / Bc) : kv_tiles;
    int k_phase = 0;

    for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
        int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
        bk::cp_async_128_zfill(&k_smem_a[swz<D>(row, col8)], &K_ptr[row * D + col8], row < N);
    }
    bk::cp_async_commit(); bk::cp_async_wait_all(); __syncthreads();

    for (int kv_t = 0; kv_t < kv_end; kv_t++) {
        const int kv_start = kv_t * Bc;
        __nv_bfloat16* k_cur = k_phase ? k_smem_b : k_smem_a;
        const bool has_next = (kv_t + 1 < kv_end);

        for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
            int row = ci / (D / 8), col8 = (ci % (D / 8)) * 8;
            int gr = kv_start + row;
            bk::cp_async_128_zfill(&v_smem[swz<D>(row, col8)], &V_ptr[gr * D + col8], gr < N);
        }
        bk::cp_async_commit();

        float s_acc[S_ACC];
        #pragma unroll
        for (int i = 0; i < S_ACC; i++) s_acc[i] = 0.0f;

        #pragma unroll
        for (int k = 0; k < QKT_K; k++) {
            uint32_t a0=q_frag[k*4],a1=q_frag[k*4+1],a2=q_frag[k*4+2],a3=q_frag[k*4+3];
            #pragma unroll
            for (int n = 0; n < QKT_N; n++) {
                uint32_t b0,b1;
                {int mat=(lane>>3)&1;bk::ldmatrix_x2(b0,b1,&k_cur[swz<D>(n*8+(lane&7),k*16+mat*8)]);}
                bk::mma_m16n8k16_bf16_nv(s_acc[n*4],s_acc[n*4+1],s_acc[n*4+2],s_acc[n*4+3],
                    a0,a1,a2,a3,b0,b1,s_acc[n*4],s_acc[n*4+1],s_acc[n*4+2],s_acc[n*4+3]);
            }
        }

        if (has_next) {
            int nxt = (kv_t+1)*Bc;
            __nv_bfloat16* k_nxt = k_phase ? k_smem_a : k_smem_b;
            for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
                int row = ci/(D/8), col8 = (ci%(D/8))*8; int gr = nxt+row;
                bk::cp_async_128_zfill(&k_nxt[swz<D>(row,col8)],&K_ptr[gr*D+col8],gr<N);
            }
            bk::cp_async_commit();
        }

        if (has_next) bk::cp_async_wait<1>(); else bk::cp_async_wait<0>();
        __syncthreads();

        float rmax_a=-FLT_MAX,rmax_b=-FLT_MAX;
        #pragma unroll
        for(int n=0;n<QKT_N;n++){
            int si=n*4,col0=n*8+(lane%4)*2,kv0=kv_start+col0,kv1=kv0+1;
            if(kv0>=N||(CAUSAL&&kv0>global_row_a))s_acc[si]=-FLT_MAX;
            if(kv1>=N||(CAUSAL&&kv1>global_row_a))s_acc[si+1]=-FLT_MAX;
            if(kv0>=N||(CAUSAL&&kv0>global_row_b))s_acc[si+2]=-FLT_MAX;
            if(kv1>=N||(CAUSAL&&kv1>global_row_b))s_acc[si+3]=-FLT_MAX;
            if(global_row_a>=N){s_acc[si]=-FLT_MAX;s_acc[si+1]=-FLT_MAX;}
            if(global_row_b>=N){s_acc[si+2]=-FLT_MAX;s_acc[si+3]=-FLT_MAX;}
            rmax_a=fmaxf(rmax_a,fmaxf(s_acc[si],s_acc[si+1]));
            rmax_b=fmaxf(rmax_b,fmaxf(s_acc[si+2],s_acc[si+3]));
        }
        #pragma unroll
        for(int d=1;d<4;d<<=1){rmax_a=fmaxf(rmax_a,__shfl_xor_sync(0xffffffff,rmax_a,d));rmax_b=fmaxf(rmax_b,__shfl_xor_sync(0xffffffff,rmax_b,d));}

        float m_new_a=fmaxf(m_a,rmax_a),m_new_b=fmaxf(m_b,rmax_b);
        float sc_a=__expf(m_a-m_new_a),sc_b=__expf(m_b-m_new_b);
        #pragma unroll
        for(int i=0;i<O_ACC;i+=4){o_acc[i]*=sc_a;o_acc[i+1]*=sc_a;o_acc[i+2]*=sc_b;o_acc[i+3]*=sc_b;}
        l_a*=sc_a;l_b*=sc_b;m_a=m_new_a;m_b=m_new_b;

        float lt_a=0,lt_b=0;
        #pragma unroll
        for(int n=0;n<QKT_N;n++){
            int si=n*4;
            float p0=__expf(s_acc[si]-m_new_a);
            float p1=__expf(s_acc[si+1]-m_new_a);
            float p2=__expf(s_acc[si+2]-m_new_b);
            float p3=__expf(s_acc[si+3]-m_new_b);
            lt_a+=p0+p1;lt_b+=p2+p3;s_acc[si]=p0;s_acc[si+1]=p1;s_acc[si+2]=p2;s_acc[si+3]=p3;
        }
        #pragma unroll
        for(int d=1;d<4;d<<=1){lt_a+=__shfl_xor_sync(0xffffffff,lt_a,d);lt_b+=__shfl_xor_sync(0xffffffff,lt_b,d);}
        l_a+=lt_a;l_b+=lt_b;

        #pragma unroll
        for(int pk=0;pk<PV_K;pk++){
            int nt0=pk*2,nt1=pk*2+1;
            uint32_t pa0=pack_bf16_pair(s_acc[nt0*4],s_acc[nt0*4+1]),pa1=pack_bf16_pair(s_acc[nt0*4+2],s_acc[nt0*4+3]);
            uint32_t pa2=pack_bf16_pair(s_acc[nt1*4],s_acc[nt1*4+1]),pa3=pack_bf16_pair(s_acc[nt1*4+2],s_acc[nt1*4+3]);
            #pragma unroll
            for(int pn=0;pn<PV_N;pn++){
                uint32_t vb0,vb1;
                {int mat=(lane>>3)&1;bk::ldmatrix_x2_trans(vb0,vb1,&v_smem[swz<D>(pk*16+mat*8+(lane&7),pn*8)]);}
                bk::mma_m16n8k16_bf16_nv(o_acc[pn*4],o_acc[pn*4+1],o_acc[pn*4+2],o_acc[pn*4+3],
                    pa0,pa1,pa2,pa3,vb0,vb1,o_acc[pn*4],o_acc[pn*4+1],o_acc[pn*4+2],o_acc[pn*4+3]);
            }
        }

        if (has_next) bk::cp_async_wait<0>();
        __syncthreads();
        k_phase ^= 1;
    }

    float inv_a=(l_a>0)?(1.0f/l_a):0.0f, inv_b=(l_b>0)?(1.0f/l_b):0.0f;
    #pragma unroll
    for(int n=0;n<PV_N;n++){
        int col0=n*8+(lane%4)*2;
        if(global_row_a<N&&col0+1<D){
            uint32_t p=pack_bf16_pair(o_acc[n*4]*inv_a, o_acc[n*4+1]*inv_a);
            *reinterpret_cast<uint32_t*>(&O_ptr[global_row_a*D+col0])=p;
        }
        if(global_row_b<N&&col0+1<D){
            uint32_t p=pack_bf16_pair(o_acc[n*4+2]*inv_b, o_acc[n*4+3]*inv_b);
            *reinterpret_cast<uint32_t*>(&O_ptr[global_row_b*D+col0])=p;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// D=40 MMA kernel: pad to DP=64 for swizzle + m16n8k16 alignment.
// Identical MMA structure to D=64, but loads zero-fill cols 40-63
// and stores only cols 0-39.
// Smem: Q/V[8KB] + K_A[8KB] + K_B[8KB] = 24KB
// ═══════════════════════════════════════════════════════════════════════
template <bool CAUSAL>
__global__ void __launch_bounds__(128)
flash_attn_mma_d40(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const int N, const float scale
) {
    constexpr int D = 40, DP = 64;
    constexpr int Br = 64, Bc = 64;
    constexpr int BLOCK = 128, WARP_ROWS = 16;
    constexpr int K_STEPS = DP / 16;   // 4
    constexpr int N_TILES = Bc / 8;    // 8
    constexpr int ACC = N_TILES * 4;   // 32
    constexpr int PV_K = Bc / 16;      // 4
    constexpr int PV_N = DP / 8;       // 8 (5 real + 3 padded)
    constexpr int O_ACC = PV_N * 4;    // 32
    constexpr int TILE_CHUNKS = Bc * (DP / 8); // 512

    const int bh = blockIdx.x;
    const int q_start = blockIdx.y * Br;
    if (q_start >= N) return;

    const int tid = threadIdx.x, warp_id = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    const __nv_bfloat16* Q_ptr = Q + (size_t)bh * N * D;
    const __nv_bfloat16* K_ptr = K + (size_t)bh * N * D;
    const __nv_bfloat16* V_ptr = V + (size_t)bh * N * D;
    __nv_bfloat16*       O_ptr = O + (size_t)bh * N * D;

    // Smem: [Q/V: Br*DP = 8KB] [K_A: Bc*DP = 8KB] [K_B: 8KB] = 24KB
    extern __shared__ char smem_raw[];
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_smem_a = q_smem + Br * DP;
    __nv_bfloat16* k_smem_b = k_smem_a + Bc * DP;

    // ─── Load Q to smem: cp.async for cols<40, zero-fill cols 40-63 ─
    constexpr int Q_CHUNKS = Br * (DP / 8);
    for (int ci = tid; ci < Q_CHUNKS; ci += BLOCK) {
        int row = ci / (DP / 8), col8 = (ci % (DP / 8)) * 8;
        int gr = q_start + row;
        bk::cp_async_128_zfill(&q_smem[swz<DP>(row, col8)],
            &Q_ptr[gr * D + col8], (gr < N) && (col8 + 8 <= D));
    }
    bk::cp_async_commit();
    bk::cp_async_wait_all();
    __syncthreads();

    // ─── Q → registers + BF16 pre-scale ─────────────────────────────
    float o_acc[O_ACC];
    #pragma unroll
    for (int i = 0; i < O_ACC; i++) o_acc[i] = 0.0f;

    uint32_t q_frag[K_STEPS * 4];
    #pragma unroll
    for (int k = 0; k < K_STEPS; k++) {
        int sub = lane / 8, sub_row = lane % 8;
        int row = warp_id * WARP_ROWS + (sub < 2 ? sub_row : 8 + sub_row);
        int col = k * 16 + (sub % 2) * 8;
        bk::ldmatrix_x4_mma(q_frag[k*4], q_frag[k*4+1], q_frag[k*4+2], q_frag[k*4+3],
            &q_smem[swz<DP>(row, col)]);
    }

    uint32_t scale_packed = pack_bf16_pair(scale, scale);
    #pragma unroll
    for (int i = 0; i < K_STEPS * 4; i++) {
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(q_frag[i]) : "r"(q_frag[i]), "r"(scale_packed));
    }
    __syncthreads();

    __nv_bfloat16* v_smem = q_smem;

    const int mma_row_a = lane / 4, mma_row_b = mma_row_a + 8;
    const int global_row_a = q_start + warp_id * WARP_ROWS + mma_row_a;
    const int global_row_b = q_start + warp_id * WARP_ROWS + mma_row_b;
    float m_a = -FLT_MAX, l_a = 0.0f, m_b = -FLT_MAX, l_b = 0.0f;

    const int kv_tiles = (N + Bc - 1) / Bc;
    const int kv_end = CAUSAL ? min(kv_tiles, (q_start + Br + Bc - 1) / Bc) : kv_tiles;
    int k_phase = 0;

    // ─── Prologue: prefetch K[0] ────────────────────────────────────
    for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
        int row = ci / (DP / 8), col8 = (ci % (DP / 8)) * 8;
        bk::cp_async_128_zfill(&k_smem_a[swz<DP>(row, col8)],
            &K_ptr[row * D + col8], (row < N) && (col8 + 8 <= D));
    }
    bk::cp_async_commit();
    bk::cp_async_wait_all();
    __syncthreads();

    // ─── Main loop (pipelined) ──────────────────────────────────────
    for (int kv_t = 0; kv_t < kv_end; kv_t++) {
        const int kv_start = kv_t * Bc;
        __nv_bfloat16* k_cur = k_phase ? k_smem_b : k_smem_a;
        const bool has_next = (kv_t + 1 < kv_end);

        // ── V async load (zero-fill padding cols) ────────────────────
        for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
            int row = ci / (DP / 8), col8 = (ci % (DP / 8)) * 8;
            int gr = kv_start + row;
            bk::cp_async_128_zfill(&v_smem[swz<DP>(row, col8)],
                &V_ptr[gr * D + col8], (gr < N) && (col8 + 8 <= D));
        }
        bk::cp_async_commit();

        // ── S = Q_regs @ K^T ────────────────────────────────────────
        float s_acc[ACC];
        #pragma unroll
        for (int i = 0; i < ACC; i++) s_acc[i] = 0.0f;

        #pragma unroll
        for (int k = 0; k < K_STEPS; k++) {
            uint32_t a0 = q_frag[k*4], a1 = q_frag[k*4+1], a2 = q_frag[k*4+2], a3 = q_frag[k*4+3];
            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                uint32_t b0, b1;
                { int mat=(lane>>3)&1; bk::ldmatrix_x2(b0,b1,&k_cur[swz<DP>(n*8+(lane&7), k*16+mat*8)]); }
                bk::mma_m16n8k16_bf16_nv(s_acc[n*4],s_acc[n*4+1],s_acc[n*4+2],s_acc[n*4+3],
                    a0,a1,a2,a3,b0,b1, s_acc[n*4],s_acc[n*4+1],s_acc[n*4+2],s_acc[n*4+3]);
            }
        }

        // ── K[next] prefetch ─────────────────────────────────────────
        if (has_next) {
            int nxt = (kv_t + 1) * Bc;
            __nv_bfloat16* k_nxt = k_phase ? k_smem_a : k_smem_b;
            for (int ci = tid; ci < TILE_CHUNKS; ci += BLOCK) {
                int row = ci / (DP / 8), col8 = (ci % (DP / 8)) * 8;
                int gr = nxt + row;
                bk::cp_async_128_zfill(&k_nxt[swz<DP>(row, col8)],
                    &K_ptr[gr * D + col8], (gr < N) && (col8 + 8 <= D));
            }
            bk::cp_async_commit();
        }

        // ── Wait V ───────────────────────────────────────────────────
        if (has_next) bk::cp_async_wait<1>();
        else          bk::cp_async_wait<0>();
        __syncthreads();

        // ── Softmax ─────────────────────────────────────────────────
        float rmax_a = -FLT_MAX, rmax_b = -FLT_MAX;
        #pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            int col0 = n*8+(lane%4)*2, kv0 = kv_start+col0, kv1 = kv0+1;
            if (kv0>=N||(CAUSAL&&kv0>global_row_a)) s_acc[n*4]=-FLT_MAX;
            if (kv1>=N||(CAUSAL&&kv1>global_row_a)) s_acc[n*4+1]=-FLT_MAX;
            if (kv0>=N||(CAUSAL&&kv0>global_row_b)) s_acc[n*4+2]=-FLT_MAX;
            if (kv1>=N||(CAUSAL&&kv1>global_row_b)) s_acc[n*4+3]=-FLT_MAX;
            if (global_row_a>=N){s_acc[n*4]=-FLT_MAX;s_acc[n*4+1]=-FLT_MAX;}
            if (global_row_b>=N){s_acc[n*4+2]=-FLT_MAX;s_acc[n*4+3]=-FLT_MAX;}
            rmax_a=fmaxf(rmax_a,fmaxf(s_acc[n*4],s_acc[n*4+1]));
            rmax_b=fmaxf(rmax_b,fmaxf(s_acc[n*4+2],s_acc[n*4+3]));
        }
        #pragma unroll
        for (int d=1;d<4;d<<=1){rmax_a=fmaxf(rmax_a,__shfl_xor_sync(0xffffffff,rmax_a,d));rmax_b=fmaxf(rmax_b,__shfl_xor_sync(0xffffffff,rmax_b,d));}

        float m_new_a=fmaxf(m_a,rmax_a), m_new_b=fmaxf(m_b,rmax_b);
        float sc_a=__expf(m_a-m_new_a), sc_b=__expf(m_b-m_new_b);
        #pragma unroll
        for(int i=0;i<O_ACC;i+=4){o_acc[i]*=sc_a;o_acc[i+1]*=sc_a;o_acc[i+2]*=sc_b;o_acc[i+3]*=sc_b;}
        l_a*=sc_a;l_b*=sc_b;m_a=m_new_a;m_b=m_new_b;

        float lt_a=0,lt_b=0;
        #pragma unroll
        for(int n=0;n<N_TILES;n++){
            float p0=__expf(s_acc[n*4]-m_new_a);
            float p1=__expf(s_acc[n*4+1]-m_new_a);
            float p2=__expf(s_acc[n*4+2]-m_new_b);
            float p3=__expf(s_acc[n*4+3]-m_new_b);
            lt_a+=p0+p1;lt_b+=p2+p3;s_acc[n*4]=p0;s_acc[n*4+1]=p1;s_acc[n*4+2]=p2;s_acc[n*4+3]=p3;
        }
        #pragma unroll
        for(int d=1;d<4;d<<=1){lt_a+=__shfl_xor_sync(0xffffffff,lt_a,d);lt_b+=__shfl_xor_sync(0xffffffff,lt_b,d);}
        l_a+=lt_a;l_b+=lt_b;

        // ── O += P @ V ──────────────────────────────────────────────
        #pragma unroll
        for(int pk=0;pk<PV_K;pk++){
            int nt0=pk*2,nt1=pk*2+1;
            uint32_t pa0=pack_bf16_pair(s_acc[nt0*4],s_acc[nt0*4+1]),pa1=pack_bf16_pair(s_acc[nt0*4+2],s_acc[nt0*4+3]);
            uint32_t pa2=pack_bf16_pair(s_acc[nt1*4],s_acc[nt1*4+1]),pa3=pack_bf16_pair(s_acc[nt1*4+2],s_acc[nt1*4+3]);
            #pragma unroll
            for(int pn=0;pn<PV_N;pn++){
                uint32_t vb0,vb1;
                {int mat=(lane>>3)&1;bk::ldmatrix_x2_trans(vb0,vb1,&v_smem[swz<DP>(pk*16+mat*8+(lane&7),pn*8)]);}
                bk::mma_m16n8k16_bf16_nv(o_acc[pn*4],o_acc[pn*4+1],o_acc[pn*4+2],o_acc[pn*4+3],
                    pa0,pa1,pa2,pa3,vb0,vb1, o_acc[pn*4],o_acc[pn*4+1],o_acc[pn*4+2],o_acc[pn*4+3]);
            }
        }

        // ── Wait K[next], ensure V consumed ──────────────────────────
        if (has_next) bk::cp_async_wait<0>();
        __syncthreads();
        k_phase ^= 1;
    }

    // ─── Final normalize + store (only D=40 columns) ────────────────
    float inv_a=(l_a>0)?(1.0f/l_a):0.0f, inv_b=(l_b>0)?(1.0f/l_b):0.0f;
    #pragma unroll
    for(int n=0;n<PV_N;n++){
        int col0=n*8+(lane%4)*2;
        if(global_row_a<N&&col0+1<D){
            uint32_t p=pack_bf16_pair(o_acc[n*4]*inv_a, o_acc[n*4+1]*inv_a);
            *reinterpret_cast<uint32_t*>(&O_ptr[global_row_a*D+col0])=p;
        }
        if(global_row_b<N&&col0+1<D){
            uint32_t p=pack_bf16_pair(o_acc[n*4+2]*inv_b, o_acc[n*4+3]*inv_b);
            *reinterpret_cast<uint32_t*>(&O_ptr[global_row_b*D+col0])=p;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Scalar fallback for D=40
// ═══════════════════════════════════════════════════════════════════════
template <int HEAD_DIM, bool CAUSAL>
__global__ void __launch_bounds__(128)
flash_attn_scalar(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const int N, const float scale
) {
    constexpr int Br = 64, Bc = 64;
    const int bh = blockIdx.x;
    const int q_start = blockIdx.y * Br;
    if (q_start >= N) return;
    const int tid = threadIdx.x;
    const int gqr = q_start + tid;
    const bool active = (tid < Br) && (gqr < N);

    const __nv_bfloat16* Qp = Q + (size_t)bh * N * HEAD_DIM;
    const __nv_bfloat16* Kp = K + (size_t)bh * N * HEAD_DIM;
    const __nv_bfloat16* Vp = V + (size_t)bh * N * HEAD_DIM;
    __nv_bfloat16*       Op = O + (size_t)bh * N * HEAD_DIM;

    constexpr int PAD = 8, KVS = HEAD_DIM + PAD, TE = Bc * KVS;
    extern __shared__ char smem_raw[];
    __nv_bfloat16* kt = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* vt = kt + TE;

    float qr[HEAD_DIM];
    if (active) for (int d = 0; d < HEAD_DIM; d++) qr[d] = __bfloat162float(Qp[gqr*HEAD_DIM+d]);
    else for (int d = 0; d < HEAD_DIM; d++) qr[d] = 0.0f;

    float or_[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) or_[d] = 0.0f;
    float mv = -FLT_MAX, lv = 0.0f;

    const int kvt = (N+Bc-1)/Bc;
    const int kve = CAUSAL ? min(kvt, (q_start+Br+Bc-1)/Bc) : kvt;

    for (int t = 0; t < kve; t++) {
        const int ks = t * Bc;
        for (int i = tid; i < TE; i += 128) {
            int r=i/KVS, c=i%KVS; int gr=ks+r;
            __nv_bfloat16 z = __float2bfloat16(0.0f);
            kt[i] = (gr<N && c<HEAD_DIM) ? Kp[gr*HEAD_DIM+c] : z;
            vt[i] = (gr<N && c<HEAD_DIM) ? Vp[gr*HEAD_DIM+c] : z;
        }
        __syncthreads();
        if (active) {
            float sc[Bc]; float tm = -FLT_MAX;
            for (int j=0;j<Bc;j++) {
                int kg=ks+j;
                if (kg>=N||(CAUSAL&&kg>gqr)) { sc[j]=-FLT_MAX; }
                else { float d=0; for(int dd=0;dd<HEAD_DIM;dd++) d+=qr[dd]*__bfloat162float(kt[j*KVS+dd]); sc[j]=d*scale; }
                tm=fmaxf(tm,sc[j]);
            }
            float mn=fmaxf(mv,tm);
            float s=(mv>-FLT_MAX)?expf(mv-mn):0.0f;
            lv*=s; for(int d=0;d<HEAD_DIM;d++) or_[d]*=s;
            float lt=0;
            for(int j=0;j<Bc;j++) {
                float p=(sc[j]>-FLT_MAX)?expf(sc[j]-mn):0.0f;
                lt+=p;
                if(p>0) for(int d=0;d<HEAD_DIM;d++) or_[d]+=p*__bfloat162float(vt[j*KVS+d]);
            }
            mv=mn; lv+=lt;
        }
        __syncthreads();
    }
    if (active) {
        float il=(lv>0)?(1.0f/lv):0.0f;
        for(int d=0;d<HEAD_DIM;d++) Op[gqr*HEAD_DIM+d]=__float2bfloat16(or_[d]*il);
    }
}

} // namespace

// ============================================================================
torch::Tensor flash_attn_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal
) {
    TORCH_CHECK(Q.is_cuda()&&K.is_cuda()&&V.is_cuda(), "CUDA");
    TORCH_CHECK(Q.dtype()==torch::kBFloat16, "BF16");
    TORCH_CHECK(Q.is_contiguous()&&K.is_contiguous()&&V.is_contiguous(), "contiguous");
    const int B=Q.size(0),H=Q.size(1),N=Q.size(2),D=Q.size(3);
    TORCH_CHECK(D==40||D==64||D==128, "head_dim 40/64/128");

    auto O = torch::zeros_like(Q);
    float sc = 1.0f/sqrtf((float)D);
    float sc_log2e = sc * LOG2E_F;
    auto d = [&](){
        struct{const __nv_bfloat16*q,*k,*v;__nv_bfloat16*o;}r;
        r.q=reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr());
        r.k=reinterpret_cast<const __nv_bfloat16*>(K.data_ptr());
        r.v=reinterpret_cast<const __nv_bfloat16*>(V.data_ptr());
        r.o=reinterpret_cast<__nv_bfloat16*>(O.data_ptr());
        return r;
    }();

    constexpr int Br64=64, Br128=128, Bc=64;

    if(D==64){
        dim3 grid(B*H, (N+Br64-1)/Br64);
        int sm = (Br64 + 2*Bc)*D*sizeof(__nv_bfloat16); // 24KB
        if (causal) {
            flash_attn_mma_d64<true><<<grid,128,sm>>>(d.q,d.k,d.v,d.o,N,sc_log2e);
        } else {
            flash_attn_mma_d64<false><<<grid,128,sm>>>(d.q,d.k,d.v,d.o,N,sc_log2e);
        }
    } else if(D==128){
        dim3 grid(B*H, (N+Br64-1)/Br64);
        int sm = (Br64 + 2*Bc)*D*sizeof(__nv_bfloat16); // 48KB
        auto fn = causal ? flash_attn_mma_d128<true> : flash_attn_mma_d128<false>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        fn<<<grid,128,sm>>>(d.q,d.k,d.v,d.o,N,sc);
    } else {
        constexpr int DP = 64; // D_PAD for D=40
        dim3 grid(B*H, (N+Br64-1)/Br64);
        int sm = (Br64 + 2*Bc)*DP*sizeof(__nv_bfloat16); // 24KB
        auto fn = causal ? flash_attn_mma_d40<true> : flash_attn_mma_d40<false>;
        fn<<<grid,128,sm>>>(d.q,d.k,d.v,d.o,N,sc);
    }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", &flash_attn_forward,
          "Flash Attention forward (sm_120a)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("causal")=false);
}
