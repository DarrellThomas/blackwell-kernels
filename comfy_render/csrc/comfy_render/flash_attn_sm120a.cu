// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Flash Attention forward for sm_120a (RTX 5090).
// MMA kernel for D=64 with cp.async + double-buffer.
// Scalar fallback for D=40, D=128.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cmath>
#include <cfloat>

#include "mma_sm120.cuh"
#include "ldmatrix.cuh"
#include "cp_async.cuh"

namespace {

constexpr int Br = 64;
constexpr int Bc = 64;
constexpr int WARP_SIZE = 32;

// ═══════════════════════════════════════════════════════════════════════
// MMA kernel for D=64 with cp.async, single-buffer for occupancy
// ═══════════════════════════════════════════════════════════════════════
template <bool CAUSAL>
__global__ void __launch_bounds__(128)
flash_attn_mma_d64(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const int N,
    const float scale
) {
    constexpr int D = 64;
    constexpr int BLOCK = 128;
    constexpr int PAD = 8;
    constexpr int STRIDE = D + PAD; // 72
    constexpr int WARP_ROWS = 16;
    constexpr int K_STEPS = 4;  // D/16
    constexpr int N_TILES = 8;  // Bc/8 = D/8
    constexpr int ACC = 32;     // N_TILES * 4
    constexpr int KV_CHUNKS = Bc * (D / 8); // 512 cp.async chunks per tile

    const int bh = blockIdx.x;
    const int q_start = blockIdx.y * Br;
    if (q_start >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const __nv_bfloat16* Q_ptr = Q + (size_t)bh * N * D;
    const __nv_bfloat16* K_ptr = K + (size_t)bh * N * D;
    const __nv_bfloat16* V_ptr = V + (size_t)bh * N * D;
    __nv_bfloat16*       O_ptr = O + (size_t)bh * N * D;

    // Smem: Q[Br*S] + K/P[Bc*S] + V[Bc*S]  (P aliases K after S computation)
    // 28 KB total → 3 blocks per SM (25% occupancy)
    extern __shared__ char smem[];
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* k_smem = q_smem + Br * STRIDE;
    __nv_bfloat16* v_smem = k_smem + Bc * STRIDE;

    // ─── Load Q to smem ───────────────────────────────────────────────
    for (int idx = tid; idx < Br * STRIDE; idx += BLOCK) {
        int r = idx / STRIDE, c = idx % STRIDE;
        int gr = q_start + r;
        q_smem[idx] = (gr < N && c < D)
            ? Q_ptr[gr * D + c] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // ─── Accumulators ────────────────────────────────────────────────
    float o_acc[ACC];
    #pragma unroll
    for (int i = 0; i < ACC; i++) o_acc[i] = 0.0f;

    const int mma_row_a = lane / 4;
    const int mma_row_b = mma_row_a + 8;
    const int global_row_a = q_start + warp_id * WARP_ROWS + mma_row_a;
    const int global_row_b = q_start + warp_id * WARP_ROWS + mma_row_b;

    float m_a = -FLT_MAX, l_a = 0.0f;
    float m_b = -FLT_MAX, l_b = 0.0f;

    const int kv_tiles = (N + Bc - 1) / Bc;
    const int kv_end = CAUSAL ? min(kv_tiles, (q_start + Br + Bc - 1) / Bc) : kv_tiles;
    const int q_base = warp_id * WARP_ROWS;

    for (int kv_t = 0; kv_t < kv_end; kv_t++) {
        const int kv_start = kv_t * Bc;

        // ─── Load K + V via cp.async ────────────────────────────────
        for (int ci = tid; ci < KV_CHUNKS; ci += BLOCK) {
            int row = ci / (D / 8);
            int col = (ci % (D / 8)) * 8;
            int gr = kv_start + row;
            bk::cp_async_128_zfill(&k_smem[row * STRIDE + col], &K_ptr[gr * D + col], gr < N);
            bk::cp_async_128_zfill(&v_smem[row * STRIDE + col], &V_ptr[gr * D + col], gr < N);
        }
        bk::cp_async_commit();
        bk::cp_async_wait_all();
        __syncthreads();

        // ─── S = Q @ K^T via MMA ────────────────────────────────────
        float s_acc[ACC];
        #pragma unroll
        for (int i = 0; i < ACC; i++) s_acc[i] = 0.0f;

        #pragma unroll
        for (int k = 0; k < K_STEPS; k++) {
            uint32_t a0, a1, a2, a3;
            {
                int sub = lane / 8;
                int sub_row = lane % 8;
                int row = q_base + (sub < 2 ? sub_row : 8 + sub_row);
                int col = k * 16 + (sub % 2) * 8;
                bk::ldmatrix_x4_mma(a0, a1, a2, a3,
                    &q_smem[row * STRIDE + col]);
            }

            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                uint32_t b0, b1;
                {
                    int mat = (lane >> 3) & 1;
                    bk::ldmatrix_x2(b0, b1,
                        &k_smem[(n * 8 + (lane & 7)) * STRIDE + k * 16 + mat * 8]);
                }
                bk::mma_m16n8k16_bf16_nv(
                    s_acc[n*4], s_acc[n*4+1], s_acc[n*4+2], s_acc[n*4+3],
                    a0, a1, a2, a3, b0, b1,
                    s_acc[n*4], s_acc[n*4+1], s_acc[n*4+2], s_acc[n*4+3]);
            }
        }

        // ─── Scale + mask + online softmax ───────────────────────────
        float rmax_a = -FLT_MAX, rmax_b = -FLT_MAX;
        #pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            int col0 = n * 8 + (lane % 4) * 2;
            int kv0 = kv_start + col0, kv1 = kv0 + 1;

            s_acc[n*4]   *= scale;
            s_acc[n*4+1] *= scale;
            s_acc[n*4+2] *= scale;
            s_acc[n*4+3] *= scale;

            if (kv0 >= N || (CAUSAL && kv0 > global_row_a)) s_acc[n*4]   = -FLT_MAX;
            if (kv1 >= N || (CAUSAL && kv1 > global_row_a)) s_acc[n*4+1] = -FLT_MAX;
            if (kv0 >= N || (CAUSAL && kv0 > global_row_b)) s_acc[n*4+2] = -FLT_MAX;
            if (kv1 >= N || (CAUSAL && kv1 > global_row_b)) s_acc[n*4+3] = -FLT_MAX;
            if (global_row_a >= N) { s_acc[n*4] = -FLT_MAX; s_acc[n*4+1] = -FLT_MAX; }
            if (global_row_b >= N) { s_acc[n*4+2] = -FLT_MAX; s_acc[n*4+3] = -FLT_MAX; }

            rmax_a = fmaxf(rmax_a, fmaxf(s_acc[n*4], s_acc[n*4+1]));
            rmax_b = fmaxf(rmax_b, fmaxf(s_acc[n*4+2], s_acc[n*4+3]));
        }

        #pragma unroll
        for (int d = 1; d < 4; d <<= 1) {
            rmax_a = fmaxf(rmax_a, __shfl_xor_sync(0xffffffff, rmax_a, d));
            rmax_b = fmaxf(rmax_b, __shfl_xor_sync(0xffffffff, rmax_b, d));
        }

        float m_new_a = fmaxf(m_a, rmax_a);
        float m_new_b = fmaxf(m_b, rmax_b);
        float sc_a = (m_a > -FLT_MAX) ? expf(m_a - m_new_a) : 0.0f;
        float sc_b = (m_b > -FLT_MAX) ? expf(m_b - m_new_b) : 0.0f;

        #pragma unroll
        for (int i = 0; i < ACC; i += 4) {
            o_acc[i]   *= sc_a; o_acc[i+1] *= sc_a;
            o_acc[i+2] *= sc_b; o_acc[i+3] *= sc_b;
        }
        l_a *= sc_a; l_b *= sc_b;
        m_a = m_new_a; m_b = m_new_b;

        // exp + store P to k_smem (reuses K buffer) + accumulate l
        float lt_a = 0.0f, lt_b = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_TILES; n++) {
            float p0 = (s_acc[n*4]   > -FLT_MAX) ? expf(s_acc[n*4]   - m_new_a) : 0.0f;
            float p1 = (s_acc[n*4+1] > -FLT_MAX) ? expf(s_acc[n*4+1] - m_new_a) : 0.0f;
            float p2 = (s_acc[n*4+2] > -FLT_MAX) ? expf(s_acc[n*4+2] - m_new_b) : 0.0f;
            float p3 = (s_acc[n*4+3] > -FLT_MAX) ? expf(s_acc[n*4+3] - m_new_b) : 0.0f;
            lt_a += p0 + p1; lt_b += p2 + p3;

            int col0 = n * 8 + (lane % 4) * 2;
            int ra = warp_id * WARP_ROWS + mma_row_a;
            int rb = warp_id * WARP_ROWS + mma_row_b;
            k_smem[ra * STRIDE + col0]     = __float2bfloat16(p0);
            k_smem[ra * STRIDE + col0 + 1] = __float2bfloat16(p1);
            k_smem[rb * STRIDE + col0]     = __float2bfloat16(p2);
            k_smem[rb * STRIDE + col0 + 1] = __float2bfloat16(p3);
        }

        #pragma unroll
        for (int d = 1; d < 4; d <<= 1) {
            lt_a += __shfl_xor_sync(0xffffffff, lt_a, d);
            lt_b += __shfl_xor_sync(0xffffffff, lt_b, d);
        }
        l_a += lt_a; l_b += lt_b;
        __syncthreads();

        // ─── O += P @ V via MMA ─────────────────────────────────────
        #pragma unroll
        for (int k = 0; k < K_STEPS; k++) {
            uint32_t pa0, pa1, pa2, pa3;
            {
                int sub = lane / 8;
                int sub_row = lane % 8;
                int row = warp_id * WARP_ROWS + (sub < 2 ? sub_row : 8 + sub_row);
                int col = k * 16 + (sub % 2) * 8;
                bk::ldmatrix_x4_mma(pa0, pa1, pa2, pa3,
                    &k_smem[row * STRIDE + col]);
            }

            #pragma unroll
            for (int n = 0; n < N_TILES; n++) {
                uint32_t vb0, vb1;
                {
                    int mat = (lane >> 3) & 1;
                    bk::ldmatrix_x2_trans(vb0, vb1,
                        &v_smem[(k * 16 + mat * 8 + (lane & 7)) * STRIDE + n * 8]);
                }
                bk::mma_m16n8k16_bf16_nv(
                    o_acc[n*4], o_acc[n*4+1], o_acc[n*4+2], o_acc[n*4+3],
                    pa0, pa1, pa2, pa3, vb0, vb1,
                    o_acc[n*4], o_acc[n*4+1], o_acc[n*4+2], o_acc[n*4+3]);
            }
        }
        __syncthreads();
    }

    // ─── Final normalize + store ─────────────────────────────────────
    float inv_a = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
    float inv_b = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;

    #pragma unroll
    for (int n = 0; n < N_TILES; n++) {
        int col0 = n * 8 + (lane % 4) * 2;
        int gr_a = q_start + warp_id * WARP_ROWS + mma_row_a;
        int gr_b = q_start + warp_id * WARP_ROWS + mma_row_b;

        if (gr_a < N && col0 < D)     O_ptr[gr_a*D + col0]   = __float2bfloat16(o_acc[n*4]   * inv_a);
        if (gr_a < N && col0+1 < D)   O_ptr[gr_a*D + col0+1] = __float2bfloat16(o_acc[n*4+1] * inv_a);
        if (gr_b < N && col0 < D)     O_ptr[gr_b*D + col0]   = __float2bfloat16(o_acc[n*4+2] * inv_b);
        if (gr_b < N && col0+1 < D)   O_ptr[gr_b*D + col0+1] = __float2bfloat16(o_acc[n*4+3] * inv_b);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Scalar fallback
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
    extern __shared__ char smem[];
    __nv_bfloat16* kt = reinterpret_cast<__nv_bfloat16*>(smem);
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
    dim3 grid(B*H, (N+Br-1)/Br);
    auto d = [&](){
        struct{const __nv_bfloat16*q,*k,*v;__nv_bfloat16*o;}r;
        r.q=reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr());
        r.k=reinterpret_cast<const __nv_bfloat16*>(K.data_ptr());
        r.v=reinterpret_cast<const __nv_bfloat16*>(V.data_ptr());
        r.o=reinterpret_cast<__nv_bfloat16*>(O.data_ptr());
        return r;
    }();

    constexpr int PAD=8;
    if(D==64){
        int sm=(Br+2*Bc)*(64+PAD)*sizeof(__nv_bfloat16); // Q + K/P + V
        auto fn=causal?flash_attn_mma_d64<true>:flash_attn_mma_d64<false>;
        fn<<<grid,128,sm>>>(d.q,d.k,d.v,d.o,N,sc);
    } else {
        int sm=2*Bc*(D+PAD)*sizeof(__nv_bfloat16);
        if(D==40){
            auto fn=causal?flash_attn_scalar<40,true>:flash_attn_scalar<40,false>;
            fn<<<grid,128,sm>>>(d.q,d.k,d.v,d.o,N,sc);
        } else {
            auto fn=causal?flash_attn_scalar<128,true>:flash_attn_scalar<128,false>;
            fn<<<grid,128,sm>>>(d.q,d.k,d.v,d.o,N,sc);
        }
    }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", &flash_attn_forward,
          "Flash Attention forward (sm_120a)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("causal")=false);
}
