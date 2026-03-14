// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Flash Attention v4 for sm_120 (RTX 5090)
// Zero-interface full PTX inner loop — salykova-style monolithic assembly.
//
// The ENTIRE KV loop (QK^T → mask → softmax → PV MMA → prefetch → barrier)
// AND the final normalization + output store are a SINGLE asm volatile block
// with ZERO "+f" outputs. All intermediate state lives in .reg.
// Outputs written via st.global from within PTX.
//
// Inputs only: smem bases, global ptrs, Q regs, thread IDs, constants.
// This eliminates all register spills at the asm boundary.
//
// Architecture: BLOCK_Q=64/128, BLOCK_KV=64, 4 warps (128 threads),
// double-buffered K/V with cp.async, XOR swizzle, ldmatrix_x4_mma for Q.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#include "mma_sm120.cuh"
#include "ldmatrix.cuh"
#include "cp_async.cuh"
#include "swizzle.cuh"

constexpr int V4_NUM_WARPS = 4;
constexpr int V4_WARP_SIZE = 32;
constexpr int V4_THREADS = V4_NUM_WARPS * V4_WARP_SIZE;

// ============================================================
// v4 kernel — D=64, BKV=64, BQ=64 only (zero-interface PTX)
// ============================================================

template <int BLOCK_Q>
__global__ void __launch_bounds__(V4_THREADS, (BLOCK_Q <= 64) ? 3 : 2)
flash_attn_v4_kernel(
    const __nv_bfloat16 *__restrict__ Q,
    const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V,
    __nv_bfloat16 *__restrict__ O_out,
    float *__restrict__ L,
    int seq_len,
    float scale,
    bool causal)
{
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int WARP_Q = BLOCK_Q / V4_NUM_WARPS;
    constexpr int WARP_Q_TILES = WARP_Q / 16;
    constexpr int D_CHUNKS = HEAD_DIM / 16;     // 4
    constexpr int KV_SMEM_ELEMS = BLOCK_KV * HEAD_DIM;

    const int bh_idx = blockIdx.y;
    const int q_block = blockIdx.x;
    const int q_start = q_block * BLOCK_Q;

    const int tid = threadIdx.x;
    const int warp_id = tid / V4_WARP_SIZE;
    const int lane_id = tid % V4_WARP_SIZE;

    const __nv_bfloat16 *Q_bh = Q + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *K_bh = K + bh_idx * seq_len * HEAD_DIM;
    const __nv_bfloat16 *V_bh = V + bh_idx * seq_len * HEAD_DIM;
    __nv_bfloat16 *O_bh = O_out + bh_idx * seq_len * HEAD_DIM;
    float *L_bh = L + bh_idx * seq_len;

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *smem_base = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 *smem_Q = smem_base;
    __nv_bfloat16 *smem_K_base = smem_base;
    __nv_bfloat16 *smem_V_base = smem_base + 2 * KV_SMEM_ELEMS;

    // ================================================================
    // Phase A: Load Q tile global → shared via cp.async
    // ================================================================
    {
        constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
        constexpr int TOTAL_CHUNKS = BLOCK_Q * CHUNKS_PER_ROW;
        for (int i = tid; i < TOTAL_CHUNKS; i += V4_THREADS) {
            int row = i / CHUNKS_PER_ROW;
            int col = (i % CHUNKS_PER_ROW) * 8;
            int global_row = q_start + row;
            bk::cp_async_128_zfill(
                &smem_Q[bk::swizzle_idx<HEAD_DIM>(row, col)],
                &Q_bh[global_row * HEAD_DIM + col],
                global_row < seq_len);
        }
    }
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    // ================================================================
    // Phase B: Load Q shared → registers via ldmatrix_x4_mma
    // ================================================================
    uint32_t Q_rmem[WARP_Q_TILES][D_CHUNKS][4];
    {
        int warp_q_off = warp_id * WARP_Q;
        int sub = lane_id / 8;
        int t_in_sub = lane_id % 8;
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            int tile_off = warp_q_off + t * 16;
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS; dc++) {
                int smem_row = tile_off + (sub / 2) * 8 + t_in_sub;
                int smem_col = dc * 16 + (sub % 2) * 8;
                const void *addr = &smem_Q[bk::swizzle_idx<HEAD_DIM>(smem_row, smem_col)];
                bk::ldmatrix_x4_mma(Q_rmem[t][dc][0], Q_rmem[t][dc][1],
                                    Q_rmem[t][dc][2], Q_rmem[t][dc][3], addr);
            }
        }
    }
    // Pre-scale Q by scale * LOG2E
    {
        __nv_bfloat162 scale_vec = __float2bfloat162_rn(scale * 1.4426950408889634f);
        #pragma unroll
        for (int t = 0; t < WARP_Q_TILES; t++) {
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS; dc++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    __nv_bfloat162 q_val = *reinterpret_cast<__nv_bfloat162*>(&Q_rmem[t][dc][i]);
                    q_val = __hmul2(q_val, scale_vec);
                    Q_rmem[t][dc][i] = *reinterpret_cast<uint32_t*>(&q_val);
                }
            }
        }
    }
    __syncthreads();

    // ================================================================
    // Compute values needed by PTX block
    // ================================================================
    int kv_end = causal ? min(seq_len, q_start + BLOCK_Q) : seq_len;
    int num_kv_blocks = (kv_end + BLOCK_KV - 1) / BLOCK_KV;

    // Prologue: load first K/V tile
    {
        constexpr int CHUNKS_PER_ROW = HEAD_DIM / 8;
        constexpr int TOTAL_CHUNKS = BLOCK_KV * CHUNKS_PER_ROW;
        for (int i = tid; i < TOTAL_CHUNKS; i += V4_THREADS) {
            int row = i / CHUNKS_PER_ROW;
            int col = (i % CHUNKS_PER_ROW) * 8;
            bk::cp_async_128_zfill(
                &smem_K_base[bk::swizzle_idx<HEAD_DIM>(row, col)],
                &K_bh[row * HEAD_DIM + col],
                row < seq_len);
            bk::cp_async_128_zfill(
                &smem_V_base[bk::swizzle_idx<HEAD_DIM>(row, col)],
                &V_bh[row * HEAD_DIM + col],
                row < seq_len);
        }
    }
    bk::cp_async_commit();
    bk::cp_async_wait<0>();
    __syncthreads();

    // Shared memory base addresses (u32 shared addrs)
    uint32_t smem_K_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_K_base));
    uint32_t smem_V_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_V_base));
    // Global byte offsets per KV element: BLOCK_KV * HEAD_DIM * sizeof(bf16)
    // = 64 * 64 * 2 = 8192 bytes
    uint32_t kv_smem_stride = KV_SMEM_ELEMS * sizeof(__nv_bfloat16); // 8192

    // Global pointers (u64)
    uint64_t K_bh_u64 = reinterpret_cast<uint64_t>(K_bh);
    uint64_t V_bh_u64 = reinterpret_cast<uint64_t>(V_bh);
    uint64_t O_bh_u64 = reinterpret_cast<uint64_t>(O_bh);
    uint64_t L_bh_u64 = reinterpret_cast<uint64_t>(L_bh);

    // Per-warp row info for this tile
    // With BQ=64, 4 warps, each warp handles 16 rows (1 MMA tile)
    // global_row0 = q_start + warp_id*16 + lane_id/4
    // global_row1 = global_row0 + 8
    int warp_q_off = warp_id * WARP_Q;

    // Prefetch: each of 128 threads loads a portion of next KV tile
    // 64*64 = 4096 bf16 values = 8192 bytes = 512 × 16-byte chunks
    // 512/128 = 4 chunks per thread
    // Thread i loads chunks: i, i+128, i+256, i+384
    // chunk_row = chunk_idx / (64/8) = chunk_idx / 8
    // chunk_col = (chunk_idx % 8) * 8
    // Prefetch: 512 chunks / 128 threads = 4 chunks per thread

    // Causal flag as u32
    uint32_t causal_u32 = causal ? 1 : 0;

    // ================================================================
    // MONOLITHIC PTX BLOCK: KV loop + output store
    // ZERO outputs — everything via st.global inside PTX
    // ================================================================
    #pragma unroll
    for (int t = 0; t < WARP_Q_TILES; t++) {

        int global_row0 = q_start + warp_q_off + t * 16 + (lane_id / 4);
        int global_row1 = global_row0 + 8;

        // Swizzle helper values for cp.async prefetch (precompute per chunk)
        // For each of the 4 chunks this thread is responsible for:
        // chunk_idx = tid + c*128, row = chunk_idx/8, col = (chunk_idx%8)*8
        // swizzle: sw_col = col ^ ((row & 7) << 3)
        // smem_byte_offset = (row * 64 + sw_col) * 2
        uint32_t pf_smem_off[4]; // byte offsets into K or V smem buffer
        uint32_t pf_gmem_off[4]; // byte offsets into global K or V
        uint32_t pf_row[4];      // row index for bounds check
        for (int c = 0; c < 4; c++) {
            int ci = tid + c * V4_THREADS; // chunk index
            int row = ci / 8;
            int col = (ci % 8) * 8;
            int sw_col = col ^ ((row & 7) << 3);
            pf_smem_off[c] = (row * HEAD_DIM + sw_col) * sizeof(__nv_bfloat16);
            pf_gmem_off[c] = (row * HEAD_DIM + col) * sizeof(__nv_bfloat16);
            pf_row[c] = row;
        }

        asm volatile(
        "{\n"
        // ================================================================
        // Register declarations
        // ================================================================
        // O accumulators (32 floats: 8 nc-chunks × 4)
        ".reg .f32 o00,o01,o02,o03, o10,o11,o12,o13;\n"
        ".reg .f32 o20,o21,o22,o23, o30,o31,o32,o33;\n"
        ".reg .f32 o40,o41,o42,o43, o50,o51,o52,o53;\n"
        ".reg .f32 o60,o61,o62,o63, o70,o71,o72,o73;\n"
        // S accumulators (32 floats)
        ".reg .f32 s00,s01,s02,s03, s10,s11,s12,s13;\n"
        ".reg .f32 s20,s21,s22,s23, s30,s31,s32,s33;\n"
        ".reg .f32 s40,s41,s42,s43, s50,s51,s52,s53;\n"
        ".reg .f32 s60,s61,s62,s63, s70,s71,s72,s73;\n"
        // Row max & sum (persistent)
        ".reg .f32 rmax0, rmax1, rsum0, rsum1;\n"
        // K/V fragment regs
        ".reg .b32 k0,k1,k2,k3;\n"
        ".reg .b32 v0,v1,v2,v3;\n"
        // P fragment regs
        ".reg .b32 pa0,pa1,pa2,pa3;\n"
        // Softmax temporaries
        ".reg .f32 mx0, mx1, nmx0, nmx1;\n"
        ".reg .f32 rs0, rs1, ls0, ls1, tmp;\n"
        // Address computation
        ".reg .u32 sub, tsub, hsub, smod2;\n"
        ".reg .u32 base_row, k_row, kcol, k_rmask, sw_col, kidx, kaddr;\n"
        ".reg .u32 v_row, vcol, v_rmask, vsw_col, vidx, vaddr;\n"
        // Loop control
        ".reg .u32 kv_block, kv_start, cur_buf, nxt_buf;\n"
        ".reg .pred p_loop, p_mask, p_causal, pm, p_last;\n"
        ".reg .pred p_row0, p_row1, p_lane0;\n"
        // Mask temporaries
        ".reg .u32 lmod4, col0_base, col0, col1;\n"
        // Prefetch temporaries
        ".reg .u64 gaddr64;\n"
        ".reg .u32 saddr, gkv, gkv_row;\n"
        ".reg .pred p_pf;\n"
        // Output temporaries
        ".reg .u32 out_col, out_off;\n"
        ".reg .u64 out_addr64;\n"
        ".reg .b32 out_packed;\n"
        ".reg .f32 inv0, inv1;\n"
        ".reg .f32 lse0, lse1, lg0, lg1;\n"
        ".reg .u32 lmod4_out;\n"
        ".reg .pred p_lse;\n"
        "\n"

        // ================================================================
        // Initialize accumulators
        // ================================================================
        "mov.f32 o00, 0f00000000; mov.f32 o01, 0f00000000;\n"
        "mov.f32 o02, 0f00000000; mov.f32 o03, 0f00000000;\n"
        "mov.f32 o10, 0f00000000; mov.f32 o11, 0f00000000;\n"
        "mov.f32 o12, 0f00000000; mov.f32 o13, 0f00000000;\n"
        "mov.f32 o20, 0f00000000; mov.f32 o21, 0f00000000;\n"
        "mov.f32 o22, 0f00000000; mov.f32 o23, 0f00000000;\n"
        "mov.f32 o30, 0f00000000; mov.f32 o31, 0f00000000;\n"
        "mov.f32 o32, 0f00000000; mov.f32 o33, 0f00000000;\n"
        "mov.f32 o40, 0f00000000; mov.f32 o41, 0f00000000;\n"
        "mov.f32 o42, 0f00000000; mov.f32 o43, 0f00000000;\n"
        "mov.f32 o50, 0f00000000; mov.f32 o51, 0f00000000;\n"
        "mov.f32 o52, 0f00000000; mov.f32 o53, 0f00000000;\n"
        "mov.f32 o60, 0f00000000; mov.f32 o61, 0f00000000;\n"
        "mov.f32 o62, 0f00000000; mov.f32 o63, 0f00000000;\n"
        "mov.f32 o70, 0f00000000; mov.f32 o71, 0f00000000;\n"
        "mov.f32 o72, 0f00000000; mov.f32 o73, 0f00000000;\n"
        "mov.f32 rmax0, 0fff7fffff;\n"  // -FLT_MAX
        "mov.f32 rmax1, 0fff7fffff;\n"
        "mov.f32 rsum0, 0f00000000;\n"
        "mov.f32 rsum1, 0f00000000;\n"
        "\n"

        // ================================================================
        // Setup lane addressing (invariant across KV loop)
        // ================================================================
        "shr.b32 sub, %2, 3;\n"            // sub = lane_id / 8
        "and.b32 tsub, %2, 7;\n"           // tsub = lane_id % 8
        "shr.b32 hsub, sub, 1;\n"          // hsub = sub / 2
        "and.b32 smod2, sub, 1;\n"         // smod2 = sub % 2
        "shl.b32 base_row, hsub, 3;\n"     // base_row = hsub * 8
        "add.u32 base_row, base_row, tsub;\n" // base_row += tsub
        "\n"

        // ================================================================
        // KV Loop
        // ================================================================
        "mov.u32 kv_block, 0;\n"
        "KV_LOOP:\n"
        "setp.ge.u32 p_loop, kv_block, %6;\n"  // kv_block >= num_kv_blocks
        "@p_loop bra KV_DONE;\n"
        "\n"

        // cur_buf = (kv_block & 1) * kv_smem_stride
        "and.b32 cur_buf, kv_block, 1;\n"
        "mul.lo.u32 cur_buf, cur_buf, %7;\n"   // cur_buf = (kv_block&1) * kv_smem_stride

        // kv_start = kv_block * 64
        "shl.b32 kv_start, kv_block, 6;\n"

        // ================================================================
        // QK^T: Zero S, then 4 dc chunks × 4 nc pairs = 16 K loads + 32 MMAs
        // ================================================================
        "mov.f32 s00, 0f00000000; mov.f32 s01, 0f00000000;\n"
        "mov.f32 s02, 0f00000000; mov.f32 s03, 0f00000000;\n"
        "mov.f32 s10, 0f00000000; mov.f32 s11, 0f00000000;\n"
        "mov.f32 s12, 0f00000000; mov.f32 s13, 0f00000000;\n"
        "mov.f32 s20, 0f00000000; mov.f32 s21, 0f00000000;\n"
        "mov.f32 s22, 0f00000000; mov.f32 s23, 0f00000000;\n"
        "mov.f32 s30, 0f00000000; mov.f32 s31, 0f00000000;\n"
        "mov.f32 s32, 0f00000000; mov.f32 s33, 0f00000000;\n"
        "mov.f32 s40, 0f00000000; mov.f32 s41, 0f00000000;\n"
        "mov.f32 s42, 0f00000000; mov.f32 s43, 0f00000000;\n"
        "mov.f32 s50, 0f00000000; mov.f32 s51, 0f00000000;\n"
        "mov.f32 s52, 0f00000000; mov.f32 s53, 0f00000000;\n"
        "mov.f32 s60, 0f00000000; mov.f32 s61, 0f00000000;\n"
        "mov.f32 s62, 0f00000000; mov.f32 s63, 0f00000000;\n"
        "mov.f32 s70, 0f00000000; mov.f32 s71, 0f00000000;\n"
        "mov.f32 s72, 0f00000000; mov.f32 s73, 0f00000000;\n"
        "\n"

        // K load + MMA macro (inline via string concat)
        // For each (dc, nc_pair): compute swizzled K smem addr, ldmatrix_x4, 2x MMA
        // dc_off = dc*16, nc_off = nc_pair*16
        // k_row = base_row + nc_off
        // kcol = smod2*8 + dc_off
        // swizzle: sw_col = kcol ^ ((k_row & 7) << 3)
        // kidx = k_row * 64 + sw_col, *2 for bf16 bytes
        // kaddr = smem_K_base + cur_buf + kidx

// Macro: K load + 2 MMA for one (dc, nc_pair)
#define V4Z_KMMA(dc_off, nc_off, q0, q1, q2, q3, \
                 sa, sb, sc, sd, se, sf, sg, sh) \
        "add.u32 k_row, base_row, " #nc_off ";\n" \
        "shl.b32 kcol, smod2, 3;\n" \
        "add.u32 kcol, kcol, " #dc_off ";\n" \
        "and.b32 k_rmask, k_row, 7;\n" \
        "shl.b32 k_rmask, k_rmask, 3;\n" \
        "xor.b32 sw_col, kcol, k_rmask;\n" \
        "mad.lo.u32 kidx, k_row, 64, sw_col;\n" \
        "shl.b32 kidx, kidx, 1;\n" \
        "add.u32 kaddr, %0, kidx;\n" \
        "add.u32 kaddr, kaddr, cur_buf;\n" \
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {k0,k1,k2,k3}, [kaddr];\n" \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{" #sa "," #sb "," #sc "," #sd "}, " \
        "{" q0 "," q1 "," q2 "," q3 "}, {k0,k1}, " \
        "{" #sa "," #sb "," #sc "," #sd "};\n" \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{" #se "," #sf "," #sg "," #sh "}, " \
        "{" q0 "," q1 "," q2 "," q3 "}, {k2,k3}, " \
        "{" #se "," #sf "," #sg "," #sh "};\n"

        // dc=0, Q at %10-%13
        V4Z_KMMA(0,  0,  "%10","%11","%12","%13", s00,s01,s02,s03, s10,s11,s12,s13)
        V4Z_KMMA(0,  16, "%10","%11","%12","%13", s20,s21,s22,s23, s30,s31,s32,s33)
        V4Z_KMMA(0,  32, "%10","%11","%12","%13", s40,s41,s42,s43, s50,s51,s52,s53)
        V4Z_KMMA(0,  48, "%10","%11","%12","%13", s60,s61,s62,s63, s70,s71,s72,s73)
        // dc=1, Q at %14-%17
        V4Z_KMMA(16, 0,  "%14","%15","%16","%17", s00,s01,s02,s03, s10,s11,s12,s13)
        V4Z_KMMA(16, 16, "%14","%15","%16","%17", s20,s21,s22,s23, s30,s31,s32,s33)
        V4Z_KMMA(16, 32, "%14","%15","%16","%17", s40,s41,s42,s43, s50,s51,s52,s53)
        V4Z_KMMA(16, 48, "%14","%15","%16","%17", s60,s61,s62,s63, s70,s71,s72,s73)
        // dc=2, Q at %18-%21
        V4Z_KMMA(32, 0,  "%18","%19","%20","%21", s00,s01,s02,s03, s10,s11,s12,s13)
        V4Z_KMMA(32, 16, "%18","%19","%20","%21", s20,s21,s22,s23, s30,s31,s32,s33)
        V4Z_KMMA(32, 32, "%18","%19","%20","%21", s40,s41,s42,s43, s50,s51,s52,s53)
        V4Z_KMMA(32, 48, "%18","%19","%20","%21", s60,s61,s62,s63, s70,s71,s72,s73)
        // dc=3, Q at %22-%25
        V4Z_KMMA(48, 0,  "%22","%23","%24","%25", s00,s01,s02,s03, s10,s11,s12,s13)
        V4Z_KMMA(48, 16, "%22","%23","%24","%25", s20,s21,s22,s23, s30,s31,s32,s33)
        V4Z_KMMA(48, 32, "%22","%23","%24","%25", s40,s41,s42,s43, s50,s51,s52,s53)
        V4Z_KMMA(48, 48, "%22","%23","%24","%25", s60,s61,s62,s63, s70,s71,s72,s73)
        "\n"

#undef V4Z_KMMA

        // ================================================================
        // Causal + OOB mask
        // ================================================================
        // Determine if masking needed: (causal && kv_start+64 > q_start) || kv_start+64 > seq_len
        "add.u32 col0, kv_start, 64;\n"         // kv_start + BLOCK_KV
        "setp.ne.u32 p_causal, %9, 0;\n"        // causal flag
        "setp.gt.s32 p_mask, col0, %8;\n"        // kv_start+64 > q_start
        "and.pred p_mask, p_mask, p_causal;\n"   // causal && ...
        "setp.gt.s32 pm, col0, %5;\n"            // kv_start+64 > seq_len
        "or.pred p_mask, p_mask, pm;\n"          // needs masking

        "@!p_mask bra SKIP_MASK;\n"
        "and.b32 lmod4, %2, 3;\n"          // lane_id % 4
        "shl.b32 lmod4, lmod4, 1;\n"       // *2
        "add.u32 col0_base, kv_start, lmod4;\n"

#define V4Z_MASK_NC(nc_off, sa, sb, sc, sd) \
        "add.u32 col0, col0_base, " #nc_off ";\n" \
        "add.u32 col1, col0, 1;\n" \
        /* causal mask: col > row → -FLT_MAX */ \
        "@p_causal setp.gt.s32 pm, col0, %26;\n" \
        "@pm mov.f32 " #sa ", 0fff7fffff;\n" \
        "@p_causal setp.gt.s32 pm, col1, %26;\n" \
        "@pm mov.f32 " #sb ", 0fff7fffff;\n" \
        "@p_causal setp.gt.s32 pm, col0, %27;\n" \
        "@pm mov.f32 " #sc ", 0fff7fffff;\n" \
        "@p_causal setp.gt.s32 pm, col1, %27;\n" \
        "@pm mov.f32 " #sd ", 0fff7fffff;\n" \
        /* OOB mask: col >= seq_len → -FLT_MAX */ \
        "setp.ge.s32 pm, col0, %5;\n" \
        "@pm mov.f32 " #sa ", 0fff7fffff;\n" \
        "@pm mov.f32 " #sc ", 0fff7fffff;\n" \
        "setp.ge.s32 pm, col1, %5;\n" \
        "@pm mov.f32 " #sb ", 0fff7fffff;\n" \
        "@pm mov.f32 " #sd ", 0fff7fffff;\n"

        V4Z_MASK_NC(0,  s00,s01,s02,s03)
        V4Z_MASK_NC(8,  s10,s11,s12,s13)
        V4Z_MASK_NC(16, s20,s21,s22,s23)
        V4Z_MASK_NC(24, s30,s31,s32,s33)
        V4Z_MASK_NC(32, s40,s41,s42,s43)
        V4Z_MASK_NC(40, s50,s51,s52,s53)
        V4Z_MASK_NC(48, s60,s61,s62,s63)
        V4Z_MASK_NC(56, s70,s71,s72,s73)

#undef V4Z_MASK_NC

        "SKIP_MASK:\n"
        "\n"

        // ================================================================
        // Row max reduction
        // ================================================================
        // Row 0 (d0/d1)
        "max.f32 mx0, s00, s01;\n"
        "max.f32 tmp, s10, s11; max.f32 mx0, mx0, tmp;\n"
        "max.f32 tmp, s20, s21; max.f32 mx0, mx0, tmp;\n"
        "max.f32 tmp, s30, s31; max.f32 mx0, mx0, tmp;\n"
        "max.f32 tmp, s40, s41; max.f32 mx0, mx0, tmp;\n"
        "max.f32 tmp, s50, s51; max.f32 mx0, mx0, tmp;\n"
        "max.f32 tmp, s60, s61; max.f32 mx0, mx0, tmp;\n"
        "max.f32 tmp, s70, s71; max.f32 mx0, mx0, tmp;\n"
        "shfl.sync.bfly.b32 tmp, mx0, 1, 31, 0xffffffff;\n"
        "max.f32 mx0, mx0, tmp;\n"
        "shfl.sync.bfly.b32 tmp, mx0, 2, 31, 0xffffffff;\n"
        "max.f32 mx0, mx0, tmp;\n"
        // Row 1 (d2/d3)
        "max.f32 mx1, s02, s03;\n"
        "max.f32 tmp, s12, s13; max.f32 mx1, mx1, tmp;\n"
        "max.f32 tmp, s22, s23; max.f32 mx1, mx1, tmp;\n"
        "max.f32 tmp, s32, s33; max.f32 mx1, mx1, tmp;\n"
        "max.f32 tmp, s42, s43; max.f32 mx1, mx1, tmp;\n"
        "max.f32 tmp, s52, s53; max.f32 mx1, mx1, tmp;\n"
        "max.f32 tmp, s62, s63; max.f32 mx1, mx1, tmp;\n"
        "max.f32 tmp, s72, s73; max.f32 mx1, mx1, tmp;\n"
        "shfl.sync.bfly.b32 tmp, mx1, 1, 31, 0xffffffff;\n"
        "max.f32 mx1, mx1, tmp;\n"
        "shfl.sync.bfly.b32 tmp, mx1, 2, 31, 0xffffffff;\n"
        "max.f32 mx1, mx1, tmp;\n"
        "\n"

        // ================================================================
        // New max + rescale O + row_sum
        // ================================================================
        "max.f32 nmx0, rmax0, mx0;\n"
        "max.f32 nmx1, rmax1, mx1;\n"
        "sub.f32 rs0, rmax0, nmx0; ex2.approx.f32 rs0, rs0;\n"
        "sub.f32 rs1, rmax1, nmx1; ex2.approx.f32 rs1, rs1;\n"
        // Rescale O
        "mul.f32 o00, o00, rs0; mul.f32 o01, o01, rs0;\n"
        "mul.f32 o02, o02, rs1; mul.f32 o03, o03, rs1;\n"
        "mul.f32 o10, o10, rs0; mul.f32 o11, o11, rs0;\n"
        "mul.f32 o12, o12, rs1; mul.f32 o13, o13, rs1;\n"
        "mul.f32 o20, o20, rs0; mul.f32 o21, o21, rs0;\n"
        "mul.f32 o22, o22, rs1; mul.f32 o23, o23, rs1;\n"
        "mul.f32 o30, o30, rs0; mul.f32 o31, o31, rs0;\n"
        "mul.f32 o32, o32, rs1; mul.f32 o33, o33, rs1;\n"
        "mul.f32 o40, o40, rs0; mul.f32 o41, o41, rs0;\n"
        "mul.f32 o42, o42, rs1; mul.f32 o43, o43, rs1;\n"
        "mul.f32 o50, o50, rs0; mul.f32 o51, o51, rs0;\n"
        "mul.f32 o52, o52, rs1; mul.f32 o53, o53, rs1;\n"
        "mul.f32 o60, o60, rs0; mul.f32 o61, o61, rs0;\n"
        "mul.f32 o62, o62, rs1; mul.f32 o63, o63, rs1;\n"
        "mul.f32 o70, o70, rs0; mul.f32 o71, o71, rs0;\n"
        "mul.f32 o72, o72, rs1; mul.f32 o73, o73, rs1;\n"
        // Rescale sums
        "mul.f32 rsum0, rsum0, rs0;\n"
        "mul.f32 rsum1, rsum1, rs1;\n"
        "mov.f32 rmax0, nmx0;\n"
        "mov.f32 rmax1, nmx1;\n"
        "\n"

        // ================================================================
        // Prefetch next K/V tile (INSIDE PTX — overlaps exp2f+PV)
        // ================================================================
        // Check if not last block
        "add.u32 nxt_buf, kv_block, 1;\n"
        "setp.ge.u32 p_last, nxt_buf, %6;\n"
        "@p_last bra SKIP_PREFETCH;\n"

        // nxt_buf_off = ((kv_block+1) & 1) * kv_smem_stride
        "and.b32 nxt_buf, nxt_buf, 1;\n"
        "mul.lo.u32 nxt_buf, nxt_buf, %7;\n"

        // kv_start_nxt = (kv_block+1) * 64
        "add.u32 gkv, kv_start, 64;\n"  // gkv = base row for next block

        // Prefetch 4 chunks for K and V
        // chunk c: row = pf_row[c], global_kv_row = gkv + row
        // K smem addr = smem_K_base + nxt_buf + pf_smem_off[c]
        // K global addr = K_bh + (gkv + pf_row[c]) * HEAD_DIM * 2 + col*2
        //               = K_bh + gkv*128 + pf_gmem_off[c]
        // Same for V

        // Compute gkv_byte_base = gkv * HEAD_DIM * 2 = gkv * 128 = gkv << 7
        ".reg .u32 gkv_byte_base;\n"
        "shl.b32 gkv_byte_base, gkv, 7;\n"  // gkv * 128

        // Prefetch chunk 0
        "add.u32 gkv_row, gkv, %28;\n"  // pf_row[0]
        "setp.lt.u32 p_pf, gkv_row, %5;\n"
        // K
        "add.u32 saddr, %0, nxt_buf;\n"
        "add.u32 saddr, saddr, %32;\n"  // + pf_smem_off[0]
        ".reg .u32 goff0;\n"
        "add.u32 goff0, gkv_byte_base, %36;\n"  // + pf_gmem_off[0]
        ".reg .u64 goff0_64;\n"
        "cvt.u64.u32 goff0_64, goff0;\n"
        "add.u64 gaddr64, %3, goff0_64;\n"       // K_bh + offset
        "@p_pf cp.async.cg.shared.global [saddr], [gaddr64], 16;\n"
        ".reg .u32 z; mov.u32 z, 0;\n"
        "@!p_pf st.shared.v4.u32 [saddr], {z, z, z, z};\n"
        // V
        "add.u32 saddr, %1, nxt_buf;\n"
        "add.u32 saddr, saddr, %32;\n"
        "add.u64 gaddr64, %4, goff0_64;\n"
        "@p_pf cp.async.cg.shared.global [saddr], [gaddr64], 16;\n"
        "@!p_pf st.shared.v4.u32 [saddr], {z, z, z, z};\n"

        // Prefetch chunk 1
        "add.u32 gkv_row, gkv, %29;\n"
        "setp.lt.u32 p_pf, gkv_row, %5;\n"
        "add.u32 saddr, %0, nxt_buf;\n"
        "add.u32 saddr, saddr, %33;\n"
        ".reg .u32 goff1;\n"
        "add.u32 goff1, gkv_byte_base, %37;\n"
        ".reg .u64 goff1_64;\n"
        "cvt.u64.u32 goff1_64, goff1;\n"
        "add.u64 gaddr64, %3, goff1_64;\n"
        "@p_pf cp.async.cg.shared.global [saddr], [gaddr64], 16;\n"
        "@!p_pf st.shared.v4.u32 [saddr], {z, z, z, z};\n"
        "add.u32 saddr, %1, nxt_buf;\n"
        "add.u32 saddr, saddr, %33;\n"
        "add.u64 gaddr64, %4, goff1_64;\n"
        "@p_pf cp.async.cg.shared.global [saddr], [gaddr64], 16;\n"
        "@!p_pf st.shared.v4.u32 [saddr], {z, z, z, z};\n"

        // Prefetch chunk 2
        "add.u32 gkv_row, gkv, %30;\n"
        "setp.lt.u32 p_pf, gkv_row, %5;\n"
        "add.u32 saddr, %0, nxt_buf;\n"
        "add.u32 saddr, saddr, %34;\n"
        ".reg .u32 goff2;\n"
        "add.u32 goff2, gkv_byte_base, %38;\n"
        ".reg .u64 goff2_64;\n"
        "cvt.u64.u32 goff2_64, goff2;\n"
        "add.u64 gaddr64, %3, goff2_64;\n"
        "@p_pf cp.async.cg.shared.global [saddr], [gaddr64], 16;\n"
        "@!p_pf st.shared.v4.u32 [saddr], {z, z, z, z};\n"
        "add.u32 saddr, %1, nxt_buf;\n"
        "add.u32 saddr, saddr, %34;\n"
        "add.u64 gaddr64, %4, goff2_64;\n"
        "@p_pf cp.async.cg.shared.global [saddr], [gaddr64], 16;\n"
        "@!p_pf st.shared.v4.u32 [saddr], {z, z, z, z};\n"

        // Prefetch chunk 3
        "add.u32 gkv_row, gkv, %31;\n"
        "setp.lt.u32 p_pf, gkv_row, %5;\n"
        "add.u32 saddr, %0, nxt_buf;\n"
        "add.u32 saddr, saddr, %35;\n"
        ".reg .u32 goff3;\n"
        "add.u32 goff3, gkv_byte_base, %39;\n"
        ".reg .u64 goff3_64;\n"
        "cvt.u64.u32 goff3_64, goff3;\n"
        "add.u64 gaddr64, %3, goff3_64;\n"
        "@p_pf cp.async.cg.shared.global [saddr], [gaddr64], 16;\n"
        "@!p_pf st.shared.v4.u32 [saddr], {z, z, z, z};\n"
        "add.u32 saddr, %1, nxt_buf;\n"
        "add.u32 saddr, saddr, %35;\n"
        "add.u64 gaddr64, %4, goff3_64;\n"
        "@p_pf cp.async.cg.shared.global [saddr], [gaddr64], 16;\n"
        "@!p_pf st.shared.v4.u32 [saddr], {z, z, z, z};\n"

        "cp.async.commit_group;\n"
        "SKIP_PREFETCH:\n"
        "\n"

        // ================================================================
        // exp2f + sum + pack P + PV MMA (interleaved)
        // ================================================================

// V load + 2 MMA macro
#define V4Z_VMMA(kc, nc_pair, oa, ob, oc, od, oe, of, og, oh) \
        /* V address: v_row = kc*16 + (sub%2)*8 + tsub */ \
        /* v_col = (nc_pair*2 + sub/2) * 8 */ \
        "mad.lo.u32 v_row, " #kc ", 16, tsub;\n" \
        "shl.b32 vidx, smod2, 3;\n" \
        "add.u32 v_row, v_row, vidx;\n" \
        "add.u32 vcol, " #nc_pair ", " #nc_pair ";\n" \
        "add.u32 vcol, vcol, hsub;\n" \
        "shl.b32 vcol, vcol, 3;\n" \
        /* swizzle */ \
        "and.b32 v_rmask, v_row, 7;\n" \
        "shl.b32 v_rmask, v_rmask, 3;\n" \
        "xor.b32 vsw_col, vcol, v_rmask;\n" \
        "mad.lo.u32 vidx, v_row, 64, vsw_col;\n" \
        "shl.b32 vidx, vidx, 1;\n" \
        "add.u32 vaddr, %1, vidx;\n" \
        "add.u32 vaddr, vaddr, cur_buf;\n" \
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {v0,v1,v2,v3}, [vaddr];\n" \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{" #oa "," #ob "," #oc "," #od "}, {pa0,pa1,pa2,pa3}, {v0,v1}, " \
        "{" #oa "," #ob "," #oc "," #od "};\n" \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{" #oe "," #of "," #og "," #oh "}, {pa0,pa1,pa2,pa3}, {v2,v3}, " \
        "{" #oe "," #of "," #og "," #oh "};\n"

        // --- kc=0: exp2f S[0..1], pack P, V load + MMA ---
        "sub.f32 s00, s00, nmx0; ex2.approx.f32 s00, s00;\n"
        "sub.f32 s01, s01, nmx0; ex2.approx.f32 s01, s01;\n"
        "sub.f32 s02, s02, nmx1; ex2.approx.f32 s02, s02;\n"
        "sub.f32 s03, s03, nmx1; ex2.approx.f32 s03, s03;\n"
        "sub.f32 s10, s10, nmx0; ex2.approx.f32 s10, s10;\n"
        "sub.f32 s11, s11, nmx0; ex2.approx.f32 s11, s11;\n"
        "sub.f32 s12, s12, nmx1; ex2.approx.f32 s12, s12;\n"
        "sub.f32 s13, s13, nmx1; ex2.approx.f32 s13, s13;\n"
        // Local sum kc=0
        "mov.f32 ls0, 0f00000000; mov.f32 ls1, 0f00000000;\n"
        "add.f32 ls0, s00, s01; add.f32 ls1, s02, s03;\n"
        "add.f32 ls0, ls0, s10; add.f32 ls0, ls0, s11;\n"
        "add.f32 ls1, ls1, s12; add.f32 ls1, ls1, s13;\n"
        // Pack P (CVT reversed: hi, lo)
        "cvt.rn.bf16x2.f32 pa0, s01, s00;\n"
        "cvt.rn.bf16x2.f32 pa1, s03, s02;\n"
        "cvt.rn.bf16x2.f32 pa2, s11, s10;\n"
        "cvt.rn.bf16x2.f32 pa3, s13, s12;\n"
        // V[kc=0][nc=0,1] + MMA — interleave exp2f for kc=1
        V4Z_VMMA(0, 0, o00,o01,o02,o03, o10,o11,o12,o13)
        "sub.f32 s20, s20, nmx0; ex2.approx.f32 s20, s20;\n"
        "sub.f32 s21, s21, nmx0; ex2.approx.f32 s21, s21;\n"
        "sub.f32 s22, s22, nmx1; ex2.approx.f32 s22, s22;\n"
        "sub.f32 s23, s23, nmx1; ex2.approx.f32 s23, s23;\n"
        V4Z_VMMA(0, 1, o20,o21,o22,o23, o30,o31,o32,o33)
        "sub.f32 s30, s30, nmx0; ex2.approx.f32 s30, s30;\n"
        "sub.f32 s31, s31, nmx0; ex2.approx.f32 s31, s31;\n"
        "sub.f32 s32, s32, nmx1; ex2.approx.f32 s32, s32;\n"
        "sub.f32 s33, s33, nmx1; ex2.approx.f32 s33, s33;\n"
        V4Z_VMMA(0, 2, o40,o41,o42,o43, o50,o51,o52,o53)
        V4Z_VMMA(0, 3, o60,o61,o62,o63, o70,o71,o72,o73)

        // --- kc=1: sum + pack + MMA ---
        "add.f32 ls0, ls0, s20; add.f32 ls0, ls0, s21;\n"
        "add.f32 ls1, ls1, s22; add.f32 ls1, ls1, s23;\n"
        "add.f32 ls0, ls0, s30; add.f32 ls0, ls0, s31;\n"
        "add.f32 ls1, ls1, s32; add.f32 ls1, ls1, s33;\n"
        "cvt.rn.bf16x2.f32 pa0, s21, s20;\n"
        "cvt.rn.bf16x2.f32 pa1, s23, s22;\n"
        "cvt.rn.bf16x2.f32 pa2, s31, s30;\n"
        "cvt.rn.bf16x2.f32 pa3, s33, s32;\n"
        V4Z_VMMA(1, 0, o00,o01,o02,o03, o10,o11,o12,o13)
        "sub.f32 s40, s40, nmx0; ex2.approx.f32 s40, s40;\n"
        "sub.f32 s41, s41, nmx0; ex2.approx.f32 s41, s41;\n"
        "sub.f32 s42, s42, nmx1; ex2.approx.f32 s42, s42;\n"
        "sub.f32 s43, s43, nmx1; ex2.approx.f32 s43, s43;\n"
        V4Z_VMMA(1, 1, o20,o21,o22,o23, o30,o31,o32,o33)
        "sub.f32 s50, s50, nmx0; ex2.approx.f32 s50, s50;\n"
        "sub.f32 s51, s51, nmx0; ex2.approx.f32 s51, s51;\n"
        "sub.f32 s52, s52, nmx1; ex2.approx.f32 s52, s52;\n"
        "sub.f32 s53, s53, nmx1; ex2.approx.f32 s53, s53;\n"
        V4Z_VMMA(1, 2, o40,o41,o42,o43, o50,o51,o52,o53)
        V4Z_VMMA(1, 3, o60,o61,o62,o63, o70,o71,o72,o73)

        // --- kc=2: sum + pack + MMA ---
        "add.f32 ls0, ls0, s40; add.f32 ls0, ls0, s41;\n"
        "add.f32 ls1, ls1, s42; add.f32 ls1, ls1, s43;\n"
        "add.f32 ls0, ls0, s50; add.f32 ls0, ls0, s51;\n"
        "add.f32 ls1, ls1, s52; add.f32 ls1, ls1, s53;\n"
        "cvt.rn.bf16x2.f32 pa0, s41, s40;\n"
        "cvt.rn.bf16x2.f32 pa1, s43, s42;\n"
        "cvt.rn.bf16x2.f32 pa2, s51, s50;\n"
        "cvt.rn.bf16x2.f32 pa3, s53, s52;\n"
        V4Z_VMMA(2, 0, o00,o01,o02,o03, o10,o11,o12,o13)
        "sub.f32 s60, s60, nmx0; ex2.approx.f32 s60, s60;\n"
        "sub.f32 s61, s61, nmx0; ex2.approx.f32 s61, s61;\n"
        "sub.f32 s62, s62, nmx1; ex2.approx.f32 s62, s62;\n"
        "sub.f32 s63, s63, nmx1; ex2.approx.f32 s63, s63;\n"
        V4Z_VMMA(2, 1, o20,o21,o22,o23, o30,o31,o32,o33)
        "sub.f32 s70, s70, nmx0; ex2.approx.f32 s70, s70;\n"
        "sub.f32 s71, s71, nmx0; ex2.approx.f32 s71, s71;\n"
        "sub.f32 s72, s72, nmx1; ex2.approx.f32 s72, s72;\n"
        "sub.f32 s73, s73, nmx1; ex2.approx.f32 s73, s73;\n"
        V4Z_VMMA(2, 2, o40,o41,o42,o43, o50,o51,o52,o53)
        V4Z_VMMA(2, 3, o60,o61,o62,o63, o70,o71,o72,o73)

        // --- kc=3: sum + pack + MMA ---
        "add.f32 ls0, ls0, s60; add.f32 ls0, ls0, s61;\n"
        "add.f32 ls1, ls1, s62; add.f32 ls1, ls1, s63;\n"
        "add.f32 ls0, ls0, s70; add.f32 ls0, ls0, s71;\n"
        "add.f32 ls1, ls1, s72; add.f32 ls1, ls1, s73;\n"
        "cvt.rn.bf16x2.f32 pa0, s61, s60;\n"
        "cvt.rn.bf16x2.f32 pa1, s63, s62;\n"
        "cvt.rn.bf16x2.f32 pa2, s71, s70;\n"
        "cvt.rn.bf16x2.f32 pa3, s73, s72;\n"
        V4Z_VMMA(3, 0, o00,o01,o02,o03, o10,o11,o12,o13)
        V4Z_VMMA(3, 1, o20,o21,o22,o23, o30,o31,o32,o33)
        V4Z_VMMA(3, 2, o40,o41,o42,o43, o50,o51,o52,o53)
        V4Z_VMMA(3, 3, o60,o61,o62,o63, o70,o71,o72,o73)

#undef V4Z_VMMA

        // Deferred sum shuffle
        "shfl.sync.bfly.b32 tmp, ls0, 1, 31, 0xffffffff;\n"
        "add.f32 ls0, ls0, tmp;\n"
        "shfl.sync.bfly.b32 tmp, ls0, 2, 31, 0xffffffff;\n"
        "add.f32 ls0, ls0, tmp;\n"
        "shfl.sync.bfly.b32 tmp, ls1, 1, 31, 0xffffffff;\n"
        "add.f32 ls1, ls1, tmp;\n"
        "shfl.sync.bfly.b32 tmp, ls1, 2, 31, 0xffffffff;\n"
        "add.f32 ls1, ls1, tmp;\n"
        "add.f32 rsum0, rsum0, ls0;\n"
        "add.f32 rsum1, rsum1, ls1;\n"
        "\n"

        // ================================================================
        // Barrier: wait for prefetch and sync
        // ================================================================
        "cp.async.wait_group 0;\n"
        "bar.sync 0;\n"

        // Loop increment
        "add.u32 kv_block, kv_block, 1;\n"
        "bra KV_LOOP;\n"
        "KV_DONE:\n"
        "\n"

        // ================================================================
        // Final normalize + output store (all via st.global)
        // ================================================================

        // inv_sum = 1/row_sum (rcp.approx)
        // Guard: if rsum <= 0 set inv to 0
        "setp.gt.f32 p_row0, rsum0, 0f00000000;\n"
        "setp.gt.f32 p_row1, rsum1, 0f00000000;\n"
        "mov.f32 inv0, 0f00000000;\n"
        "mov.f32 inv1, 0f00000000;\n"
        "@p_row0 rcp.approx.f32 inv0, rsum0;\n"
        "@p_row1 rcp.approx.f32 inv1, rsum1;\n"

        // Normalize O
        "mul.f32 o00, o00, inv0; mul.f32 o01, o01, inv0;\n"
        "mul.f32 o02, o02, inv1; mul.f32 o03, o03, inv1;\n"
        "mul.f32 o10, o10, inv0; mul.f32 o11, o11, inv0;\n"
        "mul.f32 o12, o12, inv1; mul.f32 o13, o13, inv1;\n"
        "mul.f32 o20, o20, inv0; mul.f32 o21, o21, inv0;\n"
        "mul.f32 o22, o22, inv1; mul.f32 o23, o23, inv1;\n"
        "mul.f32 o30, o30, inv0; mul.f32 o31, o31, inv0;\n"
        "mul.f32 o32, o32, inv1; mul.f32 o33, o33, inv1;\n"
        "mul.f32 o40, o40, inv0; mul.f32 o41, o41, inv0;\n"
        "mul.f32 o42, o42, inv1; mul.f32 o43, o43, inv1;\n"
        "mul.f32 o50, o50, inv0; mul.f32 o51, o51, inv0;\n"
        "mul.f32 o52, o52, inv1; mul.f32 o53, o53, inv1;\n"
        "mul.f32 o60, o60, inv0; mul.f32 o61, o61, inv0;\n"
        "mul.f32 o62, o62, inv1; mul.f32 o63, o63, inv1;\n"
        "mul.f32 o70, o70, inv0; mul.f32 o71, o71, inv0;\n"
        "mul.f32 o72, o72, inv1; mul.f32 o73, o73, inv1;\n"
        "\n"

        // Store O via st.global.b32 (pack bf16x2)
        // col0 = nc*8 + (lane_id %% 4)*2
        // O_bh byte offset for row r, col c: (r * 64 + c) * 2
        "and.b32 lmod4_out, %2, 3;\n"
        "shl.b32 lmod4_out, lmod4_out, 1;\n"  // (lane_id%4)*2

        // global_row0 check
        "setp.lt.s32 p_row0, %26, %5;\n"  // global_row0 < seq_len
        "setp.lt.s32 p_row1, %27, %5;\n"  // global_row1 < seq_len

        // Macro: store one nc chunk
#define V4Z_STORE_NC(nc, oa, ob, oc, od) \
        /* col = nc*8 + lmod4_out */ \
        "mad.lo.u32 out_col, " #nc ", 8, lmod4_out;\n" \
        /* row0 store */ \
        "@p_row0 mad.lo.u32 out_off, %26, 64, out_col;\n" \
        "@p_row0 shl.b32 out_off, out_off, 1;\n" \
        ".reg .u64 ooff_" #nc "_0;\n" \
        "@p_row0 cvt.u64.u32 ooff_" #nc "_0, out_off;\n" \
        "@p_row0 add.u64 out_addr64, %40, ooff_" #nc "_0;\n" \
        "@p_row0 cvt.rn.bf16x2.f32 out_packed, " #ob ", " #oa ";\n" \
        "@p_row0 st.global.b32 [out_addr64], out_packed;\n" \
        /* row1 store */ \
        "@p_row1 mad.lo.u32 out_off, %27, 64, out_col;\n" \
        "@p_row1 shl.b32 out_off, out_off, 1;\n" \
        ".reg .u64 ooff_" #nc "_1;\n" \
        "@p_row1 cvt.u64.u32 ooff_" #nc "_1, out_off;\n" \
        "@p_row1 add.u64 out_addr64, %40, ooff_" #nc "_1;\n" \
        "@p_row1 cvt.rn.bf16x2.f32 out_packed, " #od ", " #oc ";\n" \
        "@p_row1 st.global.b32 [out_addr64], out_packed;\n"

        V4Z_STORE_NC(0, o00,o01,o02,o03)
        V4Z_STORE_NC(1, o10,o11,o12,o13)
        V4Z_STORE_NC(2, o20,o21,o22,o23)
        V4Z_STORE_NC(3, o30,o31,o32,o33)
        V4Z_STORE_NC(4, o40,o41,o42,o43)
        V4Z_STORE_NC(5, o50,o51,o52,o53)
        V4Z_STORE_NC(6, o60,o61,o62,o63)
        V4Z_STORE_NC(7, o70,o71,o72,o73)

#undef V4Z_STORE_NC

        // Store logsumexp: L[row] = row_max * ln(2) + log(row_sum)
        // Only lane_id % 4 == 0 writes (others have duplicate)
        "and.b32 lmod4_out, %2, 3;\n"
        "setp.eq.u32 p_lse, lmod4_out, 0;\n"
        "and.pred p_lse, p_lse, p_row0;\n"  // lane0 && row0 valid

        // row0 LSE
        "@p_lse mul.f32 lse0, rmax0, 0f3f317218;\n"  // rmax0 * ln(2)
        "@p_lse lg2.approx.f32 lg0, rsum0;\n"
        "@p_lse mul.f32 lg0, lg0, 0f3f317218;\n"      // lg2 * ln(2) = ln
        "@p_lse add.f32 lse0, lse0, lg0;\n"
        // L_bh offset = global_row0 * 4
        ".reg .u32 loff0;\n"
        "@p_lse shl.b32 loff0, %26, 2;\n"
        ".reg .u64 loff0_64;\n"
        "@p_lse cvt.u64.u32 loff0_64, loff0;\n"
        ".reg .u64 laddr0;\n"
        "@p_lse add.u64 laddr0, %41, loff0_64;\n"
        "@p_lse st.global.f32 [laddr0], lse0;\n"

        // row1 LSE
        "and.b32 lmod4_out, %2, 3;\n"
        "setp.eq.u32 p_lse, lmod4_out, 0;\n"
        "and.pred p_lse, p_lse, p_row1;\n"

        "@p_lse mul.f32 lse1, rmax1, 0f3f317218;\n"
        "@p_lse lg2.approx.f32 lg1, rsum1;\n"
        "@p_lse mul.f32 lg1, lg1, 0f3f317218;\n"
        "@p_lse add.f32 lse1, lse1, lg1;\n"
        ".reg .u32 loff1;\n"
        "@p_lse shl.b32 loff1, %27, 2;\n"
        ".reg .u64 loff1_64;\n"
        "@p_lse cvt.u64.u32 loff1_64, loff1;\n"
        ".reg .u64 laddr1;\n"
        "@p_lse add.u64 laddr1, %41, loff1_64;\n"
        "@p_lse st.global.f32 [laddr1], lse1;\n"

        "}\n"  // end PTX block

        // Operand list: ZERO outputs, inputs only
        :  /* no outputs */
        :  "r"(smem_K_u32),          // %0
           "r"(smem_V_u32),          // %1
           "r"(lane_id),             // %2
           "l"(K_bh_u64),           // %3
           "l"(V_bh_u64),           // %4
           "r"(seq_len),            // %5
           "r"(num_kv_blocks),      // %6
           "r"(kv_smem_stride),     // %7
           "r"(q_start),            // %8
           "r"(causal_u32),         // %9
           // Q_rmem[t][0..3][0..3] = %10..%25
           "r"(Q_rmem[t][0][0]), "r"(Q_rmem[t][0][1]), "r"(Q_rmem[t][0][2]), "r"(Q_rmem[t][0][3]),  // %10-13
           "r"(Q_rmem[t][1][0]), "r"(Q_rmem[t][1][1]), "r"(Q_rmem[t][1][2]), "r"(Q_rmem[t][1][3]),  // %14-17
           "r"(Q_rmem[t][2][0]), "r"(Q_rmem[t][2][1]), "r"(Q_rmem[t][2][2]), "r"(Q_rmem[t][2][3]),  // %18-21
           "r"(Q_rmem[t][3][0]), "r"(Q_rmem[t][3][1]), "r"(Q_rmem[t][3][2]), "r"(Q_rmem[t][3][3]),  // %22-25
           "r"(global_row0),        // %26
           "r"(global_row1),        // %27
           // Prefetch row indices
           "r"(pf_row[0]),          // %28
           "r"(pf_row[1]),          // %29
           "r"(pf_row[2]),          // %30
           "r"(pf_row[3]),          // %31
           // Prefetch smem offsets
           "r"(pf_smem_off[0]),     // %32
           "r"(pf_smem_off[1]),     // %33
           "r"(pf_smem_off[2]),     // %34
           "r"(pf_smem_off[3]),     // %35
           // Prefetch gmem offsets
           "r"(pf_gmem_off[0]),     // %36
           "r"(pf_gmem_off[1]),     // %37
           "r"(pf_gmem_off[2]),     // %38
           "r"(pf_gmem_off[3]),     // %39
           // Output pointers
           "l"(O_bh_u64),           // %40
           "l"(L_bh_u64)            // %41
        : "memory"
        );

    } // end per-tile
}

// ============================================================
// Host launch
// ============================================================

namespace bk {

void flash_attn_v4_fwd(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    float *L,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream)
{
    int bh = batch_size * num_heads;
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_KV = 64;

    auto compute_smem = [](int hd, int bkv) {
        int kv_elems = bkv * hd;
        return 4 * kv_elems * (int)sizeof(__nv_bfloat16);
    };

    auto launch = [&](auto kernel_fn, int smem_bytes, int block_q) {
        int num_q_blocks = (seq_len + block_q - 1) / block_q;
        dim3 grid(num_q_blocks, bh);
        dim3 block(V4_THREADS);
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(kernel_fn,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        }
        kernel_fn<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, L, seq_len, scale, causal);
    };

    int smem_bytes = compute_smem(HEAD_DIM, BLOCK_KV);
    // v4 zero-interface PTX: BQ=64 only. The per-tile KV loop architecture
    // means BQ=128 would double K/V loads (each tile runs its own KV loop).
    // BQ=128 support requires restructuring PTX to process both tiles per iteration.
    launch(flash_attn_v4_kernel<64>, smem_bytes, 64);
}

} // namespace bk

// ============================================================
// PyTorch bindings
// ============================================================

torch::Tensor flash_attn_v4_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
    bool causal)
{
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be BF16");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_cuda() && K.dtype() == torch::kBFloat16 && K.is_contiguous());
    TORCH_CHECK(V.is_cuda() && V.dtype() == torch::kBFloat16 && V.is_contiguous());

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    TORCH_CHECK(D == 64, "v4 kernel only supports head_dim=64");

    Q = Q.reshape({B * H, N, D});
    K = K.reshape({B * H, N, D});
    V = V.reshape({B * H, N, D});

    auto O_out = torch::empty_like(Q);
    auto L_out = torch::empty({B * H, N}, Q.options().dtype(torch::kFloat32));

    bk::flash_attn_v4_fwd(
        reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(O_out.data_ptr()),
        L_out.data_ptr<float>(),
        B, H, N, D, scale, causal,
        at::cuda::getCurrentCUDAStream());

    return O_out.reshape({B, H, N, D});
}
