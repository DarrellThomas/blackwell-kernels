<!-- SUMMARY
SIGNAL: Fused GroupNorm+Linear GEMM: multi-stage pipeline, split-K, epilogue fusion, tile tuning to close 2-5x gap vs cuBLAS [gemm, validated+theoretical]
WHAT: Comprehensive optimization playbook for making a custom mma.sync m16n8k16 GEMM competitive with cuBLAS on sm_120a, specifically for a fused GroupNorm+Linear kernel
FOR: GEMM, fused kernel, sm_120a RTX 5090, 2-5x performance gap at C>=640
FINDING: The gap is likely caused by missing multi-stage cp.async pipelining, suboptimal tile sizes, and lack of split-K for skinny-M shapes. CUTLASS 2.x achieves 95%+ cuBLAS with 3-4 pipeline stages, 128x128 or 64x64 tiles with occupancy-first tuning, and double-buffered register fragments. On sm_120a specifically, 64x64 tiles with 6 blocks/SM (0.98x cuBLAS) is the proven sweet spot.
TECHNIQUE: Implement cp.async multi-stage pipelining (2 stages, per hard-won lessons 3-stage kills L1), use 64x64 tiles with 4 warps and 80 regs for 6 blocks/SM occupancy, add split-K for small-M cases, fuse GroupNorm epilogue into GEMM accumulator in registers, use XOR swizzle for bank-conflict-free smem.
STATUS: validated (occupancy/tile tuning) + theoretical (split-K, full pipeline for fused kernel) — based on factory hard-won lessons + CUTLASS patterns + multiple reference implementations
-->

# Fused GroupNorm+Linear GEMM -- Optimization Playbook for sm_120a

## Problem Statement

Custom mma.sync m16n8k16 BF16 GEMM is 2-5x slower than cuBLAS for C>=640 in a
fused GroupNorm+Linear kernel. The fusion saves ~5us from eliminating the
GroupNorm->GEMM memory round-trip, but the GEMM must be within ~1.2x of cuBLAS
for the fusion to be net-positive.

**Target:** GEMM within 1.0-1.2x cuBLAS so fusion savings dominate.

---

## 1. PROVEN: Occupancy-First Tile Tuning (0.88x -> 0.98x cuBLAS)

**This is the #1 technique.** Already validated on sm_120a in the factory BF16 GEMM.

### The Recipe (from 04_HARD_WON_LESSONS.md)

| Parameter | Bad Config | Good Config |
|-----------|-----------|-------------|
| Tile size | 128x128 | **64x64** |
| Warps/block | 8 | **4** |
| MMA ops/block | 32 (4x4) | **16 (2x4)** |
| Registers/thread | ~125 | **80** |
| Blocks/SM | 2 | **6** |
| Warps/SM | 16 | **24** |
| vs cuBLAS | 0.88x | **0.98x** |

### Why It Works

sm_120a has 48 warps/SM. With 6 blocks x 4 warps = 24 warps active, the hardware
warp scheduler has 5 backup warps per sub-partition when one stalls on
math_pipe_throttle. The key insight: **24 warps with 16 MMAs each >> 16 warps
with 32 MMAs each**.

### Implementation Checklist

- [ ] `__launch_bounds__(128, 6)` -- forces compiler to cap at 80 regs
- [ ] Target 16KB smem/block x 6 blocks = 96KB (within 128KB SM limit)
- [ ] Non-volatile MMA (`asm` not `asm volatile`) for free compiler reordering
- [ ] `ldmatrix_x4_mma()` with baked-in a1/a2 swap (eliminates MOV instructions)
- [ ] Stream B fragments (load per K-tile, not preload-all) to minimize live regs
- [ ] Verify: `cuobjdump --dump-resource-usage` shows 80 regs, 0 spills

### Register Budget for 64x64 Tile with m16n8k16

```
Accumulators: 2x4 MMA tiles x 4 FP32 regs each = 32 regs
A fragments:  2 ldmatrix_x4 loads x 4 regs = 8 regs (streamed)
B fragments:  4 ldmatrix_x2 loads x 2 regs = 8 regs (streamed)
Pointers/loop vars/misc:                    ~20 regs
Epilogue (GroupNorm stats):                 ~12 regs
TOTAL:                                      ~80 regs  <-- fits!
```

---

## 2. PROVEN: cp.async Pipelining (2.54x speedup, load-bearing)

### Double-Buffer Pipeline (2 stages)

Already proven on sm_120a. This is the factory's validated pattern:

```
Stage 0: cp.async.cg loads tile[k] into smem_buf[0]
Stage 1: cp.async.cg loads tile[k+1] into smem_buf[1]
         MMA computes on smem_buf[0]
Stage 2: cp.async.cg loads tile[k+2] into smem_buf[0]
         MMA computes on smem_buf[1]
...
```

**Critical:** Use `cp.async.cg` (bypass L1) with 16-byte loads for maximum
global memory throughput. Each cp.async issues a vectorized LDG.128.

### Why NOT 3+ Stages on sm_120a

**3-stage pipeline kills L1 cache.** sm_120 has a unified 128KB L1/smem. More
smem = less L1. Triple-buffering pushes total smem past the L1 thrash point.
This is a confirmed dead end from the factory -- every project that tried it
regressed. **Double-buffer is the sweet spot.**

Smem budget check for 2-stage 64x64 GEMM:
```
A buffer: 64 x 32 x 2 bytes x 2 stages = 8 KB
B buffer: 32 x 64 x 2 bytes x 2 stages = 8 KB
Total:    16 KB per block
6 blocks: 96 KB  (within 128 KB SM limit, leaves 32KB for L1)
```

### Pipeline Implementation Pattern (from CUTLASS mma_multistage.h)

```
// Prologue: fill pipeline (kStages-1 = 1 initial load)
cp_async_group_begin();
load_tile_to_smem(buf[0], k=0);
cp_async_group_commit();

// Mainloop
for (int k = 0; k < K; k += BK) {
    // Wait for current buffer to be ready
    cp_async_wait<0>();
    __syncthreads();

    // Issue next load (into other buffer)
    cp_async_group_begin();
    load_tile_to_smem(buf[(k/BK+1) % 2], k+BK);
    cp_async_group_commit();

    // Compute MMA on current buffer
    for (int warp_k = 0; warp_k < BK; warp_k += 16) {
        ldmatrix A from smem_buf[k/BK % 2]
        ldmatrix B from smem_buf[k/BK % 2]
        mma.sync accumulate
    }
}
```

### Register Double-Buffering for Warp Fragments

CUTLASS double-buffers register fragments too:
```cpp
FragmentA warp_frag_A[2];  // alternating buffers
FragmentB warp_frag_B[2];

for (int warp_k = 0; warp_k < BK/16; warp_k++) {
    // Load NEXT fragment while computing CURRENT
    ldmatrix(warp_frag_A[(warp_k+1) % 2], ...);
    ldmatrix(warp_frag_B[(warp_k+1) % 2], ...);
    mma_sync(accum, warp_frag_A[warp_k % 2], warp_frag_B[warp_k % 2]);
}
```

**WARNING from hard-won lessons:** On sm_120a, explicit B fragment
double-buffering in PTX HURT performance (0.89x -> 0.68x cuBLAS) because the
extra registers reduced occupancy and `asm volatile` fought the compiler. The
compiler + hardware warp scheduler already overlap ldmatrix and mma.sync across
warps when you use non-volatile asm and `#pragma unroll`. Let the compiler
handle register-level scheduling.

---

## 3. XOR Swizzle for Bank-Conflict-Free Shared Memory

### The Formula

For BF16 with BLOCK_K=32 (64 bytes per row):
```
NUM_CHUNKS = 32 / 8 = 4    // 8 BF16 elements = 16 bytes per chunk
SWIZZLE_BITS = 2            // log2(4) = 2
swizzled_col = col ^ ((row & 3) << 3)
```

For BF16 with BLOCK_N=64 (128 bytes per row):
```
NUM_CHUNKS = 64 / 8 = 8
SWIZZLE_BITS = 3            // log2(8) = 3
swizzled_col = col ^ ((row & 7) << 3)
```

### Critical Detail: Match Swizzle Period to Row Width

If the row width in chunks exceeds 2^SWIZZLE_BITS, the swizzle doesn't fully
cover the column space and bank conflicts remain. For BLOCK_N=128:
- Need SWIZZLE_BITS=4 (not 3) -- see gemm_bank_conflict_wide_tile_swizzle.md
- Use CUTLASS Swizzle<4,3,4> equivalent

For the fused kernel with 64x64 tiles, SWIZZLE_BITS=3 with BK=32 is sufficient
and matches the factory's proven pattern.

### ldmatrix Alignment

ldmatrix.sync.aligned.m8n8.x4 requires 16-byte aligned addresses. With XOR
swizzle, the swizzled address must preserve 16-byte alignment. The swizzle
operates on chunk indices (not byte addresses), so alignment is automatic when
chunks are 16 bytes (8 BF16 elements).

---

## 4. Split-K for Small-M Shapes

When M is small (e.g., batch_size * spatial_dim is small) but K and N are large,
the standard data-parallel decomposition launches too few thread blocks to fill
170 SMs.

### When to Use Split-K

```
tiles_M = ceil(M / 64)
tiles_N = ceil(N / 64)
total_tiles = tiles_M * tiles_N

If total_tiles < 170:  --> split-K is beneficial
If total_tiles < 85:   --> split-K is critical (>50% SMs idle)
```

Example: M=64, N=640, K=640 with 64x64 tiles:
- tiles_M=1, tiles_N=10, total_tiles=10
- Only 10 of 170 SMs occupied = 5.9% utilization
- Split-K by 17: 170 blocks, each computing K/17 ≈ 38 elements of K

### Implementation

**Two-kernel approach:**
1. Partitioned GEMM: each block computes partial C (M x N) from a K-slice
2. Reduction kernel: sum partial results across K-partitions

```
// Kernel 1: Partitioned GEMM
dim3 grid(tiles_N, tiles_M, split_k_slices);
// Each block computes C_partial[slice] = A[:, k_start:k_end] @ B[k_start:k_end, :]

// Kernel 2: Reduction
// C_final = sum(C_partial[0:split_k_slices])
```

**Single-kernel approach (atomicAdd):**
- Each block atomicAdds its partial result to global C
- Simpler but has atomic contention for large split counts
- Works well for split_k <= 8-16

### Stream-K (Advanced)

Stream-K distributes K-iterations evenly across exactly `num_SMs` persistent
blocks, avoiding wave quantization entirely. More complex to implement but
eliminates the reduction kernel. Key insight: the scheduler falls back to
data-parallel when wave quantization is minimal, and to split-K when tiles
are few.

**Recommendation for fused kernel:** Start with simple split-K (atomicAdd)
for the M < 170*64 case. Only implement stream-K if the reduction kernel
overhead is problematic.

---

## 5. Epilogue Fusion: GroupNorm into GEMM Accumulator

This is the entire point of the fused kernel. The GroupNorm output feeds the
GEMM input, but the GEMM output may also need epilogue operations.

### Strategy A: GroupNorm as GEMM Prologue (Preferred)

GroupNorm output is the GEMM's A operand. Compute GroupNorm in registers/smem,
then feed directly to the GEMM mainloop without going through global memory.

```
// Phase 1: GroupNorm (in-kernel)
// Load X from global, compute mean/var per group
// Normalize: X_norm = (X - mean) / sqrt(var + eps) * gamma + beta
// Store X_norm to shared memory (already in smem for GEMM A operand)

// Phase 2: GEMM mainloop
// A operand comes from smem (GroupNorm output)
// B operand loaded from global via cp.async
// Standard mma.sync accumulation
```

**Key constraint:** The A tile in shared memory must be populated by GroupNorm
before the GEMM mainloop reads it. For the first K-tile, GroupNorm writes to
smem_A[buf0] then the GEMM reads it. Subsequent K-tiles load A from global
memory (the rest of the normalized input).

Wait -- for a Linear layer (Y = X @ W), the full normalized X is the A matrix.
If X fits in shared memory, great. If not (K > BK), only the first BK columns
come from GroupNorm in smem; the rest need to come from somewhere.

**Two sub-strategies:**

1. **GroupNorm writes to global, GEMM reads back:** GroupNorm output goes to a
   global buffer, then the GEMM loads tiles from it via cp.async. This saves
   one kernel launch but not the memory round-trip. Still useful if the buffer
   stays in L2 cache.

2. **Full fusion (block-resident):** Requires `thread_block_tile_N == problem_N`
   so each block owns the full output row. GroupNorm computes the full normalized
   row into smem (multiple passes if K > smem capacity), and the GEMM consumes
   each chunk. This is the CUTLASS two-op-fusion pattern.

### Strategy B: Custom Epilogue on GEMM Output

If the fused operation is GEMM followed by something (bias, activation), apply
it to the FP32 accumulators before writing to global memory:

```
// After GEMM mainloop, accumulators hold C = A @ B in FP32
for each accumulator fragment:
    acc = acc * alpha + bias[col_idx]    // bias add
    acc = activation(acc)                 // e.g., GELU, ReLU
    output[row][col] = __float2bfloat16(acc)  // convert and store
```

This eliminates a separate elementwise kernel. CUTLASS LinearCombinationGeneric
does exactly this -- apply alpha*acc + beta*C + bias + activation in the epilogue.

### Register Budget for Epilogue Fusion

GroupNorm stats (mean, variance, gamma, beta per group) can be preloaded:
```
mean[G], var[G], gamma[C], beta[C]
```

For the tile's column range, gamma/beta are loaded from global memory. Mean/var
are either passed as kernel args (if GroupNorm is done separately) or computed
in a prologue phase.

---

## 6. Memory Access Pattern Optimization

### Vectorized Global Loads

Always use 16-byte (128-bit) loads:
```
cp.async.cg.shared.global [smem_ptr], [gmem_ptr], 16;
```

This issues LDG.128 instructions, achieving 4x the throughput of scalar loads.
Requires 16-byte aligned source addresses. For BF16 matrices with leading
dimension divisible by 8 (8 x 2 bytes = 16 bytes), alignment is automatic.

### L2 Cache Tiling (Threadblock Swizzle)

Instead of mapping blocks in row-major order, use a swizzle that groups nearby
tiles together for L2 locality:

```
// Standard mapping
block_m = blockIdx.y
block_n = blockIdx.x

// Swizzled mapping (swizzle_size = 2-4)
// Tiles advance along N for swizzle_size tiles, then step in M
linear_idx = blockIdx.x + blockIdx.y * gridDim.x
swizzle_group = linear_idx / (swizzle_size * tiles_N)
within_group = linear_idx % (swizzle_size * tiles_N)
block_m = swizzle_group * swizzle_size + within_group / tiles_N
block_n = within_group % tiles_N
```

This improves L2 hit rate for the B matrix, since adjacent thread blocks in the
swizzled order share B tiles. Impact is GPU-dependent -- siboehm measured L2
already at 80% hit rate without swizzle on some configs.

### Coalesced Global Stores

After the GEMM epilogue, the output C must be written with coalesced access.
With 128 threads and BF16 output:
- Each thread writes 64*64 / 128 = 32 elements
- Rearrange through smem if accumulator layout doesn't match global layout
- Or write directly if the accumulator's thread-to-element mapping is contiguous

CUTLASS exchanges accumulators through shared memory before writing to global
memory, ensuring coalesced stores.

---

## 7. Tile Size Selection Guide

### For Large Square GEMM (M, N, K >= 1024)

| Config | Smem/block | Blocks/SM | Warps/SM | Expected |
|--------|-----------|-----------|----------|----------|
| 64x64, BK=32, 4 warps | 16 KB | 6 | 24 | **0.95-0.98x cuBLAS** |
| 128x64, BK=32, 4 warps | 24 KB | 4 | 16 | ~0.90x cuBLAS |
| 128x128, BK=32, 8 warps | 32 KB | 3 | 24 | ~0.88x cuBLAS |

**Winner: 64x64.** Factory-validated.

### For Skinny-M GEMM (M < 256, N and K large)

Use split-K with 64x64 tiles. The split factor should target ~170 total blocks:
```
split_k = ceil(170 / (tiles_M * tiles_N))
```

### For the Fused Kernel Specifically

The fused GroupNorm+Linear has:
- M = batch_size * spatial_dim (varies, could be small)
- K = input_channels (the normalized dimension)
- N = output_channels (C >= 640 is where the problem starts)

For C=640: tiles_N = ceil(640/64) = 10
If M=256: tiles_M = 4, total = 40 blocks. Split-K by 4-5 would help.
If M=1024: tiles_M = 16, total = 160. Nearly full occupancy, no split-K needed.

---

## 8. Warp Tiling Within the 64x64 Block

With 4 warps computing a 64x64 output tile using m16n8k16 MMA:

### Layout: 2x2 Warp Grid (32x32 per warp)

```
Warp 0: rows 0-31,  cols 0-31
Warp 1: rows 0-31,  cols 32-63
Warp 2: rows 32-63, cols 0-31
Warp 3: rows 32-63, cols 32-63
```

Each warp computes 32x32 = 2x4 MMA tiles (m16n8k16 produces 16x8 output):
- 2 tiles in M (2 x 16 = 32)
- 4 tiles in N (4 x 8 = 32)
- = 8 MMA instructions per K-step
- With BK=32: 2 K-steps per tile = 16 MMA instructions total

### Layout: 1x4 Warp Grid (64x16 per warp)

```
Warp 0: rows 0-63, cols 0-15
Warp 1: rows 0-63, cols 16-31
Warp 2: rows 0-63, cols 32-47
Warp 3: rows 0-63, cols 48-63
```

Each warp: 4x2 MMA tiles = 8 MMAs per K-step.
**Pro:** Each warp loads the same A rows, maximizing A reuse from smem.
**Con:** 4 warps need 4 different B columns -- more B smem traffic.

### Layout: 4x1 Warp Grid (16x64 per warp)

Each warp: 1x8 MMA tiles = 8 MMAs per K-step.
**Pro:** Each warp loads the same B columns, maximizing B reuse.
**Con:** 4 warps need 4 different A rows.

**Recommendation:** Try 2x2 first (balanced A/B reuse). Profile
long_scoreboard vs math_throttle to determine if A or B loading is the
bottleneck, then skew the warp grid accordingly.

---

## 9. Dead Ends (Do NOT Try These)

### 3-Stage or Higher Pipeline
Kills L1 cache on sm_120a. Confirmed regression in GEMM, Fused-MLP, and
attention projects. Unified 128KB L1/smem means more smem = less L1.

### 128x128 Tiles
Only fits 2-3 blocks/SM, giving 16-24 warps. The occupancy deficit costs
more than the larger tile saves in global memory traffic.

### PTX Double-Buffering of B Fragments
Added 16 registers (139 vs 123), dropped a block/SM, and `asm volatile`
prevented compiler reordering. Net result: 0.68x cuBLAS (massive regression).

### Full Fusion of Multi-GEMM Operators
If intermediates exceed tile size, you get O(D_out/BLOCK_N) redundant
recomputation. For D_out=3584, BLOCK_N=128: 28x redundant work. Epilogue
fusion is the correct ceiling.

### Large Monolithic Inline PTX (>50 operands)
On CUDA 13 / sm_120, PTX blocks with >50 "+f" output operands produce
silently wrong results due to ptxas register allocator limitations.

---

## 10. Implementation Priority Order

For the fused GroupNorm+Linear kernel, implement in this order:

1. **64x64 tiles + occupancy tuning** (biggest single win: 0.88x -> 0.98x)
   - 4 warps, 80 regs, `__launch_bounds__(128, 6)`
   - Non-volatile MMA, ldmatrix_x4_mma, streaming B

2. **cp.async double-buffer pipeline** (2.54x for standalone GEMM)
   - 2 stages only (3 kills L1)
   - cp.async.cg with 16-byte loads, zero-fill on OOB
   - cp_async_wait<0> before compute

3. **XOR swizzle on shared memory** (eliminates 10K+ bank conflicts)
   - 3-bit XOR for BK=32, 3-bit for BN=64

4. **Split-K dispatch** (for small-M shapes)
   - Dispatch: if total_tiles < 170, enable split-K
   - Start with atomicAdd single-kernel approach

5. **GroupNorm epilogue/prologue fusion** (the fusion payoff)
   - Compute GroupNorm stats in prologue
   - Feed normalized output directly to GEMM A tiles
   - Apply output epilogue (bias/activation) to accumulators

6. **L2 threadblock swizzle** (incremental, 2-5% improvement)
   - Swizzle factor 2-4 for B-matrix L2 reuse

---

## Sources

- [CUDA Matmul Optimization Worklog (siboehm)](https://siboehm.com/articles/22/CUDA-MMM) -- 10-kernel progression reaching 93.7% cuBLAS on A100
- [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) -- hierarchical blocking, split-K, epilogue
- [CUTLASS Pipelining Tutorial (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) -- multi-stage pipeline, producer/consumer sync
- [CUTLASS Persistent Kernels and Stream-K (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) -- wave quantization, stream-K algorithm
- [HGEMM Advanced Optimization (Bruce-Lee-LY)](https://bruce-lee-ly.medium.com/nvidia-tensor-core-cuda-hgemm-advanced-optimization-5a17eb77dd85) -- 95%+ cuBLAS with tensor cores, pipeline stages
- [Shared Memory Swizzling (Lei Mao)](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/) -- XOR swizzle formula and performance
- [MMA Tensor Core GEMM Introduction (am17an)](https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/) -- m16n8k16 register layout, ldmatrix usage
- [SGEMM GPU Optimization (salykova)](https://salykova.github.io/sgemm-gpu) -- 128x128 tiles, PTX-level, beat cuBLAS on RTX 3090
- [Blackwell Tuning Guide (NVIDIA)](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) -- sm_120 specs: 48 warps/SM, 99KB smem, 64K regs
- [cuBLAS 60% Performance Bug on RTX 5090](https://medium.com/data-science-collective/surfacing-a-60-performance-bug-in-cublas-62d72ef5df66) -- sm_120 dispatcher issues, optimal FMA util 64-73%
- [H100 GEMM Optimization Worklog (hamzaelshafie)](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/) -- register tiling, vectorization, bank conflicts
- [CUTLASS Two-Op Fusion (Example 13)](https://github.com/NVIDIA/cutlass/blob/main/examples/13_two_tensor_op_fusion/README.md) -- register-resident accumulator reuse between ops
- [CUTLASS Epilogue Fusion](https://deepwiki.com/NVIDIA/cutlass/5.3-epilogue-fusion-and-activation-functions) -- LinearCombinationGeneric, bias+activation fusion
- [CUTLASS mma_multistage.h](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/mma_multistage.h) -- cp.async pipeline implementation
- [CUTLASS Blackwell GeForce GEMM (Example 79)](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/) -- sm_120 constraints (cluster 1x1x1, no multicast)
- [Stream-K Paper (arXiv 2301.03598)](https://arxiv.org/abs/2301.03598) -- work-centric parallel decomposition for GEMM
- Factory `04_HARD_WON_LESSONS.md` -- validated sm_120a patterns (occupancy-first, 2-stage pipeline, XOR swizzle, non-volatile MMA)
