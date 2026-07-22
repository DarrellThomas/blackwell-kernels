<!-- SUMMARY
SIGNAL: Br=128 tiling for flash attention with mma.sync m16n8k16: register budget, warp partitioning, shared memory layout, low-occupancy wins [attention, validated]
WHAT: Concrete techniques for implementing Br=128 (BLOCK_M=128) flash attention tiling with mma.sync m16n8k16 on consumer Blackwell (sm_120a), including register math, CUTLASS TiledMMA warp layout, low-occupancy ILP argument, CUDA 13.0 smem spilling, and FA4 ping-pong insights
FOR: attention kernel, register pressure bottleneck, sm_120a RTX 5090
FINDING: Br=128 with 4 warps (WARP_Q=32) is the proven configuration. Each warp handles 2 vertical MMA tiles. Register budget is ~400-500 regs/thread with d=128. Shared memory fits in 32KB with Q/KV overlap. Dao's FA2 uses Br=128,Bc=32 on sm_8x for hdim=128 non-causal (2 CTAs/SM) and Br=64,Bc=64 for causal. The gau-nernst 5090 kernel started with Br=128,Bc=64 (v1, 68% SOL) and evolved to Br=64,Bc=64 (v5, 94% SOL). NEW: CUTLASS TiledMMA confirms warps tile M-only with 16 rows each; PTX spec gives exact 14-register instruction cost; CUDA 13.0 smem spilling pragma can redirect register spills on-chip; Volkov's ILP argument explains why 15-25% occupancy wins for compute-bound kernels; FA4 uses 128-row ping-pong between two Q tiles; NVIDIA CUDA Tile recovered 256x128 with fast-math flags.
TECHNIQUE: Use 4 warps with WARP_Q=32 (each warp owns 32 Q rows = 2 MMA-M tiles). Load Q once into registers, overlap Q_smem with K+V_smem. Double-buffer K, single-buffer V. Pack S->P in registers using bf16x2 reinterpret_cast. Use XOR swizzle on shared memory. Accept moderate register spilling. NEW: try .pragma "enable_smem_spilling" in CUDA 13.0+ to redirect spills to shared memory instead of L1/L2. For large-tile compute-bound kernels, target ILP-based latency hiding (12-16 independent instructions between memory ops) rather than high occupancy.
STATUS: validated — gau-nernst 5090 blog, Dao FA2 source code, CUTLASS Ampere FA2 example, Colfax Hopper case study, NVIDIA CUDA Tile blog, FA4 paper, spatters.ca Ada GEMM analysis, Volkov GTC 2010
-->

# Flash Attention — Br=128 Tiling with mma.sync m16n8k16 on sm_120a

## 1. Register Budget Analysis (THE critical constraint)

### Per-thread register arrays for Br=128, Bc=64, d=128, 4 warps (WARP_Q=32)

From gau-nernst's validated 5090 implementation:

```
WARP_Q = BLOCK_Q / NUM_WARPS = 128 / 4 = 32
MMA_M=16, MMA_N=8, MMA_K=16

Q_rmem[WARP_Q/MMA_M][DIM/MMA_K][4]  = [2][8][4]  = 64 uint32 regs
K_rmem[BLOCK_KV/MMA_N][DIM/MMA_K][2] = [8][8][2]  = 128 uint32 regs
P_rmem[WARP_Q/MMA_M][BLOCK_KV/MMA_K][4] = [2][4][4] = 32 uint32 regs
V_rmem[BLOCK_KV/MMA_K][DIM/MMA_N][2] = [4][16][2] = 128 uint32 regs
O_rmem[WARP_Q/MMA_M][DIM/MMA_N][4]   = [2][16][4] = 128 float regs
S_rmem[WARP_Q/MMA_M][BLOCK_KV/MMA_N][4] = [2][8][4] = 64 float regs
rowmax[WARP_Q/MMA_M][2]              = [2][2]     = 4 float regs
rowsumexp[WARP_Q/MMA_M][2]           = [2][2]     = 4 float regs
```

**Total declared: ~552 registers per thread** (mix of uint32 and float).
Not all live simultaneously — compiler can reuse K_rmem/V_rmem/S_rmem across loop iterations.
Peak live pressure is around 350-450 registers, causing spills to local memory.

**sm_120a limit: 255 registers per thread.** Anything above spills. With 128 threads (4 warps), that is 32,640 registers from the 64K register file — leaves room for exactly 1 block/SM.

### What changes at Br=128 vs Br=64

| Array | Br=64 (WARP_Q=16, 1 MMA tile) | Br=128 (WARP_Q=32, 2 MMA tiles) |
|-------|-------------------------------|----------------------------------|
| Q_rmem | [1][8][4] = 32 | [2][8][4] = 64 | +32 regs |
| O_rmem | [1][16][4] = 64 | [2][16][4] = 128 | +64 regs |
| S_rmem | [1][8][4] = 32 | [2][8][4] = 64 | +32 regs |
| P_rmem | [1][4][4] = 16 | [2][4][4] = 32 | +16 regs |
| rowmax | [1][2] = 2 | [2][2] = 4 | +2 regs |
| rowsumexp | [1][2] = 2 | [2][2] = 4 | +2 regs |

**Br=128 adds ~148 registers per thread vs Br=64.** This is the core tradeoff:
doubling Br doubles Q/O/S/P accumulators. K and V registers stay constant (they
depend on Bc and d, not Br).

### Key insight from gau-nernst

> "Reducing BLOCK_Q (so that we use fewer registers to hold accumulators) would
> resolve this issue, but my manual tuning showed that some spilling was actually
> faster."

The v1 kernel (Br=128, Bc=64) achieved 68% SOL = 142.87 TFLOPS. After adding
swizzling, pipelining, and ldmatrix.x4, the v5 kernel (Br=64, Bc=64) achieved
94% SOL = 197.74 TFLOPS — essentially matching cuDNN's 203.61 TFLOPS.

**The lesson: Br=128 gives you fewer load-compute transitions but register
spilling degrades MMA throughput. The sweet spot depends on how well you pipeline.**

## 2. Warp Partitioning Strategy

### FA2 approach: split Q across warps, replicate K/V

```
4 warps, each warp handles WARP_Q = BLOCK_Q / 4 rows of Q
All warps read the SAME K and V tiles from shared memory
```

**Why this works:** After each warp computes its slice of S = Q @ K^T, it independently
computes softmax and accumulates P @ V. **No inter-warp communication needed** for the
main compute loop. Only a warp-internal butterfly reduction for row-wise softmax:

```c
// Butterfly reduction within 4 threads (not 4 warps — within a warp)
this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
```

The m16n8k16 MMA distributes 16 output rows across the 32 threads in a warp such
that threads 0-3 hold elements from rows 0-1, threads 4-7 from rows 2-3, etc.
The row-wise max/sum reduction needs butterfly across groups of 4 threads (XOR with
masks 1 and 2) — two __shfl_xor_sync calls per reduction.

### 4 warps vs 8 warps for Br=128

From Dao's FA2 source code, the definitive comment for hdim=128:

> "Using 8 warps (128x128 and 256x64) is 28% slower for seqlen=2k"

And for hdim=64:

> "Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower"

**4 warps is the sweet spot for mma.sync on sm_80/sm_89/sm_120.** With 8 warps,
each warp owns fewer Q rows (WARP_Q=16, only 1 MMA tile) which halves the
accumulator pressure, but the additional threads compete for register file and
reduce occupancy. The 4-warp config uses 128 threads = exactly 1 warp scheduler
per warp, maximizing ILP.

### Dao's actual tile configurations for hdim=128 on sm_8x

```
Non-causal: Flash_fwd_kernel_traits<128, 128, 32, 4>  (Br=128, Bc=32, 4 warps)
Causal:     Flash_fwd_kernel_traits<128, 64, 64, 4>   (Br=64, Bc=64, 4 warps)
```

The non-causal path uses Br=128 with small Bc=32. This gives 2 CTAs per SM
because the shared memory is only 48KB (fits in default allocation). The causal
path uses the square Br=64,Bc=64 because the triangular mask wastes less compute
with square tiles.

## 3. Shared Memory Layout and Budget

### Br=128 memory calculation

```
Q tile:  128 * 128 * 2 bytes (BF16)  = 32 KB
K tile:  Bc * 128 * 2 bytes
V tile:  Bc * 128 * 2 bytes

Bc=64:  K=16KB, V=16KB → K+V = 32KB
Bc=128: K=32KB, V=32KB → K+V = 64KB
Bc=32:  K=8KB, V=8KB   → K+V = 16KB (with double buffer K: 24KB)
```

### The overlap trick (critical for fitting in 99KB)

**Q is loaded once at kernel start, transferred to registers, then Q_smem is reused
for K+V storage.** From gau-nernst:

```c
const uint32_t Q_smem = __cvta_generic_to_shared(smem);
const uint32_t K_smem = Q_smem;  // OVERLAPS with Q — Q already in registers
const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);
```

**Total smem = max(Q_tile, K_double_buffer + V_single_buffer)**

| Config | Q tile | K (2x) + V | Total smem |
|--------|--------|------------|------------|
| Br=128, Bc=32 | 32KB | 8+8+8=24KB | 32KB |
| Br=128, Bc=64 | 32KB | 16+16+16=48KB | 48KB |
| Br=128, Bc=128 | 32KB | 32+32+32=96KB | 96KB |

**Br=128,Bc=64 with double-buffered K needs 48KB** — well within 99KB usable.
**Br=128,Bc=128 needs 96KB** — barely fits, leaves 3KB for nothing else.

Dao's choice of Br=128,Bc=32 (32KB) allows 2 CTAs per SM on sm_8x.

### XOR swizzle for bank conflicts

```c
template <int STRIDE>
uint32_t swizzle(uint32_t index) {
    uint32_t row_idx = (index / STRIDE) % 8;
    uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return index ^ (bits_to_xor << 4);
}
```

For DIM=128 (stride=256 bytes), this XORs address bits based on the row within
each 8-row group, preventing bank conflicts when 32 threads access the same
column across different rows.

## 4. MMA Tile Mapping for 128 Q rows

### How 128 rows decompose into MMA operations

```
128 Q rows / 4 warps = 32 rows per warp (WARP_Q)
32 rows / 16 (MMA_M) = 2 MMA tiles vertically per warp

Total MMA tiles for Q@K^T:
  mma_id_q:  0..1  (2 vertical Q tiles per warp)
  mma_id_kv: 0..7  (8 horizontal K tiles for Bc=64, since 64/8=8)
  mma_id_d:  0..7  (8 accumulation steps for d=128, since 128/16=8)

MMA calls per K block: 2 * 8 * 8 = 128 mma.sync instructions per warp
```

### The P -> V multiplication: register packing trick

After softmax, S_rmem (float32) must become P_rmem (BF16) for the P@V MMA.
The clever trick: **reinterpret_cast the P registers as bf16x2 to match
mma.m16n8k16's A operand format**.

```c
// Pack S (post-softmax, float32) into P (bf16x2) for mma input
nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
this_P_rmem[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});
this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
```

**No shared memory round-trip needed.** S_rmem's 4 float values per MMA tile
become P_rmem's 4 uint32 (bf16x2) values, and the layout naturally matches
what mma.m16n8k16 expects for operand A. This is because the m16n8 output
layout (2 consecutive elements per row, 2 rows per thread) maps directly to
the m16k16 input layout when you pack pairs of consecutive n-dimension values
into the k-dimension.

### V loading: transposed ldmatrix

V is loaded with `ldmatrix.x4.trans` to get the transposed layout needed for
the B operand of the P@V MMA:

```c
for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
  for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
    ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
  }
```

## 5. Pipelining Strategy (what makes Br=64 beat Br=128)

### The key insight: smaller tiles pipeline better

The v5 kernel (Br=64, Bc=64) achieves 94% SOL vs v1's 68% SOL (Br=128, Bc=64)
because of three pipelining techniques:

**1. K double-buffer with cp.async groups:**
```c
// Prefetch next K while computing current QK^T
load_K(kv_id + 1);  // issues cp.async for next K block
// ... compute S = Q @ K.T with current K ...
asm volatile("cp.async.wait_group 1;");  // wait for K to land
```

**2. V single-buffer with synchronization:**
```c
__syncthreads();  // ensure V_smem from prior iteration consumed
load_V(kv_id);    // start async copy of V
// ... compute softmax on S ...
asm volatile("cp.async.wait_group 1;");  // wait for V to land
// ... compute O += P @ V ...
```

**3. Overlap loading and compute within the inner loop:**
The sequence per KV block is:
1. Issue V async load (for current block)
2. Wait for K (prefetched previous iteration)
3. Load K into registers via ldmatrix
4. Compute S = Q @ K^T (MMA)
5. Issue K async load (for NEXT block — prefetch)
6. Compute softmax on S
7. Wait for V
8. Load V into registers via ldmatrix_trans
9. Compute O += P @ V (MMA)
10. Loop

**With Br=128 + register spilling, the MMA throughput drops because spilled
registers create L1TEX local load/store stalls that compete with the async
memory pipeline.**

## 6. Concrete Recommendations for Your Kernel

### Option A: Br=128, Bc=32, 4 warps (Dao's non-causal config)

- **Shared memory:** 32KB (2 CTAs/SM possible)
- **Register budget:** WARP_Q=32, 2 MMA-M tiles/warp
- **Q_rmem:** 64 regs, **O_rmem:** 128 float regs
- **S_rmem:** smaller (Bc=32 → [2][4][4] = 32 floats)
- **K_rmem:** [4][8][2] = 64 regs (half of Bc=64 version)
- **Total live regs:** ~300-350, manageable with modest spilling
- **Pro:** 2 CTAs/SM boosts occupancy; fewer load-compute transitions than Br=64
- **Con:** Bc=32 means more KV iterations; narrow K tile underutilizes MMA

### Option B: Br=128, Bc=64, 4 warps (gau-nernst v1 config)

- **Shared memory:** 48KB with K double-buffer
- **Register budget:** WARP_Q=32, full register arrays as shown above
- **Total live regs:** ~400-450, significant spilling expected
- **Pro:** Good arithmetic intensity per KV block; matches cuDNN's Bc
- **Con:** Spilling hurts MMA throughput; 1 CTA/SM

### Option C: Br=64, Bc=64, 4 warps (gau-nernst v5 config — current best)

- **Shared memory:** 32KB
- **Register budget:** WARP_Q=16, 1 MMA-M tile/warp
- **Total live regs:** ~250-300, minimal spilling
- **Pro:** 94% SOL proven; good pipelining; 2 CTAs/SM possible
- **Con:** More load-compute transitions; your cuDNN reference wins because it avoids these

### Option D: Br=128, Bc=128, 4 warps (cuDNN-like)

- **Shared memory:** 96KB (barely fits 99KB limit; 1 CTA/SM)
- **Register budget:** Catastrophic — S_rmem alone is [2][16][4] = 128 floats
- **K_rmem would be [16][8][2] = 256 regs** just for K fragments
- **This will NOT work with mma.sync on sm_120a.** The Colfax Hopper paper confirms
  128x128 causes register spills that kill GEMM-II performance even on H100 with
  wgmma. On sm_120a with mma.sync (half the throughput per instruction), it's worse.

## 7. Dead Ends (do NOT try these)

- **Br=128, Bc=128 with mma.sync:** Confirmed by Colfax paper: "128x128 suffers from
  performance degradation due to register pressure" even on Hopper with wgmma. With
  mma.sync, the register count is even worse because mma.sync uses 4+2 input regs
  per instruction vs wgmma sourcing from shared memory.

- **8 warps for Br=128:** Dao's comment: "Using 8 warps (128x128 and 256x64) is 28%
  slower for seqlen=2k." More warps means fewer registers per warp, does not help.

- **Trying to match cuDNN's exact config:** cuDNN uses `flash_fwd_kernel<64,128,128,4>`
  at 15% occupancy. This is a fundamentally different kernel architecture (likely using
  proprietary memory pipeline features or warp-specialized execution). Matching the
  tile size without matching the pipeline design will not close the gap.

## 8. Recommended Path Forward

**Start with Option A (Br=128, Bc=32)** as a stepping stone:
1. Verify correctness with Br=128 tiling
2. Profile register usage with Nsight Compute
3. If register pressure is acceptable, try Bc=64

**But the real win is likely not Br=128 — it's better pipelining at Br=64:**
- The gau-nernst blog proves that Br=64 with aggressive pipelining (v5) reaches
  94% of cuDNN — the remaining 6% gap is likely cuDNN's proprietary pipeline
  optimizations, not tile size
- cuDNN wins at 15% occupancy because it pipelines perfectly, not because Br=128
  is magic

**The specific gap to close is load-compute transition overhead, not tile size.**
Focus on: (1) tighter cp.async/MMA overlap, (2) reducing __syncthreads barriers,
(3) hiding V load latency behind softmax computation.

## 9. NEW: PTX-Level Register Cost per MMA Instruction

### Exact register allocation from NVIDIA PTX spec and forums

Each `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instruction uses:

```
D output (accumulator result): 4 x float registers  (%Rd0..%Rd3)
A input  (Q or P fragment):    4 x uint32 registers  (%Ra0..%Ra3, packed bf16x2)
B input  (K or V fragment):    2 x uint32 registers  (%Rb0..%Rb1, packed bf16x2)
C input  (accumulator seed):   4 x float registers  (%Rc0..%Rc3)

Total per MMA call: 14 register operands
```

In practice, D aliases C (accumulate-in-place), so the live register cost per
MMA tile is **4 float (accumulator) + 4 uint32 (A) + 2 uint32 (B) = 10 registers**.

### Thread-to-element mapping within the 16x8 output

Each of the 32 threads in a warp holds 4 elements of the 16x8 = 128 output:

```
groupID = lane_id >> 2          (0..7, which row pair)
tid_in_group = lane_id % 4      (0..3, which column pair)

Element 0: row = groupID,     col = tid_in_group * 2
Element 1: row = groupID,     col = tid_in_group * 2 + 1
Element 2: row = groupID + 8, col = tid_in_group * 2
Element 3: row = groupID + 8, col = tid_in_group * 2 + 1
```

**Key insight:** Each thread covers 2 rows (separated by 8) and 2 consecutive
columns. This is why the butterfly softmax reduction only needs XOR masks 1 and 2 --
the 4 threads in a group hold the same row pair, different column pairs.

## 10. NEW: CUTLASS TiledMMA Warp Layout (Verified)

### From Dao's kernel_traits.h in flash-attention

```cpp
using TiledMma = TiledMMA<
    MMA_Atom_Arch,                              // m16n8k16 atom
    Layout<Shape<Int<kNWarps>, _1, _1>>,         // warp layout: ALL warps tile M only
    Tile<Int<16 * kNWarps>, _16, _16>            // total tile: (16*kNWarps) x 16 x 16
>;
```

**This confirms:** Warps tile ONLY along the M dimension. With kNWarps=4,
the TiledMMA covers 64 rows of M in a single pass. When kBlockM=128, the
CUTLASS `partition_fragment_C` call creates an accumulator with shape
`(MMA=4, MMA_M=2, MMA_N=N)` where MMA_M=2 means each warp processes
2 vertical MMA-M tiles (2 * 16 = 32 rows = WARP_Q).

The `cute::gemm` call then internally loops over MMA_M and MMA_N modes,
issuing 2 * (kBlockN/8) MMA instructions per K-step per warp.

### Why warps tile M only (not N)

FA2's "split-Q" strategy assigns each warp a unique Q slice. Since K and V
are shared by all warps, there is zero inter-warp communication during the
main compute loop. If warps also tiled N, they would need to synchronize
and reduce partial S results across warps -- the exact problem FA1 had.

## 11. NEW: Why Low Occupancy Wins for Attention (The ILP Argument)

### Volkov's principle (GTC 2010, validated repeatedly since)

"Instruction-level parallelism can substitute for thread-level parallelism
in hiding memory latency." -- Vasily Volkov, UC Berkeley

For compute-bound kernels with 12-16 independent instructions between
memory dependencies, **25% occupancy can outperform 100% occupancy**.
The mechanism:

```
When occupancy → 100%:
  - More warps compete for register file → spills → L1TEX stalls
  - More warps compete for shared memory → bank conflicts
  - Scheduler overhead increases with warp count
  - Net effect: lower throughput per warp, higher total stalls

When occupancy → 25% with high ILP:
  - Each warp has 255 registers (no spills)
  - Independent MMA instructions chain without stalls
  - Fewer warps = less contention for shared memory ports
  - Net effect: near-peak MMA utilization per warp
```

### How this applies to flash attention

Flash attention's main loop has this instruction sequence per KV block:

```
[ldmatrix K] → independent of prior MMA
[MMA: Q@K]  → depends on ldmatrix K
[softmax: exp, max, sum] → depends on MMA result (CUDA cores, not tensor cores)
[ldmatrix V] → independent of softmax
[MMA: P@V]  → depends on ldmatrix V AND softmax
[rescale O]  → depends on P@V MMA
```

With Br=128, each warp issues 128+ MMA instructions per KV block. Between
any two memory-dependent pairs, there are 8-16 independent MMA instructions
(the inner mma_id_kv and mma_id_d loops). This is MORE than enough ILP to
hide latency at low occupancy.

### cuDNN's 15-20% occupancy explained

cuDNN's attention kernel runs at 15-20% occupancy and wins because:

1. **Fewer Q reloads:** With Br=128, Q is loaded once and reused across all
   Bc iterations. Larger Br = fewer global memory reads of Q.
   HBM traffic ~ O(N^2 * d^2 / M) where M is SRAM size (FlashAttention
   Theorem 1). Larger tiles = larger effective M.

2. **More MMA instructions per sync point:** 128 rows means 2x more MMAs
   between each __syncthreads, amortizing the barrier cost.

3. **Better softmax amortization:** Online softmax rescaling (m_new, l_new,
   O *= exp(m_old - m_new)) happens once per WARP_Q rows per KV block.
   With WARP_Q=32 instead of 16, the rescaling cost per output element halves.

4. **Register-rich execution:** At 15% occupancy on sm_120a (48 warps max),
   that is ~7 warps active. With 64K registers / 7 warps / 32 threads =
   ~292 registers per thread. Enough for Br=128 with minimal spilling.

### The register budget at different occupancies (sm_120a)

```
64K registers / SM, 48 max warps, 255 max per thread

1 CTA (4 warps, 128 threads):  64K / 128 = 512 regs/thread (capped at 255)
  → 8.3% occupancy (4/48 warps)
  → 255 regs/thread usable, zero spills for Br=128

2 CTAs (8 warps, 256 threads): 64K / 256 = 256 regs/thread (capped at 255)
  → 16.7% occupancy (8/48 warps)
  → 255 regs/thread usable, zero spills for Br=128

3 CTAs (12 warps, 384 threads): 64K / 384 = 170 regs/thread
  → 25% occupancy
  → 170 regs/thread, HEAVY spilling for Br=128 (~400+ needed)

4 CTAs (16 warps, 512 threads): 64K / 512 = 128 regs/thread
  → 33% occupancy
  → 128 regs/thread, catastrophic spilling
```

**The sweet spot for Br=128 is 1-2 CTAs/SM (8-17% occupancy).** This gives
enough registers to avoid spilling while the high ILP from large tiles
hides latency. This exactly matches cuDNN's observed behavior.

## 12. NEW: CUDA 13.0 Shared Memory Register Spilling

### The pragma

```cuda
__global__ void flash_fwd_kernel(...) {
    asm volatile (".pragma \"enable_smem_spilling\";");
    // ... kernel body ...
}
```

When enabled, the compiler spills excess registers to shared memory instead
of local memory (L1/L2 cache). This reduces spill latency from ~100 cycles
(L2 round trip) to ~20 cycles (shared memory access).

### Requirements and constraints

- CUDA 13.0+ only
- Requires whole-program compilation (`-rdc=false`, the default)
- CANNOT be used with: `-rdc=true`, `-G`, `-g`, dynamic shared memory
  (but dynamic smem is how FA typically allocates -- **may need testing**)
- Increases shared memory usage (spilled values occupy smem)
- Benchmark showed 7.8% improvement for register-heavy kernels

### Applicability to Br=128 flash attention

If using static shared memory allocation, this pragma could redirect the
~100-150 spilled registers from Br=128 to shared memory instead of L1.
Combined with the Q_smem overlap trick (Q in regs after first load, smem
reused for K/V), there may be spare smem capacity for spills:

```
Available smem: 99KB (sm_120a usable)
K double-buffer + V: 48KB (for Bc=64)
Remaining for spills: ~51KB
Spill capacity: ~51KB / 128 threads = ~400 bytes/thread = ~100 float regs
```

This could be enough to hold the ~100-150 spilled registers on-chip.
**Worth benchmarking.**

## 13. NEW: FA4's Ping-Pong for Br=128 (Blackwell Datacenter)

### Architecture (B200, not directly portable to sm_120a)

FA4 processes **two 128-row Q tiles per CTA** in a ping-pong schedule:

```
Q^H (first 128 rows)   Q^L (second 128 rows)
     │                       │
     ├─ softmax(S^H) ────── ├─ MMA(Q^L @ K^T)    ← overlap
     ├─ MMA(P^H @ V) ────── ├─ softmax(S^L)       ← overlap
     └─ store O^H            └─ MMA(P^L @ V)
```

The key idea: while one Q tile is doing softmax (CUDA cores), the other
Q tile is doing MMA (tensor cores). This hides the softmax latency that
otherwise creates a bubble in the tensor core pipeline.

### What's portable to sm_120a

The ping-pong SCHEDULE is not directly portable (FA4 uses TMEM and wgmma
which we lack). But the INSIGHT is:

**If you can overlap softmax computation of one MMA-M tile with the MMA
computation of another MMA-M tile within the same warp, you hide the
softmax bubble.**

With WARP_Q=32 (2 MMA-M tiles per warp), you could:
1. Compute S for mma_id_q=0, start softmax for mma_id_q=0
2. While softmax runs on CUDA cores, compute S for mma_id_q=1
3. Complete softmax for mma_id_q=0, start P@V for mma_id_q=0
4. Start softmax for mma_id_q=1
5. ... etc.

This requires careful register management to keep both tiles' accumulators
live simultaneously, but that is exactly what Br=128 gives you.

## 14. NEW: Large Tile Recovery via Fast Math (NVIDIA CUDA Tile Blog)

### The 256x128 tile recovery experiment

On B200, enlarging tiles from 64x64 to 256x128 initially degraded
performance by 18-43%. But with two compiler flags:

```
flush_to_zero = True     (eliminates denormal microcode overhead)
rounding_mode = APPROX   (skips iterative refinement in exp/div)
```

Performance recovered and exceeded the baseline:
- 1024 tokens: 187 → 322 TFLOPS (+72%)
- 16384 tokens: 463 → 620 TFLOPS (+34%)

### What this means for sm_120a

The lesson: large tiles expose more instruction overhead (denormals in
softmax exp(), rounding in division). On sm_120a, equivalent optimizations:

```c
// Fast exp2f for softmax (avoid denormal handling)
__expf(x)  // instead of expf(x) -- uses SFU, ~2 ULP error

// Or use the PTX directly
asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
```

The `ftz` (flush-to-zero) flag eliminates denormal handling in the
exponentiation, which is the single most common non-MMA instruction in
the attention inner loop.

## 15. NEW: Ada GEMM Analysis (spatters.ca) — Register Tiling Scaling

### How warp tile size scales with register pressure

From spatters.ca's mma-matmul on RTX 4090 (sm_89, same ISA as sm_120a):

| Warp tile | MMA calls/iter | Accum regs/thread | Performance |
|-----------|---------------|-------------------|-------------|
| 16x8 (1x1) | 1 | 4 float | baseline |
| 32x16 (2x2) | 4 | 16 float | 86.9% cuBLAS |
| 64x32 (4x4) | 32 | 64 float | 100% cuBLAS (153.6 TF) |

The key: going from 2x2 to 4x4 warp tiling (8x more MMA calls, 4x more
accumulator registers) yielded the final jump to peak performance. The
extra ILP from 32 independent MMA calls hid all remaining latency.

**For attention at Br=128:** WARP_Q=32 gives 2 MMA-M tiles, which is the
2x2 tier. Going to WARP_Q=64 (Br=256, 4 warps) would reach the 4x4 tier
but is unrealistic for register budget. The 2-tile config at WARP_Q=32
is the practical sweet spot -- enough ILP to fill the pipeline, not so
much that registers blow up.

## Sources

- https://gau-nernst.github.io/fa-5090/ — Flash Attention for RTX 5090, mma.sync m16n8k16, BLOCK_Q=128 and BLOCK_Q=64 variants with performance data
- https://github.com/gau-nernst/learn-cuda/tree/e83c256/07_attention — Full source code (v1-v5)
- https://github.com/Dao-AILab/flash-attention — Tri Dao's FA2, kernel_traits.h and flash_fwd_launch_template.h show exact tile configs per arch
- https://arxiv.org/html/2312.11918v1 — Colfax: FlashAttention-2 on Hopper, confirms 128x128 register pressure problem
- https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py — CUTLASS Ampere FA2 with BLK_M=128 default
- https://github.com/DefTruth/CUDA-Learn-Notes/blob/main/kernels/flash-attn/mma/swizzle/flash_attn_mma_tiling_qkv_swizzle_qk.cu — Br=64 mma.sync FA2 with detailed register layout
- https://lubits.ch/flash/Part-2 — MMA m16n8k16 register fragment layout diagrams
- https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/ — NVIDIA CUDA Tile FA tuning, 256x128 recovery with fast-math, autotuning tile selection
- https://crfm.stanford.edu/2023/07/17/flash2.html — FA2 paper: split-Q warp partitioning rationale
- https://forums.developer.nvidia.com/t/complete-minimal-ptx-example-for-mma-sync-aligned-m16n8k16-row-col-f32-f16-f16-f32/311164 — Exact PTX register layout for m16n8k16
- https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d — Thread-to-element mapping for MMA fragments
- https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/ — CUDA 13.0 smem spilling pragma, 7.8% improvement
- https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf — Volkov: ILP substitutes for TLP, low occupancy wins for compute-bound kernels
- https://www.spatters.ca/mma-matmul — Ada mma.sync GEMM: 4x4 warp tile at 93% peak, register scaling analysis
- https://www.together.ai/blog/flashattention-4 — FA4 ping-pong Q tiles, asymmetric hardware scaling argument
- https://leimao.github.io/blog/CuTe-Tiled-MMA/ — CuTe TiledMMA warp distribution mechanics
