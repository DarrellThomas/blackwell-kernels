# mma.sync Pipeline Depth and Latency — Architecture Reference

**Purpose:** Collected latency and throughput measurements for mma.sync m16n8k16 across architectures, with implications for instruction scheduling on sm_120.
**Last updated:** 2026-03-13

---

## 1. Measured Latency by Architecture

### Ada Lovelace (sm_89, RTX 4090)

**Source:** https://www.spatters.ca/mma-matmul

- **Instruction:** `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- **Latency:** 32 cycles (derived from peak throughput measurement)
- **Derivation:** Peak throughput = 165.2 TFLOP/s at 2520 MHz boost. With 512 tensor cores and 4096 FLOP/MMA: `165.2e12 / (512 × 4096) / 2.52e9 = 31.2 ns ≈ 32 cycles`
- **FLOP per instruction:** 2 × 16 × 8 × 16 = 4096

### Ampere (sm_80, A100 / sm_86, RTX 3090)

**Source:** https://arxiv.org/abs/2206.02874 ("Dissecting Tensor Cores via Microbenchmarks")

- **Instruction:** `mma.sync.aligned.m16n8k16` (various precisions)
- **Latency:** ~32 cycles on A100 (similar pipeline to Ada)
- **INT4 on RTX 3090:** ~17 cycles (smaller operand size, different pipeline)
- **Throughput saturation:** Requires ILP=6 at 25+ active warps for peak on GA102

### Blackwell Consumer (sm_120, RTX 5080 GB203)

**Source:** https://arxiv.org/html/2507.10789v2 ("Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks")

- **Instruction:** `mma.sync` (sm_120 still uses mma.sync, not tcgen05)
- **Completion latency:** ~1.21 cycles (this is likely initiation interval, not full pipeline latency)
- **Peak throughput:** >11 TFLOP/s with ILP=6 and 32 active warps
- **Optimal ILP:** 6 independent MMA instructions in flight per warp
- **Optimal warps:** 25+ active warps for full throughput

### Blackwell Datacenter (sm_100, B200)

**Source:** https://arxiv.org/html/2512.02189v1

- **Instruction:** `tcgen05.mma` (NOT mma.sync — different ISA)
- **Latency:** 11.0-11.4 cycles (constant across tile sizes m64-m256)
- **vs Hopper wgmma:** 2.9-11.6x lower latency
- **NOT applicable to sm_120** — tcgen05 requires TMEM hardware absent on consumer Blackwell

---

## 2. Pipeline Depth Analysis for sm_120

### How Many MMAs Can Be In-Flight?

The tensor core pipeline accepts a new MMA instruction every cycle from the same warp scheduler (initiation interval = 1 cycle). The full pipeline latency is ~32 cycles (based on Ada measurement; sm_120 likely similar since both use mma.sync).

This means up to **32 MMA instructions can be in the pipeline simultaneously** across all warps managed by a single warp scheduler.

### Warps per Sub-Partition

sm_120 has **4 warp schedulers per SM** (same as Ada). Each scheduler manages a pool of warps and issues to its own tensor core.

With 4 warps per block (128 threads) and 3 blocks/SM:
- 12 warps total per SM
- 3 warps per scheduler
- Each scheduler can context-switch between 3 warps

With 8 warps per block (256 threads) and 2 blocks/SM:
- 16 warps total per SM
- 4 warps per scheduler

### Minimum Warps to Hide Latency

With 32-cycle latency and 1-cycle initiation:
- **Minimum:** 32 warps per scheduler × 1 MMA each = fully pipelined
- **Practical:** 4-8 warps per scheduler, each with 4-8 independent MMAs = sufficient

For our kernel at 4 warps, 3 blocks/SM (12 warps, 3 per scheduler):
- Each warp needs ~11 independent MMA instructions to keep the pipeline full
- In QK^T phase: `(BLOCK_Q/16) × (BLOCK_KV/8) × (D/16)` MMA instructions
  - BLOCK_Q=64, BLOCK_KV=64, D=64: `4 × 8 × 4 = 128` MMAs — plenty for the pipeline
  - But they're not all independent! Back-to-back MMAs on the same accumulator have a RAW dependency

### Data Dependencies Limit Effective ILP

For `mma_m16n8k16(A, B, C, D)` where D feeds back as C:
- Sequential MMAs accumulating into the same D have a 32-cycle dependency chain
- The scheduler must interleave with other warps or other independent MMAs

For our QK^T with D=64 (4 K-steps), each M,N tile does:
```
D[m][n] = A[m][k=0] * B[n][k=0]
D[m][n] += A[m][k=1] * B[n][k=1]   // depends on previous D
D[m][n] += A[m][k=2] * B[n][k=2]   // depends on previous D
D[m][n] += A[m][k=3] * B[n][k=3]   // depends on previous D
```

The 4 accumulation steps are **sequential** (RAW dependency on D). But different (m,n) tiles are **independent**. With 4×8 = 32 independent (m,n) tiles, we have 32 independent chains of length 4 = enough to fill the pipeline.

**Optimal scheduling:** Interleave across (m,n) tiles, not across K-steps within a single tile.

---

## 3. Implications for Our BF16 GEMM

### Current State

- 4 warps (128 threads), 3 blocks/SM = 12 warps, 3 per scheduler
- math_pipe_throttle is the dominant stall (~48%)
- The MMA instructions arrive in bursts during QK^T and PV phases

### What math_pipe_throttle Tells Us

The tensor core input FIFO is full because MMA instructions cluster together. Between MMA bursts (during softmax for attention, during loads for GEMM), the tensor core sits idle.

### How to Reduce Throttle

1. **More independent MMAs per loop body** (tiling): Gives the scheduler more non-dependent instructions to interleave. The spatters.ca jump from 8 to 32 MMAs per iteration (3.0→3.1) gave the biggest late-stage improvement.

2. **ILP=6 target:** The GB203 microbenchmark shows peak throughput at ILP=6. Ensure at least 6 independent MMA instructions are available for scheduling at any point in the inner loop.

3. **Interleave loads between MMAs:** Instead of loading all fragments then executing all MMAs, alternate: load A[m=0], MMA[m=0,n=0], load A[m=1], MMA[m=1,n=0], etc. This spreads MMA instructions over time.

4. **Don't use `asm volatile` on MMA.** This is critical (and confirmed by spatters.ca). Let the compiler reorder MMA instructions freely. Only use volatile on ldmatrix and cp.async.

---

## 4. GEMM vs Attention Pipeline Differences

### Pure GEMM

The inner K-loop is a single unbroken MMA phase:
```
for each K tile:
    load A, B from shared → registers
    issue cp.async for next K tile
    execute MMAs (all independent across M,N tiles)
```

The compiler can freely interleave 32+ independent MMAs per iteration. math_pipe_throttle is manageable because the MMA phase is long relative to the load phase.

### Attention

The inner KV-loop has TWO MMA phases separated by non-MMA work:
```
for each KV block:
    QK^T MMAs → softmax (scalar) → PV MMAs
```

The MMA phases are shorter (16 MMAs each for typical tile sizes), and the softmax gap starves the tensor core. The only way to overlap MMA with softmax is through multiple warps on the same scheduler — but with only 3 warps per scheduler, there may not be enough warps with MMAs ready to issue during another warp's softmax.

**This is why GEMM at 0.80x cuBLAS and attention at 1.78x SDPA have different bottlenecks.** GEMM's problem is pure scheduling within a single MMA phase. Attention's problem is the gap between two MMA phases.

---

## References

- spatters.ca MMA matmul: https://www.spatters.ca/mma-matmul
- Dissecting Tensor Cores (arXiv:2206.02874): https://arxiv.org/abs/2206.02874
- Microbenchmarking Blackwell (arXiv:2512.02189): https://arxiv.org/html/2512.02189v1
- Dissecting Blackwell with Microbenchmarks (arXiv:2507.10789): https://arxiv.org/html/2507.10789v2
- NVIDIA mma.sync forum thread: https://forums.developer.nvidia.com/t/throughput-and-latency-of-mma-sync-instruction/303046
- Lei Mao MMA benchmarks: https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances/
