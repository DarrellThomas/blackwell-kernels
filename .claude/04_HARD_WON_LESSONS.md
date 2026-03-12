# Hard-Won Lessons — sm_120 Kernel Optimization

Things we learned empirically that the profiler won't tell you.
Do not unlearn these. Do not "fix" them. They are correct.

---

## ISA & Register Layout

**sm_120 is NOT datacenter Blackwell.** It uses `mma.sync` (extended Ampere), not `tcgen05`. There is no TMEM hardware. Data path is registers → tensor cores. Do not attempt datacenter Blackwell approaches (FA3, FA4, CUTLASS tcgen05 examples). They will not compile.

**The a1/a2 register swap is required, not a bug.** `ldmatrix_x4` outputs registers in a different order than `mma.sync` expects. You must swap r1 and r2: pass `(r0, r2, r1, r3)` to MMA. This was empirically verified through `test_mma4.cu`. Every correct kernel in this project does this swap. Removing it will produce silently wrong results that pass casual inspection but fail on non-trivial inputs.

**ldmatrix_x2_trans computes A*B, not A*B^T.** This is correct for P*V multiplication. Do not add a transpose for V — it's already handled by the instruction semantics.

---

## Load-Bearing Optimizations

These optimizations are structural — they're woven into the kernel's architecture. Removing any one of them will cause major regression, not just a small slowdown.

**cp.async (v3):** All global→shared loads use 16-byte `cp.async.cg` with zero-fill on OOB. This was a 2.54x speedup (313 us → 123 us). Reverting to scalar register-mediated copies would be catastrophic.

**Double-buffer pipelining (v4):** K/V tiles alternate between two shared memory buffer slots. Next block prefetches while current block computes. Removing this re-couples load and compute phases.

**XOR swizzle (v5):** Shared memory addressing uses XOR-based swizzle to spread accesses across banks. This eliminated tens of thousands of bank conflicts. Do not remove or "simplify" the swizzle indexing.

**Register-only P→A conversion (v7):** The softmax output P is converted from FP32 accumulators to BF16 MMA A-fragments entirely in registers via warp shuffles. This eliminated the shared memory round-trip that was the dominant source of bank conflicts. Do not reintroduce a shared memory path for P.

---

## Shared Memory Budget

**99 KB max per block** on sm_120 (not 128 KB — CUDA reserves 1 KB per block). The tuning guide is authoritative. Double-buffering roughly doubles shared memory usage. Always calculate before changing tile sizes:

- Current: ~55 KB with double-buffered K/V (BLOCK_Q=64, BLOCK_KV=32)
- BLOCK_KV=64 with double buffer: ~82 KB — fits but tight
- BLOCK_Q=128 with double buffer: needs careful register/smem tradeoff

**Static shared memory declarations are limited to 48 KB.** Anything above requires dynamic allocation with `cudaFuncSetAttribute`.

---

## Occupancy vs Register Pressure

sm_120 has only 48 warps/SM (vs 64 on datacenter). Each block's register usage directly constrains how many blocks fit per SM. More blocks = better latency hiding, but fewer registers per thread = more spills to local memory.

The sweet spot has been ~80-128 registers/thread. Going above 128 regs typically drops to 1-2 blocks/SM, which hurts more than the extra registers help. Use `--ptxas-options=-v` during builds to check register usage when making architectural changes.

---

## Benchmark Noise

CUDA kernel benchmarks have noise. The benchmark harness uses 10 warmup + 100 timed iterations and reports mean. Differences under ~2% are within noise. Do not keep or discard based on sub-2% changes — run again to confirm.

GPU 0 has ComfyUI running. Always use `CUDA_VISIBLE_DEVICES=1`. Forgetting this doesn't just use the wrong GPU — it can cause contention and wildly inconsistent benchmark numbers.

---

## PTX-First / Manual Scheduling

**B fragment double-buffering does NOT help on sm_120.** Attempted loading B[nt+1] while MMA executes with B[nt] using separate register sets and `asm volatile` ordering. Result: 0.68x cuBLAS (regression from 0.89x). Root cause: 139 registers (vs 123), 0 spills — the extra registers reduce occupancy, and the `asm volatile` barriers prevent the compiler from reordering instructions freely. The hardware warp scheduler (with 8 warps / 256 threads) already overlaps ldmatrix and mma.sync across warps.

**The compiler's `#pragma unroll` scheduling is surprisingly good.** On sm_120, nvcc with `-O2` and `#pragma unroll` produces near-optimal interleaving of MMA and load instructions. Simple PTX double-buffering tricks that work on Ampere (salykova) add overhead here because: (1) more registers → lower occupancy, (2) `asm volatile` constraints fight the compiler's scheduler, (3) the warp scheduler already hides latency across 8 warps.

**To actually beat the compiler, you need a radical approach.** A single large `asm` block with full manual register allocation and instruction scheduling — not incremental PTX additions to a compiler-managed kernel. The salykova approach (writing the entire inner loop in PTX with explicit register naming) is the right idea, but requires committing fully to assembly for the hot path.

---

## cp.async Visibility Gotcha

**cp.async writes ARE visible to shared memory reads before `cp.async.wait<0>`.** This is technically undefined behavior per the PTX spec, but it manifests as silent data corruption — not a crash. Experiment 39 proved this: swapping compute-before-load to load-before-compute (reading smem while cp.async writes to the same buffer) produced 70% relative errors on all sizes except the trivial case (M=2048,K=64 where there are no next tiles to load). Same-buffer load-compute overlap is **impossible** with cp.async double-buffering. You CANNOT issue cp.async to a buffer while any warp reads from it.

---

## 2-Tile Unroll Race Condition

**The 2-tile K-loop unroll has an implicit warp-synchrony requirement.** The loop computes buf[0] then buf[1], then prefetches into buf[0] and buf[1]. There is NO barrier between the compute and prefetch phases. This works with 8 warps (256 threads) because all warps complete compute_tile at nearly the same cycle (identical workloads, 2 warps per scheduler). With 16 warps (512 threads, 4 per scheduler), warp scheduling stagger causes fast warps to start cp.async into buf[0] while slow warps still read buf[0] via ldmatrix. This produces ~20% data corruption errors.

Adding `__syncthreads()` between compute and prefetch fixes correctness but doubles barrier count, negating the entire benefit of the 2-tile unroll. The current 8-warp / 2-tile-unroll configuration is the only one that works without the extra barrier. Do not increase warps beyond 8 without either adding the barrier or switching to a different loop structure.

---

## GEMM Tile Configuration

**BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 with 8 warps is the optimal configuration.** Confirmed through 40 experiments. All alternatives regressed:
- BLOCK_K=64: too much data per load, wait stalls dominate (rows 4, 7, 35)
- BLOCK_N=64: fewer smem reads but worse compute-to-load ratio (rows 15, 32)
- BLOCK_M=256 or BLOCK_N=256: excess smem kills L1 cache (rows 15, 20, 37)
- 2D warp tiling (any arrangement): more A ldmatrix loads = more bank conflicts (rows 11, 28, 34)
- 3-stage pipeline: occupancy loss offsets latency benefit (row 6, 30)
- launch_bounds(256,2): barrier stalls double (rows 14, 29)

**The 32KB shared memory sweet spot.** The GEMM kernel uses 32KB double-buffered smem, leaving 96KB for L1 cache. Any configuration exceeding 32KB (3 stages, 4 buffers, larger tiles) reduces L1 cache and degrades performance — the L1 penalty always exceeds the smem benefit.

---

## Volatile Requirements

**ldmatrix MUST remain `asm volatile`.** Non-volatile ldmatrix allows the compiler to reorder shared memory loads across `__syncthreads` and `cp_async_wait` barriers, causing massive data corruption (>100% relative error on most test cases). This is different from MMA where removing volatile is safe — ldmatrix reads from shared memory that changes after barriers, so the compiler must not hoist or delay these loads.

---

## Reference Implementations

**gau-nernst** wrote custom flash attention for RTX 5090 and achieved 94.4% of 209.5 TFLOPS peak using BLOCK_Q=128. This is our feasibility proof — we know sm_120 can get there.

**cuDNN SDPA** achieves 97.2% of peak. This is the bar. Beating it on some configs is possible (we already do at 1.33x for D=64 N=2048 causal), but matching it across all configs is the stretch goal.
