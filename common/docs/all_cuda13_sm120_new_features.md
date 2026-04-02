# CUDA 13.0 / sm_120 New Features for Kernel Optimization

**Sources:**
- https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/
- https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/
- https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- https://docs.nvidia.com/cuda/parallel-thread-execution/ (PTX ISA 9.2)
- https://forums.developer.nvidia.com/t/thread-block-clustering-in-blackwell-gpus/320471
- https://forums.developer.nvidia.com/t/does-blackwell-sm-120-have-native-f32x2-support/344788
- https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
- https://forums.developer.nvidia.com/t/run-ptx-mma-sync-aligned-kind-mxf8f6f4-block-scale-scale-vec-1x-m16n8k32-on-sm-120a/329702
- https://forums.developer.nvidia.com/t/different-ctas-accessing-the-same-shared-memory-address-on-rtx-5090-is-this-expected/352137
- https://blog.alpindale.net/posts/5090_decode_optimization/
- https://github.com/pytorch/pytorch/issues/172807
- https://github.com/nvidia/cutlass/issues/2867
- https://github.com/ggml-org/llama.cpp/issues/19662
- https://reviews.llvm.org/D100124

**Relevant to:** ALL active workers (attention, GEMM, fused-mlp, rmsnorm, dotproduct, linalg, numerical)
**Purpose:** Document confirmed sm_120-specific features and CUDA 13 improvements that workers may not be using, plus features that are NOT available to avoid wasted effort.
**Date:** 2026-03-14

---

## Confirmed New Features for sm_120

### 1. enable_smem_spilling Pragma (CUDA 13.0+) -- HIGH IMPACT

**What:** Registers can now spill to shared memory instead of local memory (L2 cache). Shared memory is ~10x lower latency than L2 for spills.

**Syntax:**
```cuda
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS) kernel(...) {
    asm volatile(".pragma \"enable_smem_spilling\";");
    // ... rest of kernel
}
```

**How it works:** ptxas redirects register spill traffic to unused shared memory. If shared memory is exhausted, remaining spills fall back to local memory as before. The pragma must be placed inside the kernel function body.

**Measured impact:** 7.76% kernel duration improvement, 8% fewer cycles, 9% reduction in SM active cycles in NVIDIA's test case. QUDA lattice QCD kernels saw 5-10% improvement.

**When to use it:**
- Kernels with register spills (check `--ptxas-options=-v` for "bytes spill stores/loads")
- Kernels with unused shared memory headroom
- Must have explicit `__launch_bounds__` (without it, ptxas assumes max threads and may degrade occupancy)

**When NOT to use it:**
- Kernels already maxing out shared memory (e.g., our attention kernel at 96 KB / 128 KB SM)
- Kernels using dynamic shared memory (incompatible)
- Compilation with `-rdc=true`, `-G`, or `-ewp` (incompatible)

**Worker impact assessment:**
- **GEMM worker:** Potentially useful. 80 regs/thread with 0 spills currently. If experimenting with larger tiles (more regs), this provides a safety net -- spills to smem instead of L2. But current 6-block config already uses 96 KB smem, leaving only 32 KB headroom per SM for spills.
- **Attention worker:** Less useful. Already at 145 regs with tight smem budget (96 KB). Little headroom for smem spills.
- **Numerical/linalg workers:** Potentially high impact. Complex kernels with many live variables benefit most. Worth trying on any kernel showing register spills.
- **RMSNorm/dotproduct workers:** Simple kernels, unlikely to benefit.

### 2. redux.sync Warp Reduction Instruction (sm_80+) -- MEDIUM IMPACT

**What:** Hardware warp-level reduction in a single instruction. Replaces multi-step `__shfl_xor_sync` reduction trees.

**Syntax:**
```ptx
redux.sync.add.s32  dst, src, membermask;
redux.sync.min.s32  dst, src, membermask;
redux.sync.max.s32  dst, src, membermask;
redux.sync.min.u32  dst, src, membermask;
redux.sync.max.u32  dst, src, membermask;
redux.sync.and.b32  dst, src, membermask;
redux.sync.or.b32   dst, src, membermask;
redux.sync.xor.b32  dst, src, membermask;
```

**Architecture:** sm_80+ (Ampere and later). Confirmed available on sm_120.

**Supported types:** `.s32`, `.u32`, `.b32` only. **NO .f32 support.** This is a significant limitation -- floating-point reductions (softmax max, sum for normalization) cannot use this instruction directly.

**Inline ASM example:**
```cuda
uint32_t result;
asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
```

**Practical applicability:**
- **NOT directly useful for softmax** (needs f32 max and f32 sum) -- would require converting to fixed-point, which adds overhead
- **Useful for integer reductions:** tile index calculations, mask operations, counting
- **Useful for RMSNorm/LayerNorm** only if the sum-of-squares can be done in fixed point (unlikely to be faster than shfl_xor tree for f32)
- The `__shfl_xor_sync` tree for f32 reductions (5 shuffles for full warp) is already well-optimized by the compiler

**Bottom line:** Available but limited utility for our workers due to no f32 support. The workers' existing `__shfl_xor_sync` reduction patterns are the correct approach for floating-point reductions.

### 3. Thread Block Clusters and Distributed Shared Memory (sm_120) -- LOW-MEDIUM IMPACT

**What:** Multiple thread blocks can form a "cluster" and access each other's shared memory. Max portable cluster size: 8 blocks.

**Confirmed on sm_120:** Yes. Thread block clusters ARE supported on consumer Blackwell. Forum confirmations from developers and NVIDIA's own tuning guide lists cluster support for sm_120.

**Distributed shared memory:** Also confirmed working. Blocks within a cluster have different DSMEM addresses that allow cross-block reads/writes/atomics. Tested on RTX 5090: blocks in a 4-block cluster show distinct addresses (e.g., `0xe93600000400`, `0xe93601000400`).

**NOT available on sm_120:** Cluster multicast (copying from global memory to shared memory of multiple SMs simultaneously). This is sm_100 only.

**NOT available on sm_120:** TMA-based `cp.async.bulk.tensor` for DSMEM. One user hit "illegal instruction" errors with this on RTX 5090.

**Access pattern requirement:** DSMEM accesses should be coalesced and aligned to 32-byte segments.

**Practical applicability:**
- Could enable cross-block cooperation for very large problem sizes
- Producer-consumer patterns across blocks
- Cross-block reductions without going through global memory
- **Not needed for current single-block-per-tile kernels** (attention, GEMM)
- Worth exploring for future multi-block attention (split-K, multi-query) or very large GEMM tiles
- Overhead of cluster synchronization vs. benefit is unknown -- needs benchmarking

### 4. MXFP8 Block-Scaled MMA (sm_120a) -- MEDIUM-HIGH IMPACT

**What:** Hardware-accelerated block-scaled FP8 MMA. Each group of elements shares a scale factor (e8m0 format), enabling higher dynamic range without per-element scaling overhead.

**Instruction:**
```ptx
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
```

**Critical detail -- sm_120 vs sm_120a:**
- `sm_120` (plain): Block-scaled MMA **does NOT compile**. Errors: "Instruction 'mma with block scale' not supported on .target 'sm_120'"
- `sm_120a` (accelerated): Block-scaled MMA **works**. The `a` suffix enables architecture-specific features.
- The RTX 5090 hardware IS sm_120a capable. The issue is the compilation target flag.
- **To compile:** Use `-gencode=arch=compute_120a,code=sm_120a` (NOT `compute_120,code=sm_120`)

**Register layout:**
- A: 4x uint32 (same as regular mma.sync m16n8k32)
- B: 2x uint32
- C/D: 4x float
- SFA, SFB: 1x uint8 each (scale factors, passed as uint32)
- Scale factors use e8m0 format (unsigned 8-bit exponent, no mantissa) with bias of 127

**Impact for workers:**
- **GEMM FP8 worker:** Could improve FP8 GEMM accuracy by allowing per-block scaling instead of per-tensor. Currently at 1.34x cuBLAS with regular FP8 MMA. Block scaling adds hardware-managed dynamic range.
- **Attention FP8 worker:** Our FP8 attention already handles scaling in software (converting BF16 to FP8 with saturation). Block-scaled MMA could reduce the need for careful numerical range management.
- **Key question:** Is the overhead of computing and managing scale factors worth the accuracy improvement? The scale factors themselves take register space and need computation.

### 5. 32-Byte Aligned Vector Types (CUDA 13.0) -- LOW IMPACT

**What:** CUDA 13 introduces `_32a` variants of large vector types for 32-byte alignment: `double4_32a`, `long4_32a`, `ulong4_32a`, `longlong4_32a`, `ulonglong4_32a`. Old types (`double4`, etc.) now emit deprecation warnings.

**However, sm_120 caps vector loads at 128 bits (16 bytes).**

A practitioner working on RTX 5090 decode optimization explicitly confirmed: "sm_120 caps vector loads at 128-bits." Attempting 256-bit `uint8` loads did not work.

The 32-byte aligned types are primarily for sm_100 (datacenter Blackwell) which supports 256-bit loads. The `_32a` alignment may still help sm_120 for cache line alignment purposes but does not enable wider loads.

**Bottom line:** Our existing 128-bit (16-byte) vectorized loads (`int4`, `float4`) are the maximum on sm_120. The 32-byte types are not useful for our workers.

### 6. Shared Memory Register Spilling -- Detailed Architecture Interaction

**Shared memory consumed by spilling:** In NVIDIA's test case, enabling smem spilling consumed 46,080 bytes (45 KB) of shared memory per block. This is significant.

**Interaction with occupancy:**
- Extra smem per block reduces blocks/SM
- Example: kernel using 32 KB smem + 45 KB spills = 77 KB total. Only 1 block fits per SM (vs. potentially 3-4 without spills).
- **Must have explicit launch_bounds to prevent this degradation.**

**Decision framework for workers:**
```
IF kernel has register spills (ptxas -v shows spill bytes > 0)
  AND kernel has unused smem headroom (total smem/block < 50% of 128 KB)
  AND kernel uses explicit __launch_bounds__
  AND kernel does NOT use dynamic shared memory
THEN try enable_smem_spilling
ELSE do not use it
```

---

## Features That Are sm_100 Only (Do NOT Use on sm_120)

### 1. tcgen05 Instructions (Tensor Core Gen 5 -- sm_100 ONLY)
- `tcgen05.mma`, `tcgen05.ld`, `tcgen05.st`, `tcgen05.cp`, `tcgen05.alloc`, etc.
- These are the datacenter Blackwell tensor core instructions
- sm_120 uses `mma.sync` (extended Ampere ISA), NOT tcgen05
- Compiling tcgen05 for sm_120 produces: "Instruction 'tcgen05.*' not supported on .target 'sm_120'"

### 2. TMEM (Tensor Memory -- sm_100 ONLY)
- 256 KB per SM dedicated tensor memory
- Accessed only via tcgen05 instructions
- sm_120 has NO TMEM hardware whatsoever

### 3. wgmma (Warp Group MMA -- sm_90/sm_100 ONLY)
- Hopper's asynchronous warp-group-level MMA
- Compiling for sm_120: "Instruction 'wgmma.fence' not supported on .target 'sm_120'"
- sm_120 uses warp-synchronous `mma.sync`, not warp-group-level MMA

### 4. setmaxnreg (Dynamic Register Reallocation -- sm_90a ONLY)
- Dynamically reallocates registers between warps within a CTA
- Enables producer-consumer warp specialization (e.g., Flash Attention 3 pattern)
- **Requires sm_90a** (Hopper accelerated features). NOT forward-compatible.
- sm_120 does NOT support this instruction
- **Workers cannot use warp specialization with dynamic register reallocation.**
- Warp specialization on sm_120 must use static register allocation only

### 5. f32x2 Packed Float Operations (sm_100 ONLY)
- `FADD2`, `FFMA2`, `FMUL2` SASS instructions for packed 32-bit float pairs
- Available on sm_100 and sm_103 to reduce instruction issue pressure
- **sm_120 does NOT have these.** PTX `add.f32x2` compiles to two separate `FADD` instructions on sm_120.
- No benefit from using f32x2 intrinsics on consumer Blackwell

### 6. cp.async.bulk / TMA (sm_90+ Datacenter ONLY)
- Tensor Memory Accelerator for multi-dimensional async copies
- `cp.async.bulk` requires sm_90+ and TMA hardware
- sm_120 does NOT have TMA
- sm_120 uses traditional `cp.async.ca` / `cp.async.cg` (same as sm_80/sm_89)
- Our existing cp.async double-buffer pipeline is the optimal approach for sm_120

### 7. Cluster Multicast (sm_100 ONLY)
- Copying from global memory to shared memory of multiple SMs simultaneously
- Part of the TMA subsystem
- sm_120 supports clusters but NOT multicast copies

### 8. MXFP4 Block-Scaled MMA (sm_120a, but impractical)
- `mma.sync.aligned.kind::mxf4nvf4.block_scale`
- Technically available on sm_120a but requires the "f" suffix: sm_120f
- Practically difficult to use -- PTX instruction support exists but library support (CUTLASS CuTe DSL) is absent
- Error on plain sm_120: "Feature '.kind::mxf4' not supported on .target 'sm_120'"

### 9. 256-bit Global Loads (sm_100 ONLY)
- `LDG.256` / `STG.256` for 32-byte vector loads/stores
- sm_120 is confirmed to cap at 128-bit (16-byte) vector loads
- The `_32a` aligned types in CUDA 13 are for sm_100, not sm_120

---

## Features Same as Ada (sm_89) -- No Change on sm_120

### 1. MUFU/SFU Throughput: Unchanged
- **16 ops per clock per SM** for MUFU.EX2 (exponential), MUFU.RCP (reciprocal), MUFU.RSQ (reciprocal sqrt)
- Same throughput as Hopper (sm_90) and Ada (sm_89)
- B300/GB300 (Blackwell Ultra datacenter) doubles this to 32 ops/clock -- but that is NOT available on RTX 5090
- **This is why math_pipe_throttle dominates our attention kernel's softmax phase.** The MUFU throughput has not improved on consumer GPUs since Ampere.

### 2. cp.async: Same as sm_80+
- `cp.async.ca.shared.global` and `cp.async.cg.shared.global`
- 4/8/16 byte variants
- No new async copy instructions on sm_120 beyond what sm_89 had

### 3. ldmatrix: Same variants
- `ldmatrix.sync.aligned.m8n8.x1/x2/x4` (.b16)
- `ldmatrix.sync.aligned.m16n16.x1` (.b8, for FP8/FP6/FP4)
- Same instruction set as Ada, no new ldmatrix variants on sm_120

### 4. mma.sync Instruction Set: Same + block_scale addition
- BF16: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` -- unchanged
- FP8: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` -- unchanged
- **New on sm_120a:** `mma.sync.aligned.kind::mxf8f6f4.block_scale.*` -- block-scaled variant (see section 4 above)

### 5. Max Vector Load Width: 128-bit
- Same as Ada, Hopper, Ampere
- `int4` / `float4` / 16-byte loads remain the widest available

---

## CUDA 13 Compiler Improvements (Architecture-Independent)

### 1. --Ofast-compile=<level>
Prioritizes compilation speed over optimization. Useful during development iteration but not for final benchmark builds.

### 2. --frandom-seed=<seed>
Deterministic compilation. Ensures identical PTX/object files across builds. Useful for reproducible benchmarking.

### 3. Improved Fatbin Compression (Zstandard)
17% smaller binaries. No runtime performance impact.

### 4. Stack Canaries (--device-stack-protector=true)
Security feature. Not relevant for kernel optimization.

### 5. Nsight Compute 2025.3
New "Instruction Mix and Scoreboard Dependency" tables in source view. New "Throughput Breakdown" section. **Workers should update ncu if not on this version** -- the new dependency stall analysis can help identify bottlenecks.

### 6. cuBLAS GEMM Autotune
Experimental `CUBLAS_GEMM_AUTOTUNE` parameter for automatic algorithm selection. Could affect our cuBLAS reference baselines -- worth re-running benchmarks after enabling to see if the reference bar moves.

---

## Impact Assessment by Worker

### Attention Worker (main/)
- **enable_smem_spilling:** Unlikely to help. Already at 145 regs, 96 KB smem (tight budget).
- **MXFP8 block_scale (sm_120a):** Worth investigating if pursuing FP8 accuracy improvements. Could reduce need for manual saturation handling.
- **Thread block clusters:** Future consideration for multi-block attention or very long sequences. Not needed now.
- **MUFU unchanged at 16 ops/clock:** Confirms math_pipe_throttle is a hardware limit. Software exp2 emulation (FMA-based) may be worth trying to supplement MUFU capacity, as FA4 does on datacenter.

### GEMM Worker (gemm/)
- **enable_smem_spilling:** Try on experimental kernels with higher reg pressure (>80 regs). Current 6-block config has 32 KB smem headroom.
- **MXFP8 block_scale (sm_120a):** High potential for FP8 GEMM accuracy. Compile with `-arch=compute_120a -code=sm_120a`.
- **Thread block clusters:** Could enable cooperative GEMM tiles across blocks. Worth future exploration for very large problem sizes.

### Fused-MLP Worker (fused-mlp/)
- **enable_smem_spilling:** Good candidate if the fused kernel has high register pressure.
- **MXFP8 block_scale:** Could improve epilogue-fused FP8 pipeline accuracy.

### RMSNorm / DotProduct / Linalg / Numerical Workers
- **enable_smem_spilling:** Try on any kernel reporting spills.
- **redux.sync (s32/u32 only):** Limited utility. Float reductions still need shfl_xor.
- **Thread block clusters:** Potentially useful for cross-block reductions in norm kernels.

---

## Actionable Recommendations

1. **Immediate (all workers):** Add `asm volatile(".pragma \"enable_smem_spilling\";");` to any kernel with register spills and smem headroom. Requires `__launch_bounds__` and no dynamic smem. Test for performance delta.

2. **Immediate (GEMM FP8 worker):** Investigate block-scaled MMA on sm_120a. Compile with `compute_120a`. This is the most impactful new hardware feature for FP8 work.

3. **Immediate (attention worker):** Consider FA4's software exp2 emulation technique to supplement MUFU.EX2. The 16 ops/clock MUFU limit is a hardware wall that won't move on consumer Blackwell. FMA-based exp2 approximation can run on units that are otherwise idle during math_pipe_throttle stalls.

4. **Near-term (all workers):** Update Nsight Compute to 2025.3 for improved dependency stall analysis.

5. **Future (GEMM/attention workers):** Explore thread block clusters for multi-block cooperation patterns. The infrastructure exists on sm_120 but is uncharted territory for our kernel designs.

6. **Do NOT attempt:** setmaxnreg (sm_90a only), f32x2 packed ops (sm_100 only), cp.async.bulk/TMA (sm_90+ only), tcgen05 (sm_100 only), 256-bit vector loads (sm_100 only). These are confirmed absent on sm_120.
