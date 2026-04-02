# gau-nernst: TMA GEMM + Warp Specialization for sm_120 (March 2026)

**Source:** https://github.com/gau-nernst/learn-cuda/tree/main/02c_matmul_sm120
**Relevant to:** GEMM worker
**Worker's current problem:** BF16 GEMM at 0.97x cuBLAS. FP8 at 1.34x cuBLAS.
**Date:** 2026-03-15

---

## What This Is

gau-nernst (Thien Tran) has published three sm_120 GEMM implementations in his
learn-cuda repository, with the latest commits from **March 12-14, 2026**. The work
explores TMA (Tensor Memory Accelerator) for global-to-shared transfers on sm_120
and compares against cp.async.

---

## Why It Matters for Us

### TMA on sm_120 Is Real and Measurable

sm_120 supports TMA (`cp.async.bulk.tensor`) via the CUDA Driver API. This is NOT
tcgen05/datacenter-only -- it works on consumer Blackwell.

**Performance comparison (RTX 5090 @ 400W, BF16, % of SOL):**

| Kernel | 2048 | 4096 | 8192 |
|--------|------|------|------|
| cuBLAS | 77.0% | 83.5% | 91.3% |
| v0 (cp.async) | 71.0% | 82.4% | 79.4% |
| v1 (TMA) | 73.3% | 85.5% | 82.3% |

**Key finding:** TMA is ~2-3% faster than cp.async at intermediate sizes (4096)
but both fall off at 8192. cuBLAS still wins overall.

### TMA Implementation Details (Transferable)

1. **Host-side setup:** `cuTensorMapEncodeTiled()` creates CUtensorMap descriptors
   encoding global memory layout, shared memory box dimensions, and swizzle mode.

2. **Swizzle modes tested:**
   - `CU_TENSOR_MAP_SWIZZLE_128B`: XOR bits [4:6] with [7:9] (128-byte blocks)
   - `CU_TENSOR_MAP_SWIZZLE_64B`: XOR bits [4:5] with [7:8] (64-byte blocks)

3. **Mbarrier synchronization:** TMA uses mbarrier (not cp.async.commit_group) for
   synchronization. Only one thread issues TMA instructions. Phase tracking enables
   ring-buffer management for multi-stage pipelining.

4. **Fence requirements:** `fence.proxy.async.shared::cta` required for async proxy
   visibility between TMA warp and MMA warps.

### v2: Warp Specialization (NEW)

The v2 kernel (March 14, 2026) implements **warp specialization** for sm_120:
- **1 warp:** Dedicated TMA producer (issues all global->shared transfers)
- **Remaining warps:** MMA consumers (compute only)
- Mbarrier-based producer-consumer coordination

This is the **first open-source warp-specialized GEMM for sm_120** we've seen.

Performance notes:
- BF16: BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, 2 stages
- INT8: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, 3 stages
- INT8 correctness issues noted (~224 errors per 16M values)

### v3: cuBLAS Layout (WIP, March 14)

A v3 kernel targeting "cuBLAS layout" was committed March 14 but is marked WIP.
This suggests the author is investigating whether matching cuBLAS's memory layout
for A/B matrices can close the performance gap.

---

## Key Techniques We Haven't Tried

1. **TMA for GEMM global loads.** Our GEMM worker uses cp.async. TMA provides
   automatic swizzling, single-thread issue, and mbarrier synchronization. The
   2-3% improvement at 4096 is modest, but combined with other optimizations, it
   may compound.

2. **Warp specialization.** Dedicating one warp to data movement while others
   compute. This separates producer/consumer concerns and may improve instruction
   scheduling. Our GEMM kernel currently has all warps doing both loads and computes.

3. **CU_TENSOR_MAP_SWIZZLE_128B vs our XOR swizzle.** TMA's hardware swizzle is
   applied automatically by the TMA unit, potentially eliminating the swizzle
   address calculation overhead in our ldmatrix code.

---

## Caveats

1. **Performance doesn't beat cuBLAS.** gau-nernst reaches 85.5% SOL at best vs
   cuBLAS's 91.3% at 8192. Our worker is at 97% of cuBLAS, which is significantly
   better.

2. **TMA setup overhead.** The CUtensorMap must be created on the host and passed
   to the kernel. For dynamic shapes, this adds host-side latency.

3. **Warp specialization reduces effective MMA warps.** If 1 of 4 warps is
   dedicated to TMA, only 3 warps do MMA. This is only beneficial if the TMA warp
   can keep shared memory full faster than 4 warps splitting load/compute.

4. **INT8 correctness issues** in v2 suggest the warp specialization synchronization
   is not fully debugged.

---

## Recommendation

**Monitor, don't adopt yet.** Our GEMM worker is already at 0.97x cuBLAS with
cp.async, which is better than gau-nernst's TMA results. However:

- If the worker is exploring new optimization directions for the last 3% to match
  cuBLAS, TMA warp specialization is a genuine alternative path worth studying.
- The mbarrier-based producer-consumer pattern could be applicable to our attention
  kernel's memory pipeline.
- Watch the v3 "cuBLAS layout" kernel for potential new insights.
