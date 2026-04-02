# CUDA 13: Shared Memory Register Spilling (10x Lower Latency Spills)

**Source:** https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/
**Relevant to:** ALL workers with register pressure (attention, gemm, fused-mlp)
**Worker's current problem:** Multiple workers are register-constrained. Attention at 165/170 regs (5 spare), GEMM FP8 at 1 block/SM. Register pressure prevents exploring approaches that need more live variables.

## What This Is

CUDA 13.0 adds a compiler feature that spills registers to **shared memory** instead
of local memory (L2 cache). Shared memory access is ~10x lower latency than L2 spills.
The feature is opt-in via an inline ASM pragma.

## How to Enable

```cuda
__launch_bounds__(128, 3)  // MUST specify launch_bounds
__global__ void kernel(...) {
    asm volatile (".pragma \"enable_smem_spilling\";");
    // rest of kernel
}
```

That's it. One line. The compiler handles everything.

## How It Works

- Compiler automatically redirects register spills to static shared memory
- If shared memory is insufficient, remaining spills fall back to local memory (L2)
- The spill space is carved from the static shared memory allocation
- Increases per-block shared memory usage

## Performance Impact

- **NVIDIA blog example:** 7.76% kernel speedup (8.35 us -> 7.71 us)
- **QUDA library:** 5-10% gains on register-heavy lattice QCD kernels
- Eliminates the ~100-cycle latency penalty of L2 spills, replacing with ~10-cycle smem access

## Per-Worker Analysis

### Attention (165 regs, 32KB smem, 3 blocks/SM)
- Shared memory budget: 99KB / 3 blocks = 33KB per block
- Current usage: 32KB (double-buffered K/V)
- **Available for spills: ~1KB** ← VERY TIGHT
- **Verdict: Probably NOT useful** — almost no headroom. Would need to drop
  to 2 blocks/SM to get smem space, which is known to regress (exp 50-53).

### GEMM FP8 (48KB smem, 1 block/SM via launch_bounds(128,1))
- Shared memory budget: 99KB (1 block = entire SM)
- Current usage: 48KB
- **Available for spills: ~51KB** ← LOTS OF HEADROOM
- **Verdict: PROMISING.** The worker could explore approaches previously rejected
  due to register pressure (e.g., B fragment double-buffering, wider tiles with
  8 warps) knowing that spills are only ~10 cycles instead of ~100. The 48KB→99KB
  gap gives ample room.

### GEMM BF16 (16KB smem, 6 blocks/SM)
- Shared memory budget: 99KB / 6 blocks = 16.5KB per block
- Current usage: 16KB
- **Available for spills: ~0.5KB** ← NO HEADROOM
- **Verdict: NOT useful** with current config. Would need to reduce blocks/SM.

### Fused-MLP (uses GEMM primitives, same constraints as GEMM)
- Same analysis as GEMM BF16/FP8 depending on path.

## Requirements and Caveats

1. **MUST use `__launch_bounds__`** — without it, compiler assumes max threads and
   over-allocates shared memory, hurting occupancy
2. **CANNOT use dynamic shared memory** — only static smem. Kernels using
   `cudaFuncSetAttribute` for dynamic smem are incompatible
3. **CANNOT use `-rdc=true`** (relocatable device code) — requires whole-program mode
4. **Compilation flag:** `-rdc=false` (default, usually fine)
5. **Increases static smem usage** — check with `--ptxas-options=-v` to see actual allocation

## When to Use

Use when:
- Your kernel has register spills (check `--ptxas-options=-v` for "bytes spill stores")
- You have shared memory headroom (total smem < 99KB at desired occupancy)
- You're exploring an approach that would need more registers than currently available

Do NOT use when:
- Shared memory is already near the 99KB limit
- Reducing blocks/SM to make smem room would hurt more than the spill improvement
- Kernel uses dynamic shared memory allocation

## Recommendation

**GEMM FP8 worker should try this immediately** — 51KB headroom, 1 block/SM config,
and register pressure has been a recurring theme in FP8 experiments. Add the one-line
pragma and re-benchmark. If spills move from local to shared, previously-rejected
approaches (wider tiles, more warps) might become viable.

Other workers should note this for future experiments that hit register ceilings.
