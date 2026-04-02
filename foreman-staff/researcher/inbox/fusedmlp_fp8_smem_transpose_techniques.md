# FP8 B-Matrix Shared Memory Transpose Techniques for GEMM on sm_120

**Sources:**
- https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
- https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8
- https://leimao.github.io/blog/CuTe-ldmatrix/
- https://research.colfax-intl.com/adding-fp8-to-flashattention/
- https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
- https://forums.developer.nvidia.com/t/understanding-cutlass-permuted-shared-memory-layout/303697
- https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
- https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/
- https://github.com/NVIDIA/cutlass/discussions/911

**Relevant to:** fused-mlp worker
**Worker's current problem:** Native FP8 B with row-major smem uses scalar byte loads at stride-64 causing bank conflicts, making it 4% slower than the ldmatrix+CVT (load as BF16, convert to FP8 in registers) approach.

---

## What This Is

A comprehensive analysis of techniques to get FP8 B-matrix data from global memory into the correct register fragment layout for `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` on sm_120, focusing on whether in-shared-memory transpose is feasible and what alternatives exist. Covers cp.async capabilities, ldmatrix variants (including the new m16n16 shape on Blackwell), warp-shuffle and PRMT-based approaches, and CUTLASS/ThunderKittens patterns.

---

## Why It Matters for Us

The worker has mapped the B fragment layout:
- `b0 = {B[k, n], B[k+1, n], B[k+16, n], B[k+17, n]}` (interleaved, not consecutive)
- `b1 = {B[k+8, n], B[k+9, n], B[k+24, n], B[k+25, n]}`

Row-major FP8 smem means stride-64 between k-rows for the same n-column, causing bank conflicts on scalar byte loads. The worker estimates column-major smem transpose could enable 32-bit aligned loads (4 consecutive k-values per load), saving ~256 cycles of CVT overhead per K-tile. The worker's existing brief in `fused_mlp_fp8_smem_transpose_for_b.md` proposed an ldmatrix+PRMT alternative. This brief provides the missing web research to validate or refine both approaches.

---

## Key Findings

### 1. cp.async CANNOT Do Layout Transformation

**Critical answer to the worker's first question:** `cp.async` (non-bulk, Ampere-style) performs a straight byte copy from global to shared memory. It does NOT support transpose, scatter, or any layout transformation. The data lands in shared memory in exactly the same layout as global memory.

On Hopper, `cp.async.bulk` with TMA (Tensor Memory Accelerator) CAN do layout transformations via tensor descriptors. However, TMA is NOT available on sm_120 (consumer Blackwell uses `mma.sync` ISA, not `tcgen05`). The Colfax CUTLASS tutorial confirms TMA transpose is sm_100+ datacenter only.

**Bottom line:** Any transpose must happen AFTER the cp.async completes, as a post-load step in shared memory or during the smem-to-register load.

### 2. ldmatrix.m16n16 with .trans and b8x16 -- New on Blackwell but Uncertain on sm_120

The PTX ISA (8.7+) defines a new ldmatrix shape for 8-bit data:

```
ldmatrix.sync.aligned.m16n16.x1{.trans}.shared.b8x16 r, [p];
```

Key details:
- `.m16n16` shape is documented as available on "sm_100 and higher GPU versions"
- `.b8x16` is the 8-bit destination format (16 bytes per row)
- `.trans` modifier IS part of the instruction syntax for m16n16
- The instruction loads a 16x16 matrix of 8-bit elements (256 bytes total per warp)

**The critical uncertainty:** "sm_100 and higher" should include sm_120, but sm_120 is consumer Blackwell with a different ISA subset than datacenter sm_100/sm_101. The NVIDIA forum thread about ldmatrix on sm_120 shows developers struggling to find the right instruction. The forum poster ended up using direct register loads, not ldmatrix.m16n16.

**Recommendation:** This is worth a 30-minute empirical test. Write a minimal kernel that attempts:
```ptx
ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16 {%0}, [%1];
```
If it compiles and runs correctly on sm_120, this is the cleanest solution: hardware-accelerated transpose of 8-bit data from shared memory to registers, matching the pattern the worker already uses for BF16 (ldmatrix.m8n8.x4.trans with b16).

### 3. The ldmatrix.b16 + PRMT Approach (Already in Worker's Brief)

The existing brief proposes keeping B as BF16 in smem and using ldmatrix.b16 + 2 PRMT to merge K-halves. This is a solid approach already documented in `fused_mlp_fp8_smem_transpose_for_b.md`. The web research confirms the viability:

- **Colfax FP8 FlashAttention** uses `__byte_perm` (the C++ intrinsic for PTX `prmt`) combined with `__shfl_sync` to rearrange FP8 data between the accumulator layout and operand layout. Their exact selectors: `upper_map[4] = {0,3,1,2}` and `lower_map[4] = {1,2,0,3}` for shuffle, plus byte selectors like `0x7654` and `0x3210` for prmt.

- **ThunderKittens** loads FP8 via ldmatrix treating 16-bit pairs as two FP8 values: "we let the instruction load in 16-bits and use this to fill 2 fp8 values per load."

**However**, the PRMT selectors `0x5410` and `0x7632` in the existing brief need empirical validation. The interleaved pattern `{k, k+1, k+16, k+17}` means the PRMT must pick bytes from two different 32-bit registers (r0 from K=0-15, r1 from K=16-31) in a specific interleaving. A test kernel printing register contents byte-by-byte after ldmatrix and after PRMT is essential.

### 4. In-Shared-Memory Transpose: Feasible but Expensive

If ldmatrix.m16n16 doesn't work on sm_120 and the PRMT approach has issues, the explicit smem transpose is the fallback. Key findings from web research:

**Bank conflict avoidance for 8-bit transpose:**
- Shared memory has 32 banks, each 4 bytes wide
- For FP8 (1 byte per element), 4 elements map to each bank
- A 32xN column-major layout where N=8 (MMA tile width) means 32 bytes per column = exactly 8 banks. Adjacent columns alias to the same banks
- **Fix 1 (padding):** Pad each column to 36 bytes (+4 bytes). Cost: 12.5% wasted smem
- **Fix 2 (XOR swizzle):** `dst_col_swizzled = dst_col ^ (dst_row >> 2)` spreads accesses across banks. The CUTLASS approach: `store_column = (lane_id % 8) ^ (lane_id / 8)` for 16-bit data; adapt for 8-bit by adjusting the shift amount

**Cost analysis for explicit transpose:**
- Write phase: 128 threads transpose a 32x64 tile (K=32, N=64). Each thread handles 32*64/128 = 16 bytes. Achievable with 4x 32-bit loads + 4x 32-bit stores with byte shuffle (prmt) between.
- Sync overhead: 2 `__syncthreads()` per K-tile (one after cp.async, one after transpose)
- Double-buffer impact: Cannot overlap the transpose with the next cp.async prefetch unless you have 3 smem buffers (cp.async target, transpose source, compute source)

**NVIDIA's blog on efficient transpose** shows the penalty of the extra sync is significant -- their optimized transpose kernel spends ~20% of time on synchronization overhead.

### 5. The Warp-Shuffle Approach (No Smem Transpose Needed)

An approach NOT in the existing briefs: use warp shuffles to rearrange FP8 bytes AFTER loading with ldmatrix.b16, bypassing the need for any smem transpose:

1. Load B with `ldmatrix.m8n8.x2.trans` (treating FP8 pairs as b16) -- same as current BF16 path
2. Use `__shfl_xor_sync` + `prmt.b32` to rearrange bytes across threads into the m16n8k32 B-fragment layout

This is similar to what Colfax does for their FP8 FlashAttention accumulator-to-operand layout conversion. The advantage: no smem overhead, no extra syncs, just a few register instructions per warp.

**Cost estimate:** 2 `prmt` + 1-2 `shfl.sync` per B fragment = ~6-8 instructions. Compare to ~7 CVT instructions in the current ldmatrix+CVT approach. Potentially break-even or slight win, especially if the shuffles can overlap with MMA execution.

### 6. Pre-Quantized FP8 B Weights (The Biggest Win)

All approaches above assume B starts as BF16 in global memory. If B is pre-quantized to FP8 (which is the standard for inference with quantized weights):

1. **cp.async halves B bandwidth** (1 byte vs 2 bytes per element)
2. **ldmatrix.b16 loads FP8 pairs** naturally (two FP8 values per 16-bit slot)
3. **PRMT or shfl** to get the interleaved layout
4. **Zero CVT instructions**

This is the path with the largest potential gain. The cp.async bandwidth savings alone could be 5-10% for GEMM1, plus the CVT elimination.

---

## Recommended Experiment Sequence

**Priority 1: Test ldmatrix.m16n16.trans.b8x16 on sm_120** (~30 min)
```cuda
// Minimal test: does this compile and produce correct output on sm_120?
uint32_t result;
asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16 {%0}, [%1];"
             : "=r"(result) : "r"(smem_addr));
```
If this works, it's the cleanest solution. The hardware does the 8-bit transpose for you.

**Priority 2: Validate ldmatrix.b16 + PRMT selectors** (~1 hour)
Write a test kernel that:
1. Fills smem with known FP8 values in row-major layout
2. Loads with `ldmatrix.m8n8.x2.trans` (interpreting FP8 pairs as b16)
3. Applies PRMT with various selectors
4. Prints byte-by-byte register contents
5. Compares against known-correct B fragment layout from `fp8_native_b_fragment_layout.md`

**Priority 3: Benchmark warp-shuffle approach vs explicit smem transpose** (~2 hours)
Compare the register-only approach (ldmatrix + prmt + shfl) against the smem transpose approach (cp.async + sync + transpose + sync + 32-bit loads). Measure with ncu focusing on `long_scoreboard_stall` and `smem_bank_conflict` metrics.

---

## Caveats

1. **ldmatrix.m16n16 availability on sm_120 is NOT confirmed.** The PTX ISA says "sm_100 and higher" but sm_120 is a different microarchitecture than sm_100. Our hard-won lesson: "Always test PTX instructions empirically on sm_120. Some instructions work, some don't."

2. **The PRMT byte selector values are theoretical.** The `0x5410`/`0x7632` selectors in the existing brief assume a specific byte ordering from ldmatrix output. The actual ordering depends on which thread in the warp you're in and how ldmatrix.trans maps the transposed data to registers. Must be validated empirically.

3. **The interleaved B fragment layout `{k, k+1, k+16, k+17}` complicates all approaches.** This is NOT a simple transpose -- it's an interleaved gather from two K-halves. Any approach must handle this interleaving, not just a straight column-major read.

4. **Double-buffer interaction:** The explicit smem transpose approach breaks the standard double-buffer pipeline because you can't overlap the transpose step with the next cp.async prefetch (both use shared memory). This could cause a pipeline bubble that negates the CVT savings. The register-only approaches (PRMT/shfl) don't have this problem.

5. **For Hopper/datacenter Blackwell content:** Most FP8 GEMM resources online target WGMMA (sm_90/sm_100), which uses a completely different programming model (B lives in smem, loaded via TMA with built-in transpose). These approaches do NOT apply to sm_120's `mma.sync` ISA. Filter aggressively.
