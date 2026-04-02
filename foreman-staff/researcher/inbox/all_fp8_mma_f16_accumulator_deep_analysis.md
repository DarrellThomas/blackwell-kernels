# FP8 MMA with FP16 Accumulator: Deep Technical Analysis

**Sources:**
- [gau-nernst NVRTC matmul benchmarks](https://gau-nernst.github.io/nvrtc-matmul/)
- [Lei Mao MMA Peak Performance Benchmarks](https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances/)
- [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [PTX ISA 8.7 (CUDA 12.8) - introduced f16 accumulator](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html)
- [NVIDIA RTX 5090 specs](https://forums.developer.nvidia.com/t/rtx-5090-specs-fp16-tensor-tflops-is-ambiguous/351063)
- [SageAttention2++ two-level accumulation](https://arxiv.org/html/2505.21136v1)
- [FlashAttention-3 FP8 techniques](https://arxiv.org/abs/2407.08608)
- [PyTorch FP16 accumulation support](https://github.com/pytorch/pytorch/issues/123558)
- [CUTLASS 4.2 changelog - FP16 accum for sm89](https://docs.nvidia.com/cutlass/4.3.2/CHANGELOG.html)
- [ldmatrix FP8 on sm120 forum](https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254)
- [gau-nernst flash attention 5090](https://gau-nernst.github.io/fa-5090/)

**Relevant to:** attention worker, GEMM worker, fused-MLP worker
**Date:** 2026-03-15

---

## Executive Summary

The `mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16` instruction provides **2x peak tensor
core throughput** compared to the current `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
On RTX 5090, this translates to a measured **1.49x real-world speedup** (692 vs 465 TFLOPS at
M=N=K=8192, power-limited). It also halves accumulator register count (2 vs 4 registers per tile).

**However, there are critical tradeoffs:**
- The K dimension per instruction halves (k16 vs k32), doubling instruction count
- FP16 accumulators risk overflow (max 65504) and precision loss
- The A operand fragment layout changes, requiring code restructuring
- For attention, softmax MUST remain FP32 -- only the MMA accumulators change

**Verdict by worker:**
- **GEMM:** HIGH priority. Direct 1.49x throughput gain. FP16 precision acceptable for training.
- **Attention:** MEDIUM priority. Register savings valuable but softmax precision needs care.
- **Fused-MLP:** MEDIUM priority. GEMM1 benefits; overall gain limited since GEMM2 uses cuBLAS.

---

## 1. The Instruction

### What Changed

PTX ISA 8.7 (CUDA 12.8) introduced FP16 accumulator support for FP8 MMA. Our CUDA 13.0
(PTX ISA 9.1) already supports this. No CUDA upgrade needed.

```
CURRENT (FP32 accumulator):
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    {d0, d1, d2, d3},     // D: 4x float32 (128 bits)
    {a0, a1, a2, a3},     // A: 4x uint32  (128 bits, 32 FP8 values)
    {b0, b1},              // B: 2x uint32  (64 bits,  16 FP8 values)
    {c0, c1, c2, c3};     // C: 4x float32 (128 bits)
    // Computes: 16*8*32*2 = 8192 FLOPs per instruction

NEW (FP16 accumulator):
mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16
    {d0, d1},              // D: 2x uint32  (64 bits, 4 FP16 values packed)
    {a0, a1, a2, a3},     // A: 4x uint32  (128 bits, 32 FP8 values)
    {b0, b1},              // B: 2x uint32  (64 bits,  16 FP8 values)
    {c0, c1};              // C: 2x uint32  (64 bits, 4 FP16 values packed)
    // Computes: 16*8*16*2 = 4096 FLOPs per instruction
```

### Critical Shape Difference

The FP16 accumulator variant uses **m16n8k16**, NOT m16n8k32. This is not a drop-in
replacement. PTX ISA does NOT support m16n8k32 with f16 accumulator for FP8.

| Property | FP32 accum (current) | FP16 accum (new) |
|----------|---------------------|-----------------|
| Shape | m16n8k**32** | m16n8k**16** |
| K per MMA | 32 | 16 |
| FLOPs per MMA | 8192 | 4096 |
| D/C registers | 4 (float32) | 2 (uint32, packed f16x2) |
| A registers | 4 (uint32) | 4 (uint32) |
| B registers | 2 (uint32) | 2 (uint32) |
| Instructions for K=64 | 2 | 4 |
| Peak throughput (5090) | ~419 TFLOPS | ~838 TFLOPS |

The A operand for m16n8k16 with FP8 inputs holds 32 FP8 values (4 regs x 8 bytes =
32 bytes = 32 FP8 values), same register count as m16n8k32. But the fragment layout
within those registers is different because the M and K mapping changes. The fragment
packing follows the m16n8k16 pattern (same as BF16/FP16 MMA), not the m16n8k32 pattern.

### A/B Fragment Layout for m16n8k16 FP8

Since m16n8k16 with FP8 uses the SAME shape as BF16 MMA (m16n8k16), the fragment
layout should follow the same thread-to-element mapping as `mma.sync.m16n8k16.f32.bf16.bf16.f32`.
This means:

- **A fragments:** Same layout as BF16 A fragments. `ldmatrix_x4` should work as-is.
  The a1/a2 register swap (`ldmatrix_x4_mma`) should still apply.
- **B fragments:** Same layout as BF16 B fragments. `ldmatrix_x2_trans` should work.
- **Key insight:** Since we already load BF16 from smem and convert to FP8, we can
  instead load FP8 from smem using the SAME ldmatrix pattern as BF16 (the bytes
  land in the right places for m16n8k16). This needs empirical verification.

**WARNING:** The A operand holds 32 FP8 values packed into 4 uint32 registers, but only
K=16 of them are used per MMA. The exact packing and which bytes map to which matrix
positions needs to be verified empirically on sm_120. Do NOT assume the m16n8k32 FP8
fragment layout -- it is different.

---

## 2. Throughput Analysis

### Peak Performance (RTX 5090, 170 SMs)

| Data Type | Accumulator | Peak TFLOPS (dense) | Peak TFLOPS (sparse) |
|-----------|-------------|---------------------|---------------------|
| FP8 e4m3 | FP16 | ~838 | ~1676 |
| FP8 e4m3 | FP32 | ~419 | ~838 |
| FP16 | FP16 | ~419 | ~838 |
| FP16 | FP32 | ~209.5 | ~419 |
| BF16 | FP32 | ~209.5 | ~419 |

The 2x relationship is exact: FP16 accumulator doubles the peak throughput for both
FP8 and FP16 input types. The hardware mechanism: with FP16 accumulators, the tensor
core can execute two m16n8k16 MMAs in the same cycles as one m16n8k32 with FP32 accumulators.
The narrower accumulator write path frees the register file port for the next instruction.

### Measured Performance (gau-nernst, RTX 5090 @ 600W, M=N=K=8192)

| Configuration | TFLOPS | % of Peak |
|--------------|--------|-----------|
| FP8 + FP32 accum | 465 | ~55% of 838 |
| FP8 + FP16 accum | 692 | ~83% of 838 |
| Speedup | **1.49x** | |

The 1.49x (not 2x) is power-limited. At 600W power cap, the RTX 5090 hits a power wall
before reaching full throughput. The B200 (datacenter, higher power budget) achieves
the full 2x (1929 vs 965 TFLOPS).

**For our kernels:** Our FP8 kernels run at lower SM utilization (43.8% for attention,
67% for GEMM) where power is not the bottleneck. The actual speedup for the MMA phase
alone could be closer to 2x.

### Why It Is Faster Despite 2x More Instructions

Each m16n8k16 FP16-accum MMA does half the FLOPs of m16n8k32 FP32-accum, but executes
in approximately half the cycles. The net effect is:

- Same K dimension requires 2x MMA instructions
- Each instruction executes ~2x faster (half the accumulator write bandwidth)
- Net throughput: ~2x (limited by power in practice to ~1.5x)
- Register pressure: lower (2 vs 4 accumulator regs per tile)

The hardware can pipeline m16n8k16 MMAs more tightly because the FP16 accumulator write
uses half the register file bandwidth of FP32.

---

## 3. Register Budget Analysis

### Attention Worker (FP8 Kernel)

**Current state:** 165 registers, 170.7 threshold for 3 blocks/SM, **5 registers spare**

The FP8 attention kernel has these accumulator arrays:

| Accumulator | Current (FP32) | With FP16 | Savings |
|------------|----------------|-----------|---------|
| S (QK^T scores) | `[WARP_Q/16][BKV/8][4]` = `[4][8][4]` = 128 regs | `[4][8][2]` = 64 regs | **64 regs** |
| O (output) | `[WARP_Q/16][DIM/8][4]` = `[4][8][4]` = 128 regs | `[4][8][2]` = 64 regs | **64 regs** |

Wait -- these numbers assume WARP_Q=64. Let me recalculate with the actual attention
kernel parameters (WARP_Q=16 for the inner warp tile, BKV=64, DIM=64):

| Accumulator | Current (FP32) | With FP16 | Savings |
|------------|----------------|-----------|---------|
| S (QK^T scores) | `[1][8][4]` = 32 regs | `[1][8][2]` = 16 regs | **16 regs** |
| O (output) | `[1][8][4]` = 32 regs | `[1][8][2]` = 16 regs | **16 regs** |
| **Total accum** | **64 regs** | **32 regs** | **32 regs** |

**32 registers saved.** This changes the register budget from 165 to ~133, pushing
well below the 170.7 threshold with 37 spare registers. This opens several options:

1. **Stay at 3 blocks/SM:** Use 37 spare registers for more aggressive prefetching,
   wider tiles, or better scheduling. Margin goes from "barely fits" to "comfortable."

2. **Try 4 blocks/SM:** With 133 registers and 4 warps/block (128 threads), 4 blocks =
   16 warps = 512 threads. Register file: 64K/512 = 128 regs/thread. Need to get from
   133 down to 128. That's only 5 more registers to shed. With `launch_bounds(128, 4)`,
   this may be achievable. The smem budget: 4 blocks x 32KB = 128KB = exactly at SM limit.
   **This was tried before (experiment 60) and regressed due to L1 cache pressure.**
   However, with FP16 accumulators, the instruction mix is different and the bottleneck
   may shift.

3. **Wider BKV tile:** More K per iteration = fewer softmax passes. BKV=128 was
   previously impossible due to registers. With 32 spare, it may fit. But smem doubles
   to 64KB, limiting to 1-2 blocks/SM.

**Most promising path:** Keep 3 blocks/SM but use the register headroom to eliminate
conversion overhead. The 32 spare registers could hold pre-converted FP8 fragments,
enabling better overlap of conversion with MMA.

### GEMM Worker (FP8 Kernel, 64x128 tile)

**Current state:** `launch_bounds(128, 1)`, 48KB smem, 67% SM throughput

The FP8 GEMM accumulator is `[WARP_M/16][WARP_N/8][4]` per warp.
For the 64x128 tile with 4 warps (2x2 warp tiling), each warp handles a 32x64 subtile:

| Accumulator | Current (FP32) | With FP16 | Savings per warp |
|------------|----------------|-----------|-----------------|
| C/D accum | `[2][8][4]` = 64 regs | `[2][8][2]` = 32 regs | **32 regs** |

32 registers saved per warp. This is massive for GEMM. Possible uses:

1. **Higher occupancy:** With 32 fewer regs, might fit 2 blocks/SM instead of 1.
   This would dramatically increase warp-level parallelism.

2. **Wider tiles:** 64x256 tile becomes feasible. Compute/load ratio increases from
   5.3 to ~10.6 MMA/KB, making the kernel strongly compute-bound.

3. **Deeper pipeline:** Extra registers allow 3-stage pipeline within the register
   budget (previously killed by occupancy loss).

### Fused-MLP Worker

The FP8 GEMM1 in fused-MLP uses the same GEMM building blocks. Benefits mirror the
GEMM worker analysis. Since GEMM2 uses cuBLAS, only GEMM1 gets the FP16 accum benefit.

---

## 4. Precision Analysis

### FP16 Accumulator Characteristics

| Property | FP16 | FP32 | Impact |
|----------|------|------|--------|
| Exponent bits | 5 | 8 | Range: +-65504 vs +-3.4e38 |
| Mantissa bits | 10 | 23 | Precision: ~3.3 vs ~7.2 decimal digits |
| Smallest subnormal | 5.96e-8 | 1.4e-45 | Underflow risk |
| Max value | 65504 | 3.4e38 | **Overflow risk for large K** |

### Risk Assessment by Use Case

#### GEMM (K=4096)

With FP8 e4m3 inputs (max value 448, ~2.4 decimal digits precision):

- Each multiply produces at most 448 * 448 = 200,704
- **This EXCEEDS FP16 max (65504)!** A single product can overflow.
- With K=4096 accumulation in FP16, overflow is virtually guaranteed for
  non-trivial inputs.

**Mitigation: Tile-level accumulation.** Accumulate within a single m16n8k16 MMA
(K=16) in FP16, then promote to FP32 after each MMA before adding to the running sum.
This is the "promoted accumulation" or "two-level accumulation" technique:

```
// Pseudocode for promoted accumulation
float32 C_fp32[...] = {0};      // FP32 running accumulator (in registers)
for (k = 0; k < K; k += 16) {
    half2 C_fp16[...] = {0};    // FP16 tile accumulator (reset each iter)
    mma.sync.m16n8k16.f16.e4m3.e4m3.f16(C_fp16, A, B, C_fp16);
    C_fp32 += cvt_f32(C_fp16);  // Promote and accumulate
}
```

This costs one `cvt.f32.f16` per accumulator register per K iteration, but:
- Prevents FP16 overflow entirely (single MMA result K=16 stays in range)
- Gets full 2x MMA throughput benefit
- FP32 accumulator is in registers, no smem needed
- The conversion is 1 instruction per 2 values (f16x2 to 2x f32)

**Cost:** 2 extra CVT instructions per MMA (converting 2 uint32 f16x2 to 4 float32).
With K=64 (BLOCK_K=64), that's 4 MMAs x 2 CVTs = 8 extra instructions per K-block.
This is negligible compared to the MMA throughput gain.

**Alternative: SageAttention2++ approach.** Accumulate 2 consecutive MMAs in FP16 before
promoting to FP32. This halves the conversion overhead at the cost of slightly larger
intermediate values (K=32 accumulated in FP16). For our FP8 inputs with values < 448,
two MMAs accumulate at most 32 * 448 * 448 products, but each thread only sees a
subset -- the per-thread accumulation is much smaller. This needs empirical testing.

#### Attention: QK^T (Score Computation)

QK^T computes dot products of Q and K rows, each of dimension D=64.
- Q values are BF16, converted to FP8 e4m3 (max 448)
- K values are BF16, converted to FP8 e4m3 (max 448)
- Each score is a D=64 dot product: sum of 64 products

Per-product max: 448 * 448 = 200,704 > 65504 (FP16 overflow).

**Same overflow risk as GEMM.** Must use promoted accumulation: accumulate each
m16n8k16 MMA (K=16 of D=64) in FP16, then promote to FP32.

After FP32 promotion, the scores feed into softmax (exp2f, max-subtract, normalize).
Softmax MUST remain FP32. This is a universal requirement -- even FlashAttention-3
on Hopper keeps softmax in FP32.

**Implementation for attention QK^T:**
```
// For each KV block, accumulate QK^T scores
float S_fp32[...] = {0};              // FP32 softmax input
for (dc = 0; dc < DIM; dc += 16) {
    half2 S_fp16[2] = {0};            // FP16 tile result
    mma.sync.m16n8k16.f16.e4m3.e4m3.f16(S_fp16, Q_frag, K_frag, {0,0});
    // Zero C input: don't accumulate in FP16 across K iterations
    S_fp32 += cvt_f32(S_fp16);        // Promote immediately
}
// S_fp32 now has QK^T scores in FP32 for softmax
```

**Register impact:** S_fp16 is 2 registers (temporary, reused each iteration).
S_fp32 is 4 registers (running sum). Total: 6 regs vs current 4 regs. But wait --
currently we accumulate QK^T directly in FP32 C/D. With promoted accumulation, we
need both the FP16 MMA output (2 regs) AND the FP32 running sum (4 regs) live
simultaneously. Net cost: +2 registers per score tile, not -2.

**This means QK^T does NOT save registers with promoted accumulation.** The register
savings come only from the OUTPUT accumulator (O), not from intermediate scores.

#### Attention: PV (Output Accumulation)

P*V accumulates across all KV blocks. The P values are softmax outputs in [0,1].
V values are BF16 (or FP8 e4m3, max 448).

- P * V per element: [0,1] * 448 = max 448. Well within FP16 range.
- Across KV blocks: O += rescale * P*V. The rescaling divides by sum of exponentials,
  keeping values bounded.

**PV is the safest candidate for FP16 accumulation.** The products are naturally bounded
by the softmax normalization. However, O accumulates across ALL KV blocks (N/BKV
iterations). With N=2048 and BKV=64, that's 32 iterations of accumulation.

Over 32 iterations, FP16 accumulation error compounds. The output values stay bounded
(no overflow), but precision degrades. For inference, this is likely acceptable.
For training, it depends on the gradient sensitivity.

**Hybrid approach (recommended for attention):**
- QK^T MMA: Use FP16 accumulator with promoted accumulation to FP32 for softmax
- PV MMA: Use FP16 accumulator, accumulate O in FP16 across KV blocks
- Final output: Convert FP16 O to BF16 for storage

This saves 16 registers on O (from 32 to 16 for the `[1][8]` accumulator array),
which is the most impactful saving.

**More conservative approach:**
- QK^T MMA: FP16 accumulator with immediate FP32 promotion (no net register savings)
- PV MMA: FP16 accumulator with periodic FP32 promotion every N KV blocks
- This bounds error while still getting ~1.5x MMA throughput

#### Precision Benchmarks from Literature

| System | Technique | Accuracy | Notes |
|--------|-----------|----------|-------|
| SageAttention2++ | FP8 PV + FP32+FP16 two-level accum | 99.97% cosine sim | 2 MMA results in FP16 before FP32 promotion |
| FlashAttention-3 | FP8 + block quantization | RMSE 9.1e-3 | 2.6x lower error than naive FP8 |
| PyTorch (Llama-2-7b) | FP16 GEMM + FP16 accum | +0.0006 perplexity | 40% end-to-end speedup on 4090 |
| cuBLAS (consumer Blackwell) | FP16 accum (default for perf) | Sufficient for training | "Full speed with FP16, half speed with FP32" |

---

## 5. Implementation Guide

### For GEMM Worker

**Priority: HIGH.** This is the simplest and most impactful change.

**Step 1:** Replace MMA instruction:
```ptx
// OLD (in gemm inner loop):
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 ...

// NEW:
mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16 ...
```

**Step 2:** Adjust K-loop iteration count (2x more iterations for same BLOCK_K).

**Step 3:** Add promoted accumulation: after each MMA, convert FP16 result to FP32
and add to FP32 running accumulator.

**Step 4:** Adjust fragment loading. Since the A operand fragment layout changes from
m16n8k32 to m16n8k16, the ldmatrix/conversion pipeline needs restructuring:
- Currently: load BF16 from smem (ldmatrix_x4), convert to FP8 (cvt.e4m3x2), feed to k32 MMA
- New: load BF16 from smem (ldmatrix_x4), convert to FP8 (cvt.e4m3x2), feed to k16 MMA
- The BF16->FP8 conversion produces 4 bytes per pair, same as before
- But each MMA only consumes K=16 values, so the conversion can be split

**Step 5:** Benchmark vs cuBLAS 13.2 FP8 (which likely already uses FP16 accum).

**Expected result:** 1.3-1.5x throughput improvement for the MMA phase. Overall
kernel speedup depends on how much time is MMA vs memory. Current math_throttle is
27%, suggesting significant MMA phase contribution.

### For Attention Worker

**Priority: MEDIUM.** Register savings are valuable but the kernel is latency-bound,
not compute-bound.

**Recommended approach: PV-only FP16 accumulation.**

Keep QK^T using the existing m16n8k32 FP32 path (or switch to m16n8k16 with promoted
accumulation -- either way, softmax stays FP32). Switch PV to m16n8k16 FP16 accumulator.

Benefits:
- O accumulator drops from 32 to 16 registers (-16 regs)
- PV MMA throughput approximately doubles
- The PV phase is a smaller fraction of total time than QK^T + softmax
- Softmax precision unaffected

Register budget after PV-only change: 165 - 16 = ~149 registers.
Margin: 170.7 - 149 = 21.7 spare registers. Comfortable.

**Alternative: Full FP16 accumulation (both QK^T and PV).**
QK^T with promoted accumulation costs +2 registers per score tile (need both FP16 temp
and FP32 running sum). S accumulator: 2 regs FP16 temp + 4 regs FP32 running = 6 total
(vs current 4). Net: +2 per score tile.

With `[1][8]` score tiles: +16 registers. Combined with O savings of -16, net is zero.
But the MMA throughput doubles for both phases.

**Best strategy:** Start with PV-only FP16 accum (-16 regs, simpler change), measure
performance and accuracy. If acceptable, try full FP16 accum for both phases.

### For Fused-MLP Worker

**Priority: MEDIUM.** Only GEMM1 benefits (GEMM2 uses cuBLAS).

Follow the GEMM worker approach for GEMM1. Since GEMM1 is ~50% of the fused kernel
time, a 1.5x MMA speedup for GEMM1 translates to ~25% overall improvement.

### Fragment Layout Considerations

**The m16n8k16 FP8 fragment layout is the SAME as BF16 m16n8k16.** This is confirmed
by gau-nernst's implementation which uses the same shape selection for FP8+FP16 accum
as for BF16+FP32 accum.

Implications for our kernels:
- `ldmatrix_x4_mma` (with a1/a2 swap) should work for A operand loading
- `ldmatrix_x2_trans` should work for B operand loading
- The BF16->FP8 conversion pipeline stays the same
- But we feed K=16 values per MMA instead of K=32

**Important: empirically verify the a1/a2 swap is still required for m16n8k16 with
FP8 inputs.** The swap was validated for m16n8k16 with BF16 and m16n8k32 with FP8,
but not for m16n8k16 with FP8. It SHOULD be the same since the shape is m16n8k16,
but test before committing to the approach.

---

## 6. Risks and Mitigations

### Risk 1: FP16 Overflow in Accumulation

**Severity: HIGH for GEMM, MEDIUM for attention PV, LOW for attention QK^T with promoted accum**

FP8 e4m3 max value is 448. Product max is 200,704 > 65504 (FP16 max).
Even a single FP8 multiply can overflow FP16.

**Mitigation:** Never accumulate more than one MMA's worth of results in FP16.
Use promoted accumulation (FP16 MMA output -> immediate FP32 promotion). This is
zero-risk for overflow since each m16n8k16 MMA accumulates only K=16 products,
and the per-thread accumulation is a subset of these.

Actually, let's verify: for m16n8k16, each thread computes one element of the 16x8
output tile. That element is a dot product of a 16-element row of A with a 16-element
column of B. Max value: 16 * 448 * 448 = 3,211,264.

**This is 49x larger than FP16 max (65504). Overflow WILL occur even within a single
MMA if inputs are near max.** However, real inputs (BF16 activations converted to FP8)
rarely approach the FP8 max of 448. Typical BF16 values in transformer layers are
in [-2, 2] range, which maps to FP8 values in [-2, 2]. Products: 16 * 2 * 2 = 64.
Well within FP16 range.

**For safety:** Use promoted accumulation AND verify that input magnitudes stay
reasonable. If any input exceeds ~64 (sqrt(65504/16)), overflow is possible.

### Risk 2: Precision Loss in Long Accumulations

**Severity: MEDIUM for attention (N=2048, 32 KV blocks), LOW for GEMM (bounded by K tile)**

FP16 has 10-bit mantissa (~3.3 decimal digits). After 32 accumulations, relative error
is approximately 32 * 2^-11 = 1.6%. For attention outputs that feed into layer norm
and further computation, this is likely acceptable for inference.

For training: accumulated rounding errors in attention outputs create structured noise
in gradients. SageAttention2++ reports 99.97% cosine similarity with two-level
accumulation, suggesting this is manageable.

### Risk 3: cuBLAS Baseline Shift

**Severity: HIGH for GEMM "vs cuBLAS" metrics**

cuBLAS on consumer Blackwell runs at "full speed with FP16 accumulation, half speed
with FP32 accumulation." This means cuBLAS likely already uses FP16 accumulators.
Our current 1.29x advantage was measured with OUR FP32 accum vs cuBLAS's potentially
FP16 accum. Switching our kernels to FP16 accum levels the playing field but may not
increase the gap.

**Must re-benchmark cuBLAS to establish the true baseline** before and after switching
to FP16 accumulators.

### Risk 4: CUDA 13.0 Compatibility

**Severity: LOW (likely works)**

PTX ISA 8.7 (CUDA 12.8) introduced FP16 accum for FP8 MMA. Our CUDA 13.0 (PTX ISA 9.1)
postdates this. The instruction should work. However, sm_120 has surprised us before
(e.g., `cvt.e4m3x2.bf16x2` fails despite being documented for sm_89+).

**Test empirically before committing.** Write a minimal test kernel with the new MMA
instruction, compile with `-arch=sm_120`, and verify it produces correct results.

---

## 7. Recommended Experiment Plan

### Phase 1: Microbenchmark (1 hour)

Write a minimal kernel that:
1. Fills shared memory with known FP8 values
2. Executes 1000 back-to-back `mma.sync.m16n8k16.f16.e4m3.e4m3.f16` instructions
3. Measures cycles per MMA
4. Compares with 1000 `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`
5. Verifies correctness against a reference computation

This establishes whether the instruction actually works on sm_120 with CUDA 13.0
and measures the raw throughput gain.

### Phase 2: GEMM Integration (2-4 hours)

1. Modify fp8_gemm_sm120.cu to use m16n8k16 f16 accumulator with promoted accumulation
2. Adjust K-loop to iterate 2x more
3. Benchmark at 4096^3 vs both the old kernel AND cuBLAS 13.0
4. Measure register usage with ncu
5. If register savings allow, try wider tiles (64x256 or 128x128 with 2 blocks)

### Phase 3: Attention Integration (2-4 hours)

1. Modify PV multiplication only to use m16n8k16 f16 accumulator
2. Keep QK^T and softmax unchanged (FP32)
3. Validate correctness against FP64 reference
4. Benchmark at primary config (B=2 H=8 N=2048 D=64)
5. Check register count -- target is 149 or lower

### Phase 4: Full FP16 Attention (if Phase 3 succeeds)

1. Switch QK^T to m16n8k16 f16 accumulator with promoted accumulation
2. Validate softmax stability
3. Benchmark and profile with ncu
4. If register savings allow, try 4 blocks/SM

---

## 8. Correction to Previous Brief

The earlier brief (`all_fp8_fp16_accumulation_half_speed_discovery.md`) showed the
instruction as `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`. **This is
incorrect.** PTX ISA does NOT support m16n8k32 with f16 accumulator for FP8.

The correct instruction is:
```
mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16
```

The shape is m16n8k**16**, not k32. This is a critical distinction:
- k16 means half the K per MMA, requiring 2x more MMA instructions
- k16 means different A operand fragment layout (m16n8k16 pattern, not m16n8k32)
- k16 means the fragment layout matches BF16 MMA, not the current FP8 MMA

The previous brief's throughput numbers and register analysis are still correct.
The shape and code changes are different.
