# FP8 MMA with FP16 Accumulation: 1.49x Speedup on sm_120

**Source:** https://gau-nernst.github.io/nvrtc-matmul/
**Relevant to:** attention worker, GEMM worker
**Worker's current problem:** Both workers are register-constrained with FP32 accumulators. Attention worker at 165 regs (5 spare before occupancy drops). GEMM worker at balanced profile with 67% SM throughput.
**Date:** 2026-03-14

## What This Is

gau-nernst demonstrated that FP8 MMA with FP16 accumulation achieves **692 TFLOPS
vs 465 TFLOPS** (1.49x speedup) for FP8+FP32 accumulation on the RTX 5090 at
M=N=K=8192. The B200 datacenter chip shows a full 2x (1929 vs 965 TFLOPS); on
sm_120 the gap is limited by power.

This is achieved by using the `mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16`
instruction, which uses **2 output registers instead of 4** per MMA tile.

## Why It Matters for Us

### Attention Worker (FP8)
The FP8 attention kernel is at 165 registers with a 170.7 threshold for 3 blocks/SM
(only 5 registers spare). The FP32 accumulator for the output matrix O takes 4
registers per 16x8 tile. With FP16 accumulation, this drops to 2 registers per tile.

For the attention kernel's output accumulator layout `[WARP_Q/MMA_M][DIM/MMA_N][4]`:
- With WARP_Q=16, DIM=64: that's `[1][8][4]` = 32 registers for O accumulators
- FP16 accumulation: `[1][8][2]` = 16 registers — **saving 16 registers**

Those 16 registers could be used for:
- More aggressive prefetch buffers
- Wider tiles (e.g., BLOCK_KV=128)
- Reducing spill pressure
- Or simply increasing occupancy margin from 5 to 21 spare registers

The P*V multiplication accumulates in FP16 instead of FP32. Since P values are
in [0,1] (softmax outputs) and V values are BF16, the FP16 accumulation may have
acceptable precision for inference. **Must validate numerically.**

### GEMM Worker (FP8)
The FP8 GEMM at 1.29x cuBLAS uses m16n8k32 with FP32 accumulators. Switching to
m16n8k16 with FP16 accumulators:
- Halves accumulator register count per tile
- Could enable 128x128 tiles at higher occupancy (currently launch_bounds(128,1))
- Or enable wider tiles (64x256) that increase compute/load ratio beyond 5.3
- Net TFLOPS improvement of ~1.49x suggests this DOES compensate for the 2x more
  instructions needed (k16 vs k32)

## Key Technique

### PTX Instruction

```ptx
// FP8 e4m3 with FP16 accumulation (NEW - PTX 8.7 / CUDA 12.8+)
mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16
    {%0, %1},           // D: 2x uint32 (4x f16 values packed)
    {%2, %3, %4, %5},   // A: 4x uint32 (FP8 fragments)
    {%6, %7},            // B: 2x uint32 (FP8 fragments)
    {%8, %9};            // C: 2x uint32 (4x f16 values packed)

// Compare: FP8 e4m3 with FP32 accumulation (current)
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    {%0, %1, %2, %3},   // D: 4x float
    {%4, %5, %6, %7},   // A: 4x uint32 (FP8 fragments)
    {%8, %9},            // B: 2x uint32 (FP8 fragments)
    {%10, %11, %12, %13}; // C: 4x float
```

### Register Comparison

| Component    | FP32 acc (current) | FP16 acc (new) | Savings |
|-------------|-------------------|----------------|---------|
| A operand   | 4 regs            | 4 regs         | 0       |
| B operand   | 2 regs            | 2 regs         | 0       |
| C/D accum   | 4 regs            | 2 regs         | **-2**  |
| K per instr | 32                | 16             | 2x instrs |

### Shape Change

**Critical:** The instruction shape changes from `m16n8k32` to `m16n8k16`:
- FP32 acc: each MMA processes K=32 elements → 16*8*32*2 = 8192 FLOPs/instr
- FP16 acc: each MMA processes K=16 elements → 16*8*16*2 = 4096 FLOPs/instr
- You need 2x more MMA instructions for the same K dimension
- BUT the hardware executes them faster (fewer register writes, simpler accumulator)
- Net throughput: 692 vs 465 TFLOPS = 1.49x faster despite 2x more instructions

### Performance Data (RTX 5090 @ 600W)

| Configuration      | TFLOPS | Notes |
|--------------------|--------|-------|
| FP8 + FP32 acc     | 465    | Current approach (m16n8k32) |
| FP8 + FP16 acc     | 692    | New approach (m16n8k16) |
| **Speedup**        | **1.49x** | Power-limited; B200 shows full 2x |
| MXFP8 (block-scaled) | ~680 | Slightly slower than plain FP8+FP16 |

### Requirements

- **CUDA 12.8+** (PTX ISA 8.7 adds `.f16` accumulator for FP8 types)
- Our setup: CUDA 13.0 — **already supported**
- Compile with `-arch=sm_120` (plain sm_120 is fine; sm_120a only needed for MXFP8)

## Caveats

1. **Precision:** FP16 has ~3.3 decimal digits (10 bits mantissa) vs FP32's ~7.2
   digits (23 bits). For GEMM with large K dimensions, accumulation error grows.
   This is mitigated by:
   - Periodic FP16→FP32 promotion and re-accumulation (e.g., every 128 K iterations)
   - For attention: P values are [0,1], V values are BF16, so products are small
   - For training: may need careful error analysis
   - cuBLAS likely uses FP32 acc internally, so our "vs cuBLAS" comparison changes

2. **Instruction count doubles:** 2x more MMA instructions for same K, but 1.49x
   net speedup shows hardware throughput improvement dominates.

3. **Fragment layout:** The A/B operand register count stays the same (4+2), but
   the K dimension per instruction halves. Need to adjust K-loop unrolling.

4. **Output conversion:** Final output in FP16 registers needs conversion to BF16
   for storage. This is a single `cvt` per value, negligible overhead.

5. **Softmax in attention:** The QK^T MMA produces scores that feed into softmax
   (exp2f, max, sum). If QK^T accumulates in FP16 instead of FP32, the softmax
   computation may need FP32 promotion for numerical stability. Consider:
   - QK^T MMA in FP16 acc → promote to FP32 for softmax → convert back for P*V
   - Or keep QK^T in FP32 acc, only use FP16 acc for P*V (saves half the accum regs)

## Suggested Experiments

### For Attention Worker
1. **P*V only with FP16 acc:** Keep QK^T at FP32 accumulation for softmax stability.
   Only switch the P*V multiplication to FP16 accumulation. This saves registers on
   the O accumulator without affecting softmax precision.
2. **Both QK^T and P*V with FP16 acc:** Maximum register savings but needs careful
   softmax validation. Promote QK^T FP16 results to FP32 before softmax operations.

### For GEMM Worker
1. **Direct substitution:** Replace m16n8k32/f32 with m16n8k16/f16. Adjust K-loop
   to do 2x iterations. Measure throughput and precision.
2. **Wider tiles:** Use register savings to try 64x256 or 128x128 tiles with higher
   compute/load ratio.
3. **Hybrid accumulation:** Accumulate in FP16 for inner K-loop, promote to FP32
   every N iterations to bound error.
