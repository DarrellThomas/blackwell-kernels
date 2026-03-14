@01_UNIVERSAL_PRINCIPLES.md
@03_FUSED_MLP_SPECIFICATION.md
@04_HARD_WON_LESSONS.md
@../docs/math_throttle_optimization.md
@../docs/reference_spatters_mma_matmul.md
@../docs/fused_mlp_agent_state.md

# blackwell-kernels — Fused MLP Branch

## What This Project Is
Custom CUDA kernels for the RTX 5090 (sm_120 / consumer Blackwell). The GPU uses `mma.sync` tensor core ISA (NOT `tcgen05` like datacenter Blackwell). We build our own optimized kernels.

**This branch focuses on fused MLP kernels** — combining two linear layers with an activation function into a single kernel to eliminate the intermediate global memory round-trip.

## Prior Art in This Repo
- **BF16 GEMM:** 0.97x cuBLAS at 4096³ (64×64 tiles, 4 warps, 6 blocks/SM)
- **FP8 GEMM:** 1.00-1.29x cuBLAS, dual-dispatch 64×64 / 128×128 tiles
- **Flash Attention BF16:** up to 1.77x cuDNN SDPA
- **Flash Attention FP8:** up to 2.79x cuDNN SDPA

The fused MLP kernel reuses GEMM building blocks (cp.async, swizzle, ldmatrix, MMA) but adds activation fusion and potentially a two-phase matmul structure.

## Build & Test

```bash
# Build (MUST use CUDA 13, system nvcc is 12.8)
CUDA_HOME=/usr/local/cuda-13 python3 setup.py build_ext --inplace

# Test
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_mlp.py

# Benchmark
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 benchmarks/bench_mlp.py

# Standalone CUDA tests
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda-13/bin/nvcc -O2 -gencode arch=compute_120,code=sm_120 -o tests/test_foo tests/test_foo.cu && CUDA_VISIBLE_DEVICES=1 ./tests/test_foo
```

- **ALWAYS** use `CUDA_VISIBLE_DEVICES=1` (GPU 0 has ComfyUI)
- **ALWAYS** use `CUDA_HOME=/usr/local/cuda-13` for builds

## Critical MMA Register Layout (Empirically Verified on sm_120)

### ldmatrix_x4 → MMA a1/a2 SWAP
ldmatrix_x4 outputs: r0=m0k0, r1=m0k1, r2=m1k0, r3=m1k1
MMA expects:         a0=m0k0, a1=m1k0, a2=m0k1, a3=m1k1
**Preferred: use `ldmatrix_x4_mma()` which bakes swap into operand order `{%0, %2, %1, %3}`.**
Legacy: pass `(r0, r2, r1, r3)` to MMA if using plain `ldmatrix_x4`.

### ldmatrix_x4 addressing (ALT mapping)
```
sub = lane_id / 8;
row = (sub/2)*8 + lane_id%8;
col = (sub%2)*8;
```

### ldmatrix_x2_trans for B (GEMM: A*B)
Gives B_col[k,n]=Bsrc[k,n] → computes A*B (not A*B^T). Correct for standard GEMM.

### D-fragment output
d0→D[T/4,(T%4)*2], d1→D[T/4,(T%4)*2+1], d2→D[T/4+8,(T%4)*2], d3→D[T/4+8,(T%4)*2+1]

## Key Lessons from GEMM (Apply Directly)

1. **Occupancy is king for compute-only loops.** 64×64 tiles, 4 warps, 6 blocks/SM beat 128×128 with 2 blocks/SM. 24 warps >> 16 warps.
2. **32 KB smem sweet spot.** Leaves 96 KB for L1 cache. Going above hurts more than it helps.
3. **Dual-dispatch for L2-bound problems.** 128×128 tiles when total data > 64MB or grid > 4096 blocks.
4. **FP8 is free performance.** `cvt.rn.satfinite.e4m3x2.f32` works on sm_120. 2x MMA throughput, ~3.7% relative error — network weights self-correct.
5. **No division on GPU.** Use power-of-2 tile sizes → bitwise shifts. Pad inputs to tile multiples in Python.
6. **Non-volatile MMA.** Always use `mma_m16n8k16_bf16_nv` / `mma_m16n8k32_e4m3_nv`. Let the compiler schedule.

## Project Conventions
- Copyright: Darrell Thomas, MIT License
- All `.cu` and `.py` files start with copyright header
- Kernel naming: `fused_mlp_sm120` (v1), `fused_mlp_v2_sm120` (v2), etc.
- Test naming: `tests/test_*.py` (Python), `tests/test_*.cu` (CUDA standalone)
