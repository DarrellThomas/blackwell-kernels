@03_PROJECT_SPECIFICATION.md
@04_HARD_WON_LESSONS.md
@../docs/nvidia_blackwell_tuning_guide_sm120.md
@../docs/cuda_best_practices.md
@../docs/cuda_programming_guide.md
@../docs/blackwell_compatibility_guide.md
@../docs/math_throttle_optimization.md

# blackwell-kernels — Project Principles

## What This Project Is
Custom CUDA kernels for the RTX 5090 (sm_120 / consumer Blackwell). The GPU uses `mma.sync` tensor core ISA (NOT `tcgen05` like datacenter Blackwell). Flash Attention 3/4 will never work on this chip. We build our own.

## Build & Test

```bash
# Build (MUST use CUDA 13, system nvcc is 12.8)
CUDA_HOME=/usr/local/cuda-13 python3 setup.py build_ext --inplace

# Test
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_attention.py

# Benchmark
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 benchmarks/bench_attention.py

# Standalone CUDA tests
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda-13/bin/nvcc -O2 -gencode arch=compute_120,code=sm_120 -o tests/test_foo tests/test_foo.cu && CUDA_VISIBLE_DEVICES=1 ./tests/test_foo
```

- **ALWAYS** use `CUDA_VISIBLE_DEVICES=1` (GPU 0 has ComfyUI)
- **ALWAYS** use `CUDA_HOME=/usr/local/cuda-13` for builds

## Critical MMA Register Layout (Empirically Verified on sm_120)

### ldmatrix_x4 → MMA a1/a2 SWAP
ldmatrix_x4 outputs: r0=m0k0, r1=m0k1, r2=m1k0, r3=m1k1
MMA expects:         a0=m0k0, a1=m1k0, a2=m0k1, a3=m1k1
**Pass (r0, r2, r1, r3) to MMA — always swap r1 and r2.**

### ldmatrix_x4 addressing (ALT mapping)
```
sub = lane_id / 8;
row = (sub/2)*8 + lane_id%8;
col = (sub%2)*8;
```

### B-fragment for A*B^T (Q*K^T)
Load consecutive elements from same row: `b0={B[T/4,(T%4)*2..+1]}`, `b1={B[T/4,(T%4)*2+8..+9]}`

### ldmatrix_x2_trans for B (P*V)
Gives B_col[k,n]=Bsrc[k,n] → computes A*B (not A*B^T). Correct for P*V.

### D-fragment output
d0→D[T/4,(T%4)*2], d1→D[T/4,(T%4)*2+1], d2→D[T/4+8,(T%4)*2], d3→D[T/4+8,(T%4)*2+1]

## Project Conventions
- Copyright: Darrell Thomas, MIT License
- All `.cu` and `.py` files start with copyright header
- Kernel naming: `flash_attn_sm120` (v1 scalar), `flash_attn_v2_sm120` (v2 MMA)
- Test naming: `tests/test_*.py` (Python), `tests/test_*.cu` (CUDA standalone)
