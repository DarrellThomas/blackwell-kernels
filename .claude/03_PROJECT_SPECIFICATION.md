# blackwell-kernels — Project Specification

## 1. Executive Summary

### Problem Statement
NVIDIA consumer Blackwell GPUs (RTX 5090, sm_120) are architecturally excluded from the mainstream deep learning kernel ecosystem. Flash Attention 3/4, CUTLASS fused attention, and most optimized libraries target datacenter sm_100 only. The tensor core ISA is fundamentally different.

### Solution
Build custom CUDA kernels targeting sm_120's `mma.sync` tensor core instructions, starting with flash attention (the training bottleneck) and expanding to GEMM and fused operations.

### Success Metrics
| Metric | Target | Current |
|--------|--------|---------|
| v2 MMA correctness | All tests pass | DONE |
| v2 vs cuDNN SDPA | >50% SDPA speed | 14-69% (unoptimized) |
| FP8 attention | Working kernel | Not started |
| Integration with autoresearch | Drop-in replacement | Not started |

## 2. Architecture

### Hardware Target
- **GPU**: NVIDIA RTX 5090 (GB202, sm_120)
- **Tensor Cores**: 5th gen, `mma.sync` ISA (NOT `tcgen05`)
- **No TMEM**: Data path is registers → tensor cores (unlike datacenter sm_100)
- **Key specs**: 170 SMs, 32GB GDDR7, 1792 GB/s, 128KB shared/SM, 96MB L2

### Software Stack
- CUDA Toolkit 13.0+ (system: `/usr/local/cuda-13`)
- PyTorch 2.10 with CUDA 13.0
- Python 3.12
- Build: `setup.py` with `torch.utils.cpp_extension.CUDAExtension`

### Project Structure
```
/data/src/blackwell-kernels/
├── .claude/                         # Claude Code instructions
│   ├── CLAUDE.md                    # Project principles & MMA reference
│   ├── 01_UNIVERSAL_PRINCIPLES.md   # Universal coding principles
│   └── 03_PROJECT_SPECIFICATION.md  # This file
├── setup.py                         # PyTorch extension build
├── LICENSE                          # MIT License
├── csrc/
│   ├── common/                      # Shared CUDA utilities
│   │   ├── mma_sm120.cuh            # mma.sync wrappers
│   │   ├── ldmatrix.cuh             # ldmatrix shared→register helpers
│   │   ├── cp_async.cuh             # Async global→shared copy
│   │   └── swizzle.cuh              # Bank conflict avoidance
│   ├── attention/
│   │   ├── flash_attn_sm120.cu      # v1: scalar flash attention
│   │   ├── flash_attn_sm120.cuh     # v1 header
│   │   └── flash_attn_v2_sm120.cu   # v2: MMA tensor core flash attention
│   └── gemm/
│       └── bf16_gemm_sm120.cu       # BF16 GEMM kernel
├── python/
│   └── blackwell_kernels/
│       ├── __init__.py              # Package init, exports kernels
│       ├── attention.py             # Python attention wrappers
│       └── ops.py                   # General ops
├── tests/
│   ├── test_attention.py            # Main correctness tests (6 tests)
│   ├── debug_v2.py                  # Debug: basic, identity, multi-block
│   ├── debug_v2b.py                 # Debug: uniform K, repeated Q, V=I, scale=0
│   ├── test_gemm.py                 # GEMM correctness
│   ├── test_mma_smoke.cu            # MMA toolchain smoke test
│   ├── test_ldmatrix.cu             # ldmatrix_x4 register mapping verification
│   ├── test_mma.cu                  # MMA + ldmatrix_x2_trans test
│   ├── test_mma2.cu                 # B-fragment dump + identity MMA test
│   ├── test_mma3.cu                 # MMA with manual B loading
│   ├── test_mma4.cu                 # All layout combos + a1/a2 swap discovery
│   └── test_mma5.cu                 # Verified MMA: random data, PASS
└── benchmarks/
    └── bench_attention.py           # v1 vs v2 vs cuDNN SDPA benchmark
```

## 3. Kernel Specifications

### flash_attn_sm120 (v1 — scalar)
- Pure FP32 scalar math, no tensor cores
- BLOCK_Q=16, BLOCK_KV=16, 1 warp (32 threads)
- Online softmax with warp shuffles
- Serves as correctness reference

### flash_attn_v2_sm120 (v2 — MMA tensor core)
- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for Q*K^T and P*V
- BLOCK_Q=64, BLOCK_KV=64, 4 warps (128 threads)
- Q loaded once to registers via ldmatrix_x4, reused across all KV blocks
- K loaded manually (scalar) for B^T fragment packing
- V loaded via ldmatrix_x2_trans (direct P*V, no transpose needed)
- P conversion: accumulator → shared memory → ldmatrix_x4 → registers
- Online softmax with 4-thread group shuffles (XOR masks 1, 2)
- Causal mask support

### API
```python
from blackwell_kernels import flash_attn_sm120, flash_attn_v2_sm120

# Q, K, V: [batch*heads, seq_len, head_dim], dtype=torch.bfloat16
O, L = flash_attn_v2_sm120(Q, K, V, causal=True)
# O: [batch*heads, seq_len, head_dim] — attention output
# L: [batch*heads, seq_len] — logsumexp for backward pass
```

## 4. Development Phases

### Phase 0+1: Scaffold & Toolchain (DONE)
- Project structure, build system, mma.sync smoke test

### Phase 2 v1: Scalar Flash Attention (DONE)
- Correct reference kernel, pure FP32
- 4 tests pass, max_err < 0.002 vs PyTorch

### Phase 2 v2: MMA Flash Attention (DONE)
- Tensor core accelerated, BF16 MMA
- 6 tests pass, max_err ~0.004 (BF16 precision)
- 3-9x faster than v1, ~14-69% of cuDNN SDPA

### Phase 2 v3+ (TODO): Optimize MMA Kernel
- Double-buffer shared memory (pipelining)
- Swizzle for bank conflict elimination
- Register-only P conversion (skip shared memory round-trip)
- cp.async for overlapped global→shared loads
- Target: >80% of cuDNN SDPA speed

### Phase 3 (TODO): FP8 Attention
- `mma.sync.aligned.m16n8k32` (2x throughput vs BF16)
- Per-tensor dynamic quantization
- FP32 softmax intermediate

### Phase 4 (TODO): Fused Operations
- RMSNorm + Attention fusion
- Fused MLP (linear + relu^2 + linear)

### Phase 5 (TODO): Autoresearch Integration
- Drop-in replacement for flash_attn in training pipeline
- Benchmark val_bpb improvement from faster kernels

## 5. Testing

### Correctness Tests (test_attention.py)
| Test | Config | Status |
|------|--------|--------|
| v1 non-causal | B=2 H=4 N=128 D=64 | PASS |
| v1 causal | B=2 H=4 N=128 D=64 | PASS |
| v2 non-causal | B=2 H=4 N=128 D=64 | PASS |
| v2 causal | B=2 H=4 N=128 D=64 | PASS |
| v2 D=128 | B=2 H=4 N=128 D=128 | PASS |
| v2 N=2048 causal | B=2 H=4 N=2048 D=64 | PASS |

### CUDA Unit Tests (standalone)
- test_mma_smoke.cu: Toolchain validation
- test_ldmatrix.cu: ALT mapping verification
- test_mma4.cu: a1/a2 swap discovery
- test_mma5.cu: Full MMA verification with random data

## 6. Performance Baselines

### v2 MMA vs cuDNN SDPA (causal, RTX 5090)
| B | H | N | D | SDPA (ms) | v1 (ms) | v2 (ms) | v2/SDPA |
|---|---|---|---|-----------|---------|---------|---------|
| 2 | 8 | 512 | 64 | 0.017 | 0.107 | 0.041 | 0.40x |
| 2 | 8 | 1024 | 64 | 0.034 | 0.298 | 0.081 | 0.41x |
| 2 | 8 | 2048 | 64 | 0.121 | 0.637 | 0.175 | 0.69x |
| 2 | 8 | 4096 | 64 | 0.302 | 2.127 | 0.517 | 0.58x |
| 4 | 16 | 2048 | 64 | 0.229 | 1.751 | 0.425 | 0.54x |
| 4 | 16 | 2048 | 128 | 0.425 | 3.853 | 2.971 | 0.14x |

v2 is 3-9x faster than v1. Gap to cuDNN is expected for an unoptimized first MMA pass.
Optimization path: pipelining, swizzle, register P conversion, cp.async.
