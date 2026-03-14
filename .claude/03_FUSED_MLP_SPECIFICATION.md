# Fused MLP Kernel — Project Specification

## 1. Executive Summary

### Problem Statement
In a transformer forward pass, the MLP block executes two back-to-back GEMMs with an activation function between them. Standard implementations write the intermediate tensor (often 4x the model dimension) to global memory, then read it back. This round-trip wastes memory bandwidth and is the dominant non-attention cost in transformer training/inference.

### Solution
Build a fused MLP kernel that keeps the intermediate activations in shared memory or registers, eliminating the global memory round-trip. Two matmuls + activation in one kernel launch.

### The Operation
```
Standard MLP:     Y = W2 @ activation(W1 @ X)
SwiGLU variant:   Y = W_down @ (SiLU(W_gate @ X) * (W_up @ X))
```

Where:
- X: [B*S, D] — input (batch × sequence, model dim)
- W1: [D, D_ff] — up-projection (D_ff typically = 4*D or 8/3*D for SwiGLU)
- W2: [D_ff, D] — down-projection
- Intermediate: [B*S, D_ff] — this is what we avoid writing to global memory

### Success Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| v1 correctness | All tests pass | max_err vs PyTorch reference |
| v1 vs separate GEMMs | >1.3x faster | Time(fused) vs Time(GEMM1) + Time(GEMM2) |
| FP8 path | Working | Same as BF16 but with FP8 MMA |
| vs cuBLAS two-call | >1.2x faster | Time(fused) vs Time(cublas×2) |

## 2. Architecture

### Why Fusion Wins

Two separate GEMMs for MLP with D=768, D_ff=3072, B*S=2048:
- GEMM1 output: [2048, 3072] = 12MB written to global memory
- GEMM2 input: [2048, 3072] = 12MB read from global memory
- Wasted: 24MB of DRAM bandwidth (write + read)

Fused kernel: intermediate stays in smem/registers. Zero global memory for the intermediate.

At 1792 GB/s DRAM bandwidth, 24MB = 13.4 μs of pure memory time saved. For a kernel running ~100-200 μs, that's 7-13% free.

### Fusion Strategy

The key insight: tile the second GEMM's K-dimension (D_ff) to match the first GEMM's N-dimension output. Each outer K-tile:

```
for k_tile in range(0, D_ff, BLOCK_K_OUTER):
    # Phase 1: GEMM1 — compute a slice of the intermediate
    # intermediate[BLOCK_M, BLOCK_K_OUTER] = X[BLOCK_M, D] @ W1[D, k_tile:k_tile+BLOCK_K_OUTER]
    # This is a full inner GEMM reducing over D

    # Phase 2: Activation — in registers or smem
    # activated = relu_sq(intermediate)  or  silu(intermediate)

    # Phase 3: GEMM2 — accumulate into output
    # Y[BLOCK_M, BLOCK_N] += activated[BLOCK_M, BLOCK_K_OUTER] @ W2[k_tile:k_tile+BLOCK_K_OUTER, BLOCK_N]
    # This is one K-tile of the outer accumulation
```

### Development Phases

#### Phase 1: Epilogue-Fused GEMM (simplest, proves the plumbing)
- Standard GEMM1 kernel with activation applied to accumulators before writing output
- Still writes intermediate to global memory, but no separate activation kernel launch
- Validates: build system, Python bindings, test infrastructure, activation math

#### Phase 2: Prologue-Fused GEMM
- Standard GEMM2 kernel that loads intermediate, applies activation, then does matmul
- Still reads intermediate from global memory
- Validates: activation-before-MMA path, data layout compatibility

#### Phase 3: Full Fusion (the real kernel)
- Single kernel: GEMM1 + activation + GEMM2, no intermediate in global memory
- Outer loop over D_ff tiles, inner loops over D (GEMM1) and D (GEMM2)
- Intermediate lives in shared memory between phases

#### Phase 4: FP8 Fusion
- Apply FP8 path to the fused kernel
- BF16 inputs → FP8 conversion → m16n8k32 MMA for both phases
- FP32 accumulators throughout, BF16 output

### Tile Configuration (Starting Point)

Based on GEMM learnings:
- **BLOCK_M:** 64 (output rows per block)
- **BLOCK_N:** 64 (output columns per block — this is D for the second GEMM)
- **BLOCK_K_OUTER:** 32-64 (intermediate dimension tile — D_ff slices)
- **BLOCK_K_INNER:** 32 (reduction dimension for each GEMM phase)
- **Warps:** 4 (128 threads), target 6 blocks/SM

Shared memory budget per block:
- GEMM1 needs: X tile [BLOCK_M, BLOCK_K_INNER] + W1 tile [BLOCK_K_INNER, BLOCK_K_OUTER]
- Intermediate: [BLOCK_M, BLOCK_K_OUTER] (activation output, stays in smem)
- GEMM2 needs: W2 tile [BLOCK_K_OUTER, BLOCK_N] (intermediate is already in smem)
- With BLOCK_M=64, BLOCK_K_OUTER=32, BLOCK_K_INNER=32, BLOCK_N=64:
  - X tile: 64×32×2 = 4 KB
  - W1 tile: 32×32×2 = 2 KB
  - Intermediate: 64×32×2 = 4 KB
  - W2 tile: 32×64×2 = 4 KB
  - Double-buffer overhead: ~2x for some tiles
  - Total: ~28 KB — fits in the 32 KB sweet spot

## 3. Activation Functions

### ReLU² (ReGLU squared)
```c
float relu_sq(float x) { float r = fmaxf(x, 0.0f); return r * r; }
```
Simplest. Pure register operation. No transcendentals.

### SiLU (Swish) — used in SwiGLU
```c
float silu(float x) { return x / (1.0f + expf(-x)); }
// With fast math: x * __frcp_rn(1.0f + __expf(-x))
```
One `expf` + one `rcp`. Under `--use_fast_math`, maps to MUFU instructions.

### GELU (approximate)
```c
float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
```
More expensive: `tanhf` is a MUFU instruction but has ~8-cycle latency.

### SwiGLU (gated)
```c
// Requires TWO up-projections: W_gate and W_up
// gate = silu(X @ W_gate)
// up   = X @ W_up
// output = gate * up
```
This changes the fusion structure — GEMM1 produces TWO outputs (gate and up), then they're element-wise multiplied before GEMM2. W1 is effectively [D, 2*D_ff] split into W_gate and W_up.

**Start with ReLU² for simplicity. Add SiLU/SwiGLU after the architecture is proven.**

## 4. Reference Dimensions (Common Transformer Configs)

| Model | D | D_ff | D_ff/D | Activation |
|-------|---|------|--------|------------|
| GPT-2 Small | 768 | 3072 | 4x | GELU |
| LLaMA 7B | 4096 | 11008 | 2.69x | SwiGLU |
| LLaMA 13B | 5120 | 13824 | 2.70x | SwiGLU |
| Mistral 7B | 4096 | 14336 | 3.5x | SwiGLU |

Primary benchmark config: **D=768, D_ff=3072, B*S=2048** (GPT-2 scale).
Secondary: **D=4096, D_ff=11008, B*S=2048** (LLaMA scale).

## 5. File Structure

```
csrc/
├── common/               # Shared (DO NOT MODIFY on this branch)
│   ├── mma_sm120.cuh
│   ├── ldmatrix.cuh
│   ├── cp_async.cuh
│   ├── swizzle.cuh
│   └── fp8_convert.cuh
└── fused/
    └── fused_mlp_sm120.cu    # The fused MLP kernel

python/blackwell_kernels/
├── ops.py                    # Add fused_mlp wrapper
└── __init__.py               # Export fused_mlp_sm120

tests/
├── test_mlp.py               # Correctness vs PyTorch
└── test_fused_mlp.cu          # Standalone CUDA tests (optional)

benchmarks/
└── bench_mlp.py              # fused vs 2× torch.mm + activation

profiles/
└── profile_mlp.py            # Minimal launch for ncu
```

## 6. Testing

### Correctness Reference
```python
def reference_mlp(X, W1, W2, activation='relu_sq'):
    """PyTorch reference — two separate matmuls + activation."""
    hidden = X @ W1
    if activation == 'relu_sq':
        hidden = torch.relu(hidden) ** 2
    elif activation == 'silu':
        hidden = torch.nn.functional.silu(hidden)
    return hidden @ W2
```

### Test Configs
| Test | B*S | D | D_ff | Activation | Tolerance |
|------|-----|---|------|------------|-----------|
| Small square | 256 | 256 | 1024 | relu_sq | 5% rel |
| GPT-2 scale | 2048 | 768 | 3072 | relu_sq | 5% rel |
| Non-aligned | 200 | 300 | 1200 | relu_sq | 5% rel |
| FP8 (when ready) | 2048 | 768 | 3072 | relu_sq | 10% rel |

## 7. Constraints

- **DO NOT modify `csrc/common/`** — shared code changes need manual promotion to main
- **ALWAYS** use `CUDA_VISIBLE_DEVICES=1` (GPU 0 has ComfyUI)
- **ALWAYS** use `CUDA_HOME=/usr/local/cuda-13` for builds
- sm_120 uses `mma.sync`, NOT `tcgen05` — no TMEM, no wgmma
- Pad all dimensions to tile multiples in the Python wrapper
- No division in kernel hot paths — power-of-2 tile sizes only
- FP32 accumulators always — never accumulate in reduced precision
