# Fused MLP Kernel — Optimization State

**Last updated:** 2026-03-13
**Status:** Not started. Workspace prepared.
**Goal:** Fused MLP (GEMM1 + activation + GEMM2) beating 2× cuBLAS calls.

-----

## Hardware

- GPU: RTX 5090, sm_120 (consumer Blackwell, `mma.sync` ISA)
- Host: Threadripper PRO 7995WX, 512GB DDR5, Ubuntu 24.04
- CUDA 13 / PyTorch 2.10

-----

## Baseline (Not Yet Measured)

Two separate cuBLAS calls + PyTorch activation:
```python
hidden = torch.mm(X, W1)          # GEMM1
hidden = torch.relu(hidden) ** 2  # activation
Y = torch.mm(hidden, W2)          # GEMM2
```

Primary config: D=768, D_ff=3072, B*S=2048.

TODO: Measure baseline timing before writing the kernel.

-----

## Development Plan

1. **Phase 1 — Epilogue-fused GEMM:** GEMM1 + activation in one kernel. Still writes to global. Proves plumbing.
2. **Phase 2 — Full fusion:** GEMM1 + activation + GEMM2 in one kernel. Intermediate in smem only.
3. **Phase 3 — FP8 path:** Apply FP8 MMA (m16n8k32) to fused kernel.
4. **Phase 4 — SwiGLU:** Gated variant (two up-projections + element-wise multiply).

-----

## Starting Points (From GEMM Optimization)

These settings are empirically validated on sm_120 for GEMM:
- 64×64 tiles, BLOCK_K=32, 4 warps, `launch_bounds(128, 6)` → 6 blocks/SM
- 32 KB smem sweet spot (leaves 96 KB for L1 cache)
- cp.async double-buffer, XOR swizzle, ldmatrix_x4_mma, non-volatile MMA
- Python-side padding to tile multiples (zero boundary checks)
- FP8: `cvt.rn.satfinite.e4m3x2.f32` works on sm_120, reversed operand order
- Dual-dispatch: 128×128 tiles when data > 64MB or grid > 4096 blocks

-----

## Experiments

| # | Description | Duration (us) | vs Baseline | Notes |
|---|-------------|---------------|-------------|-------|
| — | (none yet) | — | — | — |

-----

## References

- [GEMM agent state](../docs/gemm_agent_state.md) — GEMM optimization results (in gemm branch)
- [spatters.ca MMA matmul](../docs/reference_spatters_mma_matmul.md) — 93% peak GEMM on Ada
- [math throttle guide](../docs/math_throttle_optimization.md) — compute-bound stall diagnosis
- [hard-won lessons](../.claude/04_HARD_WON_LESSONS.md) — empirical constraints from 60+ experiments
