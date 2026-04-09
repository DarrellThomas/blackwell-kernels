# [Kernel Name] — Optimization State

**Last updated:** —
**Status:** Not started
**Goal:** [e.g., Beat cuBLAS on primary config]

-----

## Hardware

- GPU: RTX 5090, sm_120 (consumer Blackwell, `mma.sync` ISA)
- Host: Threadripper PRO 7995WX, 512GB DDR5, Ubuntu 24.04
- CUDA 13 / PyTorch 2.10

-----

## Baseline

[Measure reference implementation performance here before optimizing]

| Config | Reference (ms) | Notes |
|--------|---------------|-------|
| Primary | — | — |

-----

## Experiments

| # | Description | Config | Kernel (ms) | Baseline (ms) | Speedup | Notes |
|---|-------------|--------|-------------|---------------|---------|-------|
<!-- Fill in as you run experiments -->

-----

## Key Architectural Decisions

[Document load-bearing decisions here — things that must not be reverted]

-----

## Dead Ends

[Document approaches that were tried and failed, with root cause]

-----

## References

- [hard-won lessons](../.claude/04_HARD_WON_LESSONS.md) — empirical constraints
- [spatters.ca MMA matmul](../docs/reference_spatters_mma_matmul.md) — GEMM reference
- [math throttle guide](../docs/math_throttle_optimization.md) — compute-bound stalls
