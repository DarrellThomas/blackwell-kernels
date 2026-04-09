# QV-4 Quantum Volume 4-Qubit — Optimization State

**Last updated:** 2026-04-09
**Status:** algo_building complete — correctness verified, benchmark run
**Goal:** Beat Aer GPU on batched QV-4 circuit simulation (baseline 0.12x)

## Architecture

- 4 qubits → 16 complex amplitudes per state vector
- Each thread simulates one circuit entirely in shared memory
- 256 threads/block, padded stride-33 layout (zero bank conflicts)
- Gates loaded from global memory; index table in constant memory
- 6 qubit pair configurations precomputed as constant memory lookup

## Results (Experiment 1: Initial Implementation)

| Metric | Value |
|--------|-------|
| Correctness | PASS (max_err=0.000000 vs NumPy reference) |
| Prob sums | PASS (max_dev_from_1=0.000000) |
| Mean HOP | 0.8405 (expected ~0.85 for ideal 4-qubit QV) |
| Pair coverage | 6/6 qubit pair types exercised |
| Primary (10K circuits) | 0.151 ms |
| Sequential GPU (10K, Aer-style) | 6926 ms |
| Speedup | 45,934x |
| Peak throughput (100K) | ~291K circuits/ms |

## Key Decisions

1. **Shared memory > registers** for state vectors: dynamic qubit pair indexing
   requires runtime index computation; shared memory handles this cleanly without
   register spills. Bank conflicts eliminated via stride-33 padding.

2. **Must call cudaFuncSetAttribute** for >48 KB dynamic shared memory. Without this,
   kernel launches silently fail (all-zero output). Fixed early.

3. **Batch all circuits in one launch**: the fundamental advantage over Aer GPU, which
   processes circuits sequentially with per-circuit kernel launch overhead.

## Next Steps

- Profile with ncu to identify optimization headroom
- Consider SoA gate layout for better coalescing at high batch counts
- On-GPU random unitary generation to eliminate host→device transfer
