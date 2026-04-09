# QV-8 Agent State

## Current Status
- Phase: development / algo_building → ready for validation
- Iteration: 0 (initial implementation)
- Build: PASS
- Tests: PASS (max_err ~1e-8 vs PyTorch reference)
- Bench: 3539x vs PyTorch reference (1000 circuits, 0.042ms)

## Architecture
- Fused CUDA kernel: one block per circuit, 256 threads, state in shared memory
- Each thread owns one amplitude of the 2^8 statevector
- 2-qubit SU(4) gates applied via complex matrix-vector product per thread
- Two __syncthreads per gate (read barrier + write barrier)
- Returns probability vector per circuit (|amplitude|^2)
- Batched: processes up to 10K+ circuits in a single launch

## Benchmark Results (initial)
| Config | Custom (ms) | Ref (ms) | Speedup |
|--------|-------------|----------|---------|
| 1000 circuits (primary) | 0.042 | 146.9 | 3539x |
| 100 circuits | 0.054 | 144.6 | 2699x |
| 10000 circuits | 0.345 | 156.1 | 452x |

## Experiment Log
| # | Change | Result | Decision |
|---|--------|--------|----------|
| 0 | Initial fused kernel | 3539x vs ref, all tests pass | KEEP |
