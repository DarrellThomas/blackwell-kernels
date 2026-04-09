# QV-8 Agent State

## Current Status
- Phase: rework → rework_complete
- Iteration: 1 (Python interface optimization)
- Build: PASS
- Tests: PASS (86/86 checks, max_err ~2e-8 vs PyTorch reference)
- Bench: 1.0ms end-to-end for 1000 circuits (GPU-native pipeline)

## Architecture
- Fused CUDA kernel: one block per circuit, 256 threads, state in shared memory
- Each thread owns one amplitude of the 2^8 statevector
- 2-qubit Haar-U(4) gates applied via complex matrix-vector product per thread
- Two __syncthreads per gate (read barrier + write barrier)
- Returns probability vector per circuit (|amplitude|^2)
- Batched: processes up to 10K+ circuits in a single launch
- GPU-native circuit generation: Gram-Schmidt U(4) in real arithmetic, no CPU roundtrip

## Python Interface
- `generate_qv8_circuits()`: Batched NumPy QR (10x faster than original Python loops)
- `generate_qv8_circuits_gpu()`: Fully GPU-native via Gram-Schmidt in real arithmetic (1000x faster)
- `qv8_simulate()`: CUDA kernel wrapper (unchanged)
- `qv8_simulate_ref()`: PyTorch reference (unchanged)

## Benchmark Results
### Kernel only (pre-generated data on GPU)
| Config | Custom (ms) | Ref (ms) | Speedup |
|--------|-------------|----------|---------|
| 1000 circuits (primary) | 0.044 | 10.2 | 233x |
| 100 circuits | 0.008 | 5.8 | 696x |
| 10000 circuits | 0.161 | 14.9 | 92x |

### End-to-end GPU pipeline (generate_qv8_circuits_gpu + qv8_simulate)
| Config | End-to-end (ms) | Throughput |
|--------|----------------|------------|
| 100 circuits | 1.005 | 99 circuits/ms |
| 1000 circuits | 1.006 | 994 circuits/ms |
| 10000 circuits | 1.059 | 9445 circuits/ms |

### Performance History
| Version | End-to-end 1000 circuits | Bottleneck |
|---------|------------------------|------------|
| v0 (Python loops) | ~981 ms | Circuit gen: 789ms in Python for-loops |
| v1 (batched NumPy) | ~47 ms | Circuit gen: 52ms in batched QR |
| v2 (GPU Gram-Schmidt) | 1.0 ms | Gram-Schmidt orthogonalization |

## Experiment Log
| # | Change | Result | Decision |
|---|--------|--------|----------|
| 0 | Initial fused kernel | 3539x vs ref, all tests pass | KEEP |
| 1 | Batched NumPy QR (replace Python loops) | 21x end-to-end speedup | KEEP |
| 2 | GPU Gram-Schmidt U(4) generation | 991x end-to-end speedup | KEEP |
