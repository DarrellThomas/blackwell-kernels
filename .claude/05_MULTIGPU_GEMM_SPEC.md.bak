# Multi-GPU Tiled GEMM — Specification

## 1. Executive Summary

### Problem Statement

Large matrix multiplications that exceed a single GPU's 32GB GDDR7 memory — or that could benefit from parallel execution across multiple GPUs — have no optimized solution for consumer hardware. Existing multi-GPU GEMM implementations (NCCL, CUTLASS multi-GPU) target datacenter interconnects (NVLink, NVSwitch) and don't optimize for the PCIe-connected, heterogeneous setups that consumer users actually have.

### Solution

A tiled GEMM kernel where the large matrix resides in CPU (host) memory and tile blocks are streamed to multiple GPUs over PCIe. The kernel discovers the hardware topology at runtime, auto-tunes tile sizes and pipeline depth for the specific system, and partitions work proportionally across GPUs of potentially different capabilities.

### Why This Matters

This kernel would be genuinely novel. No consumer-oriented library does this today. The use cases are real:

- **Models that don't fit in one GPU** — a single 70B parameter model's weight matrices exceed 32GB in BF16
- **Parallel throughput** — two 5090s = 419 TFLOPS of BF16 compute, but only if you can keep both fed
- **Accessible HPC** — consumer multi-GPU setups (2-4 GPUs) are common among researchers, hobbyists, and small labs who can't afford or access datacenter hardware
- **Open-source auto-tuning** — users can run the autokernel optimization loop on *their* hardware to get a kernel tuned for their specific PCIe topology, GPU mix, and memory bandwidth

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Single-GPU overhead | <5% vs native on-GPU GEMM | Benchmark: tiled streaming vs cuBLAS for matrices that fit in VRAM |
| Multi-GPU scaling | >1.7x with 2 GPUs | Benchmark: 2-GPU vs 1-GPU on same problem |
| PCIe utilization | >80% of measured bandwidth | ncu + nsys trace of transfer overlap |
| Correctness | Bit-identical to single-GPU | Test against torch.mm reference |
| Asymmetric support | Proportional partition | Benchmark: mixed GPU configs |

## 2. Hardware Context

### Memory Hierarchy (Top to Bottom)

```
CPU DRAM          ~80 GB/s (DDR5-6000, dual channel)
    │
    PCIe 5.0 x16  ~64 GB/s per link (theoretical)
    │              ~55 GB/s (measured, typical)
    ├── GPU 0      32 GB GDDR7, 1792 GB/s internal
    └── GPU 1      32 GB GDDR7, 1792 GB/s internal
                   209.5 TFLOPS BF16 tensor each
```

### Key Ratios

| Transfer | Bandwidth | vs GPU Internal |
|----------|-----------|-----------------|
| CPU → GPU (PCIe 5.0 x16) | ~55 GB/s | 32x slower than GDDR7 |
| GPU internal (GDDR7) | 1,792 GB/s | — |
| GPU compute (BF16 tensor) | 209.5 TFLOPS | — |

### The Arithmetic Intensity Argument

For GEMM C = A × B where A is M×K and B is K×N:

- **Compute**: 2×M×N×K FLOPs
- **Transfer** (if streaming A row-tiles and B fully): M×K + K×N elements × 2 bytes = 2K(M+N) bytes

Arithmetic intensity of the *PCIe transfer* (not the GPU compute):

```
AI = 2×M×N×K / (2K(M+N)) = M×N / (M+N)
```

For square matrices (M=N):
- M=N=1024:  AI = 512 → 512 FLOPs per byte transferred
- M=N=4096:  AI = 2048 → 2048 FLOPs per byte transferred
- M=N=16384: AI = 8192 → 8192 FLOPs per byte transferred

At 55 GB/s PCIe and 209.5 TFLOPS compute, the balance point is:
```
209.5e12 / 55e9 = 3,809 FLOPs/byte
```

**Matrices larger than ~4096×4096 are compute-bound even over PCIe.** For these sizes, the PCIe transfer can be fully hidden by overlapping with compute. This is exactly the regime where this kernel shines.

### PCIe Topology Variations

Consumer systems vary. The kernel must handle:

| Config | GPUs | PCIe | Notes |
|--------|------|------|-------|
| Dual x16 | 2 identical | 2 × PCIe 5.0 x16 | Ideal case (our setup) |
| Split x16/x8 | 2 identical | x16 + x8 | Common on consumer boards |
| Mixed GPUs | 2 different | varies | e.g., 5090 + 4090 |
| Single GPU | 1 | x16 | Fallback: just stream tiles |
| Quad GPU | 4 | varies | Workstation/HEDT boards |

## 3. Architecture

### 3.1 Tiling Strategy

The core idea: partition the output matrix C into row blocks and assign blocks to GPUs.

```
        N columns
    ┌───────────────┐
    │   GPU 0 rows  │  ← rows 0..M/2-1
    │               │
M   ├───────────────┤
rows│   GPU 1 rows  │  ← rows M/2..M-1
    │               │
    └───────────────┘

Each GPU computes its row partition by streaming K-tiles:

  A_tile[m_start:m_end, k:k+BK] × B_tile[k:k+BK, :] → accumulate into C_partition
```

For each K-tile iteration:
1. Host sends A_tile (rows for this GPU) and B_tile (shared across GPUs) via pinned memory + cudaMemcpyAsync
2. GPU computes partial GEMM using the on-GPU tiled kernel
3. Accumulate into running result
4. Overlap: while GPU computes tile K, host transfers tile K+1

### 3.2 Pipeline Design

Double-buffered (minimum) or triple-buffered host↔device streaming:

```
Time →
         ┌──────────┐┌──────────┐┌──────────┐
Host→GPU │ Transfer1││ Transfer2││ Transfer3│  (CUDA stream 1: copies)
         └──────────┘└──────────┘└──────────┘
              ┌──────────┐┌──────────┐┌──────────┐
GPU compute   │ Compute0 ││ Compute1 ││ Compute2 │  (CUDA stream 2: kernels)
              └──────────┘└──────────┘└──────────┘
```

Each buffer holds one K-tile's worth of A and B data. While the kernel processes buffer 0, the host fills buffer 1 via a separate CUDA stream. cudaEvents synchronize the handoff.

### 3.3 Asymmetric Partitioning

When GPUs have different compute capabilities:

1. **Calibration** measures each GPU's effective TFLOPS (a quick GEMM microbenchmark)
2. **Partition** rows proportionally: if GPU 0 has 209 TFLOPS and GPU 1 has 165 TFLOPS, split ~56/44
3. **Straggler handling**: the faster GPU finishes first and waits. For deeply asymmetric configs, consider dynamic work stealing (GPU that finishes its partition grabs rows from the other's queue)

### 3.4 Memory Layout

**Host side** (pinned memory):
- Matrix A: full M×K in pinned host memory (or memory-mapped for truly huge matrices)
- Matrix B: full K×N in pinned host memory
- Matrix C: full M×N in pinned host memory (results gathered here)

**Device side** (per GPU):
- Double-buffer for A tiles: 2 × (M_partition × BK × 2 bytes)
- Double-buffer for B tiles: 2 × (BK × N × 2 bytes) — or partitioned if N is also large
- Accumulator C partition: M_partition × N × 4 bytes (FP32 accumulator)
- Final C partition: M_partition × N × 2 bytes (BF16 output)

### 3.5 Component Stack

```
┌─────────────────────────────────────┐
│  Python API (torch-compatible)      │  ← drop-in replacement for torch.mm
├─────────────────────────────────────┤
│  Host Orchestrator (C++)            │  ← hardware discovery, partitioning,
│  - cudaGetDeviceProperties          │     stream management, tile scheduling
│  - PCIe bandwidth probe             │
│  - Calibration cache                │
├─────────────────────────────────────┤
│  Per-GPU Tile Kernel (CUDA)         │  ← our existing GEMM kernel, invoked
│  - mma.sync BF16                    │     per-tile with accumulation
│  - cp.async, double-buffer, swizzle │
├─────────────────────────────────────┤
│  Transfer Engine                    │  ← cudaMemcpyAsync on dedicated streams
│  - Pinned memory pools              │     per GPU, double/triple buffered
│  - Per-GPU copy streams             │
└─────────────────────────────────────┘
```

## 4. Runtime Hardware Discovery

At initialization (first call or explicit init), the kernel:

1. **Enumerate GPUs**: `cudaGetDeviceCount`, `cudaGetDeviceProperties` for each
   - SM count, compute capability, memory size, clock speed
   - Filter to compatible GPUs (sm_120+ or configurable)

2. **Measure PCIe bandwidth**: for each GPU, time a pinned→device transfer
   - Use a ~64MB transfer (large enough to saturate, small enough to be fast)
   - Measure both directions (H2D and D2H)
   - Store measured bandwidth, not theoretical

3. **Measure compute throughput**: for each GPU, run a small GEMM
   - ~1024×1024 is enough to measure effective TFLOPS
   - Accounts for thermal state, power limits, background load

4. **Build execution plan**:
   - Partition rows proportionally by compute throughput
   - Set tile size BK based on: min(GPU memory budget, PCIe sweet spot)
   - Set pipeline depth based on: compute_time / transfer_time ratio
   - Choose buffer count (2 or 3) based on ratio

5. **Cache results**: write to `~/.blackwell_kernels/hw_profile.json`
   - Invalidate if GPU config changes
   - User can force recalibration

## 5. Auto-Tuning with Autokernel

This is where the project's optimization loop becomes infrastructure for everyone.

### For our hardware (the reference implementation)

We optimize the inner tile kernel via the existing autokernel loop — this gives us the fastest possible on-GPU compute for our specific 5090s.

### For other users' hardware

Users clone the repo and run the autokernel optimization loop on their own machine. The loop:

1. Discovers their hardware (GPU model, PCIe config)
2. Profiles the inner tile kernel on their specific GPU
3. Optimizes tile sizes, pipeline depth, partition strategy for their topology
4. Produces a tuned kernel binary specific to their system

The source code, build system, and optimization infrastructure are all open. A user with a 4090 + 5090 gets a kernel tuned for their asymmetric setup. A user with 4× 3090s gets one tuned for quad-GPU PCIe 4.0.

### Tunable parameters

| Parameter | What it controls | Tuning range |
|-----------|-----------------|--------------|
| BK (K-tile size) | PCIe transfer granularity | 32–512 |
| BLOCK_M, BLOCK_N | On-GPU tile dimensions | 64–256 |
| Pipeline depth | Buffers in flight | 2–4 |
| Partition ratio | Row split across GPUs | calibrated |
| Stream count | Concurrent copy streams per GPU | 1–3 |
| Accumulator precision | FP32 vs BF16 intermediate | workload-dependent |

## 6. API Design

### Python (torch-compatible)

```python
from blackwell_kernels import multi_gpu_gemm, calibrate

# One-time calibration (optional — auto-runs on first call)
hw = calibrate()
# hw = { 'gpus': [...], 'pcie_bw': [...], 'tflops': [...], 'partition': [...] }

# Drop-in matrix multiply
# A and B can be CPU tensors (pinned) or GPU tensors
A = torch.randn(16384, 16384, dtype=torch.bfloat16)  # CPU tensor, 512 MB
B = torch.randn(16384, 16384, dtype=torch.bfloat16)  # CPU tensor, 512 MB

C = multi_gpu_gemm(A, B)
# C is a CPU tensor with the result (or GPU tensor if inputs were on GPU)

# Explicit GPU selection
C = multi_gpu_gemm(A, B, devices=[0, 1])

# Single-GPU streaming (matrix too large for VRAM, stream tiles)
C = multi_gpu_gemm(A, B, devices=[0])
```

### C++ (for integration)

```cpp
#include "blackwell_kernels/multi_gpu_gemm.h"

MultiGpuGemm gemm;  // discovers hardware, calibrates on first use
gemm.execute(A_ptr, B_ptr, C_ptr, M, N, K, {0, 1});  // GPU IDs
```

## 7. Development Phases

### Phase 1: Single-GPU Tile Streaming
- CPU-resident matrices, stream K-tiles to one GPU via pinned memory
- Validate: correctness matches torch.mm, overhead <5% vs on-GPU cuBLAS for VRAM-fitting matrices
- This is the foundation — get the pipelining right on one GPU first

### Phase 2: Dual-GPU Symmetric
- Row-partitioned output, both GPUs stream from same host matrices
- Validate: >1.7x scaling on our dual 5090 setup
- Optimize: pipeline depth, tile sizes, stream synchronization

### Phase 3: Hardware Discovery & Calibration
- Runtime GPU enumeration, bandwidth probing, compute measurement
- Execution plan generation, calibration caching
- Validate: correct behavior on single-GPU fallback, asymmetric sim (artificially throttle one GPU)

### Phase 4: Asymmetric & N-GPU Generalization
- Proportional partitioning, straggler handling
- Support 1–N GPUs with mixed capabilities
- Validate: simulated mixed configs, community testing

### Phase 5: Autokernel Integration
- The optimization loop can tune tile sizes and pipeline depth per-system
- Users run `/autokernel multigpu-gemm` and get a kernel tuned for their hardware

## 8. Testing Strategy

| Test | What it validates |
|------|-------------------|
| Correctness (small) | 128×128 result matches torch.mm exactly |
| Correctness (large) | 16384×16384 result within BF16 tolerance |
| Correctness (non-square) | M≠N≠K, odd sizes, non-tile-aligned dimensions |
| Overhead (fits in VRAM) | Streaming overhead <5% vs cuBLAS for 4096×4096 |
| Scaling (2 GPU) | >1.7x speedup vs single GPU |
| Bandwidth utilization | nsys trace shows >80% PCIe saturation during transfers |
| Pipeline overlap | nsys trace shows compute/transfer overlap (no gaps) |
| Fallback (1 GPU) | Works correctly with single GPU |
| Calibration | hw_profile.json generated, correct GPU enumeration |
| Large matrix | 32768×32768 (2 GB per matrix) — exceeds single GPU memory |

## 9. Constraints & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU memory bandwidth bottleneck | CPU can't feed two x16 links simultaneously at full speed | Measure actual bandwidth; pipeline depth compensates |
| NUMA effects | GPU on remote NUMA node gets lower PCIe bandwidth | Pin host memory to correct NUMA node |
| GPU thermal throttling | Sustained dual-GPU load may throttle clocks | Calibration measures sustained (not burst) throughput |
| Pinned memory limits | OS limits on pinned (non-pageable) memory | Use memory-mapped files for truly huge matrices; chunk pinning |
| Synchronization overhead | cudaEvent waits and stream sync add latency | Minimize sync points; batch tiles |
| Portability | Different CUDA versions, driver versions | Minimum: CUDA 13+, sm_120; test on sm_89 (4090) too |

## 10. What Makes This Different

Existing multi-GPU GEMM solutions (NCCL, CUTLASS, Megatron-LM) assume:
- NVLink or NVSwitch interconnect (300+ GB/s)
- Homogeneous datacenter GPUs
- The full matrix fits in aggregate GPU memory
- A distributed framework manages the partitioning

We assume:
- PCIe 5.0 (55-64 GB/s) — 5x slower than NVLink
- Potentially heterogeneous consumer GPUs
- Matrices that may not fit in any single GPU
- No framework — just a function call that works

The optimization strategy is fundamentally different: instead of minimizing communication (the datacenter approach, because NVLink is fast but collective overhead is high), we maximize *overlap* (the consumer approach, because PCIe is slow but predictable and the compute-to-transfer ratio is enormous for large matrices).
