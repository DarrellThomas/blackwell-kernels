# CUDA C Best Practices — Kernel Optimization Reference

Source: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

## Global Memory Coalescing
- Concurrent warp accesses coalesce into 32-byte segment transactions
- Sequential aligned: full throughput; misaligned: ~90% (L1/L2 helps)
- Stride-2: 50% efficiency; high stride: degrades to 32 transactions/warp
- `cudaMalloc` guarantees 256-byte alignment
- Thread block sizes must be multiples of warp size (32)

## Shared Memory Usage Pattern
- Load coalesced from global → rearrange in shared for non-unit-stride access
- Reduces global traffic from O(M×N×K) to O(M×N + N×K) for matmul-style kernels

## L2 Cache Persistence (sm_80+)
- Set aside L2 for frequently-accessed data via `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)`
- Configure per-kernel access policy: `base_ptr`, `num_bytes`, `hitRatio`, `Persisting`/`Streaming`
- Up to 50% perf gain when persistent data fits; ~10% penalty if thrashing
- Tune `hitRatio` proportionally when data exceeds set-aside size

## Occupancy
- Ratio of active warps to max warps per SM
- Limited by: shared memory/block, registers/thread, block count
- Register pressure is the most common limiter — each thread holds registers for entire kernel
- 128–512 threads per block; grid size at least 2× SM count

## Instruction Optimization
- Division/modulo slower than multiply; use bit shifts for power-of-2
- `x % n` → `x & (n-1)` when n is power of 2
- Unroll small loops (2–4 iterations)
- `__sinf()`, `__cosf()`, `__expf()`: ~2-3× faster than standard, 2-3 ULP error

## Warp Divergence
- All threads in warp execute both branches (inactive masked)
- Worst case: 2× execution time
- Prefer simple if-statements over loops for small conditionals

## Data Transfer (Host ↔ Device)
- PCIe x16 Gen3: ~16 GB/s vs device memory: ~900 GB/s — minimize transfers
- Pinned memory (`cudaHostAlloc`): ~12 GB/s vs ~6 GB/s pageable
- `cudaMemcpyAsync` + non-default streams → overlap transfers with compute
- Two copy engines: overlap H→D, D→H, and kernel simultaneously
- Zero copy (`cudaHostAllocMapped`): beneficial on integrated GPUs or single-access coalesced patterns

## Profiling
- Effective bandwidth = (bytes_read + bytes_written) / 10^9 / time_seconds
- Load/Store Efficiency: target >80%
- Use CUDA events for GPU-clock timing (~0.5μs resolution)
