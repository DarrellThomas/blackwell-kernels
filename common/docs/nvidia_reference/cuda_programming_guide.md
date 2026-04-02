# CUDA C++ Programming Guide — Key Technical Content

Source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
Fetched: 2026-03-27

## Memory Hierarchy

### Memory Spaces
- **Local Memory**: Private per-thread storage
- **Shared Memory**: Visible to all threads of the block, same lifetime as the block
- **Global Memory**: Accessible by all threads across all blocks
- **Constant Memory**: Read-only, persistent across kernel launches
- **Texture Memory**: Optimized read-only access with filtering capabilities
- **Distributed Shared Memory** (CC 9.0+): Thread blocks within a cluster can read/write peer block shared memory

## Thread Block Programming

### Thread Organization
- **Thread Blocks**: Up to 1024 threads per block, indexed via `threadIdx`
- **Grids**: Collections of thread blocks, indexed via `blockIdx`
- **Clusters** (CC 9.0+): Optional hierarchy, max 8 blocks portable

**Thread Index Calculation:**
- 1D: `threadIdx.x`
- 2D (Dx x Dy): `x + y*Dx`
- 3D (Dx x Dy x Dz): `x + y*Dx + z*Dx*Dy`

### Synchronization
```cuda
__syncthreads();  // Block-level barrier
```
Thread blocks must execute independently — any order, parallel or serial.

## Warp-Level Primitives

- **Vote**: `__ballot_sync(mask, predicate)` — warp-wide consensus
- **Shuffle**: `__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, `__shfl_down_sync` — direct thread-to-thread exchange
- **Match**: Identify matching values across warps
- **Reduce**: Warp-level aggregation
- **Sync**: `__syncwarp()` — explicit warp barrier

## Asynchronous Operations

### cp.async & memcpy_async
```cuda
cuda::memcpy_async(shared_dst, global_src, bytes, barrier);
```

**Pipeline Pattern:** Overlap data movement with computation via double/triple buffering.

**Requirements:**
- Data aligned to 16-byte boundaries
- Only trivially copyable (POD) types
- Keep commit/arrive-on operations converged within warp

### Asynchronous Barrier
- Arrival: Thread signals participation
- Countdown: Barrier counts down arrivals
- Completion: All threads synchronized
- Reset: Barrier resets for next phase

## Cooperative Groups API

**Group Types:** Thread Block, Cluster (CC 9.0+), Grid, Tile (partitioned subset)

**Collectives:** `sync()`, `reduce()`, `inclusive_scan()`, `exclusive_scan()`, `memcpy_async()`

## Tensor Core Programming (WMMA)

```cuda
wmma::fragment<wmma::matrix_a, M, N, K, type> a_frag;
wmma::fragment<wmma::matrix_b, M, N, K, type> b_frag;
wmma::fragment<wmma::accumulator, M, N, K, type> acc_frag;
wmma::load_matrix_sync(a_frag, ptr, lda);
wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
```

## Compute Capability — Blackwell (sm_120)

| Spec | CC 12.0 (sm_120) |
|------|-------------------|
| Max warps per SM | 48 |
| Max threads per SM | 1536 |
| Max threads per block | 1024 |
| 32-bit registers per SM | 64K |
| Max registers per thread | 255 |
| Max thread blocks per SM | 32 |
| Shared memory per SM | 128 KB |
| Max shared memory per block | 99 KB |

## Occupancy

**Occupancy** = active warps / max warps per SM

**Limiting factors:**
1. Registers per thread (64K / threads = max regs)
2. Shared memory per block
3. Thread block size
4. Hardware warp capacity (48 warps on sm_120)

Higher occupancy does NOT always mean higher performance. Low occupancy always reduces latency hiding ability.

## Performance Optimization

### Three Optimization Targets
1. **Maximize Utilization** — saturate GPU at application, device, and SM level
2. **Maximize Memory Throughput** — coalesce accesses, minimize host-device transfers
3. **Maximize Instruction Throughput** — minimize dependencies, hide latency

### Memory Coalescing
- Access patterns should be sequential within a warp
- 128-byte transaction granularity
- Misaligned or scattered accesses cause multiple transactions

### Key Principles
- Minimize host-device data transfers
- Ensure coalesced global memory access patterns
- Minimize redundant global memory accesses
- Avoid warp divergence
