# NVIDIA Blackwell Tuning Guide — sm_120 (RTX 5090) Extract

**Source:** https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
**Version:** 13.2 (retrieved 2026-03-12)
**Scope:** This document extracts the sm_120 (compute capability 12.0) specs relevant to kernel optimization on RTX 5090. Datacenter sm_100 (B200) specs are omitted.

---

## SM Resources (Compute Capability 12.0)

| Resource | Limit | Notes |
|----------|-------|-------|
| Max warps per SM | 48 | (sm_100 datacenter gets 64) |
| Max threads per SM | 1536 | 48 warps × 32 threads |
| Max thread blocks per SM | 32 | |
| Register file per SM | 64K × 32-bit | 256 KB total |
| Max registers per thread | 255 | |
| Shared memory per SM | 128 KB | Unified L1/texture/shared cache |
| **Max shared memory per block** | **99 KB** | CUDA reserves 1 KB per block |
| Static shared memory limit | 48 KB | Dynamic allocation required above this |
| Max threads per block | 1024 | |

### Occupancy Implications

With 128 KB shared/SM and 99 KB max/block:
- 1 block using 99 KB → 1 block/SM (low occupancy)
- 2 blocks using 64 KB each = 128 KB → fits (2 blocks/SM)
- 2 blocks using 55 KB each = 110 KB → fits (2 blocks/SM)
- 4 blocks using 32 KB each = 128 KB → fits (4 blocks/SM)
- 5 blocks using 25 KB each = 125 KB → fits (5 blocks/SM)

Register pressure also limits occupancy:
- 64K registers / 48 warps = 1365 registers per warp max
- At 255 regs/thread × 32 threads = 8160 regs/warp → only 7 warps (very low)
- At 128 regs/thread × 32 threads = 4096 regs/warp → 15 warps
- At 64 regs/thread × 32 threads = 2048 regs/warp → 31 warps
- Sweet spot: ~80-128 registers/thread for reasonable occupancy

### Dynamic Shared Memory

Static shared memory declarations (`__shared__`) are limited to 48 KB. To use more:
```cpp
// At launch time:
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
kernel<<<grid, block, dynamic_smem_bytes>>>();
```

### Unified Cache Carveout

The L1/shared memory split is configurable:
```cpp
cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    carveout_kb);
```
Supported carveout values (KB): 0, 8, 16, 32, 64, 100, 128.
Default gives 128 KB shared memory.

---

## Thread Block Clusters (New in Blackwell)

Thread Block Clusters allow thread blocks to share memory across blocks:
- Blocks in a cluster can read/write/atomics on each other's shared memory (Distributed Shared Memory)
- Max portable cluster size: 8 blocks
- Useful for: cross-block reductions, producer-consumer patterns, large tile cooperation

**Relevance to flash attention:** Could enable cross-block cooperation for very large sequence lengths or multi-query attention patterns. Not needed for current single-block kernel but worth exploring for future optimizations.

---

## Memory Hierarchy

| Level | Size | Bandwidth | Notes |
|-------|------|-----------|-------|
| Registers | 256 KB/SM | ~fastest | Per-thread, compiler-managed |
| Shared Memory | 128 KB/SM | ~19 TB/s | Programmer-managed, 32 banks |
| L1 Cache | Unified w/ shared | | Automatic for global loads |
| L2 Cache | 96 MB | | Shared across all SMs |
| GDDR7 | 32 GB | 1,792 GB/s | 512-bit bus |

### Bank Conflict Avoidance

Shared memory has 32 banks, 4-byte stride. Conflicts when multiple threads in a warp access different addresses in the same bank.

Mitigation strategies:
1. **XOR swizzle:** `addr ^ ((addr >> 4) & 0x1F)` — spreads accesses across banks
2. **Padding:** Add 1 element per row to shift column alignment
3. **Vectorized access:** 128-bit loads (`float4`, `int4`) access 4 consecutive banks

---

## Key Optimization Priorities for sm_120 Kernels

From the tuning guide's general recommendations, applied to sm_120:

1. **Maximize occupancy** — but note sm_120 has fewer warps (48 vs 64) so each block matters more
2. **Coalesced global memory access** — 128-byte cache lines, sequential threads should access sequential addresses
3. **Use `cp.async`** — overlap global→shared loads with compute
4. **Minimize shared memory bank conflicts** — use swizzle or padding
5. **Avoid warp divergence** — especially in causal masking and boundary conditions
6. **Use `mma.sync` efficiently** — keep tensor core pipe fed, avoid burst-then-idle patterns
7. **Register pressure management** — trade registers for occupancy when latency-hiding is sufficient
