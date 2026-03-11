# v2 MMA Kernel Optimization Priorities

**Config:** B=2 H=8 N=2048 D=64, causal, RTX 5090

---

## Profile History

### Before cp.async (2026-03-11)

| Metric | Value | Notes |
|--------|-------|-------|
| Duration | 313 us | |
| Compute (SM) Throughput | 16.5% | Severely underutilized |
| Tensor Pipe Utilization | 16.5% | |
| Achieved Occupancy | 15.5% | |
| Long Scoreboard Stalls | 6.95 (62%) | #1 bottleneck |
| Math Pipe Throttle | 0.83 (7%) | |
| Smem Bank Conflicts (stores) | 61,034 | P conversion dominated |
| Smem Bank Conflicts (loads) | 8,123 | |
| v2/SDPA (D=64, N=2048) | 0.69x | |
| v2/SDPA (D=128, N=2048) | 0.14x | |

### After cp.async (2026-03-11)

| Metric | Value | Notes |
|--------|-------|-------|
| Duration | 123 us | **2.54x faster** |
| Compute (SM) Throughput | 44.5% | +170% |
| Tensor Pipe Utilization | 44.5% | +170% |
| Achieved Occupancy | 16.8% | ~same |
| Long Scoreboard Stalls | 1.59 (20%) | **-77%** — cp.async directly attacked this |
| Math Pipe Throttle | 2.28 (29%) | Now #1 stall — tensor cores saturated in bursts |
| Wait Stalls | 1.57 (20%) | cp_async_wait + barrier waits |
| Smem Bank Conflicts (stores) | 37,967 | -38% (P conversion still dominates) |
| Smem Bank Conflicts (loads) | 7,795 | ~same |
| v2/SDPA (D=64, N=2048) | **1.21x** | Now faster than cuDNN SDPA |
| v2/SDPA (D=128, N=2048) | **0.64x** | +357% improvement |

**Diagnosis: Compute-bursting.** Tensor cores are now the top stall (29% Math Pipe Throttle), meaning MMA instructions arrive in bursts during compute phases while load phases leave the pipe idle. The fix is to overlap loads with compute (double-buffering). Wait stalls (20%) are from cp_async_wait — also fixed by pipelining.

---

## Completed Optimizations

### ~~Priority 1: cp.async for Global→Shared Loads~~ ✓ DONE
**Actual speedup: 2.54x** (est. was 1.5-2x) — exceeded expectations.

Converted all three global→shared loads (Q, K, V) from scalar 2-byte register-mediated copies to 16-byte `cp.async.cg` with `cp_async_128_zfill` (zero-fills on OOB predicate). Long Scoreboard stalls dropped from 62% to 20%.

---

## Remaining Optimizations (re-prioritized after cp.async)

### Priority 2: Double-Buffer Shared Memory (Pipelining)
**Est. speedup: 1.3-1.8x** — Now attacks TWO top stalls: Math Pipe Throttle (29%) and Wait (20%).

Current flow per KV block:
```
[cp.async K] → [wait+sync] → [Compute Q*K^T] → [cp.async V] → [wait+sync] → [Compute P*V]
```

With double-buffering:
```
[cp.async K_0]
for i in 0..num_kv_blocks:
    [cp.async K_{i+1} into buf B]  ←── overlapped with compute
    [wait K_i in buf A]
    [Compute Q*K^T on buf A]       ←── overlapped with K_{i+1} load
    [swap A, B]
    [cp.async V_i into buf B]      ←── overlapped with softmax
    [wait V_i]
    [Compute P*V on buf A]         ←── overlapped with next loop's K load
```

Benefits:
- Spreads MMA instructions over time → reduces Math Pipe Throttle bursts
- Overlaps cp.async with compute → reduces Wait stalls
- Shared memory budget: 27.6 KB current → ~55 KB with double buffer (within 102.4 KB limit)

### Priority 3: Swizzle Shared Memory Addressing
**Est. speedup: 1.1-1.3x** — Eliminates 38K store bank conflicts + 8K load conflicts.

P conversion stores (FP32 accum → BF16 smem) are still the main offender at 38K conflicts. XOR-based swizzle maps consecutive thread addresses to different banks.

### Priority 4: Register-Only P Conversion
**Est. speedup: 1.1-1.2x** — Eliminates the shared memory round-trip for P.

Current P flow: FP32 accum registers → BF16 store to smem → ldmatrix_x4 from smem → MMA registers.

Possible: Convert FP32 → BF16 in registers, use `shfl.sync` to redistribute into MMA A-fragment layout. Eliminates P's smem stores (the 38K bank conflicts) and smem loads entirely.

### Priority 5: Larger Tiles (BLOCK_Q=128)
**Est. speedup: 1.2-1.5x** — More MMA ops per memory access.

gau-nernst's v5 kernel uses BLOCK_Q=128 and achieved 94.4% of peak TFLOPS. Larger Q tile = more compute per K/V load. Requires more registers and shared memory but acceptable if latency hidden by pipelining.

### Priority 6: Coalesced Global Stores for O
**Est. speedup: ~1.1x** — ncu flagged uncoalesced global stores (8.2/32 bytes utilized per sector).

Current O store writes 2 bf16 elements per thread in a scattered pattern (MMA accumulator layout). Could stage through shared memory for coalesced 128-byte writes.

---

## Target After All Optimizations

| Metric | After cp.async | Target |
|--------|---------------|--------|
| v2/SDPA (D=64, N=2048) | 1.21x | >1.5x |
| v2/SDPA (D=128, N=2048) | 0.64x | >0.85x |
| Tensor Pipe Utilization | 44.5% | >70% |
| Math Pipe Throttle | 29% | <15% |
| Wait Stalls | 20% | <10% |
| Smem Bank Conflicts (stores) | 38K | <1K |

Reference: gau-nernst achieved 94.4% of 209.5 TFLOPS peak. cuDNN SDPA is at 97.2%.
