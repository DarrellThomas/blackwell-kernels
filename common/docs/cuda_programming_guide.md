# CUDA C Programming Guide — Device-Side Reference

Source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## Thread Hierarchy
- **Thread** → **Block** (up to 1024 threads) → **Grid**
- Thread ID: 1D `threadIdx.x`; 2D `x + y*blockDim.x`
- Blocks execute independently in any order across SMs
- Warp = 32 threads; fundamental scheduling unit

## Memory Spaces

| Space | Scope | Lifetime | Use |
|-------|-------|----------|-----|
| Registers | Thread | Thread | Local variables |
| Local Memory | Thread | Thread | Register spill (backed by L1) |
| Shared Memory | Block | Block | Intra-block cooperation |
| Global Memory | All | App | Large datasets (L2 cached) |
| Constant Memory | All | App | Read-only broadcast data |

## Synchronization
- `__syncthreads()` — full block barrier
- `__syncwarp(mask)` — warp barrier
- `__threadfence()` — global memory fence (ensures writes visible to other threads)
- `__threadfence_block()` — block-scoped memory fence

## Warp-Level Primitives

### Shuffle (data exchange within warp, no shared memory)
- `__shfl_sync(mask, val, srcLane)` — read from specific lane
- `__shfl_up_sync(mask, val, delta)` — read from lane-delta
- `__shfl_down_sync(mask, val, delta)` — read from lane+delta
- `__shfl_xor_sync(mask, val, laneMask)` — read from lane XOR mask

### Vote
- `__ballot_sync(mask, pred)` — bitmask of threads where pred is true
- `__any_sync(mask, pred)` / `__all_sync(mask, pred)`

### Reduce
- `__reduce_add_sync(mask, val)`, `__reduce_min_sync`, `__reduce_max_sync`

## Shared Memory

### Allocation
- Static: `__shared__ float data[256];` (up to 48 KB)
- Dynamic: `extern __shared__ type array[];` + `kernel<<<B,T,bytes>>>()`
- Above 48 KB requires `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)`

### Bank Conflicts
- 32 banks, 4 bytes per bank
- Same-bank access by multiple threads → serialized
- Broadcast (all read same address) → no conflict
- Stride-1 (consecutive 4-byte words) → no conflict

## Asynchronous Operations

### cp.async (global → shared, bypassing registers)
- `cuda::memcpy_async(dst_shared, src_global, size, barrier)`
- Overlaps data movement with computation
- Requires `cuda::barrier` for completion

### Pipeline
- Multi-stage: `memcpy_async()` → `commit()` → `wait(stage)`
- Software pipelining of global→shared loads

## mma.sync (warp-level tensor core, sm_70+)
- All 32 warp threads cooperate on one matrix multiply-accumulate
- Fragment types: A (m×k), B (k×n), C/D accumulator (m×n)
- Layout: row/col major per fragment
- Precisions: FP16, BF16, TF32, FP8, INT8 → FP32 accumulator
- D = A * B + C in a single instruction
