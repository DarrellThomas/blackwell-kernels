# PTX ISA 9.2 Technical Reference

Source: https://docs.nvidia.com/cuda/parallel-thread-execution/
Fetched: 2026-03-27

## Overview

PTX (Parallel Thread Execution) is a low-level virtual machine and instruction set architecture for NVIDIA GPUs, enabling data-parallel computing across multiple threads organized in a hierarchical structure.

## Thread Organization

**Thread Hierarchy:**
- Individual threads execute in SIMT (single-instruction, multiple-thread) fashion
- Cooperative Thread Arrays (CTAs) group threads with shared memory and synchronization
- Clusters of CTAs enable broader synchronization (sm_90+)
- Grids organize clusters/CTAs for large-scale parallel execution

**Key identifiers:** Thread IDs use `tid.x/y/z`, CTA IDs use `ctaid.x/y/z`, cluster IDs accessible via special registers.

## Memory Hierarchy

- **Local Memory:** Per-thread private memory
- **Shared Memory:** CTA-visible with cluster visibility (sm_90+)
- **Global Memory:** All-thread accessible
- **Constant/Texture:** Read-only, cached
- **Surface:** Read-write cached memory

## Core Instruction Categories

### Integer Arithmetic
Instructions include: `add`, `sub`, `mul`, `mad`, `div`, `rem`, `abs`, `neg`, `min`, `max`, `popc`, `clz`, `bfind`, `bfe`, `bfi`

### Floating-Point Operations
- Standard: `add`, `sub`, `mul`, `fma`, `div`, `abs`, `neg`, `min`, `max`, `rcp`, `sqrt`, `rsqrt`
- Transcendental: `sin`, `cos`, `lg2`, `ex2`, `tanh`
- Half-precision variants available

### Memory Operations
- **Load/Store:** `ld`, `st` with state space qualifiers
- **Async Copy:** `cp.async`, `cp.async.bulk`, `cp.async.bulk.tensor`
- **Prefetch:** `prefetch`, `prefetchu`

### Synchronization

**Barriers:**
- `bar` / `barrier`: CTA-level synchronization
- `barrier.cluster`: Cluster-level (sm_90+)

**Atomic Operations:**
- `atom`: Atomic read-modify-write with operations (add, inc, dec, max, min, and, or, xor, cas, exch)
- Supports global and shared memory

**Memory Fences:**
- `membar` / `fence`: Enforces memory ordering with scope qualifiers (thread, block, system)

**Advanced Synchronization:**
- `mbarrier`: Multi-phase barrier supporting asynchronous operation tracking
- `vote.sync`: Warp-level voting
- `match.sync`: Warp-level matching
- `redux.sync`: Warp-level reduction

## Matrix Multiply-Accumulate (MMA)

**Warp-Level Instructions:**

`mma.sync` variants for different shapes and data types:
- **m16n8k16** float: Accumulates across K=16, produces 16x8 result
- **m16n8k32** float: Extended K dimension (FP8)
- **m8n8k16** operations for smaller matrices

**Instruction format example:**
```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
```

**Fragment Organization:**
- Matrix A (row-major): m x k elements distributed across warp threads
- Matrix B (col-major): k x n elements distributed across warp threads
- Matrix D (accumulator): m x n elements in registers

### Matrix Load/Store

**ldmatrix:** Loads matrix fragments from shared memory to registers with layout optimization
```
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
```

**stmatrix:** Stores register fragments to shared memory

## Async Copy Operations

**Non-Bulk Copy:**
```
cp.async.cg.shared.global [dst], [src], 16;
cp.async.commit_group;
cp.async.wait_group N;
cp.async.wait_all;
```

**Bulk Copy:**
```
cp.async.bulk [dst], [src], bytes, [mbarrier];
cp.async.bulk.tensor [dst], [src], [tensor_map];
```

## Data Types

**Integer:** u8, s8, u16, s16, u32, s32, u64, s64
**Floating-Point:** f16, f32, f64, bf16
**Special:** e4m3 (FP8), e5m2 (FP8), tf32 (32-bit precision, 19-bit mantissa)
**Packed Types:** u8x4, s8x4, u16x2, s16x2, bf16x2

## State Spaces

- `.reg`: Register (per-thread)
- `.sreg`: Special registers
- `.const`: Read-only constants
- `.global`: All-thread accessible
- `.local`: Per-thread local memory
- `.param`: Function parameters
- `.shared`: CTA-shared memory
- `.generic`: Unified addressing

## Special Registers

- `%tid.{x,y,z}`: Thread ID within CTA
- `%ntid.{x,y,z}`: CTA dimensions
- `%ctaid.{x,y,z}`: CTA ID in grid
- `%nctaid.{x,y,z}`: Grid dimensions
- `%laneid`: Lane ID within warp (0-31)
- `%warpid`: Warp ID within CTA
- `%lanemask_eq/lt/le/gt/ge`: Warp voting masks
- `%clock64`: 64-bit clock counter

## Version 9.2 Additions (CUDA 13.2)

- `.u8x4` / `.s8x4` support for arithmetic instructions (packed 8-bit SIMD)
- `.add.sat.{u16x2/s16x2/u32}` saturation operations
- `.b128` type for `st.async`
- `.ignore_oob` for `cp.async.bulk` (out-of-bounds handling)
- `.bf16x2` destination for FP8 conversion instructions (`cvt.rn.satfinite.e4m3x2.f32`)
