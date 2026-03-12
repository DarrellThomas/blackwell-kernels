# Spatters.ca — Implementing a Fast Tensor Core Matmul on Ada (RTX 4090)

**Source:** https://www.spatters.ca/mma-matmul
**Code:** https://github.com/spatters/mma-matmul
**Architecture:** Ada Lovelace (sm_89), RTX 4090
**Relevance to sm_120:** Both use `mma.sync` ISA (not `tcgen05`). Techniques transfer directly.
**Result:** 93% of RTX 4090 peak (153.6 TFLOP/s), matching cuBLAS.

---

## 1. Hardware Context

| Spec | RTX 4090 (Ada, sm_89) | RTX 5090 (Blackwell, sm_120) |
|------|----------------------|------------------------------|
| Tensor Cores | 512 | 170 SMs (TCs per SM varies) |
| Peak FP16/32 | 165.2 TFLOP/s | ~209.5 TFLOP/s |
| Boost Clock | 2520 MHz | ~2407 MHz |
| Warp Schedulers/SM | 4 | 4 |
| Max Warps/SM | 48 (Ada) | 48 |
| ISA | mma.sync | mma.sync |
| Shared Mem/SM | 128 KB | 128 KB |
| Max Shared/Block | 100 KB | 99 KB |

**MMA Instruction Used:** `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- A: 16x16 FP16 (row-major), B: 16x8 FP16 (col-major), C/D: 16x8 FP32
- FLOP per instruction: 2 x 16 x 8 x 16 = 4096
- Latency: 32 cycles (derived from peak throughput: 165.2 TFLOP/s / (512 TCs x 4096 FLOP) = 12.7 ns = 32 cycles at 2520 MHz)
- All 32 threads in a warp cooperate on one MMA

**Theoretical Lower Bound (M=N=K=4096):**
- Total FLOP: 2 x 4096^3 = 137.4 GFLOP
- Total MMA instructions: 256 x 512 x 256 = 33,554,432
- Card-wide waves: 65,536 (33M instructions / 512 TCs)
- Minimum cycles: 65,536 x 32 = 2,097,152 cycles
- Minimum time: 2,097,152 / 2.52 GHz = 832 us

---

## 2. Complete Optimization Progression

### Kernel 1.0: Naive MMA

| Metric | Value |
|--------|-------|
| Execution time | 4680 us |
| Throughput | 29.4 TFLOP/s |
| % of cuBLAS | 19.1% |
| % of peak | 17.8% |
| Cycles per MMA | 179.9 |
| Thread block | 16x16 (256 threads, 8 warps) |
| Output tile/block | 32x32 |
| Output tile/warp | 16x8 |
| Warp arrangement | 2 rows x 4 columns |

**Warp layout in output tile:**
```
(warp_0 | warp_1 | warp_2 | warp_3)
(warp_4 | warp_5 | warp_6 | warp_7)
```

**Problems identified via Nsight Compute warp state stats:**
- Shared memory throttle (MIO): 31 cycles/instruction average
- Barrier waits: 15 cycles/instruction
- Long scoreboard (global load deps): 11 cycles/instruction

**Root causes:**
1. Uncoalesced global loads: each thread loads individual 16-bit values
2. Shared memory bank conflicts during register loads
3. Low computation-to-memory ratio (1 MMA per sync point)

**Main loop structure:**
```c
for (int kStart = 0; kStart < K; kStart += K_BLOCK) {
    // Load from global to shared memory (uncoalesced 16-bit loads)
    for (int m = 0; m < 2; ++m) {
        As[m*K_BLOCK + ty][tx] = A[(mBlock + ty + m*K_BLOCK)*K + kStart + tx];
    }
    for (int n = 0; n < 2; ++n) {
        Bs[ty][n*K_BLOCK + tx] = B[(kStart + ty) * K + nBlock + n*K_BLOCK + tx];
    }
    __syncthreads();

    // Load from shared memory to FP16 registers (bank conflicts here)
    aReg[0] = As[mWarp + groupID][groupLaneID*2];
    aReg[1] = As[mWarp + groupID][groupLaneID*2 + 1];
    aReg[2] = As[mWarp + groupID + 8][groupLaneID*2];
    aReg[3] = As[mWarp + groupID + 8][groupLaneID*2 + 1];
    aReg[4] = As[mWarp + groupID][groupLaneID*2 + 8];
    aReg[5] = As[mWarp + groupID][groupLaneID*2 + 9];
    aReg[6] = As[mWarp + groupID + 8][groupLaneID*2 + 8];
    aReg[7] = As[mWarp + groupID + 8][groupLaneID*2 + 9];

    bReg[0] = Bs[groupLaneID*2 + 0][nWarp + groupID];
    bReg[1] = Bs[groupLaneID*2 + 1][nWarp + groupID];
    bReg[2] = Bs[groupLaneID*2 + 8][nWarp + groupID];
    bReg[3] = Bs[groupLaneID*2 + 9][nWarp + groupID];

    unsigned const *aPtr = reinterpret_cast<unsigned const *>(&aReg);
    unsigned const *bPtr = reinterpret_cast<unsigned const *>(&bReg);
    mma_m16n8k16(aPtr, bPtr, dReg, dReg);
    __syncthreads();
}
```

**Register usage per thread (Kernel 1.0):**
- A fragment: 8 FP16 values packed into 4 x 32-bit registers
- B fragment: 4 FP16 values packed into 2 x 32-bit registers
- D accumulator: 4 FP32 values = 4 registers
- Total per MMA: 4 (A) + 2 (B) + 4 (C) + 4 (D) = 14 registers minimum

---

### Kernel 1.1: Naive + 2x Tiling

| Metric | Value |
|--------|-------|
| Execution time | 2400 us |
| Throughput | 57.3 TFLOP/s |
| % of cuBLAS | 37.3% |
| % of peak | 34.7% |
| Cycles per MMA | ~90 (halved) |

**Change:** Added 2x tiling in M and N dimensions. Each warp now executes 4 MMA instructions (2x2 output tiles) per K iteration. Thread block computes 64x64 output tile instead of 32x32.

**Why it helps:** Amortizes synchronization overhead. 4 MMAs per `__syncthreads()` instead of 1. Barrier cost per MMA reduced 4x.

---

### Kernel 2.0: Vectorized Loads + Permuted Shared Memory

| Metric | Value |
|--------|-------|
| Execution time | 1080 us |
| Throughput | 127.3 TFLOP/s |
| % of cuBLAS | 82.9% |
| % of peak | 77.0% |
| Cycles per MMA | ~32.8 |
| Occupancy | 32 warps/SM |

**Three critical changes:**

#### 2.0a: Vectorized 128-bit Global Loads

Replaced per-element 16-bit loads with `uint4` (128-bit, 8 FP16 elements) loads:
```c
As[storeRow][storeCol] = globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
Bs[storeRow][storeCol] = globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
```

Consecutive threads load consecutive `uint4` values in K dimension — fully coalesced. 8 consecutive threads = 128 bytes = one cache line.

**Rationale for K-major storage:** "Working with 128b vectors is natural when using Tensor Cores as the fundamental Tensor Core operation is an 8x8x128b matrix multiply, i.e., each 128b vector forms one row of the input matrices."

**Requires:** A stored row-major in global memory, B stored column-major.

#### 2.0b: XOR-Permuted Shared Memory Layout

**Store permutation (global -> shared):**
```c
int storeRow = warpID * 4 + laneID / 8;
int storeCol = (laneID % 8) ^ (laneID / 8);
```

**Load permutation (shared -> registers via ldmatrix):**

For A (16x16 matrix):
```c
int loadRowA = (laneID % 16) / 2;
int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
```

For B (8x16 matrix):
```c
int loadRowB = (laneID % 8) / 2;
int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);
```

**Why this eliminates bank conflicts:**

Shared memory has 32 banks, 4 bytes per bank. A row of 8 `uint4` values = 128 bytes = spans all 32 banks. When each thread requests a 16-byte (128-bit) `uint4`, the warp's 512-byte request is split into **4 phases of 8 consecutive threads each** (max shared memory bandwidth = 32 banks x 4B = 128B per phase).

The XOR permutation ensures that within each 8-thread phase, no two threads access the same bank. The store-side permutation `(laneID % 8) ^ (laneID / 8)` rearranges column placement so that the load-side ldmatrix accesses hit distinct banks.

The store itself would be bank-conflict-free without permutation — the permutation exists specifically for the load (ldmatrix) phase.

**For k=2..3 subtiles:** Column indices are XORed with 2: `loadColA ^ 2`, `loadColB ^ 2`. This accesses the second half of each K tile while maintaining bank-conflict-free access.

**Verification:** Nsight Compute showed 0 bank conflicts for Kernel 2.1.

#### 2.0c: ldmatrix for Register Loading

Replaced scalar register loads with `ldmatrix.sync.aligned.x4.m8n8.shared.b16`:
```c
__device__ void load_matrix_x4(unsigned *destReg, uint4 *srcAddr) {
    unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(destReg[0]), "=r"(destReg[1]), "=r"(destReg[2]), "=r"(destReg[3])
        : "r"(ptxSrcAddr)
    );
}
```

- `__cvta_generic_to_shared` converts 64-bit C pointer to 32-bit shared memory address
- `volatile` qualifier is **critical** — "without it the loads do not get synchronized properly and threads end up with incorrect data"
- `.x4` loads 4 x 8x128-bit matrices (32 x 128-bit rows total)
- `.x2` variant available for B fragments (smaller)
- Thread-to-address mapping: threads 0-15 provide addresses for m=0..15 k=0; threads 16-31 for m=0..15 k=1

**Main loop (Kernel 2.0):**
```c
for (int k = 0; k < K/8; k += 4) {
    As[storeRow][storeCol] = globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    Bs[storeRow][storeCol] = globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    __syncthreads();

    for (int m = 0; m < 2; m++) {
        int mTile = m * 8;
        for (int n = 0; n < 2; n++) {
            int nTile = n * 4;
            // Load A and B from shared -> registers, then MMA (k=0..1)
            load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + loadColA));
            load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + loadColB));
            mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
            // Same for k=2..3 (XOR column with 2)
            load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + (loadColA^2)));
            load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + (loadColB^2)));
            mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
        }
    }
    __syncthreads();
}
```

---

### Kernel 2.1: Register Reuse Optimization

| Metric | Value |
|--------|-------|
| Execution time | 1030 us |
| Throughput | 133.4 TFLOP/s |
| % of cuBLAS | 86.9% |
| % of peak | 80.8% |
| Cycles per MMA | 38 |

**Change:** In Kernel 2.0, each A tile was reloaded from shared memory for every B tile in the inner loop. Kernel 2.1 loads each A tile **once** and reuses it across all B tiles.

**Trade-off:** Uses 8 additional FP32 registers for the second A tile copy, but eliminates redundant `ldmatrix` operations.

**Stall profile:** Primary stall is now Tensor Core availability waiting (not memory). This is the inflection point where the kernel becomes compute-bound.

---

### Kernel 3.0: N-Stage Async Pipeline (cp.async)

| Metric | Value |
|--------|-------|
| Execution time | 1000 us |
| Throughput | 137.4 TFLOP/s |
| % of cuBLAS | 89.5% |
| % of peak | 83.2% |
| Occupancy | 24 warps/SM (down from 32) |
| N_STAGES | 3 (optimal) |

**Change:** Replaced synchronous global-to-shared loads with `cp.async` (global -> shared, bypassing registers).

#### cp.async PTX Wrapper

```c
__device__ void cp_async(uint4 *dstAddr, const uint4 *srcAddr) {
    unsigned ptxDstAddr = __cvta_generic_to_shared(dstAddr);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(ptxDstAddr),
           "l"(srcAddr),
           "n"(16));
}
```

- Copies 16 bytes (128 bits) per instruction
- `.cg` cache policy: cache in L2, not L1
- `.L2::128B` specifies 128-byte cache line granularity
- Third parameter `"n"(16)` must be compile-time constant
- Grouped via `cp.async.commit_group`
- Synchronized via `cp.async.wait_group N` (N must be compile-time constant)

#### Shared Memory Layout (Circular Buffer)

```c
__shared__ uint4 As[N_STAGES * 32][8];
__shared__ uint4 Bs[N_STAGES * 32][8];
```

With N_STAGES=3: 96 rows per buffer. Total shared memory = 2 x 96 x 8 x 16 bytes = 24,576 bytes.

#### Pipeline Structure

**Prelude:** Load first N_STAGES-1 stages before entering main loop:
```c
for (int nStage = 0; nStage < N_STAGES - 1; nStage++) {
    int kStart = nStage * 4;
    aStorePtr = As + 32 * nStage;
    bStorePtr = Bs + 32 * nStage;
    cp_async(aStorePtr[storeRow] + storeCol, aGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow] + storeCol, bGlobalAddress + kStart);
    asm volatile("cp.async.commit_group;\n" ::);
}
```

**Invariant:** At start of each main loop iteration, at most N_STAGES-1 cp.async operations are pending. Load pointer and store pointer alternate between buffer slots modulo N_STAGES.

**Main loop:**
```c
for (int nStage = 0; nStage < K/32; nStage++) {
    int kStart = (N_STAGES - 1 + nStage) * 4;
    aStorePtr = As + 32 * ((nStage + N_STAGES - 1) % N_STAGES);
    bStorePtr = Bs + 32 * ((nStage + N_STAGES - 1) % N_STAGES);
    aLoadPtr = As + 32 * (nStage % N_STAGES);
    bLoadPtr = Bs + 32 * (nStage % N_STAGES);

    // Wait for current stage's data to arrive
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES - 2));
    __syncthreads();

    // Preload ALL A and B fragments into registers
    for (int m = 0; m < 2; m++) {
        load_matrix_x4(aReg[m], aLoadPtr[m*8 + warpOffsetA + loadRowA] + loadColA);
        load_matrix_x4(aReg[m] + 4, aLoadPtr[m*8 + warpOffsetA + loadRowA] + (loadColA^2));
    }
    for (int n = 0; n < 2; n++) {
        load_matrix_x2(bReg[n], bLoadPtr[n*4 + warpOffsetB + loadRowB] + loadColB);
        load_matrix_x2(bReg[n] + 2, bLoadPtr[n*4 + warpOffsetB + loadRowB] + (loadColB^2));
    }

    // Issue cp.async for NEXT stage (overlaps with MMA computation below)
    kStart = (kStart > 512 - 4) ? 512 - 4 : kStart;  // Clamp for final iterations
    cp_async(aStorePtr[storeRow] + storeCol, aGlobalAddress + kStart);
    cp_async(bStorePtr[storeRow] + storeCol, bGlobalAddress + kStart);
    asm volatile("cp.async.commit_group;\n" ::);

    // Execute MMAs (2x2 tiling = 4 MMA pairs = 8 MMA instructions)
    for (int m = 0; m < 2; m++) {
        for (int n = 0; n < 2; n++) {
            mma_m16n8k16(aReg[m], bReg[n], dReg[m][n]);
            mma_m16n8k16(aReg[m] + 4, bReg[n] + 2, dReg[m][n]);
        }
    }
}
```

#### Superfluous cp.async Hack for Final Iterations

On the last N_STAGES-1 iterations, no new data is needed from global memory. But `cp.async.wait_group` requires a **compile-time constant** argument. To keep the argument fixed at `N_STAGES-2`, the code submits superfluous copies (clamped to last valid K position) that are never consumed:

```c
// Clamp to prevent OOB access while keeping wait_group argument constant
kStart = (kStart > 512 - 4) ? 512 - 4 : kStart;
cp_async(aStorePtr[storeRow] + storeCol, aGlobalAddress + kStart);
cp_async(bStorePtr[storeRow] + storeCol, bGlobalAddress + kStart);
```

#### N_STAGES Selection

| N_STAGES | Performance |
|----------|-------------|
| 3 | 1000 us (optimal) |
| 4 | Similar to 3 |
| >4 | Decreased (occupancy loss) |

Occupancy dropped from 32 to 24 warps/SM due to extra shared memory. The modest improvement (1030 -> 1000 us, 3%) suggests occupancy loss nearly offsets latency-hiding benefit.

---

### Kernel 3.1: 4x Tiling — Matching cuBLAS

| Metric | Value |
|--------|-------|
| Execution time | 895 us |
| Throughput | 153.6 TFLOP/s |
| % of cuBLAS | 100% |
| % of peak | 93.0% |
| Cycles per MMA | 34.2 |

**Change:** Increased M/N tiling from 2x to 4x.

| Property | Kernel 3.0 | Kernel 3.1 |
|----------|-----------|-----------|
| Block output tile | 64x64 | 128x128 |
| MMAs per K iteration | 2x2x2 = 8 | 4x4x2 = 32 |
| Grid size (M=N=4096) | 64x64 = 4096 blocks | 32x32 = 1024 blocks |

**Why 4x more MMAs per iteration helps:**
1. Barrier frequency reduced: same number of `__syncthreads()` calls but 4x more MMAs between them
2. Barrier cost per MMA reduced ~4x
3. Tensor cores stay fed longer between synchronization points

**Register budget (Kernel 3.1):**
- A fragment preloads: aReg[4][8] = 32 x 32-bit = 32 registers (for 4 M tiles x 2 K subtiles)
- B fragment preloads: bReg[4][4] = 16 x 32-bit = 16 registers (for 4 N tiles x 2 K subtiles)
- D accumulators: dReg[4][4][4] = 64 FP32 = 64 registers (for 4x4 output tiles x 4 values each)
- Total fragments: ~112 registers + other locals

**Stall profile (final):**
- Dominated by Tensor Core wait: ~36 cycles average per warp
- All memory stalls effectively eliminated
- "The vast majority of stalls are now due to waiting for Tensor Cores"
- This is the goal state — compute-bound kernel

---

## 3. Performance Summary Table

| Kernel | Time (us) | TFLOP/s | % cuBLAS | % Peak | Cycles/MMA | Key Change |
|--------|-----------|---------|----------|--------|-----------|------------|
| 1.0 | 4680 | 29.4 | 19.1% | 17.8% | 179.9 | Naive MMA |
| 1.1 | 2400 | 57.3 | 37.3% | 34.7% | ~90 | 2x M/N tiling |
| 2.0 | 1080 | 127.3 | 82.9% | 77.0% | ~32.8 | Vec loads + XOR swizzle + ldmatrix |
| 2.1 | 1030 | 133.4 | 86.9% | 80.8% | 38 | A register reuse |
| 3.0 | 1000 | 137.4 | 89.5% | 83.2% | ~30.4 | cp.async 3-stage pipeline |
| 3.1 | 895 | 153.6 | 100% | 93.0% | 34.2 | 4x M/N tiling (128x128 block) |
| cuBLAS | 895 | 153.6 | 100% | 93.0% | — | Reference |

**Biggest jumps:**
- 1.0 -> 2.0: 4.3x speedup (coalescing + bank conflicts + ldmatrix)
- 3.0 -> 3.1: 1.12x speedup (tiling amortization)
- 1.0 -> 3.1: 5.2x total speedup

---

## 4. PTX Wrappers (Complete)

### MMA Instruction
```c
__device__ void mma_m16n8k16(const unsigned *A, const unsigned *B,
                             float *C, float *D) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );
}
```

**Note:** This MMA wrapper does NOT use `volatile`. The compiler is free to reorder MMA instructions relative to each other, which enables it to interleave them with loads when profitable.

### ldmatrix (x4 — for A fragments)
```c
__device__ void load_matrix_x4(unsigned *destReg, uint4 *srcAddr) {
    unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(destReg[0]), "=r"(destReg[1]), "=r"(destReg[2]), "=r"(destReg[3])
        : "r"(ptxSrcAddr)
    );
}
```

### ldmatrix (x2 — for B fragments)
```c
__device__ void load_matrix_x2(unsigned *destReg, uint4 *srcAddr) {
    unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(destReg[0]), "=r"(destReg[1])
        : "r"(ptxSrcAddr)
    );
}
```

### cp.async
```c
__device__ void cp_async(uint4 *dstAddr, const uint4 *srcAddr) {
    unsigned ptxDstAddr = __cvta_generic_to_shared(dstAddr);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(ptxDstAddr),
           "l"(srcAddr),
           "n"(16));
}
```

### cp.async Synchronization
```c
asm volatile("cp.async.commit_group;\n" ::);        // Batch pending copies
asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); // Wait until N groups remain
```

**Critical:** `volatile` is required on ldmatrix and cp.async for correctness. Without it, the compiler reorders loads across synchronization boundaries, causing threads to read stale data.

---

## 5. The Scheduling Model — How Latency Is Hidden

The blog does NOT use explicit instruction-level interleaving (no manual PTX scheduling of MMA interleaved with loads). Instead, the approach relies on three mechanisms:

### 5a. Hardware Warp Scheduling

Ada has 4 warp schedulers per SM, each with its own Tensor Core. With 24 warps active per SM (Kernel 3.0+), the hardware scheduler round-robins across warps. When one warp stalls on math_pipe_throttle (MMA input FIFO full) or wait (MMA result dependency), the scheduler issues from another warp.

### 5b. Structural Overlap (cp.async + MMA)

The loop structure places `cp.async` (non-blocking) BEFORE the MMA block:
```
1. Wait for current stage data
2. Load shared -> registers (ldmatrix)
3. Issue cp.async for NEXT stage  <-- non-blocking, runs in background
4. Execute MMAs                    <-- overlaps with cp.async in flight
```

The cp.async hardware copies data from global to shared memory while the tensor cores execute MMAs. This is structural overlap, not instruction-level interleaving.

### 5c. Tiling Amortization (Key Insight from 3.0 -> 3.1)

The biggest performance jump after Kernel 2.0 came from increasing tiling (3.0 -> 3.1: 8 MMAs -> 32 MMAs per iteration). This works because:
- More independent MMA instructions give the warp scheduler more to work with
- Barrier (`__syncthreads()`) frequency per MMA drops 4x
- The tensor core can stay busy for 32 consecutive MMAs before any sync point

**This is the most applicable lesson for sm_120 flash attention:** Larger tiles with more independent MMAs per loop iteration reduce synchronization overhead and keep the tensor core fed.

---

## 6. Nsight Compute Analysis Details

### Warp State Stats Progression

**Kernel 1.0 (memory-bound):**
- MIO throttle (shared mem): 31 cycles/instruction (dominant)
- Barrier: 15 cycles/instruction
- Long scoreboard (global load): 11 cycles/instruction
- Total overhead: ~57 cycles/instruction -> 179.9 cycles/MMA

**Kernel 2.1 (transition to compute-bound):**
- MIO throttle: eliminated (XOR swizzle + ldmatrix)
- Barrier: reduced (2x tiling amortization)
- Long scoreboard: reduced (register preloading)
- Primary stall: Tensor Core availability

**Kernel 3.1 (compute-bound):**
- Tensor Core wait: ~36 cycles average per warp (dominant)
- All other stalls: minimal
- This is the target state for an optimized kernel

### Tensor Core Utilization Measurement Error

Nsight Compute reported 47.3% Tensor Core utilization for Kernel 3.1. The author identified this as a measurement error:

- Nsight uses a fixed 16-cycle latency for `smsp__pipe_tensor_op_hmma_cycles_active`
- The actual m16n8k16 latency is 32 cycles (derived from peak throughput)
- Correct calculation: 32 x 65,536 / total_cycles = ~94.6% utilization
- 47.3% is approximately half of 94.6%, consistent with the 16-vs-32 cycle error

### Relevant Nsight Metrics Referenced
- `sm__cycles_elapsed.avg`, `.max`, `.min`, `.sum`
- `smsp__inst_executed_pipe_tensor_op_hmma.avg`, `.max`, `.min`, `.sum`
- `smsp__pipe_tensor_op_hmma_cycles_active`
- Warp state statistics (implied `smsp__warp_issue_stalled_*` counters)
- Shared memory bank conflict metrics

---

## 7. Output Store Pattern

No `stmatrix` instruction available on Ada (requires sm_90). The blog writes directly from registers to global memory:

"We write directly from registers to global memory. It may be possible to optimize this by writing first to shared and then writing to global in a coalesced pattern, but that requires more shared memory and could reduce occupancy. I experimented with this but did not see a performance improvement."

---

## 8. Numerical Precision

Direct MMA accumulation (`D = A*B + C` where C is the running accumulator) causes precision loss when `C >> A*B` magnitudes. Relative error: ~1e-5.

**Alternative: External accumulation:**
```c
float4 dRegAcc = 0;
float cReg[4] = {0.};
mma_m16n8k16(aPtr, bPtr, cReg, dReg);  // Don't accumulate in MMA
float4 *dRegPtr = reinterpret_cast<float4 *>(dReg);
dRegAcc.x += dRegPtr->x;  // Accumulate in separate FP32 registers
dRegAcc.y += dRegPtr->y;
dRegAcc.z += dRegPtr->z;
dRegAcc.w += dRegPtr->w;
```

Performance cost: ~10 us on Kernel 3.1. Reduces relative error by two orders of magnitude (~1e-7) and centers it around zero.

---

## 9. Shared Memory Layout Summary

| Kernel | Declaration | Size | Banks Used |
|--------|-------------|------|------------|
| 1.0 | `fp16 As[2*K_BLOCK][32]` + `Bs[16][2*K_BLOCK]` | Small | Bank conflicts |
| 2.x | `uint4 As[32][8]` + `Bs[32][8]` | 2 x 32 x 8 x 16B = 8 KB | XOR swizzle, 0 conflicts |
| 3.x | `uint4 As[N_STAGES*32][8]` + `Bs[N_STAGES*32][8]` | 2 x 96 x 8 x 16B = 24.6 KB (N=3) | XOR swizzle |

Row layout: 8 `uint4` values per row = 128 bytes = spans all 32 shared memory banks. Each row perfectly covers the bank space.

---

## 10. Applicability to sm_120 Flash Attention

### What Transfers Directly
1. **XOR swizzle pattern** for shared memory: identical bank structure (32 banks, 4B/bank)
2. **ldmatrix addressing formulas**: same ISA on sm_120
3. **cp.async with N-stage pipeline**: same instruction support
4. **Tiling amortization**: larger output tiles = more independent MMAs = better scheduling
5. **PTX wrappers**: `mma_m16n8k16`, `load_matrix_x4/x2`, `cp_async` are ISA-compatible

### What Differs
1. **sm_120 has 48 warps/SM** (same as Ada sm_89 — this actually matches well)
2. **sm_120 max shared/block is 99 KB** (vs 100 KB on Ada) — nearly identical
3. **We use BF16 not FP16**: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` — same tile sizes, same register layout
4. **Flash attention has softmax between MMA phases**: cannot have a single unbroken MMA block like GEMM
5. **Our a1/a2 register swap**: required on sm_120, not mentioned for Ada — may be an sm_120-specific quirk or may be needed but not discussed

### Key Lessons for Our Kernel
1. **Increase tiling to get more MMAs between sync points** — the 3.0 -> 3.1 jump (8 -> 32 MMAs per iter) gave the biggest late-stage improvement
2. **XOR swizzle is non-negotiable** — eliminating bank conflicts was the single biggest optimization (4.3x from Kernel 1.0 to 2.0)
3. **cp.async pipeline helps modestly** (3%) when occupancy drops — worth doing but not transformative
4. **The compiler handles MMA scheduling adequately** — no manual PTX interleaving was needed to reach 93% peak
5. **Register preloading (load all fragments before any MMA)** avoids redundant shared memory reads
6. **stmatrix unavailable** — must handle P conversion without it (we already do register-only P->A conversion)
