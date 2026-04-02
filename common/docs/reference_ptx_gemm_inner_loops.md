# PTX GEMM Inner Loop Implementations — Reference Collection

**Purpose:** Catalog of open-source implementations writing GEMM inner loops in inline PTX, with code snippets and scheduling patterns relevant to sm_120 BF16 GEMM.
**Last updated:** 2026-03-13

---

## 1. spatters/mma-matmul (Ada sm_89, FP16)

**Source:** https://github.com/spatters/mma-matmul | https://www.spatters.ca/mma-matmul
**Result:** 93% of RTX 4090 peak (153.6 TFLOP/s), matching cuBLAS
**See also:** `/data/src/bwk/main/docs/reference_spatters_mma_matmul.md` for full analysis

### Inner Loop (Kernel 3.1 — best performer)

The inner loop preloads ALL A and B fragments into registers, then executes all MMAs:

```c
// Structure per K iteration:
cp.async.wait_group(N_STAGES - 2);
__syncthreads();

// 1. Preload ALL shared -> register fragments
for (int m = 0; m < 2; m++) {
    load_matrix_x4(aReg[m], aLoadPtr[...] + loadColA);        // k=0..1
    load_matrix_x4(aReg[m]+4, aLoadPtr[...] + (loadColA^2));  // k=2..3
}
for (int n = 0; n < 2; n++) {
    load_matrix_x2(bReg[n], bLoadPtr[...] + loadColB);        // k=0..1
    load_matrix_x2(bReg[n]+2, bLoadPtr[...] + (loadColB^2));  // k=2..3
}

// 2. Issue cp.async for NEXT K block (non-blocking)
cp_async(aStorePtr[...], aGlobalAddress + kNext);
cp_async(bStorePtr[...], bGlobalAddress + kNext);
cp.async.commit_group;

// 3. Execute MMAs (4x4 tiling = 32 MMA instructions)
for (int m = 0; m < 4; m++)
    for (int n = 0; n < 4; n++) {
        mma_m16n8k16(aReg[m], bReg[n], dReg[m][n]);      // k=0..1
        mma_m16n8k16(aReg[m]+4, bReg[n]+2, dReg[m][n]);  // k=2..3
    }
```

**Key insight:** The compiler naturally interleaves the 32 MMA instructions with the pending cp.async. No manual PTX interleaving needed at this tiling level. The 32 independent MMAs give the warp scheduler enough work to hide all memory latency.

**Register budget (Kernel 3.1):**
- A fragments: `aReg[4][8]` = 32 registers
- B fragments: `bReg[4][4]` = 16 registers
- D accumulators: `dReg[4][4][4]` = 64 registers
- Total MMA-related: ~112 registers + locals

### PTX Wrappers

```c
// MMA — NOT volatile (compiler can reorder for scheduling)
__device__ void mma_m16n8k16(const unsigned *A, const unsigned *B,
                             float *C, float *D) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

// ldmatrix — volatile (must not reorder across sync boundaries)
__device__ void load_matrix_x4(unsigned *destReg, uint4 *srcAddr) {
    unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(destReg[0]), "=r"(destReg[1]), "=r"(destReg[2]), "=r"(destReg[3])
        : "r"(ptxSrcAddr));
}

// cp.async — volatile (must commit before wait)
__device__ void cp_async(uint4 *dstAddr, const uint4 *srcAddr) {
    unsigned ptxDstAddr = __cvta_generic_to_shared(dstAddr);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(ptxDstAddr), "l"(srcAddr), "n"(16));
}
```

**Critical pattern:** MMA is NOT volatile — the compiler is free to reorder MMAs relative to each other and relative to non-volatile instructions. This lets the compiler interleave MMAs with loads. ldmatrix and cp.async ARE volatile to prevent reordering across synchronization boundaries.

---

## 2. salykova/sgemm.cu (Ampere GA102, FP32 scalar)

**Source:** https://github.com/salykova/sgemm.cu | https://salykova.github.io/sgemm-gpu
**Result:** Beats cuBLAS by 3-4% on RTX 3090
**See also:** `/data/src/bwk/main/docs/reference_salykova_sgemm.md`

### Inner Loop (FP32 SGEMM — no tensor cores)

This is scalar FMA, not tensor core MMA, but the scheduling pattern is instructive:

```c
// Register double-buffered fragments
float frag_a[2][8];  // ping-pong buffers
float frag_b[2][8];

// Preload first fragments
// ... load frag_a[0], frag_b[0] from shared ...

for (int wk = 0; wk < K_TILE; wk++) {
    int curr = wk % 2;
    int next = (wk + 1) % 2;

    // Load NEXT fragments while computing with CURRENT
    // These loads execute while FMA pipeline processes current data
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(frag_a[next][0]), "=f"(frag_a[next][1]),
          "=f"(frag_a[next][2]), "=f"(frag_a[next][3])
        : "r"(smem_addr_a_next));

    // 8x8 outer product with current fragments
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            acc[i][j] += frag_a[curr][i] * frag_b[curr][j];
}
```

**Key technique:** Register-level double-buffering. Load `frag[next]` from shared memory while FMA executes on `frag[curr]`. This hides the ~20-30 cycle shared memory load latency behind the FMA chain.

**Why this beat cuBLAS:** The register double-buffer kept the FMA pipeline continuously fed, whereas cuBLAS apparently stalled on smem loads at these specific tile sizes. The technique is directly applicable to our MMA inner loop — we can load next ldmatrix fragments while MMA executes on current fragments.

---

## 3. Bruce-Lee-LY/cuda_hgemm (Multiple GPU archs, FP16)

**Source:** https://github.com/Bruce-Lee-LY/cuda_hgemm
**Blog:** https://bruce-lee-ly.medium.com/nvidia-tensor-core-cuda-hgemm-advanced-optimization-5a17eb77dd85
**Result:** >=95% of cuBLAS across 256-16384 dimensions

### Key Optimization Layers

1. **Pg2s (global→shared) double-buffer:** Two shared memory buffers, alternating per K iteration
2. **Ps2r (shared→register) double-buffer:** Two sets of fragment registers, load next while MMA uses current
3. **Permuted shared memory:** XOR-based swizzle for bank conflict elimination with MMA PTX ldmatrix
4. **Swizzle L2:** Block index remapping for L2 cache locality

### PTX Header (ptx.h)

The PTX helper file provides wrappers for all critical operations:

```c
// ldmatrix.x4 (loads 4 x m8n8 matrices = 16x16 tile)
// Output: 4 x uint32 registers containing packed FP16 pairs
asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
    : "r"(smem_addr));

// ldmatrix.x4.trans (loads with transpose)
asm volatile(
    "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
    : "r"(smem_addr));

// m16n8k16 MMA (FP16 input, FP32 accumulator)
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1]),
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

// m16n8k8 MMA (smaller K dimension)
asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    : "r"(a[0]), "r"(a[1]),  // Only 2 A regs for k=8
      "r"(b[0]), "r"(b[1]),  // Only 2 B regs for k=8 (but actually 1 for m16n8k8)
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
```

### Register Reuse Pattern ("Right-Left-Right-Left")

For a 2x4 MMA tile (2 M-tiles x 4 N-tiles), the inner loop alternates:
1. Load A tile [m=0], iterate N-tiles left→right: MMA with B[n=0], B[n=1], B[n=2], B[n=3]
2. Load A tile [m=1], iterate N-tiles right→left: MMA with B[n=3], B[n=2], B[n=1], B[n=0]

This pattern maximizes register reuse: B fragments loaded for the end of one sweep are still live for the start of the next sweep in reverse direction.

---

## 4. DefTruth/CUDA-Learn-Notes (LeetCUDA)

**Source:** https://github.com/DefTruth/CUDA-Learn-Notes (also https://github.com/xlite-dev/LeetCUDA)
**Key file:** `kernels/hgemm/mma/swizzle/hgemm_mma_stage_tn_swizzle_x4.cu`
**Result:** 98-100% of cuBLAS TFLOPS

### Multi-Stage Pipeline with Swizzle

The most complete open-source HGEMM using MMA PTX with all optimizations combined:
- N-stage cp.async pipeline (2-4 stages)
- XOR swizzle for bank-conflict-free ldmatrix.x4 loads
- Register double-buffer for A and B fragments
- L2 swizzle for block scheduling

This implementation achieves 98-100% of cuBLAS, demonstrating that the combination of all these techniques is sufficient to close the gap on modern architectures.

---

## 5. Alex Armbruster's GEMM Tutorial (Tesla T4, FP16)

**Source:** https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
**Result:** 96% of cuBLAS on Tesla T4

### Inner Loop Evolution

Progression from 8% to 96% cuBLAS:

| Kernel | % cuBLAS | Key Optimization |
|--------|----------|------------------|
| K1 | 8% | Naive hierarchical tiling |
| K2 | 24% | Vectorized copy + loop unroll |
| K3 | 50% | XOR swizzle eliminates bank conflicts |
| K4 | 70% | Async copy prefetch (double-buffer smem) |
| K6 | 96% | Register preload + double buffer |

### Register Pressure Progression

- K3: ~104 regs/thread (after swizzling)
- K4: 166 regs/thread (with prefetch — 62 extra for global load staging)
- K6: Similar but with register double-buffer for fragments

### Instruction Count Insight

K4 executed 216M instructions vs cuBLAS's 94M (2.3x more). The excess came from redundant index calculations during repeated ldmatrix calls. Reducing instruction count via loop unrolling and precomputed addresses was critical for the final jump.

**Lesson for our GEMM:** Instruction count itself can be a bottleneck. With math_pipe_throttle as the dominant stall, reducing non-MMA instruction count frees issue slots for MMA instructions.

---

## Summary: Common Patterns Across All Implementations

### What All High-Performance Inner Loops Share

1. **Preload all fragments before MMA burst.** Load A and B from shared→registers BEFORE executing any MMA instructions in that K iteration. Do not interleave loads and MMAs within the same K step.

2. **cp.async for global→shared overlap.** Issue cp.async for the NEXT K block, then execute MMAs for the CURRENT block. The cp.async runs in background hardware.

3. **XOR swizzle for bank-conflict-free ldmatrix.** All implementations that reach >80% cuBLAS use XOR-based shared memory permutation.

4. **MMA asm is NOT volatile; ldmatrix IS volatile.** Allowing the compiler to reorder MMA instructions enables better scheduling. ldmatrix must be volatile for correctness.

5. **4x tiling (32+ MMAs per K iteration)** gives the biggest late-stage improvement. More independent MMAs = better hardware scheduling = less synchronization overhead per MMA.

### What Differs

- spatters uses `load_matrix_x2` for B fragments; Bruce-Lee-LY uses `load_matrix_x4.trans`
- salykova uses register double-buffer (SGEMM); spatters preloads all (MMA GEMM)
- Register budgets range from ~96 (salykova FP32) to ~112+ (spatters FP16 MMA)
- Pipeline depth: 2 stages (Bruce-Lee-LY) to 3 stages (spatters) to 4+ (LeetCUDA)
