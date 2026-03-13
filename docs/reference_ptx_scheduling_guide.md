# PTX Instruction Scheduling Guide — Practical Techniques for sm_120

**Purpose:** How to manually schedule MMA and memory instructions for maximum tensor core throughput, with specific techniques applicable to our BF16 GEMM at 0.80x cuBLAS.
**Last updated:** 2026-03-13

---

## 1. The Scheduling Problem

### What math_pipe_throttle Means

The warp scheduler wants to issue an MMA instruction, but the tensor core's input FIFO is already full. This happens when MMA instructions arrive in **bursts** — the pipe saturates during bursts, then starves between them.

### The Ideal: Uniform MMA Distribution

Instead of:
```
LDSM LDSM LDSM LDSM  MMA MMA MMA MMA MMA MMA MMA MMA  (burst → throttle)
```

We want:
```
LDSM MMA LDSM MMA LDSM MMA LDSM MMA LDSM MMA LDSM MMA  (uniform → no throttle)
```

This keeps the tensor core input FIFO partially filled at all times instead of overflowing then draining.

### Three Levels of Scheduling Control

| Level | Mechanism | Effort | Expected Gain |
|-------|-----------|--------|---------------|
| 1. Compiler-driven | `#pragma unroll`, non-volatile MMA | Low | ~80-93% peak |
| 2. C++ structure | Loop ordering, register staging | Medium | ~90-95% peak |
| 3. Full PTX | Single asm block with manual scheduling | High | ~95-100% peak |

Our current GEMM is at level 1-2 (0.80x cuBLAS). Closing the gap requires level 2+ techniques.

---

## 2. Compiler-Driven Scheduling (What We Already Have)

### Making MMA Non-Volatile

**Source:** https://www.spatters.ca/mma-matmul

The single most important scheduling decision:

```c
// CORRECT — compiler can reorder freely
asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ..."
    : "=f"(D[0]), ... : "r"(A[0]), ... );

// WRONG for scheduling — prevents reordering
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ..."
    : "=f"(D[0]), ... : "r"(A[0]), ... );
```

With non-volatile MMA, the compiler (`nvcc -O2` with `#pragma unroll`) naturally interleaves:
- MMA instructions from different (m,n) tile coordinates
- ldmatrix loads for the next K-step between MMAs for the current K-step
- cp.async commit/wait between MMAs

**This alone gets spatters.ca to 93% peak.** The compiler's scheduler is surprisingly good at sm_89/sm_120 when given freedom.

### Pragma Unroll

```c
#pragma unroll
for (int m = 0; m < M_TILES; m++) {
    #pragma unroll
    for (int n = 0; n < N_TILES; n++) {
        mma_m16n8k16(a_frag[m], b_frag[n], acc[m][n]);
    }
}
```

Unrolling exposes all independent MMA instructions to the compiler simultaneously. The compiler sees the full dependency graph and can schedule optimally.

**Without unroll:** The compiler may keep the loop structure, issuing MMAs sequentially and missing scheduling opportunities.

---

## 3. C++ Structural Scheduling

### Technique: Preload All Fragments Before MMA

Load ALL shared → register fragments, THEN execute ALL MMAs:

```c
// Phase 1: All loads (fill register file)
for (m = 0; m < M_TILES; m++)
    load_matrix_x4(a_frag[m], smem_a_addr[m]);
for (n = 0; n < N_TILES; n++)
    load_matrix_x2(b_frag[n], smem_b_addr[n]);

// Phase 2: Issue cp.async for next K block (non-blocking)
cp_async(...);
cp_async_commit();

// Phase 3: All MMAs (compiler interleaves with pending cp.async)
for (m = 0; m < M_TILES; m++)
    for (n = 0; n < N_TILES; n++)
        mma_m16n8k16(a_frag[m], b_frag[n], acc[m][n]);
```

**Why this works:** The MMAs all have their inputs ready in registers. The compiler can issue them in any order, interleaving with the background cp.async. No MMA stalls on data dependencies.

**Tradeoff:** Higher register pressure. All fragments must be live simultaneously.

### Technique: Interleave Loads Between MMA Groups

Instead of all-loads-then-all-MMAs, interleave at the M-tile level:

```c
for (m = 0; m < M_TILES; m++) {
    load_matrix_x4(a_frag, smem_a_addr[m]);    // Load A for this M tile
    for (n = 0; n < N_TILES; n++) {
        load_matrix_x2(b_frag, smem_b_addr[n]); // Load B for this N tile
        mma_m16n8k16(a_frag, b_frag, acc[m][n]);
    }
}
```

**Why this helps:** Loads and MMAs alternate naturally. The compiler sees each load-MMA pair close together and can overlap them.

**Tradeoff:** Less freedom for the compiler. A fragments are only live for one M-tile's worth of MMAs.

### Technique: Register Double-Buffer for Fragments

**Source:** https://salykova.github.io/sgemm-gpu (key technique that beat cuBLAS)

```c
unsigned a_frag[2][4];  // Two sets of A fragment registers
unsigned b_frag[2][2];  // Two sets of B fragment registers

// Load first set
load_matrix_x4(a_frag[0], addr_k0);
load_matrix_x2(b_frag[0], addr_k0);

for (int k = 0; k < K_TILES; k++) {
    int curr = k % 2;
    int next = (k + 1) % 2;

    // Load NEXT fragments (k+1) while MMA uses CURRENT (k)
    if (k + 1 < K_TILES) {
        load_matrix_x4(a_frag[next], addr_next_k);
        load_matrix_x2(b_frag[next], addr_next_k);
    }

    // MMA with current fragments
    mma_m16n8k16(a_frag[curr], b_frag[curr], acc);
}
```

**Register cost:** Doubles fragment register usage (4+2 → 8+4 = +6 registers per tile).
**Benefit:** Hides shared memory load latency behind MMA execution.

**Warning from our experience (04_HARD_WON_LESSONS.md):** On sm_120, B-fragment double-buffering regressed performance from 0.89x to 0.68x cuBLAS. The extra 16 registers (139 vs 123) dropped occupancy, and `asm volatile` barriers prevented compiler reordering. The hardware warp scheduler with 8 warps already overlaps ldmatrix and mma.sync across warps.

**Lesson:** Register double-buffering helps when (a) few warps available for latency hiding, OR (b) the compiler isn't already interleaving well. On sm_120 with enough warps, the hardware scheduler does a better job than manual double-buffering.

---

## 4. The Right-Left-Right-Left Pattern

**Source:** https://github.com/Bruce-Lee-LY/cuda_hgemm

For a 2D MMA tile grid (M_TILES × N_TILES), the traversal order affects register reuse:

```
Standard (row-major):
  m=0: n=0 → n=1 → n=2 → n=3
  m=1: n=0 → n=1 → n=2 → n=3

Right-Left (serpentine):
  m=0: n=0 → n=1 → n=2 → n=3  (right)
  m=1: n=3 → n=2 → n=1 → n=0  (left)
```

**Why:** When moving from m=0 to m=1 in standard order, B[n=0] must be reloaded despite being used moments ago. In serpentine order, B[n=3] from the end of m=0's sweep is immediately reused at the start of m=1's sweep.

**Register savings:** One less B-fragment reload per M-tile transition. For 4×4 tiles, this saves 3 ldmatrix.x2 instructions per K iteration.

**Applicability:** Most useful when B fragments are expensive to reload (high latency to shared memory). Less useful when all fragments are preloaded into registers before the MMA block.

---

## 5. Full PTX Inner Loop (The Nuclear Option)

### When to Consider It

- Already at >90% of peak via C++/compiler techniques
- math_pipe_throttle is dominant stall
- SASS inspection shows suboptimal instruction ordering that can't be controlled from C++
- Specific instruction sequences (e.g., MMA interleaved with MUFU.EX2 for softmax) need exact ordering

### Structure of a Full PTX Inner Loop

```c
asm volatile(
    "{\n"
    // Declare ALL registers explicitly
    "  .reg .u32 a0, a1, a2, a3;\n"     // A fragment
    "  .reg .u32 a4, a5, a6, a7;\n"     // A fragment (next K)
    "  .reg .u32 b0, b1;\n"             // B fragment
    "  .reg .u32 b2, b3;\n"             // B fragment (next K)
    "  .reg .f32 d0, d1, d2, d3;\n"     // Accumulator tile [0,0]
    "  .reg .f32 d4, d5, d6, d7;\n"     // Accumulator tile [0,1]
    // ... more accumulators ...
    "\n"
    // Initialize accumulators from C++ inputs
    "  mov.f32 d0, %0;\n"
    "  mov.f32 d1, %1;\n"
    // ...
    "\n"
    // MANUALLY SCHEDULED INNER LOOP
    // Interleave loads and MMAs for optimal scheduling:
    "\n"
    "  ldmatrix.sync.aligned.x4.m8n8.shared.b16 {a0,a1,a2,a3}, [%N];\n"
    "  ldmatrix.sync.aligned.x2.m8n8.shared.b16 {b0,b1}, [%M];\n"
    "\n"
    "  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
    "    {d0,d1,d2,d3}, {a0,a2,a1,a3}, {b0,b1}, {d0,d1,d2,d3};\n"
    //                   ^^ a1/a2 swap!
    "\n"
    "  ldmatrix.sync.aligned.x2.m8n8.shared.b16 {b2,b3}, [%P];\n"  // Load NEXT B
    "\n"
    "  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
    "    {d4,d5,d6,d7}, {a0,a2,a1,a3}, {b2,b3}, {d4,d5,d6,d7};\n"
    //   ^^ DIFFERENT accumulator, SAME A, NEXT B — independent of first MMA
    "\n"
    // ... continue interleaving ...
    "\n"
    // Write accumulators back to C++ outputs
    "  mov.f32 %0, d0;\n"
    // ...
    "}\n"
    : "+f"(acc[0][0]), "+f"(acc[0][1]), ...
    : "r"(smem_addr_a), "r"(smem_addr_b0), "r"(smem_addr_b1), ...
);
```

### Challenges

1. **Register naming:** All registers inside the asm block must be named explicitly. The compiler doesn't optimize across the asm boundary.

2. **No loop constructs:** PTX loops (`bra`) inside inline asm are fragile. Better to fully unroll in the C++ preprocessor and emit straight-line PTX.

3. **Operand limit:** Each asm statement has a limit on operand count (varies by compiler version). For large inner loops, may need to split into multiple asm blocks with explicit register passing.

4. **Debugging:** No source-level debugging inside asm blocks. Printf via PTX is possible but painful.

5. **Maintenance:** Every architectural change (tile size, K-step, etc.) requires rewriting the entire asm block.

### Key Insight from Our Experience

Our attention kernel's SASS analysis confirmed the compiler already:
- Interleaves QK^T HMMA with K LDSM loads
- Hoists V LDSM loads into the softmax gap
- Interleaves exp2f (MUFU.EX2) with PV HMMA

The remaining 6% gap (68 μs vs 64 μs compiler ceiling) is from suboptimal ordering that C++ can't control. Full PTX would target this gap, but for the GEMM kernel at 0.80x cuBLAS, there's likely more to gain from C++ structural changes first.

---

## 6. Scheduling Strategy for Our BF16 GEMM

### Current State

- 0.80x cuBLAS, math_pipe_throttle dominant
- MMA instructions burst during the compute phase, then starve during loads
- Compiler is doing some interleaving but not enough

### Recommended Approach (Progressive)

**Step 1: Ensure MMA is non-volatile.** Check that our MMA wrapper does NOT use `asm volatile`. If it does, remove volatile. (Highest impact, lowest effort.)

**Step 2: Preload all fragments.** Load ALL A and B fragments into registers before executing any MMAs. This gives the compiler maximum freedom to schedule. (Medium effort, moderate impact.)

**Step 3: Increase tiling.** If currently using 2×2 MMA tiling (8 MMAs per K-step), move to 4×4 (32 MMAs per K-step). This was the biggest late-stage improvement in spatters.ca (89.5% → 100% cuBLAS). (Medium effort, potentially large impact.)

**Step 4: Register double-buffer evaluation.** Try loading K+1 fragments while executing K MMAs. Monitor register count — if it causes spills or occupancy loss, revert. Our past experience (04_HARD_WON_LESSONS.md) shows this can backfire on sm_120. (Medium effort, uncertain impact.)

**Step 5: SASS inspection.** Dump SASS and check if the compiler is actually interleaving HMMA with LDSM instructions. If not, restructure the C++ loop to force better ordering. (Low effort to inspect, medium effort to fix.)

```bash
# Dump SASS for inspection
CUDA_HOME=/usr/local/cuda-13 /usr/local/cuda-13/bin/cuobjdump -sass build/lib*/blackwell_kernels*.so | grep -A 5 'HMMA\|LDSM'
```

**Step 6: Full PTX inner loop (if steps 1-5 plateau).** Write the entire K-loop body in a single asm block with manual register allocation and instruction ordering. Target: eliminate all scheduling gaps between MMA instructions.

---

## 7. Measuring Scheduling Quality

### Nsight Compute Metrics

```bash
ncu --metrics \
  smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio,\
  smsp__warp_issue_stalled_wait_per_warp_active.ratio,\
  smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio,\
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio,\
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed \
  ./kernel
```

| Metric | Good | Bad | Meaning |
|--------|------|-----|---------|
| math_pipe_throttle | <20% | >40% | MMA bursts saturating FIFO |
| wait | <15% | >30% | Data dependency stalls |
| short_scoreboard | <10% | >20% | Shared memory latency |
| tensor_pipe_active | >70% | <50% | Tensor core utilization |

### SASS Interleaving Check

Look for alternating HMMA and LDSM instructions in the SASS dump:

**Good scheduling:**
```
HMMA.1688.F32  // MMA tile [0,0]
LDSM.16.MT88   // Load next B fragment
HMMA.1688.F32  // MMA tile [0,1]
LDSM.16.M88    // Load next A fragment
HMMA.1688.F32  // MMA tile [1,0]
```

**Bad scheduling:**
```
LDSM.16.MT88   // Load all fragments
LDSM.16.MT88
LDSM.16.M88
LDSM.16.M88
HMMA.1688.F32  // Then burst all MMAs
HMMA.1688.F32
HMMA.1688.F32
HMMA.1688.F32
```

---

## References

- spatters.ca MMA matmul (scheduling via non-volatile MMA): https://www.spatters.ca/mma-matmul
- salykova SGEMM (register double-buffer): https://salykova.github.io/sgemm-gpu
- Bruce-Lee-LY cuda_hgemm (right-left-right-left): https://github.com/Bruce-Lee-LY/cuda_hgemm
- math_pipe_throttle guide (our docs): `/data/src/bwk/main/docs/math_throttle_optimization.md`
- hard-won lessons (our docs): `/data/src/bwk/main/.claude/04_HARD_WON_LESSONS.md`
- GTC 2020 tensor core optimization: https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
- NVIDIA inline PTX guide: https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
