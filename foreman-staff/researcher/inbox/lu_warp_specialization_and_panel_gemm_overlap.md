# Warp Specialization for Monolithic LU: Panel-GEMM Overlap

**Source:** https://arxiv.org/abs/2512.18134 (Twill: Optimal SWP and Warp Specialization, Dec 2025)
**Source:** https://arxiv.org/abs/2510.14719 (Tawa: Automatic Warp Specialization, Oct 2025)
**Source:** https://arxiv.org/abs/2506.11209 (Performance Model for WS Kernels, Jun 2025)
**Source:** https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
**Relevant to:** LU worker
**Worker's current problem:** In a cooperative-groups monolithic LU kernel, the panel factorization leaves most SMs idle. Need techniques to overlap panel work with other useful computation.

---

## What This Is

Warp specialization (WS) is a technique where different warps within a thread
block (or across blocks) perform different tasks simultaneously. Recent research
(2025) has produced automated tools for generating warp-specialized kernels on
Blackwell. While primarily targeting GEMM and attention, the principles apply
to factorization kernels that combine panel and GEMM phases.

---

## Warp Specialization Primer

### The Idea

Instead of all warps doing the same work:
```
Traditional: [warp0: GEMM] [warp1: GEMM] [warp2: GEMM] [warp3: GEMM]
```

Assign different roles:
```
Specialized: [warp0: load data] [warp1: MMA compute] [warp2: MMA compute] [warp3: store results]
```

### On Blackwell (sm_120)

- 48 warps per SM (compute capability 12.0)
- 32 max blocks per SM
- No TMEM, no wgmma (those are sm_100 datacenter features)
- WS is done manually via warp-level control flow

For Flash Attention on Blackwell, 5+ warp roles are used:
1. Loading K/V data
2. Issuing MMA operations
3. Computing softmax
4. Scaling intermediate results
5. Storing output

---

## Application to Monolithic LU

### The Panel Idle Problem

In our cooperative-groups design:
- Phase 1 (panel): Block 0 does panel, blocks 1-339 idle
- Phase 2 (LASWP+TRSM): All blocks participate
- Phase 3 (trailing GEMM): All blocks participate

During Phase 1, 169 SMs are completely idle. This is wasted compute.

### Look-Ahead via Block-Level Specialization

Instead of strict phases with grid.sync(), use block-level specialization:

```
Block 0: Panel factorization for iteration k
Blocks 1-339: Trailing GEMM from iteration k-1 (still completing)

// Only sync when: panel k done AND trailing GEMM k-1 done
```

This is the HPL look-ahead pattern, but implemented via block specialization
rather than separate kernel launches.

### Implementation Sketch

```cpp
__global__ void monolithic_lu(float* A, int N, int NB) {
    cg::grid_group grid = cg::this_grid();

    for (int k = 0; k < N/NB; k++) {
        if (blockIdx.x == 0) {
            // Panel block: factorize panel k
            panel_factorize(A, k, NB);
            // Signal completion via atomic flag
            atomicExch(&panel_done[k], 1);
        } else {
            if (k > 0) {
                // GEMM blocks: finish trailing update from iteration k-1
                // (that wasn't completed before grid.sync)
                finish_trailing_gemm(A, k-1, NB, blockIdx.x);
            }
            // Wait for panel k to complete
            while (atomicAdd(&panel_done[k], 0) == 0) { /* spin */ }
        }

        // All blocks: LASWP + TRSM for iteration k
        distributed_laswp_trsm(A, k, NB);
        grid.sync();

        // All blocks: START trailing GEMM for iteration k
        // Do look-ahead columns first (next panel's columns)
        trailing_gemm_lookahead(A, k, NB, blockIdx.x);
        grid.sync();
    }
}
```

### Warp Specialization Within the Panel Block

Within block 0's panel factorization, warps can be specialized:

```
Warp 0-1: Argmax reduction (pivot search)
Warp 2-3: Row swap execution (loading/storing swapped rows)
Warp 4-5: Column scaling (divide by pivot)
Warp 6-7: Rank-1 update (trailing panel columns)
```

This pipelines the per-column steps: while warps 0-1 are finding the next
column's pivot, warps 6-7 are finishing the current column's rank-1 update.

---

## Twill and Tawa (2025 Research)

### Twill (Dec 2025)

Twill is a compiler that automatically derives optimal software pipelining
and warp specialization schedules. Key finding: Twill can automatically
rediscover the expert-designed schedules for Flash Attention on both Hopper
and Blackwell.

**Relevance to LU:** Twill targets iterative kernels with loop-carried
dependencies -- exactly like the column loop in panel factorization. However,
Twill currently targets GEMM-like workloads, not factorization. The algorithmic
principles (identifying pipeline stages, balancing warp groups) are transferable.

### Tawa (Oct 2025)

Tawa generates warp-specialized code from high-level tile-based programs.
"For Blackwell and later architectures that provide more warp roles, Tawa can
create additional partitions."

**Relevance:** Tawa's tile abstraction could potentially express the panel
factorization + trailing GEMM composition, but this is speculative. The tools
are research prototypes focused on GEMM/attention, not factorization.

---

## Practical Recommendations

### For v3 Monolithic Kernel (Priority Order)

1. **Start with simple grid.sync() phases** (panel -> LASWP+TRSM -> GEMM).
   Measure baseline.

2. **Add look-ahead** via atomic-flag panel completion (block 0 signals, other
   blocks start trailing GEMM for column k while panel k+1 runs). This is
   the highest-value optimization.

3. **Warp specialization within panel block** is a micro-optimization. Only
   pursue if panel factorization is measured to be the bottleneck (>20% of
   total time).

4. **Do NOT use Twill or Tawa** for the factorization kernel. These tools
   target GEMM/attention patterns. Write the warp specialization manually
   using explicit warp-level control flow.

### sm_120 Constraints for WS

- 48 warps per SM, but we typically use 8 warps (256 threads) per block
- With 2 blocks/SM: 16 warps actively scheduled
- Warp specialization within an 8-warp block: 4 roles with 2 warps each
  (reasonable granularity)
- Register file: 64K regs / 48 warps = 1365 regs per warp max. With 8
  warps per block: 8192 regs per block = plenty.

---

## Caveats

1. **Look-ahead with atomic flags introduces complexity.** Spin-waiting on
   global memory flags wastes cycles and requires careful occupancy control
   (all blocks must be resident). Cooperative launch guarantees this.

2. **Warp specialization requires warp-level divergence.** The compiler may
   not optimize specialized paths as well as uniform ones. Profile with nsys
   to verify warp utilization.

3. **Panel-GEMM overlap benefit depends on ratio.** For N=4096 NB=64, the
   panel is ~0.5ms and the trailing GEMM is ~2.7ms (with BF16x9). Overlapping
   the panel hides at most 0.5ms -- an 8% improvement. Worth doing but not
   transformative.

---

## Sources

- [Twill: Optimal SWP and Warp Specialization (Dec 2025)](https://arxiv.org/abs/2512.18134)
- [Tawa: Automatic Warp Specialization (Oct 2025)](https://arxiv.org/abs/2510.14719)
- [WS Kernel Performance Model (Jun 2025)](https://arxiv.org/abs/2506.11209)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [Unweaving Warp Specialization Blog](https://rohany.github.io/blog/warp-specialization/)
