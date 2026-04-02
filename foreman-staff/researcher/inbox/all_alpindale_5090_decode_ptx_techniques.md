# RTX 5090 Decode Kernel: PTX Techniques and sm_120 Insights

**Source:** [Hitting 1,000 tokens/sec on a single RTX 5090](https://blog.alpindale.net/posts/5090_decode_optimization/)
**Relevant to:** all workers (PTX techniques applicable to sm_120)
**Date:** 2026-03-14

---

## What This Is

A detailed blog post by alpindale about building a 1,200-line CUDA megakernel for
Qwen3-0.6B inference on RTX 5090. Achieves 1,033 tokens/sec (0.97 ms/token) at
71.2% memory bandwidth utilization. While this is a decode (memory-bound) kernel,
the PTX techniques and sm_120 findings are broadly applicable.

## sm_120 Specific Findings

### 1. Vector Load Limit: 128-bit Confirmed
"sm_120 caps vector loads at 128-bits." 256-bit `uint8` loads do NOT work.
(Already documented in our sm_120 features brief, but independently confirmed.)

### 2. Memory Fence: fence.acq_rel.gpu Preferred on Blackwell

```ptx
fence.acq_rel.gpu;
```

This is preferred over `__threadfence()` on Blackwell. Lighter overhead -- establishes
ordering without full L1 flush. The author measured measurable improvement from
switching.

**Relevance:** Our kernels use `__threadfence()` in some synchronization paths.
Switching to `fence.acq_rel.gpu` in inline PTX may reduce synchronization overhead.

### 3. L1-Bypassing Loads for Weights

```ptx
ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];
```

The `L1::no_allocate` cache hint prevents weight matrix data from polluting the L1
cache, preserving it for activations and other hot data.

**Relevance:** In our GEMM and attention kernels, we load A/B matrices through
cp.async to shared memory, bypassing L1 already. But for any direct global memory
reads (epilogue reads, scale factor reads, etc.), the L1::no_allocate hint could
help preserve L1 for working data.

### 4. Fast Exponential via ex2.approx

```ptx
ex2.approx.ftz.f32 %0, %1;   // ~10x faster than expf
```

Used with `x * 1.4427f` (log2(e)) for natural exponential. Approximately 10x faster
than `expf()`.

**Relevance:** Our attention worker already knows about MUFU.EX2. This confirms the
magnitude of the speedup. The `ftz` (flush-to-zero) flag is important for avoiding
denormal slowdowns.

### 5. L2 Prefetch for Productive Spin

```ptx
prefetch.global.L2 [ptr];
```

The breakthrough technique: 112 idle blocks during the attention phase issue L2
prefetch instructions for upcoming weight data. This "productive spin" achieved
the largest single performance jump (813 -> 1,000 tok/s).

**Relevance:** This is a multi-block cooperative technique that requires persistent
kernel design. Not directly applicable to our current single-kernel approach, but
the concept of using idle compute to warm caches is interesting for future work.

### 6. Effective Bandwidth Measurement

The author achieved 1,192 GB/s effective bandwidth out of 1,792 GB/s theoretical
(66.5%). The gap comes from:
- Synchronization barriers: ~8.8 us/layer for full-grid atomics
- Instruction overhead
- Cache miss penalties

**Relevance:** This gives us a calibration point for what's achievable on RTX 5090
for memory-bound workloads.

## Techniques NOT Applicable to Our Work

- **Persistent megakernel with atomic barriers:** Our kernels are compute-bound
  (GEMM, attention), not memory-bound decode. The multi-block cooperative pattern
  is overkill for our use cases.
- **RoPE in registers via shfl_sync:** Specific to transformer decode, not applicable
  to dense linear algebra.

## Actionable Items

1. **Try `fence.acq_rel.gpu` instead of `__threadfence()`** in any kernel using
   threadfence for synchronization. Inline PTX:
   ```cuda
   asm volatile("fence.acq_rel.gpu;");
   ```

2. **Use `L1::no_allocate` for non-reused global reads** (scale factors, one-shot
   metadata reads). Inline PTX:
   ```cuda
   asm volatile("ld.global.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(ptr));
   ```
