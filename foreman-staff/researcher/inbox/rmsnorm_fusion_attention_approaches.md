# RMSNorm Fusion with Attention: Approaches and Precedents

**Source:** Mirage auto-fusion framework, cuDNN v9.7 fusion support, FlexAttention prologue hooks
**Relevant to:** rmsnorm worker
**Worker's current problem:** Hit 4.1 μs pipelining floor. Standalone kernel is optimal. Next step is fusing RMSNorm into the attention kernel to eliminate the 4.1 μs launch entirely.

## What This Is

Survey of existing approaches for fusing normalization (RMSNorm/LayerNorm) directly
into attention kernels as a "prologue" operation — computing normalization on-the-fly
as Q, K, or V data is loaded.

## Why It Matters for Us

The rmsnorm kernel is at the hardware floor (4.1 μs). Further standalone optimization
is impossible. The only way to eliminate this cost is to fuse it into a downstream
consumer (attention). The fusion saves the entire 4.1 μs by computing normalization
within the attention kernel's input loading path.

## Key Approaches

### Approach 1: Q Prologue Fusion (Simplest)

Fuse RMSNorm into the attention kernel's Q loading path. Q is loaded once and reused
across all KV blocks, so the normalization cost is amortized.

```
Standard:                              Fused:
Q = rms_norm(X_q, w) (separate)       attention_kernel:
K = rms_norm(X_k, w) (separate)         load Q tile from global
V = X_v              (separate)          compute RMS of Q tile (in registers)
O = attention(Q, K, V)                   normalize Q in registers
                                         for each KV block:
                                           load K, V, compute attention
```

**Implementation sketch:**
1. Load Q tile rows into registers (already done in attention kernel)
2. Compute sum-of-squares via warp shuffle reduction (same as rmsnorm kernel)
3. Compute rsqrt(mean_sq + eps) × weight per element
4. Normalize Q values in registers before converting to MMA fragments

**Register overhead:** Need to hold raw Q values (FP32) while computing RMS, then
convert to BF16. Current attention kernel loads Q as BF16 via ldmatrix. The fused
version would need to load FP32 Q, normalize, convert to BF16, then build MMA
fragments. This adds ~8 registers for partial sums and the weight vector.

**Challenge:** ldmatrix expects BF16 data in shared memory. If Q arrives as FP32
(pre-normalization), we can't use ldmatrix. Options:
1. Normalize Q on the host/in a prologue kernel, store as BF16, load with ldmatrix
   (but this defeats the purpose — we're back to a separate kernel)
2. Load Q as FP32 from global → registers, normalize, convert to BF16, store to
   smem as BF16, then ldmatrix into MMA fragments. This adds a global→reg→smem
   round trip but only for Q (loaded once, amortized across all KV blocks).
3. Compute RMSNorm at the start of the attention kernel using cooperative threads,
   then proceed with standard ldmatrix-based attention. This is essentially a
   two-phase kernel: normalize Q → compute attention.

### Approach 2: K Prologue Fusion (Higher Value)

K is loaded per KV block (not reused like Q), so K normalization cost isn't amortized.
Fusing K normalization saves one kernel launch per attention call.

Same challenge as Q: need to normalize FP32 data before ldmatrix can load BF16.
Could do an in-smem normalize: load K_block as FP32, normalize cooperatively,
write BF16 to smem, then ldmatrix.

### Approach 3: Mirage-style Automatic Fusion

Mirage (from Zhihao Jia's group) automatically discovers fusion opportunities.
Their fused RMSNorm + Linear kernel achieves 1.5-1.7x over separate operations.
The technique: fuse the normalization into the GEMM prologue, computing norm
values on-the-fly as matrix tiles are loaded.

**For attention:** The analog is computing RMSNorm during the Q (or K) tile loading
phase of the attention kernel. This is conceptually simple but requires careful
engineering to avoid increasing register pressure beyond the 170.7 threshold (3
blocks/SM).

### Approach 4: cuDNN Prologue Fusion

cuDNN v9.7+ supports RMSNorm as a "prologue" to attention. This means NVIDIA's
own FlashAttention implementation can accept un-normalized inputs and apply
RMSNorm internally. We don't use cuDNN, but the existence of this feature
validates that the fusion is architecturally sound.

## Practical Recommendation

**Start with Approach 3 variant — two-phase kernel:**

```
Phase 1 (first 256 threads, ~500 cycles):
  Load Q rows from global to registers
  Compute RMS via warp reduction
  Normalize and convert to BF16
  Store normalized BF16 Q to shared memory
  __syncthreads()

Phase 2 (main attention loop):
  Load Q from shared memory via ldmatrix (standard path)
  ... rest of attention kernel unchanged ...
```

This preserves the entire attention kernel structure while fusing RMSNorm into
the prologue. Phase 1 runs once, Phase 2 is the standard attention loop.

**Cost analysis:**
- Phase 1 adds ~2 μs (256 threads computing RMS + normalize + store)
- Phase 1 eliminates: separate RMSNorm kernel launch (4.1 μs) + Q global read (shared)
- Net savings: ~2 μs per attention call

## Caveats

1. **Register pressure.** Phase 1 needs registers for FP32 Q values + partial sums.
   These registers are freed before Phase 2 starts (after __syncthreads and Q
   written to smem). The compiler should handle this via register reuse.

2. **Shared memory.** Q is already stored in shared memory for Phase 2. Phase 1
   writes to the same buffer. No additional smem needed.

3. **Only worth it if RMSNorm and attention are always paired.** If RMSNorm feeds
   into multiple consumers, separate kernels are more flexible.

4. **Backward pass.** Fusing RMSNorm into attention complicates the backward pass.
   Consider whether training workloads need this fusion.
