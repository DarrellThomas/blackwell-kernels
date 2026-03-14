# Reference: gau-nernst — Flash Attention on RTX 5090

**Source:** https://gau-nernst.github.io/fa-5090/
**Author:** Thien Tran (gau-nernst)
**Attribution:** Techniques from this work should be cited in code and commit messages.

---

## Result: 197.74 TFLOPS (94.39% of 209.5 TFLOPS peak)

Five optimization versions, each guided by NCU profiling.

## Configuration

- BLOCK_Q = 128, DIM = 128, NUM_WARPS = 4, TB_SIZE = 128 threads
- BLOCK_KV varies: 64 → 32 → 64 across versions
- mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 (same ISA as us)

## Version Progression

| Ver | TFLOPS | % SOL | Key Change | Bottleneck After |
|-----|--------|-------|------------|------------------|
| v1 | 142.87 | 68.2% | Basic cp.async + ldmatrix | Short Scoreboard (bank conflicts) |
| v2 | 181.11 | 86.5% | XOR swizzle eliminates 8-way bank conflicts | Long Scoreboard (global mem latency) |
| v3 | 189.84 | 90.6% | Double-buffered K/V, BLOCK_KV→32 | Math Pipe Throttle (good!) |
| v4 | 194.33 | 92.8% | ldmatrix.x4 for K and V (fewer instructions) | Math Pipe Throttle |
| v5 | 197.74 | 94.4% | Single-buffer V, double-buffer K, BLOCK_KV→64 | Math Pipe Throttle |

cuDNN reference: 203.61 TFLOPS (97.2% SOL)

## Key Technical Insights

### 1. Instruction Count Reduction (v4: +2.5%)
Replacing two `ldmatrix.x2` with one `ldmatrix.x4` for K and V gave a "non-trivial improvement"
despite identical arithmetic intensity. **Instruction scheduling throughput is itself a bottleneck
on sm_120.** Fewer instructions to issue = more time for MMA.

### 2. Asymmetric Buffering (v5: +1.8%)
Only K needs double-buffering. V prefetch can overlap with the first MMA (Q*K^T) execution.
- Single buffer for V (prefetched during Q*K^T MMA)
- Double-buffered K (prefetched during P*V MMA)
- This frees shared memory, allowing BLOCK_KV back to 64 (from 32)

### 3. Shared Memory Overlap
Q_smem overlaps with (K_smem + V_smem) since Q is loaded once and reused from registers.
This is how everything fits in ~96KB shared memory.

### 4. P→A Packing Requires No Shuffle
Within an 8x8 tile, the accumulator layout (m16n8) and multiplicand A layout (m16k16) are
identical. Converting P (softmax output) to A-fragments for P*V needs no inter-thread data
movement — just in-register BF16 conversion.

### 5. Online Softmax
- Thread-level reduction across columns
- Butterfly reduction within 4-thread groups: `__shfl_xor_sync()` with masks 1, 2
- Rescaling: `O *= exp(m_old - m_new)`

### 6. Execution Order Per KV Block (v3+)
1. Prefetch K[i+1] (after wait)
2. MMA: Q @ K^T (first MMA)
3. Prefetch V[i+1] (before softmax)
4. Online softmax
5. MMA: P @ V (second MMA)

## Comparison to Our Kernel

| Aspect | gau-nernst | Our kernel |
|--------|-----------|------------|
| BLOCK_Q | **128** | 64 |
| BLOCK_KV | 64 (v5) | 32 |
| Warps | 4 | 4 |
| K loading | ldmatrix.x4 | scalar manual |
| V loading | ldmatrix.x2_trans | ldmatrix.x2_trans |
| P conversion | Register-only | Register-only (v7) |
| Swizzle | XOR | XOR |
| Buffering | Asymmetric (K=2, V=1) | Double-buffer K/V |
| Performance | 94.4% SOL | ~58% SOL |

### Gap Analysis — What Would Close It

1. **BLOCK_Q=128** — Our biggest structural difference. Doubles Q reuse per KV block.
   This is the single most impactful change but requires careful smem budget management.

2. **ldmatrix.x4 for K** — We currently load K with scalar manual packing. Switching to
   ldmatrix.x4 reduces instruction count (the v4 insight).

3. **Asymmetric buffering** — Single-buffer V frees smem for larger tiles.

4. **BLOCK_KV=64** — With asymmetric buffering, we can double KV tile size without
   exceeding smem budget. This halves the number of KV iterations.
