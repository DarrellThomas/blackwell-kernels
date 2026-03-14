# Blackwell Microbenchmarking Paper

**Source:** arXiv:2512.02189v1 — "Microbenchmarking NVIDIA Blackwell GPU"
**Scope:** Datacenter B200 (sm_100) characterization. sm_120 (consumer) uses mma.sync instead of tcgen05, but memory hierarchy and some architectural insights transfer.

**Key caveat:** This paper benchmarks the B200 datacenter GPU which uses `tcgen05` tensor core ISA and has TMEM (Tensor Memory). The RTX 5090 (sm_120) does NOT have TMEM and uses `mma.sync`. Tensor core latency numbers here are for `tcgen05` and will differ from `mma.sync` on sm_120. Memory hierarchy results are more transferable.

---

## 1. Key Findings

- 1.56x higher mixed-precision training throughput vs H200
- 42% better energy efficiency than H200
- 58% reduction in memory access latency for cache-misses
- 2.5x throughput improvement with FP4 quantization (Mistral-7B)
- Tensor core latency nearly constant across tile sizes (11.0-11.4 cycles) -- spatial array design, not deeper pipeline

---

## 2. Hardware Specifications (B200 Datacenter)

- 208 billion transistors, dual-die configuration
- 148 SMs across 8 GPCs
- 4 L2 cache partitions (vs 2 in Hopper)
- 8 HBM3e memory stacks, unified 192 GB
- 256 KB Tensor Memory (TMEM) per SM -- dedicated on-chip
- TMEM bandwidth: 16 TB/s read, 8 TB/s write per SM
- 5th-generation Tensor Cores using `tcgen05` instruction set

**Note for sm_120:** Consumer Blackwell has 170 SMs (vs 148), 32 GB GDDR7 (vs 192 GB HBM3e), 128 KB shared memory per SM, NO TMEM. The tensor core hardware is similar but accessed via `mma.sync` instead of `tcgen05`.

---

## 3. Tensor Core Latency and Throughput

### Single-Instruction Latency (tcgen05.mma on B200)

| Precision | Latency (cycles) | Throughput (TFLOPS) | % Peak |
|-----------|-----------------|---------------------|--------|
| FP64 | 11.0-11.4 | 44.8 | 99.6% |
| FP32 | ~11.4 | 481.2 | 96.2% |
| BF16 | ~11.4 | 1926.8 | 96.3% |
| FP16 | 11.2 | 1929.2 | 96.5% |
| FP8 | 11.8 | 3851.4 | 96.3% |
| FP6 | 12.3 | 5134.8 | 95.8% |
| FP4 | 12.6 | 7702.5 | 96.3% |
| INT8 | 11.9 | 3927.1 | 98.2% |

**Critical insight:** Latency remains nearly constant across tile sizes (11.0-11.4 cycles), whereas Hopper scales linearly. This indicates a spatial array design where throughput is achieved through wider datapaths, not deeper pipelining.

### Hopper vs Blackwell Comparison (m64n64k16, FP16)

| Architecture | Instruction | Scope | Latency |
|-------------|-------------|-------|---------|
| Hopper | wgmma | Warp-group (128 threads) | 32.0 cycles |
| Blackwell | tcgen05.mma | Single warp | 11.0 cycles |

**2.9-11.6x lower single-instruction latency** on Blackwell.

### Relevance to sm_120

sm_120's `mma.sync.aligned.m16n8k16` has been measured at approximately 32 cycles on Ada (sm_89). sm_120 likely has similar latency since both use the `mma.sync` extended Ampere ISA. The B200's 11-cycle `tcgen05` latency does NOT apply to sm_120.

---

## 4. Detailed Precision Characterization

| Input A/B | Accumulator | Tile Shape | Latency (cycles) | Throughput (TFLOPS) |
|-----------|-------------|-----------|-----------------|---------------------|
| FP16 | FP16 | m64n8k16 | 11.2 | 964.8 |
| FP16 | FP32 | m64n8k16 | 11.5 | 482.4 |
| BF16 | FP32 | m64n8k16 | 11.4 | 481.6 |
| FP8 | FP16 | m64n8k16 | 11.8 | 1925.3 |
| FP8 | FP32 | m64n8k16 | 12.1 | 1912.8 |
| FP6 | FP16 | m64n8k16 | 12.3 | 2567.2 |
| FP4 | FP16 | m64n8k16 | 12.6 | 3850.1 |
| INT8 | INT32 | m64n8k16 | 11.9 | 3928.5 |

### Accumulator Bottleneck

FP16 inputs with FP32 accumulator vs FP16 accumulator: **throughput halved** (1929.2 -> 964.6 TFLOPS). The accumulator datapath, not the multiply units, is the throughput limiter. This means using FP32 accumulators (as we do for attention) costs 50% of peak tensor throughput.

**Implication for sm_120:** Our `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` uses FP32 accumulators. If an FP16 accumulator variant exists on sm_120, it would double tensor throughput (but reduce numerical precision).

---

## 5. Memory Subsystem

### Tensor Memory (B200 Only, NOT on sm_120)

- Access latency: 420 clock cycles (cache-miss)
- Read bandwidth: 16 TB/s per SM
- Write bandwidth: 8 TB/s per SM
- Optimal tile: 64x64 elements (4 KB FP8)
- Underutilized (<32x32): 45% peak bandwidth
- Keeps intermediate results on-chip, saving 12 TB/s data movement vs Hopper

### Global Memory Bandwidth

**STREAM Triad results:**

| Array Size | B200 (TB/s) | H200 (TB/s) | B200 % Peak |
|-----------|-----------|-----------|-----------|
| 4 GB | 4.134 | 2.88 | 51.7% |
| 64 GB | 7.42 | 4.35 | 92.8% |
| 128 GB | 7.48 | 4.38 | 93.5% |

B200 peak: 8.0 TB/s (HBM3e). RTX 5090 peak: 1.792 TB/s (GDDR7).

---

## 6. Warp Scheduler Observations

### Architectural Shift from Hopper

- **Hopper:** `wgmma` requires warp-group synchronization (128 threads, 4 warps)
- **Blackwell (B200):** `tcgen05.mma` allows per-thread dispatch (single warp)
- Result: 18-23% reduction in scheduler stalls for memory-bound kernels
- **sm_120:** `mma.sync` also requires only single warp (32 threads), similar to B200

### CTA Pair Execution (B200)

- Two CTAs share operands via intra-TPC communication
- Reduces redundant data movement
- Maps to Tensor Processing Cluster (TPC)
- Not available on sm_120 in the same form

---

## 7. SASS Instruction Mapping

| Precision | SASS Instruction (B200) | Hopper Equivalent |
|-----------|------------------------|-------------------|
| FP64 | DMMA | DMMA |
| FP32 | HMMA | HGMMA |
| FP8 | QMMA | QGMMA |
| INT8 | IMMA | IGMMA |
| FP4 | OMMA | N/A (new) |
| FP6 | (unnamed) | N/A (new) |

**sm_120 SASS:** Uses `HMMA` for BF16/FP16, `QMMA` for FP8. Same instruction mnemonics as B200 at SASS level, though the underlying hardware implementation differs (mma.sync vs tcgen05).

---

## 8. Performance Comparisons

### LLM Inference (Mistral-7B, batch 32, seq_len 2048)

| Precision | B200 tok/s | H200 tok/s | Speedup | Perplexity Impact |
|-----------|-----------|-----------|---------|-------------------|
| FP16 | 45,200 | 28,500 | 1.59x | Baseline |
| FP8 | 78,400 | 49,200 | 1.59x | +1.9% |
| FP4 | 112,800 | N/A | 2.50x vs FP16 | +8.2% |

### FP64 DGEMM

| Matrix Size | B200 (TFLOPS) | H200 (TFLOPS) | Speedup | B200 % Peak |
|------------|--------------|--------------|---------|-----------|
| 8192 | 35.45 | 18.2 | 1.95x | 78.8% |
| 16384 | 36.14 | 18.7 | 1.93x | 80.3% |
| 32768 | 36.30 | 18.9 | 1.92x | 80.7% |

### Training Performance

| Model | B200 Throughput | H200 Throughput | Speedup | Energy Eff |
|-------|----------------|----------------|---------|-----------|
| ResNet-50 | 2,436 img/s | 1,580 img/s | 1.54x | 3.77 img/s/W |
| GPT-1.3B | 14,397 tok/s | 9,240 tok/s | 1.56x | 22.2 tok/s/W |

### Training Speedup Decomposition

- SM count increase: 1.09x
- CTA pairing: 1.27x
- TMEM: 1.26x
- Combined: ~1.54-1.56x

---

## 9. FP8 and Lower Precision Details

### FP8 Formats

| Format | Bits | Exponent | Mantissa | Dynamic Range | Use Case |
|--------|------|----------|----------|--------------|----------|
| E4M3 | 8 | 4 | 3 | Moderate range, higher precision | Weights, activations |
| E5M2 | 8 | 5 | 2 | Wider range, lower precision | Gradients |

### FP4 Format (E2M1)

- 1 sign, 2 exponent, 1 mantissa bit
- Block scaling: MXFP4 (32-element blocks with E8M0 scale) or NVFP4 (16-element blocks with E4M3 scale)
- Hardware dequantization to FP8/FP16 during MMA
- 2.5x throughput vs FP16 with +8.2% perplexity on Mistral-7B

### Throughput Scaling Across Precisions

Despite 177x throughput difference (FP64 to FP4), latency varies only 1.27x (11.2-14.2 cycles). This confirms throughput scaling is achieved through wider datapaths (more parallel multipliers), not deeper pipelining.

**Implication for sm_120 FP8 attention:** Moving from m16n8k16 BF16 to m16n8k32 FP8 doubles the K dimension per instruction, giving 2x tensor throughput. The softmax phase remains unchanged, so the arithmetic intensity ratio improves favorably. This is why FP8 attention is the highest-ROI next step.

---

## 10. Methodology

### Microbenchmarking Approach

- PTX-level kernels for architectural isolation
- Explicit register/memory control
- PTX-to-SASS translation verified

### Measurement Protocols

- **Latency:** Pointer-chase benchmarks with dependent accesses (prevents pipeline overlap)
- **Throughput:** Back-to-back MMA operations until saturation
- **Memory:** Three strategies -- pointer-chase (latency), instruction comparison, bandwidth saturation

### LLM Inference

- 100 iterations (10 warmup), median/P95/P99 latencies
- Models: Mistral-7B (dense), Mixtral-8x7B/8x22B (MoE)
- Default: batch 32, seq_len 2048

---

## 11. Key Takeaways for sm_120 Development

1. **Constant MMA latency across tile sizes** means we don't benefit from larger MMA tiles on Blackwell. m16n8k16 is the sweet spot for sm_120.

2. **FP32 accumulator halves tensor throughput.** If FP16 accumulators are viable for parts of the computation, there's 2x headroom.

3. **FP8 doubles throughput vs BF16** with only +1.9% perplexity impact. The m16n8k32 instruction processes 2x the K elements per cycle.

4. **Spatial array design** means latency hiding via occupancy (more warps) is the right strategy, not instruction-level parallelism within a single warp.

5. **TMEM is B200-only.** On sm_120, all intermediate data must go through registers or shared memory. This makes register pressure and shared memory management more critical than on datacenter Blackwell.

6. **The 18-23% scheduler stall reduction** from per-warp dispatch (vs warp-group) benefits sm_120 as well, since `mma.sync` is also per-warp.
