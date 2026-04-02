# Math Pipe Throttle — Optimization Guide for sm_120

## What Is Math Throttle?

**Profiler metric:** `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio`

The warp scheduler wanted to issue a math instruction (FMA, ALU, or tensor core `mma.sync`) but the math pipeline's input FIFO was already full. The pipe cannot accept another instruction this cycle.

### Throttle vs Wait — Two Different Stalls

| Stall | Meaning | Direction |
|-------|---------|-----------|
| **math_pipe_throttle** | Input FIFO full — too many instructions arriving | Push-side congestion |
| **wait** | Waiting for MMA result due to data dependency | Pull-side dependency |

Both show up in tensor-core-heavy kernels. Throttle means you're *issuing* too fast in bursts. Wait means you're *consuming* results too soon after issue.

## When Is It Good vs Bad?

**Good:** Tensor pipe utilization >85% AND math_throttle is top stall. You are compute-bound. The tensor cores are saturated. This is the goal.

**Bad:** Tensor pipe utilization ~50% AND math_throttle is high (~40%+). MMA instructions arrive in **bursts** — the pipe saturates during bursts then starves between them. SM throughput is well below peak despite being "compute-bound."

### Diagnostic

Compare these two ncu metrics:
- `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` (tensor pipe utilization)
- `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.ratio` (throttle %)

If tensor pipe <70% but throttle >30%, you have a burst scheduling problem.

## Root Cause on sm_120

The typical inner loop pattern:

```
load K tile → load V tile → MMA MMA MMA MMA → softmax → MMA MMA MMA MMA
```

The MMA instructions cluster together. During the MMA burst, the tensor core input FIFO fills instantly. During loads/softmax, the tensor core sits idle. This burst-then-starve pattern wastes half the available tensor core bandwidth.

**sm_120 makes this worse** because it has only 48 warps/SM (vs 64 on datacenter Blackwell). With fewer warps available for latency hiding, the scheduler has fewer alternatives when a warp is throttled. At 4 warps per block (128 threads), there is exactly 1 warp per sub-partition — when it stalls on math throttle, that sub-partition's tensor core goes idle.

### mma.sync Pipeline Depth

On Ada (sm_89), `m16n8k16` has ~32-cycle latency. sm_120 likely has similar latency (both use `mma.sync`, not `tcgen05`). With 4 warp schedulers per SM and ~32-cycle MMA latency, you need at minimum 8 warps issuing MMA to keep the pipe full (32 cycles / 4 schedulers = 8 issue slots).

## Optimization Strategies

Ranked by expected impact for sm_120 kernels:

### 1. Interleave MMA with Memory Operations

The highest-impact fix. Instead of bursting all MMAs together, spread them across time with memory operations between them:

**Before (bursty):**
```
cp.async K[next]
cp.async V[next]
cp.async.commit
cp.async.wait
mma  mma  mma  mma    ← pipe saturates, then starves during loads
softmax
mma  mma  mma  mma
```

**After (interleaved):**
```
mma + cp.async K[next]    ← math and memory run in parallel
mma + cp.async V[next]
mma + cp.async.commit
mma                       ← pipe stays fed, never bursts
softmax
mma  mma  mma  mma
```

The spatters.ca Ada matmul blog achieved this and reduced per-MMA overhead from 179.9 cycles to 34.2 cycles — reaching 93% of RTX 4090 peak.

**Implementation:** Requires careful instruction ordering. The compiler's `#pragma unroll` sometimes achieves this naturally, but explicit ordering via inline PTX may be needed for fine-grained control.

### 2. Increase to 8 Warps (256 Threads)

Gives 2 warps per sub-partition. When one warp stalls on math_throttle, the scheduler issues from the other warp (which may have a load ready).

**Register budget:** 128 regs/thread × 256 threads = 32K registers/block. With 64K regs/SM, this fits 2 blocks/SM = 16 warps total. Decent occupancy.

**Tradeoff:** More warps means more register pressure. If spills to local memory occur, the cure is worse than the disease. Check `--ptxas-options=-v` for spill count.

### 3. Deeper Software Pipeline (3+ Stages)

Double-buffering gives 2 stages of prefetch. A third stage increases the prefetch distance — loads issued earlier have more time to complete before the data is needed.

**Shared memory cost:**
- 2-stage (current): ~55 KB
- 3-stage: ~82 KB (fits under 99 KB limit, but tight)
- 4-stage: ~110 KB (exceeds 99 KB — not feasible on sm_120)

### 4. Larger Tiles (More Independent MMAs)

Larger output tiles per warp mean more independent MMA instructions per loop iteration. The scheduler can issue these across multiple cycles without data dependencies stalling the pipe.

Example: going from 2×2 to 4×4 MMA tiles per warp gives 4× more independent instructions to schedule before any result is needed.

**Tradeoff:** More registers per thread for accumulator storage. A 4×4 tile needs 64 FP32 accumulators = 64 registers just for D fragments.

### 5. Warp Specialization (Producer/Consumer)

Dedicate some warps to loading data (producer) and others to MMA (consumer). Producers never issue math instructions — they never contend for the math pipe. Consumers focus exclusively on MMA.

**Coordination:** Requires shared memory barriers between producer/consumer warps. On sm_120 without TMEM, this is the only coordination mechanism.

**Complexity:** Highest implementation complexity. CUTLASS 3.x uses this on Hopper but it requires significant architectural rework.

## Applying to Our Kernels

### Flash Attention (v2 MMA)

Current: 4 warps, BLOCK_Q=64, BLOCK_KV=32, double-buffered K/V, ~93 us at N=2048 D=64.

The inner loop has two MMA phases (QK^T and PV) separated by softmax. Math throttle hits during both MMA phases. Strategy 1 (interleaving loads between MMAs) is most applicable — prefetch next K/V tile between MMA instructions rather than before/after the MMA block.

### BF16 GEMM

Current: inner loop is a tight MMA-dominated loop with less non-math work than attention.

GEMM is the best candidate for strategies 1-4. The inner loop's simplicity makes interleaving and deeper pipelining more straightforward. The spatters.ca results (93% peak on Ada) demonstrate that these strategies work on similar `mma.sync` architectures.

## References

### Primary Sources

- **NVIDIA Nsight Graphics — Warp Stall Reasons**
  https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html
  Canonical definitions: "A math pipe input FIFO is full (FMA, ALU, FP16+Tensor)"

- **NVIDIA Nsight Compute Profiling Guide v13.2**
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
  Metrics reference for all `smsp__warp_issue_stalled_*` counters

- **NVIDIA Blackwell Tuning Guide (sm_120)**
  https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
  SM resource limits: 48 warps/SM, 64K registers, 99 KB shared/block

### Implementation References

- **Implementing a fast Tensor Core matmul on Ada (spatters.ca)**
  https://www.spatters.ca/mma-matmul
  Best practical walkthrough of reducing MMA stalls on sm_89. Reached 93% peak via interleaved scheduling. Directly applicable to sm_120's `mma.sync`.

- **Dissecting Tensor Cores via Microbenchmarks (arXiv:2206.02874)**
  https://arxiv.org/abs/2206.02874
  `mma.sync` latency/throughput characterization across architectures.

- **Microbenchmarking NVIDIA Blackwell (arXiv:2512.02189)**
  https://arxiv.org/abs/2512.02189
  Blackwell tensor core latency measurements (datacenter sm_100, but useful for pipeline depth understanding).

### NVIDIA Forum Discussions

- **How to analyze stall_wait in HMMA case**
  https://forums.developer.nvidia.com/t/how-to-analysis-the-stall-wait-in-this-hmma-case/310727
  Key distinction: math_pipe_throttle = multiple warps contending for same pipe; wait = data dependency.

- **Understanding stall_wait and sampling data**
  https://forums.developer.nvidia.com/t/how-to-understanding-stall-wait-and-sampling-data/196060
  Stall_wait as unavoidable pipeline latency in MMA-heavy kernels.

### Architecture References

- **CUTLASS Efficient GEMM Pipelining (Colfax Research)**
  https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/
  Multi-stage pipeline design for tensor core GEMM.

- **CUTLASS Ping-Pong GEMM Kernel (PyTorch Blog)**
  https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/
  Warp specialization: producer/consumer separation for Hopper.

- **Pushing Tensor Cores to the Limit (GTC 2020)**
  https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
  A100 tensor core optimization strategies (general principles apply to sm_120).
