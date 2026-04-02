# Nsight Compute 2026.1: Register Dependency Visualization

**Sources:**
- [CUDA 13.2 Blog Post](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

**Relevant to:** all workers (especially gemm, attention, lu, qr, cholesky)
**Worker's current problem:** Workers profile with ncu but manually interpret stall
reasons (math_pipe_throttle, long_scoreboard, barrier, wait). The new register
dependency tool makes it easier to trace WHY a stall occurs at the instruction level.

---

## What This Is

Nsight Compute 2026.1 (shipped with CUDA 13.2) adds a **Register Dependency
correlation window** on the Source page. This tool shows register-level dependencies
between instructions, helping identify which source lines create pipeline stalls.

## Why It Matters for Us

Our workers repeatedly hit stall patterns they struggle to diagnose:

| Worker | Top Stall | How Register Deps Help |
|--------|-----------|------------------------|
| Attention | math_pipe_throttle (48%) | See which MMA output registers feed into softmax ALU instructions, creating the burst/starvation pattern |
| GEMM FP8 | long_scoreboard (17%) | Trace which global/shared loads have the longest register dependency chains |
| GEMM FP8 | wait (14%) | Identify which MMA instructions wait on prior loads to fill their input registers |
| RMSNorm | barrier (46%) | Confirm that barrier stalls correlate with shared memory reduction — irreducible |
| Cholesky | launch overhead (dominant) | Less useful (problem is kernel count, not instruction-level stalls) |
| LU | not yet engaged | Will be useful once v1 kernel is profiled |

## Key New Features

### 1. Register Dependency Correlation Window

Located on the **Source page** of an ncu report. Shows:
- Which instruction produces each register value
- Which downstream instructions consume that register
- The dependency chain length (number of cycles between produce and consume)
- Highlighted critical paths (longest dependency chains that limit throughput)

### 2. Report Clustering and Merging

New tool under **File > Merge Reports** that:
- Groups similar profiling runs for statistical analysis
- Helps identify noise vs real performance differences
- Useful for our workers who run many experiments (79+ for GEMM, 53+ for attention)

### 3. CUDA Graphs Viewer Improvements

Enhanced visualization showing:
- Graphs as built AND as profiled (side by side)
- Visual correlation between collected results and graph nodes
- Useful for Cholesky/LU workers who use CUDA Graphs to reduce launch overhead

## How to Use

```bash
# Profile a kernel with source-level metrics
ncu --set full --section SourceCounters -o report kernel_binary

# Open in Nsight Compute GUI
ncu-ui report.ncu-rep
# Navigate to Source page → Register Dependency tab
```

For workers running headless (tmux), generate the report and transfer to a machine
with the GUI, or use `ncu --page details --csv` for text-based analysis.

## Caveats

- Requires CUDA 13.2 toolkit (which we have)
- The register dependency view works best with `-lineinfo` in compilation flags
  (already standard in our setup.py configurations)
- Does NOT replace understanding of the hardware pipeline — it supplements it
- The clustering tool requires multiple profiling runs of the same kernel

## Recommendation

Workers should update their profiling workflow to use `ncu` from CUDA 13.2 and
check the Register Dependency tab when investigating stall patterns. This is
especially valuable for:
- Attention worker: understanding MMA → softmax → MMA pipeline bubble
- GEMM worker: understanding FP8 B-load → compute dependency chains
- LU/QR workers: profiling their first custom kernels
