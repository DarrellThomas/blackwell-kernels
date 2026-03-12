# blackwell-kernels

Custom CUDA kernels for the RTX 5090 — because NVIDIA gave us an amazing GPU and then forgot to ship the software for it.

## The Problem

The RTX 5090 (sm_120, consumer Blackwell) has a different tensor core ISA than its datacenter sibling. Datacenter Blackwell (sm_100) uses `tcgen05` and gets Flash Attention 3, Flash Attention 4, CUTLASS fused attention, and the full optimized kernel ecosystem. Consumer Blackwell uses `mma.sync` and gets... `cuDNN.`

Flash Attention 3 will never compile on this chip. Flash Attention 4 won't either. CUTLASS examples target `tcgen05`. If you're training models on an RTX 5090, you're leaving performance on the table because nobody has written optimized kernels for your specific hardware.

So we wrote our own.

## What's Here

**Flash Attention** — a from-scratch implementation using `mma.sync.aligned.m16n8k16` tensor core instructions, built specifically for sm_120. Not a port of FA2/FA3 — a ground-up design informed by weeks of empirical testing of the register layout, ldmatrix behavior, and MMA fragment packing on this specific GPU.

Current performance (B=2, H=8, N=2048, D=64, causal):
- **1.61x faster** than cuDNN SDPA
- **98.8 us** kernel duration
- **57.2%** SM throughput (and climbing)

The kernel is still being actively optimized by an autonomous optimization loop (more on that below).

## Background

We were running [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — an autonomous AI research loop that trains models and investigates research questions — when we noticed our training was bottlenecked by unoptimized attention kernels. The RTX 5090 was doing great on everything except the inner loop that matters most.

Karpathy's insight was that you can put an AI agent in a loop with a clear objective function and let it run autonomously. He used it for model training. We looked at that and thought: kernel optimization is the same loop. Profile, identify the bottleneck, make a targeted change, measure, keep or discard, repeat.

So we built **autokernel** — a profile-driven optimization loop where [Claude Code](https://claude.com/claude-code) (Anthropic's AI coding agent) autonomously optimizes CUDA kernels. The profiler (NVIDIA Nsight Compute) tells it what to fix, it makes one focused change per iteration, benchmarks it, and keeps only improvements. Each iteration takes about 90 seconds. You start it and walk away.

## How It Works

```
    ┌─────────┐
    │ PROFILE │ ← ncu tells us the #1 bottleneck
    └────┬────┘
         ▼
    ┌─────────┐
    │ ANALYZE │ ← map bottleneck to optimization strategy
    └────┬────┘
         ▼
    ┌─────────┐
    │  CODE   │ ← modify kernel source (one focused change)
    └────┬────┘
         ▼
    ┌─────────┐
    │  BUILD  │ ← compile with CUDA 13
    └────┬────┘
         ▼
    ┌─────────┐
    │  TEST   │ ← 6 correctness tests must pass
    └────┬────┘
         ▼
    ┌─────────┐
    │  BENCH  │ ← measure duration, compare to cuDNN SDPA
    └────┬────┘
         ▼
    ┌──────────────┐
    │ KEEP/DISCARD │ ← faster + correct → keep. else → git reset
    └──────┬───────┘
           │
           └──→ loop back to PROFILE
```

The agent writes its discoveries to a [hard-won lessons file](.claude/04_HARD_WON_LESSONS.md) that persists across runs — things like "always swap a1/a2 registers from ldmatrix_x4" and "register-only P conversion eliminated the dominant source of bank conflicts." Each run starts smarter than the last.

## The Dashboard

A live web dashboard tracks optimization progress in real-time:

```bash
python3 dashboard.py
# open http://localhost:8420
```

It shows kernel duration over time, speedup vs cuDNN SDPA, SM throughput, stall breakdowns, and a full experiment log — color-coded by keep/discard/crash. Annotation markers show when context changes happened (like adding new reference docs mid-run). Auto-refreshes every 30 seconds. Just leave it open in a browser tab.

## Quick Start

### Requirements

- NVIDIA RTX 5090 (sm_120)
- CUDA Toolkit 13.0+
- NVIDIA Nsight Compute (`ncu`) — for profiling in the optimization loop
- PyTorch 2.10+ with CUDA 13 support
- Python 3.12+
- [Claude Code](https://claude.com/claude-code) — for running the autonomous optimization loop

### Build & Test

```bash
# Build
CUDA_HOME=/usr/local/cuda-13 python3 setup.py build_ext --inplace

# Test
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 tests/test_attention.py

# Benchmark
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python3 benchmarks/bench_attention.py
```

### Use in Your Code

```python
import torch
from blackwell_kernels import flash_attn_v2_sm120

# Q, K, V: [batch*heads, seq_len, head_dim], dtype=torch.bfloat16
Q = torch.randn(2, 8, 2048, 64, dtype=torch.bfloat16, device="cuda")
K = torch.randn(2, 8, 2048, 64, dtype=torch.bfloat16, device="cuda")
V = torch.randn(2, 8, 2048, 64, dtype=torch.bfloat16, device="cuda")

O, L = flash_attn_v2_sm120(Q, K, V, causal=True)
# O: attention output, L: logsumexp (for backward pass)
```

### Run the Optimization Loop Yourself

See [AUTOKERNEL.md](AUTOKERNEL.md) for full instructions. The short version:

1. Install [Claude Code](https://claude.com/claude-code)
2. Open this directory in Claude Code
3. Type `/autokernel`
4. Walk away

The `/autokernel` skill handles everything — starts the dashboard, creates a branch, runs the baseline, and kicks off the autonomous loop. Every improvement is a git commit. Every failure is recorded in the hard-won lessons. Check the dashboard whenever you want.

## Project Structure

```
csrc/
  attention/
    flash_attn_v2_sm120.cu    ← the kernel (this is where the magic happens)
    flash_attn_sm120.cu       ← v1 scalar reference
  common/
    mma_sm120.cuh             ← mma.sync wrappers
    ldmatrix.cuh              ← shared→register helpers
    cp_async.cuh              ← async global→shared copy
    swizzle.cuh               ← bank conflict avoidance
python/
  blackwell_kernels/          ← Python bindings
tests/                        ← correctness tests (Python + standalone CUDA)
benchmarks/                   ← benchmark harness
docs/                         ← curated CUDA reference docs
.claude/
  04_HARD_WON_LESSONS.md      ← empirical knowledge (the good stuff)
  CLAUDE.md                   ← agent context (MMA register layout, build commands)
program_gemm.md               ← GEMM optimization loop instructions
eval.sh                       ← build → test → bench → profile pipeline
```

## Kernel Roadmap

| Kernel | Status | Why |
|--------|--------|-----|
| Flash Attention (BF16) | Optimizing | Training bottleneck |
| BF16 GEMM | Stub exists | Linear layers |
| Flash Attention (FP8) | Not started | 2x throughput from wider MMA |
| Fused MLP | Not started | Eliminate memory round-trips |
| RMSNorm + Attention | Not started | Fuse norm into attention |

Each new kernel goes through the same two-phase process: get it correct (human-guided), then make it fast (autonomous loop). The primitives in `csrc/common/` and the hard-won lessons carry forward — each kernel starts further ahead than the last.

## Contributing

The best way to contribute is to run the optimization loop yourself and push what you find. Fork it, let it run overnight (or for a week), and open a PR with your improvements. The hard-won lessons file and git history tell the full story of what was tried.

If you discover something new about sm_120's behavior, add it to `.claude/04_HARD_WON_LESSONS.md`. These empirical findings are the most valuable part of this project.

## Acknowledgments

**[Andrej Karpathy](https://github.com/karpathy)** — for [autoresearch](https://github.com/karpathy/autoresearch), which inspired the autonomous optimization loop. We were running autoresearch when we discovered the kernel gap, and adapted the approach for kernel optimization.

**[Anthropic](https://anthropic.com)** — this project was built in collaboration with [Claude Code](https://claude.com/claude-code) (Claude Opus). From reverse-engineering the MMA register layout to writing the kernels to building the optimization infrastructure, Claude was a genuine partner throughout. The autonomous optimization loop is Claude running independently, making real engineering decisions about CUDA kernels, hundreds of iterations at a time.

**[gau-nernst](https://github.com/gau-nernst)** — for demonstrating that 94.4% of peak TFLOPS is achievable on sm_120 with custom flash attention, proving the feasibility of this approach.

**NVIDIA** — for building a beast of a GPU. We just wish the kernel ecosystem had shipped with it. Consider this our contribution to closing that gap.

## License

MIT License. Copyright (c) 2026 Darrell Thomas. See [LICENSE](LICENSE).
