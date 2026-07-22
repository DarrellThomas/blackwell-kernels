---
name: autokernel
description: This skill should be used when the user asks to "start autokernel", "run autokernel", "resume autokernel", "optimize kernels", or wants to start or resume the profile-driven kernel optimization loop.
disable-model-invocation: true
argument-hint: [kernel] [tag]
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
---

# autokernel — Profile-Driven Kernel Optimization

Autonomous optimization loop for CUDA kernels on RTX 5090 (sm_120).
Profiles with ncu, identifies bottlenecks, implements targeted fixes, benchmarks, keeps or discards.

## Arguments

`$ARGUMENTS` format: `[kernel] [tag]`

- **kernel**: Which kernel to optimize. Options: `attention`, `attention-fp8`, `gemm`. Default: `attention`.
- **tag**: Run tag for the branch name (e.g. `mar12`). If omitted, propose today's date.

Examples:
- `/autokernel` — optimize attention kernel, propose new tag
- `/autokernel attention mar12` — optimize attention, use tag mar12
- `/autokernel gemm mar12` — optimize GEMM kernel, use tag mar12

Parse `$ARGUMENTS`:
- If one word → it's the kernel name (if it matches a known kernel) or a tag
- If two words → first is kernel, second is tag

## Current State

**Git branch:** !`git branch --show-current`
**Existing autokernel branches:** !`git branch --list 'autokernel/*' | tr -d ' ' || echo "none"`
**Uncommitted changes:** !`git status --short | head -5 || echo "clean"`
**Available kernels:** !`python3 /data/src/bwk/common/memory/factory_brain.py jobs | sed -n '4,$p' | awk '{print $4}' | grep -v '^-$' | sort -u | tr '\n' ', ' | sed 's/, $//' || echo "none yet"`
**Dashboard:** !`curl -s -o /dev/null -w "%{http_code}" http://localhost:8420/ 2>/dev/null || echo "not running"`

## Per-Kernel Configuration

### attention
- **Source**: `csrc/attention/flash_attn_v2_sm120.cu`
- **Primitives**: `csrc/common/*.cuh`
- **Test**: `tests/test_attention.py`
- **Bench**: `benchmarks/bench_attention.py`
- **Profile**: `profiles/profile_v2.py`
- **Results**: `results/attention.tsv`
- **Logs**: `logs/attention/`
- **Reference metric**: `vs_sdpa` (vs cuDNN SDPA)
- **Lessons**: `.claude/04_HARD_WON_LESSONS.md`

### attention-fp8
- **Source**: `csrc/attention/flash_attn_fp8_sm120.cu`
- **Primitives**: `csrc/common/*.cuh` (especially `fp8_convert.cuh`)
- **Test**: `tests/test_attention_fp8.py`
- **Bench**: `benchmarks/bench_attention_fp8.py`
- **Profile**: `profiles/profile_fp8.py`
- **Results**: `results/attention_fp8.tsv`
- **Logs**: `logs/attention-fp8/`
- **Reference metric**: `vs_sdpa` (vs cuDNN SDPA)
- **Lessons**: `.claude/04_HARD_WON_LESSONS.md`
- **Note**: Also runs as a secondary eval during attention (BF16) optimization cycles

### gemm
- **Source**: `csrc/gemm/bf16_gemm_sm120.cu`
- **Primitives**: `csrc/common/*.cuh`
- **Test**: `tests/test_gemm.py`
- **Bench**: `benchmarks/bench_gemm.py`
- **Profile**: `profiles/profile_gemm.py`
- **Results**: `results/gemm.tsv`
- **Logs**: `logs/gemm/`
- **Reference metric**: `vs_ref` (vs cuBLAS)
- **Lessons**: `.claude/04_HARD_WON_LESSONS.md` (shared)

## Dashboard Auto-Start

If the dashboard status above shows "not running" or anything other than "200", start it before doing anything else:

```bash
nohup python3 dashboard.py > /dev/null 2>&1 &
```

Confirm it's up: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8420/`

Tell the user: "Dashboard started at http://localhost:8420"

## Decision Logic

Based on the state above, determine which scenario applies:

### Scenario A: Fresh Start
**Conditions:** No autokernel branch exists for this kernel, or user provided a new tag.

Steps:
1. Agree on a tag. If user passed one, use it. Otherwise propose today's date (e.g., `mar12`).
2. Create branch: `git checkout -b autokernel/<tag>` (or use existing if shared)
3. Read all in-scope files for the target kernel (see Per-Kernel Configuration above).
4. Ensure `factory_brain` is the experiment source of truth. Do not initialize or depend on `results/<kernel>.tsv`.
5. Create log dir: `mkdir -p logs/<kernel>`
6. Run baseline: `./eval.sh --kernel <kernel> > eval.log 2>&1`
7. Record baseline in `factory_brain` with `experiment-add`.
8. Confirm setup with user, then begin the optimization loop per the kernel's program file (`program_attention.md` for attention, `program_gemm.md` for gemm).

### Scenario B: Resume on Same Branch
**Conditions:** Currently on an `autokernel/*` branch AND the database has experiment rows for this kernel.

Steps:
1. Read `python3 /data/src/bwk/common/memory/factory_brain.py experiment-summary --kernel <kernel> --recent 8` to understand progress.
2. Read `git log --oneline -10` to see recent commits.
3. Check for uncommitted changes.
4. Run a fresh eval: `./eval.sh --kernel <kernel> > eval.log 2>&1`
5. Report: "Resuming from iteration N. Last kept: X us / Y.Yx. Bottleneck: Z."
6. Continue the optimization loop per the kernel's program file (`program_attention.md` for attention, `program_gemm.md` for gemm).

### Scenario C: Resume on Wrong Branch
**Conditions:** NOT on an `autokernel/*` branch, but autokernel branches exist.

Steps:
1. List existing autokernel branches.
2. Ask user which branch to resume, or whether to start fresh.
3. `git checkout autokernel/<branch>` and proceed per Scenario B.

### Scenario D: Resume with Lost Results
**Conditions:** On an `autokernel/*` branch, but the database has no experiment history for this kernel.

Steps:
1. Reconstruct state from git history.
2. Rebuild baseline state in `factory_brain`.
3. Run baseline on current HEAD.
4. Record baseline with note: "resumed — prior experiment history unavailable".
5. Continue the optimization loop.

## Eval Commands

```bash
# Full eval pipeline (build + test + bench + profile)
./eval.sh --kernel <kernel> > eval.log 2>&1
grep -E "^(build|test|bench|profile|primary_|tsv_|top_)" eval.log

# Quick check (no profiling)
./eval.sh --kernel <kernel> --quick > eval.log 2>&1

# Profile only
./eval.sh --kernel <kernel> --profile > eval.log 2>&1
```

## Core References

**Each kernel has its own program file.** Read the correct one fully before beginning:

- **attention**: `program_attention.md` — attention-specific loop, metrics (`vs_sdpa`), exit criteria
- **gemm**: `program_gemm.md` — GEMM-specific loop, metrics (`vs_ref`), architecture summary

Additional files to read on startup:
- `.claude/04_HARD_WON_LESSONS.md` — invariants and guardrails
- `docs/nvidia_blackwell_tuning_guide_sm120.md` — sm_120 hardware limits
- The kernel source file for the target kernel (see Per-Kernel Configuration)
- `csrc/common/*.cuh` — shared primitives
