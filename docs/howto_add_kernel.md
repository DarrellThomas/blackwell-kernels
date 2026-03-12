# How to Add a New Kernel to the Autokernel Loop

Step-by-step guide for adding a new kernel to the optimization infrastructure.

---

## 1. Write the Kernel

Create the CUDA source in the appropriate directory:

```
csrc/<kernel_name>/<kernel_name>_sm120.cu
```

- Add copyright header (Darrell Thomas, MIT License)
- Target sm_120, use `mma.sync` ISA (not tcgen05)
- Add any new shared primitives to `csrc/common/`

## 2. Add Python Bindings

- Add the kernel to `setup.py` (CUDAExtension sources list)
- Add Python wrapper in `python/blackwell_kernels/`
- Export from `python/blackwell_kernels/__init__.py`

## 3. Create Test, Benchmark, and Profile Scripts

```
tests/test_<kernel>.py          # Correctness vs PyTorch/cuBLAS reference
benchmarks/bench_<kernel>.py    # Timing harness, must emit:
                                #   primary_custom_ms: <value>
                                #   primary_vs_ref: <value>x
profiles/profile_<kernel>.py    # Minimal launch for ncu profiling
```

The benchmark script MUST print `primary_custom_ms:` and `primary_vs_ref:` lines — `eval.sh` parses these.

## 4. Register in eval.sh

Add a case to the kernel switch in `eval.sh`:

```bash
case "$KERNEL" in
    ...
    <kernel_name>)
        TEST_SCRIPT="tests/test_<kernel_name>.py"
        BENCH_SCRIPT="benchmarks/bench_<kernel_name>.py"
        PROFILE_SCRIPT="profiles/profile_<kernel_name>.py"
        NCU_KERNEL_NAME="<cuda_kernel_function_name>"
        ;;
    ...
esac
```

**Copy eval.sh to all worktrees after editing.**

## 5. Register in the Autokernel Skill

Edit `.claude/skills/autokernel/SKILL.md`, add to the Per-Kernel Configuration section:

```markdown
### <kernel_name>
- **Source**: `csrc/<kernel_name>/<source>.cu`
- **Primitives**: `csrc/common/*.cuh`
- **Test**: `tests/test_<kernel_name>.py`
- **Bench**: `benchmarks/bench_<kernel_name>.py`
- **Profile**: `profiles/profile_<kernel_name>.py`
- **Results**: `results/<kernel_name>.tsv`
- **Logs**: `logs/<kernel_name>/`
- **Reference metric**: `vs_ref` (vs cuBLAS) or `vs_sdpa` (vs cuDNN)
- **Lessons**: `.claude/04_HARD_WON_LESSONS.md`
```

Also add the kernel name to the `Available kernels:` line in the Current State section.

## 6. Register on the Dashboard

Edit `dashboard.py` in the UI worktree (`/data/src/blackwell-kernels-ui/`). Add to `KERNEL_CONFIG`:

```python
"<kernel_name>": {
    "ref_label": "vs <Reference>",
    "ref_column": "vs_ref",         # must match TSV column name
    "ref_target": 1.0,              # target ratio (1.0 = match reference)
    "gpu": 0,                       # which GPU runs this loop
    "tsv_file": Path("/data/src/blackwell-kernels-<worktree>/results/<kernel_name>.tsv"),
    "heartbeat": Path("/data/src/blackwell-kernels-<worktree>/.autokernel.<kernel_name>.alive"),
    "theoretical_floor_us": <value>,     # roofline minimum (see theoretical_limits.md)
    "achievable_ceiling_us": <value>,    # realistic target
    "markers": [],
},
```

Restart the dashboard after editing.

## 7. Create a Worktree

```bash
cd /data/src/blackwell-kernels
git worktree add -b autokernel/<tag> /data/src/blackwell-kernels-<name> autokernel/mar12
```

Copy current infrastructure files to the new worktree:
```bash
cp eval.sh /data/src/blackwell-kernels-<name>/
```

Set up a scoped `.claude/CLAUDE.md` in the worktree if needed.

## 8. Update the Watchdog

Edit `watchdog.sh`, add to the `LOOPS` array:

```bash
LOOPS=(
    ...
    "<kernel_name>|<tmux_session>|/data/src/blackwell-kernels-<worktree>/results/<kernel_name>.tsv|/autokernel <kernel_name> <tag>"
)
```

Restart the watchdog after editing.

## 9. Initialize the TSV

Create the results file with the header:

```bash
mkdir -p /data/src/blackwell-kernels-<worktree>/results
printf 'commit\tduration_us\tvs_ref\tsm_pct\tstall_math\tstall_wait\tstall_scoreboard\tstall_barrier\ttop_stall\tstatus\tdescription\n' \
  > /data/src/blackwell-kernels-<worktree>/results/<kernel_name>.tsv
```

Use `vs_sdpa` instead of `vs_ref` for attention-family kernels.

## 10. Create a tmux Session and Launch

```bash
tmux new-session -d -s autokernel-<name> -c /data/src/blackwell-kernels-<worktree>
tmux send-keys -t autokernel-<name> 'claude' Enter
# wait for Claude to start, then:
tmux send-keys -t autokernel-<name> '/autokernel <kernel_name> <tag>' Enter
```

## Checklist

```
[ ] Kernel source in csrc/
[ ] Python bindings in setup.py + python/
[ ] test, bench, profile scripts
[ ] eval.sh case added (copied to all worktrees)
[ ] SKILL.md updated
[ ] dashboard.py KERNEL_CONFIG entry
[ ] Git worktree created
[ ] watchdog.sh LOOPS entry
[ ] TSV header initialized
[ ] tmux session launched
```

## Current Layout

```
blackwell-kernels/        ← master (orchestrator)
blackwell-kernels-gpu1/   ← attention-r2 loop (GPU 1)
blackwell-kernels-gemm/   ← gemm loop (GPU 0)
blackwell-kernels-ui/     ← dashboard UI only
```
