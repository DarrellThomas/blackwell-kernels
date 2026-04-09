# blackwell-kernels — [PROJECT NAME]

## Rules (non-negotiable)

1. **Every input size must work.** 1, 63, 64, 65, 1024, 4096. Not just tile-aligned.
2. **Every BLAS parameter must work.** Alpha, beta, lda, transpose, uplo.
3. **Test accuracy at every size** vs reference. Crashes = broken. Wrong answers = broken.
4. **If it crashes on ANY valid input, it is broken.** Fix before optimizing.
5. **Correct first. Complete second. Fast third.**

## Your Current Phase

@phase_context.md

## Factory Database

```bash
 /data/src/bwk/common/memory/msearch "your question" --kernel <your-kernel> -k 5
fb heartbeat <kernel> --task "desc" --job <id>         # REQUIRED every iteration
fb jobs                                                # see all work
fb messages --status open                              # check for messages
fb job-update <id> --state <state> --by <kernel>       # update your job state
```

## Factory Objective

Read these before optimizing:

```bash
cat /data/src/bwk/common/docs/factory_objective_profiles.md
cat program_[kernel].md
```

Do not assume "better" means one benchmark got faster.

For this project, always evaluate against:
1. Hard gates
2. Declared keep/discard rule
3. Objective vector in the project spec

If a change speeds up one benchmark but violates a hard gate or coverage goal,
discard it.

## Build & Test

```bash
# Build (MUST use CUDA 13.2)
CUDA_HOME=/usr/local/cuda-13 python3 setup.py build_ext --inplace

# Test
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3 tests/test_[kernel].py

# Full eval pipeline (build + test + bench + profile)
./eval.sh --kernel [kernel]

# Profile single function (multi-function projects)
./eval.sh --kernel [kernel] --func [op_name]
```

- **ALWAYS** use `CUDA_VISIBLE_DEVICES=1` (GPU 1 = air-cooled kernel dev GPU)
- **ALWAYS** use `CUDA_HOME=/usr/local/cuda-13` for builds

## Results Tracking

**Every experiment must be recorded in `results/[kernel].tsv`.**
No rows = invisible to dashboard and factory.

```
commit  duration_us  vs_ref  sm_pct  stall_math  stall_wait  stall_scoreboard  stall_barrier  top_stall  status  description
```

Use `-` for unmeasured metrics. Status is `keep` or `discard`.

In `description`, record the real decision basis:
- which objective axis improved
- which gate was checked
- whether the result was single-shape-only or coverage-wide

## Before You Stop

**BEFORE stopping for ANY reason:**
1. `fb heartbeat <kernel> --task "reason" --state complete --job <id>`
2. `fb message-create --from <kernel> --subject "Halt: <reason>" --type halt`
3. Update `docs/<kernel>_agent_state.md` with final state

## Hardware

- RTX 5090 (sm_120a, consumer Blackwell, `mma.sync` NOT `tcgen05`)
- 170 SMs, 32GB GDDR7, 1792 GB/s, 99 KB shared/block, 48 warps/SM
- Compile: `-gencode arch=compute_120a,code=sm_120a`

## Project Conventions
- Copyright: Darrell Thomas, MIT License
- All `.cu` and `.py` files start with copyright header
