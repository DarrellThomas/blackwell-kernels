# blackwell-kernels — comfy_render

## Context

Custom CUDA kernels for ComfyUI diffusion inference on RTX 5090.
Target installation: `/users/melanie/ComfyUI/`
4 ops: mha (#65), gnorm_linear (#66), wstream (#67), vae_conv (#68).
This is a multi-function project — work one job at a time, in priority order.

## Rules (non-negotiable)

1. **Every input size must work.** Seq lengths from 16 to 65536. Not just tile-aligned.
2. **All model families must work.** SD1.5 (head_dim=40), SDXL (64), Flux (128).
3. **Test accuracy at every size** vs PyTorch reference. Crashes = broken. Wrong answers = broken.
4. **If it crashes on ANY valid input, it is broken.** Fix before optimizing.
5. **Correct first. Complete second. Fast third.**

## Your Current Phase

@phase_context.md

## Factory Database

```bash
 /data/src/bwk/common/memory/msearch "your question" --kernel <your-kernel> -k 5
fb heartbeat <kernel> --task "desc" --job <id>         # REQUIRED every iteration
fb jobs                                                # see all work
fb job-show <id>                                       # inspect this job's objective profile
fb messages --status open                              # check for messages
fb job-update <id> --state <state> --by <kernel>       # update your job state
python3 /data/src/bwk/common/memory/factory_brain.py experiment-summary --kernel <kernel>
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

**Every experiment must be recorded in `factory_brain`.**
That is the source of truth for worker history, progress, and UI state.

Record each run with `experiment-add`, using the real decision basis:
- what changed
- which objective axis improved or regressed
- which gates were checked
- whether the result generalizes or only helps a narrow case

TSV mirrors are deprecated compatibility artifacts. Do not rely on them as the
primary record.

Before choosing the next experiment, inspect:

```bash
python3 /data/src/bwk/common/memory/factory_brain.py experiment-summary --kernel <kernel>
```

Your next step should be informed by:
- the current discard streak
- the best keep so far
- what has already failed repeatedly
- which objective axes still have headroom

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
