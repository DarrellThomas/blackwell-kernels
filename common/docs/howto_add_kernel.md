# How to Add a New Project to the Factory

This factory is now database-first.

Do not onboard a new project by wiring per-project watchdogs, seeding TSV files,
or teaching the UI about local heartbeat paths. The source of truth is
`factory_brain`.

## 1. Create the Project Workspace

From the repo root:

```bash
./new-project.sh <project-name>
```

That creates a lean project folder with shared policies and scripts pointing back
to `common/`.

## 2. Define the Objective in the Database

Create or update the job in `factory_brain` before you launch a worker.

```bash
fb job-create \
  --title "Implement <project-name>" \
  --project <project-name> \
  --state queued
```

Then fill in the objective profile:

```bash
fb job-update <id> \
  --factory-mode <fixed_shape_kernel|general_shape_library|numerical_method|alternative_arithmetic|research_exploration> \
  --optimization-scope <algorithmic|hardware_tuned|hybrid> \
  --objective-vector "..." \
  --acceptance-gates "..." \
  --keep-rule "..." \
  --benchmark-set "..." \
  --hardware-target "RTX 5090 / Blackwell" \
  --retarget-policy "..."
```

If the project compares against a named reference, also set:

```bash
fb job-update <id> --reference-label "cuBLAS"
```

## 3. Fill in the Project Spec

Update the project-specific contract:

```bash
program_<project-name>.md
```

That file should state:
- what the project is trying to optimize
- what counts as keep vs discard
- which gates must pass first
- what benchmark or validation set matters

Do not define success as a single generic `vs_ref` unless that is actually the
project objective.

## 4. Add the Implementation Surfaces

Create the normal project files:

```text
csrc/<project>/
python/blackwell_kernels/
tests/
benchmarks/
profiles/
```

Then wire the project into:
- `setup.py`
- `python/blackwell_kernels/__init__.py`
- the local `eval.sh` switch
- any project-specific bindings or wrappers

## 5. Make Eval Emit DB-Ready Metrics

`eval.sh` and the benchmark harness should emit the primary metrics the worker
needs, but experiment state belongs in `factory_brain`.

Workers record runs with:

```bash
fb heartbeat <kernel> --task "exp<N>: <desc>" --job <id>
python3 /data/src/bwk/common/memory/factory_brain.py experiment-add ...
python3 /data/src/bwk/common/memory/factory_brain.py experiment-summary --kernel <kernel>
```

TSV mirrors are deprecated compatibility artifacts. Do not create a new project
around TSV-first logging.

## 6. Use Shared Control Surfaces

The shared control plane lives in `common/`.

Use:
- `common/claude/`
- `common/scripts/watchdog.sh`
- `common/docs/`
- `common/memory/factory_brain.py`

Do not add a new project-local `watchdog.sh` unless there is a very specific
reason that cannot be handled centrally.

## 7. Launch a Worker

Launch the worker in tmux from the project directory and point it at the DB job.

Example:

```bash
tmux new-session -d -s <project-name> -c /data/src/bwk/<project-name>
tmux send-keys -t <project-name> 'codex --dangerously-bypass-approvals-and-sandbox --no-alt-screen' Enter
```

Then give it the job id and let it work from:
- `program_<project>.md`
- `.claude/CLAUDE.md`
- `fb job-show <id>`
- `fb messages --status open`

## 8. Verify the Factory View

Before you call the project onboarded, check:

```bash
fb job-show <id>
fb workers
python3 /data/src/bwk/common/memory/factory_brain.py watchdog-state
```

The UI should be able to render the project from DB state. It should not require
you to hardcode heartbeat paths or local TSV files into `dashboard.py`.

## Checklist

```text
[ ] Project created with ./new-project.sh
[ ] Job exists in factory_brain
[ ] Objective profile fields filled in
[ ] program_<project>.md reflects real keep/discard logic
[ ] eval/build/test/profile paths are wired
[ ] Worker can report heartbeat and experiment-add to the DB
[ ] Shared watchdog can observe the project
[ ] UI can surface the project from DB-backed state
```

## What Not to Do

Do not:
- seed `results/<project>.tsv` as the primary log
- add a per-project dashboard heartbeat path
- add a per-project `watchdog.sh` by default
- define "better" without an objective profile
- let the UI become the source of truth
