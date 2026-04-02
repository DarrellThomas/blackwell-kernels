# Project Workspace Guide

This workspace is intentionally thin.

The active sources of truth are:
- `factory_brain` for jobs, experiment history, worker status, messages, and watchdog state
- `common/claude/` for shared worker policy and loop behavior
- `common/docs/` for shared reference docs and onboarding guidance
- `program_<project>.md` for the project-local contract
- `docs/<project>_agent_state.md` for the project-local narrative memory

## Read This First

For any active project, read in this order:
1. `fb job-show <id>`
2. `program_<project>.md`
3. `docs/<project>_agent_state.md`
4. `common/claude/phases/development.md` or the current phase doc
5. `common/claude/07_OPTIMIZATION_LOOP.md` if this is an optimization-loop job

## Shared Commands

```bash
python3 /data/src/bwk/common/memory/factory_brain.py jobs
python3 /data/src/bwk/common/memory/factory_brain.py job-show <id>
python3 /data/src/bwk/common/memory/factory_brain.py experiment-summary --kernel <kernel> --recent 8
python3 /data/src/bwk/common/memory/factory_brain.py messages --status open
fb heartbeat <kernel> --task "exp<N>: <desc>" --job <id>
```

## Shared Policy

- Common commands, policies, prompts, and scripts belong in `common/`
- Project folders should keep only project-local files
- TSV is deprecated as a source of truth
- Project-local watchdog copies are deprecated; use `common/scripts/watchdog.sh`

## Onboarding

Use:
- `common/docs/howto_add_kernel.md`
- `common/docs/factory_objective_profiles.md`
- `new-project.sh`

If a local doc contradicts `factory_brain` or `common/`, the shared source wins.
