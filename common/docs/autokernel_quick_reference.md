# Autokernel Quick Reference

The autokernel loop is now DB-first.

## Required Loop Surfaces

- Job spec: `fb job-show <id>`
- Experiment history: `python3 /data/src/bwk/common/memory/factory_brain.py experiment-summary --kernel <kernel> --recent 8`
- Shared loop policy: `common/claude/07_OPTIMIZATION_LOOP.md`
- Phase policy: `common/claude/phases/<phase>.md`
- Project contract: `program_<project>.md`
- Narrative memory: `docs/<project>_agent_state.md`

## Required Recording

```bash
fb heartbeat <kernel> --task "exp<N>: <desc>" --job <id>
python3 /data/src/bwk/common/memory/factory_brain.py experiment-add \
  --kernel <kernel> \
  --status <keep|discard> \
  --description "<real reason>"
```

## Decision Order

1. Hard gates
2. Primary objective
3. Secondary regressions

Do not treat one benchmark speedup as automatically keep-worthy.

## Deprecated

- project-local watchdog copies
- TSV as source of truth
- UI-derived heartbeat logic
- duplicated local copies of shared onboarding or policy docs
