# Phase: Quality

Your kernel passed validation. Now it goes through code quality checks.

## Lint Check

The watchdog runs the CUDA linter automatically. If it fails, you'll be
sent to rework with the lint output in a DB message.

To run it yourself:
```bash
python3 /data/src/bwk/common/scripts/lint_cuda.py csrc/<kernel>/<kernel>_sm120.cu
```

## Common Lint Issues

- Missing copyright header
- Unused variables or includes
- Magic numbers without comments
- Overly long functions (>200 lines — consider splitting)
- Missing `__launch_bounds__` on global functions
- Hardcoded GPU constants (should reference common headers)

## Fix and Re-lint

1. Fix all lint issues
2. Re-run lint: `python3 /data/src/bwk/common/scripts/lint_cuda.py <file>`
3. Verify tests still pass after cleanup
4. The watchdog advances to `lint_pass` → `ready_to_ship` automatically
