# [Kernel Name] — Optimization Program

TEMPLATE: Rename to `program_<kernel>.md` and fill in.

## Factory Mode

Choose one:
- `fixed_shape_kernel`
- `general_shape_library`
- `numerical_method`
- `alternative_arithmetic`
- `research_exploration`

See `/data/src/bwk/common/docs/factory_objective_profiles.md`.

## Objective

Describe the real objective, not just the reference implementation.

Examples:
- fixed-shape kernel: minimize latency on the declared production shape set
- general-shape library: maximize coverage-weighted performance while preserving
  all supported shapes/strides
- numerical method: minimize residual or improve convergence at acceptable cost
- alternative arithmetic: preserve exact arithmetic semantics while improving
  throughput/packing density

## Primary Benchmark Set

Declare the evaluation set that actually decides keep/discard.

### Fixed-Shape Projects

| Case | M | N | K | Weight | Notes |
|------|---|---|---|--------|-------|
| primary | — | — | — | 1.0 | — |

### General-Shape / Numerical Projects

List representative classes:
- tile boundaries: 63, 64, 65
- tiny sizes: 1, 2, 3, 7, 15, 17
- rectangular tall / wide
- stride cases (`lda > m`, submatrix views)
- ill-conditioned / mixed-scale inputs if applicable

## Reference Metric

Reference implementations may still be used, but they are not always the whole objective.

### Objective Vector

Mark each axis as `gate`, `primary`, `secondary`, or `ignore`.

| Axis | Role | Measurement |
|------|------|-------------|
| Correctness | gate | tests / reference comparison |
| Coverage | gate or primary | shape and stride matrix pass rate |
| Numerical quality | gate or primary | residual, backward error, convergence |
| Weighted performance | primary or secondary | median/weighted `vs_ref` across benchmark set |
| Complexity | secondary | codepath count, fallback burden, maintenance risk |

### Keep / Discard Rule

Fill in explicitly:
- Keep when: [example: all gates pass and weighted performance improves >=2%]
- Discard when: [example: any gate fails, or speedup appears only on one shape while coverage regresses]

## What You MAY Modify

- `csrc/<kernel>/` — kernel source code
- `python/blackwell_kernels/<kernel>.py` — Python wrapper
- `tests/test_<kernel>.py` — add tests as needed
- `benchmarks/bench_<kernel>.py` — add benchmark configs
- `.claude/03_PROJECT_SPECIFICATION.md` — update as architecture evolves
- `docs/<kernel>_agent_state.md` — update after each experiment

## What You MUST NOT Modify

- `csrc/common/` — shared headers (symlinked from common/)
- `eval.sh` — eval pipeline (request changes via DB message)
- Other kernel directories

## Optimization Strategy

1. Start from the GEMM template (64x64 tiles, 4 warps, cp.async, XOR swizzle)
2. Get hard gates passing first
3. Profile with ncu only if this project is actually bottlenecked by kernel performance
4. Iterate: change one thing, evaluate against the declared objective vector
5. Document each experiment in agent_state.md, including why it was kept or discarded

## Hard Gates

List the non-negotiable conditions here.

Examples:
- all declared shapes correct
- all BLAS/LAPACK flag combinations correct
- no crash on degenerate inputs
- residual < tolerance
- bit-exact arithmetic semantics

## Secondary Metrics

List metrics that matter but do not override the hard gates.

Examples:
- single-shape latency
- SM throughput
- top stall
- benchmark variance
- code complexity / fallback count

## Exit Criteria

- Hard gates all pass
- Primary objective has met its target
- No obvious remaining path improves the declared objective vector
- Stopping condition is justified in `docs/<kernel>_agent_state.md`
