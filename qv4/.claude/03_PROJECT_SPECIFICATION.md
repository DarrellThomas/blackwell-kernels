# [Project Name] — Project Specification

## 1. Executive Summary

### Problem Statement
[2-3 sentences describing what this kernel/library optimizes]

### Solution
[2-3 sentences describing the approach]

### Factory Mode

Choose one:
- `fixed_shape_kernel`
- `general_shape_library`
- `numerical_method`
- `alternative_arithmetic`
- `research_exploration`

See `/data/src/bwk/common/docs/factory_objective_profiles.md`.

### Success Metrics
| Axis | Role | Target | Measurement |
|------|------|--------|-------------|
| Correctness | gate | ... | max_err / exactness |
| Coverage | gate or primary | ... | shape/stride pass rate |
| Numerical quality | gate or primary | ... | residual / backward error / convergence |
| Weighted performance | primary or secondary | ... | weighted `vs_ref` over benchmark set |
| Complexity | secondary | ... | fallback count / maintenance burden |

### Keep / Discard Rule

Write this explicitly:
- Keep when: ...
- Discard when: ...

## 2. Operations

<!-- Single-function: describe the one operation.
     Multi-function: fill in the catalog below for each op. -->

### Operations Catalog

<!-- Delete this section for single-function projects. -->

| Op | Kernel Name | Reference | Target | Status |
|----|-------------|-----------|--------|--------|
| [op1] | [op1]_sm120 | cuBLAS [fn] | >=1.0x | queued |
| [op2] | [op2]_sm120 | PyTorch [fn] | >=1.0x | queued |

**Priority order:** [op1] first because [reason], then [op2], etc.

**Eval.sh FUNCTIONS array** (must match kernel names above):
```bash
FUNCTIONS=(
    "op1|op1_sm120"
    "op2|op2_sm120"
)
```

## 3. Architecture

[Describe the kernel architecture, tile sizes, data flow.
 For multi-function projects, group ops by type:]

### Compute-bound operations (MMA / tensor core)
[ops that use MMA, their tile sizes, occupancy targets]

### Bandwidth-bound operations (vectorized loads + reductions)
[ops that are memory-bound, their load strategy, occupancy targets]

## 4. File Structure

```
csrc/
├── common/            # Shared (DO NOT MODIFY — symlinked from common/)
└── [kernel or lib]/
    └── [kernel]_sm120.cu          # Single-function
    # OR for multi-function:
    ├── [op1]_sm120.cu
    ├── [op2]_sm120.cu
    └── ...

python/blackwell_kernels/
├── [kernel or lib].py     # Python wrappers
└── __init__.py

tests/test_[kernel].py
benchmarks/bench_[kernel].py
profiles/profile_[kernel].py       # Single-function: direct launch
                                   # Multi-function: accepts --op <name>
```

## 5. Testing

### Test Configs
| Test | Config | Tolerance |
|------|--------|-----------|
| Small | ... | 5% rel |
| Primary | ... | 5% rel |

### Coverage Matrix

List the shapes and semantics that define "supported":
- tiny / degenerate:
- tile boundaries:
- rectangular tall:
- rectangular wide:
- stride / submatrix:
- flag combinations:
- ill-conditioned or mixed-scale cases:

<!-- Multi-function: add per-op test configs if they differ significantly -->

## 6. Constraints

- **DO NOT modify `csrc/common/`** — symlinked from common/
- **ALWAYS** use `CUDA_VISIBLE_DEVICES=1` (GPU 1 = air-cooled kernel dev GPU)
- **ALWAYS** use `CUDA_HOME=/usr/local/cuda-13` for builds (CUDA 13.2)
- sm_120 uses `mma.sync`, NOT `tcgen05`
