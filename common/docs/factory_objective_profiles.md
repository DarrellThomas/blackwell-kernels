# Factory Objective Profiles

This factory does not have a single objective.

Some projects are fixed-shape performance kernels. Some are general-purpose
BLAS/LAPACK libraries. Some are numerical methods where the point is stability,
accuracy, or convergence behavior. Future projects may target integer or
ternary arithmetic where throughput, exactness, and packing efficiency dominate.

The factory must therefore optimize against an **objective vector**, not a
single scalar like `vs_ref`.

## Core Principle

Before running an autonomous loop, every project must declare:

1. **Factory mode**
2. **Acceptance gates**
3. **Keep/discard rule**
4. **Primary benchmark set**
5. **Failure budget**

If these are not explicit, the worker will optimize the wrong thing.

## Factory Modes

### 1. Fixed-Shape Kernel

Use for:
- training kernels
- inference kernels
- model-specific fused ops
- workloads with known dimensions and controlled layouts

Primary objective:
- maximize throughput / minimize latency on a declared shape set

Hard gates:
- correctness on declared shape set
- no crash on nearby non-primary shapes if they are supposed to be supported

Keep rule:
- keep improvements that improve the weighted benchmark set and do not regress
  correctness or nearby-shape sanity checks

Examples:
- flash attention for a known head dimension
- GEMM tuned for a model-specific batch/hidden-size regime

### 2. General-Shape Library

Use for:
- BLAS/LAPACK exports
- Octave / NumPy / SciPy compatibility layers
- reusable primitives that will see arbitrary shapes and strides

Primary objective:
- maximize **coverage-weighted performance**

Hard gates:
- all supported shapes and strides must be correct
- non-contiguous inputs / `lda > m` style layouts must work
- edge shapes must not crash

Keep rule:
- do not keep an optimization that speeds up the primary benchmark if it breaks
  shape coverage, stride support, or causes large cliffs on nearby sizes

Crossover policy:
- CPU/GPU crossover is allowed when it is derived from empirical sweeps across
  representative shape classes.
- The decision is end-to-end placement cost, not matrix size alone.
- A nominally small operand does not automatically belong on CPU if its partner
  or the surrounding pipeline is already on GPU.
- It is often better to keep the whole operation on GPU and pay one small copy
  than to split residency and introduce synchronization and transfer overhead.
- The crossover must preserve identical semantics, flag support, stride support,
  and edge-case behavior.
- Thresholds should be treated as measured policy, not guesswork.

Suggested scorecard:
- correctness gate: pass/fail
- shape coverage: percentage of shape matrix passing
- stride coverage: percentage of stride cases passing
- robustness: singular / degenerate / mixed-scale handling
- weighted performance: benchmark median over the declared benchmark suite

Examples:
- `dgemv`, `dtrmm`, `dlange`, `dgesv`, `dpotrs`

### 3. Numerical Method

Use for:
- iterative refinement
- Krylov solvers
- conditioning-aware algorithms
- compensated summation
- mixed-precision schemes where accuracy is the point

Primary objective:
- improve numerical behavior at acceptable cost

Hard gates:
- convergence behavior defined up front
- residual / backward-error targets
- stability on ill-conditioned inputs

Keep rule:
- keep changes that improve the declared numerical metric even if raw kernel
  time is slightly worse, provided the end-to-end objective improves

Typical metrics:
- residual norm
- backward error
- forward error
- iteration count to convergence
- robustness threshold vs condition number
- end-to-end solve time

Examples:
- GMRES iterative refinement
- compensated reductions
- mixed-precision LU solve recovery

### 4. Alternative Arithmetic Primitive

Use for:
- integer math libraries
- ternary / low-bit primitives
- custom packing or exact arithmetic kernels

Primary objective:
- correctness of arithmetic semantics first, throughput second

Hard gates:
- bit-exact or rule-exact output semantics
- packing / unpacking correctness
- compatibility with the target higher-level method

Keep rule:
- do not trade semantic correctness for speed

Examples:
- ternary matmul primitives
- packed integer reductions

### 5. Research / Exploration

Use for:
- proof-of-concept implementations
- paper reproduction
- feasibility studies

Primary objective:
- answer a research question cheaply and cleanly

Hard gates:
- hypothesis stated clearly
- result recorded clearly

Keep rule:
- keep only if the experiment advances the research question, not just if it is
  faster on one benchmark

## Objective Vector

Each project spec should define a vector like:

```text
objective_vector =
  correctness
  coverage
  numerical_quality
  weighted_performance
  maintainability
```

Not every axis must be optimized equally, but every project must state:
- which axes are hard gates
- which axis is primary
- which regressions are acceptable

## Keep / Discard Rules

### Keep a change when:

- it passes all hard gates
- it improves the primary objective
- it does not create an unacceptable regression on secondary objectives

### Discard a change when:

- it fails any hard gate
- it only improves one cherry-picked benchmark while hurting declared coverage
- it moves a numerical-method project backward on residual / convergence
- it adds complexity without materially improving the objective vector

## Benchmark Sets

Every project should define one of these benchmark sets explicitly.

### Fixed-Shape Set
- 1-5 named shapes
- optional weighted traffic mix

### Coverage Set
- tiny boundary sizes: 0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65
- medium sizes around tile boundaries
- large square and rectangular sizes
- stride / submatrix cases
- degenerate and mixed-scale cases where applicable

### Numerical Set
- well-conditioned inputs
- ill-conditioned inputs
- adversarial scale separation
- convergence-stress cases

## Factory Design Rule

The factory should automate the **evaluation contract**, not just the benchmark.

That means:
- one shared loop framework
- different objective profiles
- different test suites and benchmark suites
- a stable worker prompt that reads the profile and acts accordingly

This is how token cost stays closer to O(1):
- define the evaluation contract once
- reuse it across projects
- stop rediscovering what “better” means for every new domain
