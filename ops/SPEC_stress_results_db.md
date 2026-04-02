# Spec: Stress Test Results in Factory DB (Job #39)

## Goal

Every stress test result — good, bad, ugly — goes into the factory database.
The data is searchable, queryable, and grows with every run. Workers can
query it to understand dispatch regimes. The crossover map IS the product.

## Schema: `stress_results` table

```sql
CREATE TABLE stress_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,           -- groups results from same run (UUID or timestamp)
    run_timestamp TEXT NOT NULL,    -- ISO 8601

    -- Test identity
    suite TEXT NOT NULL,            -- gemm, lu, chol, qr, syrk, gemv, dot, solve, svd, eig, ...
    case_name TEXT NOT NULL,        -- n64, n128_k1e+06, 512x64_eps1e-06, ...
    metric TEXT NOT NULL,           -- relerr, duration_ms, gflops, vs_ref, bool, pass, fail

    -- Parsed dimensions (from case_name, for querying)
    size_m INTEGER,                -- M dimension (NULL if not applicable)
    size_n INTEGER,                -- N dimension (NULL for square, or column count)

    -- Results
    value REAL NOT NULL,           -- measured value
    threshold REAL,                -- acceptance threshold (NULL for benchmarks)
    status TEXT NOT NULL,          -- ok, fail, skip, error

    -- Backend context
    backend TEXT,                  -- bwk, cublas, cpu, or NULL for correctness tests

    -- Benchmark-specific (NULL for correctness tests)
    duration_ms REAL,
    gflops REAL,
    vs_cublas REAL,
    vs_cpu REAL,

    -- Provenance
    source_file TEXT,              -- which .tsv or test script produced this
    gpu_id INTEGER,                -- which GPU (0 or 1)
    cuda_version TEXT,             -- e.g., "13.2"

    -- Profile tag (e.g., "heavy", "quick", "crossover")
    profile TEXT
);

-- Indexes for common queries
CREATE INDEX idx_stress_suite_size ON stress_results(suite, size_m);
CREATE INDEX idx_stress_status ON stress_results(status);
CREATE INDEX idx_stress_run ON stress_results(run_id);
CREATE INDEX idx_stress_backend ON stress_results(backend, suite);
```

## CLI Commands (add to factory_brain.py)

```bash
# Record a single result
fb stress-add --suite gemm --case n4096 --metric relerr \
    --value 8.1e-16 --threshold 1e-11 --status ok \
    --backend bwk --run-id "20260330T072216Z"

# Batch ingest from TSV
fb stress-ingest /path/to/results.tsv --profile heavy

# Query
fb stress-query --suite gemm --status fail              # what's broken
fb stress-query --suite gemm --metric vs_cublas          # crossover data
fb stress-query --suite all --status fail --latest       # latest failures
fb stress-summary                                        # per-suite pass/fail counts
fb stress-crossover --suite gemm                         # show size where cublas wins

# Compare runs
fb stress-diff <run_id_1> <run_id_2>                    # what changed between runs
```

## HTTP API Endpoints

```
POST /api/stress/result     — record one result (JSON body)
POST /api/stress/ingest     — batch ingest from TSV upload
GET  /api/stress/query      — query with filters
GET  /api/stress/summary    — pass/fail counts per suite
GET  /api/stress/crossover  — crossover analysis per operation
```

## Modifications to Stress Test Code

### stress_plugin_enhanced.m

After each test case, append to DB via HTTP:
```octave
function record_result(suite, case_name, metric, value, threshold, status)
    % POST to factory DB
    url = 'http://localhost:8421/api/stress/result';
    body = sprintf('{"suite":"%s","case":"%s","metric":"%s","value":%g,"threshold":%g,"status":"%s","run_id":"%s"}', ...
        suite, case_name, metric, value, threshold, status, getenv('BWK_RUN_ID'));
    urlread(url, 'post', body);  % Octave HTTP
end
```

Or shell out to the CLI:
```octave
system(sprintf('fb stress-add --suite %s --case %s --metric %s --value %g --threshold %g --status %s', ...
    suite, case_name, metric, value, threshold, status));
```

### bench_gemm_crossover.m, bench_plugin.m, showcase scripts

Same pattern — record every benchmark result with backend tag.

## Ingest Existing Data

One-time migration of existing TSVs in `octave-gpu/results/`:
```bash
fb stress-ingest octave-gpu/results/stress_plugin_enhanced_heavy_20260330T072216Z.tsv --profile heavy
fb stress-ingest octave-gpu/results/bench_gemm_crossover_20260330T072451Z.tsv --profile crossover
fb stress-ingest octave-gpu/results/showcase_bwk.tsv --backend bwk --profile showcase
fb stress-ingest octave-gpu/results/showcase_cublas.tsv --backend cublas --profile showcase
fb stress-ingest octave-gpu/results/showcase_cpu.tsv --backend cpu --profile showcase
```

## Worker Access via msearch

Once ingested, workers can query stress results through the existing msearch interface
if we index summaries as research chunks. Or add a dedicated `fb stress-query` path.

## Key Design Decisions

1. **Every result is recorded** — pass, fail, and everything in between. No filtering.
2. **Run IDs group results** — so you can compare full runs over time.
3. **Parsed dimensions** — so you can query "show me all GEMM results for N > 2048"
4. **Backend tagging** — so you can compare CPU vs cuBLAS vs BWK at the same size.
5. **TSV files still written** — the DB is additive, not a replacement. TSVs are the archive.
