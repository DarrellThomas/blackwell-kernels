#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# norm() integration tests for:
# - LD_PRELOAD BLAS override path (dnrm2_ via Octave host norm)
# - gpu_matrix plugin path (gpu_norm dispatch and host fallback)

import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "lib", "libbwk_blas.so")
LIB_DIR = os.path.join(ROOT, "lib")
PLUGIN_DIR = os.path.join(ROOT, "plugin")
PLUGIN_INST_DIR = os.path.join(PLUGIN_DIR, "inst")

BASE_ENV = dict(os.environ)
BASE_ENV["CUDA_VISIBLE_DEVICES"] = BASE_ENV.get("CUDA_VISIBLE_DEVICES", "1")

passed = 0
failed = 0


def run_command(cmd, label, env=None, timeout=120):
    global passed, failed
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env or BASE_ENV,
        timeout=timeout,
        cwd=ROOT,
    )
    ok = result.returncode == 0
    if ok:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}")
        print(f"        stdout: {result.stdout.strip()[:300]}")
        print(f"        stderr: {result.stderr.strip()[:200]}")
    return ok


def run_octave(script, label, env=None, timeout=120):
    return run_command(
        ["octave-cli", "--quiet", "--eval", script],
        label,
        env=env,
        timeout=timeout,
    )


PRELOAD_ENV = dict(BASE_ENV)
PRELOAD_ENV["LD_PRELOAD"] = LIB

PLUGIN_SETUP = f"""
  warning('off', 'Octave:shadowed-function');
  addpath('{PLUGIN_DIR}');
  addpath('{PLUGIN_INST_DIR}');
"""


run_command(
    ["make", "-C", LIB_DIR, "test-dnrm2-abi"],
    "dnrm2 abi",
)

run_octave(
    """
  x = [3; 4; 12];
  y = [1e308; 1e-308];
  z = builtin('zeros', 0, 1);
  assert(abs(norm(x) - 13.0) < 1e-15);
  assert(abs(norm(y) - 1e308) <= 1e-15 * 1e308);
  assert(norm(z) == 0.0);
""",
    "preload host vector norms",
    env=PRELOAD_ENV,
)

run_octave(
    PLUGIN_SETUP
    + """
  col_vec = [3; 4; 12];
  row_vec = [3, 4, 12];
  mixed_vec = [1e308; 1e-308];
  empty_vec = builtin('zeros', 0, 1);

  assert(abs(norm(gpu_create(col_vec)) - norm(col_vec)) < 1e-15);
  assert(abs(norm(gpu_create(col_vec), 2) - norm(col_vec, 2)) < 1e-15);
  assert(abs(norm(gpu_create(col_vec), 1) - norm(col_vec, 1)) < 1e-15);
  assert(abs(norm(gpu_create(col_vec), Inf) - norm(col_vec, Inf)) < 1e-15);
  assert(abs(norm(gpu_create(col_vec), -Inf) - norm(col_vec, -Inf)) < 1e-15);
  assert(abs(norm(gpu_create(row_vec)) - norm(row_vec)) < 1e-15);
  assert(abs(norm(gpu_create(mixed_vec)) - norm(mixed_vec)) <= 1e-15 * norm(mixed_vec));
  assert(norm(gpu_create(empty_vec)) == 0.0);
""",
    "plugin vector norms and empty vector",
)

run_octave(
    PLUGIN_SETUP
    + """
  A = reshape([1, 2, 3, 4, 5, 6], 2, 3);
  Ag = gpu_create(A);

  assert(abs(norm(Ag) - norm(A)) < 1e-12);
  assert(abs(norm(Ag, 2) - norm(A, 2)) < 1e-12);
  assert(abs(norm(Ag, 1) - norm(A, 1)) < 1e-15);
  assert(abs(norm(Ag, Inf) - norm(A, Inf)) < 1e-15);
  assert(abs(norm(Ag, 'fro') - norm(A, 'fro')) < 1e-15);
  assert(norm(norm(Ag, [], 'rows') - norm(A, [], 'rows'), 'fro') < 1e-15);
  assert(norm(norm(Ag, 1, 'cols') - norm(A, 1, 'cols'), 'fro') < 1e-15);
""",
    "plugin matrix norms and rows-cols dispatch",
)

run_octave(
    PLUGIN_SETUP
    + """
  Z = builtin('zeros', 0, 0);
  Zg = gpu_create(Z);
  rows_norm = norm(Zg, [], 'rows');
  cols_norm = norm(Zg, 1, 'cols');

  assert(norm(Zg) == 0.0);
  assert(isequal(size(rows_norm), [0, 1]));
  assert(isequal(size(cols_norm), [1, 0]));
""",
    "plugin empty matrix norms",
)

print()
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
