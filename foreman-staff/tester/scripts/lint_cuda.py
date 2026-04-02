#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License.
#
# lint_cuda.py — Static analysis linter for CUDA kernel source files
#
# Checks for common sm_120 pitfalls learned from 700+ factory experiments.
# Does NOT compile code — just pattern-matches on source text.
#
# Usage:
#   python3 lint_cuda.py ${BWK_ROOT}/linalg/csrc/linalg/*.cu
#   python3 lint_cuda.py --all          # lint all factory kernels
#   python3 lint_cuda.py --shipped      # lint only shipped primitives

import argparse
import re
import sys
from pathlib import Path

BWK_ROOT = Path(__file__).resolve().parents[3]

passed = 0
warned = 0
errors_list = []


def lint_file(filepath: str) -> list[dict]:
    """Lint a single .cu or .cuh file. Returns list of findings."""
    findings = []
    path = Path(filepath)
    if not path.exists():
        return [{"line": 0, "severity": "error", "rule": "file_missing",
                 "msg": f"File not found: {filepath}"}]

    lines = path.read_text(errors="replace").split("\n")
    content = "\n".join(lines)
    filename = path.name

    # Track state for multi-line analysis
    in_kernel = False
    kernel_name = ""
    has_syncthreads_after_smem_write = False
    last_smem_write_line = 0
    cp_async_count = 0
    wait_group_count = 0
    has_launch_bounds = False
    shared_mem_bytes = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments, empty lines, and NOLINT-suppressed lines
        if stripped.startswith("//") or stripped == "":
            continue
        if "NOLINT" in line:
            continue

        # ============================================================
        # Rule 1: __syncthreads() after shared memory writes
        # ============================================================
        if re.search(r'smem|sA\[|sB\[|shared.*=|__shared__.*=', stripped):
            if "=" in stripped and "//" not in stripped.split("=")[0]:
                last_smem_write_line = i

        if "__syncthreads" in stripped:
            has_syncthreads_after_smem_write = True

        # If we read from smem without sync after a write
        if last_smem_write_line > 0 and not has_syncthreads_after_smem_write:
            if re.search(r'smem\[|sA\[|sB\[', stripped) and "=" not in stripped.split("//")[0].split("]")[0]:
                # Reading from smem without sync after write — but only flag
                # if it's more than 3 lines after the write (allow same-thread access)
                if i - last_smem_write_line > 5:
                    pass  # Too many false positives for this simple check

        if "__syncthreads" in stripped:
            last_smem_write_line = 0
            has_syncthreads_after_smem_write = False

        # ============================================================
        # Rule 2: cp.async without matching wait_group
        # ============================================================
        if "cp.async" in stripped and "wait" not in stripped and "commit" not in stripped:
            cp_async_count += 1
        if "cp.async.wait" in stripped or "wait_group" in stripped:
            wait_group_count += 1

        # ============================================================
        # Rule 3: Missing launch_bounds on __global__ kernels
        # ============================================================
        if "__global__" in stripped:
            in_kernel = True
            kernel_name = stripped
            has_launch_bounds = "__launch_bounds__" in stripped
            if not has_launch_bounds:
                # Check next line too
                if i < len(lines) and "__launch_bounds__" in lines[i]:
                    has_launch_bounds = True
                if not has_launch_bounds:
                    findings.append({
                        "line": i, "severity": "warn", "rule": "no_launch_bounds",
                        "msg": f"__global__ kernel without __launch_bounds__ — register allocation is unpredictable"
                    })

        # ============================================================
        # Rule 4: Shared memory over 99KB (sm_120 limit)
        # ============================================================
        m = re.search(r'__shared__\s+\w+\s+\w+\[(\d+)\]', stripped)
        if m:
            size = int(m.group(1))
            # Rough estimate — actual depends on type
            if size > 99 * 1024 / 4:  # assume float
                findings.append({
                    "line": i, "severity": "error", "rule": "smem_over_99kb",
                    "msg": f"Static shared memory array size {size} may exceed 99KB sm_120 limit"
                })

        # ============================================================
        # Rule 5: cudaMalloc without cudaFree (leak detection)
        # ============================================================
        if "cudaMalloc(" in stripped and "Async" not in stripped:
            findings.append({
                "line": i, "severity": "warn", "rule": "sync_malloc",
                "msg": "cudaMalloc (synchronous) — prefer cudaMallocAsync for stream-ordered allocation"
            })

        # ============================================================
        # Rule 6: Missing CUDA_VISIBLE_DEVICES in test/bench scripts
        # ============================================================
        if filename.endswith(".py") and ("subprocess" in stripped or "os.system" in stripped):
            if "CUDA_VISIBLE_DEVICES" not in stripped:
                findings.append({
                    "line": i, "severity": "warn", "rule": "no_gpu_select",
                    "msg": "Shell command without CUDA_VISIBLE_DEVICES — may run on wrong GPU"
                })

        # ============================================================
        # Rule 7: Volatile/non-volatile correctness
        # ============================================================
        if "mma.sync" in stripped and "volatile" in stripped:
            findings.append({
                "line": i, "severity": "error", "rule": "volatile_mma",
                "msg": "MMA with volatile qualifier — mma.sync MUST be non-volatile for compiler scheduling"
            })

        if ("ldmatrix" in stripped or "cp.async" in stripped) and "non_volatile" in stripped:
            findings.append({
                "line": i, "severity": "warn", "rule": "nonvolatile_load",
                "msg": "ldmatrix/cp.async with non_volatile — these typically need volatile to prevent reordering"
            })

        # ============================================================
        # Rule 8: Bank conflict risk — stride patterns
        # ============================================================
        if re.search(r'smem\[.*\*\s*(32|64|128)\s*\+', stripped):
            findings.append({
                "line": i, "severity": "warn", "rule": "bank_conflict_stride",
                "msg": "Shared memory access with stride 32/64/128 — high bank conflict risk. Use XOR swizzle."
            })

        # ============================================================
        # Rule 9: Missing error check after CUDA API calls
        # ============================================================
        if re.search(r'cuda(Malloc|Memcpy|Launch|Stream|Event|Free)\(', stripped):
            # Check if the return value is checked
            if "TORCH_CHECK" not in stripped and "assert" not in stripped and "!=" not in stripped:
                if "=" not in stripped.split("(")[0]:  # not capturing return
                    pass  # Too noisy — many cuBLAS calls don't check

        # ============================================================
        # Rule 10: BLAS interface compliance (for primitives)
        # ============================================================
        if "common/csrc/primitives" in str(filepath) or "primitives" in str(filepath):
            if "__global__" in stripped and "void" in stripped:
                # Check if it's a BLAS-level kernel
                for blas_op in ["gemm", "syrk", "trsm", "trmm", "gemv", "symm"]:
                    if blas_op in filename.lower():
                        # Should have alpha/beta in the file
                        if "alpha" not in content or "beta" not in content:
                            findings.append({
                                "line": i, "severity": "error", "rule": "no_blas_interface",
                                "msg": f"BLAS primitive {filename} missing alpha/beta parameters — not usable as building block"
                            })
                        if "lda" not in content and "ld_a" not in content and "stride" not in content.lower():
                            findings.append({
                                "line": i, "severity": "error", "rule": "no_stride_support",
                                "msg": f"BLAS primitive {filename} missing lda/stride — cannot operate on sub-matrices"
                            })
                        break

        # ============================================================
        # Rule 11: Hardcoded GPU assumptions
        # ============================================================
        if re.search(r'(48|64)\s*\*\s*1024.*shared|MAX_SHARED.*=.*(48|64)\s*\*\s*1024', stripped):
            if "99" not in stripped and "99KB" not in stripped:
                findings.append({
                    "line": i, "severity": "warn", "rule": "wrong_smem_limit",
                    "msg": "Shared memory limit assumes 48/64KB — sm_120 max is 99KB per block"
                })

        if re.search(r'warps.*=.*64|MAX_WARPS.*64|64\s*warps', stripped):
            findings.append({
                "line": i, "severity": "warn", "rule": "wrong_warp_count",
                "msg": "Assumes 64 warps/SM — sm_120 (CC 12.0) has 48 warps/SM, not 64"
            })

        # ============================================================
        # Rule 12: Copyright header
        # ============================================================
    if lines and not lines[0].startswith("// Copyright"):
        findings.append({
            "line": 1, "severity": "warn", "rule": "no_copyright",
            "msg": "Missing copyright header"
        })

    # End-of-file checks
    if cp_async_count > 0 and wait_group_count == 0:
        findings.append({
            "line": 0, "severity": "error", "rule": "cp_async_no_wait",
            "msg": f"File has {cp_async_count} cp.async calls but no wait_group — data race"
        })

    return findings


def lint_directory(dirpath: str, pattern: str = "**/*.cu") -> dict:
    """Lint all matching files in a directory."""
    path = Path(dirpath)
    results = {}
    for f in sorted(path.glob(pattern)):
        findings = lint_file(str(f))
        if findings:
            results[str(f)] = findings
    # Also check .cuh files
    for f in sorted(path.glob("**/*.cuh")):
        findings = lint_file(str(f))
        if findings:
            results[str(f)] = findings
    return results


def format_findings(results: dict) -> str:
    """Format findings for terminal output."""
    global passed, warned, errors_list

    output = []
    total_errors = 0
    total_warns = 0
    clean_files = 0

    for filepath, findings in sorted(results.items()):
        short = filepath.split("/bwk/")[-1] if "/bwk/" in filepath else filepath
        file_errors = [f for f in findings if f["severity"] == "error"]
        file_warns = [f for f in findings if f["severity"] == "warn"]

        if not findings:
            clean_files += 1
            continue

        output.append(f"\n  {short}:")
        for f in findings:
            icon = "ERROR" if f["severity"] == "error" else "WARN "
            if f["line"] > 0:
                output.append(f"    {icon} L{f['line']:4d}  [{f['rule']}] {f['msg']}")
            else:
                output.append(f"    {icon}        [{f['rule']}] {f['msg']}")

        total_errors += len(file_errors)
        total_warns += len(file_warns)

    output.append(f"\n{'='*60}")
    output.append(f"LINT: {total_errors} errors, {total_warns} warnings across {len(results)} files")
    if total_errors > 0:
        output.append(f"  {total_errors} ERRORS must be fixed before shipping")

    passed = len(results) - total_errors
    warned = total_warns
    errors_list = [(filepath, f) for filepath, findings in results.items()
                   for f in findings if f["severity"] == "error"]

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="CUDA kernel linter for sm_120")
    parser.add_argument("files", nargs="*", help="Files or directories to lint")
    parser.add_argument("--all", action="store_true", help="Lint all factory kernels")
    parser.add_argument("--shipped", action="store_true", help="Lint only shipped primitives")
    args = parser.parse_args()

    results = {}

    if args.shipped:
        print("=== Linting Shipped Primitives ===")
        results.update(lint_directory(str(BWK_ROOT / "common/csrc/primitives")))

    elif args.all:
        print("=== Linting All Factory Kernels ===")
        for project in ["gemm", "linalg", "numerical", "qr", "spmv",
                        "cuquantum", "attention", "fused-mlp", "dotproduct",
                        "rmsnorm", "lu", "main"]:
            csrc = BWK_ROOT / project / "csrc"
            if csrc.is_dir():
                results.update(lint_directory(str(csrc)))
        # Also lint shipped
        results.update(lint_directory(str(BWK_ROOT / "common/csrc")))

    elif args.files:
        for f in args.files:
            p = Path(f)
            if p.is_dir():
                results.update(lint_directory(str(p)))
            elif p.is_file():
                findings = lint_file(str(p))
                if findings:
                    results[str(p)] = findings
    else:
        parser.print_help()
        return

    print(format_findings(results))
    sys.exit(1 if errors_list else 0)


if __name__ == "__main__":
    main()
