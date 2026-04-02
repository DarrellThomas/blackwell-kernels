#!/usr/bin/env python3
# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
#
# Mines experiment history from factory_brain and generates the
# sm_120 Optimization Playbook — a decision tree keyed by stall type.
#
# Run after every batch of experiments, or on a cron:
#   python3 common/scripts/generate_playbook.py
#
# Output: common/docs/sm120_optimization_playbook.md
#         + copies to all active worker docs/ folders

import os
import re
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import sys

WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT = WORKSPACE / "common/docs/sm120_optimization_playbook.md"
WORKER_DIRS = ["dotproduct", "rmsnorm", "fused-mlp", "linalg", "lu", "qr",
               "spmv", "numerical", "main", "gemm", "attention"]
sys.path.insert(0, str(WORKSPACE / "common/memory"))

def collect_all_experiments():
    """Collect experiments from factory_brain."""
    from factory_brain import ResearchMemory

    mem = ResearchMemory()
    rows = mem.conn.execute("""
        SELECT kernel_type, duration_us, vs_ref, sm_pct, stall_math, stall_wait,
               stall_scoreboard, stall_barrier, top_stall, status, description
        FROM experiments
        WHERE kernel_type != ''
        ORDER BY kernel_type, COALESCE(timestamp, recorded_at) ASC, id ASC
    """).fetchall()
    mem.close()

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["kernel_type"]].append({
            "project": row["kernel_type"],
            "duration_us": float(row["duration_us"] or 0.0),
            "vs_ref": float(row["vs_ref"] or 0.0),
            "sm_pct": float(row["sm_pct"] or 0.0),
            "stall_math": str(row["stall_math"] or "").strip(),
            "stall_wait": str(row["stall_wait"] or "").strip(),
            "stall_scoreboard": str(row["stall_scoreboard"] or "").strip(),
            "stall_barrier": str(row["stall_barrier"] or "").strip(),
            "top_stall": (str(row["top_stall"] or "").strip() or "unknown"),
            "status": str(row["status"] or "").strip().lower(),
            "description": str(row["description"] or "").strip(),
        })

    all_exps = []
    for kernel_name in sorted(grouped):
        exps = grouped[kernel_name]
        all_exps.extend(exps)
        kept = sum(1 for e in exps if "keep" in e["status"])
        disc = sum(1 for e in exps if "discard" in e["status"])
        print(f"  {kernel_name}: {len(exps)} experiments ({kept} kept, {disc} discarded)")

    return all_exps


# ─── Analysis ─────────────────────────────────────────────────────────────────

def group_by_stall(experiments):
    """Group experiments by top_stall and status."""
    kept_by_stall = defaultdict(list)
    discarded_by_stall = defaultdict(list)

    for e in experiments:
        if "keep" in e["status"]:
            kept_by_stall[e["top_stall"]].append(e)
        elif "discard" in e["status"]:
            discarded_by_stall[e["top_stall"]].append(e)

    return kept_by_stall, discarded_by_stall


def extract_dead_ends(discarded):
    """Find recurring failure patterns from discarded experiments."""
    # Count description patterns
    pattern_counts = defaultdict(list)
    for e in discarded:
        # Normalize description to find patterns
        desc = e["description"].lower()
        # Extract key phrases
        for pattern in [
            "3-stage", "triple buffer", "three stage",
            "launch_bounds", "register spill", "spill",
            "block index remap", "remapping",
            "two-phase", "two phase", "second kernel",
            "preload all", "pre-load all",
            "manual ptx", "ptx schedul",
            "q in shared", "q-in-smem", "q in smem",
            "full fusion", "fused gate.*down",
            "loop reorder",
            "128×128", "128x128",
            "8 warps",
            "warp-per-row",
            "cp.async.*cg", ".cg evict",
        ]:
            if re.search(pattern, desc):
                pattern_counts[pattern].append(e)
                break

    return pattern_counts


def extract_breakthroughs(kept):
    """Find the biggest improvements — techniques worth highlighting."""
    breakthroughs = []
    for e in kept:
        if e["vs_ref"] >= 1.2 and e["top_stall"] != "unknown":
            breakthroughs.append(e)
    return sorted(breakthroughs, key=lambda e: -e["vs_ref"])


def find_proven_techniques(kept_by_stall):
    """Extract concrete technique descriptions from kept experiments, deduplicated."""
    techniques = {}  # stall -> list of (description, project, vs_ref)

    for stall, exps in kept_by_stall.items():
        if stall == "unknown":
            continue
        seen_descs = set()
        techs = []
        for e in sorted(exps, key=lambda x: -x["vs_ref"]):
            # Normalize to avoid near-duplicates
            norm = re.sub(r'v\d+[:\s]', '', e["description"].lower()).strip()
            norm = re.sub(r'exp\d+[:\s]', '', norm).strip()
            norm = norm[:60]  # truncate for dedup
            if norm not in seen_descs:
                seen_descs.add(norm)
                techs.append({
                    "desc": e["description"],
                    "project": e["project"],
                    "vs_ref": e["vs_ref"],
                    "sm_pct": e["sm_pct"],
                })
        techniques[stall] = techs[:15]  # cap at 15 per stall

    return techniques


def find_failed_techniques(discarded_by_stall):
    """Extract concrete failure descriptions, deduplicated."""
    failures = {}

    for stall, exps in discarded_by_stall.items():
        if stall == "unknown":
            continue
        seen_descs = set()
        fails = []
        for e in exps:
            norm = re.sub(r'v\d+[:\s]', '', e["description"].lower()).strip()
            norm = re.sub(r'exp\d+[:\s]', '', norm).strip()
            norm = norm[:60]
            if norm not in seen_descs:
                seen_descs.add(norm)
                fails.append({
                    "desc": e["description"],
                    "project": e["project"],
                    "vs_ref": e["vs_ref"],
                })
        failures[stall] = fails[:15]

    return failures


# ─── Playbook generation ─────────────────────────────────────────────────────

STALL_EXPLANATIONS = {
    "long_scoreboard": "Warps stalled waiting for data from DRAM/L2. Kernel is bandwidth-bound or latency-bound.",
    "math_throttle": "Tensor core / FMA input FIFO full. Instructions arriving faster than pipe can consume.",
    "barrier": "Warps waiting at __syncthreads() for other warps to arrive.",
    "wait": "Warps waiting for MMA result (data dependency) or cp.async completion.",
    "not_selected": "Warp scheduler has nothing to issue. Too few warps in flight.",
    "short_scoreboard": "Shared memory load latency. Warps waiting for smem data.",
    "lg_throttle": "Local/global memory pipe throttled.",
}

STALL_ORDER = ["long_scoreboard", "math_throttle", "barrier", "wait", "not_selected"]

UNIVERSAL_DEAD_ENDS = [
    ("3-stage pipeline", "Kills L1 cache on sm_120 — L1 and smem share 128KB. Triple-buffering pushes smem past the point where L1 thrashes. Every project that tried it regressed.", "gemm, attention, fused-mlp"),
    ("Manual PTX scheduling", "ptxas reorders back to compiler-preferred schedule. 7 approaches tried across attention, all performance-neutral. Cannot beat compiler from C++.", "attention (7 attempts)"),
    ("launch_bounds forcing register spills", "Even 8 bytes of spill is catastrophic for bandwidth-bound kernels. Spill goes to local memory (backed by L1/DRAM).", "dotproduct, attention, gemm"),
    ("Full operator fusion across GEMM boundaries", "O(D_out/BLOCK_N) redundant recomputation when intermediates exceed tile size. 7.8-51.5x slowdown.", "fused-mlp"),
    ("Two-phase reduction kernels", "Second kernel launch overhead (2-3 us) exceeds any reduction benefit for bandwidth-bound kernels.", "dotproduct"),
    ("Block index remapping", "Destroys L2 locality. Sequential block indices map to sequential L2 cache lines.", "attention"),
    ("Preloading all fragments to registers", "Compiler already interleaves optimally with #pragma unroll. Extra live regs hurt occupancy.", "gemm, attention"),
    ("Scalar smem loads replacing ldmatrix", "ldmatrix is warp-collective 128-bit. Scalar uint16 = 16-bit with bank conflicts. 19% regression.", "attention FP8"),
    ("cp.async with .cg hint for large D", "L1 eviction at D>=4096. .cg bypasses L1, which is fine for streaming but bad when data is reused.", "rmsnorm"),
]

PROVEN_PATTERNS = {
    "Bandwidth-Bound (dotproduct, rmsnorm, BLAS1)": [
        "float4 vectorized loads (16B per load)",
        "Grid-stride loop with 4-8x unroll (8 independent loads in flight)",
        "FMA intrinsics (__fmaf_rn — one instruction vs MUL+ADD)",
        "Warp shuffle reduction (__shfl_xor_sync butterfly)",
        "Single atomicAdd per block (not per warp — 2720→170 atomics)",
        "Streaming loads (ld.global.cs) for data > L2 (96MB)",
        "Auto-tune block size per problem size",
        "= 89.4% of peak bandwidth (1602/1792 GB/s)",
    ],
    "Compute-Bound GEMM (BF16, FP8)": [
        "64x64 tiles, 4 warps, 80 regs, 6 blocks/SM",
        "cp.async double-buffer pipelining",
        "XOR swizzle for bank conflicts",
        "Non-volatile MMA (asm not asm volatile)",
        "ldmatrix_x4_mma (baked a1/a2 swap, eliminates MOVs)",
        "Stream B fragments per-tile (fewer live regs)",
        "__launch_bounds__(128, 6)",
        "= 0.98x cuBLAS (BF16), 1.34x cuBLAS (FP8)",
    ],
    "Compute-Bound Attention (BF16, FP8)": [
        "BQ=64 BKV=64, 4 warps, 145 regs, 3 blocks/SM",
        "cp.async double-buffer, XOR swizzle",
        "Register-only P→A conversion (no smem round-trip)",
        "exp2f softmax (LOG2E folded into Q scale, saves 34 MULs/iter)",
        "Skip mask for unmasked KV blocks (-60 conditionals/iter)",
        "Dynamic BQ dispatch (128 for large grids)",
        "Prefetch after QK^T not before (avoids load contention)",
        "Vectorized FP8 conversion (cvt.e4m3x2.f32)",
        "= 1.76x SDPA (BF16), 2.33x SDPA (FP8)",
    ],
    "Fused Epilogue (fused-mlp)": [
        "Epilogue fusion only (activation fused into GEMM output)",
        "Do NOT fuse across GEMM boundaries",
        "BLOCK_K=64 for fewer barriers",
        "ldmatrix_x4_trans for B matrix",
        "= 1.07-1.22x PyTorch",
    ],
}

REG_BUDGET = [
    (44, 11, 44),
    (64, 7, 28),
    (80, 6, 24),
    (96, 5, 20),
    (128, 3, 12),
    (145, 3, 12),
    (165, 2, 8),
    (255, 1, 4),
]

QUICK_REFERENCE = [
    ("long_scoreboard", "Bandwidth-bound", "More blocks/SM (increase occupancy)"),
    ("long_scoreboard", "Compute-bound", "Check if data fits L2 → use .ca not .cs"),
    ("math_throttle", "GEMM", "Occupancy-first (smaller tiles, more blocks)"),
    ("math_throttle", "Attention", "Probably near ceiling. Check vs our 1.76x."),
    ("math_throttle", "Fused", "BLOCK_K=64, ldmatrix_x4_trans for B"),
    ("barrier", "Reduction", "Warp-level reduction, single atomicAdd per block"),
    ("barrier", "Any", "Fewer main-loop iterations (larger BLOCK_K)"),
    ("not_selected", "Any", "Reduce registers via __launch_bounds__"),
    ("wait", "Any", "Double-buffer pipelining (NOT triple)"),
]


def generate_playbook(experiments):
    """Generate the full playbook markdown."""
    kept_by_stall, discarded_by_stall = group_by_stall(experiments)
    proven = find_proven_techniques(kept_by_stall)
    failed = find_failed_techniques(discarded_by_stall)

    total_kept = sum(len(v) for v in kept_by_stall.values())
    total_disc = sum(len(v) for v in discarded_by_stall.values())
    total = total_kept + total_disc
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("# sm_120 Optimization Playbook — Empirical Decision Tree")
    lines.append("")
    lines.append(f"**Auto-generated from {total} experiments across {len(set(e['project'] for e in experiments))} kernel projects on RTX 5090 (sm_120)**")
    lines.append(f"**{total_kept} kept, {total_disc} discarded. Every entry is measured, not theoretical.**")
    lines.append(f"**Last updated: {timestamp}**")
    lines.append("")
    lines.append("This is the optimization vocabulary for this chip. When you see a stall in ncu,")
    lines.append("look it up here and pick a technique. Do NOT invent approaches — use what's proven.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## How to Use This Document")
    lines.append("")
    lines.append("1. Run ncu, identify your **top stall**")
    lines.append("2. Find the matching section below")
    lines.append("3. Pick a technique that applies to your kernel type")
    lines.append("4. Try it, measure, keep or discard")
    lines.append("5. Check \"Universal Dead Ends\" BEFORE trying anything")
    lines.append("")
    lines.append("---")

    # Per-stall sections
    for stall in STALL_ORDER:
        explanation = STALL_EXPLANATIONS.get(stall, "")
        kept_count = len(kept_by_stall.get(stall, []))
        disc_count = len(discarded_by_stall.get(stall, []))

        lines.append("")
        lines.append(f"## {stall.upper()} — {explanation}")
        lines.append("")
        lines.append(f"*{kept_count} kept, {disc_count} discarded across all projects.*")

        # Proven techniques
        if stall in proven and proven[stall]:
            lines.append("")
            lines.append("### Proven Techniques")
            lines.append("")
            lines.append("| Technique | Project | vs_ref |")
            lines.append("|-----------|---------|--------|")
            for t in proven[stall]:
                desc_short = t["desc"][:80].replace("|", "/")
                lines.append(f"| {desc_short} | {t['project']} | {t['vs_ref']:.2f}x |")

        # Failed techniques
        if stall in failed and failed[stall]:
            lines.append("")
            lines.append("### What FAILED")
            lines.append("")
            for f in failed[stall]:
                desc_short = f["desc"][:100].replace("|", "/")
                lines.append(f"- **{f['project']}** ({f['vs_ref']:.2f}x): {desc_short}")

        lines.append("")
        lines.append("---")

    # Universal dead ends
    lines.append("")
    lines.append("## UNIVERSAL DEAD ENDS — Never Try These on sm_120")
    lines.append("")
    lines.append("These failed across multiple projects. Do not retry them.")
    lines.append("")
    lines.append("| Dead End | Projects | Root Cause |")
    lines.append("|----------|----------|------------|")
    for name, cause, projects in UNIVERSAL_DEAD_ENDS:
        cause_short = cause[:100].replace("|", "/")
        lines.append(f"| **{name}** | {projects} | {cause_short} |")

    # Proven patterns
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## PROVEN PATTERNS — Reusable Building Blocks")
    for pattern_name, steps in PROVEN_PATTERNS.items():
        lines.append("")
        lines.append(f"### {pattern_name}")
        lines.append("```")
        for step in steps:
            lines.append(step)
        lines.append("```")

    # Register budget
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## REGISTER BUDGET (sm_120, 128 threads/block)")
    lines.append("")
    lines.append("| Regs/thread | Max blocks/SM | Warps/SM |")
    lines.append("|-------------|---------------|----------|")
    for regs, blocks, warps in REG_BUDGET:
        lines.append(f"| {regs} | {blocks} | {warps} |")

    # Compiler behavior
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## COMPILER BEHAVIOR ON sm_120")
    lines.append("")
    lines.append("Things the compiler already does well (don't fight it):")
    lines.append("- Interleaves MMA with loads when using `#pragma unroll`")
    lines.append("- Hoists loads into compute gaps (V loads into softmax gap)")
    lines.append("- Produces identical SASS for most C++ loop restructurings")
    lines.append("- Near-optimal register allocation at <=128 registers with non-volatile asm")
    lines.append("")
    lines.append("Things the compiler CANNOT do:")
    lines.append("- Cross sequential phase boundaries (softmax between QK^T and PV)")
    lines.append("- Choose between streaming and cached loads based on data size")
    lines.append("- Reduce register count below the algorithm's fundamental requirements")

    # Noise floor
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## NOISE FLOOR")
    lines.append("")
    lines.append("**Differences < 2% are noise.** 10 warmup + 100 timed iterations.")
    lines.append("If the delta is 1-2%, run 3 more trials before trusting it.")

    # Quick reference
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## QUICK REFERENCE: What To Try First")
    lines.append("")
    lines.append("| Your top stall | Kernel type | Try first |")
    lines.append("|---------------|-------------|-----------|")
    for stall, ktype, technique in QUICK_REFERENCE:
        lines.append(f"| {stall} | {ktype} | {technique} |")

    return "\n".join(lines) + "\n"


# ─── Distribution ─────────────────────────────────────────────────────────────

def distribute_playbook():
    """Copy the playbook to all active worker docs/ folders."""
    count = 0
    for worker in WORKER_DIRS:
        docs_dir = WORKSPACE / worker / "docs"
        if docs_dir.is_dir():
            dest = docs_dir / "sm120_optimization_playbook.md"
            shutil.copy2(OUTPUT, dest)
            count += 1
    return count


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Generating sm_120 Optimization Playbook")
    print("=" * 60)
    print()

    print("Collecting experiments...")
    experiments = collect_all_experiments()
    total = len(experiments)
    kept = sum(1 for e in experiments if "keep" in e["status"])
    discarded = sum(1 for e in experiments if "discard" in e["status"])
    print(f"\nTotal: {total} experiments ({kept} kept, {discarded} discarded)")

    print("\nGenerating playbook...")
    playbook = generate_playbook(experiments)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(playbook)
    print(f"Written to: {OUTPUT}")
    print(f"Size: {len(playbook)} bytes, {len(playbook.splitlines())} lines")

    print("\nDistributing to workers...")
    count = distribute_playbook()
    print(f"Copied to {count} worker docs/ folders")

    print("\nDone.")


if __name__ == "__main__":
    main()
