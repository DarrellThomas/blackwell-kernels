#!/usr/bin/env python3
"""
Generate Level 2 summaries for all documents in the research memory database.

Uses Claude Sonnet via the Anthropic API to produce structured summaries
(200-500 words) and one-line signals for each document.

Usage:
    python generate_summaries.py                # Process all docs without summaries
    python generate_summaries.py --limit 10     # Process 10 docs
    python generate_summaries.py --doc-id 42    # Process specific doc
    python generate_summaries.py --dry-run      # Show what would be processed
    python generate_summaries.py --cost          # Estimate API cost

Requires: ANTHROPIC_API_KEY environment variable
"""

import argparse
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from common.memory.factory_brain import ResearchMemory, DB_PATH

SONNET_MODEL = "claude-sonnet-4-6"
MAX_INPUT_CHARS = 12000  # truncate very long docs to keep costs down
RATE_LIMIT_DELAY = 0.5   # seconds between API calls

SUMMARY_PROMPT_TEMPLATE = (
    'You are summarizing a technical document from a CUDA kernel optimization research database.\n'
    'The audience is a Claude AI agent optimizing GPU kernels for RTX 5090 (sm_120, mma.sync ISA).\n'
    '\n'
    'The document is titled: "{title}"\n'
    'It is categorized as: {doc_type} / {kernel_type} (provenance: {provenance})\n'
    '\n'
    'Produce exactly two outputs:\n'
    '\n'
    '1. SIGNAL: A single line (under 120 characters) that captures what this document is about.\n'
    '   Format: "[key topic]: [core finding or technique] [kernel_type, status]"\n'
    '   where status is one of: validated, theoretical, dead-end, reference\n'
    '   Example: "FP8 bank conflicts: 4-bit XOR swizzle eliminates 86K conflicts [gemm, validated]"\n'
    '\n'
    '2. SUMMARY: A structured summary of 200-500 words in this exact format:\n'
    '\n'
    'WHAT: [one sentence — what technique, finding, or topic this document covers]\n'
    'FOR: [kernel type, stall type or bottleneck, hardware target]\n'
    'FINDING: [2-3 sentences — what was discovered, measured, or concluded]\n'
    'TECHNIQUE: [2-3 sentences — how to implement it, what to change in code]\n'
    'STATUS: [validated/theoretical/dead-end/reference] — [evidence: experiment numbers, measurements, or source]\n'
    '\n'
    'If the document is source code, describe what the code does, its key optimizations,\n'
    'and how it could be reused. If it is a reference doc, summarize the key facts.\n'
    'If it describes a dead end, explain why it failed.\n'
    '\n'
    'Respond in this exact format (no markdown fences, no extra text):\n'
    '\n'
    'SIGNAL: <your signal line>\n'
    '\n'
    'SUMMARY:\n'
    'WHAT: <...>\n'
    'FOR: <...>\n'
    'FINDING: <...>\n'
    'TECHNIQUE: <...>\n'
    'STATUS: <...>\n'
    '\n'
    'Here is the document:\n'
    '\n'
    '{content}'
)


EXPERIMENT_PROMPT_TEMPLATE = (
    'You are summarizing a set of GPU kernel optimization experiments from the blackwell-kernels factory.\n'
    'The audience is a Claude AI agent optimizing CUDA kernels for RTX 5090 (sm_120, mma.sync ISA).\n'
    '\n'
    'This is experiment data for: {kernel_type}\n'
    'Total experiments: {total_rows} ({kept_count} kept, {discarded_count} discarded)\n'
    '\n'
    'Produce exactly two outputs:\n'
    '\n'
    '1. SIGNAL: A single line (under 120 characters) summarizing the experiment campaign.\n'
    '   Format: "[kernel] experiments: [N] runs, [K] kept. [key finding]. [best result]"\n'
    '   Example: "GEMM experiments: 78 runs, 9 kept. Dual-dispatch FP8 beats cuBLAS. Best: 1.34x [gemm, validated]"\n'
    '\n'
    '2. SUMMARY: A structured synthesis of 300-500 words:\n'
    '\n'
    'WHAT: [one sentence — what kernel was being optimized and the overall campaign scope]\n'
    'FOR: [kernel type, primary stall types encountered, hardware target]\n'
    'WINS: [3-5 sentences — what techniques were KEPT and why they worked. Include specific vs_ref numbers.]\n'
    'DEAD_ENDS: [3-5 sentences — the most informative DISCARDED experiments. What was tried, why it failed, what was learned. Group by pattern if possible.]\n'
    'CURRENT_STATE: [1-2 sentences — best result achieved, primary remaining bottleneck]\n'
    'STATUS: validated — [N] experiments, [K] kept, measured on RTX 5090\n'
    '\n'
    'Focus on PATTERNS, not individual rows. Group related failures together.\n'
    '"3-stage pipeline failed 3 times due to L1 cache loss" is better than listing 3 separate failures.\n'
    '\n'
    'Respond in this exact format (no markdown fences, no extra text):\n'
    '\n'
    'SIGNAL: <your signal line>\n'
    '\n'
    'SUMMARY:\n'
    'WHAT: <...>\n'
    'FOR: <...>\n'
    'WINS: <...>\n'
    'DEAD_ENDS: <...>\n'
    'CURRENT_STATE: <...>\n'
    'STATUS: <...>\n'
    '\n'
    'Here are the experiments (one per line, tab-separated):\n'
    '\n'
    '{content}'
)


def _parse_signal_summary(text: str, fallback_title: str, kernel_type: str) -> tuple[str, str]:
    """Parse SIGNAL: and SUMMARY: from an LLM response."""
    signal = ""
    summary = ""
    lines = text.split("\n")
    in_summary = False

    for line in lines:
        if line.startswith("SIGNAL:"):
            signal = line[len("SIGNAL:"):].strip()
        elif line.startswith("SUMMARY:"):
            in_summary = True
        elif in_summary:
            summary += line + "\n"

    summary = summary.strip()

    if not signal:
        signal = f"{fallback_title} [{kernel_type}]"
    if not summary:
        summary = text

    return signal, summary


def generate_summary(client, title: str, doc_type: str, kernel_type: str,
                     provenance: str, content: str) -> tuple[str, str]:
    """Call Sonnet to generate summary and signal for a document."""
    if len(content) > MAX_INPUT_CHARS:
        content = content[:MAX_INPUT_CHARS] + "\n\n[... truncated for summarization ...]"

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        title=title, doc_type=doc_type, kernel_type=kernel_type,
        provenance=provenance, content=content
    )

    response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return _parse_signal_summary(response.content[0].text.strip(), title, kernel_type)


def generate_experiment_summary(client, title: str, kernel_type: str,
                                tsv_path: str) -> tuple[str, str]:
    """Call Sonnet to synthesize a summary across all experiments in a TSV."""
    import csv

    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return f"{title} [empty]", "No experiment data."

    # Count kept/discarded
    status_col = next((h for h in headers if h.lower() == "status"), None)
    kept = sum(1 for r in rows if status_col and r.get(status_col, "").strip().lower() == "keep")
    discarded = len(rows) - kept

    # Build a compact representation — full header + all rows
    # Truncate to fit context if needed
    tsv_text = "\t".join(headers) + "\n"
    for row in rows:
        line = "\t".join(row.get(h, "") for h in headers)
        if len(tsv_text) + len(line) > MAX_INPUT_CHARS:
            tsv_text += f"\n[... {len(rows) - rows.index(row)} more rows truncated ...]"
            break
        tsv_text += line + "\n"

    prompt = EXPERIMENT_PROMPT_TEMPLATE.format(
        kernel_type=kernel_type,
        total_rows=len(rows),
        kept_count=kept,
        discarded_count=discarded,
        content=tsv_text
    )

    response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return _parse_signal_summary(response.content[0].text.strip(), title, kernel_type)


def estimate_cost(docs: list[dict]) -> dict:
    """Estimate API cost for summarizing documents."""
    total_chars = 0
    for doc in docs:
        filepath = doc["source_file"]
        try:
            content = Path(filepath).read_text(errors="replace")
            total_chars += min(len(content), MAX_INPUT_CHARS)
        except Exception:
            total_chars += 2000  # estimate

    # Rough token estimate: ~4 chars per token
    input_tokens = total_chars / 4
    output_tokens = len(docs) * 300  # ~300 tokens per summary
    # Sonnet pricing: $3/M input, $15/M output
    input_cost = (input_tokens / 1_000_000) * 3.0
    output_cost = (output_tokens / 1_000_000) * 15.0

    return {
        "documents": len(docs),
        "est_input_tokens": int(input_tokens),
        "est_output_tokens": int(output_tokens),
        "est_input_cost": round(input_cost, 4),
        "est_output_cost": round(output_cost, 4),
        "est_total_cost": round(input_cost + output_cost, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Level 2 summaries for research memory")
    parser.add_argument("--db", default=str(DB_PATH), help="Database path")
    parser.add_argument("--limit", type=int, help="Max documents to process")
    parser.add_argument("--doc-id", type=int, help="Process specific document ID")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--cost", action="store_true", help="Estimate API cost")
    parser.add_argument("--skip-source-code", action="store_true",
                        help="Skip source code files (summarize docs only)")
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    mem = ResearchMemory(args.db)

    if args.doc_id:
        # Process specific document
        row = mem.conn.execute(
            "SELECT id, source_file, title, doc_type, kernel_type, provenance "
            "FROM documents WHERE id = ?", (args.doc_id,)
        ).fetchone()
        if not row:
            print(f"Document ID {args.doc_id} not found.", file=sys.stderr)
            sys.exit(1)
        docs = [dict(row)]
    else:
        docs = mem.docs_without_summary()

    if args.skip_source_code:
        docs = [d for d in docs if d.get("doc_type") != "source_code"]

    if args.limit:
        docs = docs[:args.limit]

    if not docs:
        print("All documents already have summaries.")
        mem.close()
        return

    if args.cost:
        est = estimate_cost(docs)
        print(f"Documents to summarize: {est['documents']}")
        print(f"Estimated input tokens:  {est['est_input_tokens']:,}")
        print(f"Estimated output tokens: {est['est_output_tokens']:,}")
        print(f"Estimated cost: ${est['est_total_cost']:.4f}")
        mem.close()
        return

    if args.dry_run:
        print(f"Would process {len(docs)} documents:")
        for d in docs[:20]:
            print(f"  [{d['id']:3d}] {d['title'][:60]} ({d['kernel_type']}/{d['doc_type']})")
        if len(docs) > 20:
            print(f"  ... and {len(docs) - 20} more")
        mem.close()
        return

    # Process documents
    print(f"Generating summaries for {len(docs)} documents...\n")
    success = 0
    errors = 0

    for i, doc in enumerate(docs, 1):
        filepath = doc["source_file"]
        doc_type = doc.get("doc_type", "research")
        kernel_type = doc.get("kernel_type", "general")

        try:
            if doc_type == "experiment":
                # Experiment TSVs get a specialized synthesis prompt
                if not os.path.isfile(filepath):
                    print(f"  [{i}/{len(docs)}] SKIP (file gone): {filepath}")
                    errors += 1
                    continue
                signal, summary = generate_experiment_summary(
                    client, doc["title"], kernel_type, filepath
                )
            else:
                # Regular documents
                try:
                    content = Path(filepath).read_text(errors="replace")
                except Exception as e:
                    print(f"  [{i}/{len(docs)}] SKIP (can't read): {filepath} — {e}")
                    errors += 1
                    continue

                if len(content.strip()) < 100:
                    print(f"  [{i}/{len(docs)}] SKIP (too short): {filepath}")
                    errors += 1
                    continue

                signal, summary = generate_summary(
                    client, doc["title"], doc_type,
                    kernel_type, doc.get("provenance", "research"),
                    content
                )

            mem.set_summary(doc["id"], summary, signal)
            success += 1

            # Show progress
            label = "EXP" if doc_type == "experiment" else "DOC"
            print(f"  [{i}/{len(docs)}] [{label}] {doc['title'][:50]}")
            print(f"           Signal: {signal[:100]}")
            print(f"           Summary: {len(summary)} chars")

        except Exception as e:
            print(f"  [{i}/{len(docs)}] ERROR: {doc['title'][:50]} — {e}")
            errors += 1

        # Rate limiting
        if i < len(docs):
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone: {success} summaries generated, {errors} errors")
    s = mem.stats()
    print(f"Summary coverage: {s['docs_with_summary']}/{s['documents']} "
          f"({100*s['docs_with_summary']/max(s['documents'],1):.0f}%)")
    mem.close()


if __name__ == "__main__":
    main()
