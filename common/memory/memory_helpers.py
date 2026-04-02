"""Helper utilities extracted from factory_brain.

These helpers are kept separate so they can be reused by the modularized
memory components without pulling the entire factory_brain module into scope.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

# -- Helper method bindings -------------------------------------------------

def attach_helper_methods(cls):
    """Bind helper methods onto ResearchMemory.

    Caller should set cls.SOURCE_PRIORITY before binding. This keeps helper
    wiring in one place and avoids circular imports.
    """
    cls._extract_title = _extract_title
    cls.add_source = add_source
    return cls


def _extract_title(self, content: str, filepath: str) -> str:
    """Extract a document title from the first markdown heading or filename."""
    match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return Path(filepath).stem.replace("_", " ").replace("-", " ").title()


def add_source(self, path: str, doc_type: str = "research", pattern: str = "**/*.md",
               provenance: str = "research", priority: int = 50):
    """Register an additional ingest source at runtime.

    Uses self.SOURCE_PRIORITY when available so tests can override the list
    without mutating the module-level default in factory_brain.
    """
    from common.memory import factory_brain as fb
    slots = getattr(fb, "SOURCE_PRIORITY", None)
    if slots is None:
        return
    slots.append((Path(path), doc_type, provenance, priority))
    print(f"Added source: {path} ({doc_type}, {pattern})", file=sys.stderr)


# -- CLI formatting helpers -------------------------------------------------

def format_result(r: dict, index: int, verbose: bool = False, level: int = 0) -> str:
    """Format a single search result for terminal display.

    level=0: legacy format (chunk-based results)
    level=1: signal line only (one-liner)
    level=2: summary (200-500 words)
    level=3: summary + raw chunks
    """
    lines = []
    dist = r.get("effective_distance", r.get("distance", r.get("rrf_score", "")))
    dist_str = f"  score={dist:.4f}" if isinstance(dist, float) else ""

    prov = r.get("provenance", "")
    prov_badge = {"validated": "\033[32m[V]\033[0m", "reference": "\033[34m[R]\033[0m",
                  "research": "", "archive": "\033[2m[A]\033[0m"}.get(prov, "")
    emp = " \033[33m[empirical]\033[0m" if r.get("is_empirical") else ""

    if level == 1:
        signal = r.get("signal", r.get("title", ""))
        lines.append(f"\033[1m[{index}]\033[0m {prov_badge} {signal} "
                     f"\033[2m({r['kernel_type']}){dist_str}\033[0m{emp}")
        return "\n".join(lines)

    lines.append(f"\033[1m[{index}]\033[0m {prov_badge} {r['title']} "
                 f"\033[2m({r['kernel_type']}/{r['doc_type']}){dist_str}\033[0m{emp}")
    lines.append(f"    \033[36m{r['source_file']}\033[0m")

    if level >= 2 and r.get("summary"):
        lines.append(f"    \033[33m--- Summary ---\033[0m")
        for sline in r["summary"].split("\n"):
            lines.append(f"    {sline}")
    elif level == 0:
        if r.get("heading"):
            lines.append(f"    \033[33m@ {r['heading']}\033[0m")
        tags = []
        if r.get("stall_types"):
            tags.append(f"stalls: {r['stall_types']}")
        if r.get("techniques"):
            tags.append(f"techniques: {r['techniques']}")
        if tags:
            lines.append(f"    {' | '.join(tags)}")
        text = r.get("text", "")
        snippet = text[:300].replace("\n", " ")
        if len(text) > 300:
            snippet += "..."
        lines.append(f"    {snippet}")
    else:
        if r.get("techniques"):
            lines.append(f"    techniques: {r['techniques']}")
        lines.append(f"    \033[2m(no summary generated yet — use --full for raw chunks)\033[0m")

    if level >= 3 and r.get("chunks"):
        lines.append(f"    \033[33m--- Full Content ({len(r['chunks'])} chunks) ---\033[0m")
        for chunk in r["chunks"]:
            if chunk.get("heading"):
                lines.append(f"    \033[33m@ {chunk['heading']}\033[0m")
            lines.append(f"    {chunk['text'][:500]}")
            if len(chunk["text"]) > 500:
                lines.append("    ...")
            lines.append("")

    if verbose:
        lines.append(f"    \033[2m[provenance: {prov} | doc_id: {r.get('doc_id', r.get('chunk_id', ''))}]\033[0m")

    return "\n".join(lines)


def format_quality_result(report: str | Iterable[str]) -> str:
    """Normalize quality_report output to a printable string."""
    if isinstance(report, str):
        return report
    return "\n".join(report)
