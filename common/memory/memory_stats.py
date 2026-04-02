"""Stats and quality helpers extracted from factory_brain.
Expect `self` with .conn, .db_path, PROVENANCE_TIERS.
"""
import os
import re


def attach_stats_methods(cls):
    from common.memory.factory_brain import PROVENANCE_TIERS
    cls.PROVENANCE_TIERS = PROVENANCE_TIERS
    cls.stats = stats
    cls.quality_report = quality_report
    return cls

# -- Stats & Quality --

def stats(self) -> dict:
    """Return database statistics."""
    c = self.conn
    docs = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunks = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    by_type = {r[0]: r[1] for r in c.execute(
        "SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type"
    ).fetchall()}

    by_kernel = {r[0]: r[1] for r in c.execute(
        "SELECT kernel_type, COUNT(*) FROM documents GROUP BY kernel_type ORDER BY COUNT(*) DESC"
    ).fetchall()}

    by_provenance = {r[0]: r[1] for r in c.execute(
        "SELECT provenance, COUNT(*) FROM documents GROUP BY provenance ORDER BY COUNT(*) DESC"
    ).fetchall()}

    empirical = c.execute("SELECT COUNT(*) FROM documents WHERE is_empirical = 1").fetchone()[0]
    experiments = c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    # Chunk quality
    chunk_stats = c.execute("""
        SELECT COUNT(*) as total,
               AVG(LENGTH(text)) as avg_len,
               MIN(LENGTH(text)) as min_len,
               MAX(LENGTH(text)) as max_len,
               SUM(CASE WHEN LENGTH(text) < 200 THEN 1 ELSE 0 END) as tiny,
               SUM(CASE WHEN stall_types != '' THEN 1 ELSE 0 END) as with_stalls,
               SUM(CASE WHEN techniques != '' THEN 1 ELSE 0 END) as with_techniques
        FROM chunks
    """).fetchone()

    db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

    # Summary coverage
    with_summary = c.execute("SELECT COUNT(*) FROM documents WHERE has_summary = 1").fetchone()[0]
    summary_vecs = c.execute("SELECT COUNT(*) FROM vec_summaries").fetchone()[0]

    return {
        "documents": docs,
        "chunks": chunks,
        "db_size_mb": round(db_size / 1048576, 2),
        "by_doc_type": by_type,
        "by_kernel_type": by_kernel,
        "by_provenance": by_provenance,
        "empirical_docs": empirical,
        "experiments_total": experiments,
        "docs_with_summary": with_summary,
        "summary_vectors": summary_vecs,
        "chunk_avg_len": round(chunk_stats["avg_len"] or 0),
        "chunk_min_len": chunk_stats["min_len"] or 0,
        "chunk_tiny_count": chunk_stats["tiny"] or 0,
        "chunks_with_stalls": chunk_stats["with_stalls"] or 0,
        "chunks_with_techniques": chunk_stats["with_techniques"] or 0,
        "jobs_total": c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
        "jobs_active": c.execute(
            "SELECT COUNT(*) FROM jobs WHERE state NOT IN ('shipped','converged','parked','abandoned')"
        ).fetchone()[0],
        "jobs_by_state": {r[0]: r[1] for r in c.execute(
            "SELECT state, COUNT(*) FROM jobs GROUP BY state ORDER BY COUNT(*) DESC"
        ).fetchall()},
        "messages_total": c.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        "messages_open": c.execute("SELECT COUNT(*) FROM messages WHERE status = 'open'").fetchone()[0],
    }

def quality_report(self) -> str:
    """Generate a quality audit report."""
    s = self.stats()
    lines = []
    lines.append("=== RESEARCH MEMORY QUALITY REPORT ===\n")

    lines.append(f"Documents: {s['documents']}  |  Chunks: {s['chunks']}  |  DB: {s['db_size_mb']} MB\n")

    # Provenance breakdown
    lines.append("PROVENANCE TIERS:")
    for tier, count in s["by_provenance"].items():
        boost = self.PROVENANCE_TIERS.get(tier, {}).get("boost", 1.0)
        desc = self.PROVENANCE_TIERS.get(tier, {}).get("description", "")
        lines.append(f"  {tier:12s}  {count:4d} docs  (boost: {boost}x) — {desc}")

    lines.append(f"\nEMPIRICAL DOCS: {s['empirical_docs']}/{s['documents']} "
                 f"({100*s['empirical_docs']/max(s['documents'],1):.1f}%)")

    # Summary coverage (Level 2)
    lines.append(f"\nSUMMARY COVERAGE (Level 2):")
    lines.append(f"  Documents with summaries: {s['docs_with_summary']}/{s['documents']} "
                 f"({100*s['docs_with_summary']/max(s['documents'],1):.1f}%)")
    lines.append(f"  Summary vectors indexed: {s['summary_vectors']}")

    # Chunk quality
    lines.append(f"\nCHUNK QUALITY:")
    lines.append(f"  Avg length: {s['chunk_avg_len']} chars")
    lines.append(f"  Min length: {s['chunk_min_len']} chars")
    lines.append(f"  Tiny (<200 chars): {s['chunk_tiny_count']} "
                 f"({100*s['chunk_tiny_count']/max(s['chunks'],1):.1f}%)")

    # Metadata coverage
    lines.append(f"\nMETADATA COVERAGE:")
    lines.append(f"  Chunks with stall tags: {s['chunks_with_stalls']}/{s['chunks']} "
                 f"({100*s['chunks_with_stalls']/max(s['chunks'],1):.1f}%)")
    lines.append(f"  Chunks with technique tags: {s['chunks_with_techniques']}/{s['chunks']} "
                 f"({100*s['chunks_with_techniques']/max(s['chunks'],1):.1f}%)")

    # Duplicate check
    dupes = self.conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT content_hash FROM documents GROUP BY content_hash HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    lines.append(f"\nDUPLICATES: {dupes} content hashes with multiple entries "
                 f"(should be 0 after dedup)")

    return "\n".join(lines)

