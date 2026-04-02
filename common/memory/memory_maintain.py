"""Maintenance command for ResearchMemory."""

def cmd_maintain(args):
    """Run maintenance: incremental ingest, gap report, promotion candidates."""
    from common.memory.factory_brain import BWK_ROOT, ResearchMemory

    mem = ResearchMemory(args.db)

    print("=" * 60)
    print("  MEMORY DB MAINTENANCE REPORT")
    print("=" * 60)

    # Phase 1: Incremental ingest (picks up new/changed files)
    print("\n--- Phase 1: Incremental ingest ---")
    md_stats = mem.ingest_all()
    tsv_stats = mem.ingest_all_tsv()
    mem.refresh_worker_state()
    new_md = md_stats.get("files", 0)
    new_tsv = tsv_stats.get("chunks", 0)
    removed = md_stats.get("removed", 0)
    print(f"  New docs indexed: {new_md}")
    print(f"  New experiment rows: {new_tsv}")
    print(f"  Stale docs removed: {removed}")
    print(f"  Docs skipped (unchanged): {md_stats.get('skipped', 0)}")
    print(f"  Duplicates eliminated: {md_stats.get('dedup_skipped', 0)}")

    # Phase 2: Coverage gaps
    print("\n--- Phase 2: Coverage analysis ---")
    stats = mem.stats()
    total_docs = stats["documents"]
    total_chunks = stats["chunks"]
    by_kernel = stats.get("by_kernel_type", {})
    by_prov = stats.get("by_provenance", {})

    print(f"  Total: {total_docs} docs, {total_chunks} chunks, {stats['db_size_mb']:.1f} MB")
    print(f"  Empirical docs: {stats.get('empirical_docs', 0)}/{total_docs} "
          f"({100*stats.get('empirical_docs',0)/max(total_docs,1):.0f}%)")
    print(f"  Provenance: validated={by_prov.get('validated',0)}, "
          f"reference={by_prov.get('reference',0)}, "
          f"research={by_prov.get('research',0)}, "
          f"archive={by_prov.get('archive',0)}")

    # Active kernels with low coverage
    active_kernels = [
        ("attention", BWK_ROOT / "main"),
        ("gemm", BWK_ROOT / "gemm"),
        ("fused_mlp", BWK_ROOT / "fused-mlp"),
        ("dotproduct", BWK_ROOT / "dotproduct"),
        ("linalg", BWK_ROOT / "linalg"),
        ("qr", BWK_ROOT / "qr"),
        ("rmsnorm", BWK_ROOT / "rmsnorm"),
        ("spmv", BWK_ROOT / "spmv"),
        ("numerical", BWK_ROOT / "numerical"),
        ("cuquantum", BWK_ROOT / "cuquantum"),
        ("chess_training", BWK_ROOT / "chess-training"),
    ]

    print("\n  Kernel coverage:")
    gaps = []
    for kernel, proj_dir in active_kernels:
        doc_count = by_kernel.get(kernel, 0)
        # Count files on disk
        docs_dir = proj_dir / "docs"
        claude_dir = proj_dir / ".claude"
        on_disk = 0
        if docs_dir.is_dir():
            on_disk += len(list(docs_dir.glob("**/*.md")))
        if claude_dir.is_dir():
            on_disk += len(list(claude_dir.glob("**/*.md")))
        coverage = f"{doc_count}/{on_disk}" if on_disk > 0 else f"{doc_count}/?"
        flag = " *** LOW" if doc_count < 5 else ""
        print(f"    {kernel:<16} {coverage:>10} indexed/on-disk{flag}")
        if doc_count < 5:
            gaps.append(kernel)

    # Phase 3: Promotion candidates (research → validated)
    print("\n--- Phase 3: Promotion candidates ---")
    # Find research docs that are tagged empirical — these are candidates
    # for promotion to validated provenance
    candidates = mem.conn.execute("""
        SELECT d.id, d.title, d.source_file, d.kernel_type, d.provenance
        FROM documents d
        WHERE d.provenance = 'research'
          AND d.is_empirical = 1
          AND d.doc_type NOT IN ('experiment', 'dead_end')
        ORDER BY d.kernel_type, d.title
    """).fetchall()

    if candidates:
        print(f"  {len(candidates)} empirical research docs could be promoted to 'validated':")
        shown = 0
        for c in candidates:
            if shown >= 20:
                print(f"  ... and {len(candidates) - 20} more")
                break
            print(f"    [{c['kernel_type']:<12}] {c['title'][:60]}")
            shown += 1
    else:
        print("  No promotion candidates found.")

    # Phase 4: Duplicates check
    print("\n--- Phase 4: Duplicate check ---")
    dupes = mem.conn.execute("""
        SELECT content_hash, COUNT(*) as cnt
        FROM documents
        GROUP BY content_hash
        HAVING cnt > 1
    """).fetchall()
    if dupes:
        print(f"  WARNING: {len(dupes)} content hashes with multiple entries")
        for d in dupes[:5]:
            files = mem.conn.execute(
                "SELECT source_file FROM documents WHERE content_hash = ?",
                (d["content_hash"],)
            ).fetchall()
            print(f"    Hash {d['content_hash'][:12]}...: {', '.join(f['source_file'].split('/')[-1] for f in files)}")
    else:
        print("  Clean — no duplicates.")

    # Phase 5: Summary
    print("\n--- Summary ---")
    if new_md or new_tsv or removed:
        print(f"  Changes this run: +{new_md} docs, +{new_tsv} experiments, -{removed} stale")
    else:
        print("  No changes — database is current.")
    if gaps:
        print(f"  Low coverage kernels: {', '.join(gaps)}")
    if candidates:
        print(f"  {len(candidates)} docs ready for provenance promotion")

    print()
    mem.close()

