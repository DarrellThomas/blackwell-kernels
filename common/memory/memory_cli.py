"""CLI front-end for ResearchMemory commands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SUPPORTED_COMMANDS = {"ingest", "search", "fts", "stats", "quality", "maintain", "serve"}


def _runtime_bindings():
    from common.memory.factory_brain import ResearchMemory, PROVENANCE_TIERS, format_result

    return ResearchMemory, PROVENANCE_TIERS, format_result


def add_subcommands(parser: argparse.ArgumentParser):
    sub = parser.add_subparsers(dest="command")
    # ingest
    p = sub.add_parser("ingest", help="Ingest docs/experiments")
    p.add_argument("path", nargs="?", help="Optional path or TSV")
    p.add_argument("--type", default="research")
    p.add_argument("--pattern", default="**/*.md")
    p.add_argument("--force", action="store_true")

    # search
    p = sub.add_parser("search", help="Hybrid search")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--kernel")
    p.add_argument("--type")
    p.add_argument("--stall")
    p.add_argument("--technique")
    p.add_argument("--mode", choices=["hybrid","semantic","fts"], default="hybrid")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--verbose", action="store_true")

    # fts
    p = sub.add_parser("fts", help="Full-text search")
    p.add_argument("query", nargs="+")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--kernel")
    p.add_argument("--type")
    p.add_argument("--verbose", action="store_true")

    # stats
    sub.add_parser("stats", help="Database stats")
    sub.add_parser("quality", help="Quality audit")

    # maintain
    sub.add_parser("maintain", help="Ingest + gap report")

    # serve
    p = sub.add_parser("serve", help="HTTP API")
    p.add_argument("--port", type=int, default=8421)

    return parser


def run(args):
    cmd = args.command
    if cmd not in SUPPORTED_COMMANDS:
        print(f"Unknown or unsupported command for memory_cli: {cmd}", file=sys.stderr)
        sys.exit(2)

    ResearchMemory, PROVENANCE_TIERS, format_result = _runtime_bindings()

    if cmd == "ingest":
        mem = ResearchMemory(args.db)
        if args.path:
            p = Path(args.path)
            if p.is_file():
                if p.suffix == ".tsv":
                    stats = mem.ingest_tsv(str(p), force=args.force)
                    print(f"TSV rows: {stats['chunks']}")
                else:
                    n = mem.ingest_file(str(p), doc_type=args.type, force=args.force)
                    print(f"Ingested {n} chunks from {p}")
            elif p.is_dir():
                stats = mem.ingest_directory(str(p), args.type, args.pattern, args.force)
                print(f"Docs: {stats['files']} files, {stats['chunks']} chunks ({stats['skipped']} skipped)")
            else:
                print(f"Not found: {p}", file=sys.stderr); sys.exit(1)
        else:
            stats = mem.ingest_all(force=args.force)
            print(f"Docs: {stats['files']} files, {stats['chunks']} chunks ({stats['skipped']} skipped, {stats['dedup_skipped']} dedup)")
            print("Indexing experiments...")
            tsv_stats = mem.ingest_all_tsv(force=args.force)
            print(f"Experiments: {tsv_stats['files']} files, {tsv_stats['chunks']} rows")
        mem.close(); return

    if cmd in {"search", "fts"}:
        mem = ResearchMemory(args.db)
        query = " ".join(args.query)
        level = getattr(args, 'level', 0)
        if cmd == "fts":
            results = mem.search_fts(query, k=args.k, kernel_type=args.kernel, doc_type=args.type)
            level = 0
        elif level > 0:
            results = mem.search_summaries(query, k=args.k, level=level,
                                           kernel_type=args.kernel, doc_type=args.type,
                                           stall_type=args.stall, technique=args.technique)
        elif args.mode == "fts":
            results = mem.search_fts(query, k=args.k, kernel_type=args.kernel, doc_type=args.type,
                                     stall_type=args.stall, technique=args.technique)
        elif args.mode == "semantic":
            results = mem.search_semantic(query, k=args.k, kernel_type=args.kernel, doc_type=args.type,
                                          stall_type=args.stall, technique=args.technique)
        else:
            has_summaries = mem.conn.execute("SELECT COUNT(*) FROM vec_summaries").fetchone()[0]
            if has_summaries > 0:
                results = mem.search_summaries(query, k=args.k, level=2,
                                               kernel_type=args.kernel, doc_type=args.type,
                                               stall_type=args.stall, technique=args.technique)
            else:
                results = mem.search_hybrid(query, k=args.k, kernel_type=args.kernel, doc_type=args.type,
                                            stall_type=args.stall, technique=args.technique)
        if not results:
            print("No results found.")
        else:
            for i, r in enumerate(results, 1):
                print(format_result(r, i, verbose=args.verbose, level=level))
                print()
        mem.close(); return

    if cmd == "stats":
        mem = ResearchMemory(args.db)
        s = mem.stats()
        print(f"Database: {args.db}")
        print(f"Size: {s['db_size_mb']} MB")
        print(f"Documents: {s['documents']}")
        print(f"Chunks: {s['chunks']}")
        print(f"Experiments: {s['experiments_total']}")
        print("\nBy provenance:")
        for t, n in s["by_provenance"].items():
            boost = PROVENANCE_TIERS.get(t, {}).get("boost", 1.0)
            print(f"  {t}: {n} (boost: {boost}x)")
        print("\nBy document type:")
        for t, n in sorted(s["by_doc_type"].items()):
            print(f"  {t}: {n}")
        print("\nBy kernel type:")
        for t, n in s["by_kernel_type"].items():
            print(f"  {t}: {n}")
        print(f"\nEmpirically-backed: {s['empirical_docs']}")
        print(f"Summaries (Level 2): {s['docs_with_summary']}/{s['documents']} ({100*s['docs_with_summary']/max(s['documents'],1):.0f}%)")
        print(f"Chunk avg length: {s['chunk_avg_len']} chars")
        print(f"Chunks with stall tags: {s['chunks_with_stalls']}")
        print(f"Chunks with technique tags: {s['chunks_with_techniques']}")
        mem.close(); return

    if cmd == "quality":
        mem = ResearchMemory(args.db)
        print(mem.quality_report())
        mem.close(); return

    if cmd == "maintain":
        from common.memory import memory_maintain
        memory_maintain.cmd_maintain(args)
        return

    if cmd == "serve":
        from common.memory import memory_server
        mem = ResearchMemory(args.db)
        memory_server.serve(mem, args.port)
        return

    print(f"Unsupported command: {cmd}", file=sys.stderr)
    sys.exit(2)



def build_parser(default_db: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research Memory Database for blackwell-kernels")
    parser.add_argument("--db", default=default_db, help="Database path")
    add_subcommands(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(default_db="research.db")
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
