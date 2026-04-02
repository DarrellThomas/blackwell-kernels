"""Experiment helpers extracted from factory_brain.
Expect `self` with .conn, BWK_ROOT, STATE_TO_PHASE, ensure_open_message.
"""
import time, json, hashlib
from common.memory import memory_ingest as _mem_ingest

extract_techniques = _mem_ingest.extract_techniques
STALL_NAMES = _mem_ingest.STALL_NAMES

def attach_experiment_methods(cls):
    from common.memory.factory_brain import BWK_ROOT, STATE_TO_PHASE
    cls.BWK_ROOT = BWK_ROOT
    cls.STATE_TO_PHASE = STATE_TO_PHASE
    cls.record_experiment = record_experiment
    cls.get_experiments = get_experiments
    cls.summarize_experiments = summarize_experiments
    return cls

# -- Experiments --

def record_experiment(self, kernel_type: str, status: str, description: str,
                      timestamp: str = "", git_commit: str = "", duration_us=None,
                      vs_ref=None, sm_pct=None, stall_math=None, stall_wait=None,
                      stall_scoreboard=None, stall_barrier=None, top_stall: str = "",
                      job_id: int = None, source_type: str = "db",
                      source_path: str = "", experiment_index: int = 0,
                      reference_label: str = "", extra: dict | None = None) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "kernel_type": kernel_type,
        "job_id": job_id,
        "source_type": source_type,
        "source_path": source_path,
        "experiment_index": experiment_index,
        "timestamp": (timestamp or "").strip() or now,
        "git_commit": git_commit or "",
        "duration_us": duration_us,
        "vs_ref": vs_ref,
        "sm_pct": sm_pct,
        "stall_math": stall_math,
        "stall_wait": stall_wait,
        "stall_scoreboard": stall_scoreboard,
        "stall_barrier": stall_barrier,
        "top_stall": top_stall or "",
        "status": (status or "").strip().lower(),
        "description": description or "",
        "reference_label": reference_label or "",
        "extra_json": json.dumps(extra or {}, sort_keys=True),
    }
    row_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    self.conn.execute("""
        INSERT OR IGNORE INTO experiments (
            kernel_type, job_id, source_type, source_path, row_hash,
            experiment_index, timestamp, git_commit, duration_us, vs_ref, sm_pct,
            stall_math, stall_wait, stall_scoreboard, stall_barrier,
            top_stall, status, description, reference_label, extra_json,
            recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        payload["kernel_type"], payload["job_id"], payload["source_type"],
        payload["source_path"], row_hash, payload["experiment_index"],
        payload["timestamp"], payload["git_commit"], payload["duration_us"],
        payload["vs_ref"], payload["sm_pct"], payload["stall_math"],
        payload["stall_wait"], payload["stall_scoreboard"],
        payload["stall_barrier"], payload["top_stall"], payload["status"],
        payload["description"], payload["reference_label"],
        payload["extra_json"], now
    ))
    self.conn.commit()
    row = self.conn.execute(
        "SELECT * FROM experiments WHERE row_hash = ?",
        (row_hash,)
    ).fetchone()
    return dict(row) if row else {}

def get_experiments(self, kernel_type: str = None, job_id: int = None,
                    limit: int = 100, status: str = None):
    where, params = [], []
    if kernel_type:
        where.append("kernel_type = ?"); params.append(kernel_type)
    if job_id is not None:
        where.append("job_id = ?"); params.append(job_id)
    if status:
        where.append("status = ?"); params.append(status)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    rows = self.conn.execute(f"""
        SELECT * FROM experiments {where_sql}
        ORDER BY COALESCE(timestamp, recorded_at) DESC, id DESC
        LIMIT ?
    """, params + [limit]).fetchall()
    return [dict(r) for r in rows]

def summarize_experiments(self, kernel_type: str = None, job_id: int = None,
                          recent: int = 12) -> dict:
    rows = self.get_experiments(kernel_type=kernel_type, job_id=job_id, limit=10000)
    summary = {
        "kernel_type": kernel_type or "",
        "job_id": job_id,
        "total": len(rows),
        "kept": 0,
        "discarded": 0,
        "unknown": 0,
        "current_discard_streak": 0,
        "max_discard_streak": 0,
        "best_keep": None,
        "last_keep": None,
        "last_experiment": rows[0] if rows else None,
        "recent": rows[:recent],
        "top_stalls": [],
        "recent_discards": [],
        "recent_keeps": [],
    }
    if not rows:
        return summary

    ordered = list(reversed(rows))
    stall_counts = {}
    current_discard_streak = 0
    max_discard_streak = 0
    running_discard_streak = 0
    best_keep = None
    last_keep = None

    for row in ordered:
        row_status = (row.get("status") or "").strip().lower()
        if row_status in ("keep", "kept"):
            summary["kept"] += 1
            last_keep = row
            running_discard_streak = 0
            vr = row.get("vs_ref")
            dur = row.get("duration_us")
            if best_keep is None:
                best_keep = row
            else:
                best_vr = best_keep.get("vs_ref")
                best_dur = best_keep.get("duration_us")
                if vr is not None and (best_vr is None or vr > best_vr):
                    best_keep = row
                elif vr is not None and best_vr is not None and abs(vr - best_vr) < 1e-12:
                    if dur is not None and (best_dur is None or dur < best_dur):
                        best_keep = row
        elif row_status in ("discard", "discarded"):
            summary["discarded"] += 1
            running_discard_streak += 1
            max_discard_streak = max(max_discard_streak, running_discard_streak)
        else:
            summary["unknown"] += 1
            running_discard_streak = 0

        top_stall = (row.get("top_stall") or "").strip()
        if top_stall and top_stall not in ("-", "none"):
            stall_counts[top_stall] = stall_counts.get(top_stall, 0) + 1

    for row in rows:
        row_status = (row.get("status") or "").strip().lower()
        if row_status in ("discard", "discarded"):
            current_discard_streak += 1
        else:
            break

    summary["current_discard_streak"] = current_discard_streak
    summary["max_discard_streak"] = max_discard_streak
    summary["best_keep"] = best_keep
    summary["last_keep"] = last_keep
    summary["top_stalls"] = [
        {"stall": stall, "count": count}
        for stall, count in sorted(stall_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    ]
    summary["recent_discards"] = [
        r for r in rows
        if (r.get("status") or "").strip().lower() in ("discard", "discarded")
    ][:5]
    summary["recent_keeps"] = [
        r for r in rows
        if (r.get("status") or "").strip().lower() in ("keep", "kept")
    ][:5]
    return summary

# -- Ingest --

def ingest_all(self, force: bool = False) -> dict:
    """Run full ingest with deduplication across all sources.

    Scans all sources in priority order. For each unique content_hash,
    only the highest-priority (lowest number) copy is indexed. Lower-priority
    duplicates are recorded in the canonical document's 'also_at' field.
    """
    # Phase 1: Scan all files, compute hashes, pick canonical copies
    print("Phase 1: Scanning files and resolving duplicates...", file=sys.stderr)
    all_files = []  # (filepath, doc_type, provenance_hint, priority)

    for src_path, doc_type, prov_hint, priority in SOURCE_PRIORITY:
        if not src_path.is_dir():
            continue
        for f in sorted(src_path.glob("**/*.md")):
            all_files.append((str(f.resolve()), doc_type, prov_hint, priority))

    for src_path, glob_pat, priority in CODE_SOURCES:
        if not src_path.is_dir():
            continue
        for f in sorted(src_path.glob(glob_pat)):
            all_files.append((str(f.resolve()), "source_code", "research", priority))

    # Group by content hash, pick canonical (lowest priority number)
    hash_map = {}  # content_hash -> [(filepath, doc_type, prov_hint, priority)]
    file_hashes = {}  # filepath -> content_hash

    for filepath, doc_type, prov_hint, priority in all_files:
        try:
            content = Path(filepath).read_text(errors="replace")
        except Exception:
            continue
        if len(content.strip()) < 50:
            continue
        ch = file_content_hash(content)
        file_hashes[filepath] = ch
        if ch not in hash_map:
            hash_map[ch] = []
        hash_map[ch].append((filepath, doc_type, prov_hint, priority))

    # Select canonical copy for each hash
    canonical = {}  # content_hash -> (filepath, doc_type, prov_hint, [also_at])
    total_files = 0
    dedup_skipped = 0

    for ch, entries in hash_map.items():
        entries.sort(key=lambda x: x[3])  # sort by priority
        best = entries[0]
        also_at = [e[0] for e in entries[1:]]
        canonical[ch] = (best[0], best[1], best[2], also_at)
        total_files += 1
        dedup_skipped += len(also_at)

    print(f"  {total_files} unique documents ({dedup_skipped} duplicates eliminated)",
          file=sys.stderr)

    # Phase 2: Index canonical copies
    print("Phase 2: Indexing canonical copies...", file=sys.stderr)
    stats = {"files": 0, "chunks": 0, "skipped": 0, "dedup_skipped": dedup_skipped}

    for ch, (filepath, doc_type, prov_hint, also_at) in canonical.items():
        if not force:
            row = self.conn.execute(
                "SELECT id, content_hash FROM documents WHERE source_file = ?",
                (filepath,)
            ).fetchone()
            if row and row["content_hash"] == ch:
                stats["skipped"] += 1
                continue

        n = self._index_document(filepath, doc_type, prov_hint, also_at, ch)
        if n > 0:
            stats["files"] += 1
            stats["chunks"] += n
        else:
            stats["skipped"] += 1

    # Phase 3: Clean up documents that are no longer canonical
    # (were previously indexed but are now duplicates of a higher-priority copy)
    canonical_paths = set(v[0] for v in canonical.values())
    stale = self.conn.execute("SELECT id, source_file FROM documents").fetchall()
    removed = 0
    for row in stale:
        if row["source_file"] not in canonical_paths:
            self._remove_document(row["id"])
            removed += 1
    if removed:
        print(f"  Removed {removed} stale/demoted documents", file=sys.stderr)
        self.conn.commit()

    stats["removed"] = removed
    return stats

def ingest_file(self, filepath: str, doc_type: str = "research",
                force: bool = False) -> int:
    """Ingest a single file (bypasses dedup — for ad-hoc additions)."""
    filepath = str(Path(filepath).resolve())
    if not os.path.isfile(filepath):
        return 0
    content = Path(filepath).read_text(errors="replace")
    if len(content.strip()) < 50:
        return 0
    ch = file_content_hash(content)

    if not force:
        row = self.conn.execute(
            "SELECT id, content_hash FROM documents WHERE source_file = ?",
            (filepath,)
        ).fetchone()
        if row and row["content_hash"] == ch:
            return 0

    prov = infer_provenance(filepath, infer_doc_type(filepath, doc_type))
    return self._index_document(filepath, doc_type, prov, [], ch)

def ingest_directory(self, dirpath: str, doc_type: str = "research",
                     pattern: str = "**/*.md", force: bool = False) -> dict:
    """Ingest a specific directory (bypasses dedup — for ad-hoc additions)."""
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        print(f"  Skipping (not found): {dirpath}", file=sys.stderr)
        return {"files": 0, "chunks": 0, "skipped": 0}

    files = sorted(dirpath.glob(pattern))
    stats = {"files": 0, "chunks": 0, "skipped": 0}

    for f in files:
        n = self.ingest_file(str(f), doc_type=doc_type, force=force)
        if n > 0:
            stats["files"] += 1
            stats["chunks"] += n
        else:
            stats["skipped"] += 1

    return stats

def _index_document(self, filepath: str, doc_type: str, prov_hint: str,
                    also_at: list[str], content_hash: str) -> int:
    """Index a single document: chunk, embed, store."""
    content = Path(filepath).read_text(errors="replace")
    if len(content.strip()) < 50:
        return 0

    existing = self.conn.execute(
        "SELECT id, content_hash, has_summary FROM documents WHERE source_file = ?",
        (filepath,)
    ).fetchone()
    content_changed = bool(existing and existing["content_hash"] != content_hash)

    kernel_type = infer_kernel_type(filepath, content)
    real_doc_type = infer_doc_type(filepath, doc_type)
    provenance = infer_provenance(filepath, real_doc_type)
    # Use path-based hint if provenance detection gives generic result
    if provenance == "research" and prov_hint in ("validated", "reference"):
        provenance = prov_hint
    title = self._extract_title(content, filepath)
    is_empirical = detect_empirical(content)
    doc_techniques = extract_techniques(content)
    file_mod = time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(filepath)))
    also_at_str = "\n".join(also_at[:20]) if also_at else ""

    # Chunk based on file type
    if real_doc_type == "source_code":
        raw_chunks = chunk_code(content, filepath)
    else:
        raw_chunks = chunk_markdown(content, filepath, doc_title=title)

    if not raw_chunks:
        return 0

    # Embed all chunks in batch
    texts = [c["text"] for c in raw_chunks]
    embeddings = embed_texts(texts, task="search_document")

    # Upsert document
    self.conn.execute("""
        INSERT INTO documents (source_file, content_hash, doc_type, kernel_type,
                               provenance, title, date_indexed, file_modified,
                               chunk_count, is_empirical, techniques, also_at)
        VALUES (?, ?, ?, ?, ?, ?, date('now'), ?, ?, ?, ?, ?)
        ON CONFLICT(source_file) DO UPDATE SET
            content_hash=excluded.content_hash, doc_type=excluded.doc_type,
            kernel_type=excluded.kernel_type, provenance=excluded.provenance,
            title=excluded.title, date_indexed=excluded.date_indexed,
            file_modified=excluded.file_modified, chunk_count=excluded.chunk_count,
            is_empirical=excluded.is_empirical, techniques=excluded.techniques,
            also_at=excluded.also_at
    """, (filepath, content_hash, real_doc_type, kernel_type, provenance,
          title, file_mod, len(raw_chunks), int(is_empirical),
          doc_techniques, also_at_str))

    doc_id = self.conn.execute(
        "SELECT id FROM documents WHERE source_file = ?", (filepath,)
    ).fetchone()["id"]

    # If content changed, invalidate Level 2 artifacts so stale summaries do
    # not survive a re-ingest. They will be regenerated by generate_summaries.py.
    if content_changed:
        self.conn.execute(
            "UPDATE documents SET summary = '', signal = '', has_summary = 0 WHERE id = ?",
            (doc_id,)
        )
        self.conn.execute("DELETE FROM vec_summaries WHERE doc_id = ?", (doc_id,))

    # Delete old chunks for this doc
    old_chunk_ids = [r["id"] for r in self.conn.execute(
        "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()]

    if old_chunk_ids:
        for cid in old_chunk_ids:
            self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    # Insert new chunks + vectors
    for chunk, emb in zip(raw_chunks, embeddings):
        stalls = extract_stall_types(chunk["text"])
        techs = extract_techniques(chunk["text"])
        self.conn.execute(
            "INSERT INTO chunks (doc_id, position, heading, text, stall_types, techniques) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, chunk["position"], chunk["heading"], chunk["text"], stalls, techs)
        )
        chunk_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(emb))
        )

    self.conn.commit()
    return len(raw_chunks)

# -- Experiment TSV Ingest --

def ingest_tsv(self, tsv_path: str, kernel_type: str = None,
               force: bool = False) -> dict:
    """Ingest experiment results from a TSV file.

    Each row becomes one chunk. The description field is the semantic content.
    Numeric columns become metadata. Incremental: only adds rows not yet indexed.

    Returns stats dict with files, chunks, skipped counts.
    """
    import csv

    tsv_path = str(Path(tsv_path).resolve())
    if not os.path.isfile(tsv_path):
        return {"files": 0, "chunks": 0, "skipped": 0}

    # Infer kernel type from path if not given
    if not kernel_type:
        # /data/src/bwk/<project>/results/<name>.tsv
        parts = tsv_path.split("/")
        for i, p in enumerate(parts):
            if p == "results" and i > 0:
                kernel_type = parts[i - 1]
                break
        # Map project dirs to kernel types
        dir_to_kernel = {
            "main": "attention", "fused-mlp": "fused_mlp",
            "chess-training": "chess_training",
        }
        kernel_type = dir_to_kernel.get(kernel_type, kernel_type) or "general"

    # Use TSV filename as doc identity
    content_hash = file_content_hash(Path(tsv_path).read_text(errors="replace"))

    # Check if document chunks are already indexed with same hash
    existing = self.conn.execute(
        "SELECT id, content_hash, chunk_count FROM documents WHERE source_file = ?",
        (tsv_path,)
    ).fetchone()
    doc_up_to_date = bool(existing and existing["content_hash"] == content_hash and not force)

    # Parse TSV
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return {"files": 0, "chunks": 0, "skipped": 1}

    # Find the description and status columns
    desc_col = None
    status_col = None
    for h in headers:
        if h.lower() == "description":
            desc_col = h
        if h.lower() == "status":
            status_col = h

    if not desc_col:
        print(f"  WARNING: No 'description' column in {tsv_path}", file=sys.stderr)
        return {"files": 0, "chunks": 0, "skipped": 1}

    # Build chunks — one per row
    tsv_name = Path(tsv_path).stem
    title = f"Experiments: {tsv_name} ({kernel_type})"
    raw_chunks = []
    kept_count = 0
    discarded_count = 0
    try:
        self.conn.execute("DELETE FROM experiments WHERE source_type = 'tsv' AND source_path = ?", (tsv_path,))
    except sqlite3.OperationalError:
        pass

    for i, row in enumerate(rows):
        desc = row.get(desc_col, "").strip()
        if not desc or len(desc) < 10:
            continue

        status = row.get(status_col, "").strip().lower() if status_col else ""
        vs_ref = row.get("vs_ref", "-").strip()
        duration = row.get("duration_us", "-").strip()
        top_stall = row.get("top_stall", "").strip()
        sm_pct = row.get("sm_pct", "-").strip()
        commit = row.get("commit", "").strip()
        timestamp = row.get("timestamp", "").strip()

        # Build rich text for the chunk — combines description with key metrics
        parts = [desc]
        metrics = []
        if vs_ref and vs_ref != "-":
            metrics.append(f"vs_ref={vs_ref}")
        if duration and duration != "-":
            metrics.append(f"duration={duration}us")
        if sm_pct and sm_pct != "-":
            metrics.append(f"SM={sm_pct}%")
        if top_stall and top_stall != "-" and top_stall != "none":
            metrics.append(f"top_stall={top_stall}")
        if status:
            metrics.append(f"status={status}")
        if metrics:
            parts.append(f"[{', '.join(metrics)}]")

        chunk_text = " ".join(parts)

        # Extract stall types from both top_stall field and description
        stalls = set()
        if top_stall and top_stall != "-" and top_stall != "none":
            stalls.add(top_stall)
        # Also check for stall mentions in description
        for s in STALL_NAMES:
            if s in desc.lower():
                stalls.add(s)
        stall_str = ",".join(sorted(stalls))

        techs = extract_techniques(desc)

        heading = f"{'KEPT' if status == 'keep' else 'DISCARDED'}: {tsv_name} exp {i+1}"
        if status == "keep":
            kept_count += 1
        else:
            discarded_count += 1

        def _to_float(value):
            text = str(value or "").strip()
            if not text or text == "-":
                return None
            try:
                return float(text)
            except ValueError:
                return None

        extras = {
            k: v for k, v in row.items()
            if k not in {
                "timestamp", "commit", "duration_us", "vs_ref", "sm_pct",
                "stall_math", "stall_wait", "stall_scoreboard",
                "stall_barrier", "top_stall", "status", "description",
            }
        }
        self.record_experiment(
            kernel_type=kernel_type,
            status=status or "unknown",
            description=desc,
            timestamp=timestamp,
            git_commit=commit,
            duration_us=_to_float(duration),
            vs_ref=_to_float(vs_ref),
            sm_pct=_to_float(sm_pct),
            stall_math=_to_float(row.get("stall_math")),
            stall_wait=_to_float(row.get("stall_wait")),
            stall_scoreboard=_to_float(row.get("stall_scoreboard")),
            stall_barrier=_to_float(row.get("stall_barrier")),
            top_stall=top_stall,
            source_type="tsv",
            source_path=tsv_path,
            experiment_index=i + 1,
            extra=extras,
        )

        raw_chunks.append({
            "text": chunk_text,
            "heading": heading,
            "position": i,
            "stalls": stall_str,
            "techniques": techs,
            "status": status,
        })

    if not raw_chunks:
        return {"files": 0, "chunks": 0, "skipped": 1}

    if doc_up_to_date:
        self.conn.commit()
        return {"files": 0, "chunks": 0, "skipped": 1,
                "kept": kept_count, "discarded": discarded_count}

    # Embed all chunks
    texts = [c["text"] for c in raw_chunks]
    embeddings = embed_texts(texts, task="search_document")

    # Provenance: experiment files with mostly kept results are "validated"
    prov = "validated" if kept_count > discarded_count else "research"
    is_empirical = 1  # all experiments are empirical by definition

    file_mod = time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(tsv_path)))
    doc_techniques = set()
    for c in raw_chunks:
        if c["techniques"]:
            doc_techniques.update(c["techniques"].split(","))

    # Upsert document
    self.conn.execute("""
        INSERT INTO documents (source_file, content_hash, doc_type, kernel_type,
                               provenance, title, date_indexed, file_modified,
                               chunk_count, is_empirical, techniques, also_at,
                               summary, signal, has_summary)
        VALUES (?, ?, 'experiment', ?, ?, ?, date('now'), ?, ?, 1, ?, '', '', '', 0)
        ON CONFLICT(source_file) DO UPDATE SET
            content_hash=excluded.content_hash, doc_type='experiment',
            kernel_type=excluded.kernel_type, provenance=excluded.provenance,
            title=excluded.title, date_indexed=excluded.date_indexed,
            file_modified=excluded.file_modified, chunk_count=excluded.chunk_count,
            is_empirical=1, techniques=excluded.techniques
    """, (tsv_path, content_hash, kernel_type, prov, title,
          file_mod, len(raw_chunks), ",".join(sorted(doc_techniques))))

    doc_id = self.conn.execute(
        "SELECT id FROM documents WHERE source_file = ?", (tsv_path,)
    ).fetchone()["id"]

    # Delete old chunks
    old_chunk_ids = [r["id"] for r in self.conn.execute(
        "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()]
    if old_chunk_ids:
        for cid in old_chunk_ids:
            self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    # Insert new chunks + vectors
    for chunk, emb in zip(raw_chunks, embeddings):
        self.conn.execute(
            "INSERT INTO chunks (doc_id, position, heading, text, stall_types, techniques) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, chunk["position"], chunk["heading"], chunk["text"],
             chunk["stalls"], chunk["techniques"])
        )
        chunk_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(emb))
        )

    self.conn.commit()
    return {"files": 1, "chunks": len(raw_chunks), "skipped": 0,
            "kept": kept_count, "discarded": discarded_count}

def ingest_all_tsv(self, force: bool = False) -> dict:
    """Ingest all experiment TSV files from known project directories."""
    tsv_dirs = [
        self.BWK_ROOT / "main/results",
        self.BWK_ROOT / "gemm/results",
        self.BWK_ROOT / "fused-mlp/results",
        self.BWK_ROOT / "attention/results",
        self.BWK_ROOT / "dotproduct/results",
        self.BWK_ROOT / "linalg/results",
        self.BWK_ROOT / "lu/results",
        self.BWK_ROOT / "qr/results",
        self.BWK_ROOT / "rmsnorm/results",
        self.BWK_ROOT / "spmv/results",
        self.BWK_ROOT / "numerical/results",
        self.BWK_ROOT / "cuquantum/results",
        self.BWK_ROOT / "chess-training/results",
    ]

    total = {"files": 0, "chunks": 0, "skipped": 0, "kept": 0, "discarded": 0}

    for d in tsv_dirs:
        if not d.is_dir():
            continue
        for tsv_file in sorted(d.glob("*.tsv")):
            print(f"  Indexing {tsv_file} ...", file=sys.stderr)
            stats = self.ingest_tsv(str(tsv_file), force=force)
            if stats["chunks"] > 0:
                print(f"    {stats['chunks']} experiments "
                      f"({stats.get('kept', 0)} kept, {stats.get('discarded', 0)} discarded)",
                      file=sys.stderr)
            for k in total:
                total[k] += stats.get(k, 0)

    return total

# -- Worker State --

# -- Summary Management --

def set_summary(self, doc_id: int, summary: str, signal: str):
    """Store a summary and signal for a document, and embed the summary."""
    self.conn.execute(
        "UPDATE documents SET summary = ?, signal = ?, has_summary = 1 WHERE id = ?",
        (summary, signal, doc_id)
    )
    # Embed the summary and store in vec_summaries
    emb = embed_texts([summary], task="search_document")[0]
    # Upsert into vec_summaries
    self.conn.execute("DELETE FROM vec_summaries WHERE doc_id = ?", (doc_id,))
    self.conn.execute(
        "INSERT INTO vec_summaries (doc_id, embedding) VALUES (?, ?)",
        (doc_id, serialize_f32(emb))
    )
    self.conn.commit()

def docs_without_summary(self) -> list[dict]:
    """Return documents that don't have summaries yet."""
    rows = self.conn.execute("""
        SELECT id, source_file, title, doc_type, kernel_type, provenance
        FROM documents WHERE has_summary = 0
        ORDER BY provenance, kernel_type
    """).fetchall()
    return [dict(r) for r in rows]

# -- Search --

