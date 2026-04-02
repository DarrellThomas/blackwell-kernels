"""Search helpers extracted from factory_brain.
These functions expect `self` to provide: conn, embed_query, serialize_f32, PROVENANCE_TIERS."""
import sqlite3
import re

# -- Search --

def search_summaries(self, query: str, k: int = 10,
                     kernel_type: str = None,
                     doc_type: str = None,
                     stall_type: str = None,
                     technique: str = None,
                     provenance: str = None,
                     level: int = 1) -> list[dict]:
    """Tiered search against document summaries (Level 2 vectors).

    level=1: returns signal lines (one-liners)
    level=2: returns full summaries (200-500 words)
    level=3: returns raw chunks from the matching documents
    """
    vec = self.embed_query(query)

    rows = self.conn.execute(
        """
        SELECT doc_id, distance
        FROM vec_summaries
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance
        """,
        (self.serialize_f32(vec), k * 5)
    ).fetchall()

    doc_ids = [r["doc_id"] for r in rows]
    distances = {r["doc_id"]: r["distance"] for r in rows}

    if not doc_ids:
        # Fall back to chunk search if no summaries exist yet
        return self.search_semantic(
            query, k=k, kernel_type=kernel_type, doc_type=doc_type,
            stall_type=stall_type, technique=technique, provenance=provenance
        )

    # Fetch document metadata and apply filters
    placeholders = ",".join("?" * len(doc_ids))
    where_clauses = []
    params = list(doc_ids)

    if kernel_type:
        where_clauses.append("kernel_type = ?")
        params.append(kernel_type)
    if doc_type:
        where_clauses.append("doc_type = ?")
        params.append(doc_type)
    if technique:
        where_clauses.append("techniques LIKE ?")
        params.append(f"%{technique}%")
    if provenance:
        where_clauses.append("provenance = ?")
        params.append(provenance)

    extra_where = ""
    if where_clauses:
        extra_where = "AND " + " AND ".join(where_clauses)

    results = self.conn.execute(f"""
        SELECT id, source_file, title, doc_type, kernel_type, provenance,
               is_empirical, techniques, summary, signal, has_summary
        FROM documents
        WHERE id IN ({placeholders})
        {extra_where}
    """, params).fetchall()

    # Optional stall filter: check if any chunk in the doc mentions the stall
    if stall_type:
        filtered = []
        for r in results:
            has_stall = self.conn.execute(
                "SELECT 1 FROM chunks WHERE doc_id = ? AND stall_types LIKE ? LIMIT 1",
                (r["id"], f"%{stall_type}%")
            ).fetchone()
            if has_stall:
                filtered.append(r)
        results = filtered

    # Build output with provenance-boosted ranking
    output = []
    for r in results:
        raw_dist = distances.get(r["id"], 999)
        boost = self.PROVENANCE_TIERS.get(r["provenance"], {}).get("boost", 1.0)
        if r["is_empirical"]:
            boost *= 1.15
        effective_dist = raw_dist / boost

        entry = {
            "doc_id": r["id"],
            "title": r["title"],
            "source_file": r["source_file"],
            "doc_type": r["doc_type"],
            "kernel_type": r["kernel_type"],
            "provenance": r["provenance"],
            "is_empirical": bool(r["is_empirical"]),
            "techniques": r["techniques"],
            "distance": raw_dist,
            "effective_distance": effective_dist,
        }

        if level == 1:
            entry["signal"] = r["signal"] if r["has_summary"] else r["title"]
        elif level == 2:
            entry["summary"] = r["summary"] if r["has_summary"] else "(no summary — use --full for raw chunks)"
            entry["signal"] = r["signal"] if r["has_summary"] else r["title"]
        # level 3 handled below

        output.append(entry)

    output.sort(key=lambda x: x["effective_distance"])
    output = output[:k]

    # Level 3: attach raw chunks for the matched documents
    if level >= 3:
        for entry in output:
            chunks = self.conn.execute(
                "SELECT text, heading, stall_types, techniques FROM chunks "
                "WHERE doc_id = ? ORDER BY position",
                (entry["doc_id"],)
            ).fetchall()
            entry["chunks"] = [dict(c) for c in chunks]
            entry["summary"] = entry.get("summary", "")

    return output

def search_semantic(self, query: str, k: int = 10,
                    kernel_type: str = None,
                    doc_type: str = None,
                    stall_type: str = None,
                    technique: str = None,
                    provenance: str = None) -> list[dict]:
    """Semantic similarity search with provenance-weighted ranking."""
    vec = self.embed_query(query)

    # Over-fetch for post-filtering
    rows = self.conn.execute(
        """
        SELECT chunk_id, distance
        FROM vec_chunks
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance
        """,
        (self.serialize_f32(vec), k * 5)
    ).fetchall()

    chunk_ids = [r["chunk_id"] for r in rows]
    distances = {r["chunk_id"]: r["distance"] for r in rows}

    if not chunk_ids:
        return []

    # Fetch chunk + doc metadata, apply filters
    placeholders = ",".join("?" * len(chunk_ids))
    where_clauses = []
    params = list(chunk_ids)

    if kernel_type:
        where_clauses.append("d.kernel_type = ?")
        params.append(kernel_type)
    if doc_type:
        where_clauses.append("d.doc_type = ?")
        params.append(doc_type)
    if stall_type:
        where_clauses.append("c.stall_types LIKE ?")
        params.append(f"%{stall_type}%")
    if technique:
        where_clauses.append("c.techniques LIKE ?")
        params.append(f"%{technique}%")
    if provenance:
        where_clauses.append("d.provenance = ?")
        params.append(provenance)

    extra_where = ""
    if where_clauses:
        extra_where = "AND " + " AND ".join(where_clauses)

    results = self.conn.execute(f"""
        SELECT c.id as chunk_id, c.text, c.heading, c.stall_types,
               c.techniques as chunk_techniques,
               d.source_file, d.doc_type, d.kernel_type, d.title,
               d.provenance, d.is_empirical, d.techniques as doc_techniques
        FROM chunks c
        JOIN documents d ON c.doc_id = d.id
        WHERE c.id IN ({placeholders})
        {extra_where}
    """, params).fetchall()

    # Build output with provenance-boosted ranking
    output = []
    for r in results:
        raw_dist = distances.get(r["chunk_id"], 999)
        prov = r["provenance"]
        boost = self.PROVENANCE_TIERS.get(prov, {}).get("boost", 1.0)
        # Empirical results get an additional boost
        if r["is_empirical"]:
            boost *= 1.15
        # Lower distance = better; divide by boost to promote high-provenance
        effective_dist = raw_dist / boost

        output.append({
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "heading": r["heading"],
            "source_file": r["source_file"],
            "doc_type": r["doc_type"],
            "kernel_type": r["kernel_type"],
            "title": r["title"],
            "stall_types": r["stall_types"],
            "techniques": r["chunk_techniques"],
            "provenance": prov,
            "is_empirical": bool(r["is_empirical"]),
            "distance": raw_dist,
            "effective_distance": effective_dist,
        })

    output.sort(key=lambda x: x["effective_distance"])
    return output[:k]

@staticmethod
def _sanitize_fts_query(query: str) -> str:
    """Escape FTS5 special characters, wrapping terms with punctuation in quotes."""
    tokens = query.split()
    safe = []
    for t in tokens:
        if t.upper() in ("AND", "OR", "NOT"):
            safe.append(t)
        elif re.search(r'[.:()*"\[\]{}^~!@#$%&]', t):
            safe.append(f'"{t}"')
        else:
            safe.append(t)
    return " ".join(safe)

def search_fts(self, query: str, k: int = 10,
               kernel_type: str = None,
               doc_type: str = None,
               stall_type: str = None,
               technique: str = None,
               provenance: str = None) -> list[dict]:
    """Full-text search using FTS5 with provenance boost."""
    safe_query = self._sanitize_fts_query(query).strip()
    if not safe_query or k <= 0:
        return []

    where_clauses = []
    params = [safe_query]

    if kernel_type:
        where_clauses.append("d.kernel_type = ?")
        params.append(kernel_type)
    if doc_type:
        where_clauses.append("d.doc_type = ?")
        params.append(doc_type)
    if stall_type:
        where_clauses.append("c.stall_types LIKE ?")
        params.append(f"%{stall_type}%")
    if technique:
        where_clauses.append("c.techniques LIKE ?")
        params.append(f"%{technique}%")
    if provenance:
        where_clauses.append("d.provenance = ?")
        params.append(provenance)

    extra_where = ""
    if where_clauses:
        extra_where = "AND " + " AND ".join(where_clauses)

    params.append(k * 3)  # over-fetch for re-ranking

    try:
        results = self.conn.execute(f"""
            SELECT c.id as chunk_id, c.text, c.heading, c.stall_types,
                   c.techniques as chunk_techniques,
                   d.source_file, d.doc_type, d.kernel_type, d.title,
                   d.provenance, d.is_empirical, d.techniques as doc_techniques,
                   fts.rank as fts_rank
            FROM fts_chunks fts
            JOIN chunks c ON c.id = fts.rowid
            JOIN documents d ON c.doc_id = d.id
            WHERE fts_chunks MATCH ?
            {extra_where}
            ORDER BY fts.rank
            LIMIT ?
        """, params).fetchall()
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "fts5" not in message and "syntax error" not in message:
            raise
        return []

    # Re-rank with provenance boost
    output = []
    for r in results:
        raw_rank = r["fts_rank"]
        boost = self.PROVENANCE_TIERS.get(r["provenance"], {}).get("boost", 1.0)
        if r["is_empirical"]:
            boost *= 1.15
        effective_rank = raw_rank / boost  # more negative = better in FTS5

        output.append({
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "heading": r["heading"],
            "source_file": r["source_file"],
            "doc_type": r["doc_type"],
            "kernel_type": r["kernel_type"],
            "title": r["title"],
            "stall_types": r["stall_types"],
            "techniques": r["chunk_techniques"],
            "provenance": r["provenance"],
            "is_empirical": bool(r["is_empirical"]),
            "score": raw_rank,
            "effective_score": effective_rank,
        })

    output.sort(key=lambda x: x["effective_score"])
    return output[:k]

def search_hybrid(self, query: str, k: int = 10,
                  kernel_type: str = None,
                  doc_type: str = None,
                  stall_type: str = None,
                  technique: str = None,
                  provenance: str = None,
                  semantic_weight: float = 0.7) -> list[dict]:
    """Combined semantic + FTS search with RRF and provenance weighting."""
    sem_results = self.search_semantic(
        query, k=k*2, kernel_type=kernel_type,
        doc_type=doc_type, stall_type=stall_type,
        technique=technique, provenance=provenance
    )
    fts_results = self.search_fts(
        query, k=k*2, kernel_type=kernel_type, doc_type=doc_type,
        stall_type=stall_type, technique=technique, provenance=provenance
    )

    # Reciprocal Rank Fusion
    rrf_k = 60
    scores = {}
    all_results = {}

    for rank, r in enumerate(sem_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + semantic_weight / (rrf_k + rank + 1)
        all_results[cid] = r

    fts_weight = 1.0 - semantic_weight
    for rank, r in enumerate(fts_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + fts_weight / (rrf_k + rank + 1)
        if cid not in all_results:
            all_results[cid] = r

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
    output = []
    for cid, score in ranked:
        r = all_results[cid]
        r["rrf_score"] = score
        output.append(r)

    return output

