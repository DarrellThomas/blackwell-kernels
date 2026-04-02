"""Ingest, chunking, and content tagging helpers for ResearchMemory."""
from __future__ import annotations

import csv
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import List

from common.memory import memory_embeddings as _mem_embed

# Chunking thresholds
MIN_CHUNK_CHARS = 200
CHUNK_OVERLAP_CHARS = 200

# Kernel/type inference aids
KERNEL_PREFIXES = [
    "attention", "gemm", "fused_mlp", "fusedmlp", "lu", "qr", "cholesky",
    "spmv", "dotproduct", "linalg", "rmsnorm", "swiglu", "fft", "trsm",
    "cg", "gmres", "eigenvalue", "convolution", "ichol", "ldlt", "bicgstab",
    "numerical", "cross", "all",
]

STALL_NAMES = [
    "long_scoreboard", "math_throttle", "barrier", "not_selected",
    "wait", "lg_throttle", "short_scoreboard", "tex_throttle",
    "mio_throttle", "drain", "dispatch_stall",
]

STALL_SEMANTICS = {
    "long_scoreboard": [
        "memory latency", "global load", "L2 miss", "DRAM", "bandwidth bound",
        "memory bound", "load stall", "data dependency", "scoreboard",
    ],
    "math_throttle": [
        "compute bound", "compute-bound", "ALU", "tensor core utilization",
        "math throughput", "instruction throughput", "compute intensity",
        "arithmetic intensity", "FLOPS", "mma throughput",
    ],
    "barrier": [
        "syncthreads", "__syncthreads", "warp divergence", "synchronization",
        "barrier overhead", "sync overhead", "load imbalance",
    ],
    "not_selected": [
        "occupancy", "warp scheduling", "eligible warps", "active warps",
        "blocks per SM", "register pressure", "shared memory pressure",
    ],
}

TECHNIQUE_PATTERNS = {
    "swizzle": [r"swizzle", r"XOR.*swizzle", r"swizzle.*XOR", r"bank.?free"],
    "double_buffer": [r"double.?buffer", r"ping.?pong", r"multi.?stage", r"pipeline.*stage"],
    "tiling": [r"tile.?size", r"block.?tile", r"\d+x\d+.*tile", r"tile.*\d+x\d+"],
    "occupancy": [r"occupancy", r"blocks?.?per.?SM", r"registers?.?per.?thread"],
    "vectorized_load": [r"vectori[sz]ed.*load", r"float4", r"uint4", r"128.?bit.*load", r"cp\.async"],
    "register_reuse": [r"register.*reus", r"register.*pressure", r"reg.*spill"],
    "shared_memory": [r"shared.?mem", r"smem", r"__shared__"],
    "warp_specialization": [r"warp.?special", r"producer.*consumer.*warp", r"async.*warp"],
    "fusion": [r"kernel.?fusion", r"fused.*kernel", r"epilogue.*fus"],
    "mma_layout": [r"mma\.sync", r"m16n8k", r"ldmatrix", r"fragment.*layout"],
    "fp8": [r"FP8", r"e4m3", r"e5m2", r"fp8.*accum", r"mixed.?precision.*fp8"],
    "quantization": [r"quantiz", r"INT8", r"INT4", r"MXFP", r"microscaling"],
}


# ------------ helpers -----------------

def file_content_hash(content: str) -> str:
    return _hash(content)


def _hash(content: str) -> str:
    return __import__("hashlib").sha256(content.encode()).hexdigest()[:16]


def extract_techniques(text: str) -> str:
    techniques = set()
    for technique, patterns in TECHNIQUE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                techniques.add(technique)
                break
    return ",".join(sorted(techniques)) if techniques else ""


def extract_stall_types(text: str) -> str:
    text = text.lower()
    stalls = set()
    for s in STALL_NAMES:
        if s in text:
            stalls.add(s)
    for stall_name, indicators in STALL_SEMANTICS.items():
        for indicator in indicators:
            if indicator.lower() in text:
                stalls.add(stall_name)
                break
    return ",".join(sorted(stalls)) if stalls else ""


def detect_empirical(text: str) -> bool:
    indicators = [
        r"\d+\.\d+x\s*(cuBLAS|SDPA|reference|ref|baseline)",
        r"vs_ref\s*[=:]\s*\d",
        r"duration[_\s]*(us|ms)\s*[=:]\s*\d",
        r"experiment\s+\d+",
        r"measured\s+",
        r"profil(e|ing)\s+(show|reveal|indicate)",
        r"ncu\s+(show|report|output)",
        r"\d+\s*%\s*(SM|occupancy|utilization|efficiency)",
        r"TFLOPS|GFLOPS|GB/s",
    ]
    for pat in indicators:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def infer_kernel_type(filepath: str, text: str = "") -> str:
    name = os.path.basename(filepath).lower()
    path_parts = filepath.lower()
    for prefix in KERNEL_PREFIXES:
        if name.startswith(prefix + "_") or name.startswith(prefix + "-"):
            return prefix
        if f"/{prefix}/" in path_parts:
            return prefix
    for prefix in KERNEL_PREFIXES:
        if prefix in text.lower():
            return prefix
    return "general"


def infer_doc_type(filepath: str, explicit: str = "research") -> str:
    if explicit:
        return explicit
    name = os.path.basename(filepath).lower()
    if name.endswith(('.cu', '.cuh', '.c', '.cc', '.cpp', '.py', '.pt', '.rs', '.go', '.m', '.mjs', '.js')):
        return "source_code"
    if name.endswith('.tsv'):
        return "experiment"
    return "research"


def infer_provenance(filepath: str, doc_type: str, source_priority) -> str:
    if doc_type == "experiment":
        return "validated"
    for src_path, _doc_type, prov, _priority in source_priority:
        try:
            path = Path(filepath).resolve()
        except Exception:
            path = Path(filepath)
        try:
            if src_path in path.parents or src_path == path:
                return prov
        except TypeError:
            continue
    return "research"


def chunk_markdown(text: str, source_file: str, doc_title: str = "",
                   max_chars: int = 2000, overlap_chars: int = CHUNK_OVERLAP_CHARS,
                   min_chars: int = MIN_CHUNK_CHARS) -> List[dict]:
    raw_sections = []
    sections = re.split(r'(?m)^(#{1,3}\s+.+)$', text)
    current_heading = doc_title or os.path.basename(source_file)
    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r'^#{1,3}\s+', part):
            current_heading = part.lstrip('#').strip()
            continue
        raw_sections.append((current_heading, part))

    chunks, position = [], 0
    carry_text = ""
    carry_heading = ""
    for heading, section_text in raw_sections:
        if carry_text:
            section_text = carry_text + "\n\n" + section_text
            if len(heading) <= len(carry_heading):
                heading = carry_heading
            carry_text = ""
            carry_heading = ""

        if len(section_text) < min_chars:
            carry_text = section_text
            carry_heading = heading
            continue

        if len(section_text) <= max_chars:
            chunks.append({"text": section_text, "heading": heading, "position": position})
            position += 1
        else:
            paragraphs = section_text.split('\n\n')
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) > max_chars and len(buffer) >= min_chars:
                    chunks.append({"text": buffer.strip(), "heading": heading, "position": position})
                    position += 1
                    buffer = buffer[-overlap_chars:] + "\n\n" + para
                else:
                    buffer = buffer + "\n\n" + para if buffer else para
            if buffer.strip():
                if len(buffer.strip()) >= min_chars:
                    chunks.append({"text": buffer.strip(), "heading": heading, "position": position})
                    position += 1
                else:
                    carry_text = buffer.strip()
                    carry_heading = heading
    if carry_text:
        if chunks:
            chunks[-1]["text"] += "\n\n" + carry_text
        elif len(carry_text) > 50:
            chunks.append({"text": carry_text, "heading": carry_heading, "position": 0})
    return chunks


def chunk_code(text: str, source_file: str, max_chars: int = 2000,
               min_chars: int = MIN_CHUNK_CHARS) -> List[dict]:
    filename = os.path.basename(source_file)
    pattern = r'(?m)^((?:__global__|__device__|__host__|template\s*<|__forceinline__).+)'
    parts = re.split(pattern, text)
    raw_segments = []
    i = 0
    while i < len(parts):
        segment = parts[i]
        if i + 1 < len(parts):
            segment = parts[i] + parts[i + 1]
            i += 2
        else:
            i += 1
        segment = segment.strip()
        if segment:
            raw_segments.append(segment)

    chunks = []
    position = 0
    buffer = ""
    for segment in raw_segments:
        if buffer and len(buffer) + len(segment) > max_chars and len(buffer) >= min_chars:
            chunks.append({"text": buffer, "heading": f"code:{filename}", "position": position})
            position += 1
            buffer = segment
        else:
            buffer = buffer + "\n\n" + segment if buffer else segment
    if buffer and len(buffer) > 50:
        chunks.append({"text": buffer, "heading": f"code:{filename}", "position": position})
        position += 1

    final = []
    for chunk in chunks:
        if len(chunk["text"]) <= max_chars:
            final.append(chunk)
        else:
            lines = chunk["text"].split('\n')
            buf = ""
            for line in lines:
                if len(buf) + len(line) > max_chars and len(buf) >= min_chars:
                    final.append({"text": buf, "heading": chunk["heading"], "position": chunk["position"]})
                    buf = line
                else:
                    buf = buf + "\n" + line if buf else line
            if buf and len(buf) > 50:
                final.append({"text": buf, "heading": chunk["heading"], "position": chunk["position"]})
    for i, c in enumerate(final):
        c["position"] = i
    return final


# ------------ ingest binding -----------------

def attach_ingest_methods(cls):
    cls.ingest_all = ingest_all
    cls.ingest_file = ingest_file
    cls.ingest_directory = ingest_directory
    cls._index_document = _index_document
    cls.ingest_tsv = ingest_tsv
    cls.ingest_all_tsv = ingest_all_tsv
    return cls


# ------------ bound methods -----------------

def ingest_all(self, force: bool = False) -> dict:
    file_hashes = {}
    hash_map = {}
    for src_path, doc_type, prov_hint, priority in self.SOURCE_PRIORITY:
        src_path = Path(src_path).expanduser()
        if not src_path.exists():
            continue
        print(f"Scanning {src_path}...", file=sys.stderr)
        for filepath in src_path.glob('**/*'):
            if filepath.is_dir():
                continue
            if filepath.suffix.lower() in {'.cu', '.cuh', '.c', '.cc', '.cpp', '.h', '.hpp', '.py', '.rs', '.go', '.m', '.mjs', '.js'}:
                if str(filepath) not in getattr(self, 'CODE_SOURCES', []):
                    continue
            try:
                content = Path(filepath).read_text(errors="replace")
            except Exception:
                continue
            if len(content.strip()) < 50:
                continue
            ch = file_content_hash(content)
            file_hashes[str(filepath)] = ch
            if ch not in hash_map:
                hash_map[ch] = []
            hash_map[ch].append((str(filepath), doc_type, prov_hint, priority))

    canonical = {}
    total_files = 0
    dedup_skipped = 0
    for ch, entries in hash_map.items():
        entries.sort(key=lambda x: x[3])
        best = entries[0]
        also_at = [e[0] for e in entries[1:]]
        canonical[ch] = (best[0], best[1], best[2], also_at)
        total_files += 1
        dedup_skipped += len(also_at)

    print(f"  {total_files} unique documents ({dedup_skipped} duplicates eliminated)", file=sys.stderr)
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


def ingest_file(self, filepath: str, doc_type: str = "research", force: bool = False) -> int:
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
    prov = infer_provenance(filepath, infer_doc_type(filepath, doc_type), self.SOURCE_PRIORITY)
    return self._index_document(filepath, doc_type, prov, [], ch)


def ingest_directory(self, dirpath: str, doc_type: str = "research", pattern: str = "**/*.md", force: bool = False) -> dict:
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


def _index_document(self, filepath: str, doc_type: str, prov_hint: str, also_at: list[str], content_hash: str) -> int:
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
    provenance = infer_provenance(filepath, real_doc_type, self.SOURCE_PRIORITY)
    if provenance == "research" and prov_hint in ("validated", "reference"):
        provenance = prov_hint
    title = self._extract_title(content, filepath)
    is_empirical = detect_empirical(content)
    doc_techniques = extract_techniques(content)
    file_mod = time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(filepath)))
    also_at_str = "\n".join(also_at[:20]) if also_at else ""
    if real_doc_type == "source_code":
        raw_chunks = chunk_code(content, filepath)
    else:
        raw_chunks = chunk_markdown(content, filepath, doc_title=title)
    if not raw_chunks:
        return 0
    texts = [c["text"] for c in raw_chunks]
    embeddings = _mem_embed.embed_texts(texts, task="search_document")
    self.conn.execute(
        """
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
        """,
        (filepath, content_hash, real_doc_type, kernel_type, provenance,
         title, file_mod, len(raw_chunks), int(is_empirical), doc_techniques, also_at_str)
    )
    doc_id = self.conn.execute("SELECT id FROM documents WHERE source_file = ?", (filepath,)).fetchone()["id"]
    if content_changed:
        self.conn.execute("UPDATE documents SET summary = '', signal = '', has_summary = 0 WHERE id = ?", (doc_id,))
        self.conn.execute("DELETE FROM vec_summaries WHERE doc_id = ?", (doc_id,))
    old_chunk_ids = [r["id"] for r in self.conn.execute(
        "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()]
    if old_chunk_ids:
        for cid in old_chunk_ids:
            self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    for chunk, emb in zip(raw_chunks, embeddings):
        stalls = extract_stall_types(chunk["text"])
        techs = extract_techniques(chunk["text"])
        self.conn.execute(
            "INSERT INTO chunks (doc_id, position, heading, text, stall_types, techniques) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, chunk["position"], chunk["heading"], chunk["text"], stalls, techs)
        )
        chunk_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)", (chunk_id, _mem_embed.serialize_f32(emb)))
    self.conn.commit()
    return len(raw_chunks)


def ingest_tsv(self, tsv_path: str, kernel_type: str = None, force: bool = False) -> dict:
    tsv_path = str(Path(tsv_path).resolve())
    if not os.path.isfile(tsv_path):
        return {"files": 0, "chunks": 0, "skipped": 0}
    if not kernel_type:
        parts = tsv_path.split("/")
        for i, p in enumerate(parts):
            if p == "results" and i > 0:
                kernel_type = parts[i - 1]
                break
        dir_to_kernel = {"main": "attention", "fused-mlp": "fused_mlp", "chess-training": "chess_training"}
        kernel_type = dir_to_kernel.get(kernel_type, kernel_type) or "general"

    content_hash = file_content_hash(Path(tsv_path).read_text(errors="replace"))
    existing = self.conn.execute(
        "SELECT id, content_hash, chunk_count FROM documents WHERE source_file = ?",
        (tsv_path,)
    ).fetchone()
    doc_up_to_date = bool(existing and existing["content_hash"] == content_hash and not force)

    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        return {"files": 0, "chunks": 0, "skipped": 1}

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
        parts_txt = [desc]
        metrics = []
        if vs_ref and vs_ref != "-":
            metrics.append(f"vs_ref={vs_ref}")
        if duration and duration != "-":
            metrics.append(f"duration={duration}us")
        if sm_pct and sm_pct != "-":
            metrics.append(f"SM={sm_pct}%")
        if top_stall and top_stall not in {"-", "none", ""}:
            metrics.append(f"top_stall={top_stall}")
        if status:
            metrics.append(f"status={status}")
        if metrics:
            parts_txt.append(f"[{', '.join(metrics)}]")
        chunk_text = " ".join(parts_txt)
        stalls = set()
        if top_stall and top_stall not in {"-", "none", ""}:
            stalls.add(top_stall)
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
            text_val = str(value or "").strip()
            if not text_val or text_val == "-":
                return None
            try:
                return float(text_val)
            except ValueError:
                return None
        extra = {k: v for k, v in row.items() if k not in {"timestamp", "commit", "duration_us", "vs_ref", "sm_pct", "stall_math", "stall_wait", "stall_scoreboard", "stall_barrier", "top_stall", "status", "description"}}
        extras = {
            "timestamp": timestamp,
            "commit": commit,
            "duration_us": _to_float(duration),
            "vs_ref": _to_float(vs_ref),
            "sm_pct": _to_float(sm_pct),
            "stall_math": _to_float(row.get("stall_math")),
            "stall_wait": _to_float(row.get("stall_wait")),
            "stall_scoreboard": _to_float(row.get("stall_scoreboard")),
            "stall_barrier": _to_float(row.get("stall_barrier")),
            **extra,
        }
        raw_chunks.append({
            "text": chunk_text,
            "heading": heading,
            "position": i,
            "stall_types": stall_str,
            "techniques": techs,
            "extras": extras,
            "status": status,
            "top_stall": top_stall,
            "vs_ref": vs_ref,
            "duration": duration,
            "sm_pct": sm_pct,
        })
    if not raw_chunks:
        return {"files": 0, "chunks": 0, "skipped": 1}

    if doc_up_to_date:
        return {"files": 0, "chunks": existing["chunk_count"], "skipped": 1}

    if not force:
        row = self.conn.execute(
            "SELECT id, content_hash FROM documents WHERE source_file = ?",
            (tsv_path,)
        ).fetchone()
        if row and row["content_hash"] == content_hash:
            return {"files": 0, "chunks": row["chunk_count"], "skipped": 1}

    self.conn.execute(
        "INSERT INTO documents (source_file, content_hash, doc_type, kernel_type, provenance, title, date_indexed, file_modified, chunk_count, is_empirical, techniques, also_at)\n         VALUES (?, ?, 'experiment', ?, 'validated', ?, date('now'), date('now'), ?, 1, '', '')\n         ON CONFLICT(source_file) DO UPDATE SET content_hash=excluded.content_hash, doc_type=excluded.doc_type, kernel_type=excluded.kernel_type, provenance=excluded.provenance, title=excluded.title, date_indexed=excluded.date_indexed, file_modified=excluded.file_modified, chunk_count=excluded.chunk_count, is_empirical=excluded.is_empirical, techniques=excluded.techniques, also_at=excluded.also_at",
        (tsv_path, content_hash, kernel_type, title, len(raw_chunks))
    )
    doc_id = self.conn.execute("SELECT id FROM documents WHERE source_file = ?", (tsv_path,)).fetchone()["id"]
    self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)", (doc_id,))
    self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    for chunk in raw_chunks:
        self.conn.execute(
            "INSERT INTO chunks (doc_id, position, heading, text, stall_types, techniques) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, chunk["position"], chunk["heading"], chunk["text"], chunk["stall_types"], chunk["techniques"])
        )
        chunk_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        emb = _mem_embed.embed_texts([chunk["text"]], task="search_document")[0]
        self.conn.execute("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)", (chunk_id, _mem_embed.serialize_f32(emb)))
        self.conn.execute(
            "INSERT OR IGNORE INTO experiments (kernel_type, job_id, source_type, source_path, row_hash, experiment_index, timestamp, git_commit, duration_us, vs_ref, sm_pct, stall_math, stall_wait, stall_scoreboard, stall_barrier, top_stall, status, description, reference_label, extra_json, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, date('now'))",
            (
                kernel_type, None, 'tsv', tsv_path, file_content_hash(chunk["text"] + str(chunk["position"])),
                chunk["position"], chunk.get("timestamp", timestamp), chunk.get("commit", commit), chunk.get("duration"), chunk.get("vs_ref"), chunk.get("sm_pct"), chunk["extras"].get("stall_math"), chunk["extras"].get("stall_wait"), chunk["extras"].get("stall_scoreboard"), chunk["extras"].get("stall_barrier"), chunk.get("top_stall", ""), chunk.get("status", ""), chunk["text"], '', __import__('json').dumps(chunk["extras"], sort_keys=True)
            )
        )
    self.conn.commit()
    return {"files": 1, "chunks": len(raw_chunks), "skipped": 0}


def ingest_all_tsv(self, force: bool = False) -> dict:
    results = {"files": 0, "chunks": 0, "skipped": 0}
    for kernel_dir in (Path(self.BWK_ROOT) if hasattr(self, 'BWK_ROOT') else Path('.')).iterdir():
        tsv_dir = kernel_dir / 'results'
        if not tsv_dir.is_dir():
            continue
        for tsv_file in tsv_dir.glob('*.tsv'):
            stats = self.ingest_tsv(str(tsv_file), force=force)
            for k, v in stats.items():
                results[k] = results.get(k, 0) + v
    return results
