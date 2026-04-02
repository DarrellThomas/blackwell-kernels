#!/usr/bin/env python3
"""
Factory Brain — Central database for the blackwell-kernels factory.

Provides semantic search (sqlite-vec) + full-text search (FTS5) over all
research briefs, source code, documentation, and experiment results.

Quality-first design:
  - Content-hash deduplication (one canonical copy per unique document)
  - Minimum chunk size with context preservation
  - Provenance tiers (validated > reference > research > archive)
  - Rich metadata (techniques, stall types, empirical flags)
  - Provenance-weighted search ranking

Usage:
    python factory_brain.py ingest              # Index all research directories
    python factory_brain.py search "query"      # Hybrid semantic + FTS search
    python factory_brain.py fts "exact phrase"   # Full-text search
    python factory_brain.py stats                # Database statistics
    python factory_brain.py quality              # Quality audit report
    python factory_brain.py serve [port]         # HTTP API (default 8421)
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import sys
import time
from pathlib import Path
from typing import Optional

# Allow direct script execution without requiring PYTHONPATH to be preset.
if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import sqlite_vec
from common.memory import memory_config as _mem_cfg
def validate_transition(from_state: str, to_state: str) -> tuple:
    if from_state == to_state:
        return False, f"Already in state '{from_state}'"
    if from_state not in STATE_TO_PHASE:
        return False, f"Unknown source state '{from_state}'"
    if to_state not in STATE_TO_PHASE:
        return False, f"Unknown target state '{to_state}'"
    if from_state == 'shipped' and to_state not in ('converged', 'parked', 'abandoned'):
        return False, (
            "Cannot go backward from 'shipped'. Shipped work is versioned. "
            "To make changes, create a new job (fb job-create)."
        )
    if from_state in ('converged', 'parked', 'abandoned'):
        return True, ""
    return True, ""


from common.memory import memory_search as _mem_search
from common.memory import memory_messages as _mem_messages
from common.memory import memory_workers as _mem_workers
from common.memory import memory_jobs as _mem_jobs
from common.memory import memory_experiments as _mem_exps
from common.memory import memory_stats as _mem_stats
from common.memory import memory_issues as _mem_issues
from common.memory import memory_ingest as _mem_ingest
from common.memory.memory_helpers import format_result, format_quality_result, attach_helper_methods

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_DIR = _mem_cfg.DB_DIR
DB_PATH = _mem_cfg.DB_PATH
BWK_ROOT = _mem_cfg.BWK_ROOT

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
EMBEDDING_DEVICE = "cpu"  # embeddings are cheap, keep GPU free for kernels
MAX_CHUNK_TOKENS = 512
MIN_CHUNK_CHARS = 200  # chunks below this get merged with neighbors
CHUNK_OVERLAP_CHARS = 200

# ---------------------------------------------------------------------------
# Provenance: path priority determines the canonical copy of each document.
# Lower number = higher authority. When the same content_hash appears in
# multiple directories, only the highest-priority copy is indexed.
# ---------------------------------------------------------------------------

SOURCE_PRIORITY = _mem_cfg.SOURCE_PRIORITY
CODE_SOURCES = _mem_cfg.CODE_SOURCES

# ---------------------------------------------------------------------------
# Provenance tier definitions — used for search ranking
# ---------------------------------------------------------------------------

PROVENANCE_TIERS = _mem_cfg.PROVENANCE_TIERS

# ---------------------------------------------------------------------------
# Technique and stall vocabulary for semantic tagging
# ---------------------------------------------------------------------------

STALL_NAMES = [
    "long_scoreboard", "math_throttle", "barrier", "not_selected",
    "wait", "lg_throttle", "short_scoreboard", "tex_throttle",
    "mio_throttle", "drain", "dispatch_stall",
]

# Semantic stall indicators — phrases that imply a stall type even without
# using the literal stall name
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

# ---------------------------------------------------------------------------
# Job State Machine — Phase-Based
# ---------------------------------------------------------------------------

JOB_PHASES = _mem_cfg.JOB_PHASES
PHASE_ORDER = _mem_cfg.PHASE_ORDER
STATE_TO_PHASE = _mem_cfg.STATE_TO_PHASE
ALL_JOB_STATES = _mem_cfg.ALL_JOB_STATES

JOB_TYPES = _mem_cfg.JOB_TYPES
JOB_PRIORITIES = _mem_cfg.JOB_PRIORITIES
MESSAGE_TYPES = _mem_cfg.MESSAGE_TYPES
MESSAGE_STATUSES = _mem_cfg.MESSAGE_STATUSES
MESSAGE_PRIORITIES = _mem_cfg.MESSAGE_PRIORITIES
FACTORY_MODES = _mem_cfg.FACTORY_MODES
OPTIMIZATION_SCOPES = _mem_cfg.OPTIMIZATION_SCOPES
EXECUTION_LANES = _mem_cfg.EXECUTION_LANES

# ---------------------------------------------------------------------------
# Embedding Model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from transformers import AutoModel, AutoTokenizer
    import torch

    print(f"Loading embedding model: {EMBEDDING_MODEL}...", file=sys.stderr)
    _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    _model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    _model.eval()
    _model.to(EMBEDDING_DEVICE)
    print("Model loaded.", file=sys.stderr)


def embed_texts(texts: list[str], task: str = "search_document") -> list[list[float]]:
    """Embed a batch of texts. task is 'search_document' for indexing, 'search_query' for queries."""
    _load_model()
    import torch

    # nomic-embed-text-v1.5 uses task-prefixed inputs
    prefixed = [f"{task}: {t}" for t in texts]
    encoded = _tokenizer(
        prefixed, padding=True, truncation=True, max_length=MAX_CHUNK_TOKENS,
        return_tensors="pt"
    ).to(EMBEDDING_DEVICE)

    with torch.no_grad():
        output = _model(**encoded)
        # Mean pooling over token embeddings
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embeddings = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query."""
    return embed_texts([text], task="search_query")[0]


def serialize_f32(vec: list[float]) -> bytes:
    """Pack a float list into bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# Chunking — quality-aware
# ---------------------------------------------------------------------------

def chunk_markdown(text: str, source_file: str, doc_title: str = "",
                   max_chars: int = 2000, overlap_chars: int = CHUNK_OVERLAP_CHARS,
                   min_chars: int = MIN_CHUNK_CHARS) -> list[dict]:
    """Split markdown by headings, then by size. Merges small fragments.

    Each chunk carries its heading context so it's meaningful in isolation.
    Chunks below min_chars are merged with the next chunk.
    """
    raw_sections = []

    # Split on headings (## or #)
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

    # Build chunks with minimum size enforcement
    chunks = []
    position = 0
    carry_text = ""
    carry_heading = ""

    for heading, section_text in raw_sections:
        # If we have carry-over from a too-small previous section, merge it
        if carry_text:
            section_text = carry_text + "\n\n" + section_text
            # Keep the more specific heading
            if len(heading) > len(carry_heading):
                pass  # use new heading
            else:
                heading = carry_heading
            carry_text = ""
            carry_heading = ""

        if len(section_text) < min_chars:
            # Too small — carry forward to merge with next section
            carry_text = section_text
            carry_heading = heading
            continue

        if len(section_text) <= max_chars:
            chunks.append({
                "text": section_text,
                "heading": heading,
                "position": position,
            })
            position += 1
        else:
            # Split by paragraphs, respecting minimum size
            paragraphs = section_text.split('\n\n')
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) > max_chars and len(buffer) >= min_chars:
                    chunks.append({
                        "text": buffer.strip(),
                        "heading": heading,
                        "position": position,
                    })
                    position += 1
                    buffer = buffer[-overlap_chars:] + "\n\n" + para
                else:
                    buffer = buffer + "\n\n" + para if buffer else para

            if buffer.strip():
                if len(buffer.strip()) >= min_chars:
                    chunks.append({
                        "text": buffer.strip(),
                        "heading": heading,
                        "position": position,
                    })
                    position += 1
                else:
                    # Too small — carry forward
                    carry_text = buffer.strip()
                    carry_heading = heading

    # Flush any remaining carry
    if carry_text:
        if chunks:
            # Merge into last chunk
            chunks[-1]["text"] += "\n\n" + carry_text
        elif len(carry_text) > 50:
            chunks.append({
                "text": carry_text,
                "heading": carry_heading,
                "position": 0,
            })

    return chunks


def chunk_code(text: str, source_file: str, max_chars: int = 2000,
               min_chars: int = MIN_CHUNK_CHARS) -> list[dict]:
    """Split source code by function/kernel boundaries.

    Keeps complete function bodies together. Merges small fragments
    (single declarations, includes) into neighboring chunks.
    """
    filename = os.path.basename(source_file)

    # Split on function-level boundaries
    # Match: __global__, __device__, __host__, template<, standalone function defs
    pattern = r'(?m)^((?:__global__|__device__|__host__|template\s*<|__forceinline__).+)'
    parts = re.split(pattern, text)

    # Reassemble: each function signature + its body
    raw_segments = []
    i = 0
    while i < len(parts):
        segment = parts[i]
        # If next part is a function body (follows a signature match), combine
        if i + 1 < len(parts):
            segment = parts[i] + parts[i + 1]
            i += 2
        else:
            i += 1
        segment = segment.strip()
        if segment:
            raw_segments.append(segment)

    # Merge small segments and build final chunks
    chunks = []
    position = 0
    buffer = ""

    for segment in raw_segments:
        if buffer and len(buffer) + len(segment) > max_chars and len(buffer) >= min_chars:
            chunks.append({
                "text": buffer,
                "heading": f"code:{filename}",
                "position": position,
            })
            position += 1
            buffer = segment
        else:
            buffer = buffer + "\n\n" + segment if buffer else segment

    if buffer and len(buffer) > 50:
        chunks.append({
            "text": buffer,
            "heading": f"code:{filename}",
            "position": position,
        })
        position += 1

    # Handle very large chunks (shouldn't happen often with function splitting)
    final = []
    for chunk in chunks:
        if len(chunk["text"]) <= max_chars:
            final.append(chunk)
        else:
            # Split at blank lines within the function
            lines = chunk["text"].split('\n')
            buf = ""
            for line in lines:
                if len(buf) + len(line) > max_chars and len(buf) >= min_chars:
                    final.append({
                        "text": buf,
                        "heading": chunk["heading"],
                        "position": chunk["position"],
                    })
                    buf = line
                else:
                    buf = buf + "\n" + line if buf else line
            if buf and len(buf) > 50:
                final.append({
                    "text": buf,
                    "heading": chunk["heading"],
                    "position": chunk["position"],
                })

    # Re-number positions
    for i, c in enumerate(final):
        c["position"] = i

    return final


# ---------------------------------------------------------------------------
# Metadata Extraction — rich tagging
# ---------------------------------------------------------------------------

def infer_kernel_type(filepath: str, text: str = "") -> str:
    """Infer kernel type from filename, path, or content."""
    kernel_prefixes = _mem_ingest.KERNEL_PREFIXES
    name = os.path.basename(filepath).lower()
    path_parts = filepath.lower()

    for prefix in kernel_prefixes:
        if name.startswith(prefix + "_") or name.startswith(prefix + "-"):
            return prefix
        # Match directory component exactly (not substring)
        if f"/{prefix}/" in path_parts:
            return prefix

    # Content-based fallback for files that don't follow naming conventions
    if text:
        text_lower = text[:2000].lower()
        for prefix in kernel_prefixes:
            if prefix == "all" or prefix == "cross":
                continue  # too generic for content matching
            # Require the kernel name to appear as a word
            if re.search(rf'\b{prefix}\b', text_lower):
                return prefix

    return "general"


def infer_doc_type(filepath: str, explicit: str = "research") -> str:
    """Determine document type from path context."""
    fp = filepath.lower()
    if any(ext in fp for ext in [".cu", ".cuh", ".h", ".cpp"]):
        return "source_code"
    if "/playbook" in fp or "playbook" in os.path.basename(fp):
        return "playbook"
    if "dead_end" in fp or "hard_won" in fp:
        return "dead_end"
    if "/archive/" in fp:
        return "archive"
    if "agent_state" in fp:
        return "agent_state"
    return explicit


def infer_provenance(filepath: str, doc_type: str) -> str:
    """Determine provenance tier from file location and type."""
    fp = filepath.lower()

    # Validated tier: approved briefs, playbooks, hard-won lessons
    if "/approved/" in fp:
        return "validated"
    if doc_type in ("playbook", "dead_end"):
        return "validated"
    if "hard_won" in fp or "agent_state" in fp:
        return "validated"

    # Reference tier: common/docs, manuals
    if "/common/docs/" in fp:
        return "reference"
    if doc_type == "source_code":
        if "/common/csrc/" in fp:
            return "reference"  # canonical shared code
        return "research"  # worker code is active research

    # Archive tier
    if "/archive/" in fp or doc_type == "archive":
        return "archive"

    return "research"


def extract_stall_types(text: str) -> str:
    """Extract stall types from text using both literal and semantic matching."""
    stalls = set()
    text_lower = text.lower()

    # Literal matches
    for s in STALL_NAMES:
        if s in text_lower:
            stalls.add(s)

    # Semantic matches — phrases that imply a stall type
    for stall_name, indicators in STALL_SEMANTICS.items():
        for phrase in indicators:
            if phrase.lower() in text_lower:
                stalls.add(stall_name)
                break

    return ",".join(sorted(stalls)) if stalls else ""


def extract_techniques(text: str) -> str:
    """Extract optimization technique tags from text content."""
    techniques = set()
    for technique, patterns in TECHNIQUE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                techniques.add(technique)
                break
    return ",".join(sorted(techniques)) if techniques else ""


def detect_empirical(text: str) -> bool:
    """Detect if content contains empirically-measured results (not just theory)."""
    indicators = [
        r'\d+\.\d+x\s*(cuBLAS|SDPA|reference|ref|baseline)',  # "1.34x cuBLAS"
        r'vs_ref\s*[=:]\s*\d',                                  # vs_ref = 1.34
        r'duration[_\s]*(us|ms)\s*[=:]\s*\d',                   # duration_us = 52
        r'experiment\s+\d+',                                      # experiment 47
        r'measured\s+',                                           # "measured latency"
        r'profil(e|ing)\s+(show|reveal|indicate)',               # "profiling shows"
        r'ncu\s+(show|report|output)',                           # "ncu shows"
        r'\d+\s*%\s*(SM|occupancy|utilization|efficiency)',      # "43.8% SM"
        r'TFLOPS|GFLOPS|GB/s',                                  # throughput measurements
    ]
    for pat in indicators:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def file_content_hash(content: str) -> str:
    """SHA256 of content for change detection and deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Database Schema & Manager
# ---------------------------------------------------------------------------

class ResearchMemory:
    """Manages the research memory database."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        c = self.conn
        # -- Documents table (one row per unique document) --
        c.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                doc_type TEXT NOT NULL DEFAULT 'research',
                kernel_type TEXT NOT NULL DEFAULT 'general',
                provenance TEXT NOT NULL DEFAULT 'research',
                title TEXT,
                date_indexed TEXT NOT NULL,
                file_modified TEXT,
                chunk_count INTEGER DEFAULT 0,
                is_empirical INTEGER DEFAULT 0,
                techniques TEXT DEFAULT '',
                also_at TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                signal TEXT DEFAULT '',
                has_summary INTEGER DEFAULT 0
            )
        """)

        # -- Chunks table (text content for each chunk) --
        c.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                heading TEXT,
                text TEXT NOT NULL,
                stall_types TEXT DEFAULT '',
                techniques TEXT DEFAULT '',
                UNIQUE(doc_id, position)
            )
        """)

        # -- Vector index (sqlite-vec) --
        c.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            )
        """)

        # -- Vector index for summaries (Level 2 search target) --
        c.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_summaries USING vec0(
                doc_id INTEGER PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            )
        """)

        # -- Full-text search index (FTS5) --
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                text,
                heading,
                content=chunks,
                content_rowid=id,
                tokenize='porter unicode61'
            )
        """)

        # -- Triggers to keep FTS in sync --
        c.executescript("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO fts_chunks(rowid, text, heading)
                VALUES (new.id, new.text, new.heading);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, text, heading)
                VALUES ('delete', old.id, old.text, old.heading);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, text, heading)
                VALUES ('delete', old.id, old.text, old.heading);
                INSERT INTO fts_chunks(rowid, text, heading)
                VALUES (new.id, new.text, new.heading);
            END;
        """)

        # -- Indexes --
        c.execute("CREATE INDEX IF NOT EXISTS idx_docs_kernel ON documents(kernel_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(doc_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_docs_provenance ON documents(provenance)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_stall ON chunks(stall_types)")

        # -- Worker state table (computed from TSV data + filesystem signals) --
        c.execute("""
            CREATE TABLE IF NOT EXISTS worker_state (
                kernel_type TEXT PRIMARY KEY,
                tsv_path TEXT,
                total_experiments INTEGER DEFAULT 0,
                kept INTEGER DEFAULT 0,
                discarded INTEGER DEFAULT 0,
                best_vsref REAL,
                best_duration_us REAL,
                top_stall TEXT DEFAULT '',
                current_discard_streak INTEGER DEFAULT 0,
                max_discard_streak INTEGER DEFAULT 0,
                last_kept_description TEXT DEFAULT '',
                last_experiment_time TEXT DEFAULT '',
                has_halt_note INTEGER DEFAULT 0,
                status TEXT DEFAULT 'unknown',
                diagnosis TEXT DEFAULT '',
                updated_at TEXT DEFAULT '',
                heartbeat_at TEXT DEFAULT '',
                current_task TEXT DEFAULT '',
                job_id INTEGER DEFAULT NULL,
                process_state TEXT DEFAULT '',
                live_status TEXT DEFAULT '',
                live_reason TEXT DEFAULT '',
                activity_at TEXT DEFAULT ''
            )
        """)
        # Migration: add columns if upgrading from older schema
        for col, sql in [
            ('heartbeat_at', "ALTER TABLE worker_state ADD COLUMN heartbeat_at TEXT DEFAULT ''"),
            ('current_task', "ALTER TABLE worker_state ADD COLUMN current_task TEXT DEFAULT ''"),
            ('job_id', "ALTER TABLE worker_state ADD COLUMN job_id INTEGER DEFAULT NULL"),
            ('process_state', "ALTER TABLE worker_state ADD COLUMN process_state TEXT DEFAULT ''"),
            ('live_status', "ALTER TABLE worker_state ADD COLUMN live_status TEXT DEFAULT ''"),
            ('live_reason', "ALTER TABLE worker_state ADD COLUMN live_reason TEXT DEFAULT ''"),
            ('activity_at', "ALTER TABLE worker_state ADD COLUMN activity_at TEXT DEFAULT ''"),
        ]:
            try:
                c.execute(sql)
            except sqlite3.OperationalError:
                pass

        c.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kernel_type TEXT NOT NULL,
                job_id INTEGER DEFAULT NULL REFERENCES jobs(id),
                source_type TEXT NOT NULL DEFAULT 'db',
                source_path TEXT DEFAULT '',
                row_hash TEXT UNIQUE NOT NULL,
                experiment_index INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT '',
                git_commit TEXT DEFAULT '',
                duration_us REAL DEFAULT NULL,
                vs_ref REAL DEFAULT NULL,
                sm_pct REAL DEFAULT NULL,
                stall_math REAL DEFAULT NULL,
                stall_wait REAL DEFAULT NULL,
                stall_scoreboard REAL DEFAULT NULL,
                stall_barrier REAL DEFAULT NULL,
                top_stall TEXT DEFAULT '',
                status TEXT DEFAULT '',
                description TEXT DEFAULT '',
                reference_label TEXT DEFAULT '',
                extra_json TEXT DEFAULT '',
                recorded_at TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_experiments_kernel ON experiments(kernel_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_experiments_job ON experiments(job_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_experiments_time ON experiments(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER DEFAULT NULL REFERENCES jobs(id),
                kernel_type TEXT DEFAULT '',
                category TEXT NOT NULL,
                command TEXT NOT NULL,
                status TEXT NOT NULL,
                output TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_test_runs_job ON test_runs(job_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_test_runs_kernel ON test_runs(kernel_type)")

        # -- Issues table (tester → foreman → worker → tester cycle) --
        c.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'warning',
                status TEXT NOT NULL DEFAULT 'open',
                kernel_type TEXT DEFAULT '',
                source_file TEXT DEFAULT '',
                filed_by TEXT DEFAULT 'tester',
                assigned_to TEXT DEFAULT '',
                description TEXT DEFAULT '',
                reproduce TEXT DEFAULT '',
                fix_description TEXT DEFAULT '',
                filed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                closed_at TEXT DEFAULT ''
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_issues_kernel ON issues(kernel_type)")

        # -- Primitives manifest (version control for shipped kernels) --
        c.execute("""
            CREATE TABLE IF NOT EXISTS primitives (
                shelf_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                has_blas_interface INTEGER DEFAULT 0,
                vs_ref REAL,
                shipped_from TEXT DEFAULT '',
                shipped_by TEXT DEFAULT '',
                shipped_at TEXT NOT NULL,
                tests_passed TEXT DEFAULT '',
                lint_clean INTEGER DEFAULT 0
            )
        """)

        # -- Jobs table (workpiece state tracker) --
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                job_type TEXT NOT NULL DEFAULT 'kernel',
                kernel_type TEXT DEFAULT '',
                parent_job_id INTEGER DEFAULT NULL REFERENCES jobs(id),
                state TEXT NOT NULL DEFAULT 'wishlist',
                phase TEXT NOT NULL DEFAULT 'ideation',
                priority TEXT NOT NULL DEFAULT '3',
                assigned_to TEXT DEFAULT '',
                execution_lane TEXT DEFAULT '',
                vs_ref REAL DEFAULT NULL,
                target_vs_ref REAL DEFAULT 1.0,
                tags TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_by TEXT NOT NULL DEFAULT 'ops',
                updated_by TEXT NOT NULL DEFAULT 'ops',
                notes TEXT DEFAULT '',
                spec TEXT DEFAULT '',
                source_file TEXT DEFAULT '',
                factory_mode TEXT DEFAULT '',
                objective_vector TEXT DEFAULT '',
                acceptance_gates TEXT DEFAULT '',
                keep_rule TEXT DEFAULT '',
                benchmark_set TEXT DEFAULT '',
                failure_budget TEXT DEFAULT '',
                crossover_policy TEXT DEFAULT '',
                optimization_scope TEXT DEFAULT '',
                hardware_target TEXT DEFAULT '',
                retarget_policy TEXT DEFAULT '',
                reference_label TEXT DEFAULT ''
            )
        """)
        for col in ['spec', 'source_file', 'version', 'factory_mode', 'execution_lane',
                    'objective_vector', 'acceptance_gates', 'keep_rule', 'benchmark_set',
                    'failure_budget', 'crossover_policy', 'optimization_scope',
                    'hardware_target', 'retarget_policy', 'reference_label']:
            try:
                if col == 'version':
                    c.execute("ALTER TABLE jobs ADD COLUMN version REAL DEFAULT 0")
                else:
                    c.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_phase ON jobs(phase)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_kernel ON jobs(kernel_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_job_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_assigned ON jobs(assigned_to)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_lane ON jobs(execution_lane)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS job_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                from_state TEXT NOT NULL,
                to_state TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                reason TEXT DEFAULT '',
                timestamp TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobtrans_job ON job_transitions(job_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobtrans_time ON job_transitions(timestamp)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER DEFAULT NULL REFERENCES jobs(id),
                from_agent TEXT NOT NULL,
                to_agent TEXT DEFAULT '',
                message_type TEXT NOT NULL DEFAULT 'info',
                subject TEXT NOT NULL,
                body TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                priority TEXT NOT NULL DEFAULT 'normal',
                created_at TEXT NOT NULL,
                resolved_at TEXT DEFAULT '',
                resolved_by TEXT DEFAULT ''
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_job ON messages(job_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_agent)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(message_type)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS watchdog_state (
                name TEXT PRIMARY KEY,
                last_run_at TEXT DEFAULT '',
                last_status TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            )
        """)

        c.commit()

    def close(self):
        self.conn.close()

    # -- Primitives Manifest --

    def ship_primitive(self, source_path: str, shelf_subdir: str = "linalg",
                       vs_ref: float = None, shipped_by: str = "ops",
                       tests_passed: str = "", lint_clean: bool = False) -> dict:
        """Ship a kernel file to the primitives shelf with version tracking.

        Copies the file, computes hash, bumps version, records in DB.
        Returns the shipping record.
        """
        import shutil

        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source not found: {source}")

        name = source.name
        shelf_dir = Path(BWK_ROOT / "common/csrc/primitives" / shelf_subdir)
        shelf_dir.mkdir(parents=True, exist_ok=True)
        dest = shelf_dir / name
        shelf_path = f"{shelf_subdir}/{name}"

        content = source.read_text()
        new_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        has_alpha = "alpha" in content
        has_lda = "lda" in content or "ld_a" in content or "stride" in content.lower()
        has_blas = 1 if (has_alpha and has_lda) else 0

        # Get current version
        row = self.conn.execute(
            "SELECT version, content_hash FROM primitives WHERE shelf_path = ?",
            (shelf_path,)
        ).fetchone()

        if row and row["content_hash"] == new_hash:
            return {"action": "unchanged", "shelf_path": shelf_path,
                    "version": row["version"], "hash": new_hash}

        old_version = row["version"] if row else 0
        new_version = old_version + 1

        # Copy file
        shutil.copy2(str(source), str(dest))

        # Update DB
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute("""
            INSERT INTO primitives (shelf_path, content_hash, version,
                                    has_blas_interface, vs_ref, shipped_from,
                                    shipped_by, shipped_at, tests_passed, lint_clean)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(shelf_path) DO UPDATE SET
                content_hash=excluded.content_hash, version=excluded.version,
                has_blas_interface=excluded.has_blas_interface, vs_ref=excluded.vs_ref,
                shipped_from=excluded.shipped_from, shipped_by=excluded.shipped_by,
                shipped_at=excluded.shipped_at, tests_passed=excluded.tests_passed,
                lint_clean=excluded.lint_clean
        """, (shelf_path, new_hash, new_version, has_blas, vs_ref,
              str(source), shipped_by, now, tests_passed, int(lint_clean)))
        self.conn.commit()

        # Compile the kernel into a .o object file (strip PyTorch binding)
        compile_result = self._compile_primitive(dest)

        return {"action": "shipped", "shelf_path": shelf_path,
                "version": new_version, "hash": new_hash,
                "has_blas_interface": bool(has_blas),
                "compiled": compile_result.get("ok", False)}

    def _compile_primitive(self, cu_path: Path) -> dict:
        """Compile a .cu file into a .o object file for the primitives shelf.

        Strips the PyTorch binding section (everything after '// PyTorch binding')
        and compiles the pure CUDA kernel with -fPIC for linking into .so libraries.
        """
        import subprocess, tempfile

        nvcc = "/usr/local/cuda-13/bin/nvcc"
        if not Path(nvcc).is_file():
            return {"ok": False, "error": "nvcc not found at /usr/local/cuda-13/bin/nvcc"}

        source = cu_path.read_text()
        common_include = str(BWK_ROOT / "common/csrc")
        common_headers = str(BWK_ROOT / "common/csrc/common")

        # Strip PyTorch binding — keep everything above the marker
        marker = "// PyTorch binding"
        if marker in source:
            kernel_source = source[:source.index(marker)]
            # Close any open namespace
            if "namespace" in kernel_source and "} // namespace" not in kernel_source:
                kernel_source += "\n} // namespace\n"
        else:
            kernel_source = source

        # Remove torch includes, replace with minimal CUDA headers
        lines = kernel_source.split('\n')
        cleaned = []
        for line in lines:
            if 'torch/extension.h' in line or 'ATen/' in line or 'pybind' in line.lower():
                continue
            cleaned.append(line)
        kernel_source = '\n'.join(cleaned)

        # Write to temp file, compile
        out_path = cu_path.with_suffix('.o')
        with tempfile.NamedTemporaryFile(suffix='.cu', mode='w', delete=False, dir=str(cu_path.parent)) as tmp:
            tmp.write(kernel_source)
            tmp_path = tmp.name

        try:
            result = subprocess.run([
                nvcc, '-arch=sm_120a', '-O3', '-Xcompiler', '-fPIC',
                '-I', common_include, '-I', common_headers,
                '-c', tmp_path, '-o', str(out_path)
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                return {"ok": True, "output": str(out_path), "size": out_path.stat().st_size}
            else:
                return {"ok": False, "error": result.stderr[:300]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "nvcc timed out (120s)"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def link_primitives_so(output_path: str = None) -> dict:
        """Link all compiled .o files on the shelf into libbwk_primitives.so.
        Deduplicates by filename — linalg/ takes priority over gemm/ for shared names."""
        import subprocess

        shelf = BWK_ROOT / "common/csrc/primitives"
        if output_path is None:
            output_path = str(shelf / "lib" / "libbwk_primitives.so")

        # Collect .o files, deduplicate by filename (linalg/ preferred)
        all_objs = sorted(shelf.rglob("*.o"))
        seen_names = {}
        # Priority: linalg > qr > gemm > everything else
        priority = {"linalg": 0, "qr": 1, "gemm": 2}
        for o in all_objs:
            subdir = o.parent.name
            pri = priority.get(subdir, 10)
            name = o.name
            if name not in seen_names or pri < seen_names[name][1]:
                seen_names[name] = (o, pri)
        obj_files = sorted([v[0] for v in seen_names.values()])
        if not obj_files:
            return {"ok": False, "error": "No .o files found on shelf"}

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        nvcc = "/usr/local/cuda-13/bin/nvcc"

        try:
            result = subprocess.run([
                nvcc, '-arch=sm_120a', '-shared', '-Xcompiler', '-fPIC',
                '-o', output_path,
                *[str(o) for o in obj_files],
                '-lcublas', '-lcusolver',
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                size = Path(output_path).stat().st_size
                return {"ok": True, "output": output_path, "objects": len(obj_files),
                        "size_mb": round(size / 1048576, 2)}
            else:
                return {"ok": False, "error": result.stderr[:300]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_primitives(self, blas_only: bool = False) -> list[dict]:
        """Query the primitives manifest."""
        where = "WHERE has_blas_interface = 1" if blas_only else ""
        rows = self.conn.execute(f"""
            SELECT * FROM primitives {where}
            ORDER BY shelf_path
        """).fetchall()
        return [dict(r) for r in rows]

    def verify_shelf(self) -> list[dict]:
        """Check every file on the shelf matches its manifest hash."""
        results = []
        shelf = Path(BWK_ROOT / "common/csrc/primitives")

        for f in sorted(shelf.rglob("*.cu")):
            rel = str(f.relative_to(shelf))
            file_hash = hashlib.sha256(f.read_text().encode()).hexdigest()[:16]

            row = self.conn.execute(
                "SELECT * FROM primitives WHERE shelf_path = ?", (rel,)
            ).fetchone()

            if not row:
                results.append({"path": rel, "status": "untracked",
                                "file_hash": file_hash})
            elif row["content_hash"] == file_hash:
                results.append({"path": rel, "status": "ok",
                                "version": row["version"], "hash": file_hash})
            else:
                results.append({"path": rel, "status": "modified",
                                "file_hash": file_hash,
                                "manifest_hash": row["content_hash"],
                                "version": row["version"]})

        return results

    def auto_ship_job(self, job_id: int, shipped_by: str = "watchdog") -> list:
        """Ship kernel .cu file(s) for a job. Returns list of ship results.

        If job.source_file is set, ships only that one file (1 job = 1 function).
        If not set, ships all .cu files for the kernel_type (legacy monolith mode).
        """
        job = self.get_job(job_id)
        if not job:
            return []
        kernel = job.get("kernel_type", "")
        if not kernel:
            return []

        # Map kernel_type to worktree path and shelf subdirectory
        worktree_map = {
            "lu": ("lu", "lu"),
            "qr": ("qr", "qr"),
            "gemm": ("gemm", "gemm"),
            "linalg": ("linalg", "linalg"),
            "numerical": ("numerical", "numerical"),
            "spmv": ("spmv", "spmv"),
            "attention": ("main", "attention"),
            "cuquantum": ("cuquantum", "cuquantum"),
            "rmsnorm": ("rmsnorm", "rmsnorm"),
            "dotproduct": ("dotproduct", "dotproduct"),
            "fused_mlp": ("fused-mlp", "fused-mlp"),
            "chess_training": ("chess-training", "chess-training"),
        }

        info = worktree_map.get(kernel)
        if not info:
            return []

        worktree, shelf_sub = info

        # If source_file is set, ship only that one file (1 job = 1 function)
        if job.get("source_file"):
            source = Path(job["source_file"])
            if not source.is_absolute():
                source = BWK_ROOT / worktree / source
            cu_files = [source] if source.is_file() else []
        else:
            # Legacy: ship all .cu files for this kernel_type
            source_dir = BWK_ROOT / worktree / "csrc" / kernel.replace("_", "-")
            if not source_dir.is_dir():
                source_dir = BWK_ROOT / worktree / "csrc" / worktree
            if not source_dir.is_dir():
                source_dir = BWK_ROOT / worktree / "csrc"
            cu_files = list(source_dir.glob("*_sm120*.cu")) if source_dir.is_dir() else []
            if not cu_files:
                cu_files = list(source_dir.glob("*.cu")) if source_dir.is_dir() else []

        results = []

        # Get vs_ref from worker_state
        vs_ref = None
        row = self.conn.execute(
            "SELECT best_vsref FROM worker_state WHERE kernel_type = ?", (kernel,)
        ).fetchone()
        if row:
            vs_ref = row[0]

        for cu in cu_files:
            try:
                result = self.ship_primitive(
                    str(cu), shelf_subdir=shelf_sub,
                    vs_ref=vs_ref, shipped_by=shipped_by
                )
                results.append(result)
            except Exception as e:
                results.append({"action": "error", "file": str(cu), "error": str(e)})

        return results

    # -- Issues --

    def file_issue(self, title: str, severity: str, kernel_type: str,
                   source_file: str, description: str, reproduce: str = "",
                   filed_by: str = "tester") -> int:
        """File a new issue. Returns issue ID."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute("""
            INSERT INTO issues (title, severity, status, kernel_type, source_file,
                                filed_by, description, reproduce, filed_at, updated_at)
            VALUES (?, ?, 'open', ?, ?, ?, ?, ?, ?, ?)
        """, (title, severity, kernel_type, source_file, filed_by,
              description, reproduce, now, now))
        self.conn.commit()
        issue_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return issue_id

    def assign_issue(self, issue_id: int, assigned_to: str):
        """Foreman assigns an issue to a worker."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute(
            "UPDATE issues SET assigned_to = ?, status = 'assigned', updated_at = ? WHERE id = ?",
            (assigned_to, now, issue_id)
        )
        self.conn.commit()

    def rework_issue(self, issue_id: int, fix_description: str):
        """Worker submits a fix for re-testing."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute(
            "UPDATE issues SET fix_description = ?, status = 'retest', updated_at = ? WHERE id = ?",
            (fix_description, now, issue_id)
        )
        self.conn.commit()

    def close_issue(self, issue_id: int):
        """Tester verifies fix and closes the issue."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute(
            "UPDATE issues SET status = 'closed', closed_at = ?, updated_at = ? WHERE id = ?",
            (now, now, issue_id)
        )
        self.conn.commit()

    def reopen_issue(self, issue_id: int, reason: str):
        """Tester reopens if fix didn't work."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute(
            "UPDATE issues SET status = 'open', fix_description = ?, updated_at = ? WHERE id = ?",
            (f"REOPENED: {reason}", now, issue_id)
        )
        self.conn.commit()

    def get_issues(self, status: str = None, kernel_type: str = None) -> list[dict]:
        """Query issues. Sorted: open first, then by severity."""
        where = []
        params = []
        if status:
            where.append("status = ?")
            params.append(status)
        if kernel_type:
            where.append("kernel_type = ?")
            params.append(kernel_type)

        where_sql = ""
        if where:
            where_sql = "WHERE " + " AND ".join(where)

        rows = self.conn.execute(f"""
            SELECT * FROM issues {where_sql}
            ORDER BY
                CASE status
                    WHEN 'open' THEN 1
                    WHEN 'assigned' THEN 2
                    WHEN 'retest' THEN 3
                    WHEN 'closed' THEN 4
                END,
                CASE severity
                    WHEN 'blocking' THEN 1
                    WHEN 'correctness' THEN 2
                    WHEN 'warning' THEN 3
                END,
                filed_at DESC
        """, params).fetchall()
        return [dict(r) for r in rows]

    # -- Jobs (workpiece lifecycle tracking) --

    def create_job(self, name, title, description="", job_type="kernel", kernel_type="",
                   parent_job_id=None, state="wishlist", priority="3", assigned_to="",
                   target_vs_ref=1.0, tags="", created_by="ops", notes="",
                   source_file="") -> int:
        if state not in ALL_JOB_STATES:
            raise ValueError(f"Unknown state '{state}'. Valid: {sorted(ALL_JOB_STATES)}")
        phase = STATE_TO_PHASE[state]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cursor = self.conn.execute("""
            INSERT INTO jobs (name, title, description, job_type, kernel_type,
                              parent_job_id, state, phase, priority, assigned_to,
                              target_vs_ref, tags, created_at, updated_at,
                              created_by, updated_by, notes, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, title, description, job_type, kernel_type,
              parent_job_id, state, phase, priority, assigned_to,
              target_vs_ref, tags, now, now, created_by, created_by, notes, source_file))
        job_id = cursor.lastrowid
        self.conn.execute("""
            INSERT INTO job_transitions (job_id, from_state, to_state, changed_by, reason, timestamp)
            VALUES (?, '', ?, ?, 'created', ?)
        """, (job_id, state, created_by, now))
        self.conn.commit()
        return job_id

    def update_job_state(self, job_id, to_state, changed_by, reason=""):
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job #{job_id} not found")
        from_state = job["state"]
        valid, err = validate_transition(from_state, to_state)
        if not valid:
            raise ValueError(err)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        new_phase = STATE_TO_PHASE[to_state]
        self.conn.execute("UPDATE jobs SET state = ?, phase = ?, updated_at = ?, updated_by = ? WHERE id = ?",
                          (to_state, new_phase, now, changed_by, job_id))
        self.conn.execute("""
            INSERT INTO job_transitions (job_id, from_state, to_state, changed_by, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (job_id, from_state, to_state, changed_by, reason, now))
        self.conn.commit()
        return self.get_job(job_id)

    def update_job(self, job_id, updated_by="ops", **kwargs):
        allowed = {"title", "description", "priority", "assigned_to", "vs_ref",
                    "target_vs_ref", "tags", "notes", "kernel_type", "spec", "source_file"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return self.get_job(job_id)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        updates["updated_at"] = now
        updates["updated_by"] = updated_by
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [job_id]
        self.conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", values)
        self.conn.commit()
        return self.get_job(job_id)

    def get_job(self, job_id):
        row = self.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def get_job_by_name(self, name):
        row = self.conn.execute("SELECT * FROM jobs WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def get_jobs(self, state=None, phase=None, job_type=None, kernel_type=None,
                 assigned_to=None, parent_job_id=None, priority=None):
        where, params = [], []
        if state:
            where.append("state = ?"); params.append(state)
        if phase:
            where.append("phase = ?"); params.append(phase)
        if job_type:
            where.append("job_type = ?"); params.append(job_type)
        if kernel_type:
            where.append("kernel_type = ?"); params.append(kernel_type)
        if assigned_to:
            where.append("assigned_to = ?"); params.append(assigned_to)
        if parent_job_id is not None:
            where.append("parent_job_id = ?"); params.append(parent_job_id)
        if priority:
            where.append("priority = ?"); params.append(priority)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self.conn.execute(f"""
            SELECT * FROM jobs {where_sql}
            ORDER BY CAST(priority AS INTEGER), updated_at DESC
        """, params).fetchall()
        return [dict(r) for r in rows]

    def get_job_history(self, job_id):
        rows = self.conn.execute(
            "SELECT * FROM job_transitions WHERE job_id = ? ORDER BY timestamp ASC",
            (job_id,)).fetchall()
        return [dict(r) for r in rows]

    def sync_job_vsref(self, job_id):
        job = self.get_job(job_id)
        if not job or not job.get("kernel_type"):
            return job
        row = self.conn.execute(
            "SELECT best_vsref FROM worker_state WHERE kernel_type = ?",
            (job["kernel_type"],)).fetchone()
        if row and row[0] is not None:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.conn.execute("UPDATE jobs SET vs_ref = ?, updated_at = ? WHERE id = ?",
                              (row[0], now, job_id))
            self.conn.commit()
        return self.get_job(job_id)

    # -- Messages --

    def create_message(self, from_agent, subject, body="", to_agent="",
                       job_id=None, message_type="info", priority="normal"):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cursor = self.conn.execute("""
            INSERT INTO messages (job_id, from_agent, to_agent, message_type,
                                  subject, body, status, priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)
        """, (job_id, from_agent, to_agent, message_type, subject, body, priority, now))
        self.conn.commit()
        return cursor.lastrowid

    def acknowledge_message(self, message_id, by="foreman"):
        self.conn.execute("UPDATE messages SET status = 'acknowledged' WHERE id = ?", (message_id,))
        self.conn.commit()
        return self.get_message(message_id)

    def resolve_message(self, message_id, by="foreman"):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute(
            "UPDATE messages SET status = 'resolved', resolved_at = ?, resolved_by = ? WHERE id = ?",
            (now, by, message_id))
        self.conn.commit()
        return self.get_message(message_id)

    def get_message(self, message_id):
        row = self.conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return dict(row) if row else None

    def get_messages(self, status=None, job_id=None, from_agent=None,
                     to_agent=None, message_type=None):
        where, params = [], []
        if status:
            where.append("status = ?"); params.append(status)
        if job_id is not None:
            where.append("job_id = ?"); params.append(job_id)
        if from_agent:
            where.append("from_agent = ?"); params.append(from_agent)
        if to_agent:
            where.append("to_agent = ?"); params.append(to_agent)
        if message_type:
            where.append("message_type = ?"); params.append(message_type)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self.conn.execute(f"""
            SELECT * FROM messages {where_sql}
            ORDER BY
                CASE status WHEN 'open' THEN 1 WHEN 'acknowledged' THEN 2 ELSE 3 END,
                CASE priority WHEN 'urgent' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
                created_at DESC
        """, params).fetchall()
        return [dict(r) for r in rows]

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
        if provenance == "research" and prov_hint and prov_hint != "research":
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

        # Check if already indexed with same hash
        existing = self.conn.execute(
            "SELECT id, content_hash, chunk_count FROM documents WHERE source_file = ?",
            (tsv_path,)
        ).fetchone()
        content_changed = bool(existing and existing["content_hash"] != content_hash)

        if existing and existing["content_hash"] == content_hash and not force:
            return {"files": 0, "chunks": 0, "skipped": 1}

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

        def _to_float(value):
            text = str(value or "").strip()
            if not text or text == "-":
                return None
            try:
                return float(text)
            except ValueError:
                return None

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

            raw_chunks.append({
                "text": chunk_text,
                "heading": heading,
                "position": i,
                "stalls": stall_str,
                "techniques": techs,
                "status": status,
                "timestamp": timestamp,
                "git_commit": commit,
                "duration_us": _to_float(duration),
                "vs_ref": _to_float(vs_ref),
                "sm_pct": _to_float(sm_pct),
                "stall_math": _to_float(row.get("stall_math")),
                "stall_wait": _to_float(row.get("stall_wait")),
                "stall_scoreboard": _to_float(row.get("stall_scoreboard")),
                "stall_barrier": _to_float(row.get("stall_barrier")),
                "top_stall": top_stall if top_stall not in {"", "-", "none"} else "",
                "extra": {
                    key: value
                    for key, value in row.items()
                    if key not in {"timestamp", "commit", "duration_us", "vs_ref", "sm_pct",
                                   "stall_math", "stall_wait", "stall_scoreboard",
                                   "stall_barrier", "top_stall", "status", "description"}
                },
            })

        if not raw_chunks:
            return {"files": 0, "chunks": 0, "skipped": 1}

        self.conn.execute(
            "DELETE FROM experiments WHERE source_type = 'tsv' AND source_path = ?",
            (tsv_path,)
        )

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

        if content_changed:
            self.conn.execute(
                "UPDATE documents SET summary = '', signal = '', has_summary = 0 WHERE id = ?",
                (doc_id,)
            )
            self.conn.execute("DELETE FROM vec_summaries WHERE doc_id = ?", (doc_id,))

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
            self.record_experiment(
                kernel_type=kernel_type,
                status=chunk["status"],
                description=chunk["text"],
                timestamp=chunk["timestamp"],
                git_commit=chunk["git_commit"],
                duration_us=chunk["duration_us"],
                vs_ref=chunk["vs_ref"],
                sm_pct=chunk["sm_pct"],
                stall_math=chunk["stall_math"],
                stall_wait=chunk["stall_wait"],
                stall_scoreboard=chunk["stall_scoreboard"],
                stall_barrier=chunk["stall_barrier"],
                top_stall=chunk["top_stall"],
                source_type="tsv",
                source_path=tsv_path,
                experiment_index=chunk["position"] + 1,
                extra=chunk["extra"],
            )

        self.conn.commit()
        return {"files": 1, "chunks": len(raw_chunks), "skipped": 0,
                "kept": kept_count, "discarded": discarded_count}

    def ingest_all_tsv(self, force: bool = False) -> dict:
        """Ingest all experiment TSV files from known project directories."""
        tsv_dirs = [
            BWK_ROOT / "main/results",
            BWK_ROOT / "gemm/results",
            BWK_ROOT / "fused-mlp/results",
            BWK_ROOT / "attention/results",
            BWK_ROOT / "dotproduct/results",
            BWK_ROOT / "linalg/results",
            BWK_ROOT / "lu/results",
            BWK_ROOT / "qr/results",
            BWK_ROOT / "rmsnorm/results",
            BWK_ROOT / "spmv/results",
            BWK_ROOT / "numerical/results",
            BWK_ROOT / "cuquantum/results",
            BWK_ROOT / "chess-training/results",
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

    def worker_heartbeat(self, kernel_type: str, current_task: str = "",
                         process_state: str = "working", job_id: int = None):
        """Worker calls this to report it's alive and what it's doing.

        Workers self-report: heartbeat (alive), current_task, process_state (working/complete).
        The system computes: stuck (from discard streaks), idle (from stale heartbeat).
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute("""
            UPDATE worker_state SET heartbeat_at = ?, current_task = ?,
                process_state = ?, job_id = ?, updated_at = ?
            WHERE kernel_type = ?
        """, (now, current_task[:200], process_state, job_id, now, kernel_type))
        if self.conn.total_changes == 0:
            # Row doesn't exist yet — insert minimal record
            self.conn.execute("""
                INSERT OR IGNORE INTO worker_state (kernel_type, heartbeat_at,
                    current_task, process_state, job_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (kernel_type, now, current_task[:200], process_state, job_id, now))
        self.conn.commit()

    def refresh_worker_state(self) -> dict:
        """Compute worker state from TSV data + heartbeat signals.

        Hybrid approach:
          - Self-reported: heartbeat_at, current_task, process_state (working/complete)
          - Computed from TSV: stuck (discard streaks), idle (stale heartbeat)
          - 'producing': last 3 experiments include a keep
          - 'grinding':  5+ consecutive discards (spinning but not halted)
          - 'stalled':   10+ consecutive discards or no experiments in 24h
          - 'halted':    halt note exists in for_foreman-claude/
          - 'idle':      no TSV file or empty
          - 'converged': last 3 keeps are within 2% of each other
        """
        import csv

        tsv_dirs = {
            "attention": BWK_ROOT / "main/results/attention.tsv",
            "gemm": BWK_ROOT / "gemm/results/gemm.tsv",
            "fused_mlp": BWK_ROOT / "fused-mlp/results/mlp.tsv",
            "dotproduct": BWK_ROOT / "dotproduct/results/dotproduct.tsv",
            "linalg": BWK_ROOT / "linalg/results/linalg.tsv",
            "lu": BWK_ROOT / "lu/results/lu.tsv",
            "qr": BWK_ROOT / "qr/results/qr.tsv",
            "rmsnorm": BWK_ROOT / "rmsnorm/results/rmsnorm.tsv",
            "spmv": BWK_ROOT / "spmv/results/spmv.tsv",
            "numerical": BWK_ROOT / "numerical/results/numerical.tsv",
            "cuquantum": BWK_ROOT / "cuquantum/results/cuquantum.tsv",
            "chess_training": BWK_ROOT / "chess-training/results/chess-training.tsv",
        }

        # Preserve existing heartbeat/process_state data before refresh
        existing = {}
        for row in self.conn.execute(
            "SELECT kernel_type, heartbeat_at, current_task, job_id, process_state FROM worker_state"
        ).fetchall():
            existing[row[0]] = dict(row)

        results = {}

        for kernel, tsv_path in tsv_dirs.items():
            prior = existing.get(kernel, {})
            state = {
                "kernel_type": kernel,
                "tsv_path": str(tsv_path),
                "total_experiments": 0,
                "kept": 0,
                "discarded": 0,
                "best_vsref": None,
                "best_duration_us": None,
                "top_stall": "",
                "current_discard_streak": 0,
                "max_discard_streak": 0,
                "last_kept_description": "",
                "last_experiment_time": "",
                "has_halt_note": 0,
                "status": "idle",
                "diagnosis": "no TSV data",
                "heartbeat_at": prior.get("heartbeat_at", ""),
                "current_task": prior.get("current_task", ""),
                "job_id": prior.get("job_id"),
                "process_state": prior.get("process_state", ""),
            }

            # Parse TSV
            if not tsv_path.is_file():
                results[kernel] = state
                continue

            try:
                with open(tsv_path) as f:
                    rows = list(csv.DictReader(f, delimiter="\t"))
            except Exception:
                results[kernel] = state
                continue

            if not rows:
                results[kernel] = state
                continue

            # Compute metrics
            kept_rows = []
            streak = 0
            max_streak = 0

            for row in rows:
                status = row.get("status", "").strip().lower()
                if status in ("keep", "kept"):
                    kept_rows.append(row)
                    streak = 0
                elif status in ("discard", "discarded"):
                    streak += 1
                    max_streak = max(max_streak, streak)

            # Current tail discard streak
            tail_streak = 0
            for row in reversed(rows):
                if row.get("status", "").strip().lower() in ("discard", "discarded"):
                    tail_streak += 1
                else:
                    break

            # Best vs_ref from kept rows
            best_vsref = None
            best_duration = None
            for row in kept_rows:
                try:
                    vr = float(row.get("vs_ref", "0"))
                    if best_vsref is None or vr > best_vsref:
                        best_vsref = vr
                except (ValueError, TypeError):
                    pass
                try:
                    dur = float(row.get("duration_us", "0"))
                    if best_duration is None or dur < best_duration:
                        best_duration = dur
                except (ValueError, TypeError):
                    pass

            # Top stall from most recent experiment
            last_row = rows[-1]
            top_stall = last_row.get("top_stall", "").strip()
            if top_stall in ("-", "none", ""):
                top_stall = ""

            # Last kept description
            last_kept_desc = ""
            if kept_rows:
                last_kept_desc = kept_rows[-1].get("description", "").strip()

            # Timestamp from last row
            last_time = last_row.get("timestamp", "").strip()
            if not last_time:
                # Fall back to file modification time
                last_time = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(os.path.getmtime(str(tsv_path))))

            # Determine status (hybrid: self-reported + computed)
            # 1. Worker self-reported "complete" → trust it
            # 2. Heartbeat stale (>30 min) → dead/idle (computed)
            # 3. Discard streaks → stuck/grinding (computed, worker can't see this)
            # 4. Otherwise → producing
            heartbeat_stale = False
            if state.get("heartbeat_at"):
                try:
                    import calendar
                    hb_time = calendar.timegm(time.strptime(state["heartbeat_at"], "%Y-%m-%dT%H:%M:%SZ"))
                    heartbeat_stale = (time.time() - hb_time) > 1800  # 30 min
                except (ValueError, OverflowError):
                    pass

            if state.get("process_state") == "complete":
                status = "complete"
                diagnosis = f"worker self-reported complete. task: {state.get('current_task', '')[:60]}"
            elif heartbeat_stale and state.get("heartbeat_at"):
                status = "idle"
                diagnosis = f"heartbeat stale (last: {state['heartbeat_at']})"
            elif tail_streak >= 10:
                status = "stalled"
                diagnosis = f"{tail_streak} consecutive discards — likely exhausted current approach"
            elif tail_streak >= 5:
                status = "grinding"
                diagnosis = f"{tail_streak} consecutive discards — spinning without progress"
            elif len(kept_rows) >= 3:
                # Check convergence: last 3 keeps within 2%
                recent_vsrefs = []
                for kr in kept_rows[-3:]:
                    try:
                        recent_vsrefs.append(float(kr.get("vs_ref", "0")))
                    except (ValueError, TypeError):
                        pass
                if len(recent_vsrefs) == 3:
                    spread = max(recent_vsrefs) - min(recent_vsrefs)
                    avg = sum(recent_vsrefs) / 3
                    if avg > 0 and spread / avg < 0.02:
                        status = "converged"
                        diagnosis = f"last 3 keeps within 2% ({min(recent_vsrefs):.2f}–{max(recent_vsrefs):.2f}x)"
                    else:
                        status = "producing"
                        diagnosis = f"last keep: {last_kept_desc[:80]}"
                else:
                    status = "producing"
                    diagnosis = f"last keep: {last_kept_desc[:80]}"
            else:
                status = "producing"
                diagnosis = f"{len(kept_rows)} kept so far, tail streak={tail_streak}"

            state.update({
                "total_experiments": len(rows),
                "kept": len(kept_rows),
                "discarded": len(rows) - len(kept_rows),
                "best_vsref": best_vsref,
                "best_duration_us": best_duration,
                "top_stall": top_stall,
                "current_discard_streak": tail_streak,
                "max_discard_streak": max_streak,
                "last_kept_description": last_kept_desc[:200],
                "last_experiment_time": last_time,
                "status": status,
                "diagnosis": diagnosis,
            })

            results[kernel] = state

        # Write to DB
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for kernel, s in results.items():
            self.conn.execute("""
                INSERT INTO worker_state (
                    kernel_type, tsv_path, total_experiments, kept, discarded,
                    best_vsref, best_duration_us, top_stall,
                    current_discard_streak, max_discard_streak,
                    last_kept_description, last_experiment_time,
                    has_halt_note, status, diagnosis, updated_at,
                    heartbeat_at, current_task, job_id, process_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(kernel_type) DO UPDATE SET
                    tsv_path=excluded.tsv_path,
                    total_experiments=excluded.total_experiments,
                    kept=excluded.kept, discarded=excluded.discarded,
                    best_vsref=excluded.best_vsref,
                    best_duration_us=excluded.best_duration_us,
                    top_stall=excluded.top_stall,
                    current_discard_streak=excluded.current_discard_streak,
                    max_discard_streak=excluded.max_discard_streak,
                    last_kept_description=excluded.last_kept_description,
                    last_experiment_time=excluded.last_experiment_time,
                    has_halt_note=excluded.has_halt_note,
                    status=excluded.status,
                    diagnosis=excluded.diagnosis,
                    updated_at=excluded.updated_at
            """, (kernel, s["tsv_path"], s["total_experiments"], s["kept"],
                  s["discarded"], s["best_vsref"], s["best_duration_us"],
                  s["top_stall"], s["current_discard_streak"],
                  s["max_discard_streak"], s["last_kept_description"],
                  s["last_experiment_time"], s["has_halt_note"],
                  s["status"], s["diagnosis"], now,
                  s.get("heartbeat_at", ""), s.get("current_task", ""),
                  s.get("job_id"), s.get("process_state", "")))

        self.conn.commit()
        return results

    def get_worker_state(self, kernel_type: str = None) -> list[dict]:
        """Query worker state. If kernel_type is None, returns all workers."""
        if kernel_type:
            rows = self.conn.execute(
                "SELECT * FROM worker_state WHERE kernel_type = ?", (kernel_type,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM worker_state ORDER BY "
                "CASE status "
                "  WHEN 'stalled' THEN 1 "
                "  WHEN 'grinding' THEN 2 "
                "  WHEN 'halted' THEN 3 "
                "  WHEN 'producing' THEN 4 "
                "  WHEN 'converged' THEN 5 "
                "  ELSE 6 END, "
                "current_discard_streak DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def _remove_document(self, doc_id: int):
        """Remove a document and all its chunks/vectors."""
        chunk_ids = [r["id"] for r in self.conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()]
        for cid in chunk_ids:
            self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

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
        vec = embed_query(query)

        rows = self.conn.execute(
            """
            SELECT doc_id, distance
            FROM vec_summaries
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (serialize_f32(vec), k * 5)
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
            boost = PROVENANCE_TIERS.get(r["provenance"], {}).get("boost", 1.0)
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
        vec = embed_query(query)

        # Over-fetch for post-filtering
        rows = self.conn.execute(
            """
            SELECT chunk_id, distance
            FROM vec_chunks
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (serialize_f32(vec), k * 5)
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
            boost = PROVENANCE_TIERS.get(prov, {}).get("boost", 1.0)
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
                   doc_type: str = None) -> list[dict]:
        """Full-text search using FTS5 with provenance boost."""
        safe_query = self._sanitize_fts_query(query)
        where_clauses = []
        params = [safe_query]

        if kernel_type:
            where_clauses.append("d.kernel_type = ?")
            params.append(kernel_type)
        if doc_type:
            where_clauses.append("d.doc_type = ?")
            params.append(doc_type)

        extra_where = ""
        if where_clauses:
            extra_where = "AND " + " AND ".join(where_clauses)

        params.append(k * 3)  # over-fetch for re-ranking

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

        # Re-rank with provenance boost
        output = []
        for r in results:
            raw_rank = r["fts_rank"]
            boost = PROVENANCE_TIERS.get(r["provenance"], {}).get("boost", 1.0)
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
            query, k=k*2, kernel_type=kernel_type, doc_type=doc_type
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
            boost = PROVENANCE_TIERS.get(tier, {}).get("boost", 1.0)
            desc = PROVENANCE_TIERS.get(tier, {}).get("description", "")
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

    # -- Helpers --

    def _extract_title(self, content: str, filepath: str) -> str:
        """Extract title from markdown heading or filename."""
        match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return Path(filepath).stem.replace("_", " ").replace("-", " ").title()

    def add_source(self, path: str, doc_type: str = "research",
                   pattern: str = "**/*.md"):
        """Add a new ingest source at runtime."""
        SOURCE_PRIORITY.append((Path(path), doc_type, "research", 50))
        print(f"Added source: {path} ({doc_type}, {pattern})", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        # Compact: one line per result
        signal = r.get("signal", r.get("title", ""))
        lines.append(f"\033[1m[{index}]\033[0m {prov_badge} {signal} "
                     f"\033[2m({r['kernel_type']}){dist_str}\033[0m{emp}")
        return "\n".join(lines)

    # Header (levels 0, 2, 3)
    lines.append(f"\033[1m[{index}]\033[0m {prov_badge} {r['title']} "
                 f"\033[2m({r['kernel_type']}/{r['doc_type']}){dist_str}\033[0m{emp}")
    lines.append(f"    \033[36m{r['source_file']}\033[0m")

    if level >= 2 and r.get("summary"):
        # Show the structured summary
        lines.append(f"    \033[33m--- Summary ---\033[0m")
        for sline in r["summary"].split("\n"):
            lines.append(f"    {sline}")
    elif level == 0:
        # Legacy: show chunk snippet
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
        # Level 2 but no summary yet
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
                lines.append(f"    ...")
            lines.append("")

    if verbose:
        text = r.get("text", r.get("summary", ""))
        lines.append(f"    \033[2m[provenance: {prov} | doc_id: {r.get('doc_id', r.get('chunk_id', ''))}]\033[0m")

    return "\n".join(lines)


def cmd_ingest(args):
    mem = ResearchMemory(args.db)
    if args.path:
        p = Path(args.path)
        if p.is_file():
            if p.suffix == ".tsv":
                stats = mem.ingest_tsv(str(p), force=args.force)
                print(f"Ingested: {stats['chunks']} experiments from {p} "
                      f"({stats.get('kept', 0)} kept, {stats.get('discarded', 0)} discarded)")
            else:
                n = mem.ingest_file(str(p), doc_type=args.type, force=args.force)
                print(f"Ingested: {n} chunks from {p}")
        elif p.is_dir():
            stats = mem.ingest_directory(str(p), args.type, args.pattern, args.force)
            print(f"Ingested: {stats['files']} files, {stats['chunks']} chunks "
                  f"({stats['skipped']} unchanged)")
        else:
            print(f"Not found: {p}", file=sys.stderr)
            sys.exit(1)
    else:
        stats = mem.ingest_all(force=args.force)
        print(f"\nDocs: {stats['files']} files, {stats['chunks']} chunks "
              f"({stats['skipped']} unchanged, {stats['dedup_skipped']} duplicates eliminated)")
        # Also ingest experiments
        print("\nIndexing experiment results...", file=sys.stderr)
        tsv_stats = mem.ingest_all_tsv(force=args.force)
        print(f"\nExperiments: {tsv_stats['files']} TSV files, {tsv_stats['chunks']} rows "
              f"({tsv_stats.get('kept', 0)} kept, {tsv_stats.get('discarded', 0)} discarded)")
    mem.close()


def cmd_search(args):
    mem = ResearchMemory(args.db)
    query = " ".join(args.query)
    level = getattr(args, 'level', 0)

    # Tiered search (default when summaries exist)
    if level > 0:
        results = mem.search_summaries(
            query, k=args.k, level=level,
            kernel_type=args.kernel, doc_type=args.type,
            stall_type=args.stall, technique=args.technique,
            provenance=args.provenance
        )
    elif args.mode == "fts":
        results = mem.search_fts(
            query, k=args.k,
            kernel_type=args.kernel, doc_type=args.type
        )
    elif args.mode == "semantic":
        results = mem.search_semantic(
            query, k=args.k,
            kernel_type=args.kernel, doc_type=args.type,
            stall_type=args.stall, technique=args.technique,
            provenance=args.provenance
        )
    else:  # hybrid — use tiered if summaries available, else fall back
        # Check if summaries exist
        has_summaries = mem.conn.execute(
            "SELECT COUNT(*) FROM vec_summaries"
        ).fetchone()[0]
        if has_summaries > 0:
            results = mem.search_summaries(
                query, k=args.k, level=2,
                kernel_type=args.kernel, doc_type=args.type,
                stall_type=args.stall, technique=args.technique,
                provenance=args.provenance
            )
        else:
            results = mem.search_hybrid(
                query, k=args.k,
                kernel_type=args.kernel, doc_type=args.type,
                stall_type=args.stall, technique=args.technique,
                provenance=args.provenance
            )

    if not results:
        print("No results found.")
    else:
        print(f"\n{len(results)} results for: {query}\n")
        for i, r in enumerate(results, 1):
            print(format_result(r, i, verbose=args.verbose, level=level))
            print()

    mem.close()


def cmd_stats(args):
    mem = ResearchMemory(args.db)
    s = mem.stats()
    print(f"Database: {args.db}")
    print(f"Size: {s['db_size_mb']} MB")
    print(f"Documents: {s['documents']}")
    print(f"Chunks: {s['chunks']}")
    print(f"\nBy provenance:")
    for t, n in s["by_provenance"].items():
        boost = PROVENANCE_TIERS.get(t, {}).get("boost", 1.0)
        print(f"  {t}: {n} (boost: {boost}x)")
    print(f"\nBy document type:")
    for t, n in sorted(s["by_doc_type"].items()):
        print(f"  {t}: {n}")
    print(f"\nBy kernel type:")
    for t, n in s["by_kernel_type"].items():
        print(f"  {t}: {n}")
    print(f"\nEmpirically-backed: {s['empirical_docs']}")
    print(f"Summaries (Level 2): {s['docs_with_summary']}/{s['documents']} "
          f"({100*s['docs_with_summary']/max(s['documents'],1):.0f}%)")
    print(f"Chunk avg length: {s['chunk_avg_len']} chars")
    print(f"Chunks with stall tags: {s['chunks_with_stalls']}")
    print(f"Chunks with technique tags: {s['chunks_with_techniques']}")
    mem.close()


def cmd_quality(args):
    mem = ResearchMemory(args.db)
    print(mem.quality_report())
    mem.close()



from common.memory import memory_maintain



def cmd_serve(args):
    from common.memory import memory_server
    mem = ResearchMemory(args.db)
    memory_server.serve(mem, args.port)


from common.memory.memory_helpers import format_result, format_quality_result, attach_helper_methods
ResearchMemory.PROVENANCE_TIERS = PROVENANCE_TIERS
ResearchMemory.embed_query = staticmethod(embed_query)
ResearchMemory.serialize_f32 = staticmethod(serialize_f32)
ResearchMemory.validate_transition = staticmethod(validate_transition)
ResearchMemory.ALL_JOB_STATES = ALL_JOB_STATES
ResearchMemory.STATE_TO_PHASE = STATE_TO_PHASE
ResearchMemory.FACTORY_MODES = FACTORY_MODES
ResearchMemory.OPTIMIZATION_SCOPES = OPTIMIZATION_SCOPES
ResearchMemory.EXECUTION_LANES = EXECUTION_LANES
ResearchMemory.JOB_TYPES = JOB_TYPES
ResearchMemory.JOB_PRIORITIES = JOB_PRIORITIES
ResearchMemory.MESSAGE_TYPES = MESSAGE_TYPES
ResearchMemory.MESSAGE_STATUSES = MESSAGE_STATUSES
ResearchMemory.MESSAGE_PRIORITIES = MESSAGE_PRIORITIES
ResearchMemory.search_summaries = _mem_search.search_summaries
ResearchMemory.search_semantic = _mem_search.search_semantic
ResearchMemory.search_fts = _mem_search.search_fts
ResearchMemory.search_hybrid = _mem_search.search_hybrid
ResearchMemory._sanitize_fts_query = staticmethod(_mem_search._sanitize_fts_query)
_mem_messages.attach_message_methods(ResearchMemory)
_mem_workers.attach_worker_methods(ResearchMemory)
_mem_jobs.attach_job_methods(ResearchMemory)
_mem_exps.attach_experiment_methods(ResearchMemory)
_mem_stats.attach_stats_methods(ResearchMemory)
_mem_issues.attach_issue_methods(ResearchMemory)
attach_helper_methods(ResearchMemory)


def main(argv: list[str] | None = None) -> int:
    from common.memory import memory_cli

    parser = memory_cli.build_parser(default_db=str(DB_PATH))
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    memory_cli.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
