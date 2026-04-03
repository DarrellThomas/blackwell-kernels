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

import sqlite_vec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_DIR = Path(__file__).parent
DB_PATH = DB_DIR / "research.db"
BWK_ROOT = Path(__file__).resolve().parents[2]

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

SOURCE_PRIORITY = [
    # Tier 1 — Validated (approved by foreman, backed by experiments)
    (BWK_ROOT / "foreman-staff/researcher/approved", "research", "validated", 10),
    (BWK_ROOT / "common/docs", "reference", "reference", 12),
    (BWK_ROOT / "common/claude", "reference", "reference", 13),
    # Tier 2 — Research (active briefs, curated cache)
    (BWK_ROOT / "foreman-staff/researcher/archive", "research", "research", 20),
    (BWK_ROOT / "foreman-staff/researcher/cache", "research", "research", 25),
    (BWK_ROOT / "foreman-staff/researcher/inbox", "research", "research", 30),
    # Tier 3 — Worker docs (may have local copies of shared docs)
    (BWK_ROOT / "main/docs", "research", "research", 40),
    (BWK_ROOT / "gemm/docs", "research", "research", 40),
    (BWK_ROOT / "fused-mlp/docs", "research", "research", 40),
    (BWK_ROOT / "attention/docs", "research", "research", 40),
    (BWK_ROOT / "dotproduct/docs", "research", "research", 40),
    (BWK_ROOT / "linalg/docs", "research", "research", 40),
    (BWK_ROOT / "lu/docs", "research", "research", 40),
    (BWK_ROOT / "qr/docs", "research", "research", 40),
    (BWK_ROOT / "rmsnorm/docs", "research", "research", 40),
    (BWK_ROOT / "spmv/docs", "research", "research", 40),
    (BWK_ROOT / "numerical/docs", "research", "research", 40),
    (BWK_ROOT / "cuquantum/docs", "research", "research", 40),
    (BWK_ROOT / "chess-training/docs", "research", "research", 40),
    (BWK_ROOT / "octave-gpu/docs", "research", "research", 40),
    (BWK_ROOT / "ui/docs", "research", "research", 40),
    # Tier 3b — Worker .claude/ dirs (hard-won lessons, specs, feedback)
    (BWK_ROOT / "main/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "gemm/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "fused-mlp/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "attention/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "dotproduct/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "linalg/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "lu/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "qr/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "rmsnorm/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "spmv/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "numerical/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "cuquantum/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "chess-training/.claude", "agent_state", "validated", 45),
    (BWK_ROOT / "octave-gpu/.claude", "agent_state", "validated", 45),
    # Tier 4 — Archive (historical snapshots, may be stale)
]

# Source code: NOT indexed. Git is the authority for .cu/.cuh files.
# Workers search code via grep/glob, not the vector DB.
CODE_SOURCES = []

# ---------------------------------------------------------------------------
# Provenance tier definitions — used for search ranking
# ---------------------------------------------------------------------------

PROVENANCE_TIERS = {
    "validated": {"boost": 1.5, "description": "Foreman-approved, empirically backed"},
    "reference": {"boost": 1.3, "description": "Shared reference docs, manuals"},
    "research":  {"boost": 1.0, "description": "Active research briefs"},
    "archive":   {"boost": 0.7, "description": "Historical snapshots, may be stale"},
}

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

JOB_PHASES = {
    'ideation':    ['wishlist', 'planning'],
    'development': ['not_started', 'algo_building', 'algo_optimizing', 'hw_optimizing',
                    'stuck_needs_research', 'research_available'],
    'validation':  ['compiles_ok', 'tests_writing', 'testing', 'testing_pass', 'testing_fail',
                    'edge_testing', 'edge_pass', 'edge_fail'],
    'rework':      ['rework', 'rework_complete', 'retesting', 'retest_pass', 'retest_fail'],
    'quality':     ['linting', 'lint_pass', 'lint_fail'],
    'shipping':    ['ready_to_ship', 'shipping', 'shipped'],
    'terminal':    ['converged', 'parked', 'abandoned'],
}

PHASE_ORDER = ['ideation', 'development', 'validation', 'rework', 'quality', 'shipping', 'terminal']

STATE_TO_PHASE = {}
for _phase, _states in JOB_PHASES.items():
    for _state in _states:
        STATE_TO_PHASE[_state] = _phase

ALL_JOB_STATES = set(STATE_TO_PHASE.keys())

JOB_TYPES = {'kernel', 'algorithm', 'infrastructure', 'research'}
JOB_PRIORITIES = {'1', '2', '3', '4', '5'}
FACTORY_MODES = {
    'fixed_shape_kernel',
    'general_shape_library',
    'numerical_method',
    'alternative_arithmetic',
    'research_exploration',
}
OPTIMIZATION_SCOPES = {
    'algorithmic',
    'hardware_tuned',
    'hybrid',
}
EXECUTION_LANES = {'active', 'hopper', 'incubating', 'parked'}
MESSAGE_TYPES = {'halt', 'blocker', 'question', 'feedback', 'info', 'directive'}
MESSAGE_STATUSES = {'open', 'acknowledged', 'resolved'}
MESSAGE_PRIORITIES = {'urgent', 'normal', 'low'}

KERNEL_WORKTREE_MAP = {
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

OCTAVE_GPU_ASSIGNEES = {'octave-gpu', 'cx1', 'cx2', 'cx3', 'cx4'}
KNOWN_REPO_ROOTS = (
    "common",
    "octave-gpu",
    *dict.fromkeys(worktree for worktree, _ in KERNEL_WORKTREE_MAP.values()),
)


def _phase_index(phase: str) -> int:
    return PHASE_ORDER.index(phase)


def validate_transition(from_state: str, to_state: str) -> tuple:
    """Check if a state transition is legal. Returns (is_valid, error_message).

    Rules:
      - Before 'shipped': move freely in any direction. Development is iterative.
      - 'shipped' is the hard line. Versioned. Once shipped, only → terminal.
      - Terminal states can reactivate back to development.
      - 'shipped' cannot go backward. To change shipped work, create a new job.
    """
    if from_state == to_state:
        return False, f"Already in state '{from_state}'"
    if from_state not in STATE_TO_PHASE:
        return False, f"Unknown source state '{from_state}'"
    if to_state not in STATE_TO_PHASE:
        return False, f"Unknown target state '{to_state}'"

    # Shipped can go to terminal OR back to development/validation for rework.
    # Going backward from shipped triggers a version bump (handled by update_job).
    if from_state == 'shipped':
        to_phase = STATE_TO_PHASE.get(to_state, '')
        if to_phase not in ('development', 'validation', 'rework', 'terminal'):
            return False, (
                f"From 'shipped', can go to development/validation/rework (version bump) "
                f"or terminal. Cannot go to '{to_state}' ({to_phase})."
            )

    # Terminal can reactivate to development
    if from_state in ('converged', 'parked', 'abandoned'):
        return True, ""

    # Everything before shipped: move freely
    return True, ""


def get_kernel_worktree_info(kernel_type: str):
    return KERNEL_WORKTREE_MAP.get((kernel_type or "").strip())


def _repo_root_for_path(path: Path) -> Optional[Path]:
    try:
        resolved = path.resolve()
    except OSError:
        return None
    for repo_name in KNOWN_REPO_ROOTS:
        candidate = BWK_ROOT / repo_name
        try:
            resolved.relative_to(candidate)
            return candidate
        except ValueError:
            continue
    return None


def resolve_job_source_path(job) -> Optional[Path]:
    source_file = (job.get("source_file") or "").strip()
    if not source_file:
        return None

    raw = Path(source_file).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        info = get_kernel_worktree_info(job.get("kernel_type", ""))
        if info:
            candidates.append(BWK_ROOT / info[0] / raw)
        candidates.append(BWK_ROOT / raw)
        assigned = (job.get("assigned_to") or "").strip()
        if assigned in OCTAVE_GPU_ASSIGNEES:
            candidates.append(BWK_ROOT / "octave-gpu" / raw)

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved

    if candidates:
        try:
            return candidates[0].resolve()
        except OSError:
            return candidates[0]
    return None


def resolve_job_project_dir(job) -> Optional[Path]:
    source_path = resolve_job_source_path(job)
    if source_path is not None:
        repo_root = _repo_root_for_path(source_path)
        if repo_root and repo_root.is_dir():
            return repo_root

    assigned = (job.get("assigned_to") or "").strip()
    if assigned in OCTAVE_GPU_ASSIGNEES:
        candidate = BWK_ROOT / "octave-gpu"
        if candidate.is_dir():
            return candidate

    info = get_kernel_worktree_info(job.get("kernel_type", ""))
    if info:
        candidate = BWK_ROOT / info[0]
        if candidate.is_dir():
            return candidate

    kernel = (job.get("kernel_type") or "").strip()
    if kernel:
        candidate = BWK_ROOT / kernel
        if candidate.is_dir():
            return candidate

    return None


def describe_job_shipping(job) -> dict:
    job_type = (job.get("job_type") or "kernel").strip() or "kernel"
    project_dir = resolve_job_project_dir(job)

    if job_type != "kernel":
        return {
            "ok": True,
            "mode": "metadata_only",
            "detail": f"{job_type} job; ship via repo metadata only",
            "project_dir": project_dir,
        }

    info = get_kernel_worktree_info(job.get("kernel_type", ""))
    if not info:
        return {
            "ok": False,
            "mode": "primitive",
            "error": f"Kernel job requires a recognized kernel_type; got '{job.get('kernel_type', '')}'.",
            "project_dir": project_dir,
        }

    worktree, shelf_subdir = info
    kernel_root = BWK_ROOT / worktree
    source_file = (job.get("source_file") or "").strip()
    if not source_file:
        return {
            "ok": False,
            "mode": "primitive",
            "error": (
                "Kernel primitive shipping now requires source_file. "
                "Split Octave wrapper/integration work into an algorithm job, "
                "and point the kernel job at exactly one .cu primitive source."
            ),
            "project_dir": project_dir or kernel_root,
        }

    source_path = resolve_job_source_path(job)
    if source_path is None:
        return {
            "ok": False,
            "mode": "primitive",
            "error": f"Kernel source_file could not be resolved: {source_file}",
            "project_dir": project_dir or kernel_root,
        }

    if not source_path.is_file():
        return {
            "ok": False,
            "mode": "primitive",
            "error": f"Kernel source_file does not exist: {source_path}",
            "project_dir": project_dir or kernel_root,
        }

    if source_path.suffix != ".cu":
        return {
            "ok": False,
            "mode": "primitive",
            "error": f"Kernel source_file must point to one .cu primitive, not {source_path.name}",
            "project_dir": project_dir or kernel_root,
        }

    try:
        source_path.relative_to(kernel_root)
    except ValueError:
        return {
            "ok": False,
            "mode": "primitive",
            "error": (
                f"Kernel source_file must live under {kernel_root}. "
                f"Got {source_path}. Split Octave wrapper work into a non-kernel job."
            ),
            "project_dir": project_dir or kernel_root,
        }

    return {
        "ok": True,
        "mode": "primitive",
        "source_path": source_path,
        "project_dir": kernel_root,
        "shelf_subdir": shelf_subdir,
        "detail": f"{shelf_subdir}/{source_path.name}",
    }


# Kernel type inference from filename prefixes
KERNEL_PREFIXES = [
    "attention", "gemm", "fused_mlp", "fusedmlp", "lu", "qr", "cholesky",
    "spmv", "dotproduct", "linalg", "rmsnorm", "swiglu", "fft", "trsm",
    "cg", "gmres", "eigenvalue", "convolution", "ichol", "ldlt", "bicgstab",
    "numerical", "cross", "all",
]


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
    name = os.path.basename(filepath).lower()
    path_parts = filepath.lower()

    for prefix in KERNEL_PREFIXES:
        if name.startswith(prefix + "_") or name.startswith(prefix + "-"):
            return prefix
        # Match directory component exactly (not substring)
        if f"/{prefix}/" in path_parts:
            return prefix

    # Content-based fallback for files that don't follow naming conventions
    if text:
        text_lower = text[:2000].lower()
        for prefix in KERNEL_PREFIXES:
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

        # -- Test runs log for compliance and gate steps --
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
        for col in ['spec', 'source_file', 'version', 'factory_mode',
                    'execution_lane',
                    'objective_vector', 'acceptance_gates', 'keep_rule',
                    'benchmark_set', 'failure_budget', 'crossover_policy',
                    'optimization_scope', 'hardware_target', 'retarget_policy',
                    'reference_label']:
            try:
                if col == 'version':
                    c.execute(f"ALTER TABLE jobs ADD COLUMN version REAL DEFAULT 0")
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
        """Ship one kernel primitive for a job, or no-op for non-kernel jobs."""
        job = self.get_job(job_id)
        if not job:
            return []
        kernel = job.get("kernel_type", "")
        if not kernel:
            return []

        plan = describe_job_shipping(job)
        if not plan.get("ok"):
            return [{"action": "error", "error": plan["error"]}]
        if plan.get("mode") != "primitive":
            return [{"action": "noop", "detail": plan.get("detail", "non-kernel job")}]

        # Get vs_ref from worker_state
        vs_ref = None
        row = self.conn.execute(
            "SELECT best_vsref FROM worker_state WHERE kernel_type = ?", (kernel,)
        ).fetchone()
        if row:
            vs_ref = row[0]

        try:
            result = self.ship_primitive(
                str(plan["source_path"]), shelf_subdir=plan["shelf_subdir"],
                vs_ref=vs_ref, shipped_by=shipped_by
            )
            return [result]
        except Exception as e:
            return [{"action": "error", "file": str(plan["source_path"]), "error": str(e)}]

    def record_test_run(self, job_id: int, kernel_type: str, category: str, command: str, status: str, output: str = "") -> None:
        """Persist a gate/test run for compliance, edge, and stress suites."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute("INSERT INTO test_runs (job_id, kernel_type, category, command, status, output, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (job_id if job_id is not None else None, kernel_type, category, command, status, output, now))
        self.conn.commit()

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
                   execution_lane="", target_vs_ref=1.0, tags="", created_by="ops", notes="",
                   source_file="", factory_mode="", objective_vector="",
                   acceptance_gates="", keep_rule="", benchmark_set="",
                   failure_budget="", crossover_policy="", optimization_scope="",
                   hardware_target="", retarget_policy="", reference_label="") -> int:
        if state not in ALL_JOB_STATES:
            raise ValueError(f"Unknown state '{state}'. Valid: {sorted(ALL_JOB_STATES)}")
        if job_type not in JOB_TYPES:
            raise ValueError(f"Unknown job type '{job_type}'. Valid: {sorted(JOB_TYPES)}")
        if factory_mode and factory_mode not in FACTORY_MODES:
            raise ValueError(f"Unknown factory mode '{factory_mode}'. Valid: {sorted(FACTORY_MODES)}")
        if optimization_scope and optimization_scope not in OPTIMIZATION_SCOPES:
            raise ValueError(
                f"Unknown optimization scope '{optimization_scope}'. "
                f"Valid: {sorted(OPTIMIZATION_SCOPES)}"
            )
        phase = STATE_TO_PHASE[state]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cursor = self.conn.execute("""
            INSERT INTO jobs (name, title, description, job_type, kernel_type,
                              parent_job_id, state, phase, priority, assigned_to, execution_lane,
                              target_vs_ref, tags, created_at, updated_at,
                              created_by, updated_by, notes, source_file,
                              factory_mode, objective_vector, acceptance_gates,
                              keep_rule, benchmark_set, failure_budget,
                              crossover_policy, optimization_scope,
                              hardware_target, retarget_policy, reference_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, title, description, job_type, kernel_type,
              parent_job_id, state, phase, priority, assigned_to, execution_lane,
              target_vs_ref, tags, now, now, created_by, created_by, notes, source_file,
              factory_mode, objective_vector, acceptance_gates, keep_rule,
              benchmark_set, failure_budget, crossover_policy, optimization_scope,
              hardware_target, retarget_policy, reference_label))
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

        # Version bump: every time a job hits 'shipped' internally, minor version increments.
        # Internal versions are 0.x (0.1, 0.2, ...). Public release will be 1.0.
        lane_update = ""
        lane_params = []
        if to_state in ('shipped', 'converged', 'abandoned', 'parked'):
            lane_update = ", execution_lane = ?"
            lane_params = ['parked']
        if to_state == 'shipped':
            cur = job.get("version", 0) or 0
            version = round(cur + 0.1, 1)
            reason = f"[v{version}] {reason}" if reason else f"[v{version}] shipped"
            self.conn.execute(
                f"UPDATE jobs SET state = ?, phase = ?, version = ?, updated_at = ?, updated_by = ?{lane_update} WHERE id = ?",
                (to_state, new_phase, version, now, changed_by, *lane_params, job_id))
        else:
            self.conn.execute(
                f"UPDATE jobs SET state = ?, phase = ?, updated_at = ?, updated_by = ?{lane_update} WHERE id = ?",
                (to_state, new_phase, now, changed_by, *lane_params, job_id))
        self.conn.execute("""
            INSERT INTO job_transitions (job_id, from_state, to_state, changed_by, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (job_id, from_state, to_state, changed_by, reason, now))
        self.conn.commit()

        if to_state == 'rework':
            kernel = job.get('kernel_type') or '<kernel>'
            query = f"{BWK_ROOT}/common/memory/msearch \"{kernel} failure root cause\" --kernel {kernel} -k 5" if kernel != '<kernel>' else "{BWK_ROOT}/common/memory/msearch \"failure root cause\" -k 5"
            self.ensure_open_message('watchdog',
                                     f"Research checkpoint required for job #{job_id}",
                                     body=(
                                         "Job entered rework. Before continuing, read the failure messages, run a research checkpoint against the DB, and use the result to guide the next fix. Suggested query: " + query
                                     ),
                                     job_id=job_id, message_type='info', priority='normal')

        return self.get_job(job_id)

    def update_job(self, job_id, updated_by="ops", **kwargs):
        allowed = {"title", "description", "priority", "assigned_to", "execution_lane", "vs_ref",
                    "target_vs_ref", "tags", "notes", "kernel_type", "spec",
                    "source_file", "factory_mode", "objective_vector", "job_type",
                    "acceptance_gates", "keep_rule", "benchmark_set",
                    "failure_budget", "crossover_policy", "optimization_scope",
                    "hardware_target", "retarget_policy", "reference_label"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return self.get_job(job_id)
        if "job_type" in updates and updates["job_type"] and updates["job_type"] not in JOB_TYPES:
            raise ValueError(f"Unknown job type '{updates['job_type']}'. Valid: {sorted(JOB_TYPES)}")
        if "factory_mode" in updates and updates["factory_mode"] and updates["factory_mode"] not in FACTORY_MODES:
            raise ValueError(f"Unknown factory mode '{updates['factory_mode']}'. Valid: {sorted(FACTORY_MODES)}")
        if "optimization_scope" in updates and updates["optimization_scope"] and updates["optimization_scope"] not in OPTIMIZATION_SCOPES:
            raise ValueError(
                f"Unknown optimization scope '{updates['optimization_scope']}'. "
                f"Valid: {sorted(OPTIMIZATION_SCOPES)}"
            )
        if "execution_lane" in updates and updates["execution_lane"] and updates["execution_lane"] not in EXECUTION_LANES:
            raise ValueError(
                f"Unknown execution lane '{updates['execution_lane']}'. Valid: {sorted(EXECUTION_LANES)}"
            )
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
                 assigned_to=None, parent_job_id=None, priority=None, execution_lane=None):
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
        if execution_lane:
            where.append("execution_lane = ?"); params.append(execution_lane)
        if parent_job_id is not None:
            where.append("parent_job_id = ?"); params.append(parent_job_id)
        if priority:
            where.append("priority = ?"); params.append(priority)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self.conn.execute(f"""
            SELECT * FROM jobs {where_sql}
            ORDER BY CASE execution_lane WHEN 'active' THEN 0 WHEN 'hopper' THEN 1 WHEN 'incubating' THEN 2 WHEN 'parked' THEN 3 ELSE 4 END,
                     CAST(priority AS INTEGER), updated_at DESC
        """, params).fetchall()
        return [dict(r) for r in rows]

    def get_job_history(self, job_id):
        rows = self.conn.execute(
            "SELECT * FROM job_transitions WHERE job_id = ? ORDER BY timestamp ASC",
            (job_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_watchdog_state(self, name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM watchdog_state WHERE name = ?",
            (name,)
        ).fetchone()
        return dict(row) if row else None

    def touch_watchdog_state(self, name: str, status: str = "", notes: str = "") -> dict:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.conn.execute("""
            INSERT INTO watchdog_state (name, last_run_at, last_status, notes)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                last_run_at=excluded.last_run_at,
                last_status=excluded.last_status,
                notes=excluded.notes
        """, (name, now, status, notes))
        self.conn.commit()
        return self.get_watchdog_state(name)

    def set_watchdog_daemon_state(self, status: str, notes: str = "", pid: int | None = None, host: str = "") -> dict:
        parts = []
        if pid is not None:
            parts.append(f"pid={pid}")
        if host:
            parts.append(f"host={host}")
        if notes:
            parts.append(notes)
        return self.touch_watchdog_state("watchdog_daemon", status=status, notes=" | ".join(parts))

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

    def ensure_open_message(self, from_agent, subject, body="", to_agent="",
                            job_id=None, message_type="info", priority="normal"):
        row = self.conn.execute("""
            SELECT id FROM messages
            WHERE status = 'open' AND from_agent = ? AND subject = ?
              AND COALESCE(job_id, -1) = COALESCE(?, -1)
            ORDER BY id DESC
            LIMIT 1
        """, (from_agent, subject, job_id)).fetchone()
        if row:
            return row[0]
        return self.create_message(from_agent=from_agent, subject=subject, body=body,
                                   to_agent=to_agent, job_id=job_id,
                                   message_type=message_type, priority=priority)

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
        If a job id is supplied and that job has a kernel_type, the job-owned kernel identity
        wins over the caller-provided label. This keeps project/job heartbeats from being
        accidentally attributed to the wrong worker family.
        """
        canonical_kernel = (kernel_type or "").strip()
        if job_id is not None:
            job = self.get_job(job_id)
            job_worker = ((job or {}).get("assigned_to") or "").strip()
            job_kernel = ((job or {}).get("kernel_type") or "").strip()
            if job_worker:
                canonical_kernel = job_worker
            elif job_kernel:
                canonical_kernel = job_kernel
        if not canonical_kernel:
            raise ValueError("heartbeat requires a kernel type or a job with kernel_type set")

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cursor = self.conn.execute("""
            UPDATE worker_state SET heartbeat_at = ?, current_task = ?,
                process_state = ?, job_id = ?, updated_at = ?
            WHERE kernel_type = ?
        """, (now, current_task[:200], process_state, job_id, now, canonical_kernel))
        if cursor.rowcount == 0:
            self.conn.execute("""
                INSERT OR IGNORE INTO worker_state (kernel_type, heartbeat_at,
                    current_task, process_state, job_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (canonical_kernel, now, current_task[:200], process_state, job_id, now))
        self.conn.commit()
        return canonical_kernel

    def refresh_worker_state(self) -> dict:
        """Compute worker state from structured experiment history + heartbeat."""

        existing = {}
        for row in self.conn.execute(
            "SELECT kernel_type, tsv_path, heartbeat_at, current_task, job_id, process_state FROM worker_state"
        ).fetchall():
            existing[row[0]] = dict(row)

        kernels = set(existing.keys())
        kernels.update(
            r[0] for r in self.conn.execute(
                "SELECT DISTINCT kernel_type FROM experiments WHERE kernel_type != ''"
            ).fetchall()
        )
        kernels.update(
            r[0] for r in self.conn.execute(
                "SELECT DISTINCT kernel_type FROM jobs WHERE kernel_type != ''"
            ).fetchall()
        )

        results = {}

        for kernel in sorted(kernels):
            prior = existing.get(kernel, {})
            rows = [
                dict(r) for r in self.conn.execute("""
                    SELECT * FROM experiments
                    WHERE kernel_type = ?
                    ORDER BY COALESCE(timestamp, recorded_at) ASC, id ASC
                """, (kernel,)).fetchall()
            ]

            state = {
                "kernel_type": kernel,
                "tsv_path": prior.get("tsv_path", ""),
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
                "diagnosis": "no experiment data",
                "heartbeat_at": prior.get("heartbeat_at", ""),
                "current_task": prior.get("current_task", ""),
                "job_id": prior.get("job_id"),
                "process_state": prior.get("process_state", ""),
                "live_status": "historical",
                "live_reason": "no worker heartbeat recorded",
                "activity_at": "",
            }

            if not rows:
                heartbeat_recent = False
                if state.get("heartbeat_at"):
                    try:
                        import calendar
                        hb_epoch = calendar.timegm(time.strptime(state["heartbeat_at"], "%Y-%m-%dT%H:%M:%SZ"))
                        heartbeat_recent = (time.time() - hb_epoch) <= 600
                    except (ValueError, OverflowError):
                        heartbeat_recent = False
                if state.get("process_state") == "complete":
                    state["status"] = "complete"
                    state["diagnosis"] = f"worker self-reported complete. task: {state.get('current_task', '')[:60]}"
                    state["live_status"] = "complete"
                    state["live_reason"] = f"worker self-reported complete ({state.get('heartbeat_at') or 'no heartbeat'})"
                    state["activity_at"] = state.get("heartbeat_at", "")
                elif heartbeat_recent:
                    state["status"] = "producing"
                    state["diagnosis"] = f"active worker heartbeat. task: {state.get('current_task', '')[:80]}"
                    state["live_status"] = "active"
                    state["live_reason"] = f"recent heartbeat ({state['heartbeat_at']})"
                    state["activity_at"] = state.get("heartbeat_at", "")
                results[kernel] = state
                continue

            kept_rows = []
            streak = 0
            max_streak = 0
            for row in rows:
                row_status = (row.get("status") or "").strip().lower()
                if row_status in ("keep", "kept"):
                    kept_rows.append(row)
                    streak = 0
                elif row_status in ("discard", "discarded"):
                    streak += 1
                    max_streak = max(max_streak, streak)

            tail_streak = 0
            for row in reversed(rows):
                row_status = (row.get("status") or "").strip().lower()
                if row_status in ("discard", "discarded"):
                    tail_streak += 1
                else:
                    break

            best_vsref = None
            best_duration = None
            for row in kept_rows:
                vr = row.get("vs_ref")
                if vr is not None and (best_vsref is None or vr > best_vsref):
                    best_vsref = vr
                dur = row.get("duration_us")
                if dur is not None and (best_duration is None or dur < best_duration):
                    best_duration = dur

            last_row = rows[-1]
            top_stall = (last_row.get("top_stall") or "").strip()
            if top_stall in ("-", "none", ""):
                top_stall = ""

            last_kept_desc = ""
            if kept_rows:
                last_kept_desc = (kept_rows[-1].get("description") or "").strip()

            last_time = (last_row.get("timestamp") or "").strip() or (last_row.get("recorded_at") or "").strip()

            # Determine status (hybrid: self-reported + computed)
            # 1. Worker self-reported "complete" → trust it
            # 2. Heartbeat stale (>30 min) → dead/idle (computed)
            # 3. Discard streaks → stuck/grinding (computed, worker can't see this)
            # 4. Otherwise → producing
            heartbeat_stale = False
            heartbeat_recent = False
            hb_epoch = None
            last_exp_epoch = None
            if state.get("heartbeat_at"):
                try:
                    import calendar
                    hb_epoch = calendar.timegm(time.strptime(state["heartbeat_at"], "%Y-%m-%dT%H:%M:%SZ"))
                    heartbeat_recent = (time.time() - hb_epoch) <= 600   # 10 min
                    heartbeat_stale = (time.time() - hb_epoch) > 1800    # 30 min
                except (ValueError, OverflowError):
                    pass
            try:
                import calendar
                last_exp_epoch = calendar.timegm(time.strptime(last_time, "%Y-%m-%dT%H:%M:%SZ"))
            except (ValueError, OverflowError):
                last_exp_epoch = None

            if state.get("process_state") == "complete":
                status = "complete"
                diagnosis = f"worker self-reported complete. task: {state.get('current_task', '')[:60]}"
            elif heartbeat_recent:
                status = "producing"
                if tail_streak >= 5:
                    diagnosis = f"active worker investigating prior {tail_streak}-discard streak. task: {state.get('current_task', '')[:60]}"
                elif kept_rows:
                    diagnosis = f"active worker heartbeat. last keep: {last_kept_desc[:80]}"
                else:
                    diagnosis = f"active worker heartbeat. task: {state.get('current_task', '')[:60]}"
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

            if state.get("process_state") == "complete":
                live_status = "complete"
                live_reason = f"worker self-reported complete ({state.get('heartbeat_at') or 'no heartbeat'})"
            elif heartbeat_recent:
                live_status = "active"
                live_reason = f"recent heartbeat ({state['heartbeat_at']})"
            elif state.get("heartbeat_at"):
                live_status = "stale"
                live_reason = f"stale heartbeat ({state['heartbeat_at']})"
            elif last_exp_epoch and (time.time() - last_exp_epoch) <= 21600:
                live_status = "untracked_recent"
                live_reason = f"recent experiment results without heartbeat ({last_time})"
            elif last_time:
                live_status = "historical"
                live_reason = f"last experiment {last_time}"
            else:
                live_status = "historical"
                live_reason = "no recent activity"

            activity_candidates = [t for t in (state.get("heartbeat_at", ""), last_time) if t]
            activity_at = max(activity_candidates) if activity_candidates else ""

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
                "live_status": live_status,
                "live_reason": live_reason,
                "activity_at": activity_at,
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
                    heartbeat_at, current_task, job_id, process_state,
                    live_status, live_reason, activity_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    updated_at=excluded.updated_at,
                    heartbeat_at=excluded.heartbeat_at,
                    current_task=excluded.current_task,
                    job_id=excluded.job_id,
                    process_state=excluded.process_state,
                    live_status=excluded.live_status,
                    live_reason=excluded.live_reason,
                    activity_at=excluded.activity_at
            """, (kernel, s["tsv_path"], s["total_experiments"], s["kept"],
                  s["discarded"], s["best_vsref"], s["best_duration_us"],
                  s["top_stall"], s["current_discard_streak"],
                  s["max_discard_streak"], s["last_kept_description"],
                  s["last_experiment_time"], s["has_halt_note"],
                  s["status"], s["diagnosis"], now,
                  s.get("heartbeat_at", ""), s.get("current_task", ""),
                  s.get("job_id"), s.get("process_state", ""),
                  s.get("live_status", ""), s.get("live_reason", ""),
                  s.get("activity_at", "")))

        self.conn.commit()

        for kernel, s in results.items():
            if s.get("status") not in ("stalled", "grinding"):
                continue
            job_id = s.get("job_id")
            if not job_id:
                row = self.conn.execute("""
                    SELECT id FROM jobs
                    WHERE kernel_type = ?
                      AND state NOT IN ('shipped', 'converged', 'parked', 'abandoned')
                    ORDER BY CAST(priority AS INTEGER), updated_at DESC
                    LIMIT 1
                """, (kernel,)).fetchone()
                job_id = row[0] if row else None
            if not job_id:
                continue
            query = f"{BWK_ROOT}/common/memory/msearch \"{kernel} {s.get('top_stall') or 'bottleneck'}\" --kernel {kernel} -k 5"
            body = (
                f"Worker/progress status is {s.get('status')} for kernel '{kernel}'. Before continuing, run a research checkpoint against the DB and post the useful findings back into the work. Suggested query: {query}. If the playbook is empty, widen the search and then mark the job stuck_needs_research."
            )
            self.ensure_open_message('watchdog',
                                     f"Research checkpoint required for job #{job_id}",
                                     body=body, job_id=job_id,
                                     message_type='info', priority='normal')

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
                "CASE live_status "
                "  WHEN 'active' THEN 1 "
                "  WHEN 'stale' THEN 2 "
                "  WHEN 'untracked_recent' THEN 3 "
                "  WHEN 'complete' THEN 4 "
                "  ELSE 5 END, "
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
    print(f"Experiments: {s['experiments_total']}")
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


def cmd_maintain(args):
    """Run maintenance: incremental ingest, gap report, promotion candidates."""
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
    new_tsv = mem.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    removed = md_stats.get("removed", 0)
    print(f"  New docs indexed: {new_md}")
    print(f"  Experiment rows in DB: {new_tsv}")
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


def cmd_serve(args):
    """Minimal HTTP API for programmatic access."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    mem = ResearchMemory(args.db)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            if parsed.path == "/api/search":
                query = params.get("q", [""])[0]
                k = int(params.get("k", ["10"])[0])
                mode = params.get("mode", ["hybrid"])[0]
                level = int(params.get("level", ["0"])[0])
                kernel = params.get("kernel", [None])[0]
                doc_type = params.get("type", [None])[0]
                stall = params.get("stall", [None])[0]
                technique = params.get("technique", [None])[0]

                if level > 0:
                    results = mem.search_summaries(query, k, kernel, doc_type, stall, technique, level=level)
                elif mode == "fts":
                    results = mem.search_fts(query, k, kernel, doc_type)
                elif mode == "semantic":
                    results = mem.search_semantic(query, k, kernel, doc_type, stall, technique)
                else:
                    # Default: use summary search if summaries exist
                    has_summaries = mem.conn.execute("SELECT COUNT(*) FROM vec_summaries").fetchone()[0]
                    if has_summaries > 0:
                        results = mem.search_summaries(query, k, kernel, doc_type, stall, technique, level=2)
                    else:
                        results = mem.search_hybrid(query, k, kernel, doc_type, stall, technique)

                self._json_response({"results": results, "query": query, "count": len(results)})

            elif parsed.path == "/api/stats":
                self._json_response(mem.stats())

            elif parsed.path == "/api/quality":
                self._json_response({"report": mem.quality_report()})

            elif parsed.path == "/api/workers":
                if params.get("refresh", ["0"])[0] in ("1", "true", "yes"):
                    mem.refresh_worker_state()
                workers = mem.get_worker_state()
                self._json_response({"workers": workers, "count": len(workers)})

            elif parsed.path == "/api/issues":
                status = params.get("status", [None])[0]
                kernel = params.get("kernel", [None])[0]
                issues = mem.get_issues(status=status, kernel_type=kernel)
                open_count = sum(1 for i in issues if i["status"] in ("open", "assigned"))
                self._json_response({"issues": issues, "count": len(issues), "open": open_count})

            elif parsed.path.startswith("/api/issues/"):
                # /api/issues/42/assign?to=linalg
                # /api/issues/42/rework?fix=description
                # /api/issues/42/close
                # /api/issues/42/reopen?reason=still+fails
                parts = parsed.path.split("/")
                if len(parts) >= 4:
                    try:
                        issue_id = int(parts[3])
                    except ValueError:
                        self._json_response({"error": "Invalid issue ID"}, 400)
                        return
                    action = parts[4] if len(parts) > 4 else "get"
                    if action == "assign":
                        to = params.get("to", [""])[0]
                        mem.assign_issue(issue_id, to)
                        self._json_response({"ok": True, "issue_id": issue_id, "assigned_to": to})
                    elif action == "rework":
                        fix = params.get("fix", [""])[0]
                        mem.rework_issue(issue_id, fix)
                        self._json_response({"ok": True, "issue_id": issue_id, "status": "retest"})
                    elif action == "close":
                        mem.close_issue(issue_id)
                        self._json_response({"ok": True, "issue_id": issue_id, "status": "closed"})
                    elif action == "reopen":
                        reason = params.get("reason", [""])[0]
                        mem.reopen_issue(issue_id, reason)
                        self._json_response({"ok": True, "issue_id": issue_id, "status": "open"})
                    else:
                        issues = mem.get_issues()
                        match = [i for i in issues if i["id"] == issue_id]
                        self._json_response(match[0] if match else {"error": "Not found"})

            elif parsed.path == "/api/jobs":
                jobs = mem.get_jobs(state=params.get("state", [None])[0],
                                    phase=params.get("phase", [None])[0],
                                    job_type=params.get("type", [None])[0],
                                    kernel_type=params.get("kernel", [None])[0],
                                    assigned_to=params.get("assigned", [None])[0],
                                    priority=params.get("priority", [None])[0])
                active = sum(1 for j in jobs if j["state"] not in ("shipped","converged","parked","abandoned"))
                self._json_response({"jobs": jobs, "count": len(jobs), "active": active})

            elif parsed.path.startswith("/api/jobs/"):
                parts = parsed.path.split("/")
                if len(parts) >= 4:
                    if parts[3] == "new":
                        try:
                            jid = mem.create_job(
                                name=params.get("name",[""])[0], title=params.get("title",[""])[0],
                                description=params.get("description",[""])[0],
                                job_type=params.get("type",["kernel"])[0],
                                kernel_type=params.get("kernel",[""])[0],
                                parent_job_id=int(params["parent"][0]) if "parent" in params else None,
                                state=params.get("state",["wishlist"])[0],
                                priority=params.get("priority",["3"])[0],
                                assigned_to=params.get("assigned",[""])[0],
                                execution_lane=params.get("lane",[""])[0],
                                target_vs_ref=float(params["target"][0]) if "target" in params else 1.0,
                                tags=params.get("tags",[""])[0],
                                created_by=params.get("by",["ops"])[0],
                                notes=params.get("notes",[""])[0])
                            self._json_response({"ok": True, "job_id": jid})
                        except (ValueError, sqlite3.IntegrityError) as e:
                            self._json_response({"error": str(e)}, 400)
                    else:
                        try:
                            job_id = int(parts[3])
                        except ValueError:
                            self._json_response({"error": "Invalid job ID"}, 400); return
                        action = parts[4] if len(parts) > 4 else "get"
                        if action == "get":
                            job = mem.get_job(job_id)
                            self._json_response(job if job else {"error": "Not found"})
                        elif action == "transition":
                            try:
                                result = mem.update_job_state(job_id, params.get("to",[""])[0],
                                    params.get("by",["ops"])[0], params.get("reason",[""])[0])
                                self._json_response({"ok": True, "job": result})
                            except ValueError as e:
                                self._json_response({"error": str(e)}, 400)
                        elif action == "history":
                            self._json_response({"job_id": job_id, "transitions": mem.get_job_history(job_id)})
                        elif action == "update":
                            try:
                                result = mem.update_job(job_id, updated_by=params.get("by",["ops"])[0],
                                    title=params.get("title",[None])[0], description=params.get("description",[None])[0],
                                    priority=params.get("priority",[None])[0], assigned_to=params.get("assigned",[None])[0], execution_lane=params.get("lane",[None])[0],
                                    notes=params.get("notes",[None])[0], tags=params.get("tags",[None])[0])
                                self._json_response({"ok": True, "job": result})
                            except ValueError as e:
                                self._json_response({"error": str(e)}, 400)
                        elif action == "sync-vsref":
                            self._json_response({"ok": True, "job": mem.sync_job_vsref(job_id)})
                        else:
                            self._json_response({"error": f"Unknown action: {action}"}, 400)

            elif parsed.path == "/api/messages":
                msgs = mem.get_messages(status=params.get("status",[None])[0],
                    job_id=int(params["job"][0]) if "job" in params else None,
                    from_agent=params.get("from",[None])[0], to_agent=params.get("to",[None])[0],
                    message_type=params.get("type",[None])[0])
                open_count = sum(1 for m in msgs if m["status"] == "open")
                self._json_response({"messages": msgs, "count": len(msgs), "open": open_count})

            elif parsed.path.startswith("/api/messages/"):
                parts = parsed.path.split("/")
                if len(parts) >= 4:
                    if parts[3] == "new":
                        try:
                            mid = mem.create_message(from_agent=params.get("from",[""])[0],
                                subject=params.get("subject",[""])[0], body=params.get("body",[""])[0],
                                to_agent=params.get("to",[""])[0],
                                job_id=int(params["job"][0]) if "job" in params else None,
                                message_type=params.get("type",["info"])[0],
                                priority=params.get("priority",["normal"])[0])
                            self._json_response({"ok": True, "message_id": mid})
                        except ValueError as e:
                            self._json_response({"error": str(e)}, 400)
                    else:
                        try:
                            msg_id = int(parts[3])
                        except ValueError:
                            self._json_response({"error": "Invalid message ID"}, 400); return
                        action = parts[4] if len(parts) > 4 else "get"
                        if action == "get":
                            msg = mem.get_message(msg_id)
                            self._json_response(msg if msg else {"error": "Not found"})
                        elif action == "ack":
                            self._json_response({"ok": True, "message": mem.acknowledge_message(msg_id, params.get("by",["foreman"])[0])})
                        elif action == "resolve":
                            self._json_response({"ok": True, "message": mem.resolve_message(msg_id, params.get("by",["foreman"])[0])})
                        else:
                            self._json_response({"error": f"Unknown action: {action}"}, 400)

            else:
                self._json_response({"error": "Unknown endpoint"}, 404)

        def _json_response(self, data, code=200):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2, default=str).encode())

        def log_message(self, format, *a):
            pass

    port = args.port
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Research Memory API running on http://localhost:{port}")
    print(f"  Search:  http://localhost:{port}/api/search?q=bank+conflicts&kernel=gemm")
    print(f"  Stats:   http://localhost:{port}/api/stats")
    print(f"  Quality: http://localhost:{port}/api/quality")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        mem.close()


def main():
    parser = argparse.ArgumentParser(
        description="Research Memory Database for blackwell-kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--db", default=str(DB_PATH), help="Database path")

    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index research files (with deduplication)")
    p_ingest.add_argument("path", nargs="?", help="Specific file or directory (default: all sources)")
    p_ingest.add_argument("--type", default="research", help="Document type")
    p_ingest.add_argument("--pattern", default="**/*.md", help="Glob pattern for directories")
    p_ingest.add_argument("--force", action="store_true", help="Re-index even if unchanged")

    # search (default: hybrid)
    p_search = sub.add_parser("search", help="Search the research database")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("-k", type=int, default=10, help="Number of results")
    p_search.add_argument("--mode", choices=["hybrid", "semantic", "fts"], default="hybrid")
    p_search.add_argument("--kernel", help="Filter by kernel type")
    p_search.add_argument("--type", help="Filter by document type")
    p_search.add_argument("--stall", help="Filter by stall type")
    p_search.add_argument("--technique", help="Filter by technique tag")
    p_search.add_argument("--provenance", help="Filter by provenance tier")
    p_search.add_argument("--detail", action="store_const", const=2, dest="level", help="Show Level 2 summaries")
    p_search.add_argument("--full", action="store_const", const=3, dest="level", help="Show Level 3 full content")
    p_search.add_argument("--signals", action="store_const", const=1, dest="level", help="Show Level 1 signals only")
    p_search.set_defaults(level=0)
    p_search.add_argument("-v", "--verbose", action="store_true")

    # fts shortcut
    p_fts = sub.add_parser("fts", help="Full-text search (exact keyword match)")
    p_fts.add_argument("query", nargs="+")
    p_fts.add_argument("-k", type=int, default=10)
    p_fts.add_argument("--kernel", help="Filter by kernel type")
    p_fts.add_argument("--type", help="Filter by document type")
    p_fts.add_argument("-v", "--verbose", action="store_true")

    # stats
    sub.add_parser("stats", help="Show database statistics")

    # quality
    sub.add_parser("quality", help="Run quality audit report")

    # ingest-tsv (for watchdog — TSV-only incremental ingest)
    p_itsv = sub.add_parser("ingest-tsv", help="Import TSV experiments into DB and search index")
    p_itsv.add_argument("--force", action="store_true", help="Re-index even if unchanged")

    p_exp = sub.add_parser("experiment-add", help="Record one experiment row in factory_brain")
    p_exp.add_argument("--kernel", required=True)
    p_exp.add_argument("--status", required=True)
    p_exp.add_argument("--description", required=True)
    p_exp.add_argument("--timestamp", default="")
    p_exp.add_argument("--commit", default="", dest="git_commit")
    p_exp.add_argument("--duration-us", type=float, default=None)
    p_exp.add_argument("--vs-ref", type=float, default=None)
    p_exp.add_argument("--sm-pct", type=float, default=None)
    p_exp.add_argument("--stall-math", type=float, default=None)
    p_exp.add_argument("--stall-wait", type=float, default=None)
    p_exp.add_argument("--stall-scoreboard", type=float, default=None)
    p_exp.add_argument("--stall-barrier", type=float, default=None)
    p_exp.add_argument("--top-stall", default="")
    p_exp.add_argument("--job", type=int, default=None)
    p_exp.add_argument("--reference-label", default="")
    p_exp.add_argument("--source-type", default="db")
    p_exp.add_argument("--source-path", default="")
    p_exp.add_argument("--index", type=int, default=0)
    p_exp.add_argument("--extra-json", default="")

    p_exps = sub.add_parser("experiments", help="List experiments from factory_brain")
    p_exps.add_argument("--kernel")
    p_exps.add_argument("--job", type=int)
    p_exps.add_argument("--status")
    p_exps.add_argument("--limit", type=int, default=20)
    p_exp_summary = sub.add_parser("experiment-summary", help="Summarize experiment history from factory_brain")
    p_exp_summary.add_argument("--kernel")
    p_exp_summary.add_argument("--job", type=int)
    p_exp_summary.add_argument("--recent", type=int, default=8)

    # workers — show worker state table
    sub.add_parser("workers", help="Show computed worker state (stuck detection)")

    # issues — show/manage issues
    p_issues = sub.add_parser("issues", help="Show open issues from tester")
    p_issues.add_argument("--status", help="Filter by status (open/assigned/retest/closed)")
    p_issues.add_argument("--kernel", help="Filter by kernel type")

    # jobs
    p_jobs = sub.add_parser("jobs", help="List jobs (factory work items)")
    p_jobs.add_argument("--state", help="Filter by state")
    p_jobs.add_argument("--phase", help="Filter by phase")
    p_jobs.add_argument("--type", dest="job_type", help="Filter by job type")
    p_jobs.add_argument("--kernel", help="Filter by kernel type")
    p_jobs.add_argument("--assigned", help="Filter by assigned agent")
    p_jobs.add_argument("--priority", help="Filter by priority")
    p_jobs.add_argument("--lane", dest="execution_lane", help="Filter by execution lane", choices=sorted(EXECUTION_LANES))
    p_jc = sub.add_parser("job-create", help="Create a new job")
    p_jc.add_argument("name"); p_jc.add_argument("title")
    p_jc.add_argument("--description", default=""); p_jc.add_argument("--lane", dest="execution_lane", default="", choices=[''] + sorted(EXECUTION_LANES)); p_jc.add_argument("--type", default="kernel", dest="job_type", choices=sorted(JOB_TYPES))
    p_jc.add_argument("--kernel", default=""); p_jc.add_argument("--parent", type=int, default=None)
    p_jc.add_argument("--state", default="wishlist"); p_jc.add_argument("--priority", default="3")
    p_jc.add_argument("--assigned", default=""); p_jc.add_argument("--target", type=float, default=1.0)
    p_jc.add_argument("--tags", default=""); p_jc.add_argument("--by", default="ops")
    p_jc.add_argument("--notes", default="")
    p_jc.add_argument("--source-file", default="", dest="source_file", help="Primary source file. Required for kernel primitive shipping.")
    p_jc.add_argument("--factory-mode", default="", choices=sorted(FACTORY_MODES))
    p_jc.add_argument("--objective-vector", default="")
    p_jc.add_argument("--acceptance-gates", default="")
    p_jc.add_argument("--keep-rule", default="")
    p_jc.add_argument("--benchmark-set", default="")
    p_jc.add_argument("--failure-budget", default="")
    p_jc.add_argument("--crossover-policy", default="")
    p_jc.add_argument("--optimization-scope", default="", choices=sorted(OPTIMIZATION_SCOPES))
    p_jc.add_argument("--hardware-target", default="")
    p_jc.add_argument("--retarget-policy", default="")
    p_jc.add_argument("--reference-label", default="")
    p_ju = sub.add_parser("job-update", help="Update job state or fields")
    p_ju.add_argument("id", type=int); p_ju.add_argument("--state"); p_ju.add_argument("--title")
    p_ju.add_argument("--description"); p_ju.add_argument("--priority"); p_ju.add_argument("--assigned")
    p_ju.add_argument("--lane", dest="execution_lane", choices=sorted(EXECUTION_LANES))
    p_ju.add_argument("--type", dest="job_type", choices=sorted(JOB_TYPES))
    p_ju.add_argument("--notes"); p_ju.add_argument("--tags"); p_ju.add_argument("--spec")
    p_ju.add_argument("--source-file", dest="source_file")
    p_ju.add_argument("--factory-mode", choices=sorted(FACTORY_MODES))
    p_ju.add_argument("--objective-vector")
    p_ju.add_argument("--acceptance-gates")
    p_ju.add_argument("--keep-rule")
    p_ju.add_argument("--benchmark-set")
    p_ju.add_argument("--failure-budget")
    p_ju.add_argument("--crossover-policy")
    p_ju.add_argument("--optimization-scope", choices=sorted(OPTIMIZATION_SCOPES))
    p_ju.add_argument("--hardware-target")
    p_ju.add_argument("--retarget-policy")
    p_ju.add_argument("--reference-label")
    p_ju.add_argument("--by", default="ops"); p_ju.add_argument("--reason", default="")
    p_js = sub.add_parser("job-show", help="Show job details including spec")
    p_js.add_argument("id", type=int)
    p_jh = sub.add_parser("job-history", help="Show job transition history")
    p_jh.add_argument("id", type=int)
    p_msgs = sub.add_parser("messages", help="List messages between agents")
    p_msgs.add_argument("--status"); p_msgs.add_argument("--job", type=int)
    p_msgs.add_argument("--from-agent", dest="from_agent"); p_msgs.add_argument("--to-agent", dest="to_agent")
    p_msgs.add_argument("--type", dest="msg_type")
    p_mc = sub.add_parser("message-create", help="Send a message")
    p_mc.add_argument("--from", required=True, dest="from_agent"); p_mc.add_argument("--subject", required=True)
    p_mc.add_argument("--body", default=""); p_mc.add_argument("--to", default="", dest="to_agent")
    p_mc.add_argument("--job", type=int, default=None); p_mc.add_argument("--type", default="info", dest="msg_type")
    p_mc.add_argument("--priority", default="normal")
    p_ma = sub.add_parser("message-ack", help="Acknowledge a message"); p_ma.add_argument("id", type=int); p_ma.add_argument("--by", default="foreman")
    p_mr = sub.add_parser("message-resolve", help="Resolve a message"); p_mr.add_argument("id", type=int); p_mr.add_argument("--by", default="foreman")
    p_nudge = sub.add_parser("nudge", help="Run gate logic for one job immediately")
    p_nudge.add_argument("id", type=int, help="Job ID to nudge")
    p_wds = sub.add_parser("watchdog-state", help="Show watchdog tick timestamps")
    p_wds.add_argument("--name", help="Filter by watchdog tick name")
    p_link = sub.add_parser("link-so", help="Link all compiled primitives into libbwk_primitives.so")
    p_link.add_argument("--output", default=None, help="Output path (default: primitives/lib/libbwk_primitives.so)")
    p_hb = sub.add_parser("heartbeat", help="Worker reports it's alive")
    p_hb.add_argument("kernel", help="Kernel type (e.g., lu, qr)")
    p_hb.add_argument("--task", default="", help="What you're working on")
    p_hb.add_argument("--state", default="working", help="working or complete")
    p_hb.add_argument("--job", type=int, default=None, help="Job ID")

    # maintain — DB health check + gap report
    sub.add_parser("maintain", help="Run maintenance: ingest new files, report gaps, suggest promotions")

    # serve
    p_serve = sub.add_parser("serve", help="Start HTTP API server")
    p_serve.add_argument("port", nargs="?", type=int, default=8421)

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ingest-tsv":
        mem = ResearchMemory(args.db)
        stats = mem.ingest_all_tsv(force=args.force)
        if stats["chunks"] > 0:
            print(f"TSV ingest: {stats['files']} files, {stats['chunks']} rows "
                  f"({stats.get('kept', 0)} kept, {stats.get('discarded', 0)} discarded)")
        mem.close()
    elif args.command == "experiment-add":
        mem = ResearchMemory(args.db)
        extra = {}
        if args.extra_json:
            extra = json.loads(args.extra_json)
        row = mem.record_experiment(
            kernel_type=args.kernel,
            status=args.status,
            description=args.description,
            timestamp=args.timestamp,
            git_commit=args.git_commit,
            duration_us=args.duration_us,
            vs_ref=args.vs_ref,
            sm_pct=args.sm_pct,
            stall_math=args.stall_math,
            stall_wait=args.stall_wait,
            stall_scoreboard=args.stall_scoreboard,
            stall_barrier=args.stall_barrier,
            top_stall=args.top_stall,
            job_id=args.job,
            source_type=args.source_type,
            source_path=args.source_path,
            experiment_index=args.index,
            reference_label=args.reference_label,
            extra=extra,
        )
        print(f"Recorded experiment #{row.get('id', '?')} for {args.kernel}: {args.status}")
        mem.close()
    elif args.command == "experiments":
        mem = ResearchMemory(args.db)
        rows = mem.get_experiments(kernel_type=args.kernel, job_id=args.job,
                                   status=args.status, limit=args.limit)
        if not rows:
            print("No experiments found.")
        else:
            print(f"\n{'ID':>5} {'KERNEL':<14} {'STATUS':<9} {'VS_REF':>7} {'DUR(us)':>9} {'WHEN':<20} DESCRIPTION")
            print("-" * 110)
            for r in rows:
                vs = f"{r['vs_ref']:.2f}" if r.get("vs_ref") is not None else "-"
                dur = f"{r['duration_us']:.1f}" if r.get("duration_us") is not None else "-"
                when = (r.get("timestamp") or r.get("recorded_at") or "")[:20]
                print(f"{r['id']:>5} {r['kernel_type']:<14} {r['status']:<9} {vs:>7} {dur:>9} {when:<20} {(r.get('description') or '')[:40]}")
        mem.close()
    elif args.command == "experiment-summary":
        mem = ResearchMemory(args.db)
        summary = mem.summarize_experiments(kernel_type=args.kernel, job_id=args.job, recent=args.recent)
        if summary["total"] == 0:
            print("No experiments found.")
        else:
            print(f"Kernel: {summary.get('kernel_type') or '-'}")
            print(f"Job: {summary.get('job_id') if summary.get('job_id') is not None else '-'}")
            print(f"Total: {summary['total']}  kept={summary['kept']}  discarded={summary['discarded']}  unknown={summary['unknown']}")
            print(f"Discard streak: current={summary['current_discard_streak']} max={summary['max_discard_streak']}")
            best = summary.get("best_keep")
            if best:
                vs = f"{best['vs_ref']:.2f}x" if best.get("vs_ref") is not None else "-"
                dur = f"{best['duration_us']:.1f}us" if best.get("duration_us") is not None else "-"
                print(f"Best keep: #{best['id']} {vs} {dur}  {((best.get('description') or '')[:120])}")
            last_keep = summary.get("last_keep")
            if last_keep:
                print(f"Last keep: #{last_keep['id']} {(last_keep.get('timestamp') or last_keep.get('recorded_at') or '')[:20]}  {((last_keep.get('description') or '')[:120])}")
            last = summary.get("last_experiment")
            if last:
                print(f"Last experiment: #{last['id']} {last.get('status', '-'):<8} {(last.get('timestamp') or last.get('recorded_at') or '')[:20]}  {((last.get('description') or '')[:120])}")
            if summary["top_stalls"]:
                print("Top stalls:")
                for item in summary["top_stalls"]:
                    print(f"  - {item['stall']}: {item['count']}")
            if summary["recent_discards"]:
                print("Recent discards:")
                for row in summary["recent_discards"]:
                    print(f"  - #{row['id']} {(row.get('top_stall') or '-'): <18} {((row.get('description') or '')[:120])}")
            if summary["recent_keeps"]:
                print("Recent keeps:")
                for row in summary["recent_keeps"]:
                    vs = f"{row['vs_ref']:.2f}x" if row.get("vs_ref") is not None else "-"
                    print(f"  - #{row['id']} {vs:>7} {(row.get('top_stall') or '-'): <18} {((row.get('description') or '')[:120])}")
            print("Recent history:")
            for row in summary["recent"]:
                when = (row.get("timestamp") or row.get("recorded_at") or "")[:20]
                print(f"  - #{row['id']} {row.get('status', '-'):<8} {when} {((row.get('description') or '')[:100])}")
        mem.close()
    elif args.command == "workers":
        mem = ResearchMemory(args.db)
        mem.refresh_worker_state()
        workers = mem.get_worker_state()
        # Status badges
        badges = {
            "stalled": "\033[31mSTALLED\033[0m",
            "grinding": "\033[33mGRINDING\033[0m",
            "halted": "\033[35mHALTED\033[0m",
            "producing": "\033[32mPRODUCING\033[0m",
            "converged": "\033[36mCONVERGED\033[0m",
            "idle": "\033[2mIDLE\033[0m",
            "unknown": "\033[2m???\033[0m",
        }
        print(f"\n{'KERNEL':<16} {'STATUS':<12} {'BEST':>6} {'EXP':>5} {'KEPT':>5} "
              f"{'LIVE':<16} {'STREAK':>6} {'STALL':<18} DIAGNOSIS")
        print("-" * 126)
        for w in workers:
            badge = badges.get(w["status"], w["status"])
            vsref = f"{w['best_vsref']:.2f}x" if w["best_vsref"] else "-"
            print(f"{w['kernel_type']:<16} {badge:<21} {vsref:>6} {w['total_experiments']:>5} "
                  f"{w['kept']:>5} {(w.get('live_status') or '-'): <16} {w['current_discard_streak']:>6} "
                  f"{w['top_stall']:<18} {w['diagnosis'][:50]}")
        print()
        # Summary
        stuck = sum(1 for w in workers if w["status"] in ("stalled", "grinding"))
        halted = sum(1 for w in workers if w["status"] == "halted")
        producing = sum(1 for w in workers if w["status"] == "producing")
        if stuck:
            print(f"  \033[31m{stuck} workers need attention (stalled/grinding)\033[0m")
        if halted:
            print(f"  \033[35m{halted} workers halted (check halt notes)\033[0m")
        print(f"  {producing} workers producing")
        mem.close()
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "fts":
        args.mode = "fts"
        args.stall = None
        args.technique = None
        args.provenance = None
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "quality":
        cmd_quality(args)
    elif args.command == "issues":
        mem = ResearchMemory(args.db)
        issues = mem.get_issues(status=args.status, kernel_type=args.kernel)
        if not issues:
            print("No issues found.")
        else:
            # Status badges
            status_badges = {
                "open": "\033[31mOPEN\033[0m",
                "assigned": "\033[33mASSIGNED\033[0m",
                "retest": "\033[36mRETEST\033[0m",
                "closed": "\033[32mCLOSED\033[0m",
            }
            sev_badges = {
                "blocking": "\033[31mBLOCKING\033[0m",
                "correctness": "\033[33mCORRECTNESS\033[0m",
                "warning": "\033[2mWARNING\033[0m",
            }
            print(f"\n{'ID':>4}  {'STATUS':<10} {'SEVERITY':<14} {'KERNEL':<14} {'ASSIGNED':<12} TITLE")
            print("-" * 100)
            for issue in issues:
                status_b = status_badges.get(issue["status"], issue["status"])
                sev_b = sev_badges.get(issue["severity"], issue["severity"])
                assigned = issue["assigned_to"] or "-"
                print(f"{issue['id']:>4}  {status_b:<19} {sev_b:<23} {issue['kernel_type']:<14} "
                      f"{assigned:<12} {issue['title'][:50]}")
            print()
            open_count = sum(1 for i in issues if i["status"] in ("open", "assigned"))
            retest_count = sum(1 for i in issues if i["status"] == "retest")
            if open_count:
                print(f"  \033[31m{open_count} open issues need attention\033[0m")
            if retest_count:
                print(f"  \033[36m{retest_count} issues awaiting re-test\033[0m")
        mem.close()
    elif args.command == "jobs":
        mem = ResearchMemory(args.db)
        jobs = mem.get_jobs(state=args.state, phase=args.phase, job_type=getattr(args,"job_type",None),
                            kernel_type=args.kernel, assigned_to=args.assigned, priority=args.priority, execution_lane=getattr(args, 'execution_lane', None))
        if not jobs:
            print("No jobs found.")
        else:
            phase_colors = {"ideation":"\033[2m","development":"\033[33m","validation":"\033[36m",
                            "rework":"\033[31m","quality":"\033[35m","shipping":"\033[32m","terminal":"\033[2m"}
            rst = "\033[0m"
            print(f"\n{'ID':>4}  {'STATE':<20} {'PHASE':<13} {'PRI':<8} {'KERNEL':<12} {'ASSIGNED':<10} {'LANE':<11} {'V':>3} {'VS_REF':>7}  TITLE")
            print("-" * 115)
            for j in jobs:
                pc = phase_colors.get(j["phase"],"")
                vs = f"{j['vs_ref']:.2f}" if j["vs_ref"] else "-"
                ver = j.get('version', 0) or 0
                ver_s = f"{ver:.1f}" if ver > 0 else "-"
                print(f"{j['id']:>4}  {pc}{j['state']:<20}{rst} {j['phase']:<13} {j['priority']:<8} "
                      f"{(j['kernel_type'] or '-'):<12} {(j['assigned_to'] or '-'):<10} {(j.get('execution_lane') or '-'):<11} {ver_s:>5} {vs:>7}  {j['title'][:40]}")
            print()
            all_jobs = mem.get_jobs()
            lane_counts = {
                'active': sum(1 for j in all_jobs if (j.get('execution_lane') or '') == 'active'),
                'hopper': sum(1 for j in all_jobs if (j.get('execution_lane') or '') == 'hopper'),
                'incubating': sum(1 for j in all_jobs if (j.get('execution_lane') or '') == 'incubating'),
                'parked': sum(1 for j in all_jobs if (j.get('execution_lane') or '') == 'parked'),
            }
            print(f"  lanes: active={lane_counts['active']} hopper={lane_counts['hopper']} incubating={lane_counts['incubating']} parked={lane_counts['parked']}")
        mem.close()
    elif args.command == "job-create":
        mem = ResearchMemory(args.db)
        try:
            jid = mem.create_job(name=args.name, title=args.title, description=args.description,
                job_type=args.job_type, kernel_type=args.kernel, parent_job_id=args.parent,
                state=args.state, priority=args.priority, assigned_to=args.assigned, execution_lane=args.execution_lane,
                target_vs_ref=args.target, tags=args.tags, created_by=args.by,
                notes=args.notes, source_file=args.source_file,
                factory_mode=args.factory_mode, objective_vector=args.objective_vector,
                acceptance_gates=args.acceptance_gates, keep_rule=args.keep_rule,
                benchmark_set=args.benchmark_set, failure_budget=args.failure_budget,
                crossover_policy=args.crossover_policy,
                optimization_scope=args.optimization_scope,
                hardware_target=args.hardware_target,
                retarget_policy=args.retarget_policy,
                reference_label=args.reference_label)
            print(f"Created job #{jid}: {args.name} ({args.state})")
        except (ValueError, sqlite3.IntegrityError) as e:
            print(f"Error: {e}", file=sys.stderr); sys.exit(1)
        mem.close()
    elif args.command == "job-update":
        mem = ResearchMemory(args.db)
        try:
            if args.state:
                result = mem.update_job_state(args.id, args.state, changed_by=args.by, reason=args.reason)
                print(f"Job #{args.id} ({result['name']}): {result['state']} [{result['phase']}]  by {args.by}")
            else:
                result = mem.update_job(args.id, updated_by=args.by, title=args.title, description=args.description,
                    priority=args.priority, assigned_to=args.assigned, execution_lane=args.execution_lane, notes=args.notes, tags=args.tags,
                    job_type=args.job_type, source_file=args.source_file,
                    spec=args.spec, factory_mode=args.factory_mode, objective_vector=args.objective_vector,
                    acceptance_gates=args.acceptance_gates, keep_rule=args.keep_rule,
                    benchmark_set=args.benchmark_set, failure_budget=args.failure_budget,
                    crossover_policy=args.crossover_policy,
                    optimization_scope=args.optimization_scope,
                    hardware_target=args.hardware_target,
                    retarget_policy=args.retarget_policy,
                    reference_label=args.reference_label)
                print(f"Job #{args.id} ({result['name']}): updated")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr); sys.exit(1)
        mem.close()
    elif args.command == "job-show":
        mem = ResearchMemory(args.db)
        job = mem.get_job(args.id)
        if not job:
            print(f"Job #{args.id} not found.", file=sys.stderr); sys.exit(1)
        ver = job.get('version', 0) or 0
        ver_s = f"v{ver:.1f}" if ver > 0 else "unshipped"
        print(f"Job #{job['id']}: {job['title']}")
        print(f"  State: {job['state']} [{job['phase']}] | Version: {ver_s}")
        print(f"  Type: {job.get('job_type') or '-'} | Priority: {job['priority']} | Kernel: {job['kernel_type'] or '-'} | Assigned: {job['assigned_to'] or '-'} | Lane: {job.get('execution_lane') or '-'}")
        if job.get('source_file'):
            print(f"  Source File: {job['source_file']}")
        if job.get('factory_mode'):
            print(f"  Factory Mode: {job['factory_mode']}")
        if job.get('optimization_scope'):
            print(f"  Optimization Scope: {job['optimization_scope']}")
        if job.get('hardware_target'):
            print(f"  Hardware Target: {job['hardware_target']}")
        if job.get('description'): print(f"  Description: {job['description']}")
        if job.get('notes'): print(f"  Notes: {job['notes']}")
        if job.get('objective_vector'): print(f"  Objective Vector: {job['objective_vector']}")
        if job.get('acceptance_gates'): print(f"  Acceptance Gates: {job['acceptance_gates']}")
        if job.get('keep_rule'): print(f"  Keep Rule: {job['keep_rule']}")
        if job.get('benchmark_set'): print(f"  Benchmark Set: {job['benchmark_set']}")
        if job.get('failure_budget'): print(f"  Failure Budget: {job['failure_budget']}")
        if job.get('crossover_policy'): print(f"  Crossover Policy: {job['crossover_policy']}")
        if job.get('retarget_policy'): print(f"  Retarget Policy: {job['retarget_policy']}")
        if job.get('reference_label'): print(f"  Reference Label: {job['reference_label']}")
        if job.get('spec'):
            print(f"\n{job['spec']}")
        else:
            print("\n  (no spec attached)")
        mem.close()
    elif args.command == "job-history":
        mem = ResearchMemory(args.db)
        job = mem.get_job(args.id)
        if not job:
            print(f"Job #{args.id} not found.", file=sys.stderr); sys.exit(1)
        history = mem.get_job_history(args.id)
        print(f"\nJob #{args.id}: {job['name']} — {job['title']}\nCurrent: {job['state']} [{job['phase']}]\n")
        if history:
            print(f"{'TIMESTAMP':<26} {'FROM':<20} {'TO':<20} {'BY':<10} REASON")
            print("-" * 100)
            for t in history:
                print(f"{t['timestamp']:<26} {(t['from_state'] or '(new)'):<20} {t['to_state']:<20} {t['changed_by']:<10} {t['reason'][:40]}")
        mem.close()
    elif args.command == "messages":
        mem = ResearchMemory(args.db)
        msgs = mem.get_messages(status=args.status, job_id=args.job, from_agent=args.from_agent,
            to_agent=args.to_agent, message_type=args.msg_type)
        if not msgs:
            print("No messages found.")
        else:
            sb = {"open":"\033[31mOPEN\033[0m","acknowledged":"\033[33mACK\033[0m","resolved":"\033[32mDONE\033[0m"}
            print(f"\n{'ID':>4}  {'STATUS':<14} {'TYPE':<10} {'FROM':<12} {'TO':<12} {'JOB':>4}  SUBJECT")
            print("-" * 100)
            for m in msgs:
                print(f"{m['id']:>4}  {sb.get(m['status'],m['status']):<23} {m['message_type']:<10} "
                      f"{m['from_agent']:<12} {(m['to_agent'] or '*'):<12} {str(m['job_id'] or '-'):>4}  {m['subject'][:40]}")
            print(f"\n  {sum(1 for m in msgs if m['status']=='open')} open")
        mem.close()
    elif args.command == "message-create":
        mem = ResearchMemory(args.db)
        mid = mem.create_message(from_agent=args.from_agent, subject=args.subject, body=args.body,
            to_agent=args.to_agent, job_id=args.job, message_type=args.msg_type, priority=args.priority)
        print(f"Created message #{mid}: {args.subject}"); mem.close()
    elif args.command == "message-ack":
        mem = ResearchMemory(args.db)
        mem.acknowledge_message(args.id, by=args.by); print(f"Message #{args.id}: acknowledged"); mem.close()
    elif args.command == "message-resolve":
        mem = ResearchMemory(args.db)
        mem.resolve_message(args.id, by=args.by); print(f"Message #{args.id}: resolved"); mem.close()
    elif args.command == "watchdog-state":
        mem = ResearchMemory(args.db)
        if args.name:
            rows = [mem.get_watchdog_state(args.name)]
            rows = [r for r in rows if r]
        else:
            rows = [
                dict(r) for r in mem.conn.execute(
                    "SELECT * FROM watchdog_state ORDER BY CASE name WHEN 'watchdog_daemon' THEN 0 ELSE 1 END, name"
                ).fetchall()
            ]
        if not rows:
            print("No watchdog state found.")
        else:
            print(f"\n{'NAME':<20} {'LAST RUN':<22} {'STATUS':<12} NOTES")
            print("-" * 100)
            for row in rows:
                last_run = row.get('last_run_at') or '-'
                status = row.get('last_status') or '-'
                notes = row.get('notes') or ''
                print(f"{row['name']:<20} {last_run:<22} {status:<12} {notes}")
        mem.close()
    elif args.command == "link-so":
        result = ResearchMemory.link_primitives_so(output_path=args.output)
        if result["ok"]:
            print(f"Linked {result['objects']} objects → {result['output']} ({result['size_mb']} MB)")
        else:
            print(f"Error: {result['error']}", file=sys.stderr); sys.exit(1)
    elif args.command == "heartbeat":
        mem = ResearchMemory(args.db)
        canonical = mem.worker_heartbeat(args.kernel, current_task=args.task,
                                         process_state=args.state, job_id=args.job)
        print(f"Heartbeat: {canonical} [{args.state}] {args.task[:60]}")
        mem.close()
    elif args.command == "nudge":
        # Nudge = run the SAME gate logic as watchdog.sh, for one job, right now.
        # No duplicate code — just call the watchdog's gate processing inline.
        import subprocess
        job_id = args.id
        print(f"Nudging job #{job_id} through watchdog gate...")
        gate_code = f'''
from pathlib import Path
import sys
BWK_ROOT = Path({str(BWK_ROOT)!r})
sys.path.insert(0, str(BWK_ROOT / 'common' / 'memory'))
script_path = BWK_ROOT / 'common' / 'scripts' / 'gate_process.py'
namespace = {{'__name__': 'factory_brain_nudge', '__file__': str(script_path)}}
exec(compile(script_path.read_text(), str(script_path), 'exec'), namespace)
namespace['gate_process_job']({job_id})
'''
        result = subprocess.run(
            ['python3', '-c', gate_code],
            capture_output=True, text=True, timeout=900,
            env={**os.environ, 'TRANSFORMERS_NO_TF': '1', 'TF_CPP_MIN_LOG_LEVEL': '3'},
        )
        if result.stdout: print(result.stdout.rstrip())
        if result.stderr: print(result.stderr.rstrip(), file=sys.stderr)
        # Show final state
        mem = ResearchMemory(args.db)
        job = mem.get_job(job_id)
        if job:
            print(f"\nFinal: {job['state']} [{job['phase']}]")
        mem.close()
    elif args.command == "maintain":
        cmd_maintain(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
