"""Embedding helpers for ResearchMemory."""
from __future__ import annotations

import os
import struct
from typing import List

_model = None
_tokenizer = None

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
EMBEDDING_DEVICE = "cpu"  # embeddings are cheap, keep GPU free for kernels
MAX_CHUNK_TOKENS = 512


def _load_model():
    """Lazy-load the embedding model once per process."""
    global _model, _tokenizer
    if _model is not None:
        return
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from transformers import AutoModel, AutoTokenizer
    import torch

    print(f"Loading embedding model: {EMBEDDING_MODEL}...", file=os.sys.stderr)
    _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    _model = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    _model.eval()
    _model.to(EMBEDDING_DEVICE)
    print("Model loaded.", file=os.sys.stderr)


def embed_texts(texts: List[str], task: str = "search_document") -> List[List[float]]:
    """Embed a batch of texts. task is 'search_document' for indexing, 'search_query' for queries."""
    _load_model()
    import torch

    prefixed = [f"{task}: {t}" for t in texts]
    encoded = _tokenizer(
        prefixed, padding=True, truncation=True, max_length=MAX_CHUNK_TOKENS,
        return_tensors="pt"
    ).to(EMBEDDING_DEVICE)

    with torch.no_grad():
        output = _model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        embeddings = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single query."""
    return embed_texts([text], task="search_query")[0]


def serialize_f32(vec: List[float]) -> bytes:
    """Pack a float list into bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)
