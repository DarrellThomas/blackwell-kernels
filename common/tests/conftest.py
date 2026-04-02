from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _hashed_embeddings(dim: int):
    def embed_texts(texts: list[str], task: str = "search_document") -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * dim
            for token in re.findall(r"[a-z0-9_]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                idx = int.from_bytes(digest, "little") % dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    return embed_texts


@pytest.fixture
def stub_memory_embeddings(monkeypatch):
    from common.memory import factory_brain as fb

    monkeypatch.setattr(fb, 'embed_texts', _hashed_embeddings(fb.EMBEDDING_DIM))
    return fb
