"""Text embeddings for the semantic cache.

Default: a dependency-free hashed n-gram embedder (fast, deterministic, good
at catching paraphrases that share wording). If `sentence-transformers` is
installed and SWITCHYARD_EMBEDDER=st, a real sentence-embedding model is used
instead, which also catches paraphrases with different wording.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"a", "an", "the", "is", "are", "of", "to", "in", "on", "for", "and", "or",
         "me", "you", "please", "can", "could", "would", "what", "whats", "how", "i", "do"}


class HashEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, s: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOP]
        for w in words:
            i, s = self._bucket("w:" + w)
            v[i] += s * 2.0
            padded = f"#{w}#"
            for k in range(len(padded) - 2):
                i, s = self._bucket("c:" + padded[k:k + 3])
                v[i] += s * 0.5
        for a, b in zip(words, words[1:], strict=False):
            i, s = self._bucket(f"b:{a}_{b}")
            v[i] += s
        n = np.linalg.norm(v)
        return v / n if n else v


class STEmbedder:  # pragma: no cover - optional heavy dependency
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(model)

    def embed(self, text: str) -> np.ndarray:
        return self.m.encode(text, normalize_embeddings=True).astype(np.float32)


def build_embedder():
    if os.environ.get("SWITCHYARD_EMBEDDER") == "st":
        return STEmbedder()
    return HashEmbedder()
