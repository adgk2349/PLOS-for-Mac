from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EmbeddingProfile:
    name: str
    dim: int


class EmbeddingService:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_to_vector(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dim

        vector = np.zeros(self.dim, dtype=np.float32)
        words = text.split()
        if not words:
            words = [text]

        for token in words:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = (digest[5] / 255.0) + 0.5
            vector[idx] += sign * weight

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.astype(np.float32).tolist()

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(doc) for doc in docs]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_to_vector(query)
