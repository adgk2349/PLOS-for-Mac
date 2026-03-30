from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingProfile:
    name: str
    dim: int


class EmbeddingService:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._model = None
        # NOTE:
        # Indexing stability is prioritized over peak embedding throughput.
        # Apple MPS has caused hard process aborts in torch ops on some macOS builds.
        # Default to CPU unless explicitly overridden.
        requested_device = str(os.getenv("LOCAL_AI_EMBED_DEVICE", "cpu") or "cpu").strip().lower()
        allow_unsafe_mps = str(os.getenv("LOCAL_AI_ALLOW_MPS_UNSAFE", "0") or "0").strip() == "1"
        if requested_device == "mps" and not allow_unsafe_mps:
            logger.warning(
                "LOCAL_AI_EMBED_DEVICE=mps requested but disabled for stability. "
                "Using cpu. Set LOCAL_AI_ALLOW_MPS_UNSAFE=1 to force mps."
            )
            requested_device = "cpu"
        self._device = requested_device if requested_device in {"cpu", "mps", "cuda"} else "cpu"
        
    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            # all-MiniLM-L6-v2 produces 384-dimensional embeddings (perfect match for existing DB)
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device=self._device)
            logger.info(
                "Loaded semantic embedding model (sentence-transformers/all-MiniLM-L6-v2) lazily on device=%s.",
                self._device,
            )
        except ImportError:
            logger.warning("sentence-transformers is not installed. Using blake2b hash-based fallback embeddings.")

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
        self._ensure_model()
        if self._model is not None:
            embeddings = self._model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
            return [vec.tolist() for vec in embeddings]
        return [self._hash_to_vector(doc) for doc in docs]

    def embed_query(self, query: str) -> list[float]:
        self._ensure_model()
        if self._model is not None:
            embedding = self._model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
            return embedding.tolist()
        return self._hash_to_vector(query)


class RerankerService:
    def __init__(self, en_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", multi_model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._en_model_name = en_model_name
        self._multi_model_name = multi_model_name
        self._en_model = None
        self._multi_model = None
        requested_device = str(
            os.getenv("LOCAL_AI_RERANK_DEVICE", os.getenv("LOCAL_AI_EMBED_DEVICE", "cpu")) or "cpu"
        ).strip().lower()
        allow_unsafe_mps = str(os.getenv("LOCAL_AI_ALLOW_MPS_UNSAFE", "0") or "0").strip() == "1"
        if requested_device == "mps" and not allow_unsafe_mps:
            logger.warning(
                "MPS reranker requested but disabled for stability. "
                "Using cpu. Set LOCAL_AI_ALLOW_MPS_UNSAFE=1 to force mps."
            )
            requested_device = "cpu"
        self._device = requested_device if requested_device in {"cpu", "mps", "cuda"} else "cpu"
        
        # Lazy load: Both models now load on demand to speed up initial sidecar startup.

    def _load_en_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._en_model = CrossEncoder(self._en_model_name, device=self._device)
            logger.info("Loaded English neural reranker (%s) on device=%s.", self._en_model_name, self._device)
        except Exception as e:
            logger.warning(f"Could not load English reranker: {e}")

    def _load_multi_model(self):
        if self._multi_model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            # Lazy load: BAAI/bge-reranker-v2-m3 is fairly large (~2.2GB)
            self._multi_model = CrossEncoder(self._multi_model_name, device=self._device)
            logger.info("Loaded Multilingual neural reranker (%s) on device=%s.", self._multi_model_name, self._device)
        except Exception as e:
            logger.warning(f"Could not load Multilingual reranker: {e}")

    @property
    def is_available(self) -> bool:
        return self._en_model is not None or self._multi_model is not None

    def rerank(self, query: str, texts: list[str], language: str = "auto") -> list[float]:
        if not texts:
            return []
        
        # Ensure appropriate model is loaded
        if language == "en":
            self._load_en_model()
        else:
            self._load_multi_model()
            if self._multi_model is None:
                self._load_en_model()

        # Select model based on language
        # Language "ko", "ja" or ambiguous "auto" use the multilingual specialized model
        # Language "en" uses the lightweight English model
        model = None
        lang = (language or "auto").lower().strip()
        
        if lang == "en":
            model = self._en_model
        else:
            self._load_multi_model()
            model = self._multi_model or self._en_model # Fallback to EN if Multi fails

        if model is None:
            return [0.0] * len(texts)
        
        # CrossEncoder expects a list of (Query, Passage) pairs
        pairs = [[query, text] for text in texts]
        scores = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        return scores.tolist()
