from __future__ import annotations

from datetime import datetime
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import availability differs by environment
    import lancedb
    import pyarrow as pa
except Exception:  # pragma: no cover
    lancedb = None
    pa = None

# Chunk IDs are generated as "<sha1hex>:<int>".
# Only allow characters that appear in that format to prevent filter-string injection.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9:_\-]+$")


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    doc_id: str
    file_path: str
    text: str
    score: float
    modified_at: float
    reliability: float = 1.0

    def with_score(self, score: float, reliability: float | None = None) -> "VectorHit":
        """Return a new VectorHit with the given score (immutable update)."""
        rel = reliability if reliability is not None else self.reliability
        return replace(self, score=min(score, 2.0), reliability=rel)


class VectorStore:
    def __init__(self, db_path: Path, dim: int):
        self._dim = dim
        self._fallback_rows: dict[str, dict[str, Any]] = {}
        self._fallback_memories: dict[str, dict[str, Any]] = {}
        self._table = None
        self._memories_table = None
        if lancedb is not None and pa is not None:
            self._conn = lancedb.connect(str(db_path))
            self._schema = pa.schema(
                [
                    pa.field("chunk_id", pa.string()),
                    pa.field("doc_id", pa.string()),
                    pa.field("file_path", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("modified_at", pa.float64()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                ]
            )
            try:
                self._table = self._conn.open_table("chunks")
            except Exception:
                self._table = self._conn.create_table("chunks", schema=self._schema)

            try:
                self._memories_table = self._conn.open_table("memories")
            except Exception:
                self._memories_table = self._conn.create_table("memories", schema=self._get_memories_schema())
        else:
            self._conn = None

    @property
    def using_lancedb(self) -> bool:
        return self._table is not None

    def _get_memories_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("vector", pa.list_(pa.float32(), self._dim)),
            pa.field("text", pa.string()),
            pa.field("memory_id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("workspace_id", pa.string()),
            pa.field("created_at", pa.string()),
        ])

    @staticmethod
    def _safe_equals_filter(field: str, value: str) -> str | None:
        if not value:
            return None
        if not _SAFE_ID_RE.match(value):
            return None
        return f"{field} = '{value}'"

    @staticmethod
    def _safe_id_list(values: list[str]) -> list[str]:
        return [value for value in values if value and _SAFE_ID_RE.match(value)]

    @staticmethod
    def _to_modified_at(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value or "").strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except Exception:
            pass
        try:
            normalized = text.replace("Z", "+00:00")
            return float(datetime.fromisoformat(normalized).timestamp())
        except Exception:
            return 0.0

    @staticmethod
    def _cosine_similarity(query: np.ndarray, vector: list[float] | np.ndarray) -> float:
        candidate = np.array(vector, dtype=np.float32)
        denom_q = np.linalg.norm(query)
        denom_v = np.linalg.norm(candidate)
        if denom_q == 0 or denom_v == 0:
            return 0.0
        return float(np.dot(query, candidate) / (denom_q * denom_v))

    def upsert_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        if self._table is not None:
            chunk_ids = self._safe_id_list([str(row.get("chunk_id", "")).strip() for row in rows])
            if chunk_ids:
                quoted = ", ".join(f"'{chunk_id}'" for chunk_id in chunk_ids)
                self._table.delete(f"chunk_id IN ({quoted})")
            self._table.add(rows)
            return

        for row in rows:
            chunk_id = str(row.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            self._fallback_rows[chunk_id] = row

    def upsert_memories(self, rows: list[dict[str, Any]]) -> None:
        """Upsert memory vectors. memory_id is the unique key."""
        if not rows:
            return

        if self._memories_table is not None:
            memory_ids = self._safe_id_list([str(row.get("memory_id", "")).strip() for row in rows])
            if memory_ids:
                quoted = ", ".join(f"'{memory_id}'" for memory_id in memory_ids)
                self._memories_table.delete(f"memory_id IN ({quoted})")
            self._memories_table.add(rows)
            return

        for row in rows:
            memory_id = str(row.get("memory_id", "")).strip()
            if not memory_id:
                continue
            self._fallback_memories[memory_id] = row

    def delete_memories(self, memory_ids: list[str]) -> None:
        """Delete memory vectors by memory_id."""
        safe_ids = self._safe_id_list([str(item).strip() for item in (memory_ids or [])])
        if not safe_ids:
            return

        if self._memories_table is not None:
            quoted = ", ".join(f"'{memory_id}'" for memory_id in safe_ids)
            self._memories_table.delete(f"memory_id IN ({quoted})")
            return

        for memory_id in safe_ids:
            self._fallback_memories.pop(memory_id, None)

    def search_memories(
        self,
        query_vector: list[float],
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
        limit: int = 5,
    ) -> list[VectorHit]:
        """
        Search for relevant past memories.
        If session_id/workspace_id are provided, it acts as Tier 1/2 search.
        If None, it's a global Tier 3 search.
        """
        max_limit = max(1, int(limit))
        if self._memories_table is not None:
            qb = self._memories_table.search(query_vector).metric("cosine")

            filters: list[str] = []
            if session_id:
                clause = self._safe_equals_filter("session_id", session_id)
                if clause is None:
                    return []
                filters.append(clause)
            if workspace_id:
                clause = self._safe_equals_filter("workspace_id", workspace_id)
                if clause is None:
                    return []
                filters.append(clause)

            if filters:
                qb = qb.where(" AND ".join(filters))

            rows = qb.limit(max_limit).to_list()
            return [
                VectorHit(
                    chunk_id=str(row.get("memory_id", "")),
                    doc_id=str(row.get("session_id", "")),
                    file_path="",
                    text=str(row.get("text", "")),
                    score=float(1.0 - row.get("_distance", 1.0)),
                    modified_at=self._to_modified_at(row.get("created_at")),
                )
                for row in rows
            ]

        query = np.array(query_vector, dtype=np.float32)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._fallback_memories.values():
            if session_id and row.get("session_id") != session_id:
                continue
            if workspace_id and row.get("workspace_id") != workspace_id:
                continue
            vector = row.get("vector")
            if not vector:
                continue
            score = self._cosine_similarity(query, vector)
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            VectorHit(
                chunk_id=str(row.get("memory_id", "")),
                doc_id=str(row.get("session_id", "")),
                file_path="",
                text=str(row.get("text", "")),
                score=score,
                modified_at=self._to_modified_at(row.get("created_at")),
            )
            for score, row in scored[:max_limit]
        ]

    def search_memories_fts(self, query: str, limit: int = 5) -> list[VectorHit]:
        """Keyword search across memories."""
        if self._memories_table is not None:
            try:
                results = self._memories_table.search(query).limit(limit).to_list()
                return [
                    VectorHit(
                        chunk_id=str(row.get("memory_id", "")),
                        doc_id=str(row.get("session_id", "")),
                        file_path="",
                        text=str(row.get("text", "")),
                        score=float(row.get("_score", 0.0)),
                        modified_at=self._to_modified_at(row.get("created_at")),
                    )
                    for row in results
                ]
            except Exception:
                return []
        return []

    def search_memories_hybrid(
        self,
        query_text: str,
        query_vector: list[float],
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
        limit: int = 5,
    ) -> list[VectorHit]:
        """Hybrid search across memories using RRF."""
        dense = self.search_memories(query_vector, session_id=session_id, workspace_id=workspace_id, limit=limit * 2)
        sparse = self.search_memories_fts(query_text, limit=limit * 2)
        if session_id:
            sparse = [hit for hit in sparse if str(hit.doc_id or "") == str(session_id)]
        
        if not sparse: return dense[:limit]
        if not dense: return sparse[:limit]

        k = 60
        rrf_scores: dict[str, float] = {}
        hit_map: dict[str, VectorHit] = {}

        for rank, hit in enumerate(dense, 1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (1.0 / (k + rank))
            hit_map[hit.chunk_id] = hit
        for rank, hit in enumerate(sparse, 1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (1.0 / (k + rank))
            if hit.chunk_id not in hit_map: hit_map[hit.chunk_id] = hit

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [hit_map[cid] for cid in sorted_ids[:limit]]

    def delete_doc(self, doc_id: str) -> None:
        if self._table is not None:
            # doc_id is also internally generated (SHA1 hex), so safe to embed.
            if _SAFE_ID_RE.match(doc_id):
                self._table.delete(f"doc_id = '{doc_id}'")
            return

        keys_to_delete = [k for k, v in self._fallback_rows.items() if v["doc_id"] == doc_id]
        for key in keys_to_delete:
            del self._fallback_rows[key]

    def search(self, vector: list[float], limit: int) -> list[VectorHit]:
        """Standard vector (Dense) search."""
        if self._table is not None:
            results = (
                self._table.search(vector)
                .metric("cosine")
                .limit(limit)
                .to_list()
            )
            return [
                VectorHit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    file_path=row["file_path"],
                    text=row["text"],
                    score=float(1.0 - row.get("_distance", 1.0)),
                    modified_at=float(row["modified_at"]),
                )
                for row in results
            ]
        return self._search_fallback(vector, limit)

    def search_fts(self, query: str, limit: int) -> list[VectorHit]:
        """Sparse (BM25) search using Full-Text Search index."""
        if self._table is not None:
            try:
                # FTS search in LanceDB
                results = (
                    self._table.search(query)
                    .limit(limit)
                    .to_list()
                )
                return [
                    VectorHit(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        file_path=row["file_path"],
                        text=row["text"],
                        # FTS scores are BM25 scores (higher is better). 
                        # We normalize them relative to other results or just use them as-is for RRF.
                        score=float(row.get("_score", 0.0)),
                        modified_at=float(row["modified_at"]),
                    )
                    for row in results
                ]
            except Exception as e:
                logger.warning("FTS search failed (index might not exist): %s", e)
                return []
        return []

    def search_hybrid(self, query_text: str, query_vector: list[float], limit: int) -> list[VectorHit]:
        """Hybrid search combining Dense (Vector) and Sparse (FTS) results using RRF."""
        dense_hits = self.search(query_vector, limit=limit * 2)
        sparse_hits = self.search_fts(query_text, limit=limit * 2)

        if not sparse_hits:
            return dense_hits[:limit]
        if not dense_hits:
            return sparse_hits[:limit]

        # Reciprocal Rank Fusion (RRF)
        # score = sum(1 / (k + rank))
        k = 60
        rrf_scores: dict[str, float] = {}
        hit_map: dict[str, VectorHit] = {}

        for rank, hit in enumerate(dense_hits, 1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (1.0 / (k + rank))
            hit_map[hit.chunk_id] = hit

        for rank, hit in enumerate(sparse_hits, 1):
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (1.0 / (k + rank))
            # If hit was only in sparse, keep it but use a lower reliability/score for safety
            if hit.chunk_id not in hit_map:
                hit_map[hit.chunk_id] = hit

        # Sorting by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_hits = []
        for cid in sorted_ids[:limit]:
            hit = hit_map[cid]
            # Normalize RRF score to a [0, 1] range for compatibility with thresholds
            # Max RRF for k=60 rank=1 is ~0.0163. Two hits could be 0.032.
            # We'll just preserve the original Dense score if it exists, as it's our similarity anchor.
            final_hits.append(hit)

        return final_hits

    def create_fts_index(self) -> None:
        """Create or update Full-Text Search index on the 'text' column for all tables."""
        if self._table is not None:
            try:
                self._table.create_index(column="text", index_type="fts", replace=True)
                logger.info("FTS index created/updated on 'chunks.text'.")
            except Exception as e:
                logger.error("Failed to create FTS index for chunks: %s", e)
        
        if self._memories_table is not None:
            try:
                self._memories_table.create_index(column="text", index_type="fts", replace=True)
                logger.info("FTS index created/updated on 'memories.text'.")
            except Exception as e:
                logger.warning("Failed to create FTS index for memories: %s", e)

    def _search_fallback(self, vector: list[float], limit: int) -> list[VectorHit]:
        if not self._fallback_rows:
            return []
        query = np.array(vector, dtype=np.float32)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._fallback_rows.values():
            score = self._cosine_similarity(query, row["vector"])
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            VectorHit(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                file_path=row["file_path"],
                text=row["text"],
                score=score,
                modified_at=float(row["modified_at"]),
            )
            for score, row in scored[:limit]
        ]
