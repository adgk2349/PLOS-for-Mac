from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

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

    def with_score(self, score: float) -> "VectorHit":
        """Return a new VectorHit with the given score (immutable update)."""
        return replace(self, score=min(score, 2.0))  # clamp – reranking may add bonuses > 1.0


class VectorStore:
    def __init__(self, db_path: Path, dim: int):
        self._dim = dim
        self._fallback_rows: dict[str, dict[str, Any]] = {}
        self._table = None
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
        else:
            self._conn = None

    @property
    def using_lancedb(self) -> bool:
        return self._table is not None

    def upsert_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self._table is not None:
            chunk_ids = [r["chunk_id"] for r in rows]
            if chunk_ids:
                # Validate each chunk_id against the safe-character regex before
                # embedding it in the LanceDB filter string. LanceDB does not support
                # standard SQL parameter binding, so we sanitise manually.
                safe_quoted = []
                for cid in chunk_ids:
                    if _SAFE_ID_RE.match(cid):
                        safe_quoted.append(f"'{cid}'")
                    # Skip IDs that contain unexpected characters – they cannot
                    # exist in the table anyway (generated internally).
                if safe_quoted:
                    condition = "chunk_id IN ({})".format(", ".join(safe_quoted))
                    self._table.delete(condition)
            self._table.add(rows)
            return

        for row in rows:
            self._fallback_rows[row["chunk_id"]] = row

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
        if self._table is not None:
            results = (
                self._table.search(vector)
                .metric("cosine")
                .limit(limit)
                .to_list()
            )
            hits = [
                VectorHit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    file_path=row["file_path"],
                    text=row["text"],
                    # LanceDB returns cosine *distance* in [0, 2]; convert to similarity.
                    score=float(1.0 - row.get("_distance", 1.0)),
                    modified_at=float(row["modified_at"]),
                )
                for row in results
            ]
            return hits

        if not self._fallback_rows:
            return []

        q = np.array(vector, dtype=np.float32)
        denom_q = np.linalg.norm(q)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._fallback_rows.values():
            v = np.array(row["vector"], dtype=np.float32)
            denom_v = np.linalg.norm(v)
            if denom_q == 0 or denom_v == 0:
                score = 0.0
            else:
                score = float(np.dot(q, v) / (denom_q * denom_v))
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        output = []
        for score, row in scored[:limit]:
            output.append(
                VectorHit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    file_path=row["file_path"],
                    text=row["text"],
                    score=score,
                    modified_at=float(row["modified_at"]),
                )
            )
        return output
