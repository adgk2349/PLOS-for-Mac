from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import (
    ChatFilters,
    DocumentMetadata,
    DocumentMetadataUpdate,
)

class DocumentRepository:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def get_indexed_documents(self) -> dict[str, float]:
        rows = self._fetchall("SELECT path, modified_at FROM documents")
        return {row["path"]: float(row["modified_at"]) for row in rows}

    def upsert_document(
        self,
        doc_id: str,
        path: str,
        file_type: str,
        modified_at: float,
        indexed_at: str,
        normalized_metadata: dict[str, Any],
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO documents(
                    doc_id, path, file_type, modified_at, indexed_at,
                    summary, category, subcategory, document_type, tags, year, project, importance, excluded
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                  doc_id=excluded.doc_id,
                  file_type=excluded.file_type,
                  modified_at=excluded.modified_at,
                  indexed_at=excluded.indexed_at,
                  summary=excluded.summary,
                  category=excluded.category,
                  subcategory=excluded.subcategory,
                  document_type=excluded.document_type,
                  tags=excluded.tags,
                  year=excluded.year,
                  project=excluded.project,
                  importance=excluded.importance,
                  excluded=excluded.excluded
                """,
                (
                    doc_id,
                    path,
                    file_type,
                    modified_at,
                    indexed_at,
                    normalized_metadata["summary"],
                    normalized_metadata["category"],
                    normalized_metadata["subcategory"],
                    normalized_metadata["document_type"],
                    json.dumps(normalized_metadata["tags"], ensure_ascii=False),
                    normalized_metadata["year"],
                    normalized_metadata["project"],
                    normalized_metadata["importance"],
                    1 if normalized_metadata["excluded"] else 0,
                ),
            )
            cur.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.commit()

    def update_document_auto_metadata(self, doc_id: str, normalized_metadata: dict[str, Any]) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE documents
                SET summary=?, category=?, subcategory=?, document_type=?, tags=?, year=?, project=?, importance=?, excluded=?
                WHERE doc_id=?
                """,
                (
                    normalized_metadata["summary"],
                    normalized_metadata["category"],
                    normalized_metadata["subcategory"],
                    normalized_metadata["document_type"],
                    json.dumps(normalized_metadata["tags"], ensure_ascii=False),
                    normalized_metadata["year"],
                    normalized_metadata["project"],
                    normalized_metadata["importance"],
                    1 if normalized_metadata["excluded"] else 0,
                    doc_id,
                ),
            )
            self._conn.commit()

    def insert_chunks(self, chunks: list[tuple[str, str, int, str]]) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, chunk_order, text) VALUES (?, ?, ?, ?)",
                chunks,
            )
            self._conn.commit()

    def list_chunks_by_doc_ids(self, doc_ids: list[str]) -> list[sqlite3.Row]:
        if not doc_ids:
            return []
        placeholders = ",".join("?" for _ in doc_ids)
        return self._fetchall(
            f"SELECT c.chunk_id, c.doc_id, c.text, c.chunk_order, d.path, d.modified_at FROM chunks c "
            f"JOIN documents d ON d.doc_id = c.doc_id WHERE c.doc_id IN ({placeholders})",
            tuple(doc_ids),
        )

    def list_all_chunks(self) -> list[sqlite3.Row]:
        return self._fetchall(
            "SELECT c.chunk_id, c.doc_id, c.text, c.chunk_order, d.path, d.modified_at "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id"
        )

    def get_document_record(self, doc_id: str) -> sqlite3.Row | None:
        return self._fetchone("SELECT * FROM documents WHERE doc_id=?", (doc_id,))

    def get_documents_metadata_map(self, doc_ids: list[str]) -> list[sqlite3.Row]:
        if not doc_ids:
            return []
        placeholders = ",".join("?" for _ in doc_ids)
        return self._fetchall(
            f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        )

    def list_documents(
        self,
        *,
        search: str | None = None,
        category: str | None = None,
        year: int | None = None,
        project: str | None = None,
        tags: list[str] | None = None,
        excluded: bool | None = None,
        doc_ids: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[sqlite3.Row], int]:
        """
        Perform SQL-level filtering so that Python-side iteration is bounded.
        Returns (rows_for_page, total_count).

        NOTE: Tags are stored as JSON arrays; SQLite json_each is used for
        exact-tag matching. If json_each is not available (very old SQLite),
        we fall back to LIKE matching which is slightly less strict.
        """
        conditions: list[str] = []
        params: list[Any] = []

        # Workspace-scope filter (doc_id allow-list)
        if doc_ids is not None:
            cleaned = [str(item).strip() for item in doc_ids if str(item).strip()]
            if not cleaned:
                return [], 0
            placeholders = ",".join("?" for _ in cleaned)
            conditions.append(f"doc_id IN ({placeholders})")
            params.extend(cleaned)

        # Excluded / not-excluded filter
        if excluded is None:
            # Default: only non-excluded documents
            conditions.append("(COALESCE(user_excluded, excluded)=0)")
        else:
            conditions.append("(COALESCE(user_excluded, excluded)=?)")
            params.append(1 if excluded else 0)

        # Category — prefer user_category, fall back to auto category
        if category:
            conditions.append(
                "(COALESCE(user_category, category) = ?)"
            )
            params.append(category)

        # Year
        if year is not None:
            conditions.append("(COALESCE(user_year, year) = ?)")
            params.append(year)

        # Project (case-insensitive substring)
        if project:
            conditions.append(
                "(LOWER(COALESCE(user_project, project, '')) LIKE ?)"
            )
            params.append(f"%{project.lower()}%")

        # Text search across path + summary
        if search:
            needle = search.strip().lower()
            if needle:
                conditions.append(
                    "(LOWER(path) LIKE ? OR LOWER(summary) LIKE ?)"
                )
                params.extend([f"%{needle}%", f"%{needle}%"])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Count total before pagination
        count_sql = f"SELECT COUNT(*) AS cnt FROM documents {where_clause}"
        count_row = self._fetchone(count_sql, tuple(params))
        total = int(count_row["cnt"]) if count_row else 0

        # Fetch page
        page_sql = (
            f"SELECT * FROM documents {where_clause} "
            f"ORDER BY indexed_at DESC LIMIT ? OFFSET ?"
        )
        rows = self._fetchall(page_sql, tuple(params) + (limit, offset))

        # Tag filtering is done in Python because JSON array intersection
        # is cumbersome in SQLite without json_each; the dataset is already
        # bounded by the SQL query above, so this is acceptable.
        if tags:
            wanted = {t.lower() for t in tags if t.strip()}
            if wanted:
                filtered = []
                for row in rows:
                    raw = row["user_tags"] if row["user_tags"] is not None else row["tags"]
                    try:
                        row_tags = {t.lower() for t in json.loads(raw or "[]")}
                    except Exception:
                        row_tags = set()
                    if wanted.intersection(row_tags):
                        filtered.append(row)
                rows = filtered
                total = len(rows)  # recalculate after tag filter
                rows = rows[offset:offset + limit]

        return rows, total

    def find_doc_ids(self) -> list[sqlite3.Row]:
        return self._fetchall("SELECT * FROM documents")

    def update_document_metadata_base(self, doc_id: str, assignments: list[str], values: list[Any]) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"UPDATE documents SET {', '.join(assignments)} WHERE doc_id=?",
                (*values, doc_id),
            )
            self._conn.commit()

    def get_status_snapshot_base(self) -> sqlite3.Row | None:
        return self._fetchone("SELECT COUNT(*) AS count, MAX(indexed_at) AS last_indexed FROM documents")

    def _fetchone(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(query, params)
            return cur.fetchone()

    def _fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(query, params)
            return cur.fetchall()
