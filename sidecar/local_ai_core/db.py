from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    ChatFilters,
    DocumentMetadata,
    DocumentMetadataUpdate,
    SettingsModel,
    StartupProfile,
    WorkMode,
    WorkspaceResponse,
    WorkspaceUpdateRequest,
)


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class Database:
    def __init__(self, sqlite_path: Path):
        self._path = sqlite_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS workspace (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    included_paths TEXT NOT NULL,
                    excluded_paths TEXT NOT NULL,
                    startup_profile TEXT NOT NULL,
                    default_mode TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    file_type TEXT NOT NULL,
                    modified_at REAL NOT NULL,
                    indexed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_order INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    reason TEXT NOT NULL,
                    last_attempt_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS external_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    sent_chars INTEGER NOT NULL,
                    approved_by_user INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                );
                """
            )

            self._ensure_document_columns(cur)
            self._conn.commit()

        self._bootstrap_defaults()

    def _ensure_document_columns(self, cur: sqlite3.Cursor) -> None:
        cur.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cur.fetchall()}

        additions = [
            ("summary", "TEXT NOT NULL DEFAULT ''"),
            ("category", "TEXT NOT NULL DEFAULT '참고자료'"),
            ("subcategory", "TEXT NOT NULL DEFAULT ''"),
            ("document_type", "TEXT NOT NULL DEFAULT ''"),
            ("tags", "TEXT NOT NULL DEFAULT '[]'"),
            ("year", "INTEGER"),
            ("project", "TEXT"),
            ("importance", "REAL NOT NULL DEFAULT 0.5"),
            ("excluded", "INTEGER NOT NULL DEFAULT 0"),
            ("user_category", "TEXT"),
            ("user_subcategory", "TEXT"),
            ("user_document_type", "TEXT"),
            ("user_tags", "TEXT"),
            ("user_year", "INTEGER"),
            ("user_project", "TEXT"),
            ("user_importance", "REAL"),
            ("user_excluded", "INTEGER"),
        ]
        for name, ddl in additions:
            if name not in columns:
                cur.execute(f"ALTER TABLE documents ADD COLUMN {name} {ddl}")

    def _bootstrap_defaults(self) -> None:
        if self._fetchone("SELECT id FROM workspace WHERE id=1") is None:
            self.update_workspace(
                WorkspaceUpdateRequest(
                    included_paths=[],
                    excluded_paths=[],
                    startup_profile=StartupProfile.RECOMMENDED,
                    default_mode=WorkMode.GENERAL,
                )
            )
        if self._fetchone("SELECT id FROM settings WHERE id=1") is None:
            self.update_settings(SettingsModel())

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

    def update_workspace(self, request: WorkspaceUpdateRequest) -> WorkspaceResponse:
        now = utc_now().isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO workspace(id, included_paths, excluded_paths, startup_profile, default_mode, updated_at)
                VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  included_paths=excluded.included_paths,
                  excluded_paths=excluded.excluded_paths,
                  startup_profile=excluded.startup_profile,
                  default_mode=excluded.default_mode,
                  updated_at=excluded.updated_at
                """,
                (
                    json.dumps(request.included_paths),
                    json.dumps(request.excluded_paths),
                    request.startup_profile.value,
                    request.default_mode.value,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_workspace()

    def get_workspace(self) -> WorkspaceResponse:
        row = self._fetchone("SELECT * FROM workspace WHERE id=1")
        if row is None:
            raise RuntimeError("workspace not initialized")
        return WorkspaceResponse(
            included_paths=json.loads(row["included_paths"]),
            excluded_paths=json.loads(row["excluded_paths"]),
            startup_profile=row["startup_profile"],
            default_mode=row["default_mode"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def update_settings(self, settings: SettingsModel) -> SettingsModel:
        now = utc_now().isoformat()
        payload = settings.model_dump_json()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO settings(id, payload, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  payload=excluded.payload,
                  updated_at=excluded.updated_at
                """,
                (payload, now),
            )
            self._conn.commit()
        return self.get_settings()

    def get_settings(self) -> SettingsModel:
        row = self._fetchone("SELECT payload FROM settings WHERE id=1")
        if row is None:
            raise RuntimeError("settings not initialized")
        return SettingsModel.model_validate_json(row["payload"])

    def get_indexed_documents(self) -> dict[str, float]:
        rows = self._fetchall("SELECT path, modified_at FROM documents")
        return {row["path"]: float(row["modified_at"]) for row in rows}

    def upsert_document(
        self,
        doc_id: str,
        path: str,
        file_type: str,
        modified_at: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        indexed_at = utc_now().isoformat()
        normalized = self._normalize_metadata(metadata)
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
                    normalized["summary"],
                    normalized["category"],
                    normalized["subcategory"],
                    normalized["document_type"],
                    json.dumps(normalized["tags"], ensure_ascii=False),
                    normalized["year"],
                    normalized["project"],
                    normalized["importance"],
                    1 if normalized["excluded"] else 0,
                ),
            )
            cur.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.commit()

    def update_document_auto_metadata(self, doc_id: str, metadata: dict[str, Any]) -> DocumentMetadata | None:
        normalized = self._normalize_metadata(metadata)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE documents
                SET summary=?, category=?, subcategory=?, document_type=?, tags=?, year=?, project=?, importance=?, excluded=?
                WHERE doc_id=?
                """,
                (
                    normalized["summary"],
                    normalized["category"],
                    normalized["subcategory"],
                    normalized["document_type"],
                    json.dumps(normalized["tags"], ensure_ascii=False),
                    normalized["year"],
                    normalized["project"],
                    normalized["importance"],
                    1 if normalized["excluded"] else 0,
                    doc_id,
                ),
            )
            self._conn.commit()
        return self.get_document_metadata(doc_id)

    def insert_chunks(self, doc_id: str, chunks: list[tuple[str, str, int, str]]) -> None:
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

    def get_document_record(self, doc_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
        if not row:
            return None
        return self._row_to_raw_dict(row)

    def get_document_metadata(self, doc_id: str) -> DocumentMetadata | None:
        row = self._fetchone("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
        if row is None:
            return None
        return DocumentMetadata(**self._row_to_effective_dict(row))

    def get_documents_metadata_map(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self._fetchall(
            f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        )
        return {row["doc_id"]: self._row_to_effective_dict(row) for row in rows}

    def list_documents(
        self,
        *,
        search: str | None = None,
        filters: ChatFilters | None = None,
        included_paths: list[str] | None = None,
        excluded_paths: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[DocumentMetadata], int]:
        rows = self._fetchall("SELECT * FROM documents ORDER BY indexed_at DESC")
        filtered = [self._row_to_effective_dict(row) for row in rows]
        filtered = self._apply_doc_filters(
            filtered,
            search=search,
            filters=filters,
            included_paths=included_paths,
            excluded_paths=excluded_paths,
        )

        total = len(filtered)
        page = filtered[offset : offset + limit]
        return [DocumentMetadata(**item) for item in page], total

    def find_doc_ids(
        self,
        *,
        filters: ChatFilters | None,
        search: str | None = None,
        included_paths: list[str] | None = None,
        excluded_paths: list[str] | None = None,
    ) -> set[str]:
        rows = self._fetchall("SELECT * FROM documents")
        effective = [self._row_to_effective_dict(row) for row in rows]
        filtered = self._apply_doc_filters(
            effective,
            search=search,
            filters=filters,
            included_paths=included_paths,
            excluded_paths=excluded_paths,
        )
        return {item["doc_id"] for item in filtered}

    def update_document_metadata(
        self,
        doc_id: str,
        payload: DocumentMetadataUpdate,
    ) -> DocumentMetadata:
        row = self._fetchone("SELECT doc_id FROM documents WHERE doc_id=?", (doc_id,))
        if row is None:
            raise KeyError(f"document not found: {doc_id}")

        assignments: list[str] = []
        values: list[Any] = []
        provided = payload.model_fields_set

        mapping = {
            "category": "user_category",
            "subcategory": "user_subcategory",
            "document_type": "user_document_type",
            "tags": "user_tags",
            "year": "user_year",
            "project": "user_project",
            "importance": "user_importance",
            "excluded": "user_excluded",
        }

        for key, column in mapping.items():
            if key not in provided:
                continue

            value = getattr(payload, key)
            if key == "tags":
                value = json.dumps(value, ensure_ascii=False) if value is not None else None
            if key == "excluded" and value is not None:
                value = 1 if bool(value) else 0
            assignments.append(f"{column}=?")
            values.append(value)

        if assignments:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    f"UPDATE documents SET {', '.join(assignments)} WHERE doc_id=?",
                    (*values, doc_id),
                )
                self._conn.commit()

        metadata = self.get_document_metadata(doc_id)
        if metadata is None:
            raise KeyError(f"document not found after update: {doc_id}")
        return metadata

    def record_failure(self, path: str, reason: str) -> None:
        now = utc_now().isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO failures(path, reason, last_attempt_at)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                  reason=excluded.reason,
                  last_attempt_at=excluded.last_attempt_at
                """,
                (path, reason, now),
            )
            self._conn.commit()

    def delete_documents_by_paths(self, paths: list[str]) -> list[str]:
        if not paths:
            return []
        placeholders = ",".join("?" for _ in paths)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT doc_id, path FROM documents WHERE path IN ({placeholders})",
                tuple(paths),
            )
            rows = cur.fetchall()
            if not rows:
                return []

            cur.execute(
                f"DELETE FROM documents WHERE path IN ({placeholders})",
                tuple(paths),
            )
            cur.execute(
                f"DELETE FROM failures WHERE path IN ({placeholders})",
                tuple(paths),
            )
            self._conn.commit()
        return [str(row["doc_id"]) for row in rows]

    def clear_failure(self, path: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM failures WHERE path=?", (path,))
            self._conn.commit()

    def clear_all_failures(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM failures")
            self._conn.commit()

    def list_failures(self) -> list[dict[str, Any]]:
        rows = self._fetchall("SELECT path, reason, last_attempt_at FROM failures ORDER BY last_attempt_at DESC")
        return [
            {
                "path": row["path"],
                "reason": row["reason"],
                "last_attempt_at": datetime.fromisoformat(row["last_attempt_at"]),
            }
            for row in rows
        ]

    def record_external_call(self, provider: str, sent_chars: int, approved_by_user: bool) -> datetime:
        timestamp = utc_now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO external_calls(provider, sent_chars, approved_by_user, timestamp) VALUES (?, ?, ?, ?)",
                (provider, sent_chars, 1 if approved_by_user else 0, timestamp.isoformat()),
            )
            self._conn.commit()
        return timestamp

    def get_status_snapshot(self) -> dict[str, Any]:
        row = self._fetchone("SELECT COUNT(*) AS count, MAX(indexed_at) AS last_indexed FROM documents")
        ext = self._fetchone("SELECT provider, timestamp FROM external_calls ORDER BY id DESC LIMIT 1")
        workspace = self.get_workspace()
        return {
            "indexed_docs": int(row["count"]) if row else 0,
            "last_indexed_at": row["last_indexed"] if row and row["last_indexed"] else None,
            "latest_external_call": {
                "provider": ext["provider"],
                "timestamp": ext["timestamp"],
            }
            if ext
            else None,
            "included_paths": workspace.included_paths,
        }

    @staticmethod
    def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        payload = metadata or {}
        tags = payload.get("tags")
        if not isinstance(tags, list):
            tags = []
        tags = [str(tag).strip() for tag in tags if str(tag).strip()][:8]
        importance = payload.get("importance", 0.5)
        try:
            importance = max(0.0, min(1.0, float(importance)))
        except Exception:
            importance = 0.5
        return {
            "summary": str(payload.get("summary") or "")[:260],
            "category": str(payload.get("category") or "참고자료"),
            "subcategory": str(payload.get("subcategory") or "")[:40],
            "document_type": str(payload.get("document_type") or "")[:40],
            "tags": tags,
            "year": payload.get("year"),
            "project": str(payload.get("project") or "")[:48] or None,
            "importance": importance,
            "excluded": bool(payload.get("excluded", False)),
        }

    def _apply_doc_filters(
        self,
        rows: list[dict[str, Any]],
        *,
        search: str | None,
        filters: ChatFilters | None,
        included_paths: list[str] | None = None,
        excluded_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []

        result = rows
        if included_paths is not None:
            result = self._apply_workspace_scope(
                result,
                included_paths=included_paths,
                excluded_paths=excluded_paths or [],
            )
        if search:
            needle = search.strip().lower()
            if needle:
                result = [
                    row
                    for row in result
                    if needle in row["path"].lower()
                    or needle in row["summary"].lower()
                    or needle in " ".join(row["tags"]).lower()
                ]

        if filters is None:
            return result

        if filters.category:
            result = [row for row in result if row["category"] == filters.category]

        if filters.year is not None:
            result = [row for row in result if row.get("year") == filters.year]

        if filters.project:
            needle = filters.project.lower()
            result = [row for row in result if (row.get("project") or "").lower().find(needle) >= 0]

        if filters.tags:
            wanted = {tag.lower() for tag in filters.tags if tag.strip()}
            if wanted:
                result = [
                    row
                    for row in result
                    if wanted.intersection({tag.lower() for tag in row.get("tags", [])})
                ]

        if filters.excluded is not None:
            result = [row for row in result if bool(row.get("excluded", False)) == bool(filters.excluded)]
        else:
            result = [row for row in result if not bool(row.get("excluded", False))]

        return result

    @staticmethod
    def _apply_workspace_scope(
        rows: list[dict[str, Any]],
        *,
        included_paths: list[str],
        excluded_paths: list[str],
    ) -> list[dict[str, Any]]:
        included_roots = [Database._normalize_path(path) for path in included_paths if str(path).strip()]
        excluded_roots = [Database._normalize_path(path) for path in excluded_paths if str(path).strip()]
        if not included_roots:
            return []

        scoped: list[dict[str, Any]] = []
        for row in rows:
            candidate = Database._normalize_path(str(row.get("path") or ""))
            if not candidate:
                continue
            if any(Database._is_same_or_child(candidate, excluded) for excluded in excluded_roots):
                continue
            if any(Database._is_same_or_child(candidate, included) for included in included_roots):
                scoped.append(row)
        return scoped

    @staticmethod
    def _normalize_path(path: str) -> Path | None:
        stripped = (path or "").strip()
        if not stripped:
            return None
        try:
            return Path(stripped).expanduser().resolve(strict=False)
        except Exception:
            return None

    @staticmethod
    def _is_same_or_child(path: Path, root: Path) -> bool:
        return path == root or root in path.parents

    @staticmethod
    def _row_to_raw_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {key: row[key] for key in row.keys()}

    @staticmethod
    def _parse_tags(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(tag) for tag in raw]
        try:
            value = json.loads(raw)
            if isinstance(value, list):
                return [str(tag) for tag in value]
        except Exception:
            pass
        return []

    @staticmethod
    def _row_to_effective_dict(row: sqlite3.Row) -> dict[str, Any]:
        category = row["user_category"] if row["user_category"] is not None else row["category"]
        subcategory = row["user_subcategory"] if row["user_subcategory"] is not None else row["subcategory"]
        document_type = row["user_document_type"] if row["user_document_type"] is not None else row["document_type"]
        year = row["user_year"] if row["user_year"] is not None else row["year"]
        project = row["user_project"] if row["user_project"] is not None else row["project"]
        importance = row["user_importance"] if row["user_importance"] is not None else row["importance"]
        excluded = row["user_excluded"] if row["user_excluded"] is not None else row["excluded"]

        tags_raw = row["user_tags"] if row["user_tags"] is not None else row["tags"]
        tags = Database._parse_tags(tags_raw)

        modified = datetime.fromtimestamp(float(row["modified_at"]), tz=timezone.utc)
        indexed = datetime.fromisoformat(row["indexed_at"])
        return {
            "doc_id": row["doc_id"],
            "path": row["path"],
            "file_type": row["file_type"],
            "modified_at": modified,
            "indexed_at": indexed,
            "summary": row["summary"] or "",
            "category": category or "참고자료",
            "subcategory": subcategory or "",
            "document_type": document_type or "",
            "tags": tags,
            "year": year,
            "project": project,
            "importance": float(importance or 0.5),
            "excluded": bool(excluded or 0),
        }
