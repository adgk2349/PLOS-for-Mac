from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import SettingsModel, StartupProfile, WorkMode, WorkspaceResponse, WorkspaceUpdateRequest


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
            self._conn.commit()

        self._bootstrap_defaults()

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

    def upsert_document(self, doc_id: str, path: str, file_type: str, modified_at: float) -> None:
        indexed_at = utc_now().isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO documents(doc_id, path, file_type, modified_at, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                  doc_id=excluded.doc_id,
                  file_type=excluded.file_type,
                  modified_at=excluded.modified_at,
                  indexed_at=excluded.indexed_at
                """,
                (doc_id, path, file_type, modified_at, indexed_at),
            )
            cur.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.commit()

    def insert_chunks(self, doc_id: str, chunks: list[tuple[str, int, str]]) -> None:
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

    def clear_failure(self, path: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM failures WHERE path=?", (path,))
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
