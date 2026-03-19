from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from ..models import (
    MemoryClearScope,
    EpisodicMemoryEvent,
    PinnedMemoryItem,
    SessionMemoryItem,
    UserPreferenceItem,
    WorkspaceMemoryItem,
)

class MemoryRepository:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def write_session_memory(
        self,
        *,
        item_id: str,
        session_id: str,
        key: str,
        value_json: str,
        created_at: str,
        updated_at: str,
        expires_at: str | None,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO session_memory(id, session_id, key, value_json, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (item_id, session_id, key, value_json, created_at, updated_at, expires_at),
            )
            self._conn.commit()

    def get_session_memory(self, session_id: str, limit: int) -> list[sqlite3.Row]:
        return self._fetchall(
            """
            SELECT * FROM session_memory
            WHERE session_id=?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        )

    def delete_expired_session_memory(self, now_iso: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "DELETE FROM session_memory WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now_iso,),
            )
            self._conn.commit()

    def get_session_memory_ids(self, session_id: str) -> list[sqlite3.Row]:
        return self._fetchall(
            "SELECT id FROM session_memory WHERE session_id=? ORDER BY updated_at DESC",
            (session_id,),
        )

    def delete_session_memory_by_ids(self, ids: list[str]) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executemany("DELETE FROM session_memory WHERE id=?", [(item_id,) for item_id in ids])
            self._conn.commit()

    def clear_session_memory(self, session_id: str | None = None) -> int:
        with self._lock:
            cur = self._conn.cursor()
            if session_id:
                cur.execute("DELETE FROM session_memory WHERE session_id=?", (session_id,))
            else:
                cur.execute("DELETE FROM session_memory")
            count = cur.rowcount if cur.rowcount is not None else 0
            self._conn.commit()
        return max(0, int(count))

    def get_workspace_memory_existing(self, workspace_id: str, memory_type: str, key: str, source: str) -> sqlite3.Row | None:
        return self._fetchone(
            """
            SELECT id, created_at FROM workspace_memory
            WHERE workspace_id=? AND memory_type=? AND key=? AND source=?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (workspace_id, memory_type, key, source),
        )

    def upsert_workspace_memory(
        self,
        *,
        id: str,
        workspace_id: str,
        memory_type: str,
        key: str,
        value_json: str,
        confidence: float,
        source: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO workspace_memory(id, workspace_id, memory_type, key, value_json, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  value_json=excluded.value_json,
                  confidence=excluded.confidence,
                  updated_at=excluded.updated_at
                """,
                (id, workspace_id, memory_type, key, value_json, confidence, source, created_at, updated_at),
            )
            self._conn.commit()

    def get_workspace_memory(self, workspace_id: str, min_confidence: float, limit: int) -> list[sqlite3.Row]:
        return self._fetchall(
            """
            SELECT * FROM workspace_memory
            WHERE workspace_id=? AND confidence>=?
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            (workspace_id, min_confidence, limit),
        )

    def clear_workspace_memory(self, workspace_id: str | None = None, inferred_only: bool = False) -> int:
        with self._lock:
            cur = self._conn.cursor()
            if workspace_id and inferred_only:
                cur.execute("DELETE FROM workspace_memory WHERE workspace_id=? AND source='inferred'", (workspace_id,))
            elif workspace_id:
                cur.execute("DELETE FROM workspace_memory WHERE workspace_id=?", (workspace_id,))
            elif inferred_only:
                cur.execute("DELETE FROM workspace_memory WHERE source='inferred'")
            else:
                cur.execute("DELETE FROM workspace_memory")
            count = cur.rowcount if cur.rowcount is not None else 0
            self._conn.commit()
        return max(0, int(count))

    def get_user_preference_existing(self, key: str, source: str) -> sqlite3.Row | None:
        return self._fetchone(
            "SELECT id, created_at FROM user_preferences WHERE key=? AND source=? ORDER BY updated_at DESC LIMIT 1",
            (key, source),
        )

    def upsert_user_preference(
        self,
        *,
        id: str,
        key: str,
        value_json: str,
        confidence: float,
        source: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO user_preferences(id, key, value_json, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  value_json=excluded.value_json,
                  confidence=excluded.confidence,
                  updated_at=excluded.updated_at
                """,
                (id, key, value_json, confidence, source, created_at, updated_at),
            )
            self._conn.commit()

    def list_user_preferences(self) -> list[sqlite3.Row]:
        return self._fetchall("SELECT * FROM user_preferences ORDER BY updated_at DESC")

    def clear_user_preferences(self, inferred_only: bool = False) -> int:
        with self._lock:
            cur = self._conn.cursor()
            if inferred_only:
                cur.execute("DELETE FROM user_preferences WHERE source='inferred'")
            else:
                cur.execute("DELETE FROM user_preferences")
            count = cur.rowcount if cur.rowcount is not None else 0
            self._conn.commit()
        return max(0, int(count))

    def insert_episodic_memory(
        self,
        *,
        id: str,
        workspace_id: str | None,
        event_type: str,
        summary: str,
        related_file_ids: str,
        related_action_ids: str,
        metadata_json: str,
        importance: float,
        created_at: str,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO episodic_memory(id, workspace_id, event_type, summary, related_file_ids, related_action_ids, metadata_json, importance, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (id, workspace_id, event_type, summary, related_file_ids, related_action_ids, metadata_json, importance, created_at),
            )
            self._conn.commit()

    def get_relevant_episodic_memory(self, workspace_id: str | None, limit: int) -> list[sqlite3.Row]:
        return self._fetchall(
            """
            SELECT * FROM episodic_memory
            WHERE (? IS NULL OR workspace_id=? OR workspace_id IS NULL)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (workspace_id, workspace_id, limit),
        )

    def list_recent_episodic_memory(self, workspace_id: str | None, cutoff: str, limit: int) -> list[sqlite3.Row]:
        if workspace_id:
            return self._fetchall(
                """
                SELECT * FROM episodic_memory
                WHERE workspace_id=? AND created_at>=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (workspace_id, cutoff, limit),
            )
        else:
            return self._fetchall(
                """
                SELECT * FROM episodic_memory
                WHERE created_at>=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (cutoff, limit),
            )

    def clear_episodic_memory(self, workspace_id: str | None = None) -> int:
        with self._lock:
            cur = self._conn.cursor()
            if workspace_id:
                cur.execute("DELETE FROM episodic_memory WHERE workspace_id=?", (workspace_id,))
            else:
                cur.execute("DELETE FROM episodic_memory")
            count = cur.rowcount if cur.rowcount is not None else 0
            self._conn.commit()
        return max(0, int(count))

    def get_episodic_memory_for_pruning(self) -> list[sqlite3.Row]:
        return self._fetchall("SELECT id, importance, created_at FROM episodic_memory ORDER BY created_at DESC")

    def delete_episodic_memory_by_ids(self, ids: list[str]) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executemany("DELETE FROM episodic_memory WHERE id=?", [(item_id,) for item_id in ids])
            self._conn.commit()

    def insert_pinned_memory(
        self,
        *,
        id: str,
        scope: str,
        workspace_id: str | None,
        title: str,
        content: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO pinned_memory(id, scope, workspace_id, title, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (id, scope, workspace_id, title, content, created_at, updated_at),
            )
            self._conn.commit()

    def list_pinned_memory(self, scope: str | None, workspace_id: str | None, limit: int) -> list[sqlite3.Row]:
        if scope and workspace_id:
            return self._fetchall(
                """
                SELECT * FROM pinned_memory
                WHERE scope=? AND (workspace_id=? OR workspace_id IS NULL)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (scope, workspace_id, limit),
            )
        elif scope:
            return self._fetchall(
                "SELECT * FROM pinned_memory WHERE scope=? ORDER BY updated_at DESC LIMIT ?",
                (scope, limit),
            )
        elif workspace_id:
            return self._fetchall(
                """
                SELECT * FROM pinned_memory
                WHERE scope='global' OR workspace_id=?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (workspace_id, limit),
            )
        else:
            return self._fetchall("SELECT * FROM pinned_memory ORDER BY updated_at DESC LIMIT ?", (limit,))

    def delete_pinned_memory(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM pinned_memory WHERE id=?", (memory_id,))
            removed = (cur.rowcount or 0) > 0
            self._conn.commit()
        return bool(removed)

    def clear_all_pinned_memory(self) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM pinned_memory")
            count = cur.rowcount if cur.rowcount is not None else 0
            self._conn.commit()
        return max(0, int(count))

    def clear_pinned_memory_by_workspace(self, workspace_id: str | None) -> int:
        with self._lock:
            cur = self._conn.cursor()
            if workspace_id:
                cur.execute("DELETE FROM pinned_memory WHERE workspace_id=?", (workspace_id,))
            else:
                cur.execute("DELETE FROM pinned_memory WHERE scope='workspace'")
            count = cur.rowcount if cur.rowcount is not None else 0
            self._conn.commit()
        return max(0, int(count))

    def get_memory_content_by_id(self, table: str, memory_id: str) -> sqlite3.Row | None:
        # Note: Be careful with 'table' parameter to avoid injection if it comes from user input.
        # Here it is internal.
        return self._fetchone(f"SELECT * FROM {table} WHERE id=?", (memory_id,))

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
