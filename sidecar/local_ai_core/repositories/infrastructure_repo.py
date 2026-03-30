from __future__ import annotations

import sqlite3
import threading
from typing import Any

class InfrastructureRepository:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def record_failure(self, path: str, reason: str, now_iso: str) -> None:
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
                (path, reason, now_iso),
            )
            self._conn.commit()

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

    def list_failures(self) -> list[sqlite3.Row]:
        return self._fetchall("SELECT path, reason, last_attempt_at FROM failures ORDER BY last_attempt_at DESC")

    def record_external_call(self, provider: str, sent_chars: int, approved_by_user: bool, timestamp_iso: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO external_calls(provider, sent_chars, approved_by_user, timestamp) VALUES (?, ?, ?, ?)",
                (provider, sent_chars, 1 if approved_by_user else 0, timestamp_iso),
            )
            self._conn.commit()

    def get_latest_external_call(self) -> sqlite3.Row | None:
        return self._fetchone("SELECT provider, timestamp FROM external_calls ORDER BY id DESC LIMIT 1")

    def list_plugin_registry(self) -> list[sqlite3.Row]:
        return self._fetchall(
            """
            SELECT plugin_id, manifest_json, enabled, state, updated_at
            FROM plugin_registry
            ORDER BY plugin_id
            """
        )

    def get_plugin_registry(self, plugin_id: str) -> sqlite3.Row | None:
        return self._fetchone(
            """
            SELECT plugin_id, manifest_json, enabled, state, updated_at
            FROM plugin_registry
            WHERE plugin_id=?
            LIMIT 1
            """,
            (plugin_id,),
        )

    def upsert_plugin_registry(
        self,
        *,
        plugin_id: str,
        manifest_json: str,
        enabled: bool,
        state: str,
        updated_at: str,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO plugin_registry(plugin_id, manifest_json, enabled, state, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(plugin_id) DO UPDATE SET
                  manifest_json=excluded.manifest_json,
                  enabled=excluded.enabled,
                  state=excluded.state,
                  updated_at=excluded.updated_at
                """,
                (plugin_id, manifest_json, 1 if enabled else 0, state, updated_at),
            )
            self._conn.commit()

    def list_platform_adapters(self) -> list[sqlite3.Row]:
        return self._fetchall(
            """
            SELECT adapter_key, adapter_class, health, updated_at
            FROM platform_adapters
            ORDER BY adapter_key
            """
        )

    def upsert_platform_adapter(
        self,
        *,
        adapter_key: str,
        adapter_class: str,
        health: str,
        updated_at: str,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO platform_adapters(adapter_key, adapter_class, health, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(adapter_key) DO UPDATE SET
                  adapter_class=excluded.adapter_class,
                  health=excluded.health,
                  updated_at=excluded.updated_at
                """,
                (adapter_key, adapter_class, health, updated_at),
            )
            self._conn.commit()

    def delete_plugin_registry(self, plugin_id: str) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM plugin_registry WHERE plugin_id=?", (plugin_id,))
            removed = cur.rowcount > 0
            self._conn.commit()
            return removed

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
