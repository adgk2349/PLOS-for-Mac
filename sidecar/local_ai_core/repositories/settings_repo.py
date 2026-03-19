from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from ..models import (
    BehaviorPolicy,
)

class SettingsRepository:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def get_workspace(self) -> sqlite3.Row | None:
        return self._fetchone("SELECT * FROM workspace WHERE id=1")

    def update_workspace(
        self,
        *,
        included_paths: str,
        excluded_paths: str,
        startup_profile: str,
        default_mode: str,
        updated_at: str,
    ) -> None:
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
                (included_paths, excluded_paths, startup_profile, default_mode, updated_at),
            )
            self._conn.commit()

    def get_settings_payload(self) -> sqlite3.Row | None:
        return self._fetchone("SELECT payload FROM settings WHERE id=1")

    def update_settings_payload(self, payload_json: str, updated_at: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO settings(id, payload, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (payload_json, updated_at),
            )
            self._conn.commit()

    def get_behavior_policy_legacy(self) -> sqlite3.Row | None:
        return self._fetchone("SELECT * FROM behavior_policies WHERE id=1")

    def get_workspace_weights_legacy(self) -> list[sqlite3.Row]:
        return self._fetchall("SELECT path, weight FROM workspace_weights")

    def update_behavior_policy_legacy(
        self,
        *,
        preferred_mode: str | None,
        preferred_action_order: str,
        preferred_response_length: str,
        updated_at: str,
        weights: list[tuple[str, float, str]],
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO behavior_policies(id, preferred_mode, preferred_action_order, preferred_response_length, updated_at)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  preferred_mode=excluded.preferred_mode,
                  preferred_action_order=excluded.preferred_action_order,
                  preferred_response_length=excluded.preferred_response_length,
                  updated_at=excluded.updated_at
                """,
                (preferred_mode, preferred_action_order, preferred_response_length, updated_at),
            )
            cur.execute("DELETE FROM workspace_weights")
            if weights:
                cur.executemany(
                    "INSERT INTO workspace_weights(path, weight, updated_at) VALUES (?, ?, ?)",
                    weights,
                )
            self._conn.commit()

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
