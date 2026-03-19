from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import schema
from .models import (
    BehaviorPolicy,
    ChatFilters,
    EpisodicMemoryEvent,
    MemoryClearScope,
    PinnedMemoryItem,
    DocumentMetadata,
    DocumentMetadataUpdate,
    SessionMemoryItem,
    SettingsModel,
    StartupProfile,
    UserPreferenceItem,
    WorkMode,
    WorkspaceIdentity,
    WorkspaceMemoryItem,
    WorkspaceResponse,
    WorkspaceUpdateRequest,
)
from .repositories.document_repo import DocumentRepository
from .repositories.memory_repo import MemoryRepository
from .repositories.settings_repo import SettingsRepository
from .repositories.infrastructure_repo import InfrastructureRepository

def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)

class Database:
    def __init__(self, sqlite_path: Path):
        self._path = sqlite_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        # Initialize repositories
        self.documents = DocumentRepository(self._conn, self._lock)
        self.memory = MemoryRepository(self._conn, self._lock)
        self.settings = SettingsRepository(self._conn, self._lock)
        self.infra = InfrastructureRepository(self._conn, self._lock)
        
        self._migrate()
        self._bootstrap_defaults()

    def _migrate(self) -> None:
        with self._lock:
            schema.migrate(self._conn)

    def _bootstrap_defaults(self) -> None:
        if self.settings.get_workspace() is None:
            self.update_workspace(
                WorkspaceUpdateRequest(
                    included_paths=[],
                    excluded_paths=[],
                    startup_profile=StartupProfile.RECOMMENDED,
                    default_mode=WorkMode.GENERAL,
                )
            )
        if self.settings.get_settings_payload() is None:
            self.update_settings(SettingsModel())
        if self.settings.get_behavior_policy_legacy() is None:
            self.update_behavior_policy(BehaviorPolicy())
        self._bootstrap_memory_defaults()

    def _bootstrap_memory_defaults(self) -> None:
        defaults: list[tuple[str, dict[str, Any]]] = [
            ("response_length", {"value": "medium"}),
            ("show_citations", {"value": True}),
            ("confirm_external_calls", {"value": False}),
            ("default_action_order", {"value": []}),
            ("prefer_draft_over_overwrite", {"value": True}),
            ("prefer_action_suggestions", {"value": True}),
        ]
        for key, value in defaults:
            row = self.memory.get_user_preference_existing(key=key, source="explicit")
            if row is not None:
                continue
            self.upsert_user_preference(
                key=key,
                value_json=value,
                source="explicit",
                confidence=1.0,
            )

    # --- Workspace & Settings ---

    def update_workspace(self, request: WorkspaceUpdateRequest) -> WorkspaceResponse:
        now = utc_now().isoformat()
        self.settings.update_workspace(
            included_paths=json.dumps(request.included_paths),
            excluded_paths=json.dumps(request.excluded_paths),
            startup_profile=request.startup_profile.value,
            default_mode=request.default_mode.value,
            updated_at=now,
        )
        return self.get_workspace()

    def get_workspace(self) -> WorkspaceResponse:
        row = self.settings.get_workspace()
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
        self.settings.update_settings_payload(
            payload_json=settings.model_dump_json(),
            updated_at=now,
        )
        return self.get_settings()

    def get_settings(self) -> SettingsModel:
        row = self.settings.get_settings_payload()
        if row is None:
            return SettingsModel()
        try:
            return SettingsModel.model_validate_json(row["payload"])
        except Exception:
            return SettingsModel()

    # --- Document & Chunks ---

    def get_indexed_documents(self) -> dict[str, float]:
        return self.documents.get_indexed_documents()

    def upsert_document(
        self,
        doc_id: str,
        path: str,
        file_type: str,
        modified_at: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized = self._normalize_metadata(metadata)
        self.documents.upsert_document(
            doc_id=doc_id,
            path=path,
            file_type=file_type,
            modified_at=modified_at,
            indexed_at=utc_now().isoformat(),
            normalized_metadata=normalized,
        )

    def update_document_auto_metadata(self, doc_id: str, metadata: dict[str, Any]) -> DocumentMetadata | None:
        normalized = self._normalize_metadata(metadata)
        self.documents.update_document_auto_metadata(doc_id, normalized)
        return self.get_document_metadata(doc_id)

    def insert_chunks(self, doc_id: str, chunks: list[tuple[str, str, int, str]]) -> None:
        self.documents.insert_chunks(chunks)

    def list_chunks_by_doc_ids(self, doc_ids: list[str]) -> list[sqlite3.Row]:
        return self.documents.list_chunks_by_doc_ids(doc_ids)

    def list_all_chunks(self) -> list[sqlite3.Row]:
        return self.documents.list_all_chunks()

    def get_document_record(self, doc_id: str) -> dict[str, Any] | None:
        row = self.documents.get_document_record(doc_id)
        if not row:
            return None
        return self._row_to_raw_dict(row)

    def get_document_metadata(self, doc_id: str) -> DocumentMetadata | None:
        row = self.documents.get_document_record(doc_id)
        if row is None:
            return None
        return DocumentMetadata(**self._row_to_effective_dict(row))

    def get_documents_metadata_map(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        rows = self.documents.get_documents_metadata_map(doc_ids)
        return {row["doc_id"]: self._row_to_effective_dict(row) for row in rows}

    def list_documents(
        self,
        *,
        search: str | None = None,
        filters: ChatFilters | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[DocumentMetadata], int]:
        rows = self.documents.list_documents()
        filtered = [self._row_to_effective_dict(row) for row in rows]
        filtered = self._apply_doc_filters(filtered, search=search, filters=filters)
        total = len(filtered)
        page = filtered[offset : offset + limit]
        return [DocumentMetadata(**item) for item in page], total

    def find_doc_ids(self, *, filters: ChatFilters | None, search: str | None = None) -> set[str]:
        rows = self.documents.find_doc_ids()
        effective = [self._row_to_effective_dict(row) for row in rows]
        filtered = self._apply_doc_filters(effective, search=search, filters=filters)
        return {item["doc_id"] for item in filtered}

    def find_doc_ids_for_workspace(
        self,
        *,
        included_paths: list[str],
        excluded_paths: list[str],
        filters: ChatFilters | None,
        search: str | None = None,
    ) -> set[str]:
        rows = self.documents.find_doc_ids()
        effective = [self._row_to_effective_dict(row) for row in rows]
        filtered = self._apply_doc_filters(effective, search=search, filters=filters)

        include_roots = [Path(item).expanduser().resolve() for item in included_paths if str(item).strip()]
        exclude_roots = [Path(item).expanduser().resolve() for item in excluded_paths if str(item).strip()]
        output: set[str] = set()
        for item in filtered:
            path = Path(item["path"]).expanduser().resolve()
            if include_roots and not any(root == path or root in path.parents for root in include_roots):
                continue
            if any(root == path or root in path.parents for root in exclude_roots):
                continue
            output.add(item["doc_id"])
        return output

    def update_document_metadata(self, doc_id: str, payload: DocumentMetadataUpdate) -> DocumentMetadata:
        row = self.documents.get_document_record(doc_id)
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
            elif key == "excluded" and value is not None:
                value = 1 if bool(value) else 0
            assignments.append(f"{column}=?")
            values.append(value)

        if assignments:
            self.documents.update_document_metadata_base(doc_id, assignments, values)

        metadata = self.get_document_metadata(doc_id)
        if metadata is None:
            raise KeyError(f"document not found after update: {doc_id}")
        return metadata

    # --- Memory ---

    def write_session_memory(
        self,
        *,
        session_id: str,
        key: str,
        value_json: dict[str, Any],
        ttl_hours: int = 24,
        keep_recent: int = 40,
    ) -> SessionMemoryItem:
        now = utc_now()
        item_id = str(uuid.uuid4())
        created_at = now.isoformat()
        expires_at = (now + timedelta(hours=ttl_hours)).isoformat() if ttl_hours > 0 else None
        
        self.memory.write_session_memory(
            item_id=item_id,
            session_id=session_id,
            key=key,
            value_json=json.dumps(value_json, ensure_ascii=False),
            created_at=created_at,
            updated_at=created_at,
            expires_at=expires_at,
        )
        self._prune_session_memory(session_id=session_id, keep_recent=keep_recent)
        return self.get_session_memory_item(item_id)

    def get_session_memory_item(self, item_id: str) -> SessionMemoryItem:
        row = self.memory.get_memory_content_by_id("session_memory", item_id)
        if not row:
            raise KeyError(f"Session memory not found: {item_id}")
        return self._row_to_session_memory(row)

    def get_relevant_session_memory(self, *, session_id: str, limit: int = 20) -> list[SessionMemoryItem]:
        self.memory.delete_expired_session_memory(utc_now().isoformat())
        rows = self.memory.get_session_memory(session_id, limit)
        return [self._row_to_session_memory(row) for row in rows]

    def clear_session_memory(self, session_id: str | None = None) -> int:
        return self.memory.clear_session_memory(session_id)

    def upsert_workspace_memory(self, *, workspace_id: str, memory_type: str, key: str, value_json: dict[str, Any], confidence: float = 0.62, source: str = "inferred") -> WorkspaceMemoryItem:
        now = utc_now().isoformat()
        existing = self.memory.get_workspace_memory_existing(workspace_id, memory_type, key, source)
        item_id = existing["id"] if existing else str(uuid.uuid4())
        created_at = existing["created_at"] if existing else now
        
        self.memory.upsert_workspace_memory(
            id=item_id,
            workspace_id=workspace_id,
            memory_type=memory_type,
            key=key,
            value_json=json.dumps(value_json, ensure_ascii=False),
            confidence=confidence,
            source=source,
            created_at=created_at,
            updated_at=now,
        )
        return self._row_to_workspace_memory(self.memory.get_memory_content_by_id("workspace_memory", item_id))

    def get_relevant_workspace_memory(self, *, workspace_id: str, intent: str | None = None, min_confidence: float = 0.5, limit: int = 30) -> list[WorkspaceMemoryItem]:
        # Note: In the original monolithic version, 'intent' might have been used for filtering.
        # Here we delegate basic retrieval to the repo.
        rows = self.memory.get_workspace_memory(workspace_id, min_confidence, limit)
        return [self._row_to_workspace_memory(row) for row in rows]

    def clear_workspace_memory(self, workspace_id: str | None = None, inferred_only: bool = False) -> int:
        return self.memory.clear_workspace_memory(workspace_id, inferred_only)

    def upsert_user_preference(self, *, key: str, value_json: dict[str, Any], confidence: float = 0.62, source: str = "inferred") -> UserPreferenceItem:
        now = utc_now().isoformat()
        existing = self.memory.get_user_preference_existing(key, source)
        item_id = existing["id"] if existing else str(uuid.uuid4())
        created_at = existing["created_at"] if existing else now
        
        self.memory.upsert_user_preference(
            id=item_id,
            key=key,
            value_json=json.dumps(value_json, ensure_ascii=False),
            confidence=confidence,
            source=source,
            created_at=created_at,
            updated_at=now,
        )
        return self._row_to_user_preference(self.memory.get_memory_content_by_id("user_preferences", item_id))

    def get_user_preferences(self) -> list[UserPreferenceItem]:
        rows = self.memory.list_user_preferences()
        return [self._row_to_user_preference(row) for row in rows]

    def get_resolved_user_preferences(self) -> dict[str, UserPreferenceItem]:
        prefs = self.get_user_preferences()
        resolved: dict[str, UserPreferenceItem] = {}
        for p in reversed(prefs):
            if p.key not in resolved or p.source == "explicit":
                resolved[p.key] = p
        return resolved

    def clear_user_preferences(self, inferred_only: bool = False) -> int:
        return self.memory.clear_user_preferences(inferred_only)

    def insert_episodic_memory(self, *, workspace_id: str | None, event_type: str, summary: str, related_file_ids: list[str] | None = None, related_action_ids: list[str] | None = None, metadata_json: dict[str, Any] | None = None, importance: float = 0.5) -> EpisodicMemoryEvent:
        item_id = str(uuid.uuid4())
        now = utc_now().isoformat()
        self.memory.insert_episodic_memory(
            id=item_id,
            workspace_id=workspace_id,
            event_type=event_type,
            summary=summary,
            related_file_ids=json.dumps(related_file_ids or []),
            related_action_ids=json.dumps(related_action_ids or []),
            metadata_json=json.dumps(metadata_json or {}, ensure_ascii=False),
            importance=importance,
            created_at=now,
        )
        self._prune_episodic_memory()
        return self._row_to_episodic_memory(self.memory.get_memory_content_by_id("episodic_memory", item_id))

    def get_relevant_episodic_memory(self, *, workspace_id: str | None = None, intent: str | None = None, related_file_ids: list[str] | None = None, limit: int = 15) -> list[EpisodicMemoryEvent]:
        # 'intent' and 'related_file_ids' are passed by service but original code was mostly relying on workspace_id and limit 
        # in the basic episodic retrieval.
        rows = self.memory.get_relevant_episodic_memory(workspace_id, limit)
        return [self._row_to_episodic_memory(row) for row in rows]

    def list_recent_episodic_memory(self, *, workspace_id: str | None = None, days: int = 7, limit: int = 50) -> list[EpisodicMemoryEvent]:
        cutoff = (utc_now() - timedelta(days=days)).isoformat()
        rows = self.memory.list_recent_episodic_memory(workspace_id, cutoff, limit)
        return [self._row_to_episodic_memory(row) for row in rows]

    def clear_episodic_memory(self, workspace_id: str | None = None) -> int:
        return self.memory.clear_episodic_memory(workspace_id)

    def create_pinned_memory(self, *, scope: str, workspace_id: str | None, title: str, content: str) -> PinnedMemoryItem:
        item_id = str(uuid.uuid4())
        now = utc_now().isoformat()
        self.memory.insert_pinned_memory(
            id=item_id,
            scope=scope,
            workspace_id=workspace_id,
            title=title,
            content=content,
            created_at=now,
            updated_at=now,
        )
        return self._row_to_pinned_memory(self.memory.get_memory_content_by_id("pinned_memory", item_id))

    def list_pinned_memory(self, *, scope: str | None = None, workspace_id: str | None = None, limit: int = 50) -> list[PinnedMemoryItem]:
        rows = self.memory.list_pinned_memory(scope, workspace_id, limit)
        return [self._row_to_pinned_memory(row) for row in rows]

    def delete_pinned_memory(self, memory_id: str) -> bool:
        return self.memory.delete_pinned_memory(memory_id)

    def create_pin_from_memory(self, *, memory_id: str, scope: str, workspace_id: str | None) -> PinnedMemoryItem | None:
        row = self.memory.get_memory_content_by_id("episodic_memory", memory_id)
        if row:
            return self.create_pinned_memory(scope=scope, workspace_id=workspace_id, title="Episodic Memory", content=row["summary"])
        
        row = self.memory.get_memory_content_by_id("workspace_memory", memory_id)
        if row:
            return self.create_pinned_memory(scope=scope, workspace_id=workspace_id, title=f"Workspace {row['key']}", content=row["value_json"])
            
        row = self.memory.get_memory_content_by_id("user_preferences", memory_id)
        if row:
            return self.create_pinned_memory(scope=scope, workspace_id=workspace_id, title=f"Preference {row['key']}", content=row["value_json"])

        row = self.memory.get_memory_content_by_id("session_memory", memory_id)
        if row:
            return self.create_pinned_memory(scope=scope, workspace_id=workspace_id, title=f"Session {row['key']}", content=row["value_json"])
        
        return None

    def clear_memory(self, *, scope: MemoryClearScope, workspace_id: str | None = None, session_id: str | None = None) -> int:
        cleared = 0
        if scope == MemoryClearScope.ALL:
            cleared += self.clear_session_memory()
            cleared += self.clear_workspace_memory()
            cleared += self.clear_user_preferences()
            cleared += self.clear_episodic_memory()
            cleared += self.memory.clear_all_pinned_memory()
            return cleared
        if scope == MemoryClearScope.SESSION:
            return self.clear_session_memory(session_id=session_id)
        if scope == MemoryClearScope.WORKSPACE:
            cleared += self.clear_workspace_memory(workspace_id=workspace_id)
            cleared += self.clear_episodic_memory(workspace_id=workspace_id)
            cleared += self.memory.clear_pinned_memory_by_workspace(workspace_id)
            return cleared
        if scope == MemoryClearScope.INFERRED_ONLY:
            cleared += self.clear_workspace_memory(workspace_id=workspace_id, inferred_only=True)
            cleared += self.clear_user_preferences(inferred_only=True)
            return cleared
        if scope == MemoryClearScope.EPISODIC:
            return self.clear_episodic_memory(workspace_id=workspace_id)
        return 0

    # --- Behavior Policy ---

    def get_behavior_policy(self) -> BehaviorPolicy:
        legacy_row = self.settings.get_behavior_policy_legacy()
        legacy_weights_rows = self.settings.get_workspace_weights_legacy()
        legacy_weights = {item["path"]: float(item["weight"]) for item in legacy_weights_rows}
        
        legacy_mode = legacy_row["preferred_mode"] if legacy_row and legacy_row["preferred_mode"] else None
        try:
            legacy_actions = json.loads((legacy_row["preferred_action_order"] if legacy_row else "[]") or "[]")
        except Exception:
            legacy_actions = []
        legacy_length = legacy_row["preferred_response_length"] if legacy_row else "medium"

        resolved_preferences = self.get_resolved_user_preferences()
        response_length_pref = resolved_preferences.get("response_length")
        action_order_pref = resolved_preferences.get("default_action_order")
        default_mode_pref = resolved_preferences.get("default_mode")

        weights = dict(legacy_weights)
        workspace_weight_rows = self.memory.get_workspace_memory(workspace_id="", min_confidence=0.0, limit=1000) # Simplified retrieval
        # Note: the original code had a more specific query for retrieval_weight. 
        # For simplicity in this facade, it might need refinement if performance is an issue.
        # But retrieval_weight is just one type of workspace memory.
        
        # Let's re-fetch specifically for weights to be accurate to original logic.
        weight_rows = self._fetchall("SELECT key, value_json FROM workspace_memory WHERE memory_type='retrieval_weight' ORDER BY updated_at DESC")
        for row in weight_rows:
            payload = self._parse_json_dict(row["value_json"])
            value = payload.get("weight")
            try:
                weights[row["key"]] = max(0.5, min(float(value), 1.8))
            except Exception: continue

        preferred_mode = legacy_mode
        if default_mode_pref:
            preferred_mode = str(default_mode_pref.value_json.get("value") or preferred_mode or "")
        preferred_actions = list(legacy_actions)
        if action_order_pref:
            raw = action_order_pref.value_json.get("value")
            if isinstance(raw, list): preferred_actions = raw
        preferred_response_length = legacy_length or "medium"
        if response_length_pref:
            preferred_response_length = str(response_length_pref.value_json.get("value") or preferred_response_length)

        return BehaviorPolicy(
            workspace_weights=weights,
            preferred_mode=preferred_mode or None,
            preferred_action_order=preferred_actions,
            preferred_response_length=preferred_response_length,
        )

    def update_behavior_policy(self, policy: BehaviorPolicy) -> BehaviorPolicy:
        now = utc_now().isoformat()
        weights_data = []
        for path, weight in (policy.workspace_weights or {}).items():
            if not str(path).strip(): continue
            weights_data.append((str(path), max(0.5, min(1.8, float(weight))), now))
        
        self.settings.update_behavior_policy_legacy(
            preferred_mode=policy.preferred_mode.value if policy.preferred_mode else None,
            preferred_action_order=json.dumps([item.value for item in policy.preferred_action_order], ensure_ascii=False),
            preferred_response_length=policy.preferred_response_length.value,
            updated_at=now,
            weights=weights_data,
        )
        
        self.upsert_user_preference(key="response_length", value_json={"value": policy.preferred_response_length.value})
        self.upsert_user_preference(key="default_action_order", value_json={"value": [item.value for item in policy.preferred_action_order]})
        if policy.preferred_mode:
            self.upsert_user_preference(key="default_mode", value_json={"value": policy.preferred_mode.value})
            
        workspace = self.get_workspace()
        identity = self.get_workspace_identity(workspace)
        for path, weight in (policy.workspace_weights or {}).items():
            if not str(path).strip(): continue
            self.upsert_workspace_memory(
                workspace_id=identity.workspace_id,
                memory_type="retrieval_weight",
                key=str(path),
                value_json={"weight": max(0.5, min(float(weight), 1.8))},
            )
        return self.get_behavior_policy()

    # --- Infrastructure ---

    def record_failure(self, path: str, reason: str) -> None:
        self.infra.record_failure(path, reason, utc_now().isoformat())

    def clear_failure(self, path: str) -> None:
        self.infra.clear_failure(path)

    def clear_all_failures(self) -> None:
        self.infra.clear_all_failures()

    def list_failures(self) -> list[dict[str, Any]]:
        rows = self.infra.list_failures()
        return [
            {
                "path": row["path"],
                "reason": row["reason"],
                "last_attempt_at": datetime.fromisoformat(row["last_attempt_at"]),
            }
            for row in rows
        ]

    def record_external_call(self, provider: str, sent_chars: int, approved_by_user: bool) -> datetime:
        now = utc_now()
        self.infra.record_external_call(provider, sent_chars, approved_by_user, now.isoformat())
        return now

    def get_status_snapshot(self) -> dict[str, Any]:
        row = self.documents.get_status_snapshot_base()
        ext = self.infra.get_latest_external_call()
        workspace = self.get_workspace()
        return {
            "indexed_docs": int(row["count"]) if row else 0,
            "last_indexed_at": row["last_indexed"] if row and row["last_indexed"] else None,
            "latest_external_call": {"provider": ext["provider"], "timestamp": ext["timestamp"]} if ext else None,
            "included_paths": workspace.included_paths,
        }

    # --- Utils ---

    def get_workspace_identity(self, workspace: WorkspaceResponse) -> WorkspaceIdentity:
        paths = sorted(workspace.included_paths)
        raw = "|".join(paths)
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return WorkspaceIdentity(
            workspace_id=sha[:12],
            included_paths_hash=sha,
            version=1,
        )

    def _prune_session_memory(self, *, session_id: str, keep_recent: int) -> None:
        self.memory.delete_expired_session_memory(utc_now().isoformat())
        rows = self.memory.get_session_memory_ids(session_id)
        if len(rows) <= keep_recent: return
        stale_ids = [row["id"] for row in rows[keep_recent:]]
        self.memory.delete_session_memory_by_ids(stale_ids)

    def _prune_episodic_memory(self) -> None:
        rows = self.memory.get_episodic_memory_for_pruning()
        now = utc_now()
        stale_ids: list[str] = []
        for row in rows:
            created = datetime.fromisoformat(row["created_at"])
            age_days = (now - created).days
            importance = float(row["importance"] or 0.0)
            ttl_days = 90 if importance >= 0.4 else 14
            if age_days > ttl_days: stale_ids.append(row["id"])
        if stale_ids:
            self.memory.delete_episodic_memory_by_ids(stale_ids)

    @staticmethod
    def _parse_json_dict(raw: Any) -> dict[str, Any]:
        if raw is None: return {}
        if isinstance(raw, dict): return raw
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else {}
        except: return {}

    @staticmethod
    def _parse_json_list(raw: Any) -> list[str]:
        if raw is None: return []
        if isinstance(raw, list): return [str(i) for i in raw]
        try:
            val = json.loads(raw)
            return [str(i) for i in val] if isinstance(val, list) else []
        except: return []

    @staticmethod
    def _row_to_session_memory(row: sqlite3.Row) -> SessionMemoryItem:
        return SessionMemoryItem(
            id=row["id"], session_id=row["session_id"], key=row["key"],
            value_json=Database._parse_json_dict(row["value_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
        )

    @staticmethod
    def _row_to_workspace_memory(row: sqlite3.Row) -> WorkspaceMemoryItem:
        return WorkspaceMemoryItem(
            id=row["id"], workspace_id=row["workspace_id"], memory_type=row["memory_type"], key=row["key"],
            value_json=Database._parse_json_dict(row["value_json"]),
            confidence=float(row["confidence"] or 0.0), source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_user_preference(row: sqlite3.Row) -> UserPreferenceItem:
        return UserPreferenceItem(
            id=row["id"], key=row["key"],
            value_json=Database._parse_json_dict(row["value_json"]),
            confidence=float(row["confidence"] or 0.0), source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_episodic_memory(row: sqlite3.Row) -> EpisodicMemoryEvent:
        return EpisodicMemoryEvent(
            id=row["id"], workspace_id=row["workspace_id"], event_type=row["event_type"], summary=row["summary"],
            related_file_ids=Database._parse_json_list(row["related_file_ids"]),
            related_action_ids=Database._parse_json_list(row["related_action_ids"]),
            metadata_json=Database._parse_json_dict(row["metadata_json"]),
            importance=float(row["importance"] or 0.0),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_pinned_memory(row: sqlite3.Row) -> PinnedMemoryItem:
        return PinnedMemoryItem(
            id=row["id"], scope=row["scope"], workspace_id=row["workspace_id"], title=row["title"], content=row["content"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        p = metadata or {}
        tags = [str(t).strip() for t in p.get("tags", []) if str(t).strip()][:8] if isinstance(p.get("tags"), list) else []
        try: importance = max(0.0, min(1.0, float(p.get("importance", 0.5))))
        except: importance = 0.5
        return {
            "summary": str(p.get("summary") or "")[:260],
            "category": str(p.get("category") or "참고자료"),
            "subcategory": str(p.get("subcategory") or "")[:40],
            "document_type": str(p.get("document_type") or "")[:40],
            "tags": tags, "year": p.get("year"), "project": str(p.get("project") or "")[:48] or None,
            "importance": importance, "excluded": bool(p.get("excluded", False)),
        }

    def _apply_doc_filters(self, rows: list[dict[str, Any]], *, search: str | None, filters: ChatFilters | None) -> list[dict[str, Any]]:
        result = rows
        if search:
            n = search.strip().lower()
            if n: result = [r for r in result if n in r["path"].lower() or n in r["summary"].lower() or n in " ".join(r["tags"]).lower()]
        if not filters: return result
        if filters.category: result = [r for r in result if r["category"] == filters.category]
        if filters.year is not None: result = [r for r in result if r.get("year") == filters.year]
        if filters.project:
            n = filters.project.lower()
            result = [r for r in result if (r.get("project") or "").lower().find(n) >= 0]
        if filters.tags:
            w = {t.lower() for t in filters.tags if t.strip()}
            if w: result = [r for r in result if w.intersection({t.lower() for t in r.get("tags", [])})]
        if filters.excluded is not None: result = [r for r in result if bool(r.get("excluded")) == bool(filters.excluded)]
        else: result = [r for r in result if not bool(r.get("excluded"))]
        return result

    @staticmethod
    def _row_to_raw_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {k: row[k] for k in row.keys()}

    @staticmethod
    def _row_to_effective_dict(row: sqlite3.Row) -> dict[str, Any]:
        c = row["user_category"] if row["user_category"] is not None else row["category"]
        sc = row["user_subcategory"] if row["user_subcategory"] is not None else row["subcategory"]
        dt = row["user_document_type"] if row["user_document_type"] is not None else row["document_type"]
        y = row["user_year"] if row["user_year"] is not None else row["year"]
        pj = row["user_project"] if row["user_project"] is not None else row["project"]
        imp = row["user_importance"] if row["user_importance"] is not None else row["importance"]
        ex = row["user_excluded"] if row["user_excluded"] is not None else row["excluded"]
        ts = Database._parse_json_list(row["user_tags"] if row["user_tags"] is not None else row["tags"])
        return {
            "doc_id": row["doc_id"], "path": row["path"], "file_type": row["file_type"],
            "modified_at": datetime.fromtimestamp(float(row["modified_at"]), tz=timezone.utc),
            "indexed_at": datetime.fromisoformat(row["indexed_at"]),
            "summary": row["summary"] or "", "category": c or "참고자료", "subcategory": sc or "",
            "document_type": dt or "", "tags": ts, "year": y, "project": pj,
            "importance": float(imp or 0.5), "excluded": bool(ex or 0),
        }

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
