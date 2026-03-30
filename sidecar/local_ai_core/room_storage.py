from __future__ import annotations

from dataclasses import dataclass
import asyncio
from pathlib import Path
import hashlib
import inspect
import os
import shutil
import sqlite3
import threading
import time
from typing import Any, Callable

from .indexing import DocumentClassifier
from .storage.db import Database
from .indexing import IndexingService
from .memory_service import MemoryService
from .models import SettingsModel, StartupProfile, WorkMode, WorkspaceResponse, WorkspaceUpdateRequest
from .storage.async_adapter import AsyncAdapter
from .storage.vector_store import VectorStore

@dataclass(frozen=True, slots=True)
class RoomStorageKey:
    room_id: str
    room_id_hash: str
    scope_hash: str
    included_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...]


@dataclass(slots=True)
class RoomStorageHandle:
    key: RoomStorageKey
    storage_id: str
    room_key: str
    root_dir: Path
    data_dir: Path
    db: Database
    vector_store: VectorStore
    indexing: IndexingService
    memory: MemoryService
    chat_service: Any
    created_at: float
    last_access_at: float
    last_job_id: str | None = None


class RoomStorageRegistry:
    def __init__(
        self,
        *,
        base_data_dir: Path,
        embedding_service: Any,
        provider_router: Any,
        local_inference: Any,
        capability_router: Any,
        docker_service: Any,
        settings_loader: Callable[[], SettingsModel],
        workspace_loader: Callable[[], WorkspaceResponse],
        embedding_dim: int,
    ):
        self._base_data_dir = base_data_dir.expanduser().resolve()
        self._base_data_dir.mkdir(parents=True, exist_ok=True)
        self._embedding = embedding_service
        self._providers = provider_router
        self._local_inference = local_inference
        self._capability_router = capability_router
        self._docker = docker_service
        self._settings_loader = settings_loader
        self._workspace_loader = workspace_loader
        self._embedding_dim = int(embedding_dim)
        self._lock = threading.RLock()
        self._handles: dict[tuple[str, str], RoomStorageHandle] = {}
        self._by_storage_id: dict[str, RoomStorageHandle] = {}
        self._latest_scope_by_room_key: dict[str, str] = {}
        self._async_adapter = AsyncAdapter()
        self._room_write_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _room_key(room_id: str) -> str:
        raw = str(room_id or "").strip()
        if not raw:
            raw = "room"
        digest = hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()
        return digest[:20]

    @staticmethod
    def _normalize_path_list(values: list[str] | None) -> list[str]:
        if not values:
            return []
        output: list[str] = []
        seen: set[str] = set()
        for item in values:
            text = str(item or "").strip()
            if not text:
                continue
            expanded = os.path.abspath(os.path.expanduser(text))
            if expanded in seen:
                continue
            seen.add(expanded)
            output.append(expanded)
        return output

    @classmethod
    def _scope_hash(cls, *, included_paths: list[str], excluded_paths: list[str]) -> str:
        normalized_included = cls._normalize_path_list(included_paths)
        normalized_excluded = cls._normalize_path_list(excluded_paths)
        merged = [f"+:{item}" for item in sorted(normalized_included)] + [f"-:{item}" for item in sorted(normalized_excluded)]
        raw = "\n".join(merged).encode("utf-8")
        return hashlib.sha1(raw, usedforsecurity=False).hexdigest()

    def _build_key(self, *, room_id: str, included_paths: list[str], excluded_paths: list[str]) -> RoomStorageKey:
        safe_room_id = str(room_id or "").strip() or "room"
        normalized_included = tuple(sorted(self._normalize_path_list(included_paths)))
        normalized_excluded = tuple(sorted(self._normalize_path_list(excluded_paths)))
        scope_hash = self._scope_hash(
            included_paths=list(normalized_included),
            excluded_paths=list(normalized_excluded),
        )
        return RoomStorageKey(
            room_id=safe_room_id,
            room_id_hash=self._room_key(safe_room_id),
            scope_hash=scope_hash,
            included_paths=normalized_included,
            excluded_paths=normalized_excluded,
        )

    def _storage_paths(self, key: RoomStorageKey) -> tuple[str, Path, Path]:
        room_key = key.room_id_hash
        storage_id = f"{room_key}:{key.scope_hash[:12]}"
        root_dir = (self._base_data_dir / room_key / key.scope_hash).resolve()
        data_dir = root_dir
        return storage_id, root_dir, data_dir

    def _build_room_chat_service(self, *, db: Database, vector_store: VectorStore, indexing: IndexingService, memory: MemoryService):
        from .chat import ChatService

        return ChatService(
            db,
            vector_store,
            self._embedding,
            self._providers,
            self._local_inference,
            memory,
            indexing,
            capability_router=self._capability_router,
            docker_service=self._docker,
            room_registry=None,
        )

    def _apply_templates(
        self,
        *,
        handle: RoomStorageHandle,
        startup_profile: StartupProfile,
        default_mode: WorkMode,
        settings_template: SettingsModel,
    ) -> None:
        handle.db.update_workspace(
            WorkspaceUpdateRequest(
                included_paths=list(handle.key.included_paths),
                excluded_paths=list(handle.key.excluded_paths),
                startup_profile=startup_profile,
                default_mode=default_mode,
            )
        )
        handle.db.update_settings(settings_template)

    def _create_handle(
        self,
        *,
        key: RoomStorageKey,
        startup_profile: StartupProfile,
        default_mode: WorkMode,
        settings_template: SettingsModel,
    ) -> RoomStorageHandle:
        storage_id, root_dir, data_dir = self._storage_paths(key)
        root_dir.mkdir(parents=True, exist_ok=True)

        db = Database(data_dir / "local_ai_core.sqlite3", skip_init=True, docker=self._docker)
        db.initialize()

        vector_store = VectorStore(data_dir / "lancedb", dim=self._embedding_dim)
        indexing = IndexingService(
            db,
            vector_store,
            self._embedding,
            DocumentClassifier(self._embedding, self._local_inference),
            capability_router=self._capability_router,
            settings_loader=db.get_settings,
        )
        memory = MemoryService(db)
        memory.set_dependencies(vector_store=vector_store, embedding_service=self._embedding)
        chat_service = self._build_room_chat_service(
            db=db,
            vector_store=vector_store,
            indexing=indexing,
            memory=memory,
        )
        handle = RoomStorageHandle(
            key=key,
            storage_id=storage_id,
            room_key=key.room_id_hash,
            root_dir=root_dir,
            data_dir=data_dir,
            db=db,
            vector_store=vector_store,
            indexing=indexing,
            memory=memory,
            chat_service=chat_service,
            created_at=time.time(),
            last_access_at=time.time(),
        )
        self._apply_templates(
            handle=handle,
            startup_profile=startup_profile,
            default_mode=default_mode,
            settings_template=settings_template,
        )
        return handle

    def _current_workspace_templates(self) -> tuple[StartupProfile, WorkMode, SettingsModel]:
        workspace = self._workspace_loader()
        settings = self._settings_loader()
        return workspace.startup_profile, workspace.default_mode, settings

    def resolve_chat_service_for_request(
        self,
        *,
        room_id: str,
        included_paths: list[str],
        excluded_paths: list[str],
    ) -> tuple[Any, dict[str, Any]]:
        requested_room_id = str(room_id or "").strip() or "room"
        requested_room_key = self._room_key(requested_room_id)
        normalized_included = self._normalize_path_list(included_paths)
        normalized_excluded = self._normalize_path_list(excluded_paths)
        route_reason = "room_scope_request"
        if not normalized_included:
            latest = self.resolve_last_chat_service(room_id=requested_room_id)
            if latest is not None:
                room_chat, room_meta = latest
                room_meta = dict(room_meta)
                room_meta["room_route_reason"] = "room_scope_cached"
                return room_chat, room_meta
            # No explicit room scope and no cached room handle: let caller decide
            # fallback behavior (global workspace path for backward compatibility).
            return None, {"room_route_reason": "room_scope_missing"}

        startup_profile, default_mode, settings_template = self._current_workspace_templates()
        key = self._build_key(
            room_id=requested_room_id,
            included_paths=normalized_included,
            excluded_paths=normalized_excluded,
        )

        created = False
        with self._lock:
            handle = self._handles.get((key.room_id_hash, key.scope_hash))
            if handle is None:
                handle = self._create_handle(
                    key=key,
                    startup_profile=startup_profile,
                    default_mode=default_mode,
                    settings_template=settings_template,
                )
                self._handles[(key.room_id_hash, key.scope_hash)] = handle
                self._by_storage_id[handle.storage_id] = handle
                created = True
            else:
                self._apply_templates(
                    handle=handle,
                    startup_profile=startup_profile,
                    default_mode=default_mode,
                    settings_template=settings_template,
                )
                handle.last_access_at = time.time()
            self._latest_scope_by_room_key[requested_room_key] = key.scope_hash

        self.ensure_auto_index(handle=handle, force_full=created)
        metadata = self.metadata_for_handle(handle)
        metadata["room_route_reason"] = route_reason
        return handle.chat_service, metadata

    def _latest_handle_for_room_key(self, room_key: str) -> RoomStorageHandle | None:
        latest_scope = self._latest_scope_by_room_key.get(room_key)
        if latest_scope:
            by_scope = self._handles.get((room_key, latest_scope))
            if by_scope is not None:
                return by_scope
        candidates = [handle for handle in self._handles.values() if handle.key.room_id_hash == room_key]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.last_access_at)

    def resolve_last_chat_service(self, *, room_id: str, scope_hash: str | None = None) -> tuple[Any, dict[str, Any]] | None:
        room_key = self._room_key(room_id)
        target_scope = str(scope_hash or "").strip()
        with self._lock:
            handle = self._handles.get((room_key, target_scope)) if target_scope else self._latest_handle_for_room_key(room_key)
        with self._lock:
            if handle is None:
                return None
            handle.last_access_at = time.time()
            self._latest_scope_by_room_key[room_key] = handle.key.scope_hash
        return handle.chat_service, self.metadata_for_handle(handle)

    def resolve_last_memory_service(
        self,
        *,
        room_id: str,
        scope_hash: str | None = None,
    ) -> tuple[MemoryService, RoomStorageHandle] | None:
        room_key = self._room_key(room_id)
        target_scope = str(scope_hash or "").strip()
        with self._lock:
            handle = self._handles.get((room_key, target_scope)) if target_scope else self._latest_handle_for_room_key(room_key)
        with self._lock:
            if handle is None:
                return None
            handle.last_access_at = time.time()
            self._latest_scope_by_room_key[room_key] = handle.key.scope_hash
        return handle.memory, handle

    def metadata_for_handle(self, handle: RoomStorageHandle) -> dict[str, Any]:
        return {
            "room_storage_id": handle.storage_id,
            "room_scope_hash": handle.key.scope_hash,
            "room_index_state": self.room_index_state(handle=handle),
        }

    def room_index_state(self, *, handle: RoomStorageHandle) -> str:
        if handle.indexing.has_running_job():
            return "indexing"
        if handle.last_job_id:
            job = handle.indexing.get_job(handle.last_job_id)
            if job is not None:
                if job.status in {"queued", "running"}:
                    return "indexing"
                if job.status == "failed":
                    return "failed"
        snapshot = handle.db.get_status_snapshot()
        indexed_docs = int(snapshot.get("indexed_docs") or 0)
        return "ready" if indexed_docs > 0 else "idle"

    def room_index_state_by_storage_id(self, storage_id: str) -> str:
        with self._lock:
            handle = self._by_storage_id.get(str(storage_id or ""))
        if handle is None:
            return "idle"
        return self.room_index_state(handle=handle)

    def _index_job_snapshot_for_handle(self, handle: RoomStorageHandle) -> dict[str, Any]:
        job_id = str(handle.last_job_id or "").strip()
        if not job_id:
            return {}
        job = handle.indexing.get_job(job_id)
        if job is None:
            return {"job_id": job_id}
        progress = float(getattr(job, "progress", 0.0) or 0.0)
        progress = max(0.0, min(1.0, progress))
        return {
            "job_id": job_id,
            "index_progress": progress,
            "index_stage": str(getattr(job, "stage", "") or ""),
            "processed_files": int(getattr(job, "processed_files", 0) or 0),
            "failed_files": int(getattr(job, "failed_files", 0) or 0),
            "job_status": str(getattr(job, "status", "") or ""),
        }

    def _index_job_snapshot_by_storage_id(self, storage_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._by_storage_id.get(str(storage_id or ""))
        if handle is None:
            return {}
        return self._index_job_snapshot_for_handle(handle)

    def ensure_auto_index(self, *, handle: RoomStorageHandle, force_full: bool = False) -> str:
        snapshot = handle.db.get_status_snapshot()
        indexed_docs = int(snapshot.get("indexed_docs") or 0)
        has_scope = bool(handle.key.included_paths)
        should_index = has_scope and (force_full or indexed_docs <= 0)
        if should_index and not handle.indexing.has_running_job():
            workspace = handle.db.get_workspace()
            job = handle.indexing.start_job("full", workspace)
            handle.last_job_id = job.job_id
            handle.last_access_at = time.time()
        return self.room_index_state(handle=handle)

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        total = 0
        if not path.exists():
            return 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                file_path = Path(root) / name
                try:
                    total += file_path.stat().st_size
                except Exception:
                    continue
        return total

    @staticmethod
    def _count_table(db_path: Path, table: str) -> int:
        if not db_path.exists():
            return 0
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                return int(row[0] if row else 0)
        except Exception:
            return 0

    def _latest_handle_for_room(self, room_id: str) -> RoomStorageHandle | None:
        room_key = self._room_key(room_id)
        with self._lock:
            return self._latest_handle_for_room_key(room_key)

    def _get_room_write_lock(self, room_key: str) -> asyncio.Lock:
        with self._lock:
            lock = self._room_write_locks.get(room_key)
            if lock is None:
                lock = asyncio.Lock()
                self._room_write_locks[room_key] = lock
            return lock

    async def run_room_write(
        self,
        *,
        room_id: str,
        operation: Callable[..., Any],
        scope_hash: str | None = None,
        **kwargs: Any,
    ) -> Any:
        room_key = self._room_key(room_id)
        lock = self._get_room_write_lock(room_key)
        async with lock:
            call_kwargs = dict(kwargs)
            try:
                signature = inspect.signature(operation)
                has_room_id = "room_id" in signature.parameters
                has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
                if has_room_id or has_kwargs:
                    call_kwargs["room_id"] = room_id
            except (TypeError, ValueError):
                pass
            return await self._async_adapter.run(operation, **call_kwargs)

    async def reindex_room_async(
        self,
        *,
        room_id: str,
        scope: str = "full",
        included_paths: list[str] | None = None,
        excluded_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.run_room_write(
            room_id=room_id,
            operation=self.reindex_room,
            scope=scope,
            included_paths=included_paths,
            excluded_paths=excluded_paths,
        )

    async def delete_room_storage_async(self, *, room_id: str) -> dict[str, Any]:
        return await self.run_room_write(
            room_id=room_id,
            operation=self.delete_room_storage,
        )

    def room_storage_status(self, *, room_id: str) -> dict[str, Any]:
        room_key = self._room_key(room_id)
        room_dir = (self._base_data_dir / room_key).resolve()
        variants: list[dict[str, Any]] = []
        total_bytes = 0
        if room_dir.exists():
            for candidate in sorted([item for item in room_dir.iterdir() if item.is_dir()], key=lambda item: item.stat().st_mtime, reverse=True):
                scope_hash = candidate.name
                storage_id = f"{room_key}:{scope_hash[:12]}"
                db_path = candidate / "local_ai_core.sqlite3"
                indexed_docs = self._count_table(db_path, "documents")
                chunk_count = self._count_table(db_path, "chunks")
                session_count = self._count_table(db_path, "session_memory")
                workspace_count = self._count_table(db_path, "workspace_memory")
                bytes_used = self._dir_size_bytes(candidate)
                total_bytes += bytes_used
                state = self.room_index_state_by_storage_id(storage_id)
                if state == "idle" and indexed_docs > 0:
                    state = "ready"
                job_snapshot = self._index_job_snapshot_by_storage_id(storage_id)
                variants.append(
                    {
                        "room_storage_id": storage_id,
                        "scope_hash": scope_hash,
                        "data_dir": str(candidate),
                        "indexed_docs": indexed_docs,
                        "chunk_count": chunk_count,
                        "session_memory_count": session_count,
                        "workspace_memory_count": workspace_count,
                        "bytes_used": bytes_used,
                        "room_index_state": state,
                        "index_progress": job_snapshot.get("index_progress"),
                        "index_stage": job_snapshot.get("index_stage"),
                        "processed_files": job_snapshot.get("processed_files"),
                        "failed_files": job_snapshot.get("failed_files"),
                        "job_id": job_snapshot.get("job_id"),
                        "job_status": job_snapshot.get("job_status"),
                    }
                )
        return {
            "room_id": room_id,
            "room_key": room_key,
            "variant_count": len(variants),
            "total_bytes": total_bytes,
            "variants": variants,
        }

    def reindex_room(
        self,
        *,
        room_id: str,
        scope: str = "full",
        included_paths: list[str] | None = None,
        excluded_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        safe_scope = str(scope or "full").strip().lower()
        if safe_scope not in {"full", "incremental"}:
            safe_scope = "full"

        handle: RoomStorageHandle | None = None
        if included_paths:
            startup_profile, default_mode, settings_template = self._current_workspace_templates()
            key = self._build_key(
                room_id=room_id,
                included_paths=included_paths or [],
                excluded_paths=excluded_paths or [],
            )
            with self._lock:
                handle = self._handles.get((key.room_id_hash, key.scope_hash))
            if handle is None:
                handle = self._create_handle(
                    key=key,
                    startup_profile=startup_profile,
                    default_mode=default_mode,
                    settings_template=settings_template,
                )
                with self._lock:
                    self._handles[(key.room_id_hash, key.scope_hash)] = handle
                    self._by_storage_id[handle.storage_id] = handle
                    self._latest_scope_by_room_key[key.room_id_hash] = key.scope_hash
            else:
                self._apply_templates(
                    handle=handle,
                    startup_profile=startup_profile,
                    default_mode=default_mode,
                    settings_template=settings_template,
                )
        else:
            handle = self._latest_handle_for_room(room_id)

        if handle is None:
            return {"ok": False, "error": "room storage not found"}

        if handle.indexing.has_running_job():
            state = self.room_index_state(handle=handle)
            snapshot = self._index_job_snapshot_for_handle(handle)
            return {
                "ok": True,
                "room_storage_id": handle.storage_id,
                "room_scope_hash": handle.key.scope_hash,
                "room_index_state": state,
                "job_id": handle.last_job_id,
                "index_progress": snapshot.get("index_progress"),
                "index_stage": snapshot.get("index_stage"),
                "processed_files": snapshot.get("processed_files"),
                "failed_files": snapshot.get("failed_files"),
                "job_status": snapshot.get("job_status"),
                "started": False,
            }

        workspace = handle.db.get_workspace()
        job = handle.indexing.start_job(safe_scope, workspace)
        handle.last_job_id = job.job_id
        handle.last_access_at = time.time()
        return {
            "ok": True,
            "room_storage_id": handle.storage_id,
            "room_scope_hash": handle.key.scope_hash,
            "room_index_state": self.room_index_state(handle=handle),
            "job_id": job.job_id,
            "index_progress": float(getattr(job, "progress", 0.0) or 0.0),
            "index_stage": str(getattr(job, "stage", "") or ""),
            "processed_files": int(getattr(job, "processed_files", 0) or 0),
            "failed_files": int(getattr(job, "failed_files", 0) or 0),
            "job_status": str(getattr(job, "status", "") or ""),
            "started": True,
        }

    def delete_room_storage(self, *, room_id: str) -> dict[str, Any]:
        room_key = self._room_key(room_id)
        target_dir = (self._base_data_dir / room_key).resolve()
        removed_bytes = self._dir_size_bytes(target_dir)

        with self._lock:
            drop_keys = [key for key, handle in self._handles.items() if handle.key.room_id_hash == room_key]
            for key in drop_keys:
                handle = self._handles.pop(key)
                self._by_storage_id.pop(handle.storage_id, None)
                try:
                    handle.indexing.stop_watcher()
                except Exception:
                    pass
            self._latest_scope_by_room_key.pop(room_key, None)
            self._room_write_locks.pop(room_key, None)

        removed = False
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=False)
            removed = True

        return {
            "room_id": room_id,
            "removed": removed,
            "removed_bytes": removed_bytes if removed else 0,
        }

    def close_all(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
            self._by_storage_id.clear()
            self._latest_scope_by_room_key.clear()
        for handle in handles:
            try:
                handle.indexing.stop_watcher()
            except Exception:
                pass
