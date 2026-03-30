from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ..models import LocalEngine, SettingsModel, WorkspaceResponse
from .async_adapter import AsyncAdapter

logger = logging.getLogger(__name__)


class WorkspaceRepo(Protocol):
    async def get_workspace(self): ...
    async def get_settings(self): ...


class MemoryRepo(Protocol):
    async def initialize(self) -> None: ...


class InfraRepo(Protocol):
    async def start_docker(self, *, keep_running: bool) -> bool: ...
    async def stop_docker(self, *, shutdown_desktop: bool, remove_stack: bool) -> bool: ...


class AdapterWorkspaceRepo:
    def __init__(self, db, adapter: AsyncAdapter):
        self._db = db
        self._adapter = adapter

    async def get_workspace(self):
        return await self._adapter.run(self._db.get_workspace)

    async def get_settings(self):
        return await self._adapter.run(self._db.get_settings)


class AdapterMemoryRepo:
    def __init__(self, db, adapter: AsyncAdapter):
        self._db = db
        self._adapter = adapter

    async def initialize(self) -> None:
        await self._adapter.run(self._db.initialize)


class AdapterInfraRepo:
    def __init__(self, docker, adapter: AsyncAdapter):
        self._docker = docker
        self._adapter = adapter

    async def start_docker(self, *, keep_running: bool) -> bool:
        return bool(await self._adapter.run(self._docker.start, keep_running=keep_running))

    async def stop_docker(self, *, shutdown_desktop: bool, remove_stack: bool) -> bool:
        return bool(
            await self._adapter.run(
                self._docker.stop,
                shutdown_desktop=shutdown_desktop,
                remove_stack=remove_stack,
            )
        )


class AioSqliteWorkspaceRepo:
    """Async workspace/settings reads using aiosqlite with safe adapter fallback."""

    def __init__(self, *, sqlite_path: Path, fallback: AdapterWorkspaceRepo):
        self._sqlite_path = sqlite_path
        self._fallback = fallback
        self._aiosqlite = None
        try:
            import aiosqlite  # type: ignore

            self._aiosqlite = aiosqlite
        except Exception:
            self._aiosqlite = None

    @property
    def available(self) -> bool:
        return self._aiosqlite is not None

    async def get_workspace(self):
        if not self.available:
            return await self._fallback.get_workspace()
        try:
            assert self._aiosqlite is not None
            async with self._aiosqlite.connect(self._sqlite_path) as conn:
                conn.row_factory = self._aiosqlite.Row
                async with conn.execute(
                    """
                    SELECT included_paths, excluded_paths, startup_profile, default_mode, updated_at
                    FROM workspace
                    WHERE id=1
                    LIMIT 1
                    """
                ) as cur:
                    row = await cur.fetchone()
            if row is None:
                return await self._fallback.get_workspace()
            return WorkspaceResponse(
                included_paths=self._decode_json_list(row["included_paths"]),
                excluded_paths=self._decode_json_list(row["excluded_paths"]),
                startup_profile=row["startup_profile"],
                default_mode=row["default_mode"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
        except Exception:
            logger.exception("AioSqliteWorkspaceRepo.get_workspace failed; falling back to adapter backend.")
            return await self._fallback.get_workspace()

    async def get_settings(self):
        if not self.available:
            return await self._fallback.get_settings()
        try:
            assert self._aiosqlite is not None
            async with self._aiosqlite.connect(self._sqlite_path) as conn:
                conn.row_factory = self._aiosqlite.Row
                async with conn.execute(
                    """
                    SELECT payload
                    FROM settings
                    WHERE id=1
                    LIMIT 1
                    """
                ) as cur:
                    row = await cur.fetchone()
            if row is None:
                return SettingsModel()
            settings = SettingsModel.model_validate_json(row["payload"])
            return self._sanitize_engine_model_paths(settings)
        except Exception:
            logger.exception("AioSqliteWorkspaceRepo.get_settings failed; falling back to adapter backend.")
            return await self._fallback.get_settings()

    @staticmethod
    def _decode_json_list(payload: str) -> list[str]:
        try:
            import json

            parsed = json.loads(payload or "[]")
            if isinstance(parsed, list):
                return [str(item) for item in parsed if isinstance(item, str)]
        except Exception:
            pass
        return []

    @staticmethod
    def _sanitize_engine_model_paths(settings: SettingsModel) -> SettingsModel:
        mlx_path = str(settings.mlx_model_path or "").strip()
        llama_path = str(settings.llama_model_path or "").strip()
        if settings.local_engine.value == "mlx":
            if mlx_path:
                settings.llama_model_path = None
            elif llama_path:
                settings.local_engine = LocalEngine.LLAMA_CPP
                settings.mlx_model_path = None
        else:
            if llama_path:
                settings.mlx_model_path = None
            elif mlx_path:
                settings.local_engine = LocalEngine.MLX
                settings.llama_model_path = None
        return settings


class AsyncRepositoryFactory:
    """Build repository implementations with optional backend selection."""

    @staticmethod
    def create_workspace_repo(*, db, adapter: AsyncAdapter, backend: str) -> WorkspaceRepo:
        adapter_repo = AdapterWorkspaceRepo(db, adapter)
        selected = str(backend or "").strip().lower()
        if selected != "aiosqlite":
            return adapter_repo

        repo = AioSqliteWorkspaceRepo(sqlite_path=db.sqlite_path, fallback=adapter_repo)
        if not repo.available:
            logger.warning("ASYNC_REPO_BACKEND=aiosqlite requested but aiosqlite is unavailable; using adapter backend.")
            return adapter_repo
        return repo
