from __future__ import annotations

import logging

from ..config import settings
from ..storage.async_adapter import AsyncAdapter
from ..storage.async_repositories import (
    AdapterInfraRepo,
    AdapterMemoryRepo,
    AsyncRepositoryFactory,
)

logger = logging.getLogger(__name__)


class SystemOrchestrator:
    """Lifecycle orchestration for DB/indexer/docker with async-adapter boundary."""

    def __init__(self, *, db, indexing, docker):
        self._db = db
        self._indexing = indexing
        self._docker = docker
        self._adapter = AsyncAdapter()
        self.workspace_repo = AsyncRepositoryFactory.create_workspace_repo(
            db=db,
            adapter=self._adapter,
            backend=settings.async_repo_backend,
        )
        self.memory_repo = AdapterMemoryRepo(db, self._adapter)
        self.infra_repo = AdapterInfraRepo(docker, self._adapter)

    async def initialize(self) -> None:
        await self.memory_repo.initialize()
        self._indexing.start_watcher()
        app_settings = await self.workspace_repo.get_settings()
        if bool(getattr(app_settings, "auto_start_searxng", False)):
            started = await self.infra_repo.start_docker(keep_running=True)
            if started:
                logger.info("Docker services (SearXNG) started.")
            else:
                logger.warning("Docker services could not be started automatically.")
        else:
            logger.info("SearXNG auto-start disabled. Using on-demand startup.")

    async def shutdown(self) -> bool:
        return await self.infra_repo.stop_docker(shutdown_desktop=False, remove_stack=False)
