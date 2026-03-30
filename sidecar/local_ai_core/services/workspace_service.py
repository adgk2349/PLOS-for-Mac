from __future__ import annotations

import asyncio
from typing import Any, Callable

from ..storage.async_adapter import AsyncAdapter


class WorkspaceService:
    def __init__(self, *, db, indexing):
        self._db = db
        self._indexing = indexing
        self._adapter = AsyncAdapter()
        self._write_lock = asyncio.Lock()

    async def run_read(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return await self._adapter.run(fn, *args, **kwargs)

    async def run_write(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        async with self._write_lock:
            return await self._adapter.run(fn, *args, **kwargs)

    async def get_workspace(self):
        return await self.run_read(self._db.get_workspace)

    async def update_workspace(self, payload):
        return await self.run_write(self._db.update_workspace, payload)

    async def get_settings(self):
        return await self.run_read(self._db.get_settings)

    async def update_settings(self, payload):
        return await self.run_write(self._db.update_settings, payload)

    async def get_status_snapshot(self):
        return await self.run_read(self._db.get_status_snapshot)

    async def find_doc_ids_for_workspace(self, *, included_paths, excluded_paths, filters, search):
        return await self.run_read(
            self._db.find_doc_ids_for_workspace,
            included_paths=included_paths,
            excluded_paths=excluded_paths,
            filters=filters,
            search=search,
        )

    async def list_documents(self, *, search, filters, allowed_doc_ids, limit, offset):
        return await self.run_read(
            self._db.list_documents,
            search=search,
            filters=filters,
            allowed_doc_ids=allowed_doc_ids,
            limit=limit,
            offset=offset,
        )

    async def update_document_metadata(self, doc_id, payload):
        return await self.run_write(self._db.update_document_metadata, doc_id, payload)

    def start_watcher(self) -> None:
        self._indexing.start_watcher()
