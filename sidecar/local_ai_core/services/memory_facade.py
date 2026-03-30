from __future__ import annotations


class MemoryFacade:
    """Async facade for global/room memory operations."""

    def __init__(self, *, workspace_service, room_storage):
        self._workspace = workspace_service
        self._room_storage = room_storage

    async def get_relevant_session(self, *, memory, session_id: str):
        return await self._workspace.run_read(memory.get_relevant_session_memory, session_id=session_id)

    async def get_relevant_workspace(self, *, memory, workspace_id: str, intent: str | None = None):
        return await self._workspace.run_read(
            memory.get_relevant_workspace_memory,
            workspace_id=workspace_id,
            intent=intent,
        )

    async def get_relevant_episodic(
        self,
        *,
        memory,
        workspace_id: str | None,
        intent: str | None,
        related_files: list[str],
    ):
        return await self._workspace.run_read(
            memory.get_relevant_episodic_memory,
            workspace_id=workspace_id,
            intent=intent,
            related_files=related_files,
        )

    async def list_preferences(self, *, memory):
        return await self._workspace.run_read(memory.get_user_preferences)

    async def write_event(self, *, memory, payload):
        return await self._workspace.run_write(memory.writeMemoryEvent, event=payload)

    async def clear(self, *, memory, payload):
        return await self._workspace.run_write(
            memory.clearMemory,
            scope=payload.scope,
            workspace_id=payload.workspace_id,
            session_id=payload.session_id,
        )

    async def pin(self, *, memory, payload):
        return await self._workspace.run_write(
            memory.pinMemory,
            memory_id=payload.memory_id,
            scope=payload.scope,
            workspace_id=payload.workspace_id,
            title=payload.title,
            content=payload.content,
        )

    async def unpin(self, *, memory, memory_id: str) -> bool:
        out = await self._workspace.run_write(memory.unpinMemory, memory_id)
        return bool(out)

    async def list_pins(self, *, memory, scope: str | None = None, workspace_id: str | None = None):
        return await self._workspace.run_read(memory.listPinnedMemory, scope=scope, workspace_id=workspace_id)

    async def room_write_event(self, *, room_id: str, room_scope_hash: str | None, memory, payload):
        return await self._room_storage.run_room_write(
            room_id=room_id,
            scope_hash=room_scope_hash,
            operation=memory.write_memory_event,
            event=payload,
        )

    async def room_clear(self, *, room_id: str, room_scope_hash: str | None, memory, payload):
        return await self._room_storage.run_room_write(
            room_id=room_id,
            scope_hash=room_scope_hash,
            operation=memory.clear_memory,
            scope=payload.scope,
            workspace_id=payload.workspace_id,
            session_id=payload.session_id,
        )

    async def room_pin(self, *, room_id: str, room_scope_hash: str | None, memory, payload):
        return await self._room_storage.run_room_write(
            room_id=room_id,
            scope_hash=room_scope_hash,
            operation=memory.pin_memory,
            memory_id=payload.memory_id,
            scope=payload.scope,
            workspace_id=payload.workspace_id,
            title=payload.title,
            content=payload.content,
        )

    async def room_unpin(self, *, room_id: str, room_scope_hash: str | None, memory, memory_id: str) -> bool:
        out = await self._room_storage.run_room_write(
            room_id=room_id,
            scope_hash=room_scope_hash,
            operation=memory.unpin_memory,
            memory_id=memory_id,
        )
        return bool(out)
