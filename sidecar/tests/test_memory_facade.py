from __future__ import annotations

import asyncio

from local_ai_core.services.memory_facade import MemoryFacade


class _DummyWorkspaceService:
    async def run_read(self, fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    async def run_write(self, fn, /, *args, **kwargs):
        return fn(*args, **kwargs)


class _DummyRoomStorage:
    async def run_room_write(self, *, room_id: str, operation, scope_hash=None, **kwargs):
        return operation(**kwargs)


class _DummyMemory:
    def listPinnedMemory(self, *, scope=None, workspace_id=None):  # noqa: N802
        return [{"scope": scope, "workspace_id": workspace_id}]

    def unpinMemory(self, memory_id):  # noqa: N802
        return memory_id == "ok"

    def write_memory_event(self, event):
        return {"event_id": "e1", "accepted": True, "summary": getattr(event, "summary", "")}


def test_memory_facade_global_pins_and_unpin():
    facade = MemoryFacade(workspace_service=_DummyWorkspaceService(), room_storage=_DummyRoomStorage())
    mem = _DummyMemory()

    pins = asyncio.run(facade.list_pins(memory=mem, scope="global", workspace_id=None))
    removed_true = asyncio.run(facade.unpin(memory=mem, memory_id="ok"))
    removed_false = asyncio.run(facade.unpin(memory=mem, memory_id="bad"))

    assert pins == [{"scope": "global", "workspace_id": None}]
    assert removed_true is True
    assert removed_false is False
