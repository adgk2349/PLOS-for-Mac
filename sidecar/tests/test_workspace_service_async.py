from __future__ import annotations

import asyncio
import time

from local_ai_core.services.workspace_service import WorkspaceService


class _DummyDB:
    def __init__(self):
        self.value = 0

    def increment_with_delay(self, delta: int, delay: float) -> int:
        current = self.value
        time.sleep(delay)
        self.value = current + delta
        return self.value


class _DummyIndexing:
    def start_watcher(self) -> None:
        return None


def test_workspace_service_run_write_serializes():
    service = WorkspaceService(db=_DummyDB(), indexing=_DummyIndexing())

    async def _run():
        t1 = asyncio.create_task(service.run_write(service._db.increment_with_delay, 1, 0.06))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(service.run_write(service._db.increment_with_delay, 1, 0.0))
        return await asyncio.gather(t1, t2)

    out = asyncio.run(_run())
    assert out == [1, 2]
