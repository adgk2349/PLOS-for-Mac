from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

from local_ai_core.models import SettingsModel, StartupProfile, WorkMode, WorkspaceResponse
from local_ai_core.room_storage import RoomStorageRegistry


def _make_registry(tmp_path: Path) -> RoomStorageRegistry:
    return RoomStorageRegistry(
        base_data_dir=tmp_path / "rooms",
        embedding_service=None,
        provider_router=None,
        local_inference=None,
        capability_router=None,
        docker_service=None,
        settings_loader=lambda: SettingsModel(),
        workspace_loader=lambda: WorkspaceResponse(
            included_paths=[],
            excluded_paths=[],
            startup_profile=StartupProfile.RECOMMENDED,
            default_mode=WorkMode.GENERAL,
            updated_at=datetime.now(timezone.utc),
        ),
        embedding_dim=384,
    )


def test_run_room_write_serializes_same_room(tmp_path: Path):
    registry = _make_registry(tmp_path)
    order: list[str] = []

    def first() -> str:
        time.sleep(0.06)
        order.append("first")
        return "first"

    def second() -> str:
        order.append("second")
        return "second"

    async def _run():
        t1 = asyncio.create_task(registry.run_room_write(room_id="room-a", operation=first))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(registry.run_room_write(room_id="room-a", operation=second))
        return await asyncio.gather(t1, t2)

    r1, r2 = asyncio.run(_run())

    assert (r1, r2) == ("first", "second")
    assert order == ["first", "second"]
