from __future__ import annotations

import asyncio
from pathlib import Path

from local_ai_core.db import Database
from local_ai_core.storage.async_adapter import AsyncAdapter
from local_ai_core.storage.async_repositories import AsyncRepositoryFactory


def test_workspace_repo_factory_adapter_backend(tmp_path: Path):
    db = Database(tmp_path / "local_ai_core.sqlite3")
    adapter = AsyncAdapter()
    repo = AsyncRepositoryFactory.create_workspace_repo(db=db, adapter=adapter, backend="adapter")
    assert repo.__class__.__name__ == "AdapterWorkspaceRepo"


def test_workspace_repo_factory_aiosqlite_backend_reads_workspace_and_settings(tmp_path: Path):
    db = Database(tmp_path / "local_ai_core.sqlite3")
    adapter = AsyncAdapter()
    repo = AsyncRepositoryFactory.create_workspace_repo(db=db, adapter=adapter, backend="aiosqlite")

    workspace = asyncio.run(repo.get_workspace())
    settings = asyncio.run(repo.get_settings())

    assert workspace is not None
    assert settings is not None
    assert hasattr(settings, "privacy_mode")
