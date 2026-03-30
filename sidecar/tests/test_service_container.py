from __future__ import annotations

import sys
from importlib import import_module, reload
from pathlib import Path

from local_ai_core import config as config_module


def test_service_container_smoke(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOCAL_AI_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOCAL_AI_SESSION_TOKEN", "token")
    reload(config_module)
    if "local_ai_core.container" in sys.modules:
        container_module = reload(sys.modules["local_ai_core.container"])
    else:
        container_module = import_module("local_ai_core.container")
    ServiceContainer = getattr(container_module, "ServiceContainer")
    container = ServiceContainer()
    db = container.db()
    workspace_service = container.workspace_service()
    assert str(db.sqlite_path).startswith(str((tmp_path / "data").resolve()))
    assert db is not None
    assert workspace_service is not None
