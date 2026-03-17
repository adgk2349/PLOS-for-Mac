from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _reload_sidecar_modules() -> None:
    for module_name in [
        "local_ai_core.config",
        "local_ai_core.db",
        "local_ai_core.main",
    ]:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("LOCAL_AI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("LOCAL_AI_SESSION_TOKEN", "test-token")
    _reload_sidecar_modules()

    from local_ai_core.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    return {"x-session-token": "test-token"}
