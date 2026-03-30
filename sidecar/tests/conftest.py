from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

def _patch_llama_cpp_unraisable_destructor() -> None:
    """
    Test-only guard for a known llama-cpp-python destructor issue:
    partially initialized LlamaModel can raise AttributeError('sampler') in __del__.
    """
    try:
        internals = importlib.import_module("llama_cpp._internals")
        cls = getattr(internals, "LlamaModel", None)
        if cls is None:
            return
        original = getattr(cls, "__del__", None)
        if original is None or getattr(cls, "_plos_safe_del_patched", False):
            return

        def _safe_del(self):
            try:
                original(self)
            except AttributeError as exc:
                if "sampler" in str(exc):
                    return
                raise

        cls.__del__ = _safe_del
        setattr(cls, "_plos_safe_del_patched", True)
    except Exception:
        return


def pytest_configure(config: pytest.Config) -> None:
    _patch_llama_cpp_unraisable_destructor()


def _reload_sidecar_modules() -> None:
    for module_name in [
        "local_ai_core.config",
        "local_ai_core.container",
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
