from __future__ import annotations

from local_ai_core.db import Database
from local_ai_core.models import PrivacyMode
import local_ai_core.db as db_module


class _ImmediateThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self.daemon = daemon

    def start(self):
        if self._target is not None:
            self._target()


class _DummyDocker:
    def __init__(self):
        self.calls: list[tuple] = []

    def start(self, *, keep_running: bool = False) -> bool:
        self.calls.append(("start", keep_running))
        return True

    def stop(self, *, shutdown_desktop: bool = False, remove_stack: bool = False) -> bool:
        self.calls.append(("stop", shutdown_desktop, remove_stack))
        return True


def test_update_settings_does_not_stop_searxng_on_unrelated_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module.threading, "Thread", _ImmediateThread)
    docker = _DummyDocker()
    db = Database(tmp_path / "settings.sqlite3", docker=docker)
    docker.calls.clear()

    updated = db.get_settings().model_copy(update={"privacy_mode": PrivacyMode.LOCAL_ONLY})
    db.update_settings(updated)

    assert docker.calls == []


def test_update_settings_only_reacts_to_auto_start_toggle(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module.threading, "Thread", _ImmediateThread)
    docker = _DummyDocker()
    db = Database(tmp_path / "settings.sqlite3", docker=docker)
    docker.calls.clear()

    enable = db.get_settings().model_copy(update={"auto_start_searxng": True})
    db.update_settings(enable)
    assert docker.calls == [("start", True)]

    docker.calls.clear()
    still_enabled = db.get_settings().model_copy(
        update={
            "auto_start_searxng": True,
            "privacy_mode": PrivacyMode.LOCAL_ONLY,
        }
    )
    db.update_settings(still_enabled)
    assert docker.calls == []

    disable = db.get_settings().model_copy(update={"auto_start_searxng": False})
    db.update_settings(disable)
    assert docker.calls == [("stop", False, False)]
