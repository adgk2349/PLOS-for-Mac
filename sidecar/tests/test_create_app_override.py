from __future__ import annotations

from local_ai_core import main


def test_create_app_accepts_state_override():
    state = main.AppState()
    _ = main.create_app(state=state)
    assert main.app_state is state
