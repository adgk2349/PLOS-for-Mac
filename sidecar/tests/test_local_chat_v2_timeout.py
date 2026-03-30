from __future__ import annotations

import asyncio

from local_ai_core import main


def test_local_chat_v2_timeout_returns_504(client, auth_headers):
    async def _timeout(_payload):
        raise asyncio.TimeoutError()

    main.app_state.chat_facade.local_chat_v2 = _timeout
    resp = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={"query": "timeout check", "mode": "GENERAL"},
    )
    assert resp.status_code == 504
    assert resp.json().get("detail") == "local inference timeout"
