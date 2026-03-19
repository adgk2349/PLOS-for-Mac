from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from local_ai_core.db import Database
from local_ai_core.models import WorkspaceResponse


def test_workspace_identity_hash_is_order_independent(tmp_path: Path):
    db = Database(tmp_path / "memory.sqlite3")
    a = WorkspaceResponse(
        included_paths=["/tmp/a", "/tmp/b"],
        excluded_paths=["/tmp/x", "/tmp/y"],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    b = WorkspaceResponse(
        included_paths=["/tmp/b", "/tmp/a"],
        excluded_paths=["/tmp/y", "/tmp/x"],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )

    id_a = db.get_workspace_identity(a)
    id_b = db.get_workspace_identity(b)
    assert id_a.workspace_id == id_b.workspace_id
    assert id_a.included_paths_hash == id_b.included_paths_hash


def test_memory_endpoints_flow(client, auth_headers):
    write = client.post(
        "/v1/memory/events",
        headers=auth_headers,
        json={
            "event_type": "query",
            "session_id": "s1",
            "workspace_id": "w1",
            "summary": "자료구조 시험 범위 정리",
            "related_file_ids": ["doc-1"],
            "related_action_ids": [],
            "metadata_json": {"mode": "GENERAL", "intent": "find_file"},
            "importance": 0.5,
        },
    )
    assert write.status_code == 200
    assert write.json()["accepted"] is True

    session = client.get("/v1/memory/session/relevant", headers=auth_headers, params={"session_id": "s1"})
    assert session.status_code == 200
    session_items = session.json()["items"]
    assert len(session_items) >= 1

    recent_query = next((item for item in session_items if item["key"] == "recent_query"), None)
    assert recent_query is not None
    assert "summary" in recent_query["value_json"]

    pin = client.post(
        "/v1/memory/pin",
        headers=auth_headers,
        json={
            "memory_id": recent_query["id"],
            "scope": "workspace",
            "workspace_id": "w1",
            "title": None,
            "content": None,
        },
    )
    assert pin.status_code == 200
    pin_id = pin.json()["item"]["id"]

    pins = client.get("/v1/memory/pins", headers=auth_headers, params={"workspace_id": "w1"})
    assert pins.status_code == 200
    assert any(item["id"] == pin_id for item in pins.json()["items"])

    clear = client.post(
        "/v1/memory/clear",
        headers=auth_headers,
        json={"scope": "session", "session_id": "s1"},
    )
    assert clear.status_code == 200
    assert clear.json()["scope"] == "session"

    session_after = client.get("/v1/memory/session/relevant", headers=auth_headers, params={"session_id": "s1"})
    assert session_after.status_code == 200
    assert session_after.json()["items"] == []


def test_workspace_memory_mode_pinned_only_disables_workspace_learning(client, auth_headers):
    settings = client.get("/v1/settings", headers=auth_headers)
    assert settings.status_code == 200
    payload = settings.json()
    payload["workspace_memory_mode"] = "pinned_only"
    payload["workspace_memory_enabled"] = True
    save = client.put("/v1/settings", headers=auth_headers, json=payload)
    assert save.status_code == 200

    write = client.post(
        "/v1/memory/events",
        headers=auth_headers,
        json={
            "event_type": "manual_override",
            "session_id": "s-mode",
            "workspace_id": "w-mode",
            "summary": "set defaults",
            "related_file_ids": [],
            "related_action_ids": [],
            "metadata_json": {"default_mode": "DEVELOPMENT"},
            "importance": 0.9,
        },
    )
    assert write.status_code == 200

    workspace = client.get(
        "/v1/memory/workspace/relevant",
        headers=auth_headers,
        params={"workspace_id": "w-mode", "intent": "compare_files"},
    )
    assert workspace.status_code == 200
    assert workspace.json()["items"] == []

    session = client.get("/v1/memory/session/relevant", headers=auth_headers, params={"session_id": "s-mode"})
    assert session.status_code == 200
    assert len(session.json()["items"]) >= 1
