from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def _poll_room_ready(client, headers: dict[str, str], room_id: str, timeout: float = 20.0) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f"/v1/rooms/{room_id}/storage/status", headers=headers)
        res.raise_for_status()
        payload = res.json()
        variants = payload.get("variants") or []
        if variants:
            top = variants[0]
            if int(top.get("indexed_docs") or 0) > 0 and int(top.get("chunk_count") or 0) > 0:
                return payload
        time.sleep(0.2)
    raise TimeoutError(f"room {room_id} storage did not become ready")


def test_room_storage_reindex_status_and_delete(client, auth_headers, tmp_path: Path):
    room_a = "room-a"
    room_b = "room-b"
    folder_a = tmp_path / "room-a-files"
    folder_b = tmp_path / "room-b-files"
    folder_a.mkdir(parents=True, exist_ok=True)
    folder_b.mkdir(parents=True, exist_ok=True)
    (folder_a / "alpha.txt").write_text("alpha_room_token", encoding="utf-8")
    (folder_b / "beta.txt").write_text("beta_room_token", encoding="utf-8")

    reindex_a = client.post(
        f"/v1/rooms/{room_a}/storage/reindex",
        headers=auth_headers,
        json={"scope": "full", "included_paths": [str(folder_a)], "excluded_paths": []},
    )
    assert reindex_a.status_code == 200
    assert reindex_a.json()["ok"] is True

    ready_a = _poll_room_ready(client, auth_headers, room_a)
    assert ready_a["variant_count"] >= 1
    variant_a = ready_a["variants"][0]
    db_path_a = Path(variant_a["data_dir"]) / "local_ai_core.sqlite3"
    assert db_path_a.exists()
    with sqlite3.connect(db_path_a) as conn:
        rows = conn.execute("SELECT path FROM documents").fetchall()
        assert rows
        assert all(str(folder_a) in str(row[0]) for row in rows)

    reindex_b = client.post(
        f"/v1/rooms/{room_b}/storage/reindex",
        headers=auth_headers,
        json={"scope": "full", "included_paths": [str(folder_b)], "excluded_paths": []},
    )
    assert reindex_b.status_code == 200
    assert reindex_b.json()["ok"] is True

    ready_b = _poll_room_ready(client, auth_headers, room_b)
    assert ready_b["variant_count"] >= 1
    variant_b = ready_b["variants"][0]
    assert variant_a["room_storage_id"] != variant_b["room_storage_id"]

    delete_a = client.delete(f"/v1/rooms/{room_a}/storage", headers=auth_headers)
    assert delete_a.status_code == 200
    assert delete_a.json()["removed"] is True

    after_delete = client.get(f"/v1/rooms/{room_a}/storage/status", headers=auth_headers)
    assert after_delete.status_code == 200
    assert after_delete.json()["variant_count"] == 0


def test_v2_chat_room_metadata_present(client, auth_headers, tmp_path: Path):
    room_id = "room-meta"
    folder = tmp_path / "room-meta-files"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "notes.txt").write_text("room_meta_token", encoding="utf-8")

    client.post(
        f"/v1/rooms/{room_id}/storage/reindex",
        headers=auth_headers,
        json={"scope": "full", "included_paths": [str(folder)], "excluded_paths": []},
    )
    _poll_room_ready(client, auth_headers, room_id)

    response = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "notes.txt 내용 요약해줘",
            "mode": "SUMMARY",
            "conversation_id": room_id,
            "session_id": room_id,
            "included_paths": [str(folder)],
            "excluded_paths": [],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    metadata = payload.get("metadata") or {}
    assert isinstance(metadata.get("room_storage_id"), str)
    assert isinstance(metadata.get("room_scope_hash"), str)
    assert metadata.get("room_index_state") in {"idle", "indexing", "ready", "failed"}


def test_room_storage_hash_isolation_for_slug_like_ids(client, auth_headers, tmp_path: Path):
    room_a = "team+a"
    room_b = "team_a"
    folder_a = tmp_path / "hash-room-a"
    folder_b = tmp_path / "hash-room-b"
    folder_a.mkdir(parents=True, exist_ok=True)
    folder_b.mkdir(parents=True, exist_ok=True)
    (folder_a / "a.txt").write_text("alpha", encoding="utf-8")
    (folder_b / "b.txt").write_text("beta", encoding="utf-8")

    res_a = client.post(
        f"/v1/rooms/{room_a}/storage/reindex",
        headers=auth_headers,
        json={"scope": "full", "included_paths": [str(folder_a)], "excluded_paths": []},
    )
    assert res_a.status_code == 200
    assert res_a.json().get("ok") is True
    res_b = client.post(
        f"/v1/rooms/{room_b}/storage/reindex",
        headers=auth_headers,
        json={"scope": "full", "included_paths": [str(folder_b)], "excluded_paths": []},
    )
    assert res_b.status_code == 200
    assert res_b.json().get("ok") is True

    ready_a = _poll_room_ready(client, auth_headers, room_a)
    ready_b = _poll_room_ready(client, auth_headers, room_b)
    assert ready_a["variant_count"] >= 1
    assert ready_b["variant_count"] >= 1
    assert ready_a["room_key"] != ready_b["room_key"]
    assert ready_a["variants"][0]["room_storage_id"] != ready_b["variants"][0]["room_storage_id"]


def test_v2_chat_room_scope_cached_when_included_paths_missing(client, auth_headers, tmp_path: Path):
    room_id = "room-scope-cache"
    folder = tmp_path / "scope-cache-folder"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "memo.md").write_text("scope cache works", encoding="utf-8")

    first = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "memo 내용 알려줘",
            "mode": "SUMMARY",
            "conversation_id": room_id,
            "session_id": room_id,
            "included_paths": [str(folder)],
            "excluded_paths": [],
        },
    )
    assert first.status_code == 200
    first_meta = first.json().get("metadata") or {}
    assert first_meta.get("memory_backend") == "room"
    assert isinstance(first_meta.get("room_storage_id"), str)

    second = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "방금 내용 다시 요약해줘",
            "mode": "SUMMARY",
            "conversation_id": room_id,
            "session_id": room_id,
            "included_paths": [],
            "excluded_paths": [],
        },
    )
    assert second.status_code == 200
    second_meta = second.json().get("metadata") or {}
    assert second_meta.get("memory_backend") == "room"
    assert second_meta.get("room_route_reason") == "room_scope_cached"


def test_v2_chat_without_scope_does_not_fallback_to_global_workspace(client, auth_headers):
    room_id = f"room-no-scope-first-turn-{time.time_ns()}"
    response = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "현재 방 상태 알려줘",
            "mode": "GENERAL",
            "conversation_id": room_id,
            "session_id": room_id,
            "included_paths": [],
            "excluded_paths": [],
        },
    )
    assert response.status_code == 200
    metadata = response.json().get("metadata") or {}
    assert metadata.get("memory_backend") == "global"
    assert metadata.get("room_route_reason") == "room_scope_missing"
    assert not metadata.get("room_storage_id")


def test_room_memory_api_isolation(client, auth_headers, tmp_path: Path):
    room_a = "room-memory-a"
    room_b = "room-memory-b"
    folder_a = tmp_path / "room-memory-a-folder"
    folder_b = tmp_path / "room-memory-b-folder"
    folder_a.mkdir(parents=True, exist_ok=True)
    folder_b.mkdir(parents=True, exist_ok=True)
    (folder_a / "a.md").write_text("alpha", encoding="utf-8")
    (folder_b / "b.md").write_text("beta", encoding="utf-8")

    for room, folder in ((room_a, folder_a), (room_b, folder_b)):
        reindex = client.post(
            f"/v1/rooms/{room}/storage/reindex",
            headers=auth_headers,
            json={"scope": "full", "included_paths": [str(folder)], "excluded_paths": []},
        )
        assert reindex.status_code == 200
        assert reindex.json().get("ok") is True
        _poll_room_ready(client, auth_headers, room)

    event_a = client.post(
        f"/v1/rooms/{room_a}/memory/events",
        headers=auth_headers,
        json={
            "event_type": "query",
            "session_id": room_a,
            "workspace_id": "ws-a",
            "summary": "alpha-event",
            "related_file_ids": [],
            "related_action_ids": [],
            "metadata_json": {"source": "test-a"},
            "importance": 0.7,
        },
    )
    assert event_a.status_code == 200
    assert event_a.json().get("accepted") is True

    event_b = client.post(
        f"/v1/rooms/{room_b}/memory/events",
        headers=auth_headers,
        json={
            "event_type": "query",
            "session_id": room_b,
            "workspace_id": "ws-b",
            "summary": "beta-event",
            "related_file_ids": [],
            "related_action_ids": [],
            "metadata_json": {"source": "test-b"},
            "importance": 0.7,
        },
    )
    assert event_b.status_code == 200
    assert event_b.json().get("accepted") is True

    episodic_a = client.get(
        f"/v1/rooms/{room_a}/memory/episodic/relevant",
        headers=auth_headers,
        params={"workspace_id": "ws-a", "intent": "query"},
    )
    assert episodic_a.status_code == 200
    summaries_a = [str(item.get("summary") or "") for item in (episodic_a.json().get("items") or [])]
    assert any("alpha-event" in s for s in summaries_a)
    assert all("beta-event" not in s for s in summaries_a)

    episodic_b = client.get(
        f"/v1/rooms/{room_b}/memory/episodic/relevant",
        headers=auth_headers,
        params={"workspace_id": "ws-b", "intent": "query"},
    )
    assert episodic_b.status_code == 200
    summaries_b = [str(item.get("summary") or "") for item in (episodic_b.json().get("items") or [])]
    assert any("beta-event" in s for s in summaries_b)
    assert all("alpha-event" not in s for s in summaries_b)


def test_development_mode_includes_review_metadata(client, auth_headers, tmp_path: Path):
    room_id = "room-dev-review"
    folder = tmp_path / "room-dev-review-folder"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "main.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    reindex = client.post(
        f"/v1/rooms/{room_id}/storage/reindex",
        headers=auth_headers,
        json={"scope": "full", "included_paths": [str(folder)], "excluded_paths": []},
    )
    assert reindex.status_code == 200
    assert reindex.json().get("ok") is True
    _poll_room_ready(client, auth_headers, room_id)

    response = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "main.py 코드 리뷰해줘",
            "mode": "DEVELOPMENT",
            "conversation_id": room_id,
            "session_id": room_id,
            "included_paths": [str(folder)],
            "excluded_paths": [],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    metadata = payload.get("metadata") or {}
    assert metadata.get("review_mode") is True
    assert isinstance(metadata.get("grounded_file_count"), int)
    assert isinstance(metadata.get("grounded_line_refs"), list)
