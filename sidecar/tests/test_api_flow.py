from __future__ import annotations

import time
from pathlib import Path


def _poll_job(client, headers: dict[str, str], job_id: str, timeout: float = 20.0) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f"/v1/index/jobs/{job_id}", headers=headers)
        res.raise_for_status()
        payload = res.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not finish in time")


def test_folder_to_index_to_chat_flow(client, auth_headers, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "project.md"
    source.write_text("Local AI Core project goal is private RAG on Mac.", encoding="utf-8")

    res = client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(workspace)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )
    assert res.status_code == 200

    job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "full"})
    assert job.status_code == 200
    job_id = job.json()["job_id"]

    status_payload = _poll_job(client, auth_headers, job_id)
    assert status_payload["status"] == "completed"
    assert status_payload["processed_files"] >= 1

    chat = client.post(
        "/v1/chat/local",
        headers=auth_headers,
        json={
            "query": "What is the project goal?",
            "mode": "GENERAL",
            "conversation_id": "conv-1",
        },
    )
    assert chat.status_code == 200
    body = chat.json()
    assert body["is_local"] is True
    assert body["lead"]
    assert body["result_summary"]
    assert isinstance(body["actions"], list)
    assert len(body["actions"]) >= 1
    action_kinds = {item["kind"] for item in body["actions"]}
    assert "ASK_FOLLOWUP" in action_kinds
    assert "OPEN_FILE" in action_kinds
    assert len(body["citations"]) >= 1


def test_incremental_reindex_updates_content(client, auth_headers, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "notes.txt"
    source.write_text("alpha baseline", encoding="utf-8")

    client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(workspace)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )

    full_job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "full"})
    _poll_job(client, auth_headers, full_job.json()["job_id"])

    source.write_text("alpha baseline\nnewtoken123 appears here", encoding="utf-8")
    time.sleep(0.05)

    inc_job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "incremental"})
    status_payload = _poll_job(client, auth_headers, inc_job.json()["job_id"])
    assert status_payload["status"] == "completed"

    chat = client.post(
        "/v1/chat/local",
        headers=auth_headers,
        json={
            "query": "newtoken123",
            "mode": "GENERAL",
        },
    )
    assert chat.status_code == 200
    snippets = [c["snippet"] for c in chat.json()["citations"]]
    assert any("newtoken123" in snippet for snippet in snippets)


def test_incremental_reindex_prunes_deleted_file(client, auth_headers, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "to-delete.txt"
    source.write_text("deleteme_unique_token_9281", encoding="utf-8")

    client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(workspace)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )

    full_job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "full"})
    _poll_job(client, auth_headers, full_job.json()["job_id"])

    source.unlink()
    inc_job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "incremental"})
    _poll_job(client, auth_headers, inc_job.json()["job_id"])

    chat = client.post(
        "/v1/chat/local",
        headers=auth_headers,
        json={
            "query": "deleteme_unique_token_9281",
            "mode": "GENERAL",
        },
    )
    assert chat.status_code == 200
    payload = chat.json()
    assert all(item["file_path"] != str(source) for item in payload["citations"])
    assert all("deleteme_unique_token_9281" not in item["snippet"] for item in payload["citations"])


def test_workspace_scope_blocks_removed_folder_without_reindex(client, auth_headers, tmp_path: Path):
    folder_a = tmp_path / "folderA"
    folder_b = tmp_path / "folderB"
    folder_a.mkdir(parents=True, exist_ok=True)
    folder_b.mkdir(parents=True, exist_ok=True)
    file_a = folder_a / "a.txt"
    file_b = folder_b / "b.txt"
    file_a.write_text("datastructure_unique_a_token", encoding="utf-8")
    file_b.write_text("out_of_scope_unique_b_token", encoding="utf-8")

    client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(folder_a), str(folder_b)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )
    full_job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "full"})
    _poll_job(client, auth_headers, full_job.json()["job_id"])

    # Narrow workspace to folderA only, without running another index job.
    client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(folder_a)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )

    docs = client.get("/v1/docs", headers=auth_headers)
    assert docs.status_code == 200
    doc_paths = [item["path"] for item in docs.json()["documents"]]
    assert str(file_a) in doc_paths
    assert str(file_b) not in doc_paths

    chat = client.post(
        "/v1/chat/local",
        headers=auth_headers,
        json={
            "query": "out_of_scope_unique_b_token",
            "mode": "GENERAL",
        },
    )
    assert chat.status_code == 200
    payload = chat.json()
    assert all(item["file_path"] != str(file_b) for item in payload["citations"])


def test_external_call_privacy_gate(client, auth_headers, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(workspace)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )

    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={
            "privacy_mode": "LOCAL_ONLY",
            "startup_profile": "RECOMMENDED",
            "model_profile": "recommended",
            "reindex_policy": "filewatch_incremental",
            "language": "ko-KR",
        },
    )

    blocked = client.post(
        "/v1/chat/deep-analysis",
        headers=auth_headers,
        json={
            "query": "Need deeper analysis",
            "mode": "GENERAL",
            "provider": "openai",
            "selected_citations": [],
            "user_confirmed": True,
        },
    )
    assert blocked.status_code == 403

    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={
            "privacy_mode": "HYBRID",
            "startup_profile": "RECOMMENDED",
            "model_profile": "recommended",
            "reindex_policy": "filewatch_incremental",
            "language": "ko-KR",
        },
    )

    allowed = client.post(
        "/v1/chat/deep-analysis",
        headers=auth_headers,
        json={
            "query": "Need deeper analysis",
            "mode": "GENERAL",
            "provider": "openai",
            "selected_citations": [],
            "user_confirmed": True,
        },
    )
    assert allowed.status_code == 200
    assert allowed.json()["provider"] == "openai"


def test_strict_search_returns_composed_insufficient_shape(client, auth_headers, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "notes.txt"
    source.write_text("totally unrelated content", encoding="utf-8")

    client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(workspace)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )
    job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "full"})
    _poll_job(client, auth_headers, job.json()["job_id"])

    strict = client.post(
        "/v1/chat/local",
        headers=auth_headers,
        json={
            "query": "quantum tunneling theorem in local files",
            "mode": "STRICT_SEARCH",
            "conversation_id": "conv-strict",
        },
    )
    assert strict.status_code == 200
    payload = strict.json()
    assert payload["intent"] in {"DOCUMENT_QA", "AMBIGUOUS", "TASK_REQUEST", "FILE_SEARCH"}
    assert payload["lead"]
    assert payload["result_summary"]
    assert payload["citations"] == []
    assert any(action["kind"] == "ASK_FOLLOWUP" for action in payload["actions"])
