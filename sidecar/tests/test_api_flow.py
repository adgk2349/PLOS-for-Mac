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
