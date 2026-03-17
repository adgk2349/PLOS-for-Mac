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


def test_docs_list_and_metadata_override_flow(client, auth_headers, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "proposal_2025.md"
    source.write_text("프로젝트 기획안 초안. 일정과 요구사항 정리.", encoding="utf-8")

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

    docs = client.get("/v1/docs", headers=auth_headers)
    assert docs.status_code == 200
    payload = docs.json()
    assert payload["total"] >= 1
    doc = payload["documents"][0]
    assert "category" in doc
    assert "tags" in doc

    updated = client.put(
        f"/v1/docs/{doc['doc_id']}/metadata",
        headers=auth_headers,
        json={
            "category": "코드관련",
            "tags": ["Swift", "RAG"],
            "importance": 0.9,
            "excluded": True,
        },
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["category"] == "코드관련"
    assert "Swift" in body["tags"]
    assert body["excluded"] is True

    # Default listing excludes excluded=true documents.
    visible_docs = client.get("/v1/docs", headers=auth_headers)
    assert all(item["doc_id"] != doc["doc_id"] for item in visible_docs.json()["documents"])

    hidden_docs = client.get("/v1/docs?excluded=true", headers=auth_headers)
    assert any(item["doc_id"] == doc["doc_id"] for item in hidden_docs.json()["documents"])

    reclassified = client.post(f"/v1/docs/{doc['doc_id']}/reclassify", headers=auth_headers)
    assert reclassified.status_code == 200
    # User overrides should still win after reclassify.
    assert reclassified.json()["category"] == "코드관련"
