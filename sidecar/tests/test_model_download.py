from __future__ import annotations

from datetime import datetime, timezone

from local_ai_core.models import (
    DownloadProgressItem,
    DownloadProgressKind,
    DownloadProgressStatus,
    LocalEngine,
    ModelDownloadResponse,
    ModelListItem,
)


def test_model_download_and_list_endpoints(client, auth_headers, monkeypatch):
    from local_ai_core import main

    def fake_download_model(*, url: str, engine: LocalEngine, filename: str | None):
        return ModelDownloadResponse(
            file_name=filename or "demo.gguf",
            saved_path="/tmp/models/demo.gguf",
            engine=engine,
            bytes_written=1234,
            total_bytes=4321,
            progress_percent=100.0,
            download_id="direct:llama_cpp:demo.gguf:test",
        )

    def fake_list_models():
        return [
            ModelListItem(
                file_name="demo.gguf",
                path="/tmp/models/demo.gguf",
                engine=LocalEngine.LLAMA_CPP,
                size_bytes=1234,
                modified_at=datetime.now(tz=timezone.utc),
            )
        ]

    def fake_download_progress():
        return [
            DownloadProgressItem(
                download_id="direct:llama_cpp:demo.gguf:test",
                kind=DownloadProgressKind.DIRECT,
                status=DownloadProgressStatus.COMPLETED,
                model_id=None,
                engine=LocalEngine.LLAMA_CPP,
                file_name="demo.gguf",
                downloaded_bytes=1234,
                total_bytes=4321,
                progress_percent=100.0,
                detail="done",
                error=None,
                updated_at=datetime.now(tz=timezone.utc),
            )
        ]

    monkeypatch.setattr(main.app_state.model_manager, "download_model", fake_download_model)
    monkeypatch.setattr(main.app_state.model_manager, "list_models", fake_list_models)
    monkeypatch.setattr(main.app_state.model_manager, "get_download_progress", fake_download_progress)

    downloaded = client.post(
        "/v1/models/download",
        headers=auth_headers,
        json={"url": "https://example.com/demo.gguf", "engine": "llama_cpp", "filename": "demo.gguf"},
    )
    assert downloaded.status_code == 200
    assert downloaded.json()["engine"] == "llama_cpp"
    assert downloaded.json()["saved_path"].endswith("demo.gguf")

    listed = client.get("/v1/models", headers=auth_headers)
    assert listed.status_code == 200
    assert len(listed.json()["models"]) == 1

    progress = client.get("/v1/models/download/progress", headers=auth_headers)
    assert progress.status_code == 200
    items = progress.json()["items"]
    assert len(items) == 1
    assert items[0]["kind"] == "direct"
    assert items[0]["status"] == "completed"
