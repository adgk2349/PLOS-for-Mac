from __future__ import annotations

from local_ai_core.models import LocalEngine, RuntimePrepareResponse


def test_runtime_prepare_endpoint(client, auth_headers, monkeypatch):
    from local_ai_core import main

    def fake_prepare_runtime(*, engine: LocalEngine, profile: str, mlx_model_path: str | None, llama_model_path: str | None):
        return RuntimePrepareResponse(
            engine=engine,
            ready=True,
            package_available=True,
            model_path=llama_model_path if engine == LocalEngine.LLAMA_CPP else mlx_model_path,
            model_exists=True,
            accelerator="Metal GPU offload 가능",
            detail="runtime ready",
        )

    monkeypatch.setattr(main.app_state.local_inference, "prepare_runtime", fake_prepare_runtime)

    response = client.post(
        "/v1/models/runtime/prepare",
        headers=auth_headers,
        json={"engine": "llama_cpp", "model_path": "/tmp/demo.gguf"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "llama_cpp"
    assert payload["ready"] is True
    assert payload["package_available"] is True
