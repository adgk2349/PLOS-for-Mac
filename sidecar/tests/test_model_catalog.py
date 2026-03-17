from __future__ import annotations

from local_ai_core.models import (
    LocalEngine,
    ModelCatalogActivateResponse,
    ModelCatalogDeleteResponse,
    ModelCatalogInstallResponse,
    ModelCatalogItemState,
    ModelCatalogResponse,
    ModelInstallStatus,
    ModelSupportFlags,
)


def _sample_catalog_item() -> ModelCatalogItemState:
    return ModelCatalogItemState(
        id="llama32_3b_mlx_balanced",
        name="Llama 3.2 3B Instruct",
        profile="balanced",
        engine=LocalEngine.MLX,
        distribution_type="huggingface_repo",
        repo_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        filename=None,
        download_label="추천 설정",
        description="balanced preset",
        size_gb=2.3,
        recommended_for=["M1", "M2"],
        recommended_memory_gb=8,
        tags=["balanced", "mlx"],
        supports=ModelSupportFlags(chat=True, rag=True, tool_use=False, vision=False),
        default=True,
        status=ModelInstallStatus.NOT_INSTALLED,
        installed_path=None,
        active=False,
        failure_reason=None,
    )


def test_model_catalog_endpoints(client, auth_headers, monkeypatch):
    from local_ai_core import main

    sample_catalog = ModelCatalogResponse(version=1, default_profile="balanced", models=[_sample_catalog_item()])

    monkeypatch.setattr(main.app_state.model_manager, "catalog_with_status", lambda _settings: sample_catalog)

    listed = client.get("/v1/models/catalog", headers=auth_headers)
    assert listed.status_code == 200
    body = listed.json()
    assert body["default_profile"] == "balanced"
    assert len(body["models"]) == 1

    def fake_install(model_id: str):
        return ModelCatalogInstallResponse(
            model_id=model_id,
            status=ModelInstallStatus.INSTALLED,
            engine=LocalEngine.MLX,
            saved_path="/tmp/mlx-model",
            detail="ok",
        )

    monkeypatch.setattr(main.app_state.model_manager, "install_catalog_model", fake_install)

    installed = client.post(
        "/v1/models/catalog/install",
        headers=auth_headers,
        json={"model_id": "llama32_3b_mlx_balanced"},
    )
    assert installed.status_code == 200
    assert installed.json()["status"] == "installed"

    def fake_activate(model_id: str):
        return ModelCatalogActivateResponse(
            model_id=model_id,
            engine=LocalEngine.MLX,
            model_path="/tmp/mlx-model",
            profile="fast",
        )

    monkeypatch.setattr(main.app_state.model_manager, "activate_catalog_model", fake_activate)

    activated = client.post(
        "/v1/models/catalog/activate",
        headers=auth_headers,
        json={"model_id": "llama32_3b_mlx_balanced"},
    )
    assert activated.status_code == 200
    assert activated.json()["engine"] == "mlx"

    settings = client.get("/v1/settings", headers=auth_headers)
    assert settings.status_code == 200
    settings_body = settings.json()
    assert settings_body["local_engine"] == "mlx"
    assert settings_body["mlx_model_path"] == "/tmp/mlx-model"
    assert settings_body["startup_profile"] == "FAST"

    def fake_delete(model_id: str):
        return ModelCatalogDeleteResponse(model_id=model_id, removed=True)

    monkeypatch.setattr(main.app_state.model_manager, "delete_catalog_model", fake_delete)

    deleted = client.delete("/v1/models/catalog/llama32_3b_mlx_balanced", headers=auth_headers)
    assert deleted.status_code == 200
    assert deleted.json()["removed"] is True
