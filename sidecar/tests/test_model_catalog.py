from __future__ import annotations

from pathlib import Path

import pytest

from local_ai_core.model_manager import ModelManager
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


def test_catalog_manifest_includes_high_quality_options(tmp_path: Path):
    manager = ModelManager(tmp_path)
    manifest = manager._load_catalog_manifest()
    ids = {item.id for item in manifest.models}
    assert "qwen25_7b_mlx_advanced" in ids
    assert "gemma3_12b_gguf" in ids
    assert "deepseek_r1_distill_qwen14b_gguf" in ids
    assert "llama31_8b_mlx_advanced" in ids
    assert "gpt_oss_20b_gguf" in ids
    assert "gpt_oss_120b_gguf" in ids
    assert "qwen35_397b_a17b_gguf" in ids
    assert "kimi25_gguf" in ids


def test_install_catalog_model_blocks_when_memory_is_insufficient(tmp_path: Path, monkeypatch):
    manager = ModelManager(tmp_path)
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "16")

    with monkeypatch.context() as ctx:
        ctx.setattr(manager, "_download_huggingface_repo", lambda _model: str(tmp_path / "dummy"))
        ctx.setattr(manager, "_download_huggingface_file", lambda _model: str(tmp_path / "dummy.gguf"))
        with pytest.raises(ValueError) as excinfo:
            manager.install_catalog_model("gpt_oss_20b_gguf")
        assert "최소 권장 사양은 64GB" in str(excinfo.value)
