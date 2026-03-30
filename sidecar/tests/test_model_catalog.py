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
    SettingsModel,
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
    assert "qwen35_9b_gguf" in ids
    assert "gemma3_12b_gguf" in ids
    assert "deepseek_r1_distill_qwen14b_gguf" in ids
    assert "llama31_8b_mlx_advanced" not in ids
    assert "qwen3_8b_gguf_advanced" not in ids
    assert "gpt_oss_20b_gguf" in ids
    assert "gpt_oss_120b_gguf" in ids
    assert "qwen35_397b_a17b_gguf" in ids
    assert "kimi25_gguf" in ids


def test_install_catalog_model_blocks_when_memory_is_insufficient(tmp_path: Path, monkeypatch):
    manager = ModelManager(tmp_path)
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "8")

    with monkeypatch.context() as ctx:
        ctx.setattr(manager, "_download_huggingface_repo", lambda _model: str(tmp_path / "dummy"))
        ctx.setattr(manager, "_download_huggingface_file", lambda _model: str(tmp_path / "dummy.gguf"))
        with pytest.raises(ValueError) as excinfo:
            manager.install_catalog_model("gpt_oss_20b_gguf")
        assert "최소 권장 사양은 16GB" in str(excinfo.value)


def test_catalog_status_ignores_stale_engine_mismatch_install_record(tmp_path: Path):
    manager = ModelManager(tmp_path)

    stale_dir = tmp_path / "models" / "llama_cpp" / "qwen35_9b_gguf"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_file = stale_dir / "Qwen3.5-9B-Q4_K_M.gguf"
    stale_file.write_text("placeholder", encoding="utf-8")

    state = manager._load_state()
    manager._set_state_record(
        state,
        "qwen35_9b_gguf",
        status=ModelInstallStatus.INSTALLED,
        installed_path=str(stale_file),
        failure_reason=None,
    )
    manager._save_state(state)

    catalog = manager.catalog_with_status(SettingsModel(local_engine=LocalEngine.MLX))
    target = next(item for item in catalog.models if item.id == "qwen35_9b_gguf")
    assert target.status == ModelInstallStatus.NOT_INSTALLED
    assert target.installed_path is None


def test_activate_catalog_model_rejects_stale_engine_mismatch_record(tmp_path: Path):
    manager = ModelManager(tmp_path)

    stale_dir = tmp_path / "models" / "llama_cpp" / "qwen35_9b_gguf"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_file = stale_dir / "Qwen3.5-9B-Q4_K_M.gguf"
    stale_file.write_text("placeholder", encoding="utf-8")

    state = manager._load_state()
    manager._set_state_record(
        state,
        "qwen35_9b_gguf",
        status=ModelInstallStatus.INSTALLED,
        installed_path=str(stale_file),
        failure_reason=None,
    )
    manager._save_state(state)

    with pytest.raises(FileNotFoundError) as excinfo:
        manager.activate_catalog_model("qwen35_9b_gguf")
    assert "호환되지 않습니다" in str(excinfo.value)


def test_activate_catalog_model_normalizes_llama_repo_directory_to_latest_gguf(tmp_path: Path):
    manager = ModelManager(tmp_path)

    model_id = "gpt_oss_120b_gguf"
    install_dir = tmp_path / "models" / "llama_cpp" / model_id
    install_dir.mkdir(parents=True, exist_ok=True)
    older = install_dir / "gpt-oss-120b-mxfp4-00001-of-00002.gguf"
    newer = install_dir / "gpt-oss-120b-mxfp4-00002-of-00002.gguf"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")

    state = manager._load_state()
    manager._set_state_record(
        state,
        model_id,
        status=ModelInstallStatus.INSTALLED,
        installed_path=str(install_dir),
        failure_reason=None,
    )
    manager._save_state(state)

    activated = manager.activate_catalog_model(model_id)
    assert activated.engine == LocalEngine.LLAMA_CPP
    assert activated.model_path.endswith("gpt-oss-120b-mxfp4-00002-of-00002.gguf")
