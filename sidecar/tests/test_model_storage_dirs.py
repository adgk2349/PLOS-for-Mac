from __future__ import annotations

from pathlib import Path

from local_ai_core.local_inference import LocalInferenceEngine
from local_ai_core.model_manager import ModelManager
from local_ai_core.models import LocalEngine


def test_model_manager_uses_explicit_models_dir(tmp_path: Path):
    data_dir = tmp_path / "data"
    custom_models_dir = tmp_path / "custom-models"
    manager = ModelManager(data_dir, models_dir=custom_models_dir)

    sample = custom_models_dir / "llama_cpp" / "sample.gguf"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"GGUF")

    listed = manager.list_models()
    assert any(item.path == str(sample) for item in listed)


def test_runtime_discovery_uses_local_ai_models_dir(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    target = models_dir / "llama_cpp" / "demo.gguf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"GGUF")

    monkeypatch.setenv("LOCAL_AI_MODELS_DIR", str(models_dir))
    engine = LocalInferenceEngine()
    discovered = engine._discover_downloaded_model(LocalEngine.LLAMA_CPP)

    assert discovered == str(target)

