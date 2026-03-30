from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from local_ai_core.local_inference import LocalInferenceEngine
from local_ai_core.model_manager import ModelManager
from local_ai_core.models import LocalEngine


def test_local_inference_unload_and_health_smoke():
    engine = LocalInferenceEngine()
    detail = engine.unload("all")
    assert "model_unload" in detail or "noop" in detail
    health = engine.health()
    assert isinstance(health, dict)
    assert "loaded" in health
    assert "policy" in health


def test_model_manager_runtime_lifecycle_without_controller(tmp_path: Path):
    manager = ModelManager(tmp_path)
    ok, detail = manager.load(engine=LocalEngine.MLX, model_ref=None)
    assert ok is False
    assert "no_runtime_controller" in detail
    out = manager.unload("all")
    assert "no_runtime_controller" in out


def test_model_list_marks_loaded_from_runtime_health(tmp_path: Path):
    manager = ModelManager(tmp_path)
    model_dir = (tmp_path / "models" / "mlx")
    model_dir.mkdir(parents=True, exist_ok=True)
    f = model_dir / "demo.gguf"
    f.write_bytes(b"test")
    now = datetime.now(timezone.utc).isoformat()
    models = manager.list_models(
        runtime_health={
            "loaded": [
                {
                    "engine": "mlx",
                    "path": str(f.resolve()),
                    "last_used_at": now,
                }
            ]
        }
    )
    target = next(item for item in models if item.path == str(f))
    assert target.loaded is True
    assert target.resident_engine == LocalEngine.MLX
    assert target.last_used_at is not None

