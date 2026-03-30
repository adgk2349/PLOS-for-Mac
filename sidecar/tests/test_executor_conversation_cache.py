from __future__ import annotations

from types import SimpleNamespace

from local_ai_core.executor import LocalExecutor
from local_ai_core.models import LocalEngine, StartupProfile, WorkMode


class _FakeInference:
    def __init__(self) -> None:
        self.calls = 0

    def generate_conversational(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            answer=f"ok-{self.calls}",
            engine_used=kwargs.get("engine", LocalEngine.MLX),
            used_fallback=False,
            detail="detail",
        )


def test_conversation_cache_hit_reuses_recent_result(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_CONV_CACHE_MAX", "8")
    monkeypatch.setenv("LOCAL_AI_CONV_CACHE_TTL_SECONDS", "60")
    fake = _FakeInference()
    executor = LocalExecutor(fake, capability_router=None)

    first = executor.execute_conversation(
        query="hello",
        mode=WorkMode.GENERAL,
        startup_profile=StartupProfile.RECOMMENDED,
        engine=LocalEngine.MLX,
        mlx_model_path="/tmp/a",
        llama_model_path=None,
        language_preference="ko",
        session_summary="s1",
        max_tokens=200,
    )
    second = executor.execute_conversation(
        query="hello",
        mode=WorkMode.GENERAL,
        startup_profile=StartupProfile.RECOMMENDED,
        engine=LocalEngine.MLX,
        mlx_model_path="/tmp/a",
        llama_model_path=None,
        language_preference="ko",
        session_summary="s1",
        max_tokens=200,
    )
    assert fake.calls == 1
    assert first.generated_text == second.generated_text
    assert any("prompt_cache:conversation_hit" in row for row in second.tool_logs)
