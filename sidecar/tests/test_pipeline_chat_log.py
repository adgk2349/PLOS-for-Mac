from __future__ import annotations

from pathlib import Path

from local_ai_core.reasoning.pipeline import ReasoningPipeline


def test_append_chat_log_is_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LOCAL_AI_CHAT_LOG_ENABLED", raising=False)
    monkeypatch.setenv("LOCAL_AI_CHAT_LOG_PATH", str(tmp_path / "chat.jsonl"))

    pipeline = ReasoningPipeline()
    pipeline._append_chat_log({"event": "request_start", "query": "hello"})

    assert not (tmp_path / "chat.jsonl").exists()
