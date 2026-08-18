from pathlib import Path

from local_ai_core.reasoning.conversation_context_policy import resolve_conversation_context_policy


def test_policy_uses_verbatim_history_without_duplicate_prompt_digest():
    policy = resolve_conversation_context_policy(
        startup_profile="RECOMMENDED",
        model_path=None,
        digest={"recent_turns": [{"role": "user", "text": "이전 대화"}]},
    )

    assert policy.history_turn_pairs == 6
    assert policy.inject_digest_into_legacy_prompt is False


def test_policy_falls_back_to_digest_prompt_without_verbatim_history():
    policy = resolve_conversation_context_policy(
        startup_profile="FAST",
        model_path=None,
        digest={"rolling_summary": "이전 대화 요약"},
    )

    assert policy.history_turn_pairs == 4
    assert policy.inject_digest_into_legacy_prompt is True


def test_deep_large_context_model_expands_verbatim_window(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 65536}', encoding="utf-8")

    policy = resolve_conversation_context_policy(
        startup_profile="DEEP",
        model_path=str(tmp_path),
        digest={"recent_turns": [{"role": "assistant", "text": "이전 결론"}]},
    )

    assert policy.history_turn_pairs == 10
    assert policy.rolling_summary_chars == 700
