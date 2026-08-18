from local_ai_core.inference.utils import inject_system_instruction_if_needed


def test_chat_message_state_moves_late_system_hints_before_transcript():
    messages = inject_system_instruction_if_needed(
        [
            {"role": "system", "content": "Previous conversation summary: hackathon planning."},
            {"role": "user", "content": "해커톤을 시작할까?"},
            {"role": "assistant", "content": "좋은 시기입니다."},
            {"role": "system", "content": "Primary context anchor: hackathon."},
            {"role": "user", "content": "큰 규모 위주가 좋을까?"},
            {"role": "system", "content": "Answer the current message directly."},
        ],
        response_language="ko",
    )

    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user"]
    assert "Primary context anchor" in messages[0]["content"]
    assert "Answer the current message directly" in messages[0]["content"]
    assert messages[-1]["content"] == "큰 규모 위주가 좋을까?"


def test_chat_message_state_drops_leading_assistant_fragment():
    messages = inject_system_instruction_if_needed(
        [
            {"role": "assistant", "content": "."},
            {"role": "user", "content": "이어서 설명해줘"},
        ],
        response_language="ko",
    )

    assert [item["role"] for item in messages] == ["system", "user"]
