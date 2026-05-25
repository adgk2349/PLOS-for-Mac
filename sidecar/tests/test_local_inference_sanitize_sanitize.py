from __future__ import annotations

# Auto-split from large test module for maintainability.

import importlib

from datetime import datetime, timezone

from pathlib import Path

from local_ai_core.local_inference import LocalInferenceEngine

from local_ai_core.models import Citation, LocalEngine, WorkMode

class _StubInferenceEngine(LocalInferenceEngine):
    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs

    def _generate_with_engine(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
        style: str = "grounded",
        message_state: list[dict[str, str]] | None = None,
        response_language: str | None = None,
    ) -> str | None:
        output = self._outputs.get(engine)
        if output is None:
            self._set_engine_error(engine, f"{engine.value} stub failure")
            return None
        return output

    def _discover_downloaded_model(self, engine: LocalEngine) -> str | None:
        _ = engine
        return None

class _SequentialStubInferenceEngine(LocalInferenceEngine):
    def __init__(self, outputs_by_engine):
        super().__init__()
        self._outputs_by_engine = {engine: list(outputs) for engine, outputs in outputs_by_engine.items()}

    def _generate_with_engine(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
        style: str = "grounded",
        message_state: list[dict[str, str]] | None = None,
        response_language: str | None = None,
    ) -> str | None:
        values = self._outputs_by_engine.get(engine) or []
        if not values:
            self._set_engine_error(engine, f"{engine.value} stub failure")
            return None
        value = values.pop(0)
        if value is None:
            self._set_engine_error(engine, f"{engine.value} stub failure")
            return None
        return value

    def _discover_downloaded_model(self, engine: LocalEngine) -> str | None:
        _ = engine
        return None

def test_sanitize_generated_answer_dedupes_repeated_sentences():
    raw = "찾았습니다. 찾았습니다. 찾았습니다. 가장 관련 파일은 notes.md입니다. 가장 관련 파일은 notes.md입니다."
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert cleaned.count("찾았습니다.") <= 2
    assert "notes.md" in cleaned

def test_sanitize_generated_answer_removes_prompt_echo():
    prompt = "Mode: GENERAL Question: test"
    raw = f"{prompt} Mode: GENERAL Question: test 결과입니다."
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt=prompt)
    assert cleaned
    assert cleaned != raw

def test_sanitize_generated_answer_removes_answer_prefix_and_repeated_blocks():
    raw = (
        "Answer: 지금 강의는 컴퓨터 공학부 2024년 2학기 3~4학년에 대한 강의입니다. "
        "강의는 NE322-A 강의실에서 진행됩니다. "
        "지금 강의는 컴퓨터 공학부 2024년 2학기 3~4학년에 대한 강의입니다. "
        "강의는 NE322-A 강의실에서 진행됩니다."
    )
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert not cleaned.lower().startswith("answer:")
    assert cleaned.count("NE322-A") == 1

def test_sanitize_generated_answer_caps_long_unpunctuated_segment():
    raw = "token " * 900
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert len(cleaned) <= 1210
    assert cleaned

def test_sanitize_generated_answer_removes_reasoning_leak_lines():
    raw = (
        "User: 안녕하세요\n"
        "Follow-up question: 오늘 날씨는?\n"
        "Okay, let's see. The user asked about weather.\n"
        "안녕하세요. 오늘은 어느 지역 날씨를 확인할까요?"
    )
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert "Follow-up question" not in cleaned
    assert "Okay, let's see" not in cleaned
    assert "어느 지역 날씨" in cleaned

def test_sanitize_generated_answer_returns_empty_when_only_thought_leak():
    raw = "User: hi\nOkay, let's see. I should infer what the user means."
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert cleaned == ""

def test_sanitize_generated_answer_collapses_repeated_phrase_run():
    raw = (
        "좋아, 이 맥락 기준으로 바로 정리해볼게. "
        + ("전송을 시작하기 전에 " * 20)
        + "전송을 시작하기 전에 확인하세요."
    )
    cleaned = LocalInferenceEngine._sanitize_generated_answer(raw, prompt="")
    assert cleaned.count("전송을 시작하기 전에") <= 2
    assert len(cleaned) < len(raw)

def test_looks_conversational_answer_accepts_short_korean_greeting():
    assert LocalInferenceEngine._looks_conversational_answer(
        "안녕하세요!",
        response_language="ko",
        query="안녕",
    )

def test_looks_conversational_answer_rejects_role_only_token():
    assert not LocalInferenceEngine._looks_conversational_answer(
        "assistant.",
        response_language="ko",
        query="안녕",
    )

def test_postprocess_conversational_answer_strips_meta_prefix_phrase():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "Okay, I'll go with that response. 안녕하세요!",
        query="안녕",
        response_language="ko",
    )
    assert cleaned == "안녕하세요!"

def test_postprocess_conversational_answer_strips_you_a_prefixes():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "You: 네, 로컬 모델은 어디서든 쓸 수 있어요.\nA: 물 조금 마셔요.",
        query="로컬 모델을 우주에서도 쓸 수 있을까?",
        response_language="ko",
    )
    assert "You:" not in cleaned
    assert "A:" not in cleaned
    assert "로컬 모델은 어디서든" in cleaned

def test_postprocess_conversational_answer_dedupes_adjacent_repeat():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "네, 그냥 자연스럽게 말하려고 해요. 네, 그냥 자연스럽게 말하려고 해요.",
        query="너 왜케 클로드같이 말해",
        response_language="ko",
    )
    assert cleaned.count("자연스럽게 말하려고 해요") == 1

def test_postprocess_conversational_answer_collapses_comma_loop():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "말씀하세요. 귀 기울여 들을게요,,,,,,,",
        query="고민이 있는데 들어줄 수 있나",
        response_language="ko",
    )
    assert ",,,,," not in cleaned
    assert cleaned.endswith(".")

def test_postprocess_conversational_answer_limits_questions_to_one():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "어떤 메뉴 좋아하세요? 어떤 종류를 생각 중이세요? 김치찌개가 무난해요.",
        query="오늘 저녁 뭐 먹을까",
        response_language="ko",
    )
    assert cleaned.count("?") <= 1
    assert "김치찌개" in cleaned

def test_normalize_three_option_recommendation_formats_numbered_options():
    normalized = LocalInferenceEngine._normalize_three_option_recommendation(
        "김치찌개는 든든해요. 된장찌개는 담백해요. 순두부찌개는 가볍게 먹기 좋아요.",
        response_language="ko",
    )
    assert normalized.startswith("1. ")
    assert "\n2. " in normalized
    assert "\n3. " in normalized

def test_looks_conversational_answer_rejects_instructional_meta_response():
    assert not LocalInferenceEngine._looks_conversational_answer(
        "최대한 한 번만 물어보세요. 최대한 1~3문장으로만 답하세요.",
        response_language="ko",
        query="여긴 많은편이야",
    )

def test_postprocess_conversational_answer_strips_question_mark_rule_leak():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "사용자에게 물어볼 때는 반드시 '?'를 붙여주세요. 최종 답변: 현재 시간은 10시 30분입니다.",
        query="지금 몇 시야?",
        response_language="ko",
    )
    assert cleaned == "현재 시간은 10시 30분입니다."

def test_postprocess_conversational_answer_strips_three_sentence_rule_leak():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "단, 사용자의 질문에 대한 명확한 답변이 필요할 경우 3문장까지 가능합니다.",
        query="오늘 피곤한데 몇 시에 잘까?",
        response_language="ko",
    )
    assert cleaned == ""

def test_postprocess_conversational_answer_strips_insufficient_answer_rule_leak():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "단, 사용자의 질문에 대한 답변이 부족할 경우 추가적인 질문을 덧붙일 수 있습니다. Okay,.",
        query="라마 아니었어?",
        response_language="ko",
    )
    assert cleaned == ""

def test_looks_conversational_answer_rejects_insufficient_answer_rule_leak():
    assert not LocalInferenceEngine._looks_conversational_answer(
        "단, 사용자의 질문에 대한 답변이 부족할 경우 추가적인 질문을 덧붙일 수 있습니다. Okay,.",
        response_language="ko",
        query="라마 아니었어?",
    )

def test_postprocess_conversational_answer_strips_help_user_directly_leak():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "사용자에게 직접 도움을 주세요. 안녕하세요!",
        query="안녕",
        response_language="ko",
    )
    assert cleaned == "안녕하세요!"

def test_looks_conversational_answer_rejects_help_user_directly_leak():
    assert not LocalInferenceEngine._looks_conversational_answer(
        "사용자에게 직접 도움을 주세요.",
        response_language="ko",
        query="나 햄버거 먹고싶은데",
    )

def test_looks_conversational_answer_accepts_brief_ack_reply():
    assert LocalInferenceEngine._looks_conversational_answer(
        "맞아요.",
        response_language="ko",
        query="그렇구나!",
    )

def test_looks_conversational_answer_accepts_short_math_reply():
    assert LocalInferenceEngine._looks_conversational_answer(
        "2입니다.",
        response_language="ko",
        query="1 더하기 1은?",
    )

def test_looks_conversational_answer_rejects_react_immediately_rule_leak():
    assert not LocalInferenceEngine._looks_conversational_answer(
        "사용자의 말에 바로 반응하세요.",
        response_language="ko",
        query="그렇구나!",
    )

def test_postprocess_conversational_answer_strips_user_message_rule_leak():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        "사용자 메시지에 바로 반응하세요. 사용자 메시지에 명확한 답을 하세요.",
        query="매일 늦게자서 고민이네",
        response_language="ko",
    )
    assert cleaned == ""

def test_looks_conversational_answer_rejects_user_message_rule_leak():
    assert not LocalInferenceEngine._looks_conversational_answer(
        "사용자 메시지에 바로 반응하세요. 사용자 메시지에 명확한 답을 하세요.",
        response_language="ko",
        query="매일 늦게자서 고민이네",
    )

def test_korean_quality_issues_detects_query_echo():
    issues = LocalInferenceEngine._korean_quality_issues(
        query="오늘 점심에 뭐 먹을지 메뉴 추천해줘",
        answer="오늘 점심에 뭐 먹을지 메뉴 추천해줘",
        response_language="ko",
    )
    assert "query_echo" in issues

def test_korean_quality_issues_detects_informal_tone():
    issues = LocalInferenceEngine._korean_quality_issues(
        query="몇 시쯤 자는게 좋을까",
        answer="늦게 자는 건 건강에 좋지 않아. 자는 시간을 맞춰봐.",
        response_language="ko",
    )
    assert "informal_tone" in issues

def test_korean_quality_issues_allows_polite_tone():
    issues = LocalInferenceEngine._korean_quality_issues(
        query="몇 시쯤 자는게 좋을까",
        answer="늦게 자는 건 건강에 좋지 않아요. 자는 시간을 조금 앞당겨 보세요.",
        response_language="ko",
    )
    assert "informal_tone" not in issues

def test_generate_conversational_no_longer_retries_on_leak():
    # In Turbo mode, leaks should be handled by post-processing or prevented by the prompt.
    # We no longer do a full second inference pass for leaks.
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: [
                "알겠어요. 편하게 이어서 얘기해요.",
            ],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    result = engine.generate_conversational(
        query="여긴 많은편이야",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.used_fallback is False
    assert "편하게" in result.answer

def test_sanitize_generated_answer_removes_file_marker_prefix():
    cleaned = LocalInferenceEngine._sanitize_generated_answer(
        raw="(데통10주1차.txt) 오류율 계산 핵심은 신호 대 잡음비 해석입니다.",
        prompt="",
    )
    assert "데통10주1차.txt" not in cleaned
    assert "오류율 계산" in cleaned

def test_sanitize_generated_answer_preserves_python_minus_operator_in_code():
    cleaned = LocalInferenceEngine._sanitize_generated_answer(
        raw=(
            "```python\n"
            "def two_sum(nums, target):\n"
            "    for i, num in enumerate(nums):\n"
            "        complement = target - num\n"
            "        if complement in {}:\n"
            "            return [0, i]\n"
            "```"
        ),
        prompt="",
    )
    assert "target - num" in cleaned
    assert "```python" in cleaned

def test_sanitize_generated_answer_autocloses_unbalanced_code_fence():
    cleaned = LocalInferenceEngine._sanitize_generated_answer(
        raw="```python\ndef add(a, b):\n    return a + b",
        prompt="",
    )
    assert cleaned.count("```") % 2 == 0
    assert cleaned.endswith("```")

def test_postprocess_conversational_answer_preserves_fenced_code_block():
    cleaned = LocalInferenceEngine._postprocess_conversational_answer(
        (
            "좋아요. 아래 명령을 실행하세요.\n\n"
            "```bash\n"
            "python app.py --help\n"
            "echo \"ok\"\n"
            "```"
        ),
        query="실행 명령 알려줘",
        response_language="ko",
    )
    assert "```bash" in cleaned
    assert "python app.py --help" in cleaned
    assert "echo \"ok\"" in cleaned

def test_strip_reasoning_leak_removes_qwen_think_block():
    raw = "<think>\nstep1\nstep2\n</think>\n\n최종 답변은 여기입니다."
    cleaned = LocalInferenceEngine._strip_reasoning_leak(raw)
    assert "<think>" not in cleaned.lower()
    assert "최종 답변은 여기입니다." in cleaned

def test_strip_reasoning_leak_removes_unclosed_think_tail_and_keeps_prefix():
    raw = (
        "만나서 반갑습니다. 궁금한 점을 말씀해 주세요. "
        "<think> Thinking Process: 1. Analyze the Request: user asked greeting..."
    )
    cleaned = LocalInferenceEngine._strip_reasoning_leak(raw)
    assert "<think>" not in cleaned.lower()
    assert "Thinking Process" not in cleaned
    assert "만나서 반갑습니다." in cleaned

def test_strip_reasoning_leak_removes_constraint_meta_block():
    engine = LocalInferenceEngine()
    raw = (
        "<think> Thinking Process: 1. Analyze the Request:\n"
        "* Constraint 1: Start with the answer directly\n"
        "* Constraint 2: Do not output system logs.\n"
        "* Constraint 3: Do not role-play both user and assistant.\n"
        "실제 답변은 이 아래에 있지 않습니다."
    )
    cleaned = engine._strip_reasoning_leak(raw)
    assert "<think>" not in cleaned.lower()
    assert "Thinking Process" not in cleaned
    assert "Constraint 1" not in cleaned
    assert "Do not output system logs" not in cleaned

def test_strip_reasoning_leak_preserves_real_answer_after_constraint_block():
    engine = LocalInferenceEngine()
    raw = (
        "<think> Thinking Process: 1. Analyze the Request:\n"
        "* Constraint 1: Start with the answer directly\n"
        "* Constraint 2: Do not output system logs.\n"
        "* Constraint 3: Do not role-play both user and assistant.\n"
        "</think>\n"
        "제 성능은 모델 크기, 컨텍스트 길이, 장치 상태에 따라 달라집니다."
    )
    cleaned = engine._strip_reasoning_leak(raw)
    assert "Thinking Process" not in cleaned
    assert "Constraint 1" not in cleaned
    assert "제 성능은 모델 크기" in cleaned

def test_postprocess_conversational_answer_keeps_tail_after_inline_constraint_marker():
    engine = LocalInferenceEngine()
    cleaned = engine._postprocess_conversational_answer(
        "Constraint 1: Start with the answer directly. 제 성능은 현재 로컬 설정 기준으로 중간 수준입니다.",
        query="Explain your performance.",
        response_language="ko",
    )
    assert "Constraint 1" not in cleaned
    assert "제 성능은 현재 로컬 설정 기준으로 중간 수준입니다." in cleaned
