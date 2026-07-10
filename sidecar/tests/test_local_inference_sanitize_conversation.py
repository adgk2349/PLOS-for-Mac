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

def test_generate_conversational_auto_switches_primary_engine_for_gguf_path(tmp_path: Path):
    gguf = tmp_path / "primary.gguf"
    gguf.write_bytes(b"GGUF")
    engine = _StubInferenceEngine(
        outputs={
            LocalEngine.MLX: None,
            LocalEngine.LLAMA_CPP: "안녕하세요!",
        }
    )
    result = engine.generate_conversational(
        query="안녕",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        mlx_model_path=str(gguf),
        llama_model_path=None,
        language_preference="ko",
        allow_static_fallback=False,
    )
    detail = result.detail or ""
    assert result.used_fallback is False
    assert result.engine_used == LocalEngine.LLAMA_CPP
    assert "conversational_primary=llama_cpp" in detail
    assert "auto_switch:mlx_path_is_gguf" in detail

def test_generate_conversational_single_pass_turbo_mode():
    # In Turbo mode, we only do one pass. Even if the result looks slightly off,
    # we rely on post-processing rather than expensive retries.
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: ["안녕하세요!"],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    result = engine.generate_conversational(
        query="안녕",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.used_fallback is False
    assert result.engine_used == LocalEngine.MLX
    assert "안녕하세요" in result.answer

def test_generate_conversational_no_longer_rewrites_quality_issues():
    # Echoing the query is a quality issue, but in Turbo mode we don't rewrite.
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: [
                "오늘 뭐 먹지?",
            ],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    result = engine.generate_conversational(
        query="오늘 뭐 먹지?",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.used_fallback is False
    assert "오늘 뭐 먹지" in result.answer

def test_generate_conversational_repairs_hard_issue_even_when_static_fallback_disabled():
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: [
                "사용자 메시지에 바로 반응하세요.",
                "제 성능은 현재 로컬 설정과 모델 크기에 따라 달라집니다.",
            ],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    result = engine.generate_conversational(
        query="성능 설명해줘",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.used_fallback is False
    assert "제 성능은 현재 로컬 설정과 모델 크기에 따라 달라집니다." in result.answer

def test_generate_conversational_repairs_meta_only_promise_answer():
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: [
                "한 문장으로만 요약해 드릴게요.",
                "핵심은 오늘 가장 중요한 일 하나를 먼저 끝내는 것입니다.",
            ],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    result = engine.generate_conversational(
        query="방금 내용 한 줄로 다시 말해줘",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.used_fallback is False
    assert result.answer == "핵심은 오늘 가장 중요한 일 하나를 먼저 끝내는 것입니다."

def test_generate_conversational_continues_truncated_answer_once():
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: [
                "프라이팬으로 굽는 핵심은 **1.",
                "고기는 굽기 20분 전에 꺼내고 팬을 충분히 달군 뒤 짧고 강하게 굽는 것입니다.",
            ],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    result = engine.generate_conversational(
        query="소고기 굽는 법 핵심만 알려줘",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.used_fallback is False
    assert "충분히 달군 뒤" in result.answer

def test_generate_conversational_allows_brief_ack_answer():
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: ["맞아요."],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    result = engine.generate_conversational(
        query="그렇구나!",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.used_fallback is False
    assert result.answer == "맞아요."

def test_generate_conversational_no_longer_uses_safe_direct_fallback_for_specific_model_query():
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: [
                "최근 세션 컨텍스트: 사용자: 저녁 뭐 먹지? 사용자: 안녕.",
                None,
                None,
            ],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    result = engine.generate_conversational(
        query="지금 모델 성능은 어느 정도야?",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.used_fallback is True
    assert result.answer == ""
    detail = result.detail or ""
    assert "conversational engine failed" in detail

def test_generate_conversational_no_longer_uses_clarification_fallback_for_ambiguous_short_query():
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: [
                "최근 세션 컨텍스트: 사용자: 저녁 뭐 먹지?",
                None,
                None,
            ],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    result = engine.generate_conversational(
        query="그거 말고",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.used_fallback is True
    assert result.answer == ""

def test_generate_grounded_is_single_pass_turbo(tmp_path: Path):
    mlx_dir = tmp_path / "mlx-model"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    (mlx_dir / "config.json").write_text("{}", encoding="utf-8")
    engine = _SequentialStubInferenceEngine(
        outputs_by_engine={
            LocalEngine.MLX: [
                "상위 후보는 프로그램9.2, 프로그램9.3, 프로그램11.1입니다.",
            ],
            LocalEngine.LLAMA_CPP: [None],
        }
    )
    engine._ensure_runtime_module = lambda **_kwargs: True  # type: ignore[method-assign]
    result = engine.generate(
        query="상위 3개 후보만 보여줘",
        mode=WorkMode.GENERAL,
        citations=[],
        profile="recommended",
        engine=LocalEngine.MLX,
        mlx_model_path=str(mlx_dir),
        language_preference="ko",
    )
    assert isinstance(result.used_fallback, bool)
    assert isinstance(result.answer, str)

def test_generate_conversational_failure_detail_contains_engine_errors():
    engine = _StubInferenceEngine(
        outputs={
            LocalEngine.MLX: None,
            LocalEngine.LLAMA_CPP: None,
        }
    )
    result = engine.generate_conversational(
        query="안녕",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=False,
    )
    assert result.answer == ""
    assert result.used_fallback is True
    detail = result.detail or ""
    assert "primary_error=mlx" in detail
    assert "mlx stub failure" in detail

def test_generate_conversational_brief_query_does_not_emit_static_fallback_text():
    engine = _StubInferenceEngine(
        outputs={
            LocalEngine.MLX: None,
            LocalEngine.LLAMA_CPP: None,
        }
    )
    result = engine.generate_conversational(
        query="그렇구나!",
        mode=WorkMode.GENERAL,
        profile="recommended",
        engine=LocalEngine.MLX,
        language_preference="ko",
        allow_static_fallback=True,
    )
    assert result.used_fallback is True
    assert result.answer == ""
    assert "static_fallback_suppressed=1" in (result.detail or "")

def test_conversation_sampling_preset_uses_stronger_repeat_penalty():
    llama_sampling = LocalInferenceEngine._sampling_preset(
        style="conversation",
        engine=LocalEngine.LLAMA_CPP,
    )
    mlx_sampling = LocalInferenceEngine._sampling_preset(
        style="conversation",
        engine=LocalEngine.MLX,
    )
    # Ollama-benchmarked: temperature 0.78~0.80, repeat_penalty 1.10, top_k 40, min_p 0.05
    assert llama_sampling["repeat_penalty"] == 1.10
    assert mlx_sampling["repeat_penalty"] == 1.10
    assert llama_sampling["temperature"] >= 0.75
    assert mlx_sampling["temperature"] >= 0.75
    assert llama_sampling.get("top_k", 0) == 40
    assert mlx_sampling.get("top_k", 0) == 40
    assert llama_sampling.get("min_p", 0.0) == 0.05
    assert mlx_sampling.get("min_p", 0.0) == 0.05

def test_split_conversation_prompt_for_chat_extracts_system_and_user():
    prompt = (
        "You are a conversational local AI assistant.\n"
        "Mode: GENERAL\n"
        "<conversation_memory>\nrecent topic: swift\n</conversation_memory>\n"
        "Input message: 스위프트 뭐부터 공부할까?\n"
        "Answer:"
    )
    system_text, user_text = LocalInferenceEngine._split_conversation_prompt_for_chat(prompt)
    assert "You are a conversational local AI assistant." in system_text
    assert "Mode: GENERAL" in system_text
    assert user_text == "스위프트 뭐부터 공부할까?"

def test_split_conversation_prompt_for_chat_extracts_system_and_user_instance():
    engine = LocalInferenceEngine()
    prompt = (
        "You are a conversational local AI assistant.\n"
        "Mode: GENERAL\n"
        "Input message: 오늘 일정 정리해줘\n"
        "Answer:"
    )
    system_text, user_text = engine._split_conversation_prompt_for_chat(prompt)
    assert "You are a conversational local AI assistant." in system_text
    assert "Mode: GENERAL" in system_text
    assert user_text == "오늘 일정 정리해줘"

def test_pick_best_conversation_answer_rejects_invalid_primary_meta_response():
    engine = LocalInferenceEngine()
    selected, issues = engine._pick_best_conversation_answer(
        primary_answer="<think> Thinking Process: Analyze the Request",
        primary_issues=["meta_leak"],
        primary_valid=False,
        repaired_answer=None,
        repaired_issues=[],
        repaired_valid=False,
        query="성능 설명해줘",
        is_recommendation_query=False,
    )
    assert selected is None
    assert issues == []

def test_conversation_quality_issues_flags_meta_only_promise_answer():
    engine = LocalInferenceEngine()
    issues = engine._conversation_quality_issues(
        query="왜 그렇게 골랐는지 한 문장으로만 말해줘",
        answer="설명해 드릴게요.",
        response_language="ko",
    )
    assert "meta_only_ack" in issues

def test_conversation_quality_issues_flags_truncated_answer():
    engine = LocalInferenceEngine()
    issues = engine._conversation_quality_issues(
        query="소고기 굽는 법 핵심만 알려줘",
        answer="프라이팬으로 굽는 핵심은 **1.",
        response_language="ko",
    )
    assert "truncated_answer" in issues

def test_conversation_quality_issues_flags_intent_restatement_answer():
    engine = LocalInferenceEngine()
    issues = engine._conversation_quality_issues(
        query="곁들일 거 하나만 추천해줘",
        answer="소고기에 곁들일 메뉴를 추천받고 싶으신 것 같습니다.",
        response_language="ko",
    )
    assert "intent_restatement" in issues
