from datetime import datetime, timezone
import re

from local_ai_core.models import (
    Citation,
    ExecutionResult,
    LocalPlan,
    ParsedEntities,
    ParsedIntent,
    ParsedTimeFilters,
    ParsedWorkspaceFilters,
    ReasoningIntent,
    StartupProfile,
    SuggestedActionKind,
    VerificationResult,
    WorkMode,
)
from local_ai_core.response_composer import ResponseComposer


def _sample_citation(path: str, score: float = 0.8) -> Citation:
    return Citation(
        doc_id="doc-1",
        chunk_id=f"chunk-{path}",
        file_path=path,
        snippet="sample snippet",
        score=score,
        modified_at=datetime.now(tz=timezone.utc),
    )


def test_intent_file_search_classification():
    composer = ResponseComposer()
    intent = composer.classify_intent(
        query="2025 의사결정나무 파일 찾아줘",
        mode=WorkMode.GENERAL,
        citations=[],
    )
    assert intent.value == "FILE_SEARCH"


def test_compose_returns_three_layer_with_actions():
    composer = ResponseComposer()
    citations = [
        _sample_citation("/tmp/a.md", 0.9),
        _sample_citation("/tmp/b.md", 0.7),
    ]
    intent, lead, summary, actions = composer.compose(
        query="두 파일 비교해줘",
        mode=WorkMode.RESEARCH,
        response_language="ko",
        citations=citations,
        result_summary="비교 결과입니다.",
    )
    assert intent.value in {"TASK_REQUEST", "DOCUMENT_QA"}
    assert lead
    assert "비교 결과" in summary
    kinds = {action.kind for action in actions}
    assert SuggestedActionKind.OPEN_FILE in kinds
    assert SuggestedActionKind.SUMMARIZE_TOP in kinds
    assert SuggestedActionKind.COMPARE_TOP in kinds
    assert SuggestedActionKind.ASK_FOLLOWUP in kinds


def test_compose_insufficient_contains_followup():
    composer = ResponseComposer()
    _, _, summary, actions = composer.compose(
        query="이거 뭐지",
        mode=WorkMode.STRICT_SEARCH,
        response_language="ko",
        citations=[],
        result_summary="",
        insufficient=True,
    )
    assert "근거 부족" in summary
    assert any(action.kind == SuggestedActionKind.ASK_FOLLOWUP for action in actions)


def test_compose_v2_candidate_mode_hides_noisy_generated_text():
    composer = ResponseComposer()
    citation = _sample_citation("/tmp/data_structures.md", 0.22)
    response = composer.compose_v2(
        query="자료구조에서 중요한게 뭐였지",
        mode=WorkMode.GENERAL,
        response_language="ko",
        parsed_intent=ParsedIntent(
            intent=ReasoningIntent.EXPLAIN_CONTENT,
            entities=ParsedEntities(),
            time_filters=ParsedTimeFilters(),
            workspace_filters=ParsedWorkspaceFilters(),
            confidence=0.45,
        ),
        plan=LocalPlan(
            plan_type="explanation",
            selected_files=[citation.doc_id],
            selected_chunks=[citation.chunk_id],
            response_strategy="direct_grounded_explanation",
            allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
            external_reasoning_needed=False,
        ),
        execution_result=ExecutionResult(
            result_type="answer",
            structured_payload={"text": "console.log(...) console.log(...)"},
            citations=[citation],
            tool_logs=[],
            generated_text="console.log(...) console.log(...) console.log(...)",
        ),
        verification=VerificationResult(
            is_valid=False,
            confidence=0.25,
            issues=["low_relevance"],
            ambiguity_level=0.75,
            candidate_mode=True,
        ),
        behavior_policy=None,
        response_length="medium",
        show_citations=True,
        prefer_action_suggestions=True,
        used_profile=StartupProfile.RECOMMENDED,
        engine_used=None,
        used_fallback=False,
        runtime_detail=None,
    )
    assert "console.log" not in response.structured_result.summary
    assert "data_structures.md" in response.structured_result.summary


def test_compose_v2_candidate_mode_adds_clarifying_questions_for_find_file():
    composer = ResponseComposer()
    response = composer.compose_v2(
        query="자료구조 폴더에 지금 뭐있어",
        mode=WorkMode.GENERAL,
        response_language="ko",
        parsed_intent=ParsedIntent(
            intent=ReasoningIntent.FIND_FILE,
            entities=ParsedEntities(topics=["자료구조"], file_names=[], tags=[], projects=[]),
            time_filters=ParsedTimeFilters(),
            workspace_filters=ParsedWorkspaceFilters(),
            confidence=0.52,
        ),
        plan=LocalPlan(
            plan_type="file_lookup",
            selected_files=[],
            selected_chunks=[],
            response_strategy="list_then_offer_actions",
            allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
            external_reasoning_needed=False,
        ),
        execution_result=ExecutionResult(
            result_type="candidate",
            structured_payload={"reason": "low_relevance_precheck", "items": []},
            citations=[],
            tool_logs=[],
            generated_text="근거 부족",
        ),
        verification=VerificationResult(
            is_valid=False,
            confidence=0.3,
            issues=["low_relevance"],
            ambiguity_level=0.7,
            candidate_mode=True,
        ),
        behavior_policy=None,
        response_length="medium",
        show_citations=True,
        prefer_action_suggestions=True,
        used_profile=StartupProfile.RECOMMENDED,
        engine_used=None,
        used_fallback=False,
        runtime_detail=None,
    )
    assert "확인 질문" in response.structured_result.summary
    assert any(action.label == "질문 좁히기" for action in response.actions)


def test_compose_v2_general_chat_recommendation_normalizes_three_options_and_metadata():
    composer = ResponseComposer()
    response = composer.compose_v2(
        query="오늘 저녁 메뉴 추천해줘",
        mode=WorkMode.GENERAL,
        response_language="ko",
        parsed_intent=ParsedIntent(
            intent=ReasoningIntent.GENERAL_CHAT,
            entities=ParsedEntities(),
            time_filters=ParsedTimeFilters(),
            workspace_filters=ParsedWorkspaceFilters(),
            confidence=0.88,
        ),
        plan=LocalPlan(
            plan_type="conversation",
            selected_files=[],
            selected_chunks=[],
            response_strategy="conversational_assistant",
            allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
            external_reasoning_needed=False,
        ),
        execution_result=ExecutionResult(
            result_type="conversation",
            structured_payload={
                "style": "general_chat",
                "ungrounded_allowed": True,
            },
            citations=[],
            tool_logs=[],
            generated_text="김치찌개는 든든해요. 된장찌개는 담백해요. 순두부찌개는 가볍게 먹기 좋아요.",
        ),
        verification=VerificationResult(
            is_valid=True,
            confidence=0.9,
            issues=[],
            ambiguity_level=0.08,
            candidate_mode=False,
        ),
        behavior_policy=None,
        response_length="medium",
        show_citations=True,
        prefer_action_suggestions=True,
        used_profile=StartupProfile.RECOMMENDED,
        engine_used=None,
        used_fallback=False,
        runtime_detail=(
            "conversational_primary=mlx; direct_first_applied=1; "
            "question_count_after_postprocess=0; recommendation_shape=three_options"
        ),
    )
    summary = response.structured_result.summary
    assert summary.startswith("1. ")
    assert "\n2. " in summary
    assert "\n3. " in summary
    assert response.metadata["direct_first_applied"] is True
    assert response.metadata["question_count_after_postprocess"] == 0
    assert response.metadata["recommendation_shape"] == "three_options"
    assert all(action.kind != SuggestedActionKind.ASK_FOLLOWUP for action in response.actions)


def test_compose_v2_conversational_direct_does_not_append_clarification_block():
    composer = ResponseComposer()
    response = composer.compose_v2(
        query="핵심만 말해줘",
        mode=WorkMode.GENERAL,
        response_language="ko",
        parsed_intent=ParsedIntent(
            intent=ReasoningIntent.EXPLAIN_CONTENT,
            entities=ParsedEntities(),
            time_filters=ParsedTimeFilters(),
            workspace_filters=ParsedWorkspaceFilters(),
            confidence=0.75,
        ),
        plan=LocalPlan(
            plan_type="explanation",
            selected_files=[],
            selected_chunks=[],
            response_strategy="direct_grounded_explanation",
            allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
            external_reasoning_needed=False,
        ),
        execution_result=ExecutionResult(
            result_type="answer",
            structured_payload={"text": ""},
            citations=[],
            tool_logs=[],
            generated_text="핵심만 먼저 정리하면 우선순위를 빠르게 잡을 수 있어요.",
        ),
        verification=VerificationResult(
            is_valid=True,
            confidence=0.55,
            issues=[],
            ambiguity_level=0.82,
            candidate_mode=False,
        ),
        behavior_policy=None,
        response_length="medium",
        show_citations=True,
        prefer_action_suggestions=True,
        used_profile=StartupProfile.RECOMMENDED,
        engine_used=None,
        used_fallback=False,
        runtime_detail=None,
    )
    assert "확인 질문" not in response.structured_result.summary


def test_normalize_korean_summary_conversation_trims_ramble():
    raw = (
        "안녕하세요. 도움이 필요하신가요? 오늘은 어떤 일이 있으신가요? "
        "뉴욕에서 온 한국어 학생입니다 환영합니다! 한국어 배우는 건 어때요? 잘 배우고 있어요."
    )
    normalized = ResponseComposer._normalize_korean_summary(raw, conversation_mode=True, query="안녕")
    assert normalized
    assert len(normalized) <= 140
    assert normalized.endswith((".", "!", "?"))


def test_dedupe_sentence_lines_conversation_mode_is_less_aggressive():
    raw = "집에서 먹는 건 편해요. 집에서 먹는 건 꽤 편한 편이에요. 배달도 괜찮아요."
    deduped = ResponseComposer._dedupe_sentence_lines(raw, aggressive=False)
    assert "배달도 괜찮아요." in deduped
    assert "꽤 편한 편" in deduped


def test_compose_v2_summary_formats_numbered_points():
    composer = ResponseComposer()
    citation = _sample_citation("/tmp/데통10주1차.txt", 0.81)
    response = composer.compose_v2(
        query="데통10주1차.txt 핵심을 5줄로 요약해줘",
        mode=WorkMode.SUMMARY,
        response_language="ko",
        parsed_intent=ParsedIntent(
            intent=ReasoningIntent.SUMMARIZE_FILE,
            entities=ParsedEntities(file_names=["데통10주1차.txt"], tags=[], topics=[], projects=[]),
            time_filters=ParsedTimeFilters(),
            workspace_filters=ParsedWorkspaceFilters(),
            confidence=0.88,
        ),
        plan=LocalPlan(
            plan_type="summary",
            selected_files=[citation.doc_id],
            selected_chunks=[citation.chunk_id],
            response_strategy="focused_file_grounded_summary",
            allowed_actions=[SuggestedActionKind.OPEN_FILE],
            external_reasoning_needed=False,
        ),
        execution_result=ExecutionResult(
            result_type="summary",
            structured_payload={"text": ""},
            citations=[citation],
            tool_logs=[],
            generated_text=(
                "08:20 통신 측에는 프레임 데이터를 임시 저장할 버퍼가 필요합니다. "
                "08:30 ACK가 안 오면 재전송해야 합니다. "
                "오류율 계산은 신호 품질 해석과 함께 봐야 합니다. "
                "프로토콜 단계별 실패 조건을 점검해야 합니다. "
                "실험 조건에 따라 결과 편차가 생깁니다."
            ),
        ),
        verification=VerificationResult(
            is_valid=True,
            confidence=0.82,
            issues=[],
            ambiguity_level=0.12,
            candidate_mode=False,
        ),
        behavior_policy=None,
        response_length="medium",
        show_citations=True,
        prefer_action_suggestions=True,
        used_profile=StartupProfile.RECOMMENDED,
        engine_used=None,
        used_fallback=False,
        runtime_detail=None,
    )
    summary = response.structured_result.summary
    assert summary.startswith("1. ")
    assert "\n2. " in summary
    assert "\n3. " in summary
    assert "\n4. " in summary
    assert "\n5. " in summary


def test_compose_v2_summary_formats_numbered_points_even_in_candidate_mode():
    composer = ResponseComposer()
    citation = _sample_citation("/tmp/데통10주1차.txt", 0.66)
    response = composer.compose_v2(
        query="데통10주1차.txt 핵심을 5줄로 요약해줘",
        mode=WorkMode.SUMMARY,
        response_language="ko",
        parsed_intent=ParsedIntent(
            intent=ReasoningIntent.SUMMARIZE_FILE,
            entities=ParsedEntities(file_names=["데통10주1차.txt"], tags=[], topics=[], projects=[]),
            time_filters=ParsedTimeFilters(),
            workspace_filters=ParsedWorkspaceFilters(),
            confidence=0.84,
        ),
        plan=LocalPlan(
            plan_type="summary",
            selected_files=[citation.doc_id],
            selected_chunks=[citation.chunk_id],
            response_strategy="focused_file_grounded_summary",
            allowed_actions=[SuggestedActionKind.OPEN_FILE],
            external_reasoning_needed=False,
        ),
        execution_result=ExecutionResult(
            result_type="summary",
            structured_payload={"text": ""},
            citations=[citation],
            tool_logs=[],
            generated_text=(
                "08:20 통신 측에는 프레임 임시 저장용 버퍼가 필요합니다. "
                "08:30 ACK가 없으면 송신 측은 재전송 절차로 돌아갑니다. "
                "08:40 수신 준비가 안 되면 흐름 제어로 속도를 조절합니다. "
                "08:50 오류율 해석은 신호 품질과 함께 판단해야 합니다. "
                "09:10 단계별 실패 조건을 미리 점검해야 재시도 비용이 줄어듭니다."
            ),
        ),
        verification=VerificationResult(
            is_valid=True,
            confidence=0.67,
            issues=[],
            ambiguity_level=0.2,
            candidate_mode=True,
        ),
        behavior_policy=None,
        response_length="medium",
        show_citations=True,
        prefer_action_suggestions=True,
        used_profile=StartupProfile.RECOMMENDED,
        engine_used=None,
        used_fallback=False,
        runtime_detail=None,
    )
    summary = response.structured_result.summary
    assert summary.startswith("1. ")
    assert "\n2. " in summary
    assert "\n3. " in summary
    assert "\n4. " in summary
    assert "\n5. " in summary


def test_extract_summary_point_candidates_removes_inline_timecodes():
    points = ResponseComposer._extract_summary_point_candidates(
        "08:20 수신 측 버퍼가 필요합니다. 08:30 수신 측 버퍼가 필요합니다. 09:10 ACK 누락 시 재전송합니다.",
        response_language="ko",
    )
    assert points
    assert all(re.search(r"\b\d{1,2}[:：]\d{2}\b", line) is None for line in points)
    assert sum(1 for line in points if "수신 측 버퍼가 필요합니다" in line) <= 1


def test_file_list_summary_lists_all_for_small_result_set():
    summary = ResponseComposer._file_list_summary(
        query="데통 파일 찾아줘",
        payload={
            "items": [
                {"file_path": "/tmp/데통1주차.txt"},
                {"file_path": "/tmp/데통2주차.txt"},
                {"file_path": "/tmp/데통3주차.txt"},
            ]
        },
        response_language="ko",
        candidate_mode=False,
        fallback="fallback",
    )
    assert "총 3개" in summary
    assert "1. 데통1주차.txt" in summary
    assert "2. 데통2주차.txt" in summary
    assert "3. 데통3주차.txt" in summary


def test_file_list_summary_lists_extended_when_query_requests_all():
    items = [{"file_path": f"/tmp/데통{idx}주차.txt"} for idx in range(1, 16)]
    summary = ResponseComposer._file_list_summary(
        query="데통 파일 전부 찾아줘",
        payload={"items": items},
        response_language="ko",
        candidate_mode=False,
        fallback="fallback",
    )
    assert "총 15개" in summary
    assert "1. 데통1주차.txt" in summary
    assert "15. 데통15주차.txt" in summary


def test_file_list_summary_respects_scope_all_even_without_all_token():
    items = [{"file_path": f"/tmp/데통{idx}주차.txt"} for idx in range(1, 36)]
    summary = ResponseComposer._file_list_summary(
        query="파일 목록 보여줘",
        payload={"items": items},
        response_language="ko",
        candidate_mode=False,
        fallback="fallback",
        requested_scope="all",
    )
    assert "총 35개" in summary
    assert "30. 데통30주차.txt" in summary
    assert "원하면 '계속 보여줘'" in summary


def test_naturalize_summary_text_removes_repeated_blocks():
    raw = (
        "좋아, 이 맥락 기준으로 바로 정리해볼게. "
        "전송을 시작하기 전에 전송을 시작하기 전에 전송을 시작하기 전에 "
        "전송을 시작하기 전에 버퍼를 확인합니다."
    )
    cleaned = ResponseComposer._naturalize_summary_text(
        summary=raw,
        query="데통10주차 핵심 5줄 요약해줘",
        response_language="ko",
        result_type="summary",
        intent=ReasoningIntent.SUMMARIZE_FILE,
    )
    assert cleaned.count("전송을 시작하기 전에") <= 1
    assert "좋아, 이 맥락 기준으로" not in cleaned


def test_compose_v2_summary_uses_natural_body_without_template_lead():
    composer = ResponseComposer()
    citation = _sample_citation("/tmp/데통10주1차.txt", 0.78)
    response = composer.compose_v2(
        query="데통10주1차.txt 핵심 5줄로 요약해줘",
        mode=WorkMode.SUMMARY,
        response_language="ko",
        parsed_intent=ParsedIntent(
            intent=ReasoningIntent.SUMMARIZE_FILE,
            entities=ParsedEntities(file_names=["데통10주1차.txt"], tags=[], topics=[], projects=[]),
            time_filters=ParsedTimeFilters(),
            workspace_filters=ParsedWorkspaceFilters(),
            confidence=0.9,
        ),
        plan=LocalPlan(
            plan_type="summary",
            selected_files=[citation.doc_id],
            selected_chunks=[citation.chunk_id],
            response_strategy="focused_file_grounded_summary",
            allowed_actions=[SuggestedActionKind.OPEN_FILE],
            external_reasoning_needed=False,
        ),
        execution_result=ExecutionResult(
            result_type="summary",
            structured_payload={"text": ""},
            citations=[citation],
            tool_logs=[],
            generated_text=(
                "08:20 ACK가 누락되면 재전송한다. "
                "수신 측 버퍼 상태를 먼저 점검해야 한다. "
                "오류율 해석은 신호 품질과 함께 본다."
            ),
        ),
        verification=VerificationResult(
            is_valid=True,
            confidence=0.85,
            issues=[],
            ambiguity_level=0.1,
            candidate_mode=False,
        ),
        behavior_policy=None,
        response_length="medium",
        show_citations=True,
        prefer_action_suggestions=True,
        used_profile=StartupProfile.RECOMMENDED,
        engine_used=None,
        used_fallback=False,
        runtime_detail=None,
    )
    assert response.lead == ""
    assert response.structured_result.summary.startswith("1. ")


def test_naturalize_summary_text_strips_user_message_instruction_leak():
    cleaned = ResponseComposer._naturalize_summary_text(
        summary="사용자 메시지에 바로 반응하세요. 사용자 메시지에 명확한 답을 하세요.",
        query="매일 늦게자서 고민이네",
        response_language="ko",
        result_type="conversation",
        intent=ReasoningIntent.EXPLAIN_CONTENT,
    )
    assert cleaned == ""
