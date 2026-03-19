from datetime import datetime, timezone

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


def test_normalize_korean_summary_conversation_trims_ramble():
    raw = (
        "안녕하세요. 도움이 필요하신가요? 오늘은 어떤 일이 있으신가요? "
        "뉴욕에서 온 한국어 학생입니다 환영합니다! 한국어 배우는 건 어때요? 잘 배우고 있어요."
    )
    normalized = ResponseComposer._normalize_korean_summary(raw, conversation_mode=True, query="안녕")
    assert normalized
    assert len(normalized) <= 140
    assert normalized.endswith((".", "!", "?"))
