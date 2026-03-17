from datetime import datetime, timezone

from local_ai_core.models import Citation, SuggestedActionKind, WorkMode
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

