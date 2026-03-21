from __future__ import annotations

from datetime import datetime, timezone

from local_ai_core.executor import LocalExecutor
from local_ai_core.local_inference import InferenceResult
from local_ai_core.models import (
    Citation,
    LocalEngine,
    LocalPlan,
    ParsedIntent,
    ParsedEntities,
    ParsedTimeFilters,
    ParsedWorkspaceFilters,
    ReasoningIntent,
    StartupProfile,
    SuggestedActionKind,
    WorkMode,
)


class _StubInference:
    def __init__(self):
        self.calls: list[str] = []

    def generate(self, **kwargs):
        query = str(kwargs.get("query") or "")
        self.calls.append(query)
        if "부분 요약" in query or "partial summaries" in query:
            return InferenceResult(
                answer="전체 핵심은 데이터 통신 절차와 오류 처리 흐름입니다.",
                engine_used=LocalEngine.MLX,
                used_fallback=False,
                detail="reduce_ok",
            )
        return InferenceResult(
            answer="파일 핵심 요약입니다.",
            engine_used=LocalEngine.MLX,
            used_fallback=False,
            detail="map_ok",
        )

    def generate_conversational(self, **kwargs):
        return InferenceResult(
            answer="안녕하세요.",
            engine_used=LocalEngine.MLX,
            used_fallback=False,
            detail=None,
        )


def _citation(doc_id: str, chunk_id: str, file_name: str, score: float) -> Citation:
    return Citation(
        doc_id=doc_id,
        chunk_id=chunk_id,
        file_path=f"/tmp/{file_name}",
        snippet=f"{file_name} snippet",
        score=score,
        modified_at=datetime.now(timezone.utc),
    )


def test_executor_runs_map_reduce_for_multi_file_summary_strategy():
    executor = LocalExecutor(local_inference=_StubInference())
    citations = [
        _citation("doc-a", "doc-a:0", "a.txt", 0.81),
        _citation("doc-b", "doc-b:0", "b.txt", 0.73),
        _citation("doc-c", "doc-c:0", "c.txt", 0.66),
    ]
    parsed = ParsedIntent(
        intent=ReasoningIntent.SUMMARIZE_FILE,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.8,
    )
    plan = LocalPlan(
        plan_type="summary",
        selected_files=["doc-a", "doc-b", "doc-c"],
        selected_chunks=["doc-a:0", "doc-b:0", "doc-c:0"],
        response_strategy="map_reduce_grounded_summary",
        allowed_actions=[SuggestedActionKind.OPEN_FILE],
        external_reasoning_needed=False,
    )

    result = executor.execute(
        query="파일 여러개 전부 읽어보고 요약해줘",
        mode=WorkMode.SUMMARY,
        parsed_intent=parsed.intent,
        plan=plan,
        citations=citations,
        startup_profile=StartupProfile.RECOMMENDED,
        engine=LocalEngine.MLX,
        mlx_model_path=None,
        llama_model_path=None,
        language_preference="ko",
        response_length="medium",
    )

    assert result.result_type == "summary"
    assert result.structured_payload.get("aggregation") == "map_reduce"
    assert result.structured_payload.get("files_considered") == 3
    assert "summary:map_reduce" in result.tool_logs
    assert "데이터 통신 절차" in result.generated_text


def test_executor_runs_focused_file_summary_strategy():
    executor = LocalExecutor(local_inference=_StubInference())
    citations = [
        _citation("doc-a", "doc-a:0", "a.txt", 0.91),
        _citation("doc-a", "doc-a:1", "a.txt", 0.88),
        _citation("doc-a", "doc-a:2", "a.txt", 0.84),
    ]
    parsed = ParsedIntent(
        intent=ReasoningIntent.SUMMARIZE_FILE,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.8,
    )
    plan = LocalPlan(
        plan_type="summary",
        selected_files=["doc-a"],
        selected_chunks=["doc-a:0", "doc-a:1", "doc-a:2"],
        response_strategy="focused_file_grounded_summary",
        allowed_actions=[SuggestedActionKind.OPEN_FILE],
        external_reasoning_needed=False,
    )

    result = executor.execute(
        query="a.txt 핵심 5줄 요약해줘",
        mode=WorkMode.SUMMARY,
        parsed_intent=parsed.intent,
        plan=plan,
        citations=citations,
        startup_profile=StartupProfile.RECOMMENDED,
        engine=LocalEngine.MLX,
        mlx_model_path=None,
        llama_model_path=None,
        language_preference="ko",
        response_length="medium",
    )

    assert result.result_type == "summary"
    assert result.structured_payload.get("aggregation") == "focused_file"
    assert result.structured_payload.get("target_doc_id") == "doc-a"
    assert "summary:focused_file" in result.tool_logs
