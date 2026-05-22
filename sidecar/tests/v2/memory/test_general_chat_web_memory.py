from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from local_ai_core.models import (
    ExecutionResult,
    LocalChatRequestV2,
    ParsedIntent,
    PrivacyMode,
    ReasoningIntent,
    RelevantMemoryBundle,
    SettingsModel,
    SessionMemoryItem,
    StructuredResult,
    VerificationResult,
    WorkMode,
    WorkspaceIdentity,
    WorkspaceResponse,
    ComposedChatResponseV2,
)
from local_ai_core.nlu.followup_resolver import FollowUpResolution
from local_ai_core.reasoning.context import ReasoningContext
from local_ai_core.reasoning.strategies.general_chat import GeneralChatStrategy
from local_ai_core.web_retrieval import WebRetrievalReport, WebSearchSource


class _FakeExecutor:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.max_tokens_seen: list[int] = []

    def execute_conversation(
        self,
        *,
        query,
        mode,
        startup_profile,
        engine,
        mlx_model_path,
        llama_model_path,
        language_preference,
        session_summary,
        max_tokens,
    ) -> ExecutionResult:
        self.queries.append(str(query))
        self.max_tokens_seen.append(int(max_tokens))
        if "저장된 웹 메모리" in str(query):
            answer = "저장 메모리 기반 요약 답변 [1]"
        else:
            answer = f"일반 로컬 답변: {query}"
        return ExecutionResult(
            result_type="conversation",
            structured_payload={},
            citations=[],
            tool_logs=[],
            generated_text=answer,
            engine_used=None,
            used_fallback=False,
            runtime_detail="ok",
        )


class _FakeComposer:
    def compose_v2(
        self,
        *,
        query,
        mode,
        response_language,
        parsed_intent,
        plan,
        execution_result,
        verification,
        behavior_policy,
        response_length,
        show_citations,
        prefer_action_suggestions,
        used_profile,
        engine_used,
        used_fallback,
        runtime_detail,
        followup_resolution,
        allow_clarification,
        conversation_path,
        is_local,
        prompt_cache_hit,
    ) -> ComposedChatResponseV2:
        return ComposedChatResponseV2(
            lead=str(execution_result.generated_text or "")[:40],
            structured_result=StructuredResult(
                result_type="conversation",
                summary=str(execution_result.generated_text or ""),
                details=[],
                data={},
            ),
            execution_result=execution_result,
            generated_text=str(execution_result.generated_text or ""),
            citations=execution_result.citations if show_citations else [],
            actions=[],
            prompt_cache_hit=prompt_cache_hit,
            metadata={"conversation_path": conversation_path},
            parsed_intent=parsed_intent,
            plan=plan,
            verification=verification or VerificationResult(is_valid=True, confidence=0.8),
            mode=mode,
            used_profile=used_profile,
            is_local=is_local,
            engine_used=engine_used,
            used_fallback=used_fallback,
            runtime_detail=runtime_detail,
        )


class _FakeMemoryService:
    def __init__(self, ranked_entries: list[dict]) -> None:
        self._ranked_entries = ranked_entries

    def get_ranked_web_memory_entries(self, *, session_id: str, query: str, limit: int = 4) -> list[dict]:
        return list(self._ranked_entries)[:limit]


def _build_context(*, query: str, privacy_mode: PrivacyMode = PrivacyMode.HYBRID, hybrid_enabled: bool = True) -> ReasoningContext:
    workspace = WorkspaceResponse(
        included_paths=[],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    workspace_identity = WorkspaceIdentity(workspace_id="ws", included_paths_hash="h", version=1)
    settings = SettingsModel(
        privacy_mode=privacy_mode,
        hybrid_web_search_enabled=hybrid_enabled,
        language="ko",
        model_profile="advanced",
    )
    req = LocalChatRequestV2(
        query=query,
        mode=WorkMode.GENERAL,
        conversation_id="conv-web-mem",
        session_id="conv-web-mem",
    )
    return ReasoningContext(
        req=req,
        workspace=workspace,
        workspace_identity=workspace_identity,
        settings=settings,
        session_id="conv-web-mem",
        response_language="ko",
        parsed_intent=ParsedIntent(intent=ReasoningIntent.GENERAL_CHAT),
        followup_resolution=FollowUpResolution(),
        memory_bundle=RelevantMemoryBundle(workspace_identity=workspace_identity),
        behavior_policy={},
        memory_prefs=None,
        last_context={},
        session_digest=None,
        effective_query=query,
        force_web_search=False,
    )


def _entry(*, confidence: float, source_count: int = 1) -> dict:
    return {
        "entry_id": "e1",
        "query": "아이폰 비교",
        "answer_summary": "아이폰 모델별 차이 요약",
        "sources": [
            {
                "title": "compare",
                "url": "https://example.com/compare",
                "snippet": "모델별 비교",
            }
        ],
        "source_count": source_count,
        "confidence": confidence,
        "lexical_score": 0.22,
        "vector_score": 0.35,
        "created_at": "2026-03-27T00:00:00+00:00",
        "conversation_path": "external_web_search_direct",
        "vector_memory_id": "webmem:conv-web-mem:e1",
    }


def test_general_chat_web_memory_threshold_rejects_059():
    context = _build_context(query="아이폰 차이 정리해줘")
    executor = _FakeExecutor()
    strategy = GeneralChatStrategy()

    composed = asyncio.run(
        strategy.execute(
            context=context,
            dependencies={
                "executor": executor,
                "composer": _FakeComposer(),
                "memory_service": _FakeMemoryService([_entry(confidence=0.59)]),
            },
        )
    )

    assert composed.metadata["conversation_path"] == "local_conversation"
    assert composed.metadata["web_memory_reused"] is False
    assert composed.metadata["web_memory_rank_score"] == 0.0
    assert executor.queries and executor.queries[0] == "아이폰 차이 정리해줘"
    assert executor.max_tokens_seen and executor.max_tokens_seen[0] >= 1024


def test_general_chat_web_memory_threshold_accepts_060():
    context = _build_context(query="아이폰 차이 정리해줘")
    executor = _FakeExecutor()
    strategy = GeneralChatStrategy()

    composed = asyncio.run(
        strategy.execute(
            context=context,
            dependencies={
                "executor": executor,
                "composer": _FakeComposer(),
                "memory_service": _FakeMemoryService([_entry(confidence=0.60)]),
            },
        )
    )

    assert composed.metadata["conversation_path"] == "session_web_memory_reused"
    assert composed.metadata["web_memory_reused"] is True
    assert composed.metadata["web_memory_rank_score"] >= 0.60
    assert composed.metadata["web_path"] == "session_memory"


def test_general_chat_web_memory_rejects_unrelated_query_even_with_high_confidence():
    context = _build_context(query="문제를 하나 작성해줘")
    executor = _FakeExecutor()
    strategy = GeneralChatStrategy()

    composed = asyncio.run(
        strategy.execute(
            context=context,
            dependencies={
                "executor": executor,
                "composer": _FakeComposer(),
                "memory_service": _FakeMemoryService(
                    [
                        {
                            **_entry(confidence=0.95),
                            "lexical_score": 0.02,
                            "vector_score": 0.0,
                        }
                    ]
                ),
            },
        )
    )

    assert composed.metadata["conversation_path"] == "local_conversation"
    assert composed.metadata["web_memory_reused"] is False
    assert executor.queries and "저장된 웹 메모리" not in executor.queries[0]


def test_general_chat_followup_query_injects_previous_context():
    context = _build_context(query="그럼 문제 하나 작성해줘")
    context.last_context = {
        "last_user_query": "스위프트와 파이썬 변수 차이를 설명해줘",
        "result_summary": "스위프트는 정적 타입, 파이썬은 동적 타입입니다.",
    }
    executor = _FakeExecutor()
    strategy = GeneralChatStrategy()

    composed = asyncio.run(
        strategy.execute(
            context=context,
            dependencies={
                "executor": executor,
                "composer": _FakeComposer(),
                "memory_service": _FakeMemoryService([]),
            },
        )
    )

    assert composed.metadata["conversation_path"] == "local_conversation"
    assert executor.queries
    # Follow-up context may be tracked in metadata/state without hard prompt injection text.
    assert executor.queries[0].strip() == "그럼 문제 하나 작성해줘"


def test_general_chat_freshness_query_forces_web_route_even_with_memory_candidate():
    context = _build_context(
        query="아이폰 최신 업데이트 알려줘",
        privacy_mode=PrivacyMode.LOCAL_ONLY,
        hybrid_enabled=False,
    )
    executor = _FakeExecutor()
    strategy = GeneralChatStrategy()

    composed = asyncio.run(
        strategy.execute(
            context=context,
            dependencies={
                "executor": executor,
                "composer": _FakeComposer(),
                "memory_service": _FakeMemoryService([_entry(confidence=0.95)]),
            },
        )
    )

    assert composed.metadata["conversation_path"] == "session_web_memory_reused"
    assert composed.metadata["web_path"] == "session_memory"
    assert composed.metadata["web_memory_reused"] is True


def test_general_chat_web_reasoning_loop_refines_and_converges(monkeypatch):
    context = _build_context(query="아이폰 17 최신 정보 알려줘")
    context.force_web_search = True
    executor = _FakeExecutor()
    strategy = GeneralChatStrategy()
    calls: list[str] = []

    reports = [
        WebRetrievalReport(
            query="round1",
            round_query="round1",
            sources=[
                WebSearchSource(title="블로그", url="https://blog.example.com/iphone17", snippet="루머", content="루머"),
            ],
            logs=["retrieving:http://localhost:8080/search"],
            discovered_count=1,
            fetch_success_count=1,
            fetch_failure_count=0,
            usable_source_count=1,
            unique_domain_count=1,
            quality_score=0.41,
        ),
        WebRetrievalReport(
            query="round2",
            round_query="round2",
            sources=[
                WebSearchSource(title="Apple Newsroom", url="https://www.apple.com/newsroom/", snippet="공식 발표", content="공식 발표"),
                WebSearchSource(title="Reuters", url="https://www.reuters.com/technology/", snippet="보도", content="보도"),
                WebSearchSource(title="The Verge", url="https://www.theverge.com/apple", snippet="분석", content="분석"),
            ],
            logs=["retrieving:https://www.apple.com/newsroom/"],
            discovered_count=3,
            fetch_success_count=3,
            fetch_failure_count=0,
            usable_source_count=3,
            unique_domain_count=3,
            quality_score=0.86,
        ),
    ]

    def _fake_run(self, *, query, **kwargs):
        calls.append(str(query))
        return reports[min(len(calls) - 1, len(reports) - 1)]

    monkeypatch.setattr("local_ai_core.reasoning.helpers.web.general_chat_web_execution_helpers.WebRetriever.run", _fake_run)

    composed = asyncio.run(
        strategy.execute(
            context=context,
            dependencies={
                "executor": executor,
                "composer": _FakeComposer(),
                "memory_service": _FakeMemoryService([]),
            },
        )
    )

    assert composed.metadata["conversation_path"] == "external_web_search_direct"
    assert composed.metadata["web_loop_rounds"] == 2
    assert composed.metadata["web_loop_converged"] is True
    assert len(composed.metadata["web_loop_queries"]) == 2
    assert any(str(log).startswith("web_loop:round=1|") for log in composed.execution_result.tool_logs)
    assert any(str(log) == "web_loop:refine_triggered" for log in composed.execution_result.tool_logs)
    assert any(str(log) == "web_loop:converged" for log in composed.execution_result.tool_logs)


def test_general_chat_memory_recall_prefers_fact_store_response():
    context = _build_context(query="내 이름이 뭐야?")
    context.memory_bundle.session_items = [
        SessionMemoryItem(
            id="fact-1",
            session_id=context.session_id,
            key="fact:user_name",
            value_json={"memory_type": "fact", "value": "민수", "summary": "사용자 이름은 민수입니다."},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            expires_at=None,
        )
    ]
    executor = _FakeExecutor()
    strategy = GeneralChatStrategy()

    composed = asyncio.run(
        strategy.execute(
            context=context,
            dependencies={
                "executor": executor,
                "composer": _FakeComposer(),
                "memory_service": _FakeMemoryService([]),
            },
        )
    )

    assert composed.metadata["conversation_path"] == "local_conversation"
    assert composed.metadata["recall_path"] == "fact_store"
    assert composed.metadata["fact_hit_subject"] == "user_name"
    assert "민수" in str(composed.generated_text or "")
    assert any(
        "memory_recall:fact_store_injected_generation" == str(log)
        for log in (composed.execution_result.tool_logs or [])
    )
