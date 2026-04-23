from __future__ import annotations

# Auto-split submodule

from __future__ import annotations

from __future__ import annotations

import asyncio

import json

import time

import unicodedata

from datetime import datetime, timezone

from pathlib import Path

from types import SimpleNamespace

from local_ai_core.external_providers import ProviderResult

from local_ai_core.nlu.intent_parser import IntentParser

from local_ai_core.local_planner import LocalPlanner

from local_ai_core.reasoning.pipeline import ReasoningPipeline

from local_ai_core.reasoning.helpers.retrieval_helpers import RetrievalHelpers

from local_ai_core.reasoning.helpers.settings_sys_helpers import SettingsSysHelpers

from local_ai_core.nlu.followup_resolver import FollowUpResolution, FollowUpResolver

from local_ai_core.nlu.clarification_budget import ClarificationBudget, ClarificationBudgetState

from local_ai_core.models import (
    Citation,
    ComposedChatResponseV2,
    ExecutionResult,
    LocalPlan,
    ParsedEntities,
    ParsedIntent,
    ParsedTimeFilters,
    ParsedWorkspaceFilters,
    LocalEngine,
    PrivacyMode,
    ReasoningIntent,
    StartupProfile,
    StructuredResult,
    SuggestedActionKind,
    VerificationResult,
    WorkMode,
    WorkspaceResponse,
)

from local_ai_core.verifier import ResultVerifier

def _patch_direct_web_search(monkeypatch, fake_direct_web_search):
    def _fake_loop(
        self,
        *,
        retriever,
        base_query,
        freshness_sensitive_query,
        searxng_url,
        prefer_searxng,
        max_rounds=3,
        max_total_seconds=18.0,
        round_timeout_seconds=6.0,
    ):
        _ = (self, retriever, freshness_sensitive_query, searxng_url, prefer_searxng, max_rounds, max_total_seconds, round_timeout_seconds)
        response_length = "long" if any(token in str(base_query) for token in ("자세히", "상세", "길게", "deep", "detail")) else "medium"
        execution = fake_direct_web_search(
            query=base_query,
            mode=WorkMode.GENERAL,
            response_language="ko",
            workspace=None,
            settings=None,
            response_length=response_length,
        )
        if execution is None:
            return [], ["web_search:unavailable"], {
                "web_loop_rounds": 1,
                "web_loop_converged": False,
                "web_loop_quality_score": 0.0,
                "web_loop_queries": [base_query],
            }
        rows: list[dict[str, str]] = []
        logs = list(execution.tool_logs or [])
        for line in logs:
            text = str(line or "")
            if text.startswith("retrieved:"):
                url = text.split(":", 1)[1].strip()
                if not url:
                    continue
                rows.append(
                    {
                        "title": url,
                        "url": url,
                        "snippet": "",
                    }
                )
        meta = {
            "web_loop_rounds": 1,
            "web_loop_converged": True,
            "web_loop_quality_score": 0.9,
            "web_loop_queries": [base_query],
        }
        return rows[:3], logs, meta

    monkeypatch.setattr(
        "local_ai_core.reasoning.strategies.general_chat.GeneralChatStrategy._run_web_reasoning_loop",
        _fake_loop,
    )

def _poll_job(client, headers: dict[str, str], job_id: str, timeout: float = 20.0) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f"/v1/index/jobs/{job_id}", headers=headers)
        res.raise_for_status()
        payload = res.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not finish in time")

def test_v2_chat_auto_web_search_for_latest_uncertain_query(client, auth_headers, monkeypatch):
    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={
            "privacy_mode": PrivacyMode.HYBRID.value,
            "hybrid_web_search_enabled": True,
            "startup_profile": "RECOMMENDED",
            "model_profile": "advanced",
            "local_engine": LocalEngine.LLAMA_CPP.value,
            "llama_model_path": "/tmp/Qwen3-8B-Q4_K_M.gguf",
            "mlx_model_path": None,
            "reindex_policy": "filewatch_incremental",
            "language": "ko",
            "action_permission_mode": "ASK_PER_ACTION",
            "adaptive_personalization_enabled": True,
            "session_memory_enabled": True,
            "workspace_memory_enabled": True,
            "local_memory_only": True,
            "workspace_memory_mode": "normal",
        },
    )

    from local_ai_core import main

    pipeline = main.app_state.chat._pipeline

    def _fake_direct_web_search(*, query, mode, response_language, workspace, settings, response_length):
        assert "최신버전" in query
        return ExecutionResult(
            result_type="conversation",
            structured_payload={"style": "general_chat", "source": "web_search_direct", "ungrounded_allowed": True},
            citations=[],
            tool_logs=[
                "planning:web_search_requested",
                "web_search:requested",
                "web_search:direct",
                "retrieving:https://duckduckgo.com/html",
                "retrieved:https://duckduckgo.com/html",
            ],
            generated_text="최신 버전 정보를 웹에서 확인했습니다.",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="direct_web_search=1",
        )

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "지금 최신버전이 맞아? 모르겠으면 검색해봐",
            "mode": "GENERAL",
            "conversation_id": "conv-web-auto-latest",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["metadata"]["conversation_path"] in {
        "external_web_search_direct",
        "external_web_search_unavailable",
        "local_conversation",
    }
    if payload["metadata"]["conversation_path"] == "external_web_search_direct":
        assert payload["metadata"]["web_path"] == "direct"
    elif payload["metadata"]["conversation_path"] == "external_web_search_unavailable":
        assert payload["metadata"]["web_path"] == "unavailable"
    assert payload["metadata"]["web_auto_triggered"] is True
    assert payload["structured_result"]["summary"]

def test_v2_chat_web_search_detail_query_uses_long_response_length(client, auth_headers, monkeypatch):
    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={
            "privacy_mode": PrivacyMode.HYBRID.value,
            "hybrid_web_search_enabled": True,
            "startup_profile": "RECOMMENDED",
            "model_profile": "advanced",
            "local_engine": LocalEngine.LLAMA_CPP.value,
            "llama_model_path": "/tmp/Qwen3-8B-Q4_K_M.gguf",
            "mlx_model_path": None,
            "reindex_policy": "filewatch_incremental",
            "language": "ko",
            "action_permission_mode": "ASK_PER_ACTION",
            "adaptive_personalization_enabled": True,
            "session_memory_enabled": True,
            "workspace_memory_enabled": True,
            "local_memory_only": True,
            "workspace_memory_mode": "normal",
        },
    )

    from local_ai_core import main

    pipeline = main.app_state.chat._pipeline

    def _fake_direct_web_search(*, query, mode, response_language, workspace, settings, response_length):
        assert response_length == "long"
        return ExecutionResult(
            result_type="conversation",
            structured_payload={"style": "general_chat", "source": "web_search_direct", "ungrounded_allowed": True},
            citations=[],
            tool_logs=[
                "web_search:requested",
                "web_search:direct",
                "retrieving:https://api.duckduckgo.com/",
                "retrieved:https://api.duckduckgo.com/",
            ],
            generated_text="웹 검색 상세 결과입니다.",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="direct_web_search=1",
        )

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "인터넷에서 감스트 프로필 자세히 알려줘",
            "mode": "GENERAL",
            "conversation_id": "conv-web-length-long",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["metadata"]["conversation_path"] in {"external_web_search_direct", "local_conversation"}
    if "effective_response_length" in payload["metadata"]:
        assert payload["metadata"]["effective_response_length"] == "long"

def test_v2_chat_web_search_followup_keeps_web_path(client, auth_headers, monkeypatch):
    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={
            "privacy_mode": PrivacyMode.HYBRID.value,
            "hybrid_web_search_enabled": True,
            "startup_profile": "RECOMMENDED",
            "model_profile": "advanced",
            "local_engine": LocalEngine.LLAMA_CPP.value,
            "llama_model_path": "/tmp/Qwen3-8B-Q4_K_M.gguf",
            "mlx_model_path": None,
            "reindex_policy": "filewatch_incremental",
            "language": "ko",
            "action_permission_mode": "ASK_PER_ACTION",
            "adaptive_personalization_enabled": True,
            "session_memory_enabled": True,
            "workspace_memory_enabled": True,
            "local_memory_only": True,
            "workspace_memory_mode": "normal",
        },
    )

    from local_ai_core import main

    pipeline = main.app_state.chat._pipeline
    calls = {"count": 0, "queries": []}

    def _fake_direct_web_search(*, query, mode, response_language, workspace, settings, response_length):
        calls["count"] += 1
        calls["queries"].append(query)
        return ExecutionResult(
            result_type="conversation",
            structured_payload={"style": "general_chat", "source": "web_search_direct", "ungrounded_allowed": True},
            citations=[],
            tool_logs=[
                "web_search:requested",
                "web_search:direct",
                "retrieving:https://api.duckduckgo.com/",
                "retrieved:https://api.duckduckgo.com/",
            ],
            generated_text=f"웹 검색 처리: {query}",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="direct_web_search=1",
        )

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)

    first = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "인터넷에서 감스트 검색해봐",
            "mode": "GENERAL",
            "conversation_id": "conv-web-followup",
            "top_k": 6,
            "filters": None,
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["metadata"]["conversation_path"] in {
        "external_web_search_direct",
        "external_web_search_unavailable",
    }

    second = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "프로필 검색해봐",
            "mode": "GENERAL",
            "conversation_id": "conv-web-followup",
            "top_k": 6,
            "filters": None,
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["metadata"]["conversation_path"] in {
        "external_web_search_direct",
        "external_web_search_unavailable",
        "local_conversation",
    }
    if second_payload["metadata"]["conversation_path"] == "external_web_search_direct":
        assert second_payload["metadata"]["web_path"] == "direct"
    elif second_payload["metadata"]["conversation_path"] == "external_web_search_unavailable":
        assert second_payload["metadata"]["web_path"] == "unavailable"
    assert isinstance(second_payload["citations"], list)
    assert calls["count"] >= 1
    if second_payload["metadata"]["conversation_path"] == "external_web_search_direct":
        assert calls["count"] >= 2
        assert any("프로필 검색해봐" in item for item in calls["queries"])

def test_v2_chat_web_search_followup_detail_question_stays_on_web_path(client, auth_headers, monkeypatch):
    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={
            "privacy_mode": PrivacyMode.HYBRID.value,
            "hybrid_web_search_enabled": True,
            "startup_profile": "RECOMMENDED",
            "model_profile": "advanced",
            "local_engine": LocalEngine.LLAMA_CPP.value,
            "llama_model_path": "/tmp/Qwen3-8B-Q4_K_M.gguf",
            "mlx_model_path": None,
            "reindex_policy": "filewatch_incremental",
            "language": "ko",
            "action_permission_mode": "ASK_PER_ACTION",
            "adaptive_personalization_enabled": True,
            "session_memory_enabled": True,
            "workspace_memory_enabled": True,
            "local_memory_only": True,
            "workspace_memory_mode": "normal",
        },
    )

    from local_ai_core import main

    pipeline = main.app_state.chat._pipeline
    queries: list[str] = []

    def _fake_direct_web_search(*, query, mode, response_language, workspace, settings, response_length):
        queries.append(query)
        return ExecutionResult(
            result_type="conversation",
            structured_payload={"style": "general_chat", "source": "web_search_direct", "ungrounded_allowed": True},
            citations=[],
            tool_logs=[
                "web_search:requested",
                "web_search:direct",
                "retrieving:https://api.duckduckgo.com/",
                "retrieved:https://api.duckduckgo.com/",
            ],
            generated_text="감스트는 아프리카TV BJ예요.",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="direct_web_search=1",
        )

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)

    first = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "인터넷에서 감스트 검색해봐",
            "mode": "GENERAL",
            "conversation_id": "conv-web-followup-detail",
            "top_k": 6,
            "filters": None,
        },
    )
    assert first.status_code == 200
    assert first.json()["metadata"]["conversation_path"] == "external_web_search_direct"

    second = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "몇년도부터 방송을 시작했지? 레전드 영상 있나",
            "mode": "GENERAL",
            "conversation_id": "conv-web-followup-detail",
            "top_k": 6,
            "filters": None,
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["metadata"]["conversation_path"] in {"external_web_search_direct", "local_conversation"}
    if second_payload["metadata"]["conversation_path"] == "external_web_search_direct":
        assert second_payload["metadata"]["web_path"] == "direct"
    assert len(queries) >= 1
    if len(queries) >= 2:
        assert "후속 질문: 몇년도부터 방송을 시작했지? 레전드 영상 있나" in queries[-1]

def test_reasoning_pipeline_contextless_web_search_directive_uses_previous_topic():
    last_context = {
        "conversation_path": "local_conversation",
        "last_user_query": "오늘 날짜가 뭐야?",
        "result_summary": "날짜 정보를 확신하지 못함",
    }
    assert ReasoningPipeline._is_followup_web_search_request(
        query="인터넷 검색해봐",
        last_context=last_context,
    ) is True
    expanded = ReasoningPipeline._web_search_query_for_turn(
        query="인터넷 검색해봐",
        last_context=last_context,
        is_followup_web_search=True,
    )
    assert "오늘 날짜가 뭐야?" in expanded
    assert "후속 질문: 인터넷 검색해봐" in expanded

def test_reasoning_pipeline_topicful_web_search_request_does_not_force_previous_topic():
    last_context = {
        "conversation_path": "local_conversation",
        "last_user_query": "오늘 날짜가 뭐야?",
        "result_summary": "날짜 정보를 확신하지 못함",
    }
    assert ReasoningPipeline._is_followup_web_search_request(
        query="인터넷에서 감스트 검색해봐",
        last_context=last_context,
    ) is True

def test_v2_chat_contextless_web_search_directive_inherits_previous_local_question(client, auth_headers, monkeypatch):
    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={
            "privacy_mode": PrivacyMode.HYBRID.value,
            "hybrid_web_search_enabled": True,
            "startup_profile": "RECOMMENDED",
            "model_profile": "advanced",
            "local_engine": LocalEngine.LLAMA_CPP.value,
            "llama_model_path": "/tmp/Qwen3-8B-Q4_K_M.gguf",
            "mlx_model_path": None,
            "reindex_policy": "filewatch_incremental",
            "language": "ko",
            "action_permission_mode": "ASK_PER_ACTION",
            "adaptive_personalization_enabled": True,
            "session_memory_enabled": True,
            "workspace_memory_enabled": True,
            "local_memory_only": True,
            "workspace_memory_mode": "normal",
        },
    )

    from local_ai_core import main

    pipeline = main.app_state.chat._pipeline
    web_queries: list[str] = []

    def _fake_execute_conversation(*, query, mode, startup_profile, engine, mlx_model_path, llama_model_path, language_preference, session_summary, max_tokens):
        return ExecutionResult(
            result_type="conversation",
            structured_payload={"style": "general_chat", "source": "local_model", "ungrounded_allowed": True},
            citations=[],
            tool_logs=["router:intent=general_chat"],
            generated_text=f"로컬 답변: {query}",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="local_only=1",
        )

    def _fake_direct_web_search(*, query, mode, response_language, workspace, settings, response_length):
        web_queries.append(query)
        return ExecutionResult(
            result_type="conversation",
            structured_payload={"style": "general_chat", "source": "web_search_direct", "ungrounded_allowed": True},
            citations=[],
            tool_logs=[
                "web_search:requested",
                "web_search:direct",
                "retrieving:https://api.duckduckgo.com/",
                "retrieved:https://api.duckduckgo.com/",
            ],
            generated_text="웹 검색 결과",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="direct_web_search=1",
        )

    monkeypatch.setattr(pipeline._executor, "execute_conversation", _fake_execute_conversation)
    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)

    first = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "파이썬이 뭐야?",
            "mode": "GENERAL",
            "conversation_id": "conv-contextless-web-carry",
            "top_k": 6,
            "filters": None,
        },
    )
    assert first.status_code == 200
    assert str(first.json()["metadata"]["conversation_path"]).startswith("local_conversation")

    second = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "인터넷 검색해봐",
            "mode": "GENERAL",
            "conversation_id": "conv-contextless-web-carry",
            "top_k": 6,
            "filters": None,
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["metadata"]["conversation_path"] in {
        "external_web_search_direct",
        "external_web_search_unavailable",
    }
    if second_payload["metadata"]["conversation_path"] == "external_web_search_direct":
        assert second_payload["metadata"]["web_path"] == "direct"
        assert len(web_queries) >= 1
        assert "파이썬이 뭐야?" in web_queries[-1]
        assert "후속 질문: 인터넷 검색해봐" in web_queries[-1]
    else:
        assert second_payload["metadata"]["web_path"] == "unavailable"

def test_v2_chat_web_search_target_with_info_question_is_explicit(client, auth_headers, monkeypatch):
    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={
            "privacy_mode": PrivacyMode.HYBRID.value,
            "hybrid_web_search_enabled": True,
            "startup_profile": "RECOMMENDED",
            "model_profile": "advanced",
            "local_engine": LocalEngine.LLAMA_CPP.value,
            "llama_model_path": "/tmp/Qwen3-8B-Q4_K_M.gguf",
            "mlx_model_path": None,
            "reindex_policy": "filewatch_incremental",
            "language": "ko",
            "action_permission_mode": "ASK_PER_ACTION",
            "adaptive_personalization_enabled": True,
            "session_memory_enabled": True,
            "workspace_memory_enabled": True,
            "local_memory_only": True,
            "workspace_memory_mode": "normal",
        },
    )

    from local_ai_core import main

    pipeline = main.app_state.chat._pipeline

    def _fake_direct_web_search(*, query, mode, response_language, workspace, settings, response_length):
        assert "인터넷에 감스트 정보" in query
        return ExecutionResult(
            result_type="conversation",
            structured_payload={"style": "general_chat", "source": "web_search_direct", "ungrounded_allowed": True},
            citations=[],
            tool_logs=[
                "web_search:requested",
                "web_search:direct",
                "retrieving:https://api.duckduckgo.com/",
                "retrieved:https://api.duckduckgo.com/",
            ],
            generated_text="웹 근거를 찾았습니다.",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="direct_web_search=1",
        )

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "인터넷에 감스트 정보 있어?",
            "mode": "GENERAL",
            "conversation_id": "conv-web-target-info",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["metadata"]["conversation_path"] in {"external_web_search_direct", "local_conversation"}

def test_reasoning_pipeline_forces_general_chat_for_latest_search_query_without_file_cues():
    parsed = ParsedIntent(
        intent=ReasoningIntent.EXPLAIN_CONTENT,
        entities=ParsedEntities(topics=["클로드", "소넷"], file_names=[], tags=[], projects=[]),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.58,
        operation="summarize",
        target="소넷",
        scope="single",
        ambiguity="clear",
    )
    assert (
        ReasoningPipeline._should_force_general_chat(
            query="지금 최신버전이 맞아? 모르겠으면 검색해봐",
            parsed_intent=parsed,
            last_context=None,
        )
        is True
    )

def test_should_auto_web_search_triggers_for_freshness_and_uncertainty():
    parsed = ParsedIntent(
        intent=ReasoningIntent.GENERAL_CHAT,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.52,
        operation="chat",
        target=None,
        scope="single",
        ambiguity="unclear",
    )
    assert (
        ReasoningPipeline._should_auto_web_search(
            query="지금 최신버전이 맞아? 모르겠으면 검색해봐",
            parsed_intent=parsed,
            last_context=None,
        )
        is True
    )

def test_should_auto_web_search_does_not_trigger_for_generic_howto_coding_query():
    parsed = ParsedIntent(
        intent=ReasoningIntent.GENERAL_CHAT,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.92,
        operation="chat",
        target=None,
        scope="single",
        ambiguity="clear",
    )
    assert (
        ReasoningPipeline._should_auto_web_search(
            query="문제가 지금 어떻게 되는거야? two sum 풀이를 설명해줘",
            parsed_intent=parsed,
            last_context=None,
        )
        is False
    )
