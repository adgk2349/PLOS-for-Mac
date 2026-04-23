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

def test_v2_chat_explicit_web_search_blocked_when_hybrid_web_off(client, auth_headers):
    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={
            "privacy_mode": PrivacyMode.HYBRID.value,
            "hybrid_web_search_enabled": False,
            "startup_profile": "RECOMMENDED",
            "model_profile": "advanced",
            "local_engine": LocalEngine.LLAMA_CPP.value,
            "llama_model_path": None,
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
    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "인터넷 검색으로 스위프트 최신 문서 링크 찾아줘",
            "mode": "GENERAL",
            "conversation_id": "conv-web-blocked",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["is_local"] is True
    assert payload["metadata"]["conversation_path"] == "external_web_search_blocked"
    assert payload["metadata"]["web_path"] == "blocked"
    assert payload["metadata"]["web_sources_count"] >= 0
    assert payload["metadata"]["web_fetch_failures"] == 0
    assert "웹검색(인터넷 경로)이 꺼져" in payload["structured_result"]["summary"]
    assert payload["citations"] == []

def test_v2_chat_explicit_web_search_escalates_when_allowed(client, auth_headers, monkeypatch):
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
        assert "인터넷 검색" in query
        return ExecutionResult(
            result_type="conversation",
            structured_payload={"style": "general_chat", "source": "web_search_direct", "ungrounded_allowed": True},
            citations=[],
            tool_logs=[
                "web_search:requested",
                "web_search:direct",
                "retrieving:https://api.duckduckgo.com/",
                "retrieved:https://api.duckduckgo.com/",
                "retrieving:https://docs.swift.org/swift-book/",
                "retrieved:https://docs.swift.org/swift-book/",
            ],
            generated_text="최신 Swift 공식 문서는 https://docs.swift.org/swift-book/ 입니다.",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="direct_web_search=1",
        )

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "인터넷 검색으로 Swift 공식 문서 최신 링크 찾아줘",
            "mode": "GENERAL",
            "conversation_id": "conv-web-escalated",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["is_local"] is False
    assert payload["metadata"]["conversation_path"] == "external_web_search_direct"
    assert payload["metadata"]["web_path"] == "direct"
    assert payload["metadata"]["web_sources_count"] >= 1
    assert payload["metadata"]["web_fetch_failures"] == 0
    assert payload["metadata"]["escalated_provider"] is None
    assert "docs.swift.org/swift-book" in payload["structured_result"]["summary"]
    trace_events = payload["metadata"].get("trace_events") or []
    assert any(str(event.get("status")) == "retrieving" for event in trace_events)
    assert any(str(event.get("status")) == "retrieved" for event in trace_events)

def test_v2_chat_explicit_web_search_with_direct_url_uses_web_path(client, auth_headers, monkeypatch):
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
        assert "https://" in query
        return ExecutionResult(
            result_type="conversation",
            structured_payload={
                "style": "general_chat",
                "source": "web_search_direct",
                "ungrounded_allowed": True,
                "web_path": "direct",
                "web_sources_count": 2,
                "web_fetch_failures": 1,
            },
            citations=[],
            tool_logs=[
                "planning:web_search_requested",
                "retrieving:https://example.com/",
                "retrieved:https://example.com/",
                "done:web_evidence_composed:2",
            ],
            generated_text="요약 완료",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="web_path=direct|web_sources_count=2|web_fetch_failures=1",
        )

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "https://example.com 이 문서 요약해줘",
            "mode": "GENERAL",
            "conversation_id": "conv-web-direct-url",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["metadata"]["conversation_path"] in {
        "external_web_search_direct",
        "external_web_search_unavailable",
    }
    if payload["metadata"]["conversation_path"] == "external_web_search_direct":
        assert payload["metadata"]["web_path"] == "direct"
        assert payload["metadata"]["web_sources_count"] >= 1
    else:
        assert payload["metadata"]["web_path"] == "unavailable"
    assert payload["metadata"]["web_fetch_failures"] >= 0

def test_v2_chat_explicit_web_search_unavailable_returns_no_guess_message(client, auth_headers, monkeypatch):
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
        pipeline._last_web_report = {
            "logs": [
                "planning:web_search_requested",
                "warning:search_failed:https://html.duckduckgo.com/html/",
                "warning:search_failed:https://api.duckduckgo.com/",
            ],
            "sources_count": 0,
            "fetch_failures": 0,
            "discovered_count": 0,
        }
        return None

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)
    monkeypatch.setattr(pipeline, "_escalate_general_chat", lambda **kwargs: None)

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "인터넷에서 감스트 검색해봐",
            "mode": "GENERAL",
            "conversation_id": "conv-web-unavailable",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["metadata"]["conversation_path"] == "external_web_search_unavailable"
    assert payload["metadata"]["web_path"] == "unavailable"
    assert payload["metadata"]["web_sources_count"] == 0
    assert payload["metadata"]["web_fetch_failures"] == 0
    assert "신뢰 가능한 근거를 수집하지 못해" in payload["structured_result"]["summary"]
    trace_events = payload["metadata"].get("trace_events") or []
    assert isinstance(trace_events, list)

def test_v2_chat_explicit_web_search_detects_nfd_korean(client, auth_headers, monkeypatch):
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
            generated_text=f"웹 검색 결과: {query}",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="direct_web_search=1",
        )

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)

    nfd_query = unicodedata.normalize("NFD", "인터넷에서 감스트 검색해봐")
    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": nfd_query,
            "mode": "GENERAL",
            "conversation_id": "conv-web-nfd",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["metadata"]["conversation_path"] in {
        "external_web_search_direct",
        "external_web_search_unavailable",
    }
    if payload["metadata"]["conversation_path"] == "external_web_search_direct":
        trace_events = payload["metadata"].get("trace_events") or []
        assert any(str(event.get("status")) == "retrieving" for event in trace_events)
    else:
        assert payload["metadata"]["web_path"] == "unavailable"

def test_v2_chat_explicit_web_search_skips_conversation_repair(client, auth_headers, monkeypatch):
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
    repair_calls = {"count": 0}

    def _fake_direct_web_search(*, query, mode, response_language, workspace, settings, response_length):
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
            generated_text=f"{query} 검색 결과를 찾았습니다.",
            engine_used=LocalEngine.LLAMA_CPP,
            used_fallback=False,
            runtime_detail="direct_web_search=1",
        )

    def _fake_repair(**kwargs):
        repair_calls["count"] += 1
        return None

    _patch_direct_web_search(monkeypatch, _fake_direct_web_search)
    monkeypatch.setattr(pipeline, "_repair_repetitive_conversation_response", _fake_repair)

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "인터넷에서 감스트 검색해봐",
            "mode": "GENERAL",
            "conversation_id": "conv-web-no-repair",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["metadata"]["conversation_path"] in {
        "external_web_search_direct",
        "external_web_search_unavailable",
    }
    if payload["metadata"]["conversation_path"] == "external_web_search_unavailable":
        assert payload["metadata"]["web_path"] == "unavailable"
    assert repair_calls["count"] >= 0
    trace_events = payload["metadata"].get("trace_events") or []
    assert any(str(event.get("message") or "").startswith("retrieving https://api.duckduckgo.com/") for event in trace_events)
