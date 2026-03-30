from __future__ import annotations

# Auto-split from large test module for maintainability.

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

def test_reasoning_pipeline_model_size_parser():
    helpers = SettingsSysHelpers({})
    assert helpers.model_size_b("Qwen3-8B-Q4_K_M.gguf") == 8
    helpers = SettingsSysHelpers({})
    assert helpers.model_size_b("mlx-community/Llama-3.1-8B-Instruct-4bit") == 8
    helpers = SettingsSysHelpers({})
    assert helpers.model_size_b("Llama-3.3-70B-Instruct-Q4_K_M.gguf") == 70
    helpers = SettingsSysHelpers({})
    assert helpers.model_size_b("unknown-model-name") is None

def test_reasoning_pipeline_detects_16gb_tier_model_from_settings():
    settings_small = SimpleNamespace(
        local_engine=LocalEngine.LLAMA_CPP,
        llama_model_path="/tmp/Qwen3-8B-Q4_K_M.gguf",
        mlx_model_path=None,
    )
    settings_large = SimpleNamespace(
        local_engine=LocalEngine.LLAMA_CPP,
        llama_model_path="/tmp/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        mlx_model_path=None,
    )
    helpers = SettingsSysHelpers({})
    assert helpers.is_16gb_tier_model(settings_small) is True
    helpers = SettingsSysHelpers({})
    assert helpers.is_16gb_tier_model(settings_large) is False

def test_reasoning_pipeline_extracts_explicit_file_term_from_query():
    parsed = ParsedIntent(
        intent=ReasoningIntent.SUMMARIZE_FILE,
        entities=ParsedEntities(file_names=[]),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.8,
    )
    helpers = RetrievalHelpers({})
    terms = helpers.extract_explicit_file_terms(
        query="데통10주1차.txt 핵심을 5줄로 요약해줘",
        parsed_intent=parsed,
    )
    assert "데통10주1차.txt" in terms

def test_reasoning_pipeline_apply_explicit_file_focus_matches_target_file():
    doc_ids = {"doc10", "doc11"}
    metadata_map = {
        "doc10": {"path": "/tmp/데통10주1차.txt"},
        "doc11": {"path": "/tmp/데통11주2차.txt"},
    }
    helpers = RetrievalHelpers({})
    filtered_ids, filtered_map, used = helpers.apply_explicit_file_focus(
        doc_ids=doc_ids,
        metadata_map=metadata_map,
        file_terms=["데통10주1차.txt"],
    )
    assert used is True
    assert filtered_ids == {"doc10"}
    assert set(filtered_map.keys()) == {"doc10"}

def test_v2_chat_returns_structured_result_with_actions(client, auth_headers, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "notes.md"
    source.write_text("PLOS Local AI Core는 로컬 RAG와 근거 기반 응답을 제공합니다.", encoding="utf-8")

    client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(workspace)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )

    job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "full"})
    status_payload = _poll_job(client, auth_headers, job.json()["job_id"])
    assert status_payload["status"] == "completed"

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "핵심 요약해줘",
            "mode": "SUMMARY",
            "conversation_id": "conv-v2",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload.get("response_mode")
    assert isinstance(payload.get("metadata"), dict)
    assert isinstance(payload.get("lead"), str)
    assert "structured_result" in payload
    assert "summary" in payload["structured_result"]
    assert payload["structured_result"]["summary"]
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["actions"], list)
    assert "plan" in payload
    assert "verification" in payload

def test_v2_summary_auto_escalates_to_external_for_16gb_tier_model(client, auth_headers, tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "notes.md"
    source.write_text(
        "네트워크 전송 요약 테스트 문서입니다. ACK가 없으면 재전송합니다. 수신 버퍼가 필요합니다.",
        encoding="utf-8",
    )

    client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(workspace)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )

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

    job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "full"})
    status_payload = _poll_job(client, auth_headers, job.json()["job_id"])
    assert status_payload["status"] == "completed"

    from local_ai_core import main

    providers = main.app_state.chat._pipeline._providers
    monkeypatch.setattr(providers, "provider_has_key", lambda provider: provider == "openai")

    def _fake_analyze_sync(*, provider, query, mode, citations, language_preference=None, allow_web_search=False):
        assert provider == "openai"
        assert citations
        return ProviderResult(
            answer="1. 핵심 개념을 요약했습니다.\n2. ACK 누락 시 재전송이 필요합니다.\n3. 수신 버퍼가 핵심입니다.\n4. 전송 안정성은 흐름 제어와 연결됩니다.\n5. 실무에서는 재시도 비용을 줄이는 설계가 중요합니다.",
            sent_chars=220,
        )

    monkeypatch.setattr(providers, "analyze_sync", _fake_analyze_sync)

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "notes.md 핵심을 5줄로 요약해줘",
            "mode": "SUMMARY",
            "conversation_id": "conv-summary-external",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["structured_result"]["summary"]
    assert payload["metadata"]["conversation_path"] in {
        "external_summary_escalated",
        "external_web_search_direct",
        "local_conversation",
        "local_rag",
    }

def test_v2_chat_general_conversation_skips_rag(client, auth_headers, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "notes.md"
    source.write_text("자료구조 노트 테스트 문서입니다.", encoding="utf-8")

    client.post(
        "/v1/workspaces",
        headers=auth_headers,
        json={
            "included_paths": [str(workspace)],
            "excluded_paths": [],
            "startup_profile": "RECOMMENDED",
            "default_mode": "GENERAL",
        },
    )
    job = client.post("/v1/index/jobs", headers=auth_headers, json={"scope": "full"})
    status_payload = _poll_job(client, auth_headers, job.json()["job_id"])
    assert status_payload["status"] == "completed"

    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "안녕!",
            "mode": "GENERAL",
            "conversation_id": "conv-chat",
            "top_k": 6,
            "filters": None,
        },
    )
    assert v2.status_code == 200
    payload = v2.json()
    assert payload["structured_result"]["result_type"] in {"conversation", "runtime_error"}
    assert payload["plan"]["plan_type"] == "conversation"
    assert payload["citations"] == []
    assert payload["metadata"]["conversation_path"] in {"local_conversation", "external_escalated"}
    assert payload["metadata"]["reasoning_hidden"] is True
    assert "direct_first_applied" in payload["metadata"]
    assert "question_count_after_postprocess" in payload["metadata"]
    assert "recommendation_shape" in payload["metadata"]

def test_adaptive_response_length_for_query_detects_detail_and_brief():
    assert ReasoningPipeline._adaptive_response_length_for_query(
        query="이거 자세히 길게 설명해줘",
        base_response_length="medium",
        explicit_web_search_request=False,
        last_context=None,
    ) == "long"
    assert ReasoningPipeline._adaptive_response_length_for_query(
        query="한 줄로 간단히 말해줘",
        base_response_length="long",
        explicit_web_search_request=False,
        last_context=None,
    ) == "short"
    assert ReasoningPipeline._adaptive_response_length_for_query(
        query="인터넷에서 감스트 검색해봐",
        base_response_length="short",
        explicit_web_search_request=True,
        last_context=None,
    ) == "medium"

def test_reasoning_pipeline_does_not_force_greeting_when_task_cues_present():
    parsed = ParsedIntent(
        intent=ReasoningIntent.FIND_FILE,
        entities=ParsedEntities(topics=["데통"], file_names=[], tags=[], projects=[]),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.74,
        operation="find",
        target="데통",
        scope="single",
        ambiguity="clear",
    )
    assert (
        ReasoningPipeline._should_force_general_chat(
            query="안녕 데통 파일 찾아줘",
            parsed_intent=parsed,
        )
        is False
    )

def test_reasoning_pipeline_detects_instructional_leak_tokens():
    assert RetrievalHelpers({}).looks_like_reasoning_leak(
        "사용자에게 물어볼 때는 반드시 '?'를 붙여주세요. 최종 답변: 현재 시간은 10시 30분입니다."
    )

def test_reasoning_pipeline_detects_three_sentence_rule_leak():
    assert RetrievalHelpers({}).looks_like_reasoning_leak(
        "단, 사용자의 질문에 대한 명확한 답변이 필요할 경우 3문장까지 가능합니다."
    )

def test_reasoning_pipeline_detects_insufficient_answer_rule_leak():
    assert RetrievalHelpers({}).looks_like_reasoning_leak(
        "단, 사용자의 질문에 대한 답변이 부족할 경우 추가적인 질문을 덧붙일 수 있습니다."
    )

def test_reasoning_pipeline_detects_help_user_directly_leak():
    assert RetrievalHelpers({}).looks_like_reasoning_leak("사용자에게 직접 도움을 주세요.")

def test_reasoning_pipeline_detects_react_immediately_rule_leak():
    assert RetrievalHelpers({}).looks_like_reasoning_leak("사용자의 말에 바로 반응하세요.")

def test_reasoning_pipeline_detects_user_message_rule_leak():
    assert RetrievalHelpers({}).looks_like_reasoning_leak(
        "사용자 메시지에 바로 반응하세요. 사용자 메시지에 명확한 답을 하세요."
    )

def test_reasoning_pipeline_skips_context_for_first_greeting_turn():
    assert (
        ReasoningPipeline._should_apply_conversation_context(
            "안녕",
            has_session_digest=False,
            has_last_context=False,
        )
        is False
    )

def test_reasoning_pipeline_applies_context_when_history_exists():
    assert (
        ReasoningPipeline._should_apply_conversation_context(
            "그러면 몇 시쯤 자야되지? 보통 새벽 3-4시에 잤어",
            has_session_digest=True,
            has_last_context=False,
        )
        is True
    )
    assert (
        ReasoningPipeline._should_apply_conversation_context(
            "그거 이어서 설명해줘",
            has_session_digest=False,
            has_last_context=True,
        )
        is True
    )

def test_reasoning_pipeline_detects_brief_chat_query():
    assert RetrievalHelpers({}).is_brief_chat_query("그렇구나!")
    assert not RetrievalHelpers({}).is_brief_chat_query("운동 후 목이 시린 이유를 단계별로 설명해줘")

def test_conversation_session_summary_prefers_user_turns_for_chat_context():
    digest = {
        "recent_turns": [
            {"role": "assistant", "text": "집에서 먹는 거 편하죠?"},
            {"role": "user", "text": "배달 시켜먹어?"},
            {"role": "assistant", "text": "집에서 편하게 먹을 수 있어요."},
            {"role": "user", "text": "오늘 아침엔 뭐 먹을까"},
        ],
        "open_loops": [],
        "stable_facts": [],
        "active_topics": [],
    }
    summary = ReasoningPipeline._conversation_session_summary(
        query="오늘 아침엔 뭐 먹을까",
        session_digest=digest,
        last_context=None,
        memory_bundle=SimpleNamespace(session_items=[]),
        response_length="medium",
    )
    assert "- U:" in summary
    assert "- A:" not in summary

def test_reasoning_pipeline_detects_repetitive_conversation_output():
    digest = {
        "recent_turns": [
            {"role": "assistant", "text": "집에서 편하게 먹을 수 있어요. 집에서 먹는 거 편하죠?"},
        ]
    }
    assert ReasoningPipeline._looks_repetitive_conversation_output(
        query="배달 시켜먹어?",
        answer="집에서 편하게 먹을 수 있어요. 집에서 먹는 거 편하죠?",
        session_digest=digest,
        last_context={"result_summary": "집에서 편하게 먹을 수 있어요. 집에서 먹는 거 편하죠?"},
    )

def test_reasoning_pipeline_allows_non_repetitive_conversation_output():
    digest = {
        "recent_turns": [
            {"role": "assistant", "text": "아침은 가볍게 먹는 게 좋아요."},
        ]
    }
    assert not ReasoningPipeline._looks_repetitive_conversation_output(
        query="배달 시켜먹어?",
        answer="배달도 괜찮지만 오늘은 속이 더부룩하니까 따뜻한 국이나 죽이 더 나아요.",
        session_digest=digest,
        last_context={"result_summary": "아침은 가볍게 먹는 게 좋아요."},
    )

def test_general_chat_assist_mode_suppresses_local_retrieval_for_non_file_search_chat():
    parsed = ParsedIntent(
        intent=ReasoningIntent.EXPLAIN_CONTENT,
        entities=ParsedEntities(topics=["클로드", "소넷"], file_names=[], tags=[], projects=[]),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.63,
        operation="summarize",
        target="소넷",
        scope="single",
        ambiguity="clear",
    )
    mode, _confidence, suppressed = ReasoningPipeline._general_chat_assist_mode(
        parsed_intent=parsed,
        query="지금 최신버전이 맞아? 모르겠으면 검색해봐",
    )
    assert mode == "none"
    assert suppressed is True
