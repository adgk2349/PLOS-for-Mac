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

def test_planner_for_multi_file_summary_uses_map_reduce_strategy():
    planner = LocalPlanner()
    parsed = ParsedIntent(
        intent=ReasoningIntent.SUMMARIZE_FILE,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.75,
    )
    plan = planner.build_plan(
        parsed_intent=parsed,
        mode=WorkMode.SUMMARY,
        file_doc_ids=["a", "b", "c", "d"],
        chunk_ids=["a:0", "b:0", "c:0", "d:0"],
        top_score=0.41,
        prefer_multi_file_summary=True,
    )
    assert plan.plan_type == "summary"
    assert plan.response_strategy == "map_reduce_grounded_summary"
    assert SuggestedActionKind.MAKE_SHORTER in plan.allowed_actions

def test_planner_for_explicit_file_summary_uses_focused_strategy():
    planner = LocalPlanner()
    parsed = ParsedIntent(
        intent=ReasoningIntent.SUMMARIZE_FILE,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.8,
    )
    plan = planner.build_plan(
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        file_doc_ids=["doc10"],
        chunk_ids=["doc10:0", "doc10:1", "doc10:2"],
        top_score=0.62,
        prefer_focused_file_summary=True,
    )
    assert plan.plan_type == "summary"
    assert plan.response_strategy == "focused_file_grounded_summary"
    assert len(plan.selected_chunks) >= 3

def test_planner_for_compare_uses_compact_actions():
    planner = LocalPlanner()
    parsed = ParsedIntent(
        intent=ReasoningIntent.COMPARE_FILES,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.75,
    )
    plan = planner.build_plan(
        parsed_intent=parsed,
        mode=WorkMode.RESEARCH,
        file_doc_ids=["a", "b", "c"],
        chunk_ids=["a:0", "b:0"],
        top_score=0.4,
    )
    assert plan.plan_type == "comparison"
    assert SuggestedActionKind.COMPARE_TOP in plan.allowed_actions
    assert SuggestedActionKind.OPEN_FILE in plan.allowed_actions
    assert SuggestedActionKind.ASK_FOLLOWUP in plan.allowed_actions

def test_planner_for_find_file_keeps_large_candidate_window():
    planner = LocalPlanner()
    parsed = ParsedIntent(
        intent=ReasoningIntent.FIND_FILE,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.78,
    )
    file_doc_ids = [f"doc-{idx}" for idx in range(180)]
    chunk_ids = [f"doc-{idx}:chunk-0" for idx in range(180)]
    plan = planner.build_plan(
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        file_doc_ids=file_doc_ids,
        chunk_ids=chunk_ids,
        top_score=0.52,
    )
    assert plan.plan_type == "file_lookup"
    assert len(plan.selected_files) == 140

def test_verifier_marks_candidate_when_strict_threshold_not_met():
    verifier = ResultVerifier()
    parsed = ParsedIntent(
        intent=ReasoningIntent.EXPLAIN_CONTENT,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.6,
    )
    result = ExecutionResult(
        result_type="answer",
        structured_payload={},
        citations=[
            Citation(
                doc_id="doc",
                chunk_id="chunk",
                file_path="/tmp/a.md",
                snippet="text",
                score=0.21,
                modified_at=datetime.now(timezone.utc),
            )
        ],
        tool_logs=[],
        generated_text="answer",
    )
    verified = verifier.verify(parsed_intent=parsed, execution_result=result, mode=WorkMode.STRICT_SEARCH)
    assert isinstance(verified, VerificationResult)
    assert verified.candidate_mode is True
    assert "strict_threshold_not_met" in verified.issues

def test_conversation_max_tokens_uses_ram_cap(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "16")
    assert ReasoningPipeline._conversation_max_tokens("long", model_profile="advanced") >= 1024

def test_conversation_context_budget_tokens_uses_ram_cap(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "16")
    assert ReasoningPipeline._conversation_context_budget_tokens("long", model_profile="advanced") == 320

def test_v2_chat_token_budget_scales_to_fast(client, auth_headers):
    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={"model_profile": "fast"}
    )
    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "간단하게 답해줘",
            "mode": "GENERAL",
            "conversation_id": "conv-token-fast",
        },
    )
    assert v2.status_code == 200
    # 320 (short) * 0.8 (fast) = 256
    assert v2.json()["metadata"]["conversation_path"] in {"local_conversation", "external_web_search_direct", "external_web_search_unavailable"}

def test_v2_chat_token_budget_scales_to_deep(client, auth_headers):
    client.put(
        "/v1/settings",
        headers=auth_headers,
        json={"model_profile": "deep"}
    )
    v2 = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "매우 상세하게 분석해서 알려줘",
            "mode": "GENERAL",
            "conversation_id": "conv-token-deep",
        },
    )
    assert v2.status_code == 200
    assert v2.json()["metadata"]["conversation_path"] in {"local_conversation", "external_web_search_direct", "external_web_search_unavailable"}

def test_quality_rollup_summary_tracks_rewrite_distribution():
    summary = ReasoningPipeline._quality_rollup_summary(
        [
            {
                "korean_rewrite_used": True,
                "quality_repair_reason": "duplicate_sentence|instruction_leak",
                "assistive_retrieval_suppressed": False,
            },
            {
                "korean_rewrite_used": False,
                "quality_repair_reason": "",
                "assistive_retrieval_suppressed": True,
            },
            {
                "korean_rewrite_used": True,
                "quality_repair_reason": "duplicate_sentence",
                "assistive_retrieval_suppressed": True,
            },
            {
                "korean_rewrite_used": False,
                "quality_repair_reason": "query_echo",
                "assistive_retrieval_suppressed": False,
            },
        ]
    )
    assert summary["sample_size"] == 4
    assert summary["rewrite_count"] == 2
    assert summary["rewrite_rate"] == 0.5
    assert summary["suppressed_count"] == 2
    assert summary["top_repair_reason"] == "duplicate_sentence"

def test_clarification_budget_blocks_consecutive_questions():
    budget = ClarificationBudget()
    state = ClarificationBudgetState(
        clarification_count_current_turn=0,
        previous_turn_was_clarification=True,
        partial_user_answer_received=True,
    )
    allowed = budget.allow_clarification(
        state=state,
        query="1주차?",
        ambiguity_level=0.85,
        risk_level="low",
        candidate_gap_small=True,
    )
    assert allowed is False
