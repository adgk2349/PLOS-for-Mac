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
        "local_ai_core.reasoning.strategies.general_chat_sections.general_chat_web_mixin.GeneralChatWebMixin._run_web_reasoning_loop",
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

def test_intent_parser_extracts_year_and_tags():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=["/tmp/workspace/ignore"],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="2025 #swift 프로젝트 문서 비교해줘",
        mode=WorkMode.RESEARCH,
        workspace=workspace,
    )
    assert parsed.intent == ReasoningIntent.COMPARE_FILES
    assert parsed.time_filters.year == 2025
    assert "swift" in [tag.lower() for tag in parsed.entities.tags]

def test_intent_parser_broad_korean_inventory_question_maps_to_find_file():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="지금 강의 뭐뭐 있지",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.intent == ReasoningIntent.FIND_FILE

def test_intent_parser_folder_query_maps_to_find_file():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="자료구조 폴더에 지금 뭐있어",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.intent == ReasoningIntent.FIND_FILE

def test_intent_parser_greeting_maps_to_general_chat():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="안녕, 오늘 어때?",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.intent == ReasoningIntent.GENERAL_CHAT

def test_intent_parser_everyday_chat_maps_to_general_chat():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="나 배고파, 햄버거 어때?",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.intent == ReasoningIntent.GENERAL_CHAT

def test_intent_parser_search_without_local_file_target_maps_to_general_chat():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="클로드 소넷 최신버전 맞아? 모르겠으면 검색해봐",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.intent == ReasoningIntent.GENERAL_CHAT

def test_intent_parser_important_question_maps_to_summary():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="자료구조에서 중요한 게 뭐였지",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.intent == ReasoningIntent.SUMMARIZE_FILE

def test_intent_parser_extracts_korean_file_name_entity():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="데통10주1차.txt 핵심을 5줄로 요약해줘",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert "데통10주1차.txt" in parsed.entities.file_names

def test_intent_parser_extracts_scope_target_slots_for_all_request():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="데통 파일 전부 찾아줘",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.operation == "find"
    assert parsed.scope == "all"
    assert parsed.target == "데통"
    assert parsed.ambiguity == "clear"

def test_intent_parser_marks_ambiguous_all_scope_without_target():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="파일 전부 보여줘",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.operation == "find"
    assert parsed.scope == "all"
    assert parsed.target is None
    assert parsed.ambiguity == "unclear"

def test_reasoning_pipeline_detects_multi_file_summary_scope():
    parsed = ParsedIntent(
        intent=ReasoningIntent.SUMMARIZE_FILE,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.8,
    )
    helpers = RetrievalHelpers({})
    assert helpers.should_expand_summary_scope(
        query="파일 여러개 전부 읽어보고 핵심 요약해줘",
        parsed_intent=parsed,
    )
    helpers = RetrievalHelpers({})
    assert not helpers.should_expand_summary_scope(
        query="데통9주2차.txt 핵심만 요약해줘",
        parsed_intent=parsed,
    )

def test_reasoning_pipeline_week_exact_filter_matches_requested_week_only():
    doc_ids = {"doc7", "doc14"}
    metadata_map = {
        "doc7": {"path": "/tmp/데통7주차.txt"},
        "doc14": {"path": "/tmp/데통14주차.txt"},
    }
    helpers = RetrievalHelpers({})
    filtered_ids, filtered_map, applied, no_match = helpers.apply_week_exact_filter(
        doc_ids=doc_ids,
        metadata_map=metadata_map,
        requested_weeks=[7],
    )
    assert applied is True
    assert no_match is False
    assert filtered_ids == {"doc7"}
    assert set(filtered_map.keys()) == {"doc7"}

def test_reasoning_pipeline_week_exact_filter_returns_empty_when_not_found():
    doc_ids = {"doc10", "doc14"}
    metadata_map = {
        "doc10": {"path": "/tmp/데통10주차.txt"},
        "doc14": {"path": "/tmp/데통14주차.txt"},
    }
    helpers = RetrievalHelpers({})
    filtered_ids, filtered_map, applied, no_match = helpers.apply_week_exact_filter(
        doc_ids=doc_ids,
        metadata_map=metadata_map,
        requested_weeks=[7],
    )
    assert applied is True
    assert no_match is True
    assert filtered_ids == set()
    assert filtered_map == {}

def test_reasoning_pipeline_extract_requested_weeks_excludes_negative_week():
    helpers = RetrievalHelpers({})
    weeks = helpers.extract_requested_weeks(
        query="14주차 말고 7주차 파일 보여줘",
        followup_resolution=None,
    )
    assert weeks == [7]

def test_reasoning_pipeline_extract_requested_weeks_ignores_excluded_followup_week():
    helpers = RetrievalHelpers({})
    weeks = helpers.extract_requested_weeks(
        query="14주차 말고 7주차 파일 보여줘",
        followup_resolution=FollowUpResolution(
            is_followup=True,
            followup_type="refine_filter",
            resolved_filters={"week": 14},
        ),
    )
    assert weeks == [7]

def test_short_circuit_candidate_is_disabled_for_multi_file_summary_scope():
    helpers = RetrievalHelpers({})
    assert not helpers.should_short_circuit_candidate(
        mode=WorkMode.SUMMARY,
        top_score=0.11,
        intent=ReasoningIntent.SUMMARIZE_FILE,
        file_count=4,
        force_multi_file_summary=True,
    )

def test_short_circuit_candidate_is_disabled_for_focused_file_summary_scope():
    helpers = RetrievalHelpers({})
    assert not helpers.should_short_circuit_candidate(
        mode=WorkMode.GENERAL,
        top_score=0.09,
        intent=ReasoningIntent.SUMMARIZE_FILE,
        file_count=1,
        force_focused_file_summary=True,
    )

def test_path_focus_terms_detect_folder_query():
    terms, strict = RetrievalHelpers({}).extract_path_focus_terms(
        query="자료구조 폴더에 몇주차 자료까지 있어?",
        topics=["자료구조", "몇주차"],
    )
    assert strict is True
    assert "자료구조" in terms

def test_path_focus_filter_matches_korean_normalization():
    folder_ds = unicodedata.normalize("NFD", "자료구조")
    folder_web = unicodedata.normalize("NFD", "웹프")
    nfd_path = f"/Users/me/Desktop/{folder_ds}/언론.pdf"
    metadata = {
        "doc_ds": {"path": nfd_path},
        "doc_web": {"path": f"/Users/me/Desktop/{folder_web}/고장.pdf"},
    }
    filtered = ReasoningPipeline._filter_doc_ids_by_path_focus(
        doc_ids={"doc_ds", "doc_web"},
        metadata_map=metadata,
        focus_terms=["자료구조"],
    )
    assert filtered == {"doc_ds"}

def test_followup_resolver_refine_week_query():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(query="1주차?", mode=WorkMode.GENERAL, workspace=workspace)
    resolved = FollowUpResolver.resolve(
        query="1주차?",
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        last_context={"top_candidates": ["/tmp/workspace/a.md"]},
        last_candidates=["/tmp/workspace/a.md", "/tmp/workspace/b.md"],
        last_selected_file="/tmp/workspace/a.md",
        last_actions=["OPEN_FILE", "SUMMARIZE_TOP"],
    )
    assert resolved.is_followup is True
    assert resolved.followup_type == "refine_filter"
    assert resolved.resolved_intent == ReasoningIntent.FOLLOWUP_REFINE
    assert resolved.resolved_filters.get("week") == 1

def test_followup_resolver_refine_week_query_supports_higher_week_range():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(query="21주차?", mode=WorkMode.GENERAL, workspace=workspace)
    resolved = FollowUpResolver.resolve(
        query="21주차?",
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        last_context={"top_candidates": ["/tmp/workspace/a.md"]},
        last_candidates=["/tmp/workspace/a.md", "/tmp/workspace/b.md"],
        last_selected_file="/tmp/workspace/a.md",
        last_actions=["OPEN_FILE", "SUMMARIZE_TOP"],
    )
    assert resolved.is_followup is True
    assert resolved.followup_type == "refine_filter"
    assert resolved.resolved_intent == ReasoningIntent.FOLLOWUP_REFINE
    assert resolved.resolved_filters.get("week") == 21

def test_intent_parser_detects_followup_refine_for_higher_week_range():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="21주차만 다시 보여줘",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.intent == ReasoningIntent.FOLLOWUP_REFINE

def test_followup_resolver_lightweight_summary_query():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(query="요약만", mode=WorkMode.GENERAL, workspace=workspace)
    resolved = FollowUpResolver.resolve(
        query="요약만",
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        last_context={"top_candidates": ["/tmp/workspace/a.md"]},
        last_candidates=["/tmp/workspace/a.md"],
        last_selected_file="/tmp/workspace/a.md",
        last_actions=["OPEN_FILE"],
    )
    assert resolved.is_followup is True
    assert resolved.resolved_intent == ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST
    assert resolved.resolved_target_files == ["/tmp/workspace/a.md"]

def test_followup_resolver_does_not_override_general_chat_greeting():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(query="안녕", mode=WorkMode.GENERAL, workspace=workspace)
    assert parsed.intent == ReasoningIntent.GENERAL_CHAT

    resolved = FollowUpResolver.resolve(
        query="안녕",
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        last_context={"top_candidates": ["/tmp/workspace/a.md"]},
        last_candidates=["/tmp/workspace/a.md", "/tmp/workspace/b.md"],
        last_selected_file="/tmp/workspace/a.md",
        last_actions=["OPEN_FILE", "SUMMARIZE_TOP"],
    )
    assert resolved.is_followup is False

def test_followup_resolver_scope_all_target_clarification_is_one_shot():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(query="파일 전부 보여줘", mode=WorkMode.GENERAL, workspace=workspace)

    first = FollowUpResolver.resolve(
        query="파일 전부 보여줘",
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        last_context={},
        last_candidates=[],
        last_selected_file=None,
        last_actions=[],
    )
    assert first.is_followup is True
    assert first.followup_type == "clarify_scope_target"

    second = FollowUpResolver.resolve(
        query="파일 전부 보여줘",
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        last_context={"scope_clarification_pending": True},
        last_candidates=[],
        last_selected_file=None,
        last_actions=[],
    )
    assert second.is_followup is False

def test_followup_resolver_keeps_explicit_find_file_intent():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(query="파일 지금 뭐뭐있지", mode=WorkMode.GENERAL, workspace=workspace)
    assert parsed.intent == ReasoningIntent.FIND_FILE

    resolved = FollowUpResolver.resolve(
        query="파일 지금 뭐뭐있지",
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        last_context={"top_candidates": ["/tmp/workspace/a.md"]},
        last_candidates=["/tmp/workspace/a.md"],
        last_selected_file="/tmp/workspace/a.md",
        last_actions=["ASK_FOLLOWUP"],
    )
    assert resolved.is_followup is False

def test_intent_parser_routes_sleep_schedule_query_to_general_chat():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="그러면 몇 시쯤 자야되지? 보통 새벽 3-4시에 잤어",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    assert parsed.intent == ReasoningIntent.GENERAL_CHAT

def test_followup_resolver_does_not_force_followup_for_sleep_chat_query():
    parser = IntentParser()
    workspace = WorkspaceResponse(
        included_paths=["/tmp/workspace"],
        excluded_paths=[],
        startup_profile="RECOMMENDED",
        default_mode="GENERAL",
        updated_at=datetime.now(timezone.utc),
    )
    parsed = parser.parse(
        query="그러면 몇 시쯤 자야되지? 보통 새벽 3-4시에 잤어",
        mode=WorkMode.GENERAL,
        workspace=workspace,
    )
    resolved = FollowUpResolver.resolve(
        query="그러면 몇 시쯤 자야되지? 보통 새벽 3-4시에 잤어",
        parsed_intent=parsed,
        mode=WorkMode.GENERAL,
        last_context={"top_candidates": ["/tmp/workspace/데통10주1차.txt"]},
        last_candidates=["/tmp/workspace/데통10주1차.txt", "/tmp/workspace/데통11주2차.txt"],
        last_selected_file="/tmp/workspace/데통10주1차.txt",
        last_actions=["OPEN_FILE", "SUMMARIZE_TOP", "ASK_FOLLOWUP"],
    )
    assert resolved.is_followup is False

def test_general_chat_assist_mode_requires_explicit_retrieval_request():
    parsed = ParsedIntent(
        intent=ReasoningIntent.FIND_FILE,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.77,
        operation="find",
        target="데통",
        scope="all",
        ambiguity="clear",
    )
    mode, confidence, suppressed = ReasoningPipeline._general_chat_assist_mode(
        parsed_intent=parsed,
        query="요즘 좀 피곤하네",
    )
    assert mode == "none"
    assert confidence >= 0.7
    assert suppressed is True

def test_general_chat_assist_mode_allows_explicit_retrieval_request():
    parsed = ParsedIntent(
        intent=ReasoningIntent.FIND_FILE,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.8,
        operation="find",
        target="데통",
        scope="all",
        ambiguity="clear",
    )
    mode, _confidence, suppressed = ReasoningPipeline._general_chat_assist_mode(
        parsed_intent=parsed,
        query="데통 파일 전부 찾아줘",
    )
    assert mode == "light"
    assert suppressed is False

def test_explicit_retrieval_request_detects_exists_verb_with_target():
    assert ReasoningPipeline._has_explicit_retrieval_request("데통 파일 7주차 있나?")

def test_looks_general_chat_query_rejects_file_target_queries():
    assert not ReasoningPipeline._looks_general_chat_query("데통 파일 7주차 있나?")

def test_v2_followup_prompt_like_query_returns_natural_answer_not_candidate(client, auth_headers, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "dummy.md"
    source.write_text("dummy", encoding="utf-8")

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

    first = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "자료구조 파일 찾아줘",
            "mode": "GENERAL",
            "conversation_id": "conv-followup-natural",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "지금 질문을 더 정확하게 만들 수 있는 재질문 예시 3개를 만들어줘. (파일명/연도/태그 포함)",
            "mode": "GENERAL",
            "conversation_id": "conv-followup-natural",
        },
    )
    assert second.status_code == 200
    payload = second.json()
    assert payload["structured_result"]["result_type"] != "candidate"


def test_v2_short_followup_one_more_after_general_chat_stays_conversational(client, auth_headers):
    first = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "스위프트와 파이썬 변수 차이를 보여주는 예시 문제 하나 만들어줘",
            "mode": "GENERAL",
            "conversation_id": "conv-one-more-general",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/v2/chat/local",
        headers=auth_headers,
        json={
            "query": "하나 더 보여줘",
            "mode": "GENERAL",
            "conversation_id": "conv-one-more-general",
        },
    )
    assert second.status_code == 200
    payload = second.json()
    assert payload["structured_result"]["result_type"] != "candidate"
    lead = str(payload.get("lead") or "")
    summary = str(payload.get("result_summary") or "")
    merged = f"{lead}\n{summary}"
    assert "검색 결과가 명확하지 않습니다. 다음 옵션 중 선택해주세요." not in merged
