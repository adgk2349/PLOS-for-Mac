from __future__ import annotations

import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from local_ai_core.intent_parser import IntentParser
from local_ai_core.local_planner import LocalPlanner
from local_ai_core.reasoning_pipeline import ReasoningPipeline
from local_ai_core.followup_resolver import FollowUpResolver
from local_ai_core.clarification_budget import ClarificationBudget, ClarificationBudgetState
from local_ai_core.models import (
    Citation,
    ExecutionResult,
    ParsedEntities,
    ParsedIntent,
    ParsedTimeFilters,
    ParsedWorkspaceFilters,
    ReasoningIntent,
    SuggestedActionKind,
    VerificationResult,
    WorkMode,
    WorkspaceResponse,
)
from local_ai_core.verifier import ResultVerifier


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
    assert payload["lead"]
    assert "structured_result" in payload
    assert "summary" in payload["structured_result"]
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["actions"], list)
    assert "plan" in payload
    assert "verification" in payload


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


def test_path_focus_terms_detect_folder_query():
    terms, strict = ReasoningPipeline._extract_path_focus_terms(
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
    assert payload["structured_result"]["result_type"] in {"answer", "summary"}
