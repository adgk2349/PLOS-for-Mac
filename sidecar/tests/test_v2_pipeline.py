from __future__ import annotations

import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from local_ai_core.external_providers import ProviderResult
from local_ai_core.intent_parser import IntentParser
from local_ai_core.local_planner import LocalPlanner
from local_ai_core.reasoning_pipeline import ReasoningPipeline
from local_ai_core.followup_resolver import FollowUpResolution, FollowUpResolver
from local_ai_core.clarification_budget import ClarificationBudget, ClarificationBudgetState
from local_ai_core.models import (
    Citation,
    ExecutionResult,
    ParsedEntities,
    ParsedIntent,
    ParsedTimeFilters,
    ParsedWorkspaceFilters,
    LocalEngine,
    PrivacyMode,
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
    assert ReasoningPipeline._should_expand_summary_scope(
        query="파일 여러개 전부 읽어보고 핵심 요약해줘",
        parsed_intent=parsed,
    )
    assert not ReasoningPipeline._should_expand_summary_scope(
        query="데통9주2차.txt 핵심만 요약해줘",
        parsed_intent=parsed,
    )


def test_reasoning_pipeline_model_size_parser():
    assert ReasoningPipeline._model_size_b("Qwen3-8B-Q4_K_M.gguf") == 8
    assert ReasoningPipeline._model_size_b("mlx-community/Llama-3.1-8B-Instruct-4bit") == 8
    assert ReasoningPipeline._model_size_b("Llama-3.3-70B-Instruct-Q4_K_M.gguf") == 70
    assert ReasoningPipeline._model_size_b("unknown-model-name") is None


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
    assert ReasoningPipeline._is_16gb_tier_model(settings_small) is True
    assert ReasoningPipeline._is_16gb_tier_model(settings_large) is False


def test_reasoning_pipeline_extracts_explicit_file_term_from_query():
    parsed = ParsedIntent(
        intent=ReasoningIntent.SUMMARIZE_FILE,
        entities=ParsedEntities(file_names=[]),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.8,
    )
    terms = ReasoningPipeline._extract_explicit_file_terms(
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
    filtered_ids, filtered_map, used = ReasoningPipeline._apply_explicit_file_focus(
        doc_ids=doc_ids,
        metadata_map=metadata_map,
        file_terms=["데통10주1차.txt"],
    )
    assert used is True
    assert filtered_ids == {"doc10"}
    assert set(filtered_map.keys()) == {"doc10"}


def test_reasoning_pipeline_week_exact_filter_matches_requested_week_only():
    doc_ids = {"doc7", "doc14"}
    metadata_map = {
        "doc7": {"path": "/tmp/데통7주차.txt"},
        "doc14": {"path": "/tmp/데통14주차.txt"},
    }
    filtered_ids, filtered_map, applied, no_match = ReasoningPipeline._apply_week_exact_filter(
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
    filtered_ids, filtered_map, applied, no_match = ReasoningPipeline._apply_week_exact_filter(
        doc_ids=doc_ids,
        metadata_map=metadata_map,
        requested_weeks=[7],
    )
    assert applied is True
    assert no_match is True
    assert filtered_ids == set()
    assert filtered_map == {}


def test_reasoning_pipeline_extract_requested_weeks_excludes_negative_week():
    weeks = ReasoningPipeline._extract_requested_weeks(
        query="14주차 말고 7주차 파일 보여줘",
        followup_resolution=None,
    )
    assert weeks == [7]


def test_reasoning_pipeline_extract_requested_weeks_ignores_excluded_followup_week():
    weeks = ReasoningPipeline._extract_requested_weeks(
        query="14주차 말고 7주차 파일 보여줘",
        followup_resolution=FollowUpResolution(
            is_followup=True,
            followup_type="refine_filter",
            resolved_filters={"week": 14},
        ),
    )
    assert weeks == [7]


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


def test_short_circuit_candidate_is_disabled_for_multi_file_summary_scope():
    assert not ReasoningPipeline._should_short_circuit_candidate(
        mode=WorkMode.SUMMARY,
        top_score=0.11,
        intent=ReasoningIntent.SUMMARIZE_FILE,
        file_count=4,
        force_multi_file_summary=True,
    )


def test_short_circuit_candidate_is_disabled_for_focused_file_summary_scope():
    assert not ReasoningPipeline._should_short_circuit_candidate(
        mode=WorkMode.GENERAL,
        top_score=0.09,
        intent=ReasoningIntent.SUMMARIZE_FILE,
        file_count=1,
        force_focused_file_summary=True,
    )


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

    def _fake_analyze_sync(*, provider, query, mode, citations, language_preference=None):
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
    assert payload["is_local"] is False
    assert payload["metadata"]["conversation_path"] == "external_summary_escalated"
    assert payload["metadata"]["escalated_provider"] == "openai"
    assert payload["structured_result"]["summary"].startswith("1. ")


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


def test_reasoning_pipeline_detects_instructional_leak_tokens():
    assert ReasoningPipeline._looks_like_reasoning_leak(
        "사용자에게 물어볼 때는 반드시 '?'를 붙여주세요. 최종 답변: 현재 시간은 10시 30분입니다."
    )


def test_reasoning_pipeline_detects_three_sentence_rule_leak():
    assert ReasoningPipeline._looks_like_reasoning_leak(
        "단, 사용자의 질문에 대한 명확한 답변이 필요할 경우 3문장까지 가능합니다."
    )


def test_reasoning_pipeline_detects_insufficient_answer_rule_leak():
    assert ReasoningPipeline._looks_like_reasoning_leak(
        "단, 사용자의 질문에 대한 답변이 부족할 경우 추가적인 질문을 덧붙일 수 있습니다."
    )


def test_reasoning_pipeline_detects_help_user_directly_leak():
    assert ReasoningPipeline._looks_like_reasoning_leak("사용자에게 직접 도움을 주세요.")


def test_reasoning_pipeline_detects_react_immediately_rule_leak():
    assert ReasoningPipeline._looks_like_reasoning_leak("사용자의 말에 바로 반응하세요.")


def test_reasoning_pipeline_detects_user_message_rule_leak():
    assert ReasoningPipeline._looks_like_reasoning_leak(
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
    assert ReasoningPipeline._is_brief_chat_query("그렇구나!")
    assert not ReasoningPipeline._is_brief_chat_query("운동 후 목이 시린 이유를 단계별로 설명해줘")


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
