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
from local_ai_core.chat import ChatService

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

def test_v2_chat_stream_endpoint_returns_done_event(monkeypatch):
    async def _fake_run(self, req):
        return ComposedChatResponseV2(
            lead="테스트 응답",
            structured_result=StructuredResult(
                result_type="conversation",
                summary="테스트 응답",
                details=[],
                data={},
            ),
            execution_result=ExecutionResult(
                result_type="conversation",
                structured_payload={},
                citations=[],
                tool_logs=[],
                generated_text="테스트 응답",
                engine_used=None,
                used_fallback=False,
                runtime_detail="stream_test",
            ),
            generated_text="테스트 응답",
            citations=[],
            actions=[],
            prompt_cache_hit=False,
            metadata={
                "conversation_path": "local_conversation",
                "trace_events": [
                    {"status": "planning", "message": "planning", "source": "pipeline"},
                    {"status": "done", "message": "done", "source": "pipeline"},
                ],
            },
            parsed_intent=ParsedIntent(intent=ReasoningIntent.GENERAL_CHAT),
            plan=LocalPlan(plan_type="conversation"),
            verification=VerificationResult(is_valid=True, confidence=0.9),
            mode=WorkMode.GENERAL,
            used_profile=StartupProfile.RECOMMENDED,
            is_local=True,
            engine_used=None,
            used_fallback=False,
            runtime_detail="stream_test",
        )

    monkeypatch.setattr(ReasoningPipeline, "run", _fake_run)

    pipeline = ReasoningPipeline()

    async def _collect_events():
        events = []
        async for line in pipeline.run_stream(
            SimpleNamespace(
                query="안녕",
                mode=WorkMode.GENERAL,
                conversation_id="stream-room",
                session_id="stream-room",
            )
        ):
            events.append(json.loads(line))
        return events

    events = asyncio.run(_collect_events())

    assert [event["type"] for event in events] == ["status", "status", "chunk", "done"]
    assert events[-1]["result"]["generated_text"] == "테스트 응답"

def test_v2_chat_stream_splits_chunks_by_sentence(monkeypatch):
    async def _fake_run(self, req):
        return ComposedChatResponseV2(
            lead="테스트",
            structured_result=StructuredResult(
                result_type="conversation",
                summary="첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
                details=[],
                data={},
            ),
            execution_result=ExecutionResult(
                result_type="conversation",
                structured_payload={},
                citations=[],
                tool_logs=[],
                generated_text="첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
                engine_used=None,
                used_fallback=False,
                runtime_detail="stream_chunk_test",
            ),
            generated_text="첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
            citations=[],
            actions=[],
            prompt_cache_hit=False,
            metadata={"conversation_path": "local_conversation"},
            parsed_intent=ParsedIntent(intent=ReasoningIntent.GENERAL_CHAT),
            plan=LocalPlan(plan_type="conversation"),
            verification=VerificationResult(is_valid=True, confidence=0.9),
            mode=WorkMode.GENERAL,
            used_profile=StartupProfile.RECOMMENDED,
            is_local=True,
            engine_used=None,
            used_fallback=False,
            runtime_detail="stream_chunk_test",
        )

    monkeypatch.setattr(ReasoningPipeline, "run", _fake_run)
    pipeline = ReasoningPipeline()

    async def _collect_events():
        events = []
        async for line in pipeline.run_stream(
            SimpleNamespace(
                query="안녕",
                mode=WorkMode.GENERAL,
                conversation_id="stream-room-2",
                session_id="stream-room-2",
            )
        ):
            events.append(json.loads(line))
        return events

    events = asyncio.run(_collect_events())
    chunk_texts = [item["text"] for item in events if item.get("type") == "chunk"]
    assert len(chunk_texts) == 3
    assert chunk_texts[0] == "첫 문장입니다."
    assert chunk_texts[1] == "둘째 문장입니다."
    assert chunk_texts[2] == "셋째 문장입니다."


def test_chat_service_stream_hides_room_scope_missing_status():
    service = ChatService.__new__(ChatService)
    service._room_registry = None

    async def _source():
        yield json.dumps({"type": "chunk", "text": "테스트"}) + "\n"
        yield json.dumps({"type": "done", "result": {"generated_text": "테스트", "metadata": {}}}) + "\n"

    class _Pipeline:
        def run_stream(self, req):
            return _source()

    service._pipeline = _Pipeline()
    service._resolve_v2_chat_target = lambda req: (  # type: ignore[attr-defined]
        service,
        {"memory_backend": "global", "room_route_reason": "room_scope_missing"},
    )

    async def _collect():
        events = []
        async for line in service.local_chat_v2_stream(
            SimpleNamespace(
                query="안녕",
                mode=WorkMode.GENERAL,
                conversation_id="no-scope-room",
                session_id="no-scope-room",
            )
        ):
            events.append(json.loads(line))
        return events

    events = asyncio.run(_collect())
    status_messages = [str(item.get("message") or "") for item in events if item.get("type") == "status"]
    assert all("room routing:" not in msg for msg in status_messages)
    assert any(item.get("type") == "done" for item in events)
