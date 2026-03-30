from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from local_ai_core.models import (
    ComposedChatResponseV2,
    ExecutionResult,
    LocalChatRequestV2,
    LocalPlan,
    ParsedIntent,
    ReasoningIntent,
    RelevantMemoryBundle,
    SettingsModel,
    StructuredResult,
    VerificationResult,
    WorkMode,
    WorkspaceIdentity,
    WorkspaceResponse,
)
from local_ai_core.nlu.followup_resolver import FollowUpResolution
from local_ai_core.reasoning.pipeline import ReasoningPipeline


class _DBStub:
    def get_workspace(self) -> WorkspaceResponse:
        return WorkspaceResponse(
            included_paths=[],
            excluded_paths=[],
            startup_profile="RECOMMENDED",
            default_mode="GENERAL",
            updated_at=datetime.now(timezone.utc),
        )

    def get_settings(self) -> SettingsModel:
        return SettingsModel(language="ko")


class _MemoryStub:
    def __init__(self) -> None:
        self.last_followup_context = {
            "last_user_query": "이전 질문",
            "result_summary": "이전 답변",
            "conversation_path": "local_conversation",
        }
        self.digest_updated = False

    def get_workspace_identity(self) -> WorkspaceIdentity:
        return WorkspaceIdentity(workspace_id="ws", included_paths_hash="h", version=1)

    def get_last_conversational_context(self, session_id: str):
        return dict(self.last_followup_context)

    def get_last_candidate_set(self, session_id: str):
        return ["/tmp/a.md", "/tmp/b.md"]

    def get_last_selected_file(self, session_id: str):
        return "/tmp/a.md"

    def get_last_shown_actions(self, session_id: str):
        return ["OPEN_FILE", "ASK_FOLLOWUP"]

    def get_session_digest(self, session_id: str):
        return {
            "active_topics": ["topic-a"],
            "stable_facts": ["fact-a"],
            "open_loops": [],
            "recent_turns": [{"role": "user", "text": "이전 질문"}],
            "turn_count": 3,
        }

    def get_relevant_memory_bundle(self, *, session_id, workspace_id, intent, related_file_ids, query):
        return RelevantMemoryBundle(workspace_identity=self.get_workspace_identity())

    def resolve_preferences(self, memory_bundle):
        return None

    def update_session_digest(self, session_id: str, user_query: str, assistant_summary: str, mode: str = "hybrid"):
        self.digest_updated = True
        return {"turn_count": 4, "digest_refresh": "rule"}

    def write_conversational_context(self, *, session_id: str, context: dict):
        return None


class _FollowupStub:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def resolve(self, **kwargs):
        self.kwargs = dict(kwargs)
        return FollowUpResolution()


class _ExecutorStub:
    def __init__(self) -> None:
        self.session_summaries: list[str] = []

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
        self.session_summaries.append(str(session_summary or ""))
        return ExecutionResult(
            result_type="conversation",
            structured_payload={},
            citations=[],
            tool_logs=[],
            generated_text="ok",
            engine_used=None,
            used_fallback=False,
            runtime_detail="ok",
        )


class _ComposerStub:
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
            lead="ok",
            structured_result=StructuredResult(result_type="conversation", summary="ok", details=[], data={}),
            execution_result=execution_result,
            generated_text=execution_result.generated_text,
            citations=[],
            actions=[],
            prompt_cache_hit=False,
            metadata={"conversation_path": conversation_path},
            parsed_intent=parsed_intent,
            plan=plan if isinstance(plan, LocalPlan) else LocalPlan(plan_type="conversation"),
            verification=verification if isinstance(verification, VerificationResult) else VerificationResult(is_valid=True),
            mode=mode,
            used_profile=used_profile,
            is_local=is_local,
            engine_used=engine_used,
            used_fallback=used_fallback,
            runtime_detail=runtime_detail,
        )


def test_pipeline_loads_previous_context_before_followup_and_injects_digest_summary():
    memory = _MemoryStub()
    followup = _FollowupStub()
    executor = _ExecutorStub()

    pipeline = ReasoningPipeline(
        db=_DBStub(),
        memory=memory,
        memory_service=memory,
        executor=executor,
        composer=_ComposerStub(),
    )
    pipeline.followup_resolver = followup
    pipeline.intent_parser.parse = lambda query, mode, workspace: ParsedIntent(intent=ReasoningIntent.GENERAL_CHAT)

    req = LocalChatRequestV2(
        query="이어서 설명해줘",
        mode=WorkMode.GENERAL,
        conversation_id="sess-ctx",
        session_id="sess-ctx",
    )
    composed = asyncio.run(pipeline.run(req))

    assert followup.kwargs["last_context"]["last_user_query"] == "이전 질문"
    assert followup.kwargs["last_candidates"] == ["/tmp/a.md", "/tmp/b.md"]
    assert followup.kwargs["last_selected_file"] == "/tmp/a.md"
    assert followup.kwargs["last_actions"] == ["OPEN_FILE", "ASK_FOLLOWUP"]

    assert executor.session_summaries
    injected = executor.session_summaries[0]
    assert "topics: topic-a" in injected
    assert "last_query: 이전 질문" in injected

    assert composed.metadata["context_digest_used"] is True
    assert composed.metadata["context_injected"] is True
    assert composed.metadata["digest_turn_count"] == 4
    assert memory.digest_updated is True
