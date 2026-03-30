from __future__ import annotations
import asyncio
import logging
logger = logging.getLogger(__name__)
import os
from typing import Any
from pathlib import Path

from ... import utils

from ....nlu.followup_resolver import FollowUpResolution, FollowUpResolver
from ....models import *
from ....retrieval import extract_query_hints, merge_filters, retrieve_bundle
from .core_chat_post_helpers import CoreChatPostHelpers


class CoreChatHelpers(CoreChatPostHelpers):
    def __init__(self, dependencies: dict[str, Any]):
        self._db = dependencies.get('db')
        self._memory = dependencies.get('memory')
        self._embedding = dependencies.get('embedding_service')
        self._vector_store = dependencies.get('vector_store')
        self._composer = dependencies.get('composer')
        self._executor = dependencies.get('executor')
        self._intent_parser = dependencies.get('intent_parser')
        self._followup = dependencies.get('followup_resolver')
        self._reranker = getattr(dependencies.get('embedding_service'), '_reranker', None)
        self._clarification_budget = dependencies.get('clarification_budget')
        self._capabilities = dependencies.get('capabilities')

    @staticmethod
    def _is_detailed_explanation_requested(query: str) -> bool:
        lowered = utils._normalized_match_text(query)
        if not lowered:
            return False
        detailed_cues = (
            "자세히", "상세히", "더 길게", "분석", "심층", "상세하게", "설명해줘", "알려줘",
            "자세한", "설명", "길게", "분석해", "상세", "제한없이", "끝까지", "다써줘", "끊지마", "전부", "모두",
            "detail", "analyze", "explain", "deeply", "long", "comprehensive", "full", "complete"
        )
        return any(cue in lowered for cue in detailed_cues)

    def _effective_query(
        self,
        *,
        session_id: str,
        query: str,
        parsed_intent,
        memory_bundle,
        followup_resolution: FollowUpResolution | None = None,
    ) -> str:
        if parsed_intent.intent not in {
            ReasoningIntent.FOLLOWUP_QUESTION,
            ReasoningIntent.FOLLOWUP_REFINE,
            ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
            ReasoningIntent.SOFT_CONFIRM,
            ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
            ReasoningIntent.NEXT_CANDIDATE,
            ReasoningIntent.REDUCE_SCOPE,
            ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
        }:
            if followup_resolution and followup_resolution.query_hint:
                return f"{query}\n    \n    Constraint: {followup_resolution.query_hint}"
            return query

        recent_queries: list[str] = []
        for item in memory_bundle.session_items:
            if item.key != "recent_query":
                continue
            text = str(item.value_json.get("summary") or "").strip()
            if text:
                recent_queries.append(text)
            if len(recent_queries) >= 3:
                break
        if not recent_queries:
            if followup_resolution and followup_resolution.query_hint:
                return f"{query}\n    \n    Follow-up hint: {followup_resolution.query_hint}"
            return query
        seed = recent_queries[0]
        joined = f"{seed}\n    \n    {query}"
        if followup_resolution and followup_resolution.query_hint:
            joined += f"\n    \n    Hint: {followup_resolution.query_hint}"
        return joined

    @staticmethod
    def _candidate_gap_small(citations: list[Citation]) -> bool:
        if len(citations) < 2:
            return False
        first = float(citations[0].score)
        second = float(citations[1].score)
        return abs(first - second) <= 0.04

    @staticmethod
    def _should_short_circuit_candidate(
        *,
        mode: WorkMode,
        top_score: float,
        intent: ReasoningIntent,
        file_count: int,
        force_multi_file_summary: bool = False,
        force_focused_file_summary: bool = False,
    ) -> bool:
        if mode == WorkMode.STRICT_SEARCH:
            return top_score < 0.6
        if intent == ReasoningIntent.SUMMARIZE_FILE and force_focused_file_summary:
            return file_count <= 0
        if intent == ReasoningIntent.SUMMARIZE_FILE and force_multi_file_summary:
            return file_count <= 0
        if intent in {
            ReasoningIntent.FOLLOWUP_QUESTION,
            ReasoningIntent.FOLLOWUP_REFINE,
            ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
            ReasoningIntent.SOFT_CONFIRM,
            ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
            ReasoningIntent.NEXT_CANDIDATE,
            ReasoningIntent.REDUCE_SCOPE,
            ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
            ReasoningIntent.OPEN_FILE,
        }:
            return False
        if intent == ReasoningIntent.FIND_FILE:
            return file_count <= 0
        if file_count <= 0:
            return True
        return top_score < 0.26

    def _build_candidate_execution(*, response_language: str, citations: list[Citation], reason: str) -> ExecutionResult:
        items = []
        for citation in citations[:4]:
            items.append(
                {
                    "doc_id": citation.doc_id,
                    "file_path": citation.file_path,
                    "score": round(citation.score, 3),
                }
            )
        if response_language == "ko":
            text = "완전히 확실하진 않지만, 지금은 이쪽 후보가 가장 유력해 보여요. 파일명/기간/태그를 조금만 더 주시면 정확도를 더 올릴 수 있어요."
        else:
            text = "Evidence confidence was low, so I returned candidates instead of a final answer. Please retry with file name, year, or tag."
        return ExecutionResult(
            result_type="candidate",
            structured_payload={"reason": reason, "items": items},
            citations=citations[:6],
            tool_logs=[f"candidate:{reason}"],
            generated_text=text,
            engine_used=None,
            used_fallback=False,
            runtime_detail=None,
        )

    def _outcome_event_type(result_type: str) -> MemoryEventType | None:
        if result_type == "comparison":
            return MemoryEventType.COMPARISON
        if result_type == "draft":
            return MemoryEventType.DRAFT_CREATED
        if result_type in {"answer", "summary"}:
            return MemoryEventType.SUMMARY_CREATED
        return None

    def _should_run_secondary_reasoner(
        *,
        mode: WorkMode,
        parsed_intent: ParsedIntent,
        execution: ExecutionResult,
        verification: VerificationResult,
    ) -> bool:
        if mode == WorkMode.STRICT_SEARCH:
            return False
        if execution.result_type in {"file_list", "classification", "insufficient"}:
            return False
        # Phase 2: Disable secondary reasoner for common chats to avoid multi-pass latency
        if execution.result_type == "conversation":
            return False
        if parsed_intent.intent == ReasoningIntent.FIND_FILE:
            return False
        if parsed_intent.intent in {
            ReasoningIntent.FOLLOWUP_REFINE,
            ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
            ReasoningIntent.SOFT_CONFIRM,
            ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
            ReasoningIntent.NEXT_CANDIDATE,
            ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
            ReasoningIntent.OPEN_FILE,
        }:
            return False
        issue_set = set(verification.issues)
        if {"repetitive_output", "raw_dump_output", "intent_mismatch"}.intersection(issue_set):
            return True
        if verification.candidate_mode or verification.confidence < 0.52:
            return True
        return False

    def _build_secondary_reasoner_prompt(*, query: str, parsed_intent: ParsedIntent, response_language: str) -> str:
        if response_language == "ko":
            intent_hint = parsed_intent.intent.value
            return (
                f"{query}\n    \n    "
                "추가 지시:\n    "
                f"- intent={intent_hint}에 맞춰 답변합니다.\n    "
                "- 코드 원문/문장 반복을 금지합니다.\n    "
                "- 근거 기반 핵심만 간결하게 정리합니다.\n    "
                "- 불확실하면 마지막에 확인 질문 1개를 추가합니다."
            )
        return (
            f"{query}\n    \n    "
            "Additional constraints:\n    "
            f"- Keep alignment with intent={parsed_intent.intent.value}.\n    "
            "- Avoid repetitive output and raw code dump.\n    "
            "- Keep concise, grounded summary.\n    "
            "- If uncertain, end with one clarifying question."
        )

    def _pick_better_reasoner_result(
        *,
        primary_execution: ExecutionResult,
        primary_verification: VerificationResult,
        secondary_execution: ExecutionResult,
        secondary_verification: VerificationResult,
    ) -> tuple[ExecutionResult, VerificationResult]:
        primary_score = float(primary_verification.confidence)
        secondary_score = float(secondary_verification.confidence)
        primary_issues = set(primary_verification.issues)
        secondary_issues = set(secondary_verification.issues)

        primary_bad = {"repetitive_output", "raw_dump_output"}.intersection(primary_issues)
        secondary_bad = {"repetitive_output", "raw_dump_output"}.intersection(secondary_issues)

        choose_secondary = False
        if secondary_score >= primary_score + 0.05:
            choose_secondary = True
        elif primary_bad and not secondary_bad:
            choose_secondary = True
        elif not secondary_verification.candidate_mode and primary_verification.candidate_mode:
            choose_secondary = True

        if choose_secondary and secondary_execution.generated_text.strip():
            secondary_execution.tool_logs.extend(primary_execution.tool_logs)
            return secondary_execution, secondary_verification
        return primary_execution, primary_verification

    async def _run_general_chat(
        self,
        *,
        req: LocalChatRequestV2,
        settings,
        workspace,
        session_id: str,
        workspace_id: str,
        response_language: str,
        parsed_intent: ParsedIntent,
        behavior_policy: BehaviorPolicy,
        memory_prefs,
        memory_bundle,
        last_context: dict | None,
        session_digest: dict | None,
        force_web_search: bool = False,
    ) -> ComposedChatResponseV2:
        plan = LocalPlan(
            plan_type="conversation",
            selected_files=[],
            selected_chunks=[],
            response_strategy="conversational_assistant",
            allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
            external_reasoning_needed=False,
        )
        assist_mode, assist_confidence, assistive_retrieval_suppressed = self._general_chat_assist_mode(
            parsed_intent=parsed_intent,
            query=req.query,
        )
        if assist_mode == "clarify":
            return self._compose_scope_target_clarification(
                req=req,
                response_language=response_language,
                parsed_intent=parsed_intent,
                behavior_policy=behavior_policy,
                memory_prefs=memory_prefs,
                workspace=workspace,
                session_id=session_id,
                workspace_id=workspace_id,
                conversation_path="general_chat_assist_clarification",
            )
        is_followup_web_search = self._is_followup_web_search_request(
            query=req.query,
            last_context=last_context,
        )
        explicit_web_search_request = self._is_explicit_web_search_request(req.query) or is_followup_web_search
        auto_web_search_request = (
            not explicit_web_search_request
            and self._should_auto_web_search(
                query=req.query,
                parsed_intent=parsed_intent,
                last_context=last_context,
            )
        )
        web_search_requested = explicit_web_search_request or auto_web_search_request or force_web_search
        effective_response_length = self._adaptive_response_length_for_query(
            query=req.query,
            base_response_length=memory_prefs.response_length,
            explicit_web_search_request=web_search_requested,
            last_context=last_context,
        )
        context_summary = ""
        should_apply_context = self._should_apply_conversation_context(
            req.query,
            has_session_digest=bool(session_digest),
            has_last_context=bool(last_context),
            session_digest=session_digest,
            last_context=last_context,
        )
        if should_apply_context:
            context_summary = self._conversation_session_summary(
                query=req.query,
                session_digest=session_digest,
                last_context=last_context,
                memory_bundle=memory_bundle,
                response_length=effective_response_length,
                model_profile=str(getattr(settings, "model_profile", "recommended") or "recommended"),
            )
        digest_context_used = bool(context_summary.strip())
        execution: ExecutionResult | None = None
        conversation_path = "local_conversation"
        escalated_provider: str | None = None
        is_local = True
        brief_chat = self._is_brief_chat_query(req.query)
        hybrid_external_enabled = settings.privacy_mode == PrivacyMode.HYBRID and bool(
            getattr(settings, "hybrid_web_search_enabled", False)
        )

        web_report: dict[str, Any] = {}
        if web_search_requested:
            self._last_web_report = {}
            permission_reason = self._external_web_search_permission_reason(settings)
            if permission_reason is None:
                direct_execution = await self._execute_direct_web_search(
                    query=self._web_search_query_for_turn(
                        query=req.query,
                        last_context=last_context,
                        is_followup_web_search=is_followup_web_search,
                    ),
                    mode=req.mode,
                    response_language=response_language,
                    workspace=workspace,
                    settings=settings,
                    response_length=effective_response_length,
                )
                if direct_execution is not None:
                    execution = direct_execution
                    conversation_path = "external_web_search_direct"
                    is_local = False
                    if auto_web_search_request:
                        execution.tool_logs.insert(0, "web_search:auto_triggered")
                else:
                    web_report = dict(getattr(self, "_last_web_report", {}) or {})
                    if explicit_web_search_request:
                        execution = self._web_search_blocked_execution(
                            response_language=response_language,
                            reason="provider_unavailable",
                            additional_tool_logs=list(web_report.get("logs") or []),
                            web_sources_count=int(web_report.get("sources_count") or 0),
                            web_fetch_failures=int(web_report.get("fetch_failures") or 0),
                        )
                        execution.structured_payload["web_path"] = "unavailable"
                        conversation_path = "external_web_search_unavailable"
                    else:
                        web_report.setdefault("logs", []).append("web_search:auto_unavailable")
                        web_report.setdefault("failure_reason", "provider_unavailable")
            else:
                if explicit_web_search_request:
                    execution = self._web_search_blocked_execution(
                        response_language=response_language,
                        reason=permission_reason,
                    )
                    conversation_path = "external_web_search_blocked"
                else:
                    web_report = {
                        "logs": [f"web_search:auto_suppressed:{permission_reason}"],
                        "sources_count": 0,
                        "fetch_failures": 0,
                        "discovered_count": 0,
                        "failure_reason": permission_reason,
                    }

        if execution is None:
            execution = await self._executor.execute_conversation_async(
                query=req.query,
                mode=req.mode,
                startup_profile=workspace.startup_profile,
                engine=settings.local_engine,
                mlx_model_path=settings.mlx_model_path,
                llama_model_path=settings.llama_model_path,
                language_preference=settings.language,
                session_summary=context_summary,
                max_tokens=self._conversation_max_tokens(
                    effective_response_length,
                    model_profile=getattr(settings, "model_profile", "recommended"),
                    query=req.query,
                ),
                timeout_seconds=float(os.getenv("LOCAL_AI_INFERENCE_TIMEOUT_SECONDS", "40")),
            )
            execution.tool_logs.insert(0, f"router:intent={ReasoningIntent.GENERAL_CHAT.value}")
            if auto_web_search_request and web_report:
                trace_logs = list(web_report.get("logs") or [])
                execution.tool_logs = [*trace_logs, *execution.tool_logs]
            execution.tool_logs.append("agent:conversation_assistant")

        needs_recovery = (not execution.generated_text) or (execution.used_fallback and not brief_chat)
        if needs_recovery and execution.used_fallback:
            execution.tool_logs.append("conversation_retry_exhausted")

        if not execution.generated_text:
            execution = self._runtime_error_execution(response_language, execution.runtime_detail)
            if hybrid_external_enabled:
                execution.tool_logs.append("runtime_error:conversation_local_and_external_failed")
            else:
                execution.tool_logs.append("runtime_error:conversation_local_failed_local_only")
        should_attempt_conversation_repair = False # Phase 2 Bypass
        if should_attempt_conversation_repair:
            repaired = await self._repair_repetitive_conversation_response(
                query=req.query,
                execution=execution,
                session_digest=session_digest,
                last_context=last_context,
                response_language=response_language,
                mode=req.mode,
                workspace=workspace,
                settings=settings,
                response_length=effective_response_length,
            )
            if repaired is not None:
                execution = repaired
                execution.tool_logs.append("conversation_repair:anti_repeat")
                conversation_path = "local_conversation_repaired"

        assist_citations: list[Citation] = []
        if assist_mode == "light" and execution.result_type == "conversation" and not web_search_requested:
            assist_citations = self._assistive_retrieval_citations(
                query=req.query,
                workspace=workspace,
                parsed_intent=parsed_intent,
                user_filters=req.filters,
            )
            if assist_citations:
                execution.generated_text = self._append_source_line(
                    text=execution.generated_text,
                    citations=assist_citations,
                    response_language=response_language,
                )
                execution.citations = assist_citations
                execution.structured_payload["assist_mode"] = "light"
                execution.structured_payload["assist_confidence"] = round(float(assist_confidence), 3)
                execution.tool_logs.append("assistive_retrieval:light")
                conversation_path = "local_conversation_assist"

        verification = VerificationResult(
            is_valid=(execution.result_type != "runtime_error"),
            confidence=(0.84 if execution.result_type == "conversation" else 0.2),
            issues=([] if execution.result_type == "conversation" else ["runtime_unavailable"]),
            ambiguity_level=(0.16 if execution.result_type == "conversation" else 0.8),
            candidate_mode=False,
        )

        await asyncio.to_thread(
            self._memory.write_memory_event,
            MemoryEventRequest(
                event_type=MemoryEventType.QUERY,
                session_id=session_id,
                workspace_id=workspace_id,
                summary=req.query[:220],
                related_file_ids=[],
                metadata_json={
                    "mode": req.mode.value,
                    "intent": ReasoningIntent.GENERAL_CHAT.value,
                    "result_type": execution.result_type,
                    "conversation_path": conversation_path,
                },
                importance=0.32,
            ),
        )

        composed = self._composer.compose_v2(
            query=req.query,
            mode=req.mode,
            response_language=response_language,
            parsed_intent=parsed_intent,
            plan=plan,
            execution_result=execution,
            verification=verification,
            behavior_policy=behavior_policy,
            response_length=effective_response_length,
            show_citations=False,
            prefer_action_suggestions=memory_prefs.prefer_action_suggestions,
            used_profile=workspace.startup_profile,
            engine_used=execution.engine_used,
            used_fallback=execution.used_fallback,
            runtime_detail=execution.runtime_detail,
            followup_resolution=None,
            allow_clarification=False,
            conversation_path=conversation_path,
            escalated_provider=escalated_provider,
            is_local=is_local,
        )
        if assist_mode == "light":
            composed.metadata["assist_mode"] = "light"
            composed.metadata["assist_confidence"] = round(float(assist_confidence), 3)
        composed.metadata["assistive_retrieval_suppressed"] = bool(assistive_retrieval_suppressed)
        quality_debug = self._conversation_quality_debug_from_detail(execution.runtime_detail)
        composed.metadata["korean_rewrite_used"] = bool(quality_debug.get("korean_rewrite_used", False))
        composed.metadata["quality_repair_reason"] = str(quality_debug.get("quality_repair_reason") or "")
        composed.metadata["repair_triggered"] = bool(quality_debug.get("repair_triggered", False))
        composed.metadata["repair_success"] = bool(quality_debug.get("repair_success", False))
        composed.metadata["leak_blocked"] = bool(quality_debug.get("leak_blocked", False))
        composed.metadata["direct_first_applied"] = bool(quality_debug.get("direct_first_applied", False))
        composed.metadata["question_count_after_postprocess"] = int(
            quality_debug.get("question_count_after_postprocess", 0) or 0
        )
        composed.metadata["recommendation_shape"] = str(quality_debug.get("recommendation_shape") or "")
        composed.metadata["effective_response_length"] = str(effective_response_length)
        composed.metadata["conversation_token_budget"] = int(
            self._conversation_max_tokens(
                effective_response_length,
                model_profile=getattr(settings, "model_profile", "recommended"),
                query=req.query,
            )
        )
        if web_search_requested:
            web_path = ""
            if conversation_path == "external_web_search_direct":
                web_path = "direct"
            elif conversation_path == "external_web_search_requested":
                web_path = "fallback_provider"
            elif conversation_path == "external_web_search_blocked":
                web_path = "blocked"
            elif conversation_path == "external_web_search_unavailable":
                web_path = "unavailable"
            if web_path:
                composed.metadata["web_path"] = web_path
                composed.metadata["web_sources_count"] = int(execution.structured_payload.get("web_sources_count") or 0)
                composed.metadata["web_fetch_failures"] = int(
                    execution.structured_payload.get("web_fetch_failures") or 0
                )
        if auto_web_search_request:
            composed.metadata["web_auto_triggered"] = True
            if web_report:
                composed.metadata["web_auto_failure_reason"] = str(web_report.get("failure_reason") or "")
        composed.metadata["context_injected"] = bool(digest_context_used)
        quality_rollup = self._record_conversation_quality_event(
            session_id=session_id,
            query=req.query,
            execution=execution,
            metadata=composed.metadata,
        )
        if quality_rollup:
            composed.metadata["quality_log_sample_size"] = int(quality_rollup.get("sample_size") or 0)
            composed.metadata["quality_rewrite_rate"] = float(quality_rollup.get("rewrite_rate") or 0.0)
            composed.metadata["quality_top_repair_reason"] = str(quality_rollup.get("top_repair_reason") or "")
        conversation_summary = str(composed.structured_result.summary or "").strip()
        if self._looks_like_reasoning_leak(conversation_summary):
            conversation_summary = ""
        await asyncio.to_thread(
            self._memory.write_conversational_context,
            session_id=session_id,
            context={
                "intent": ReasoningIntent.GENERAL_CHAT.value,
                "top_candidates": [],
                "candidate_doc_ids": [],
                "filters": {},
                "shown_actions": [item.kind.value for item in composed.actions],
                "result_summary": conversation_summary[:260],
                "ambiguity": verification.ambiguity_level,
                "used_clarification": False,
                "response_mode": composed.response_mode,
                "selected_file": None,
                "parsed_operation": str(getattr(parsed_intent, "operation", "chat") or "chat"),
                "parsed_scope": str(getattr(parsed_intent, "scope", "single") or "single"),
                "parsed_target": str(getattr(parsed_intent, "target", "") or ""),
                "conversation_path": conversation_path,
                "last_user_query": req.query[:220],
                "scope_clarification_pending": False,
                "scope_clarification_asked": False,
            },
        )
        await asyncio.to_thread(
            self._update_session_digest_metadata,
            composed=composed,
            session_id=session_id,
            query=req.query,
            assistant_summary=conversation_summary or execution.generated_text,
            context_digest_used=digest_context_used,
        )
        return composed

    def _general_chat_assist_mode(*, parsed_intent: ParsedIntent, query: str) -> tuple[str, float, bool]:
        operation = str(getattr(parsed_intent, "operation", "chat") or "chat")
        target = str(getattr(parsed_intent, "target", "") or "").strip()
        ambiguity = str(getattr(parsed_intent, "ambiguity", "clear") or "clear")
        confidence = float(getattr(parsed_intent, "confidence", 0.0) or 0.0)
        if utils._is_explicit_web_search_request(query):
            return "none", confidence, True
        if operation == "chat":
            return "none", confidence, False
        local_target_hint = target if utils._has_local_file_target_cues(query) else None
        if not utils._has_explicit_retrieval_request(query, target_hint=local_target_hint):
            return "none", confidence, True
        if not target:
            if ambiguity == "unclear":
                return "clarify", confidence, False
            return "none", confidence, True
        if ambiguity == "unclear":
            if confidence >= 0.45:
                return "clarify", confidence, False
            return "none", confidence, False
        if confidence >= 0.62:
            return "light", confidence, False
        return "none", confidence, False

    def _append_source_line(*, text: str, citations: list[Citation], response_language: str) -> str:
        base = (text or "").strip()
        if not citations:
            return base
        names: list[str] = []
        seen: set[str] = set()
        for citation in citations:
            name = Path(citation.file_path).name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= 3:
                break
        if not names:
            return base
        if response_language == "ko":
            source_line = f"참고 자료: {', '.join(names)}"
        else:
            source_line = f"References: {', '.join(names)}"
        if source_line in base:
            return base
        if not base:
            return source_line
        return f"{base}\n    \n    {source_line}"


# Backward-compatible alias
CoreChatHelpersHelpers = CoreChatHelpers
