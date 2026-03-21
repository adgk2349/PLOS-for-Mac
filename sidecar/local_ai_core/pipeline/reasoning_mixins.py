from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import unicodedata

from ..clarification_budget import ClarificationBudget, ClarificationBudgetState
from ..db import Database
from ..embedding import EmbeddingService
from ..executor import LocalExecutor
from ..followup_resolver import FollowUpResolution, FollowUpResolver
from ..intent_parser import IntentParser
from ..language_utils import insufficient_evidence_message, resolve_response_language
from ..local_planner import LocalPlanner
from ..memory_service import MemoryService
from ..models import (
    BehaviorPolicy,
    ChatFilters,
    Citation,
    ComposedChatResponseV2,
    ExecutionResult,
    LocalPlan,
    LocalChatRequestV2,
    MemoryEventRequest,
    MemoryEventType,
    LocalEngine,
    ParsedIntent,
    PrivacyMode,
    ReasoningIntent,
    SuggestedActionKind,
    VerificationResult,
    WorkMode,
)
from ..response_composer import ResponseComposer
from ..retrieval import extract_query_hints, merge_filters, retrieve_bundle
from ..vector_store import VectorStore
from ..verifier import ResultVerifier


ReasoningPipeline = None  # patched by reasoning_pipeline.py

class ReasoningPipelineMethodsMixin:
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
                return f"{query}\n\nConstraint: {followup_resolution.query_hint}"
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
                return f"{query}\n\nFollow-up hint: {followup_resolution.query_hint}"
            return query
        seed = recent_queries[0]
        joined = f"{seed}\n\n{query}"
        if followup_resolution and followup_resolution.query_hint:
            joined += f"\n\nHint: {followup_resolution.query_hint}"
        return joined

    @staticmethod
    def _candidate_gap_small(citations: list[Citation]) -> bool:
        if len(citations) < 2:
            return False
        first = float(citations[0].score)
        second = float(citations[1].score)
        return abs(first - second) <= 0.04

    def _effective_behavior_policy(
        self,
        *,
        req: LocalChatRequestV2,
        memory_bundle,
        default_action_order: list[str],
        default_mode: str | None,
        workspace_weights: dict[str, float],
    ) -> BehaviorPolicy:
        stored = self._db.get_behavior_policy()
        overrides = req.behavior_overrides
        preferred_action_order = list(stored.preferred_action_order)
        if default_action_order:
            parsed_order: list[SuggestedActionKind] = []
            for raw in default_action_order:
                try:
                    parsed_order.append(SuggestedActionKind(raw))
                except Exception:
                    continue
            if parsed_order:
                preferred_action_order = parsed_order

        preferred_mode = stored.preferred_mode
        if preferred_mode is None and default_mode:
            try:
                preferred_mode = WorkMode(default_mode)
            except Exception:
                preferred_mode = None
        if preferred_mode is None:
            for item in memory_bundle.workspace_items:
                if item.memory_type == "default_mode":
                    mode_value = str(item.value_json.get("value") or "").strip()
                    if mode_value:
                        try:
                            preferred_mode = WorkMode(mode_value)
                        except Exception:
                            preferred_mode = None
                        break

        merged = BehaviorPolicy(
            workspace_weights=dict(workspace_weights or stored.workspace_weights),
            preferred_mode=preferred_mode,
            preferred_action_order=preferred_action_order,
            preferred_response_length=stored.preferred_response_length,
        )
        for item in memory_bundle.workspace_items:
            if item.memory_type == "retrieval_weight":
                try:
                    merged.workspace_weights[item.key] = float(item.value_json.get("weight"))
                except Exception:
                    continue
            if item.memory_type == "preferred_actions":
                actions = item.value_json.get("actions")
                if isinstance(actions, list):
                    parsed_actions: list[SuggestedActionKind] = []
                    for raw in actions:
                        try:
                            parsed_actions.append(SuggestedActionKind(str(raw)))
                        except Exception:
                            continue
                    if parsed_actions:
                        merged.preferred_action_order = parsed_actions

        if overrides is not None:
            if overrides.workspace_weights is not None:
                merged.workspace_weights = dict(overrides.workspace_weights)
            if overrides.preferred_mode is not None:
                merged.preferred_mode = overrides.preferred_mode
            if overrides.preferred_action_order is not None:
                merged.preferred_action_order = list(overrides.preferred_action_order)
            if overrides.preferred_response_length is not None:
                merged.preferred_response_length = overrides.preferred_response_length
            # Keep local personalization policy synced with explicit override input.
            self._db.update_behavior_policy(merged)
        return merged

    @staticmethod
    def _citation_from_chunk(chunk, metadata_map: dict) -> Citation:
        row = metadata_map.get(chunk.doc_id) or {}
        return Citation(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            snippet=chunk.snippet,
            score=chunk.score,
            modified_at=chunk.modified_at,
            category=row.get("category", "참고자료"),
            subcategory=row.get("subcategory", ""),
            tags=row.get("tags", []),
            document_type=row.get("document_type", ""),
            importance=row.get("importance", 0.5),
        )

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

    @staticmethod
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

    @staticmethod
    def _outcome_event_type(result_type: str) -> MemoryEventType | None:
        if result_type == "comparison":
            return MemoryEventType.COMPARISON
        if result_type == "draft":
            return MemoryEventType.DRAFT_CREATED
        if result_type in {"answer", "summary"}:
            return MemoryEventType.SUMMARY_CREATED
        return None

    @staticmethod
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

    @staticmethod
    def _build_secondary_reasoner_prompt(*, query: str, parsed_intent: ParsedIntent, response_language: str) -> str:
        if response_language == "ko":
            intent_hint = parsed_intent.intent.value
            return (
                f"{query}\n\n"
                "추가 지시:\n"
                f"- intent={intent_hint}에 맞춰 답변합니다.\n"
                "- 코드 원문/문장 반복을 금지합니다.\n"
                "- 근거 기반 핵심만 간결하게 정리합니다.\n"
                "- 불확실하면 마지막에 확인 질문 1개를 추가합니다."
            )
        return (
            f"{query}\n\n"
            "Additional constraints:\n"
            f"- Keep alignment with intent={parsed_intent.intent.value}.\n"
            "- Avoid repetitive output and raw code dump.\n"
            "- Keep concise, grounded summary.\n"
            "- If uncertain, end with one clarifying question."
        )

    @staticmethod
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

    @staticmethod
    def _conversation_max_tokens(response_length: str) -> int:
        """Map response_length preference to a max_tokens cap for conversation turns."""
        mapping = {
            "short": 100,
            "medium": 160,
            "long": 260,
        }
        return mapping.get(str(response_length).lower(), 160)

    def _run_general_chat(
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
                response_length=memory_prefs.response_length,
            )
        digest_context_used = bool(context_summary.strip())
        execution = self._executor.execute_conversation(
            query=req.query,
            mode=req.mode,
            startup_profile=workspace.startup_profile,
            engine=settings.local_engine,
            mlx_model_path=settings.mlx_model_path,
            llama_model_path=settings.llama_model_path,
            language_preference=settings.language,
            session_summary=context_summary,
            max_tokens=self._conversation_max_tokens(memory_prefs.response_length),
        )
        execution.tool_logs.insert(0, f"router:intent={ReasoningIntent.GENERAL_CHAT.value}")
        execution.tool_logs.append("agent:conversation_assistant")
        conversation_path = "local_conversation"
        escalated_provider: str | None = None
        is_local = True
        brief_chat = self._is_brief_chat_query(req.query)

        needs_recovery = (not execution.generated_text) or (execution.used_fallback and not brief_chat)
        if needs_recovery and settings.privacy_mode == PrivacyMode.HYBRID:
            escalated = self._escalate_general_chat(
                query=req.query,
                mode=req.mode,
                context_summary=context_summary,
                last_context=last_context or {},
                settings=settings,
            )
            if escalated is not None:
                execution, escalated_provider = escalated
                conversation_path = "external_escalated"
                is_local = False
            elif execution.used_fallback:
                execution.tool_logs.append("conversation_fallback:local_failure")

        if not execution.generated_text:
            execution = self._runtime_error_execution(response_language, execution.runtime_detail)
            if settings.privacy_mode == PrivacyMode.HYBRID:
                execution.tool_logs.append("runtime_error:conversation_local_and_external_failed")
            else:
                execution.tool_logs.append("runtime_error:conversation_local_failed_local_only")
        elif execution.result_type == "conversation":
            repaired = self._repair_repetitive_conversation_response(
                query=req.query,
                execution=execution,
                session_digest=session_digest,
                last_context=last_context,
                response_language=response_language,
                mode=req.mode,
                workspace=workspace,
                settings=settings,
                response_length=memory_prefs.response_length,
            )
            if repaired is not None:
                execution = repaired
                execution.tool_logs.append("conversation_repair:anti_repeat")
                conversation_path = "local_conversation_repaired"

        assist_citations: list[Citation] = []
        if assist_mode == "light" and execution.result_type == "conversation":
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

        self._memory.write_memory_event(
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
            )
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
            response_length=memory_prefs.response_length,
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
        self._memory.write_conversational_context(
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
                "scope_clarification_pending": False,
                "scope_clarification_asked": False,
            },
        )
        self._update_session_digest_metadata(
            composed=composed,
            session_id=session_id,
            query=req.query,
            assistant_summary=conversation_summary or execution.generated_text,
            context_digest_used=digest_context_used,
        )
        return composed

    @staticmethod
    def _general_chat_assist_mode(*, parsed_intent: ParsedIntent, query: str) -> tuple[str, float, bool]:
        operation = str(getattr(parsed_intent, "operation", "chat") or "chat")
        target = str(getattr(parsed_intent, "target", "") or "").strip()
        ambiguity = str(getattr(parsed_intent, "ambiguity", "clear") or "clear")
        confidence = float(getattr(parsed_intent, "confidence", 0.0) or 0.0)
        if operation == "chat":
            return "none", confidence, False
        if not ReasoningPipeline._has_explicit_retrieval_request(query, target_hint=target):
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

    def _assistive_retrieval_citations(
        self,
        *,
        query: str,
        workspace,
        parsed_intent: ParsedIntent,
        user_filters: ChatFilters | None,
    ) -> list[Citation]:
        hint_filters = extract_query_hints(query)
        merged_filters = merge_filters(user_filters, hint_filters) or ChatFilters()
        if merged_filters.excluded is None:
            merged_filters.excluded = False
        allowed_doc_ids, metadata_map = self._resolve_workspace_docs(
            workspace=workspace,
            filters=merged_filters,
        )
        if not allowed_doc_ids:
            return []
        focus_terms, strict_focus = self._extract_path_focus_terms(
            query=query,
            topics=parsed_intent.entities.topics,
        )
        allowed_doc_ids, metadata_map = self._apply_focus_filter(
            doc_ids=allowed_doc_ids,
            metadata_map=metadata_map,
            focus_terms=focus_terms,
            strict_focus=strict_focus,
        )
        if getattr(parsed_intent, "target", None):
            allowed_doc_ids, metadata_map, _ = self._apply_explicit_file_focus(
                doc_ids=allowed_doc_ids,
                metadata_map=metadata_map,
                file_terms=[str(parsed_intent.target)],
            )
        if not allowed_doc_ids:
            return []
        raw = self._fallback_file_citations(
            query=query,
            allowed_doc_ids=allowed_doc_ids,
            metadata_map=metadata_map,
            limit=30 if str(getattr(parsed_intent, "scope", "single") or "single") == "all" else 8,
        )
        return raw[:3]

    @staticmethod
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
        return f"{base}\n\n{source_line}"

    def _compose_scope_target_clarification(
        self,
        *,
        req: LocalChatRequestV2,
        response_language: str,
        parsed_intent: ParsedIntent,
        behavior_policy: BehaviorPolicy,
        memory_prefs,
        workspace,
        session_id: str,
        workspace_id: str,
        conversation_path: str,
    ) -> ComposedChatResponseV2:
        plan = LocalPlan(
            plan_type="conversation",
            selected_files=[],
            selected_chunks=[],
            response_strategy="conversational_assistant",
            allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
            external_reasoning_needed=False,
        )
        prompt = self._scope_target_clarification_prompt(
            response_language=response_language,
            operation=str(getattr(parsed_intent, "operation", "find") or "find"),
        )
        execution = ExecutionResult(
            result_type="conversation",
            structured_payload={
                "style": "clarification",
                "ungrounded_allowed": True,
                "scope_clarification": True,
            },
            citations=[],
            tool_logs=["clarification:scope_target"],
            generated_text=prompt,
            engine_used=None,
            used_fallback=False,
            runtime_detail=None,
        )
        verification = VerificationResult(
            is_valid=True,
            confidence=0.76,
            issues=[],
            ambiguity_level=0.14,
            candidate_mode=False,
        )
        self._memory.write_memory_event(
            MemoryEventRequest(
                event_type=MemoryEventType.QUERY,
                session_id=session_id,
                workspace_id=workspace_id,
                summary=req.query[:220],
                related_file_ids=[],
                metadata_json={
                    "mode": req.mode.value,
                    "intent": parsed_intent.intent.value,
                    "result_type": execution.result_type,
                    "conversation_path": conversation_path,
                    "scope_clarification": True,
                },
                importance=0.34,
            )
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
            response_length=memory_prefs.response_length,
            show_citations=False,
            prefer_action_suggestions=memory_prefs.prefer_action_suggestions,
            used_profile=workspace.startup_profile,
            engine_used=None,
            used_fallback=False,
            runtime_detail=None,
            followup_resolution=None,
            allow_clarification=False,
            conversation_path=conversation_path,
            escalated_provider=None,
            is_local=True,
        )
        composed.metadata["assist_mode"] = "clarify"
        self._memory.write_conversational_context(
            session_id=session_id,
            context={
                "intent": ReasoningIntent.FOLLOWUP_QUESTION.value,
                "top_candidates": [],
                "candidate_doc_ids": [],
                "filters": {},
                "shown_actions": [item.kind.value for item in composed.actions],
                "result_summary": str(composed.structured_result.summary or "")[:260],
                "ambiguity": verification.ambiguity_level,
                "used_clarification": True,
                "response_mode": composed.response_mode,
                "selected_file": None,
                "parsed_operation": str(getattr(parsed_intent, "operation", "find") or "find"),
                "parsed_scope": str(getattr(parsed_intent, "scope", "single") or "single"),
                "parsed_target": str(getattr(parsed_intent, "target", "") or ""),
                "scope_clarification_pending": True,
                "scope_clarification_asked": True,
            },
        )
        self._update_session_digest_metadata(
            composed=composed,
            session_id=session_id,
            query=req.query,
            assistant_summary=str(composed.structured_result.summary or ""),
            context_digest_used=False,
        )
        return composed

    @staticmethod
    def _scope_target_clarification_prompt(*, response_language: str, operation: str) -> str:
        op = (operation or "find").lower()
        if response_language == "ko":
            if op == "summarize":
                return "좋아요. 어떤 자료를 기준으로 요약할까요? 파일명 일부나 폴더명을 한 번만 알려주세요."
            if op == "open":
                return "좋아요. 어떤 파일을 열면 될까요? 파일명 일부만 알려주시면 바로 찾을게요."
            return "좋아요. 어떤 대상을 기준으로 찾을까요? 예: 데통 파일 전부 / 자료구조 폴더 전부"
        if op == "summarize":
            return "Got it. Which material should I summarize? Share a file name fragment or folder name."
        if op == "open":
            return "Got it. Which file should I open? A partial file name is enough."
        return "Got it. What target should I search? Example: all files under a specific topic or folder."

    @staticmethod
    def _is_greeting_query(query: str) -> bool:
        lowered = (query or "").strip().lower()
        if not lowered:
            return False
        if ReasoningPipeline._contains_task_cues(lowered):
            return False
        if not any(
            token in lowered
            for token in (
                "안녕",
                "반가워",
                "고마워",
                "감사",
                "hello",
                "hi",
                "hey",
                "thanks",
                "thank you",
                "how are you",
            )
        ):
            return False
        token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered))
        return token_count <= 8

    @staticmethod
    def _contains_task_cues(lowered: str) -> bool:
        if ReasoningPipeline._has_explicit_retrieval_request(lowered):
            return True
        action_cues = (
            "비교",
            "분석",
            "작성",
            "초안",
            "rewrite",
            "draft",
            "compare",
            "analysis",
        )
        return any(token in lowered for token in action_cues)

    @staticmethod
    def _has_explicit_retrieval_request(query: str, *, target_hint: str | None = None) -> bool:
        lowered = (query or "").strip().lower()
        if not lowered:
            return False
        retrieval_targets = (
            "파일",
            "문서",
            "폴더",
            "디렉토리",
            "경로",
            "주차",
            "태그",
            "open",
            "file",
            "document",
            "folder",
            "directory",
            "path",
            "tag",
            ".txt",
            ".pdf",
            ".md",
            ".docx",
        )
        retrieval_actions = (
            "찾아",
            "검색",
            "보여",
            "열어",
            "요약",
            "정리",
            "리스트",
            "목록",
            "find",
            "search",
            "show",
            "open",
            "list",
            "summary",
            "summarize",
        )
        has_target = any(token in lowered for token in retrieval_targets)
        if target_hint and str(target_hint).strip():
            has_target = True
        has_action = any(token in lowered for token in retrieval_actions)
        if has_target and has_action:
            return True
        scope_all_tokens = ("전체", "전부", "모두", "모든", "all", "every", "entire")
        if has_target and any(token in lowered for token in scope_all_tokens):
            return True
        return False

    @classmethod
    def _should_force_general_chat(cls, *, query: str, parsed_intent: ParsedIntent) -> bool:
        if cls._is_greeting_query(query):
            return True
        if str(getattr(parsed_intent, "operation", "chat") or "chat") != "chat":
            return False
        return cls._looks_general_chat_query(query)

    @staticmethod
    def _looks_general_chat_query(query: str) -> bool:
        lowered = (query or "").strip().lower()
        if not lowered:
            return False
        if ReasoningPipeline._contains_task_cues(lowered):
            return False
        chat_cues = (
            "몇 시",
            "몇시",
            "잠",
            "자야",
            "새벽",
            "아침",
            "피곤",
            "고민",
            "괜찮아",
            "괜찮을까",
            "배고파",
            "뭐 먹",
            "추천",
            "기분",
            "운동",
            "목이",
            "아파",
            "오늘",
            "내일",
            "night",
            "sleep",
            "tired",
            "hungry",
        )
        if any(token in lowered for token in chat_cues):
            return True
        token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered))
        if token_count <= 10 and lowered.endswith(("?", "요", "까", "냐", "니", "지", "네", "!")):
            return True
        return False

    @staticmethod
    def _is_brief_chat_query(query: str) -> bool:
        raw = (query or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        cues = (
            "그렇구나",
            "알겠",
            "오케이",
            "그래",
            "아하",
            "맞아",
            "ㅇㅋ",
            "ok",
            "okay",
            "got it",
            "makes sense",
            "cool",
            "thanks",
            "thank you",
        )
        if any(cue in lowered for cue in cues):
            return True
        compact = re.sub(r"\s+", "", raw)
        if len(compact) <= 8:
            return True
        return False

    @staticmethod
    def _should_apply_conversation_context(
        query: str,
        *,
        has_session_digest: bool = False,
        has_last_context: bool = False,
        session_digest: dict[str, Any] | None = None,
        last_context: dict[str, Any] | None = None,
    ) -> bool:
        lowered = (query or "").strip().lower()
        if not lowered:
            return False
        has_history = bool(has_session_digest or has_last_context)
        if not has_history:
            return False
        if ReasoningPipeline._is_greeting_query(query) and not has_history:
            return False
        if ReasoningPipeline._is_brief_chat_query(query):
            return False
        if ReasoningPipeline._has_followup_context_signal(query):
            return True
        relevance = ReasoningPipeline._conversation_context_relevance(
            query=query,
            session_digest=session_digest,
            last_context=last_context,
        )
        return relevance >= 0.35

    @staticmethod
    def _has_followup_context_signal(query: str) -> bool:
        lowered_query = (query or "").strip().lower()
        if not lowered_query:
            return False
        followup_tokens = (
            "그럼",
            "그러면",
            "그리고",
            "근데",
            "아까",
            "방금",
            "이어서",
            "그거",
            "이거",
            "계속",
            "다시",
            "then",
            "and",
            "continue",
            "as above",
            "follow up",
        )
        return any(token in lowered_query for token in followup_tokens)

    @staticmethod
    def _has_strong_followup_context_signal(query: str) -> bool:
        lowered_query = (query or "").strip().lower()
        if not lowered_query:
            return False
        strong_tokens = (
            "그거",
            "이거",
            "아까",
            "방금",
            "이어서",
            "그 파일",
            "그 문서",
            "위에서",
            "that one",
            "this one",
            "previous",
            "above",
            "continue",
        )
        if not any(token in lowered_query for token in strong_tokens):
            return False
        token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered_query))
        return token_count <= 18

    @staticmethod
    def _conversation_context_relevance(
        *,
        query: str,
        session_digest: dict[str, Any] | None,
        last_context: dict[str, Any] | None,
    ) -> float:
        query_terms = set(ReasoningPipeline._tokenize_query_terms(query))
        if not query_terms:
            return 0.0
        context_terms: set[str] = set()
        digest = session_digest or {}
        raw_turns = digest.get("recent_turns")
        if isinstance(raw_turns, list):
            for item in raw_turns[-8:]:
                if not isinstance(item, dict):
                    continue
                if str(item.get("role") or "").strip().lower() != "user":
                    continue
                for token in ReasoningPipeline._tokenize_query_terms(str(item.get("text") or "")):
                    context_terms.add(token)
        for item in (digest.get("active_topics") or []):
            for token in ReasoningPipeline._tokenize_query_terms(str(item or "")):
                context_terms.add(token)
        if last_context:
            fields = (
                str(last_context.get("parsed_target") or ""),
                str(last_context.get("result_summary") or ""),
                Path(str(last_context.get("selected_file") or "")).name,
            )
            for field in fields:
                for token in ReasoningPipeline._tokenize_query_terms(field):
                    context_terms.add(token)
        if not context_terms:
            return 0.0
        inter = len(query_terms.intersection(context_terms))
        union = len(query_terms.union(context_terms))
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _conversation_session_summary(
        *,
        query: str,
        session_digest: dict[str, Any] | None,
        last_context: dict | None,
        memory_bundle,
        response_length: str,
    ) -> str:
        digest = session_digest or {}
        followup_context = ReasoningPipeline._has_followup_context_signal(query)
        strong_followup_context = ReasoningPipeline._has_strong_followup_context_signal(query)
        user_recent_turns: list[str] = []
        assistant_recent_turns: list[str] = []
        raw_turns = digest.get("recent_turns")
        if isinstance(raw_turns, list):
            for entry in raw_turns:
                if not isinstance(entry, dict):
                    continue
                role = str(entry.get("role") or "user").strip().lower()
                text = str(entry.get("text") or "").strip()
                if not text:
                    continue
                if role == "assistant":
                    assistant_recent_turns.append(f"- A: {text[:120]}")
                else:
                    user_recent_turns.append(f"- U: {text[:160]}")
        recent_turns: list[str] = user_recent_turns[-4:]
        if not recent_turns and assistant_recent_turns and strong_followup_context:
            recent_turns = assistant_recent_turns[-1:]
        if not recent_turns:
            for item in memory_bundle.session_items:
                if item.key != "recent_query":
                    continue
                query_summary = str(item.value_json.get("summary") or "").strip()
                if query_summary:
                    recent_turns.append(f"- U: {query_summary[:160]}")
                if len(recent_turns) >= 4:
                    break

        open_loops = [
            f"- open: {str(item).strip()[:160]}"
            for item in (digest.get("open_loops") or [])
            if str(item).strip()
        ]
        stable_facts = [
            f"- fact: {str(item).strip()[:160]}"
            for item in (digest.get("stable_facts") or [])
            if str(item).strip()
        ]
        active_topics = [
            f"- topic: {str(item).strip()[:40]}"
            for item in (digest.get("active_topics") or [])
            if str(item).strip()
        ]
        last_context_lines: list[str] = []
        if last_context:
            selected = str(last_context.get("selected_file") or "").strip()
            if selected and followup_context:
                last_context_lines.append(f"- last_file: {Path(selected).name}")
            parsed_target = str(last_context.get("parsed_target") or "").strip()
            if parsed_target and followup_context:
                last_context_lines.append(f"- last_target: {parsed_target[:60]}")

        sections = [recent_turns]
        if followup_context:
            sections.extend([last_context_lines, open_loops])
            if strong_followup_context:
                sections.extend([stable_facts, active_topics])
        budget = ReasoningPipeline._conversation_context_budget_tokens(response_length)
        lines: list[str] = []
        used = 0
        for section_lines in sections:
            for line in section_lines:
                tokens = ReasoningPipeline._estimate_context_tokens(line)
                if used + tokens > budget:
                    continue
                lines.append(line)
                used += tokens
        return "\n".join(lines).strip()

    @staticmethod
    def _conversation_context_budget_tokens(response_length: str) -> int:
        mapping = {
            "short": 120,
            "medium": 220,
            "long": 320,
        }
        return mapping.get(str(response_length).lower(), 220)

    @staticmethod
    def _estimate_context_tokens(text: str) -> int:
        value = (text or "").strip()
        if not value:
            return 0
        return max(1, len(value) // 4)

    def _repair_repetitive_conversation_response(
        self,
        *,
        query: str,
        execution: ExecutionResult,
        session_digest: dict[str, Any] | None,
        last_context: dict[str, Any] | None,
        response_language: str,
        mode: WorkMode,
        workspace,
        settings,
        response_length: str,
    ) -> ExecutionResult | None:
        answer = str(execution.generated_text or "").strip()
        if not answer:
            return None
        if not self._looks_repetitive_conversation_output(
            query=query,
            answer=answer,
            session_digest=session_digest,
            last_context=last_context,
        ):
            return None

        repair_query = self._anti_repeat_query(
            query=query,
            previous_answer=answer,
            response_language=response_language,
        )
        repaired = self._executor.execute_conversation(
            query=repair_query,
            mode=mode,
            startup_profile=workspace.startup_profile,
            engine=settings.local_engine,
            mlx_model_path=settings.mlx_model_path,
            llama_model_path=settings.llama_model_path,
            language_preference=settings.language,
            session_summary=None,
            max_tokens=self._conversation_max_tokens(response_length),
        )
        repaired_text = str(repaired.generated_text or "").strip()
        if not repaired_text:
            return None
        if self._looks_repetitive_conversation_output(
            query=query,
            answer=repaired_text,
            session_digest=session_digest,
            last_context=last_context,
        ):
            return None
        return repaired

    @staticmethod
    def _anti_repeat_query(*, query: str, previous_answer: str, response_language: str) -> str:
        prior = re.sub(r"\s+", " ", (previous_answer or "").strip())[:140]
        if response_language == "ko":
            return (
                f"{query}\n\n"
                "바로 답변만 작성해줘. "
                "직전 답변과 겹치는 표현은 피하고, 자연스러운 한국어 존댓말 한두 문장으로 답해줘. "
                "역할 라벨이나 규칙 문장은 쓰지 마.\n"
                f"이전 답변 핵심: {prior}"
            )
        return (
            f"{query}\n\n"
            "Answer directly in one or two natural sentences. "
            "Avoid repeating wording from the previous answer. "
            "Do not output role labels or rule text.\n"
            f"Previous answer core: {prior}"
        )

    @classmethod
    def _looks_repetitive_conversation_output(
        cls,
        *,
        query: str,
        answer: str,
        session_digest: dict[str, Any] | None,
        last_context: dict[str, Any] | None,
    ) -> bool:
        cleaned_answer = ResponseComposer._strip_instruction_leakage(answer or "")
        cleaned_answer = re.sub(r"\s+", " ", cleaned_answer).strip()
        if not cleaned_answer:
            return True

        if cls._has_duplicate_sentences(cleaned_answer):
            return True

        previous_assistant_texts: list[str] = []
        digest = session_digest or {}
        raw_turns = digest.get("recent_turns")
        if isinstance(raw_turns, list):
            for entry in raw_turns:
                if not isinstance(entry, dict):
                    continue
                role = str(entry.get("role") or "").strip().lower()
                if role != "assistant":
                    continue
                text = str(entry.get("text") or "").strip()
                if text:
                    previous_assistant_texts.append(text)
        context = last_context or {}
        recent_summary = str(context.get("result_summary") or "").strip()
        if recent_summary:
            previous_assistant_texts.append(recent_summary)

        for prev in previous_assistant_texts[-3:]:
            if cls._text_similarity(cleaned_answer, prev) >= 0.76:
                return True

        if cls._text_similarity(cleaned_answer, query) >= 0.86 and len(cleaned_answer) <= 120:
            return True
        return False

    @staticmethod
    def _has_duplicate_sentences(text: str) -> bool:
        parts = [
            seg.strip()
            for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", text or "")
            if seg.strip()
        ]
        if len(parts) < 2:
            return False
        seen: set[str] = set()
        for seg in parts:
            key = re.sub(r"[^\w가-힣]+", "", seg).casefold()
            if not key:
                continue
            if key in seen:
                return True
            seen.add(key)
        return False

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        a_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (a or "").lower()))
        b_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (b or "").lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        union = len(a_tokens.union(b_tokens))
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _conversation_quality_debug_from_detail(detail: str | None) -> dict[str, Any]:
        value = str(detail or "").strip()
        if not value:
            return {
                "korean_rewrite_used": False,
                "quality_repair_reason": "",
                "repair_triggered": False,
                "repair_success": False,
                "leak_blocked": False,
                "direct_first_applied": False,
                "question_count_after_postprocess": 0,
                "recommendation_shape": "",
            }
        lowered = value.lower()
        rewrite_used = "korean_rewrite_used=1" in lowered
        repair_triggered = "repair_triggered=1" in lowered
        repair_success = "repair_success=1" in lowered
        leak_blocked = "leak_blocked=1" in lowered
        match = re.search(r"quality_repair_reason=([a-z0-9_\-|]+)", lowered)
        reason = match.group(1) if match else ""
        direct_first_applied = "direct_first_applied=1" in lowered
        question_match = re.search(r"question_count_after_postprocess=([0-9]+)", lowered)
        question_count_after_postprocess = int(question_match.group(1)) if question_match else 0
        shape_match = re.search(r"recommendation_shape=([a-z0-9_\-]+)", lowered)
        recommendation_shape = shape_match.group(1) if shape_match else ""
        return {
            "korean_rewrite_used": rewrite_used,
            "quality_repair_reason": reason,
            "repair_triggered": repair_triggered,
            "repair_success": repair_success,
            "leak_blocked": leak_blocked,
            "direct_first_applied": direct_first_applied,
            "question_count_after_postprocess": question_count_after_postprocess,
            "recommendation_shape": recommendation_shape,
        }

    @staticmethod
    def _flag_enabled(value: str | None, *, default: bool) -> bool:
        if value is None:
            return default
        lowered = str(value).strip().lower()
        if lowered in {"0", "false", "off", "no", "n"}:
            return False
        if lowered in {"1", "true", "on", "yes", "y"}:
            return True
        return default

    @staticmethod
    def _parse_positive_int(
        value: str | None,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            parsed = int(str(value).strip()) if value is not None else default
        except Exception:
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _load_recent_quality_events(path: Path, *, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        recent: deque[dict[str, Any]] = deque(maxlen=limit)
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        recent.append(payload)
        except Exception:
            return []
        return list(recent)

    @staticmethod
    def _quality_rollup_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
        sample_size = len(events)
        if sample_size <= 0:
            return {}
        rewrite_count = 0
        suppressed_count = 0
        repair_triggered_count = 0
        repair_success_count = 0
        leak_blocked_count = 0
        reason_counter: Counter[str] = Counter()
        for event in events:
            if bool(event.get("korean_rewrite_used")):
                rewrite_count += 1
            if bool(event.get("assistive_retrieval_suppressed")):
                suppressed_count += 1
            if bool(event.get("repair_triggered")):
                repair_triggered_count += 1
            if bool(event.get("repair_success")):
                repair_success_count += 1
            if bool(event.get("leak_blocked")):
                leak_blocked_count += 1
            reason_text = str(event.get("quality_repair_reason") or "").strip().lower()
            if reason_text:
                for token in reason_text.split("|"):
                    normalized = token.strip()
                    if normalized:
                        reason_counter[normalized] += 1
        top_reason = ""
        if reason_counter:
            top_reason = reason_counter.most_common(1)[0][0]
        return {
            "sample_size": sample_size,
            "rewrite_count": rewrite_count,
            "rewrite_rate": round(rewrite_count / sample_size, 3),
            "suppressed_count": suppressed_count,
            "suppressed_rate": round(suppressed_count / sample_size, 3),
            "repair_triggered_count": repair_triggered_count,
            "repair_triggered_rate": round(repair_triggered_count / sample_size, 3),
            "repair_success_count": repair_success_count,
            "repair_success_rate": round(repair_success_count / sample_size, 3),
            "leak_blocked_count": leak_blocked_count,
            "leak_blocked_rate": round(leak_blocked_count / sample_size, 3),
            "top_repair_reason": top_reason,
        }

    def _record_conversation_quality_event(
        self,
        *,
        session_id: str,
        query: str,
        execution: ExecutionResult,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._quality_log_enabled:
            return {}

        quality_repair_reason = str(metadata.get("quality_repair_reason") or "").strip().lower()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "engine": execution.engine_used.value if execution.engine_used else "none",
            "result_type": execution.result_type,
            "query_length": len(str(query or "")),
            "answer_length": len(str(execution.generated_text or "")),
            "korean_rewrite_used": bool(metadata.get("korean_rewrite_used", False)),
            "quality_repair_reason": quality_repair_reason,
            "assistive_retrieval_suppressed": bool(metadata.get("assistive_retrieval_suppressed", False)),
            "assist_mode": str(metadata.get("assist_mode") or "none"),
            "repair_triggered": bool(metadata.get("repair_triggered", False)),
            "repair_success": bool(metadata.get("repair_success", False)),
            "leak_blocked": bool(metadata.get("leak_blocked", False)),
        }

        self._quality_rollup_window.append(payload)
        try:
            self._quality_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._quality_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Debug logging should not affect response generation.
            pass
        return self._quality_rollup_summary(list(self._quality_rollup_window))

    def _update_session_digest_metadata(
        self,
        *,
        composed: ComposedChatResponseV2,
        session_id: str,
        query: str,
        assistant_summary: str,
        context_digest_used: bool,
    ) -> None:
        refresh_mode = "rule"
        turn_count = 0
        try:
            digest = self._memory.update_session_digest(
                session_id=session_id,
                user_query=query,
                assistant_summary=assistant_summary,
                mode="hybrid",
            )
            turn_count = int(digest.get("turn_count") or 0)
            refresh_mode = str(digest.get("digest_refresh") or "rule")
        except Exception:
            refresh_mode = "fallback_rule"
            turn_count = 0
        composed.metadata["context_digest_used"] = bool(context_digest_used)
        composed.metadata["context_injected"] = bool(context_digest_used)
        composed.metadata["digest_turn_count"] = turn_count
        composed.metadata["digest_refresh"] = refresh_mode

    def _refresh_digest_with_local_model(self, session_id: str, digest: dict[str, Any]) -> dict[str, Any] | None:
        settings = self._db.get_settings()
        workspace = self._db.get_workspace()
        prompt = self._digest_model_refresh_prompt(digest)
        inference = self._executor._local_inference.generate_conversational(  # noqa: SLF001
            query=prompt,
            mode=WorkMode.GENERAL,
            profile=workspace.startup_profile.value,
            engine=settings.local_engine,
            mlx_model_path=settings.mlx_model_path,
            llama_model_path=settings.llama_model_path,
            language_preference=settings.language,
            max_tokens=220,
            session_summary=None,
            allow_static_fallback=False,
        )
        answer = str(inference.answer or "").strip()
        if not answer or self._looks_like_reasoning_leak(answer):
            return None
        parsed = self._parse_digest_model_output(answer)
        if parsed is None:
            return None
        return parsed

    @staticmethod
    def _digest_model_refresh_prompt(digest: dict[str, Any]) -> str:
        compact = {
            "active_topics": digest.get("active_topics") or [],
            "stable_facts": digest.get("stable_facts") or [],
            "open_loops": digest.get("open_loops") or [],
            "recent_turns": digest.get("recent_turns") or [],
        }
        payload = json.dumps(compact, ensure_ascii=False)
        return (
            "다음은 대화 메모리 digest입니다. 잡음을 제거하고 다음 JSON 형식으로만 답하세요.\n"
            '{"active_topics":["..."],"stable_facts":["..."],"open_loops":["..."],"recent_turns":[{"role":"user|assistant","text":"..."}]}\n'
            "규칙: active_topics<=8, stable_facts<=10, open_loops<=6, recent_turns<=8. "
            "중복/지시문/정책문구는 제거하세요.\n"
            f"digest={payload}"
        )

    @staticmethod
    def _parse_digest_model_output(text: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        raw = match.group(0)
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        output: dict[str, Any] = {}
        for key, cap in (
            ("active_topics", 8),
            ("stable_facts", 10),
            ("open_loops", 6),
        ):
            value = parsed.get(key)
            if not isinstance(value, list):
                continue
            cleaned: list[str] = []
            seen: set[str] = set()
            for item in value:
                text_item = str(item or "").strip()
                if not text_item:
                    continue
                dedupe_key = text_item.casefold()
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                cleaned.append(text_item[:180])
                if len(cleaned) >= cap:
                    break
            output[key] = cleaned

        turns = parsed.get("recent_turns")
        if isinstance(turns, list):
            cleaned_turns: list[dict[str, str]] = []
            for item in turns:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "").strip().lower()
                role = "assistant" if role == "assistant" else "user"
                text_item = str(item.get("text") or "").strip()
                if not text_item:
                    continue
                cleaned_turns.append({"role": role, "text": text_item[:220]})
                if len(cleaned_turns) >= 8:
                    break
            if cleaned_turns:
                output["recent_turns"] = cleaned_turns
        return output or None

    def _escalate_general_chat(
        self,
        *,
        query: str,
        mode: WorkMode,
        context_summary: str,
        last_context: dict,
        settings,
    ) -> tuple[ExecutionResult, str] | None:
        if self._providers is None:
            return None
        provider_order = ["anthropic", "openai"]
        mini_citations = self._context_citation_summaries(context_summary=context_summary, last_context=last_context)
        payload_query = query
        if context_summary:
            payload_query = f"{query}\n\nSession summary:\n{context_summary}"
        for provider in provider_order:
            if not self._providers.provider_has_key(provider):
                continue
            try:
                result = self._providers.analyze_sync(
                    provider=provider,
                    query=payload_query,
                    mode=mode,
                    citations=mini_citations,
                    language_preference=settings.language,
                )
            except Exception:
                continue
            if not result.answer.strip():
                continue
            execution = ExecutionResult(
                result_type="conversation",
                structured_payload={
                    "style": "general_chat",
                    "source": "external_escalated",
                    "provider": provider,
                    "ungrounded_allowed": True,
                },
                citations=[],
                tool_logs=[f"external_escalated:{provider}"],
                generated_text=result.answer.strip(),
                engine_used=None,
                used_fallback=False,
                runtime_detail=f"external_escalated_provider={provider}",
            )
            return execution, provider
        return None

    @staticmethod
    def _context_citation_summaries(*, context_summary: str, last_context: dict) -> list[Citation]:
        snippets: list[tuple[str, str]] = []
        if context_summary:
            snippets.append(("session_context.txt", context_summary[:220]))
        top_candidates = last_context.get("top_candidates")
        if isinstance(top_candidates, list):
            for raw in top_candidates[:2]:
                path = str(raw or "").strip()
                if not path:
                    continue
                snippets.append((Path(path).name or "candidate.txt", f"Previous candidate file name: {Path(path).name}"))
                if len(snippets) >= 2:
                    break
        now = datetime.now(timezone.utc)
        output: list[Citation] = []
        for idx, (name, snippet) in enumerate(snippets[:2], start=1):
            output.append(
                Citation(
                    doc_id=f"session-summary-{idx}",
                    chunk_id=f"session-summary-chunk-{idx}",
                    file_path=name,
                    snippet=snippet,
                    score=0.45,
                    modified_at=now,
                )
            )
        return output

    @staticmethod
    def _runtime_error_execution(response_language: str, detail: str | None) -> ExecutionResult:
        if response_language == "ko":
            text = "로컬 대화 엔진을 실행하지 못했습니다. 모델 설치/경로를 확인한 뒤 다시 시도해 주세요."
        else:
            text = "The local conversation engines are unavailable. Verify model installation/path and try again."
        return ExecutionResult(
            result_type="runtime_error",
            structured_payload={"reason": "conversation_engine_unavailable"},
            citations=[],
            tool_logs=[],
            generated_text=text,
            engine_used=None,
            used_fallback=True,
            runtime_detail=detail,
        )

    def _should_escalate_summary_to_external(
        self,
        *,
        req: LocalChatRequestV2,
        parsed_intent: ParsedIntent,
        settings,
        citations: list[Citation],
    ) -> bool:
        if parsed_intent.intent != ReasoningIntent.SUMMARIZE_FILE:
            return False
        if settings.privacy_mode != PrivacyMode.HYBRID:
            return False
        if self._providers is None:
            return False
        if not citations:
            return False
        if not self._is_16gb_tier_model(settings):
            return False
        return self._providers.provider_has_key("anthropic") or self._providers.provider_has_key("openai")

    def _escalate_summary_to_external(
        self,
        *,
        query: str,
        mode: WorkMode,
        citations: list[Citation],
        settings,
    ) -> tuple[ExecutionResult, str] | None:
        if self._providers is None:
            return None

        provider_order = ["anthropic", "openai"]
        request_query = self._summary_external_query(query=query, language_preference=settings.language)

        for provider in provider_order:
            if not self._providers.provider_has_key(provider):
                continue
            try:
                result = self._providers.analyze_sync(
                    provider=provider,
                    query=request_query,
                    mode=mode,
                    citations=citations[:8],
                    language_preference=settings.language,
                )
            except Exception:
                continue
            text = str(result.answer or "").strip()
            if not text:
                continue
            execution = ExecutionResult(
                result_type="summary",
                structured_payload={
                    "source": "external_summary_escalated",
                    "provider": provider,
                },
                citations=citations,
                tool_logs=[f"external_escalated_summary:{provider}"],
                generated_text=text,
                engine_used=None,
                used_fallback=False,
                runtime_detail=f"external_escalated_provider={provider}",
            )
            return execution, provider
        return None

    @staticmethod
    def _summary_external_query(*, query: str, language_preference: str | None) -> str:
        language = resolve_response_language(query, language_preference)
        if language == "ko":
            return (
                f"{query}\n\n"
                "출력 형식 규칙:\n"
                "1) 반드시 번호 목록 1~5로 작성\n"
                "2) 원문 문장 복붙/중복 금지\n"
                "3) 각 줄은 핵심 개념 1개만 간결하게 작성"
            )
        return (
            f"{query}\n\n"
            "Output rules:\n"
            "1) Return exactly 5 numbered points.\n"
            "2) Avoid verbatim copy and repetition.\n"
            "3) Each line should contain one concise core idea."
        )

    @staticmethod
    def _is_16gb_tier_model(settings) -> bool:
        reference = ""
        try:
            if settings.local_engine == LocalEngine.MLX:
                reference = str(settings.mlx_model_path or "")
            else:
                reference = str(settings.llama_model_path or "")
        except Exception:
            reference = ""
        if not reference:
            return False
        size_b = ReasoningPipeline._model_size_b(reference)
        if size_b is None:
            return False
        return size_b <= 8

    @staticmethod
    def _model_size_b(reference: str) -> int | None:
        text = str(reference or "").lower()
        if not text:
            return None
        # Prefer explicit "<N>b" captures while avoiding "4bit"-like matches.
        matches = re.findall(r"(?<!\d)(\d{1,3})(?:\.\d+)?\s*b(?!it)", text)
        if matches:
            for raw in reversed(matches):
                try:
                    value = int(raw)
                except Exception:
                    continue
                if 1 <= value <= 180:
                    return value
        if "phi-4-mini" in text:
            return 4
        return None

    def _resolve_workspace_docs(
        self,
        *,
        workspace,
        filters: ChatFilters,
    ) -> tuple[set[str], dict[str, dict]]:
        doc_ids = self._db.find_doc_ids_for_workspace(
            included_paths=workspace.included_paths,
            excluded_paths=workspace.excluded_paths,
            filters=filters,
            search=None,
        )
        metadata = self._db.get_documents_metadata_map(list(doc_ids))
        return doc_ids, metadata

    @classmethod
    def _apply_focus_filter(
        cls,
        *,
        doc_ids: set[str],
        metadata_map: dict[str, dict],
        focus_terms: list[str],
        strict_focus: bool,
    ) -> tuple[set[str], dict[str, dict]]:
        if not focus_terms:
            return doc_ids, metadata_map
        focused_doc_ids = cls._filter_doc_ids_by_path_focus(
            doc_ids=doc_ids,
            metadata_map=metadata_map,
            focus_terms=focus_terms,
        )
        if focused_doc_ids:
            return focused_doc_ids, {doc_id: row for doc_id, row in metadata_map.items() if doc_id in focused_doc_ids}
        if strict_focus:
            return set(), {}
        return doc_ids, metadata_map

    @staticmethod
    def _find_like_query(query: str) -> bool:
        lowered = (query or "").lower()
        cues = (
            "찾아",
            "어디",
            "뭐 있",
            "목록",
            "리스트",
            "폴더",
            "디렉토리",
            "find",
            "where",
            "list",
            "folder",
            "directory",
        )
        return any(token in lowered for token in cues)

    def _should_trigger_auto_index(
        self,
        *,
        req: LocalChatRequestV2,
        parsed_intent: ReasoningIntent,
        allowed_doc_ids: set[str],
        strict_focus: bool,
    ) -> bool:
        if self._indexing is None:
            return False
        if req.mode == WorkMode.STRICT_SEARCH:
            return False
        if not self._find_like_query(req.query):
            return False
        if parsed_intent not in {
            ReasoningIntent.FIND_FILE,
            ReasoningIntent.SUMMARIZE_FILE,
            ReasoningIntent.EXPLAIN_CONTENT,
            ReasoningIntent.FOLLOWUP_QUESTION,
            ReasoningIntent.FOLLOWUP_REFINE,
            ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
            ReasoningIntent.SOFT_CONFIRM,
            ReasoningIntent.NEXT_CANDIDATE,
            ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
            ReasoningIntent.OPEN_FILE,
        }:
            return False
        # If focus is strict and no matching docs, or workspace docs are empty, refresh index first.
        if strict_focus and not allowed_doc_ids:
            return True
        return not allowed_doc_ids

    def _run_auto_index(self, workspace) -> bool:
        if self._indexing is None:
            return False
        now = time.monotonic()
        if now - self._last_auto_index_started_at < 2.0:
            return False
        self._last_auto_index_started_at = now
        try:
            job = self._indexing.start_job("incremental", workspace)
            status = self._indexing.get_job(job.job_id)
            if status is not None and status.status == "completed":
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _looks_like_reasoning_leak(text: str) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return True
        leak_tokens = (
            "user:",
            "assistant:",
            "you:",
            "a:",
            "follow-up question:",
            "the user",
            "okay, let's see",
            "alright, let me",
            "i should",
            "i need to",
            "_continuation",
            "recent session context",
            "최근 세션 컨텍스트",
            "세션 컨텍스트",
            "이전 문장에 대한 답변으로",
            "input message:",
            "<conversation_memory>",
            "session summary:",
            "규칙:",
            "추가 지시:",
            "evidence:",
            "explanation:",
            "the question asks",
            "based on the evidence provided",
            "therefore, the answer is",
            "(more)",
            "최종 답변:",
            "사용자에게 물어볼 때는",
            "반드시 '?'를 붙여주세요",
            "ask at most one follow-up question",
            "keep response to 1-3 sentences",
            "명확한 답변이 필요할 경우",
            "3문장까지 가능합니다",
            "답변이 부족할 경우",
            "추가적인 질문",
            "덧붙일 수 있습니다",
            "사용자에게 직접 도움을 주세요",
            "사용자에게 도움을 주세요",
            "사용자의 말에 바로 반응하세요",
            "사용자의 질문에 바로 반응하세요",
            "사용자 메시지에 바로 반응하세요",
            "사용자 메시지에 명확한 답을 하세요",
            "사용자 메시지에 명확한 답변을 하세요",
        )
        if any(token in lowered for token in leak_tokens):
            return True
        if re.search(
            r"사용자의?\s*질문에\s*대한\s*(?:답변|답이).{0,50}(?:부족|충분하지|명확하지).{0,80}(?:추가적인?\s*(?:질문|설명)|질문을?\s*덧붙)",
            lowered,
        ):
            return True
        if re.search(
            r"사용자에게\s*(?:직접\s*)?(?:도움을?|답변을?|질문을?)\s*(?:주세요|주십시오|제공하세요)",
            lowered,
        ):
            return True
        if re.search(
            r"(?:사용자의?\s*(?:말|질문|요청)|사용자\s*메시지)에\s*바로\s*반응하(?:세요|십시오)",
            lowered,
        ):
            return True
        if re.search(
            r"(?:recent\s*session\s*context|최근\s*세션\s*컨텍스트|세션\s*컨텍스트|이전\s*문장에\s*대한\s*답변으로)",
            lowered,
        ):
            return True
        if re.search(
            r"사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)",
            lowered,
        ):
            return True
        return False

    @staticmethod
    def _tokenize_query_terms(query: str) -> list[str]:
        raw = re.findall(r"[A-Za-z가-힣0-9_+\-]{2,32}", query or "")
        stop = {
            "자료",
            "문서",
            "파일",
            "폴더",
            "디렉토리",
            "찾아",
            "정리",
            "요약",
            "설명",
            "지금",
            "아까",
            "그거",
            "where",
            "find",
            "file",
            "files",
            "folder",
            "directory",
            "what",
            "have",
            "list",
        }
        terms: list[str] = []
        seen: set[str] = set()
        for token in raw:
            key = token.casefold()
            if key in stop:
                continue
            if key in seen:
                continue
            seen.add(key)
            terms.append(token)
            if len(terms) >= 8:
                break
        return terms

    @staticmethod
    def _extract_explicit_file_terms(*, query: str, parsed_intent: ParsedIntent) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for name in parsed_intent.entities.file_names:
            token = str(name or "").strip()
            if not token:
                continue
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(token)
        pattern = re.compile(
            r"([A-Za-z가-힣0-9_+\-().\[\]]+\.(?:txt|md|markdown|pdf|docx|py|swift|json|yaml|yml))",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(query or ""):
            token = str(match.group(1) or "").strip()
            if not token:
                continue
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(token)
        return terms

    @staticmethod
    def _extract_requested_weeks(
        *,
        query: str,
        followup_resolution: FollowUpResolution | None = None,
    ) -> list[int]:
        excluded = ReasoningPipeline._extract_excluded_weeks(query)
        weeks: list[int] = []
        seen: set[int] = set()
        for match in re.finditer(r"([1-9]|1[0-9]|2[0-4])\s*주차", query or "", flags=re.IGNORECASE):
            try:
                value = int(match.group(1))
            except Exception:
                continue
            if value in excluded:
                continue
            if value in seen:
                continue
            seen.add(value)
            weeks.append(value)
        if followup_resolution is not None:
            value = followup_resolution.resolved_filters.get("week")
            if isinstance(value, int) and value not in seen and value not in excluded:
                seen.add(value)
                weeks.append(value)
        return weeks[:3]

    @staticmethod
    def _extract_excluded_weeks(query: str) -> set[int]:
        excluded: set[int] = set()
        for match in re.finditer(
            r"([1-9]|1[0-9]|2[0-4])\s*주차\s*(?:는\s*)?(?:말고|말구|빼고|제외(?:하고)?|아니고)",
            query or "",
            flags=re.IGNORECASE,
        ):
            try:
                excluded.add(int(match.group(1)))
            except Exception:
                continue
        return excluded

    @classmethod
    def _apply_week_exact_filter(
        cls,
        *,
        doc_ids: set[str],
        metadata_map: dict[str, dict],
        requested_weeks: list[int],
    ) -> tuple[set[str], dict[str, dict], bool, bool]:
        if not doc_ids or not requested_weeks:
            return doc_ids, metadata_map, False, False
        wanted = {int(item) for item in requested_weeks if isinstance(item, int)}
        if not wanted:
            return doc_ids, metadata_map, False, False

        matched: set[str] = set()
        for doc_id in doc_ids:
            row = metadata_map.get(doc_id) or {}
            path = str(row.get("path") or "").strip()
            if not path:
                continue
            week_hits = set(cls._extract_weeks_from_text(path))
            if week_hits.intersection(wanted):
                matched.add(doc_id)
        if not matched:
            return set(), {}, True, True
        filtered = {doc_id: row for doc_id, row in metadata_map.items() if doc_id in matched}
        return matched, filtered, True, False

    @staticmethod
    def _extract_weeks_from_text(text: str) -> list[int]:
        output: list[int] = []
        seen: set[int] = set()
        for match in re.finditer(r"([1-9]|1[0-9]|2[0-4])\s*주차", text or "", flags=re.IGNORECASE):
            try:
                value = int(match.group(1))
            except Exception:
                continue
            if value in seen:
                continue
            seen.add(value)
            output.append(value)
        return output

    @staticmethod
    def _focused_summary_chunk_limit(startup_profile) -> int:
        key = str(getattr(startup_profile, "value", startup_profile) or "").upper()
        if key == "FAST":
            return 12
        if key == "DEEP":
            return 30
        return 20

    def _build_focused_file_summary_citations(
        self,
        *,
        doc_ids: list[str],
        metadata_map: dict[str, dict],
        max_chunks_per_file: int,
    ) -> list[Citation]:
        cleaned_ids: list[str] = []
        seen: set[str] = set()
        for raw in doc_ids:
            doc_id = str(raw or "").strip()
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            cleaned_ids.append(doc_id)
        if not cleaned_ids:
            return []
        rows = self._db.list_chunks_by_doc_ids(cleaned_ids)
        chunks_by_doc: dict[str, list] = {}
        for row in rows:
            doc_id = str(row["doc_id"])
            chunks_by_doc.setdefault(doc_id, []).append(row)
        if not chunks_by_doc:
            return []
        for items in chunks_by_doc.values():
            items.sort(key=lambda item: int(item["chunk_order"] or 0))

        citations: list[Citation] = []
        for doc_rank, doc_id in enumerate(cleaned_ids):
            items = chunks_by_doc.get(doc_id) or []
            if not items:
                continue
            meta = metadata_map.get(doc_id) or {}
            category = str(meta.get("category") or "참고자료")
            subcategory = str(meta.get("subcategory") or "")
            document_type = str(meta.get("document_type") or "")
            tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
            importance = float(meta.get("importance", 0.5) or 0.5)
            for row_idx, row in enumerate(items[: max(1, int(max_chunks_per_file))]):
                snippet = self._clip_summary_snippet(str(row["text"] or ""), max_chars=320)
                if not snippet:
                    continue
                modified_raw = row["modified_at"]
                if isinstance(modified_raw, datetime):
                    modified_at = modified_raw
                else:
                    try:
                        modified_at = datetime.fromtimestamp(float(modified_raw), tz=timezone.utc)
                    except Exception:
                        modified_at = datetime.now(timezone.utc)
                base_score = max(0.22, 0.78 - (doc_rank * 0.05) - (row_idx * 0.008))
                citations.append(
                    Citation(
                        doc_id=doc_id,
                        chunk_id=str(row["chunk_id"] or f"{doc_id}:{row_idx}"),
                        file_path=str(row["path"] or meta.get("path") or doc_id),
                        snippet=snippet,
                        score=min(max(base_score, 0.16), 0.92),
                        modified_at=modified_at,
                        category=category,
                        subcategory=subcategory,
                        tags=[str(tag) for tag in tags][:8],
                        document_type=document_type,
                        importance=importance,
                    )
                )
        citations.sort(key=lambda item: item.score, reverse=True)
        return citations

    @classmethod
    def _apply_explicit_file_focus(
        cls,
        *,
        doc_ids: set[str],
        metadata_map: dict[str, dict],
        file_terms: list[str],
    ) -> tuple[set[str], dict[str, dict], bool]:
        if not doc_ids or not file_terms:
            return doc_ids, metadata_map, False
        normalized_terms: list[tuple[set[str], set[str], bool]] = []
        has_extension_term = False
        for raw in file_terms:
            name = Path(str(raw)).name.strip()
            if not name:
                continue
            ext_present = "." in name
            has_extension_term = has_extension_term or ext_present
            normalized_terms.append(
                (
                    cls._normalize_variants(name),
                    cls._normalize_variants(Path(name).stem),
                    ext_present,
                )
            )
        if not normalized_terms:
            return doc_ids, metadata_map, False

        matched: set[str] = set()
        for doc_id in doc_ids:
            row = metadata_map.get(doc_id) or {}
            raw_path = str(row.get("path") or "").strip()
            if not raw_path:
                continue
            base = Path(raw_path).name
            base_variants = cls._normalize_variants(base)
            stem_variants = cls._normalize_variants(Path(base).stem)
            for name_variants, stem_name_variants, ext_present in normalized_terms:
                if ext_present:
                    if name_variants.intersection(base_variants):
                        matched.add(doc_id)
                        break
                else:
                    if stem_name_variants.intersection(stem_variants):
                        matched.add(doc_id)
                        break

        if matched:
            return matched, {doc_id: row for doc_id, row in metadata_map.items() if doc_id in matched}, True
        if has_extension_term:
            return set(), {}, True
        return doc_ids, metadata_map, False

    @staticmethod
    def _should_expand_summary_scope(*, query: str, parsed_intent: ParsedIntent) -> bool:
        if parsed_intent.intent != ReasoningIntent.SUMMARIZE_FILE:
            return False
        if str(getattr(parsed_intent, "scope", "single") or "single") == "all":
            return True
        text = (query or "").strip().lower()
        if not text:
            return False
        scope_tokens = ("파일", "문서", "자료", "강의", "노트", "file", "files", "document", "documents", "docs")
        if not any(token in text for token in scope_tokens):
            return False
        all_tokens = ("전체", "전부", "모든", "모두", "all")
        multi_tokens = ("여러", "multiple", "across")
        if any(token in text for token in all_tokens):
            return True
        if any(token in text for token in multi_tokens) and any(token in text for token in ("요약", "summary", "핵심")):
            return True
        explicit_patterns = (
            r"(전체|전부|모든|모두)\s*(파일|문서|자료|강의|노트)",
            r"(파일|문서|자료|강의|노트)\s*(전체|전부|모든|모두)",
            r"여러\s*개?\s*(파일|문서|자료)",
            r"all\s+(?:files?|documents?|docs?)",
            r"(?:across|over)\s+all\s+(?:files?|documents?|docs?)",
            r"multiple\s+(?:files?|documents?)",
        )
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in explicit_patterns)

    @staticmethod
    def _summary_scope_doc_limit(startup_profile) -> int:
        key = str(getattr(startup_profile, "value", startup_profile) or "").upper()
        if key == "FAST":
            return 8
        if key == "DEEP":
            return 16
        return 12

    def _expand_summary_citations(
        self,
        *,
        query: str,
        allowed_doc_ids: set[str],
        metadata_map: dict[str, dict],
        base_citations: list[Citation],
        max_files: int,
        max_chunks_per_file: int = 2,
    ) -> list[Citation]:
        if not allowed_doc_ids:
            return base_citations
        max_files = max(1, int(max_files))
        max_chunks_per_file = max(1, int(max_chunks_per_file))
        query_terms = [item.casefold() for item in self._tokenize_query_terms(query)]

        base_doc_scores: dict[str, float] = {}
        ordered_doc_ids: list[str] = []
        for citation in base_citations:
            existing = base_doc_scores.get(citation.doc_id)
            if existing is None:
                ordered_doc_ids.append(citation.doc_id)
                base_doc_scores[citation.doc_id] = float(citation.score)
            else:
                base_doc_scores[citation.doc_id] = max(existing, float(citation.score))

        now = datetime.now(timezone.utc)
        extra_docs: list[tuple[float, str]] = []
        for doc_id in allowed_doc_ids:
            if doc_id in base_doc_scores:
                continue
            row = metadata_map.get(doc_id) or {}
            path = str(row.get("path") or "").strip()
            if not path:
                continue
            summary = str(row.get("summary") or "")
            importance = float(row.get("importance", 0.5) or 0.5)
            score = 0.22 + min(max(importance, 0.0), 1.0) * 0.08
            low_path = path.casefold()
            low_summary = summary.casefold()
            for term in query_terms:
                if term in low_path:
                    score += 0.13
                if term in low_summary:
                    score += 0.06
            modified_at = row.get("modified_at")
            if isinstance(modified_at, datetime):
                age_days = max(0.0, (now - modified_at).total_seconds() / 86400.0)
                if age_days <= 30:
                    score += 0.05
                elif age_days <= 180:
                    score += 0.03
            extra_docs.append((score, doc_id))
        extra_docs.sort(key=lambda item: item[0], reverse=True)

        for _, doc_id in extra_docs:
            ordered_doc_ids.append(doc_id)
            if len(ordered_doc_ids) >= max_files:
                break
        if not ordered_doc_ids:
            return base_citations
        ordered_doc_ids = ordered_doc_ids[:max_files]

        rows = self._db.list_chunks_by_doc_ids(ordered_doc_ids)
        chunks_by_doc: dict[str, list] = {}
        for row in rows:
            doc_id = str(row["doc_id"])
            chunks_by_doc.setdefault(doc_id, []).append(row)
        for items in chunks_by_doc.values():
            items.sort(key=lambda item: int(item["chunk_order"] or 0))

        citations: list[Citation] = []
        for doc_index, doc_id in enumerate(ordered_doc_ids):
            meta = metadata_map.get(doc_id) or {}
            category = str(meta.get("category") or "참고자료")
            subcategory = str(meta.get("subcategory") or "")
            document_type = str(meta.get("document_type") or "")
            tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
            importance = float(meta.get("importance", 0.5) or 0.5)
            file_path = str(meta.get("path") or "")
            modified_default = meta.get("modified_at")
            if not isinstance(modified_default, datetime):
                modified_default = now

            rows_for_doc = chunks_by_doc.get(doc_id, [])
            picked_rows: list = []
            for row in rows_for_doc:
                text = str(row["text"] or "").strip()
                if not text:
                    continue
                picked_rows.append(row)
                break
            if len(picked_rows) < max_chunks_per_file:
                best_overlap = -1
                best_row = None
                selected_chunk_ids = {str(item["chunk_id"]) for item in picked_rows}
                for row in rows_for_doc:
                    chunk_id = str(row["chunk_id"] or "")
                    if not chunk_id or chunk_id in selected_chunk_ids:
                        continue
                    text = str(row["text"] or "").strip()
                    if not text:
                        continue
                    low_text = text.casefold()
                    overlap = sum(1 for term in query_terms if term and term in low_text)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_row = row
                if best_row is not None and (best_overlap > 0 or not picked_rows):
                    picked_rows.append(best_row)
                elif best_row is None and len(rows_for_doc) >= 2 and not picked_rows:
                    picked_rows.append(rows_for_doc[1])

            base_score = base_doc_scores.get(doc_id, max(0.3, 0.58 - (doc_index * 0.02)))
            if not picked_rows:
                snippet = self._clip_summary_snippet(str(meta.get("summary") or "") or Path(file_path).name)
                if not snippet:
                    continue
                citations.append(
                    Citation(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}:meta",
                        file_path=file_path or doc_id,
                        snippet=snippet,
                        score=max(0.16, min(0.92, base_score)),
                        modified_at=modified_default,
                        category=category,
                        subcategory=subcategory,
                        tags=[str(tag) for tag in tags][:8],
                        document_type=document_type,
                        importance=importance,
                    )
                )
                continue

            seen_snippets: set[str] = set()
            for row_index, row in enumerate(picked_rows[:max_chunks_per_file]):
                snippet = self._clip_summary_snippet(str(row["text"] or ""))
                if not snippet:
                    continue
                key = re.sub(r"[^\w가-힣]+", "", snippet).casefold()
                if key and key in seen_snippets:
                    continue
                if key:
                    seen_snippets.add(key)
                modified_raw = row["modified_at"]
                if isinstance(modified_raw, datetime):
                    modified_at = modified_raw
                else:
                    try:
                        modified_at = datetime.fromtimestamp(float(modified_raw), tz=timezone.utc)
                    except Exception:
                        modified_at = modified_default
                score = max(0.16, min(0.92, base_score - (row_index * 0.03)))
                citations.append(
                    Citation(
                        doc_id=doc_id,
                        chunk_id=str(row["chunk_id"] or f"{doc_id}:chunk{row_index}"),
                        file_path=str(row["path"] or file_path or doc_id),
                        snippet=snippet,
                        score=score,
                        modified_at=modified_at,
                        category=category,
                        subcategory=subcategory,
                        tags=[str(tag) for tag in tags][:8],
                        document_type=document_type,
                        importance=importance,
                    )
                )

        if not citations:
            return base_citations
        citations.sort(key=lambda item: item.score, reverse=True)
        return citations[: max_files * max_chunks_per_file]

    @staticmethod
    def _clip_summary_snippet(text: str, *, max_chars: int = 300) -> str:
        compact = re.sub(r"\s+", " ", (text or "").strip())
        if not compact:
            return ""
        if len(compact) <= max_chars:
            return compact
        head = compact[:max_chars]
        cut = head.rsplit(" ", 1)[0].strip()
        if not cut:
            cut = head.strip()
        return f"{cut}..."

    @classmethod
    def _fallback_file_citations(
        cls,
        *,
        query: str,
        allowed_doc_ids: set[str],
        metadata_map: dict[str, dict],
        limit: int = 140,
    ) -> list[Citation]:
        terms = cls._tokenize_query_terms(query)
        ranked: list[tuple[float, Citation]] = []
        now_ts = time.time()
        for doc_id in allowed_doc_ids:
            row = metadata_map.get(doc_id) or {}
            file_path = str(row.get("path") or "")
            if not file_path:
                continue
            category = str(row.get("category") or "참고자료")
            tags = row.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            summary = str(row.get("summary") or "")
            subcategory = str(row.get("subcategory") or "")
            document_type = str(row.get("document_type") or "")
            modified_at = row.get("modified_at")
            if modified_at is None:
                continue
            score = 0.28
            path_lc = file_path.casefold()
            summary_lc = summary.casefold()
            category_lc = f"{category} {subcategory} {document_type}".casefold()
            tag_set = {str(tag).casefold() for tag in tags}
            for term in terms:
                key = term.casefold()
                if key in path_lc:
                    score += 0.14
                if key in summary_lc:
                    score += 0.08
                if any(key in tag for tag in tag_set):
                    score += 0.10
                if key in category_lc:
                    score += 0.06

            age_days = max(0.0, (now_ts - modified_at.timestamp()) / 86400.0)
            if age_days <= 30:
                score += 0.08
            elif age_days <= 180:
                score += 0.04
            score = max(0.01, min(score, 0.92))

            snippet = summary.strip() or Path(file_path).name
            if len(snippet) > 260:
                snippet = snippet[:260].rstrip() + "..."
            citation = Citation(
                doc_id=doc_id,
                chunk_id=f"{doc_id}:meta",
                file_path=file_path,
                snippet=snippet,
                score=score,
                modified_at=modified_at,
                category=category,
                subcategory=subcategory,
                tags=[str(tag) for tag in tags][:8],
                document_type=document_type,
                importance=float(row.get("importance", 0.5) or 0.5),
            )
            ranked.append((score, citation))
        ranked.sort(key=lambda item: item[0], reverse=True)
        cap = max(10, min(int(limit), 220))
        return [item[1] for item in ranked[:cap]]

    @staticmethod
    def _merge_find_file_citations(
        *,
        primary: list[Citation],
        fallback: list[Citation],
        limit: int,
    ) -> list[Citation]:
        merged: dict[str, Citation] = {}
        for item in [*primary, *fallback]:
            existing = merged.get(item.doc_id)
            if existing is None or float(item.score) > float(existing.score):
                merged[item.doc_id] = item
        output = sorted(merged.values(), key=lambda item: float(item.score), reverse=True)
        cap = max(10, min(int(limit), 220))
        return output[:cap]

    @classmethod
    def _extract_path_focus_terms(cls, *, query: str, topics: list[str]) -> tuple[list[str], bool]:
        text = (query or "").strip()
        if not text:
            return [], False

        lowered = text.lower()
        strict_focus = any(token in lowered for token in ("폴더", "디렉토리", "folder", "directory"))

        found: list[str] = []
        patterns = (
            r"([A-Za-z가-힣0-9_+\-]{2,32})\s*(?:폴더|디렉토리)",
            r"(?:folder|directory)\s*[:\-]?\s*([A-Za-z가-힣0-9_+\-]{2,32})",
            r"([A-Za-z가-힣0-9_+\-]{2,32})에서",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidate = (match.group(1) or "").strip()
                if cls._is_focus_candidate(candidate):
                    found.append(candidate)

        if strict_focus:
            for topic in topics:
                if cls._is_focus_candidate(topic):
                    found.append(topic)
                if len(found) >= 3:
                    break

        unique: list[str] = []
        seen: set[str] = set()
        for token in found:
            key = unicodedata.normalize("NFC", token).casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(token)
            if len(unique) >= 3:
                break
        return unique, strict_focus

    @classmethod
    def _is_focus_candidate(cls, token: str) -> bool:
        value = (token or "").strip()
        if len(value) < 2:
            return False
        if value.lower() in cls._FOCUS_STOPWORDS:
            return False
        return True

    @staticmethod
    def _normalize_variants(value: str) -> set[str]:
        raw = (value or "").strip()
        if not raw:
            return set()
        nfc = unicodedata.normalize("NFC", raw).casefold()
        nfd = unicodedata.normalize("NFD", raw).casefold()
        variants = {nfc, nfd, nfc.replace(" ", ""), nfd.replace(" ", "")}
        return {item for item in variants if item}

    @classmethod
    def _filter_doc_ids_by_path_focus(
        cls,
        *,
        doc_ids: set[str],
        metadata_map: dict[str, dict],
        focus_terms: list[str],
    ) -> set[str]:
        if not doc_ids or not focus_terms:
            return set()

        normalized_terms = [cls._normalize_variants(term) for term in focus_terms]
        normalized_terms = [terms for terms in normalized_terms if terms]
        if not normalized_terms:
            return set()

        filtered: set[str] = set()
        for doc_id in doc_ids:
            path = str((metadata_map.get(doc_id) or {}).get("path") or "")
            if not path:
                continue
            path_variants = cls._normalize_variants(path)
            if not path_variants:
                continue
            matched = False
            for term_variants in normalized_terms:
                if any(term in path_variant for term in term_variants for path_variant in path_variants):
                    matched = True
                    break
            if matched:
                filtered.add(doc_id)
        return filtered
