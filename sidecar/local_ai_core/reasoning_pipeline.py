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

from .clarification_budget import ClarificationBudget, ClarificationBudgetState
from .db import Database
from .embedding import EmbeddingService
from .executor import LocalExecutor
from .followup_resolver import FollowUpResolution, FollowUpResolver
from .intent_parser import IntentParser
from .language_utils import insufficient_evidence_message, resolve_response_language
from .local_planner import LocalPlanner
from .memory_service import MemoryService
from .models import (
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
from .response_composer import ResponseComposer
from .retrieval import extract_query_hints, merge_filters, retrieve_bundle
from .vector_store import VectorStore
from .verifier import ResultVerifier


from .pipeline import reasoning_mixins as _reasoning_pipeline_mixins

class ReasoningPipeline(_reasoning_pipeline_mixins.ReasoningPipelineMethodsMixin):
    _FOCUS_STOPWORDS = {
        "폴더",
        "디렉토리",
        "자료",
        "문서",
        "파일",
        "강의",
        "노트",
        "folder",
        "directory",
        "files",
        "file",
        "docs",
        "document",
    }

    def __init__(
        self,
        *,
        db: Database,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        local_inference,
        composer: ResponseComposer,
        memory_service: MemoryService,
        provider_router=None,
        indexing_service=None,
    ):
        self._db = db
        self._vector_store = vector_store
        self._embedding = embedding_service
        self._composer = composer
        self._intent_parser = IntentParser()
        self._planner = LocalPlanner()
        self._executor = LocalExecutor(local_inference)
        self._verifier = ResultVerifier()
        self._memory = memory_service
        self._providers = provider_router
        self._followup = FollowUpResolver()
        self._clarification_budget = ClarificationBudget()
        self._indexing = indexing_service
        self._last_auto_index_started_at = 0.0
        self._memory.set_digest_model_refresher(self._refresh_digest_with_local_model)
        try:
            self._memory.clear_session_context_memory_once()
        except Exception:
            # Startup hygiene should not block runtime initialization.
            pass
        self._quality_log_enabled = self._flag_enabled(
            os.getenv("LOCAL_AI_QUALITY_LOG_ENABLED"),
            default=True,
        )
        self._quality_log_window = self._parse_positive_int(
            os.getenv("LOCAL_AI_QUALITY_LOG_WINDOW"),
            default=50,
            minimum=10,
            maximum=200,
        )
        data_dir = Path(os.getenv("LOCAL_AI_DATA_DIR", "./data")).expanduser().resolve()
        self._quality_log_path = data_dir / "conversation_quality.jsonl"
        self._quality_rollup_window: deque[dict[str, Any]] = deque(
            self._load_recent_quality_events(self._quality_log_path, limit=self._quality_log_window),
            maxlen=self._quality_log_window,
        )

    def run(self, req: LocalChatRequestV2) -> ComposedChatResponseV2:
        workspace = self._db.get_workspace()
        settings = self._db.get_settings()
        response_language = resolve_response_language(req.query, settings.language)
        session_id = req.session_id or req.conversation_id or "default-session"
        workspace_identity = self._memory.get_workspace_identity()

        parsed_intent = self._intent_parser.parse(query=req.query, mode=req.mode, workspace=workspace)
        if self._should_force_general_chat(query=req.query, parsed_intent=parsed_intent):
            parsed_intent.intent = ReasoningIntent.GENERAL_CHAT
            parsed_intent.operation = "chat"
            parsed_intent.scope = "single"
            parsed_intent.target = None
            parsed_intent.ambiguity = "clear"
            parsed_intent.confidence = max(parsed_intent.confidence, 0.9)
        memory_bundle = self._memory.get_relevant_memory_bundle(
            session_id=session_id,
            workspace_id=workspace_identity.workspace_id,
            intent=parsed_intent.intent.value,
            related_file_ids=[],
        )
        last_context = self._memory.get_last_conversational_context(session_id)
        session_digest = self._memory.get_session_digest(session_id)
        followup_resolution = self._followup.resolve(
            query=req.query,
            parsed_intent=parsed_intent,
            mode=req.mode,
            last_context=last_context,
            last_candidates=self._memory.get_last_candidate_set(session_id),
            last_selected_file=self._memory.get_last_selected_file(session_id),
            last_actions=self._memory.get_last_shown_actions(session_id),
        )
        needs_scope_target_clarification = followup_resolution.followup_type == "clarify_scope_target"
        if followup_resolution.is_followup and followup_resolution.resolved_intent is not None:
            parsed_intent.intent = followup_resolution.resolved_intent
            parsed_intent.confidence = max(parsed_intent.confidence, min(0.92, followup_resolution.confidence))
        memory_prefs = self._memory.resolve_preferences(memory_bundle)
        effective_query = self._effective_query(
            session_id=session_id,
            query=req.query,
            parsed_intent=parsed_intent,
            memory_bundle=memory_bundle,
            followup_resolution=followup_resolution,
        )
        behavior_policy = self._effective_behavior_policy(
            req=req,
            memory_bundle=memory_bundle,
            default_action_order=memory_prefs.default_action_order,
            default_mode=memory_prefs.default_mode,
            workspace_weights=memory_prefs.workspace_weights,
        )
        if needs_scope_target_clarification:
            return self._compose_scope_target_clarification(
                req=req,
                response_language=response_language,
                parsed_intent=parsed_intent,
                behavior_policy=behavior_policy,
                memory_prefs=memory_prefs,
                workspace=workspace,
                session_id=session_id,
                workspace_id=workspace_identity.workspace_id,
                conversation_path="scope_target_clarification",
            )
        if parsed_intent.intent == ReasoningIntent.GENERAL_CHAT:
            return self._run_general_chat(
                req=req,
                settings=settings,
                workspace=workspace,
                session_id=session_id,
                workspace_id=workspace_identity.workspace_id,
                response_language=response_language,
                parsed_intent=parsed_intent,
                behavior_policy=behavior_policy,
                memory_prefs=memory_prefs,
                memory_bundle=memory_bundle,
                last_context=last_context,
                session_digest=session_digest,
            )
        prefer_multi_file_summary = self._should_expand_summary_scope(
            query=req.query,
            parsed_intent=parsed_intent,
        )
        prefer_focused_file_summary = False
        hint_filters = extract_query_hints(req.query)
        merged_filters = merge_filters(req.filters, hint_filters) or ChatFilters()
        if followup_resolution.resolved_filters:
            if merged_filters.year is None and isinstance(followup_resolution.resolved_filters.get("year"), int):
                merged_filters.year = int(followup_resolution.resolved_filters["year"])
        if parsed_intent.time_filters.year and merged_filters.year is None:
            merged_filters.year = parsed_intent.time_filters.year
        if merged_filters.excluded is None:
            merged_filters.excluded = False

        auto_indexed = False
        auto_index_triggered = False
        metadata_fallback_used = False
        summary_scope_expanded = False
        explicit_file_focus_used = False
        focused_file_summary_used = False
        week_exact_filter_applied = False
        week_exact_no_match = False

        allowed_doc_ids, metadata_map = self._resolve_workspace_docs(
            workspace=workspace,
            filters=merged_filters,
        )
        if req.filters is None and not allowed_doc_ids:
            merged_filters = ChatFilters(excluded=False)
            allowed_doc_ids, metadata_map = self._resolve_workspace_docs(
                workspace=workspace,
                filters=merged_filters,
            )
        focus_terms, strict_focus = self._extract_path_focus_terms(
            query=req.query,
            topics=parsed_intent.entities.topics,
        )
        allowed_doc_ids, metadata_map = self._apply_focus_filter(
            doc_ids=allowed_doc_ids,
            metadata_map=metadata_map,
            focus_terms=focus_terms,
            strict_focus=strict_focus,
        )
        if followup_resolution.resolved_target_files:
            resolved_set = set(followup_resolution.resolved_target_files)
            matched = set()
            for doc_id in allowed_doc_ids:
                row = metadata_map.get(doc_id) or {}
                path = str(row.get("path") or "")
                if path in resolved_set or doc_id in resolved_set:
                    matched.add(doc_id)
            if matched:
                allowed_doc_ids = matched
                metadata_map = {doc_id: row for doc_id, row in metadata_map.items() if doc_id in matched}
        explicit_file_terms = self._extract_explicit_file_terms(
            query=req.query,
            parsed_intent=parsed_intent,
        )
        if explicit_file_terms:
            allowed_doc_ids, metadata_map, explicit_file_focus_used = self._apply_explicit_file_focus(
                doc_ids=allowed_doc_ids,
                metadata_map=metadata_map,
                file_terms=explicit_file_terms,
            )
        requested_weeks = self._extract_requested_weeks(
            query=req.query,
            followup_resolution=followup_resolution,
        )
        allowed_doc_ids, metadata_map, week_exact_filter_applied, week_exact_no_match = self._apply_week_exact_filter(
            doc_ids=allowed_doc_ids,
            metadata_map=metadata_map,
            requested_weeks=requested_weeks,
        )
        if parsed_intent.intent == ReasoningIntent.SUMMARIZE_FILE and explicit_file_terms and not prefer_multi_file_summary:
            prefer_focused_file_summary = True

        if self._should_trigger_auto_index(
            req=req,
            parsed_intent=parsed_intent.intent,
            allowed_doc_ids=allowed_doc_ids,
            strict_focus=strict_focus,
        ):
            auto_index_triggered = True
            auto_indexed = self._run_auto_index(workspace)
            if auto_indexed:
                allowed_doc_ids, metadata_map = self._resolve_workspace_docs(
                    workspace=workspace,
                    filters=merged_filters,
                )
                allowed_doc_ids, metadata_map = self._apply_focus_filter(
                    doc_ids=allowed_doc_ids,
                    metadata_map=metadata_map,
                    focus_terms=focus_terms,
                    strict_focus=strict_focus,
                )
                if explicit_file_terms:
                    allowed_doc_ids, metadata_map, explicit_file_focus_used = self._apply_explicit_file_focus(
                        doc_ids=allowed_doc_ids,
                        metadata_map=metadata_map,
                        file_terms=explicit_file_terms,
                    )
        query_vector = self._embedding.embed_query(effective_query)
        effective_top_k = req.top_k
        if effective_top_k is None and parsed_intent.intent == ReasoningIntent.FIND_FILE:
            effective_top_k = min(64, max(20, len(allowed_doc_ids)))
        if effective_top_k is None and prefer_multi_file_summary:
            effective_top_k = min(28, max(12, len(allowed_doc_ids)))
        preset, retrieval = retrieve_bundle(
            vector_store=self._vector_store,
            query_vector=query_vector,
            mode=req.mode,
            startup_profile=workspace.startup_profile,
            query=effective_query,
            allowed_doc_ids=allowed_doc_ids,
            filters=merged_filters,
            metadata_map=metadata_map,
            behavior_policy=behavior_policy,
            explicit_top_k=effective_top_k,
        )
        _ = preset  # explicit for readability

        citations = [
            self._citation_from_chunk(chunk, metadata_map)
            for chunk in retrieval.chunk_candidates
        ]
        file_doc_ids = [item.doc_id for item in retrieval.file_candidates]
        chunk_ids = [item.chunk_id for item in retrieval.chunk_candidates]
        top_score = retrieval.chunk_candidates[0].score if retrieval.chunk_candidates else 0.0
        if parsed_intent.intent == ReasoningIntent.FIND_FILE and not file_doc_ids and allowed_doc_ids:
            fallback_citations = self._fallback_file_citations(
                query=effective_query,
                allowed_doc_ids=allowed_doc_ids,
                metadata_map=metadata_map,
            )
            if fallback_citations:
                citations = fallback_citations
                file_doc_ids = [item.doc_id for item in fallback_citations]
                chunk_ids = [item.chunk_id for item in fallback_citations]
                top_score = max(top_score, fallback_citations[0].score)
                metadata_fallback_used = True
        if parsed_intent.intent == ReasoningIntent.FIND_FILE and allowed_doc_ids:
            fallback_citations = self._fallback_file_citations(
                query=effective_query,
                allowed_doc_ids=allowed_doc_ids,
                metadata_map=metadata_map,
            )
            if fallback_citations:
                citations = self._merge_find_file_citations(
                    primary=citations,
                    fallback=fallback_citations,
                    limit=140,
                )
                file_doc_ids = []
                seen_doc_ids: set[str] = set()
                for item in citations:
                    if item.doc_id in seen_doc_ids:
                        continue
                    seen_doc_ids.add(item.doc_id)
                    file_doc_ids.append(item.doc_id)
                chunk_ids = [item.chunk_id for item in citations]
                top_score = citations[0].score if citations else top_score
                metadata_fallback_used = True
        if prefer_focused_file_summary and allowed_doc_ids:
            target_doc_ids = file_doc_ids[:2] if file_doc_ids else sorted(allowed_doc_ids)[:2]
            focused = self._build_focused_file_summary_citations(
                doc_ids=target_doc_ids,
                metadata_map=metadata_map,
                max_chunks_per_file=self._focused_summary_chunk_limit(workspace.startup_profile),
            )
            if focused:
                focused_file_summary_used = True
                citations = focused
                file_doc_ids = []
                seen_doc_ids: set[str] = set()
                for item in citations:
                    if item.doc_id in seen_doc_ids:
                        continue
                    seen_doc_ids.add(item.doc_id)
                    file_doc_ids.append(item.doc_id)
                chunk_ids = [item.chunk_id for item in citations]
                top_score = citations[0].score if citations else top_score
        if prefer_multi_file_summary and allowed_doc_ids:
            expanded = self._expand_summary_citations(
                query=effective_query,
                allowed_doc_ids=allowed_doc_ids,
                metadata_map=metadata_map,
                base_citations=citations,
                max_files=self._summary_scope_doc_limit(workspace.startup_profile),
                max_chunks_per_file=2,
            )
            if expanded:
                summary_scope_expanded = True
                citations = expanded
                file_doc_ids = []
                seen_doc_ids: set[str] = set()
                for item in citations:
                    if item.doc_id in seen_doc_ids:
                        continue
                    seen_doc_ids.add(item.doc_id)
                    file_doc_ids.append(item.doc_id)
                chunk_ids = [item.chunk_id for item in citations]
                top_score = citations[0].score if citations else top_score
        plan = self._planner.build_plan(
            parsed_intent=parsed_intent,
            mode=req.mode,
            file_doc_ids=file_doc_ids,
            chunk_ids=chunk_ids,
            top_score=top_score,
            prefer_multi_file_summary=prefer_multi_file_summary,
            prefer_focused_file_summary=prefer_focused_file_summary,
        )
        if self._should_short_circuit_candidate(
            mode=req.mode,
            top_score=top_score,
            intent=parsed_intent.intent,
            file_count=len(file_doc_ids),
            force_multi_file_summary=prefer_multi_file_summary,
            force_focused_file_summary=prefer_focused_file_summary,
        ):
            execution = self._build_candidate_execution(
                response_language=response_language,
                citations=citations,
                reason="low_relevance_precheck",
            )
        else:
            execution = self._executor.execute(
                query=effective_query,
                mode=req.mode,
                parsed_intent=parsed_intent.intent,
                plan=plan,
                citations=citations,
                startup_profile=workspace.startup_profile,
                engine=settings.local_engine,
                mlx_model_path=settings.mlx_model_path,
                llama_model_path=settings.llama_model_path,
                language_preference=settings.language,
                response_length=memory_prefs.response_length,
            )
        execution.tool_logs.append(f"router:plan={plan.plan_type}")
        execution.tool_logs.append("agent:reasoner_primary")
        if auto_indexed:
            execution.tool_logs.append("auto_index:incremental")
            execution.structured_payload["auto_indexed"] = True
        elif auto_index_triggered:
            execution.tool_logs.append("auto_index:started")
            execution.structured_payload["auto_indexing_started"] = True
        if metadata_fallback_used:
            execution.tool_logs.append("fallback:metadata_file_search")
            execution.structured_payload["metadata_fallback_used"] = True
        if week_exact_filter_applied:
            execution.tool_logs.append("focus:week_exact")
            execution.structured_payload["week_exact_filter_used"] = True
            execution.structured_payload["requested_weeks"] = requested_weeks
            if week_exact_no_match:
                execution.structured_payload["week_exact_no_match"] = True
        if explicit_file_focus_used:
            execution.tool_logs.append("focus:explicit_file_name")
            execution.structured_payload["explicit_file_focus_used"] = True
        if focused_file_summary_used:
            execution.tool_logs.append("summary_scope:focused_file")
            execution.structured_payload["focused_file_summary_used"] = True
        if summary_scope_expanded:
            execution.tool_logs.append("summary_scope:multi_file")
            execution.structured_payload["summary_scope_expanded"] = True
        verification = self._verifier.verify(
            parsed_intent=parsed_intent,
            execution_result=execution,
            mode=req.mode,
        )
        if self._should_run_secondary_reasoner(
            mode=req.mode,
            parsed_intent=parsed_intent,
            execution=execution,
            verification=verification,
        ):
            refined_query = self._build_secondary_reasoner_prompt(
                query=req.query,
                parsed_intent=parsed_intent,
                response_language=response_language,
            )
            refined_execution = self._executor.execute(
                query=refined_query,
                mode=req.mode,
                parsed_intent=parsed_intent.intent,
                plan=plan,
                citations=citations,
                startup_profile=workspace.startup_profile,
                engine=settings.local_engine,
                mlx_model_path=settings.mlx_model_path,
                llama_model_path=settings.llama_model_path,
                language_preference=settings.language,
                response_length=memory_prefs.response_length,
            )
            refined_execution.tool_logs.append("agent:reasoner_secondary")
            refined_verification = self._verifier.verify(
                parsed_intent=parsed_intent,
                execution_result=refined_execution,
                mode=req.mode,
            )
            execution, verification = self._pick_better_reasoner_result(
                primary_execution=execution,
                primary_verification=verification,
                secondary_execution=refined_execution,
                secondary_verification=refined_verification,
            )
        execution.tool_logs.append("agent:verifier_multistage")

        conversation_path = "local_rag"
        escalated_provider: str | None = None
        is_local = True
        if self._should_escalate_summary_to_external(
            req=req,
            parsed_intent=parsed_intent,
            settings=settings,
            citations=citations,
        ):
            escalated = self._escalate_summary_to_external(
                query=effective_query,
                mode=req.mode,
                citations=citations,
                settings=settings,
            )
            if escalated is not None:
                execution, escalated_provider = escalated
                verification = VerificationResult(
                    is_valid=True,
                    confidence=max(float(verification.confidence), 0.86),
                    issues=[],
                    ambiguity_level=min(float(verification.ambiguity_level), 0.18),
                    candidate_mode=False,
                )
                conversation_path = "external_summary_escalated"
                is_local = False
            else:
                execution.tool_logs.append("external_escalation:summary_unavailable")

        candidate_gap_small = self._candidate_gap_small(citations)
        clarification_state = ClarificationBudgetState(
            clarification_count_current_turn=0,
            previous_turn_was_clarification=bool((last_context or {}).get("used_clarification", False)),
            partial_user_answer_received=followup_resolution.is_followup,
        )
        risk_level = "low"
        if parsed_intent.intent in {ReasoningIntent.DRAFT_EDIT}:
            risk_level = "medium"
        allow_clarification = self._clarification_budget.allow_clarification(
            state=clarification_state,
            query=req.query,
            ambiguity_level=verification.ambiguity_level,
            risk_level=risk_level,
            candidate_gap_small=candidate_gap_small,
        )

        if (
            req.mode != WorkMode.STRICT_SEARCH
            and verification.candidate_mode
            and execution.result_type != "file_list"
            and not bool(execution.structured_payload.get("ungrounded_allowed", False))
            and (
                execution.used_fallback
                or not execution.generated_text.strip()
                or execution.result_type in {"candidate", "insufficient"}
                or self._looks_like_reasoning_leak(execution.generated_text)
            )
        ):
            execution = self._build_candidate_execution(
                response_language=response_language,
                citations=citations,
                reason="verification_candidate_mode",
            )

        if req.mode == WorkMode.STRICT_SEARCH and (not execution.citations or verification.candidate_mode):
            execution.generated_text = insufficient_evidence_message(response_language)
            execution.result_type = "insufficient"
            execution.structured_payload = {"reason": "strict_threshold_not_met"}
        if execution.result_type == "file_list":
            execution.structured_payload.setdefault("requested_scope", str(getattr(parsed_intent, "scope", "single") or "single"))
            execution.structured_payload.setdefault("requested_operation", str(getattr(parsed_intent, "operation", "find") or "find"))
            if getattr(parsed_intent, "target", None):
                execution.structured_payload.setdefault("requested_target", str(parsed_intent.target))

        top_file_ids = [item.doc_id for item in retrieval.file_candidates[:3]]
        self._memory.write_memory_event(
            MemoryEventRequest(
                event_type=MemoryEventType.QUERY,
                session_id=session_id,
                workspace_id=workspace_identity.workspace_id,
                summary=req.query[:220],
                related_file_ids=top_file_ids,
                metadata_json={
                    "mode": req.mode.value,
                    "intent": parsed_intent.intent.value,
                    "result_type": execution.result_type,
                },
                importance=0.45,
            )
        )
        outcome_type = self._outcome_event_type(execution.result_type)
        if outcome_type is not None:
            self._memory.write_memory_event(
                MemoryEventRequest(
                    event_type=outcome_type,
                    session_id=session_id,
                    workspace_id=workspace_identity.workspace_id,
                    summary=f"{execution.result_type} generated",
                    related_file_ids=top_file_ids,
                    metadata_json={"mode": req.mode.value, "result_type": execution.result_type},
                    importance=0.5,
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
            show_citations=memory_prefs.show_citations,
            prefer_action_suggestions=memory_prefs.prefer_action_suggestions,
            used_profile=workspace.startup_profile,
            engine_used=execution.engine_used,
            used_fallback=execution.used_fallback,
            runtime_detail=execution.runtime_detail,
            followup_resolution=followup_resolution,
            allow_clarification=allow_clarification,
            conversation_path=conversation_path,
            escalated_provider=escalated_provider,
            is_local=is_local,
        )
        conversation_summary = str(composed.structured_result.summary or "").strip()
        if self._looks_like_reasoning_leak(conversation_summary):
            conversation_summary = ""
        self._memory.write_conversational_context(
            session_id=session_id,
            context={
                "intent": parsed_intent.intent.value,
                "top_candidates": [item.file_path for item in citations[:5]],
                "candidate_doc_ids": [item.doc_id for item in citations[:5]],
                "filters": merged_filters.model_dump(),
                "shown_actions": [item.kind.value for item in composed.actions],
                "result_summary": conversation_summary[:260],
                "ambiguity": verification.ambiguity_level,
                "used_clarification": bool(composed.metadata.get("used_clarification", False)),
                "response_mode": composed.response_mode,
                "selected_file": (citations[0].file_path if citations else None),
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
            context_digest_used=False,
        )
        return composed


_reasoning_pipeline_mixins.ReasoningPipeline = ReasoningPipeline
