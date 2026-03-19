from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import time
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


class ReasoningPipeline:
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

    def run(self, req: LocalChatRequestV2) -> ComposedChatResponseV2:
        workspace = self._db.get_workspace()
        settings = self._db.get_settings()
        response_language = resolve_response_language(req.query, settings.language)
        session_id = req.session_id or req.conversation_id or "default-session"
        workspace_identity = self._memory.get_workspace_identity()

        parsed_intent = self._intent_parser.parse(query=req.query, mode=req.mode, workspace=workspace)
        memory_bundle = self._memory.get_relevant_memory_bundle(
            session_id=session_id,
            workspace_id=workspace_identity.workspace_id,
            intent=parsed_intent.intent.value,
            related_file_ids=[],
        )
        last_context = self._memory.getLastConversationalContext(session_id)
        followup_resolution = self._followup.resolve(
            query=req.query,
            parsed_intent=parsed_intent,
            mode=req.mode,
            last_context=last_context,
            last_candidates=self._memory.getLastCandidateSet(session_id),
            last_selected_file=self._memory.getLastSelectedFile(session_id),
            last_actions=self._memory.getLastShownActions(session_id),
        )
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
            )
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
        metadata_fallback_used = False

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

        if self._should_trigger_auto_index(
            req=req,
            parsed_intent=parsed_intent.intent,
            allowed_doc_ids=allowed_doc_ids,
            strict_focus=strict_focus,
        ):
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
        query_vector = self._embedding.embed_query(effective_query)

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
            explicit_top_k=req.top_k,
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
        plan = self._planner.build_plan(
            parsed_intent=parsed_intent,
            mode=req.mode,
            file_doc_ids=file_doc_ids,
            chunk_ids=chunk_ids,
            top_score=top_score,
        )
        if self._should_short_circuit_candidate(
            mode=req.mode,
            top_score=top_score,
            intent=parsed_intent.intent,
            file_count=len(file_doc_ids),
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
        if metadata_fallback_used:
            execution.tool_logs.append("fallback:metadata_file_search")
            execution.structured_payload["metadata_fallback_used"] = True
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

        top_file_ids = [item.doc_id for item in retrieval.file_candidates[:3]]
        self._memory.writeMemoryEvent(
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
            self._memory.writeMemoryEvent(
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
            conversation_path="local_rag",
            escalated_provider=None,
            is_local=True,
        )
        self._memory.write_conversational_context(
            session_id=session_id,
            context={
                "intent": parsed_intent.intent.value,
                "top_candidates": [item.file_path for item in citations[:5]],
                "candidate_doc_ids": [item.doc_id for item in citations[:5]],
                "filters": merged_filters.model_dump(),
                "shown_actions": [item.kind.value for item in composed.actions],
                "result_summary": composed.structured_result.summary[:260],
                "ambiguity": verification.ambiguity_level,
                "used_clarification": bool(composed.metadata.get("used_clarification", False)),
                "response_mode": composed.response_mode,
                "selected_file": (citations[0].file_path if citations else None),
            },
        )
        return composed

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
    ) -> bool:
        if mode == WorkMode.STRICT_SEARCH:
            return top_score < 0.6
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
            text = "현재 자료로는 확정 단정이 어려워서 먼저 가능성이 높은 후보부터 보여드립니다. 파일명/기간/태그를 조금만 더 주시면 더 정확하게 찾아드리겠습니다."
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
    ) -> ComposedChatResponseV2:
        plan = LocalPlan(
            plan_type="conversation",
            selected_files=[],
            selected_chunks=[],
            response_strategy="conversational_assistant",
            allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
            external_reasoning_needed=False,
        )
        context_summary = ""
        if not self._is_greeting_query(req.query):
            context_summary = self._conversation_session_summary(
                memory_bundle=memory_bundle,
                last_context=last_context,
            )
        execution = self._executor.execute_conversation(
            query=req.query,
            mode=req.mode,
            startup_profile=workspace.startup_profile,
            engine=settings.local_engine,
            mlx_model_path=settings.mlx_model_path,
            llama_model_path=settings.llama_model_path,
            language_preference=settings.language,
            session_summary=context_summary,
            max_tokens=160,
        )
        execution.tool_logs.insert(0, f"router:intent={ReasoningIntent.GENERAL_CHAT.value}")
        execution.tool_logs.append("agent:conversation_assistant")
        conversation_path = "local_conversation"
        escalated_provider: str | None = None
        is_local = True

        if not execution.generated_text:
            if settings.privacy_mode == PrivacyMode.HYBRID:
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
                else:
                    execution = self._runtime_error_execution(response_language, execution.runtime_detail)
                    execution.tool_logs.append("runtime_error:conversation_local_and_external_failed")
            else:
                execution = self._runtime_error_execution(response_language, execution.runtime_detail)
                execution.tool_logs.append("runtime_error:conversation_local_failed_local_only")

        verification = VerificationResult(
            is_valid=(execution.result_type != "runtime_error"),
            confidence=(0.84 if execution.result_type == "conversation" else 0.2),
            issues=([] if execution.result_type == "conversation" else ["runtime_unavailable"]),
            ambiguity_level=(0.16 if execution.result_type == "conversation" else 0.8),
            candidate_mode=False,
        )

        self._memory.writeMemoryEvent(
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
        self._memory.write_conversational_context(
            session_id=session_id,
            context={
                "intent": ReasoningIntent.GENERAL_CHAT.value,
                "top_candidates": [],
                "candidate_doc_ids": [],
                "filters": {},
                "shown_actions": [item.kind.value for item in composed.actions],
                "result_summary": composed.structured_result.summary[:260],
                "ambiguity": verification.ambiguity_level,
                "used_clarification": False,
                "response_mode": composed.response_mode,
                "selected_file": None,
            },
        )
        return composed

    @staticmethod
    def _is_greeting_query(query: str) -> bool:
        lowered = (query or "").strip().lower()
        if not lowered:
            return False
        return any(
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
        )

    @staticmethod
    def _conversation_session_summary(*, memory_bundle, last_context: dict | None) -> str:
        lines: list[str] = []
        if last_context:
            summary = str(last_context.get("result_summary") or "").strip()
            if summary:
                lines.append(f"- 직전 응답 요약: {summary[:220]}")
            selected = str(last_context.get("selected_file") or "").strip()
            if selected:
                lines.append(f"- 직전 선택 파일: {Path(selected).name}")
        for item in memory_bundle.session_items:
            if item.key != "recent_query":
                continue
            query_summary = str(item.value_json.get("summary") or "").strip()
            if query_summary:
                lines.append(f"- 최근 질문: {query_summary[:160]}")
            if len(lines) >= 3:
                break
        return "\n".join(lines[:3]).strip()

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
            deadline = now + 15.0
            while time.monotonic() < deadline:
                status = self._indexing.get_job(job.job_id)
                if status is None:
                    return False
                if status.status == "completed":
                    return True
                if status.status == "failed":
                    return False
                time.sleep(0.2)
        except Exception:
            return False
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

    @classmethod
    def _fallback_file_citations(
        cls,
        *,
        query: str,
        allowed_doc_ids: set[str],
        metadata_map: dict[str, dict],
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
        return [item[1] for item in ranked[:8]]

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
