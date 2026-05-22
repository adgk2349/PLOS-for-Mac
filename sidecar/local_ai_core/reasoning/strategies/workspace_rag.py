import logging
import re
import json
import asyncio
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
from .base import ReasoningStrategy
from .workspace_rag_components import WorkspaceRagComponents
from .workspace_rag_development import WorkspaceRagDevelopment
from .workspace_rag_modules import (
    WorkspaceRagPrompter,
    WorkspaceRagRetriever,
    WorkspaceRagReranker,
    WorkspaceRagSearchFlow,
    WorkspaceRagMaterializer,
    WorkspaceRagFinalizer,
)
from ..context import ReasoningContext
from .. import utils
from ...models import *
from ...nlu.clarification_budget import ClarificationBudgetState
from ...nlu.followup_resolver import FollowUpResolution
from ...retrieval import extract_query_hints, merge_filters, retrieve_bundle
from ...language_utils import insufficient_evidence_message
from ...storage.async_adapter import AsyncAdapter


class WorkspaceRagStrategy(ReasoningStrategy):
    _CODE_EXTENSIONS = {
        ".swift", ".m", ".mm", ".h", ".hpp", ".c", ".cc", ".cpp",
        ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
        ".rb", ".php", ".cs", ".scala", ".sql", ".sh", ".zsh", ".bash",
        ".yaml", ".yml", ".json", ".toml", ".ini", ".xml",
    }
    _DEVELOPMENT_BATCH_FILE_LIMIT = 6
    _DEVELOPMENT_BATCH_CHUNKS_PER_FILE = 3
    _DEVELOPMENT_BATCH_CHAR_LIMIT = 7500
    _DEVELOPMENT_BATCH_MAX_FILES = 30
    _DEVELOPMENT_BATCH_TRIGGER_FILES = 12
    _DEVELOPMENT_BATCH_TRIGGER_CITATIONS = 36
    _DEVELOPMENT_BATCH_TIMEOUT_SEC = 14.0
    _DEVELOPMENT_PATCH_MAX_ITEMS = 5

    def __init__(self):
        self._development_batch_lock = asyncio.Lock()
        self._async_adapter = AsyncAdapter()
        self._prompter = WorkspaceRagPrompter()
        self._retriever = WorkspaceRagRetriever(
            file_limit=self._DEVELOPMENT_BATCH_FILE_LIMIT,
            chunks_per_file=self._DEVELOPMENT_BATCH_CHUNKS_PER_FILE,
            char_limit=self._DEVELOPMENT_BATCH_CHAR_LIMIT,
            max_files=self._DEVELOPMENT_BATCH_MAX_FILES,
        )
        self._reranker = WorkspaceRagReranker()

    async def _run_development_batch_review(
        self,
        *,
        executor,
        settings,
        workspace,
        req: LocalChatRequestV2,
        response_language: str,
        citations: list[Citation],
    ) -> tuple[list[dict[str, Any]], list[str], int, list[str]]:
        batches = self._retriever.build_development_batches(citations)
        if not batches:
            return [], ["review_batch:empty"], 0, []

        batch_count = len(batches)
        partial_failures = 0
        collected: list[dict[str, Any]] = []
        tool_logs: list[str] = [f"review_batch:enabled", f"review_batch:count={batch_count}"]
        details: list[str] = []

        async with self._development_batch_lock:
            for idx, batch in enumerate(batches, start=1):
                tool_logs.append(f"review_batch:round={idx}|citations={len(batch)}")
                try:
                    issues, detail, used_fallback = await asyncio.wait_for(
                        self._async_adapter.run(
                            WorkspaceRagDevelopment.generate_batch_review_issues,
                            executor=executor,
                            settings=settings,
                            workspace=workspace,
                            batch_citations=batch,
                        ),
                        timeout=self._DEVELOPMENT_BATCH_TIMEOUT_SEC,
                    )
                    collected.extend(issues)
                    tool_logs.append(f"review_batch:issues={len(issues)}")
                    if used_fallback:
                        tool_logs.append("review_batch:fallback_issue_extractor")
                    if detail:
                        details.append(str(detail))
                except TimeoutError:
                    partial_failures += 1
                    tool_logs.append(f"review_batch:timeout=round{idx}")
                except (RuntimeError, ValueError, TypeError, OSError) as exc:
                    partial_failures += 1
                    tool_logs.append(f"review_batch:error=round{idx}:{str(exc)[:120]}")

        reduced = self._reranker.reduce_and_rank_issues(collected)
        tool_logs.append(f"review_batch:reduced={len(reduced)}")
        if partial_failures > 0:
            tool_logs.append(f"review_batch:partial_failures={partial_failures}")
        return reduced, tool_logs, partial_failures, details

    def handles_intent(self, intent: ParsedIntent, followup: FollowUpResolution | None) -> bool:
        return intent.intent in {
            ReasoningIntent.FIND_FILE,
            ReasoningIntent.SUMMARIZE_FILE,
            ReasoningIntent.COMPARE_FILES,
            ReasoningIntent.EXPLAIN_CONTENT,
            ReasoningIntent.DRAFT_EDIT,
            ReasoningIntent.CLASSIFY,
            ReasoningIntent.OPEN_FILE,
            ReasoningIntent.FOLLOWUP_QUESTION,
            ReasoningIntent.FOLLOWUP_REFINE,
            ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
            ReasoningIntent.SOFT_CONFIRM,
            ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
            ReasoningIntent.NEXT_CANDIDATE,
            ReasoningIntent.REDUCE_SCOPE,
            ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
        }

    async def execute(
        self,
        *,
        context: ReasoningContext,
        dependencies: dict[str, Any],
    ) -> ComposedChatResponseV2:
        state = {
            "context": context,
            "dependencies": dependencies,
            "executor": dependencies.get("executor"),
            "memory": dependencies.get("memory"),
            "composer": dependencies.get("composer"),
            "pipeline_helpers": dependencies.get("helpers"),
            "embedding": dependencies.get("embedding_service"),
            "vector_store": dependencies.get("vector_store"),
            "reranker": getattr(dependencies.get("embedding_service"), "_reranker", None),
            "db": dependencies.get("db"),
            "req": context.req,
            "parsed_intent": context.parsed_intent,
            "workspace": context.workspace,
            "followup_resolution": context.followup_resolution,
            "effective_query": context.effective_query,
            "behavior_policy": context.behavior_policy,
            "response_language": context.response_language,
            "session_id": context.session_id,
            "workspace_identity": context.workspace_identity,
            "memory_prefs": context.memory_prefs,
            "memory_bundle": context.memory_bundle,
            "last_context": context.last_context,
            "session_digest": context.session_digest,
            "settings": context.settings,
            "perf": dependencies.get("helpers").get_performance_config() if hasattr(dependencies.get("helpers"), "get_performance_config") else {"rerank_top_k": 5, "retrieval_limit": 20},
        }

        # 1단계: 필터 및 쿼리 준비
        bypass_response = await self._prepare_filters_and_query(state)
        if bypass_response is not None:
            return bypass_response

        # 2단계: Retrieval 및 Materialization
        await self._retrieve_and_materialize(state)

        # 3단계: Inference 수행 및 1차 검증
        await self._run_inference(state)

        # 4단계: Self-Reflection (CRAG)
        bypass_response = await self._run_crag_reflection_if_needed(state)
        if bypass_response is not None:
            return bypass_response

        # 5단계: Secondary Reasoner (필요시)
        await self._run_secondary_reasoner_if_needed(state)

        # 6단계: Development 및 Patch 처리
        await self._handle_development_and_patches(state)

        # 7단계: 최종 응답 및 메타데이터, 메모리 이벤트 기록
        return await self._finalize_response(state)

    async def _prepare_filters_and_query(self, state: dict[str, Any]) -> ComposedChatResponseV2 | None:
        context: ReasoningContext = state["context"]
        dependencies: dict[str, Any] = state["dependencies"]
        req: LocalChatRequestV2 = state["req"]
        parsed_intent: ParsedIntent = state["parsed_intent"]
        workspace = state["workspace"]
        followup_resolution: FollowUpResolution = state["followup_resolution"]
        pipeline_helpers = state["pipeline_helpers"]
        response_language: str = state["response_language"]
        behavior_policy = state["behavior_policy"]
        session_id: str = state["session_id"]
        workspace_identity = state["workspace_identity"]
        memory_prefs = state["memory_prefs"]
        memory_bundle = state["memory_bundle"]
        last_context = state["last_context"]
        session_digest = state["session_digest"]
        settings = state["settings"]
        embedding = state["embedding"]

        # Guard: conversational follow-up summary requests ("방금 말한 ... 3줄")
        # should stay in GeneralChat unless the user clearly targets local files.
        lowered_query = str(req.query or "").strip().lower()
        followup_summary_tokens = (
            "방금",
            "아까",
            "다시",
            "핵심만",
            "3줄",
            "두 줄",
            "한 줄",
            "요약해",
            "줄여줘",
            "말한",
        )
        looks_like_followup_summary = any(token in lowered_query for token in followup_summary_tokens)
        if (
            looks_like_followup_summary
            and not utils._has_local_file_target_cues(req.query)
            and not utils._is_explicit_web_search_request(req.query)
            and not utils._is_explicit_freshness_web_request(req.query)
        ):
            logger.info("WorkspaceRAG bypassed for conversational follow-up summary query.")
            return await pipeline_helpers.run_general_chat(
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
                force_web_search=False,
            )

        prefer_multi_file_summary = pipeline_helpers.should_expand_summary_scope(
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

        allowed_doc_ids, metadata_map = pipeline_helpers.resolve_workspace_docs(
            workspace=workspace,
            filters=merged_filters,
        )
        if req.filters is None and not allowed_doc_ids:
            merged_filters = ChatFilters(excluded=False)
            allowed_doc_ids, metadata_map = pipeline_helpers.resolve_workspace_docs(
                workspace=workspace,
                filters=merged_filters,
            )
        focus_terms, strict_focus = pipeline_helpers.extract_path_focus_terms(
            query=req.query,
            topics=parsed_intent.entities.topics,
        )
        allowed_doc_ids, metadata_map = pipeline_helpers.apply_focus_filter(
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
        explicit_file_terms = pipeline_helpers.extract_explicit_file_terms(
            query=req.query,
            parsed_intent=parsed_intent,
        )
        explicit_file_focus_used = False
        if explicit_file_terms:
            allowed_doc_ids, metadata_map, explicit_file_focus_used = pipeline_helpers.apply_explicit_file_focus(
                doc_ids=allowed_doc_ids,
                metadata_map=metadata_map,
                file_terms=explicit_file_terms,
            )
        requested_weeks = pipeline_helpers.extract_requested_weeks(
            query=req.query,
            followup_resolution=followup_resolution,
        )
        allowed_doc_ids, metadata_map, week_exact_filter_applied, week_exact_no_match = pipeline_helpers.apply_week_exact_filter(
            doc_ids=allowed_doc_ids,
            metadata_map=metadata_map,
            requested_weeks=requested_weeks,
        )
        if parsed_intent.intent == ReasoningIntent.SUMMARIZE_FILE and explicit_file_terms and not prefer_multi_file_summary:
            prefer_focused_file_summary = True

        if pipeline_helpers.should_trigger_auto_index(
            req=req,
            parsed_intent=parsed_intent.intent,
            allowed_doc_ids=allowed_doc_ids,
            strict_focus=strict_focus,
        ):
            auto_index_triggered = True
            auto_indexed = pipeline_helpers.run_auto_index(workspace)
            if auto_indexed:
                allowed_doc_ids, metadata_map = pipeline_helpers.resolve_workspace_docs(
                    workspace=workspace,
                    filters=merged_filters,
                )
                allowed_doc_ids, metadata_map = pipeline_helpers.apply_focus_filter(
                    doc_ids=allowed_doc_ids,
                    metadata_map=metadata_map,
                    focus_terms=focus_terms,
                    strict_focus=strict_focus,
                )
                if explicit_file_terms:
                    allowed_doc_ids, metadata_map, explicit_file_focus_used = pipeline_helpers.apply_explicit_file_focus(
                        doc_ids=allowed_doc_ids,
                        metadata_map=metadata_map,
                        file_terms=explicit_file_terms,
                    )
        effective_query = context.effective_query
        query_vector = embedding.embed_query(effective_query)
        if pipeline_helpers.capabilities is not None:
            query_hook = pipeline_helpers.capabilities.process_retrieval_query_transform(query=effective_query)
            transformed_query = str(query_hook.value or "").strip()
            if transformed_query and transformed_query != effective_query:
                effective_query = transformed_query
                query_vector = embedding.embed_query(effective_query)
                if query_hook.source != PluginCapabilitySource.BUILT_IN:
                    dependencies["executor"].log_tool(f"plugin_capability:{query_hook.source.value}:retrieval.query_transform")
            if query_hook.error_code == PluginErrorCode.PLUGIN_PERMISSION_DENIED:
                dependencies["executor"].log_tool(f"plugin_privacy_blocked:retrieval.query_transform:{query_hook.error_message or ''}")

        perf = state["perf"]
        effective_top_k = req.top_k or perf["rerank_top_k"]
        if parsed_intent.intent == ReasoningIntent.FIND_FILE and req.top_k is None:
            effective_top_k = min(100, max(perf["rerank_top_k"], len(allowed_doc_ids)))

        state.update({
            "merged_filters": merged_filters,
            "allowed_doc_ids": allowed_doc_ids,
            "metadata_map": metadata_map,
            "prefer_focused_file_summary": prefer_focused_file_summary,
            "prefer_multi_file_summary": prefer_multi_file_summary,
            "auto_indexed": auto_indexed,
            "auto_index_triggered": auto_index_triggered,
            "explicit_file_focus_used": explicit_file_focus_used,
            "week_exact_filter_applied": week_exact_filter_applied,
            "week_exact_no_match": week_exact_no_match,
            "requested_weeks": requested_weeks,
            "query_vector": query_vector,
            "effective_query": effective_query,
            "effective_top_k": effective_top_k,
            "metadata_fallback_used": False,
            "focused_file_summary_used": False,
            "summary_scope_expanded": False,
        })
        return None

    async def _retrieve_and_materialize(self, state: dict[str, Any]) -> None:
        vector_store = state["vector_store"]
        req = state["req"]
        workspace = state["workspace"]
        query_vector = state["query_vector"]
        effective_query = state["effective_query"]
        allowed_doc_ids = state["allowed_doc_ids"]
        merged_filters = state["merged_filters"]
        metadata_map = state["metadata_map"]
        behavior_policy = state["behavior_policy"]
        reranker = state["reranker"]
        response_language = state["response_language"]
        effective_top_k = state["effective_top_k"]
        perf = state["perf"]
        pipeline_helpers = state["pipeline_helpers"]
        parsed_intent = state["parsed_intent"]
        prefer_focused_file_summary = state["prefer_focused_file_summary"]
        prefer_multi_file_summary = state["prefer_multi_file_summary"]
        focused_file_summary_used = state["focused_file_summary_used"]
        summary_scope_expanded = state["summary_scope_expanded"]
        metadata_fallback_used = state["metadata_fallback_used"]

        flow_result = WorkspaceRagSearchFlow.run(
            vector_store=vector_store,
            req=req,
            workspace=workspace,
            query_vector=query_vector,
            effective_query=effective_query,
            allowed_doc_ids=allowed_doc_ids,
            merged_filters=merged_filters,
            metadata_map=metadata_map,
            behavior_policy=behavior_policy,
            reranker=reranker,
            response_language=response_language,
            effective_top_k=effective_top_k,
            retrieval_limit=perf["retrieval_limit"],
            pipeline_helpers=pipeline_helpers,
        )
        preset = flow_result.preset
        retrieval = flow_result.retrieval
        if flow_result.tier2_triggered:
            state["executor"].log_tool("retrieval:tier2_global_search_triggered")

        grounding_reliability = 1.0
        if retrieval.chunk_candidates:
            rel_subset = retrieval.chunk_candidates[:3]
            grounding_reliability = sum(c.reliability for c in rel_subset) / len(rel_subset)

        _ = preset
        if (
            pipeline_helpers.capabilities is not None
            and retrieval.chunk_candidates
        ):
            post_filter_hook = pipeline_helpers.capabilities.process_retrieval_post_filter(
                query=effective_query,
                chunk_candidates=retrieval.chunk_candidates,
            )
            if post_filter_hook.error_code == PluginErrorCode.PLUGIN_PERMISSION_DENIED:
                state["executor"].log_tool(f"plugin_privacy_blocked:retrieval.post_filter:{post_filter_hook.error_message or ''}")

        citations = [
            pipeline_helpers.citation_from_chunk(chunk, metadata_map)
            for chunk in retrieval.chunk_candidates
        ]
        file_doc_ids = [item.doc_id for item in retrieval.file_candidates]
        chunk_ids = [item.chunk_id for item in retrieval.chunk_candidates]
        top_score = retrieval.chunk_candidates[0].score if retrieval.chunk_candidates else 0.0

        materialized = WorkspaceRagMaterializer.run(
            parsed_intent=parsed_intent,
            effective_query=effective_query,
            allowed_doc_ids=allowed_doc_ids,
            metadata_map=metadata_map,
            workspace=workspace,
            pipeline_helpers=pipeline_helpers,
            citations=citations,
            file_doc_ids=file_doc_ids,
            chunk_ids=chunk_ids,
            top_score=top_score,
            metadata_fallback_used=metadata_fallback_used,
            prefer_focused_file_summary=prefer_focused_file_summary,
            prefer_multi_file_summary=prefer_multi_file_summary,
            focused_file_summary_used=focused_file_summary_used,
            summary_scope_expanded=summary_scope_expanded,
        )

        state.update({
            "citations": materialized.citations,
            "file_doc_ids": materialized.file_doc_ids,
            "chunk_ids": materialized.chunk_ids,
            "top_score": materialized.top_score,
            "metadata_fallback_used": materialized.metadata_fallback_used,
            "focused_file_summary_used": materialized.focused_file_summary_used,
            "summary_scope_expanded": materialized.summary_scope_expanded,
            "grounding_reliability": grounding_reliability,
            "retrieval": retrieval,
        })

    async def _run_inference(self, state: dict[str, Any]) -> None:
        req = state["req"]
        citations = state["citations"]
        file_doc_ids = state["file_doc_ids"]
        executor = state["executor"]
        settings = state["settings"]
        workspace = state["workspace"]
        response_language = state["response_language"]
        parsed_intent = state["parsed_intent"]
        chunk_ids = state["chunk_ids"]
        top_score = state["top_score"]
        prefer_multi_file_summary = state["prefer_multi_file_summary"]
        prefer_focused_file_summary = state["prefer_focused_file_summary"]
        pipeline_helpers = state["pipeline_helpers"]
        effective_query = state["effective_query"]
        memory_prefs = state["memory_prefs"]
        auto_indexed = state["auto_indexed"]
        auto_index_triggered = state["auto_index_triggered"]
        metadata_fallback_used = state["metadata_fallback_used"]
        week_exact_filter_applied = state["week_exact_filter_applied"]
        requested_weeks = state["requested_weeks"]
        week_exact_no_match = state["week_exact_no_match"]
        explicit_file_focus_used = state["explicit_file_focus_used"]
        focused_file_summary_used = state["focused_file_summary_used"]
        summary_scope_expanded = state["summary_scope_expanded"]
        grounding_reliability = state["grounding_reliability"]

        development_batch_issues: list[dict[str, Any]] = []
        review_batch_enabled = False
        review_batch_count = 0
        review_batch_partial_failures = 0
        batch_logs = []
        batch_details = []

        if req.mode == WorkMode.DEVELOPMENT and self._retriever.is_large_development_review_request(
            req=req,
            citations=citations,
            file_doc_ids=file_doc_ids,
        ):
            review_batch_enabled = True
            development_batch_issues, batch_logs, review_batch_partial_failures, batch_details = await self._run_development_batch_review(
                executor=executor,
                settings=settings,
                workspace=workspace,
                req=req,
                response_language=response_language,
                citations=citations,
            )
            review_batch_count = max(1, len(self._retriever.build_development_batches(citations)))

        plan = pipeline_helpers.planner.build_plan(
            parsed_intent=parsed_intent,
            mode=req.mode,
            file_doc_ids=file_doc_ids,
            chunk_ids=chunk_ids,
            top_score=top_score,
            prefer_multi_file_summary=prefer_multi_file_summary,
            prefer_focused_file_summary=prefer_focused_file_summary,
        )

        # Level 1 Override
        hybrid_external_enabled = settings.privacy_mode == PrivacyMode.HYBRID and bool(
            getattr(settings, "hybrid_web_search_enabled", False)
        )
        if hybrid_external_enabled and top_score < 0.40 and not pipeline_helpers.is_brief_chat_query(req.query):
            if bool(
                utils._is_explicit_web_search_request(req.query)
                or utils._is_explicit_freshness_web_request(req.query)
            ):
                logger.info("Universal RAG web escalation (top_score=%.2f).", top_score)
                escalated_resp = await pipeline_helpers.run_general_chat(
                    req=req,
                    settings=settings,
                    workspace=workspace,
                    session_id=state["session_id"],
                    workspace_id=state["workspace_identity"].workspace_id,
                    response_language=response_language,
                    parsed_intent=parsed_intent,
                    behavior_policy=state["behavior_policy"],
                    memory_prefs=memory_prefs,
                    memory_bundle=state["memory_bundle"],
                    last_context=state["last_context"],
                    session_digest=state["session_digest"],
                    force_web_search=True,
                )
                state["bypass_escalation"] = escalated_resp
                return

        execution = await executor.execute_async(
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
            response_length=getattr(memory_prefs, "response_length", "long") if memory_prefs else "long",
        )

        if review_batch_enabled and development_batch_issues:
            execution.generated_text = self._prompter.build_batch_review_text(
                query=req.query,
                issues=development_batch_issues,
                response_language=response_language,
            )
            execution.result_type = "summary"
            execution.used_fallback = False
            execution.structured_payload["review_batch_issues"] = development_batch_issues[:20]
            execution.structured_payload["review_batch_issue_count"] = len(development_batch_issues)
            execution.runtime_detail = "; ".join(batch_details[:3]) if batch_details else execution.runtime_detail
            execution.tool_logs.extend(batch_logs)
            execution.tool_logs.append("review_batch:reduced_output_applied")
        elif review_batch_enabled:
            execution.tool_logs.extend(batch_logs)
            execution.tool_logs.append("review_batch:no_grounded_issue")

        WorkspaceRagFinalizer.apply_execution_metadata(
            execution=execution,
            plan=plan,
            auto_indexed=auto_indexed,
            auto_index_triggered=auto_index_triggered,
            metadata_fallback_used=metadata_fallback_used,
            week_exact_filter_applied=week_exact_filter_applied,
            requested_weeks=requested_weeks,
            week_exact_no_match=week_exact_no_match,
            explicit_file_focus_used=explicit_file_focus_used,
            focused_file_summary_used=focused_file_summary_used,
            summary_scope_expanded=summary_scope_expanded,
            req_mode=req.mode,
            work_mode_development=WorkMode.DEVELOPMENT,
            review_batch_enabled=review_batch_enabled,
            review_batch_count=review_batch_count,
            review_batch_partial_failures=review_batch_partial_failures,
        )

        verification = pipeline_helpers.verifier.verify(
            parsed_intent=parsed_intent,
            execution_result=execution,
            mode=req.mode,
            reliability=grounding_reliability,
        )

        state.update({
            "review_batch_enabled": review_batch_enabled,
            "review_batch_count": review_batch_count,
            "review_batch_partial_failures": review_batch_partial_failures,
            "development_batch_issues": development_batch_issues,
            "batch_logs": batch_logs,
            "batch_details": batch_details,
            "plan": plan,
            "execution": execution,
            "verification": verification,
        })

    async def _run_crag_reflection_if_needed(self, state: dict[str, Any]) -> ComposedChatResponseV2 | None:
        if "bypass_escalation" in state:
            return state["bypass_escalation"]

        req = state["req"]
        execution = state["execution"]
        verification = state["verification"]
        citations = state["citations"]
        executor = state["executor"]
        settings = state["settings"]
        response_language = state["response_language"]
        pipeline_helpers = state["pipeline_helpers"]
        workspace = state["workspace"]
        session_id = state["session_id"]
        workspace_identity = state["workspace_identity"]
        parsed_intent = state["parsed_intent"]
        behavior_policy = state["behavior_policy"]
        memory_prefs = state["memory_prefs"]
        memory_bundle = state["memory_bundle"]
        last_context = state["last_context"]
        session_digest = state["session_digest"]

        if (
            req.mode != WorkMode.STRICT_SEARCH
            and execution.result_type in {"answer", "summary"}
            and (verification.confidence < 0.7 or req.mode in {WorkMode.RESEARCH, WorkMode.DEVELOPMENT})
        ):
            reflection_context = "\n".join([f"[{c.file_path}] {c.snippet}" for c in citations[:5]])
            reflection_cat, reflection_reason = await executor.generate_reflection_async(
                engine=settings.local_engine,
                query=req.query,
                context=reflection_context,
                answer=execution.generated_text,
                mlx_model_path=settings.mlx_model_path,
                llama_model_path=settings.llama_model_path,
                response_language=response_language,
            )
            execution.tool_logs.append(f"reflection:{reflection_cat}|{reflection_reason[:40]}...")

            hybrid_external_enabled = settings.privacy_mode == PrivacyMode.HYBRID and bool(
                getattr(settings, "hybrid_web_search_enabled", False)
            )
            if reflection_cat in {"HALLUCINATED", "IRRELEVANT", "INSUFFICIENT"} and hybrid_external_enabled:
                if bool(
                    utils._is_explicit_web_search_request(req.query)
                    or utils._is_explicit_freshness_web_request(req.query)
                ):
                    logger.info("CRAG: Reflection triggered explicit web escalation (Category: %s)", reflection_cat)
                    return await pipeline_helpers.run_general_chat(
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
                        force_web_search=True,
                    )
        return None

    async def _run_secondary_reasoner_if_needed(self, state: dict[str, Any]) -> None:
        if "bypass_escalation" in state:
            return

        req = state["req"]
        parsed_intent = state["parsed_intent"]
        execution = state["execution"]
        verification = state["verification"]
        pipeline_helpers = state["pipeline_helpers"]
        response_language = state["response_language"]
        executor = state["executor"]
        settings = state["settings"]
        plan = state["plan"]
        citations = state["citations"]
        workspace = state["workspace"]
        memory_prefs = state["memory_prefs"]
        grounding_reliability = state["grounding_reliability"]

        if pipeline_helpers.should_run_secondary_reasoner(
            mode=req.mode,
            parsed_intent=parsed_intent,
            execution=execution,
            verification=verification,
        ):
            refined_query = pipeline_helpers.build_secondary_reasoner_prompt(
                query=req.query,
                parsed_intent=parsed_intent,
                response_language=response_language,
            )
            refined_execution = await executor.execute_async(
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
                response_length=getattr(memory_prefs, "response_length", "long") if memory_prefs else "long",
            )
            refined_execution.tool_logs.append("agent:reasoner_secondary")
            refined_verification = pipeline_helpers.verifier.verify(
                parsed_intent=parsed_intent,
                execution_result=refined_execution,
                mode=req.mode,
                reliability=grounding_reliability,
            )
            better_execution, better_verification = pipeline_helpers.pick_better_reasoner_result(
                primary_execution=execution,
                primary_verification=verification,
                secondary_execution=refined_execution,
                secondary_verification=refined_verification,
            )
            state["execution"] = better_execution
            state["verification"] = better_verification

        state["execution"].tool_logs.append("agent:verifier_multistage")

    async def _handle_development_and_patches(self, state: dict[str, Any]) -> None:
        if "bypass_escalation" in state:
            return

        req = state["req"]
        parsed_intent = state["parsed_intent"]
        settings = state["settings"]
        citations = state["citations"]
        pipeline_helpers = state["pipeline_helpers"]
        execution = state["execution"]
        verification = state["verification"]
        effective_query = state["effective_query"]
        response_language = state["response_language"]
        review_batch_enabled = state["review_batch_enabled"]
        development_batch_issues = state["development_batch_issues"]
        executor = state["executor"]
        workspace = state["workspace"]
        last_context = state["last_context"]
        followup_resolution = state["followup_resolution"]

        conversation_path = "local_rag"
        escalated_provider = None
        is_local = True

        if pipeline_helpers.should_escalate_summary_to_external(
            req=req,
            parsed_intent=parsed_intent,
            settings=settings,
            citations=citations,
        ):
            escalated = pipeline_helpers.escalate_summary_to_external(
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
                state["execution"] = execution
                state["verification"] = verification
            else:
                execution.tool_logs.append("external_escalation:summary_unavailable")

        candidate_gap_small = pipeline_helpers.candidate_gap_small(citations)
        clarification_state = ClarificationBudgetState(
            clarification_count_current_turn=0,
            previous_turn_was_clarification=bool((last_context or {}).get("used_clarification", False)),
            partial_user_answer_received=followup_resolution.is_followup,
        )
        risk_level = "low"
        if parsed_intent.intent in {ReasoningIntent.DRAFT_EDIT}:
            risk_level = "medium"
        allow_clarification = pipeline_helpers.clarification_budget.allow_clarification(
            state=clarification_state,
            query=req.query,
            ambiguity_level=verification.ambiguity_level,
            risk_level=risk_level,
            candidate_gap_small=candidate_gap_small,
        )

        patch_plan_count = 0
        patch_applied_count = 0
        patch_failed_count = 0
        patch_verification_status = "not_requested"

        if req.mode == WorkMode.STRICT_SEARCH and (not execution.citations or verification.candidate_mode):
            execution.generated_text = insufficient_evidence_message(response_language)
            execution.result_type = "insufficient"
            execution.structured_payload = {"reason": "strict_threshold_not_met"}
        if req.mode == WorkMode.DEVELOPMENT and not execution.citations:
            execution.generated_text = insufficient_evidence_message(response_language)
            execution.result_type = "insufficient"
            execution.structured_payload = {"reason": "development_requires_grounded_evidence"}
        elif req.mode == WorkMode.DEVELOPMENT:
            development_action = (req.development_action or "review").strip().lower() if req.mode == WorkMode.DEVELOPMENT else "review"
            requested_fix_mode = (req.fix_mode or "apply_patch").strip().lower() if req.mode == WorkMode.DEVELOPMENT else "plan_only"

            if development_action == "fix":
                if review_batch_enabled:
                    source_issues = development_batch_issues
                else:
                    source_issues = [WorkspaceRagDevelopment.issue_from_citation(item) for item in execution.citations[:16]]
                patch_plan = WorkspaceRagDevelopment.build_patch_plan(source_issues, max_items=self._DEVELOPMENT_PATCH_MAX_ITEMS)
                patch_plan_count = len(patch_plan)
                execution.structured_payload["patch_plan"] = patch_plan
                execution.structured_payload["patch_plan_count"] = patch_plan_count
                execution.tool_logs.append(f"patch_plan:count={patch_plan_count}")

                force_plan_only = requested_fix_mode != "apply_patch"
                if any(str(item.get("risk_level") or "").lower() == "high" for item in patch_plan):
                    force_plan_only = True
                    execution.tool_logs.append("patch_plan:downgraded=high_risk")
                if patch_plan_count >= 4:
                    force_plan_only = True
                    execution.tool_logs.append("patch_plan:downgraded=wide_change")

                if force_plan_only:
                    patch_verification_status = "skipped"
                    execution.tool_logs.append("patch_apply:skipped_plan_only")
                else:
                    patch_applied_count, patch_failed_count, changed_files = await asyncio.to_thread(
                        WorkspaceRagDevelopment.apply_patch_items,
                        executor=executor,
                        settings=settings,
                        workspace=workspace,
                        query=req.query,
                        patch_plan=patch_plan,
                    )
                    execution.tool_logs.append(f"patch_apply:applied={patch_applied_count}")
                    execution.tool_logs.append(f"patch_apply:failed={patch_failed_count}")
                    patch_verification_status, verify_logs = WorkspaceRagComponents.run_patch_verification(changed_files)
                    for row in verify_logs:
                         execution.tool_logs.append(f"patch_verify:{row}")
                    if patch_verification_status == "failed":
                         execution.tool_logs.append("patch_verify:failed_retry_needed")

                if response_language == "en":
                     execution.generated_text = (
                         "Patch planning completed.\n"
                         f"- fix_mode: {'plan_only' if force_plan_only else 'apply_patch'}\n"
                         f"- patch_plan_count: {patch_plan_count}\n"
                         f"- patch_applied_count: {patch_applied_count}\n"
                         f"- patch_failed_count: {patch_failed_count}\n"
                         f"- verification_status: {patch_verification_status}"
                     )
                elif response_language == "ja":
                     execution.generated_text = (
                         "パッチ計画を完了しました。\n"
                         f"- fix_mode: {'plan_only' if force_plan_only else 'apply_patch'}\n"
                         f"- patch_plan_count: {patch_plan_count}\n"
                         f"- patch_applied_count: {patch_applied_count}\n"
                         f"- patch_failed_count: {patch_failed_count}\n"
                         f"- verification_status: {patch_verification_status}"
                     )
                else:
                     execution.generated_text = (
                         "패치 계획을 완료했습니다.\n"
                         f"- fix_mode: {'plan_only' if force_plan_only else 'apply_patch'}\n"
                         f"- patch_plan_count: {patch_plan_count}\n"
                         f"- patch_applied_count: {patch_applied_count}\n"
                         f"- patch_failed_count: {patch_failed_count}\n"
                         f"- verification_status: {patch_verification_status}"
                     )
                execution.result_type = "summary"

            execution.generated_text = self._prompter.ensure_development_answer_template(
                query=req.query,
                answer=execution.generated_text,
                citations=execution.citations,
                response_language=response_language,
            )
            execution.tool_logs.append("development:review_template_applied")

        state.update({
            "conversation_path": conversation_path,
            "escalated_provider": escalated_provider,
            "is_local": is_local,
            "allow_clarification": allow_clarification,
            "patch_plan_count": patch_plan_count,
            "patch_applied_count": patch_applied_count,
            "patch_failed_count": patch_failed_count,
            "patch_verification_status": patch_verification_status,
        })

    async def _finalize_response(self, state: dict[str, Any]) -> ComposedChatResponseV2:
        if "bypass_escalation" in state:
            return state["bypass_escalation"]

        execution = state["execution"]
        parsed_intent = state["parsed_intent"]
        retrieval = state["retrieval"]
        pipeline_helpers = state["pipeline_helpers"]
        session_id = state["session_id"]
        workspace_identity = state["workspace_identity"]
        req = state["req"]
        composer = state["composer"]
        response_language = state["response_language"]
        plan = state["plan"]
        verification = state["verification"]
        behavior_policy = state["behavior_policy"]
        memory_prefs = state["memory_prefs"]
        workspace = state["workspace"]
        allow_clarification = state["allow_clarification"]
        conversation_path = state["conversation_path"]
        escalated_provider = state["escalated_provider"]
        is_local = state["is_local"]
        review_batch_enabled = state["review_batch_enabled"]
        review_batch_count = state["review_batch_count"]
        review_batch_partial_failures = state["review_batch_partial_failures"]
        patch_plan_count = state["patch_plan_count"]
        patch_applied_count = state["patch_applied_count"]
        patch_failed_count = state["patch_failed_count"]
        patch_verification_status = state["patch_verification_status"]

        if execution.result_type == "file_list":
            execution.structured_payload.setdefault("requested_scope", str(getattr(parsed_intent, "scope", "single") or "single"))
            execution.structured_payload.setdefault("requested_operation", str(getattr(parsed_intent, "operation", "find") or "find"))
            if getattr(parsed_intent, "target", None):
                execution.structured_payload.setdefault("requested_target", str(parsed_intent.target))

        top_file_ids = [item.doc_id for item in retrieval.file_candidates[:3]]
        pipeline_helpers.memory.write_memory_event(
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
        outcome_type = pipeline_helpers.outcome_event_type(execution.result_type)
        if outcome_type is not None:
            pipeline_helpers.memory.write_memory_event(
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

        inference_engine = getattr(pipeline_helpers.executor, "_local_inference", None)
        prompt_cache_hit = bool(getattr(inference_engine, "_last_mlx_cache_hit", False))

        composed = composer.compose_v2(
            query=req.query,
            mode=req.mode,
            response_language=response_language,
            parsed_intent=parsed_intent,
            plan=plan,
            execution_result=execution,
            verification=verification,
            behavior_policy=behavior_policy,
            response_length=getattr(memory_prefs, 'response_length', 'long') if memory_prefs else 'long',
            show_citations=getattr(memory_prefs, 'show_citations', True) if memory_prefs else True,
            prefer_action_suggestions=getattr(memory_prefs, 'prefer_action_suggestions', True) if memory_prefs else True,
            used_profile=workspace.startup_profile,
            engine_used=execution.engine_used,
            used_fallback=execution.used_fallback,
            runtime_detail=execution.runtime_detail,
            followup_resolution=state["followup_resolution"],
            allow_clarification=allow_clarification,
            conversation_path=conversation_path,
            escalated_provider=escalated_provider,
            is_local=is_local,
            prompt_cache_hit=prompt_cache_hit,
        )

        if req.mode == WorkMode.DEVELOPMENT:
            WorkspaceRagFinalizer.apply_development_composed_metadata(
                composed=composed,
                execution=execution,
                review_batch_enabled=review_batch_enabled,
                review_batch_count=review_batch_count,
                review_batch_partial_failures=review_batch_partial_failures,
                patch_plan_count=patch_plan_count,
                patch_applied_count=patch_applied_count,
                patch_failed_count=patch_failed_count,
                patch_verification_status=patch_verification_status,
                extract_grounded_line_refs=self._prompter.extract_grounded_line_refs,
            )
        return composed
