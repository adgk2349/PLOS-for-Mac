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
from ..context import ReasoningContext
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

    @classmethod
    def _is_code_file(cls, file_path: str) -> bool:
        return Path(str(file_path or "")).suffix.lower() in cls._CODE_EXTENSIONS

    @staticmethod
    def _extract_grounded_line_refs(citations: list[Citation]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for citation in citations[:10]:
            path = str(citation.file_path or "").strip()
            if not path:
                continue
            snippet = str(citation.snippet or "")
            line_match = re.search(r"(?:\bline\s*|라인\s*)(\d{1,6})\b|\bL(\d{1,6})\b", snippet, re.IGNORECASE)
            line_number = ""
            if line_match:
                line_number = str(line_match.group(1) or line_match.group(2) or "").strip()
            ref = f"{path}:L{line_number}" if line_number else f"{path}:{citation.chunk_id}"
            if ref in seen:
                continue
            seen.add(ref)
            output.append(ref)
            if len(output) >= 8:
                break
        return output

    @staticmethod
    def _ensure_development_answer_template(
        *,
        query: str,
        answer: str,
        citations: list[Citation],
        response_language: str,
    ) -> str:
        text = str(answer or "").strip()
        if not text:
            return text
        lowered = text.lower()
        if "근거 파일/라인" in text or "evidence files/lines" in lowered:
            return text

        file_refs: list[str] = []
        seen_paths: set[str] = set()
        for citation in citations[:4]:
            path = str(citation.file_path or "").strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            file_refs.append(path)

        if response_language == "en":
            refs_text = "\n".join(f"- {item}" for item in file_refs) or "- (no grounded file references)"
            return (
                f"Issue Summary:\n{query.strip()}\n\n"
                f"Evidence files/lines:\n{refs_text}\n\n"
                f"Proposed change:\n{text}\n\n"
                "Validation:\n- Run impacted unit/integration tests\n- Verify no regression in related paths"
            )
        if response_language == "ja":
            refs_text = "\n".join(f"- {item}" for item in file_refs) or "- （根拠ファイルなし）"
            return (
                f"問題要約:\n{query.strip()}\n\n"
                f"根拠ファイル/行:\n{refs_text}\n\n"
                f"修正提案:\n{text}\n\n"
                "検証:\n- 影響範囲のテストを実行\n- 関連パスの回帰を確認"
            )

        refs_text = "\n".join(f"- {item}" for item in file_refs) or "- (근거 파일 없음)"
        return (
            f"문제 요약:\n{query.strip()}\n\n"
            f"근거 파일/라인:\n{refs_text}\n\n"
            f"수정 제안:\n{text}\n\n"
            "검증:\n- 영향 범위 단위/통합 테스트 실행\n- 관련 경로 회귀 여부 확인"
        )

    @staticmethod
    def _extract_json_value(raw: str) -> dict[str, Any] | list[Any] | None:
        return WorkspaceRagDevelopment.extract_json_value(raw)

    @staticmethod
    def _severity_rank(value: str) -> int:
        token = str(value or "").strip().upper()
        if token == "P0":
            return 0
        if token == "P1":
            return 1
        if token == "P2":
            return 2
        return 3

    @staticmethod
    def _parse_line_ref(line_ref: str | None) -> int | None:
        return WorkspaceRagDevelopment.parse_line_ref(line_ref)

    @classmethod
    def _is_large_development_review_request(
        cls,
        *,
        req: LocalChatRequestV2,
        citations: list[Citation],
        file_doc_ids: list[str],
    ) -> bool:
        if req.mode != WorkMode.DEVELOPMENT:
            return False
        if len(file_doc_ids) >= cls._DEVELOPMENT_BATCH_TRIGGER_FILES:
            return True
        if len(citations) >= cls._DEVELOPMENT_BATCH_TRIGGER_CITATIONS:
            return True
        lowered = str(req.query or "").strip().lower()
        trigger_tokens = (
            "전체",
            "프로젝트 전체",
            "전부",
            "full repo",
            "entire project",
            "code review all",
        )
        return any(token in lowered for token in trigger_tokens)

    @classmethod
    def _build_development_batches(cls, citations: list[Citation]) -> list[list[Citation]]:
        grouped: dict[str, list[Citation]] = {}
        for item in citations:
            grouped.setdefault(str(item.doc_id), []).append(item)
        ranked_docs = sorted(
            grouped.items(),
            key=lambda row: max((c.score for c in row[1]), default=0.0),
            reverse=True,
        )[: cls._DEVELOPMENT_BATCH_MAX_FILES]
        batches: list[list[Citation]] = []
        current: list[Citation] = []
        current_chars = 0
        current_files = 0
        for _, doc_citations in ranked_docs:
            picked = sorted(doc_citations, key=lambda c: c.score, reverse=True)[: cls._DEVELOPMENT_BATCH_CHUNKS_PER_FILE]
            picked_chars = sum(len(str(c.snippet or "")) for c in picked)
            next_files = current_files + 1
            if (
                current
                and (
                    next_files > cls._DEVELOPMENT_BATCH_FILE_LIMIT
                    or current_chars + picked_chars > cls._DEVELOPMENT_BATCH_CHAR_LIMIT
                )
            ):
                batches.append(current)
                current = []
                current_chars = 0
                current_files = 0
            current.extend(picked)
            current_chars += picked_chars
            current_files += 1
        if current:
            batches.append(current)
        return batches[:10]

    def _issue_from_citation(self, citation: Citation) -> dict[str, Any]:
        return WorkspaceRagDevelopment.issue_from_citation(citation)

    def _generate_batch_review_issues(
        self,
        *,
        executor,
        settings,
        workspace,
        req: LocalChatRequestV2,
        response_language: str,
        batch_citations: list[Citation],
    ) -> tuple[list[dict[str, Any]], str | None, bool]:
        _ = req
        _ = response_language
        return WorkspaceRagDevelopment.generate_batch_review_issues(
            executor=executor,
            settings=settings,
            workspace=workspace,
            batch_citations=batch_citations,
        )

    @classmethod
    def _reduce_and_rank_issues(cls, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return WorkspaceRagComponents.reduce_and_rank_issues(items, severity_rank=cls._severity_rank)

    def _build_patch_plan(self, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return WorkspaceRagDevelopment.build_patch_plan(issues, max_items=self._DEVELOPMENT_PATCH_MAX_ITEMS)

    @staticmethod
    def _build_patch_prompt(*, query: str, issue: dict[str, Any], source_excerpt: str) -> str:
        return WorkspaceRagDevelopment.build_patch_prompt(query=query, issue=issue, source_excerpt=source_excerpt)

    def _apply_patch_items(
        self,
        *,
        executor,
        settings,
        workspace,
        query: str,
        patch_plan: list[dict[str, Any]],
    ) -> tuple[int, int, list[str]]:
        return WorkspaceRagDevelopment.apply_patch_items(
            executor=executor,
            settings=settings,
            workspace=workspace,
            query=query,
            patch_plan=patch_plan,
        )

    @staticmethod
    def _run_patch_verification(changed_files: list[str]) -> tuple[str, list[str]]:
        return WorkspaceRagComponents.run_patch_verification(changed_files)

    def _build_batch_review_text(self, *, query: str, issues: list[dict[str, Any]], response_language: str) -> str:
        return WorkspaceRagComponents.build_batch_review_text(
            query=query,
            issues=issues,
            response_language=response_language,
        )

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
        batches = self._build_development_batches(citations)
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
                            self._generate_batch_review_issues,
                            executor=executor,
                            settings=settings,
                            workspace=workspace,
                            req=req,
                            response_language=response_language,
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

        reduced = self._reduce_and_rank_issues(collected)
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
        executor = dependencies.get("executor")
        memory = dependencies.get("memory")
        composer = dependencies.get("composer")
        pipeline_helpers = dependencies.get("helpers")
        embedding = dependencies.get("embedding_service")
        vector_store = dependencies.get("vector_store")
        reranker = getattr(embedding, "_reranker", None)
        db = dependencies.get("db")
        
        req = context.req
        parsed_intent = context.parsed_intent
        workspace = context.workspace
        followup_resolution = context.followup_resolution
        merged_filters = getattr(req, "filters", ChatFilters())
        effective_query = context.effective_query
        perf = pipeline_helpers.get_performance_config() if hasattr(pipeline_helpers, "get_performance_config") else {"rerank_top_k": 5, "retrieval_limit": 20}
        behavior_policy = context.behavior_policy
        response_language = context.response_language
        session_id = context.session_id
        workspace_identity = context.workspace_identity
        memory_prefs = context.memory_prefs
        memory_bundle = context.memory_bundle
        last_context = context.last_context
        session_digest = context.session_digest
        settings = context.settings
        
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
        metadata_fallback_used = False
        summary_scope_expanded = False
        explicit_file_focus_used = False
        focused_file_summary_used = False
        week_exact_filter_applied = False
        week_exact_no_match = False
        review_batch_enabled = False
        review_batch_count = 0
        review_batch_partial_failures = 0
        patch_plan_count = 0
        patch_applied_count = 0
        patch_failed_count = 0
        patch_verification_status = "not_requested"
        development_action = (req.development_action or "review").strip().lower() if req.mode == WorkMode.DEVELOPMENT else "review"
        requested_fix_mode = (req.fix_mode or "plan_only").strip().lower() if req.mode == WorkMode.DEVELOPMENT else "plan_only"

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
        query_vector = embedding.embed_query(effective_query)
        if pipeline_helpers.capabilities is not None:
            query_hook = pipeline_helpers.capabilities.process_retrieval_query_transform(query=effective_query)
            transformed_query = str(query_hook.value or "").strip()
            if transformed_query and transformed_query != effective_query:
                effective_query = transformed_query
                query_vector = embedding.embed_query(effective_query)
                if query_hook.source != PluginCapabilitySource.BUILT_IN:
                    executor.log_tool(f"plugin_capability:{query_hook.source.value}:retrieval.query_transform")
            if query_hook.error_code == PluginErrorCode.PLUGIN_PERMISSION_DENIED:
                executor.log_tool(f"plugin_privacy_blocked:retrieval.query_transform:{query_hook.error_message or ''}")

        effective_top_k = req.top_k or perf["rerank_top_k"]
        if parsed_intent.intent == ReasoningIntent.FIND_FILE and req.top_k is None:
            effective_top_k = min(100, max(perf["rerank_top_k"], len(allowed_doc_ids)))
        
        # --- Phase 17: Tiered Retrieval (Ripple-out Search) ---
        # Tier 1: Focused search (current chat workspace)
        preset, retrieval = retrieve_bundle(
            vector_store=vector_store,
            query_vector=query_vector,
            mode=req.mode,
            startup_profile=workspace.startup_profile,
            query=effective_query,
            allowed_doc_ids=allowed_doc_ids,
            filters=merged_filters,
            metadata_map=metadata_map,
            behavior_policy=behavior_policy,
            reranker=reranker,
            response_language=response_language,
            explicit_top_k=effective_top_k,
            search_limit_override=perf["retrieval_limit"],
        )

        # Tier 2: Global Search (if Tier 1 is insufficient)
        # If Tier 1 has low scores, broaden to everything indexed locally.
        tier2_triggered = False
        top_score = retrieval.chunk_candidates[0].score if retrieval.chunk_candidates else 0.0
        # Multi-tiered threshold: broaden search if local workspace matches are weak (< 0.5)
        if top_score < 0.50 and not req.included_paths: 
             # Search without doc_id constraints
             _, global_retrieval = retrieve_bundle(
                 vector_store=vector_store,
                 query_vector=query_vector,
                 mode=req.mode,
                 startup_profile=workspace.startup_profile,
                 query=effective_query,
                 allowed_doc_ids=set(), # Empty set = all documents in LanceDB
                 filters=merged_filters,
                 metadata_map=metadata_map,
                 behavior_policy=behavior_policy,
                 reranker=reranker,
                 response_language=response_language,
                 explicit_top_k=preset.top_k,
                 search_limit_override=perf["retrieval_limit"] * 2, # Broader global search
             )
             
             # Combine results, prioritizing unique chunks.
             if global_retrieval.chunk_candidates:
                 tier2_triggered = True
                 existing_ids = {c.chunk_id for c in retrieval.chunk_candidates}
                 for gc in global_retrieval.chunk_candidates:
                     if gc.chunk_id not in existing_ids:
                         retrieval.chunk_candidates.append(gc)
                 
                 # Re-sort combined results
                 retrieval.chunk_candidates.sort(key=lambda x: x.score, reverse=True)
                 # Re-aggregate files
                 retrieval.file_candidates = pipeline_helpers.aggregate_file_candidates(retrieval.chunk_candidates)

        if tier2_triggered:
            executor.log_tool("retrieval:tier2_global_search_triggered")

        # Phase 20: RA-RAG Reliability Calculation
        grounding_reliability = 1.0
        if retrieval.chunk_candidates:
            rel_subset = retrieval.chunk_candidates[:3]
            grounding_reliability = sum(c.reliability for c in rel_subset) / len(rel_subset)

        if pipeline_helpers.capabilities is not None:
            retrieval_hook = pipeline_helpers.capabilities.process_retriever_search(
                query=effective_query,
                bundle=retrieval,
            )
            retrieval = retrieval_hook.value
            if retrieval_hook.source != PluginCapabilitySource.BUILT_IN:
                execution_log = f"capability:{retrieval_hook.source.value}:retriever.search"
                if retrieval_hook.plugin_id:
                    execution_log += f":{retrieval_hook.plugin_id}"
                # Persist visibility for future plugin-runtime rollout diagnostics.
                retrieval.rerank_features["capability_retriever_source"] = retrieval_hook.source.value
                retrieval.rerank_features["capability_retriever_log"] = execution_log
        _ = preset  # explicit for readability

        if pipeline_helpers.capabilities is not None and retrieval.chunk_candidates:
            post_filter_hook = pipeline_helpers.capabilities.process_retrieval_post_filter(
                query=effective_query,
                chunk_candidates=retrieval.chunk_candidates,
            )
            if isinstance(post_filter_hook.value, list):
                retrieval = retrieval.model_copy(update={"chunk_candidates": post_filter_hook.value})
                retrieval.file_candidates = pipeline_helpers.aggregate_file_candidates(retrieval.chunk_candidates)
            if post_filter_hook.source != PluginCapabilitySource.BUILT_IN:
                retrieval.rerank_features["capability_post_filter_source"] = post_filter_hook.source.value
                if post_filter_hook.plugin_id:
                    retrieval.rerank_features["capability_post_filter_plugin_id"] = post_filter_hook.plugin_id
            if post_filter_hook.error_code == PluginErrorCode.PLUGIN_PERMISSION_DENIED:
                executor.log_tool(f"plugin_privacy_blocked:retrieval.post_filter:{post_filter_hook.error_message or ''}")

        if pipeline_helpers.capabilities is not None and retrieval.chunk_candidates:
            rerank_hook = pipeline_helpers.capabilities.process_reranker_rank(
                query=effective_query,
                chunk_candidates=retrieval.chunk_candidates,
            )
            if rerank_hook.value is not None:
                retrieval = retrieval.model_copy(
                    update={"chunk_candidates": rerank_hook.value}
                )
            if rerank_hook.source != PluginCapabilitySource.BUILT_IN:
                retrieval.rerank_features["capability_reranker_source"] = rerank_hook.source.value
                if rerank_hook.plugin_id:
                    retrieval.rerank_features["capability_reranker_plugin_id"] = rerank_hook.plugin_id

        citations = [
            pipeline_helpers.citation_from_chunk(chunk, metadata_map)
            for chunk in retrieval.chunk_candidates
        ]
        file_doc_ids = [item.doc_id for item in retrieval.file_candidates]
        chunk_ids = [item.chunk_id for item in retrieval.chunk_candidates]
        top_score = retrieval.chunk_candidates[0].score if retrieval.chunk_candidates else 0.0
        if parsed_intent.intent == ReasoningIntent.FIND_FILE and not file_doc_ids and allowed_doc_ids:
            fallback_citations = pipeline_helpers.fallback_file_citations(
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
            fallback_citations = pipeline_helpers.fallback_file_citations(
                query=effective_query,
                allowed_doc_ids=allowed_doc_ids,
                metadata_map=metadata_map,
            )
            if fallback_citations:
                citations = pipeline_helpers.merge_find_file_citations(
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
            focused = pipeline_helpers.build_focused_file_summary_citations(
                doc_ids=target_doc_ids,
                metadata_map=metadata_map,
                max_chunks_per_file=pipeline_helpers.focused_summary_chunk_limit(workspace.startup_profile),
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
            expanded = pipeline_helpers.expand_summary_citations(
                query=effective_query,
                allowed_doc_ids=allowed_doc_ids,
                metadata_map=metadata_map,
                base_citations=citations,
                max_files=pipeline_helpers.summary_scope_doc_limit(workspace.startup_profile),
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

        development_batch_issues: list[dict[str, Any]] = []
        if req.mode == WorkMode.DEVELOPMENT and self._is_large_development_review_request(
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
            review_batch_count = max(1, len(self._build_development_batches(citations)))
        else:
            batch_logs = []
            batch_details = []

        plan = pipeline_helpers.planner.build_plan(
            parsed_intent=parsed_intent,
            mode=req.mode,
            file_doc_ids=file_doc_ids,
            chunk_ids=chunk_ids,
            top_score=top_score,
            prefer_multi_file_summary=prefer_multi_file_summary,
            prefer_focused_file_summary=prefer_focused_file_summary,
        )

        # Level 1 Override: If retrieval is still very weak after Tier 2, and we have hybrid enabled,
        # redirect to web search or general chat instead of hallucinating on irrelevant data.
        hybrid_external_enabled = settings.privacy_mode == PrivacyMode.HYBRID and bool(
            getattr(settings, "hybrid_web_search_enabled", False)
        )
        if hybrid_external_enabled and top_score < 0.40 and not pipeline_helpers.is_brief_chat_query(req.query):
            # This handles cases like "iPad performance" being misclassified as SUMMARIZE_FILE
            # but getting CS notes with a score like 0.3.
            logger.info("Universal RAG fallback triggered (top_score=%.2f). Escalating to web search.", top_score)
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

        if pipeline_helpers.should_short_circuit_candidate(
            mode=req.mode,
            top_score=top_score,
            intent=parsed_intent.intent,
            file_count=len(file_doc_ids),
            force_multi_file_summary=prefer_multi_file_summary,
            force_focused_file_summary=prefer_focused_file_summary,
        ):
            execution = pipeline_helpers.build_candidate_execution(
                response_language=response_language,
                citations=citations,
                reason="low_relevance_precheck",
            )
        else:
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
            execution.generated_text = self._build_batch_review_text(
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
        if req.mode == WorkMode.DEVELOPMENT and review_batch_enabled:
            execution.structured_payload["review_batch_enabled"] = True
            execution.structured_payload["review_batch_count"] = review_batch_count
            execution.structured_payload["review_batch_partial_failures"] = review_batch_partial_failures
        verification = pipeline_helpers.verifier.verify(
            parsed_intent=parsed_intent,
            execution_result=execution,
            mode=req.mode,
            reliability=grounding_reliability,
        )

        # Level 3: Agentic Self-Reflection (Self-RAG / CRAG)
        # If confidence is low or mode is RESEARCH/DEVELOPMENT, perform a reflection check.
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
            
            # Corrective Action: If hallucinated or irrelevant, and we have web search, trigger it!
            hybrid_external_enabled = settings.privacy_mode == PrivacyMode.HYBRID and bool(
                getattr(settings, "hybrid_web_search_enabled", False)
            )
            if reflection_cat in {"HALLUCINATED", "IRRELEVANT", "INSUFFICIENT"} and hybrid_external_enabled:
                logger.info("CRAG: Reflection triggered web search fallback (Category: %s)", reflection_cat)
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
            execution, verification = pipeline_helpers.pick_better_reasoner_result(
                primary_execution=execution,
                primary_verification=verification,
                secondary_execution=refined_execution,
                secondary_verification=refined_verification,
            )
        execution.tool_logs.append("agent:verifier_multistage")

        conversation_path = "local_rag"
        escalated_provider: str | None = None
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

        if (
            req.mode != WorkMode.STRICT_SEARCH
            and verification.candidate_mode
            and execution.result_type != "file_list"
            and not bool(execution.structured_payload.get("ungrounded_allowed", False))
            and (
                execution.used_fallback
                or not execution.generated_text.strip()
                or execution.result_type in {"candidate", "insufficient"}
                or pipeline_helpers.looks_like_reasoning_leak(execution.generated_text)
            )
        ):
            execution = pipeline_helpers.build_candidate_execution(
                response_language=response_language,
                citations=citations,
                reason="verification_candidate_mode",
            )

        if req.mode == WorkMode.STRICT_SEARCH and (not execution.citations or verification.candidate_mode):
            execution.generated_text = insufficient_evidence_message(response_language)
            execution.result_type = "insufficient"
            execution.structured_payload = {"reason": "strict_threshold_not_met"}
        if req.mode == WorkMode.DEVELOPMENT and not execution.citations:
            execution.generated_text = insufficient_evidence_message(response_language)
            execution.result_type = "insufficient"
            execution.structured_payload = {"reason": "development_requires_grounded_evidence"}
        elif req.mode == WorkMode.DEVELOPMENT:
            if development_action == "fix":
                if review_batch_enabled:
                    source_issues = development_batch_issues
                else:
                    source_issues = [self._issue_from_citation(item) for item in execution.citations[:16]]
                patch_plan = self._build_patch_plan(source_issues)
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
                        self._apply_patch_items,
                        executor=executor,
                        settings=settings,
                        workspace=workspace,
                        query=req.query,
                        patch_plan=patch_plan,
                    )
                    execution.tool_logs.append(f"patch_apply:applied={patch_applied_count}")
                    execution.tool_logs.append(f"patch_apply:failed={patch_failed_count}")
                    patch_verification_status, verify_logs = self._run_patch_verification(changed_files)
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
            execution.generated_text = self._ensure_development_answer_template(
                query=req.query,
                answer=execution.generated_text,
                citations=execution.citations,
                response_language=response_language,
            )
            execution.tool_logs.append("development:review_template_applied")
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
            followup_resolution=followup_resolution,
            allow_clarification=allow_clarification,
            conversation_path=conversation_path,
            escalated_provider=escalated_provider,
            is_local=is_local,
            prompt_cache_hit=prompt_cache_hit,
        )
        if req.mode == WorkMode.DEVELOPMENT:
            grounded_paths = []
            seen_paths: set[str] = set()
            for citation in execution.citations:
                path = str(citation.file_path or "").strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                grounded_paths.append(path)
            grounded_line_refs = self._extract_grounded_line_refs(execution.citations)
            composed.metadata["review_mode"] = True
            composed.metadata["grounded_file_count"] = len(grounded_paths)
            composed.metadata["grounded_line_refs"] = grounded_line_refs
            composed.metadata["review_batch_enabled"] = review_batch_enabled
            composed.metadata["review_batch_count"] = review_batch_count
            composed.metadata["review_batch_partial_failures"] = review_batch_partial_failures
            composed.metadata["patch_plan_count"] = patch_plan_count
            composed.metadata["patch_applied_count"] = patch_applied_count
            composed.metadata["patch_failed_count"] = patch_failed_count
            composed.metadata["verification_status"] = patch_verification_status
        return composed
