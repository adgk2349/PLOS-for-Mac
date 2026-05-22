from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....models import PluginCapabilitySource
from ....retrieval import retrieve_bundle


@dataclass(slots=True)
class WorkspaceRagSearchFlowResult:
    preset: Any
    retrieval: Any
    tier2_triggered: bool


class WorkspaceRagSearchFlow:
    @staticmethod
    def run(
        *,
        vector_store,
        req,
        workspace,
        query_vector,
        effective_query: str,
        allowed_doc_ids: set[str],
        merged_filters,
        metadata_map: dict[str, Any],
        behavior_policy,
        reranker,
        response_language: str,
        effective_top_k: int,
        retrieval_limit: int,
        pipeline_helpers,
    ) -> WorkspaceRagSearchFlowResult:
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
            search_limit_override=retrieval_limit,
        )

        tier2_triggered = False
        top_score = retrieval.chunk_candidates[0].score if retrieval.chunk_candidates else 0.0
        if top_score < 0.50 and not req.included_paths:
            _, global_retrieval = retrieve_bundle(
                vector_store=vector_store,
                query_vector=query_vector,
                mode=req.mode,
                startup_profile=workspace.startup_profile,
                query=effective_query,
                allowed_doc_ids=set(),
                filters=merged_filters,
                metadata_map=metadata_map,
                behavior_policy=behavior_policy,
                reranker=reranker,
                response_language=response_language,
                explicit_top_k=preset.top_k,
                search_limit_override=retrieval_limit * 2,
            )
            if global_retrieval.chunk_candidates:
                tier2_triggered = True
                existing_ids = {c.chunk_id for c in retrieval.chunk_candidates}
                for gc in global_retrieval.chunk_candidates:
                    if gc.chunk_id not in existing_ids:
                        retrieval.chunk_candidates.append(gc)
                retrieval.chunk_candidates.sort(key=lambda x: x.score, reverse=True)
                retrieval.file_candidates = pipeline_helpers.aggregate_file_candidates(retrieval.chunk_candidates)

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
                retrieval.rerank_features["capability_retriever_source"] = retrieval_hook.source.value
                retrieval.rerank_features["capability_retriever_log"] = execution_log

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

        if pipeline_helpers.capabilities is not None and retrieval.chunk_candidates:
            rerank_hook = pipeline_helpers.capabilities.process_reranker_rank(
                query=effective_query,
                chunk_candidates=retrieval.chunk_candidates,
            )
            if rerank_hook.value is not None:
                retrieval = retrieval.model_copy(update={"chunk_candidates": rerank_hook.value})
            if rerank_hook.source != PluginCapabilitySource.BUILT_IN:
                retrieval.rerank_features["capability_reranker_source"] = rerank_hook.source.value
                if rerank_hook.plugin_id:
                    retrieval.rerank_features["capability_reranker_plugin_id"] = rerank_hook.plugin_id

        return WorkspaceRagSearchFlowResult(
            preset=preset,
            retrieval=retrieval,
            tier2_triggered=tier2_triggered,
        )
