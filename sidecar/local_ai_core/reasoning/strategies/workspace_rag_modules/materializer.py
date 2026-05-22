from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....models import Citation, ReasoningIntent


@dataclass(slots=True)
class WorkspaceRagMaterialized:
    citations: list[Citation]
    file_doc_ids: list[str]
    chunk_ids: list[str]
    top_score: float
    metadata_fallback_used: bool
    focused_file_summary_used: bool
    summary_scope_expanded: bool


class WorkspaceRagMaterializer:
    @staticmethod
    def run(
        *,
        parsed_intent,
        effective_query: str,
        allowed_doc_ids: set[str],
        metadata_map: dict[str, Any],
        workspace,
        pipeline_helpers,
        citations: list[Citation],
        file_doc_ids: list[str],
        chunk_ids: list[str],
        top_score: float,
        metadata_fallback_used: bool,
        prefer_focused_file_summary: bool,
        prefer_multi_file_summary: bool,
        focused_file_summary_used: bool,
        summary_scope_expanded: bool,
    ) -> WorkspaceRagMaterialized:
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

        return WorkspaceRagMaterialized(
            citations=citations,
            file_doc_ids=file_doc_ids,
            chunk_ids=chunk_ids,
            top_score=top_score,
            metadata_fallback_used=metadata_fallback_used,
            focused_file_summary_used=focused_file_summary_used,
            summary_scope_expanded=summary_scope_expanded,
        )

