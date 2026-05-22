from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
from typing import Any
import os
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from collections import Counter, deque

from ....nlu.clarification_budget import ClarificationBudget, ClarificationBudgetState
from ....db import Database
from ....embedding import EmbeddingService
from ....executor import LocalExecutor
from ....nlu.followup_resolver import FollowUpResolution, FollowUpResolver
from ....nlu.intent_parser import IntentParser
from ....language_utils import insufficient_evidence_message, resolve_response_language
from ....local_planner import LocalPlanner
from ....memory_service import MemoryService
from ....models import *
from ....composition.composer import ResponseComposer
from ...context import ReasoningContext
from ...strategies.general_chat import GeneralChatStrategy
from ....retrieval import extract_query_hints, merge_filters, retrieve_bundle
from ....vector_store import VectorStore
from ....verifier import ResultVerifier


class RetrievalHelpers:
    def __init__(self, dependencies: dict[str, Any]):
        self._db = dependencies.get('db')
        self._memory = dependencies.get('memory') or dependencies.get('memory_service')
        self._embedding = dependencies.get('embedding_service')
        self._vector_store = dependencies.get('vector_store')
        self._composer = dependencies.get('composer')
        self._executor = dependencies.get('executor')
        self._intent_parser = dependencies.get('intent_parser')
        self._followup = dependencies.get('followup_resolver')
        self._reranker = getattr(dependencies.get('embedding_service'), '_reranker', None)
        self._clarification_budget = dependencies.get('clarification_budget')
        self._capabilities = dependencies.get('capabilities') or dependencies.get('capability_router')
        self._indexing = dependencies.get('indexing_service')
        self._planner = dependencies.get('local_planner')
        self._verifier = dependencies.get('verifier')
        self._provider_router = dependencies.get('provider_router')
        self._last_auto_index_started_at = 0
    def looks_like_reasoning_leak(self, text: str) -> bool:
        from ... import utils
        return utils._looks_like_reasoning_leak(text)

    def is_brief_chat_query(self, query: str) -> bool:
        from ... import utils
        return utils._is_brief_chat_query(query)

    def extract_path_focus_terms(self, query: str, topics: list[str]) -> tuple[list[str], bool]:
        from ... import utils
        return utils._extract_path_focus_terms(query=query, topics=topics)

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def planner(self):
        return self._planner

    @property
    def composer(self):
        return self._composer

    @property
    def verifier(self):
        return self._verifier

    @property
    def executor(self):
        return self._executor

    @property
    def memory(self):
        return self._memory

    @property
    def clarification_budget(self):
        return self._clarification_budget

    # ─── Session Digest ───────────────────────────────────────────────────────

    def update_session_digest_metadata(
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
            if self._memory:
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

    # ─── Workspace Doc Resolution ─────────────────────────────────────────────

    def resolve_workspace_docs(
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

    # ─── Focus Filters ────────────────────────────────────────────────────────

    def apply_focus_filter(
        self,
        *,
        doc_ids: set[str],
        metadata_map: dict[str, dict],
        focus_terms: list[str],
        strict_focus: bool,
    ) -> tuple[set[str], dict[str, dict]]:
        if not focus_terms:
            return doc_ids, metadata_map
        focused_doc_ids = self.filter_doc_ids_by_path_focus(
            doc_ids=doc_ids,
            metadata_map=metadata_map,
            focus_terms=focus_terms,
        )
        if focused_doc_ids:
            return focused_doc_ids, {doc_id: row for doc_id, row in metadata_map.items() if doc_id in focused_doc_ids}
        if strict_focus:
            return set(), {}
        return doc_ids, metadata_map

    def should_trigger_auto_index(
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
        return not allowed_doc_ids

    def _find_like_query(self, query: str) -> bool:
        keywords = ["찾아", "보여", "어딨", "있어", "어케", "어떻게", "핵심", "요약", "비교"]
        return any(k in query for k in keywords)

    def run_auto_index(self, workspace) -> bool:
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

    def apply_week_exact_filter(
        self,
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
            week_hits = set(self.extract_weeks_from_text(path))
            if week_hits.intersection(wanted):
                matched.add(doc_id)
        if not matched:
            return set(), {}, True, True
        filtered = {doc_id: row for doc_id, row in metadata_map.items() if doc_id in matched}
        return matched, filtered, True, False

    def filter_doc_ids_by_path_focus(
        self,
        *,
        doc_ids: set[str],
        metadata_map: dict[str, dict],
        focus_terms: list[str],
    ) -> set[str]:
        if not doc_ids or not focus_terms:
            return set()
        normalized_terms = [self._normalize_variants(term) for term in focus_terms]
        normalized_terms = [terms for terms in normalized_terms if terms]
        if not normalized_terms:
            return set()
        filtered: set[str] = set()
        for doc_id in doc_ids:
            path = str((metadata_map.get(doc_id) or {}).get("path") or "")
            if not path:
                continue
            path_variants = self._normalize_variants(path)
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

    def _normalize_variants(self, text: str) -> list[str]:
        text = text.lower().strip()
        nfc = unicodedata.normalize("NFC", text)
        nfd = unicodedata.normalize("NFD", text)
        return list({text, nfc, nfd})

    def extract_weeks_from_text(self, text: str) -> list[int]:
        matches = re.findall(r"(\d+)\s*(?:주차|주)", text)
        return [int(m) for m in matches]

    # ─── Summary Scope ────────────────────────────────────────────────────────

    def should_expand_summary_scope(self, query: str, parsed_intent: ParsedIntent) -> bool:
        if parsed_intent.intent != ReasoningIntent.SUMMARIZE_FILE:
            return False
        keywords = ["전부", "모두", "여러개", "파일들", "전체", "싹다"]
        return any(k in query for k in keywords)

    def summary_scope_doc_limit(self, startup_profile) -> int:
        profile = str(startup_profile).upper() if startup_profile else "RECOMMENDED"
        return {"FAST": 3, "RECOMMENDED": 5, "DEEP": 10}.get(profile, 5)

    def focused_summary_chunk_limit(self, startup_profile) -> int:
        profile = str(startup_profile).upper() if startup_profile else "RECOMMENDED"
        return {"FAST": 4, "RECOMMENDED": 8, "DEEP": 16}.get(profile, 8)

    def expand_summary_citations(
        self,
        *,
        query: str,
        allowed_doc_ids: set[str],
        metadata_map: dict[str, dict],
        base_citations: list[Citation],
        max_files: int,
        max_chunks_per_file: int,
    ) -> list[Citation]:
        """Expand citations to include more files for multi-file summaries."""
        seen_doc_ids = {c.doc_id for c in base_citations}
        extra = []
        for doc_id in list(allowed_doc_ids):
            if len(seen_doc_ids) >= max_files:
                break
            if doc_id in seen_doc_ids:
                continue
            row = metadata_map.get(doc_id) or {}
            path = row.get("path", "")
            seen_doc_ids.add(doc_id)
            extra.append(Citation(
                doc_id=doc_id,
                chunk_id=f"{doc_id}:0",
                file_path=str(path),
                snippet="",
                score=0.1,
                modified_at=datetime.now(timezone.utc),
                category=row.get("category", "참고자료"),
            ))
        return base_citations + extra if extra else []

    def build_focused_file_summary_citations(
        self,
        *,
        doc_ids: list[str],
        metadata_map: dict[str, dict],
        max_chunks_per_file: int,
    ) -> list[Citation]:
        """Build dense citations for a focused single-file summary."""
        citations = []
        for doc_id in doc_ids[:2]:
            row = metadata_map.get(doc_id) or {}
            path = row.get("path", "")
            for i in range(max_chunks_per_file):
                citations.append(Citation(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}:{i}",
                    file_path=str(path),
                    snippet="",
                    score=0.5,
                    modified_at=datetime.now(timezone.utc),
                    category=row.get("category", "참고자료"),
                ))
        return citations

    # ─── Explicit File Terms ──────────────────────────────────────────────────

    def extract_explicit_file_terms(self, query: str, parsed_intent: ParsedIntent) -> list[str]:
        terms = list(parsed_intent.entities.file_names)
        potential = re.findall(r"[\w가-힣\.-]+\.(?:pdf|docx|txt|md|swift|py|js|ts|java|c|cpp|h)", query)
        for p in potential:
            if p not in terms:
                terms.append(p)
        return terms

    def apply_explicit_file_focus(
        self,
        *,
        doc_ids: set[str],
        metadata_map: dict[str, dict],
        file_terms: list[str],
    ) -> tuple[set[str], dict[str, dict], bool]:
        if not doc_ids or not file_terms:
            return doc_ids, metadata_map, False
        matched: set[str] = set()
        normalized_terms = [t.lower() for t in file_terms]
        for doc_id in doc_ids:
            row = metadata_map.get(doc_id) or {}
            path = str(row.get("path") or "").lower()
            if any(term in path for term in normalized_terms):
                matched.add(doc_id)
        if matched:
            return matched, {doc_id: row for doc_id, row in metadata_map.items() if doc_id in matched}, True
        return doc_ids, metadata_map, False

    def extract_path_focus_terms(self, query: str, topics: list[str]) -> tuple[list[str], bool]:
        terms = list(topics)
        strict = "만" in query or "로만" in query
        paths = re.findall(r"/(?:[\w가-힣\.-]+/)+", query)
        terms.extend([p.strip("/") for p in paths])
        return terms, strict

    def extract_requested_weeks(self, query: str, followup_resolution: FollowUpResolution | None) -> list[int]:
        weeks = self.extract_weeks_from_text(query)
        negations = re.findall(r"(\d+)\s*(?:주차|주)\s*(?:말고|제외)", query)
        negated = {int(n) for n in negations}
        weeks = [w for w in weeks if w not in negated]
        if followup_resolution and followup_resolution.resolved_filters:
            f_week = followup_resolution.resolved_filters.get("week")
            if isinstance(f_week, int) and f_week not in negated:
                if f_week not in weeks:
                    weeks.append(f_week)
        return weeks

    # ─── Short-circuit & Verification ────────────────────────────────────────

    def should_short_circuit_candidate(
        self,
        mode: WorkMode,
        top_score: float,
        intent: ReasoningIntent,
        file_count: int,
        force_multi_file_summary: bool = False,
        force_focused_file_summary: bool = False,
    ) -> bool:
        if force_multi_file_summary or force_focused_file_summary:
            return False
        if mode == WorkMode.SUMMARY and file_count > 0:
            return False
        if intent == ReasoningIntent.FIND_FILE:
            return False
        return top_score < 0.15

    def outcome_event_type(self, result_type: str) -> MemoryEventType | None:
        mapping = {
            "answer": MemoryEventType.ANSWER,
            "summary": MemoryEventType.SUMMARY,
            "file_list": MemoryEventType.FILE_LIST,
        }
        return mapping.get(result_type)

    # ─── Query Analysis ───────────────────────────────────────────────────────

    def is_brief_chat_query(self, query: str) -> bool:
        return len(query.strip()) < 10 and not any(k in query for k in ["찾아", "요약"])

    def looks_like_reasoning_leak(self, text: str) -> bool:
        leaks = ["system prompt", "rule:", "instruction:", "user message:"]
        return any(leak in text.lower() for leak in leaks)

    def is_detailed_explanation_requested(self, query: str) -> bool:
        return any(keyword in query.lower() for keyword in ["detailed", "explain in depth", "핵심", "자세히"])

    def get_performance_config(self) -> dict[str, int]:
        return {"rerank_top_k": 5, "retrieval_limit": 20}

    # ─── Citations ────────────────────────────────────────────────────────────

    def citation_from_chunk(self, chunk: ChunkCandidate, metadata_map: dict[str, dict]) -> Citation:
        meta = metadata_map.get(chunk.doc_id) or {}
        return Citation(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            snippet=chunk.snippet,
            score=chunk.score,
            modified_at=chunk.modified_at,
            category=meta.get("category", "참고자료"),
            subcategory=meta.get("subcategory", ""),
            tags=meta.get("tags", []),
            document_type=meta.get("document_type", ""),
            importance=meta.get("importance", 0.5),
        )

    def fallback_file_citations(
        self,
        *,
        query: str,
        allowed_doc_ids: set[str],
        metadata_map: dict[str, dict],
    ) -> list[Citation]:
        results = []
        query_terms = query.lower().split()
        for doc_id in list(allowed_doc_ids)[:5]:
            path = metadata_map.get(doc_id, {}).get("path", "").lower()
            if any(term in path for term in query_terms):
                results.append(Citation(
                    doc_id=doc_id,
                    chunk_id="fallback",
                    file_path=metadata_map[doc_id]["path"],
                    snippet="파일명이 검색어와 일치합니다.",
                    score=0.1,
                    modified_at=datetime.now(timezone.utc),
                    category=metadata_map[doc_id].get("category", "참고자료"),
                ))
        return results

    def merge_find_file_citations(
        self,
        *,
        primary: list[Citation],
        fallback: list[Citation],
        limit: int = 140,
    ) -> list[Citation]:
        """Merge primary chunk citations with fallback file-name citations, deduplicating by doc_id."""
        seen: set[str] = set()
        merged: list[Citation] = []
        for c in primary + fallback:
            if c.doc_id not in seen:
                seen.add(c.doc_id)
                merged.append(c)
            if len(merged) >= limit:
                break
        return merged

    def aggregate_file_candidates(self, chunk_candidates: list[ChunkCandidate]) -> list[FileCandidate]:
        """Aggregate chunk-level results into file-level candidates."""
        seen: dict[str, FileCandidate] = {}
        for chunk in chunk_candidates:
            if chunk.doc_id not in seen:
                seen[chunk.doc_id] = FileCandidate(
                    doc_id=chunk.doc_id,
                    file_path=chunk.file_path,
                    score=chunk.score,
                    modified_at=chunk.modified_at,
                    category=chunk.category,
                    tags=chunk.tags,
                )
            else:
                seen[chunk.doc_id] = seen[chunk.doc_id].model_copy(
                    update={"score": max(seen[chunk.doc_id].score, chunk.score)}
                )
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)

    # ─── Secondary Reasoner ───────────────────────────────────────────────────

    def should_run_secondary_reasoner(
        self,
        *,
        mode: WorkMode,
        parsed_intent: ParsedIntent,
        execution: Any = None,
        verification: Any = None,
    ) -> bool:
        top_score = 0.0
        if execution and hasattr(execution, "citations") and execution.citations:
            top_score = execution.citations[0].score
        if mode == WorkMode.RESEARCH and top_score > 0.4:
            return True
        if parsed_intent.intent in {ReasoningIntent.FIND_FILE, ReasoningIntent.SUMMARIZE_FILE} and top_score > 0.5:
            return True
        return False

    def build_secondary_reasoner_prompt(
        self,
        *,
        query: str,
        parsed_intent: ParsedIntent,
        response_language: str,
    ) -> str:
        if response_language == "ko":
            return (
                f"사용자 질문: {query}\n"
                f"의도: {parsed_intent.intent.value}\n"
                "위 질문에 대한 답변이 충분하지 않거나 보강이 필요한지 검토하고, "
                "더 정교한 답변을 위해 근거 데이터를 재해석해서 보강된 답변을 작성해줘."
            )
        return (
            f"User Query: {query}\n"
            f"Intent: {parsed_intent.intent.value}\n"
            "Review if the current answer is sufficient. Re-interpret the grounded evidence "
            "to provide a more refined and comprehensive response."
        )

    def pick_better_reasoner_result(
        self,
        *,
        primary_execution: Any,
        primary_verification: Any,
        secondary_execution: Any,
        secondary_verification: Any,
    ) -> tuple[Any, Any]:
        if secondary_verification.confidence > primary_verification.confidence:
            return secondary_execution, secondary_verification
        return primary_execution, primary_verification

    # ─── Clarification ────────────────────────────────────────────────────────

    def candidate_gap_small(self, citations: list[Citation]) -> bool:
        if len(citations) < 2:
            return False
        return (citations[0].score - citations[1].score) < 0.1

    # ─── Candidate Execution ─────────────────────────────────────────────────

    def build_candidate_execution(
        self,
        *,
        response_language: str,
        citations: list[Citation],
        reason: str,
    ) -> ExecutionResult:
        options: list[str] = []
        for citation in list(citations or [])[:3]:
            title = str(getattr(citation, "source", "") or "").strip()
            if not title:
                title = str(getattr(citation, "snippet", "") or "").strip()[:42]
            title = re.sub(r"\s+", " ", title).strip()
            if title and title not in options:
                options.append(title)
        if response_language == "ko":
            if options:
                text = "\n".join(
                    f"{idx}. {item}" for idx, item in enumerate(options, start=1)
                )
            else:
                text = ""
        else:
            if options:
                text = "\n".join(
                    f"{idx}. {item}" for idx, item in enumerate(options, start=1)
                )
            else:
                text = ""
        return ExecutionResult(
            generated_text=text,
            result_type="candidate",
            citations=citations,
            structured_payload={"reason": reason},
        )

    # ─── External Escalation ──────────────────────────────────────────────────

    def should_escalate_summary_to_external(
        self,
        *,
        req: Any,
        parsed_intent: Any,
        settings: Any,
        citations: list[Citation],
    ) -> bool:
        if not citations:
            return False
        if req.mode != WorkMode.RESEARCH:
            return False
        if len(citations) > 10 or parsed_intent.intent == ReasoningIntent.SUMMARIZE_FILE:
            return True
        return False

    def escalate_summary_to_external(
        self,
        *,
        query: str,
        mode: Any,
        citations: list[Citation],
        settings: Any,
    ) -> tuple[ExecutionResult, str] | None:
        # Placeholder: external escalation not implemented in this environment
        return None

    def should_apply_formatting_optimization(self, query: str, result_type: str) -> bool:
        return result_type in {"answer", "summary"} and len(query) > 20

    # ─── General Chat Fallback ────────────────────────────────────────────────

    async def run_general_chat(
        self,
        *,
        req: Any,
        settings: Any,
        workspace: Any,
        session_id: str | None,
        workspace_id: str,
        response_language: str,
        parsed_intent: Any,
        behavior_policy: Any,
        memory_prefs: Any,
        memory_bundle: Any,
        last_context: dict | None,
        session_digest: str | None,
        force_web_search: bool = False,
    ) -> ComposedChatResponseV2:
        """
        Delegate to the actual GeneralChatStrategy when workspace RAG needs fallback.
        """
        logger.info("[RetrievalHelpers] run_general_chat delegated (force_web_search=%s)", force_web_search)
        context = ReasoningContext(
            req=req,
            workspace=workspace,
            workspace_identity=WorkspaceIdentity(
                workspace_id=workspace_id,
                included_paths_hash="",
                version=1,
            ),
            settings=settings,
            session_id=str(session_id or req.conversation_id or "default-session"),
            response_language=response_language,
            parsed_intent=parsed_intent,
            followup_resolution=FollowUpResolution(),
            memory_bundle=memory_bundle,
            behavior_policy=behavior_policy,
            memory_prefs=memory_prefs,
            last_context=last_context,
            session_digest=session_digest,
            effective_query=req.query,
            force_web_search=force_web_search,
        )
        strategy = GeneralChatStrategy()
        dependencies = {
            "executor": self._executor,
            "composer": self._composer or ResponseComposer(),
            "memory": self._memory,
            "memory_service": self._memory,
            "docker_service": self._db.docker_service if hasattr(self._db, "docker_service") else None,
            "sys_helpers": self,
        }
        return await strategy.execute(context=context, dependencies=dependencies)
    def looks_like_reasoning_leak(self, text: str) -> bool:
        from ... import utils
        return utils._looks_like_reasoning_leak(text)

    def is_brief_chat_query(self, query: str) -> bool:
        from ... import utils
        return utils._is_brief_chat_query(query)

    def extract_path_focus_terms(self, query: str, topics: list[str]) -> tuple[list[str], bool]:
        from ... import utils
        return utils._extract_path_focus_terms(query=query, topics=topics)
