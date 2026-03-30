from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
from typing import Any
import os
import json
import re
from datetime import datetime, timezone
from collections import Counter, deque
from pathlib import Path

from ... import utils

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
from ....retrieval import extract_query_hints, merge_filters, retrieve_bundle
from ....vector_store import VectorStore
from ....verifier import ResultVerifier

class FormattingHelpers:
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
    def _conversation_session_summary(
        *,
        query: str,
        session_digest: dict[str, Any] | None,
        last_context: dict | None,
        memory_bundle,
        response_length: str,
        model_profile: str = "recommended",
    ) -> str:
        digest = session_digest or {}
        followup_context = utils._has_followup_context_signal(query)
        strong_followup_context = utils._has_strong_followup_context_signal(query)
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

        semantic_memories = [
            f"- related_past: {m['text'][:180]}" + (" (from other session)" if m.get("is_global") else "")
            for m in getattr(memory_bundle, "semantic_memories", [])
        ]

        sections = [recent_turns]
        if semantic_memories:
            sections.append(semantic_memories)
        if followup_context:
            sections.extend([last_context_lines, open_loops])
            if strong_followup_context:
                sections.extend([stable_facts, active_topics])
        budget = utils._conversation_context_budget_tokens(
            response_length,
            model_profile=model_profile,
        )
        lines: list[str] = []
        used = 0
        for section_lines in sections:
            for line in section_lines:
                tokens = utils._estimate_context_tokens(line)
                if used + tokens > budget:
                    continue
                lines.append(line)
                used += tokens
        return "\n    ".join(lines).strip()

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

    async def _refresh_digest_with_local_model(self, session_id: str, digest: dict[str, Any]) -> dict[str, Any] | None:
        settings = self._db.get_settings()
        workspace = self._db.get_workspace()
        prompt = self._digest_model_refresh_prompt(digest)
        inference = await self._executor.generate_conversational_async(
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
            timeout_seconds=float(os.getenv("LOCAL_AI_INFERENCE_TIMEOUT_SECONDS", "40")),
        )
        answer = str(inference.answer or "").strip()
        if not answer or self._looks_like_reasoning_leak(answer):
            return None
        parsed = self._parse_digest_model_output(answer)
        if parsed is None:
            return None
        return parsed

    def _digest_model_refresh_prompt(digest: dict[str, Any]) -> str:
        compact = {
            "active_topics": digest.get("active_topics") or [],
            "stable_facts": digest.get("stable_facts") or [],
            "open_loops": digest.get("open_loops") or [],
            "recent_turns": digest.get("recent_turns") or [],
        }
        payload = json.dumps(compact, ensure_ascii=False)
        return (
            "다음은 대화 메모리 digest입니다. 잡음을 제거하고 다음 JSON 형식으로만 답하세요.\n    "
            '{"active_topics":["..."],"stable_facts":["..."],"open_loops":["..."],"recent_turns":[{"role":"user|assistant","text":"..."}]}\n    '
            "규칙: active_topics<=8, stable_facts<=10, open_loops<=6, recent_turns<=8. "
            "중복/지시문/정책문구는 제거하세요.\n    "
            f"digest={payload}"
        )

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
