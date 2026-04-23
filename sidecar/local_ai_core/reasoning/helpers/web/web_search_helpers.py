from __future__ import annotations
import asyncio
import logging
logger = logging.getLogger(__name__)
from typing import Any
import os
import json
import re
import time
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

class WebSearchHelpers:
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
    def _should_auto_web_search(
        *,
        query: str,
        parsed_intent: ParsedIntent,
        last_context: dict | None,
    ) -> bool:
        lowered = utils._normalized_match_text(query)
        if not lowered:
            return False
        if utils._is_followup_web_search_request(query=query, last_context=last_context):
            return True
        
        # Soften suppression: only block if it's strictly a local file task
        if utils._has_local_file_target_cues(lowered):
            # If it also looks like it needs freshness/external info, allow it
            if not utils._is_freshness_sensitive_query(lowered):
                return False
                
        if utils._is_explicit_web_search_request(lowered):
            return True
        if utils._is_freshness_sensitive_query(lowered):
            return True
            
        confidence = float(getattr(parsed_intent, "confidence", 0.0) or 0.0)
        ambiguity = str(getattr(parsed_intent, "ambiguity", "clear") or "clear").lower()
        has_search_action = any(token in lowered for token in ("검색", "search", "찾아", "crawl", "크롤", "확인", "check"))
        low_confidence = confidence < 0.58 or ambiguity == "unclear"
        
        if low_confidence and has_search_action:
            return True
        return False

    @staticmethod
    def _is_context_carryover_web_search_request(*, query: str, last_context: dict | None) -> bool:
        if not isinstance(last_context, dict):
            return False
        lowered = utils._normalized_match_text(query)
        if not lowered:
            return False
        previous_query = str(last_context.get("last_user_query") or "").strip()
        if not previous_query:
            return False
        local_doc_tokens = (
            "파일",
            "문서",
            "폴더",
            ".txt",
            ".pdf",
            ".md",
            ".docx",
            "file",
            "document",
            "folder",
            "directory",
        )
        if any(token in lowered for token in local_doc_tokens):
            return False
        if not utils._is_explicit_web_search_request(lowered):
            return False

        # Short directives like "인터넷 검색해봐" should inherit the previous topic.
        stripped = lowered
        generic_tokens = (
            "인터넷에서",
            "인터넷",
            "웹에서",
            "웹",
            "online",
            "web",
            "search",
            "검색해봐",
            "검색해 줘",
            "검색해줘",
            "검색",
            "찾아봐",
            "찾아 줘",
            "찾아줘",
            "look up",
            "look it up",
            "해봐",
            "해 줘",
            "해줘",
            "좀",
            "한번",
            "다시",
        )
        for token in generic_tokens:
            stripped = stripped.replace(token, " ")
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if not stripped:
            return True
        deictic_tokens = {"그거", "그걸", "그거를", "이거", "이걸", "저거", "that", "this", "it"}
        residue = [token for token in re.findall(r"[A-Za-z가-힣0-9_]+", stripped) if token]
        return bool(residue) and all(token in deictic_tokens for token in residue)

    @staticmethod
    def _is_followup_web_search_request(*, query: str, last_context: dict | None) -> bool:
        if not isinstance(last_context, dict):
            return False
        lowered = utils._normalized_match_text(query)
        if not lowered:
            return False
        if utils._contains_explicit_local_only_constraint(lowered):
            return False
        if utils._is_context_carryover_web_search_request(query=lowered, last_context=last_context):
            return True
        path = str(last_context.get("conversation_path") or "").strip().lower()
        if not path.startswith("external_web_search"):
            return False
        local_doc_tokens = (
            "파일",
            "문서",
            "폴더",
            ".txt",
            ".pdf",
            ".md",
            ".docx",
            "file",
            "document",
            "folder",
            "directory",
        )
        if any(token in lowered for token in local_doc_tokens):
            return False
        if utils._is_explicit_web_search_request(lowered):
            return True
        # Avoid dragging unrelated runtime/model questions into web mode just because
        # the previous turn happened to be a web-search turn.
        non_web_topic_tokens = (
            "모델",
            "성능",
            "benchmark",
            "벤치",
            "엔진",
            "파이프라인",
            "pipeline",
            "설정",
            "runtime",
            "로컬",
            "local",
            "메모리",
            "토큰",
            "context",
            "qwen",
            "llama",
            "gemma",
            "gpt",
            "mlx",
        )
        if any(token in lowered for token in non_web_topic_tokens):
            return False
        strong_followup_tokens = (
            "검색",
            "search",
            "찾아",
            "재검색",
            "다시검색",
            "다시 검색",
            "다시 찾아",
            "크롤",
            "crawl",
            "프로필 검색",
            "링크",
            "url",
            "출처",
            "source",
            "결과",
            "나왔",
            "더 자세",
            "자세히",
            "근거",
            "근거 더",
            "공식 링크",
        )
        if any(token in lowered for token in strong_followup_tokens):
            return True
        contextual_followup_tokens = (
            "언제",
            "몇년",
            "몇 년",
            "누구",
            "어디",
            "왜",
            "어떻게",
            "방송",
            "영상",
            "레전드",
            "더",
            "자세",
            "알려",
            "결과",
            "나왔",
        )
        if any(token in lowered for token in contextual_followup_tokens):
            return True
        return False

    def _web_search_query_for_turn(*, query: str, last_context: dict | None, is_followup_web_search: bool) -> str:
        cleaned = str(query or "").strip()
        if not cleaned or not is_followup_web_search or not isinstance(last_context, dict):
            return cleaned
        previous_query = str(last_context.get("last_user_query") or "").strip()
        previous_summary = str(last_context.get("result_summary") or "").strip()
        anchor = previous_query or previous_summary
        if not anchor:
            return cleaned
        if utils._has_token_overlap(cleaned, anchor, min_overlap=1):
            return cleaned
        return f"{anchor}\n    \n    후속 질문: {cleaned}"

    @staticmethod
    def _external_web_search_permission_reason(settings) -> str | None:
        mode = getattr(settings, "privacy_mode", PrivacyMode.LOCAL_ONLY)
        if mode == PrivacyMode.LOCAL_ONLY:
            return "local_only"
        if mode == PrivacyMode.CONFIRM_BEFORE_EXTERNAL:
            return "confirm_required"
        if mode == PrivacyMode.HYBRID and not bool(getattr(settings, "hybrid_web_search_enabled", False)):
            return "hybrid_web_off"
        return None

    def _web_search_blocked_execution(
        *,
        response_language: str,
        reason: str,
        additional_tool_logs: list[str] | None = None,
        web_sources_count: int = 0,
        web_fetch_failures: int = 0,
    ) -> ExecutionResult:
        if response_language == "ko":
            mapping = {
                "local_only": "현재 프라이버시 모드가 로컬 전용이라 인터넷 검색을 실행할 수 없습니다. 설정에서 하이브리드로 전환해 주세요.",
                "hybrid_web_off": "하이브리드 모드지만 웹검색(인터넷 경로)이 꺼져 있어 인터넷 검색을 실행할 수 없습니다. 프라이버시 설정에서 웹검색 허용을 켜주세요.",
                "confirm_required": "외부 호출은 승인 모드입니다. 외부 호출을 승인한 뒤 다시 요청해 주세요.",
                "provider_unavailable": "인터넷에서 신뢰 가능한 근거를 수집하지 못해 답을 확정할 수 없습니다. 네트워크 상태를 확인한 뒤 다시 시도해 주세요.",
            }
        else:
            mapping = {
                "local_only": "Internet search is blocked because privacy mode is LOCAL_ONLY. Switch to HYBRID in settings.",
                "hybrid_web_off": "Internet search is disabled because HYBRID web search is turned off. Enable it in privacy settings.",
                "confirm_required": "External calls require approval. Approve external access and retry.",
                "provider_unavailable": "I could not gather reliable web evidence, so I cannot provide a grounded answer yet. Check network and retry.",
            }
        text = mapping.get(reason, mapping.get("provider_unavailable", "Internet search is unavailable right now."))
        merged_logs = ["planning:web_search_requested", "web_search:requested", *(additional_tool_logs or [])]
        merged_logs.append(f"web_search:blocked:{reason}")
        return ExecutionResult(
            result_type="conversation",
            structured_payload={
                "style": "general_chat",
                "source": "web_search_gate",
                "ungrounded_allowed": True,
                "web_path": "blocked",
                "web_sources_count": max(0, int(web_sources_count)),
                "web_fetch_failures": max(0, int(web_fetch_failures)),
            },
            citations=[],
            tool_logs=merged_logs,
            generated_text=text,
            engine_used=None,
            used_fallback=False,
            runtime_detail=(
                f"web_search_blocked_reason={reason}|"
                f"web_sources_count={max(0, int(web_sources_count))}|"
                f"web_fetch_failures={max(0, int(web_fetch_failures))}"
            ),
        )

    async def _execute_direct_web_search(
        self,
        *,
        query: str,
        mode: WorkMode,
        response_language: str,
        workspace,
        settings,
        response_length: str,
    ) -> ExecutionResult | None:
        if self._web_retriever is None:
            return None
        report = await asyncio.to_thread(
            self._web_retriever.run,
            query=query,
            max_candidates=8,
            max_sources=3,
        )
        self._last_web_report = {
            "logs": list(report.logs),
            "sources_count": len(report.sources),
            "fetch_failures": int(report.fetch_failure_count),
            "discovered_count": int(report.discovered_count),
            "failure_reason": str(report.failure_reason or ""),
        }
        if not report.sources:
            return None

        now = datetime.now(timezone.utc)
        web_citations: list[Citation] = []
        for idx, source in enumerate(report.sources, start=1):
            score = max(0.2, 0.86 - ((idx - 1) * 0.15))
            snippet = source.content.strip() or source.snippet.strip()
            if not snippet:
                continue
            web_citations.append(
                Citation(
                    doc_id=f"web:{idx}",
                    chunk_id=f"web:{idx}:chunk",
                    file_path=source.url,
                    snippet=snippet[:900],
                    score=score,
                    modified_at=now,
                    category="web",
                    subcategory="internet",
                    tags=["web"],
                    document_type="web_page",
                    importance=0.6,
                )
            )
        if not web_citations:
            return None

        if response_language == "ko":
            web_prompt = (
                f"질문: {query}\n    \n    "
                "아래 웹 출처 스니펫만 근거로 답하세요.\n    "
                "모르는 내용은 추측하지 말고 모른다고 말하세요.\n    "
                "답변 끝에 참고 링크를 1~3개 줄바꿈으로 제시하세요."
            )
        else:
            web_prompt = (
                f"Question: {query}\n    \n    "
                "Answer only from the web source snippets below.\n    "
                "Do not guess when evidence is insufficient.\n    "
                "End with 1-3 reference URLs."
            )

        inference = await self._executor.generate_async(
            query=web_prompt,
            mode=mode if mode != WorkMode.GENERAL else WorkMode.RESEARCH,
            citations=web_citations,
            profile=workspace.startup_profile.value,
            engine=settings.local_engine,
            mlx_model_path=settings.mlx_model_path,
            llama_model_path=settings.llama_model_path,
            language_preference=settings.language,
            max_tokens=self._conversation_max_tokens(
                response_length,
                model_profile=getattr(settings, "model_profile", "recommended"),
                query=query,
            )
            + 80,
            timeout_seconds=float(os.getenv("LOCAL_AI_INFERENCE_TIMEOUT_SECONDS", "40")),
        )

        answer = str(inference.answer or "").strip()
        if not answer:
            if response_language == "ko":
                lines = []
                for idx, source in enumerate(report.sources[:3], start=1):
                    title = source.title or source.url
                    lines.append(f"{idx}. {title}\n       {source.url}")
                answer = "웹에서 찾은 관련 자료입니다:\n    " + "\n    ".join(lines)
            else:
                lines = []
                for idx, source in enumerate(report.sources[:3], start=1):
                    title = source.title or source.url
                    lines.append(f"{idx}. {title}\n       {source.url}")
                answer = "I found these related web sources:\n    " + "\n    ".join(lines)

        tool_logs = [
            "planning:web_search_requested",
            "web_search:requested",
            "web_search:direct",
            *report.logs,
            f"done:web_evidence_composed:{len(web_citations)}",
            f"inference:{inference.engine_used.value}",
        ]
        detail_prefix = str(inference.detail or "").strip()
        detail_suffix = (
            f"web_path=direct|web_sources_count={len(web_citations)}|"
            f"web_fetch_failures={int(report.fetch_failure_count)}"
        )
        runtime_detail = f"{detail_prefix}|{detail_suffix}" if detail_prefix else detail_suffix
        return ExecutionResult(
            result_type="conversation",
            structured_payload={
                "style": "general_chat",
                "source": "web_search_direct",
                "ungrounded_allowed": True,
                "web_path": "direct",
                "web_sources_count": len(web_citations),
                "web_fetch_failures": int(report.fetch_failure_count),
            },
            citations=web_citations[:3],
            tool_logs=tool_logs,
            generated_text=answer,
            engine_used=inference.engine_used,
            used_fallback=inference.used_fallback,
            runtime_detail=runtime_detail,
        )

    def _external_provider_endpoint(provider: str) -> str:
        mapping = {
            "openai": "https://api.openai.com/v1/responses",
            "anthropic": "https://api.anthropic.com/v1/messages",
        }
        return mapping.get(str(provider or "").strip().lower(), "")

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
        if not bool(getattr(settings, "hybrid_web_search_enabled", False)):
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
            endpoint = self._external_provider_endpoint(provider)
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
            trace_logs: list[str] = []
            if endpoint:
                trace_logs.append(f"retrieving:{endpoint}")
                trace_logs.append(f"retrieved:{endpoint}")
            execution = ExecutionResult(
                result_type="summary",
                structured_payload={
                    "source": "external_summary_escalated",
                    "provider": provider,
                },
                citations=citations,
                tool_logs=[*trace_logs, f"external_escalated_summary:{provider}"],
                generated_text=text,
                engine_used=None,
                used_fallback=False,
                runtime_detail=f"external_escalated_provider={provider}",
            )
            return execution, provider
        return None

    def _summary_external_query(*, query: str, language_preference: str | None) -> str:
        language = resolve_response_language(query, language_preference)
        if language == "ko":
            return (
                f"{query}\n    \n    "
                "출력 형식 규칙:\n    "
                "1) 반드시 번호 목록 1~5로 작성\n    "
                "2) 원문 문장 복붙/중복 금지\n    "
                "3) 각 줄은 핵심 개념 1개만 간결하게 작성"
            )
        return (
            f"{query}\n    \n    "
            "Output rules:\n    "
            "1) Return exactly 5 numbered points.\n    "
            "2) Avoid verbatim copy and repetition.\n    "
            "3) Each line should contain one concise core idea."
        )

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
