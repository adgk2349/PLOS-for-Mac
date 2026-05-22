from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from ... import utils
from ....models import (
    BehaviorPolicy,
    Citation,
    ComposedChatResponseV2,
    ExecutionResult,
    LocalPlan,
    ParsedIntent,
    PrivacyMode,
    ReasoningIntent,
    SuggestedActionKind,
    VerificationResult,
)
from ....nlu.followup_resolver import FollowUpResolution
from ...context import ReasoningContext
from ...executor_contract import bind_async_executor_contract, require_executor_methods
from ....web_retrieval import WebRetrievalReport, WebRetriever
from ....language_utils import detect_query_language, normalize_language_code
from ...helpers.web.general_chat_web_gate_helpers import GeneralChatWebGateHelpers
from ...helpers.web.general_chat_web_execution_helpers import GeneralChatWebExecutionHelpers
from ...helpers.chat.general_chat_recall_gate_helpers import GeneralChatRecallGateHelpers
from ...helpers.chat.general_chat_recall_execution_helpers import GeneralChatRecallExecutionHelpers
from ...helpers.chat.general_chat_conversation_execution_helpers import GeneralChatConversationExecutionHelpers
from ...answer_contract import (
    coerce_answer_type_hint,
    extract_contract_response,
    infer_answer_type_hint,
    validate_contract_response,
)


class GeneralChatWebMixin:

    @staticmethod
    def _is_explicit_web_search_request(query: str) -> bool:
        # Keep web-intent detection aligned with centralized routing utilities.
        # This prevents stale per-strategy rules from overriding local-only cues.
        try:
            return bool(GeneralChatWebGateHelpers.is_explicit_web_search_request(query))
        except Exception:
            lowered = unicodedata.normalize("NFC", (query or "").strip()).lower()
            if not lowered:
                return False
            if "http://" in lowered or "https://" in lowered:
                return True
            web_targets = ("인터넷", "웹", "web", "online", "링크", "url", "사이트", "site")
            web_actions = ("검색", "search", "찾아", "look up", "크롤", "crawl")
            return any(t in lowered for t in web_targets) and any(a in lowered for a in web_actions)

    @staticmethod
    def _searxng_connection_refused(logs: list[str]) -> bool:
        return GeneralChatWebExecutionHelpers.searxng_connection_refused(logs)

    @staticmethod
    def _wait_for_searxng_http(base_url: str, *, timeout_seconds: float = 8.0) -> bool:
        return GeneralChatWebExecutionHelpers.wait_for_searxng_http(base_url, timeout_seconds=timeout_seconds)

    @staticmethod
    async def _wait_for_searxng_http_async(base_url: str, *, timeout_seconds: float = 8.0) -> bool:
        return await GeneralChatWebExecutionHelpers.wait_for_searxng_http_async(base_url, timeout_seconds=timeout_seconds)

    @staticmethod
    async def _docker_is_running_async(docker_service: Any) -> bool:
        return await GeneralChatWebExecutionHelpers.docker_is_running_async(docker_service)

    @staticmethod
    async def _docker_start_async(docker_service: Any, *, keep_running: bool) -> bool:
        return await GeneralChatWebExecutionHelpers.docker_start_async(docker_service, keep_running=keep_running)

    async def ensure_local_searxng_ready_async(
        self,
        *,
        docker_service: Any,
        keep_running: bool,
        allow_auto_stop: bool,
        host: str,
        port: int,
        searxng_url: str,
        port_timeout_seconds: float = 6.0,
        http_timeout_seconds: float = 8.0,
    ) -> tuple[bool, list[str]]:
        return await GeneralChatWebExecutionHelpers.ensure_local_searxng_ready_async(
            self,
            docker_service=docker_service,
            keep_running=keep_running,
            allow_auto_stop=allow_auto_stop,
            host=host,
            port=port,
            searxng_url=searxng_url,
            port_timeout_seconds=port_timeout_seconds,
            http_timeout_seconds=http_timeout_seconds,
        )

    @staticmethod
    def _recent_web_memory(last_context: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(last_context, dict):
            return {}
        web_summary = str(last_context.get("web_summary") or "").strip()
        web_query = str(last_context.get("web_query") or "").strip()
        raw_sources = last_context.get("web_sources")
        web_sources: list[dict[str, str]] = []
        if isinstance(raw_sources, list):
            for item in raw_sources[:4]:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                web_sources.append(
                    {
                        "title": str(item.get("title") or "").strip()[:160],
                        "url": url[:500],
                        "snippet": str(item.get("snippet") or "").strip()[:280],
                    }
                )
        if not web_summary and not web_sources:
            return {}
        return {
            "query": web_query,
            "summary": web_summary,
            "sources": web_sources,
        }

    @staticmethod
    def _web_sources_for_prompt(sources: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for idx, item in enumerate(sources[:4], start=1):
            title = str(item.get("title") or "").strip() or f"source {idx}"
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            lines.append(f"[{idx}] {title}")
            if url:
                lines.append(f"URL: {url}")
            if snippet:
                lines.append(f"요약: {snippet}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _deterministic_web_summary(*, query: str, sources: list[dict[str, str]], language: str) -> str:
        if language == "ko":
            header = "웹 검색 근거를 정리하면 아래와 같습니다."
            lead = f"질문: {query.strip()}" if query.strip() else ""
            bullets: list[str] = []
            for idx, item in enumerate(sources[:3], start=1):
                title = str(item.get("title") or "").strip() or "제목 없음"
                snippet = str(item.get("snippet") or "").strip()
                if snippet:
                    bullets.append(f"- [{idx}] {title}: {snippet}")
                else:
                    bullets.append(f"- [{idx}] {title}")
            refs = [f"[{i}] {s.get('url','')}" for i, s in enumerate(sources[:3], start=1) if str(s.get("url") or "").strip()]
            body = "\n".join([header, lead, *bullets, "출처:", *refs]).strip()
            return body
        header = "Here is a concise summary from the web evidence."
        bullets = []
        for idx, item in enumerate(sources[:3], start=1):
            title = str(item.get("title") or "").strip() or "untitled"
            snippet = str(item.get("snippet") or "").strip()
            bullets.append(f"- [{idx}] {title}" + (f": {snippet}" if snippet else ""))
        refs = [f"[{i}] {s.get('url','')}" for i, s in enumerate(sources[:3], start=1) if str(s.get("url") or "").strip()]
        return "\n".join([header, *bullets, "Sources:", *refs]).strip()

    @staticmethod
    def _is_web_context_followup(*, query: str, last_context: dict[str, Any] | None, followup_web_search: bool) -> bool:
        if followup_web_search:
            return True
        if not isinstance(last_context, dict):
            return False
        path = str(last_context.get("conversation_path") or "").strip().lower()
        if not (path.startswith("external_web_search") or path == "session_web_memory_reused"):
            return False
        return utils._has_followup_context_signal(query)

    @classmethod
    def _is_relevant_web_memory_entry(
        cls,
        *,
        query: str,
        entry: dict[str, Any],
        last_context: dict[str, Any] | None,
        followup_web_search: bool,
    ) -> bool:
        lexical = max(0.0, min(1.0, float(entry.get("lexical_score") or 0.0)))
        vector = max(0.0, min(1.0, float(entry.get("vector_score") or 0.0)))
        entry_query = str(entry.get("query") or "").strip()
        entry_summary = str(entry.get("answer_summary") or "").strip()
        if cls._is_web_context_followup(
            query=query,
            last_context=last_context,
            followup_web_search=followup_web_search,
        ):
            return lexical >= 0.03 or vector >= 0.45
        if lexical < 0.08:
            return False
        if utils._has_token_overlap(query, entry_query, min_overlap=1):
            return True
        if utils._has_token_overlap(query, entry_summary, min_overlap=2):
            return True
        return False

    def _query_variant_for_round(
        self,
        *,
        original_query: str,
        round_index: int,
        round1_sources: list[dict[str, str]],
    ) -> str:
        return GeneralChatWebExecutionHelpers.query_variant_for_round(
            self,
            original_query=original_query,
            round_index=round_index,
            round1_sources=round1_sources,
        )

    @staticmethod
    def _source_rows_from_report(report: WebRetrievalReport) -> list[dict[str, str]]:
        return GeneralChatWebExecutionHelpers.source_rows_from_report(report)

    @staticmethod
    def _is_uncertain_web_round(
        *,
        report: WebRetrievalReport,
        freshness_sensitive_query: bool,
    ) -> bool:
        return GeneralChatWebExecutionHelpers.is_uncertain_web_round(
            report=report,
            freshness_sensitive_query=freshness_sensitive_query,
        )

    def _run_web_reasoning_loop(
        self,
        *,
        retriever: WebRetriever,
        base_query: str,
        freshness_sensitive_query: bool,
        searxng_url: str | None,
        prefer_searxng: bool,
        max_rounds: int = 3,
        max_total_seconds: float = 18.0,
        round_timeout_seconds: float = 6.0,
    ) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
        return GeneralChatWebExecutionHelpers.run_web_reasoning_loop(
            self,
            retriever=retriever,
            base_query=base_query,
            freshness_sensitive_query=freshness_sensitive_query,
            searxng_url=searxng_url,
            prefer_searxng=prefer_searxng,
            max_rounds=max_rounds,
            max_total_seconds=max_total_seconds,
            round_timeout_seconds=round_timeout_seconds,
        )
