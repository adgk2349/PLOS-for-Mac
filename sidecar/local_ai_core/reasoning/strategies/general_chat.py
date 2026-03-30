import asyncio
from typing import Any
from datetime import datetime, timezone
import os
import re
import socket
import time
import unicodedata
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .base import ReasoningStrategy
from .. import utils
from ...models import (
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
from ...nlu.followup_resolver import FollowUpResolution
from ..context import ReasoningContext
from ..executor_contract import bind_async_executor_contract, require_executor_methods
from ...web_retrieval import WebRetrievalReport, WebRetriever


class GeneralChatStrategy(ReasoningStrategy):
    """
    Handles unstructured conversation, conversational memory, and optional fallback to web search.
    """

    @staticmethod
    def _is_explicit_web_search_request(query: str) -> bool:
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
        for item in logs or []:
            low = str(item or "").lower()
            if "searxng" not in low:
                continue
            if "connection refused" in low or "errno 61" in low:
                return True
        return False

    @staticmethod
    def _wait_for_port(host: str, port: int, *, timeout_seconds: float = 6.0) -> bool:
        deadline = time.time() + max(0.5, timeout_seconds)
        while time.time() < deadline:
            try:
                with socket.create_connection((host, int(port)), timeout=0.8):
                    return True
            except Exception:
                time.sleep(0.25)
        return False

    @staticmethod
    def _wait_for_searxng_http(base_url: str, *, timeout_seconds: float = 8.0) -> bool:
        parsed = urlparse(str(base_url or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        query = urlencode({"q": "ping"})
        probe_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path or '/search'}?{query}"
        deadline = time.time() + max(1.0, timeout_seconds)
        while time.time() < deadline:
            try:
                req = Request(
                    probe_url,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Accept-Language": "ko,en-US;q=0.9",
                    },
                )
                with urlopen(req, timeout=1.2) as resp:  # nosec B310
                    code = int(getattr(resp, "status", 200))
                    if 200 <= code < 500:
                        return True
            except Exception:
                time.sleep(0.3)
        return False

    @staticmethod
    def _has_memory_recall_cue(query: str) -> bool:
        lowered = unicodedata.normalize("NFC", (query or "").strip()).lower()
        if not lowered:
            return False
        cues = (
            "그거", "이거", "방금", "아까", "요약", "정리", "다시", "핵심", "출처", "근거", "비교",
            "that", "this", "summarize", "summary", "again", "source", "evidence",
        )
        return any(token in lowered for token in cues)

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

    async def _run_conversation_inference(
        self,
        *,
        executor,
        query: str,
        context: ReasoningContext,
        max_tokens: int,
    ) -> ExecutionResult:
        timeout_seconds = float(os.getenv("LOCAL_AI_INFERENCE_TIMEOUT_SECONDS", "40"))
        try:
            executor = bind_async_executor_contract(executor)
            require_executor_methods(executor, "execute_conversation_async")
            return await executor.execute_conversation_async(
                query=query,
                mode=context.req.mode,
                startup_profile=context.workspace.startup_profile,
                engine=context.settings.local_engine,
                mlx_model_path=context.settings.mlx_model_path,
                llama_model_path=context.settings.llama_model_path,
                language_preference=context.settings.language,
                session_summary=context.session_digest or "",
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timeout_text = (
                "로컬 추론이 제한 시간 내에 완료되지 않았습니다. 질문을 조금 나누어 다시 시도해 주세요."
                if context.response_language == "ko"
                else "Local inference exceeded the time limit. Please retry with a narrower prompt."
            )
            return ExecutionResult(
                result_type="conversation",
                structured_payload={"reason": "inference_timeout", "ungrounded_allowed": True},
                citations=[],
                tool_logs=["inference_timeout:conversation"],
                generated_text=timeout_text,
                engine_used=context.settings.local_engine,
                used_fallback=True,
                runtime_detail=f"inference_timeout:conversation:{timeout_seconds:.1f}s",
            )

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
    def _normalize_space(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @classmethod
    def _trim_redundant_opening_from_last_context(
        cls,
        *,
        answer: str,
        last_context: dict[str, Any] | None,
    ) -> str:
        candidate = str(answer or "").strip()
        if not candidate or not isinstance(last_context, dict):
            return candidate
        previous = cls._normalize_space(str(last_context.get("result_summary") or ""))
        if not previous:
            return candidate

        normalized_answer = cls._normalize_space(candidate)
        if not normalized_answer:
            return candidate

        previous_head = previous
        sentence_split = re.split(r"(?<=[.!?。！？])\s+|\n+", previous)
        if sentence_split:
            first_sentence = cls._normalize_space(sentence_split[0])
            if len(first_sentence) >= 18:
                previous_head = first_sentence

        if normalized_answer.startswith(previous):
            stripped = normalized_answer[len(previous):].lstrip(" \n\t:;,-")
            if len(stripped) >= 12:
                return stripped
        if previous_head and normalized_answer.startswith(previous_head):
            stripped = normalized_answer[len(previous_head):].lstrip(" \n\t:;,-")
            if len(stripped) >= 12:
                return stripped
        return candidate

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

    @staticmethod
    def _conversation_query_with_context(
        *,
        query: str,
        response_language: str,
        followup_resolution: FollowUpResolution | None,
        last_context: dict[str, Any] | None,
    ) -> str:
        text = str(query or "").strip()
        if not text or not isinstance(last_context, dict):
            return text
        lowered = unicodedata.normalize("NFC", text).lower()
        token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered))
        has_followup_signal = bool(
            (followup_resolution and followup_resolution.is_followup)
            or utils._has_followup_context_signal(text)
            or utils._has_strong_followup_context_signal(text)
        )
        if not has_followup_signal and token_count > 7:
            return text

        previous_query = str(last_context.get("last_user_query") or "").strip()
        previous_summary = str(last_context.get("result_summary") or "").strip()
        if not previous_query and not previous_summary:
            return text
        previous_query = previous_query[:220]
        previous_summary = previous_summary[:360]
        if response_language == "ko":
            return (
                "이전 대화 맥락을 참고해 답하세요.\n"
                f"- 이전 사용자 질문: {previous_query}\n"
                f"- 이전 답변 요약: {previous_summary}\n\n"
                f"현재 요청: {text}\n"
                "규칙: 현재 요청이 후속 질문이면 같은 주제로 이어서 답하고, 새 주제가 명확할 때만 전환하세요."
            )
        return (
            "Use prior conversation context before answering.\n"
            f"- Previous user question: {previous_query}\n"
            f"- Previous answer summary: {previous_summary}\n\n"
            f"Current request: {text}\n"
            "Rule: If this is a follow-up, keep the same topic unless the user clearly switches topics."
        )

    @staticmethod
    def _contains_korean(text: str) -> bool:
        return bool(re.search(r"[가-힣]", str(text or "")))

    @staticmethod
    def _tokenize_keywords(text: str, *, max_tokens: int = 4) -> list[str]:
        raw_tokens = re.findall(r"[A-Za-z가-힣0-9_]{2,24}", str(text or "").lower())
        stop = {
            "latest", "update", "official", "source", "search", "info", "information",
            "최신", "업데이트", "공식", "출처", "검색", "정보", "정리", "요약", "알려", "말해",
        }
        output: list[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            if token in stop:
                continue
            if token in seen:
                continue
            seen.add(token)
            output.append(token)
            if len(output) >= max(1, max_tokens):
                break
        return output

    def _query_variant_for_round(
        self,
        *,
        original_query: str,
        round_index: int,
        round1_sources: list[dict[str, str]],
    ) -> str:
        base = " ".join(str(original_query or "").split()).strip()
        if round_index <= 1 or not base:
            return base
        if round_index == 2:
            entities = self._tokenize_keywords(base, max_tokens=2)
            suffix = "latest update"
            if self._contains_korean(base):
                suffix = "latest update 최신 업데이트"
            return " ".join([base, *entities, suffix]).strip()
        title_blob = " ".join(str(item.get("title") or "") for item in round1_sources[:3] if isinstance(item, dict))
        title_terms = self._tokenize_keywords(title_blob, max_tokens=2)
        return " ".join([base, *title_terms, "official source"]).strip()

    @staticmethod
    def _source_rows_from_report(report: WebRetrievalReport) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for source in report.sources[:6]:
            title = str(source.title or source.url)[:160]
            url = str(source.url or "").strip()[:500]
            snippet = str(source.snippet or source.content or "")[:320]
            if not url:
                continue
            rows.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )
        return rows

    @staticmethod
    def _is_uncertain_web_round(
        *,
        report: WebRetrievalReport,
        freshness_sensitive_query: bool,
    ) -> bool:
        if int(report.usable_source_count) < 2:
            return True
        if int(report.unique_domain_count) < 2:
            return True
        if int(report.fetch_failure_count) > int(report.fetch_success_count):
            return True
        if freshness_sensitive_query and int(report.usable_source_count) < 3:
            return True
        return False

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
        start_ts = time.time()
        collected_logs: list[str] = []
        round_queries: list[str] = []
        all_candidates: list[dict[str, Any]] = []
        round1_sources: list[dict[str, str]] = []
        converged = False
        last_quality = 0.0

        for round_index in range(1, max(1, int(max_rounds)) + 1):
            if (time.time() - start_ts) >= max_total_seconds:
                collected_logs.append("web_loop:budget_exhausted")
                break
            round_query = self._query_variant_for_round(
                original_query=base_query,
                round_index=round_index,
                round1_sources=round1_sources,
            )
            round_queries.append(round_query)
            collected_logs.append(f"web_loop:round={round_index}|query={round_query[:200]}")
            collected_logs.append("web_loop:retrieving")
            report = retriever.run(
                query=round_query or base_query,
                max_candidates=8,
                max_sources=3,
                searxng_url=searxng_url,
                prefer_searxng=prefer_searxng,
                freshness_sensitive=freshness_sensitive_query,
            )
            collected_logs.extend(list(report.logs or []))
            round_sources = self._source_rows_from_report(report)
            if round_index == 1:
                round1_sources = list(round_sources)
            last_quality = float(max(0.0, min(1.0, report.quality_score or 0.0)))
            collected_logs.append(
                "web_loop:quality="
                f"{last_quality:.3f}|usable={int(report.usable_source_count)}|domains={int(report.unique_domain_count)}"
                f"|success={int(report.fetch_success_count)}|failure={int(report.fetch_failure_count)}"
            )

            for idx, source in enumerate(round_sources, start=1):
                candidate_score = max(0.0, min(1.0, last_quality - ((idx - 1) * 0.05)))
                all_candidates.append(
                    {
                        **source,
                        "score": candidate_score,
                    }
                )

            uncertain = self._is_uncertain_web_round(
                report=report,
                freshness_sensitive_query=freshness_sensitive_query,
            )
            if not uncertain:
                converged = True
                collected_logs.append("web_loop:converged")
                break
            if round_index < max_rounds and (time.time() - start_ts) < max_total_seconds:
                collected_logs.append("web_loop:refine_triggered")
                continue
            collected_logs.append("web_loop:budget_exhausted")
            break

        deduped: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        sorted_candidates = sorted(all_candidates, key=lambda row: float(row.get("score") or 0.0), reverse=True)
        for row in sorted_candidates:
            url = str(row.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(
                {
                    "title": str(row.get("title") or "").strip()[:160],
                    "url": url[:500],
                    "snippet": str(row.get("snippet") or "").strip()[:320],
                }
            )
            if len(deduped) >= 3:
                break
        collected_logs.append("web_loop:finalizing")
        metadata = {
            "web_loop_rounds": len(round_queries),
            "web_loop_converged": bool(converged),
            "web_loop_quality_score": float(max(0.0, min(1.0, last_quality))),
            "web_loop_queries": round_queries[:3],
            "web_loop_timed_out": bool((time.time() - start_ts) >= max_total_seconds),
            "round_timeout_seconds": float(max(1.0, round_timeout_seconds)),
        }
        return deduped, collected_logs, metadata

    def handles_intent(self, intent: ParsedIntent, followup: FollowUpResolution | None) -> bool:
        return intent.intent == ReasoningIntent.GENERAL_CHAT

    async def execute(
        self,
        *,
        context: ReasoningContext,
        dependencies: dict[str, Any],
    ) -> ComposedChatResponseV2:
        executor = dependencies["executor"]
        composer = dependencies["composer"]

        plan = LocalPlan(
            plan_type="conversation",
            selected_files=[],
            selected_chunks=[],
            response_strategy="conversational_assistant",
            allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
            external_reasoning_needed=False,
        )

        execution: ExecutionResult | None = None
        conversation_path = "local_conversation"
        is_local = True
        response_length = getattr(context.memory_prefs, "response_length", "long") if context.memory_prefs else "long"
        model_profile = str(getattr(context.settings, "model_profile", "recommended") or "recommended")
        max_tokens = 1024 if response_length == "long" else 640 if response_length == "medium" else 320
        sys_helpers = dependencies.get("sys_helpers")
        if sys_helpers is not None:
            try:
                max_tokens = int(
                    sys_helpers.conversation_max_tokens(
                        response_length,
                        model_profile=model_profile,
                        query=context.req.query,
                    )
                )
            except Exception:
                pass
        web_memory_for_metadata: dict[str, Any] = {}
        web_memory_reused = False
        web_memory_rank_score = 0.0
        last_context = context.last_context if isinstance(context.last_context, dict) else {}
        memory_service = dependencies.get("memory") or dependencies.get("memory_service")
        freshness_sensitive_query = utils._is_freshness_sensitive_query(context.req.query)
        followup_web_search = utils._is_followup_web_search_request(
            query=context.req.query,
            last_context=last_context,
        )
        
        # Combined trigger for web search: explicit / follow-up / forced by assistant mode
        web_search_triggered = (
            context.force_web_search
            or self._is_explicit_web_search_request(context.req.query)
            or followup_web_search
            or freshness_sensitive_query
        )

        if web_search_triggered:
            mode = context.settings.privacy_mode
            if mode == PrivacyMode.LOCAL_ONLY or (
                mode == PrivacyMode.HYBRID and not bool(getattr(context.settings, "hybrid_web_search_enabled", False))
            ):
                blocked_text = (
                    "하이브리드 모드지만 웹검색(인터넷 경로)이 꺼져 있어 인터넷 검색을 실행할 수 없습니다. 프라이버시 설정에서 웹검색 허용을 켜주세요."
                    if context.response_language == "ko"
                    else "Internet search is disabled in current privacy settings. Enable hybrid web search."
                )
                execution = ExecutionResult(
                    result_type="conversation",
                    structured_payload={"web_path": "blocked", "ungrounded_allowed": True},
                    citations=[],
                    tool_logs=["web_search:blocked:privacy"],
                    generated_text=blocked_text,
                    engine_used=None,
                    used_fallback=False,
                    runtime_detail="web_search_blocked",
                )
                conversation_path = "external_web_search_blocked"
            else:
                retriever = WebRetriever()
                configured_searxng_url = str(getattr(context.settings, "searxng_url", "") or "").strip()
                if not configured_searxng_url:
                    configured_searxng_url = str(os.getenv("LOCAL_AI_SEARXNG_URL", "") or "").strip()
                if not configured_searxng_url:
                    configured_searxng_url = "http://127.0.0.1:8080/search"

                docker_service = dependencies.get("docker_service")
                keep_running = bool(getattr(context.settings, "auto_start_searxng", False))
                allow_auto_stop = not keep_running
                parsed_searx = urlparse(configured_searxng_url)
                is_local_searx = (
                    parsed_searx.scheme in {"http", "https"}
                    and (parsed_searx.hostname or "").strip().lower() in {"localhost", "127.0.0.1"}
                    and (parsed_searx.port in {None, 8080})
                )
                if is_local_searx and docker_service is not None:
                    try:
                        docker_service.mark_usage(allow_auto_stop=allow_auto_stop)
                        if not docker_service.is_running():
                            docker_service.start(keep_running=keep_running)
                        host = (parsed_searx.hostname or "127.0.0.1").strip() or "127.0.0.1"
                        port = int(parsed_searx.port or 8080)
                        self._wait_for_port(host, port, timeout_seconds=6.0)
                        self._wait_for_searxng_http(configured_searxng_url, timeout_seconds=8.0)
                    except Exception:
                        pass

                effective_web_query = utils._web_search_query_for_turn(
                    query=context.req.query,
                    last_context=last_context,
                    is_followup_web_search=followup_web_search,
                )
                source_rows, web_loop_logs, web_loop_meta = self._run_web_reasoning_loop(
                    retriever=retriever,
                    base_query=effective_web_query or context.req.query,
                    freshness_sensitive_query=bool(freshness_sensitive_query),
                    searxng_url=configured_searxng_url or None,
                    prefer_searxng=True,
                    max_rounds=3,
                    max_total_seconds=18.0,
                    round_timeout_seconds=6.0,
                )
                if (
                    not source_rows
                    and is_local_searx
                    and docker_service is not None
                    and self._searxng_connection_refused(web_loop_logs)
                ):
                    try:
                        docker_service.start(keep_running=keep_running)
                        host = (parsed_searx.hostname or "127.0.0.1").strip() or "127.0.0.1"
                        port = int(parsed_searx.port or 8080)
                        self._wait_for_port(host, port, timeout_seconds=20.0)
                        self._wait_for_searxng_http(configured_searxng_url, timeout_seconds=20.0)
                        retriever_retry = WebRetriever()
                        source_rows, retried_logs, retried_meta = self._run_web_reasoning_loop(
                            retriever=retriever_retry,
                            base_query=effective_web_query or context.req.query,
                            freshness_sensitive_query=bool(freshness_sensitive_query),
                            searxng_url=configured_searxng_url or None,
                            prefer_searxng=True,
                            max_rounds=3,
                            max_total_seconds=18.0,
                            round_timeout_seconds=6.0,
                        )
                        web_loop_logs.extend(retried_logs)
                        web_loop_meta = retried_meta
                    except Exception:
                        pass
                if is_local_searx and docker_service is not None:
                    try:
                        docker_service.mark_usage(allow_auto_stop=allow_auto_stop)
                    except Exception:
                        pass
                if source_rows:
                    now = datetime.now(timezone.utc)
                    citations: list[Citation] = []
                    for idx, source in enumerate(source_rows[:3], start=1):
                        citations.append(
                            Citation(
                                doc_id=f"web:{idx}",
                                chunk_id=f"web:{idx}:chunk",
                                file_path=str(source.get("url") or ""),
                                snippet=str(source.get("snippet") or "")[:320],
                                score=max(0.2, 0.9 - (idx * 0.15)),
                                modified_at=now,
                                category="web",
                                subcategory="internet",
                                tags=["web"],
                                document_type="web_page",
                                importance=0.6,
                            )
                        )

                    synth_prompt = (
                        "아래 웹 근거를 바탕으로 사용자의 질문에 직접 답하세요.\n"
                        "- 한국어로 자연스럽고 간결하게 작성\n"
                        "- 핵심 주장 2~4개로 정리\n"
                        "- 불확실하면 단정하지 말 것\n"
                        "- 출처 번호를 [1], [2] 형태로 문장 끝에 표기\n\n"
                        f"사용자 질문:\n{context.req.query}\n\n"
                        f"웹 근거:\n{self._web_sources_for_prompt(source_rows)}"
                    ) if context.response_language == "ko" else (
                        "Answer the user question directly using only the web evidence below.\n"
                        "- Be concise and natural\n"
                        "- Keep 2-4 key points\n"
                        "- Do not overclaim uncertain facts\n"
                        "- Cite with [1], [2] markers\n\n"
                        f"User question:\n{context.req.query}\n\n"
                        f"Web evidence:\n{self._web_sources_for_prompt(source_rows)}"
                    )
                    synthesized = await self._run_conversation_inference(
                        executor=executor,
                        query=synth_prompt,
                        context=context,
                        max_tokens=max_tokens,
                    )
                    answer_text = str(synthesized.generated_text or "").strip()
                    if not answer_text:
                        answer_text = self._deterministic_web_summary(
                            query=context.req.query,
                            sources=source_rows,
                            language=context.response_language,
                        )
                    web_summary = answer_text[:500]
                    web_memory_for_metadata = {
                        "web_query": (effective_web_query or context.req.query)[:260],
                        "web_summary": web_summary,
                        "web_sources": source_rows,
                    }
                    execution = ExecutionResult(
                        result_type="conversation",
                        structured_payload={
                            "web_path": "direct",
                            "web_sources_count": len(citations),
                            "ungrounded_allowed": True,
                            "web_summary": web_summary,
                            "web_loop_rounds": int(web_loop_meta.get("web_loop_rounds") or 0),
                            "web_loop_converged": bool(web_loop_meta.get("web_loop_converged")),
                            "web_loop_quality_score": float(web_loop_meta.get("web_loop_quality_score") or 0.0),
                            "web_loop_queries": list(web_loop_meta.get("web_loop_queries") or [])[:3],
                        },
                        citations=citations,
                        tool_logs=[*web_loop_logs, "web_search:direct", "web_answer:synthesized"],
                        generated_text=answer_text,
                        engine_used=None,
                        used_fallback=False,
                        runtime_detail="web_search_direct",
                    )
                    conversation_path = "external_web_search_direct"
                    is_local = False
                else:
                    unavailable_text = (
                        "인터넷에서 신뢰 가능한 근거를 수집하지 못해 답을 확정할 수 없습니다. 잠시 후 다시 시도해 주세요."
                        if context.response_language == "ko"
                        else "I couldn't gather reliable web results right now. Please try again shortly."
                    )
                    execution = ExecutionResult(
                        result_type="conversation",
                        structured_payload={
                            "web_path": "unavailable",
                            "ungrounded_allowed": True,
                            "web_loop_rounds": int(web_loop_meta.get("web_loop_rounds") or 0),
                            "web_loop_converged": bool(web_loop_meta.get("web_loop_converged")),
                            "web_loop_quality_score": float(web_loop_meta.get("web_loop_quality_score") or 0.0),
                            "web_loop_queries": list(web_loop_meta.get("web_loop_queries") or [])[:3],
                        },
                        citations=[],
                        tool_logs=["web_search:unavailable", *web_loop_logs],
                        generated_text=unavailable_text,
                        engine_used=None,
                        used_fallback=False,
                        runtime_detail="web_search_unavailable",
                    )
                    conversation_path = "external_web_search_unavailable"

        if execution is None and not web_search_triggered:
            ranked_entries: list[dict[str, Any]] = []
            if memory_service is not None:
                try:
                    ranked_entries = list(
                        memory_service.get_ranked_web_memory_entries(
                            session_id=context.session_id,
                            query=context.req.query,
                            limit=4,
                        )
                    )
                except Exception:
                    ranked_entries = []

            selected_entry: dict[str, Any] | None = None
            if ranked_entries:
                top = ranked_entries[0]
                top_confidence = float(top.get("confidence") or 0.0)
                top_source_count = int(top.get("source_count") or 0)
                if (
                    top_confidence >= 0.60
                    and top_source_count >= 1
                    and self._is_relevant_web_memory_entry(
                        query=context.req.query,
                        entry=top,
                        last_context=last_context,
                        followup_web_search=followup_web_search,
                    )
                ):
                    selected_entry = top
                    web_memory_rank_score = max(0.0, min(1.0, top_confidence))

            if selected_entry is not None:
                selected_sources = selected_entry.get("sources")
                if not isinstance(selected_sources, list):
                    selected_sources = []
                memory_prompt = (
                    "아래는 같은 채팅방에서 저장된 웹 검색 메모리입니다.\n"
                    "새 검색 없이 이 메모리만으로 질문 중심으로 답하세요.\n"
                    "- 한국어 자연어 요약으로 답할 것\n"
                    "- 핵심 2~4개를 정리할 것\n"
                    "- 메모리에 없는 내용은 추측하지 말 것\n"
                    "- 출처 번호 [1], [2]를 문장 끝에 유지할 것\n\n"
                    f"사용자 질문:\n{context.req.query}\n\n"
                    f"저장된 웹 질의:\n{selected_entry.get('query','')}\n\n"
                    f"저장된 웹 요약:\n{selected_entry.get('answer_summary','')}\n\n"
                    f"저장된 출처:\n{self._web_sources_for_prompt(selected_sources)}"
                ) if context.response_language == "ko" else (
                    "Use only the saved web memory from this same chat session.\n"
                    "Do not run a new search and do not hallucinate.\n"
                    "- Respond with natural language summary\n"
                    "- Keep 2-4 key points\n"
                    "- Keep citation markers like [1], [2]\n\n"
                    f"User question:\n{context.req.query}\n\n"
                    f"Saved web query:\n{selected_entry.get('query','')}\n\n"
                    f"Saved summary:\n{selected_entry.get('answer_summary','')}\n\n"
                    f"Saved sources:\n{self._web_sources_for_prompt(selected_sources)}"
                )
                memory_inference = await self._run_conversation_inference(
                    executor=executor,
                    query=memory_prompt,
                    context=context,
                    max_tokens=max_tokens,
                )
                memory_answer = str(memory_inference.generated_text or "").strip()
                if not memory_answer:
                    memory_answer = str(selected_entry.get("answer_summary") or "").strip()
                if memory_answer:
                    web_memory_for_metadata = {
                        "web_query": str(selected_entry.get("query") or "")[:260],
                        "web_summary": memory_answer[:500],
                        "web_sources": selected_sources[:4],
                    }
                    now = datetime.now(timezone.utc)
                    citations: list[Citation] = []
                    for idx, source in enumerate(selected_sources[:3], start=1):
                        if not isinstance(source, dict):
                            continue
                        url = str(source.get("url") or "").strip()
                        if not url:
                            continue
                        citations.append(
                            Citation(
                                doc_id=f"web-memory:{idx}",
                                chunk_id=f"web-memory:{idx}:chunk",
                                file_path=url,
                                snippet=str(source.get("snippet") or "")[:320],
                                score=max(0.2, 0.9 - (idx * 0.15)),
                                modified_at=now,
                                category="web",
                                subcategory="session_memory",
                                tags=["web", "session-memory"],
                                document_type="web_page",
                                importance=0.55,
                            )
                        )
                    execution = ExecutionResult(
                        result_type="conversation",
                        structured_payload={"web_path": "session_memory", "ungrounded_allowed": True},
                        citations=citations,
                        tool_logs=[
                            f"web_memory:candidates={len(ranked_entries)}",
                            f"web_memory:selected_score={web_memory_rank_score:.3f}",
                            "web_memory:reuse",
                            "web_answer:session_memory",
                        ],
                        generated_text=memory_answer,
                        engine_used=memory_inference.engine_used,
                        used_fallback=memory_inference.used_fallback,
                        runtime_detail=memory_inference.runtime_detail or "web_memory_reused",
                    )
                    conversation_path = "session_web_memory_reused"
                    is_local = True
                    web_memory_reused = True

        if execution is None:
            conversation_query = self._conversation_query_with_context(
                query=context.req.query,
                response_language=context.response_language,
                followup_resolution=context.followup_resolution,
                last_context=context.last_context,
            )
            execution = await self._run_conversation_inference(
                executor=executor,
                query=conversation_query,
                context=context,
                max_tokens=max_tokens,
            )
            execution.tool_logs.insert(0, f"router:intent={ReasoningIntent.GENERAL_CHAT.value}")
            execution.tool_logs.append("agent:conversation_assistant")

        execution.tool_logs.append(f"conversation:max_tokens={max_tokens}")
        detail = str(execution.runtime_detail or "").strip()
        if detail:
            if "max_tokens=" not in detail:
                execution.runtime_detail = f"{detail};max_tokens={max_tokens}"
        else:
            execution.runtime_detail = f"max_tokens={max_tokens}"

        execution_text = self._trim_redundant_opening_from_last_context(
            answer=str(execution.generated_text or ""),
            last_context=context.last_context,
        )
        if execution_text != str(execution.generated_text or ""):
            execution = execution.model_copy(
                update={
                    "generated_text": execution_text,
                    "runtime_detail": (
                        f"{execution.runtime_detail};deduped_opening=True"
                        if execution.runtime_detail
                        else "deduped_opening=True"
                    ),
                }
            )
            execution.tool_logs.append("conversation:deduped_opening")

        if not execution.generated_text:
            normalized_detail = str(execution.runtime_detail or "").strip()
            configured_engine = str(getattr(context.settings, "local_engine", "") or "").strip() or "unknown"
            configured_model_path = ""
            try:
                if str(configured_engine).lower() == "llama_cpp":
                    configured_model_path = str(getattr(context.settings, "llama_model_path", "") or "").strip()
                else:
                    configured_model_path = str(getattr(context.settings, "mlx_model_path", "") or "").strip()
            except Exception:
                configured_model_path = ""
            fallback_text = (
                "로컬 대화 엔진에서 답변 생성을 완료하지 못했습니다. 모델 설치/경로를 확인해 주세요."
                if context.response_language == "ko"
                else "The local conversation engine could not complete generation. Verify model installation/path."
            )
            if normalized_detail:
                fallback_text = (
                    f"{fallback_text}\n\n상세 원인: {normalized_detail}"
                    if context.response_language == "ko"
                    else f"{fallback_text}\n\nDetails: {normalized_detail}"
                )
            else:
                engine_hint = (
                    f"engine={configured_engine}; model_path={configured_model_path or '(empty)'}"
                )
                fallback_text = (
                    f"{fallback_text}\n\n상세 원인: 런타임 상세 오류가 비어 있습니다. {engine_hint}"
                    if context.response_language == "ko"
                    else f"{fallback_text}\n\nDetails: runtime detail is empty. {engine_hint}"
                )
            execution = ExecutionResult(
                result_type="conversation",
                structured_payload={"reason": "conversation_generation_failed"},
                citations=[],
                tool_logs=["runtime_error:conversation_local_failed"],
                generated_text=fallback_text,
                engine_used=execution.engine_used,
                used_fallback=True,
                runtime_detail=execution.runtime_detail,
            )

        assist_citations = execution.citations if (
            conversation_path.startswith("external_web_search")
            or conversation_path == "session_web_memory_reused"
        ) else []

        verification = VerificationResult(
            is_valid=(execution.result_type != "runtime_error"),
            confidence=(0.84 if execution.result_type == "conversation" else 0.2),
            issues=([] if execution.result_type == "conversation" else ["runtime_unavailable"]),
            ambiguity_level=(0.16 if execution.result_type == "conversation" else 0.8),
            candidate_mode=False,
        )

        composed = composer.compose_v2(
            query=context.req.query,
            mode=context.req.mode,
            response_language=context.response_language,
            parsed_intent=context.parsed_intent,
            plan=plan,
            execution_result=execution,
            verification=verification,
            behavior_policy=context.behavior_policy,
            response_length=getattr(context.memory_prefs, "response_length", "long") if context.memory_prefs else "long",
            show_citations=bool(assist_citations),
            prefer_action_suggestions=getattr(context.memory_prefs, "prefer_action_suggestions", True) if context.memory_prefs else True,
            used_profile=context.workspace.startup_profile,
            engine_used=execution.engine_used,
            used_fallback=execution.used_fallback,
            runtime_detail=execution.runtime_detail,
            followup_resolution=context.followup_resolution,
            allow_clarification=None,
            conversation_path=conversation_path,
            is_local=is_local,
            prompt_cache_hit=False,
        )
        web_path = execution.structured_payload.get("web_path")
        if isinstance(web_path, str) and web_path:
            composed.metadata["web_path"] = web_path
            composed.metadata["web_sources_count"] = int(execution.structured_payload.get("web_sources_count") or 0)
            composed.metadata["web_fetch_failures"] = int(execution.structured_payload.get("web_fetch_failures") or 0)
            composed.metadata["web_loop_rounds"] = int(execution.structured_payload.get("web_loop_rounds") or 0)
            composed.metadata["web_loop_converged"] = bool(execution.structured_payload.get("web_loop_converged"))
            composed.metadata["web_loop_quality_score"] = float(
                max(0.0, min(1.0, float(execution.structured_payload.get("web_loop_quality_score") or 0.0)))
            )
            composed.metadata["web_loop_queries"] = list(execution.structured_payload.get("web_loop_queries") or [])[:3]
        if web_memory_for_metadata:
            composed.metadata["web_query"] = str(web_memory_for_metadata.get("web_query") or "")[:260]
            composed.metadata["web_summary"] = str(web_memory_for_metadata.get("web_summary") or "")[:500]
            raw_sources = web_memory_for_metadata.get("web_sources")
            if isinstance(raw_sources, list):
                composed.metadata["web_sources"] = raw_sources[:4]
        composed.metadata["web_memory_reused"] = bool(web_memory_reused)
        composed.metadata["web_memory_rank_score"] = (
            float(max(0.0, min(1.0, web_memory_rank_score)))
            if web_memory_reused
            else 0.0
        )
        return composed
