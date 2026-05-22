from __future__ import annotations

import asyncio
import socket
import time
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from ....web_retrieval import WebRetrievalReport, WebRetriever


class GeneralChatWebExecutionHelpers:
    @staticmethod
    def searxng_connection_refused(logs: list[str]) -> bool:
        for item in logs or []:
            low = str(item or "").lower()
            if "searxng" not in low:
                continue
            if "connection refused" in low or "errno 61" in low:
                return True
        return False

    @staticmethod
    def wait_for_port(host: str, port: int, *, timeout_seconds: float = 6.0) -> bool:
        deadline = time.time() + max(0.5, timeout_seconds)
        while time.time() < deadline:
            try:
                with socket.create_connection((host, int(port)), timeout=0.8):
                    return True
            except Exception:
                time.sleep(0.25)
        return False

    @staticmethod
    def wait_for_searxng_http(base_url: str, *, timeout_seconds: float = 8.0) -> bool:
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
    async def wait_for_port_async(host: str, port: int, *, timeout_seconds: float = 6.0) -> bool:
        return await asyncio.to_thread(
            GeneralChatWebExecutionHelpers.wait_for_port,
            host,
            port,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    async def wait_for_searxng_http_async(base_url: str, *, timeout_seconds: float = 8.0) -> bool:
        return await asyncio.to_thread(
            GeneralChatWebExecutionHelpers.wait_for_searxng_http,
            base_url,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    async def docker_is_running_async(docker_service: Any) -> bool:
        return bool(await asyncio.to_thread(docker_service.is_running))

    @staticmethod
    async def docker_start_async(docker_service: Any, *, keep_running: bool) -> bool:
        return bool(await asyncio.to_thread(docker_service.start, keep_running=keep_running))

    @staticmethod
    async def ensure_local_searxng_ready_async(
        strategy,
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
        logs: list[str] = []
        if docker_service is None:
            return False, ["web_search:docker_service_missing"]
        docker_service.mark_usage(allow_auto_stop=allow_auto_stop)
        try:
            is_running = await strategy._docker_is_running_async(docker_service)
            logs.append(f"web_search:docker_running={int(is_running)}")
            if not is_running:
                started = await strategy._docker_start_async(docker_service, keep_running=keep_running)
                logs.append(f"web_search:docker_start={int(started)}")
                if not started:
                    return False, logs
            port_ready = await strategy._wait_for_port_async(host, port, timeout_seconds=port_timeout_seconds)
            http_ready = await strategy._wait_for_searxng_http_async(searxng_url, timeout_seconds=http_timeout_seconds)
            if not (port_ready and http_ready):
                logs.append("web_search:searxng_probe_failed")
                return False, logs
            return True, logs
        except Exception as exc:
            logs.append(f"web_search:searxng_ready_exception={str(exc)[:120]}")
            return False, logs

    @staticmethod
    def query_variant_for_round(
        strategy,
        *,
        original_query: str,
        round_index: int,
        round1_sources: list[dict[str, str]],
    ) -> str:
        base = " ".join(str(original_query or "").split()).strip()
        if round_index <= 1 or not base:
            return base
        if round_index == 2:
            entities = strategy._tokenize_keywords(base, max_tokens=2)
            suffix = "latest update"
            if strategy._contains_korean(base):
                suffix = "latest update 최신 업데이트"
            return " ".join([base, *entities, suffix]).strip()
        title_blob = " ".join(str(item.get("title") or "") for item in round1_sources[:3] if isinstance(item, dict))
        title_terms = strategy._tokenize_keywords(title_blob, max_tokens=2)
        return " ".join([base, *title_terms, "official source"]).strip()

    @staticmethod
    def source_rows_from_report(report: WebRetrievalReport) -> list[dict[str, str]]:
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
    def is_uncertain_web_round(
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

    @staticmethod
    def run_web_reasoning_loop(
        strategy,
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
            round_query = strategy._query_variant_for_round(
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
            round_sources = strategy._source_rows_from_report(report)
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

            uncertain = strategy._is_uncertain_web_round(
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

