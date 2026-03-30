from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import os
import re
import time
from urllib.parse import parse_qs, unquote, urlparse, urlsplit, urlunsplit

try:
    import httpx
except ImportError:
    httpx = None


@dataclass(slots=True)
class WebSearchSource:
    title: str
    url: str
    snippet: str
    content: str


@dataclass(slots=True)
class WebRetrievalReport:
    query: str
    sources: list[WebSearchSource]
    logs: list[str]
    discovered_count: int
    fetch_success_count: int
    fetch_failure_count: int
    unique_domain_count: int = 0
    usable_source_count: int = 0
    quality_score: float = 0.0
    round_query: str = ""
    cache_hit: bool = False
    failure_reason: str = ""


@dataclass(slots=True)
class _DiscoveredURL:
    title: str
    url: str
    snippet: str
    source: str


@dataclass(slots=True)
class _FetchedPage:
    title: str
    url: str
    snippet: str
    content: str


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        text = " ".join(str(data or "").split()).strip()
        if not text:
            return
        self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks).strip()


class _DuckDuckGoHTMLResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_result_link = False
        self._current_href = ""
        self._current_title_chunks: list[str] = []
        self._in_snippet = False
        self._current_snippet_chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        attr_map = {k: v for k, v in attrs}
        css_class = str(attr_map.get("class") or "")
        if tag == "a" and "result__a" in css_class:
            self._in_result_link = True
            self._current_href = str(attr_map.get("href") or "")
            self._current_title_chunks = []
            return
        if "result__snippet" in css_class:
            self._in_snippet = True
            self._current_snippet_chunks = []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_result_link:
            title = " ".join(" ".join(self._current_title_chunks).split()).strip()
            href = " ".join(self._current_href.split()).strip()
            if href:
                self.results.append(
                    {
                        "title": title,
                        "url": href,
                        "snippet": "",
                    }
                )
            self._in_result_link = False
            self._current_href = ""
            self._current_title_chunks = []
            return
        if self._in_snippet and tag in {"div", "span", "a"}:
            snippet = " ".join(" ".join(self._current_snippet_chunks).split()).strip()
            if snippet and self.results and not self.results[-1].get("snippet"):
                self.results[-1]["snippet"] = snippet
            self._in_snippet = False
            self._current_snippet_chunks = []

    def handle_data(self, data):
        text = str(data or "")
        if self._in_result_link and text:
            self._current_title_chunks.append(text)
        if self._in_snippet and text:
            self._current_snippet_chunks.append(text)


class _SearxngHTMLResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._inside_result_article = False
        self._article_depth = 0
        self._inside_h3_link = False
        self._inside_snippet = False
        self._current_href = ""
        self._title_chunks: list[str] = []
        self._snippet_chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        attr_map = {k: v for k, v in attrs}
        css_class = str(attr_map.get("class") or "")

        if tag == "article" and "result" in css_class:
            self._inside_result_article = True
            self._article_depth = 1
            self._current_href = ""
            self._title_chunks = []
            self._snippet_chunks = []
            return

        if self._inside_result_article and tag == "article":
            self._article_depth += 1

        if not self._inside_result_article:
            return

        if tag == "h3":
            return
        if tag == "a" and not self._inside_h3_link and str(attr_map.get("href") or "").strip().startswith("http"):
            self._inside_h3_link = True
            self._current_href = str(attr_map.get("href") or "").strip()
            return
        if tag == "p" and "content" in css_class:
            self._inside_snippet = True
            return

    def handle_endtag(self, tag):
        if self._inside_result_article and tag == "article":
            self._article_depth -= 1
            if self._article_depth <= 0:
                title = " ".join(" ".join(self._title_chunks).split()).strip()
                snippet = " ".join(" ".join(self._snippet_chunks).split()).strip()
                href = str(self._current_href or "").strip()
                if href:
                    self.results.append(
                        {
                            "title": title,
                            "url": href,
                            "snippet": snippet,
                        }
                    )
                self._inside_result_article = False
                self._inside_h3_link = False
                self._inside_snippet = False
                self._current_href = ""
                self._title_chunks = []
                self._snippet_chunks = []
                self._article_depth = 0
            return

        if not self._inside_result_article:
            return

        if tag == "a" and self._inside_h3_link:
            self._inside_h3_link = False
            return
        if tag == "p" and self._inside_snippet:
            self._inside_snippet = False
            return

    def handle_data(self, data):
        if not self._inside_result_article:
            return
        text = str(data or "")
        if not text:
            return
        if self._inside_h3_link:
            self._title_chunks.append(text)
        if self._inside_snippet:
            self._snippet_chunks.append(text)


class WebRetriever:
    _DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
    _DDG_INSTANT_ENDPOINT = "https://api.duckduckgo.com/"
    _UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    _ACCEPT_LANGUAGE = "ko,en-US;q=0.9,en;q=0.8"
    _ALLOWED_TEXT_TYPES = (
        "text/",
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    )

    def __init__(self):
        if httpx is not None:
            self._timeout = httpx.Timeout(8.0, connect=4.0)
        else:
            self._timeout = None
        self._cache_ttl_seconds = 300
        self._cache: dict[str, tuple[float, WebRetrievalReport]] = {}

    def run(
        self,
        *,
        query: str,
        max_candidates: int = 8,
        max_sources: int = 3,
        searxng_url: str | None = None,
        prefer_searxng: bool = True,
        freshness_sensitive: bool = False,
    ) -> WebRetrievalReport:
        key = self._cache_key(
            query,
            searxng_url=searxng_url,
            prefer_searxng=prefer_searxng,
            freshness_sensitive=freshness_sensitive,
        )
        now_ts = time.time()
        cached = self._cache.get(key)
        if cached is not None and now_ts - cached[0] <= self._cache_ttl_seconds:
            return self._clone_report(cached[1], cache_hit=True)

        logs: list[str] = ["planning:web_search_requested"]
        direct_urls = self._extract_direct_urls(query)
        if direct_urls:
            discovered = [
                _DiscoveredURL(title=self._domain_title(url), url=url, snippet="", source="direct_url")
                for url in direct_urls[:max_candidates]
            ]
            logs.append("web_discovery:direct_url")
        else:
            discovered = self.discover_urls(
                query=query,
                limit=max_candidates,
                logs=logs,
                searxng_url=searxng_url,
                prefer_searxng=prefer_searxng,
            )
            logs.append(f"web_discovery:count={len(discovered)}")

        if not discovered:
            report = WebRetrievalReport(
                query=query,
                sources=[],
                logs=logs,
                discovered_count=0,
                fetch_success_count=0,
                fetch_failure_count=0,
                unique_domain_count=0,
                usable_source_count=0,
                quality_score=0.0,
                round_query=query,
                cache_hit=False,
                failure_reason="no_discovery",
            )
            return report

        pages, fetch_failures = self.fetch_pages(
            discovered=discovered,
            max_sources=max_sources,
            logs=logs,
        )
        sources = self.build_evidence(pages=pages, max_sources=max_sources, logs=logs)
        usable_source_count = self._usable_source_count(sources)
        unique_domain_count = self._unique_domain_count(sources)
        quality_score = self._quality_score(
            usable_source_count=usable_source_count,
            unique_domain_count=unique_domain_count,
            fetch_success_count=len(pages),
            fetch_failure_count=fetch_failures,
            freshness_sensitive=freshness_sensitive,
        )

        report = WebRetrievalReport(
            query=query,
            sources=sources,
            logs=logs,
            discovered_count=len(discovered),
            fetch_success_count=len(pages),
            fetch_failure_count=fetch_failures,
            unique_domain_count=unique_domain_count,
            usable_source_count=usable_source_count,
            quality_score=quality_score,
            round_query=query,
            cache_hit=False,
            failure_reason=("" if sources else "no_fetch_content"),
        )
        if sources:
            self._cache[key] = (now_ts, self._clone_report(report))
        return report

    def search_and_fetch(
        self,
        *,
        query: str,
        max_sources: int = 3,
        searxng_url: str | None = None,
        prefer_searxng: bool = True,
    ) -> tuple[list[WebSearchSource], list[str]]:
        report = self.run(
            query=query,
            max_candidates=max(4, max_sources * 2),
            max_sources=max_sources,
            searxng_url=searxng_url,
            prefer_searxng=prefer_searxng,
        )
        return report.sources, report.logs

    def discover_urls(
        self,
        *,
        query: str,
        limit: int,
        logs: list[str],
        searxng_url: str | None = None,
        prefer_searxng: bool = True,
    ) -> list[_DiscoveredURL]:
        effective_searxng_url = str(searxng_url or os.getenv("LOCAL_AI_SEARXNG_URL", "")).strip()
        discovered = []
        if prefer_searxng and effective_searxng_url:
            discovered = self._discover_from_searxng(
                query=query,
                limit=limit,
                logs=logs,
                base_url=effective_searxng_url,
            )
        
        if not discovered:
            discovered = self._discover_from_ddg_html(query=query, limit=limit, logs=logs)
        if not discovered:
            discovered = self._discover_from_ddg_instant(query=query, limit=limit, logs=logs)
        if not discovered and (not prefer_searxng) and effective_searxng_url:
            discovered = self._discover_from_searxng(
                query=query,
                limit=limit,
                logs=logs,
                base_url=effective_searxng_url,
            )
        return self._dedupe_discovered(discovered, limit=limit)

    def _discover_from_searxng(self, *, query: str, limit: int, logs: list[str], base_url: str) -> list[_DiscoveredURL]:
        if httpx is None:
            return []
        
        # Ensure base_url ends with /search or similar if not provided
        url = base_url.rstrip("/")
        if not url.endswith("/search"):
            url += "/search"
            
        params = {"q": query, "format": "json"}
        logs.append(f"retrieving:searxng:{url}")
        headers = {
            "User-Agent": self._UA,
            "Accept-Language": self._ACCEPT_LANGUAGE,
            "Accept": "application/json,text/plain,*/*",
        }
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = int(getattr(exc.response, "status_code", 0) or 0)
            if status_code == 403:
                logs.append(f"notice:searxng_json_forbidden:{url}")
            else:
                logs.append(f"warning:searxng_json_failed:{url}:{exc}")
            return self._discover_from_searxng_html(query=query, limit=limit, logs=logs, url=url)
        except Exception as exc:
            logs.append(f"warning:searxng_json_failed:{url}:{exc}")
            return self._discover_from_searxng_html(query=query, limit=limit, logs=logs, url=url)

        results = payload.get("results")
        if not isinstance(results, list):
            return self._discover_from_searxng_html(query=query, limit=limit, logs=logs, url=url)

        output: list[_DiscoveredURL] = []
        for row in results:
            target_url = str(row.get("url") or "").strip()
            if not self._is_http_url(target_url):
                continue
            output.append(
                _DiscoveredURL(
                    title=str(row.get("title") or self._domain_title(target_url))[:120],
                    url=target_url,
                    snippet=str(row.get("content") or row.get("snippet") or "")[:320],
                    source="searxng",
                )
            )
            if len(output) >= limit:
                break
        return output

    def _discover_from_searxng_html(self, *, query: str, limit: int, logs: list[str], url: str) -> list[_DiscoveredURL]:
        if httpx is None:
            return []
        logs.append(f"retrieving:searxng_html:{url}")
        headers = {
            "User-Agent": self._UA,
            "Accept-Language": self._ACCEPT_LANGUAGE,
        }
        try:
            with httpx.Client(timeout=self._timeout, headers=headers, follow_redirects=True) as client:
                response = client.get(url, params={"q": query})
                response.raise_for_status()
                html = response.text
        except Exception as exc:
            logs.append(f"warning:searxng_html_failed:{url}:{exc}")
            return []

        normalized_html = str(html or "").casefold()
        if (
            "class=\"result" not in normalized_html
            and "class='result" not in normalized_html
            and "id=\"urls\"" not in normalized_html
            and "id='urls'" not in normalized_html
        ):
            logs.append("notice:searxng_html_non_result_page")
            return []

        parser = _SearxngHTMLResultParser()
        parser.feed(html)
        parser.close()

        output: list[_DiscoveredURL] = []
        for row in parser.results:
            target_url = str(row.get("url") or "").strip()
            if not self._is_http_url(target_url):
                continue
            output.append(
                _DiscoveredURL(
                    title=str(row.get("title") or self._domain_title(target_url))[:120],
                    url=target_url,
                    snippet=str(row.get("snippet") or "")[:320],
                    source="searxng",
                )
            )
            if len(output) >= limit:
                break
        return output

    def fetch_pages(
        self,
        *,
        discovered: list[_DiscoveredURL],
        max_sources: int,
        logs: list[str],
    ) -> tuple[list[_FetchedPage], int]:
        pages: list[_FetchedPage] = []
        failures = 0
        for item in discovered:
            if len(pages) >= max_sources:
                break
            fetched = self._fetch_page(url=item.url, logs=logs)
            if fetched is None:
                failures += 1
                continue
            title = item.title or self._domain_title(item.url)
            pages.append(
                _FetchedPage(
                    title=title[:120],
                    url=item.url,
                    snippet=(item.snippet or "")[:320],
                    content=fetched[:3000],
                )
            )
        return pages, failures

    def build_evidence(self, *, pages: list[_FetchedPage], max_sources: int, logs: list[str]) -> list[WebSearchSource]:
        output: list[WebSearchSource] = []
        for page in pages[:max_sources]:
            snippet = page.snippet.strip() or page.content[:320]
            content = page.content.strip() or snippet
            if not content:
                continue
            output.append(
                WebSearchSource(
                    title=page.title,
                    url=page.url,
                    snippet=snippet,
                    content=content,
                )
            )
        logs.append(f"done:web_evidence_composed:{len(output)}")
        return output

    def _discover_from_ddg_html(self, *, query: str, limit: int, logs: list[str]) -> list[_DiscoveredURL]:
        if httpx is None:
            logs.append("error:httpx_not_installed")
            return []
            
        params = {"q": query, "ia": "web"}
        logs.append(f"retrieving:{self._DDG_HTML_ENDPOINT}")
        headers = {
            "User-Agent": self._UA,
            "Accept-Language": self._ACCEPT_LANGUAGE,
        }
        try:
            with httpx.Client(timeout=self._timeout, headers=headers, follow_redirects=True) as client:
                response = client.get(self._DDG_HTML_ENDPOINT, params=params)
                response.raise_for_status()
                html = response.text
        except Exception:
            logs.append(f"warning:search_failed:{self._DDG_HTML_ENDPOINT}")
            return []
        logs.append(f"retrieved:{self._DDG_HTML_ENDPOINT}")

        parser = _DuckDuckGoHTMLResultParser()
        parser.feed(html)
        parser.close()

        output: list[_DiscoveredURL] = []
        for row in parser.results:
            raw_url = self._resolve_duckduckgo_redirect(row.get("url", ""))
            if not self._is_http_url(raw_url):
                continue
            output.append(
                _DiscoveredURL(
                    title=str(row.get("title") or self._domain_title(raw_url))[:120],
                    url=raw_url,
                    snippet=str(row.get("snippet") or "")[:320],
                    source="ddg_html",
                )
            )
            if len(output) >= limit:
                break
        return output

    def _discover_from_ddg_instant(self, *, query: str, limit: int, logs: list[str]) -> list[_DiscoveredURL]:
        if httpx is None:
            logs.append("error:httpx_not_installed")
            return []
            
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        logs.append(f"retrieving:{self._DDG_INSTANT_ENDPOINT}")
        headers = {
            "User-Agent": self._UA,
            "Accept-Language": self._ACCEPT_LANGUAGE,
        }
        try:
            with httpx.Client(timeout=self._timeout, headers=headers, follow_redirects=True) as client:
                response = client.get(self._DDG_INSTANT_ENDPOINT, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            logs.append(f"warning:search_failed:{self._DDG_INSTANT_ENDPOINT}")
            return []
        logs.append(f"retrieved:{self._DDG_INSTANT_ENDPOINT}")

        output: list[_DiscoveredURL] = []
        abstract_url = str(payload.get("AbstractURL") or "").strip()
        abstract_text = str(payload.get("AbstractText") or "").strip()
        heading = str(payload.get("Heading") or "").strip()
        if self._is_http_url(abstract_url):
            output.append(
                _DiscoveredURL(
                    title=(heading or self._domain_title(abstract_url))[:120],
                    url=abstract_url,
                    snippet=abstract_text[:320],
                    source="ddg_instant",
                )
            )

        def walk_topics(raw_items):
            if not isinstance(raw_items, list):
                return
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("Topics"), list):
                    walk_topics(item.get("Topics"))
                    continue
                url = str(item.get("FirstURL") or "").strip()
                text = str(item.get("Text") or "").strip()
                if not self._is_http_url(url):
                    continue
                title = self._title_from_text(text) or self._domain_title(url)
                output.append(
                    _DiscoveredURL(
                        title=title[:120],
                        url=url,
                        snippet=text[:320],
                        source="ddg_instant",
                    )
                )
                if len(output) >= limit:
                    return

        walk_topics(payload.get("RelatedTopics"))
        return output[:limit]

    def _fetch_page(self, *, url: str, logs: list[str]) -> str | None:
        if httpx is None:
            logs.append("error:httpx_not_installed")
            return None
            
        logs.append(f"retrieving:{url}")
        headers = {
            "User-Agent": self._UA,
            "Accept-Language": self._ACCEPT_LANGUAGE,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Cache-Control": "no-cache",
        }
        try:
            with httpx.Client(timeout=self._timeout, headers=headers, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                content_type = str(response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
                if content_type and not self._is_text_content_type(content_type):
                    logs.append(f"warning:fetch_skipped_content_type:{url}")
                    return None
                body = response.text
        except Exception as exc:
            fallback_text = self._fetch_page_via_reader_proxy(url=url, logs=logs)
            if fallback_text:
                return fallback_text
            logs.append(f"warning:fetch_failed:{url}:{exc.__class__.__name__}")
            return None

        if self._looks_binary_blob(body):
            logs.append(f"warning:fetch_skipped_binary:{url}")
            return None

        if "<html" in body[:400].lower() or "text/html" in content_type or not content_type:
            parser = _HTMLTextExtractor()
            parser.feed(body)
            parser.close()
            text = parser.text()
        else:
            text = body

        compact = " ".join((text or "").split()).strip()
        if not compact:
            logs.append(f"warning:fetch_empty:{url}")
            return None

        logs.append(f"retrieved:{url}")
        return compact

    def _fetch_page_via_reader_proxy(self, *, url: str, logs: list[str]) -> str | None:
        if httpx is None:
            return None
        reader_url = self._reader_proxy_url(url)
        if not reader_url:
            return None
        logs.append(f"retrieving:{reader_url}")
        headers = {
            "User-Agent": self._UA,
            "Accept-Language": self._ACCEPT_LANGUAGE,
        }
        try:
            with httpx.Client(timeout=self._timeout, headers=headers, follow_redirects=True) as client:
                response = client.get(reader_url)
                response.raise_for_status()
                body = str(response.text or "")
        except Exception:
            logs.append(f"warning:reader_proxy_failed:{url}")
            return None

        compact = " ".join(unescape(body).split()).strip()
        if not compact:
            logs.append(f"warning:reader_proxy_empty:{url}")
            return None
        logs.append(f"retrieved:{reader_url}")
        return compact[:3000]

    @staticmethod
    def _reader_proxy_url(url: str) -> str:
        parsed = urlsplit(str(url or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        path = parsed.path or "/"
        query = f"?{parsed.query}" if parsed.query else ""
        return f"https://r.jina.ai/http://{parsed.netloc}{path}{query}"

    @staticmethod
    def _cache_key(
        query: str,
        *,
        searxng_url: str | None = None,
        prefer_searxng: bool = True,
        freshness_sensitive: bool = False,
    ) -> str:
        base = " ".join(str(query or "").strip().casefold().split())
        provider = "searxng" if prefer_searxng else "crawler"
        endpoint = " ".join(str(searxng_url or "").strip().casefold().split())
        freshness = "1" if freshness_sensitive else "0"
        return f"{base}|provider:{provider}|searxng:{endpoint}|fresh:{freshness}"

    @staticmethod
    def _clone_report(report: WebRetrievalReport, *, cache_hit: bool | None = None) -> WebRetrievalReport:
        return WebRetrievalReport(
            query=report.query,
            sources=[
                WebSearchSource(title=item.title, url=item.url, snippet=item.snippet, content=item.content)
                for item in report.sources
            ],
            logs=list(report.logs),
            discovered_count=int(report.discovered_count),
            fetch_success_count=int(report.fetch_success_count),
            fetch_failure_count=int(report.fetch_failure_count),
            unique_domain_count=int(report.unique_domain_count),
            usable_source_count=int(report.usable_source_count),
            quality_score=float(report.quality_score or 0.0),
            round_query=str(report.round_query or ""),
            cache_hit=(report.cache_hit if cache_hit is None else bool(cache_hit)),
            failure_reason=str(report.failure_reason or ""),
        )

    @staticmethod
    def _usable_source_count(sources: list[WebSearchSource]) -> int:
        count = 0
        for item in sources:
            url = str(getattr(item, "url", "") or "").strip()
            snippet = str(getattr(item, "snippet", "") or "").strip()
            content = str(getattr(item, "content", "") or "").strip()
            if url and (snippet or content):
                count += 1
        return count

    @classmethod
    def _unique_domain_count(cls, sources: list[WebSearchSource]) -> int:
        domains: set[str] = set()
        for item in sources:
            url = str(getattr(item, "url", "") or "").strip()
            canonical = cls._canonicalize_url(url)
            if not canonical:
                continue
            parsed = urlparse(canonical)
            host = str(parsed.netloc or "").strip().lower()
            if host:
                domains.add(host)
        return len(domains)

    @staticmethod
    def _quality_score(
        *,
        usable_source_count: int,
        unique_domain_count: int,
        fetch_success_count: int,
        fetch_failure_count: int,
        freshness_sensitive: bool,
    ) -> float:
        usable_ratio = min(max(int(usable_source_count), 0) / 3.0, 1.0)
        domain_ratio = min(max(int(unique_domain_count), 0) / 3.0, 1.0)
        total_fetch = max(int(fetch_success_count) + int(fetch_failure_count), 1)
        failure_ratio = min(max(int(fetch_failure_count), 0) / float(total_fetch), 1.0)
        freshness_bonus = 1.0 if freshness_sensitive and int(usable_source_count) >= 3 else 0.0
        score = (
            (0.40 * usable_ratio)
            + (0.30 * domain_ratio)
            + (0.20 * (1.0 - failure_ratio))
            + (0.10 * freshness_bonus)
        )
        return max(0.0, min(1.0, float(score)))

    @classmethod
    def _dedupe_discovered(cls, items: list[_DiscoveredURL], *, limit: int) -> list[_DiscoveredURL]:
        output: list[_DiscoveredURL] = []
        seen: set[str] = set()
        for item in items:
            canonical = cls._canonicalize_url(item.url)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            output.append(
                _DiscoveredURL(
                    title=item.title,
                    url=canonical,
                    snippet=item.snippet,
                    source=item.source,
                )
            )
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _extract_direct_urls(query: str) -> list[str]:
        text = str(query or "")
        if not text:
            return []
        candidates = re.findall(r"https?://[^\s<>)\]\}]+", text, flags=re.IGNORECASE)
        output: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.rstrip(".,;:!?)]}")
            if not WebRetriever._is_http_url(normalized):
                continue
            canonical = WebRetriever._canonicalize_url(normalized)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            output.append(canonical)
        return output

    @staticmethod
    def _resolve_duckduckgo_redirect(url: str) -> str:
        raw = unescape(str(url or "").strip())
        if not raw:
            return ""
        if raw.startswith("//"):
            raw = f"https:{raw}"
        parsed = urlparse(raw)
        if "duckduckgo.com" not in parsed.netloc.lower():
            return WebRetriever._canonicalize_url(raw)
        if parsed.path.startswith("/l/"):
            qs = parse_qs(parsed.query)
            target = (qs.get("uddg") or [""])[0]
            target = unquote(target)
            if target:
                return WebRetriever._canonicalize_url(target)
        return WebRetriever._canonicalize_url(raw)

    @staticmethod
    def _is_http_url(url: str) -> bool:
        if not url:
            return False
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        raw = str(url or "").strip()
        if not raw:
            return ""
        parsed = urlsplit(raw)
        scheme = (parsed.scheme or "").lower()
        netloc = (parsed.netloc or "").lower()
        if scheme not in {"http", "https"} or not netloc:
            return ""
        path = parsed.path or "/"
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return urlunsplit((scheme, netloc, path, parsed.query, ""))

    @classmethod
    def _is_text_content_type(cls, content_type: str) -> bool:
        value = str(content_type or "").strip().lower()
        if not value:
            return True
        return any(value.startswith(prefix) for prefix in cls._ALLOWED_TEXT_TYPES)

    @staticmethod
    def _looks_binary_blob(text: str) -> bool:
        value = str(text or "")
        if not value:
            return False
        sample = value[:1024]
        if "\x00" in sample:
            return True
        non_printables = 0
        for ch in sample:
            code = ord(ch)
            if code in {9, 10, 13}:
                continue
            if 32 <= code <= 126:
                continue
            if 0xAC00 <= code <= 0xD7A3:
                continue
            if 0x1100 <= code <= 0x11FF:
                continue
            if 0x3130 <= code <= 0x318F:
                continue
            if 0x4E00 <= code <= 0x9FFF:
                continue
            non_printables += 1
        return non_printables > max(8, len(sample) // 8)

    @staticmethod
    def _title_from_text(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        if " - " in value:
            return value.split(" - ", 1)[0].strip()
        if "|" in value:
            return value.split("|", 1)[0].strip()
        return value[:80]

    @staticmethod
    def _domain_title(url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc or "web"
