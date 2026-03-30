from local_ai_core.web_retrieval import (
    WebRetriever,
    WebSearchSource,
    _DiscoveredURL,
    _FetchedPage,
    _SearxngHTMLResultParser,
)
import httpx
import local_ai_core.web_retrieval as web_retrieval_module


def test_extract_direct_url_skips_search(monkeypatch):
    retriever = WebRetriever()

    called = {"html": 0, "instant": 0}

    def _fail_html(**kwargs):
        called["html"] += 1
        return []

    def _fail_instant(**kwargs):
        called["instant"] += 1
        return []

    monkeypatch.setattr(retriever, "_discover_from_ddg_html", _fail_html)
    monkeypatch.setattr(retriever, "_discover_from_ddg_instant", _fail_instant)

    report = retriever.run(query="https://example.com 문서 요약해줘", max_candidates=8, max_sources=3)

    assert report.discovered_count >= 1
    assert called["html"] == 0
    assert called["instant"] == 0
    assert any(log == "web_discovery:direct_url" for log in report.logs)


def test_discover_urls_falls_back_to_instant(monkeypatch):
    retriever = WebRetriever()
    logs: list[str] = []

    def _empty_html(**kwargs):
        return []

    def _instant(**kwargs):
        return [
            _DiscoveredURL(
                title="Swift",
                url="https://docs.swift.org/swift-book/",
                snippet="Swift docs",
                source="ddg_instant",
            )
        ]

    monkeypatch.setattr(retriever, "_discover_from_ddg_html", _empty_html)
    monkeypatch.setattr(retriever, "_discover_from_ddg_instant", _instant)

    discovered = retriever.discover_urls(query="Swift 공식 문서", limit=8, logs=logs)
    assert len(discovered) == 1
    assert discovered[0].url == "https://docs.swift.org/swift-book"


def test_discover_urls_prefers_searxng_when_url_is_configured(monkeypatch):
    retriever = WebRetriever()
    logs: list[str] = []
    called = {"searx": 0, "html": 0, "instant": 0}

    def _searx(**kwargs):
        called["searx"] += 1
        return [
            _DiscoveredURL(
                title="SearXNG Hit",
                url="https://example.com/searx",
                snippet="from searxng",
                source="searxng",
            )
        ]

    def _html(**kwargs):
        called["html"] += 1
        return []

    def _instant(**kwargs):
        called["instant"] += 1
        return []

    monkeypatch.setattr(retriever, "_discover_from_searxng", _searx)
    monkeypatch.setattr(retriever, "_discover_from_ddg_html", _html)
    monkeypatch.setattr(retriever, "_discover_from_ddg_instant", _instant)

    discovered = retriever.discover_urls(
        query="아이폰 최신 비교",
        limit=8,
        logs=logs,
        searxng_url="http://127.0.0.1:8080/search",
        prefer_searxng=True,
    )
    assert len(discovered) == 1
    assert discovered[0].source == "searxng"
    assert called["searx"] == 1
    assert called["html"] == 0
    assert called["instant"] == 0


def test_run_uses_cache(monkeypatch):
    retriever = WebRetriever()
    calls = {"discover": 0, "fetch": 0, "build": 0}

    def _discover(*, query, limit, logs, searxng_url=None, prefer_searxng=True):
        calls["discover"] += 1
        return [
            _DiscoveredURL(
                title="Example",
                url="https://example.com",
                snippet="example",
                source="ddg_html",
            )
        ]

    def _fetch(*, discovered, max_sources, logs):
        calls["fetch"] += 1
        return (
            [
                _FetchedPage(
                    title="Example",
                    url="https://example.com",
                    snippet="example",
                    content="example content",
                )
            ],
            0,
        )

    def _build(*, pages, max_sources, logs):
        calls["build"] += 1
        return [
            WebSearchSource(
                title="Example",
                url="https://example.com",
                snippet="example",
                content="example content",
            )
        ]

    monkeypatch.setattr(retriever, "discover_urls", _discover)
    monkeypatch.setattr(retriever, "fetch_pages", _fetch)
    monkeypatch.setattr(retriever, "build_evidence", _build)

    first = retriever.run(query="example", max_candidates=8, max_sources=3)
    second = retriever.run(query="example", max_candidates=8, max_sources=3)

    assert len(first.sources) == 1
    assert len(second.sources) == 1
    assert second.cache_hit is True
    assert calls == {"discover": 1, "fetch": 1, "build": 1}


def test_searxng_html_parser_extracts_results():
    html = """
    <div id="urls">
      <article class="result result-default category-general">
        <h3><a href="https://www.apple.com/iphone/" rel="noreferrer">iPhone - Apple</a></h3>
        <p class="content">Apple iPhone 공식 페이지</p>
      </article>
      <article class="result result-default category-general">
        <h3><a href="https://en.wikipedia.org/wiki/IPhone">iPhone - Wikipedia</a></h3>
        <p class="content">Wikipedia entry</p>
      </article>
    </div>
    """
    parser = _SearxngHTMLResultParser()
    parser.feed(html)
    parser.close()
    assert len(parser.results) == 2
    assert parser.results[0]["url"] == "https://www.apple.com/iphone/"
    assert "Apple" in parser.results[0]["title"]


def test_build_evidence_caps_count_and_logs():
    retriever = WebRetriever()
    logs: list[str] = []
    pages = [
        _FetchedPage(title=f"t{idx}", url=f"https://e{idx}.com", snippet="", content=f"content {idx}")
        for idx in range(5)
    ]
    output = retriever.build_evidence(pages=pages, max_sources=3, logs=logs)

    assert len(output) == 3
    assert logs[-1] == "done:web_evidence_composed:3"


def test_run_does_not_cache_failure_reports(monkeypatch):
    retriever = WebRetriever()
    calls = {"discover": 0}

    def _discover(*, query, limit, logs, searxng_url=None, prefer_searxng=True):
        calls["discover"] += 1
        return []

    monkeypatch.setattr(retriever, "discover_urls", _discover)

    first = retriever.run(query="unreachable", max_candidates=4, max_sources=2)
    second = retriever.run(query="unreachable", max_candidates=4, max_sources=2)

    assert first.sources == []
    assert second.sources == []
    assert first.cache_hit is False
    assert second.cache_hit is False
    assert calls["discover"] == 2


def test_searxng_json_403_uses_notice_and_falls_back_html(monkeypatch):
    retriever = WebRetriever()
    logs: list[str] = []

    class _ForbiddenResponse:
        def __init__(self, url: str):
            self.status_code = 403
            self.request = httpx.Request("GET", url)

        def raise_for_status(self):
            raise httpx.HTTPStatusError("403 Forbidden", request=self.request, response=self)

        def json(self):
            return {}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, headers=None):
            return _ForbiddenResponse(url)

    monkeypatch.setattr(web_retrieval_module.httpx, "Client", _Client)
    monkeypatch.setattr(retriever, "_discover_from_searxng_html", lambda **kwargs: [])

    discovered = retriever._discover_from_searxng(
        query="아이폰 최신 정보",
        limit=5,
        logs=logs,
        base_url="http://127.0.0.1:8080/search",
    )
    assert discovered == []
    assert any(log.startswith("notice:searxng_json_forbidden:") for log in logs)


def test_searxng_html_skips_non_result_page(monkeypatch):
    retriever = WebRetriever()
    logs: list[str] = []

    class _Response:
        status_code = 200
        text = "<html><body><h1>Forbidden</h1><a href='https://example.com'>docs</a></body></html>"

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(web_retrieval_module.httpx, "Client", _Client)
    discovered = retriever._discover_from_searxng_html(
        query="아이폰 최신 정보",
        limit=5,
        logs=logs,
        url="http://127.0.0.1:8080/search",
    )
    assert discovered == []
    assert "notice:searxng_html_non_result_page" in logs


def test_run_populates_quality_metrics(monkeypatch):
    retriever = WebRetriever()

    monkeypatch.setattr(
        retriever,
        "discover_urls",
        lambda **kwargs: [
            _DiscoveredURL(title="A", url="https://a.example.com/p", snippet="a", source="ddg_html"),
            _DiscoveredURL(title="B", url="https://b.example.com/p", snippet="b", source="ddg_html"),
        ],
    )
    monkeypatch.setattr(
        retriever,
        "fetch_pages",
        lambda **kwargs: (
            [
                _FetchedPage(title="A", url="https://a.example.com/p", snippet="a", content="a content"),
                _FetchedPage(title="B", url="https://b.example.com/p", snippet="b", content="b content"),
            ],
            0,
        ),
    )
    monkeypatch.setattr(
        retriever,
        "build_evidence",
        lambda **kwargs: [
            WebSearchSource(title="A", url="https://a.example.com/p", snippet="a", content="a content"),
            WebSearchSource(title="B", url="https://b.example.com/p", snippet="b", content="b content"),
        ],
    )

    report = retriever.run(query="아이폰 최신", freshness_sensitive=True)
    assert report.round_query == "아이폰 최신"
    assert report.usable_source_count == 2
    assert report.unique_domain_count == 2
    assert 0.0 <= report.quality_score <= 1.0


def test_quality_score_formula_freshness_bonus():
    score = WebRetriever._quality_score(
        usable_source_count=3,
        unique_domain_count=3,
        fetch_success_count=3,
        fetch_failure_count=0,
        freshness_sensitive=True,
    )
    assert round(score, 4) == 1.0
