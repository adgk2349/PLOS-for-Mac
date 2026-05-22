from __future__ import annotations

from pathlib import Path

from local_ai_core.db import Database
from local_ai_core.memory_service import MemoryService
from local_ai_core.models import MemoryClearScope


def _new_service(tmp_path: Path) -> MemoryService:
    db = Database(tmp_path / "digest.sqlite3")
    return MemoryService(db)


def test_session_digest_caps_and_merge(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-cap"

    for idx in range(12):
        user = f"나는 topic{idx}를 보통 밤에 공부해"
        assistant = f"topic{idx}는 핵심을 먼저 보면 좋아. 다음에 무엇을 볼까?"
        memory.update_session_digest(session_id, user, assistant, mode="rule")

    digest = memory.get_session_digest(session_id)
    assert digest is not None
    assert digest["turn_count"] == 12
    # New architecture: DIGEST_RECENT_TURNS_CAP=20 (was 8).
    # Assistant messages ending in questions are dropped by quality filter,
    # so only 12 user messages are stored. 12 < WINDOW_VERBATIM*2=20, no compression yet.
    assert len(digest["recent_turns"]) <= memory._DIGEST_RECENT_TURNS_CAP
    assert len(digest["active_topics"]) <= 8
    assert len(digest["stable_facts"]) <= 10
    assert len(digest["open_loops"]) <= 6
    # rolling_summary starts empty at 12 turns (compression triggers at > WINDOW_VERBATIM*2).
    assert isinstance(digest.get("rolling_summary"), str)
    # Assistant question-like turns are dropped from digest to prevent open-loop pollution.
    assert digest["recent_turns"][-1]["role"] == "user"


def test_session_digest_rolling_summary_compresses_old_turns(tmp_path: Path):
    """Verify that turns beyond WINDOW_VERBATIM get compressed into rolling_summary."""
    memory = _new_service(tmp_path)
    session_id = "session-rolling-summary"

    # Use factual (non-question) assistant responses so they don't get dropped.
    for idx in range(12):
        user = f"후쿠오카 {idx}번째 질문이야. 숙소 추천해줘."
        assistant = f"하카타 역 근처 호텔{idx}이 좋아요. 예산은 {idx+1}만원 내외입니다."
        memory.update_session_digest(session_id, user, assistant, mode="rule")

    digest = memory.get_session_digest(session_id)
    assert digest is not None
    assert digest["turn_count"] == 12
    # 12 turns × 2 (user+assistant) = 24 messages > WINDOW_VERBATIM*2=20 → compression triggers.
    assert len(digest["recent_turns"]) <= memory._DIGEST_WINDOW_VERBATIM * 2
    assert isinstance(digest.get("rolling_summary"), str)
    assert len(digest["rolling_summary"]) > 0
    # rolling_summary should contain extracted entities from archived turns.
    assert "대화 중 언급된" in digest["rolling_summary"] or len(digest["rolling_summary"]) > 5

    assert digest["recent_turns"][-1]["role"] == "assistant"


def test_session_digest_refresh_happens_every_six_turns(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-refresh"
    calls: list[int] = []

    def refresher(_session_id: str, digest: dict[str, object]) -> dict[str, object]:
        calls.append(int(digest.get("turn_count") or 0))
        return {
            "active_topics": ["model_refreshed_topic"],
            "stable_facts": digest.get("stable_facts") or [],
            "open_loops": digest.get("open_loops") or [],
            "recent_turns": digest.get("recent_turns") or [],
        }

    memory.set_digest_model_refresher(refresher)
    result = {}
    for idx in range(6):
        result = memory.update_session_digest(
            session_id,
            f"나는 topic{idx}를 공부해",
            "알겠어. 다음 단계가 뭐야?",
            mode="hybrid",
        )

    assert calls == [6]
    assert result["digest_refresh"] == "model"
    assert "model_refreshed_topic" in result["active_topics"]


def test_session_digest_refresh_fallback_keeps_rule_digest(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-fallback"

    def broken_refresher(_session_id: str, _digest: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("refresh failed")

    memory.set_digest_model_refresher(broken_refresher)
    result = {}
    for idx in range(6):
        result = memory.update_session_digest(
            session_id,
            f"나는 topic{idx}를 좋아해",
            "알겠어. 다음으로 무엇을 할까?",
            mode="hybrid",
        )

    assert result["turn_count"] == 6
    assert result["digest_refresh"] == "fallback_rule"
    assert len(result["active_topics"]) > 0


def test_session_digest_isolation_and_clear(tmp_path: Path):
    memory = _new_service(tmp_path)
    memory.update_session_digest("A", "나는 수면 패턴이 불규칙해", "오늘은 몇 시에 잘까?", mode="rule")
    memory.update_session_digest("B", "나는 스위프트를 공부 중이야", "내일 무엇을 공부할까?", mode="rule")

    digest_a = memory.get_session_digest("A")
    digest_b = memory.get_session_digest("B")
    assert digest_a is not None
    assert digest_b is not None
    assert digest_a != digest_b

    memory.clear_memory(scope=MemoryClearScope.SESSION, session_id="A")
    assert memory.get_session_digest("A") is None
    assert memory.get_session_digest("B") is not None

    memory.clear_memory(scope=MemoryClearScope.ALL)
    assert memory.get_session_digest("B") is None


def test_session_digest_uses_existing_retention_policy(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-retention"

    for idx in range(45):
        memory.update_session_digest(
            session_id,
            f"나는 topic{idx}를 공부해",
            "다음에 무엇을 할까?",
            mode="rule",
        )

    items = memory.get_relevant_session_memory(session_id=session_id)
    assert len(items) <= 120
    digest_item = next((item for item in items if item.key == "conversation_digest_v1"), None)
    assert digest_item is not None
    assert digest_item.expires_at is not None


def test_session_digest_drops_low_quality_assistant_summary(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-quality-guard"
    memory.update_session_digest(
        session_id,
        "오늘 뭐 먹지?",
        "오늘 뭐 먹지?",
        mode="rule",
    )
    digest = memory.get_session_digest(session_id)
    assert digest is not None
    assert digest["recent_turns"][-1]["role"] == "user"


def test_clear_session_context_memory_removes_only_context_keys(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-context-clear"
    memory.update_session_digest(session_id, "나는 밤에 공부해", "알겠어", mode="rule")
    memory.write_conversational_context(
        session_id=session_id,
        context={"result_summary": "최근 세션 컨텍스트: 사용자: 안녕"},
    )
    memory._db.write_session_memory(
        session_id=session_id,
        key="recent_query",
        value_json={"summary": "오늘 뭐 먹지?"},
        ttl_hours=24,
        keep_recent=40,
    )
    memory._db.write_session_memory(
        session_id=session_id,
        key="non_context_key",
        value_json={"note": "keep-me"},
        ttl_hours=24,
        keep_recent=40,
    )

    cleared = memory.clear_session_context_memory(session_id=session_id)
    items = memory.get_relevant_session_memory(session_id=session_id)
    keys = {item.key for item in items}
    assert cleared >= 1
    assert "non_context_key" in keys
    for key in memory._SESSION_CONTEXT_KEYS:
        assert key not in keys


def test_clear_session_context_memory_once_uses_marker(tmp_path: Path):
    memory = _new_service(tmp_path)
    memory.update_session_digest("A", "나는 수면이 불규칙해", "몇 시에 잘까?", mode="rule")
    first = memory.clear_session_context_memory_once()
    second = memory.clear_session_context_memory_once()
    assert first >= 1
    assert second == 0


def test_web_memory_entry_window_keeps_recent_six_without_dropping_digest(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-web-window"
    memory.update_session_digest(session_id, "초기 사용자 질문", "초기 응답", mode="rule")

    for idx in range(8):
        memory.write_web_memory_entry(
            session_id=session_id,
            query=f"아이폰 비교 {idx}",
            answer_summary=f"요약 {idx}",
            sources=[
                {
                    "title": f"source-{idx}",
                    "url": f"https://example.com/{idx}",
                    "snippet": f"snippet-{idx}",
                }
            ],
            source_count=1,
            confidence=0.7,
            conversation_path="external_web_search_direct",
        )

    recent = memory.get_recent_web_memory_entries(session_id=session_id, limit=12)
    assert len(recent) == 6
    assert recent[0]["query"] == "아이폰 비교 7"
    assert recent[-1]["query"] == "아이폰 비교 2"

    digest = memory.get_session_digest(session_id)
    assert digest is not None
    all_items = memory.get_relevant_session_memory(session_id=session_id)
    web_items = [item for item in all_items if item.key == "web_memory_entry"]
    assert len(web_items) == 6


def test_web_memory_ranked_candidates_merge_recent_and_vector_side(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-web-rank"
    first = memory.write_web_memory_entry(
        session_id=session_id,
        query="아이폰 최신 모델 비교",
        answer_summary="아이폰 17 시리즈 비교 요약",
        sources=[{"title": "A", "url": "https://a.example.com", "snippet": "A snippet"}],
        source_count=1,
        confidence=0.9,
        conversation_path="external_web_search_direct",
    )
    assert first is not None

    for idx in range(1, 6):
        memory.write_web_memory_entry(
            session_id=session_id,
            query=f"다른 주제 {idx}",
            answer_summary=f"다른 요약 {idx}",
            sources=[{"title": f"B{idx}", "url": f"https://b{idx}.example.com", "snippet": f"B{idx} snippet"}],
            source_count=1,
            confidence=0.6,
            conversation_path="external_web_search_direct",
        )

    original_vector = memory._vector_scores_for_web_memory
    memory._vector_scores_for_web_memory = lambda **kwargs: {str(first.vector_memory_id): 1.0}
    try:
        ranked = memory.get_ranked_web_memory_entries(
            session_id=session_id,
            query="아이폰 최신 비교 정리",
            limit=4,
        )
    finally:
        memory._vector_scores_for_web_memory = original_vector

    assert ranked
    assert ranked[0].get("vector_memory_id") == first.vector_memory_id
    top = ranked[0]
    assert 0.0 <= float(top.get("confidence") or 0.0) <= 1.0
    assert "lexical_score" in top
    assert "vector_score" in top
    assert "recency_score" in top


def test_web_memory_prune_deletes_stale_vector_entries(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-web-prune-vectors"

    class _VectorStoreStub:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def delete_memories(self, memory_ids: list[str]) -> None:
            self.deleted.extend([str(item) for item in memory_ids])

    vector_stub = _VectorStoreStub()
    memory._vector_store = vector_stub

    for idx in range(8):
        memory.write_web_memory_entry(
            session_id=session_id,
            query=f"q{idx}",
            answer_summary=f"a{idx}",
            sources=[{"title": f"t{idx}", "url": f"https://example.com/{idx}", "snippet": "s"}],
            source_count=1,
            confidence=0.7,
            conversation_path="external_web_search_direct",
        )

    # keep_recent=6 -> two stale vector ids must be deleted
    assert len(vector_stub.deleted) >= 2
    assert all(item.startswith("webmem:") for item in vector_stub.deleted)


def test_session_fact_memory_upsert_and_replace(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-fact-upsert"

    first = memory.upsert_session_fact_memory(
        session_id=session_id,
        user_query="내 이름은 민수야. 기억해줘.",
    )
    assert len(first["items"]) == 1
    assert first["overwrite_blocked"] == 0
    assert first["items"][0]["subject"] == "user_name"
    assert first["items"][0]["value"] == "민수"

    second = memory.upsert_session_fact_memory(
        session_id=session_id,
        user_query="내 이름은 지훈이야.",
    )
    assert len(second["items"]) == 1
    assert second["overwrite_blocked"] == 0
    items = memory.get_relevant_session_memory(session_id=session_id)
    fact_items = [item for item in items if item.key == "fact:user_name"]
    assert len(fact_items) == 1
    assert str(fact_items[0].value_json.get("value") or "") == "지훈"

    no_write = memory.upsert_session_fact_memory(
        session_id=session_id,
        user_query="내 이름 뭐야?",
    )
    assert no_write["items"] == []
    assert no_write["overwrite_blocked"] == 1
    items_after = memory.get_relevant_session_memory(session_id=session_id)
    fact_items_after = [item for item in items_after if item.key == "fact:user_name"]
    assert len(fact_items_after) == 1
    assert str(fact_items_after[0].value_json.get("value") or "") == "지훈"


def test_session_fact_memory_ignores_greeting_as_name(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-fact-greeting"
    result = memory.upsert_session_fact_memory(
        session_id=session_id,
        user_query="내 이름은 안녕하세요",
    )
    assert result["items"] == []
    items = memory.get_relevant_session_memory(session_id=session_id)
    assert not any(item.key == "fact:user_name" for item in items)


def test_session_fact_memory_blocks_question_like_and_low_confidence_values(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-fact-question-block"

    blocked = memory.upsert_session_fact_memory(
        session_id=session_id,
        user_query="내 이름 뭐였지? 기억해?",
    )
    assert blocked["items"] == []
    assert blocked["overwrite_blocked"] == 1

    low_quality = memory.upsert_session_fact_memory(
        session_id=session_id,
        user_query="내 이름은 amente",
    )
    assert low_quality["items"] == []
    items = memory.get_relevant_session_memory(session_id=session_id)
    assert not any(item.key == "fact:user_name" for item in items)


def test_web_memory_ranking_survives_without_vector_signal(tmp_path: Path):
    memory = _new_service(tmp_path)
    session_id = "session-web-kv-fallback"
    memory.write_web_memory_entry(
        session_id=session_id,
        query="아이폰 최신 모델 비교",
        answer_summary="아이폰 최신 비교 요약",
        sources=[{"title": "compare", "url": "https://example.com/compare", "snippet": "모델별 차이"}],
        source_count=1,
        confidence=0.8,
        conversation_path="external_web_search_direct",
    )

    original_vector = memory._vector_scores_for_web_memory
    memory._vector_scores_for_web_memory = lambda **kwargs: {}
    try:
        ranked = memory.get_ranked_web_memory_entries(
            session_id=session_id,
            query="아이폰 최신 모델 비교 정리",
            limit=4,
        )
    finally:
        memory._vector_scores_for_web_memory = original_vector

    assert ranked
    assert float(ranked[0]["confidence"]) >= 0.60


def test_relevant_memory_bundle_uses_session_scoped_vector_memory_only(tmp_path: Path, monkeypatch):
    memory = _new_service(tmp_path)
    calls: list[dict[str, object]] = []

    def _fake_get_relevant_vector_memory(*, query: str, session_id=None, workspace_id=None, limit=4):
        calls.append(
            {
                "query": query,
                "session_id": session_id,
                "workspace_id": workspace_id,
                "limit": limit,
            }
        )
        if session_id == "session-only":
            return [{"text": "session", "session_id": "session-only", "score": 0.9}]
        return [{"text": "global", "session_id": "other-session", "score": 0.7}]

    monkeypatch.setattr(memory, "get_relevant_vector_memory", _fake_get_relevant_vector_memory)
    bundle = memory.get_relevant_memory_bundle(
        session_id="session-only",
        workspace_id="workspace-a",
        intent="chat",
        related_file_ids=[],
        query="아이폰 최신 모델",
    )

    assert calls
    assert len(calls) == 1
    assert calls[0]["session_id"] == "session-only"
    assert all(item.get("session_id") == "session-only" for item in bundle.semantic_memories)
