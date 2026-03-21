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
    assert len(digest["recent_turns"]) <= 8
    assert len(digest["active_topics"]) <= 8
    assert len(digest["stable_facts"]) <= 10
    assert len(digest["open_loops"]) <= 6
    # Assistant question-like turns are dropped from digest to prevent open-loop pollution.
    assert digest["recent_turns"][-1]["role"] == "user"


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

    items = memory.get_relevant_session_memory(session_id)
    assert len(items) <= 40
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
    items = memory.get_relevant_session_memory(session_id)
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
