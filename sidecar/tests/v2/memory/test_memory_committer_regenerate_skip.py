from __future__ import annotations

from types import SimpleNamespace

from local_ai_core.reasoning.orchestrator.memory_committer import MemoryCommitter


class _MemoryProbe:
    def __init__(self) -> None:
        self.digest_calls = 0
        self.context_calls = 0
        self.fact_calls = 0

    def update_session_digest(self, session_id: str, user_query: str, assistant_summary: str, mode: str = "hybrid"):
        self.digest_calls += 1
        return {"turn_count": 1, "digest_refresh": "rule"}

    def write_conversational_context(self, *, session_id: str, context: dict):
        self.context_calls += 1

    def upsert_session_fact_memory(self, *, session_id: str, user_query: str):
        self.fact_calls += 1
        return {"items": [{"subject": "user_name", "value": "민수"}], "overwrite_blocked": 0}


def test_memory_committer_skips_retry_only_runtime_message():
    memory = _MemoryProbe()
    composed = SimpleNamespace(
        generated_text="로컬 생성이 시간 내 완료되지 않았습니다. '응답 다시 생성'으로 재시도해 주세요.",
        metadata={},
        actions=[],
        verification=SimpleNamespace(confidence=0.5),
        execution_result=SimpleNamespace(
            structured_payload={
                "reason": "generation_retry_exhausted",
                "offer_regenerate": True,
            }
        ),
    )
    req = SimpleNamespace(query="테스트 질의")
    context = SimpleNamespace(parsed_intent=SimpleNamespace(intent=SimpleNamespace(value="general_chat")))

    MemoryCommitter().commit(
        memory=memory,
        composed=composed,
        req=req,
        context=context,
        session_id="sess-1",
        session_digest_text="",
    )

    assert memory.digest_calls == 0
    assert memory.context_calls == 0
    assert memory.fact_calls == 1
    assert composed.metadata["digest_refresh"] == "skipped_regenerate"
    assert composed.metadata["fact_memory_upserted"] == 1
