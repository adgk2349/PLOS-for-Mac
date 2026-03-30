from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class MemoryCommitter:
    def commit(
        self,
        *,
        memory: Any,
        composed: Any,
        req: Any,
        context: Any,
        session_id: str,
        session_digest_text: str,
    ) -> None:
        if not memory or not composed:
            return
        try:
            composed.metadata["context_digest_used"] = bool(session_digest_text)
            composed.metadata["context_injected"] = bool(session_digest_text)

            digest_refresh = "fallback_rule"
            digest_turn_count = 0
            try:
                updated_digest = memory.update_session_digest(
                    session_id=session_id,
                    user_query=req.query,
                    assistant_summary=str(composed.generated_text or ""),
                    mode="hybrid",
                )
                digest_turn_count = int(updated_digest.get("turn_count") or 0)
                digest_refresh = str(updated_digest.get("digest_refresh") or "rule")
            except (AttributeError, TypeError, ValueError, RuntimeError):
                digest_refresh = "fallback_rule"
            composed.metadata["digest_turn_count"] = digest_turn_count
            composed.metadata["digest_refresh"] = digest_refresh

            safe_web_sources: list[dict[str, str]] = []
            raw_web_sources = composed.metadata.get("web_sources")
            if isinstance(raw_web_sources, list):
                for row in raw_web_sources[:4]:
                    if not isinstance(row, dict):
                        continue
                    title = str(row.get("title") or "").strip()[:160]
                    url = str(row.get("url") or "").strip()[:500]
                    snippet = str(row.get("snippet") or "").strip()[:280]
                    if not url:
                        continue
                    safe_web_sources.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                        }
                    )
            shown_actions = [
                str(getattr(action, "kind", "") or "").strip()
                for action in (composed.actions or [])
                if str(getattr(action, "kind", "") or "").strip()
            ][:8]
            conversation_path = str(composed.metadata.get("conversation_path") or "local_conversation")
            memory.write_conversational_context(
                session_id=session_id,
                context={
                    "conversation_path": conversation_path,
                    "last_user_query": req.query,
                    "result_summary": str(composed.generated_text or "")[:500],
                    "intent": context.parsed_intent.intent.value,
                    "selected_file": composed.metadata.get("selected_file"),
                    "top_candidates": composed.metadata.get("top_candidates") or [],
                    "shown_actions": shown_actions,
                    "web_query": str(composed.metadata.get("web_query") or "")[:260],
                    "web_summary": str(composed.metadata.get("web_summary") or "")[:500],
                    "web_sources": safe_web_sources,
                },
            )

            web_summary = str(composed.metadata.get("web_summary") or "").strip()
            web_query = str(composed.metadata.get("web_query") or req.query).strip()
            if web_summary and safe_web_sources and (
                conversation_path.startswith("external_web_search")
                or conversation_path == "session_web_memory_reused"
            ):
                web_memory_rank = float(composed.metadata.get("web_memory_rank_score") or 0.0)
                if web_memory_rank <= 0:
                    web_memory_rank = float(getattr(composed.verification, "confidence", 0.62) or 0.62)
                memory.write_web_memory_entry(
                    session_id=session_id,
                    query=web_query,
                    answer_summary=web_summary,
                    sources=safe_web_sources,
                    source_count=len(safe_web_sources),
                    confidence=max(0.0, min(1.0, web_memory_rank)),
                    conversation_path=conversation_path,
                )
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning("[Orchestrator] Failed to save conversation context: %s", e)
