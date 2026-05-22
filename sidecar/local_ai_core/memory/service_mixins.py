from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import re
import time
import uuid
from typing import Any, Callable, Literal

from ..models import (
    WebMemoryEntry,
)
from ..composition.composer import ResponseComposer
from .mixins.workspace_methods import MemoryServiceWorkspaceMethodsMixin


logger = logging.getLogger(__name__)


class MemoryServiceMethodsMixin(MemoryServiceWorkspaceMethodsMixin):
    _WEB_MEMORY_KEY = "web_memory_entry"
    _WEB_MEMORY_KEEP_RECENT = 6
    _WEB_MEMORY_VECTOR_LIMIT = 4
    _WEB_MEMORY_VECTOR_PREFIX = "webmem"
    _SESSION_CONTEXT_KEYS = (
        "conversation_digest_v1",
        "last_conversational_context",
        _WEB_MEMORY_KEY,
        "recent_query",
        "recent_event",
        "recent_file_ids",
        "recent_action",
    )
    _SESSION_CONTEXT_CLEANUP_MARKER = "__session_context_cleanup_v2"
    _FACT_KEY_PREFIX = "fact:"
    _SCENE_KEY_PREFIX = "scene:"

    def get_relevant_session_memory(self, *, session_id: str) -> list:
        settings = self._db.get_settings()
        if not settings.session_memory_enabled:
            return []
        return self._db.get_relevant_session_memory(session_id=session_id, limit=120)

    def get_last_conversational_context(self, session_id: str) -> dict[str, Any] | None:
        for item in self.get_relevant_session_memory(session_id=session_id):
            if item.key != "last_conversational_context":
                continue
            payload = item.value_json
            if isinstance(payload, dict):
                return payload
        return None

    def get_last_candidate_set(self, session_id: str) -> list[str]:
        context = self.get_last_conversational_context(session_id)
        output: list[str] = []
        if context:
            candidates = context.get("top_candidates")
            if isinstance(candidates, list):
                for item in candidates:
                    if isinstance(item, str) and item.strip():
                        output.append(item.strip())
        if output:
            return output[:8]
        for item in self.get_relevant_session_memory(session_id=session_id):
            if item.key != "recent_file_ids":
                continue
            file_ids = item.value_json.get("file_ids")
            if isinstance(file_ids, list):
                return [str(v).strip() for v in file_ids if str(v).strip()][:8]
        return []

    def get_last_selected_file(self, session_id: str) -> str | None:
        context = self.get_last_conversational_context(session_id)
        if context:
            value = context.get("selected_file")
            if isinstance(value, str) and value.strip():
                return value.strip()
        for item in self.get_relevant_session_memory(session_id=session_id):
            if item.key != "recent_file_ids":
                continue
            file_ids = item.value_json.get("file_ids")
            if isinstance(file_ids, list) and file_ids:
                first = file_ids[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
        return None

    def get_last_shown_actions(self, session_id: str) -> list[str]:
        context = self.get_last_conversational_context(session_id)
        if not context:
            return []
        actions = context.get("shown_actions")
        if not isinstance(actions, list):
            return []
        output: list[str] = []
        for item in actions:
            if isinstance(item, str) and item.strip():
                output.append(item.strip())
        return output[:8]

    def write_web_memory_entry(
        self,
        *,
        session_id: str,
        query: str,
        answer_summary: str,
        sources: list[dict[str, Any]] | None,
        source_count: int,
        confidence: float,
        conversation_path: str,
    ) -> WebMemoryEntry | None:
        settings = self._db.get_settings()
        if not settings.session_memory_enabled:
            return None

        clean_query = self._sanitize_digest_text(str(query or ""), max_chars=260)
        clean_answer = self._sanitize_digest_text(str(answer_summary or ""), max_chars=500)
        clean_sources = self._normalize_web_memory_sources(sources or [])
        if not clean_sources or not clean_answer:
            return None

        entry_id = str(uuid.uuid4())
        created_at = self._now_iso()
        safe_confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        normalized_source_count = max(1, int(source_count or len(clean_sources)))
        vector_memory_id = self._web_memory_vector_id(session_id=session_id, entry_id=entry_id)

        entry_payload = {
            "entry_id": entry_id,
            "query": clean_query,
            "answer_summary": clean_answer,
            "sources": clean_sources[:4],
            "source_count": normalized_source_count,
            "confidence": safe_confidence,
            "created_at": created_at,
            "conversation_path": str(conversation_path or "").strip()[:80],
            "vector_memory_id": vector_memory_id,
        }
        self._db.write_session_memory(
            session_id=session_id,
            key=self._WEB_MEMORY_KEY,
            value_json=entry_payload,
            ttl_hours=24,
            keep_recent=40,
        )
        self._prune_web_memory_entries(session_id=session_id, keep_recent=self._WEB_MEMORY_KEEP_RECENT)
        self._vectorize_web_memory_entry(session_id=session_id, payload=entry_payload)
        try:
            return WebMemoryEntry.model_validate(entry_payload)
        except Exception:
            return None

    def get_recent_web_memory_entries(self, *, session_id: str, limit: int = 6) -> list[dict[str, Any]]:
        max_limit = max(1, int(limit or self._WEB_MEMORY_KEEP_RECENT))
        items = self.get_relevant_session_memory(session_id=session_id)
        output: list[dict[str, Any]] = []
        for item in items:
            if item.key != self._WEB_MEMORY_KEY:
                continue
            normalized = self._normalize_web_memory_entry(item.value_json)
            if not normalized:
                continue
            output.append(normalized)
            if len(output) >= max_limit:
                break
        return output

    def get_ranked_web_memory_entries(self, *, session_id: str, query: str, limit: int = 4) -> list[dict[str, Any]]:
        if not str(query or "").strip():
            return []

        max_limit = max(1, int(limit or self._WEB_MEMORY_VECTOR_LIMIT))
        recent = self.get_recent_web_memory_entries(session_id=session_id, limit=self._WEB_MEMORY_KEEP_RECENT)
        if not recent:
            return []

        vector_scores = self._vector_scores_for_web_memory(
            session_id=session_id,
            query=str(query or ""),
            limit=max(self._WEB_MEMORY_VECTOR_LIMIT, max_limit * 2),
        )
        has_vector_signal = bool(vector_scores)
        candidate_map: dict[str, dict[str, Any]] = {}
        for entry in recent:
            vector_memory_id = str(entry.get("vector_memory_id") or "").strip()
            if vector_memory_id:
                candidate_map[vector_memory_id] = entry
        if vector_scores:
            for item in self.get_relevant_session_memory(session_id=session_id):
                if item.key != self._WEB_MEMORY_KEY:
                    continue
                normalized = self._normalize_web_memory_entry(item.value_json)
                if not normalized:
                    continue
                vector_memory_id = str(normalized.get("vector_memory_id") or "").strip()
                if not vector_memory_id:
                    continue
                if vector_memory_id in vector_scores:
                    candidate_map.setdefault(vector_memory_id, normalized)
                if len(candidate_map) >= max(self._WEB_MEMORY_KEEP_RECENT + self._WEB_MEMORY_VECTOR_LIMIT, max_limit * 3):
                    break
        candidates = list(candidate_map.values()) if candidate_map else recent
        now_ts = time.time()
        ranked: list[dict[str, Any]] = []
        for entry in candidates:
            lexical = self._web_memory_lexical_score(query=str(query or ""), entry=entry)
            vector = max(0.0, min(1.0, float(vector_scores.get(str(entry.get("vector_memory_id") or ""), 0.0))))
            recency = self._web_memory_recency_score(created_at=str(entry.get("created_at") or ""), now_ts=now_ts)
            prior_confidence = max(0.0, min(1.0, float(entry.get("confidence") or 0.0)))
            if has_vector_signal:
                rank_score = max(0.0, min(1.0, (0.45 * lexical) + (0.40 * vector) + (0.15 * recency)))
            else:
                # VectorDB unavailable or empty result set: keep KV-only reuse path alive.
                rank_score = max(
                    0.0,
                    min(1.0, (0.55 * lexical) + (0.25 * recency) + (0.20 * prior_confidence)),
                )
            ranked.append(
                {
                    **entry,
                    "lexical_score": lexical,
                    "vector_score": vector,
                    "recency_score": recency,
                    "confidence": rank_score,
                }
            )
        ranked.sort(key=lambda row: float(row.get("confidence") or 0.0), reverse=True)
        return ranked[:max_limit]

    def _prune_web_memory_entries(self, *, session_id: str, keep_recent: int) -> None:
        target = max(1, int(keep_recent or self._WEB_MEMORY_KEEP_RECENT))
        try:
            rows = self._db.memory.get_session_memory(session_id, 200)
        except Exception:
            return
        web_ids: list[str] = []
        row_map: dict[str, Any] = {}
        for row in rows:
            try:
                key = str(row["key"] or "").strip()
            except Exception:
                key = ""
            if key != self._WEB_MEMORY_KEY:
                continue
            memory_id = str(row["id"] or "").strip()
            if memory_id:
                web_ids.append(memory_id)
                row_map[memory_id] = row
        if len(web_ids) <= target:
            return
        stale = web_ids[target:]
        if not stale:
            return
        stale_vector_ids: list[str] = []
        for stale_id in stale:
            row = row_map.get(stale_id)
            if row is None:
                continue
            payload = row["value_json"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
            if not isinstance(payload, dict):
                continue
            vector_id = str(payload.get("vector_memory_id") or "").strip()
            if vector_id:
                stale_vector_ids.append(vector_id)
        try:
            self._db.memory.delete_session_memory_by_ids(stale)
        except Exception:
            return
        if stale_vector_ids and self._vector_store:
            try:
                self._vector_store.delete_memories(stale_vector_ids)
            except Exception:
                return

    def _normalize_web_memory_sources(self, sources: list[dict[str, Any]]) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for raw in sources[:6]:
            if not isinstance(raw, dict):
                continue
            url = str(raw.get("url") or "").strip()[:500]
            if not url:
                continue
            title = self._sanitize_digest_text(str(raw.get("title") or ""), max_chars=160)
            snippet = self._sanitize_digest_text(str(raw.get("snippet") or ""), max_chars=280)
            output.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )
        return output

    def _normalize_web_memory_entry(self, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        entry_id = str(payload.get("entry_id") or "").strip()
        query = self._sanitize_digest_text(str(payload.get("query") or ""), max_chars=260)
        answer = self._sanitize_digest_text(str(payload.get("answer_summary") or ""), max_chars=500)
        sources = self._normalize_web_memory_sources(payload.get("sources") if isinstance(payload.get("sources"), list) else [])
        if not entry_id or not answer or not sources:
            return None
        source_count = max(1, int(payload.get("source_count") or len(sources)))
        confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
        created_at = str(payload.get("created_at") or "").strip() or self._now_iso()
        conversation_path = str(payload.get("conversation_path") or "").strip()[:80]
        vector_memory_id = str(payload.get("vector_memory_id") or "").strip()
        if not vector_memory_id:
            vector_memory_id = self._web_memory_vector_id(session_id="session", entry_id=entry_id)
        return {
            "entry_id": entry_id,
            "query": query,
            "answer_summary": answer,
            "sources": sources[:4],
            "source_count": source_count,
            "confidence": confidence,
            "created_at": created_at,
            "conversation_path": conversation_path,
            "vector_memory_id": vector_memory_id,
        }

    @staticmethod
    def _web_memory_vector_id(*, session_id: str, entry_id: str) -> str:
        safe_session = re.sub(r"[^A-Za-z0-9:_-]+", "_", str(session_id or "session")).strip("_") or "session"
        safe_entry = re.sub(r"[^A-Za-z0-9:_-]+", "_", str(entry_id or "")).strip("_") or str(uuid.uuid4()).replace("-", "")
        return f"webmem:{safe_session[:40]}:{safe_entry[:64]}"

    def _vectorize_web_memory_entry(self, *, session_id: str, payload: dict[str, Any]) -> None:
        if not self._vector_store or not self._embedding_service:
            return
        snippets: list[str] = []
        for source in payload.get("sources", [])[:4]:
            if not isinstance(source, dict):
                continue
            title = str(source.get("title") or "").strip()
            snippet = str(source.get("snippet") or "").strip()
            if title:
                snippets.append(title)
            if snippet:
                snippets.append(snippet)
        text = "\n".join(
            [
                f"query: {str(payload.get('query') or '').strip()}",
                f"summary: {str(payload.get('answer_summary') or '').strip()}",
                f"sources: {' | '.join(snippets[:8])}",
            ]
        ).strip()
        if len(text) < 16:
            return
        try:
            vector = self._embedding_service.embed_query(text)
            self._vector_store.upsert_memories(
                [
                    {
                        "memory_id": str(payload.get("vector_memory_id") or ""),
                        "session_id": session_id,
                        "workspace_id": "",
                        "text": text[:2000],
                        "vector": vector,
                        "created_at": str(payload.get("created_at") or self._now_iso()),
                    }
                ]
            )
        except Exception as exc:
            logger.warning("Failed to vectorize web memory for session %s: %s", session_id, exc)

    def _vector_scores_for_web_memory(self, *, session_id: str, query: str, limit: int) -> dict[str, float]:
        if not self._vector_store or not self._embedding_service:
            return {}
        try:
            query_vector = self._embedding_service.embed_query(query)
            hits = self._vector_store.search_memories_hybrid(
                query_text=query,
                query_vector=query_vector,
                session_id=session_id,
                limit=max(1, int(limit or self._WEB_MEMORY_VECTOR_LIMIT)),
            )
        except Exception:
            return {}

        scores: dict[str, float] = {}
        for hit in hits:
            memory_id = str(getattr(hit, "chunk_id", "") or "").strip()
            if not memory_id.startswith(f"{self._WEB_MEMORY_VECTOR_PREFIX}:"):
                continue
            hit_session = str(getattr(hit, "doc_id", "") or "").strip()
            if hit_session and hit_session != session_id:
                continue
            scores[memory_id] = max(scores.get(memory_id, 0.0), float(getattr(hit, "score", 0.0) or 0.0))
        if not scores:
            return {}
        peak = max(scores.values()) if scores else 0.0
        if peak <= 0.0:
            return {}
        for key in list(scores.keys()):
            scores[key] = max(0.0, min(1.0, float(scores[key]) / float(peak)))
        return scores

    def _web_memory_lexical_score(self, *, query: str, entry: dict[str, Any]) -> float:
        query_terms = set(re.findall(r"[A-Za-z가-힣0-9_]{2,24}", str(query or "").lower()))
        if not query_terms:
            return 0.0
        texts = [str(entry.get("query") or ""), str(entry.get("answer_summary") or "")]
        for source in entry.get("sources", [])[:4]:
            if not isinstance(source, dict):
                continue
            texts.append(str(source.get("title") or ""))
            texts.append(str(source.get("snippet") or ""))
        entry_terms = set(re.findall(r"[A-Za-z가-힣0-9_]{2,24}", " ".join(texts).lower()))
        if not entry_terms:
            return 0.0
        inter = len(query_terms.intersection(entry_terms))
        union = len(query_terms.union(entry_terms))
        if union <= 0:
            return 0.0
        return max(0.0, min(1.0, inter / union))

    @staticmethod
    def _web_memory_recency_score(*, created_at: str, now_ts: float) -> float:
        try:
            ts = datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.25
        elapsed_hours = max(0.0, (now_ts - ts) / 3600.0)
        return max(0.05, min(1.0, math.exp(-elapsed_hours / 36.0)))

    def set_digest_model_refresher(
        self,
        refresher: Callable[[str, dict[str, Any]], dict[str, Any] | None] | None,
    ) -> None:
        self._digest_model_refresher = refresher

    def get_session_digest(self, session_id: str) -> dict[str, Any] | None:
        for item in self.get_relevant_session_memory(session_id=session_id):
            if item.key != self._DIGEST_KEY:
                continue
            payload = item.value_json
            if not isinstance(payload, dict):
                continue
            return self._normalize_digest_payload(payload)
        return None

    def update_session_digest(
        self,
        session_id: str,
        user_query: str,
        assistant_summary: str,
        mode: Literal["rule", "hybrid"] = "rule",
    ) -> dict[str, Any]:
        settings = self._db.get_settings()
        if not settings.session_memory_enabled:
            return self._empty_digest()

        digest = self.get_session_digest(session_id) or self._empty_digest()
        user_text = self._sanitize_digest_text(user_query, max_chars=self._DIGEST_RECENT_TURN_MAX_CHARS)
        assistant_text = self._sanitize_digest_text(assistant_summary, max_chars=self._DIGEST_RECENT_TURN_MAX_CHARS)
        assistant_text = self._strip_leading_noise_token(assistant_text)
        if assistant_text and self._should_drop_assistant_digest_text(
            assistant_text=assistant_text,
            user_query=user_text,
        ):
            assistant_text = ""

        recent_turns = list(digest.get("recent_turns") or [])
        if user_text:
            recent_turns.append({"role": "user", "text": user_text})
        if assistant_text:
            recent_turns.append({"role": "assistant", "text": assistant_text})

        # Rolling summary compression: when recent_turns exceeds the verbatim window,
        # archive older turns into rolling_summary (L2 memory) so the context window
        # is not saturated while still preserving the gist of earlier conversation.
        verbatim_window = getattr(self, "_DIGEST_WINDOW_VERBATIM", 10)
        summary_max = getattr(self, "_DIGEST_ROLLING_SUMMARY_MAX_CHARS", 600)
        if len(recent_turns) > verbatim_window * 2:
            archive = recent_turns[: -(verbatim_window * 2)]
            recent_turns = recent_turns[-(verbatim_window * 2) :]
            existing_summary = str(digest.get("rolling_summary") or "").strip()
            digest["rolling_summary"] = self._compress_turns_to_summary(
                archive,
                existing_summary=existing_summary,
                max_chars=summary_max,
            )

        digest["recent_turns"] = self._normalize_recent_turns(recent_turns)

        new_topics = self._extract_topics(user_text)
        new_topics.extend(self._extract_topics(assistant_text))
        digest["active_topics"] = self._merge_ranked_values(
            existing=digest.get("active_topics"),
            additions=new_topics,
            cap=self._DIGEST_TOPICS_CAP,
        )

        fact_candidates: list[str] = []
        if self._looks_like_stable_fact(user_text):
            fact_candidates.append(user_text)
        if self._looks_like_stable_fact(assistant_text):
            fact_candidates.append(assistant_text)
        digest["stable_facts"] = self._merge_ranked_values(
            existing=digest.get("stable_facts"),
            additions=fact_candidates,
            cap=self._DIGEST_FACTS_CAP,
        )

        loop_candidates: list[str] = []
        if self._looks_like_open_loop(user_text):
            loop_candidates.append(user_text)
        merged_loops = self._merge_ranked_values(
            existing=digest.get("open_loops"),
            additions=loop_candidates,
            cap=self._DIGEST_OPEN_LOOPS_CAP,
        )
        if assistant_text and not self._looks_like_open_loop(assistant_text):
            merged_loops = self._resolve_closed_loops(merged_loops, assistant_text)
        digest["open_loops"] = merged_loops[: self._DIGEST_OPEN_LOOPS_CAP]

        digest["version"] = self._DIGEST_VERSION
        digest["turn_count"] = int(digest.get("turn_count") or 0) + 1
        digest["updated_at"] = self._now_iso()
        self._write_digest(session_id=session_id, payload=digest)

        refresh_mode = "rule"
        if mode == "hybrid" and digest["turn_count"] % 6 == 0:
            refresh_mode, refreshed = self.refresh_digest_with_model(session_id)
            digest = refreshed
        digest["digest_refresh"] = refresh_mode
        return digest

    def refresh_digest_with_model(self, session_id: str) -> tuple[str, dict[str, Any]]:
        digest = self.get_session_digest(session_id)
        if digest is None:
            return "fallback_rule", self._empty_digest()
        if self._digest_model_refresher is None:
            return "fallback_rule", digest
        try:
            refreshed = self._digest_model_refresher(session_id, dict(digest))
        except Exception as exc:
            logger.warning("Digest model refresh failed for session=%s: %s", session_id, exc)
            return "fallback_rule", digest
        if not isinstance(refreshed, dict):
            return "fallback_rule", digest
        merged = dict(digest)
        for key in ("active_topics", "stable_facts", "open_loops", "recent_turns"):
            if key in refreshed:
                merged[key] = refreshed.get(key)
        merged["version"] = self._DIGEST_VERSION
        merged["turn_count"] = int(digest.get("turn_count") or 0)
        merged["updated_at"] = self._now_iso()
        normalized = self._normalize_digest_payload(merged)
        self._write_digest(session_id=session_id, payload=normalized)
        return "model", normalized

    def write_conversational_context(
        self,
        *,
        session_id: str,
        context: dict[str, Any],
    ) -> None:
        settings = self._db.get_settings()
        print(f"\n[DEBUG_MEM] write_conversational_context: session_id={session_id}, session_memory_enabled={settings.session_memory_enabled}")
        if not settings.session_memory_enabled:
            print("[DEBUG_MEM] write_conversational_context: session_memory_enabled is False, returning early!")
            return
        safe: dict[str, Any] = dict(context)
        safe["updated_at"] = time.time()
        result_summary = self._strip_leading_noise_token(str(safe.get("result_summary") or ""))
        if result_summary:
            cleaned_summary = self._sanitize_digest_text(result_summary, max_chars=260)
            if cleaned_summary and not self._should_drop_assistant_digest_text(
                assistant_text=cleaned_summary,
                user_query="",
            ):
                safe["result_summary"] = cleaned_summary
            else:
                safe["result_summary"] = ""
        print(f"[DEBUG_MEM] write_conversational_context: writing safe context={safe}")
        self._db.write_session_memory(
            session_id=session_id,
            key="last_conversational_context",
            value_json=safe,
            ttl_hours=24,
            keep_recent=120,
        )

    def upsert_session_fact_memory(
        self,
        *,
        session_id: str,
        user_query: str,
    ) -> dict[str, Any]:
        settings = self._db.get_settings()
        if not settings.session_memory_enabled:
            return {"items": [], "overwrite_blocked": 0}
        facts = self._extract_fact_items_from_user_query(str(user_query or ""))
        if not facts:
            return {"items": [], "overwrite_blocked": 1}
        written: list[dict[str, Any]] = []
        for fact in facts:
            subject = str(fact.get("subject") or "").strip()
            if not subject:
                continue
            key = f"{self._FACT_KEY_PREFIX}{subject}"
            try:
                self._db.clear_session_memory_by_keys(keys=[key], session_id=session_id)
            except Exception:
                pass
            payload = {
                "memory_type": "fact",
                "memory_scope": "session",
                "subject": subject,
                "predicate": str(fact.get("predicate") or "").strip(),
                "value": str(fact.get("value") or "").strip(),
                "summary": str(fact.get("summary") or "").strip(),
                "confidence": float(fact.get("confidence") or 0.9),
                "updated_at": self._now_iso(),
            }
            self._db.write_session_memory(
                session_id=session_id,
                key=key,
                value_json=payload,
                ttl_hours=24 * 30,
                keep_recent=80,
            )
            written.append(payload)
        return {"items": written, "overwrite_blocked": 0}

    def upsert_session_scene_memory(
        self,
        *,
        session_id: str,
        user_query: str,
        assistant_summary: str = "",
    ) -> dict[str, Any]:
        settings = self._db.get_settings()
        if not settings.session_memory_enabled:
            return {"items": []}
        query = self._sanitize_digest_text(str(user_query or ""), max_chars=220)
        if not query:
            return {"items": []}
        if self._is_fact_query_or_command(query):
            return {"items": []}

        summary = self._sanitize_digest_text(str(assistant_summary or ""), max_chars=260)
        tags = self._extract_scene_tags(query)
        if not tags and not summary:
            return {"items": []}

        payload = {
            "memory_type": "scene",
            "memory_scope": "session",
            "query": query,
            "summary": summary,
            "tags": tags[:8],
            "confidence": 0.72,
            "updated_at": self._now_iso(),
        }
        scene_key = f"{self._SCENE_KEY_PREFIX}{uuid.uuid4().hex[:12]}"
        self._db.write_session_memory(
            session_id=session_id,
            key=scene_key,
            value_json=payload,
            ttl_hours=24 * 14,
            keep_recent=220,
        )
        return {"items": [payload]}

    @staticmethod
    def _extract_scene_tags(text: str) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        lowered = raw.lower()
        tokens = [tok for tok in re.findall(r"[A-Za-z가-힣0-9_]{2,24}", lowered)]
        stop = {
            "그냥", "진짜", "지금", "이번", "저번", "요즘", "please",
            "알려줘", "해줘", "해봐", "정리해줘", "추천해줘", "말해줘",
            "what", "when", "where", "which", "tell", "remember",
            "그리고", "근데", "그래서", "그러면", "the", "and",
        }
        out: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok in stop:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= 10:
                break
        return out

    def _extract_fact_items_from_user_query(self, query: str) -> list[dict[str, Any]]:
        text = self._sanitize_digest_text(str(query or ""), max_chars=220)
        if not text:
            return []
        if self._is_fact_query_or_command(text):
            return []
        out: list[dict[str, Any]] = []

        def _add(subject: str, predicate: str, value: str, summary: str, confidence: float = 0.92) -> None:
            clean_value = self._sanitize_digest_text(value, max_chars=80)
            if not clean_value or not self._is_valid_fact_value(clean_value):
                return
            out.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "value": clean_value,
                    "summary": self._sanitize_digest_text(summary, max_chars=180),
                    "confidence": max(0.0, min(1.0, float(confidence))),
                }
            )

        m_name = re.search(r"(?:내\s*이름(?:은|이야|은요)?\s*)([A-Za-z가-힣0-9]{2,20})", text, flags=re.IGNORECASE)
        if m_name:
            value = re.sub(r"(이야|야|입니다|예요|이에요)$", "", m_name.group(1).strip())
            if self._is_valid_fact_value(value):
                _add("user_name", "name", value, f"사용자 이름은 {value}입니다.")

        m_drink = re.search(r"(?:좋아하는\s*음료(?:는|가)?\s*)([A-Za-z가-힣0-9 ]{1,24})", text, flags=re.IGNORECASE)
        if m_drink:
            value = re.sub(r"(이야|야|입니다|예요|이에요)$", "", m_drink.group(1).strip().rstrip(".!,"))
            lowered_value = value.lower()
            if not any(token in lowered_value for token in ("뭐", "기억", "했", "what", "remember", "said")):
                _add("favorite_drink", "likes", value, f"사용자는 {value}를 좋아합니다.")

        m_pet = re.search(r"(?:반려(?:묘|견|동물)\s*이름(?:은|이야|은요)?\s*)([A-Za-z가-힣0-9]{1,20})", text, flags=re.IGNORECASE)
        if m_pet:
            value = re.sub(r"(이야|야|입니다|예요|이에요)$", "", m_pet.group(1).strip())
            _add("pet_name", "pet_name", value, f"사용자 반려동물 이름은 {value}입니다.")
        m_pet_habit = re.search(
            r"(?:고양이|반려묘|반려동물).{0,16}?(?:습관|버릇)(?:은|이|이야|은요)?\s*([A-Za-z가-힣0-9 ,]{2,60})",
            text,
            flags=re.IGNORECASE,
        )
        if m_pet_habit:
            value = m_pet_habit.group(1).strip().rstrip(".!,")
            _add("pet_habit", "pet_habit", value, f"사용자 반려동물 습관은 {value}입니다.", confidence=0.88)

        # Generic multi-domain conversational facts (travel/preferences/schedule).
        # Keep this deterministic and low-risk: only extract declarative patterns.
        m_destination = re.search(
            r"(?:로|으로)\s*([A-Za-z가-힣]{2,24})\s*(?:갈까|갈|여행|출장)",
            text,
            flags=re.IGNORECASE,
        )
        if m_destination:
            value = m_destination.group(1).strip()
            _add("trip_destination", "destination", value, f"여행 목적지는 {value}입니다.", confidence=0.86)

        m_duration = re.search(
            r"(\d+\s*박\s*\d+\s*일)",
            text,
            flags=re.IGNORECASE,
        )
        if m_duration:
            value = re.sub(r"\s+", "", m_duration.group(1).strip())
            _add("trip_duration", "duration", value, f"여행 기간은 {value}입니다.", confidence=0.86)

        m_arrival = re.search(
            r"((?:월|화|수|목|금|토|일)?요일?\s*)?(오전|오후|아침|점심|저녁|밤)?\s*(\d{1,2}\s*시(?:\s*반)?)",
            text,
            flags=re.IGNORECASE,
        )
        if m_arrival and ("도착" in text or "arriv" in text.lower()):
            parts = [str(part or "").strip() for part in m_arrival.groups() if str(part or "").strip()]
            value = " ".join(parts).strip().rstrip(".!,")
            if value:
                _add("arrival_time", "arrival_time", value, f"도착 시각/조건은 {value}입니다.", confidence=0.84)

        m_lodging = re.search(
            r"(?:숙소(?:는|가)?\s*)([^.!\n]{3,60})",
            text,
            flags=re.IGNORECASE,
        )
        if m_lodging:
            value = m_lodging.group(1).strip().rstrip(".!,")
            lowered_lodging = value.lower()
            has_lodging_cue = any(token in value for token in ("근처", "역", "호텔", "숙박", "오호리", "하카타", "후보"))
            has_budget_like = bool(
                ("예산" in value)
                or ("만원" in value)
                or re.search(r"\d+\s*원", value)
                or any(token in lowered_lodging for token in ("budget", "cost", "price"))
            )
            if has_lodging_cue and not has_budget_like:
                _add("lodging_candidates", "lodging_candidates", value, f"숙소 후보는 {value}입니다.", confidence=0.84)

        m_budget = re.search(
            r"(?:예산(?:은|이)?\s*)([^.!\n]{2,40})",
            text,
            flags=re.IGNORECASE,
        )
        if m_budget:
            value = m_budget.group(1).strip().rstrip(".!,")
            _add("budget", "budget", value, f"예산은 {value}입니다.", confidence=0.84)

        if "아침형" in text:
            _add("preference_morning", "preference", "아침형", "사용자는 아침형 성향입니다.", confidence=0.82)
        if any(token in text for token in ("조용한", "한산한")) and any(token in text for token in ("산책", "카페", "동네")):
            _add("preference_quiet", "preference", "조용한 동선 선호", "사용자는 조용하고 덜 붐비는 동선을 선호합니다.", confidence=0.82)

        return out

    @staticmethod
    def _is_fact_query_or_command(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return True
        if "?" in lowered:
            return True
        blocked_tokens = (
            "뭐야",
            "뭐였",
            "기억나",
            "알려줘",
            "what is",
            "what was",
            "remember",
            "tell me",
            "can you",
            "please",
        )
        return any(token in lowered for token in blocked_tokens)

    @staticmethod
    def _is_valid_fact_value(value: str) -> bool:
        raw = str(value or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        blocked_exact = {
            "안녕",
            "안녕하세요",
            "hello",
            "hi",
            "hey",
            "amente",
            "unknown",
            "none",
            "null",
        }
        if lowered in blocked_exact:
            return False
        if len(raw) <= 1:
            return False
        if re.search(r"[?？!！]$", raw):
            return False
        if re.search(r"(?:뭐|무엇|what|remember|기억|알려)", lowered):
            return False
        return True

    def _write_digest(self, *, session_id: str, payload: dict[str, Any]) -> None:
        normalized = self._normalize_digest_payload(payload)
        self._db.write_session_memory(
            session_id=session_id,
            key=self._DIGEST_KEY,
            value_json=normalized,
            ttl_hours=24,
            keep_recent=120,
        )
        # Phase 18: Vectorize memory for long-term retrieval
        self._vectorize_memory(session_id=session_id, payload=normalized)

    def _vectorize_memory(self, *, session_id: str, payload: dict[str, Any]) -> None:
        """Embed and store memory in LanceDB."""
        if not self._vector_store or not self._embedding_service:
            return

        # Combine topics and facts for a rich semantic representation
        topics = ", ".join(payload.get("active_topics", []))
        facts = " ".join(payload.get("stable_facts", []))
        text = f"Topics: {topics}. Facts: {facts}".strip()
        
        if not text or len(text) < 10:
            return

        try:
            vector = self._embedding_service.embed_query(text)
            try:
                workspace_id = str(self.get_workspace_identity().workspace_id or "default")
            except Exception:
                workspace_id = "default"
            
            self._vector_store.upsert_memories([{
                "memory_id": f"digest:{session_id}", # Overwrite existing digest for same session
                "session_id": session_id,
                "workspace_id": workspace_id,
                "text": text,
                "vector": vector,
                "created_at": payload.get("updated_at", ""),
            }])
        except Exception as exc:
            logger.warning("Failed to vectorize memory for session %s: %s", session_id, exc)

    def get_relevant_vector_memory(
        self, 
        *, 
        query: str, 
        session_id: str | None = None,
        workspace_id: str | None = None,
        limit: int = 4
    ) -> list[dict[str, Any]]:
        """Perform semantic search across memories."""
        if not self._vector_store or not self._embedding_service:
            return []
            
        try:
            query_vector = self._embedding_service.embed_query(query)
            # Level 2: Use hybrid search for memories (Dense + Sparse)
            hits = self._vector_store.search_memories_hybrid(
                query_text=query,
                query_vector=query_vector, 
                session_id=session_id, 
                workspace_id=workspace_id,
                limit=limit
            )
            return [
                {"text": hit.text, "session_id": hit.doc_id, "score": hit.score}
                for hit in hits
            ]
        except Exception:
            return []

    def _empty_digest(self) -> dict[str, Any]:
        return {
            "version": self._DIGEST_VERSION,
            "turn_count": 0,
            "active_topics": [],
            "stable_facts": [],
            "open_loops": [],
            "recent_turns": [],
            "rolling_summary": "",
            "updated_at": self._now_iso(),
        }

    def _normalize_digest_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(self._empty_digest())
        normalized["version"] = str(payload.get("version") or self._DIGEST_VERSION)
        normalized["turn_count"] = max(0, int(payload.get("turn_count") or 0))
        normalized["active_topics"] = self._normalize_str_list(
            payload.get("active_topics"),
            cap=self._DIGEST_TOPICS_CAP,
        )
        normalized["stable_facts"] = self._normalize_str_list(
            payload.get("stable_facts"),
            cap=self._DIGEST_FACTS_CAP,
        )
        normalized["open_loops"] = self._normalize_str_list(
            payload.get("open_loops"),
            cap=self._DIGEST_OPEN_LOOPS_CAP,
        )
        normalized["recent_turns"] = self._normalize_recent_turns(payload.get("recent_turns"))
        normalized["rolling_summary"] = str(payload.get("rolling_summary") or "").strip()
        updated_at = str(payload.get("updated_at") or "").strip()
        normalized["updated_at"] = updated_at or self._now_iso()
        return normalized

    def _normalize_recent_turns(self, turns: Any) -> list[dict[str, str]]:
        if not isinstance(turns, list):
            return []
        output: list[dict[str, str]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role_raw = str(turn.get("role") or "").strip().lower()
            role = "assistant" if role_raw == "assistant" else "user"
            text = self._sanitize_digest_text(
                str(turn.get("text") or ""),
                max_chars=self._DIGEST_RECENT_TURN_MAX_CHARS,
            )
            if not text:
                continue
            output.append({"role": role, "text": text})
        return output[-self._DIGEST_RECENT_TURNS_CAP :]

    @staticmethod
    def _compress_turns_to_summary(
        turns: list[dict[str, Any]],
        *,
        existing_summary: str,
        max_chars: int = 600,
    ) -> str:
        """Rule-based compression of archived conversation turns into a rolling summary.

        Extracts key information (named entities, numbers, topics, open questions)
        from the provided turns and merges them with an existing summary.
        No LLM is called — this is purely heuristic to avoid latency.
        """
        if not isinstance(turns, list):
            return existing_summary[:max_chars].strip()

        # Collect assistant and user text fragments
        user_fragments: list[str] = []
        assistant_fragments: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "").strip().lower()
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            if role == "user":
                user_fragments.append(text)
            elif role == "assistant":
                assistant_fragments.append(text)

        # Extract named entities and numbers (Korean + Latin)
        def _extract_entities(texts: list[str]) -> list[str]:
            found: list[str] = []
            seen: set[str] = set()
            for text in texts:
                # Korean proper nouns: 2-6 char sequences
                for match in re.findall(r"[가-힣]{2,6}", text):
                    key = match.casefold()
                    if key not in seen and len(found) < 12:
                        seen.add(key)
                        found.append(match)
                # Numeric facts (e.g. "3박4일", "15만원", "10시")
                for match in re.findall(r"\d+\s*[박일시원만천백]+", text):
                    key = re.sub(r"\s+", "", match).casefold()
                    if key not in seen and len(found) < 16:
                        seen.add(key)
                        found.append(match.strip())
                # Latin proper nouns: capitalized words
                for match in re.findall(r"\b[A-Z][a-z]{1,15}\b", text):
                    key = match.casefold()
                    if key not in seen and len(found) < 16:
                        seen.add(key)
                        found.append(match)
            return found[:14]

        # Extract unresolved questions from user turns
        def _extract_questions(texts: list[str]) -> list[str]:
            questions: list[str] = []
            seen: set[str] = set()
            for text in texts:
                sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    is_question = (
                        "?" in sent
                        or re.search(r"(까요|나요|인가요|어때요|할까요|될까요)\s*[.!?]?$", sent)
                    )
                    if is_question:
                        key = re.sub(r"[^가-힣A-Za-z0-9]", "", sent).casefold()[:30]
                        if key and key not in seen:
                            seen.add(key)
                            questions.append(sent[:80])
                    if len(questions) >= 3:
                        break
            return questions

        entities = _extract_entities(user_fragments + assistant_fragments)
        questions = _extract_questions(user_fragments)

        parts: list[str] = []
        if existing_summary:
            parts.append(existing_summary.rstrip(".").strip())
        if entities:
            parts.append("대화 중 언급된 주요 키워드는 " + ", ".join(entities[:10]) + "입니다")
        if questions:
            q_desc = [f"'{q.rstrip('?').strip()}'" for q in questions[:2]]
            parts.append("사용자가 주로 " + ", ".join(q_desc) + "에 대해 질문하거나 언급했습니다")

        merged = ". ".join(p for p in parts if p)
        if merged and not merged.endswith("."):
            merged += "."

        if not merged:
            return ""
        # Truncate gracefully at sentence boundary
        if len(merged) > max_chars:
            truncated = merged[:max_chars]
            last_break = max(
                truncated.rfind(". "),
                truncated.rfind("。"),
                truncated.rfind("! "),
                truncated.rfind("? "),
            )
            if last_break > max_chars // 2:
                merged = truncated[: last_break + 1].strip()
            else:
                merged = truncated.rstrip() + "…"
        return merged.strip()

    def _normalize_str_list(self, values: Any, *, cap: int) -> list[str]:
        if not isinstance(values, list):
            return []
        output: list[str] = []
        seen: set[str] = set()
        for raw in values:
            text = self._sanitize_digest_text(str(raw or ""), max_chars=self._DIGEST_ITEM_MAX_CHARS)
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(text)
            if len(output) >= cap:
                break
        return output

    def _merge_ranked_values(self, *, existing: Any, additions: list[str], cap: int) -> list[str]:
        merged: list[str] = self._normalize_str_list(existing, cap=cap)
        for raw in additions:
            text = self._sanitize_digest_text(raw, max_chars=self._DIGEST_ITEM_MAX_CHARS)
            if not text:
                continue
            key = text.casefold()
            merged = [item for item in merged if item.casefold() != key]
            merged.append(text)
            if len(merged) > cap:
                merged = merged[-cap:]
        return merged

    def _sanitize_digest_text(self, text: str, *, max_chars: int) -> str:
        cleaned = ResponseComposer._strip_instruction_leakage(text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""
        if self._looks_like_instruction_leak(cleaned):
            return ""
        return cleaned[:max_chars].strip()

    @classmethod
    def _should_drop_assistant_digest_text(cls, *, assistant_text: str, user_query: str) -> bool:
        cleaned = (assistant_text or "").strip()
        if not cleaned:
            return True
        if cls._looks_like_instruction_leak(cleaned):
            return True
        if cls._has_duplicate_sentence(cleaned):
            return True
        if cls._is_high_repetition_text(cleaned):
            return True
        if cls._looks_like_open_loop(cleaned):
            return True
        if cls._token_overlap(cleaned, user_query) >= 0.82 and len(cleaned) <= 220:
            return True
        if cls._contains_context_leak_phrase(cleaned):
            return True
        return False

    @staticmethod
    def _has_duplicate_sentence(text: str) -> bool:
        parts = [
            seg.strip()
            for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", text or "")
            if seg.strip()
        ]
        if len(parts) < 2:
            return False
        seen: set[str] = set()
        for seg in parts:
            key = re.sub(r"[^\w가-힣]+", "", seg).casefold()
            if not key:
                continue
            if key in seen:
                return True
            seen.add(key)
        return False

    @staticmethod
    def _is_high_repetition_text(text: str) -> bool:
        tokens = re.findall(r"[A-Za-z0-9가-힣_]+", (text or "").lower())
        if len(tokens) < 20:
            return False
        n = 4
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not grams:
            return False
        counts: dict[tuple[str, ...], int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        top = max(counts.values(), default=0)
        return top >= 5 or (top / len(grams)) >= 0.24

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        a_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (a or "").lower()))
        b_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (b or "").lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        union = len(a_tokens.union(b_tokens))
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _looks_like_instruction_leak(text: str) -> bool:
        lowered = (text or "").lower()
        leak_tokens = (
            "user:",
            "assistant:",
            "you:",
            "a:",
            "follow-up question:",
            "the question asks",
            "based on the evidence",
            "therefore, the answer is",
            "사용자에게 물어볼 때는",
            "추가적인 질문",
            "사용자 메시지에",
            "명확한 답변",
            "최종 답변:",
            "recent session context",
            "최근 세션 컨텍스트",
            "세션 컨텍스트",
            "이전 문장에 대한 답변으로",
            "직전 답변:",
            "사용자 질문:",
            "input message:",
            "conversation memory",
            "<conversation_memory>",
            "session summary:",
            "최종 답변 규칙",
            "규칙:",
            "추가 지시:",
            "사용자 마지막 메시지에",
            "역할 라벨",
        )
        return any(token in lowered for token in leak_tokens)

    @staticmethod
    def _contains_context_leak_phrase(text: str) -> bool:
        lowered = (text or "").lower()
        leak_terms = (
            "recent session context",
            "최근 세션 컨텍스트",
            "세션 컨텍스트",
            "이전 문장에 대한 답변으로",
            "직전 답변:",
            "사용자 질문:",
            "사용자:",
            "input message:",
            "<conversation_memory>",
            "session summary:",
            "추가 지시:",
        )
        return any(term in lowered for term in leak_terms)

    def clear_session_context_memory(self, *, session_id: str | None = None) -> int:
        keys = [key for key in self._SESSION_CONTEXT_KEYS if str(key).strip()]
        if not keys:
            return 0
        return self._db.clear_session_memory_by_keys(keys=keys, session_id=session_id)

    def clear_session_context_memory_once(self) -> int:
        marker = self._SESSION_CONTEXT_CLEANUP_MARKER
        prefs = self._db.get_user_preferences()
        for item in prefs:
            if item.key != marker:
                continue
            value = item.value_json if isinstance(item.value_json, dict) else {}
            if bool(value.get("done")):
                return 0
            break
        cleared = self.clear_session_context_memory(session_id=None)
        self._db.upsert_user_preference(
            key=marker,
            value_json={"done": True, "cleared_rows": int(cleared)},
            confidence=1.0,
            source="explicit",
        )
        return int(cleared)

    def _extract_topics(self, text: str) -> list[str]:
        raw = re.findall(r"[A-Za-z가-힣0-9_+\-]{2,24}", text or "")
        topics: list[str] = []
        seen: set[str] = set()
        for token in raw:
            key = token.casefold()
            if key in self._TOPIC_STOPWORDS:
                continue
            if key in seen:
                continue
            seen.add(key)
            topics.append(token)
            if len(topics) >= self._DIGEST_TOPICS_CAP:
                break
        return topics

    @staticmethod
    def _looks_like_stable_fact(text: str) -> bool:
        value = (text or "").strip()
        if not value:
            return False
        if "?" in value:
            return False
        if len(value) < 12:
            return False
        cues = (
            "나는",
            "난 ",
            "보통",
            "항상",
            "평소",
            "my ",
            "i am",
            "i usually",
            "i often",
            "i prefer",
        )
        lowered = value.lower()
        return any(cue in lowered for cue in cues)
    @staticmethod
    def _strip_leading_noise_token(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        # Prevent recurrence amplification in long sessions for known stray prefix token.
        value = re.sub(r"(?i)^\s*amente[\s,:\-\.]+\s*", "", value).strip()
        return value
