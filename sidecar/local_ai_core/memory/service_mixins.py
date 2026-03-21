from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import re
import time
from typing import Any, Callable, Literal

from ..db import Database
from ..models import (
    MemoryClearResponse,
    MemoryClearScope,
    MemoryEventRequest,
    MemoryEventResponse,
    PinnedMemoryItem,
    RelevantMemoryBundle,
    UserPreferenceItem,
    WorkspaceIdentity,
    WorkspaceMemoryMode,
)
from ..response_composer import ResponseComposer


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResolvedMemoryPreferences:
    response_length: str = "medium"
    show_citations: bool = True
    confirm_external_calls: bool = False
    prefer_action_suggestions: bool = True
    default_action_order: list[str] = None  # type: ignore[assignment]
    default_mode: str | None = None
    workspace_weights: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.default_action_order is None:
            self.default_action_order = []
        if self.workspace_weights is None:
            self.workspace_weights = {}


def _episodic_disabled(mode: WorkspaceMemoryMode) -> bool:
    """Return True when episodic/workspace memory should not be used."""
    return mode in {WorkspaceMemoryMode.DISABLED, WorkspaceMemoryMode.PINNED_ONLY}


ResolvedMemoryPreferences = None  # patched by memory_service.py
_episodic_disabled = None  # patched by memory_service.py

class MemoryServiceMethodsMixin:
    _SESSION_CONTEXT_KEYS = (
        "conversation_digest_v1",
        "last_conversational_context",
        "recent_query",
        "recent_event",
        "recent_file_ids",
        "recent_action",
    )
    _SESSION_CONTEXT_CLEANUP_MARKER = "__session_context_cleanup_v2"

    def get_relevant_session_memory(self, session_id: str) -> list:
        settings = self._db.get_settings()
        if not settings.session_memory_enabled:
            return []
        return self._db.get_relevant_session_memory(session_id=session_id, limit=40)

    def get_last_conversational_context(self, session_id: str) -> dict[str, Any] | None:
        for item in self.get_relevant_session_memory(session_id):
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
        for item in self.get_relevant_session_memory(session_id):
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
        for item in self.get_relevant_session_memory(session_id):
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

    def set_digest_model_refresher(
        self,
        refresher: Callable[[str, dict[str, Any]], dict[str, Any] | None] | None,
    ) -> None:
        self._digest_model_refresher = refresher

    def get_session_digest(self, session_id: str) -> dict[str, Any] | None:
        for item in self.get_relevant_session_memory(session_id):
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
        if not settings.session_memory_enabled:
            return
        safe: dict[str, Any] = dict(context)
        safe["updated_at"] = time.time()
        result_summary = str(safe.get("result_summary") or "").strip()
        if result_summary:
            cleaned_summary = self._sanitize_digest_text(result_summary, max_chars=260)
            if cleaned_summary and not self._should_drop_assistant_digest_text(
                assistant_text=cleaned_summary,
                user_query="",
            ):
                safe["result_summary"] = cleaned_summary
            else:
                safe["result_summary"] = ""
        self._db.write_session_memory(
            session_id=session_id,
            key="last_conversational_context",
            value_json=safe,
            ttl_hours=24,
            keep_recent=40,
        )

    def _write_digest(self, *, session_id: str, payload: dict[str, Any]) -> None:
        self._db.write_session_memory(
            session_id=session_id,
            key=self._DIGEST_KEY,
            value_json=self._normalize_digest_payload(payload),
            ttl_hours=24,
            keep_recent=40,
        )

    def _empty_digest(self) -> dict[str, Any]:
        return {
            "version": self._DIGEST_VERSION,
            "turn_count": 0,
            "active_topics": [],
            "stable_facts": [],
            "open_loops": [],
            "recent_turns": [],
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
    def _looks_like_open_loop(text: str) -> bool:
        value = (text or "").strip()
        if not value:
            return False
        lowered = value.lower()
        if "?" in value:
            return True
        ask_cues = (
            "어떻게",
            "어디",
            "무엇",
            "뭐",
            "언제",
            "왜",
            "될까",
            "해줘",
            "추천",
            "how",
            "what",
            "where",
            "when",
            "why",
            "can you",
            "should i",
        )
        return any(cue in lowered for cue in ask_cues)

    def _resolve_closed_loops(self, loops: list[str], assistant_text: str) -> list[str]:
        if not loops:
            return []
        answer_tokens = set(self._extract_topics(assistant_text))
        if not answer_tokens:
            return loops
        unresolved: list[str] = []
        for loop in loops:
            loop_tokens = set(self._extract_topics(loop))
            if not loop_tokens:
                unresolved.append(loop)
                continue
            overlap = len(loop_tokens.intersection(answer_tokens))
            ratio = overlap / max(1, len(loop_tokens))
            if ratio >= 0.45:
                continue
            unresolved.append(loop)
        return unresolved

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Workspace / episodic memory
    # ------------------------------------------------------------------

    def get_relevant_workspace_memory(self, workspace_id: str, intent: str | None) -> list:
        settings = self._db.get_settings()
        if not settings.workspace_memory_enabled:
            return []
        if _episodic_disabled(settings.workspace_memory_mode):
            return []
        return self._db.get_relevant_workspace_memory(workspace_id=workspace_id, intent=intent, limit=30)

    def get_relevant_episodic_memory(self, workspace_id: str | None, intent: str | None, related_files: list[str]) -> list:
        settings = self._db.get_settings()
        if not settings.workspace_memory_enabled:
            return []
        if _episodic_disabled(settings.workspace_memory_mode):
            return []
        return self._db.get_relevant_episodic_memory(
            workspace_id=workspace_id,
            intent=intent,
            related_file_ids=related_files,
            limit=20,
        )

    def get_user_preferences(self) -> list[UserPreferenceItem]:
        settings = self._db.get_settings()
        items = self._db.get_user_preferences()
        if settings.adaptive_personalization_enabled:
            return items
        return [item for item in items if item.source == "explicit"]

    def get_relevant_memory_bundle(
        self,
        *,
        session_id: str,
        workspace_id: str,
        intent: str,
        related_file_ids: list[str],
    ) -> RelevantMemoryBundle:
        settings = self._db.get_settings()
        session_items = self.get_relevant_session_memory(session_id)
        pref_items = self.get_user_preferences()
        pinned_items = self._db.list_pinned_memory(workspace_id=workspace_id, limit=30)

        # Episodic and workspace memory are skipped together when disabled.
        if _episodic_disabled(settings.workspace_memory_mode) or not settings.workspace_memory_enabled:
            workspace_items: list = []
            episodic_items: list = []
        else:
            workspace_items = self._db.get_relevant_workspace_memory(
                workspace_id=workspace_id, intent=intent, limit=30
            )
            episodic_items = self._db.get_relevant_episodic_memory(
                workspace_id=workspace_id,
                intent=intent,
                related_file_ids=related_file_ids,
                limit=20,
            )

        identity = self.get_workspace_identity()
        return RelevantMemoryBundle(
            workspace_identity=identity,
            session_items=session_items,
            workspace_items=workspace_items,
            preference_items=pref_items,
            episodic_items=episodic_items,
            pinned_items=pinned_items,
        )

    # ------------------------------------------------------------------
    # Write memory event
    # ------------------------------------------------------------------

    def write_memory_event(self, event: MemoryEventRequest) -> MemoryEventResponse:
        settings = self._db.get_settings()
        event_id = ""

        if settings.session_memory_enabled and event.session_id:
            self._write_session_event_batch(event)

        if (
            settings.workspace_memory_enabled
            and event.workspace_id
            and settings.workspace_memory_mode != WorkspaceMemoryMode.DISABLED
            and settings.workspace_memory_mode != WorkspaceMemoryMode.PINNED_ONLY
        ):
            # Explicit workspace rules from user-initiated setting changes.
            if event.event_type.value == "manual_override":
                if "default_mode" in event.metadata_json:
                    self._db.upsert_workspace_memory(
                        workspace_id=event.workspace_id,
                        memory_type="default_mode",
                        key="default_mode",
                        value_json={"value": event.metadata_json.get("default_mode")},
                        confidence=1.0,
                        source="explicit",
                    )
                if "privacy_rule" in event.metadata_json:
                    self._db.upsert_workspace_memory(
                        workspace_id=event.workspace_id,
                        memory_type="privacy_rule",
                        key="privacy_rule",
                        value_json={"value": event.metadata_json.get("privacy_rule")},
                        confidence=1.0,
                        source="explicit",
                    )

        episodic_enabled = not _episodic_disabled(settings.workspace_memory_mode)
        if settings.workspace_memory_enabled and episodic_enabled:
            record = self._db.insert_episodic_memory(
                workspace_id=event.workspace_id,
                event_type=event.event_type.value,
                summary=event.summary,
                related_file_ids=event.related_file_ids,
                related_action_ids=event.related_action_ids,
                metadata_json=event.metadata_json,
                importance=event.importance,
            )
            event_id = record.id
            if (
                event.workspace_id
                and settings.adaptive_personalization_enabled
                and self._should_refresh_inferred(event.event_type)
            ):
                self._refresh_inferred_with_throttle(event.workspace_id, min_interval_seconds=30.0)

        return MemoryEventResponse(event_id=event_id or "session-only", accepted=True)

    def _write_session_event_batch(self, event: MemoryEventRequest) -> None:
        """Write all session-level memory entries for one event in close succession."""
        self._db.write_session_memory(
            session_id=event.session_id,
            key="recent_event",
            value_json={
                "event_type": event.event_type.value,
                "summary": event.summary,
                "metadata": event.metadata_json,
            },
            ttl_hours=24,
            keep_recent=40,
        )
        if event.event_type.value == "query":
            self._db.write_session_memory(
                session_id=event.session_id,
                key="recent_query",
                value_json={
                    "summary": event.summary,
                    "mode": event.metadata_json.get("mode"),
                    "workspace_id": event.workspace_id,
                },
                ttl_hours=24,
                keep_recent=40,
            )
        if event.related_file_ids:
            self._db.write_session_memory(
                session_id=event.session_id,
                key="recent_file_ids",
                value_json={"file_ids": event.related_file_ids[:8]},
                ttl_hours=24,
                keep_recent=40,
            )
        if event.related_action_ids:
            self._db.write_session_memory(
                session_id=event.session_id,
                key="recent_action",
                value_json={"action_ids": event.related_action_ids[:4]},
                ttl_hours=24,
                keep_recent=40,
            )

    # kept for backward compat with callers that haven't migrated to snake_case
    def writeMemoryEvent(self, event: MemoryEventRequest) -> MemoryEventResponse:  # noqa: N802
        return self.write_memory_event(event)

    # ------------------------------------------------------------------
    # Clear / pin memory
    # ------------------------------------------------------------------

    def clear_memory(
        self,
        *,
        scope: MemoryClearScope,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryClearResponse:
        count = self._db.clear_memory(scope=scope, workspace_id=workspace_id, session_id=session_id)
        return MemoryClearResponse(cleared_rows=count, scope=scope)

    def pin_memory(
        self,
        *,
        memory_id: str | None,
        scope: str,
        workspace_id: str | None,
        title: str | None,
        content: str | None,
    ) -> PinnedMemoryItem:
        if memory_id:
            item = self._db.create_pin_from_memory(memory_id=memory_id, scope=scope, workspace_id=workspace_id)
            if item is not None:
                return item

        safe_title = (title or "Pinned Memory").strip() or "Pinned Memory"
        safe_content = (content or "").strip()
        if not safe_content:
            safe_content = "No content"
        return self._db.create_pinned_memory(
            scope=scope,
            workspace_id=workspace_id,
            title=safe_title,
            content=safe_content,
        )

    def unpin_memory(self, memory_id: str) -> bool:
        return self._db.delete_pinned_memory(memory_id=memory_id)

    def list_pinned_memory(self, *, scope: str | None = None, workspace_id: str | None = None) -> list[PinnedMemoryItem]:
        return self._db.list_pinned_memory(scope=scope, workspace_id=workspace_id, limit=120)

    # Backward-compat shims (camelCase aliases)
    def clearMemory(self, *, scope: MemoryClearScope, workspace_id: str | None = None, session_id: str | None = None) -> MemoryClearResponse:  # noqa: N802
        return self.clear_memory(scope=scope, workspace_id=workspace_id, session_id=session_id)

    def pinMemory(self, *, memory_id: str | None, scope: str, workspace_id: str | None, title: str | None, content: str | None) -> PinnedMemoryItem:  # noqa: N802
        return self.pin_memory(memory_id=memory_id, scope=scope, workspace_id=workspace_id, title=title, content=content)

    def unpinMemory(self, memory_id: str) -> bool:  # noqa: N802
        return self.unpin_memory(memory_id)

    def listPinnedMemory(self, *, scope: str | None = None, workspace_id: str | None = None) -> list[PinnedMemoryItem]:  # noqa: N802
        return self.list_pinned_memory(scope=scope, workspace_id=workspace_id)

    def getRelevantSessionMemory(self, session_id: str):  # noqa: N802
        return self.get_relevant_session_memory(session_id)

    def getRelevantWorkspaceMemory(self, workspace_id: str, intent: str | None = None):  # noqa: N802
        return self.get_relevant_workspace_memory(workspace_id, intent)

    def getUserPreferences(self):  # noqa: N802
        return self.get_user_preferences()

    def getRelevantEpisodicMemory(self, workspace_id: str | None, intent: str | None, related_files: list[str]):  # noqa: N802
        return self.get_relevant_episodic_memory(workspace_id, intent, related_files)

    # ------------------------------------------------------------------
    # Preference resolution
    # ------------------------------------------------------------------

    def resolve_preferences(self, bundle: RelevantMemoryBundle) -> ResolvedMemoryPreferences:
        explicit_map: dict[str, UserPreferenceItem] = {}
        inferred_map: dict[str, UserPreferenceItem] = {}
        for item in bundle.preference_items:
            if item.source == "explicit":
                explicit_map[item.key] = item
            else:
                existing = inferred_map.get(item.key)
                if existing is None or item.confidence > existing.confidence:
                    inferred_map[item.key] = item

        merged = dict(inferred_map)
        merged.update(explicit_map)

        resolved = ResolvedMemoryPreferences()
        if "response_length" in merged:
            resolved.response_length = str(merged["response_length"].value_json.get("value") or "medium")
        if "show_citations" in merged:
            resolved.show_citations = bool(merged["show_citations"].value_json.get("value", True))
        if "confirm_external_calls" in merged:
            resolved.confirm_external_calls = bool(merged["confirm_external_calls"].value_json.get("value", False))
        if "prefer_action_suggestions" in merged:
            resolved.prefer_action_suggestions = bool(merged["prefer_action_suggestions"].value_json.get("value", True))
        action_order: list[str] = []
        explicit_action = explicit_map.get("default_action_order")
        inferred_action = inferred_map.get("default_action_order")
        if explicit_action:
            action_order = self._as_str_list(explicit_action.value_json.get("value"))
        if not action_order:
            action_order = self._action_order_from_pins(bundle.pinned_items)
        if not action_order:
            action_order = self._action_order_from_workspace(bundle.workspace_items)
        if not action_order:
            action_order = self._action_order_from_episodic(bundle.episodic_items)
        if not action_order and inferred_action:
            action_order = self._as_str_list(inferred_action.value_json.get("value"))
        resolved.default_action_order = action_order

        default_mode: str | None = None
        explicit_mode = explicit_map.get("default_mode")
        inferred_mode = inferred_map.get("default_mode")
        if explicit_mode:
            value = explicit_mode.value_json.get("value")
            default_mode = str(value).strip() if value else None
        if not default_mode:
            default_mode = self._default_mode_from_pins(bundle.pinned_items)
        if not default_mode:
            default_mode = self._default_mode_from_workspace(bundle.workspace_items)
        if not default_mode:
            default_mode = self._default_mode_from_episodic(bundle.episodic_items)
        if not default_mode and inferred_mode:
            value = inferred_mode.value_json.get("value")
            default_mode = str(value).strip() if value else None
        resolved.default_mode = default_mode

        weights: dict[str, float] = {}
        for item in bundle.workspace_items:
            if item.memory_type != "retrieval_weight":
                continue
            weight = item.value_json.get("weight")
            try:
                weights[item.key] = max(0.5, min(float(weight), 1.8))
            except Exception:
                continue
        resolved.workspace_weights = weights
        return resolved

    # ------------------------------------------------------------------
    # Inferred memory refresh
    # ------------------------------------------------------------------

    def _refresh_inferred_workspace_memory(self, workspace_id: str) -> None:
        events = self._db.list_recent_episodic_memory(workspace_id=workspace_id, days=7, limit=240)
        if not events:
            return
        # Single pass — count modes and actions together.
        mode_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()
        for event in events:
            mode = str(event.metadata_json.get("mode") or "").strip()
            if mode:
                mode_counter[mode] += 1
            action = str(event.metadata_json.get("action_kind") or "").strip()
            if action:
                action_counter[action] += 1

        if mode_counter:
            mode, count = mode_counter.most_common(1)[0]
            if count >= 3:
                self._db.upsert_workspace_memory(
                    workspace_id=workspace_id,
                    memory_type="default_mode",
                    key="default_mode",
                    value_json={"value": mode},
                    confidence=0.62,
                    source="inferred",
                )

        if action_counter:
            ordered = [item for item, count in action_counter.most_common(5) if count >= 3]
            if ordered:
                self._db.upsert_workspace_memory(
                    workspace_id=workspace_id,
                    memory_type="preferred_actions",
                    key="preferred_actions",
                    value_json={"actions": ordered},
                    confidence=0.62,
                    source="inferred",
                )

    def _refresh_inferred_user_preferences(self, workspace_id: str, events: list) -> None:
        """Refresh user preference inferences from the *already-fetched* events list."""
        action_counter: Counter[str] = Counter()
        for event in events:
            action = str(event.metadata_json.get("action_kind") or "").strip()
            if action:
                action_counter[action] += 1
        ordered = [item for item, count in action_counter.most_common(5) if count >= 3]
        if ordered:
            self._db.upsert_user_preference(
                key="default_action_order",
                value_json={"value": ordered},
                source="inferred",
                confidence=0.62,
            )

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        output: list[str] = []
        for item in value:
            raw = str(item).strip()
            if raw:
                output.append(raw)
        return output

    def _action_order_from_workspace(self, items: list) -> list[str]:
        for item in items:
            if item.memory_type != "preferred_actions":
                continue
            actions = self._as_str_list(item.value_json.get("actions"))
            if actions:
                return actions
        return []

    @staticmethod
    def _default_mode_from_workspace(items: list) -> str | None:
        for item in items:
            if item.memory_type != "default_mode":
                continue
            value = str(item.value_json.get("value") or "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _default_mode_from_episodic(items: list) -> str | None:
        counter: Counter[str] = Counter()
        for item in items:
            mode = str(item.metadata_json.get("mode") or "").strip()
            if mode:
                counter[mode] += 1
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def _action_order_from_episodic(self, items: list) -> list[str]:
        counter: Counter[str] = Counter()
        for item in items:
            action = str(item.metadata_json.get("action_kind") or "").strip()
            if action:
                counter[action] += 1
        return [action for action, count in counter.most_common(5) if count >= 2]

    def _action_order_from_pins(self, pins: list[PinnedMemoryItem]) -> list[str]:
        for pin in pins:
            payload = self._parse_pin_content(pin.content)
            for key in ("actions", "value"):
                actions = self._as_str_list(payload.get(key))
                if actions:
                    return actions
        return []

    def _default_mode_from_pins(self, pins: list[PinnedMemoryItem]) -> str | None:
        for pin in pins:
            payload = self._parse_pin_content(pin.content)
            for key in ("default_mode", "value", "mode"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _parse_pin_content(content: str) -> dict[str, Any]:
        raw = (content or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    @staticmethod
    def _should_refresh_inferred(event_type) -> bool:
        return event_type.value in {
            "action_executed",
            "manual_override",
            "external_analysis",
            "comparison",
            "summary_created",
            "draft_created",
        }

    def _refresh_inferred_with_throttle(self, workspace_id: str, *, min_interval_seconds: float) -> None:
        now = time.monotonic()
        last = self._last_inferred_refresh_by_workspace.get(workspace_id, 0.0)
        if (now - last) < min_interval_seconds:
            return
        self._last_inferred_refresh_by_workspace[workspace_id] = now
        # Fetch events once and share with both refresh methods to avoid duplicate DB queries.
        events = self._db.list_recent_episodic_memory(workspace_id=workspace_id, days=7, limit=240)
        if events:
            # Reuse the count loop from _refresh_inferred_workspace_memory inline
            mode_counter: Counter[str] = Counter()
            action_counter: Counter[str] = Counter()
            for event in events:
                mode = str(event.metadata_json.get("mode") or "").strip()
                if mode:
                    mode_counter[mode] += 1
                action = str(event.metadata_json.get("action_kind") or "").strip()
                if action:
                    action_counter[action] += 1

            if mode_counter:
                mode, count = mode_counter.most_common(1)[0]
                if count >= 3:
                    self._db.upsert_workspace_memory(
                        workspace_id=workspace_id,
                        memory_type="default_mode",
                        key="default_mode",
                        value_json={"value": mode},
                        confidence=0.62,
                        source="inferred",
                    )
            if action_counter:
                ordered = [a for a, c in action_counter.most_common(5) if c >= 3]
                if ordered:
                    self._db.upsert_workspace_memory(
                        workspace_id=workspace_id,
                        memory_type="preferred_actions",
                        key="preferred_actions",
                        value_json={"actions": ordered},
                        confidence=0.62,
                        source="inferred",
                    )

            # User preferences — reuse same events list, no second DB query.
            self._refresh_inferred_user_preferences(workspace_id, events)
