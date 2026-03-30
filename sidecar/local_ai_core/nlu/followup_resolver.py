from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from ..models import ParsedIntent, ReasoningIntent, WorkMode


@dataclass(slots=True)
class FollowUpResolution:
    is_followup: bool = False
    followup_type: str | None = None
    resolved_intent: ReasoningIntent | None = None
    resolved_filters: dict[str, Any] = field(default_factory=dict)
    resolved_target_files: list[str] = field(default_factory=list)
    resolved_mode: WorkMode | None = None
    confidence: float = 0.0
    query_hint: str | None = None


class FollowUpResolver:
    _DEICTIC = (
        "그거",
        "그 파일",
        "그중",
        "그거 말고",
        "다른 거",
        "다음 거",
        "다음거",
        "요약만",
    )
    _ACTION_KO = {
        "열어": ReasoningIntent.OPEN_FILE,
        "열기": ReasoningIntent.OPEN_FILE,
        "요약": ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
        "비교": ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
        "정리": ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
    }
    _SMALLTALK = (
        "안녕",
        "반가워",
        "고마워",
        "감사",
        "오늘 어때",
        "잘 지내",
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "how are you",
    )
    _DOC_CUES = (
        "파일",
        "문서",
        "폴더",
        "디렉토리",
        "경로",
        "주차",
        "태그",
        "검색",
        "요약",
        "비교",
        "정리",
        "열어",
        "open",
        "file",
        "document",
        "folder",
        "directory",
        "path",
        "tag",
        "search",
        "summary",
        "summarize",
        "compare",
    )
    _CHAT_CUES = (
        "몇 시",
        "몇시",
        "잠",
        "자야",
        "새벽",
        "아침",
        "피곤",
        "고민",
        "괜찮아",
        "괜찮을까",
        "배고파",
        "뭐 먹",
        "추천",
        "기분",
        "운동",
        "목이",
        "아파",
        "오늘",
        "내일",
        "night",
        "sleep",
        "tired",
        "eat",
        "hungry",
        "feel",
    )
    _SCOPE_CLARIFICATION_HINT = "scope_target_needed"

    @staticmethod
    def _token_len(query: str) -> int:
        return len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", query or ""))

    @classmethod
    def _looks_standalone_general_chat(cls, *, query: str, lowered: str) -> bool:
        if not query:
            return False
        if any(token in lowered for token in cls._DOC_CUES):
            return False
        if any(token in lowered for token in cls._CHAT_CUES):
            return True
        token_len = cls._token_len(query)
        if token_len <= 10 and lowered.endswith(("?", "요", "까", "냐", "니", "지", "네", "!")):
            return True
        return False

    @classmethod
    def resolve(
        cls,
        *,
        query: str,
        parsed_intent: ParsedIntent,
        mode: WorkMode,
        last_context: dict[str, Any] | None,
        last_candidates: list[str],
        last_selected_file: str | None,
        last_actions: list[str],
    ) -> FollowUpResolution:
        text = (query or "").strip()
        lowered = text.lower()
        if not text:
            return FollowUpResolution()

        if parsed_intent.intent == ReasoningIntent.GENERAL_CHAT:
            return FollowUpResolution()
        if any(token in lowered for token in cls._SMALLTALK):
            return FollowUpResolution()
        if cls._looks_standalone_general_chat(query=text, lowered=lowered):
            return FollowUpResolution()
        if cls._needs_scope_target_clarification(parsed_intent=parsed_intent, last_context=last_context):
            return FollowUpResolution(
                is_followup=True,
                followup_type="clarify_scope_target",
                resolved_intent=None,
                resolved_mode=mode,
                confidence=0.66,
                query_hint=cls._SCOPE_CLARIFICATION_HINT,
            )

        token_len = cls._token_len(text)
        has_deictic = any(token in lowered for token in cls._DEICTIC)
        has_week = re.search(r"([1-9]|1[0-9]|2[0-4])\s*주차", lowered) is not None
        short_query = token_len <= 3 or len(text) <= 8
        weak_entities = not parsed_intent.entities.file_names and not parsed_intent.entities.projects and not parsed_intent.entities.tags
        previous_has_candidates = bool(last_candidates)
        previous_has_actions = bool(last_actions)

        # If the parser already identified a concrete retrieval/task intent, do not
        # override it into follow-up without explicit deictic cues.
        explicit_intents = {
            ReasoningIntent.FIND_FILE,
            ReasoningIntent.SUMMARIZE_FILE,
            ReasoningIntent.COMPARE_FILES,
            ReasoningIntent.EXPLAIN_CONTENT,
            ReasoningIntent.DRAFT_EDIT,
            ReasoningIntent.CLASSIFY,
        }
        if parsed_intent.intent in explicit_intents and not has_deictic and not has_week:
            return FollowUpResolution()

        is_followup = bool(
            (short_query and weak_entities and previous_has_candidates)
            or has_deictic
            or (short_query and previous_has_actions and parsed_intent.intent in {ReasoningIntent.FOLLOWUP_QUESTION, ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST})
            or (parsed_intent.intent == ReasoningIntent.FOLLOWUP_QUESTION and previous_has_candidates)
        )
        if not is_followup:
            return FollowUpResolution()

        resolution = FollowUpResolution(
            is_followup=True,
            followup_type="continue_previous_result",
            resolved_intent=ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
            resolved_mode=mode,
            confidence=0.62,
        )

        if has_week:
            week_text = re.search(r"([1-9]|1[0-9]|2[0-4])\s*주차", lowered)
            if week_text:
                week_value = week_text.group(1)
                resolution.followup_type = "refine_filter"
                resolution.resolved_intent = ReasoningIntent.FOLLOWUP_REFINE
                resolution.query_hint = f"{week_value}주차"
                resolution.resolved_filters["week"] = int(week_value)
                resolution.confidence = 0.75
                return resolution

        if "다른 거" in lowered or "그거 말고" in lowered or "다음 거" in lowered:
            resolution.followup_type = "next_candidate"
            resolution.resolved_intent = ReasoningIntent.NEXT_CANDIDATE
            resolution.resolved_target_files = last_candidates[1:4] if len(last_candidates) > 1 else []
            resolution.confidence = 0.78
            return resolution

        if "요약만" in lowered or (("요약" in lowered or "정리" in lowered) and short_query):
            resolution.followup_type = "request_action"
            resolution.resolved_intent = ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST
            if last_selected_file:
                resolution.resolved_target_files = [last_selected_file]
            elif last_candidates:
                resolution.resolved_target_files = [last_candidates[0]]
            resolution.query_hint = "핵심 요약"
            resolution.confidence = 0.83
            return resolution

        for token, mapped_intent in cls._ACTION_KO.items():
            if token in lowered:
                resolution.followup_type = "request_action"
                resolution.resolved_intent = mapped_intent
                if last_selected_file:
                    resolution.resolved_target_files = [last_selected_file]
                elif last_candidates:
                    resolution.resolved_target_files = [last_candidates[0]]
                resolution.confidence = 0.8
                if mapped_intent == ReasoningIntent.OPEN_FILE:
                    resolution.query_hint = "파일 열기"
                return resolution

        if "그거" in lowered or "그 파일" in lowered or "이거지" in lowered:
            resolution.followup_type = "soft_confirmation"
            resolution.resolved_intent = ReasoningIntent.SOFT_CONFIRM
            if last_selected_file:
                resolution.resolved_target_files = [last_selected_file]
            elif last_candidates:
                resolution.resolved_target_files = [last_candidates[0]]
            resolution.confidence = 0.74
            return resolution

        if short_query and previous_has_candidates:
            resolution.followup_type = "continue_previous_result"
            resolution.resolved_intent = ReasoningIntent.CONTINUE_PREVIOUS_RESULT
            if last_candidates:
                resolution.resolved_target_files = [last_candidates[0]]
            resolution.confidence = 0.66

        return resolution

    @staticmethod
    def _needs_scope_target_clarification(*, parsed_intent: ParsedIntent, last_context: dict[str, Any] | None) -> bool:
        operation = str(getattr(parsed_intent, "operation", "chat") or "chat")
        scope = str(getattr(parsed_intent, "scope", "single") or "single")
        ambiguity = str(getattr(parsed_intent, "ambiguity", "clear") or "clear")
        target = str(getattr(parsed_intent, "target", "") or "").strip()
        if operation not in {"find", "summarize", "open"}:
            return False
        if scope != "all":
            return False
        if ambiguity != "unclear" and target:
            return False
        context = last_context or {}
        # Ask at most once for ambiguous "all" scope queries.
        already_asked = bool(context.get("scope_clarification_pending")) or bool(context.get("scope_clarification_asked"))
        return not already_asked
