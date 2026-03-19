from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .models import ParsedIntent, ReasoningIntent, WorkMode


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
        "그럼",
        "그중",
        "그거 말고",
        "다른 거",
        "다음 거",
        "요약만",
        "1주차",
        "2주차",
        "3주차",
        "4주차",
        "5주차",
        "6주차",
        "7주차",
        "8주차",
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

    @staticmethod
    def _token_len(query: str) -> int:
        return len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", query or ""))

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

        token_len = cls._token_len(text)
        has_deictic = any(token in lowered for token in cls._DEICTIC)
        has_week = re.search(r"([1-9]|1[0-6])\s*주차", lowered) is not None
        short_query = token_len <= 3 or len(text) <= 8
        weak_entities = not parsed_intent.entities.file_names and not parsed_intent.entities.projects and not parsed_intent.entities.tags
        previous_has_candidates = bool(last_candidates)
        previous_has_actions = bool(last_actions)

        is_followup = bool(
            (short_query and weak_entities and previous_has_candidates)
            or has_deictic
            or (short_query and previous_has_actions)
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
            week_text = re.search(r"([1-9]|1[0-6])\s*주차", lowered)
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
