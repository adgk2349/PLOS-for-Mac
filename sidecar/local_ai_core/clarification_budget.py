from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ClarificationBudgetState:
    clarification_count_current_turn: int = 0
    previous_turn_was_clarification: bool = False
    partial_user_answer_received: bool = False


class ClarificationBudget:
    @staticmethod
    def _is_partial_answer(query: str) -> bool:
        text = (query or "").strip()
        if not text:
            return False
        if len(text) <= 12:
            return True
        keywords = ("주차", "pdf", "md", "파일", "태그", "연도", "최근", "그거", "다른 거", "요약만", "열어")
        lowered = text.lower()
        return any(token in lowered for token in keywords)

    def allow_clarification(
        self,
        *,
        state: ClarificationBudgetState,
        query: str,
        ambiguity_level: float,
        risk_level: str,
        candidate_gap_small: bool,
    ) -> bool:
        if state.clarification_count_current_turn >= 1:
            return False
        if state.previous_turn_was_clarification:
            if state.partial_user_answer_received or self._is_partial_answer(query):
                return False
        if risk_level == "high":
            return True
        if ambiguity_level >= 0.7 and candidate_gap_small:
            return True
        return False

