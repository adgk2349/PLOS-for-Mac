from __future__ import annotations

import re
from datetime import datetime, timezone

from .models import ExecutionResult, ParsedIntent, VerificationResult, WorkMode


class ResultVerifier:
    def verify(
        self,
        *,
        parsed_intent: ParsedIntent,
        execution_result: ExecutionResult,
        mode: WorkMode,
    ) -> VerificationResult:
        issues: list[str] = []
        confidence = parsed_intent.confidence

        # Stage 1: retrieval grounding check.
        citations = execution_result.citations
        ungrounded_allowed = bool(execution_result.structured_payload.get("ungrounded_allowed", False))
        if not citations and not ungrounded_allowed:
            issues.append("no_citations")
            confidence -= 0.3
        elif citations:
            top_score = citations[0].score
            if top_score < 0.25:
                issues.append("low_relevance")
                confidence -= 0.2

        # Stage 2: time filter consistency check.
        if parsed_intent.time_filters.year is not None and citations:
            wanted = parsed_intent.time_filters.year
            if not any(c.modified_at.year == wanted for c in citations):
                issues.append("time_filter_mismatch")
                confidence -= 0.18

        if parsed_intent.time_filters.relative_days is not None and citations:
            cutoff = datetime.now(timezone.utc).timestamp() - (parsed_intent.time_filters.relative_days * 86400)
            if not any(c.modified_at.timestamp() >= cutoff for c in citations):
                issues.append("relative_time_mismatch")
                confidence -= 0.12

        # Stage 3: intent/result alignment check.
        if self._intent_mismatch(parsed_intent.intent.value, execution_result.result_type):
            issues.append("intent_mismatch")
            confidence -= 0.14

        # Stage 4: answer quality sanity checks (anti-loop / anti-dump).
        generated = execution_result.generated_text.strip()
        if self._is_repetitive_output(generated):
            issues.append("repetitive_output")
            confidence -= 0.16
        if self._is_raw_dump_output(generated):
            issues.append("raw_dump_output")
            confidence -= 0.14

        if not execution_result.generated_text.strip():
            issues.append("empty_result")
            confidence -= 0.22

        if mode == WorkMode.STRICT_SEARCH and (not citations or (citations and citations[0].score < 0.6)):
            issues.append("strict_threshold_not_met")
            confidence = min(confidence, 0.2)

        confidence = max(0.05, min(confidence, 0.99))
        ambiguity = max(0.0, 1.0 - confidence)
        candidate_mode = confidence < 0.45
        if execution_result.result_type == "file_list":
            has_items = isinstance(execution_result.structured_payload.get("items"), list) and bool(
                execution_result.structured_payload.get("items")
            )
            candidate_mode = not has_items and confidence < 0.3
        if ungrounded_allowed and execution_result.result_type in {"answer", "summary"}:
            candidate_mode = False
            confidence = max(confidence, 0.58)

        return VerificationResult(
            is_valid=not issues or confidence >= 0.4,
            confidence=confidence,
            issues=issues,
            ambiguity_level=ambiguity,
            candidate_mode=candidate_mode,
        )

    @staticmethod
    def _intent_mismatch(intent: str, result_type: str) -> bool:
        matrix = {
            "general_chat": {"conversation", "answer", "runtime_error"},
            "find_file": {"file_list", "candidate"},
            "summarize_file": {"answer", "summary", "candidate"},
            "compare_files": {"comparison", "answer", "candidate"},
            "explain_content": {"answer", "summary", "candidate"},
            "draft_edit": {"draft", "answer", "candidate"},
            "classify": {"classification", "answer", "candidate"},
            "followup_question": {"answer", "summary", "candidate", "file_list"},
            "followup_refine": {"file_list", "answer", "summary", "candidate"},
            "continue_previous_result": {"answer", "summary", "file_list", "candidate"},
            "soft_confirm": {"answer", "summary", "file_list", "candidate"},
            "select_previous_candidate": {"file_list", "answer", "summary", "candidate"},
            "next_candidate": {"file_list", "candidate"},
            "reduce_scope": {"file_list", "answer", "summary", "candidate"},
            "lightweight_action_request": {"answer", "summary", "file_list", "candidate"},
            "open_file": {"file_list", "candidate"},
        }
        allowed = matrix.get(intent)
        if not allowed:
            return False
        return result_type not in allowed

    @staticmethod
    def _is_repetitive_output(text: str) -> bool:
        if not text:
            return False
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) < 140:
            return False
        segments = [seg.strip() for seg in re.split(r"[.!?]\s+|\n+", compact) if seg.strip()]
        if len(segments) < 3:
            return False
        keys: list[str] = []
        for seg in segments:
            key = re.sub(r"[^\w가-힣]+", "", seg).lower()
            if len(key) < 20:
                continue
            keys.append(key)
        if len(keys) < 3:
            return False
        unique = len(set(keys))
        repeat_ratio = 1.0 - (unique / max(1, len(keys)))
        return repeat_ratio >= 0.35

    @staticmethod
    def _is_raw_dump_output(text: str) -> bool:
        if not text:
            return False
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) < 220:
            return False
        code_signals = (
            compact.count("{")
            + compact.count("}")
            + compact.count(";")
            + compact.lower().count("#include")
            + compact.lower().count("printf(")
            + compact.lower().count("void ")
        )
        return code_signals >= 8
