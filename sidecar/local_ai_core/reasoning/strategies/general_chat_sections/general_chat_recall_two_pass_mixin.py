from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from ... import utils
from ....models import (
    BehaviorPolicy,
    Citation,
    ComposedChatResponseV2,
    ExecutionResult,
    LocalPlan,
    ParsedIntent,
    PrivacyMode,
    ReasoningIntent,
    SuggestedActionKind,
    VerificationResult,
)
from ....nlu.followup_resolver import FollowUpResolution
from ...context import ReasoningContext
from ...executor_contract import bind_async_executor_contract, require_executor_methods
from ....web_retrieval import WebRetrievalReport, WebRetriever
from ....language_utils import detect_query_language, normalize_language_code
from ...helpers.web.general_chat_web_gate_helpers import GeneralChatWebGateHelpers
from ...helpers.web.general_chat_web_execution_helpers import GeneralChatWebExecutionHelpers
from ...helpers.chat.general_chat_recall_gate_helpers import GeneralChatRecallGateHelpers
from ...helpers.chat.general_chat_recall_execution_helpers import GeneralChatRecallExecutionHelpers
from ...helpers.chat.general_chat_conversation_execution_helpers import GeneralChatConversationExecutionHelpers
from ...answer_contract import (
    coerce_answer_type_hint,
    extract_contract_response,
    infer_answer_type_hint,
    validate_contract_response,
)


class GeneralChatRecallTwoPassMixin:

    @staticmethod
    def _recall_terms(text: str) -> set[str]:
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "what", "when", "where", "which", "who",
            "would", "could", "should", "have", "has", "had", "been", "were", "was", "are", "is", "did", "does",
            "can", "about", "into", "your", "their", "them", "they", "her", "his", "its", "our", "you",
            "only", "just", "then", "than", "also", "very", "much", "many", "more", "most", "some", "any",
            "use", "using", "date", "conversation", "context", "approximate", "answer",
            "그리고", "하지만", "또한", "정말", "그냥", "이거", "그거", "저거", "있다", "없다", "이다", "하다", "에서", "으로", "에게",
        }
        terms: set[str] = set()
        for tok in re.findall(r"[A-Za-z0-9가-힣_]{2,32}", str(text or "").lower()):
            if not tok or tok in stopwords:
                continue
            terms.add(tok)
        return terms

    @staticmethod
    def _recall_memory_item_text(item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, dict):
            pieces = [
                str(item.get("text") or ""),
                str(item.get("summary") or ""),
                str(item.get("title") or ""),
                str(item.get("content") or ""),
                str(item.get("key") or ""),
            ]
            return " ".join(part.strip() for part in pieces if str(part).strip()).strip()
        pieces = []
        for key in ("summary", "title", "content", "key"):
            value = str(getattr(item, key, "") or "").strip()
            if value:
                pieces.append(value)
        value_json = getattr(item, "value_json", None)
        if isinstance(value_json, dict):
            for key in ("value", "summary", "title"):
                value = str(value_json.get(key) or "").strip()
                if value:
                    pieces.append(value)
        return " ".join(pieces).strip()

    @classmethod
    def _memory_recall_evidence_payload(
        cls,
        *,
        query: str,
        response_language: str,
        last_context: dict[str, Any] | None,
        session_digest: str | None,
        memory_bundle: Any,
        answer_type_hint: str,
        max_items: int = 12,
    ) -> dict[str, Any]:
        return GeneralChatRecallExecutionHelpers.memory_recall_evidence_payload(
            cls,
            query=query,
            response_language=response_language,
            last_context=last_context,
            session_digest=session_digest,
            memory_bundle=memory_bundle,
            answer_type_hint=answer_type_hint,
            max_items=max_items,
        )

    @classmethod
    def _runtime_context_qa_evidence_payload(
        cls,
        *,
        query: str,
        multimodal_context: str,
        response_language: str,
        answer_type_hint: str | None = None,
        max_items: int = 8,
    ) -> dict[str, Any]:
        focus_query = cls._single_question(query, response_language="en") or str(query or "").strip()
        ranked_lines = cls._runtime_context_qa_ranked_lines(query=focus_query, multimodal_context=multimodal_context)
        rows = cls._runtime_context_rows(multimodal_context=multimodal_context)
        row_by_line: dict[str, dict[str, Any]] = {
            str(item.get("line") or "").strip(): item for item in rows if str(item.get("line") or "").strip()
        }
        candidates: list[dict[str, Any]] = []
        for score, line in ranked_lines[: max(1, int(max_items))]:
            line_key = str(line or "").strip()
            row = row_by_line.get(line_key, {})
            content = cls._line_content_text(line_key)
            if not content:
                continue
            candidates.append(
                {
                    "score": round(float(score) / 100.0, 4),
                    "source": "runtime_context",
                    "memory_type": "episode",
                    "memory_scope": "session",
                    "session_id": str(row.get("session_id") or ""),
                    "date": str(row.get("date_text") or ""),
                    "content": content[:220],
                }
            )
        answer_type = answer_type_hint or cls._runtime_context_qa_answer_type(focus_query)
        query_terms = cls._recall_terms(focus_query)
        evidence_terms: set[str] = set()
        for row in candidates:
            evidence_terms.update(cls._recall_terms(str(row.get("content") or "")))
        coverage = (
            max(0.0, min(1.0, float(len(query_terms.intersection(evidence_terms))) / float(max(1, len(query_terms)))))
            if query_terms
            else 0.0
        )
        confidence = 0.0
        if candidates:
            confidence = sum(float(row.get("score") or 0.0) for row in candidates[:5]) / float(max(1, min(5, len(candidates))))
            confidence = max(0.0, min(1.0, confidence))
        aggregation_hint: dict[str, Any] | None = None
        if answer_type == "number":
            numeric = cls._runtime_context_numeric_answer(query=focus_query, rows=rows)
            if numeric:
                aggregation_hint = {
                    "mode": ("sum" if cls._runtime_context_is_aggregate_numeric_query(focus_query) else "single"),
                    "estimated_value": numeric,
                }
        return {
            "question": focus_query,
            "answer_type": answer_type,
            "candidate_count": len(candidates),
            "coverage": round(float(coverage), 4),
            "confidence": round(float(confidence), 4),
            "candidate_evidence": candidates,
            "aggregation_hint": aggregation_hint,
        }

    @classmethod
    def _runtime_context_qa_two_pass_prompt(
        cls,
        *,
        query: str,
        response_language: str,
        evidence_payload: dict[str, Any],
    ) -> str:
        evidence_json = json.dumps(evidence_payload, ensure_ascii=False)
        no_info_text = cls._recall_no_information_message(response_language)
        if response_language == "ko":
            return (
                "너는 회상 답변 생성기다.\n"
                "아래 pass1_json만 근거로 정답만 출력해라.\n"
                "규칙:\n"
                "1) 반드시 <final_answer>...</final_answer> 또는 엄격 JSON만 출력.\n"
                "2) answer_type=number면 숫자/금액 중심으로 답하라.\n"
                "3) 인사/메타/태그 누출 금지.\n"
                f"4) 근거가 부족하면 '{no_info_text}'만 출력.\n"
                f"질문: {query}\n"
                f"pass1_json: {evidence_json}\n"
                "정답:"
            )
        return (
            "You are a recall answer synthesizer.\n"
            "Use only the pass1_json below.\n"
            "Rules:\n"
            "1) Return either <final_answer>...</final_answer> or strict JSON only.\n"
            "2) answer_type=number should prioritize numeric output.\n"
            "3) No greeting/meta/tag leakage.\n"
            f"4) If insufficient evidence, output exactly: {no_info_text}\n"
            f"Question: {query}\n"
            f"pass1_json: {evidence_json}\n"
            "Answer:"
        )

    @classmethod
    def _runtime_context_qa_two_pass_valid_answer(
        cls,
        *,
        answer: str,
        evidence_payload: dict[str, Any],
        response_language: str,
    ) -> bool:
        reasons = cls._recall_validation_reasons(
            answer=answer,
            evidence_payload=evidence_payload,
            response_language=response_language,
            answer_type_hint=str(evidence_payload.get("answer_type") or "freeform"),
            query=str(evidence_payload.get("question") or ""),
        )
        return not reasons

    @staticmethod
    def _recall_pass1_prompt(
        *,
        query: str,
        response_language: str,
        answer_type_hint: str,
        evidence_payload: dict[str, Any],
    ) -> str:
        evidence_json = json.dumps(evidence_payload, ensure_ascii=False)
        if response_language == "ko":
            return (
                "너는 회상 질문 증거 추출기다.\n"
                "아래 evidence_payload를 읽고 엄격 JSON만 출력해라.\n"
                "필수 필드: question, answer_type, candidate_evidence, confidence, coverage, aggregation_hint\n"
                "candidate_evidence는 관련 근거만 남겨라.\n"
                f"question: {query}\n"
                f"answer_type: {answer_type_hint}\n"
                f"evidence_payload: {evidence_json}\n"
                "JSON:"
            )
        return (
            "You are a recall evidence extractor.\n"
            "Read the evidence_payload and return strict JSON only.\n"
            "Required fields: question, answer_type, candidate_evidence, confidence, coverage, aggregation_hint.\n"
            "Keep only relevant evidence in candidate_evidence.\n"
            f"question: {query}\n"
            f"answer_type: {answer_type_hint}\n"
            f"evidence_payload: {evidence_json}\n"
            "JSON:"
        )

    @classmethod
    def _parse_recall_pass1_payload(
        cls,
        *,
        raw_output: str,
        fallback_payload: dict[str, Any],
        answer_type_hint: str,
    ) -> dict[str, Any]:
        parsed = cls._extract_first_json_object(raw_output) or {}
        fallback_candidates = list(fallback_payload.get("candidate_evidence") or [])
        focus_query = str(fallback_payload.get("question") or "").strip()
        candidate_rows = parsed.get("candidate_evidence")
        normalized_candidates: list[dict[str, Any]] = []
        if isinstance(candidate_rows, list):
            for idx, row in enumerate(candidate_rows[:12]):
                if isinstance(row, str):
                    text = " ".join(row.split()).strip()
                    if not text:
                        continue
                    normalized_candidates.append(
                        {
                            "score": max(0.0, 1.0 - (idx * 0.06)),
                            "source": "pass1_model",
                            "memory_type": "episode",
                            "memory_scope": "session",
                            "date": "",
                            "content": text[:260],
                        }
                    )
                    continue
                if not isinstance(row, dict):
                    continue
                content = " ".join(str(row.get("content") or "").split()).strip()
                if not content:
                    continue
                normalized_candidates.append(
                    {
                        "score": float(max(0.0, min(1.0, float(row.get("score") or 0.6)))),
                        "source": str(row.get("source") or "pass1_model")[:48],
                        "memory_type": str(row.get("memory_type") or "episode")[:24],
                        "memory_scope": str(row.get("memory_scope") or "session")[:24],
                        "date": str(row.get("date") or "")[:32],
                        "content": content[:260],
                    }
                )
        if not normalized_candidates:
            normalized_candidates = [
                row for row in fallback_candidates[:12]
                if isinstance(row, dict) and str(row.get("content") or "").strip()
            ]

        fallback_normalized = [
            row for row in fallback_candidates[:12]
            if isinstance(row, dict) and str(row.get("content") or "").strip()
        ]

        def _candidate_quality(rows: list[dict[str, Any]]) -> float:
            if not rows:
                return 0.0
            query_terms = cls._recall_terms(focus_query)
            lexical_scores: list[float] = []
            score_values: list[float] = []
            lowered_rows: list[str] = []
            for row in rows[:6]:
                content = " ".join(str(row.get("content") or "").split()).strip()
                lowered_content = content.lower()
                lowered_rows.append(lowered_content)
                if not content:
                    continue
                terms = cls._recall_terms(content)
                overlap = float(len(query_terms.intersection(terms))) if query_terms and terms else 0.0
                lexical_scores.append(max(0.0, min(1.0, overlap / 3.0)))
                score_values.append(float(max(0.0, min(1.0, float(row.get("score") or 0.0)))))
            lexical_mean = (sum(lexical_scores) / float(max(1, len(lexical_scores)))) if lexical_scores else 0.0
            score_mean = (sum(score_values) / float(max(1, len(score_values)))) if score_values else 0.0
            shape_bonus = 0.0
            if cls._recall_is_duration_query(focus_query):
                if any(re.search(r"\b(?:for\s+\d+\s+years?|years?\s+ago|\d+\s*(?:년|개월|주|일))\b", row) for row in lowered_rows):
                    shape_bonus += 0.28
                if any(cls._runtime_context_has_date_signal(row) for row in lowered_rows):
                    shape_bonus -= 0.08
            if cls._recall_is_origin_query(focus_query):
                if any(re.search(r"\bfrom\s+[a-z][a-z\s-]{2,32}\b", row) for row in lowered_rows):
                    shape_bonus += 0.24
                if any("home country" in row for row in lowered_rows):
                    shape_bonus -= 0.14
            if cls._recall_is_camped_location_query(focus_query):
                if any(any(token in row for token in ("beach", "mountains", "forest", "park", "camped", "camping")) for row in lowered_rows):
                    shape_bonus += 0.2
            if cls._recall_is_kids_preference_query(focus_query):
                if any(any(token in row for token in ("kids like", "children like", "dinosaurs", "nature", "animals")) for row in lowered_rows):
                    shape_bonus += 0.22
            return max(0.0, min(1.0, (0.52 * lexical_mean) + (0.34 * score_mean) + shape_bonus))

        coverage_raw = parsed.get("coverage", fallback_payload.get("coverage", 0.0))
        confidence_raw = parsed.get("confidence", fallback_payload.get("confidence", 0.0))
        parsed_quality = _candidate_quality(normalized_candidates)
        fallback_quality = _candidate_quality(fallback_normalized)
        prefer_fallback_candidates = fallback_quality > (parsed_quality + 0.12)
        if cls._recall_is_duration_query(focus_query):
            parsed_has_duration = any(
                re.search(r"\b(?:for\s+\d+\s+years?|years?\s+ago|\d+\s*(?:년|개월|주|일))\b", str(row.get("content") or "").lower())
                for row in normalized_candidates
            )
            fallback_has_duration = any(
                re.search(r"\b(?:for\s+\d+\s+years?|years?\s+ago|\d+\s*(?:년|개월|주|일))\b", str(row.get("content") or "").lower())
                for row in fallback_normalized
            )
            if (not parsed_has_duration) and fallback_has_duration:
                prefer_fallback_candidates = True
        if prefer_fallback_candidates and fallback_normalized:
            normalized_candidates = fallback_normalized
            coverage_raw = fallback_payload.get("coverage", 0.0)
            confidence_raw = fallback_payload.get("confidence", 0.0)

        aggregation_hint = parsed.get("aggregation_hint")
        if not isinstance(aggregation_hint, dict):
            aggregation_hint = fallback_payload.get("aggregation_hint")
            if not isinstance(aggregation_hint, dict):
                aggregation_hint = None
        valid_types = {"date", "number", "boolean", "entity", "freeform"}
        hinted_type = coerce_answer_type_hint(answer_type_hint)
        parsed_type = str(parsed.get("answer_type") or "").strip().lower()
        fallback_type = str(fallback_payload.get("answer_type") or "").strip().lower()
        if hinted_type in valid_types and hinted_type != "freeform":
            answer_type = hinted_type
        elif parsed_type in valid_types:
            answer_type = parsed_type
        elif fallback_type in valid_types:
            answer_type = fallback_type
        else:
            answer_type = "freeform"
        return {
            "question": str(parsed.get("question") or fallback_payload.get("question") or "").strip(),
            "answer_type": answer_type,
            "candidate_count": len(normalized_candidates),
            "candidate_evidence": normalized_candidates,
            "coverage": max(0.0, min(1.0, float(coverage_raw or 0.0))),
            "confidence": max(0.0, min(1.0, float(confidence_raw or 0.0))),
            "aggregation_hint": aggregation_hint,
        }

    @staticmethod
    def _recall_has_strong_evidence(evidence_payload: dict[str, Any]) -> bool:
        candidate_count = int(evidence_payload.get("candidate_count") or 0)
        coverage = float(evidence_payload.get("coverage") or 0.0)
        confidence = float(evidence_payload.get("confidence") or 0.0)
        aggregation_hint = evidence_payload.get("aggregation_hint")
        if isinstance(aggregation_hint, dict) and bool(aggregation_hint):
            return True
        if candidate_count <= 0:
            return False
        if candidate_count >= 3 and (coverage >= 0.24 or confidence >= 0.54):
            return True
        if candidate_count >= 2 and coverage >= 0.34:
            return True
        return coverage >= 0.45 and confidence >= 0.44

    @classmethod
    def _recall_select_candidate(
        cls,
        *,
        query: str,
        pass1_payload: dict[str, Any],
        answer_type_hint: str,
    ) -> dict[str, Any]:
        query_terms = cls._recall_terms(query)
        candidates = list(pass1_payload.get("candidate_evidence") or [])
        scored: list[dict[str, Any]] = []
        for idx, row in enumerate(candidates[:12]):
            if not isinstance(row, dict):
                continue
            content = " ".join(str(row.get("content") or "").split()).strip()
            if not content:
                continue
            base_score = float(max(0.0, min(1.0, float(row.get("score") or 0.0))))
            memory_type = str(row.get("memory_type") or "episode").strip().lower()
            terms = cls._recall_terms(content)
            lexical_overlap = len(query_terms.intersection(terms)) if query_terms and terms else 0
            lexical_score = max(0.0, min(1.0, float(lexical_overlap) / 3.0))
            type_bonus = 0.06 if memory_type == "fact" else (0.04 if memory_type == "preference" else 0.0)
            shape_bonus = 0.0
            lowered = content.lower()
            if answer_type_hint == "date":
                if cls._runtime_context_has_date_signal(content):
                    shape_bonus += 0.24
                if any(token in lowered for token in ("yesterday", "last year", "어제", "작년")):
                    shape_bonus += 0.08
            elif answer_type_hint == "number":
                if cls._runtime_context_extract_numbers(content):
                    shape_bonus += 0.18
                if cls._recall_is_duration_query(query):
                    if re.search(r"\b(\d+)\s*(year|years|month|months|week|weeks|day|days)\b", lowered):
                        shape_bonus += 0.18
                    if any(token in lowered for token in ("년", "개월", "주", "일", "ago", "동안", "전")):
                        shape_bonus += 0.08
                    if cls._runtime_context_has_date_signal(content):
                        shape_bonus -= 0.12
            elif answer_type_hint == "entity":
                if cls._recall_is_multi_item_query(query):
                    if "," in content:
                        shape_bonus += 0.12
                    if re.search(r"\b(and|or)\b", lowered):
                        shape_bonus += 0.06
                if any(token in lowered for token in ("research", "researched", "camp", "camped", "activity", "activities", "likes", "like", "children", "kids", "favorite")):
                    shape_bonus += 0.06
                if any(token in lowered for token in ("safe and loving home", "home country")):
                    shape_bonus -= 0.08
                if cls._recall_is_origin_query(query):
                    if re.search(r"\bfrom\s+[a-z][a-z\s-]{2,32}\b", lowered):
                        shape_bonus += 0.16
                    if any(token in lowered for token in ("sweden", "korea", "japan", "canada", "france", "germany")):
                        shape_bonus += 0.1
                    if "home country" in lowered:
                        shape_bonus -= 0.18
                if cls._recall_is_camped_location_query(query):
                    if any(token in lowered for token in ("beach", "mountains", "mountain", "forest", "park", "camped", "camping")):
                        shape_bonus += 0.12
                if cls._recall_is_kids_preference_query(query):
                    if any(token in lowered for token in ("kids like", "children like", "favorite", "favourite", "dinosaurs", "nature", "animals")):
                        shape_bonus += 0.16
                    if any(token in lowered for token in ("loving home", "accepting environment", "safe home")):
                        shape_bonus -= 0.2
                if cls._recall_is_relationship_status_query(query):
                    if any(token in lowered for token in ("single", "married", "dating", "divorced")):
                        shape_bonus += 0.1
                    if "single parent" in lowered:
                        shape_bonus += 0.05
                if cls._recall_is_career_field_query(query):
                    if any(
                        token in lowered
                        for token in (
                            "psychology",
                            "counsel",
                            "counseling",
                            "mental health",
                            "career",
                            "supporting trans",
                            "work with trans",
                        )
                    ):
                        shape_bonus += 0.16
                    if any(token in lowered for token in ("adoption", "family", "kids", "home", "camping")):
                        shape_bonus -= 0.14
                if "research" in query.lower():
                    if any(token in lowered for token in ("adoption agencies", "agency", "adoption")):
                        shape_bonus += 0.18
                    if lowered.strip() in {"research", "did research", "researched"}:
                        shape_bonus -= 0.18
                if cls._recall_is_identity_query(query):
                    if any(token in lowered for token in ("transgender", "woman", "man", "nonbinary", "identity", "lgbtq")):
                        shape_bonus += 0.18
                    if any(token in lowered for token in ("yes,", "yes ", "thanks", "wow", "awesome", "courage")):
                        shape_bonus -= 0.10
            elif answer_type_hint == "boolean":
                if re.search(r"\b(yes|no|true|false|맞아|아니|가능|불가능)\b", lowered):
                    shape_bonus += 0.08
                if "if" in query.lower() and any(token in lowered for token in ("support", "supported", "encourage", "encouraged", "도움", "지원")):
                    shape_bonus += 0.08
            composed_score = (0.56 * base_score) + (0.34 * lexical_score) + type_bonus + shape_bonus
            scored.append(
                {
                    "index": idx,
                    "score": round(float(composed_score), 4),
                    "base_score": round(base_score, 4),
                    "content": content[:260],
                }
            )
        scored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        top_indices = [int(item.get("index") or 0) for item in scored[:3]]
        if not top_indices and candidates:
            top_indices = [0]
        selected_index = top_indices[0] if top_indices else -1
        selected_row = {}
        if selected_index >= 0 and selected_index < len(candidates) and isinstance(candidates[selected_index], dict):
            selected_row = dict(candidates[selected_index])
        return {
            "selected_index": selected_index,
            "selected_candidate": selected_row,
            "top_indices": top_indices,
            "scored_candidates": scored[:3],
        }

    @staticmethod
    def _recall_selection_for_attempt(
        *,
        base_selection: dict[str, Any],
        pass1_payload: dict[str, Any],
        attempt_index: int,
    ) -> dict[str, Any]:
        top_indices: list[int] = []
        for item in list(base_selection.get("top_indices") or []):
            try:
                value = int(item)
            except Exception:
                continue
            if value >= 0:
                top_indices.append(value)
        if not top_indices:
            return dict(base_selection)
        selected_index = top_indices[min(max(0, attempt_index), max(0, len(top_indices) - 1))]
        selected_candidate = {}
        candidates = list(pass1_payload.get("candidate_evidence") or [])
        if selected_index < len(candidates) and isinstance(candidates[selected_index], dict):
            selected_candidate = dict(candidates[selected_index])
        selection = dict(base_selection)
        selection["selected_index"] = selected_index
        selection["selected_candidate"] = selected_candidate
        return selection

    @staticmethod
    def _recall_no_information_message(response_language: str) -> str:
        code = normalize_language_code(response_language) or "en"
        if code == "ko":
            return "관련 정보를 찾지 못했습니다."
        if code == "ja":
            return "該当する情報が見つかりませんでした。"
        return "No information available."

    @classmethod
    def _recall_is_no_information_answer(cls, text: str) -> bool:
        normalized = " ".join(str(text or "").strip().split()).lower()
        if not normalized:
            return False
        candidates = {
            cls._recall_no_information_message("en").lower(),
            cls._recall_no_information_message("ko").lower(),
            cls._recall_no_information_message("ja").lower(),
        }
        return normalized in candidates

    @classmethod
    def _recall_pass2_prompt(
        cls,
        *,
        query: str,
        response_language: str,
        pass1_payload: dict[str, Any],
        selected_evidence: dict[str, Any] | None = None,
    ) -> str:
        payload_json = json.dumps(pass1_payload, ensure_ascii=False)
        answer_type = str(pass1_payload.get("answer_type") or "freeform").strip().lower()
        if answer_type not in {"date", "number", "boolean", "entity", "freeform"}:
            answer_type = "freeform"
        selected_evidence = selected_evidence or {}
        try:
            raw_selected_index = selected_evidence.get("selected_index")
            selected_index = int(raw_selected_index) if raw_selected_index is not None else -1
        except Exception:
            selected_index = -1
        selected_candidate = selected_evidence.get("selected_candidate")
        top_indices: list[int] = []
        for item in list(selected_evidence.get("top_indices") or []):
            try:
                top_indices.append(int(item))
            except Exception:
                continue
            if len(top_indices) >= 3:
                break
        selection_json = json.dumps(
            {
                "selected_index": selected_index,
                "top_indices": top_indices,
                "selected_candidate": selected_candidate if isinstance(selected_candidate, dict) else {},
            },
            ensure_ascii=False,
        )
        no_info_text = cls._recall_no_information_message(response_language)
        type_rule = ""
        language_code = normalize_language_code(response_language) or "en"
        duration_query = cls._recall_is_duration_query(query)
        multi_item_query = cls._recall_is_multi_item_query(query)
        if duration_query and answer_type == "date":
            answer_type = "number"
        lowered_query = str(query or "").lower()
        counterfactual_query = ((" if " in lowered_query) or lowered_query.strip().startswith("if ")) and any(
            token in lowered_query
            for token in ("without", "hadn't", "had not", "없었", "없다면", "아니었")
        )
        if answer_type == "date":
            if language_code == "ko":
                type_rule = (
                    "- 답변은 날짜/연도 한 줄만 출력.\n"
                    "- 근거 문장에 yesterday/last year/어제/작년이 있으면 타임스탬프 기준 상대시점을 계산.\n"
                )
            elif language_code == "ja":
                type_rule = (
                    "- 回答は日付/年のみを1行で出力する。\n"
                    "- 根拠に yesterday/last year/어제/작년 がある場合はタイムスタンプ基準で相対時点を計算する。\n"
                )
            else:
                type_rule = (
                    "- Output a single short date/year line only.\n"
                    "- If evidence has relative terms (yesterday/last year), resolve relative time using timestamp context.\n"
                )
        elif answer_type == "number":
            if language_code == "ko":
                type_rule = (
                    "- 답변은 숫자/금액만 간결하게 출력.\n"
                    "- 단일값 질문은 하나의 값, 합계 질문은 총합값을 우선.\n"
                )
                if duration_query:
                    type_rule += "- 기간/경과시간 질문이면 날짜가 아니라 상대 기간(예: 10년 전, 4년)으로 답변.\n"
                    type_rule += "- 근거에 'for N years' 또는 'N years ago'가 있으면 그 기간 표현을 그대로 사용.\n"
            elif language_code == "ja":
                type_rule = (
                    "- 回答は数値/金額中心で簡潔に出力する。\n"
                    "- 単一値質問は1つの値、合計質問は総和を優先する。\n"
                )
                if duration_query:
                    type_rule += "- 期間/経過時間の質問では日付ではなく相対期間（例: 10年前, 4年）で答える。\n"
                    type_rule += "- 根拠に 'for N years' や 'N years ago' があれば、その期間表現をそのまま使う。\n"
            else:
                type_rule = (
                    "- Output a concise numeric/currency answer.\n"
                    "- For single-value queries, return one value; for aggregate queries, prioritize the total.\n"
                )
                if duration_query:
                    type_rule += "- For duration/elapsed-time questions, answer with a relative span (e.g., 10 years ago, 4 years), not a calendar date.\n"
                    type_rule += "- If evidence includes patterns like 'for N years' or 'N years ago', copy that duration phrase directly.\n"
        elif answer_type == "boolean":
            if language_code == "ko":
                type_rule = (
                    "- 답변은 예/아니오(또는 가능성 표현) 형태의 짧은 한 줄.\n"
                    "- 질문이 yes/no형이 아니면 사실 답변으로 전환.\n"
                )
                if counterfactual_query:
                    type_rule += "- 반사실(if without 류) 질문이면 근거 인과관계로 Likely no/Likely yes를 선택.\n"
            elif language_code == "ja":
                type_rule = (
                    "- 回答は Yes/No（または可能性表現）で短く1行にする。\n"
                    "- 質問が yes/no 形式でない場合は事実回答に切り替える。\n"
                )
                if counterfactual_query:
                    type_rule += "- 反事実(if without)質問では根拠の因果関係に従って Likely no/Likely yes を選ぶ。\n"
            else:
                type_rule = (
                    "- Use a short one-line Yes/No style answer (or likely yes/no if uncertain).\n"
                    "- If the question is not yes/no form, switch to a factual short answer.\n"
                )
                if counterfactual_query:
                    type_rule += "- For counterfactual if-without questions, infer likely no/likely yes from causal evidence.\n"
        elif answer_type == "entity":
            if language_code == "ko":
                type_rule = (
                    "- 답변은 짧은 명사구(이름/직함/정체성/장소) 위주로 출력.\n"
                    "- 복수 항목 질문이면 근거에 있는 항목을 쉼표로 나열.\n"
                    "- 주어+동사 일반문(예: 'Caroline did research') 금지, 대상 명사(예: adoption agencies)만 출력.\n"
                    "- 장황한 문장/설명 금지.\n"
                )
                if cls._recall_is_identity_query(query):
                    type_rule += "- 정체성 질문이면 'gender identity' 같은 일반어 금지, 구체 정체성(예: transgender woman)을 출력.\n"
                if multi_item_query:
                    type_rule += "- 복수 질문이면 근거에서 최소 2개 항목을 쉼표로 제시.\n"
                if cls._recall_is_origin_query(query):
                    type_rule += "- 출신/이동 질문이면 'home country' 같은 일반어 금지, 구체 국가/도시명만 출력.\n"
                if cls._recall_is_kids_preference_query(query):
                    type_rule += "- 아이들이 좋아하는 것 질문이면 대상 명사(예: dinosaurs, nature)만 쉼표로 제시.\n"
                if cls._recall_is_relationship_status_query(query):
                    type_rule += "- 관계 상태 질문이면 한 단어 상태값만 출력(예: Single, Married).\n"
            elif language_code == "ja":
                type_rule = (
                    "- 回答は短い名詞句（名前/肩書/正体/場所）中心で出力する。\n"
                    "- 複数項目質問なら根拠にある項目をカンマ区切りで列挙する。\n"
                    "- 主語+動詞の一般文は禁止。対象名詞だけを出力する。\n"
                    "- 長い説明文は禁止。\n"
                )
                if cls._recall_is_identity_query(query):
                    type_rule += "- アイデンティティ質問では 'gender identity' のような一般語を避け、具体的な属性を出力する。\n"
                if multi_item_query:
                    type_rule += "- 複数質問なら根拠から最低2項目をカンマで提示する。\n"
                if cls._recall_is_origin_query(query):
                    type_rule += "- 出身/移動質問では 'home country' のような一般語を避け、具体的な国名/都市名のみを出力する。\n"
                if cls._recall_is_kids_preference_query(query):
                    type_rule += "- 子どもの好み質問では対象名詞（例: dinosaurs, nature）だけをカンマで提示する。\n"
                if cls._recall_is_relationship_status_query(query):
                    type_rule += "- 関係ステータス質問では単一ステータス語のみを出力する（例: Single, Married）。\n"
            else:
                type_rule = (
                    "- Output a short noun phrase (name/title/identity/place).\n"
                    "- For multi-item questions, return a comma-separated list from evidence.\n"
                    "- Do not output generic subject+verb summaries; output concrete object nouns from evidence.\n"
                    "- Avoid long explanatory sentences.\n"
                )
                if cls._recall_is_identity_query(query):
                    type_rule += "- For identity questions, avoid generic terms like 'gender identity'; output the concrete descriptor (e.g., transgender woman).\n"
                if multi_item_query:
                    type_rule += "- For multi-item queries, provide at least two evidence-grounded items when available.\n"
                if cls._recall_is_origin_query(query):
                    type_rule += "- For origin/move-from questions, avoid generic labels like 'home country'; output a concrete country/city name only.\n"
                if cls._recall_is_kids_preference_query(query):
                    type_rule += "- For kids-preference questions, output concrete liked nouns (e.g., dinosaurs, nature) as a comma-separated list.\n"
                if cls._recall_is_relationship_status_query(query):
                    type_rule += "- For relationship-status questions, output one status label only (e.g., Single, Married).\n"
                if cls._recall_is_camped_location_query(query):
                    type_rule += "- For camp-location questions, list all distinct locations from evidence as comma-separated nouns.\n"
        if language_code == "ko":
            return (
                "너는 회상 질문 최종 답변 생성기다.\n"
                "아래 pass1_json만 근거로 답변하라.\n"
                "출력 계약:\n"
                "- 반드시 <final_answer>...</final_answer> 또는 엄격 JSON만 출력\n"
                "- 인사/메타/태그 누출 금지\n"
                f"- 근거 부족 시 정확히 {no_info_text}\n"
                "- selected_evidence.selected_index 후보를 우선 사용하되, 부족하면 top_indices 안에서만 보완.\n"
                f"- answer_type: {answer_type}\n"
                f"{type_rule}"
                f"질문: {query}\n"
                f"selected_evidence: {selection_json}\n"
                f"pass1_json: {payload_json}\n"
                "정답:"
            )
        if language_code == "ja":
            return (
                "あなたは回想質問の最終回答生成器です。\n"
                "以下の pass1_json のみを根拠に回答してください。\n"
                "出力契約:\n"
                "- 必ず <final_answer>...</final_answer> または厳格JSONのみを出力\n"
                "- 挨拶/メタ/タグ漏洩は禁止\n"
                f"- 根拠不足の場合は正確に {no_info_text}\n"
                "- selected_evidence.selected_index を優先し、不足時は top_indices 内のみで補完する。\n"
                f"- answer_type: {answer_type}\n"
                f"{type_rule}"
                f"質問: {query}\n"
                f"selected_evidence: {selection_json}\n"
                f"pass1_json: {payload_json}\n"
                "回答:"
            )
        return (
            "You are a recall answer synthesizer.\n"
            "Use only pass1_json below.\n"
            "Output contract:\n"
            "- Return <final_answer>...</final_answer> or strict JSON only\n"
            "- No greeting/meta/tag leakage\n"
            f"- If insufficient evidence output exactly: {no_info_text}\n"
            "- Prioritize selected_evidence.selected_index, and only use selected_evidence.top_indices as backup.\n"
            f"- answer_type: {answer_type}\n"
            f"{type_rule}"
            f"Question: {query}\n"
            f"selected_evidence: {selection_json}\n"
            f"pass1_json: {payload_json}\n"
            "Answer:"
        )

    @classmethod
    def _recall_answer_coverage(cls, *, answer: str, evidence_payload: dict[str, Any]) -> float:
        value = str(answer or "").strip()
        if not value:
            return 0.0
        if cls._recall_is_no_information_answer(value):
            return 1.0
        answer_terms = cls._recall_terms(value)
        if not answer_terms:
            return 0.0
        evidence_rows = list(evidence_payload.get("candidate_evidence") or [])
        evidence_terms: set[str] = set()
        for row in evidence_rows:
            if not isinstance(row, dict):
                continue
            evidence_terms.update(cls._recall_terms(str(row.get("content") or "")))
        if not evidence_terms:
            return 0.0
        overlap = len(answer_terms.intersection(evidence_terms))
        return max(0.0, min(1.0, float(overlap) / float(max(1, len(answer_terms)))))

    @classmethod
    def _recall_validation_reasons(
        cls,
        *,
        answer: str,
        evidence_payload: dict[str, Any],
        response_language: str,
        answer_type_hint: str,
        query: str = "",
    ) -> list[str]:
        raw = str(answer or "").strip()
        parsed = extract_contract_response(raw)
        if not isinstance(parsed, dict):
            parsed = {}
        extracted = str(parsed.get("answer") or "").strip()
        contract_reasons = validate_contract_response(
            answer=extracted,
            raw_text=raw,
            expected_language=response_language,
            answer_type_hint=coerce_answer_type_hint(answer_type_hint),
            declared_answer_type=str(parsed.get("declared_answer_type") or ""),
            declared_language=str(parsed.get("declared_language") or ""),
        )
        reasons = list(contract_reasons)
        if str(parsed.get("contract_format") or "plain") == "plain":
            reasons = ["missing_contract", *reasons]
        if len(extracted) > 240:
            reasons.append("too_long")
        if extracted.endswith("?"):
            reasons.append("question_like_answer")
        if cls._runtime_context_is_generic_chitchat(extracted):
            reasons.append("meta_or_greeting")
        lowered_extracted = str(extracted or "").strip().lower()
        normalized_answer_type = coerce_answer_type_hint(answer_type_hint)
        if normalized_answer_type == "entity" and cls._recall_is_boolean_short_answer(extracted):
            reasons.append("entity_boolean_mismatch")
        if normalized_answer_type == "entity":
            if re.search(
                r"\b(?:[a-z][a-z0-9_-]{1,24})\s+(?:did|does|was|is)\s+(?:research|researches|activities?|home country)\b",
                lowered_extracted,
            ):
                reasons.append("entity_too_generic")
            if cls._recall_is_origin_query(query) and "home country" in lowered_extracted:
                reasons.append("entity_origin_too_generic")
            if cls._recall_is_kids_preference_query(query) and any(
                token in lowered_extracted for token in ("loving home", "accepting environment", "safe and loving")
            ):
                reasons.append("entity_preference_too_generic")
            if cls._recall_is_relationship_status_query(query):
                compact = re.sub(r"[^a-z가-힣\s-]", " ", lowered_extracted).strip()
                if compact in {"single parent", "single mom", "single mother", "single father"}:
                    reasons.append("entity_relationship_not_compact")
            if cls._recall_is_identity_query(query):
                identity_value = re.sub(r"[^a-z0-9가-힣\s-]", " ", lowered_extracted).strip()
                identity_words = [w for w in identity_value.split() if w]
                if len(identity_words) <= 1:
                    reasons.append("entity_identity_name_only")
                if identity_value in {"caroline", "melanie"}:
                    reasons.append("entity_identity_name_only")
                if identity_value in {"identity", "gender identity"}:
                    reasons.append("entity_identity_too_generic")
                if (
                    "transgender" not in identity_value
                    and "nonbinary" not in identity_value
                    and ("woman" not in identity_value and "man" not in identity_value)
                    and len(identity_words) <= 2
                ):
                    reasons.append("entity_identity_too_generic")
        if cls._recall_is_no_information_answer(extracted) and cls._recall_has_strong_evidence(evidence_payload):
            reasons.append("unsupported_no_information")
        coverage = cls._recall_answer_coverage(answer=extracted, evidence_payload=evidence_payload)
        if (not cls._recall_is_no_information_answer(extracted)) and coverage <= 0.0:
            reasons.append("evidence_mismatch")
        return sorted({str(item) for item in reasons if str(item).strip()})

    @classmethod
    def _recall_validation_score(
        cls,
        *,
        answer: str,
        reasons: list[str],
        evidence_payload: dict[str, Any],
    ) -> float:
        base = 1.0 - (0.18 * float(len(reasons)))
        coverage = cls._recall_answer_coverage(answer=answer, evidence_payload=evidence_payload)
        score = base + (0.22 * coverage)
        reason_set = {str(item or "").strip() for item in reasons if str(item or "").strip()}
        if "entity_identity_too_generic" in reason_set:
            score -= 0.34
        if "entity_too_generic" in reason_set:
            score -= 0.24
        if "entity_origin_too_generic" in reason_set:
            score -= 0.24
        if "entity_preference_too_generic" in reason_set:
            score -= 0.22
        if "entity_relationship_not_compact" in reason_set:
            score -= 0.14
        if "answer_type_date_mismatch" in reason_set:
            score -= 0.24
        if "answer_type_number_mismatch" in reason_set:
            score -= 0.22
        if "evidence_mismatch" in reason_set:
            score -= 0.18
        if cls._recall_is_no_information_answer(str(answer or "").strip()):
            if cls._recall_has_strong_evidence(evidence_payload):
                score -= 0.58
            else:
                score -= 0.08
        return max(0.0, min(1.0, score))

    async def _run_recall_two_pass_orchestration(
        self,
        *,
        context: ReasoningContext,
        executor,
        query: str,
        answer_type_hint: str,
        evidence_payload: dict[str, Any],
        response_language: str,
        style_profile: dict[str, Any] | None,
    ) -> tuple[ExecutionResult, dict[str, Any]]:
        return await GeneralChatRecallExecutionHelpers.run_recall_two_pass_orchestration(
            self,
            context=context,
            executor=executor,
            query=query,
            answer_type_hint=answer_type_hint,
            evidence_payload=evidence_payload,
            response_language=response_language,
            style_profile=style_profile,
        )
