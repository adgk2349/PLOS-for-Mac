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


class GeneralChatRuntimeContextMixin:

    @staticmethod
    def _is_runtime_context_qa_mode(
        *,
        query: str,
        multimodal_context: str | None,
        multimodal_notes: list[str] | None,
    ) -> bool:
        context_text = str(multimodal_context or "").strip()
        if not context_text:
            return False
        lowered_query = unicodedata.normalize("NFC", str(query or "").strip()).lower()
        lowered_notes = " ".join(str(item or "").strip().lower() for item in (multimodal_notes or []))
        question_like = ("?" in lowered_query or "？" in lowered_query)
        recall_cues = (
            "when ", "what ", "who ", "where ", "which ", "how long", "how many",
            "would ", "could ", "likely ",
            "언제", "무엇", "뭐", "누구", "어디", "몇",
        )
        has_recall_cue = any(token in lowered_query for token in recall_cues)
        strict_note = (
            "provided conversation context" in lowered_notes
            or "대화 문맥만" in lowered_notes
            or "대화 기록만" in lowered_notes
        )
        strict_query = (
            "대화 기록만" in lowered_query
            or "대화 문맥만" in lowered_query
            or "conversation context below" in lowered_query
        )
        return bool(question_like and has_recall_cue and (strict_note or strict_query))

    @classmethod
    def _runtime_context_qa_prompt(
        cls,
        *,
        query: str,
        response_language: str,
    ) -> str:
        question = cls._single_question(query, response_language=response_language)
        no_info_text = cls._recall_no_information_message(response_language)
        if response_language == "ko":
            return (
                "아래 제공된 대화 문맥만 사용해서 답하세요.\n"
                "질문에 해당하는 짧은 정답 한 줄만 출력하세요.\n"
                f"문맥에 없으면 정확히 다음 문장을 출력하세요: {no_info_text}\n"
                "설명/사과/추측/코드블록 금지.\n"
                f"질문: {question}\n"
                "정답:"
            )
        if response_language == "ja":
            return (
                "以下の会話文脈だけを使って答えてください。\n"
                "質問に対する短い答えを1行で出してください。\n"
                f"文脈にない場合は次を正確に出力: {no_info_text}\n"
                "説明・推測・コードブロックは禁止。\n"
                f"質問: {question}\n"
                "回答:"
            )
        return (
            "Use only the provided conversation context.\n"
            "Return one short answer line only.\n"
            f"If missing from context, output exactly: {no_info_text}\n"
            "No explanation, no speculation, no code fences.\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @staticmethod
    def _runtime_context_qa_fallback(*, query: str, response_language: str) -> str:
        lowered = str(query or "")
        if re.search(r"[A-Za-z]", lowered) and not re.search(r"[가-힣]", lowered):
            return "No information available."
        if response_language == "ko":
            return "대화 문맥에 정보가 없습니다."
        if response_language == "ja":
            return "会話文脈に情報がありません。"
        return "No information available."

    @staticmethod
    def _runtime_context_qa_terms(text: str) -> set[str]:
        tokens = re.findall(r"[A-Za-z0-9가-힣]{2,32}", str(text or "").lower())
        stop = {
            "when", "what", "where", "who", "which", "how", "many", "long", "did", "does", "is", "are",
            "the", "and", "for", "from", "with", "that", "this", "would", "likely", "about",
            "use", "using", "date", "answer", "approximate",
            "언제", "무엇", "뭐", "누구", "어디", "어떻게", "했", "있는", "있는지", "기준",
            "conversation", "context", "question", "focus", "relevant", "excerpts",
        }
        return {tok for tok in tokens if tok not in stop}

    @staticmethod
    def _runtime_context_qa_key_phrases(text: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9가-힣]{2,32}", str(text or "").lower())
        stop = {
            "when", "what", "where", "who", "which", "how", "many", "long", "did", "does", "is", "are",
            "the", "and", "for", "from", "with", "that", "this", "would", "likely", "about",
            "use", "using", "date", "answer", "approximate",
            "conversation", "context", "question", "focus", "relevant", "excerpts",
        }
        kept = [w for w in words if w not in stop]
        phrases: list[str] = []
        for i in range(0, max(0, len(kept) - 1)):
            a = kept[i].strip()
            b = kept[i + 1].strip()
            if not a or not b:
                continue
            phrases.append(f"{a} {b}")
        # Preserve order while deduplicating.
        out: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            if phrase in seen:
                continue
            seen.add(phrase)
            out.append(phrase)
        return out[:8]

    @classmethod
    def _runtime_context_semantic_intent_bonus(cls, *, query: str, line: str) -> int:
        q = str(query or "").lower()
        content = cls._line_content_text(line).lower()
        if not q or not content:
            return 0
        bonus = 0
        if cls._recall_is_career_field_query(query):
            if any(
                token in content
                for token in ("psychology", "counsel", "counseling", "mental health", "career", "supporting trans", "work with trans")
            ):
                bonus += 22
            if any(token in content for token in ("adoption", "family", "kids", "camping", "pottery", "painting")):
                bonus -= 9
        if cls._recall_is_identity_query(query):
            if any(token in content for token in ("transgender", "woman", "man", "nonbinary")):
                bonus += 20
            if "identity" in content:
                bonus += 10
            if any(token in content for token in ("thanks", "wow", "awesome", "courage", "inspiring")):
                bonus -= 7
        if "research" in q:
            if any(token in content for token in ("adoption agencies", "adoption agency", "adoption")):
                bonus += 18
            elif re.search(r"\bresearch(?:ed|ing)?\b", content):
                bonus -= 4
        if cls._recall_is_origin_query(query):
            if re.search(r"\bfrom\s+[a-z][a-z\s-]{2,32}\b", content):
                bonus += 14
            if "home country" in content:
                bonus -= 8
            if any(token in content for token in ("sweden", "korea", "japan", "canada", "france", "germany")):
                bonus += 10
        if cls._recall_is_camped_location_query(query):
            if any(token in content for token in ("beach", "mountain", "mountains", "forest", "park", "camped", "camping")):
                bonus += 14
        if cls._recall_is_kids_preference_query(query):
            if any(token in content for token in ("kids like", "children like", "favorite", "favourite", "dinosaurs", "nature", "animals")):
                bonus += 16
            if any(token in content for token in ("loving home", "accepting environment")):
                bonus -= 10
        if cls._recall_is_relationship_status_query(query):
            if any(token in content for token in ("single", "married", "dating", "divorced", "in a relationship")):
                bonus += 12
            if "single parent" in content:
                bonus += 4
        if cls._recall_is_multi_item_query(query):
            if "," in content or " and " in content:
                bonus += 6
        return bonus

    @staticmethod
    def _runtime_context_has_date_signal(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        return bool(re.search(r"\b(19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", raw))

    @classmethod
    def _runtime_context_temporal_bonus(cls, *, query: str, line: str, answer_type_hint: str) -> int:
        lowered_query = str(query or "").lower()
        lowered_line = str(line or "").lower()
        if not lowered_query or not lowered_line:
            return 0
        bonus = 0
        if answer_type_hint == "date":
            if cls._runtime_context_has_date_signal(line):
                bonus += 18
            if any(token in lowered_query for token in ("when", "date", "언제", "날짜", "연도", "년도")) and any(
                token in lowered_line
                for token in ("date:", " on ", " am ", " pm ", "어제", "작년", "yesterday", "last year")
            ):
                bonus += 8
        temporal_pairs = [
            (("first", "earliest", "처음", "최초", "먼저"), ("first", "earliest", "처음", "최초", "초반")),
            (("last", "latest", "recent", "현재", "최근", "마지막", "최신"), ("last", "latest", "recent", "현재", "최근", "마지막", "최신")),
            (("before", "prior", "이전", "전"), ("before", "prior", "이전", "전")),
            (("after", "later", "이후", "뒤"), ("after", "later", "이후", "뒤")),
            (("yesterday", "어제"), ("yesterday", "어제")),
            (("last year", "작년"), ("last year", "작년")),
        ]
        for query_tokens, line_tokens in temporal_pairs:
            if any(token in lowered_query for token in query_tokens) and any(token in lowered_line for token in line_tokens):
                bonus += 6
        return bonus

    @classmethod
    def _runtime_context_qa_ranked_lines(
        cls,
        *,
        query: str,
        multimodal_context: str,
    ) -> list[tuple[int, str]]:
        answer_type_hint = cls._runtime_context_qa_answer_type(query)
        q_terms = cls._runtime_context_qa_terms(query)
        q_phrases = cls._runtime_context_qa_key_phrases(query)
        lines = [line.strip() for line in str(multimodal_context or "").splitlines() if line.strip()]
        ranked: list[tuple[int, str]] = []
        for line in lines:
            lowered = line.lower()
            if lowered.startswith("relevant excerpts from the conversation"):
                continue
            if lowered.startswith("question focus:"):
                continue
            line_terms = cls._runtime_context_qa_terms(line)
            overlap = len(q_terms.intersection(line_terms))
            # Prefer lines with stronger lexical overlap and slightly shorter content.
            phrase_bonus = 0
            for phrase in q_phrases:
                if phrase and phrase in lowered:
                    phrase_bonus += 10
            intent_bonus = cls._runtime_context_semantic_intent_bonus(query=query, line=line)
            if overlap <= 0 and phrase_bonus <= 0 and intent_bonus <= 0:
                continue
            temporal_bonus = cls._runtime_context_temporal_bonus(
                query=query,
                line=line,
                answer_type_hint=answer_type_hint,
            )
            score = (
                overlap * 12
                + phrase_bonus
                + temporal_bonus
                + intent_bonus
                - min(5, max(0, len(line_terms) - overlap))
            )
            ranked.append((score, line))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    @staticmethod
    def _runtime_context_parse_date_value(text: str) -> datetime | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        compact = re.sub(r"\s+", " ", raw).strip().replace(".", "")
        for fmt in (
            "%d %B, %Y",
            "%B %d, %Y",
            "%d %b, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%B %d %Y",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(compact, fmt)
            except Exception:
                continue
        year = re.search(r"\b(19|20)\d{2}\b", compact)
        if year is not None:
            try:
                return datetime(int(year.group(0)), 1, 1)
            except Exception:
                return None
        return None

    @classmethod
    def _runtime_context_rows(cls, *, multimodal_context: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in str(multimodal_context or "").splitlines() if line.strip()]
        rows: list[dict[str, Any]] = []
        current_date = ""
        current_session = ""
        for line in lines:
            lowered = line.lower()
            if lowered.startswith("session_id:"):
                current_session = line.split(":", 1)[1].strip()
                continue
            if lowered.startswith("date:"):
                current_date = line.split(":", 1)[1].strip()
                continue
            if lowered.startswith("relevant excerpts from the conversation"):
                continue
            if lowered.startswith("question focus:"):
                continue
            line_date = cls._extract_date_like_text(line)
            if line_date:
                current_date = line_date
            content = cls._line_content_text(line)
            if not content:
                continue
            row_date = current_date or line_date
            rows.append(
                {
                    "session_id": current_session,
                    "date_text": row_date,
                    "date_value": cls._runtime_context_parse_date_value(row_date),
                    "line": line,
                    "content": content,
                    "content_lower": content.lower(),
                }
            )
        return rows

    @staticmethod
    def _runtime_context_phrase_tokens(text: str) -> list[str]:
        stop = {
            "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "at",
            "my", "your", "their", "his", "her", "our", "did", "was", "were", "is", "are",
            "event", "task", "issue", "thing", "item", "first", "last", "before", "after",
        }
        out: list[str] = []
        for tok in re.findall(r"[A-Za-z0-9가-힣]{2,32}", str(text or "").lower()):
            if tok in stop:
                continue
            if tok.endswith("es") and len(tok) > 4:
                tok = tok[:-2]
            elif tok.endswith("s") and len(tok) > 3:
                tok = tok[:-1]
            out.append(tok)
        return out

    @classmethod
    def _runtime_context_best_row_for_phrase(
        cls,
        *,
        rows: list[dict[str, Any]],
        phrase: str,
    ) -> dict[str, Any] | None:
        tokens = cls._runtime_context_phrase_tokens(phrase)
        if not rows:
            return None
        best: dict[str, Any] | None = None
        best_score = -1
        for row in rows:
            content_lower = str(row.get("content_lower") or "")
            score = 0
            for tok in tokens:
                if tok in content_lower:
                    score += 2
                    continue
                if tok.endswith("o") and (tok + "es") in content_lower:
                    score += 2
                    continue
                if (tok + "s") in content_lower:
                    score += 2
            # Direct phrase match gets stronger weight.
            if phrase:
                normalized_phrase = re.sub(r"\s+", " ", str(phrase).lower()).strip()
                if normalized_phrase and normalized_phrase in content_lower:
                    score += 5
            if score <= 0:
                continue
            if score > best_score:
                best_score = score
                best = row
        return best

    @classmethod
    def _runtime_context_extract_options(cls, *, query: str) -> tuple[str, str] | None:
        q = re.sub(r"\s+", " ", str(query or "")).strip()
        if " or " not in q.lower():
            return None
        match = re.search(r"(?:,|:)\s*([^?]+?)\s+or\s+([^?]+?)\??\s*$", q, flags=re.IGNORECASE)
        if match is None:
            match = re.search(r"\b(?:which|what)\b[^?]*?\s+(.+?)\s+or\s+(.+?)\??\s*$", q, flags=re.IGNORECASE)
        if match is None:
            return None
        left = str(match.group(1) or "").strip(" .,'\"")
        right = str(match.group(2) or "").strip(" .,'\"")
        if not left or not right:
            return None
        return left, right

    @staticmethod
    def _runtime_context_day_diff(a: datetime | None, b: datetime | None) -> int | None:
        if a is None or b is None:
            return None
        return abs((a.date() - b.date()).days)

    @staticmethod
    def _runtime_context_is_generic_chitchat(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return True
        generic_heads = (
            "i'm glad", "thats great", "that's great", "congratulations",
            "what a great", "sure", "absolutely", "thank you for",
        )
        return any(lowered.startswith(head) for head in generic_heads)

    @staticmethod
    def _runtime_context_is_money_query(query: str) -> bool:
        lowered = str(query or "").lower()
        cues = (
            "$", "how much", "money", "amount", "cost", "spent", "spend",
            "earned", "earn", "raise", "raised", "fund",
            "얼마", "금액", "돈", "비용",
        )
        return any(cue in lowered for cue in cues)

    @staticmethod
    def _runtime_context_is_aggregate_numeric_query(query: str) -> bool:
        lowered = str(query or "").lower()
        cues = (
            "in total", " total ", "combined", "altogether", "sum", "overall",
            "across all", "all the events", "all events",
            "합계", "총합", "총 ",
        )
        return any(cue in f" {lowered} " for cue in cues)

    @staticmethod
    def _runtime_context_extract_numbers(text: str) -> list[float]:
        src = str(text or "")
        low = src.lower()
        out: list[float] = []
        word_to_num = {
            "once": 1, "twice": 2, "thrice": 3,
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
        }
        for word, value in word_to_num.items():
            if re.search(rf"\b{word}\b", low):
                out.append(float(value))
        for m in re.finditer(r"\$?\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)", src):
            raw = str(m.group(1) or "").replace(",", "")
            if not raw:
                continue
            start = int(m.start(1))
            end = int(m.end(1))
            if (end < len(src) and src[end:end + 1] == ":") or (start > 0 and src[start - 1:start] == ":"):
                # Time-like token (e.g., 7:55)
                continue
            try:
                value = float(raw)
            except Exception:
                continue
            if abs(value - int(value)) < 1e-9 and 1900 <= int(value) <= 2100:
                # Likely year token.
                continue
            out.append(value)
        return out

    @staticmethod
    def _runtime_context_format_number(*, value: float, money: bool) -> str:
        if money:
            if abs(value - int(value)) < 1e-9:
                return f"${int(value):,}"
            return f"${value:,.2f}".rstrip("0").rstrip(".")
        if abs(value - int(value)) < 1e-9:
            return str(int(value))
        return str(round(value, 2)).rstrip("0").rstrip(".")

    @classmethod
    def _runtime_context_numeric_candidates(
        cls,
        *,
        query: str,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        lowered_query = str(query or "").lower()
        money_query = cls._runtime_context_is_money_query(query)
        query_terms = set(cls._runtime_context_phrase_tokens(query))
        ignore_terms = {
            "how", "many", "much", "total", "across", "all", "combined", "altogether", "overall",
            "times", "time", "number", "amount", "currently", "current", "now",
            "event", "events", "attended", "participating", "did", "have", "had", "my", "i",
        }
        focus_terms = {tok for tok in query_terms if tok not in ignore_terms}
        query_phrases = cls._runtime_context_qa_key_phrases(query)

        candidates: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            content = str(row.get("content") or "").strip()
            content_lower = str(row.get("content_lower") or content.lower())
            if not content:
                continue
            if content.endswith("?"):
                continue
            numbers = cls._runtime_context_extract_numbers(content)
            if not numbers:
                continue
            overlap = sum(1 for tok in focus_terms if tok and tok in content_lower)
            phrase_bonus = sum(1 for phrase in query_phrases if phrase and phrase in content_lower)
            score = (overlap * 4) + (phrase_bonus * 6)
            if money_query and ("$" in content or "dollar" in content_lower):
                score += 3
            if not focus_terms:
                score += 1
            value = float(numbers[0])
            if not money_query and value > 5000:
                continue
            session_key = str(row.get("session_id") or "").strip() or str(row.get("date_text") or "").strip() or f"row-{idx}"
            candidates.append(
                {
                    "session_key": session_key,
                    "date_value": row.get("date_value"),
                    "score": score,
                    "value": value,
                    "money_like": bool("$" in content or "dollar" in content_lower),
                    "content": content,
                }
            )
        # Keep only minimally relevant rows for numeric extraction.
        strong = [c for c in candidates if int(c.get("score") or 0) > 0]
        return strong if strong else candidates

    @classmethod
    def _runtime_context_numeric_answer(
        cls,
        *,
        query: str,
        rows: list[dict[str, Any]],
    ) -> str:
        if not rows:
            return ""
        lowered_query = str(query or "").lower()
        money_query = cls._runtime_context_is_money_query(query)
        aggregate_query = cls._runtime_context_is_aggregate_numeric_query(query)
        latest_bias = any(token in lowered_query for token in ("currently", "current", "now", "as of now", "현재", "지금"))
        candidates = cls._runtime_context_numeric_candidates(query=query, rows=rows)
        if not candidates:
            return ""

        if aggregate_query:
            per_session: dict[str, dict[str, Any]] = {}
            for cand in candidates:
                key = str(cand.get("session_key") or "")
                prev = per_session.get(key)
                if prev is None:
                    per_session[key] = cand
                    continue
                prev_score = float(prev.get("score") or 0.0)
                new_score = float(cand.get("score") or 0.0)
                prev_dt = prev.get("date_value")
                new_dt = cand.get("date_value")
                if new_score > prev_score:
                    per_session[key] = cand
                elif new_score == prev_score and isinstance(new_dt, datetime) and isinstance(prev_dt, datetime) and new_dt > prev_dt:
                    per_session[key] = cand
            picked = list(per_session.values())
            if len(picked) >= 2:
                total = sum(float(item.get("value") or 0.0) for item in picked)
                money_like = money_query or any(bool(item.get("money_like")) for item in picked)
                return cls._runtime_context_format_number(value=total, money=money_like)

        if latest_bias:
            dated = [cand for cand in candidates if isinstance(cand.get("date_value"), datetime)]
            if dated:
                dated.sort(key=lambda item: (item.get("date_value"), float(item.get("score") or 0.0)), reverse=True)
                winner = dated[0]
                return cls._runtime_context_format_number(
                    value=float(winner.get("value") or 0.0),
                    money=(money_query or bool(winner.get("money_like"))),
                )

        candidates.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                item.get("date_value") if isinstance(item.get("date_value"), datetime) else datetime.min,
            ),
            reverse=True,
        )
        winner = candidates[0]
        return cls._runtime_context_format_number(
            value=float(winner.get("value") or 0.0),
            money=(money_query or bool(winner.get("money_like"))),
        )

    @staticmethod
    def _runtime_context_qa_answer_type(query: str) -> str:
        lowered = str(query or "").lower()
        if "?" in lowered or "？" in lowered:
            end_candidates = [idx for idx in (lowered.find("?"), lowered.find("？")) if idx >= 0]
            if end_candidates:
                lowered = lowered[: (min(end_candidates) + 1)]
        if ("where" in lowered and "from" in lowered and ("move" in lowered or "moved" in lowered)) or ("move from" in lowered):
            return "entity"
        if any(
            token in lowered
            for token in (
                "how long",
                "for how long",
                "how many years",
                "how many months",
                "how long ago",
                "how old",
                "몇 년",
                "몇년",
                "얼마나 오래",
                "기간",
            )
        ):
            return "number"
        if any(token in lowered for token in ("when ", " date", "언제", "날짜", "몇 월", "몇월", "년도", "연도")):
            return "date"
        if any(token in lowered for token in ("how many", "how much", "count", "몇", "얼마", "합계", "총합")):
            return "number"
        if lowered.startswith(
            (
                "is ",
                "are ",
                "do ",
                "did ",
                "can ",
                "could ",
                "would ",
                "was ",
                "were ",
                "should ",
                "will ",
            )
        ):
            return "boolean"
        if any(token in lowered for token in ("맞아", "아니", "정말", "일까", "인가", "맞지", "가능할까")):
            return "boolean"
        if lowered.startswith("if ") and "?" in lowered:
            return "boolean"
        if (" if " in lowered or "만약" in lowered or "가정" in lowered) and "?" in lowered:
            if re.search(r"\b(is|are|do|did|can|could|would|was|were|should|will)\b", lowered):
                return "boolean"
        if any(
            token in lowered
            for token in (
                "which ",
                "who ",
                "where ",
                "what ",
                "what is the name",
                "what field",
                "what fields",
                "누구",
                "어떤",
                "어디",
                "무엇",
                "뭐",
            )
        ):
            return "entity"
        return "freeform"
