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


class GeneralChatRecallQueryMixin:

    @staticmethod
    def _recall_query_is_boolean_like(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
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
            return True
        return any(token in lowered for token in ("맞아", "아니", "일까", "인가", "맞지", "if "))

    @staticmethod
    def _recall_is_duration_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        duration_tokens = (
            "how long",
            "for how long",
            "how many years",
            "how many months",
            "how many weeks",
            "how long ago",
            "years ago",
            "months ago",
            "weeks ago",
            "duration",
            "몇 년",
            "몇년",
            "몇 개월",
            "몇개월",
            "얼마나 오래",
            "기간",
            "동안",
            "전",
        )
        return any(token in lowered for token in duration_tokens)

    @staticmethod
    def _recall_is_multi_item_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        multi_tokens = (
            "what activities",
            "which activities",
            "what fields",
            "which fields",
            "what do",
            "where has",
            "what are",
            "list",
            "kinds of",
            "종류",
            "목록",
            "어떤 것들",
            "무엇을 좋아",
            "어디에서",
            "활동",
        )
        return any(token in lowered for token in multi_tokens)

    @staticmethod
    def _recall_is_boolean_short_answer(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False
        return normalized in {"yes", "no", "likely yes", "likely no", "예", "아니오", "아니", "네"}

    @staticmethod
    def _recall_is_career_field_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        return any(
            token in lowered
            for token in (
                "what fields",
                "which fields",
                "career",
                "career path",
                "pursue",
                "profession",
                "진로",
                "직업",
                "커리어",
                "분야",
            )
        )

    @staticmethod
    def _recall_is_identity_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        return any(
            token in lowered
            for token in (
                "identity",
                "gender identity",
                "who is",
                "정체성",
                "정체",
                "어떤 사람",
            )
        )

    @staticmethod
    def _recall_is_origin_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        return (
            ("where" in lowered and "from" in lowered and ("move" in lowered or "moved" in lowered))
            or ("move from" in lowered)
            or ("어디" in lowered and "출신" in lowered)
        )

    @staticmethod
    def _recall_is_camped_location_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        return (("where has" in lowered and "camp" in lowered) or ("camped" in lowered and "where" in lowered))

    @staticmethod
    def _recall_is_kids_preference_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        return (
            ("kids" in lowered and any(token in lowered for token in ("like", "likes", "favorite", "favourite")))
            or ("children" in lowered and "like" in lowered)
            or ("아이" in lowered and "좋아" in lowered)
        )

    @staticmethod
    def _recall_is_relationship_status_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        return any(token in lowered for token in ("relationship status", "single or", "연애 상태", "관계 상태"))

    @classmethod
    def _recall_answer_type_hint(cls, *, query: str, request_hint: str | None) -> str:
        focus_query = cls._single_question(str(query or ""), response_language="en") or str(query or "")
        inferred = cls._runtime_context_qa_answer_type(focus_query)
        if request_hint:
            hinted = str(request_hint or "").strip().lower()
            if hinted in {"date", "number", "boolean", "entity", "freeform"}:
                # Prefer semantic inference when upstream hint is too coarse.
                if hinted == "boolean" and inferred in {"entity", "freeform", "date", "number"} and not cls._recall_query_is_boolean_like(focus_query):
                    return inferred
                if hinted == "entity" and inferred in {"date", "number"}:
                    return inferred
                if hinted == "date" and cls._recall_is_origin_query(focus_query):
                    return "entity"
                if hinted == "date" and cls._recall_is_duration_query(focus_query):
                    return "number"
                if hinted in {"date", "number"} and inferred in {"date", "number"} and hinted != inferred:
                    return inferred
                return hinted
        if inferred in {"date", "number", "boolean", "entity", "freeform"}:
            return inferred
        return coerce_answer_type_hint(infer_answer_type_hint(focus_query))

    @classmethod
    def _recall_response_language(
        cls,
        *,
        query: str,
        default_language: str,
        multimodal_notes: list[str] | None,
    ) -> str:
        fallback = normalize_language_code(default_language) or "en"
        notes_blob = " ".join(str(item or "").strip().lower() for item in (multimodal_notes or []))
        if "answer in english only" in notes_blob or "english only" in notes_blob:
            return "en"
        if "한국어로만" in notes_blob or "answer in korean only" in notes_blob:
            return "ko"
        if "日本語のみ" in notes_blob or "answer in japanese only" in notes_blob:
            return "ja"
        question_focus = cls._single_question(str(query or ""), response_language="en") or str(query or "")
        detected = normalize_language_code(detect_query_language(question_focus))
        if detected in {"en", "ko", "ja"}:
            return detected
        return "en" if fallback == "en" else fallback

    @staticmethod
    def _recall_pipeline_version() -> str:
        return "recall.v2.4.2pass.selective"

    @staticmethod
    def _recall_time_budget_seconds() -> float:
        try:
            raw = float(str(os.getenv("LOCAL_AI_RECALL_TIME_BUDGET_SECONDS", "24")).strip() or "24")
        except Exception:
            raw = 24.0
        return max(12.0, min(60.0, raw))

    @staticmethod
    def _recall_disable_timeouts() -> bool:
        raw = str(os.getenv("LOCAL_AI_RECALL_DISABLE_TIMEOUT", "1") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _runtime_context_qa_deterministic_first_enabled() -> bool:
        raw = str(os.getenv("LOCAL_AI_RUNTIME_CONTEXT_QA_DETERMINISTIC_FIRST", "0") or "0").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _recall_max_regeneration_attempts() -> int:
        try:
            raw = int(float(str(os.getenv("LOCAL_AI_RECALL_REGEN_MAX_ATTEMPTS", "4")).strip() or "4"))
        except Exception:
            raw = 4
        return max(1, min(4, raw))
