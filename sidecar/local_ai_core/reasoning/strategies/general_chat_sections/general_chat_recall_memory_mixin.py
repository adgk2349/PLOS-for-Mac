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


class GeneralChatRecallMemoryMixin:
    @staticmethod
    def _ko_topic_particle(word: str) -> str:
        value = str(word or "").strip()
        if not value:
            return "는"
        last = value[-1]
        code = ord(last)
        if 0xAC00 <= code <= 0xD7A3:
            jong = (code - 0xAC00) % 28
            return "은" if jong != 0 else "는"
        return "는"

    @staticmethod
    def _render_fact_line(*, key: str, value: str, response_language: str) -> str:
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            return ""
        if response_language == "ko":
            label_map = {
                "user_name": "이름",
                "favorite_drink": "좋아하는 음료",
                "pet_name": "반려동물 이름",
                "arrival_time": "도착 시간",
                "lodging_candidates": "숙소 후보",
                "budget": "예산",
                "trip_destination": "여행지",
                "trip_duration": "여행 기간",
                "preference_morning": "생활 패턴",
                "preference_quiet": "선호 스타일",
            }
            label = label_map.get(k, k)
            particle = GeneralChatRecallMemoryMixin._ko_topic_particle(label)
            return f"{label}{particle} {v}"
        if response_language == "ja":
            return f"{k}は{v}"
        return f"{k}: {v}"

    @staticmethod
    def _has_memory_recall_cue(query: str) -> bool:
        return bool(GeneralChatRecallGateHelpers.has_memory_recall_cue(query))

    @staticmethod
    def _memory_recall_fact_subject(query: str) -> str | None:
        lowered = unicodedata.normalize("NFC", str(query or "").strip()).lower()
        if not lowered:
            return None
        if any(token in lowered for token in ("고양이", "반려묘", "pet", "반려견", "반려동물")):
            return "pet_name"
        if any(token in lowered for token in ("음료", "drink", "beverage", "커피")):
            return "favorite_drink"
        if any(token in lowered for token in ("도착 시간", "도착", "arrival", "arrive")):
            return "arrival_time"
        if any(token in lowered for token in ("숙소", "호텔", "lodging", "hotel")) and any(
            token in lowered for token in ("후보", "근처", "위치", "candidate", "location")
        ):
            return "lodging_candidates"
        if any(token in lowered for token in ("예산", "budget")):
            return "budget"
        if any(token in lowered for token in ("여행 성향", "travel style", "취향")):
            return "all"
        if any(token in lowered for token in ("개인 정보", "개인정보", "personal info", "기억해달라고 한")):
            return "all"
        if any(token in lowered for token in ("이름", "name")):
            return "user_name"
        return None

    @classmethod
    def _fact_map_from_memory_bundle(cls, memory_bundle: Any) -> dict[str, str]:
        items = list(getattr(memory_bundle, "session_items", []) or []) if memory_bundle is not None else []
        fact_map: dict[str, str] = {}
        for item in items:
            key = str(getattr(item, "key", "") or "").strip()
            if not key.startswith("fact:"):
                continue
            subject = key.split(":", 1)[1].strip()
            value_json = getattr(item, "value_json", None)
            if not isinstance(value_json, dict):
                continue
            value = str(value_json.get("value") or "").strip()
            if not value:
                summary = str(value_json.get("summary") or "").strip()
                if summary:
                    value = summary
            if value:
                fact_map[subject] = value
        return fact_map

    @classmethod
    def _scene_rows_from_memory_bundle(cls, memory_bundle: Any) -> list[dict[str, Any]]:
        items = list(getattr(memory_bundle, "session_items", []) or []) if memory_bundle is not None else []
        rows: list[dict[str, Any]] = []
        for item in items:
            key = str(getattr(item, "key", "") or "").strip()
            if not key.startswith("scene:"):
                continue
            value_json = getattr(item, "value_json", None)
            if not isinstance(value_json, dict):
                continue
            query = str(value_json.get("query") or "").strip()
            summary = str(value_json.get("summary") or "").strip()
            tags = value_json.get("tags") if isinstance(value_json.get("tags"), list) else []
            tags_norm = [str(tag).strip() for tag in tags if str(tag).strip()]
            if not query and not summary:
                continue
            rows.append(
                {
                    "query": query,
                    "summary": summary,
                    "tags": tags_norm[:8],
                    "updated_at": str(value_json.get("updated_at") or ""),
                }
            )
        return rows[-24:]

    @classmethod
    def _memory_recall_response_from_fact_store(
        cls,
        *,
        query: str,
        response_language: str,
        memory_bundle: Any,
    ) -> dict[str, str]:
        result = {"answer": "", "hit_subject": "", "miss_reason": ""}
        subject = cls._memory_recall_fact_subject(query)
        fact_map = cls._fact_map_from_memory_bundle(memory_bundle)
        if not fact_map:
            result["miss_reason"] = "empty_fact_store"
            return result

        def _ko_label(sub: str) -> str:
            if sub == "user_name":
                return "이름"
            if sub == "favorite_drink":
                return "좋아하는 음료"
            if sub == "pet_name":
                return "반려동물 이름"
            return sub

        if subject == "all":
            ordered = [("user_name", fact_map.get("user_name")), ("favorite_drink", fact_map.get("favorite_drink")), ("pet_name", fact_map.get("pet_name"))]
            pairs = [(k, v) for k, v in ordered if v]
            if not pairs:
                result["miss_reason"] = "no_fact_for_subject"
                return result
            if response_language == "ko":
                result["answer"] = ", ".join(
                    cls._render_fact_line(key=k, value=str(v), response_language=response_language) for k, v in pairs
                )
                result["hit_subject"] = "all"
                return result
            if response_language == "ja":
                result["answer"] = "、".join(
                    cls._render_fact_line(key=k, value=str(v), response_language=response_language) for k, v in pairs
                )
                result["hit_subject"] = "all"
                return result
            result["answer"] = ", ".join(
                cls._render_fact_line(key=k, value=str(v), response_language=response_language) for k, v in pairs
            )
            result["hit_subject"] = "all"
            return result

        scene_rows = cls._scene_rows_from_memory_bundle(memory_bundle)

        if subject:
            value = str(fact_map.get(subject) or "").strip()
            if not value:
                result["miss_reason"] = "no_fact_for_subject"
                return result
            else:
                if response_language == "ko":
                    result["answer"] = cls._render_fact_line(
                        key=subject,
                        value=value,
                        response_language=response_language,
                    )
                    result["hit_subject"] = subject
                    return result
                if response_language == "ja":
                    result["answer"] = cls._render_fact_line(
                        key=subject,
                        value=value,
                        response_language=response_language,
                    )
                    result["hit_subject"] = subject
                    return result
                result["answer"] = cls._render_fact_line(
                    key=subject,
                    value=value,
                    response_language=response_language,
                )
                if scene_rows:
                    line = cls._best_scene_line_for_query(query=query, scene_rows=scene_rows)
                    if line:
                        if response_language == "ko":
                            result["answer"] = f"{result['answer']}. 참고로 {line}"
                        elif response_language == "ja":
                            result["answer"] = f"{result['answer']}。参考として、{line}"
                        else:
                            result["answer"] = f"{result['answer']}. For context, {line}"
                result["hit_subject"] = subject
                return result

        # Generic recall fallback over fact_store (no forced routing).
        query_terms = set(re.findall(r"[A-Za-z가-힣0-9_]{2,32}", unicodedata.normalize("NFC", str(query or "").lower())))
        scored: list[tuple[float, str, str]] = []
        for key, val in fact_map.items():
            value = str(val or "").strip()
            if not value:
                continue
            surface = f"{key} {value}".lower()
            terms = set(re.findall(r"[A-Za-z가-힣0-9_]{2,32}", surface))
            overlap = len(query_terms.intersection(terms)) if query_terms else 0
            score = float(overlap)
            if any(token in key for token in ("trip_", "arrival_", "lodging_", "budget", "preference_")):
                score += 0.25
            scored.append((score, key, value))
        scored.sort(key=lambda row: row[0], reverse=True)
        top = [row for row in scored if row[0] > 0][:3]
        if not top:
            result["miss_reason"] = result["miss_reason"] or "unknown_subject"
            return result

        if response_language == "ko":
            answer = ", ".join(
                cls._render_fact_line(key=k, value=v, response_language=response_language) for _, k, v in top
            )
        elif response_language == "ja":
            answer = "、".join(
                cls._render_fact_line(key=k, value=v, response_language=response_language) for _, k, v in top
            )
        else:
            answer = ", ".join(
                cls._render_fact_line(key=k, value=v, response_language=response_language) for _, k, v in top
            )
        result["answer"] = answer
        result["hit_subject"] = str(top[0][1])
        if scene_rows:
            line = cls._best_scene_line_for_query(query=query, scene_rows=scene_rows)
            if line:
                if response_language == "ko":
                    result["answer"] = f"{result['answer']}. 맥락으로는 {line}"
                elif response_language == "ja":
                    result["answer"] = f"{result['answer']}。文脈としては{line}"
                else:
                    result["answer"] = f"{result['answer']}. In context, {line}"
        return result

    @staticmethod
    def _best_scene_line_for_query(*, query: str, scene_rows: list[dict[str, Any]]) -> str:
        q_terms = set(re.findall(r"[A-Za-z가-힣0-9_]{2,32}", unicodedata.normalize("NFC", str(query or "").lower())))
        best = ""
        best_score = -1.0
        for row in scene_rows[-12:]:
            summary = str(row.get("summary") or "").strip()
            query_text = str(row.get("query") or "").strip()
            base = summary or query_text
            if not base:
                continue
            text = f"{base} {' '.join(row.get('tags') or [])}".lower()
            terms = set(re.findall(r"[A-Za-z가-힣0-9_]{2,32}", text))
            overlap = len(q_terms.intersection(terms)) if q_terms else 0
            score = float(overlap)
            if query_text:
                score += 0.1
            if score > best_score:
                best_score = score
                best = base
        return re.sub(r"\s+", " ", best).strip()[:140]

    @staticmethod
    def _normalize_memory_recall_surface(*, text: str, response_language: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        # Strip contract wrappers or malformed leftovers.
        value = re.sub(r"</?final_answer>", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"<[^>]+>", "", value).strip()
        # Drop obviously broken single-fragment outputs (e.g., stray suffix chunks).
        if re.fullmatch(r"[A-Za-z]{2,12}", value) and len(value) <= 8:
            return ""
        return value

    @classmethod
    def _memory_recall_slots(
        cls,
        *,
        query: str,
        last_context: dict[str, Any] | None,
        session_digest: str | None,
    ) -> dict[str, Any]:
        sources: list[str] = []
        if isinstance(last_context, dict):
            for key in ("last_user_query", "result_summary", "parsed_target"):
                value = str(last_context.get(key) or "").strip()
                if value:
                    sources.append(value[:280])
        digest_text = str(session_digest or "").strip()
        if digest_text:
            sources.append(digest_text[:900])
        corpus = "\n".join(sources).strip()

        nickname = cls._extract_memory_nickname(corpus)
        lifestyle = cls._extract_memory_lifestyle(corpus)
        lowered_query = unicodedata.normalize("NFC", str(query or "").strip()).lower()
        asks_nickname = any(token in lowered_query for token in ("별명", "닉네임", "코드네임", "코드명", "nickname", "codename"))
        asks_lifestyle = any(token in lowered_query for token in ("생활 패턴", "패턴", "아침형", "저녁형", "lifestyle"))
        if not asks_nickname and not asks_lifestyle:
            asks_nickname = bool(nickname)
            asks_lifestyle = bool(lifestyle)
        return {
            "nickname": nickname,
            "lifestyle": lifestyle,
            "asks_nickname": bool(asks_nickname),
            "asks_lifestyle": bool(asks_lifestyle),
        }

    @staticmethod
    def _memory_recall_has_requested_fact(slots: dict[str, Any]) -> bool:
        asks_nickname = bool(slots.get("asks_nickname"))
        asks_lifestyle = bool(slots.get("asks_lifestyle"))
        nickname = str(slots.get("nickname") or "").strip()
        lifestyle = str(slots.get("lifestyle") or "").strip()
        if asks_nickname and not nickname:
            return False
        if asks_lifestyle and not lifestyle:
            return False
        return bool(asks_nickname or asks_lifestyle)

    @staticmethod
    def _memory_recall_missing_message(*, response_language: str) -> str:
        if response_language == "ko":
            return "지금은 제가 기억한 별명/코드네임/패턴 정보가 없어요. 다시 알려주시면 바로 반영할게요."
        if response_language == "ja":
            return "今はニックネームや生活パターンの記憶が見つかりません。もう一度教えていただければ反映します。"
        return "I do not have your nickname/lifestyle memory right now. Tell me once and I will keep it."

    @staticmethod
    def _memory_recall_prompt(
        *,
        query: str,
        response_language: str,
        slots: dict[str, Any],
        last_context: dict[str, Any] | None = None,
        session_digest: str | None = None,
    ) -> str:
        nickname = str(slots.get("nickname") or "").strip() or "(unknown)"
        lifestyle = str(slots.get("lifestyle") or "").strip() or "(unknown)"
        asks_nickname = bool(slots.get("asks_nickname"))
        asks_lifestyle = bool(slots.get("asks_lifestyle"))
        context_lines: list[str] = []
        if isinstance(last_context, dict):
            last_user_query = str(last_context.get("last_user_query") or "").strip()
            result_summary = str(last_context.get("result_summary") or "").strip()
            if last_user_query:
                context_lines.append(f"- last_user_query: {last_user_query[:120]}")
            if result_summary:
                context_lines.append(f"- last_result_summary: {result_summary[:180]}")
        digest_text = str(session_digest or "").strip()
        if digest_text:
            recall_match = re.search(
                r"<memory_recall_context>[\s\S]*?</memory_recall_context>",
                digest_text,
                flags=re.IGNORECASE,
            )
            recall_block = str(recall_match.group(0) or "").strip() if recall_match else ""
            if recall_block:
                context_lines.append("- memory_recall_context:")
                context_lines.append(recall_block[:1800])
            else:
                context_lines.append(f"- session_digest: {digest_text[:320]}")
        memory_context_block = ""
        if context_lines:
            memory_context_block = "최근 대화 맥락:\n" + "\n".join(context_lines) + "\n"
        if response_language == "ko":
            return (
                "너는 메모리 회수 질문에 답하는 로컬 어시스턴트다.\n"
                f"사용자 질문: {query}\n"
                "기억된 사실:\n"
                f"- nickname: {nickname}\n"
                f"- lifestyle: {lifestyle}\n"
                f"- asks_nickname: {int(asks_nickname)}\n"
                f"- asks_lifestyle: {int(asks_lifestyle)}\n"
                f"{memory_context_block}"
                "규칙:\n"
                "1) 주어진 사실만 사용하고 추측하지 마라.\n"
                "2) 질문이 요청한 슬롯을 포함해 자연스럽게 1~2문장으로 답하라.\n"
                "3) 시스템/토픽전환/메모리부족 사과문은 쓰지 마라.\n"
                "4) 코드 블록/마크다운 fence(```)를 출력하지 마라.\n"
                "답변:"
            )
        if response_language == "ja":
            return (
                "あなたは記憶呼び出し質問に答えるローカルアシスタントです。\n"
                f"ユーザー質問: {query}\n"
                "記憶事実:\n"
                f"- nickname: {nickname}\n"
                f"- lifestyle: {lifestyle}\n"
                f"- asks_nickname: {int(asks_nickname)}\n"
                f"- asks_lifestyle: {int(asks_lifestyle)}\n"
                f"{memory_context_block}"
                "ルール:\n"
                "1) 与えられた事実のみを使い、推測しないこと。\n"
                "2) 求められた項目を含めて自然な1〜2文で回答すること。\n"
                "3) システム説明や話題転換の文言は出さないこと。\n"
                "4) コードブロックや```を出さないこと。\n"
                "回答:"
            )
        return (
            "You answer memory recall questions.\n"
            f"User query: {query}\n"
            "Remembered facts:\n"
            f"- nickname: {nickname}\n"
            f"- lifestyle: {lifestyle}\n"
            f"- asks_nickname: {int(asks_nickname)}\n"
            f"- asks_lifestyle: {int(asks_lifestyle)}\n"
            f"{memory_context_block}"
            "Rules:\n"
            "1) Use only the provided facts, no guessing.\n"
            "2) Answer naturally in 1-2 sentences and include requested slots.\n"
            "3) Do not mention internal process.\n"
            "4) Do not output code fences or markdown blocks.\n"
            "Answer:"
        )

    @staticmethod
    def _is_invalid_memory_recall_output(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        if value.count("```") >= 2:
            return True
        collapsed = re.sub(r"\s+", "", value)
        if re.fullmatch(r"(?:`{3}){3,}", collapsed):
            return True
        if len(set(collapsed)) <= 2 and len(collapsed) >= 12:
            return True
        return False

    @classmethod
    def _memory_recall_generation_valid(cls, *, generated_text: str, slots: dict[str, Any]) -> bool:
        text = str(generated_text or "").strip()
        if cls._is_invalid_memory_recall_output(text):
            return False
        asks_nickname = bool(slots.get("asks_nickname"))
        asks_lifestyle = bool(slots.get("asks_lifestyle"))
        nickname = str(slots.get("nickname") or "").strip()
        lifestyle = str(slots.get("lifestyle") or "").strip()
        lowered = unicodedata.normalize("NFC", text).lower()
        if asks_nickname and nickname:
            if unicodedata.normalize("NFC", nickname).lower() not in lowered:
                return False
        if asks_lifestyle and lifestyle:
            if unicodedata.normalize("NFC", lifestyle).lower() not in lowered:
                return False
        return True

    @staticmethod
    def _memory_recall_context_tail(*, query: str, last_context: dict[str, Any] | None, response_language: str) -> str:
        if not isinstance(last_context, dict):
            return ""
        last_user_query = str(last_context.get("last_user_query") or "").strip()
        if not last_user_query:
            return ""
        lowered_last = unicodedata.normalize("NFC", last_user_query).lower()
        lowered_now = unicodedata.normalize("NFC", str(query or "").strip()).lower()
        if not lowered_last or lowered_last == lowered_now:
            return ""
        if utils._is_memory_recall_query(last_user_query):
            return ""
        if response_language == "ko":
            return f" 이어서 '{last_user_query[:40]}' 맥락으로 계속 도와드릴게요."
        if response_language == "ja":
            return f" 続けて『{last_user_query[:40]}』の文脈で進めます。"
        return f" I can continue with the context of '{last_user_query[:40]}'."

    @classmethod
    def _build_memory_recall_response(
        cls,
        *,
        query: str,
        last_context: dict[str, Any] | None,
        session_digest: str | None,
        response_language: str,
    ) -> str:
        if not cls._has_memory_recall_cue(query):
            return ""
        slots = cls._memory_recall_slots(
            query=query,
            last_context=last_context,
            session_digest=session_digest,
        )
        nickname = str(slots.get("nickname") or "").strip()
        lifestyle = str(slots.get("lifestyle") or "").strip()
        asks_nickname = bool(slots.get("asks_nickname"))
        asks_lifestyle = bool(slots.get("asks_lifestyle"))
        if not cls._memory_recall_has_requested_fact(slots):
            return cls._memory_recall_missing_message(response_language=response_language)

        if response_language == "ko":
            if asks_nickname and asks_lifestyle:
                base = f"네, 별명은 {nickname}, 생활 패턴은 {lifestyle}로 기억하고 있어요."
                return base + cls._memory_recall_context_tail(query=query, last_context=last_context, response_language=response_language)
            if asks_nickname:
                base = f"{nickname}로 기억하고 있어요."
                return base + cls._memory_recall_context_tail(query=query, last_context=last_context, response_language=response_language)
            if asks_lifestyle:
                base = f"{lifestyle}로 기억하고 있어요."
                return base + cls._memory_recall_context_tail(query=query, last_context=last_context, response_language=response_language)
            return ""
        if response_language == "ja":
            if asks_nickname and asks_lifestyle:
                base = f"はい、ニックネームは{nickname}、生活パターンは{lifestyle}として覚えています。"
                return base + cls._memory_recall_context_tail(query=query, last_context=last_context, response_language=response_language)
            if asks_nickname:
                base = f"{nickname}として覚えています。"
                return base + cls._memory_recall_context_tail(query=query, last_context=last_context, response_language=response_language)
            if asks_lifestyle:
                base = f"{lifestyle}として覚えています。"
                return base + cls._memory_recall_context_tail(query=query, last_context=last_context, response_language=response_language)
            return ""
        if asks_nickname and asks_lifestyle:
            base = f"I remember your nickname as {nickname}, and your lifestyle pattern as {lifestyle}."
            return base + cls._memory_recall_context_tail(query=query, last_context=last_context, response_language=response_language)
        if asks_nickname:
            base = f"I remember it as {nickname}."
            return base + cls._memory_recall_context_tail(query=query, last_context=last_context, response_language=response_language)
        if asks_lifestyle:
            base = f"I remember your lifestyle pattern as {lifestyle}."
            return base + cls._memory_recall_context_tail(query=query, last_context=last_context, response_language=response_language)
        return ""
