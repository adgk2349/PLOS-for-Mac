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


class GeneralChatUtilityMixin:

    @staticmethod
    def _looks_leading_fragment(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        return bool(
            re.match(
                r"^(?:께(?:서는|요)?|을|를|이|가|은|는|도)\s*(?:붙여드리겠습니다|도와드리겠습니다|안내해드리겠습니다|질문하신|오늘|집중)",
                value,
            )
        )

    @staticmethod
    def _wait_for_port(host: str, port: int, *, timeout_seconds: float = 6.0) -> bool:
        return GeneralChatWebExecutionHelpers.wait_for_port(host, port, timeout_seconds=timeout_seconds)

    @staticmethod
    async def _wait_for_port_async(host: str, port: int, *, timeout_seconds: float = 6.0) -> bool:
        return await GeneralChatWebExecutionHelpers.wait_for_port_async(host, port, timeout_seconds=timeout_seconds)

    @staticmethod
    def _extract_memory_nickname(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        patterns = (
            r"(?:내\s*)?(?:별명|닉네임|코드네임|코드명)\s*(?:은|는|이|가|:|=)?\s*[\"'“”]?([A-Za-z0-9가-힣_\-]{2,24})",
            r"\bnickname\s*(?:is|:|=)\s*[\"'“”]?([A-Za-z0-9_\-]{2,24})",
            r"\bcodename\s*(?:is|:|=)\s*[\"'“”]?([A-Za-z0-9_\-]{2,24})",
        )
        for pattern in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if match is None:
                continue
            candidate = str(match.group(1) or "").strip().strip("\"'“”")
            if candidate in {"뭐였지", "뭐지", "무엇", "별명", "닉네임", "코드네임", "코드명", "nickname", "codename"}:
                continue
            return candidate[:24]
        return ""

    @staticmethod
    def _answer_type_hint(*, query: str, request_hint: str | None) -> str:
        if request_hint:
            return coerce_answer_type_hint(request_hint)
        return infer_answer_type_hint(query)

    @classmethod
    def _should_prefer_last_resort_clarification(
        cls,
        *,
        query: str,
        last_context: dict[str, Any] | None,
        ambiguity_score: float,
        answerability_score: float,
    ) -> bool:
        if cls._has_memory_recall_cue(query):
            return False
        lowered = str(query or "").strip().lower()
        if re.search(r"\b\d{1,2}\s*(월|month)\s*쯤", lowered):
            return True
        if ambiguity_score >= 0.62:
            return True
        if ambiguity_score >= 0.54 and answerability_score < 0.44:
            return True
        if ambiguity_score >= 0.48 and cls._context_overlap_score(query=query, last_context=last_context) < 0.14:
            return True
        return False

    @staticmethod
    def _default_last_resort_direct_message(*, response_language: str) -> str:
        # Keep empty to avoid fixed-template fallback replies.
        return ""

    @staticmethod
    def _split_answer_tokens_cap() -> int:
        try:
            raw = int(float(str(os.getenv("GEN_RETRY_SPLIT_MAX_TOKENS_CAP", "3072")).strip() or "3072"))
            return max(768, min(8192, raw))
        except Exception:
            return 3072

    @classmethod
    def _sampling_for_attempt(
        cls,
        *,
        profile: str,
        attempt_index: int,
        generation_style: str,
    ) -> dict[str, float | int]:
        normalized_profile = str(profile or "balanced").strip().lower()
        normalized_style = str(generation_style or "conversation").strip().lower()
        if normalized_style == "rewrite":
            if normalized_profile == "roleplay":
                return {"temperature": 0.40, "top_p": 0.90, "top_k": 40, "repeat_penalty": 1.08}
            if normalized_profile == "coding":
                return {"temperature": 0.25, "top_p": 0.84, "top_k": 22, "repeat_penalty": 1.10}
            if normalized_profile == "concise":
                return {"temperature": 0.30, "top_p": 0.86, "top_k": 26, "repeat_penalty": 1.10}
            if normalized_profile == "analytic":
                return {"temperature": 0.28, "top_p": 0.86, "top_k": 24, "repeat_penalty": 1.10}
            return {"temperature": 0.30, "top_p": 0.86, "top_k": 24, "repeat_penalty": 1.10}

        if normalized_profile == "roleplay":
            if attempt_index <= 0:
                return {"temperature": 0.72, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.06}
            return {"temperature": 0.64, "top_p": 0.93, "top_k": 56, "repeat_penalty": 1.07}
        if normalized_profile == "coding":
            if attempt_index <= 0:
                return {"temperature": 0.38, "top_p": 0.90, "top_k": 32, "repeat_penalty": 1.15}
            return {"temperature": 0.34, "top_p": 0.88, "top_k": 28, "repeat_penalty": 1.14}
        if normalized_profile == "concise":
            if attempt_index <= 0:
                return {"temperature": 0.34, "top_p": 0.84, "top_k": 24, "repeat_penalty": 1.18}
            return {"temperature": 0.30, "top_p": 0.82, "top_k": 22, "repeat_penalty": 1.18}
        if normalized_profile == "analytic":
            if attempt_index <= 0:
                return {"temperature": 0.50, "top_p": 0.92, "top_k": 38, "repeat_penalty": 1.16}
            return {"temperature": 0.46, "top_p": 0.90, "top_k": 34, "repeat_penalty": 1.15}
        if attempt_index <= 0:
            return {"temperature": 0.55, "top_p": 0.92, "top_k": 40, "repeat_penalty": 1.16}
        return {"temperature": 0.50, "top_p": 0.90, "top_k": 36, "repeat_penalty": 1.15}

    @staticmethod
    def _looks_incomplete_answer(text: str) -> bool:
        from ... import language_profiles
        value = str(text or "").strip()
        if not value:
            return False
        if re.search(r"(?im)(?:^|\n)\s*\d{1,2}[.)]\s*$", value):
            return True
        if value.count("```") % 2 == 1:
            return True
        if re.search(r"[:;,(\[{`-]\s*$", value):
            return True
        terminals = language_profiles.sentence_terminal_chars_for_text(value)
        if len(value) >= 120 and value[-1] not in terminals:
            return True
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if lines:
            last = lines[-1]
            if len(last) >= 32 and re.search(r"[A-Za-z가-힣0-9]$", last) and not re.search(r"[.!?。！？]$", last):
                return True
        return False

    @staticmethod
    def _merge_continuation(base: str, tail: str) -> str:
        head = str(base or "").rstrip()
        cont = str(tail or "").lstrip()
        if not head:
            return cont
        if not cont:
            return head
        tail_norm = " ".join(cont.split())
        head_tail_norm = " ".join(head[-180:].split())
        if tail_norm and tail_norm in head_tail_norm:
            return head
        if head.endswith(("```", "\n")) or cont.startswith(("```", "\n")):
            return f"{head}{cont}".strip()
        return f"{head}\n{cont}".strip()

    @staticmethod
    def _trim_incomplete_tail(text: str) -> str:
        from ... import language_profiles
        value = str(text or "").strip()
        if not value:
            return ""
        if value.count("```") % 2 == 1:
            return value.rstrip() + "\n```"
        terminals = language_profiles.sentence_terminal_chars_for_text(value)
        if value[-1] in terminals:
            return value
        sentence_cut = max(
            value.rfind("."),
            value.rfind("!"),
            value.rfind("?"),
            value.rfind("。"),
            value.rfind("！"),
            value.rfind("？"),
        )
        if sentence_cut >= int(len(value) * 0.45):
            trimmed = value[: sentence_cut + 1].rstrip()
            if trimmed:
                return trimmed
        newline_cut = value.rfind("\n")
        if newline_cut >= int(len(value) * 0.5):
            trimmed = value[:newline_cut].rstrip()
            if trimmed:
                return trimmed
        space_cut = value.rfind(" ")
        if space_cut >= int(len(value) * 0.6):
            trimmed = value[:space_cut].rstrip()
            if trimmed and trimmed[-1] not in terminals:
                trimmed += "."
            return trimmed or value
        return value

    @staticmethod
    def _single_question(text: str, *, response_language: str) -> str:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip(" -\t") for line in raw.split("\n") if line.strip()]
        if not lines:
            return ""
        candidate = ""
        for line in lines:
            lowered = line.lower()
            if lowered.startswith("question:") or lowered.startswith("질문:") or lowered.startswith("質問:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    break
        for line in lines:
            if candidate:
                break
            if "?" in line or "？" in line:
                candidate = line
                break
        if not candidate:
            candidate = lines[0]
        if "?" in candidate or "？" in candidate:
            end_candidates = [idx for idx in (candidate.find("?"), candidate.find("？")) if idx >= 0]
            if end_candidates:
                end = min(end_candidates)
                candidate = candidate[: end + 1]
        if not candidate.endswith(("?", "？")):
            candidate += "?" if response_language != "ko" else "?"
        return candidate[:180].strip()

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z가-힣0-9_]{2,32}", str(text or "").lower())}

    @classmethod
    def _context_overlap_score(cls, *, query: str, last_context: dict[str, Any] | None) -> float:
        q_terms = cls._query_terms(query)
        if not q_terms or not isinstance(last_context, dict):
            return 0.0
        context_text = " ".join(
            [
                str(last_context.get("last_user_query") or ""),
                str(last_context.get("result_summary") or ""),
                str(last_context.get("parsed_target") or ""),
            ]
        )
        c_terms = cls._query_terms(context_text)
        if not c_terms:
            return 0.0
        return max(0.0, min(1.0, len(q_terms.intersection(c_terms)) / max(1, len(q_terms))))

    @classmethod
    def _estimate_last_resort_scores(
        cls,
        *,
        query: str,
        last_context: dict[str, Any] | None,
    ) -> tuple[float, float]:
        lowered = str(query or "").strip().lower()
        pronoun_tokens = ("그때", "그거", "이거", "저거", "that", "this", "previous", "around")
        fuzzy_tokens = ("쯤", "대충", "비슷", "roughly", "similar", "around")
        pronoun_hit = any(token in lowered for token in pronoun_tokens)
        fuzzy_hit = any(token in lowered for token in fuzzy_tokens) or bool(
            re.search(r"\b\d{1,2}\s*(월|month)\s*쯤", lowered)
        )
        explicit_hint = bool(re.search(r"\.(txt|pdf|md|docx|py|swift|json|yaml|yml)\b", lowered))
        context_overlap = cls._context_overlap_score(query=query, last_context=last_context)
        token_count = len(cls._query_terms(query))
        specificity = min(1.0, max(0.0, token_count / 12.0))
        if explicit_hint:
            specificity = min(1.0, specificity + 0.18)

        ambiguity_score = 0.0
        if pronoun_hit:
            ambiguity_score += 0.28
        if fuzzy_hit:
            ambiguity_score += 0.24
        ambiguity_score += (1.0 - context_overlap) * (0.20 if pronoun_hit else 0.08)
        if explicit_hint:
            ambiguity_score -= 0.14
        ambiguity_score = max(0.0, min(1.0, ambiguity_score))

        answerability_score = max(
            0.0,
            min(1.0, (specificity * 0.58) + (context_overlap * 0.34) - (ambiguity_score * 0.16)),
        )
        return ambiguity_score, answerability_score

    async def _generate_last_resort_clarification(
        self,
        *,
        executor,
        context: ReasoningContext,
        query: str,
        timeout_seconds: float,
        style_profile: dict[str, Any] | None = None,
    ) -> str:
        focus = self._general_clarify_focus(query=query, last_context=context.last_context)
        last_query = str((context.last_context or {}).get("last_user_query") or "").strip()[:180]
        last_summary = str((context.last_context or {}).get("result_summary") or "").strip()[:220]
        prompt = (
            "You are a local assistant. Create one short clarification question only.\n"
            f"Language: {context.response_language}\n"
            f"User query: {query}\n"
            f"Missing focus: {focus}\n"
            f"Previous user query: {last_query}\n"
            f"Previous answer summary: {last_summary}\n"
            "Ask only for the missing part and avoid re-asking known context.\n"
            "No bullets. No explanation. One question:"
        )
        try:
            execution = await self._run_conversation_inference(
                executor=executor,
                query=prompt,
                context=context,
                max_tokens=72,
                generation_style="rewrite",
                sampling_overrides={"temperature": 0.28, "top_p": 0.86, "top_k": 24},
                timeout_seconds=max(4.0, timeout_seconds),
                style_profile=style_profile,
            )
            if execution.used_fallback:
                return ""
            return self._single_question(str(execution.generated_text or ""), response_language=context.response_language)
        except asyncio.CancelledError:
            raise
        except Exception:
            return ""

    async def _generate_last_resort_direct_answer(
        self,
        *,
        executor,
        context: ReasoningContext,
        query: str,
        timeout_seconds: float,
        style_profile: dict[str, Any] | None = None,
    ) -> str:
        prompt = (
            "You are a local assistant. Answer the user directly in one short paragraph.\n"
            f"Language: {context.response_language}\n"
            f"User query: {query}\n"
            "Rules: no clarification question, no mention of system/internal process.\n"
            "Answer:"
        )
        try:
            execution = await self._run_conversation_inference(
                executor=executor,
                query=prompt,
                context=context,
                max_tokens=96,
                generation_style="rewrite",
                sampling_overrides={"temperature": 0.28, "top_p": 0.86, "top_k": 24},
                timeout_seconds=max(4.0, timeout_seconds),
                style_profile=style_profile,
            )
            if execution.used_fallback:
                return ""
            text = str(execution.generated_text or "").strip()
            if not text:
                return ""
            if "?" in text or "？" in text:
                return ""
            return text[:280].strip()
        except asyncio.CancelledError:
            raise
        except Exception:
            return ""

    def _build_personal_memory_context(self, memory_bundle: Optional[Any]) -> str:
        if not memory_bundle:
            return ""
        identity = []
        # Support both object and dict memory bundles
        nickname = getattr(memory_bundle, "user_nickname", None) or (
            memory_bundle.get("user_nickname") if isinstance(memory_bundle, dict) else None
        )
        if nickname:
            identity.append(f"- 사용자의 닉네임: {nickname}")
        
        tone_pref = getattr(memory_bundle, "tone_preference", None) or (
            memory_bundle.get("tone_preference") if isinstance(memory_bundle, dict) else None
        )
        if tone_pref:
            identity.append(f"- 선호하는 대화 스타일: {tone_pref}")

        session_items = list(getattr(memory_bundle, "session_items", []) or [])
        fact_lines: list[str] = []
        for item in session_items:
            key = str(getattr(item, "key", "") or "").strip()
            if not key.startswith("fact:"):
                continue
            value_json = getattr(item, "value_json", None)
            if not isinstance(value_json, dict):
                continue
            subject = str(value_json.get("subject") or key.split(":", 1)[1]).strip()
            value = str(value_json.get("value") or "").strip()
            if not value:
                continue
            fact_lines.append(f"- {subject}: {value}")
            if len(fact_lines) >= 8:
                break
        if fact_lines:
            identity.append("[기억된 사용자 사실]")
            identity.extend(fact_lines)

        pinned_items = list(getattr(memory_bundle, "pinned_items", []) or [])
        pinned_lines: list[str] = []
        for item in pinned_items:
            scope = str(getattr(item, "scope", "") or "").strip().lower()
            if scope != "global":
                continue
            title = str(getattr(item, "title", "") or "").strip()
            content = str(getattr(item, "content", "") or "").strip()
            merged = " - ".join(part for part in [title, content] if part).strip(" -")
            if not merged:
                continue
            pinned_lines.append(f"- {merged}")
            if len(pinned_lines) >= 5:
                break
        if pinned_lines:
            identity.append("[직접 저장한 전역 메모]")
            identity.extend(pinned_lines)
            
        if not identity:
            return ""
            
        return "\n[사용자 개인화 정보]\n" + "\n".join(identity) + "\n"

    def _build_conversation_memory_context(self, memory_bundle: Optional[Any], *, response_language: str) -> str:
        if not memory_bundle:
            return ""
        session_items = list(getattr(memory_bundle, "session_items", []) or [])
        fact_lines: list[str] = []
        scene_lines: list[str] = []
        for item in session_items:
            key = str(getattr(item, "key", "") or "").strip()
            value_json = getattr(item, "value_json", None)
            if not isinstance(value_json, dict):
                continue
            if key.startswith("fact:"):
                summary = str(value_json.get("summary") or "").strip()
                if summary:
                    fact_lines.append(summary)
                if len(fact_lines) >= 6:
                    continue
            if key.startswith("scene:"):
                scene_q = str(value_json.get("query") or "").strip()
                if scene_q:
                    scene_lines.append(scene_q)
                if len(scene_lines) >= 4:
                    continue
        if not fact_lines and not scene_lines:
            return ""
        if response_language == "ko":
            fact_blob = "; ".join(fact_lines[:4]).strip()
            scene_blob = "; ".join(scene_lines[:2]).strip()
            context_blob = " | ".join([chunk for chunk in (fact_blob, scene_blob) if chunk]).strip()
            if not context_blob:
                return ""
            return (
                "내부 참고 메모리: "
                f"{context_blob}\n"
                "지시: 위 메모리는 내부 참고용이며, 그대로 복사하지 말고 현재 질문에 맞게 자연스럽게 반영해서 답하세요.\n"
            )
        fact_blob = "; ".join(fact_lines[:4]).strip()
        scene_blob = "; ".join(scene_lines[:2]).strip()
        context_blob = " | ".join([chunk for chunk in (fact_blob, scene_blob) if chunk]).strip()
        if not context_blob:
            return ""
        return (
            "Internal memory reference: "
            f"{context_blob}\n"
            "Instruction: use this only as hidden context and respond naturally to the current user request.\n"
        )

    @classmethod
    def _trim_redundant_opening_from_last_context(
        cls,
        *,
        answer: str,
        last_context: dict[str, Any] | None,
    ) -> str:
        candidate = str(answer or "").strip()
        if not candidate or not isinstance(last_context, dict):
            return candidate
        previous = cls._normalize_space(str(last_context.get("result_summary") or ""))
        if not previous:
            return candidate

        normalized_answer = cls._normalize_space(candidate)
        if not normalized_answer:
            return candidate

        previous_head = previous
        sentence_split = re.split(r"(?<=[.!?。！？])\s+|\n+", previous)
        if sentence_split:
            first_sentence = cls._normalize_space(sentence_split[0])
            if len(first_sentence) >= 18:
                previous_head = first_sentence

        # Safety-first trim policy:
        # trim only when the opening sentence is an exact duplicate of previous summary head,
        # and never trim partial-token overlaps such as "당신의" -> "의".
        ans_sentences = [cls._normalize_space(s) for s in re.split(r"(?<=[.!?。！？])\s+|\n+", normalized_answer) if cls._normalize_space(s)]
        if not ans_sentences:
            return candidate
        first_sentence = ans_sentences[0]
        prev_sentence = cls._normalize_space(previous_head)
        if len(first_sentence) >= 18 and first_sentence == prev_sentence:
            remainder = normalized_answer[len(first_sentence):].lstrip(" \n\t:;,-")
            if len(remainder) >= 12:
                return remainder
        return candidate

    @staticmethod
    def _extract_date_like_text(line: str) -> str:
        raw = str(line or "").strip()
        if not raw:
            return ""
        m = re.search(
            r"\b\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+(\d{1,2}\s+[A-Za-z]+,?\s*\d{4})\b",
            raw,
            flags=re.IGNORECASE,
        )
        if m:
            return str(m.group(1)).strip()
        m = re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\b", raw, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()
        m = re.search(r"\bon\s+(\d{1,2}\s+[A-Za-z]+,?\s*\d{4})\b", raw, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()
        m = re.search(r"\bon\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\b", raw, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()
        m = re.search(r"\b(\d{1,2}\s+[A-Za-z]+,?\s*\d{4})\b", raw, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()
        m = re.search(r"\b([A-Za-z]+\s+\d{4})\b", raw, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()
        m = re.search(r"\b(20\d{2}|19\d{2})\b", raw)
        if m:
            return str(m.group(1)).strip()
        return ""

    @staticmethod
    def _resolve_relative_date_answer(*, line: str, date_text: str) -> str:
        lowered = str(line or "").lower()
        base = str(date_text or "").strip()
        if not base:
            return base
        if "last year" in lowered or "작년" in lowered:
            year_match = re.search(r"\b(19|20)\d{2}\b", base)
            if year_match:
                try:
                    return str(int(year_match.group(0)) - 1)
                except Exception:
                    return base
            return base
        if "yesterday" in lowered or "어제" in lowered:
            parsed = None
            for fmt in ("%d %B, %Y", "%B %d, %Y", "%d %B %Y", "%B %d %Y"):
                try:
                    parsed = datetime.strptime(base, fmt)
                    break
                except Exception:
                    continue
            if parsed is not None:
                shifted = parsed - timedelta(days=1)
                return f"{shifted.day} {shifted.strftime('%B')} {shifted.year}"
        return base

    @staticmethod
    def _line_content_text(line: str) -> str:
        raw = str(line or "").strip()
        if not raw:
            return ""
        # Typical benchmark line: "{date} | {speaker}: {utterance}"
        m = re.match(r"^\s*[^|]+\|\s*[^:]+:\s*(.+)\s*$", raw)
        if m:
            return str(m.group(1)).strip()
        # Fallback: strip first date-time segment before "|"
        if "|" in raw:
            rhs = raw.split("|", 1)[1].strip()
            if ":" in rhs:
                return rhs.split(":", 1)[1].strip()
            return rhs
        return raw

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any] | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        candidate_texts: list[str] = [raw]
        fenced = re.search(r"(?is)```(?:json)?\s*(\{.*?\})\s*```", raw)
        if fenced:
            candidate_texts.insert(0, str(fenced.group(1) or "").strip())
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            candidate_texts.insert(0, raw[brace_start : brace_end + 1])
        for candidate in candidate_texts:
            candidate = str(candidate or "").strip()
            if not candidate.startswith("{"):
                continue
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @classmethod
    def _followup_memory_hint(
        cls,
        *,
        query: str,
        response_language: str,
        followup_resolution: FollowUpResolution | None,
        last_context: dict[str, Any] | None,
    ) -> str:
        enabled = str(os.getenv("LOCAL_AI_CONVERSATION_FOLLOWUP_HINT_ENABLED", "0") or "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if not enabled:
            return ""
        if not isinstance(last_context, dict):
            return ""
        text = str(query or "").strip()
        if not text:
            return ""
        lowered = unicodedata.normalize("NFC", text).lower()
        token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered))
        has_followup_signal = bool(
            (followup_resolution and followup_resolution.is_followup)
            or utils._has_followup_context_signal(text)
            or utils._has_strong_followup_context_signal(text)
        )
        overlap = cls._context_overlap_score(query=text, last_context=last_context)
        if not has_followup_signal and not (token_count <= 6 and overlap >= 0.16):
            return ""

        previous_query = str(last_context.get("last_user_query") or "").strip()[:180]
        if not previous_query:
            return ""

        if response_language == "ko":
            return (
                "<followup_hint>\n"
                "후속 질문이면 같은 맥락으로 이어서 답하고, 새 주제가 명확하면 전환하세요.\n"
                f"- 이전 질문: {previous_query}\n"
                "</followup_hint>"
            )
        return (
            "<followup_hint>\n"
            "If this is a follow-up, continue the same context. Switch topics only when explicitly requested.\n"
            f"- Previous question: {previous_query}\n"
            "</followup_hint>"
        )

    @staticmethod
    def _merge_session_summary_with_hint(*, session_summary: str | None, followup_hint: str) -> str:
        base = str(session_summary or "").strip()
        hint = str(followup_hint or "").strip()
        if not hint:
            return base
        if not base:
            return hint
        return f"{base}\n{hint}"

    @staticmethod
    def _contains_korean(text: str) -> bool:
        return bool(re.search(r"[가-힣]", str(text or "")))

    @staticmethod
    def _tokenize_keywords(text: str, *, max_tokens: int = 4) -> list[str]:
        raw_tokens = re.findall(r"[A-Za-z가-힣0-9_]{2,24}", str(text or "").lower())
        stop = {
            "latest", "update", "official", "source", "search", "info", "information",
            "최신", "업데이트", "공식", "출처", "검색", "정보", "정리", "요약", "알려", "말해",
        }
        output: list[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            if token in stop:
                continue
            if token in seen:
                continue
            seen.add(token)
            output.append(token)
            if len(output) >= max(1, max_tokens):
                break
        return output
