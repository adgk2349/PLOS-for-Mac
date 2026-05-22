from __future__ import annotations
import re
import json
import os
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from ..base import BaseDelegate
from ..types import InferenceResult, _ConversationCandidateResult
from ...models import LocalEngine, WorkMode, AgentAction, Citation

if TYPE_CHECKING:
    from ...local_inference import LocalInferenceEngine

# Component: result_sanitizer.py
class ResultSanitizer(BaseDelegate):

    @staticmethod
    def _normalize_korean_leading_address(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        # Remove awkward formal-leading address that frequently appears as
        # model artifact and degrades conversational tone.
        value = re.sub(r"^\s*께서는\s+", "", value).strip()
        value = re.sub(r"^\s*당신은\s+", "", value).strip()
        return value
    @staticmethod
    def _heuristic_korean_spacing(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        # conservative fallback: avoid heavy particle-splitting.
        # only add spacing at likely sentence/phrase boundaries.
        value = re.sub(r"([가-힣]{3,})(합니다|했습니다|해요|했어요|입니다|이에요|예요|네요|군요|죠|요)(?=[가-힣])", r"\1\2 ", value)
        value = re.sub(r"([.!?])([가-힣A-Za-z0-9])", r"\1 \2", value)
        value = re.sub(r"([가-힣])([A-Za-z0-9])", r"\1 \2", value)
        value = re.sub(r"([A-Za-z0-9])([가-힣])", r"\1 \2", value)
        value = re.sub(r"\s{2,}", " ", value).strip()
        return value
    @staticmethod
    def _normalize_echo_text(text: str) -> str:
        value = str(text or "").strip().lower()
        if not value:
            return ""
        value = re.sub(r"^[\"'`“”‘’\(\[\{<\s]+|[\"'`“”‘’\)\]\}>\s]+$", "", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _is_hard_query_echo(self, answer: str, query: str) -> bool:
        a = self._normalize_echo_text(answer)
        q = self._normalize_echo_text(query)
        if not a or not q:
            return False
        if a == q:
            return True
        if a.rstrip(".!?") == q.rstrip(".!?"):
            return True
        if (a.startswith(q) or q.startswith(a)) and self._text_similarity(a, q) >= 0.92:
            return True
        return False

    @staticmethod
    def _is_action_request_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        cues = (
            "해줘", "해 줘", "해봐", "정리해줘", "나눠줘", "뽑아줘", "남겨줘", "만들어줘",
            "줘", "해", "해주세요", "please", "do this", "give me",
        )
        return any(token in lowered for token in cues)

    def _directness_score(self, answer: str | None, *, query: str, is_recommendation_query: bool) -> int:
        text = str(answer or "").strip()
        if not text:
            return 999
        score = 0
        question_count = self._question_sentence_count(text)
        score += question_count
        first_sentence = self._first_sentence(text)
        if self._looks_question_sentence(first_sentence):
            score += 2
        if self._is_action_request_query(query):
            score += question_count * 2
            if self._looks_question_sentence(first_sentence):
                score += 3
        if is_recommendation_query and not self._looks_three_option_shape(text):
            score += 2
        if self._text_similarity(text, query) >= 0.88:
            score += 1
        return score

    def _first_sentence(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        first = re.split(r"(?<=[.!?。！？])\s+|\n+", value, maxsplit=1)[0]
        return first.strip()

    def _looks_question_sentence(self, sentence: str) -> bool:
        value = str(sentence or "").strip()
        if not value:
            return False
        if "?" in value:
            return True
        return re.search(r"(까요|나요|인가요|어때요|어떨까요|할까요|될까요)\s*[.!?]?$", value) is not None

    def _question_sentence_count(self, text: str) -> int:
        value = str(text or "").strip()
        if not value:
            return 0
        count = 0
        for line in value.splitlines():
            line = line.strip()
            if not line:
                continue
            for sentence in re.split(r"(?<=[.!?。！？])\s+", line):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if self._looks_question_sentence(sentence):
                    count += 1
        return count

    def _limit_question_sentences(self, text: str, *, max_questions: int) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        kept_lines: list[str] = []
        used_questions = 0
        for line in value.splitlines():
            line = line.strip()
            if not line:
                continue
            kept_sentences: list[str] = []
            for sentence in re.split(r"(?<=[.!?。！？])\s+", line):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if self._looks_question_sentence(sentence):
                    if used_questions >= max_questions:
                        continue
                    used_questions += 1
                kept_sentences.append(sentence)
            if kept_sentences:
                kept_lines.append(" ".join(kept_sentences).strip())
        return "\n".join(kept_lines).strip()

    def _looks_three_option_shape(self, text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        numbered = re.findall(r"(?m)^\s*[1-3]\.\s+", value)
        if len(numbered) >= 3:
            return True
        bullets = re.findall(r"(?m)^\s*[-•·]\s+", value)
        return len(bullets) >= 3

    def _normalize_three_option_recommendation(self, answer: str, *, response_language: str) -> str:
        value = str(answer or "").strip()
        if not value:
            return ""
        if self._looks_three_option_shape(value):
            return value
        candidates: list[str] = []
        seen: set[str] = set()
        line_matches = re.findall(r"(?m)^\s*(?:\d+[.)]|[-•·])\s*(.+?)\s*$", value)
        for raw in line_matches:
            item = re.sub(r"\s+", " ", str(raw or "").strip()).strip(" .")
            if not item:
                continue
            key = re.sub(r"[^\w가-힣]+", "", item).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(item)
            if len(candidates) >= 3:
                break
        if len(candidates) < 3:
            for raw in re.split(r"(?<=[.!?。！？])\s+|,\s+|\n+", value):
                item = re.sub(r"\s+", " ", str(raw or "").strip()).strip(" .")
                if len(item) < 6:
                    continue
                key = re.sub(r"[^\w가-힣]+", "", item).lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                candidates.append(item)
                if len(candidates) >= 3:
                    break
        if len(candidates) < 3:
            return value
        lines = [f"{idx}. {item}" for idx, item in enumerate(candidates[:3], start=1)]
        return "\n".join(lines).strip()

    def _postprocess_conversational_answer(self, answer: str, *, query: str, response_language: str) -> str:
        raw_pass_mode = str(os.getenv("LOCAL_AI_CONVERSATION_RAW_PASS_ENABLED", "0") or "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if raw_pass_mode:
            text = str(answer or "").strip()
            if not text:
                return ""
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = re.sub(r"(?is)<\|channel[^>]*>", "", text).strip()
            text = re.sub(r"(?im)^\s*(?:answer|response|final answer|답변|최종 답변)\s*[:：]\s*", "", text).strip()
            text = re.sub(r"(?:,\s*){4,}", ", ", text).strip(" ,")
            # If the model already provides actionable content, drop leading
            # "lack-of-info" disclaimers that make normal chat feel blocked.
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if lines:
                first = lines[0]
                lack_info_preamble = bool(
                    re.search(r"(알지 못해서|정보가 부족|추천해\s*드리기\s*어렵)", first)
                )
                has_actionable_body = bool(
                    re.search(r"(?m)^\s*(?:\d+\.\s+|[-*]\s+)", text)
                    or ("추천" in text and len(lines) >= 2)
                )
                if lack_info_preamble and has_actionable_body:
                    text = "\n".join(lines[1:]).strip()
            # Strip stray Korean honorific/particle fragments that appear at the very start
            # (e.g. "께서도", "에서도", "으로도") — these are model hallucination artifacts
            # caused by continuing a cut-off previous assistant turn.
            text = re.sub(r"^(?:께서는|께서도|에서도|으로도|부터도|에게도|한테도|로도|도요|이에요|예요)\b\s*", "", text).strip()
            # Strip role labels that still leak through
            text = re.sub(r"(?im)^\s*(?:user|assistant|사용자|어시스턴트)\s*[:：]\s*", "", text).strip()
            # Guard: if the answer is a near-verbatim copy of recent history,
            # find and remove the echoed prefix (up to 120 chars).
            if "\n" in text:
                first_line = text.split("\n")[0].strip()
                if len(first_line) >= 8 and self._is_hard_query_echo(first_line, query):
                    text = "\n".join(text.split("\n")[1:]).strip()
            # Keep model-native wording; only reject obvious hard prompt echo.
            if len(re.sub(r"\s+", "", query or "")) >= 10 and self._is_hard_query_echo(text, query):
                return ""
            return self._normalize_korean_leading_address(text)

        light_postprocess = str(os.getenv("LOCAL_AI_CONVERSATION_LIGHT_POSTPROCESS", "0") or "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if light_postprocess:
            original = str(answer or "").strip()
            text = original
            if not text:
                return ""
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = re.sub(r"(?im)^\s*continuation\s*:\s*", "", text).strip()
            text = re.sub(r"(?is)^```+\s*$", "", text).strip()
            text = re.sub(r"(?is)^```+\s*[a-zA-Z0-9_-]*\s*$", "", text).strip()
            text = re.sub(r"(?im)^:?\s*please\s*provide\s*the\s*text\s*you\s*would\s*like\s*me\s*to\s*continue\.?\s*$", "", text).strip()
            text = re.sub(r"(?i)pleaseprovidethetextyouwouldlikemetocontinue\.?", "", text).strip()
            text = re.sub(r"(?i),?\s*please\s*provide\s*the\s*conversation\s*history[^.!?\n]*", "", text).strip()
            text = re.sub(r"(?i),?pleaseprovidetheconversationhistory[^.!?\n]*", "", text).strip()
            text = re.sub(r"(?is)<\|channel[^>]*>", "", text).strip()
            text = re.sub(r"\[\s*\*\*[^][]{1,80}\*\*\s*\]", "", text).strip()
            text = re.sub(r"\(\s*\*\*[^()]{1,80}\*\*\s*\)", "", text).strip()
            text = re.sub(r"(?im)^\s*(?:answer|response|final answer|답변|최종 답변)\s*[:：]\s*", "", text).strip()
            text = re.sub(r"(?im)^\s*(?:user|assistant|you|a|q)\s*[:：]?\s*", "", text).strip()
            text = re.sub(r"(?i)^\s*amente[\s,:\-]+", "", text).strip()
            text = re.sub(r"(?im)^\s*mente[\s,:\-]+.*$", "", text).strip()
            text = re.sub(r"(?:,\s*){3,}", ", ", text).strip(" ,")
            text = re.sub(r"(?m)^\s*#{2,}\s*.*$", "", text).strip()
            text = re.sub(r"(?is),?\s*\)+\s*;\s*route\s*=\s*engine_[^\\n]*$", "", text).strip()
            text = re.sub(r"(?is)\broute\s*=\s*engine_[^\\n]*$", "", text).strip()
            text = re.sub(r"(?is)\bmemory_guard:[^\\n]*$", "", text).strip()
            text = re.sub(r"(?is)\bmodel_residency=[^\\n]*$", "", text).strip()
            text = re.sub(r"(?im)^\s*지시\s*:\s*위\s*메모리.*$", "", text).strip()
            text = re.sub(r"(?im)^\s*instruction\s*:\s*use\s*this\s*only\s*as\s*hidden\s*context.*$", "", text).strip()
            if response_language == "ko":
                ko_chars = len(re.findall(r"[가-힣]", text))
                latin_words = len(re.findall(r"[A-Za-z]{3,}", text))
                latin_chars = len(re.findall(r"[A-Za-z]", text))
                if latin_words >= 8 and ko_chars < 6:
                    return ""
                if ko_chars == 0 and latin_chars >= 12:
                    return ""
                if ko_chars >= 24 and len(re.findall(r"\s+", text)) <= 1 and len(text) >= 40:
                    spaced = self._heuristic_korean_spacing(text)
                    if spaced:
                        text = spaced
                # If still densely packed Korean, apply a second-pass conservative split
                # around common particles/connectors to avoid no-space blobs.
                if ko_chars >= 24 and len(re.findall(r"\s+", text)) <= 1 and len(text) >= 40:
                    text = re.sub(r"(은|는|이|가|을|를|에|에서|으로|와|과|도|만|의)([가-힣]{2,})", r"\1 \2", text)
                    text = re.sub(r"(고|며|면|서|라서|지만)([가-힣]{2,})", r"\1 \2", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            if not text:
                # prevent empty-output regression by returning a minimally cleaned original
                fallback = re.sub(r"(?is)<\|channel[^>]*>", "", original).strip()
                fallback = re.sub(r"(?im)^\s*(?:answer|response|final answer|답변|최종 답변)\s*[:：]\s*", "", fallback).strip()
                fallback = re.sub(r"\s{2,}", " ", fallback).strip()
                if response_language == "ko" and fallback:
                    ko_chars = len(re.findall(r"[가-힣]", fallback))
                    latin_chars = len(re.findall(r"[A-Za-z]", fallback))
                    if ko_chars == 0 and latin_chars >= 12:
                        return ""
                    if ko_chars >= 24 and len(re.findall(r"\s+", fallback)) <= 1 and len(fallback) >= 40:
                        fallback = self._heuristic_korean_spacing(fallback)
                return self._normalize_korean_leading_address(fallback[:320].strip())
            return self._normalize_korean_leading_address(text)

        minimal_mode = str(os.getenv("LOCAL_AI_MINIMAL_POSTPROCESS", "0") or "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if minimal_mode:
            text = (answer or "").strip()
            if not text:
                return ""
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = re.sub(r"(?is)```+\s*generationresponse\([^`]*```+", "", text).strip()
            text = re.sub(r"(?is)generationresponse\([^)]*\)", "", text).strip()
            text = re.sub(r"(?im)^\s*token\s*=\s*\d+\s*$", "", text).strip()
            text = re.sub(r"(?im)^\s*from_draft\s*=\s*(?:true|false)\s*$", "", text).strip()
            text = re.sub(r"(?im)^\s*prompt_tokens\s*=\s*\d+\s*$", "", text).strip()
            text = re.sub(r"(?i)\bfrom_draft\s*=\s*(?:true|false)\b", "", text).strip()
            text = re.sub(r"(?i)\bprompt_tokens\s*=\s*\d+\b", "", text).strip()
            text = re.sub(r"(?i)\bprompt_tps\s*=\s*[0-9.]+\b", "", text).strip()
            text = re.sub(r"(?i)\bgeneration_tokens\s*=\s*\d+\b", "", text).strip()
            text = re.sub(r"(?i)\bgeneration_tps\s*=\s*[0-9.]+\b", "", text).strip()
            text = re.sub(r"(?i)\bpeak_memory\s*=\s*[0-9.]+\b", "", text).strip()
            text = re.sub(r"(?i)\bfinish_reason\s*=\s*['\"]?[a-z_]+['\"]?\)?", "", text).strip()
            text = re.sub(r"(?i)\bcontinuation\s*:\s*", "", text).strip()
            text = re.sub(r"(?i)\brecent_user\s*:\s*[^\\n]{0,160}", "", text).strip()
            text = re.sub(r"(?i)\brecent_assistant\s*:\s*[^\\n]{0,160}", "", text).strip()
            text = re.sub(r"(?i)\blast_query\s*:\s*[^\\n]{0,160}", "", text).strip()
            text = re.sub(r"(?i)\bnt_user\s*:\s*[^\\n]{0,120}", "", text).strip()
            text = re.sub(r"(?i)\bnt_assistant\s*:\s*[^\\n]{0,120}", "", text).strip()
            text = re.sub(r"(?i)\b사용자\s*메시지\s*:\s*", "", text).strip()
            text = re.sub(r"(?is)<\|channel[^>]*>", "", text).strip()
            text = re.sub(r"(?is)<followup_hint>.*?(?:</followup_hint>|$)", "", text).strip()
            text = re.sub(r"(?i)\b(?:a|an|nswer|answer)\s*[:：]\s*", "", text).strip()
            text = re.sub(r"(?i)\bnt_u\b", "", text).strip()
            text = re.sub(r"(?im)^\s*(?:answer|response|final answer|답변|최종 답변)\s*[:：]\s*", "", text).strip()
            text = re.sub(r"(?im)^\s*(?:user|assistant|you|a|q)\s*[:：]?\s*", "", text).strip()
            text = re.sub(r"(?is)<\|channel\|?>\s*(?:thought|analysis|final)?\s*", "", text).strip()
            text = re.sub(r"(?is)<\|start_header_id\|>.*?<\|end_header_id\|>\s*", "", text).strip()
            text = re.sub(r"(?i)^\s*amente[\s,:\-]+\s*", "", text).strip()
            # Avoid over-filtering for short casual turns (e.g., greetings).
            # Hard echo rejection stays for longer queries only.
            if len(re.sub(r"\s+", "", query or "")) >= 10 and self._is_hard_query_echo(text, query):
                return ""
            # Guard against runaway repetition loops from the model.
            if self._has_pathological_repetition(text):
                text = self._compress_repetition_fallback(text, max_tokens=24)
            text = re.sub(r"(?:답변\s*[:：]\s*){2,}", "답변: ", text)
            text = re.sub(r"(?:사용자\s*메시지\s*:\s*){2,}", "", text)
            text = re.sub(r"(?:안녕\s*){6,}", "안녕하세요.", text)
            text = re.sub(r"[؟]{2,}", "?", text)
            text = re.sub(r"(?:,\s*){4,}", "", text).strip()
            punct_only = re.sub(r"[\s,.;:!?\-_/\\|()\[\]{}'\"`~]+", "", text)
            if not punct_only:
                return ""
            # Guard against character-level loops like "거나거나..." or "녕녕녕...".
            text = re.sub(r"([가-힣]{1,2})\1{6,}", r"\1", text)
            # Guard against punctuation loops like "( ) ) ) ) ...".
            text = re.sub(r"(?:\(\s*\)\s*){6,}", "", text).strip()
            text = re.sub(r"(?:\)\s*){10,}", "", text).strip()
            # Collapse trivial same-line loops (e.g., emoji repeated per line).
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) >= 4 and len(set(lines)) == 1:
                text = lines[0]
            # Enforce Korean when Korean was requested/query is Korean.
            if response_language == "ko" and re.search(r"[가-힣]", query or ""):
                ko_chars = len(re.findall(r"[가-힣]", text))
                ja_chars = len(re.findall(r"[\u3040-\u30ff]", text))
                en_words = len(re.findall(r"[A-Za-z]{3,}", text))
                if ja_chars >= 6 and ko_chars <= ja_chars:
                    return ""
                if en_words >= 8 and ko_chars < 6:
                    return ""
            text = re.sub(r"\s{2,}", " ", text).strip()
            return self._normalize_korean_leading_address(text)

        text = (answer or "").strip()
        if not text:
            return ""
        text = self._strip_reasoning_leak(text)
        if not text:
            return ""
        text = self._strip_speaker_prefixes(text)
        if self._looks_code_heavy_answer(text):
            text = self._sanitize_code_heavy_answer(text)
            text = re.sub(r"(?im)^\s*(?:최종\s*답변|final\s*answer)\s*[:：]\s*", "", text).strip()
            return text
        text = re.sub(r"\.\s*입니다\.$", ".", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"(?im)^\s*(?:최종\s*답변|final\s*answer)\s*[:：]\s*", "", text).strip()
        text = re.sub(r"(?i)\bokay,\s*let\s*me\s*process\s*this\.?\s*", "", text).strip()
        text = re.sub(r"(?i)\bthat'?s\s*straightforward\.?\s*", "", text).strip()
        meta_regex = r"(?im)\b(?:단,\s*)?(?:사용자에게\s*물어볼\s*때는\s*반드시\s*['\"“”]?\?['\"“”]?\s*를?\s*붙여주세요|사용자의\s*질문에\s*대한\s*답이\s*명확하지\s*않을\s*경우\s*추가로\s*설명해\s*주세요|사용자의\s*질문에\s*대한\s*명확한\s*답변(?:이)?\s*필요할\s*경우\s*3\s*문장까지\s*가능(?:합니다|해요)|사용자의?\s*질문에\s*대한\s*(?:답변|답이)\s*(?:부족|충분하지\s*않|명확하지\s*않)[^.!?\n]{0,100}\s*추가(?:적인)?\s*(?:질문|설명)[^.!?\n]{0,100}(?:가능(?:합니다|해요)|할\s*수\s*있습니다|주세요))\.?\s*"
        text = re.sub(meta_regex, "", text).strip()
        text = re.sub(r"(?im)^\s*(?:recent\s*session\s*context|최근\s*세션\s*컨텍스트)\s*[:：].*$", "", text).strip()
        text = re.sub(r"(?im)\(\s*이전\s*문장에\s*대한\s*답변으로[^)]*\)\s*", "", text).strip()
        text = re.sub(r"(?im)\b이전\s*문장에\s*대한\s*답변으로[^.!?\n]{0,160}\.?\s*", "", text).strip()
        text = re.sub(r"(?im)^\s*사용자에게\s*(?:직접\s*)?(?:도움을?|답변을?|질문을?)\s*(?:주세요|주십시오|제공하세요)\.?\s*", "", text).strip()
        text = re.sub(r"(?im)^\s*(?:사용자의?\s*(?:말|질문|요청|메시지)|사용자\s*메시지)에\s*(?:바로\s*반응하|명확한\s*답(?:변)?을?\s*하)(?:세요|하십시오)\.?\s*", "", text).strip()
        text = re.sub(r"(?i)\bokay,\s*[.!,]*\s*$", "", text).strip()
        text = re.sub(r"(?i)^okay,\s*i'?ll\s*go\s*with\s*that\s*response\.?\s*", "", text).strip()
        text = re.sub(r"(?i)\b(?:okay,\s*let'?s\s*see|alright,\s*let\s*me|alright,\s*something\s*like|hmm,|wait,)\b.*", "", text).strip()
        text = re.sub(r"(?i)\b(?:user|assistant|you)\s*:\s*.*", "", text).strip()
        text = re.sub(r"(?im)^(?:최종 답변 규칙|final response rule):.*$", "", text).strip()
        text = re.sub(r"최대한\s*짧고\s*명확하게\s*답하세요\.?\s*", "", text).strip()
        text = self._dedupe_conversation_sentences(text)
        text = self._limit_question_sentences(text, max_questions=1)
        if not text:
            return ""

        if response_language == "ko" and re.search(r"[가-힣]", query):
            ko_chars = len(re.findall(r"[가-힣]", text))
            en_words = len(re.findall(r"[A-Za-z]{3,}", text))
            if en_words >= 6 and ko_chars < 10:
                return ""
        if self._looks_instructional_meta_response(text):
            return ""

        lowered_query = (query or "").lower()
        is_greeting = any(token in lowered_query for token in ("안녕", "hello", "hi", "hey"))
        if not is_greeting:
            return self._normalize_korean_leading_address(text)

        segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not segments:
            return self._normalize_korean_leading_address(text)
        limited = segments[:2]
        joined = " ".join(limited).strip()
        if response_language == "ko" and len(joined) > 130:
            joined = limited[0]
        return self._normalize_korean_leading_address(joined.strip())

    def _looks_conversational_answer(self, text: str, *, response_language: str, query: str) -> bool:
        content = (text or "").strip()
        if not content:
            return False
        lowered = content.lower()
        blocked = (
            "user:", "assistant:", "you:", "a:", "q:", "follow-up question:",
            "okay, let's see", "okay, i'll go with that response", "okay let me",
            "alright, let me", "alright, something like", "i should", "i need to",
            "the user", "let's think", "my reasoning", "thought:",
            "recent session context", "최근 세션 컨텍스트", "세션 컨텍스트",
            "이전 문장에 대한 답변으로", "mode:", "question:", "최종 답변 규칙",
            "final response rule", "짧고 명확하게 답하세요", "1~3문장", "1-3문장",
            "한 번만 물어보세요", "한번만 물어보세요", "ask at most one follow-up question",
            "keep response to 1-3 sentences", "최종 답변:", "사용자에게 물어볼 때는",
            "반드시 '?'를 붙여주세요", "답이 명확하지 않을 경우", "추가로 설명해 주세요",
            "okay, let me process this", "that's straightforward", "명확한 답변이 필요할 경우",
            "3문장까지 가능합니다", "답변이 부족할 경우", "추가적인 질문", "덧붙일 수 있습니다",
            "사용자에게 직접 도움을 주세요", "사용자에게 직접 도움", "사용자에게 직접",
            "사용자에게 도움을 주세요", "사용자의 말에 바로 반응하세요",
            "사용자의 질문에 바로 반응하세요", "사용자 메시지에 바로 반응하세요",
            "사용자 메시지에 명확한 답을 하세요", "사용자 메시지에 명확한 답변을 하세요",
        )
        if any(token in lowered for token in blocked):
            return False
        if self._looks_instructional_meta_response(content):
            return False
        lowered_query = (query or "").strip().lower()
        is_greeting_query = any(token in lowered_query for token in ("안녕", "hello", "hi", "hey", "thanks", "thank you"))
        is_brief_query = self._is_brief_chat_query(query)
        compact_query = re.sub(r"\s+", "", query or "")
        min_length = 2 if (is_greeting_query or is_brief_query) else (5 if len(compact_query) <= 14 else 9)
        if not self._looks_model_answer(content, min_length=min_length):
            return False

        if response_language == "ko" and re.search(r"[가-힣]", query):
            ko_chars = len(re.findall(r"[가-힣]", content))
            en_words = len(re.findall(r"[A-Za-z]{3,}", content))
            min_ko_chars = 1 if is_brief_query else 4
            if ko_chars < min_ko_chars or (en_words >= 8 and ko_chars <= (en_words * 2)):
                return False
        return True

    def _contains_context_leak_phrase(self, text: str) -> bool:
        lowered = (text or "").lower()
        leak_terms = (
            "recent session context", "최근 세션 컨텍스트", "세션 컨텍스트",
            "이전 문장에 대한 답변으로", "user:", "input message:",
        )
        return any(term in lowered for term in leak_terms)

    def _is_informal_korean_tone(self, text: str) -> bool:
        value = (text or "").strip()
        if not value: return False
        sentences = [seg.strip() for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", value) if seg.strip() and re.search(r"[가-힣]", seg)]
        if not sentences: return False
        polite_end = re.compile(r"(?:요|니다|습니다|세요|까요|군요|네요|입니다|이에요|예요)\s*[.!?]?$")
        polite_count = sum(1 for seg in sentences if polite_end.search(re.sub(r"\s+", " ", seg).strip()))
        return (len(sentences) - polite_count) >= max(1, (len(sentences) + 1) // 2)

    def _has_duplicate_sentences(self, text: str) -> bool:
        parts = [seg.strip() for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", text or "") if seg.strip()]
        if len(parts) < 2: return False
        seen = set()
        for seg in parts:
            key = re.sub(r"[^\w가-힣]+", "", seg).casefold()
            if not key: continue
            if key in seen: return True
            seen.add(key)
        return False

    def _text_similarity(self, a: str, b: str) -> float:
        a_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (a or "").lower()))
        b_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (b or "").lower()))
        if not a_tokens or not b_tokens: return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        union = len(a_tokens.union(b_tokens))
        return inter / union if union > 0 else 0.0

    def _sanitize_generated_answer(self, raw: str, *, prompt: str) -> str:
        text = (raw or "").strip()
        if not text: return ""
        if prompt and text.startswith(prompt): text = text[len(prompt):].strip()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(?im)^(?:answer|final answer|response|답변|최종 답변)\s*[:：]\s*", "", text).strip()
        text = re.sub(r"(?im)\b(?:answer|response|final answer|최종 답변)\s*[:：]\s*", "", text).strip()
        text = self._strip_reasoning_leak(text)
        if not text: return ""
        if self._looks_code_heavy_answer(text):
            return self._sanitize_code_heavy_answer(text)
        segments = [seg.strip(" \t-•") for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+|(?:\s+-\s+))", text) if seg.strip(" \t-•")]
        if not segments:
            compact = re.sub(r"\s+", " ", text).strip()
            if not compact: return ""
            segments = [compact]
        deduped = []
        seen_keys = set()
        prev_key = ""
        for seg in segments:
            key = re.sub(r"[^\w가-힣]+", "", seg).lower()
            if not key or key == prev_key or key in seen_keys: continue
            if any(self._near_duplicate(seg, prior) for prior in deduped): continue
            seen_keys.add(key)
            deduped.append(seg)
            prev_key = key
        compact_segments = self._remove_repeated_blocks(deduped)
        if not compact_segments: return ""
        normalized = " ".join(compact_segments).strip()
        if not normalized: return ""
        normalized = self._cap_by_sentence(compact_segments, max_chars=1200)
        if not normalized:
            first = re.sub(r"\s+", " ", compact_segments[0]).strip()
            normalized = first[:1200].rstrip() + ("..." if len(first) > 1200 else "")
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()
        normalized = re.sub(r"\(\s*[^()\n]{1,60}\.(?:txt|md|markdown|pdf|docx|py|swift|json|ya?ml)\s*\)\s*", "", normalized, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"^(?:좋아|좋아요|알겠어|알겠어요|알겠습니다|오케이|okay|alright)[,!\.\s]+", "", normalized, flags=re.IGNORECASE).strip()
        normalized = self._collapse_repeated_phrase_runs(normalized)
        if self._has_pathological_repetition(normalized):
            normalized = self._compress_repetition_fallback(normalized, max_tokens=28)
        if normalized and not (normalized.endswith("습니다") or normalized.endswith("입니다")) and normalized[-1] not in {".", "!", "?", "다", "요"}:
            normalized += "."
        return normalized

    def _looks_code_heavy_answer(self, text: str) -> bool:
        value = str(text or "")
        if not value.strip():
            return False
        if "```" in value or "'''" in value:
            return True
        code_patterns = (
            r"(?m)^\s*(?:def|class|for|while|if|elif|else|try|except|with|return|import|from)\b",
            r"\bprint\s*\(",
            r"\btarget\s*-\s*\w+",
            r"\benumerate\s*\(",
        )
        return any(re.search(pattern, value) for pattern in code_patterns)

    def _sanitize_code_heavy_answer(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"(?im)^\s*(?:answer|final answer|response|답변|최종 답변)\s*[:：]\s*", "", value).strip()
        value = re.sub(r"(?m)^([ \t]{0,3})'''[ \t]*([A-Za-z0-9_+.-]+)?[ \t]*$", r"\1```\2", value)
        value = re.sub(r"```([A-Za-z0-9_+.-]+)[ \t]+([^\n`][^\n]*)", r"```\1\n\2", value)
        value = re.sub(r"```[ \t]+([^\n`][^\n]*)", r"```\n\1", value)
        value = re.sub(
            r"(?m)^\(\s*[^()\n]{1,60}\.(?:txt|md|markdown|pdf|docx|py|swift|json|ya?ml)\s*\)\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        if value.count("```") % 2 == 1:
            value = value.rstrip() + "\n```"
        value = re.sub(r"\n{4,}", "\n\n\n", value).strip()
        return value

    def _strip_reasoning_leak(self, text: str) -> str:
        if not text: return ""
        content = text.strip()
        lowered = content.lower()
        leak_markers = (
            "<think>", "</think>", "thinking process:", "analyze the request",
            "<|channel|>", "<|channel>", "<|start_header_id|>", "<|end_header_id|>",
            "constraint 1:", "constraint 2:", "constraint 3:",
            "do not output system logs", "never role-play both user and assistant",
            "role-play both user and assistant", "start with the answer directly",
            "okay, let's see", "okay, i'll go with that response", "hmm,", "i should",
            "follow-up question:", "recent session context",
            "최근 세션 컨텍스트", "세션 컨텍스트", "이전 문장에 대한 답변으로",
            "user:", "assistant:", "a:", "q:", "mode:", "_continuation",
            "evidence:", "explanation:", "the question asks", "based on the evidence provided",
            "therefore, the answer is", "(more)", "최종 답변:", "사용자에게 물어볼 때는",
            "반드시 '?'를 붙여주세요", "답이 명확하지 않을 경우", "추가로 설명해 주세요",
            "okay, let me process this", "that's straightforward", "명확한 답변이 필요할 경우",
            "3문장까지 가능합니다", "답변이 부족할 경우", "추가적인 질문", "덧붙일 수 있습니다",
            "사용자에게 직접 도움을 주세요", "사용자에게 직접 도움", "사용자에게 직접",
            "사용자에게 도움을 주세요", "사용자의 말에 바로 반응하세요",
            "사용자의 질문에 바로 반응하세요",
        )
        if not any(marker in lowered for marker in leak_markers): return content

        content = re.sub(r"(?is)<think>.*?</think>", "", content).strip()
        if re.search(r"(?i)<think>", content) and not re.search(r"(?i)</think>", content):
            head, tail = re.split(r"(?is)<think>", content, maxsplit=1)
            if head.strip():
                content = head.strip()
            else:
                content = tail.strip()
        content = re.sub(r"(?is)</?think>", "", content).strip()

        cut_match = re.search(r"(?i)(okay,\s*let'?s\s*see|alright,\s*let\s*me|alright,\s*something\s*like|hmm,|wait,|the user asked|i should|i need to)", content)
        if cut_match and cut_match.start() > 0:
            prefix = content[:cut_match.start()].strip()
            prefix = re.sub(r"(?im)\b(?:user|assistant)\s*[:：]\s*", "", prefix).strip()
            prefix = re.sub(r"(?im)\bfollow-up question:\s*.*", "", prefix).strip()
            prefix = re.sub(r"(?im)\bmode:\s*[A-Z_]+\s*", "", prefix).strip()
            prefix = re.sub(r"\s{2,}", " ", prefix).strip(" -:\n")
            if prefix and len(prefix) >= 8: return prefix

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        cleaned_lines = []
        for line in lines:
            low = line.lower()
            if low.startswith("user:"): continue
            if low.startswith("assistant:"):
                line = re.sub(r"(?im)^\s*assistant\s*[:：]\s*", "", line).strip()
                low = line.lower()
                if not line: continue
            if low.startswith(("you:", "a:", "q:")):
                line = re.sub(r"(?im)^\s*(?:you|a|q)\s*[:：]\s*", "", line).strip()
                low = line.lower()
                if not line: continue
            if any(m in low for m in ("follow-up question:", "recent session context", "최근 세션 컨텍스트", "세션 컨텍스트", "이전 문장에 대한 답변으로")): continue
            if "okay, i'll go with that response" in low:
                line = re.sub(r"(?i)okay,\s*i'?ll\s*go\s*with\s*that\s*response\.?\s*", "", line).strip()
                if not line: continue
            if "okay, let's see" in low or low.startswith(("hmm", "wait,", "i should", "the user asked", "_continuation", "(more)", "evidence:", "explanation:", "input message:", "the question asks", "based on the evidence provided", "therefore, the answer is", "okay, let me process this", "that's straightforward")): continue
            
            line = re.sub(r"(?im)^\s*(?:최종\s*답변|final\s*answer)\s*[:：]\s*", "", line).strip()
            line = re.sub(r"(?im)^\s*[*-]?\s*(?:thinking\s*process\s*:|analyze\s+the\s+request\s*:|constraint\s*[1-3]\s*:)\s*", "", line).strip()
            line = re.sub(
                r"(?i)\b(?:do\s*not\s*output\s*system\s*logs|do\s*not\s*role-?play\s*both\s*user\s*and\s*assistant|never\s*role-?play\s*both\s*user\s*and\s*assistant|role-?play\s*both\s*user\s*and\s*assistant|start\s*with\s*the\s*answer\s*directly)\b\s*[:.]?\s*",
                "",
                line,
            ).strip()
            line = re.sub(r"(?im)\b(?:단,\s*)?(?:사용자에게\s*물어볼\s*때는\s*반드시\s*['\"“”]?\?['\"“”]?\s*를?\s*붙여주세요|사용자의\s*질문에\s*대한\s*답이\s*명확하지\s*않을\s*경우\s*추가로\s*설명해\s*주세요|사용자의\s*질문에\s*대한\s*명확한\s*답변(?:이)?\s*필요할\s*경우\s*3\s*문장까지\s*가능(?:합니다|해요)|사용자의?\s*질문에\s*대한\s*(?:답변|답이)\s*(?:부족|충분하지\s*않|명확하지\s*않)[^.!?\n]{0,100}\s*추가(?:적인)?\s*(?:질문|설명)[^.!?\n]{0,100}(?:가능(?:합니다|해요)|할\s*수\s*있습니다|주세요)|사용자에게\s*(?:직접\s*)?(?:도움을?|답변을?|질문을?)\s*(?:주세요|주십시오|제공하세요)|(?:사용자의?\s*(?:말|질문|요청)|사용자\s*메시지)에\s*(?:바로\s*반응하|명확한\s*답(?:변)?을?\s*하)(?:세요|하십시오))\.?\s*", "", line).strip()
            line = re.sub(r"(?i)\bokay,\s*[.!,]*\s*$", "", line).strip()
            line = re.sub(r"(?im)^mode:\s*[A-Z_]+\s*", "", line).strip()
            line = re.sub(r"(?im)\bquestion:\s*", "", line).strip()
            if not line or self._looks_instructional_meta_response(line): continue
            cleaned_lines.append(line)
        return self._strip_speaker_prefixes("\n".join(cleaned_lines).strip())

    def _strip_speaker_prefixes(self, text: str) -> str:
        if not text: return ""
        cleaned = re.sub(r"(?im)^\s*(?:user|assistant|you|질문|답변|assistant answer|ai)\s*[:：]\s*", "", text).strip()
        cleaned = re.sub(r"(?im)^\s*(?:a|q)\s*[:：]\s*", "", cleaned).strip()
        return cleaned

    def _dedupe_conversation_sentences(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw: return ""
        parts = [seg.strip() for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", raw) if seg.strip()]
        if not parts: return raw
        deduped = []
        prev_key = ""
        for seg in parts:
            key = re.sub(r"[^\w가-힣]+", "", seg).lower()
            if not key or key == prev_key: continue
            if deduped and self._near_duplicate(seg, deduped[-1]): continue
            deduped.append(seg)
            prev_key = key
        return " ".join(deduped).strip()

    def _looks_model_answer(self, text: str, *, min_length: int = 12) -> bool:
        content = (text or "").strip()
        if not content: return False
        if re.fullmatch(r"(?i)(assistant|user)[.!?]?", content): return False
        if self._looks_instructional_meta_response(content) or self._has_pathological_repetition(content): return False
        lowered = content.lower()
        error_signals = (
            "engine failed", "런타임", "모델 경로", "설치되어 있지", "설치 실패",
            "no relevant evidence", "_continuation", "evidence:", "explanation:",
            "recent session context", "최근 세션 컨텍스트", "이전 문장에 대한 답변으로",
            "the question asks", "based on the evidence provided", "therefore, the answer is",
            "ask at most one follow-up question", "keep response to 1-3 sentences",
            "명확한 답변이 필요할 경우", "3문장까지 가능합니다", "답변이 부족할 경우",
            "추가적인 질문", "덧붙일 수 있습니다", "사용자에게 직접 도움을 주세요",
            "사용자에게 도움을 주세요", "사용자의 말에 바로 반응하세요",
            "사용자의 질문에 바로 반응하세요", "사용자 메시지",
        )
        if any(token in lowered for token in error_signals): return False
        return len(content) >= max(2, int(min_length))

    def _prepare_evidence_lines(self, citations: list[Citation], max_items: int, response_language: str) -> list[str]:
        lines = []
        seen = []
        for citation in citations:
            snippet = self._normalize_evidence_snippet(citation.snippet)
            if not snippet or any(self._near_duplicate(snippet, prior) for prior in seen): continue
            seen.append(snippet)
            name = Path(citation.file_path).name or "source.txt"
            lines.append(f"- ({name}) {snippet}")
            if len(lines) >= max_items: break
        if lines: return lines
        return ["- 근거 문장을 찾지 못했습니다."] if response_language == "ko" else ["- No evidence snippet was available."]

    def _normalize_evidence_snippet(self, snippet: str) -> str:
        text = re.sub(r"\s+", " ", (snippet or "").strip())
        if not text: return ""
        text = self._collapse_repeated_phrase_runs(text)
        if self._has_pathological_repetition(text):
            text = self._compress_repetition_fallback(text, max_tokens=22)
        if len(text) > 220:
            text = text[:220].rsplit(" ", 1)[0].rstrip() + "..."
        return self._normalize_korean_leading_address(text.strip())

    def _looks_instructional_meta_response(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered: return False
        explicit_markers = (
            "<think>", "</think>", "thinking process:", "analyze the request",
            "constraint 1:", "constraint 2:", "constraint 3:",
            "do not output system logs", "never role-play both user and assistant",
            "role-play both user and assistant", "start with the answer directly",
            "ask at most one follow-up question", "keep response to 1-3 sentences",
            "한 번만 물어보세요", "한번만 물어보세요", "1~3문장으로", "1-3문장으로",
            "사용자에게 물어볼 때는", "반드시 '?'를 붙여주세요", "답이 명확하지 않을 경우",
            "추가로 설명해 주세요", "okay, let me process this", "that's straightforward",
            "명확한 답변이 필요할 경우", "3문장까지 가능합니다", "답변이 부족할 경우",
            "추가적인 질문", "덧붙일 수 있습니다", "사용자에게 직접 도움을 주세요",
            "사용자에게 도움을 주세요", "사용자의 말에 바로 반응하세요",
            "사용자의 질문에 바로 반응하세요", "recent session context",
            "최근 세션 컨텍스트", "세션 컨텍스트", "이전 문장에 대한 답변으로",
        )
        if any(marker in lowered for marker in explicit_markers): return True
        if "사용자의 질문" in lowered and "문장" in lowered and ("답변" in lowered or "답이" in lowered): return True
        meta_regexes = (
            r"사용자의?\s*질문에\s*대한\s*(?:답변|답이).{0,50}(?:부족|충분하지|명확하지).{0,80}(?:추가적인?\s*(?:질문|설명)|질문을?\s*덧붙)",
            r"사용자에게\s*(?:직접\s*)?(?:도움을?|답변을?|질문을?)\s*(?:주세요|주십시오|제공하세요)",
            r"(?:사용자의?\s*(?:말|질문|요청)|사용자\s*메시지)에\s*바로\s*반응하(?:세요|십시오)",
            r"사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)",
            r"(?:thinking\s*process|analyze\s+the\s+request|constraint\s*[123])",
        )
        if any(re.search(rx, lowered) for rx in meta_regexes): return True
        return ("물어보세요" in lowered and "답하세요" in lowered) or ("ask" in lowered and "respond" in lowered and "sentence" in lowered)

    def _is_brief_chat_query(self, query: str) -> bool:
        raw = (query or "").strip()
        if not raw: return False
        lowered = raw.lower()
        ack_tokens = (
            "그렇구나", "알겠", "오케이", "그래", "아하", "맞아", "ㅇㅋ", "ok", "okay",
            "got it", "makes sense", "cool", "thanks", "thank you",
        )
        if any(token in lowered for token in ack_tokens): return True
        if len(re.sub(r"\s+", "", raw)) <= 8: return True
        return re.fullmatch(r"[0-9+\-*/().=\s?]+", raw) is not None

    def _is_recommendation_chat_query(self, query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered: return False
        task_cues = ("파일", "문서", "요약", "정리해", "찾아", "열어", "비교", "주차", "file", "document", "summary", "summarize", "find", "open", "compare")
        if any(token in lowered for token in task_cues): return False
        recommendation_cues = (
            "추천", "메뉴", "뭐 먹", "어때", "골라", "선택", "뭐가 좋아", "뭐가 나아", "뭐할까", "어떤 게 좋아",
            "결정해", "정해줘", "정해 봐", "pick one", "choose for me",
            "recommend", "suggest", "what should i", "which one", "choice",
        )
        return any(token in lowered for token in recommendation_cues)

    def _near_duplicate(self, a: str, b: str) -> bool:
        norm_a = re.sub(r"\s+", " ", a).strip().lower()
        norm_b = re.sub(r"\s+", " ", b).strip().lower()
        if not norm_a or not norm_b: return False
        if norm_a == norm_b: return True
        if (norm_a in norm_b or norm_b in norm_a) and min(len(norm_a), len(norm_b)) >= 24: return True
        tokens_a = set(re.findall(r"[a-z0-9가-힣]+", norm_a))
        tokens_b = set(re.findall(r"[a-z0-9가-힣]+", norm_b))
        if not tokens_a or not tokens_b: return False
        overlap = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return union > 0 and (overlap / union) >= 0.88

    def _remove_repeated_blocks(self, segments: list[str]) -> list[str]:
        if len(segments) < 4: return segments
        output = []
        seen = set()
        for seg in segments:
            key = re.sub(r"[^\w가-힣]+", "", seg).lower()
            if key in seen: continue
            seen.add(key)
            output.append(seg)
        return output

    def _collapse_repeated_phrase_runs(self, text: str) -> str:
        content = (text or "").strip()
        if not content: return ""
        collapsed = re.sub(r"(?i)\b([A-Za-z0-9가-힣_]+)(?:\s+\1){3,}\b", r"\1", content)
        chunk_pattern = re.compile(r"(?P<chunk>(?:\S+\s+){1,7}\S+)(?:\s+(?P=chunk)){1,}")
        for _ in range(4):
            updated = chunk_pattern.sub(lambda m: m.group("chunk"), collapsed)
            if updated == collapsed: break
            collapsed = updated
        return re.sub(r"\s{2,}", " ", collapsed).strip()

    def _has_pathological_repetition(self, text: str) -> bool:
        content = (text or "").strip()
        if not content: return False
        tokens = re.findall(r"[A-Za-z0-9가-힣_]+", content.lower())
        if len(tokens) < 20: return False
        n = 4
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not grams: return False
        counts = {}
        for gram in grams: counts[gram] = counts.get(gram, 0) + 1
        top = max(counts.values(), default=0)
        return top >= 5 or (top / len(grams)) >= 0.24

    def _compress_repetition_fallback(self, text: str, *, max_tokens: int) -> str:
        content = re.sub(r"\s+", " ", (text or "").strip())
        if not content: return ""
        tokens = re.findall(r"\S+", content)
        if len(tokens) <= max_tokens: return content
        shortened = " ".join(tokens[:max_tokens]).strip()
        if shortened and shortened[-1] not in {".", "!", "?", "다", "요"}: shortened += "."
        return shortened

    def _cap_by_sentence(self, segments: list[str], *, max_chars: int) -> str:
        selected = []
        length = 0
        for seg in segments:
            proposed = length + (1 if selected else 0) + len(seg)
            if proposed > max_chars: break
            selected.append(seg)
            length = proposed
        return " ".join(selected).strip()
