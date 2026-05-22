from __future__ import annotations

import re
from pathlib import Path

from ..models import Citation, ReasoningIntent
from .composer_trace_helpers import ComposerTraceHelpers


class ComposerSummaryHelpers:
    _FENCED_CODE_PATTERN = re.compile(r"```[^\n`]*\n[\s\S]*?```")
    _KOREAN_ENDING_REGEX = r"(습니다|입니다|해요|해줘요|해줄게요|할게요|이에요|예요|됐어요|됐습니다)[.!?]?$"

    @staticmethod
    def _truncate(text: str, *, max_chars: int) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        limit = max(8, int(max_chars))
        if len(value) <= limit:
            return value
        clipped = value[:limit].rstrip()
        # Prefer trimming at a natural sentence boundary near the tail.
        boundary = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        if boundary >= max(20, int(limit * 0.55)):
            clipped = clipped[: boundary + 1].rstrip()
        return clipped

    @staticmethod
    def _is_noisy_generated_text(text: str) -> bool:
        compact = " ".join((text or "").split()).lower()
        if not compact:
            return True
        if "console.log" in compact or "#include" in compact:
            return True
        if compact.count("{") + compact.count("}") >= 8:
            return True
        # Relaxed link limit for richer web responses
        if len(compact) > 800 and compact.count("http://") + compact.count("https://") >= 5:
            return True
        tokens = re.findall(r"[A-Za-z0-9가-힣_]+", compact)
        if len(tokens) >= 60:
            unique_ratio = len(set(tokens)) / max(1, len(tokens))
            if unique_ratio < 0.22:
                return True
        return False

    @staticmethod
    def _naturalize_summary_text(
        *,
        summary: str,
        query: str,
        response_language: str,
        result_type: str,
        intent: ReasoningIntent,
        allow_unverified_urls: bool = True,
    ) -> str:
        text = (summary or "").strip()
        if not text:
            return ""
        if result_type == "file_list":
            return text

        text, code_blocks = ComposerSummaryHelpers._stash_fenced_code_blocks(text)
        text = ComposerSummaryHelpers._strip_instruction_leakage(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = ComposerSummaryHelpers._collapse_repeated_word_chunks(text)
        text = ComposerSummaryHelpers._dedupe_sentence_lines(
            text,
            aggressive=(result_type != "conversation"),
        )

        if intent == ReasoningIntent.SUMMARIZE_FILE:
            looks_like_points = bool(re.search(r"(?m)^\s*1\.\s+", text))
            if not looks_like_points:
                text = ComposerSummaryHelpers._format_summary_points(
                    summary=text,
                    query=query,
                    response_language=response_language,
                )

        if ComposerSummaryHelpers._is_noisy_generated_text(text):
            clauses = ComposerSummaryHelpers._extract_summary_point_candidates(text, response_language=response_language)
            if not clauses:
                clauses = ComposerSummaryHelpers._extract_clause_candidates(text, response_language=response_language)
            if clauses:
                text = "\n".join(f"{idx}. {item}" for idx, item in enumerate(clauses[:5], start=1))

        if not allow_unverified_urls:
            text = ComposerSummaryHelpers._remove_plain_urls(text)
            text = ComposerSummaryHelpers._strip_unverified_lookup_claims(text)
        text = ComposerSummaryHelpers._linkify_urls(text)
        text = ComposerSummaryHelpers._medium_paragraph_wrap(text)
        text = ComposerSummaryHelpers._restore_fenced_code_blocks(text, code_blocks)

        return text.strip()

    @staticmethod
    def _stash_fenced_code_blocks(text: str) -> tuple[str, dict[str, str]]:
        value = str(text or "")
        if "```" not in value:
            return value, {}
        blocks: dict[str, str] = {}

        def repl(match: re.Match[str]) -> str:
            token = f"__CODE_BLOCK_{len(blocks)}__"
            blocks[token] = match.group(0)
            return token

        protected = ComposerSummaryHelpers._FENCED_CODE_PATTERN.sub(repl, value)
        return protected, blocks

    @staticmethod
    def _restore_fenced_code_blocks(text: str, blocks: dict[str, str]) -> str:
        restored = str(text or "")
        if not blocks:
            return restored
        for token, block in blocks.items():
            restored = restored.replace(token, block)
        return restored

    @staticmethod
    def _medium_paragraph_wrap(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        paragraphs = re.split(r"\n{2,}", value)
        output: list[str] = []
        for raw in paragraphs:
            paragraph = raw.strip()
            if not paragraph:
                continue
            if "__CODE_BLOCK_" in paragraph:
                output.append(paragraph)
                continue
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if any(re.match(r"^\s*(?:[-*•#]|\d+[.)])\s*", line) for line in lines):
                output.append("\n".join(lines))
                continue
            compact = " ".join(lines)
            compact = re.sub(r"[ \t]{2,}", " ", compact).strip()
            if not compact:
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", compact) if s.strip()]
            if len(sentences) >= 6 and len(compact) >= 60:
                grouped: list[str] = []
                # Group by 4 sentences for a more natural GPT-like paragraph feel
                for idx in range(0, len(sentences), 4):
                    grouped.append(" ".join(sentences[idx : idx + 4]).strip())
                output.append("\n\n".join(grouped))
            else:
                output.append(compact)
        return "\n\n".join(output).strip()

    @staticmethod
    def _remove_plain_urls(text: str) -> str:
        if not text:
            return ""
        stripped = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1", text)
        stripped = re.sub(r"<https?://[^>\s]+>", "", stripped)
        stripped = re.sub(r"(?<!\]\()https?://[^\s<>()]+", "", stripped)
        stripped = re.sub(r"\(\s*\)", "", stripped)
        stripped = re.sub(r"\s{2,}", " ", stripped)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()

    @staticmethod
    def _strip_unverified_lookup_claims(text: str) -> str:
        if not text:
            return ""
        stripped = text
        patterns = (
            r"(?im)\b(?:확인|찾아)\s*(?:해|해서)?\s*보(?:겠습니다|겠어요|겠습니다만|겠습니다\.)\b[^.!?\n]*[.!?]?",
            r"(?im)\b(?:확인|찾아)\s*(?:해|해서)\s*볼(?:게요|까요)\b[^.!?\n]*[.!?]?",
            r"(?im)\b(?:잠시만|잠깐)\s*기다(?:려|려주세요|려\s*주세요)\b[^.!?\n]*[.!?]?",
            r"(?im)\b(?:let\s*me\s*check|i'?ll\s*check|hold\s*on)\b[^.!?\n]*[.!?]?",
        )
        for pattern in patterns:
            stripped = re.sub(pattern, "", stripped)
        stripped = re.sub(r"\s{2,}", " ", stripped)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()

    @staticmethod
    def _linkify_urls(text: str) -> str:
        if not text:
            return ""

        def repl(match: re.Match[str]) -> str:
            raw = match.group(0)
            url = raw
            suffix = ""
            while url and url[-1] in ".,;!?)]":
                suffix = url[-1] + suffix
                url = url[:-1]
            if not url:
                return raw
            return f"<{url}>{suffix}"

        return re.sub(r"(?<!\]\()https?://[^\s<>()]+", repl, text)

    @staticmethod
    def _strip_instruction_leakage(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        # Remove leaked role markers and policy-like lines.
        cleaned = re.sub(r"(?i)\buser\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\bassistant\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\bfollow-?up question\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\bthink step by step\b[:：]?\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\bokay,\s*let'?s see\.?", "", cleaned).strip()
        cleaned = re.sub(r"(?i)ask at most one follow-up question\.?", "", cleaned).strip()
        cleaned = re.sub(r"(?i)keep response to 1-3 sentences\.?", "", cleaned).strip()
        cleaned = re.sub(r"최대한\s*한\s*번만\s*물어보세요\.?", "", cleaned).strip()
        cleaned = re.sub(r"최대한\s*1[-~]\s*3문장으로만\s*답하세요\.?", "", cleaned).strip()
        cleaned = re.sub(r"(?im)^\s*(?:최종\s*답변|final\s*answer)\s*[:：]\s*", "", cleaned).strip()
        cleaned = re.sub(
            r"(?im)\b사용자에게\s*물어볼\s*때는\s*반드시\s*['\"“”]?\?['\"“”]?\s*를?\s*붙여주세요\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)\b(?:단,\s*)?사용자의?\s*질문에\s*대한\s*(?:답변|답이)\s*(?:부족|충분하지\s*않|명확하지\s*않)[^.!?\n]{0,120}\s*추가(?:적인)?\s*(?:질문|설명)[^.!?\n]{0,120}(?:가능(?:합니다|해요)|할\s*수\s*있습니다|주세요)?\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)^\s*사용자에게\s*(?:직접\s*)?(?:도움을?|답변을?|질문을?)\s*(?:주세요|주십시오|제공하세요)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)^\s*사용자\s*메시지에\s*바로\s*반응하(?:세요|십시오)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)^\s*사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)\b사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)^\s*(?:좋아|좋아요|알겠어|알겠어요|알겠습니다|응)\s*,?\s*(?:이|그)?\s*맥락[^.!?\n]{0,60}(?:볼게|해볼게|정리해볼게|이어서\s*볼게)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(r"(?i)\bokay,\s*[.!,]*\s*$", "", cleaned).strip()
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    @staticmethod
    def _collapse_repeated_word_chunks(text: str) -> str:
        words = (text or "").split()
        if len(words) < 12:
            return text
        out: list[str] = []
        i = 0
        min_chunk = 3
        max_chunk = 12
        min_repeats = 3
        while i < len(words):
            matched = False
            max_try = min(max_chunk, (len(words) - i) // min_repeats)
            for size in range(max_try, min_chunk - 1, -1):
                chunk = words[i : i + size]
                repeats = 1
                while i + (repeats + 1) * size <= len(words):
                    next_chunk = words[i + repeats * size : i + (repeats + 1) * size]
                    if next_chunk != chunk:
                        break
                    repeats += 1
                if repeats >= min_repeats:
                    out.extend(chunk)
                    i += size * repeats
                    matched = True
                    break
            if not matched:
                out.append(words[i])
                i += 1
        return " ".join(out).strip()

    @staticmethod
    def _dedupe_sentence_lines(text: str, *, aggressive: bool = True) -> str:
        if not text:
            return ""
        normalized = re.sub(r"\s+(?=\d+\.\s+)", "\n", text)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        is_numbered = all(re.match(r"^\d+\.\s+", line) for line in lines) if lines else False
        is_bulleted = all(re.match(r"^\s*[-*•]\s+", line) for line in lines) if lines else False
        if is_numbered:
            deduped_points: list[str] = []
            for line in lines:
                value = re.sub(r"^\d+\.\s+", "", line).strip()
                if any(ComposerSummaryHelpers._is_near_duplicate_point(value, prior) for prior in deduped_points):
                    continue
                deduped_points.append(value)
            return "\n".join(f"{idx}. {item}" for idx, item in enumerate(deduped_points, start=1)).strip()
        if is_bulleted:
            deduped_points: list[str] = []
            for line in lines:
                value = re.sub(r"^\s*[-*•]\s+", "", line).strip()
                if any(ComposerSummaryHelpers._is_near_duplicate_point(value, prior) for prior in deduped_points):
                    continue
                deduped_points.append(value)
            return "\n".join(f"- {item}" for item in deduped_points).strip()

        parts = [
            seg.strip()
            for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", text)
            if seg.strip()
        ]
        deduped: list[str] = []
        seen_keys: set[str] = set()
        for seg in parts:
            normalized = re.sub(r"\s{2,}", " ", seg).strip()
            if len(normalized) < 8:
                continue
            key = re.sub(r"[^\w가-힣]+", "", normalized).casefold()
            if not key:
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if aggressive and any(ComposerSummaryHelpers._is_near_duplicate_point(normalized, prior) for prior in deduped):
                continue
            deduped.append(normalized)
        if not deduped:
            return text.strip()
        joined = " ".join(deduped).strip()
        return re.sub(r"\s{2,}", " ", joined).strip()

    @staticmethod
    def _candidate_summary_from_citations(
        *,
        citations: list[Citation],
        confidence: float,
        response_language: str,
    ) -> str:
        names: list[str] = []
        seen: set[str] = set()
        for citation in citations:
            name = Path(citation.file_path).name
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= 3:
                break
        if response_language == "ko":
            uncertainty = ComposerTraceHelpers.render_uncertainty(
                confidence=confidence,
                ambiguity_type="candidate",
                risk_level="low",
                response_language=response_language,
            )
            if names:
                return f"{uncertainty} 우선 {', '.join(names)} 순서로 확인하면 가장 빠르게 맞출 수 있어요."
            return "완전히 확실하진 않지만, 지금은 이쪽이 가장 가능성이 높아 보여요. 파일명/주차/태그를 붙이면 더 정확해져요."
        if names:
            return (
                "This is not fully certain yet, but these are the most likely candidates right now: "
                f"{', '.join(names)}."
            )
        return "This is still ambiguous. Please add one more clue such as file name, year, or tag."

    @staticmethod
    def _normalize_korean_summary(summary: str, *, conversation_mode: bool = False, query: str = "") -> str:
        text = (summary or "").strip()
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]{2,}", " ", segment).strip() for segment in text.split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            return ""
        compact = "\n".join(lines).strip()
        compact = re.sub(r"\n{3,}", "\n\n", compact)
        if conversation_mode:
            if ComposerSummaryHelpers._is_recommendation_chat_query(query):
                compact = compact.strip()
            else:
                trimmed = ComposerSummaryHelpers._trim_conversation_summary(compact, query=query)
                if trimmed:
                    compact = trimmed
        # Avoid unfinished, clipped endings.
        if compact and "\n" not in compact and not re.search(ComposerSummaryHelpers._KOREAN_ENDING_REGEX, compact):
            if compact[-1] not in {".", "!", "?"}:
                compact += "."
        return compact

    @staticmethod
    def _format_summary_points(*, summary: str, query: str, response_language: str) -> str:
        text = (summary or "").strip()
        if not text:
            return ""
        desired = ComposerSummaryHelpers._requested_point_count(query=query)
        candidates = ComposerSummaryHelpers._extract_summary_point_candidates(text, response_language=response_language)
        if not candidates:
            return text
        if len(candidates) < desired:
            extra = ComposerSummaryHelpers._extract_clause_candidates(text, response_language=response_language)
            for item in extra:
                if item in candidates:
                    continue
                candidates.append(item)
                if len(candidates) >= desired:
                    break
        points = candidates[:desired]
        if not points:
            return text
        lines = [f"{idx}. {item}" for idx, item in enumerate(points, start=1)]
        return "\n".join(lines).strip()

    @staticmethod
    def _requested_point_count(*, query: str) -> int:
        text = (query or "").lower()
        match = re.search(r"([3-7])\s*(?:줄|개|포인트|문장|lines?|points?)", text)
        if match:
            return max(3, min(7, int(match.group(1))))
        return 5

    @staticmethod
    def _extract_summary_point_candidates(text: str, *, response_language: str) -> list[str]:
        compact = text.replace("\r\n", "\n").replace("\r", "\n")
        compact = re.sub(r"\(\s*[^()\n]{1,60}\.(?:txt|md|markdown|pdf|docx|py|swift|json|ya?ml)\s*\)", " ", compact, flags=re.IGNORECASE)
        compact = re.sub(r"\b\d{1,2}[:：]\d{2}\b", "\n", compact)
        compact = re.sub(r"(?m)^\s*\d{1,2}[:：]\d{2}\s*", "", compact)
        compact = re.sub(r"(?m)^\s*\d+\s*[\.\)]\s*", "", compact)
        compact = re.sub(r"[•·■◆▶]", "\n", compact)
        compact = re.sub(r"\s{2,}", " ", compact).strip()
        if not compact:
            return []
        parts = [
            seg.strip(" \t-")
            for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+|;\s+)", compact)
            if seg.strip(" \t-")
        ]
        output: list[str] = []
        seen: set[str] = set()
        for seg in parts:
            line = re.sub(r"^\d+\s*[\.\)]\s*", "", seg).strip()
            line = re.sub(r"(?i)^(?:좋아|좋아요|알겠어|알겠어요|알겠습니다|오케이|okay|alright)\s*,?\s*", "", line).strip()
            line = re.sub(r"\s{2,}", " ", line).strip(" .")
            if len(line) < 10:
                continue
            key = re.sub(r"[^\w가-힣]+", "", line).lower()
            if not key or key in seen:
                continue
            if any(ComposerSummaryHelpers._is_near_duplicate_point(line, prior) for prior in output):
                continue
            seen.add(key)
            if response_language == "ko":
                line = ComposerSummaryHelpers._trim_point_length(line, max_chars=90)
            else:
                line = ComposerSummaryHelpers._trim_point_length(line, max_chars=120)
            output.append(line)
        return output

    @staticmethod
    def _extract_clause_candidates(text: str, *, response_language: str) -> list[str]:
        clauses = [
            seg.strip(" \t-")
            for seg in re.split(r"[,\n]", text or "")
            if seg.strip(" \t-")
        ]
        output: list[str] = []
        for clause in clauses:
            line = re.sub(r"\(\s*[^()\n]{1,60}\.(?:txt|md|markdown|pdf|docx|py|swift|json|ya?ml)\s*\)", "", clause, flags=re.IGNORECASE)
            line = re.sub(r"\b\d{1,2}[:：]\d{2}\b", " ", line)
            line = re.sub(r"\s{2,}", " ", line).strip(" .")
            if len(line) < 10:
                continue
            if response_language == "ko":
                line = ComposerSummaryHelpers._trim_point_length(line, max_chars=90)
            else:
                line = ComposerSummaryHelpers._trim_point_length(line, max_chars=120)
            if line and line not in output:
                output.append(line)
        return output

    @staticmethod
    def _is_near_duplicate_point(a: str, b: str) -> bool:
        normalized_a = re.sub(r"\b\d{1,2}[:：]\d{2}\b", " ", a.lower())
        normalized_b = re.sub(r"\b\d{1,2}[:：]\d{2}\b", " ", b.lower())
        normalized_a = re.sub(r"\b\d+\b", " ", normalized_a)
        normalized_b = re.sub(r"\b\d+\b", " ", normalized_b)
        tokens_a = set(re.findall(r"[A-Za-z0-9가-힣_]+", normalized_a))
        tokens_b = set(re.findall(r"[A-Za-z0-9가-힣_]+", normalized_b))
        if not tokens_a or not tokens_b:
            return False
        overlap = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        if union <= 0:
            return False
        return (overlap / union) >= 0.82

    @staticmethod
    def _trim_point_length(text: str, *, max_chars: int) -> str:
        value = (text or "").strip()
        if len(value) <= max_chars:
            return value
        head = value[:max_chars]
        cut = head.rsplit(" ", 1)[0].strip()
        if not cut:
            cut = head.strip()
        return cut + "..."

    @staticmethod
    def _trim_conversation_summary(summary: str, *, query: str) -> str:
        if not summary:
            return ""
        text = ComposerSummaryHelpers._strip_instruction_leakage(summary)
        text = re.sub(
            r"(?im)^\s*(?:사용자의?\s*(?:말|질문|요청)|사용자\s*메시지)에\s*바로\s*반응하(?:세요|십시오)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)^\s*사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(r"\s{2,}", " ", text).strip()
        if not text:
            return ""

        sentence_candidates = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentence_candidates:
            return text[:180].strip()

        lowered_query = (query or "").lower()
        greeting_query = any(token in lowered_query for token in ("안녕", "hello", "hi", "hey"))
        max_sentences = 2 if greeting_query else 6
        deduped: list[str] = []
        seen: set[str] = set()
        for sentence in sentence_candidates:
            key = re.sub(r"[^\w가-힣]+", "", sentence).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(sentence)
            if len(deduped) >= max_sentences:
                break

        output = " ".join(deduped).strip()
        if greeting_query and len(output) > 160:
            output = deduped[0] if deduped else ComposerSummaryHelpers._truncate(output, max_chars=160)
        elif not greeting_query and len(output) > 900:
            output = ComposerSummaryHelpers._truncate(output, max_chars=900)
        return output.strip()

    @staticmethod
    def _looks_question_sentence(sentence: str) -> bool:
        value = str(sentence or "").strip()
        if not value:
            return False
        if "?" in value:
            return True
        return re.search(r"(까요|나요|인가요|어때요|어떨까요|할까요|될까요)\s*[.!?]?$", value) is not None

    @staticmethod
    def _question_sentence_count(text: str) -> int:
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
                if ComposerSummaryHelpers._looks_question_sentence(sentence):
                    count += 1
        return count

    @staticmethod
    def _enforce_direct_first_summary(*, summary: str, response_language: str) -> str:
        text = str(summary or "").strip()
        if not text:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
        if not sentences:
            return text
        first = sentences[0]
        if not ComposerSummaryHelpers._looks_question_sentence(first):
            return text
        for idx, sentence in enumerate(sentences[1:], start=1):
            if ComposerSummaryHelpers._looks_question_sentence(sentence):
                continue
            reordered = [sentence] + [s for i, s in enumerate(sentences) if i != idx]
            candidate = " ".join(reordered).strip()
            if response_language == "ko" and candidate and candidate[-1] not in {".", "!", "?"}:
                candidate += "."
            return candidate
        return text

    @staticmethod
    def _is_recommendation_chat_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        task_cues = (
            "파일",
            "문서",
            "요약",
            "정리해",
            "찾아",
            "열어",
            "비교",
            "주차",
            "file",
            "document",
            "summary",
            "summarize",
            "find",
            "open",
            "compare",
        )
        if any(token in lowered for token in task_cues):
            return False
        recommendation_cues = (
            "추천",
            "메뉴",
            "뭐 먹",
            "어때",
            "골라",
            "선택",
            "뭐가 좋아",
            "뭐가 나아",
            "뭐할까",
            "어떤 게 좋아",
            "recommend",
            "suggest",
            "what should i",
            "which one",
            "choice",
        )
        return any(token in lowered for token in recommendation_cues)

    @staticmethod
    def _normalize_recommendation_three_options(*, summary: str, response_language: str) -> str:
        text = str(summary or "").strip()
        if not text:
            return ""
        numbered = re.findall(r"(?m)^\s*[1-3]\.\s+.+$", text)
        if len(numbered) >= 3:
            return "\n".join([line.strip() for line in numbered[:3]]).strip()

        items: list[str] = []
        seen: set[str] = set()
        inline_numbered = re.findall(
            r"(?:^|\s)([1-3])\.\s*([^0-9]+?)(?=(?:\s[1-3]\.\s)|$)",
            text,
        )
        for _, raw in inline_numbered:
            item = re.sub(r"\s+", " ", str(raw or "").strip()).strip(" .")
            if len(item) < 6:
                continue
            key = re.sub(r"[^\w가-힣]+", "", item).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(item)
            if len(items) >= 3:
                break
        candidates = re.findall(r"(?m)^\s*(?:\d+[.)]|[-•·])\s*(.+?)\s*$", text)
        if not candidates:
            candidates = [
                seg.strip()
                for seg in re.split(r"(?<=[.!?。！？])\s+|,\s+|\n+", text)
                if seg.strip()
            ]
        for raw in candidates:
            item = re.sub(r"\s+", " ", str(raw or "").strip()).strip(" .")
            if len(item) < 6:
                continue
            key = re.sub(r"[^\w가-힣]+", "", item).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(item)
            if len(items) >= 3:
                break
        if len(items) < 3:
            return text
        lines = [f"{idx}. {item}" for idx, item in enumerate(items[:3], start=1)]
        if response_language == "ko":
            return "\n".join(lines).strip()
        return "\n".join(lines).strip()
