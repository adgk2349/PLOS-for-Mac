from __future__ import annotations
import re
import json
import os
import importlib
import importlib.util
from typing import Any, Literal, TYPE_CHECKING

from ..base import BaseDelegate
from ..types import InferenceResult, _ConversationCandidateResult
from ...models import LocalEngine, WorkMode, Citation

if TYPE_CHECKING:
    from ...local_inference import LocalInferenceEngine

# Component: conversational_logic.py
class ConversationalLogic(BaseDelegate):
    @staticmethod
    def _looks_meta_only_promise_answer(text: str) -> bool:
        value = re.sub(r"\s+", " ", str(text or "").strip())
        if not value:
            return False
        trimmed = re.sub(r"^(?:그럼|그러면|좋아요|알겠습니다|네|자|이제)\s*[,.! ]*", "", value)
        if re.search(r"[:\n]", trimmed):
            return False
        if len(re.findall(r"[.!?]", trimmed)) > 1:
            return False
        return bool(
            re.match(
                r"^(?:한\s*번에\s*)?(?:바로\s*)?(?:한\s*줄로\s*|한\s*문장으로만\s*|간단히\s*|짧게\s*|본문만\s*)?"
                r"(?:정리|요약|설명|답변|말씀|말해|출력|안내|추천)(?:해\s*|해드리\s*|드리\s*)?"
                r"(?:볼게요|할게요|드릴게요|해드릴게요)\.?\s*$",
                trimmed,
            )
        )

    @staticmethod
    def _looks_user_role_confusion_answer(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        lowered = value.lower()
        if lowered.startswith("오늘은") and lowered.endswith("?"):
            if re.search(r"(추천해\s*줄래요|알려\s*줄래요|정해\s*줄래요|해\s*줄래요|해줄래요)\??$", value):
                return True
        if re.search(r"(추천해\s*줄래요|알려\s*줄래요|정해\s*줄래요|해\s*줄래요|해줄래요)\??$", value):
            return True
        return False
    @staticmethod
    def _is_action_request_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        cues = (
            "해줘", "해 줘", "해봐", "나눠줘", "뽑아줘", "남겨줘", "정리해줘", "만들어줘",
            "해주세요", "please", "give me", "do this",
        )
        return any(token in lowered for token in cues)
    @staticmethod
    def _looks_truncated_conversation_answer(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        if re.search(r"(?m)(?:^|[\n ])(?:\*\*)?(?:\d+\.|[-*•])\s*$", value):
            return True
        if re.search(r"(?:\*\*)?(?:\d+\.)\s*$", value):
            return True
        if value.count("**") % 2 == 1:
            return True
        if re.search(r"[:;,(\[{`\-]\s*$", value):
            return True
        return False

    def _continue_conversation_once(
        self,
        *,
        engine: LocalEngine,
        query: str,
        draft_answer: str,
        response_language: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
    ) -> str | None:
        prompt = (
            "Continue only the unfinished tail.\n"
            "Do not restart and do not repeat previous text.\n"
            f"Language: {response_language}\n"
            f"User query: {query}\n"
            f"Current answer: {draft_answer}\n"
            "Continuation:"
        )
        raw = self._generate_with_engine(
            engine=engine,
            prompt=prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=min(96, max(24, int(max_tokens * 0.4))),
            style="rewrite",
        )
        if not raw:
            return None
        tail = self._postprocess_conversational_answer(
            raw,
            query=query,
            response_language=response_language,
        ).strip()
        if not tail:
            return None
        base = str(draft_answer or "").rstrip()
        base = re.sub(r"(?:\s*)(?:\*\*)?(?:\d+\.)\s*$", "", base).rstrip()
        base = re.sub(r"(?:\s*)[:;,(\[{`\-]\s*$", "", base).rstrip()
        merged = f"{base} {tail}".strip()
        merged = re.sub(r"\s{2,}", " ", merged).strip()
        return merged

    def _generate_conversation_candidate(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        query: str,
        response_language: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
        allow_repair_fallbacks: bool = True,
        message_state: list[dict[str, str]] | None = None,
    ) -> _ConversationCandidateResult:
        is_recommendation_query = self._is_recommendation_chat_query(query)
        try:
            raw = self._generate_with_engine(
                engine=engine,
                prompt=prompt,
                profile=profile,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                max_tokens=max_tokens,
                style="conversation",
                message_state=message_state,
                response_language=response_language,
            )
        except TypeError as exc:
            if "unexpected keyword argument" in str(exc) and "message_state" in str(exc):
                raw = self._generate_with_engine(
                    engine=engine,
                    prompt=prompt,
                    profile=profile,
                    mlx_model_path=mlx_model_path,
                    llama_model_path=llama_model_path,
                    max_tokens=max_tokens,
                    style="conversation",
                    response_language=response_language,
                )
            else:
                raise
        if not raw:
            err = self._last_engine_error.get(engine, f"{engine.value} engine failed")
            self._set_engine_error(
                engine,
                f"{engine.value} conversational response invalid (attempt1:{err})",
            )
            return _ConversationCandidateResult(answer=None)

        previous_user_text = ""
        if message_state:
            user_seen = 0
            for item in reversed(message_state):
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "").strip().lower()
                if role == "user":
                    user_seen += 1
                    if user_seen < 2:
                        continue
                    content = str(item.get("content") or "").strip()
                    if content:
                        previous_user_text = content
                        break

        answer = self._postprocess_conversational_answer(
            raw,
            query=query,
            response_language=response_language,
        )
        if answer and not self._looks_meta_only_promise_answer(answer) and self._looks_truncated_conversation_answer(answer):
            continued = self._continue_conversation_once(
                engine=engine,
                query=query,
                draft_answer=answer,
                response_language=response_language,
                profile=profile,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                max_tokens=max_tokens,
            )
            if continued:
                answer = continued
        raw_preview = re.sub(r"\s+", " ", str(raw or "")).strip()[:140]
        quality_issues = self._conversation_quality_issues(
            query=query,
            answer=answer,
            response_language=response_language,
        ) if answer else ["empty_after_sanitize"]
        if (
            answer
            and previous_user_text
            and previous_user_text.strip() != str(query or "").strip()
        ):
            sim_prev = self._text_similarity(answer, previous_user_text)
            sim_curr = self._text_similarity(answer, query)
            if sim_prev >= 0.86 and sim_curr <= 0.48:
                quality_issues.append("stale_user_echo")
        
        hard_issues = self._conversation_hard_issues(quality_issues)
        # Accept model output unless it hits hard safety/style failures.
        # Soft quality issues are tracked but should not hard-fail generation.
        is_valid_answer = bool(answer and not hard_issues)
        streaming_active = self._is_streaming_active()
        # Model-native first: if there is no hard issue, keep the answer as-is.
        if is_valid_answer:
            return _ConversationCandidateResult(
                answer=answer,
                rewrite_used=False,
                repair_triggered=False,
                repair_success=False,
                leak_blocked=False,
                direct_first_applied=True,
                question_count_after_postprocess=self._question_sentence_count(answer),
                recommendation_shape=(
                    "three_options"
                    if is_recommendation_query and self._looks_three_option_shape(answer)
                    else None
                ),
            )

        leak_blocked = bool(
            any(issue in {"meta_leak", "context_leak"} for issue in hard_issues)
            or self._looks_instructional_meta_response(str(raw or ""))
            or self._contains_context_leak_phrase(str(raw or ""))
        )
        repair_reason = "|".join(quality_issues[:6])
        repaired_answer: str | None = None
        rewrite_used = False
        repaired_issues: list[str] = []
        repaired_valid = False
        
        # Even when static fallbacks are disabled, allow a model-based rewrite
        # if the primary output is invalid (empty/hard issues). This avoids
        # failing fast into repeated runtime-error fallback text.
        rewrite_enabled = str(os.getenv("LOCAL_AI_CONVERSATION_REWRITE_ENABLED", "1") or "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        allow_model_rewrite = bool(
            rewrite_enabled
            and (allow_repair_fallbacks or not is_valid_answer)
            and (not streaming_active or bool(hard_issues))
        )
        if allow_model_rewrite:
            if response_language == "ko":
                repaired_answer = self._rewrite_korean_conversation_answer(
                    engine=engine,
                    query=query,
                    draft_answer=answer or raw,
                    profile=profile,
                    mlx_model_path=mlx_model_path,
                    llama_model_path=llama_model_path,
                    max_tokens=max_tokens,
                )
                rewrite_used = repaired_answer is not None
            else:
                repaired_answer = self._rewrite_conversation_answer(
                    engine=engine,
                    query=query,
                    draft_answer=answer or raw,
                    response_language=response_language,
                    profile=profile,
                    mlx_model_path=mlx_model_path,
                    llama_model_path=llama_model_path,
                    max_tokens=max_tokens,
                )

        if repaired_answer:
            if not self._looks_meta_only_promise_answer(repaired_answer) and self._looks_truncated_conversation_answer(repaired_answer):
                continued = self._continue_conversation_once(
                    engine=engine,
                    query=query,
                    draft_answer=repaired_answer,
                    response_language=response_language,
                    profile=profile,
                    mlx_model_path=mlx_model_path,
                    llama_model_path=llama_model_path,
                    max_tokens=max_tokens,
                )
                if continued:
                    repaired_answer = continued
            repaired_issues = self._conversation_quality_issues(
                query=query,
                answer=repaired_answer,
                response_language=response_language,
            )
            repaired_valid = bool(repaired_answer and not self._conversation_hard_issues(repaired_issues))
        if response_language == "ko":
            ko_rewrite_triggers = {"language_mismatch", "english_meta_mix", "korean_spacing_degraded"}
            should_force_strict_ko = bool(
                ko_rewrite_triggers.intersection(set(quality_issues))
                or ko_rewrite_triggers.intersection(set(repaired_issues or []))
                or ko_rewrite_triggers.intersection(set(hard_issues))
            )
        else:
            should_force_strict_ko = False
        if should_force_strict_ko:
            strict_ko = self._rewrite_korean_conversation_answer(
                engine=engine,
                query=query,
                draft_answer=repaired_answer or answer or raw,
                profile=profile,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                max_tokens=max_tokens,
            )
            if strict_ko:
                strict_issues = self._conversation_quality_issues(
                    query=query,
                    answer=strict_ko,
                    response_language=response_language,
                )
                if not self._conversation_hard_issues(strict_issues):
                    repaired_answer = strict_ko
                    repaired_issues = strict_issues
                    repaired_valid = True
                    rewrite_used = True

        selected_answer, selected_issues = self._pick_best_conversation_answer(
            primary_answer=answer,
            primary_issues=quality_issues,
            primary_valid=is_valid_answer,
            repaired_answer=repaired_answer,
            repaired_issues=repaired_issues,
            repaired_valid=repaired_valid,
            query=query,
            is_recommendation_query=is_recommendation_query,
        )
        
        if selected_answer:
            final_answer = selected_answer
            # Preserve model-native output without continuation/direct-rewrite overrides.
            # Keep model-native conversational shape; avoid template normalization.
            # Keep model-native conversational shape; do not hard-cap question count.
            question_count = self._question_sentence_count(final_answer)
            recommendation_shape = (
                "three_options"
                if is_recommendation_query and self._looks_three_option_shape(final_answer)
                else None
            )
            return _ConversationCandidateResult(
                answer=final_answer,
                rewrite_used=rewrite_used,
                quality_repair_reason=repair_reason if rewrite_used else None,
                repair_triggered=bool(rewrite_used),
                repair_success=bool(rewrite_used and selected_answer == repaired_answer and not selected_issues),
                leak_blocked=leak_blocked,
                direct_first_applied=True,
                question_count_after_postprocess=question_count,
                recommendation_shape=recommendation_shape,
            )

        # Model-native fallback: if we have a non-empty primary answer and no critical leak,
        # prefer returning it over forcing quality-guard retry loops.
        if answer and not leak_blocked and not hard_issues:
            return _ConversationCandidateResult(
                answer=answer,
                rewrite_used=rewrite_used,
                quality_repair_reason=repair_reason or None,
                repair_triggered=bool(rewrite_used),
                repair_success=False,
                leak_blocked=False,
                direct_first_applied=True,
                question_count_after_postprocess=self._question_sentence_count(answer),
                recommendation_shape=(
                    "three_options"
                    if is_recommendation_query and self._looks_three_option_shape(answer)
                    else None
                ),
            )

        last_resort_enabled = str(
            os.getenv("LOCAL_AI_CONVERSATION_LAST_RESORT_ENABLED", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if last_resort_enabled:
            last_resort = self._generate_last_resort_direct_answer(
                engine=engine,
                query=query,
                response_language=response_language,
                profile=profile,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                max_tokens=max_tokens,
            )
            if last_resort and self._looks_conversational_answer(
                last_resort,
                response_language=response_language,
                query=query,
            ):
                final_answer = last_resort
                question_count = self._question_sentence_count(final_answer)
                recommendation_shape = (
                    "three_options"
                    if is_recommendation_query and self._looks_three_option_shape(final_answer)
                    else None
                )
                return _ConversationCandidateResult(
                    answer=final_answer,
                    rewrite_used=rewrite_used,
                    quality_repair_reason="|".join(
                        [item for item in [repair_reason, "last_resort_direct"] if item][:2]
                    ),
                    repair_triggered=True,
                    repair_success=True,
                    leak_blocked=leak_blocked,
                    direct_first_applied=True,
                    question_count_after_postprocess=question_count,
                    recommendation_shape=recommendation_shape,
                )

        clarification_enabled = str(
            os.getenv("LOCAL_AI_CONVERSATION_CLARIFICATION_FALLBACK_ENABLED", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        disable_template_fallbacks = str(
            os.getenv("LOCAL_AI_DISABLE_TEMPLATE_FALLBACKS", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if disable_template_fallbacks:
            clarification_enabled = False
        if allow_repair_fallbacks and clarification_enabled:
            clarification = self._minimal_conversation_clarification(
                query=query,
                response_language=response_language,
            )
            if clarification and self._looks_conversational_answer(
                clarification,
                response_language=response_language,
                query=query,
            ):
                final_answer = clarification
                return _ConversationCandidateResult(
                    answer=final_answer,
                    rewrite_used=rewrite_used,
                    quality_repair_reason="|".join(
                        [item for item in [repair_reason, "clarification_fallback"] if item][:2]
                    ),
                    repair_triggered=True,
                    repair_success=False,
                    leak_blocked=leak_blocked,
                    direct_first_applied=True,
                    question_count_after_postprocess=self._question_sentence_count(final_answer),
                    recommendation_shape=None,
                )

        self._set_engine_error(
            engine,
            f"{engine.value} conversational response invalid (attempt1:quality_guard_blocked;raw_preview={raw_preview})",
        )
        # If quality guard rejects output, do not return raw preview fragments.
        # Let upper recovery loop retry generation to preserve conversational quality.
        return _ConversationCandidateResult(
            answer=None,
            rewrite_used=rewrite_used,
            quality_repair_reason=repair_reason or "quality_guard_blocked",
            repair_triggered=True,
            repair_success=False,
            leak_blocked=leak_blocked,
        )

    def _conversation_hard_issues(self, issues: list[str]) -> list[str]:
        hard = {
            "meta_leak",
            "context_leak",
            "clarification_template_leak",
            "meta_only_ack",
            "role_confusion",
            "leading_fragment",
            "pathological_repetition",
            "query_echo_hard",
            "stale_user_echo",
            "language_mismatch",
            "avoidable_clarification",
            "continuation_artifact",
            "comma_loop_artifact",
            "truncated_answer",
            "intent_restatement",
        }
        return [item for item in issues if item in hard]

    def _pick_best_conversation_answer(
        self,
        *,
        primary_answer: str | None,
        primary_issues: list[str],
        primary_valid: bool,
        repaired_answer: str | None,
        repaired_issues: list[str],
        repaired_valid: bool,
        query: str,
        is_recommendation_query: bool,
    ) -> tuple[str | None, list[str]]:
        primary_clean = str(primary_answer or "").strip()
        repaired_clean = str(repaired_answer or "").strip()

        def score(answer: str, issues: list[str], valid: bool) -> tuple[int, int, int]:
            hard_count = len(self._conversation_hard_issues(issues))
            soft_count = len(issues)
            recommendation_bonus = 0
            if is_recommendation_query and self._looks_three_option_shape(answer):
                recommendation_bonus = 1
            validity = 1 if valid else 0
            return (validity, recommendation_bonus, -hard_count - soft_count)

        if primary_clean and repaired_clean:
            primary_score = score(primary_clean, primary_issues, primary_valid)
            repaired_score = score(repaired_clean, repaired_issues, repaired_valid)
            if repaired_score > primary_score:
                return repaired_clean, repaired_issues
            if primary_valid:
                return primary_clean, primary_issues
            if repaired_valid:
                return repaired_clean, repaired_issues
            return None, []

        if primary_clean and primary_valid:
            return primary_clean, primary_issues
        if repaired_clean and repaired_valid:
            return repaired_clean, repaired_issues

        return None, []

    def _generate_last_resort_direct_answer(
        self,
        *,
        engine: LocalEngine,
        query: str,
        response_language: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
    ) -> str | None:
        prompt = self._last_resort_direct_reply_prompt(
            query=query,
            response_language=response_language,
        )
        raw = self._generate_with_engine(
            engine=engine,
            prompt=prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=min(max_tokens, 120),
            style="rewrite",
        )
        if not raw:
            return None
        cleaned = self._postprocess_conversational_answer(
            raw,
            query=query,
            response_language=response_language,
        )
        if not cleaned:
            return None
        issues = self._conversation_quality_issues(
            query=query,
            answer=cleaned,
            response_language=response_language,
        )
        if self._conversation_hard_issues(issues):
            return None
        return cleaned

    def _minimal_conversation_clarification(self, *, query: str, response_language: str) -> str:
        lowered = (query or "").strip().lower()
        if response_language == "ko":
            if any(token in lowered for token in ("뭐 먹", "메뉴", "점심", "저녁", "아침", "먹을")):
                return "원하시면 오늘 메뉴를 두 가지로 바로 좁혀드릴까요?"
            if any(token in lowered for token in ("파일", "문서", "요약", "찾아", "주차")):
                return "원하시는 범위를 한 줄로만 알려주실래요? 바로 정확히 찾아드릴게요."
            return "원하시는 방향 한 가지만 알려주실래요? 바로 맞춰서 답해드릴게요."
        
        if any(token in lowered for token in ("eat", "food", "lunch", "dinner", "breakfast")):
            return "If you want, I can narrow this down to two quick meal options."
        if any(token in lowered for token in ("file", "document", "summary", "find")):
            return "Could you give me one short scope hint so I can answer accurately?"
        return "Could you give me one short direction so I can answer precisely?"

    def _rewrite_korean_conversation_answer(
        self,
        *,
        engine: LocalEngine,
        query: str,
        draft_answer: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
    ) -> str | None:
        rewrite_prompt = self._korean_rewrite_prompt(query=query, draft_answer=draft_answer)
        raw = self._generate_with_engine(
            engine=engine,
            prompt=rewrite_prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=max_tokens,
            style="rewrite",
        )
        if not raw:
            return None
        rewritten = self._postprocess_conversational_answer(raw, query=query, response_language="ko")
        if not rewritten:
            return None
        if self._korean_quality_issues(query=query, answer=rewritten, response_language="ko"):
            return None
        return rewritten

    def _rewrite_conversation_answer(
        self,
        *,
        engine: LocalEngine,
        query: str,
        draft_answer: str,
        response_language: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
    ) -> str | None:
        rewrite_prompt = self._conversation_rewrite_prompt(
            query=query,
            draft_answer=draft_answer,
            response_language=response_language,
        )
        raw = self._generate_with_engine(
            engine=engine,
            prompt=rewrite_prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=max_tokens,
            style="rewrite",
        )
        if not raw:
            return None
        rewritten = self._postprocess_conversational_answer(
            raw,
            query=query,
            response_language=response_language,
        )
        if not rewritten:
            return None
        if self._conversation_quality_issues(
            query=query,
            answer=rewritten,
            response_language=response_language,
        ):
            return None
        return rewritten

    def _conversation_quality_issues(self, *, query: str, answer: str, response_language: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", (answer or "").strip())
        if not cleaned:
            return ["empty"]
        issues: list[str] = []
        lowered_clean = cleaned.lower()
        if self._looks_instructional_meta_response(cleaned):
            issues.append("meta_leak")
        if self._contains_context_leak_phrase(cleaned):
            issues.append("context_leak")
        if "continuation:" in lowered_clean or ":pleaseprovidethetextyouwouldlikemetocontinue." in lowered_clean:
            issues.append("continuation_artifact")
        if re.search(r"(?:,\s*){5,}", cleaned):
            issues.append("comma_loop_artifact")
        if self._has_duplicate_sentences(cleaned):
            issues.append("duplicate_sentence")
        if self._has_pathological_repetition(cleaned):
            issues.append("pathological_repetition")
        if len(re.sub(r"\s+", "", query or "")) >= 10:
            if self._text_similarity(cleaned, query) >= 0.86 and len(cleaned) <= 220:
                issues.append("query_echo")
            if self._is_hard_query_echo(cleaned, query):
                issues.append("query_echo_hard")
        if self._looks_leading_fragment(cleaned):
            issues.append("leading_fragment")
        if self._looks_clarification_template_leak(cleaned):
            issues.append("clarification_template_leak")
        if self._looks_avoidable_clarification_answer(query=query, answer=cleaned):
            issues.append("avoidable_clarification")
        if self._looks_meta_only_promise_answer(cleaned):
            issues.append("meta_only_ack")
        if self._looks_truncated_conversation_answer(cleaned):
            issues.append("truncated_answer")
        if self._looks_user_role_confusion_answer(cleaned):
            issues.append("role_confusion")
        if self._looks_intent_restatement_answer(query=query, answer=cleaned):
            issues.append("intent_restatement")
        
        if response_language == "ko":
            ko_chars = len(re.findall(r"[가-힣]", cleaned))
            en_words = len(re.findall(r"[A-Za-z]{3,}", cleaned))
            ja_chars = len(re.findall(r"[\u3040-\u30ff]", cleaned))
            if en_words >= 8 and ko_chars <= (en_words * 2):
                issues.append("english_meta_mix")
            if ja_chars >= 6 and ko_chars <= ja_chars:
                issues.append("language_mismatch")
            # Long Korean text with almost no spaces usually indicates degraded decode.
            if ko_chars >= 24 and len(re.findall(r"\s+", cleaned)) <= 1 and len(cleaned) >= 40:
                issues.append("korean_spacing_degraded")
            if self._is_informal_korean_tone(cleaned):
                issues.append("informal_tone")
        return issues

    @staticmethod
    def _looks_intent_restatement_answer(*, query: str, answer: str) -> bool:
        q = str(query or "").strip()
        a = str(answer or "").strip()
        if not q or not a:
            return False
        lowered = a.lower()
        markers = (
            "원하시는 것 같",
            "알고 싶으신 것 같",
            "추천받고 싶으신 것 같",
            "고민 중이시군요",
            "찾으시는군요",
            "것 같습니다",
        )
        if not any(marker in lowered for marker in markers):
            return False
        if len(re.findall(r"[.!?]", a)) > 2:
            return False
        substantive_cues = ("1.", "2.", "-", "*", "예를", "굽", "방법", "먼저", "다음", "뒤집", "온도", "소금", "후추")
        if any(token in a for token in substantive_cues):
            return False
        return True

    @staticmethod
    def _looks_avoidable_clarification_answer(*, query: str, answer: str) -> bool:
        q = str(query or "").strip().lower()
        a = str(answer or "").strip().lower()
        if not q or not a:
            return False
        direct_task_request = bool(
            any(token in q for token in ("정리해줘", "정리", "요약", "뽑아줘", "추천해줘", "알려줘"))
            and any(token in q for token in ("3개", "세 개", "두 개", "한 줄", "짧게", "간단히"))
        )
        if not direct_task_request:
            return False
        clarification_markers = (
            "알려주시면",
            "말씀해주시면",
            "구체적으로",
            "어떤",
            "무엇을",
            "어떤 종류",
            "더 알려",
            "provide",
            "tell me",
            "which one",
        )
        if any(marker in a for marker in clarification_markers):
            has_answer_shape = bool(re.search(r"(?:^|\n)\s*(?:1[.)]|-|\*)\s+", answer))
            return not has_answer_shape
        return False

    @staticmethod
    def _looks_leading_fragment(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        if re.match(r"^(?:으로는|로는|에는|에서는|와는|과는)\s+", value):
            return True
        return bool(
            re.match(
                r"^(?:께(?:서는|요)?|을|를|이|가|은|는|도)\s*(?:붙여드리겠습니다|도와드리겠습니다|안내해드리겠습니다|질문하신|오늘|집중)",
                value,
            )
        )

    @staticmethod
    def _looks_clarification_template_leak(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        lowered = value.lower()
        has_lack_of_info = ("정보가 부족" in value) or ("구체적인 추천을 드리기 어렵" in value)
        has_numbered_questions = bool(re.search(r"(?:^|\s)1\.\s*\*\*.*2\.\s*\*\*.*3\.\s*\*\*", value))
        has_example_block = ("예시" in value) and ("라고 말씀해주시면" in value or "알려주세요" in value)
        has_meta_request = ("원하시는 답변을 얻으시려면" in value) or ("아래 정보" in value)
        if has_lack_of_info and (has_numbered_questions or has_meta_request):
            return True
        if has_numbered_questions and has_example_block:
            return True
        if "please provide" in lowered and "1." in lowered and "2." in lowered:
            return True
        if "한 번에 바로 본문만 출력할게요" in value or "바로 본문만 출력할게요" in value:
            return True
        if re.match(r"^\s*한\s*번에\s*.*(?:답변|정리).*(?:드릴게요|해드릴게요)\.?\s*$", value):
            return True
        return False

    def _korean_quality_issues(self, *, query: str, answer: str, response_language: str) -> list[str]:
        if response_language != "ko":
            return []
        return self._conversation_quality_issues(
            query=query,
            answer=answer,
            response_language=response_language,
        )

    def _minimal_safe_conversation_answer(self, *, query: str, response_language: str) -> str:
        lowered = (query or "").strip().lower()
        if response_language == "ko":
            if re.search(r"(내일|오늘).*(할\s*일|할일).*(3개|세\s*개)", query):
                return "1. 내일 가장 중요한 일 1개를 먼저 끝내기.\n2. 25분 집중 + 5분 휴식으로 두 번 진행하기.\n3. 마감 10분 전에 결과 점검하고 정리하기."
            if any(token in lowered for token in ("뭐 먹", "메뉴", "점심", "저녁", "아침", "먹을")):
                return "지금은 속이 편한 메뉴 한 가지(국밥, 죽, 비빔밥 중 하나)로 고르는 게 가장 무난해요."
            if any(token in lowered for token in ("몇 시", "몇시", "자야", "수면", "잠")):
                return "지금 패턴 기준으로는 평소보다 30~60분만 앞당겨 자는 게 가장 현실적인 선택이에요."
            return "질문하신 내용에 바로 답하면, 지금은 가장 부담이 적은 선택 하나부터 정해 진행하는 게 좋아요."
        
        if any(token in lowered for token in ("eat", "lunch", "dinner", "breakfast", "food")):
            return "A simple, easy-to-digest meal is the safest choice right now."
        if any(token in lowered for token in ("sleep", "bedtime", "time to sleep")):
            return "A realistic next step is to sleep 30-60 minutes earlier than your usual time."
        return "Direct answer: start with the lowest-effort option first, then adjust from there."

    def _fallback_answer(self, query: str, mode: WorkMode, citations: list[Citation], response_language: str) -> str:
        lang = self._resolve_response_language(query, response_language)
        if not citations:
            if lang == "ko":
                return "선택된 로컬 문서에서 관련 근거를 찾지 못했습니다. 폴더/인덱싱 상태를 확인해 주세요."
            return "No relevant evidence was found in selected local documents. Check folder selection and indexing state."

        snippets = [c.snippet for c in citations[:3]]
        joined = "\n".join(f"- {snippet}" for snippet in snippets)

        if lang == "ko":
            if mode == WorkMode.SUMMARY:
                return f"핵심 요약:\n{joined}"
            if mode == WorkMode.RESEARCH:
                return f"근거 비교 관점에서 정리했습니다:\n{joined}\n\n질문: {query}"
            if mode == WorkMode.DEVELOPMENT:
                return f"개발 관점 단계형 정리:\n1) 문제 맥락 파악\n2) 관련 근거\n{joined}"
            if mode == WorkMode.WRITING:
                return f"글쓰기 초안 재료:\n{joined}"
            if mode == WorkMode.PLANNING:
                return f"기획 관점 액션 아이템:\n{joined}"
            if mode == WorkMode.STRICT_SEARCH:
                return f"근거 기반 응답:\n{joined}"
            return f"로컬 자료 기반 답변:\n{joined}"

        if mode == WorkMode.SUMMARY:
            return f"Key summary:\n{joined}"
        if mode == WorkMode.RESEARCH:
            return f"Evidence comparison summary:\n{joined}\n\nQuestion: {query}"
        if mode == WorkMode.DEVELOPMENT:
            return f"Development-oriented steps:\n1) Understand context\n2) Gather evidence\n{joined}"
        if mode == WorkMode.WRITING:
            return f"Draft materials:\n{joined}"
        if mode == WorkMode.PLANNING:
            return f"Planning action items:\n{joined}"
        if mode == WorkMode.STRICT_SEARCH:
            return f"Evidence-based response:\n{joined}"
        return f"Local source-based answer:\n{joined}"

    def classify_document(
        self,
        *,
        path: str,
        text: str,
        fixed_categories: list[str],
        fallback: dict,
    ) -> dict:
        if self._ensure_mlx_loaded("recommended", explicit_model_path=None):
            prompt = (
                "너는 로컬 문서 분류기다. 반드시 JSON 객체 하나만 출력한다.\n"
                f"category는 다음 중 하나여야 한다: {', '.join(fixed_categories)}\n"
                "JSON schema keys: summary, category, subcategory, document_type, tags, year, project, importance\n"
                f"path: {path}\n"
                f"text: {text[:5000]}"
            )
            try:
                mlx_lm = importlib.import_module("mlx_lm")
                generate = mlx_lm.generate

                raw = generate(self._mlx_model, self._mlx_tokenizer, prompt=prompt, max_tokens=320)
                parsed = self._extract_json_object(raw)
                if parsed:
                    return parsed
            except Exception:
                pass

        return self._fallback_classification(path=path, text=text, fixed_categories=fixed_categories, fallback=fallback)

    def _extract_json_object(self, raw: str) -> dict:
        if not raw:
            return {}
        candidate = raw.strip()
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            snippet = candidate[start : end + 1]
            try:
                value = json.loads(snippet)
                if isinstance(value, dict):
                    return value
            except Exception:
                return {}
        return {}

    def _fallback_classification(self, *, path: str, text: str, fixed_categories: list[str], fallback: dict) -> dict:
        compact = re.sub(r"\s+", " ", text).strip()
        summary = fallback.get("summary") or compact[:220]
        tags = fallback.get("tags") or []
        year = fallback.get("year")
        if year is None:
            match = re.search(r"(19|20)\d{2}", f"{path} {compact[:2000]}")
            if match:
                year = int(match.group(0))

        category = fallback.get("category") or "참고자료"
        if category not in fixed_categories:
            category = "참고자료"

        importance = fallback.get("importance", 0.5)
        try:
            importance = max(0.0, min(1.0, float(importance)))
        except Exception:
            importance = 0.5

        return {
            "summary": str(summary)[:260],
            "category": category,
            "subcategory": str(fallback.get("subcategory") or "")[:40],
            "document_type": str(fallback.get("document_type") or ""),
            "tags": tags[:8] if isinstance(tags, list) else [],
            "year": year,
            "project": fallback.get("project"),
            "importance": importance,
        }
