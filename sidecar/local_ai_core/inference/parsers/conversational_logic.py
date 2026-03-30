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
    ) -> _ConversationCandidateResult:
        is_recommendation_query = self._is_recommendation_chat_query(query)
        raw = self._generate_with_engine(
            engine=engine,
            prompt=prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=max_tokens,
            style="conversation",
        )
        if not raw:
            err = self._last_engine_error.get(engine, f"{engine.value} engine failed")
            self._set_engine_error(
                engine,
                f"{engine.value} conversational response invalid (attempt1:{err})",
            )
            return _ConversationCandidateResult(answer=None)

        answer = self._postprocess_conversational_answer(
            raw,
            query=query,
            response_language=response_language,
        )
        quality_issues = self._conversation_quality_issues(
            query=query,
            answer=answer,
            response_language=response_language,
        ) if answer else ["empty_after_sanitize"]
        
        hard_issues = self._conversation_hard_issues(quality_issues)
        # Accept model output unless it hits hard safety/style failures.
        # Soft quality issues are tracked but should not hard-fail generation.
        is_valid_answer = bool(answer and not hard_issues)
        streaming_active = self._is_streaming_active()
        if is_valid_answer and not quality_issues:
            return _ConversationCandidateResult(answer=answer)

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
        allow_model_rewrite = bool((allow_repair_fallbacks or not is_valid_answer) and not streaming_active)
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
            repaired_issues = self._conversation_quality_issues(
                query=query,
                answer=repaired_answer,
                response_language=response_language,
            )
            repaired_valid = bool(repaired_answer and not self._conversation_hard_issues(repaired_issues))

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
            if is_recommendation_query:
                final_answer = self._normalize_three_option_recommendation(
                    final_answer,
                    response_language=response_language,
                )
            final_answer = self._limit_question_sentences(final_answer, max_questions=1)
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

        if allow_repair_fallbacks and not streaming_active:
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
                final_answer = self._limit_question_sentences(last_resort, max_questions=1)
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

        if allow_repair_fallbacks and not streaming_active:
            clarification = self._minimal_conversation_clarification(
                query=query,
                response_language=response_language,
            )
            if clarification and self._looks_conversational_answer(
                clarification,
                response_language=response_language,
                query=query,
            ):
                final_answer = self._limit_question_sentences(clarification, max_questions=1)
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
            f"{engine.value} conversational response invalid (attempt1:quality_guard_blocked)",
        )
        return _ConversationCandidateResult(
            answer=None,
            rewrite_used=rewrite_used,
            quality_repair_reason=repair_reason or "quality_guard_blocked",
            repair_triggered=True,
            repair_success=False,
            leak_blocked=leak_blocked,
        )

    def _conversation_hard_issues(self, issues: list[str]) -> list[str]:
        hard = {"meta_leak", "context_leak", "pathological_repetition"}
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
        if self._looks_instructional_meta_response(cleaned):
            issues.append("meta_leak")
        if self._contains_context_leak_phrase(cleaned):
            issues.append("context_leak")
        if self._has_duplicate_sentences(cleaned):
            issues.append("duplicate_sentence")
        if self._has_pathological_repetition(cleaned):
            issues.append("pathological_repetition")
        if self._text_similarity(cleaned, query) >= 0.86 and len(cleaned) <= 220:
            issues.append("query_echo")
        
        if response_language == "ko":
            ko_chars = len(re.findall(r"[가-힣]", cleaned))
            en_words = len(re.findall(r"[A-Za-z]{3,}", cleaned))
            if en_words >= 8 and ko_chars <= (en_words * 2):
                issues.append("english_meta_mix")
            if self._is_informal_korean_tone(cleaned):
                issues.append("informal_tone")
        return issues

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
