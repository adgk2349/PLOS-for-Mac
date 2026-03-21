from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..language_utils import (
    insufficient_evidence_message,
    resolve_response_language,
    response_language_instruction,
)
from ..models import Citation, LocalEngine, RuntimePrepareResponse, WorkMode


@dataclass(slots=True)
class LocalInferenceConfig:
    model_path: str | None
    max_tokens: int


@dataclass(slots=True)
class InferenceResult:
    answer: str
    engine_used: LocalEngine
    used_fallback: bool = False
    detail: str | None = None


@dataclass(slots=True)
class _ConversationCandidateResult:
    answer: str | None
    rewrite_used: bool = False
    quality_repair_reason: str | None = None
    repair_triggered: bool = False
    repair_success: bool = False
    leak_blocked: bool = False
    direct_first_applied: bool = False
    question_count_after_postprocess: int = 0
    recommendation_shape: str | None = None


LocalInferenceEngine = None  # patched by local_inference.py
LocalInferenceConfig = None  # patched by local_inference.py
_ConversationCandidateResult = None  # patched by local_inference.py

class LocalInferenceEngineMethodsMixin:
    def _generate_grounded_candidate(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        response_language: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
    ) -> str | None:
        prompt_variants = [
            prompt,
            self._grounded_repair_prompt(prompt, response_language=response_language),
        ]
        attempts: list[str] = []
        for idx, prompt_variant in enumerate(prompt_variants, start=1):
            answer = self._generate_with_engine(
                engine=engine,
                prompt=prompt_variant,
                profile=profile,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                max_tokens=max_tokens,
                style="grounded",
            )
            if answer and self._looks_model_answer(answer):
                return answer
            if answer:
                attempts.append(f"attempt{idx}:filtered")
            else:
                err = self._last_engine_error.get(engine, f"{engine.value} engine failed")
                attempts.append(f"attempt{idx}:{err}")
        if attempts:
            self._set_engine_error(engine, f"{engine.value} grounded response invalid ({'; '.join(attempts)})")
        return None

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
        is_valid_answer = bool(
            answer
            and not hard_issues
            and self._looks_conversational_answer(
                answer,
                response_language=response_language,
                query=query,
            )
        )
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
            repaired_valid = bool(
                not self._conversation_hard_issues(repaired_issues)
                and self._looks_conversational_answer(
                    repaired_answer,
                    response_language=response_language,
                    query=query,
                )
            )

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

    @staticmethod
    def _conversation_hard_issues(issues: list[str]) -> list[str]:
        hard = {"meta_leak", "context_leak", "pathological_repetition"}
        return [item for item in issues if item in hard]

    @staticmethod
    def _pick_best_conversation_answer(
        *,
        primary_answer: str,
        primary_issues: list[str],
        primary_valid: bool,
        repaired_answer: str | None,
        repaired_issues: list[str],
        repaired_valid: bool,
        query: str,
        is_recommendation_query: bool,
    ) -> tuple[str | None, list[str]]:
        if primary_valid and not repaired_valid:
            return primary_answer, primary_issues
        if repaired_valid and not primary_valid:
            return repaired_answer, repaired_issues
        if not primary_valid and not repaired_valid:
            return None, []
        primary_soft = len(primary_issues)
        repaired_soft = len(repaired_issues)
        if repaired_soft < primary_soft:
            return repaired_answer, repaired_issues
        if repaired_soft == primary_soft:
            primary_direct = LocalInferenceEngine._directness_score(
                primary_answer,
                query=query,
                is_recommendation_query=is_recommendation_query,
            )
            repaired_direct = LocalInferenceEngine._directness_score(
                repaired_answer,
                query=query,
                is_recommendation_query=is_recommendation_query,
            )
            if repaired_direct < primary_direct:
                return repaired_answer, repaired_issues
            if repaired_direct == primary_direct and len(str(repaired_answer or "")) <= len(str(primary_answer or "")) + 8:
                return repaired_answer, repaired_issues
        return primary_answer, primary_issues

    @staticmethod
    def _directness_score(answer: str | None, *, query: str, is_recommendation_query: bool) -> int:
        text = str(answer or "").strip()
        if not text:
            return 999
        score = 0
        question_count = LocalInferenceEngine._question_sentence_count(text)
        score += question_count
        first_sentence = LocalInferenceEngine._first_sentence(text)
        if LocalInferenceEngine._looks_question_sentence(first_sentence):
            score += 2
        if is_recommendation_query and not LocalInferenceEngine._looks_three_option_shape(text):
            score += 2
        if LocalInferenceEngine._text_similarity(text, query) >= 0.88:
            score += 1
        return score

    @staticmethod
    def _first_sentence(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        first = re.split(r"(?<=[.!?。！？])\s+|\n+", value, maxsplit=1)[0]
        return first.strip()

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
                if LocalInferenceEngine._looks_question_sentence(sentence):
                    count += 1
        return count

    @staticmethod
    def _limit_question_sentences(text: str, *, max_questions: int) -> str:
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
                if LocalInferenceEngine._looks_question_sentence(sentence):
                    if used_questions >= max_questions:
                        continue
                    used_questions += 1
                kept_sentences.append(sentence)
            if kept_sentences:
                kept_lines.append(" ".join(kept_sentences).strip())
        return "\n".join(kept_lines).strip()

    @staticmethod
    def _looks_three_option_shape(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        numbered = re.findall(r"(?m)^\s*[1-3]\.\s+", value)
        if len(numbered) >= 3:
            return True
        bullets = re.findall(r"(?m)^\s*[-•·]\s+", value)
        return len(bullets) >= 3

    @staticmethod
    def _normalize_three_option_recommendation(answer: str, *, response_language: str) -> str:
        value = str(answer or "").strip()
        if not value:
            return ""
        if LocalInferenceEngine._looks_three_option_shape(value):
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
        if response_language == "ko":
            return "\n".join(lines).strip()
        return "\n".join(lines).strip()

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

    @staticmethod
    def _last_resort_direct_reply_prompt(*, query: str, response_language: str) -> str:
        if response_language == "ko":
            return (
                "사용자 마지막 메시지에 바로 답하세요. "
                "한 문장으로 자연스러운 한국어 존댓말만 사용하세요. "
                "규칙/메타/역할 라벨은 출력하지 마세요.\n"
                f"사용자 메시지: {query}\n"
                "답변:"
            )
        return (
            "Answer the user's last message directly in one natural sentence. "
            "Do not output rules, meta text, or role labels.\n"
            f"User message: {query}\n"
            "Answer:"
        )

    @staticmethod
    def _minimal_conversation_clarification(*, query: str, response_language: str) -> str:
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

    def _generate_with_engine(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
        style: Literal["grounded", "conversation", "rewrite"] = "grounded",
    ) -> str | None:
        if engine == LocalEngine.LLAMA_CPP:
            return self._generate_with_llama(prompt, llama_model_path, max_tokens=max_tokens, style=style)
        return self._generate_with_mlx(prompt, profile, mlx_model_path, max_tokens=max_tokens, style=style)

    @staticmethod
    def _conversation_repair_prompt(prompt: str, *, response_language: str) -> str:
        if response_language == "ko":
            repair = (
                "보정 지시: 사용자 마지막 메시지에 바로 답하세요. "
                "자연스러운 한국어 존댓말만 출력하고, 메타 문장/규칙 문장/역할 라벨은 금지합니다."
            )
        else:
            repair = (
                "Repair instruction: answer the user's latest message directly in natural language. "
                "Do not output meta commentary, rules, or role labels."
            )
        return f"{prompt}\n{repair}"

    @staticmethod
    def _grounded_repair_prompt(prompt: str, *, response_language: str) -> str:
        if response_language == "ko":
            repair = (
                "최종 답변 규칙: 근거를 자연스럽게 재서술해 답하세요. "
                "근거 문장을 길게 그대로 복붙하지 말고, 같은 내용은 하나로 합치세요. "
                "반복 문구가 있으면 핵심만 한 번만 정리하세요. "
                "Evidence/Explanation/Question/Mode/Continuation 같은 메타 문구를 출력하지 마세요."
            )
        else:
            repair = (
                "Final response rule: Paraphrase the evidence naturally. "
                "Do not copy long evidence phrases verbatim, and merge duplicates into one point. "
                "Output only the final grounded answer without meta labels such as Evidence/Explanation/Question/Mode/Continuation."
            )
        return f"{prompt}\n{repair}"

    def prepare_runtime(
        self,
        *,
        engine: LocalEngine,
        profile: str,
        mlx_model_path: str | None = None,
        llama_model_path: str | None = None,
    ) -> RuntimePrepareResponse:
        if engine == LocalEngine.LLAMA_CPP:
            resolved_path = self._resolve_llama_model_path(llama_model_path)
            package_ok = self._ensure_runtime_module(
                engine=LocalEngine.LLAMA_CPP,
                module_name="llama_cpp",
                package_spec="llama-cpp-python>=0.3.9",
                allow_install=True,
            )
            model_exists = bool(resolved_path and Path(resolved_path).expanduser().exists())
            if not resolved_path:
                self._set_engine_error(
                    LocalEngine.LLAMA_CPP,
                    "llama.cpp 모델 경로가 비어 있습니다. GGUF 파일 경로를 지정하거나 다운로드된 모델을 경로 적용해 주세요.",
                )
            elif not model_exists:
                self._set_engine_error(
                    LocalEngine.LLAMA_CPP,
                    f"llama.cpp 모델 파일을 찾지 못했습니다: {resolved_path}",
                )

            ready = False
            if package_ok and model_exists:
                ready = self._ensure_llama_loaded(
                    resolved_path,
                    allow_runtime_install=False,
                )
            detail = self._last_engine_error.get(
                LocalEngine.LLAMA_CPP,
                "llama.cpp 런타임 준비 완료" if ready else "llama.cpp 런타임 준비 실패",
            )
            return RuntimePrepareResponse(
                engine=LocalEngine.LLAMA_CPP,
                ready=ready,
                package_available=package_ok,
                model_path=resolved_path,
                model_exists=model_exists,
                accelerator=self._accelerator_hint(LocalEngine.LLAMA_CPP),
                detail=detail,
            )

        resolved_path = self._resolve_mlx_model_path(profile, mlx_model_path)
        package_ok = self._ensure_runtime_module(
            engine=LocalEngine.MLX,
            module_name="mlx_lm",
            package_spec="mlx-lm>=0.26.0",
            allow_install=True,
        )
        model_exists = self._is_mlx_model_reference_valid(resolved_path)
        if not resolved_path:
            self._set_engine_error(
                LocalEngine.MLX,
                "MLX 모델 경로가 비어 있습니다. MLX 모델 경로를 지정하거나 HuggingFace repo-id를 입력해 주세요.",
            )
        elif not model_exists:
            self._set_engine_error(
                LocalEngine.MLX,
                f"MLX 모델 경로를 검증하지 못했습니다: {resolved_path}",
            )

        ready = False
        if package_ok and model_exists:
            ready = self._ensure_mlx_loaded(
                profile,
                resolved_path,
                allow_runtime_install=False,
            )
        detail = self._last_engine_error.get(
            LocalEngine.MLX,
            "MLX 런타임 준비 완료" if ready else "MLX 런타임 준비 실패",
        )
        return RuntimePrepareResponse(
            engine=LocalEngine.MLX,
            ready=ready,
            package_available=package_ok,
            model_path=resolved_path,
            model_exists=model_exists,
            accelerator=self._accelerator_hint(LocalEngine.MLX),
            detail=detail,
        )

    def _generate_with_mlx(
        self,
        prompt: str,
        profile: str,
        explicit_model_path: str | None,
        *,
        max_tokens: int,
        style: Literal["grounded", "conversation", "rewrite"] = "grounded",
    ) -> str | None:
        if self._ensure_mlx_loaded(profile, explicit_model_path, allow_runtime_install=False):
            try:
                from mlx_lm import generate

                sampling = self._sampling_preset(style=style, engine=LocalEngine.MLX)
                kwargs = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                }
                kwargs.update(
                    {
                        "temp": sampling["temperature"],
                        "top_p": sampling["top_p"],
                        "repetition_penalty": sampling["repeat_penalty"],
                    }
                )
                try:
                    output = generate(self._mlx_model, self._mlx_tokenizer, **kwargs)
                except TypeError:
                    output = generate(self._mlx_model, self._mlx_tokenizer, prompt=prompt, max_tokens=max_tokens)
                text = self._sanitize_generated_answer(str(output), prompt=prompt)
                if text:
                    self._clear_engine_error(LocalEngine.MLX)
                    return text
                self._set_engine_error(LocalEngine.MLX, "MLX 응답이 비어 있습니다.")
            except Exception as exc:
                self._set_engine_error(LocalEngine.MLX, f"MLX 추론 실패: {exc}")
                return None
        return None

    def _generate_with_llama(
        self,
        prompt: str,
        explicit_model_path: str | None,
        *,
        max_tokens: int,
        style: Literal["grounded", "conversation", "rewrite"] = "grounded",
    ) -> str | None:
        if not self._ensure_llama_loaded(explicit_model_path, allow_runtime_install=False):
            return None

        try:
            sampling = self._sampling_preset(style=style, engine=LocalEngine.LLAMA_CPP)
            result = self._llama_model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                repeat_penalty=sampling["repeat_penalty"],
                top_k=sampling["top_k"],
            )
            choices = result.get("choices") or []
            if not choices:
                self._set_engine_error(LocalEngine.LLAMA_CPP, "llama.cpp 응답이 비어 있습니다.")
                return None
            text = self._sanitize_generated_answer(str(choices[0].get("text") or ""), prompt=prompt)
            if text:
                self._clear_engine_error(LocalEngine.LLAMA_CPP)
                return text
            self._set_engine_error(LocalEngine.LLAMA_CPP, "llama.cpp 응답 텍스트가 비어 있습니다.")
            return None
        except Exception as exc:
            self._set_engine_error(LocalEngine.LLAMA_CPP, f"llama.cpp 추론 실패: {exc}")
            return None

    @staticmethod
    def _sampling_preset(
        *,
        style: Literal["grounded", "conversation", "rewrite"],
        engine: LocalEngine,
    ) -> dict[str, float | int]:
        if style == "conversation":
            if engine == LocalEngine.LLAMA_CPP:
                return {"temperature": 0.55, "top_p": 0.92, "repeat_penalty": 1.14, "top_k": 48}
            return {"temperature": 0.52, "top_p": 0.92, "repeat_penalty": 1.14, "top_k": 0}
        if style == "rewrite":
            if engine == LocalEngine.LLAMA_CPP:
                return {"temperature": 0.35, "top_p": 0.9, "repeat_penalty": 1.1, "top_k": 40}
            return {"temperature": 0.34, "top_p": 0.9, "repeat_penalty": 1.1, "top_k": 0}
        if engine == LocalEngine.LLAMA_CPP:
            return {"temperature": 0.22, "top_p": 0.9, "repeat_penalty": 1.15, "top_k": 32}
        return {"temperature": 0.2, "top_p": 0.9, "repeat_penalty": 1.14, "top_k": 0}

    def _ensure_mlx_loaded(
        self,
        profile: str,
        explicit_model_path: str | None = None,
        *,
        allow_runtime_install: bool = False,
    ) -> bool:
        model_path = self._resolve_mlx_model_path(profile, explicit_model_path)
        if not model_path:
            self._set_engine_error(
                LocalEngine.MLX,
                "MLX 모델 경로가 비어 있습니다. 설정에서 MLX 모델 경로를 지정하거나 MLX 모델을 다운로드해 주세요.",
            )
            return False

        if not self._ensure_runtime_module(
            engine=LocalEngine.MLX,
            module_name="mlx_lm",
            package_spec="mlx-lm>=0.26.0",
            allow_install=allow_runtime_install,
        ):
            return False

        if self._mlx_model is not None and self._mlx_tokenizer is not None and self._mlx_model_path == model_path:
            self._clear_engine_error(LocalEngine.MLX)
            return True

        try:
            from mlx_lm import load

            self._mlx_model, self._mlx_tokenizer = load(model_path)
            self._mlx_model_path = model_path
            self._clear_engine_error(LocalEngine.MLX)
            return True
        except Exception as exc:
            self._mlx_model = None
            self._mlx_tokenizer = None
            self._mlx_model_path = None
            self._set_engine_error(LocalEngine.MLX, f"MLX 모델 로드 실패({model_path}): {exc}")
            return False

    def _ensure_llama_loaded(
        self,
        explicit_model_path: str | None = None,
        *,
        allow_runtime_install: bool = False,
    ) -> bool:
        model_path = self._resolve_llama_model_path(explicit_model_path)
        if not model_path:
            self._set_engine_error(
                LocalEngine.LLAMA_CPP,
                "llama.cpp 모델 경로가 비어 있습니다. GGUF 파일 경로를 지정하거나 모델 다운로드를 먼저 실행해 주세요.",
            )
            return False

        resolved = Path(model_path).expanduser()
        if not resolved.exists() or not resolved.is_file():
            self._set_engine_error(
                LocalEngine.LLAMA_CPP,
                f"llama.cpp 모델 파일을 찾지 못했습니다: {resolved}",
            )
            return False

        if not self._ensure_runtime_module(
            engine=LocalEngine.LLAMA_CPP,
            module_name="llama_cpp",
            package_spec="llama-cpp-python>=0.3.9",
            allow_install=allow_runtime_install,
        ):
            return False

        normalized_path = str(resolved)
        if self._llama_model is not None and self._llama_model_path == normalized_path:
            self._clear_engine_error(LocalEngine.LLAMA_CPP)
            return True

        try:
            from llama_cpp import Llama

            cpu_count = os.cpu_count() or 4
            self._llama_model = Llama(
                model_path=normalized_path,
                n_ctx=4096,
                n_threads=max(2, cpu_count // 2),
                n_gpu_layers=-1,
                verbose=False,
            )
            self._llama_model_path = normalized_path
            self._clear_engine_error(LocalEngine.LLAMA_CPP)
            return True
        except Exception as exc:
            self._llama_model = None
            self._llama_model_path = None
            self._set_engine_error(LocalEngine.LLAMA_CPP, f"llama.cpp 모델 로드 실패({normalized_path}): {exc}")
            return False

    @staticmethod
    def _profile_to_model(profile: str) -> str | None:
        key = profile.lower()
        if key == "fast":
            return os.getenv("LOCAL_AI_MODEL_FAST")
        if key == "deep":
            return os.getenv("LOCAL_AI_MODEL_DEEP")
        return os.getenv("LOCAL_AI_MODEL_RECOMMENDED")

    def _resolve_mlx_model_path(self, profile: str, explicit_model_path: str | None) -> str | None:
        candidate = (explicit_model_path or "").strip() or (self._profile_to_model(profile) or "").strip()
        if candidate:
            return candidate
        return self._discover_downloaded_model(LocalEngine.MLX)

    def _resolve_llama_model_path(self, explicit_model_path: str | None) -> str | None:
        candidate = (explicit_model_path or "").strip() or (os.getenv("LOCAL_AI_MODEL_LLAMA") or "").strip()
        if candidate:
            return str(Path(candidate).expanduser())
        discovered = self._discover_downloaded_model(LocalEngine.LLAMA_CPP)
        if discovered:
            return str(Path(discovered).expanduser())
        return None

    def _discover_downloaded_model(self, engine: LocalEngine) -> str | None:
        data_dir = Path(os.getenv("LOCAL_AI_DATA_DIR", "./data")).expanduser().resolve()
        root = data_dir / "models" / engine.value
        if not root.exists():
            return None

        if engine == LocalEngine.LLAMA_CPP:
            candidates = [path for path in root.rglob("*.gguf") if path.is_file()]
            if not candidates:
                candidates = [path for path in root.rglob("*") if path.is_file()]
            if not candidates:
                return None
            return str(max(candidates, key=lambda item: item.stat().st_mtime))

        directories = [path for path in root.iterdir() if path.is_dir()]
        if directories:
            return str(max(directories, key=lambda item: item.stat().st_mtime))

        files = [path for path in root.rglob("*") if path.is_file()]
        if files:
            return str(max(files, key=lambda item: item.stat().st_mtime).parent)
        return None

    @staticmethod
    def _is_mlx_model_reference_valid(model_path: str | None) -> bool:
        if not model_path:
            return False
        candidate = model_path.strip()
        if not candidate:
            return False

        if "://" in candidate:
            return True
        # Hugging Face repo-id style (e.g. mlx-community/Llama-3.2-3B-Instruct-4bit)
        if "/" in candidate and not candidate.startswith("/"):
            return True

        return Path(candidate).expanduser().exists()

    def _ensure_runtime_module(
        self,
        *,
        engine: LocalEngine,
        module_name: str,
        package_spec: str,
        allow_install: bool,
    ) -> bool:
        if importlib.util.find_spec(module_name) is not None:
            self._clear_engine_error(engine)
            return True

        if not allow_install:
            self._set_engine_error(
                engine,
                f"{engine.value} 런타임 패키지({package_spec})가 설치되어 있지 않습니다. 설정에서 엔진 준비를 먼저 실행해 주세요.",
            )
            return False

        command = [sys.executable, "-m", "pip", "install", "--upgrade", package_spec]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            log = (proc.stderr or proc.stdout or "").strip()
            tail = "\n".join(log.splitlines()[-8:]) if log else "(로그 없음)"
            self._set_engine_error(engine, f"{package_spec} 설치 실패 (exit {proc.returncode})\n{tail}")
            return False

        if importlib.util.find_spec(module_name) is None:
            self._set_engine_error(engine, f"{package_spec} 설치 후 모듈({module_name}) 확인 실패")
            return False

        self._clear_engine_error(engine)
        return True

    @staticmethod
    def _accelerator_hint(engine: LocalEngine) -> str:
        if engine == LocalEngine.LLAMA_CPP:
            try:
                import llama_cpp

                supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
                if callable(supports) and bool(supports()):
                    return "Metal GPU offload 가능"
            except Exception:
                pass
            return "CPU 또는 GPU offload 미확인"

        try:
            import mlx.core as mx

            return f"MLX device: {mx.default_device()}"
        except Exception:
            return "MLX 장치 정보 미확인"

    def _set_engine_error(self, engine: LocalEngine, message: str) -> None:
        self._last_engine_error[engine] = message

    def _clear_engine_error(self, engine: LocalEngine) -> None:
        self._last_engine_error.pop(engine, None)

    @staticmethod
    def _resolve_max_tokens(max_tokens: int | None) -> int:
        if max_tokens is None:
            return 320
        return max(96, min(int(max_tokens), 640))

    @staticmethod
    def _prompt(query: str, mode: WorkMode, citations: list[Citation], response_language: str) -> str:
        evidence_lines = LocalInferenceEngine._prepare_evidence_lines(
            citations=citations[:8],
            max_items=5,
            response_language=response_language,
        )
        snippets = "\n".join(evidence_lines)
        strict_rule = ""
        strict_message = insufficient_evidence_message(response_language)
        if mode == WorkMode.STRICT_SEARCH:
            strict_rule = (
                "STRICT RULE: If evidence is insufficient, output exactly "
                f"'{strict_message}' "
                "Do not speculate.\n"
            )
        ko_tone = ""
        if response_language == "ko":
            ko_tone = (
                "Korean style rule: Use natural polite Korean with concise, direct phrasing. "
                "Avoid repetitive or copy-paste wording.\n"
            )
        output_guard = (
            "Output guard: Return only the final answer text. "
            "Never print labels like Evidence:, Explanation:, Question:, Mode:, Continuation:, User:, Assistant:.\n"
        )
        synthesis_rule = (
            "Synthesis rule: Paraphrase evidence naturally instead of copying long phrases verbatim. "
            "Merge duplicate evidence into one concise point.\n"
        )
        return (
            "You are a local-first assistant. Answer only from citation evidence.\n"
            f"{response_language_instruction(response_language)}\n"
            f"{ko_tone}"
            f"{strict_rule}"
            f"{output_guard}"
            f"{synthesis_rule}"
            f"Mode: {mode.value}\n"
            f"Question: {query}\n"
            f"Evidence:\n{snippets}"
        )

    @staticmethod
    def _conversational_prompt(
        *,
        query: str,
        mode: WorkMode,
        response_language: str,
        session_summary: str | None = None,
    ) -> str:
        is_recommendation_query = LocalInferenceEngine._is_recommendation_chat_query(query)
        ko_tone = ""
        if response_language == "ko":
            ko_tone = (
                "한국어 규칙: 자연스러운 존댓말로 답하세요. "
                "공문체/정책문/로그 문구를 피하고, 첫 문장은 바로 답변 본문으로 시작하세요.\n"
            )
        direct_first_rule = (
            "Direct-first rule: Start with a concrete answer in the first sentence. "
            "Do not respond with only a question.\n"
        )
        recommendation_rule = ""
        if is_recommendation_query:
            if response_language == "ko":
                recommendation_rule = (
                    "추천/선택 요청이면 번호 1~3으로 3가지 옵션을 제시하고, "
                    "각 옵션마다 한 줄 근거를 붙이세요. "
                    "확인 질문은 필요할 때만 마지막에 1개 이하로 하세요.\n"
                )
            else:
                recommendation_rule = (
                    "For recommendation/choice requests, provide exactly 3 numbered options "
                    "with a one-line reason each. Ask at most one follow-up question at the end only if essential.\n"
                )
        context_block = ""
        if session_summary:
            context_block = f"<conversation_memory>\n{session_summary}\n</conversation_memory>\n"
        return (
            "You are a conversational local AI assistant.\n"
            f"{response_language_instruction(response_language)}\n"
            f"{ko_tone}"
            "Do not output system logs. Provide concise, practical help.\n"
            "Never role-play both user and assistant in one response.\n"
            "Do not include labels like 'User:' or 'Assistant:'.\n"
            "Do not invent personal facts (location, identity, background) unless user stated them in this turn.\n"
            "Answer naturally and stay concise by default; expand only when the user asks for detail.\n"
            f"{direct_first_rule}"
            f"{recommendation_rule}"
            "Never repeat instruction text, policy wording, or internal rules in the final answer.\n"
            "If conversation memory is provided, use it silently as background context and never reveal or quote it.\n"
            f"Mode: {mode.value}\n"
            f"{context_block}"
            f"Input message: {query}\n"
            "Answer:"
        )

    @staticmethod
    def _postprocess_conversational_answer(answer: str, *, query: str, response_language: str) -> str:
        text = (answer or "").strip()
        if not text:
            return ""
        text = LocalInferenceEngine._strip_reasoning_leak(text)
        if not text:
            return ""
        text = LocalInferenceEngine._strip_speaker_prefixes(text)
        text = re.sub(r"\.\s*입니다\.$", ".", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"(?im)^\s*(?:최종\s*답변|final\s*answer)\s*[:：]\s*", "", text).strip()
        text = re.sub(r"(?i)\bokay,\s*let\s*me\s*process\s*this\.?\s*", "", text).strip()
        text = re.sub(r"(?i)\bthat'?s\s*straightforward\.?\s*", "", text).strip()
        text = re.sub(
            r"(?im)^\s*사용자에게\s*물어볼\s*때는\s*반드시\s*['\"“”]?\?['\"“”]?\s*를?\s*붙여주세요\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)\b사용자에게\s*물어볼\s*때는\s*반드시\s*['\"“”]?\?['\"“”]?\s*를?\s*붙여주세요\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)\b(?:단,\s*)?사용자의\s*질문에\s*대한\s*답이\s*명확하지\s*않을\s*경우\s*추가로\s*설명해\s*주세요\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)\b(?:단,\s*)?사용자의\s*질문에\s*대한\s*명확한\s*답변(?:이)?\s*필요할\s*경우\s*3\s*문장까지\s*가능(?:합니다|해요)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)\b(?:단,\s*)?사용자의?\s*질문에\s*대한\s*(?:답변|답이)\s*(?:부족|충분하지\s*않|명확하지\s*않)[^.!?\n]{0,100}\s*추가(?:적인)?\s*(?:질문|설명)[^.!?\n]{0,100}(?:가능(?:합니다|해요)|할\s*수\s*있습니다|주세요)?\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)^\s*(?:recent\s*session\s*context|최근\s*세션\s*컨텍스트)\s*[:：].*$",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)\(\s*이전\s*문장에\s*대한\s*답변으로[^)]*\)\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)\b이전\s*문장에\s*대한\s*답변으로[^.!?\n]{0,160}\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)^\s*사용자에게\s*(?:직접\s*)?(?:도움을?|답변을?|질문을?)\s*(?:주세요|주십시오|제공하세요)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)^\s*사용자의?\s*(?:말|질문|요청)에\s*바로\s*반응하(?:세요|십시오)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)^\s*사용자\s*메시지에\s*바로\s*반응하(?:세요|십시오)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)^\s*사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)\b사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(r"(?i)\bokay,\s*[.!,]*\s*$", "", text).strip()
        text = re.sub(r"(?i)^okay,\s*i'?ll\s*go\s*with\s*that\s*response\.?\s*", "", text).strip()
        text = re.sub(
            r"(?i)\b(?:okay,\s*let'?s\s*see|alright,\s*let\s*me|alright,\s*something\s*like|hmm,|wait,)\b.*",
            "",
            text,
        ).strip()
        text = re.sub(r"(?i)\b(?:user|assistant|you)\s*:\s*.*", "", text).strip()
        text = re.sub(r"(?im)^최종 답변 규칙:.*$", "", text).strip()
        text = re.sub(r"(?im)^final response rule:.*$", "", text).strip()
        text = re.sub(r"최대한\s*짧고\s*명확하게\s*답하세요\.?\s*", "", text).strip()
        text = LocalInferenceEngine._dedupe_conversation_sentences(text)
        text = LocalInferenceEngine._limit_question_sentences(text, max_questions=1)
        if not text:
            return ""

        if response_language == "ko" and re.search(r"[가-힣]", query):
            ko_chars = len(re.findall(r"[가-힣]", text))
            en_words = len(re.findall(r"[A-Za-z]{3,}", text))
            if en_words >= 6 and ko_chars < 10:
                return ""
        if LocalInferenceEngine._looks_instructional_meta_response(text):
            return ""

        lowered_query = (query or "").lower()
        is_greeting = any(token in lowered_query for token in ("안녕", "hello", "hi", "hey"))
        if not is_greeting:
            return text

        segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not segments:
            return text
        # Greeting replies should be short and direct.
        limited = segments[:2]
        joined = " ".join(limited).strip()
        if response_language == "ko" and len(joined) > 130:
            joined = limited[0]
        return joined.strip()

    @staticmethod
    def _looks_conversational_answer(text: str, *, response_language: str, query: str) -> bool:
        content = (text or "").strip()
        if not content:
            return False
        lowered = content.lower()
        blocked = (
            "user:",
            "assistant:",
            "you:",
            "a:",
            "q:",
            "follow-up question:",
            "okay, let's see",
            "okay, i'll go with that response",
            "okay let me",
            "alright, let me",
            "alright, something like",
            "i should",
            "i need to",
            "the user",
            "let's think",
            "my reasoning",
            "thought:",
            "recent session context",
            "최근 세션 컨텍스트",
            "세션 컨텍스트",
            "이전 문장에 대한 답변으로",
            "mode:",
            "question:",
            "최종 답변 규칙",
            "final response rule",
            "짧고 명확하게 답하세요",
            "1~3문장",
            "1-3문장",
            "한 번만 물어보세요",
            "한번만 물어보세요",
            "ask at most one follow-up question",
            "keep response to 1-3 sentences",
            "최종 답변:",
            "사용자에게 물어볼 때는",
            "반드시 '?'를 붙여주세요",
            "답이 명확하지 않을 경우",
            "추가로 설명해 주세요",
            "okay, let me process this",
            "that's straightforward",
            "명확한 답변이 필요할 경우",
            "3문장까지 가능합니다",
            "답변이 부족할 경우",
            "추가적인 질문",
            "덧붙일 수 있습니다",
            "사용자에게 직접 도움을 주세요",
            "사용자에게 직접 도움",
            "사용자에게 직접",
            "사용자에게 도움을 주세요",
            "사용자의 말에 바로 반응하세요",
            "사용자의 질문에 바로 반응하세요",
            "사용자 메시지에 바로 반응하세요",
            "사용자 메시지에 명확한 답을 하세요",
            "사용자 메시지에 명확한 답변을 하세요",
        )
        if any(token in lowered for token in blocked):
            return False
        if LocalInferenceEngine._looks_instructional_meta_response(content):
            return False
        lowered_query = (query or "").strip().lower()
        is_greeting_query = any(
            token in lowered_query for token in ("안녕", "hello", "hi", "hey", "thanks", "thank you")
        )
        is_brief_query = LocalInferenceEngine._is_brief_chat_query(query)
        compact_query = re.sub(r"\s+", "", query or "")
        if is_greeting_query or is_brief_query:
            min_length = 2
        elif len(compact_query) <= 14:
            min_length = 5
        else:
            min_length = 9
        if not LocalInferenceEngine._looks_model_answer(content, min_length=min_length):
            return False

        if response_language == "ko" and re.search(r"[가-힣]", query):
            ko_chars = len(re.findall(r"[가-힣]", content))
            en_words = len(re.findall(r"[A-Za-z]{3,}", content))
            min_ko_chars = 1 if is_brief_query else 4
            if ko_chars < min_ko_chars:
                return False
            if en_words >= 8 and ko_chars <= (en_words * 2):
                return False
        return True

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

    @staticmethod
    def _conversation_rewrite_prompt(*, query: str, draft_answer: str, response_language: str) -> str:
        draft = re.sub(r"\s+", " ", (draft_answer or "").strip())[:240]
        if response_language == "ko":
            return (
                "다음 초안 답변을 자연스러운 한국어 존댓말로 다시 작성해 주세요.\n"
                "규칙:\n"
                "- 사용자 마지막 질문에 직접 답변\n"
                "- 정책/지시문/메타 문장 금지\n"
                "- 역할 라벨(User/Assistant/You/A) 금지\n"
                "- 같은 문장 반복 금지\n"
                "- 핵심 의미는 유지하고 간결하게 작성\n"
                f"사용자 질문: {query}\n"
                f"초안 답변: {draft}\n"
                "최종 답변:"
            )
        return (
            "Rewrite the draft answer naturally.\n"
            "Rules:\n"
            "- Directly answer the user's latest message\n"
            "- No policy text, no meta commentary, no role labels\n"
            "- No repeated sentences\n"
            f"User message: {query}\n"
            f"Draft answer: {draft}\n"
            "Final answer:"
        )

    @staticmethod
    def _korean_rewrite_prompt(*, query: str, draft_answer: str) -> str:
        draft = re.sub(r"\s+", " ", (draft_answer or "").strip())[:220]
        return (
            "다음 초안 답변을 자연스러운 한국어 존댓말로 다시 작성해 주세요.\n"
            "규칙:\n"
            "- 사용자 마지막 질문에 직접 답변\n"
            "- 정책/지시문/메타 문장 금지\n"
            "- 역할 라벨(User/Assistant/You/A) 금지\n"
            "- 같은 문장 반복 금지\n"
            "- 핵심 의미는 유지하고 간결하게 작성\n"
            f"사용자 질문: {query}\n"
            f"초안 답변: {draft}\n"
            "최종 답변:"
        )

    @staticmethod
    def _conversation_quality_issues(*, query: str, answer: str, response_language: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", (answer or "").strip())
        if not cleaned:
            return ["empty"]
        issues: list[str] = []
        if LocalInferenceEngine._looks_instructional_meta_response(cleaned):
            issues.append("meta_leak")
        if LocalInferenceEngine._contains_context_leak_phrase(cleaned):
            issues.append("context_leak")
        if LocalInferenceEngine._has_duplicate_sentences(cleaned):
            issues.append("duplicate_sentence")
        if LocalInferenceEngine._has_pathological_repetition(cleaned):
            issues.append("pathological_repetition")
        if LocalInferenceEngine._text_similarity(cleaned, query) >= 0.86 and len(cleaned) <= 220:
            issues.append("query_echo")
        if response_language == "ko":
            ko_chars = len(re.findall(r"[가-힣]", cleaned))
            en_words = len(re.findall(r"[A-Za-z]{3,}", cleaned))
            if en_words >= 8 and ko_chars <= (en_words * 2):
                issues.append("english_meta_mix")
            if LocalInferenceEngine._is_informal_korean_tone(cleaned):
                issues.append("informal_tone")
        return issues

    @staticmethod
    def _korean_quality_issues(*, query: str, answer: str, response_language: str) -> list[str]:
        if response_language != "ko":
            return []
        return LocalInferenceEngine._conversation_quality_issues(
            query=query,
            answer=answer,
            response_language=response_language,
        )

    @staticmethod
    def _minimal_safe_conversation_answer(*, query: str, response_language: str) -> str:
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

    @staticmethod
    def _contains_context_leak_phrase(text: str) -> bool:
        lowered = (text or "").lower()
        leak_terms = (
            "recent session context",
            "최근 세션 컨텍스트",
            "세션 컨텍스트",
            "이전 문장에 대한 답변으로",
            "사용자:",
            "input message:",
        )
        return any(term in lowered for term in leak_terms)

    @staticmethod
    def _is_informal_korean_tone(text: str) -> bool:
        value = (text or "").strip()
        if not value:
            return False
        sentences = [
            seg.strip()
            for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", value)
            if seg.strip() and re.search(r"[가-힣]", seg)
        ]
        if not sentences:
            return False

        # 존댓말 종료 표현(요/니다 계열) 비중이 낮으면 반말/비격식 톤으로 간주.
        polite_end = re.compile(
            r"(?:요|니다|습니다|세요|까요|군요|네요|입니다|이에요|예요)\s*[.!?]?$"
        )
        polite_count = 0
        for seg in sentences:
            normalized = re.sub(r"\s+", " ", seg).strip()
            if polite_end.search(normalized):
                polite_count += 1
        informal_count = len(sentences) - polite_count
        return informal_count >= max(1, (len(sentences) + 1) // 2)

    @staticmethod
    def _has_duplicate_sentences(text: str) -> bool:
        parts = [
            seg.strip()
            for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", text or "")
            if seg.strip()
        ]
        if len(parts) < 2:
            return False
        seen: set[str] = set()
        for seg in parts:
            key = re.sub(r"[^\w가-힣]+", "", seg).casefold()
            if not key:
                continue
            if key in seen:
                return True
            seen.add(key)
        return False

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        a_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (a or "").lower()))
        b_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (b or "").lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        union = len(a_tokens.union(b_tokens))
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _fallback_answer(query: str, mode: WorkMode, citations: list[Citation], response_language: str) -> str:
        lang = resolve_response_language(query, response_language)
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
                from mlx_lm import generate

                raw = generate(self._mlx_model, self._mlx_tokenizer, prompt=prompt, max_tokens=320)
                parsed = self._extract_json_object(raw)
                if parsed:
                    return parsed
            except Exception:
                pass

        return self._fallback_classification(path=path, text=text, fixed_categories=fixed_categories, fallback=fallback)

    @staticmethod
    def _extract_json_object(raw: str) -> dict:
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

    @staticmethod
    def _fallback_classification(*, path: str, text: str, fixed_categories: list[str], fallback: dict) -> dict:
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

    @staticmethod
    def _sanitize_generated_answer(raw: str, *, prompt: str) -> str:
        text = (raw or "").strip()
        if not text:
            return ""

        # Remove prompt echo when completion model repeats the input block.
        if prompt and text.startswith(prompt):
            text = text[len(prompt) :].strip()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(?im)^(?:answer|final answer|response|답변|최종 답변)\s*[:：]\s*", "", text).strip()
        text = re.sub(r"(?im)\b(?:answer|response|final answer|최종 답변)\s*[:：]\s*", "", text).strip()
        text = LocalInferenceEngine._strip_reasoning_leak(text)
        if not text:
            return ""

        # Keep line boundaries as segmentation hints to avoid single-runaway sentences.
        segments = [
            seg.strip(" \t-•")
            for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+|(?:\s+-\s+))", text)
            if seg.strip(" \t-•")
        ]
        if not segments:
            compact = re.sub(r"\s+", " ", text).strip()
            if not compact:
                return ""
            segments = [compact]

        deduped: list[str] = []
        seen_counts: dict[str, int] = {}
        prev_key = ""
        for segment in segments:
            key = re.sub(r"[^\w가-힣]+", "", segment).lower()
            if not key:
                continue
            if key == prev_key:
                continue
            count = seen_counts.get(key, 0)
            if count >= 1:
                continue
            if any(LocalInferenceEngine._near_duplicate(segment, prior) for prior in deduped):
                continue
            seen_counts[key] = count + 1
            deduped.append(segment)
            prev_key = key

        compact_segments = LocalInferenceEngine._remove_repeated_blocks(deduped)
        if not compact_segments:
            return ""
        normalized = " ".join(compact_segments).strip()
        if not normalized:
            return ""

        # Hard limit to avoid long repetitive spillover without cutting mid-sentence.
        normalized = LocalInferenceEngine._cap_by_sentence(compact_segments, max_chars=1200)
        if not normalized:
            first = re.sub(r"\s+", " ", compact_segments[0]).strip()
            normalized = first[:1200].rstrip()
            if len(first) > 1200:
                normalized += "..."

        normalized = re.sub(r"\s{2,}", " ", normalized).strip()
        normalized = re.sub(
            r"\(\s*[^()\n]{1,60}\.(?:txt|md|markdown|pdf|docx|py|swift|json|ya?ml)\s*\)\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()
        normalized = re.sub(
            r"^(?:좋아|좋아요|알겠어|알겠어요|알겠습니다|오케이|okay|alright)[,!\.\s]+",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()
        normalized = LocalInferenceEngine._collapse_repeated_phrase_runs(normalized)
        if LocalInferenceEngine._has_pathological_repetition(normalized):
            normalized = LocalInferenceEngine._compress_repetition_fallback(normalized, max_tokens=28)
        if normalized and not (normalized.endswith("습니다") or normalized.endswith("입니다")) and normalized[-1] not in {".", "!", "?", "다", "요"}:
            normalized += "."

        return normalized

    @staticmethod
    def _strip_reasoning_leak(text: str) -> str:
        if not text:
            return ""
        content = text.strip()
        lowered = content.lower()
        leak_markers = (
            "okay, let's see",
            "okay, i'll go with that response",
            "hmm,",
            "i should",
            "the user",
            "follow-up question:",
            "recent session context",
            "최근 세션 컨텍스트",
            "세션 컨텍스트",
            "이전 문장에 대한 답변으로",
            "user:",
            "assistant:",
            "you:",
            "a:",
            "q:",
            "mode:",
            "_continuation",
            "evidence:",
            "explanation:",
            "the question asks",
            "based on the evidence provided",
            "therefore, the answer is",
            "(more)",
            "최종 답변:",
            "사용자에게 물어볼 때는",
            "반드시 '?'를 붙여주세요",
            "답이 명확하지 않을 경우",
            "추가로 설명해 주세요",
            "okay, let me process this",
            "that's straightforward",
            "명확한 답변이 필요할 경우",
            "3문장까지 가능합니다",
            "답변이 부족할 경우",
            "추가적인 질문",
            "덧붙일 수 있습니다",
            "사용자에게 직접 도움을 주세요",
            "사용자에게 직접 도움",
            "사용자에게 직접",
            "사용자에게 도움을 주세요",
            "사용자의 말에 바로 반응하세요",
            "사용자의 질문에 바로 반응하세요",
        )
        if not any(marker in lowered for marker in leak_markers):
            return content

        cut_match = re.search(
            r"(?i)(okay,\s*let'?s\s*see|alright,\s*let\s*me|alright,\s*something\s*like|hmm,|wait,|the user|i should|i need to)",
            content,
        )
        if cut_match and cut_match.start() > 0:
            prefix = content[: cut_match.start()].strip()
            prefix = re.sub(r"(?im)\b(?:user|assistant)\s*:\s*", "", prefix).strip()
            prefix = re.sub(r"(?im)\bfollow-up question:\s*.*", "", prefix).strip()
            prefix = re.sub(r"(?im)\bmode:\s*[A-Z_]+\s*", "", prefix).strip()
            prefix = re.sub(r"\s{2,}", " ", prefix).strip(" -:\n")
            if prefix and len(prefix) >= 8:
                return prefix

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        cleaned_lines: list[str] = []
        for line in lines:
            low = line.lower()
            if low.startswith("user:"):
                continue
            if low.startswith("assistant:"):
                line = re.sub(r"(?im)^\s*assistant\s*[:：]\s*", "", line).strip()
                low = line.lower()
                if not line:
                    continue
            if low.startswith(("you:", "a:", "q:")):
                line = re.sub(r"(?im)^\s*(?:you|a|q)\s*[:：]\s*", "", line).strip()
                low = line.lower()
                if not line:
                    continue
            if "follow-up question:" in low or "recent session context" in low:
                continue
            if "최근 세션 컨텍스트" in line or "세션 컨텍스트" in line:
                continue
            if "이전 문장에 대한 답변으로" in line:
                continue
            if "okay, i'll go with that response" in low:
                line = re.sub(r"(?i)okay,\s*i'?ll\s*go\s*with\s*that\s*response\.?\s*", "", line).strip()
                low = line.lower()
                if not line:
                    continue
            if "okay, let's see" in low or low.startswith("hmm") or low.startswith("wait,"):
                continue
            if "i should" in low or "the user" in low:
                continue
            if "_continuation" in low or "(more)" in low:
                continue
            if "evidence:" in low or "explanation:" in low:
                continue
            if "input message:" in low:
                continue
            if "the question asks" in low or "based on the evidence provided" in low:
                continue
            if "therefore, the answer is" in low:
                continue
            if "okay, let me process this" in low:
                continue
            if "that's straightforward" in low:
                continue
            line = re.sub(r"(?im)^\s*(?:최종\s*답변|final\s*answer)\s*[:：]\s*", "", line).strip()
            line = re.sub(
                r"(?im)^\s*사용자에게\s*물어볼\s*때는\s*반드시\s*['\"“”]?\?['\"“”]?\s*를?\s*붙여주세요\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(
                r"(?im)\b사용자에게\s*물어볼\s*때는\s*반드시\s*['\"“”]?\?['\"“”]?\s*를?\s*붙여주세요\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(
                r"(?im)\b(?:단,\s*)?사용자의\s*질문에\s*대한\s*답이\s*명확하지\s*않을\s*경우\s*추가로\s*설명해\s*주세요\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(
                r"(?im)\b(?:단,\s*)?사용자의\s*질문에\s*대한\s*명확한\s*답변(?:이)?\s*필요할\s*경우\s*3\s*문장까지\s*가능(?:합니다|해요)\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(
                r"(?im)\b(?:단,\s*)?사용자의?\s*질문에\s*대한\s*(?:답변|답이)\s*(?:부족|충분하지\s*않|명확하지\s*않)[^.!?\n]{0,100}\s*추가(?:적인)?\s*(?:질문|설명)[^.!?\n]{0,100}(?:가능(?:합니다|해요)|할\s*수\s*있습니다|주세요)?\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(
                r"(?im)^\s*사용자에게\s*(?:직접\s*)?(?:도움을?|답변을?|질문을?)\s*(?:주세요|주십시오|제공하세요)\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(
                r"(?im)^\s*사용자의?\s*(?:말|질문|요청)에\s*바로\s*반응하(?:세요|십시오)\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(
                r"(?im)^\s*사용자\s*메시지에\s*바로\s*반응하(?:세요|십시오)\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(
                r"(?im)^\s*사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(
                r"(?im)\b사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
                "",
                line,
            ).strip()
            line = re.sub(r"(?i)\bokay,\s*[.!,]*\s*$", "", line).strip()
            line = re.sub(r"(?im)^mode:\s*[A-Z_]+\s*", "", line).strip()
            line = re.sub(r"(?im)^question:\s*", "", line).strip()
            line = re.sub(r"(?im)\bquestion:\s*", "", line).strip()
            if not line:
                continue
            if LocalInferenceEngine._looks_instructional_meta_response(line):
                continue
            cleaned_lines.append(line)

        if not cleaned_lines:
            return ""
        return LocalInferenceEngine._strip_speaker_prefixes("\n".join(cleaned_lines).strip())

    @staticmethod
    def _strip_speaker_prefixes(text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(
            r"(?im)^\s*(?:user|assistant|you|질문|답변|assistant answer|ai)\s*[:：]\s*",
            "",
            text,
        ).strip()
        cleaned = re.sub(r"(?im)^\s*(?:a|q)\s*[:：]\s*", "", cleaned).strip()
        return cleaned

    @staticmethod
    def _dedupe_conversation_sentences(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        parts = [seg.strip() for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", raw) if seg.strip()]
        if not parts:
            return raw
        deduped: list[str] = []
        prev = ""
        for seg in parts:
            key = re.sub(r"[^\w가-힣]+", "", seg).lower()
            if not key:
                continue
            if key == prev:
                continue
            if deduped and LocalInferenceEngine._near_duplicate(seg, deduped[-1]):
                continue
            deduped.append(seg)
            prev = key
        if not deduped:
            return ""
        return " ".join(deduped).strip()

    @staticmethod
    def _looks_model_answer(text: str, *, min_length: int = 12) -> bool:
        content = (text or "").strip()
        if not content:
            return False
        if re.fullmatch(r"(?i)(assistant|user)[.!?]?", content):
            return False
        if LocalInferenceEngine._looks_instructional_meta_response(content):
            return False
        if LocalInferenceEngine._has_pathological_repetition(content):
            return False
        lowered = content.lower()
        error_signals = (
            "engine failed",
            "런타임",
            "모델 경로",
            "설치되어 있지",
            "설치 실패",
            "no relevant evidence",
            "_continuation",
            "evidence:",
            "explanation:",
            "recent session context",
            "최근 세션 컨텍스트",
            "이전 문장에 대한 답변으로",
            "the question asks",
            "based on the evidence provided",
            "therefore, the answer is",
            "ask at most one follow-up question",
            "keep response to 1-3 sentences",
            "명확한 답변이 필요할 경우",
            "3문장까지 가능합니다",
            "답변이 부족할 경우",
            "추가적인 질문",
            "덧붙일 수 있습니다",
            "사용자에게 직접 도움을 주세요",
            "사용자에게 도움을 주세요",
            "사용자의 말에 바로 반응하세요",
            "사용자의 질문에 바로 반응하세요",
            "사용자 메시지에 바로 반응하세요",
            "사용자 메시지에 명확한 답을 하세요",
            "사용자 메시지에 명확한 답변을 하세요",
        )
        if any(token in lowered for token in error_signals):
            return False
        if len(content) < max(2, int(min_length)):
            return False
        return True

    @staticmethod
    def _prepare_evidence_lines(*, citations: list[Citation], max_items: int, response_language: str) -> list[str]:
        lines: list[str] = []
        seen: list[str] = []
        for citation in citations:
            snippet = LocalInferenceEngine._normalize_evidence_snippet(citation.snippet)
            if not snippet:
                continue
            if any(LocalInferenceEngine._near_duplicate(snippet, prior) for prior in seen):
                continue
            seen.append(snippet)
            name = Path(citation.file_path).name or "source.txt"
            lines.append(f"- ({name}) {snippet}")
            if len(lines) >= max_items:
                break
        if lines:
            return lines
        if response_language == "ko":
            return ["- 근거 문장을 찾지 못했습니다."]
        return ["- No evidence snippet was available."]

    @staticmethod
    def _normalize_evidence_snippet(snippet: str) -> str:
        text = re.sub(r"\s+", " ", (snippet or "").strip())
        if not text:
            return ""
        text = LocalInferenceEngine._collapse_repeated_phrase_runs(text)
        if LocalInferenceEngine._has_pathological_repetition(text):
            text = LocalInferenceEngine._compress_repetition_fallback(text, max_tokens=22)
        if len(text) > 220:
            head = text[:220]
            text = head.rsplit(" ", 1)[0].rstrip() + "..."
        return text.strip()

    @staticmethod
    def _looks_instructional_meta_response(text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        explicit_markers = (
            "ask at most one follow-up question",
            "keep response to 1-3 sentences",
            "한 번만 물어보세요",
            "한번만 물어보세요",
            "1~3문장으로",
            "1-3문장으로",
            "사용자에게 물어볼 때는",
            "반드시 '?'를 붙여주세요",
            "답이 명확하지 않을 경우",
            "추가로 설명해 주세요",
            "okay, let me process this",
            "that's straightforward",
            "명확한 답변이 필요할 경우",
            "3문장까지 가능합니다",
            "답변이 부족할 경우",
            "추가적인 질문",
            "덧붙일 수 있습니다",
            "사용자에게 직접 도움을 주세요",
            "사용자에게 도움을 주세요",
            "사용자의 말에 바로 반응하세요",
            "사용자의 질문에 바로 반응하세요",
            "사용자 메시지에 바로 반응하세요",
            "사용자 메시지에 명확한 답을 하세요",
            "사용자 메시지에 명확한 답변을 하세요",
            "recent session context",
            "최근 세션 컨텍스트",
            "세션 컨텍스트",
            "이전 문장에 대한 답변으로",
        )
        if any(marker in lowered for marker in explicit_markers):
            return True
        if "사용자의 질문" in lowered and "문장" in lowered and ("답변" in lowered or "답이" in lowered):
            return True
        if re.search(
            r"사용자의?\s*질문에\s*대한\s*(?:답변|답이).{0,50}(?:부족|충분하지|명확하지).{0,80}(?:추가적인?\s*(?:질문|설명)|질문을?\s*덧붙)",
            lowered,
        ):
            return True
        if re.search(
            r"사용자에게\s*(?:직접\s*)?(?:도움을?|답변을?|질문을?)\s*(?:주세요|주십시오|제공하세요)",
            lowered,
        ):
            return True
        if re.search(
            r"(?:사용자의?\s*(?:말|질문|요청)|사용자\s*메시지)에\s*바로\s*반응하(?:세요|십시오)",
            lowered,
        ):
            return True
        if re.search(
            r"사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)",
            lowered,
        ):
            return True
        if ("물어보세요" in lowered and "답하세요" in lowered) or (
            "ask" in lowered and "respond" in lowered and "sentence" in lowered
        ):
            return True
        return False

    @staticmethod
    def _is_brief_chat_query(query: str) -> bool:
        raw = (query or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        ack_tokens = (
            "그렇구나",
            "알겠",
            "오케이",
            "그래",
            "아하",
            "맞아",
            "ㅇㅋ",
            "ok",
            "okay",
            "got it",
            "makes sense",
            "cool",
            "thanks",
            "thank you",
        )
        if any(token in lowered for token in ack_tokens):
            return True
        compact = re.sub(r"\s+", "", raw)
        if len(compact) <= 8:
            return True
        if re.fullmatch(r"[0-9+\-*/().=\s?]+", raw):
            return True
        return False

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
    def _near_duplicate(a: str, b: str) -> bool:
        norm_a = re.sub(r"\s+", " ", a).strip().lower()
        norm_b = re.sub(r"\s+", " ", b).strip().lower()
        if not norm_a or not norm_b:
            return False
        if norm_a == norm_b:
            return True
        if norm_a in norm_b or norm_b in norm_a:
            shorter = min(len(norm_a), len(norm_b))
            return shorter >= 24
        # rough token overlap to suppress repetitive loops
        tokens_a = set(re.findall(r"[a-z0-9가-힣]+", norm_a))
        tokens_b = set(re.findall(r"[a-z0-9가-힣]+", norm_b))
        if not tokens_a or not tokens_b:
            return False
        overlap = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return union > 0 and (overlap / union) >= 0.88

    @staticmethod
    def _remove_repeated_blocks(segments: list[str]) -> list[str]:
        if len(segments) < 4:
            return segments
        output: list[str] = []
        seen: set[str] = set()
        for segment in segments:
            key = re.sub(r"[^\w가-힣]+", "", segment).lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(segment)
        return output

    @staticmethod
    def _collapse_repeated_phrase_runs(text: str) -> str:
        content = (text or "").strip()
        if not content:
            return ""
        collapsed = content
        # Collapse repeated single-word runs.
        collapsed = re.sub(r"(?i)\b([A-Za-z0-9가-힣_]+)(?:\s+\1){3,}\b", r"\1", collapsed)

        # Collapse repeated multi-word chunk runs (2~8 tokens).
        chunk_pattern = re.compile(r"(?P<chunk>(?:\S+\s+){1,7}\S+)(?:\s+(?P=chunk)){1,}")
        for _ in range(4):
            updated = chunk_pattern.sub(lambda m: m.group("chunk"), collapsed)
            if updated == collapsed:
                break
            collapsed = updated

        return re.sub(r"\s{2,}", " ", collapsed).strip()

    @staticmethod
    def _has_pathological_repetition(text: str) -> bool:
        content = (text or "").strip()
        if not content:
            return False
        tokens = re.findall(r"[A-Za-z0-9가-힣_]+", content.lower())
        if len(tokens) < 20:
            return False
        n = 4
        grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not grams:
            return False
        counts: dict[tuple[str, ...], int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        top = max(counts.values(), default=0)
        return top >= 5 or (top / len(grams)) >= 0.24

    @staticmethod
    def _compress_repetition_fallback(text: str, *, max_tokens: int) -> str:
        content = re.sub(r"\s+", " ", (text or "").strip())
        if not content:
            return ""
        tokens = re.findall(r"\S+", content)
        if len(tokens) <= max_tokens:
            return content
        shortened = " ".join(tokens[:max_tokens]).strip()
        if shortened and shortened[-1] not in {".", "!", "?", "다", "요"}:
            shortened += "."
        return shortened

    @staticmethod
    def _cap_by_sentence(segments: list[str], *, max_chars: int) -> str:
        selected: list[str] = []
        length = 0
        for segment in segments:
            proposed = length + (1 if selected else 0) + len(segment)
            if proposed > max_chars:
                break
            selected.append(segment)
            length = proposed
        return " ".join(selected).strip()
