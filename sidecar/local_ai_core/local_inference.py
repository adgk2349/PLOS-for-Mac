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

from .language_utils import (
    insufficient_evidence_message,
    resolve_response_language,
    response_language_instruction,
)
from .models import Citation, LocalEngine, RuntimePrepareResponse, WorkMode


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


from .inference import local_mixins as _local_inference_mixins

class LocalInferenceEngine(_local_inference_mixins.LocalInferenceEngineMethodsMixin):
    """Local inference router for MLX and llama.cpp with deterministic fallback."""

    def __init__(self):
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._mlx_model_path: str | None = None

        self._llama_model = None
        self._llama_model_path: str | None = None

        self._last_engine_error: dict[LocalEngine, str] = {}

    def generate(
        self,
        query: str,
        mode: WorkMode,
        citations: list[Citation],
        profile: str,
        *,
        engine: LocalEngine = LocalEngine.MLX,
        mlx_model_path: str | None = None,
        llama_model_path: str | None = None,
        language_preference: str | None = None,
        max_tokens: int | None = None,
    ) -> InferenceResult:
        response_language = resolve_response_language(query, language_preference)
        prompt = self._prompt(query, mode, citations, response_language)
        token_budget = self._resolve_max_tokens(max_tokens)
        primary = engine
        secondary = LocalEngine.LLAMA_CPP if primary == LocalEngine.MLX else LocalEngine.MLX

        answer = self._generate_grounded_candidate(
            engine=primary,
            prompt=prompt,
            response_language=response_language,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
        )
        if answer:
            return InferenceResult(
                answer=answer,
                engine_used=primary,
                used_fallback=False,
                detail=f"primary_engine={primary.value}",
            )

        primary_error = self._last_engine_error.get(primary, f"{primary.value} engine failed")
        secondary_answer = self._generate_grounded_candidate(
            engine=secondary,
            prompt=prompt,
            response_language=response_language,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
        )
        if secondary_answer:
            detail = (
                f"primary_engine_failed={primary.value}; fallback_engine_used={secondary.value}; "
                f"reason={primary_error}"
            )
            return InferenceResult(
                answer=secondary_answer,
                engine_used=secondary,
                used_fallback=False,
                detail=detail,
            )

        secondary_error = self._last_engine_error.get(secondary, f"{secondary.value} engine failed")
        detail = (
            f"{primary.value} 실패: {primary_error}\n"
            f"{secondary.value} 실패: {secondary_error}"
        )
        fallback = self._fallback_answer(query, mode, citations, response_language)
        return InferenceResult(
            answer=f"{detail}\n\n{fallback}",
            engine_used=primary,
            used_fallback=True,
            detail=detail,
        )

    def generate_conversational(
        self,
        *,
        query: str,
        mode: WorkMode,
        profile: str,
        engine: LocalEngine = LocalEngine.MLX,
        mlx_model_path: str | None = None,
        llama_model_path: str | None = None,
        language_preference: str | None = None,
        max_tokens: int | None = None,
        session_summary: str | None = None,
        allow_static_fallback: bool = True,
    ) -> InferenceResult:
        response_language = resolve_response_language(query, language_preference)
        prompt = self._conversational_prompt(
            query=query,
            mode=mode,
            response_language=response_language,
            session_summary=session_summary,
        )
        token_budget = self._resolve_max_tokens(max_tokens)
        primary = engine
        secondary = LocalEngine.LLAMA_CPP if primary == LocalEngine.MLX else LocalEngine.MLX

        primary_candidate = self._generate_conversation_candidate(
            engine=primary,
            prompt=prompt,
            query=query,
            response_language=response_language,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
        )
        if primary_candidate.answer:
            detail = f"conversational_primary={primary.value}"
            if primary_candidate.rewrite_used:
                detail += "; korean_rewrite_used=1"
            if primary_candidate.quality_repair_reason:
                detail += f"; quality_repair_reason={primary_candidate.quality_repair_reason}"
            if primary_candidate.repair_triggered:
                detail += "; repair_triggered=1"
            if primary_candidate.repair_success:
                detail += "; repair_success=1"
            if primary_candidate.leak_blocked:
                detail += "; leak_blocked=1"
            if primary_candidate.direct_first_applied:
                detail += "; direct_first_applied=1"
            detail += f"; question_count_after_postprocess={max(0, int(primary_candidate.question_count_after_postprocess))}"
            if primary_candidate.recommendation_shape:
                detail += f"; recommendation_shape={str(primary_candidate.recommendation_shape).strip()}"
            return InferenceResult(
                answer=primary_candidate.answer,
                engine_used=primary,
                used_fallback=False,
                detail=detail,
            )

        primary_error = self._last_engine_error.get(primary, f"{primary.value} engine failed")
        secondary_candidate = self._generate_conversation_candidate(
            engine=secondary,
            prompt=prompt,
            query=query,
            response_language=response_language,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
        )
        if secondary_candidate.answer:
            detail = f"conversational_secondary={secondary.value}; primary_error={primary_error}"
            if secondary_candidate.rewrite_used:
                detail += "; korean_rewrite_used=1"
            if secondary_candidate.quality_repair_reason:
                detail += f"; quality_repair_reason={secondary_candidate.quality_repair_reason}"
            if secondary_candidate.repair_triggered:
                detail += "; repair_triggered=1"
            if secondary_candidate.repair_success:
                detail += "; repair_success=1"
            if secondary_candidate.leak_blocked:
                detail += "; leak_blocked=1"
            if secondary_candidate.direct_first_applied:
                detail += "; direct_first_applied=1"
            detail += f"; question_count_after_postprocess={max(0, int(secondary_candidate.question_count_after_postprocess))}"
            if secondary_candidate.recommendation_shape:
                detail += f"; recommendation_shape={str(secondary_candidate.recommendation_shape).strip()}"
            return InferenceResult(
                answer=secondary_candidate.answer,
                engine_used=secondary,
                used_fallback=False,
                detail=detail,
            )

        secondary_error = self._last_engine_error.get(secondary, f"{secondary.value} engine failed")
        detail = (
            f"conversational engines failed; primary={primary.value}; secondary={secondary.value}; "
            f"primary_error={primary_error}; secondary_error={secondary_error}"
        )
        if not allow_static_fallback:
            return InferenceResult(
                answer="",
                engine_used=primary,
                used_fallback=True,
                detail=detail,
            )
        is_brief_query = self._is_brief_chat_query(query)
        if response_language == "ko":
            if is_brief_query:
                fallback = "응, 알겠어요. 이어서 편하게 말해줘요."
            else:
                fallback = "좋아요. 바로 도와드릴게요. 기준을 조금만 알려주시면 더 정확하게 이어갈 수 있어요."
        else:
            if is_brief_query:
                fallback = "Got it. Tell me a bit more and I'll help right away."
            else:
                fallback = "Sure, I can help right away. Give me one more hint and I'll narrow it down."
        return InferenceResult(
            answer=fallback,
            engine_used=primary,
            used_fallback=True,
            detail=detail,
        )


_local_inference_mixins.LocalInferenceEngine = LocalInferenceEngine
_local_inference_mixins.LocalInferenceConfig = LocalInferenceConfig
_local_inference_mixins._ConversationCandidateResult = _ConversationCandidateResult
