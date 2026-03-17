from __future__ import annotations

import os
from dataclasses import dataclass

from .models import Citation, WorkMode


@dataclass(slots=True)
class LocalInferenceConfig:
    model_path: str | None
    max_tokens: int


class LocalInferenceEngine:
    """MLX-backed local inference with a deterministic fallback for dev environments."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._mlx_ready = False

    def generate(self, query: str, mode: WorkMode, citations: list[Citation], profile: str) -> str:
        if self._ensure_mlx_loaded(profile):
            prompt = self._prompt(query, mode, citations)
            try:
                from mlx_lm import generate

                return generate(self._model, self._tokenizer, prompt=prompt, max_tokens=480)
            except Exception:
                # Fall back when model runtime errors occur.
                pass

        return self._fallback_answer(query, mode, citations)

    def _ensure_mlx_loaded(self, profile: str) -> bool:
        if self._mlx_ready:
            return True

        model_path = self._profile_to_model(profile)
        if not model_path:
            return False

        try:
            from mlx_lm import load

            self._model, self._tokenizer = load(model_path)
            self._mlx_ready = True
            return True
        except Exception:
            self._mlx_ready = False
            return False

    @staticmethod
    def _profile_to_model(profile: str) -> str | None:
        key = profile.lower()
        if key == "fast":
            return os.getenv("LOCAL_AI_MODEL_FAST")
        if key == "deep":
            return os.getenv("LOCAL_AI_MODEL_DEEP")
        return os.getenv("LOCAL_AI_MODEL_RECOMMENDED")

    @staticmethod
    def _prompt(query: str, mode: WorkMode, citations: list[Citation]) -> str:
        snippets = "\n".join(f"- {c.snippet}" for c in citations[:5])
        return (
            "You are a local-first assistant. Answer only from citation evidence.\n"
            f"Mode: {mode.value}\n"
            f"Question: {query}\n"
            f"Evidence:\n{snippets}"
        )

    @staticmethod
    def _fallback_answer(query: str, mode: WorkMode, citations: list[Citation]) -> str:
        if not citations:
            return "선택된 로컬 문서에서 관련 근거를 찾지 못했습니다. 폴더/인덱싱 상태를 확인해 주세요."

        snippets = [c.snippet for c in citations[:3]]
        joined = "\n".join(f"- {snippet}" for snippet in snippets)

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
