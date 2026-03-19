from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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


class LocalInferenceEngine:
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

        answer = self._generate_with_engine(
            engine=primary,
            prompt=prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
        )
        if answer and self._looks_model_answer(answer):
            return InferenceResult(
                answer=answer,
                engine_used=primary,
                used_fallback=False,
                detail=f"primary_engine={primary.value}",
            )

        primary_error = self._last_engine_error.get(primary, f"{primary.value} engine failed")
        secondary_answer = self._generate_with_engine(
            engine=secondary,
            prompt=prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
        )
        if secondary_answer and self._looks_model_answer(secondary_answer):
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

        answer = self._generate_with_engine(
            engine=primary,
            prompt=prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
        )
        answer = self._postprocess_conversational_answer(
            answer or "",
            query=query,
            response_language=response_language,
        )
        if answer and self._looks_conversational_answer(
            answer,
            response_language=response_language,
            query=query,
        ):
            return InferenceResult(
                answer=answer,
                engine_used=primary,
                used_fallback=False,
                detail=f"conversational_primary={primary.value}",
            )

        primary_error = self._last_engine_error.get(primary, f"{primary.value} engine failed")
        secondary_answer = self._generate_with_engine(
            engine=secondary,
            prompt=prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
        )
        secondary_answer = self._postprocess_conversational_answer(
            secondary_answer or "",
            query=query,
            response_language=response_language,
        )
        if secondary_answer and self._looks_conversational_answer(
            secondary_answer,
            response_language=response_language,
            query=query,
        ):
            return InferenceResult(
                answer=secondary_answer,
                engine_used=secondary,
                used_fallback=False,
                detail=f"conversational_secondary={secondary.value}; primary_error={primary_error}",
            )

        detail = f"conversational engines failed; primary={primary.value}; secondary={secondary.value}"
        if not allow_static_fallback:
            return InferenceResult(
                answer="",
                engine_used=primary,
                used_fallback=True,
                detail=detail,
            )
        if response_language == "ko":
            fallback = "좋아요. 바로 도와드릴게요. 기준을 조금만 알려주시면 더 정확하게 이어갈 수 있어요."
        else:
            fallback = "Sure, I can help right away. Give me one more hint and I'll narrow it down."
        return InferenceResult(
            answer=fallback,
            engine_used=primary,
            used_fallback=True,
            detail=detail,
        )

    def _generate_with_engine(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
    ) -> str | None:
        if engine == LocalEngine.LLAMA_CPP:
            return self._generate_with_llama(prompt, llama_model_path, max_tokens=max_tokens)
        return self._generate_with_mlx(prompt, profile, mlx_model_path, max_tokens=max_tokens)

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
    ) -> str | None:
        if self._ensure_mlx_loaded(profile, explicit_model_path, allow_runtime_install=False):
            try:
                from mlx_lm import generate

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

    def _generate_with_llama(self, prompt: str, explicit_model_path: str | None, *, max_tokens: int) -> str | None:
        if not self._ensure_llama_loaded(explicit_model_path, allow_runtime_install=False):
            return None

        try:
            result = self._llama_model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.2,
                repeat_penalty=1.15,
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
        snippets = "\n".join(f"- {c.snippet}" for c in citations[:5])
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
                "Korean style rule: Use concise formal Korean ending with '~습니다' or '~입니다'. "
                "Never repeat the same sentence.\n"
            )
        return (
            "You are a local-first assistant. Answer only from citation evidence.\n"
            f"{response_language_instruction(response_language)}\n"
            f"{ko_tone}"
            f"{strict_rule}"
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
        ko_tone = ""
        if response_language == "ko":
            ko_tone = (
                "한국어 규칙: 자연스러운 존댓말로 짧고 명확하게 답합니다. "
                "공문체나 딱딱한 로그 문구를 피하고, 첫 문장은 사용자 메시지에 바로 반응하세요.\n"
            )
        context_block = ""
        if session_summary:
            context_block = f"Recent session context (summary only):\n{session_summary}\n"
        return (
            "You are a conversational local AI assistant.\n"
            f"{response_language_instruction(response_language)}\n"
            f"{ko_tone}"
            "Do not output system logs. Provide concise, practical help.\n"
            "Never role-play both user and assistant in one response.\n"
            "Do not include labels like 'User:' or 'Assistant:'.\n"
            "Do not invent personal facts (location, identity, background) unless user stated them in this turn.\n"
            "Keep response to 1-3 sentences. Ask at most one follow-up question.\n"
            f"Mode: {mode.value}\n"
            f"{context_block}"
            f"User: {query}\n"
        )

    @staticmethod
    def _postprocess_conversational_answer(answer: str, *, query: str, response_language: str) -> str:
        text = (answer or "").strip()
        if not text:
            return ""
        text = LocalInferenceEngine._strip_reasoning_leak(text)
        if not text:
            return ""
        text = re.sub(r"\.\s*입니다\.$", ".", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(
            r"(?i)\b(?:okay,\s*let'?s\s*see|alright,\s*let\s*me|alright,\s*something\s*like|hmm,|wait,)\b.*",
            "",
            text,
        ).strip()
        text = re.sub(r"(?i)\b(?:user|assistant)\s*:\s*.*", "", text).strip()
        if not text:
            return ""

        if response_language == "ko" and re.search(r"[가-힣]", query):
            ko_chars = len(re.findall(r"[가-힣]", text))
            en_words = len(re.findall(r"[A-Za-z]{3,}", text))
            if en_words >= 6 and ko_chars < 10:
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
            "follow-up question:",
            "okay, let's see",
            "okay let me",
            "alright, let me",
            "alright, something like",
            "i should",
            "i need to",
            "the user",
            "let's think",
            "my reasoning",
            "thought:",
            "mode:",
            "question:",
        )
        if any(token in lowered for token in blocked):
            return False
        if not LocalInferenceEngine._looks_model_answer(content):
            return False

        if response_language == "ko" and re.search(r"[가-힣]", query):
            ko_chars = len(re.findall(r"[가-힣]", content))
            en_words = len(re.findall(r"[A-Za-z]{3,}", content))
            if ko_chars < 4:
                return False
            if en_words >= 8 and ko_chars <= (en_words * 2):
                return False
        return True

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
        text = re.sub(r"(?im)^(?:answer|final answer|response|답변)\s*[:：]\s*", "", text).strip()
        text = re.sub(r"(?im)\b(?:answer|response)\s*[:：]\s*", "", text).strip()
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
            "hmm,",
            "i should",
            "the user",
            "follow-up question:",
            "recent session context",
            "user:",
            "assistant:",
            "mode:",
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
            if low.startswith("user:") or low.startswith("assistant:"):
                continue
            if "follow-up question:" in low or "recent session context" in low:
                continue
            if "okay, let's see" in low or low.startswith("hmm") or low.startswith("wait,"):
                continue
            if "i should" in low or "the user" in low:
                continue
            line = re.sub(r"(?im)^mode:\s*[A-Z_]+\s*", "", line).strip()
            line = re.sub(r"(?im)^question:\s*", "", line).strip()
            line = re.sub(r"(?im)\bquestion:\s*", "", line).strip()
            if not line:
                continue
            cleaned_lines.append(line)

        if not cleaned_lines:
            return ""
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _looks_model_answer(text: str) -> bool:
        content = (text or "").strip()
        if not content:
            return False
        lowered = content.lower()
        error_signals = (
            "engine failed",
            "런타임",
            "모델 경로",
            "설치되어 있지",
            "설치 실패",
            "no relevant evidence",
        )
        if any(token in lowered for token in error_signals):
            return False
        if len(content) < 12:
            return False
        return True

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
