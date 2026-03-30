from __future__ import annotations
import re
import json
import os
import subprocess
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
from ..base import BaseDelegate
from ..types import InferenceResult, _ConversationCandidateResult
from ...models import LocalEngine, WorkMode, AgentAction, Citation, RuntimePrepareResponse

if TYPE_CHECKING:
    from ...local_inference import LocalInferenceEngine

# Component: llama_handler.py
class LlamaHandler(BaseDelegate):
    def _generate_with_llama(
        self,
        prompt: str,
        explicit_model_path: str | None,
        *,
        max_tokens: int,
        style: Literal["grounded", "conversation", "rewrite"] = "grounded",
    ) -> str | None:
        with self._llama_lock:
            if not self._ensure_llama_loaded(explicit_model_path, allow_runtime_install=False):
                return None

            try:
                sampling = self._sampling_preset(style=style, engine=LocalEngine.LLAMA_CPP)
                kwargs = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": sampling["temperature"],
                    "top_p": sampling["top_p"],
                    "repeat_penalty": sampling["repeat_penalty"],
                    "top_k": sampling["top_k"],
                }

                text: str = ""
                stream_cb = self._current_stream_token_callback()
                if stream_cb is not None:
                    try:
                        token_stream = self._llama_model.create_completion(**kwargs, stream=True)
                        raw_parts: list[str] = []
                        for item in token_stream:
                            if not isinstance(item, dict):
                                continue
                            choices = item.get("choices") or []
                            if not choices:
                                continue
                            piece = str(choices[0].get("text") or "")
                            if not piece:
                                continue
                            raw_parts.append(piece)
                            self._emit_stream_token(piece)
                        text = "".join(raw_parts)
                    except Exception:
                        text = ""

                if not text:
                    result = self._llama_model.create_completion(**kwargs)
                    choices = result.get("choices") or []
                    if not choices:
                        self._set_engine_error(LocalEngine.LLAMA_CPP, "llama.cpp 응답이 비어 있습니다.")
                        return None
                    text = str(choices[0].get("text") or "")

                if style == "conversation":
                    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                    if prompt and text.startswith(prompt):
                        text = text[len(prompt):].strip()
                else:
                    text = self._sanitize_generated_answer(text, prompt=prompt)
                if text:
                    self._clear_engine_error(LocalEngine.LLAMA_CPP)
                    return text
                self._set_engine_error(LocalEngine.LLAMA_CPP, "llama.cpp 응답 텍스트가 비어 있습니다.")
                return None
            except Exception as exc:
                self._set_engine_error(LocalEngine.LLAMA_CPP, f"llama.cpp 추론 실패: {exc}")
                return None
        return None

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
        with self._llama_lock:
            if self._llama_model is not None and self._llama_model_path == normalized_path:
                self._clear_engine_error(LocalEngine.LLAMA_CPP)
                return True

            try:
                llama_mod = importlib.import_module("llama_cpp")
                Llama = llama_mod.Llama

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

    def _resolve_llama_model_path(self, explicit_model_path: str | None) -> str | None:
        candidate = (explicit_model_path or "").strip() or (os.getenv("LOCAL_AI_MODEL_LLAMA") or "").strip()
        if candidate:
            return str(Path(candidate).expanduser())
        discovered = self._discover_downloaded_model(LocalEngine.LLAMA_CPP)
        if discovered:
            return str(Path(discovered).expanduser())
        return None
