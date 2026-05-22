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
from ..utils import inject_system_instruction_if_needed
from ...models import LocalEngine, WorkMode, AgentAction, Citation, RuntimePrepareResponse

if TYPE_CHECKING:
    from ...local_inference import LocalInferenceEngine

# Component: llama_handler.py
class LlamaHandler(BaseDelegate):
    def _inject_system_instruction_if_needed(
        self, message_state: list[dict[str, str]], response_language: str | None = None
    ) -> list[dict[str, str]]:
        return inject_system_instruction_if_needed(message_state, response_language)


    def _generate_with_llama(
        self,
        prompt: str,
        explicit_model_path: str | None,
        *,
        max_tokens: int,
        style: Literal["grounded", "conversation", "rewrite"] = "grounded",
        message_state: list[dict[str, str]] | None = None,
        response_language: str | None = None,
    ) -> str | None:
        with self._llama_lock:
            if not self._ensure_llama_loaded(explicit_model_path, allow_runtime_install=False):
                return None

            try:
                model_ref = self._resolve_llama_model_path(explicit_model_path)
                sampling = self._sampling_preset_for_model(
                    style=style,
                    engine=LocalEngine.LLAMA_CPP,
                    model_path=model_ref,
                )
                stop_seqs = self._stop_sequences_for_model(
                    style=style,
                    model_path=model_ref,
                    default_sequences=["User:", "Assistant:", "<|end_of_turn|>", "<|eot_id|>", "<|endoftext|>"],
                )

                text: str = ""
                stream_cb = self._current_stream_token_callback()

                # Native Chat Template handling via create_chat_completion
                if message_state and hasattr(self._llama_model, "create_chat_completion"):
                    chat_kwargs = {
                        "messages": self._inject_system_instruction_if_needed(message_state, response_language),
                        "max_tokens": max_tokens,
                        "temperature": sampling["temperature"],
                        "top_p": sampling["top_p"],
                        "repeat_penalty": sampling["repeat_penalty"],
                        "top_k": int(sampling.get("top_k", 40)),
                        "min_p": float(sampling.get("min_p", 0.0)),
                        "stop": stop_seqs,
                    }
                    try:
                        if stream_cb is not None:
                            token_stream = self._llama_model.create_chat_completion(**chat_kwargs, stream=True)
                            raw_parts: list[str] = []
                            for item in token_stream:
                                if not isinstance(item, dict):
                                    continue
                                choices = item.get("choices") or []
                                if not choices:
                                    continue
                                delta = choices[0].get("delta") or {}
                                piece = str(delta.get("content") or "")
                                if piece:
                                    raw_parts.append(piece)
                                    self._emit_stream_token(piece)
                            text = "".join(raw_parts)
                        else:
                            result = self._llama_model.create_chat_completion(**chat_kwargs)
                            choices = result.get("choices") or []
                            if choices:
                                msg = choices[0].get("message") or {}
                                text = str(msg.get("content") or "")
                    except Exception as chat_exc:
                        # Fallback to standard prompt completion if chat completion fails
                        text = ""

                # Fallback to legacy completion if no message_state, or if create_chat_completion failed
                if not text:
                    kwargs = {
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": sampling["temperature"],
                        "top_p": sampling["top_p"],
                        "repeat_penalty": sampling["repeat_penalty"],
                        "top_k": int(sampling.get("top_k", 40)),
                        "min_p": float(sampling.get("min_p", 0.0)),
                        "stop": stop_seqs,
                    }
                    if stream_cb is not None:
                        token_stream = self._llama_model.create_completion(**kwargs, stream=True)
                        raw_parts = []
                        for item in token_stream:
                            if not isinstance(item, dict):
                                continue
                            choices = item.get("choices") or []
                            if not choices:
                                continue
                            piece = str(choices[0].get("text") or "")
                            if piece:
                                raw_parts.append(piece)
                                self._emit_stream_token(piece)
                        text = "".join(raw_parts)
                    else:
                        result = self._llama_model.create_completion(**kwargs)
                        choices = result.get("choices") or []
                        if choices:
                            text = str(choices[0].get("text") or "")

                if style == "conversation":
                    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                    if not message_state and prompt and text.startswith(prompt):
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
