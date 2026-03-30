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

# Component: mlx_handler.py
class MlxHandler(BaseDelegate):
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
                mlx_lm = importlib.import_module("mlx_lm")
                generate = mlx_lm.generate
                model_ref = (explicit_model_path or self._mlx_model_path or self._profile_to_model(profile) or "").strip()
                prepared_prompt = self._mlx_prepare_prompt(
                    prompt=prompt,
                    style=style,
                    model_path=model_ref,
                )

                sampling = self._sampling_preset(style=style, engine=LocalEngine.MLX)
                kwargs = {
                    "prompt": prepared_prompt,
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
                    output = None
                    stream_cb = self._current_stream_token_callback()
                    if stream_cb is not None:
                        try:
                            stream_result = generate(
                                self._mlx_model,
                                self._mlx_tokenizer,
                                **kwargs,
                                stream=True,
                            )
                            if isinstance(stream_result, str):
                                output = stream_result
                            else:
                                raw_parts: list[str] = []
                                for chunk in stream_result:
                                    piece = self._mlx_chunk_to_text(chunk)
                                    if not piece:
                                        continue
                                    raw_parts.append(piece)
                                    self._emit_stream_token(piece)
                                output = "".join(raw_parts)
                        except Exception:
                            output = None
                    if output is None:
                        output = generate(self._mlx_model, self._mlx_tokenizer, **kwargs)
                except TypeError:
                    output = generate(self._mlx_model, self._mlx_tokenizer, prompt=prepared_prompt, max_tokens=max_tokens)

                # Conversational path should pass through raw generation text and let
                # conversational post-processing/quality gates decide validity.
                if style == "conversation":
                    text = str(output or "").replace("\r\n", "\n").replace("\r", "\n").strip()
                    if prepared_prompt and text.startswith(prepared_prompt):
                        text = text[len(prepared_prompt):].strip()
                    if prompt and text.startswith(prompt):
                        text = text[len(prompt):].strip()
                else:
                    text = self._sanitize_generated_answer(str(output), prompt=prepared_prompt or prompt)

                if text:
                    self._clear_engine_error(LocalEngine.MLX)
                    return text
                self._set_engine_error(LocalEngine.MLX, "MLX 응답이 비어 있습니다.")
            except Exception as exc:
                self._set_engine_error(LocalEngine.MLX, f"MLX 추론 실패: {exc}")
                return None
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
            mlx_lm = importlib.import_module("mlx_lm")
            load = mlx_lm.load

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

    def _resolve_mlx_model_path(self, profile: str, explicit_model_path: str | None) -> str | None:
        candidate = (explicit_model_path or "").strip() or (self._profile_to_model(profile) or "").strip()
        if candidate:
            return candidate
        return self._discover_downloaded_model(LocalEngine.MLX)

    def _is_qwen35_model_reference(self, model_path: str | None) -> bool:
        value = str(model_path or "").strip().lower()
        if not value:
            return False
        compact = re.sub(r"[^a-z0-9]+", "", value)
        if "qwen35" in compact:
            return True
        return bool(re.search(r"qwen\s*3(?:[._\-\s]?5)", value))

    def _split_conversation_prompt_for_chat(self, prompt: str) -> tuple[str, str]:
        text = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            return "", ""

        match = re.search(r"(?is)\bInput message:\s*(.+?)\n\s*Answer:\s*$", text)
        if match:
            user_text = match.group(1).strip()
            system_text = text[:match.start()].strip()
            return system_text, user_text

        marker = re.search(r"(?is)\bInput message:\s*", text)
        if not marker:
            return text.strip(), ""

        user_text = text[marker.end():].strip()
        user_text = re.sub(r"(?is)\n\s*Answer:\s*$", "", user_text).strip()
        system_text = text[:marker.start()].strip()
        return system_text, user_text

    def _mlx_prepare_prompt(
        self,
        *,
        prompt: str,
        style: Literal["grounded", "conversation", "rewrite"],
        model_path: str | None,
    ) -> str:
        if style != "conversation":
            return prompt
        if not self._is_qwen35_model_reference(model_path):
            return prompt

        tokenizer = self._mlx_tokenizer
        if tokenizer is None:
            return prompt
        if not getattr(tokenizer, "has_chat_template", False):
            return prompt
        if not hasattr(tokenizer, "apply_chat_template"):
            return prompt

        system_text, user_text = self._split_conversation_prompt_for_chat(prompt)
        if not user_text:
            return prompt

        messages = []
        system_clean = str(system_text or "").strip()
        if system_clean:
            messages.append({"role": "system", "content": system_clean})
        messages.append({"role": "user", "content": str(user_text).strip()})

        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

        rendered_text = str(rendered or "").strip()
        return rendered_text or prompt
