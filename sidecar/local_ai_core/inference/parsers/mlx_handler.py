from __future__ import annotations
import re
import json
import os
import subprocess
import sys
import importlib
import importlib.util
import threading
import time
import select
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from ..base import BaseDelegate
from ..types import InferenceResult, _ConversationCandidateResult
from ..utils import inject_system_instruction_if_needed
from ...models import LocalEngine, WorkMode, AgentAction, Citation, RuntimePrepareResponse

if TYPE_CHECKING:
    from ...local_inference import LocalInferenceEngine

# Component: mlx_handler.py
class MlxHandler(BaseDelegate):
    def __init__(self, engine: LocalInferenceEngine):
        super().__init__(engine)
        self._worker_process = None
        self._worker_lock = threading.Lock()
    def _generate_with_mlx(
        self,
        prompt: str,
        profile: str,
        explicit_model_path: str | None,
        *,
        max_tokens: int,
        style: Literal["grounded", "conversation", "rewrite"] = "grounded",
        sampling_overrides: dict[str, float | int] | None = None,
        message_state: list[dict[str, str]] | None = None,
        response_language: str | None = None,
    ) -> str | None:
        if self._mlx_isolated_generation_enabled():
            isolated_text = self._generate_with_mlx_isolated(
                prompt,
                profile,
                explicit_model_path,
                max_tokens=max_tokens,
                style=style,
                sampling_overrides=sampling_overrides,
                message_state=message_state,
                response_language=response_language,
            )
            if isolated_text:
                return isolated_text
            # Default: do not retry in-process because Metal OOM can terminate the main server process.
            if not self._mlx_allow_inprocess_fallback():
                return None

        if self._ensure_mlx_loaded(profile, explicit_model_path, allow_runtime_install=False):
            try:
                mlx_lm = importlib.import_module("mlx_lm")
                generate = mlx_lm.generate
                model_ref = (explicit_model_path or self._mlx_model_path or self._profile_to_model(profile) or "").strip()
                prepared_prompt = self._mlx_prepare_prompt(
                    prompt=prompt,
                    style=style,
                    model_path=model_ref,
                    message_state=message_state,
                    response_language=response_language,
                )

                sampling = self._sampling_preset(style=style, engine=LocalEngine.MLX)
                if sampling_overrides:
                    sampling.update(sampling_overrides)
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

    def _inject_system_instruction_if_needed(
        self, message_state: list[dict[str, str]], response_language: str | None = None
    ) -> list[dict[str, str]]:
        return inject_system_instruction_if_needed(message_state, response_language)

    def _mlx_prepare_prompt(
        self,
        *,
        prompt: str,
        style: Literal["grounded", "conversation", "rewrite"],
        model_path: str | None,
        message_state: list[dict[str, str]] | None = None,
        response_language: str | None = None,
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

        if message_state:
            messages = self._inject_system_instruction_if_needed(message_state, response_language)
        else:
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

    @staticmethod
    def _mlx_isolated_generation_enabled() -> bool:
        return True

    @staticmethod
    def _mlx_allow_inprocess_fallback() -> bool:
        raw = str(os.getenv("LOCAL_AI_MLX_ALLOW_INPROCESS_FALLBACK", "0") or "0").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _mlx_isolated_generation_timeout_seconds() -> float:
        try:
            return float(os.getenv("LOCAL_AI_MLX_ISOLATED_TIMEOUT", "0"))
        except Exception:
            return 0.0

    @staticmethod
    def _mlx_worker_startup_timeout_seconds() -> float:
        try:
            return float(os.getenv("LOCAL_AI_MLX_WORKER_STARTUP_TIMEOUT", "45"))
        except Exception:
            return 45.0

    def _terminate_worker_process(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            for _ in range(20):
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
            else:
                proc.kill()
        except Exception:
            pass

    def _ensure_mlx_worker(self) -> subprocess.Popen | None:
        with self._worker_lock:
            if self._worker_process is not None:
                if self._worker_process.poll() is None:
                    return self._worker_process
                else:
                    self._worker_process = None

            try:
                env = os.environ.copy()
                cwd = str(Path(__file__).resolve().parents[3]) # PLOS directory
                env["PYTHONPATH"] = cwd

                cmd = [
                    sys.executable,
                    "-m",
                    "local_ai_core.inference.mlx_isolated_worker",
                    "--mode",
                    "persistent"
                ]
                
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    # Prevent deadlock from unread stderr pipe filling up.
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    env=env,
                    cwd=cwd
                )

                startup_timeout = max(3.0, self._mlx_worker_startup_timeout_seconds())
                start_at = time.time()
                ready_line = ""
                while (time.time() - start_at) < startup_timeout:
                    if proc.poll() is not None:
                        break
                    r, _, _ = select.select([proc.stdout], [], [], 0.25)
                    if proc.stdout not in r:
                        continue
                    ready_line = proc.stdout.readline()
                    if not ready_line:
                        continue
                    try:
                        status = json.loads(ready_line)
                    except Exception:
                        continue
                    if status.get("status") == "ready":
                        self._worker_process = proc
                        self._clear_engine_error(LocalEngine.MLX)
                        return proc
                    error_msg = status.get("error", "Unknown initialization status")
                    self._set_engine_error(LocalEngine.MLX, f"worker_start_failed: {error_msg}")
                    break
                else:
                    self._set_engine_error(LocalEngine.MLX, "worker_start_failed: startup_timeout")

                self._terminate_worker_process(proc)
                return None
            except Exception as exc:
                self._set_engine_error(LocalEngine.MLX, f"worker_start_failed: {str(exc)}")
                return None

    def _read_worker_json_line(self, proc: subprocess.Popen, timeout_seconds: float) -> dict | None:
        if timeout_seconds <= 0:
            while True:
                if proc.poll() is not None:
                    return None
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        return None
                    continue
                try:
                    return json.loads(line)
                except Exception:
                    continue
        start_time = time.time()
        while True:
            remaining = timeout_seconds - (time.time() - start_time)
            if remaining <= 0:
                return None
            
            r, _, _ = select.select([proc.stdout], [], [], min(1.0, remaining))
            if proc.stdout in r:
                line = proc.stdout.readline()
                if not line:
                    return None
                try:
                    return json.loads(line)
                except Exception:
                    continue
            
            if proc.poll() is not None:
                return None

    def _read_worker_stream_result(self, proc: subprocess.Popen, timeout_seconds: float) -> dict | None:
        if timeout_seconds <= 0:
            while True:
                if proc.poll() is not None:
                    return None
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        return None
                    continue
                try:
                    data = json.loads(line)
                    event = data.get("event")
                    if event == "token":
                        piece = data.get("text", "")
                        if piece:
                            self._emit_stream_token(piece)
                        continue
                    return data
                except Exception:
                    continue
        start_time = time.time()
        while True:
            remaining = timeout_seconds - (time.time() - start_time)
            if remaining <= 0:
                return None
            
            r, _, _ = select.select([proc.stdout], [], [], min(1.0, remaining))
            if proc.stdout in r:
                line = proc.stdout.readline()
                if not line:
                    return None
                try:
                    data = json.loads(line)
                    event = data.get("event")
                    if event == "token":
                        piece = data.get("text", "")
                        if piece:
                            self._emit_stream_token(piece)
                    else:
                        return data
                except Exception:
                    continue
            
            if proc.poll() is not None:
                return None

    def _generate_with_mlx_isolated(
        self,
        prompt: str,
        profile: str,
        explicit_model_path: str | None,
        *,
        max_tokens: int,
        style: Literal["grounded", "conversation", "rewrite"],
        sampling_overrides: dict[str, float | int] | None = None,
        message_state: list[dict[str, str]] | None = None,
        response_language: str | None = None,
    ) -> str | None:
        model_ref = (explicit_model_path or self._resolve_mlx_model_path(profile, explicit_model_path) or "").strip()
        if not model_ref:
            self._set_engine_error(LocalEngine.MLX, "MLX 모델 경로가 비어 있습니다.")
            return None

        proc = self._ensure_mlx_worker()
        if not proc:
            return None

        sampling = self._sampling_preset(style=style, engine=LocalEngine.MLX)
        if sampling_overrides:
            sampling.update(sampling_overrides)

        stop_sequences = self._stop_sequences_for_model(
            style=style,
            model_path=model_ref,
            default_sequences=[
                "User:",
                "Assistant:",
                "<|end_of_turn|>",
                "<|eot_id|>",
                "<|endoftext|>",
            ],
        )

        prepared_prompt = self._mlx_prepare_prompt(
            prompt=prompt,
            style=style,
            model_path=model_ref,
            message_state=message_state,
            response_language=response_language,
        )

        payload = {
            "model_path": model_ref,
            "kwargs": {
                "prompt": prepared_prompt,
                "max_tokens": max_tokens,
                "temp": sampling["temperature"],
                "top_p": sampling["top_p"],
                "repetition_penalty": sampling["repeat_penalty"],
                "stop_sequences": stop_sequences,
                "message_state": message_state,
                "response_language": response_language,
            },
            "style": str(style),
            "stream": True,
        }

        try:
            with self._worker_lock:
                proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                proc.stdin.flush()

                timeout = self._mlx_isolated_generation_timeout_seconds()
                result = self._read_worker_stream_result(proc, timeout)

                if not result:
                    self._terminate_worker_process(proc)
                    self._worker_process = None
                    self._set_engine_error(LocalEngine.MLX, "워커 응답 타임아웃 또는 프로세스 종료")
                    return None

                if not result.get("ok"):
                    error_msg = result.get("error", "Unknown worker error")
                    self._set_engine_error(LocalEngine.MLX, f"워커 추론 실패: {error_msg}")
                    return None

                text = result.get("text", "")
                if text:
                    self._clear_engine_error(LocalEngine.MLX)
                    return text
                
                self._set_engine_error(LocalEngine.MLX, "워커 응답이 비어 있습니다.")
                return None
        except Exception as e:
            self._terminate_worker_process(proc)
            self._worker_process = None
            self._set_engine_error(LocalEngine.MLX, f"워커 통신 실패: {e}")
            return None
