from __future__ import annotations

# Auto-split from large test module for maintainability.

import importlib

from datetime import datetime, timezone

from pathlib import Path

from local_ai_core.local_inference import LocalInferenceEngine

from local_ai_core.models import Citation, LocalEngine, WorkMode

class _StubInferenceEngine(LocalInferenceEngine):
    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs

    def _generate_with_engine(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
        style: str = "grounded",
    ) -> str | None:
        output = self._outputs.get(engine)
        if output is None:
            self._set_engine_error(engine, f"{engine.value} stub failure")
            return None
        return output

    def _discover_downloaded_model(self, engine: LocalEngine) -> str | None:
        _ = engine
        return None

class _SequentialStubInferenceEngine(LocalInferenceEngine):
    def __init__(self, outputs_by_engine):
        super().__init__()
        self._outputs_by_engine = {engine: list(outputs) for engine, outputs in outputs_by_engine.items()}

    def _generate_with_engine(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
        style: str = "grounded",
    ) -> str | None:
        values = self._outputs_by_engine.get(engine) or []
        if not values:
            self._set_engine_error(engine, f"{engine.value} stub failure")
            return None
        value = values.pop(0)
        if value is None:
            self._set_engine_error(engine, f"{engine.value} stub failure")
            return None
        return value

    def _discover_downloaded_model(self, engine: LocalEngine) -> str | None:
        _ = engine
        return None

def test_resolve_engine_with_compatibility_switches_mlx_to_llama_for_gguf(tmp_path: Path):
    engine = LocalInferenceEngine()
    gguf = tmp_path / "demo.gguf"
    gguf.write_bytes(b"GGUF")

    effective, mlx_path, llama_path, route = engine._resolve_engine_with_compatibility(
        engine=LocalEngine.MLX,
        mlx_model_path=str(gguf),
        llama_model_path=None,
    )

    assert effective == LocalEngine.LLAMA_CPP
    assert mlx_path == str(gguf)
    assert llama_path == str(gguf)
    assert "auto_switch:mlx_path_is_gguf" in route

def test_resolve_engine_with_compatibility_switches_llama_to_mlx_for_directory(tmp_path: Path):
    engine = LocalInferenceEngine()
    mlx_dir = tmp_path / "mlx-model"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    (mlx_dir / "config.json").write_text("{}", encoding="utf-8")

    effective, mlx_path, llama_path, route = engine._resolve_engine_with_compatibility(
        engine=LocalEngine.LLAMA_CPP,
        mlx_model_path=None,
        llama_model_path=str(mlx_dir),
    )

    assert effective == LocalEngine.MLX
    assert mlx_path == str(mlx_dir)
    assert llama_path == str(mlx_dir)
    assert "auto_switch:llama_path_is_mlx" in route

def test_mlx_prepare_prompt_uses_chat_template_without_thinking_for_qwen_conversation():
    class _DummyTokenizer:
        has_chat_template = True

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            self.calls.append(
                {
                    "messages": messages,
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "kwargs": kwargs,
                }
            )
            return "TEMPLATED_PROMPT"

    engine = LocalInferenceEngine()
    dummy = _DummyTokenizer()
    engine._mlx_tokenizer = dummy
    prompt = (
        "You are a conversational local AI assistant.\n"
        "Mode: GENERAL\n"
        "Input message: 안녕\n"
        "Answer:"
    )

    rendered = engine._mlx_prepare_prompt(
        prompt=prompt,
        style="conversation",
        model_path="/tmp/qwen35_9b_gguf",
    )

    assert rendered == "TEMPLATED_PROMPT"
    assert len(dummy.calls) == 1
    call = dummy.calls[0]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    assert call["kwargs"].get("enable_thinking") is False
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"

def test_mlx_prepare_prompt_uses_chat_template_without_thinking_for_qwen_conversation_instance():
    class _DummyTokenizer:
        has_chat_template = True

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            self.calls.append(
                {
                    "messages": messages,
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "kwargs": kwargs,
                }
            )
            return "INSTANCE_TEMPLATED_PROMPT"

    engine = LocalInferenceEngine()
    dummy = _DummyTokenizer()
    engine._mlx_tokenizer = dummy
    rendered = engine._mlx_prepare_prompt(
        prompt="You are a conversational local AI assistant.\nMode: GENERAL\nInput message: 안녕\nAnswer:",
        style="conversation",
        model_path="/tmp/qwen35_9b_gguf",
    )
    assert rendered == "INSTANCE_TEMPLATED_PROMPT"
    assert len(dummy.calls) == 1
    assert dummy.calls[0]["kwargs"].get("enable_thinking") is False

def test_mlx_prepare_prompt_skips_template_for_non_qwen_model():
    class _DummyTokenizer:
        has_chat_template = True

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            return "UNEXPECTED_TEMPLATE"

    engine = LocalInferenceEngine()
    engine._mlx_tokenizer = _DummyTokenizer()
    prompt = "Input message: hello\nAnswer:"
    rendered = engine._mlx_prepare_prompt(
        prompt=prompt,
        style="conversation",
        model_path="/tmp/gemma-3-12b-it",
    )
    assert rendered == prompt

def test_mlx_context_window_hint_reads_max_position_embeddings(tmp_path: Path):
    model_dir = tmp_path / "qwen35_9b_gguf"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        '{"text_config":{"max_position_embeddings":262144}}',
        encoding="utf-8",
    )
    assert LocalInferenceEngine._mlx_context_window_hint(str(model_dir)) == 262144

def test_llama_runtime_context_window_uses_env_override(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "64")
    monkeypatch.setenv("LOCAL_AI_LLAMA_N_CTX", "8192")
    assert LocalInferenceEngine._llama_runtime_context_window() == 8192

def test_llama_runtime_context_window_applies_memory_cap(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "16")
    monkeypatch.setenv("LOCAL_AI_LLAMA_N_CTX", "32768")
    assert LocalInferenceEngine._llama_runtime_context_window(profile="advanced") == 6144

def test_generate_with_llama_conversation_path_preserves_long_text(monkeypatch):
    class _DummyLlama:
        def create_completion(self, **_kwargs):
            return {"choices": [{"text": "긴 대화 응답 " * 260}]}

    engine = LocalInferenceEngine()
    engine._llama_model = _DummyLlama()
    monkeypatch.setattr(engine.llama_h, "_ensure_llama_loaded", lambda *_args, **_kwargs: True)

    output = engine._generate_with_llama(
        "Input message: 테스트\nAnswer:",
        explicit_model_path=None,
        max_tokens=1536,
        style="conversation",
    )

    assert output is not None
    assert len(output) > 1200

def test_mlx_chunk_to_text_accepts_generation_response_like_object():
    class _DummyChunk:
        def __init__(self):
            self.text = "hello"

    assert LocalInferenceEngine._mlx_chunk_to_text(_DummyChunk()) == "hello"

def test_llama_runtime_context_window_for_model_caps_20b_on_16gb(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "16")
    monkeypatch.delenv("LOCAL_AI_LLAMA_N_CTX", raising=False)
    engine = LocalInferenceEngine()
    context = engine._llama_runtime_context_window_for_model(
        profile="advanced",
        model_path="/tmp/gpt-oss-20b-mxfp4.gguf",
    )
    assert context == 3072

def test_reload_llama_with_reduced_context_halves_current_window(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "16")
    monkeypatch.delenv("LOCAL_AI_LLAMA_N_CTX", raising=False)
    engine = LocalInferenceEngine()
    assert engine._reload_llama_with_reduced_context(
        profile="advanced",
        explicit_model_path="/tmp/gpt-oss-20b-mxfp4.gguf",
    )
    overrides = getattr(engine, "_llama_n_ctx_overrides")
    assert overrides["/tmp/gpt-oss-20b-mxfp4.gguf"] == 1536

def test_llama_gpu_layers_for_20b_on_16gb_uses_cpu(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "16")
    engine = LocalInferenceEngine()
    assert engine._llama_gpu_layers_for_model("/tmp/gpt-oss-20b-mxfp4.gguf") == 0

def test_ensure_runtime_module_reports_import_failure(monkeypatch):
    engine = LocalInferenceEngine()
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())

    def _raise_import_error(_name: str):
        raise RuntimeError("broken import")

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)

    ok = engine._ensure_runtime_module(
        engine=LocalEngine.MLX,
        module_name="mlx_lm",
        package_spec="mlx-lm>=0.26.0",
        allow_install=False,
    )
    assert ok
