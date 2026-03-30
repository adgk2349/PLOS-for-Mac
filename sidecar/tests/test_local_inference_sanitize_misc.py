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

def test_looks_model_answer_rejects_pathological_repetition():
    raw = ("전송을 시작하기 전에 " * 28).strip()
    assert not LocalInferenceEngine._looks_model_answer(raw, min_length=10)

def test_prepare_evidence_lines_dedupes_and_compresses_repeated_snippets():
    now = datetime.now(timezone.utc)
    citations = [
        Citation(
            doc_id="doc-a",
            chunk_id="a-1",
            file_path="/tmp/a.txt",
            snippet="전송을 시작하기 전에 " * 20,
            score=0.9,
            modified_at=now,
        ),
        Citation(
            doc_id="doc-a",
            chunk_id="a-2",
            file_path="/tmp/a.txt",
            snippet="전송을 시작하기 전에 확인 절차를 점검하세요.",
            score=0.88,
            modified_at=now,
        ),
        Citation(
            doc_id="doc-b",
            chunk_id="b-1",
            file_path="/tmp/b.txt",
            snippet="프로토콜 초기화 단계에서는 체크리스트를 먼저 확인합니다.",
            score=0.8,
            modified_at=now,
        ),
    ]
    lines = LocalInferenceEngine._prepare_evidence_lines(
        citations=citations,
        max_items=5,
        response_language="ko",
    )
    assert len(lines) <= 3
    assert lines[0].count("전송을 시작하기 전에") <= 2
    assert "(a.txt)" in lines[0]

def test_is_recommendation_chat_query_excludes_file_tasks():
    assert LocalInferenceEngine._is_recommendation_chat_query("오늘 저녁 메뉴 추천해줘")
    assert not LocalInferenceEngine._is_recommendation_chat_query("7주차 파일 찾아줘")

def test_is_qwen35_model_reference_detects_variants():
    assert LocalInferenceEngine._is_qwen35_model_reference("/tmp/qwen35_9b_gguf")
    assert LocalInferenceEngine._is_qwen35_model_reference("/tmp/Qwen3.5-9B-MLX-4bit")
    assert LocalInferenceEngine._is_qwen35_model_reference("/tmp/QWEN3_5_model")
    assert not LocalInferenceEngine._is_qwen35_model_reference("/tmp/gemma-3-12b")

def test_resolve_max_tokens_applies_memory_cap_even_for_advanced(monkeypatch):
    monkeypatch.setenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "16")
    assert LocalInferenceEngine._resolve_max_tokens(4096, profile="advanced") == 1536
