from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from .language_utils import resolve_response_language, response_language_instruction
from .models import Citation, WorkMode


def _context_block(citations: list[Citation]) -> str:
    if not citations:
        return "No local citations provided."
    lines = []
    for c in citations:
        lines.append(f"- {c.file_path}: {c.snippet[:240]}")
    return "\n".join(lines)


@dataclass(slots=True)
class ProviderResult:
    answer: str
    sent_chars: int


class OpenAIProvider:
    def __init__(self, api_key: str | None, model: str = "gpt-5-mini"):
        self.api_key = api_key
        self.model = model

    async def analyze(
        self,
        query: str,
        mode: WorkMode,
        citations: list[Citation],
        language_preference: str | None = None,
    ) -> ProviderResult:
        response_language = resolve_response_language(query, language_preference)
        prompt = (
            "You are a deep-analysis assistant. Respect local citations and avoid hallucinations.\n"
            f"{response_language_instruction(response_language)}\n"
            f"Mode: {mode.value}\n"
            "Local citations:\n"
            f"{_context_block(citations)}\n\n"
            f"User query: {query}"
        )
        sent_chars = len(prompt)

        if not self.api_key:
            return ProviderResult(
                answer=_missing_api_key_message("OpenAI", response_language),
                sent_chars=sent_chars,
            )

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "input": prompt,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = _extract_openai_text(data)
        return ProviderResult(answer=text, sent_chars=sent_chars)


class AnthropicProvider:
    def __init__(self, api_key: str | None, model: str = "claude-3-7-sonnet-latest"):
        self.api_key = api_key
        self.model = model

    async def analyze(
        self,
        query: str,
        mode: WorkMode,
        citations: list[Citation],
        language_preference: str | None = None,
    ) -> ProviderResult:
        response_language = resolve_response_language(query, language_preference)
        prompt = (
            "Use only grounded reasoning from local citation context when possible.\n"
            f"{response_language_instruction(response_language)}\n"
            f"Mode: {mode.value}\n"
            f"Local citations:\n{_context_block(citations)}\n\n"
            f"Question: {query}"
        )
        sent_chars = len(prompt)

        if not self.api_key:
            return ProviderResult(
                answer=_missing_api_key_message("Anthropic", response_language),
                sent_chars=sent_chars,
            )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = _extract_anthropic_text(data)
        return ProviderResult(answer=text, sent_chars=sent_chars)


class ProviderRouter:
    def __init__(self):
        self._openai = OpenAIProvider(os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
        self._anthropic = AnthropicProvider(
            os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest"),
        )

    async def analyze(
        self,
        provider: str,
        query: str,
        mode: WorkMode,
        citations: list[Citation],
        language_preference: str | None = None,
    ) -> ProviderResult:
        if provider == "openai":
            return await self._openai.analyze(query, mode, citations, language_preference=language_preference)
        if provider == "anthropic":
            return await self._anthropic.analyze(query, mode, citations, language_preference=language_preference)
        raise ValueError(f"Unsupported provider: {provider}")


def _missing_api_key_message(provider: str, response_language: str) -> str:
    if response_language == "ko":
        return f"{provider} API 키가 설정되지 않아 외부 분석을 수행하지 못했습니다."
    return f"{provider} API key is not configured, so deep analysis could not run."


def _extract_openai_text(payload: dict) -> str:
    if "output_text" in payload and isinstance(payload["output_text"], str):
        return payload["output_text"]

    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                chunks.append(text)
    if chunks:
        return "\n".join(chunks)
    return "Failed to parse external analysis response."


def _extract_anthropic_text(payload: dict) -> str:
    chunks: list[str] = []
    for item in payload.get("content", []):
        text = item.get("text")
        if text:
            chunks.append(text)
    if chunks:
        return "\n".join(chunks)
    return "Failed to parse external analysis response."
