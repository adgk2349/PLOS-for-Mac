import pytest

from local_ai_core.external_providers import AnthropicProvider, OpenAIProvider
from local_ai_core.models import WorkMode


@pytest.mark.asyncio
async def test_openai_provider_without_key_returns_guidance_message():
    provider = OpenAIProvider(api_key=None)
    result = await provider.analyze("test", WorkMode.GENERAL, citations=[])
    assert "API key" in result.answer or "API 키" in result.answer
    assert result.sent_chars > 0


@pytest.mark.asyncio
async def test_anthropic_provider_without_key_returns_guidance_message():
    provider = AnthropicProvider(api_key=None)
    result = await provider.analyze("test", WorkMode.GENERAL, citations=[])
    assert "API key" in result.answer or "API 키" in result.answer
    assert result.sent_chars > 0
