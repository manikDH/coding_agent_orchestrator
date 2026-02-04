"""Tests for LLM client."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from orch.llm.client import LLMResponse


def test_llm_response_creation():
    """Test LLMResponse dataclass creation."""
    response = LLMResponse(content="test content", model="claude-3-haiku-20240307", tokens_used=50)

    assert response.content == "test content"
    assert response.model == "claude-3-haiku-20240307"
    assert response.tokens_used == 50


@pytest.mark.asyncio
async def test_openai_client_complete():
    """Test OpenAILLMClient.complete() call."""
    with patch("orch.llm.client.openai") as mock_openai:
        # Setup mock response
        mock_choice = Mock()
        mock_choice.message.content = "openai response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage.total_tokens = 45

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.AsyncOpenAI.return_value = mock_client

        from orch.llm.client import OpenAILLMClient

        client = OpenAILLMClient("test-key")

        result = await client.complete(
            prompt="test prompt", model="gpt-4o-mini", max_tokens=100, system="You are helpful"
        )

        assert result.content == "openai response"
        assert result.tokens_used == 45


def test_anthropic_client_creation():
    """Test AnthropicLLMClient initialization."""
    with patch("orch.llm.client.anthropic") as mock_anthropic:
        mock_client = Mock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        from orch.llm.client import AnthropicLLMClient

        client = AnthropicLLMClient("test-api-key")
        assert client.api_key == "test-api-key"
        mock_anthropic.AsyncAnthropic.assert_called_once_with(api_key="test-api-key")


@pytest.mark.asyncio
async def test_anthropic_client_complete():
    """Test AnthropicLLMClient.complete() call."""
    with patch("orch.llm.client.anthropic") as mock_anthropic:
        # Setup mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="response text")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        from orch.llm.client import AnthropicLLMClient

        client = AnthropicLLMClient("test-key")

        result = await client.complete(
            prompt="test prompt", model="claude-3-haiku-20240307", max_tokens=100, temperature=0.0
        )

        assert result.content == "response text"
        assert result.tokens_used == 30


def test_factory_infer_anthropic_provider():
    """Test factory infers Anthropic from claude- prefix."""
    from orch.llm.client import LLMClientFactory

    provider = LLMClientFactory._infer_provider("claude-3-haiku-20240307")
    assert provider == "anthropic"


def test_factory_infer_openai_provider():
    """Test factory infers OpenAI from gpt- prefix."""
    from orch.llm.client import LLMClientFactory

    provider = LLMClientFactory._infer_provider("gpt-4o-mini")
    assert provider == "openai"


def test_factory_returns_none_without_api_key():
    """Test factory returns None when no API key available."""
    from orch.llm.client import LLMClientFactory

    mock_config = Mock()
    mock_config.orchestration.detection_model = "claude-3-haiku-20240307"

    with patch.dict(os.environ, {}, clear=True):
        client = LLMClientFactory.create(mock_config)

    assert client is None


def test_factory_creates_anthropic_client():
    """Test factory creates Anthropic client with API key."""
    from orch.llm.client import AnthropicLLMClient, LLMClientFactory

    mock_config = Mock()
    mock_config.orchestration.detection_model = "claude-3-haiku-20240307"

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch('orch.llm.client.anthropic'):
            client = LLMClientFactory.create(mock_config)

    assert isinstance(client, AnthropicLLMClient)


def test_factory_is_available():
    """Test is_available returns True with API key."""
    from orch.llm.client import LLMClientFactory

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
        assert LLMClientFactory.is_available() is True

    with patch.dict(os.environ, {}, clear=True):
        assert LLMClientFactory.is_available() is False
