"""LLM client for complexity detection."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    tokens_used: int


class LLMClient(ABC):
    """Abstract LLM client for complexity detection."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Send prompt to LLM and get response."""
        pass


class AnthropicLLMClient(LLMClient):
    """Anthropic API client for complexity detection."""

    def __init__(self, api_key: str):
        if anthropic is None:
            raise ImportError("anthropic package is required for AnthropicLLMClient")
        self.api_key = api_key
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Call Anthropic API."""
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )


class OpenAILLMClient(LLMClient):
    """OpenAI API client for complexity detection."""

    def __init__(self, api_key: str):
        if openai is None:
            raise ImportError("openai package is required for OpenAILLMClient")
        self.api_key = api_key
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Call OpenAI API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=temperature, messages=messages
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )
