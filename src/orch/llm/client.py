"""LLM client for complexity detection."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

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


class LLMClientFactory:
    """
    Factory for creating LLM clients based on configuration.

    Supports multiple providers with automatic API key detection.
    Falls back gracefully if no API key is available.
    """

    # Provider -> (env var, model prefix, client class)
    PROVIDERS: ClassVar[dict[str, tuple[str, str, type]]] = {
        "anthropic": ("ANTHROPIC_API_KEY", "claude-", AnthropicLLMClient),
        "openai": ("OPENAI_API_KEY", "gpt-", OpenAILLMClient),
    }

    @classmethod
    def create(
        cls,
        config,
        preferred_provider: str | None = None
    ) -> LLMClient | None:
        """
        Create an LLM client based on config and available API keys.

        Returns LLMClient if API key available, None otherwise.
        """
        detection_model = config.orchestration.detection_model

        # Determine provider from model name or preference
        provider = preferred_provider
        if not provider:
            provider = cls._infer_provider(detection_model)

        if not provider:
            # Try to find any available provider
            provider = cls._find_available_provider()

        if not provider:
            logger.warning("No LLM provider available (no API keys found)")
            return None

        # Get API key and create client
        env_var, _, client_class = cls.PROVIDERS[provider]
        api_key = os.getenv(env_var)

        if not api_key:
            logger.warning(f"No API key found for {provider} (set {env_var})")
            return None

        logger.info(f"Using {provider} LLM client for complexity detection")
        return client_class(api_key)

    @classmethod
    def _infer_provider(cls, model: str) -> str | None:
        """Infer provider from model name prefix."""
        for provider, (_, prefix, _) in cls.PROVIDERS.items():
            if model.startswith(prefix):
                return provider
        return None

    @classmethod
    def _find_available_provider(cls) -> str | None:
        """Find first provider with available API key."""
        for provider, (env_var, _, _) in cls.PROVIDERS.items():
            if os.getenv(env_var):
                return provider
        return None

    @classmethod
    def is_available(cls) -> bool:
        """Check if any LLM provider is available."""
        return cls._find_available_provider() is not None
