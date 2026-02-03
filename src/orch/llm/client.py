"""LLM client for complexity detection."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        system: str | None = None
    ) -> LLMResponse:
        """Send prompt to LLM and get response."""
        pass
