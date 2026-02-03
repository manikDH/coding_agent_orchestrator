"""Tests for LLM client."""
from orch.llm.client import LLMResponse


def test_llm_response_creation():
    """Test LLMResponse dataclass creation."""
    response = LLMResponse(
        content="test content",
        model="claude-3-haiku-20240307",
        tokens_used=50
    )

    assert response.content == "test content"
    assert response.model == "claude-3-haiku-20240307"
    assert response.tokens_used == 50
