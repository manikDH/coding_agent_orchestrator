# ComplexityAnalyzer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement LLM-based complexity detection that auto-routes tasks to appropriate model tiers.

**Architecture:** ComplexityAnalyzer uses LLM (Haiku) for fast classification, returns ComplexityResult with tier recommendations. LLMClientFactory provides provider-agnostic access (Anthropic/OpenAI). TeamOrchestrator integrates detection before agent creation.

**Tech Stack:** Python 3.11+, pytest, pytest-asyncio, anthropic, openai, pydantic

---

## Task 1: Create LLMClient Base and LLMResponse

**Files:**
- Create: `src/orch/llm/__init__.py`
- Create: `src/orch/llm/client.py`
- Create: `tests/unit/llm/__init__.py`
- Create: `tests/unit/llm/test_client.py`

**Step 1: Create package init**

Create the directory and init files:

```bash
mkdir -p src/orch/llm tests/unit/llm
touch src/orch/llm/__init__.py tests/unit/llm/__init__.py
```

**Step 2: Write the failing test for LLMResponse**

```python
# tests/unit/llm/test_client.py
"""Tests for LLM client."""
import pytest
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
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/unit/llm/test_client.py::test_llm_response_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'orch.llm.client'"

**Step 4: Write minimal implementation**

```python
# src/orch/llm/client.py
"""LLM client for complexity detection."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

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
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/unit/llm/test_client.py::test_llm_response_creation -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/orch/llm/ tests/unit/llm/
git commit -m "feat(llm): add LLMClient base class and LLMResponse dataclass"
```

---

## Task 2: Implement AnthropicLLMClient

**Files:**
- Modify: `src/orch/llm/client.py`
- Modify: `tests/unit/llm/test_client.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/llm/test_client.py
from unittest.mock import AsyncMock, Mock, patch
import pytest


@pytest.fixture
def mock_anthropic():
    """Mock anthropic module."""
    with patch.dict('sys.modules', {'anthropic': Mock()}):
        yield


def test_anthropic_client_creation(mock_anthropic):
    """Test AnthropicLLMClient initialization."""
    from orch.llm.client import AnthropicLLMClient

    client = AnthropicLLMClient("test-api-key")
    assert client.api_key == "test-api-key"


@pytest.mark.asyncio
async def test_anthropic_client_complete():
    """Test AnthropicLLMClient.complete() call."""
    with patch('orch.llm.client.anthropic') as mock_anthropic:
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
            prompt="test prompt",
            model="claude-3-haiku-20240307",
            max_tokens=100,
            temperature=0.0
        )

        assert result.content == "response text"
        assert result.tokens_used == 30
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/llm/test_client.py::test_anthropic_client_creation -v`
Expected: FAIL with "cannot import name 'AnthropicLLMClient'"

**Step 3: Write minimal implementation**

```python
# Add to src/orch/llm/client.py after LLMClient class

class AnthropicLLMClient(LLMClient):
    """Anthropic API client for complexity detection."""

    def __init__(self, api_key: str):
        import anthropic
        self.api_key = api_key
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        system: str | None = None
    ) -> LLMResponse:
        """Call Anthropic API."""
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/llm/test_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/llm/client.py tests/unit/llm/test_client.py
git commit -m "feat(llm): add AnthropicLLMClient implementation"
```

---

## Task 3: Implement OpenAILLMClient

**Files:**
- Modify: `src/orch/llm/client.py`
- Modify: `tests/unit/llm/test_client.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/llm/test_client.py

@pytest.mark.asyncio
async def test_openai_client_complete():
    """Test OpenAILLMClient.complete() call."""
    with patch('orch.llm.client.openai') as mock_openai:
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
            prompt="test prompt",
            model="gpt-4o-mini",
            max_tokens=100,
            system="You are helpful"
        )

        assert result.content == "openai response"
        assert result.tokens_used == 45
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/llm/test_client.py::test_openai_client_complete -v`
Expected: FAIL with "cannot import name 'OpenAILLMClient'"

**Step 3: Write minimal implementation**

```python
# Add to src/orch/llm/client.py after AnthropicLLMClient

class OpenAILLMClient(LLMClient):
    """OpenAI API client for complexity detection."""

    def __init__(self, api_key: str):
        import openai
        self.api_key = api_key
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        system: str | None = None
    ) -> LLMResponse:
        """Call OpenAI API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            tokens_used=response.usage.total_tokens if response.usage else 0
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/llm/test_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/llm/client.py tests/unit/llm/test_client.py
git commit -m "feat(llm): add OpenAILLMClient implementation"
```

---

## Task 4: Implement LLMClientFactory

**Files:**
- Modify: `src/orch/llm/client.py`
- Modify: `tests/unit/llm/test_client.py`

**Step 1: Write the failing tests**

```python
# Add to tests/unit/llm/test_client.py
import os


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
    from orch.llm.client import LLMClientFactory, AnthropicLLMClient

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/llm/test_client.py::test_factory_infer_anthropic_provider -v`
Expected: FAIL with "cannot import name 'LLMClientFactory'"

**Step 3: Write minimal implementation**

```python
# Add to src/orch/llm/client.py after OpenAILLMClient
import os


class LLMClientFactory:
    """
    Factory for creating LLM clients based on configuration.

    Supports multiple providers with automatic API key detection.
    Falls back gracefully if no API key is available.
    """

    # Provider -> (env var, model prefix, client class)
    PROVIDERS: dict[str, tuple[str, str, type]] = {
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/llm/test_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/llm/client.py tests/unit/llm/test_client.py
git commit -m "feat(llm): add LLMClientFactory for provider-agnostic client creation"
```

---

## Task 5: Add Orchestration Config Schema

**Files:**
- Modify: `src/orch/config/schema.py`
- Create: `tests/unit/config/test_schema.py`

**Step 1: Write the failing test**

```python
# tests/unit/config/test_schema.py
"""Tests for configuration schema."""
import pytest
from orch.config.schema import OrchestrationConfig, ComplexityConfig, OrchConfig


def test_orchestration_config_defaults():
    """Test OrchestrationConfig has correct defaults."""
    config = OrchestrationConfig()

    assert config.auto_detect is True
    assert config.default_complexity == "auto"
    assert config.detection_model == "claude-3-haiku-20240307"
    assert config.complexity.confidence_threshold == 0.7


def test_complexity_config_validation():
    """Test ComplexityConfig validates confidence threshold."""
    config = ComplexityConfig(confidence_threshold=0.85)
    assert config.confidence_threshold == 0.85


def test_orch_config_includes_orchestration():
    """Test OrchConfig includes orchestration section."""
    config = OrchConfig.default()

    assert hasattr(config, 'orchestration')
    assert config.orchestration.auto_detect is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/config/test_schema.py -v`
Expected: FAIL with "cannot import name 'OrchestrationConfig'"

**Step 3: Write minimal implementation**

```python
# Add to src/orch/config/schema.py after imports, before ModelTiers

class ComplexityConfig(BaseModel):
    """Complexity detection configuration."""
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600


class OrchestrationConfig(BaseModel):
    """Orchestration configuration for team-of-rivals."""
    auto_detect: bool = True
    default_complexity: str = "auto"  # "auto" | "simple" | "standard" | "complex"
    detection_model: str = "claude-3-haiku-20240307"
    complexity: ComplexityConfig = Field(default_factory=ComplexityConfig)
```

Then update OrchConfig class:

```python
# In OrchConfig class, add after tmux field:
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/config/test_schema.py -v`
Expected: PASS

**Step 5: Commit**

```bash
mkdir -p tests/unit/config
touch tests/unit/config/__init__.py
git add src/orch/config/schema.py tests/unit/config/
git commit -m "feat(config): add OrchestrationConfig with complexity detection settings"
```

---

## Task 6: Create ComplexityResult and DetectionSource

**Files:**
- Create: `src/orch/orchestration/complexity.py`
- Create: `tests/unit/orchestration/test_complexity.py`

**Step 1: Write the failing test**

```python
# tests/unit/orchestration/test_complexity.py
"""Tests for ComplexityAnalyzer."""
import pytest
from orch.orchestration.complexity import (
    ComplexityResult, DetectionSource,
    VALID_COMPLEXITY_LEVELS, VALID_TASK_TYPES
)


def test_detection_source_enum():
    """Test DetectionSource enum values."""
    assert DetectionSource.LLM_DETECTED.value == "llm_detected"
    assert DetectionSource.CONFIG_DEFAULT.value == "config_default"
    assert DetectionSource.ERROR_FALLBACK.value == "error_fallback"


def test_complexity_result_creation():
    """Test ComplexityResult dataclass."""
    result = ComplexityResult(
        complexity_level="complex",
        task_types=["security_sensitive"],
        reasoning="Auth task is security-critical",
        confidence=0.92,
        recommended_models={"planner": "high", "executor": "highest"},
        source=DetectionSource.LLM_DETECTED
    )

    assert result.complexity_level == "complex"
    assert "security_sensitive" in result.task_types
    assert result.confidence == 0.92


def test_complexity_result_to_dict():
    """Test ComplexityResult serialization."""
    result = ComplexityResult(
        complexity_level="standard",
        task_types=["testing_required"],
        reasoning="Needs tests",
        confidence=0.85,
        recommended_models={"planner": "medium"},
        source=DetectionSource.LLM_DETECTED
    )

    data = result.to_dict()

    assert data["complexity_level"] == "standard"
    assert data["source"] == "llm_detected"
    assert "recommended_models" in data


def test_valid_constants():
    """Test validation constants are defined."""
    assert "simple" in VALID_COMPLEXITY_LEVELS
    assert "standard" in VALID_COMPLEXITY_LEVELS
    assert "complex" in VALID_COMPLEXITY_LEVELS

    assert "security_sensitive" in VALID_TASK_TYPES
    assert "architectural" in VALID_TASK_TYPES
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/orchestration/test_complexity.py::test_detection_source_enum -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/orch/orchestration/complexity.py
"""Complexity analysis for automatic task classification."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import logging

logger = logging.getLogger(__name__)


class DetectionSource(Enum):
    """How complexity was determined."""
    LLM_DETECTED = "llm_detected"
    CONFIG_DEFAULT = "config_default"
    LOW_CONFIDENCE_FALLBACK = "low_confidence_fallback"
    ERROR_FALLBACK = "error_fallback"
    CLI_OVERRIDE = "cli_override"


# Valid values for JSON schema validation
VALID_COMPLEXITY_LEVELS = {"simple", "standard", "complex"}
VALID_TASK_TYPES = {
    "security_sensitive", "architectural", "performance_critical",
    "data_sensitive", "testing_required"
}


@dataclass
class ComplexityResult:
    """Result of complexity analysis."""
    complexity_level: Literal["simple", "standard", "complex"]
    task_types: list[str]
    reasoning: str
    confidence: float
    recommended_models: dict[str, str]
    source: DetectionSource = field(default=DetectionSource.LLM_DETECTED)

    def to_dict(self) -> dict:
        """Serialize for logging/checkpoints."""
        return {
            "complexity_level": self.complexity_level,
            "task_types": self.task_types,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "recommended_models": self.recommended_models,
            "source": self.source.value
        }


class ComplexityAnalysisError(Exception):
    """Base exception for complexity analysis errors."""
    pass


class LLMResponseError(ComplexityAnalysisError):
    """LLM returned invalid/unparseable response."""
    pass
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/orchestration/test_complexity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/orchestration/complexity.py tests/unit/orchestration/test_complexity.py
git commit -m "feat(complexity): add ComplexityResult dataclass and DetectionSource enum"
```

---

## Task 7: Implement ComplexityAnalyzer._get_model_recommendations

**Files:**
- Modify: `src/orch/orchestration/complexity.py`
- Modify: `tests/unit/orchestration/test_complexity.py`

**Step 1: Write the failing tests**

```python
# Add to tests/unit/orchestration/test_complexity.py
from unittest.mock import Mock


def test_model_recommendations_simple():
    """Test tier mapping for simple task."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)
    models = analyzer._get_model_recommendations("simple", [])

    assert models["planner"] == "low"
    assert models["executor"] == "low"
    assert models["security_critic"] == "medium"


def test_model_recommendations_complex_security():
    """Test tier mapping for complex security task."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)
    models = analyzer._get_model_recommendations("complex", ["security_sensitive"])

    assert models["planner"] == "high"
    assert models["executor"] == "highest"
    assert models["security_critic"] == "highest"


def test_model_recommendations_standard_performance():
    """Test task-type boost for performance task."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)
    models = analyzer._get_model_recommendations("standard", ["performance_critical"])

    assert models["planner"] == "medium"
    assert models["executor"] == "high"  # Boosted from medium
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/orchestration/test_complexity.py::test_model_recommendations_simple -v`
Expected: FAIL with "cannot import name 'ComplexityAnalyzer'"

**Step 3: Write minimal implementation**

```python
# Add to src/orch/orchestration/complexity.py after LLMResponseError

class ComplexityAnalyzer:
    """Analyzes task complexity using LLM and routes to appropriate models."""

    MAX_RETRIES = 2
    RETRY_DELAY_MS = 500

    def __init__(self, llm_client, config):
        self.llm_client = llm_client
        self.config = config
        self._confidence_threshold = config.orchestration.complexity.confidence_threshold

    def _get_model_recommendations(
        self,
        complexity_level: str,
        task_types: list[str]
    ) -> dict[str, str]:
        """
        Maps complexity and task types to tier levels for each role.

        Tier levels: "low" | "medium" | "high" | "highest"
        """
        recommendations = {}

        # Planner: Scales with base complexity
        recommendations["planner"] = {
            "simple": "low",
            "standard": "medium",
            "complex": "high"
        }[complexity_level]

        # Executor: Base tier + task-type boost
        base_executor_tier = {
            "simple": "low",
            "standard": "medium",
            "complex": "high"
        }[complexity_level]

        # Task-type scaling for executor
        if "security_sensitive" in task_types or "architectural" in task_types:
            executor_tier = "highest"
        elif "performance_critical" in task_types or "data_sensitive" in task_types:
            executor_tier = "high"
        else:
            executor_tier = base_executor_tier

        recommendations["executor"] = executor_tier

        # Critics: Scale with complexity and task type
        if complexity_level == "complex" or "security_sensitive" in task_types:
            recommendations["security_critic"] = "highest"
            recommendations["correctness_critic"] = "highest"
        elif complexity_level == "standard":
            recommendations["security_critic"] = "high"
            recommendations["correctness_critic"] = "high"
        else:
            recommendations["security_critic"] = "medium"
            recommendations["correctness_critic"] = "medium"

        return recommendations
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/orchestration/test_complexity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/orchestration/complexity.py tests/unit/orchestration/test_complexity.py
git commit -m "feat(complexity): add _get_model_recommendations tier mapping"
```

---

## Task 8: Implement Response Validation

**Files:**
- Modify: `src/orch/orchestration/complexity.py`
- Modify: `tests/unit/orchestration/test_complexity.py`

**Step 1: Write the failing tests**

```python
# Add to tests/unit/orchestration/test_complexity.py

def test_validate_response_valid():
    """Test validation of valid LLM response."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)

    content = '{"complexity_level": "standard", "task_types": ["testing_required"], "reasoning": "needs tests", "confidence": 0.85}'
    result = analyzer._validate_response(content)

    assert result["complexity_level"] == "standard"
    assert result["confidence"] == 0.85


def test_validate_response_with_markdown():
    """Test extraction from markdown-wrapped JSON."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)

    content = '```json\n{"complexity_level": "simple", "task_types": [], "reasoning": "trivial", "confidence": 0.9}\n```'
    result = analyzer._validate_response(content)

    assert result["complexity_level"] == "simple"


def test_validate_response_missing_field():
    """Test validation rejects missing required fields."""
    from orch.orchestration.complexity import ComplexityAnalyzer, LLMResponseError

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)

    content = '{"complexity_level": "simple"}'  # Missing fields

    with pytest.raises(LLMResponseError, match="Missing required fields"):
        analyzer._validate_response(content)


def test_validate_response_invalid_complexity():
    """Test validation rejects invalid complexity level."""
    from orch.orchestration.complexity import ComplexityAnalyzer, LLMResponseError

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)

    content = '{"complexity_level": "extreme", "task_types": [], "reasoning": "test", "confidence": 0.9}'

    with pytest.raises(LLMResponseError, match="Invalid complexity_level"):
        analyzer._validate_response(content)


def test_validate_response_filters_unknown_task_types():
    """Test unknown task types are filtered out."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)

    content = '{"complexity_level": "standard", "task_types": ["security_sensitive", "made_up_type"], "reasoning": "test", "confidence": 0.85}'
    result = analyzer._validate_response(content)

    assert result["task_types"] == ["security_sensitive"]
    assert "made_up_type" not in result["task_types"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/orchestration/test_complexity.py::test_validate_response_valid -v`
Expected: FAIL with "has no attribute '_validate_response'"

**Step 3: Write minimal implementation**

```python
# Add to ComplexityAnalyzer class in src/orch/orchestration/complexity.py
import json

    def _extract_json(self, content: str) -> str:
        """Extract JSON from content that may have markdown wrapper."""
        content = content.strip()

        # Remove markdown code block if present
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline != -1:
                last_fence = content.rfind("```")
                if last_fence > first_newline:
                    content = content[first_newline + 1:last_fence].strip()

        return content

    def _validate_response(self, content: str) -> dict:
        """
        Validate LLM response against expected schema.

        Raises:
            LLMResponseError: If validation fails
        """
        json_content = self._extract_json(content)

        try:
            detection = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Invalid JSON response: {e}") from e

        # Validate required fields
        required_fields = ["complexity_level", "task_types", "reasoning", "confidence"]
        missing = [f for f in required_fields if f not in detection]
        if missing:
            raise LLMResponseError(f"Missing required fields: {missing}")

        # Validate complexity_level
        if detection["complexity_level"] not in VALID_COMPLEXITY_LEVELS:
            raise LLMResponseError(
                f"Invalid complexity_level: {detection['complexity_level']}. "
                f"Must be one of: {VALID_COMPLEXITY_LEVELS}"
            )

        # Validate task_types - filter to valid ones
        if not isinstance(detection["task_types"], list):
            raise LLMResponseError("task_types must be a list")
        detection["task_types"] = [
            t for t in detection["task_types"]
            if t in VALID_TASK_TYPES
        ]

        # Validate confidence
        try:
            detection["confidence"] = float(detection["confidence"])
            if not 0.0 <= detection["confidence"] <= 1.0:
                raise ValueError()
        except (ValueError, TypeError):
            raise LLMResponseError(
                f"Invalid confidence: {detection['confidence']}. Must be float 0.0-1.0"
            )

        # Validate reasoning
        if not isinstance(detection["reasoning"], str) or not detection["reasoning"].strip():
            detection["reasoning"] = "No reasoning provided"

        return detection
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/orchestration/test_complexity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/orchestration/complexity.py tests/unit/orchestration/test_complexity.py
git commit -m "feat(complexity): add response validation with JSON schema enforcement"
```

---

## Task 9: Implement Fallback Logic

**Files:**
- Modify: `src/orch/orchestration/complexity.py`
- Modify: `tests/unit/orchestration/test_complexity.py`

**Step 1: Write the failing tests**

```python
# Add to tests/unit/orchestration/test_complexity.py

def test_create_fallback_result_config_default():
    """Test fallback uses config default."""
    from orch.orchestration.complexity import ComplexityAnalyzer, DetectionSource

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7
    mock_config.orchestration.default_complexity = "standard"

    analyzer = ComplexityAnalyzer(None, mock_config)
    result = analyzer._create_fallback_result(
        source=DetectionSource.ERROR_FALLBACK,
        reason="LLM failed"
    )

    assert result.complexity_level == "standard"
    assert result.source == DetectionSource.ERROR_FALLBACK
    assert result.confidence == 0.0


def test_create_fallback_result_auto_becomes_standard():
    """Test 'auto' default becomes 'standard' in fallback."""
    from orch.orchestration.complexity import ComplexityAnalyzer, DetectionSource

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7
    mock_config.orchestration.default_complexity = "auto"

    analyzer = ComplexityAnalyzer(None, mock_config)
    result = analyzer._create_fallback_result(
        source=DetectionSource.CONFIG_DEFAULT,
        reason="Auto-detect disabled"
    )

    assert result.complexity_level == "standard"


def test_create_fallback_low_confidence_complex_stays_complex():
    """Test low confidence complex detection stays complex (conservative)."""
    from orch.orchestration.complexity import ComplexityAnalyzer, DetectionSource

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7
    mock_config.orchestration.default_complexity = "standard"

    analyzer = ComplexityAnalyzer(None, mock_config)
    result = analyzer._create_fallback_result(
        source=DetectionSource.LOW_CONFIDENCE_FALLBACK,
        reason="Confidence 0.5 below threshold",
        detected_level="complex",
        detected_types=["security_sensitive"]
    )

    # Complex stays complex even with low confidence
    assert result.complexity_level == "complex"
    assert "security_sensitive" in result.task_types
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/orchestration/test_complexity.py::test_create_fallback_result_config_default -v`
Expected: FAIL with "has no attribute '_create_fallback_result'"

**Step 3: Write minimal implementation**

```python
# Add to ComplexityAnalyzer class

    def _create_fallback_result(
        self,
        source: DetectionSource,
        reason: str,
        detected_level: str | None = None,
        detected_types: list[str] | None = None
    ) -> ComplexityResult:
        """
        Create fallback result using config default.

        For low-confidence fallback, be conservative:
        - If detected as "complex" but low confidence, still use "complex"
        - Otherwise use config default
        """
        default = self.config.orchestration.default_complexity

        # Handle "auto" default - use "standard" as safe middle ground
        if default == "auto":
            default = "standard"

        # For low confidence, be conservative - if detected complex, stay complex
        if source == DetectionSource.LOW_CONFIDENCE_FALLBACK and detected_level == "complex":
            complexity = "complex"
            task_types = detected_types or []
        else:
            complexity = default
            task_types = []

        recommended_models = self._get_model_recommendations(complexity, task_types)

        return ComplexityResult(
            complexity_level=complexity,
            task_types=task_types,
            reasoning=f"Fallback: {reason}",
            confidence=0.0,
            recommended_models=recommended_models,
            source=source
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/orchestration/test_complexity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/orchestration/complexity.py tests/unit/orchestration/test_complexity.py
git commit -m "feat(complexity): add fallback logic with conservative complex handling"
```

---

## Task 10: Implement Prompt Building with Sanitization

**Files:**
- Modify: `src/orch/orchestration/complexity.py`
- Modify: `tests/unit/orchestration/test_complexity.py`

**Step 1: Write the failing tests**

```python
# Add to tests/unit/orchestration/test_complexity.py

def test_sanitize_string():
    """Test string sanitization."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)

    # Test newline and brace removal
    result = analyzer._sanitize_string("test\n{injection}")
    assert "\n" not in result
    assert "{" not in result

    # Test length limit
    long_string = "x" * 100
    result = analyzer._sanitize_string(long_string, max_length=50)
    assert len(result) == 50


def test_build_context_none():
    """Test context building with None workspace."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)
    context = analyzer._build_context(None)

    assert context["file_count"] == 0
    assert context["project_type"] == "unknown"


def test_build_context_sanitizes_paths():
    """Test context uses basenames only for privacy."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    mock_workspace = Mock()
    mock_workspace.relevant_files = ["a.py", "b.py"]
    mock_workspace.recent_changes = ["/home/user/secrets/auth.py", "/home/user/db.py"]
    mock_workspace.project_type = "python"
    mock_workspace.has_tests = True

    analyzer = ComplexityAnalyzer(None, mock_config)
    context = analyzer._build_context(mock_workspace)

    # Should be basenames only
    assert context["recent_files"] == ["auth.py", "db.py"]
    assert "/home" not in str(context)


def test_build_detection_prompt():
    """Test prompt building with sanitization."""
    from orch.orchestration.complexity import ComplexityAnalyzer

    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)

    context = {"file_count": 5, "recent_files": ["a.py"], "project_type": "python", "has_tests": True}
    prompt = analyzer._build_detection_prompt("implement auth", context)

    assert "=== TASK START ===" in prompt
    assert "=== TASK END ===" in prompt
    assert "implement auth" in prompt
    assert "complexity_level" in prompt
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/orchestration/test_complexity.py::test_sanitize_string -v`
Expected: FAIL with "has no attribute '_sanitize_string'"

**Step 3: Write minimal implementation**

```python
# Add to ComplexityAnalyzer class
from pathlib import Path

    def _sanitize_string(self, s: str, max_length: int = 50) -> str:
        """Sanitize string for inclusion in prompt."""
        if not s:
            return "unknown"
        s = s.replace("{", "").replace("}", "").replace("\n", " ")
        return s[:max_length].strip() or "unknown"

    def _build_context(self, workspace_context) -> dict:
        """
        Build sanitized context dict for LLM prompt.

        Privacy: Only include metadata, not file contents.
        """
        if workspace_context is None:
            return {
                "file_count": 0,
                "recent_files": [],
                "project_type": "unknown",
                "has_tests": False,
            }

        # Sanitize file names - only basenames, limited count
        recent_files = [
            Path(f).name for f in workspace_context.recent_changes[:5]
        ]

        return {
            "file_count": min(len(workspace_context.relevant_files), 1000),
            "recent_files": recent_files,
            "project_type": self._sanitize_string(workspace_context.project_type),
            "has_tests": bool(workspace_context.has_tests),
        }

    def _build_detection_prompt(self, user_prompt: str, context: dict) -> str:
        """
        Build structured prompt for LLM classification with injection protection.
        """
        sanitized_prompt = user_prompt[:2000]
        sanitized_prompt = sanitized_prompt.replace("```", "'''")

        return f"""Analyze the software development task below and classify its complexity.

=== TASK START ===
{sanitized_prompt}
=== TASK END ===

Workspace metadata:
- File count: {context['file_count']}
- Recent files: {', '.join(context['recent_files']) if context['recent_files'] else 'none'}
- Project type: {context['project_type']}
- Has tests: {context['has_tests']}

INSTRUCTIONS (follow exactly):

1. Complexity levels:
   - "simple": Single file, clear requirements, no edge cases
   - "standard": Multiple files or moderate complexity
   - "complex": Architectural changes, security-sensitive, or high risk

2. Task types (select all that apply):
   - security_sensitive: auth, crypto, tokens, permissions, secrets
   - architectural: refactoring, redesign, migration, restructuring
   - performance_critical: optimization, scaling, caching
   - data_sensitive: database schema, migrations, data transforms
   - testing_required: needs comprehensive test coverage

3. Output ONLY this JSON (no explanation, no markdown):
{{"complexity_level": "<level>", "task_types": ["<type1>"], "reasoning": "<why>", "confidence": <0.0-1.0>}}"""
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/orchestration/test_complexity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/orchestration/complexity.py tests/unit/orchestration/test_complexity.py
git commit -m "feat(complexity): add prompt building with sanitization and injection protection"
```

---

## Task 11: Implement ComplexityAnalyzer.analyze() Main Method

**Files:**
- Modify: `src/orch/orchestration/complexity.py`
- Modify: `tests/unit/orchestration/test_complexity.py`

**Step 1: Write the failing tests**

```python
# Add to tests/unit/orchestration/test_complexity.py
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_analyze_auto_detect_disabled():
    """Test analyze returns config default when auto_detect disabled."""
    from orch.orchestration.complexity import ComplexityAnalyzer, DetectionSource

    mock_config = Mock()
    mock_config.orchestration.auto_detect = False
    mock_config.orchestration.default_complexity = "standard"
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)
    result = await analyzer.analyze("any task", None)

    assert result.source == DetectionSource.CONFIG_DEFAULT
    assert result.complexity_level == "standard"


@pytest.mark.asyncio
async def test_analyze_no_llm_client():
    """Test analyze falls back when no LLM client."""
    from orch.orchestration.complexity import ComplexityAnalyzer, DetectionSource

    mock_config = Mock()
    mock_config.orchestration.auto_detect = True
    mock_config.orchestration.default_complexity = "standard"
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(None, mock_config)
    result = await analyzer.analyze("any task", None)

    assert result.source == DetectionSource.ERROR_FALLBACK
    assert "No LLM client" in result.reasoning


@pytest.mark.asyncio
async def test_analyze_successful_detection():
    """Test successful LLM detection."""
    from orch.orchestration.complexity import ComplexityAnalyzer, DetectionSource
    from orch.llm.client import LLMResponse

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = LLMResponse(
        content='{"complexity_level": "complex", "task_types": ["security_sensitive"], "reasoning": "Auth task", "confidence": 0.92}',
        model="claude-3-haiku",
        tokens_used=50
    )

    mock_config = Mock()
    mock_config.orchestration.auto_detect = True
    mock_config.orchestration.default_complexity = "standard"
    mock_config.orchestration.detection_model = "claude-3-haiku"
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(mock_llm, mock_config)
    result = await analyzer.analyze("refactor auth", None)

    assert result.source == DetectionSource.LLM_DETECTED
    assert result.complexity_level == "complex"
    assert result.confidence == 0.92


@pytest.mark.asyncio
async def test_analyze_low_confidence_fallback():
    """Test low confidence triggers fallback."""
    from orch.orchestration.complexity import ComplexityAnalyzer, DetectionSource
    from orch.llm.client import LLMResponse

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = LLMResponse(
        content='{"complexity_level": "simple", "task_types": [], "reasoning": "unclear", "confidence": 0.4}',
        model="claude-3-haiku",
        tokens_used=50
    )

    mock_config = Mock()
    mock_config.orchestration.auto_detect = True
    mock_config.orchestration.default_complexity = "standard"
    mock_config.orchestration.detection_model = "claude-3-haiku"
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(mock_llm, mock_config)
    result = await analyzer.analyze("ambiguous task", None)

    assert result.source == DetectionSource.LOW_CONFIDENCE_FALLBACK


@pytest.mark.asyncio
async def test_analyze_llm_error_retries():
    """Test LLM errors trigger retries then fallback."""
    from orch.orchestration.complexity import ComplexityAnalyzer, DetectionSource

    mock_llm = AsyncMock()
    mock_llm.complete.side_effect = Exception("API timeout")

    mock_config = Mock()
    mock_config.orchestration.auto_detect = True
    mock_config.orchestration.default_complexity = "standard"
    mock_config.orchestration.detection_model = "claude-3-haiku"
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(mock_llm, mock_config)
    result = await analyzer.analyze("some task", None)

    # Should have retried
    assert mock_llm.complete.call_count == 3  # 1 + MAX_RETRIES
    assert result.source == DetectionSource.ERROR_FALLBACK
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/orchestration/test_complexity.py::test_analyze_auto_detect_disabled -v`
Expected: FAIL with "has no attribute 'analyze'"

**Step 3: Write minimal implementation**

```python
# Add to ComplexityAnalyzer class
import asyncio

    async def analyze(
        self,
        user_prompt: str,
        workspace_context = None
    ) -> ComplexityResult:
        """
        Analyzes task complexity using LLM with robust error handling.
        """
        # Check if auto_detect is disabled in config
        if not self.config.orchestration.auto_detect:
            return self._create_fallback_result(
                source=DetectionSource.CONFIG_DEFAULT,
                reason="Auto-detection disabled in config"
            )

        # Check if LLM client is available
        if self.llm_client is None:
            logger.warning("No LLM client available, using config default complexity")
            return self._create_fallback_result(
                source=DetectionSource.ERROR_FALLBACK,
                reason="No LLM client configured (missing API key?)"
            )

        # Build context
        context = self._build_context(workspace_context)

        # Try LLM detection with retries
        detection = None
        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                detection = await self._call_llm_with_validation(user_prompt, context)
                break
            except LLMResponseError as e:
                last_error = e
                logger.warning(f"LLM detection attempt {attempt + 1} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.RETRY_DELAY_MS / 1000)
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error in complexity detection: {e}")
                break

        # Handle detection failure
        if detection is None:
            logger.warning(f"All LLM detection attempts failed: {last_error}")
            return self._create_fallback_result(
                source=DetectionSource.ERROR_FALLBACK,
                reason=f"LLM detection failed: {last_error}"
            )

        # Check confidence threshold
        if detection["confidence"] < self._confidence_threshold:
            logger.info(
                f"Low confidence ({detection['confidence']:.2f} < {self._confidence_threshold}), "
                "using conservative fallback"
            )
            return self._create_fallback_result(
                source=DetectionSource.LOW_CONFIDENCE_FALLBACK,
                reason=f"Confidence {detection['confidence']:.2f} below threshold {self._confidence_threshold}",
                detected_level=detection.get("complexity_level"),
                detected_types=detection.get("task_types", [])
            )

        # Map to model recommendations
        recommended_models = self._get_model_recommendations(
            detection["complexity_level"],
            detection["task_types"]
        )

        return ComplexityResult(
            complexity_level=detection["complexity_level"],
            task_types=detection["task_types"],
            reasoning=detection["reasoning"],
            confidence=detection["confidence"],
            recommended_models=recommended_models,
            source=DetectionSource.LLM_DETECTED
        )

    async def _call_llm_with_validation(
        self,
        user_prompt: str,
        context: dict
    ) -> dict:
        """
        Call LLM and validate response with JSON schema enforcement.
        """
        prompt = self._build_detection_prompt(user_prompt, context)

        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                model=self.config.orchestration.detection_model,
                max_tokens=500,
                temperature=0.0,
                system="You are a task complexity analyzer. Return ONLY valid JSON, no markdown."
            )
        except Exception as e:
            raise LLMResponseError(f"LLM API call failed: {e}") from e

        return self._validate_response(response.content)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/orchestration/test_complexity.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/orchestration/complexity.py tests/unit/orchestration/test_complexity.py
git commit -m "feat(complexity): implement ComplexityAnalyzer.analyze() with full error handling"
```

---

## Task 12: Export from Package Init

**Files:**
- Modify: `src/orch/llm/__init__.py`
- Modify: `src/orch/orchestration/__init__.py`

**Step 1: Update llm package init**

```python
# src/orch/llm/__init__.py
"""LLM client module."""
from orch.llm.client import (
    LLMClient,
    LLMResponse,
    AnthropicLLMClient,
    OpenAILLMClient,
    LLMClientFactory,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "AnthropicLLMClient",
    "OpenAILLMClient",
    "LLMClientFactory",
]
```

**Step 2: Update orchestration package init**

```python
# src/orch/orchestration/__init__.py - add to existing exports
from orch.orchestration.complexity import (
    ComplexityAnalyzer,
    ComplexityResult,
    DetectionSource,
    ComplexityAnalysisError,
    LLMResponseError,
    VALID_COMPLEXITY_LEVELS,
    VALID_TASK_TYPES,
)
```

**Step 3: Run all tests**

Run: `pytest tests/unit/llm/ tests/unit/orchestration/test_complexity.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/orch/llm/__init__.py src/orch/orchestration/__init__.py
git commit -m "feat: export complexity and LLM modules from package inits"
```

---

## Task 13: Integrate ComplexityAnalyzer into TeamOrchestrator

**Files:**
- Modify: `src/orch/orchestration/team.py`
- Modify: `tests/unit/orchestration/test_team.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/orchestration/test_team.py
from unittest.mock import Mock, patch, AsyncMock
import pytest


@pytest.mark.asyncio
async def test_orchestrate_with_complexity_detection():
    """Test orchestration includes complexity detection."""
    from orch.orchestration.team import TeamOrchestrator

    with patch('orch.orchestration.team.LLMClientFactory') as mock_factory:
        mock_factory.create.return_value = None  # No LLM client

        orchestrator = TeamOrchestrator()
        result = await orchestrator.orchestrate("test task", {})

        # Should complete even without LLM (uses fallback)
        assert result.success


@pytest.mark.asyncio
async def test_orchestrate_with_manual_complexity():
    """Test orchestration with manual complexity override."""
    from orch.orchestration.team import TeamOrchestrator

    orchestrator = TeamOrchestrator()
    result = await orchestrator.orchestrate(
        "test task",
        {"complexity": "complex"}
    )

    assert result.success
```

**Step 2: Run tests to verify current behavior**

Run: `pytest tests/unit/orchestration/test_team.py -v`

**Step 3: Update TeamOrchestrator**

Add to `src/orch/orchestration/team.py`:

```python
# Add imports at top
from orch.llm.client import LLMClientFactory
from orch.orchestration.complexity import ComplexityAnalyzer, DetectionSource
from orch.config.manager import ConfigManager
```

Then update the `orchestrate` method:

```python
    async def orchestrate(self, user_prompt: str, options: dict | None = None) -> OrchestrationResult:
        """Main entry point - runs team-of-rivals workflow"""
        options = options or {}

        # Initialize session
        session = self._create_session(user_prompt, options)
        self.session = session

        # Create checkpoint manager
        checkpoint_mgr = CheckpointManager(session.checkpoint_dir)

        try:
            # Checkpoint: init
            await self._checkpoint(checkpoint_mgr, "init")

            # === Complexity Detection ===
            if not options.get("complexity") or options.get("complexity") == "auto":
                session.state = "analyzing_complexity"

                config = ConfigManager.get_config()
                llm_client = LLMClientFactory.create(config)

                analyzer = ComplexityAnalyzer(llm_client, config)
                complexity_result = await analyzer.analyze(
                    user_prompt,
                    None  # workspace_context - TODO: add in future
                )

                session.complexity_level = complexity_result.complexity_level

                await self._checkpoint(
                    checkpoint_mgr,
                    "complexity_detected",
                    complexity=complexity_result.to_dict()
                )
            else:
                # Manual complexity specified
                session.complexity_level = options["complexity"]

            # ... rest of existing orchestration code ...
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/orchestration/test_team.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/orchestration/team.py tests/unit/orchestration/test_team.py
git commit -m "feat(team): integrate ComplexityAnalyzer into orchestration flow"
```

---

## Task 14: Run Full Test Suite

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

**Step 2: Fix any failures**

Address any test failures from integration.

**Step 3: Commit fixes if needed**

```bash
git add -A
git commit -m "fix: address test failures from complexity integration"
```

---

## Task 15: Final Verification and Summary

**Step 1: Verify all tests pass**

```bash
pytest tests/ -v
```
Expected: All tests PASS

**Step 2: Verify type checking**

```bash
mypy src/orch/llm src/orch/orchestration/complexity.py --ignore-missing-imports
```
Expected: No errors

**Step 3: Verify linting**

```bash
ruff check src/orch/llm src/orch/orchestration/complexity.py
```
Expected: No errors

**Step 4: Create summary commit**

```bash
git add -A
git commit -m "feat(complexity): complete ComplexityAnalyzer implementation

- Add LLMClient base class with Anthropic and OpenAI implementations
- Add LLMClientFactory for provider-agnostic client creation
- Add ComplexityAnalyzer with LLM-based detection
- Add response validation with JSON schema enforcement
- Add fallback logic with conservative complex handling
- Add prompt sanitization for injection protection
- Add config schema for orchestration settings
- Integrate into TeamOrchestrator
- Add comprehensive test suite (20+ tests)"
```

---

## Summary

This plan implements:

1. **LLM Client Layer** (Tasks 1-4)
   - LLMResponse dataclass
   - LLMClient abstract base
   - AnthropicLLMClient
   - OpenAILLMClient
   - LLMClientFactory with provider detection

2. **ComplexityAnalyzer** (Tasks 6-11)
   - DetectionSource enum
   - ComplexityResult dataclass
   - Tier recommendation mapping
   - Response validation with JSON schema
   - Fallback logic with conservative handling
   - Prompt building with sanitization
   - Main analyze() method with retry logic

3. **Config Schema** (Task 5)
   - OrchestrationConfig
   - ComplexityConfig

4. **Integration** (Tasks 12-15)
   - Package exports
   - TeamOrchestrator integration
   - Full test suite verification

Total: ~50 implementation items across 15 tasks
Estimated test count: 25+ unit tests
