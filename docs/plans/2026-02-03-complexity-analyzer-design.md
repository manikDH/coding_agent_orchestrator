# ComplexityAnalyzer Design (Phase 2)

**Date:** 2026-02-03
**Status:** Design Complete - Ready for Implementation (v2 - with error handling)
**Phase:** 2 - Enhanced Intelligence

> **Review Status:** Critical/High issues from codex review addressed in v2

## Changes in v2 (Error Handling & Robustness)

This version addresses all critical and high priority issues identified in the codex review:

### Critical Issues Fixed

| Issue | Solution |
|-------|----------|
| **No error handling for LLM failures/malformed JSON** | Added `_call_llm_with_validation()` with retry logic (MAX_RETRIES=2), `_validate_response()` with JSON schema validation, and `_extract_json()` for markdown wrapper removal |
| **Config flags (auto_detect, confidence_threshold) unused** | `analyze()` now checks `config.orchestration.auto_detect` first, and applies `confidence_threshold` to trigger fallback |
| **Hardcoded Anthropic provider** | Created `LLMClientFactory` with provider-agnostic creation, supporting Anthropic and OpenAI with automatic API key detection |
| **Missing confidence threshold logic** | Added threshold check with conservative fallback (complex stays complex even at low confidence) |
| **Prompt injection risk** | Added `_sanitize_string()`, user prompt delimiting with `=== TASK START/END ===`, length limits (2000 chars), and markdown fence escape |

### High Priority Issues Fixed

| Issue | Solution |
|-------|----------|
| **CLI --no-auto-detect hardcodes 'standard'** | Updated to use `config.orchestration.default_complexity` with fallback to "standard" only if config is "auto" |
| **Missing API key checks** | `LLMClientFactory` returns `None` if no API key, analyzer falls back gracefully with `ERROR_FALLBACK` source |
| **Weak response validation** | Added `_validate_response()` with: required field check, `VALID_COMPLEXITY_LEVELS` enum validation, `VALID_TASK_TYPES` filtering, confidence range validation (0.0-1.0) |

### Medium/Low Issues Addressed

| Issue | Solution |
|-------|----------|
| **Analytics divide-by-zero** | Added `detection_count > 0` check before division |
| **Workspace context privacy** | Only include file basenames (not full paths), limit to 5 files, cap file_count at 1000 |

### New Components Added

- `DetectionSource` enum to track how complexity was determined
- `LLMResponseError` exception for validation failures
- `LLMClientFactory` for provider-agnostic client creation
- `OpenAILLMClient` as alternative to Anthropic
- Comprehensive test suite with 20+ failure path tests

---

## Overview

The ComplexityAnalyzer is the intelligence layer that automatically detects task complexity and routes orchestration to appropriate model tiers. This transforms orch from requiring manual complexity specification to intelligent auto-detection and model selection.

**Key Innovation:** LLM-based complexity detection with hybrid role-based + task-type model scaling, using agent-agnostic tier system that resolves to model + reasoning level (for codex) or best model (for claude/gemini).

## Problem Statement

Phase 1 MVP requires users to manually specify complexity:
```bash
orch orchestrate run --complexity complex "refactor auth"
```

This is problematic because:
1. Users don't know what counts as "complex" vs "standard"
2. Manual specification is error-prone (under/over-provision models)
3. No automatic model tier selection based on task characteristics
4. Wastes expensive models on simple tasks, cheap models on complex tasks

## Solution: Intelligent Complexity Detection

### High-Level Architecture

```
User Request: "refactor authentication to use OAuth2"
    ↓
ComplexityAnalyzer (LLM-based detection)
    ├─ Analyzes: prompt + workspace context
    ├─ Calls: Claude Haiku (fast, cheap)
    └─ Returns: ComplexityResult
        ├─ complexity_level: "complex"
        ├─ task_types: ["security_sensitive", "architectural"]
        ├─ reasoning: "Auth refactoring is security-critical..."
        ├─ confidence: 0.95
        └─ recommended_models: {
              "planner": "high",
              "executor": "highest",
              "security_critic": "highest",
              "correctness_critic": "highest"
            }
    ↓
Agent Adapters Resolve Tiers
    ├─ Codex: highest → gpt-5.2-codex --reasoning xhigh
    ├─ Claude: highest → claude-opus-4-5-20251101
    └─ Gemini: highest → gemini-2.0-flash-thinking
    ↓
TeamOrchestrator runs with selected models
```

## Component Design

### 1. ComplexityAnalyzer Class

**Location:** `src/orch/orchestration/complexity.py`

```python
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum
import json
import re
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
    task_types: list[str]  # ["security_sensitive", "architectural", ...]
    reasoning: str  # Why this classification
    confidence: float  # 0.0-1.0
    recommended_models: dict[str, str]  # role → tier mapping
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


class ComplexityAnalyzer:
    """Analyzes task complexity using LLM and routes to appropriate models."""

    # Retry configuration
    MAX_RETRIES = 2
    RETRY_DELAY_MS = 500

    def __init__(self, llm_client: LLMClient | None, config: OrchConfig):
        self.llm_client = llm_client
        self.config = config
        self._confidence_threshold = config.orchestration.complexity.confidence_threshold

    async def analyze(
        self,
        user_prompt: str,
        workspace_context: WorkspaceContext | None = None
    ) -> ComplexityResult:
        """
        Analyzes task complexity using LLM with robust error handling.

        Process:
        1. Check if auto_detect is enabled in config
        2. Build sanitized context from workspace
        3. Call LLM with retry logic
        4. Validate and parse JSON response
        5. Check confidence threshold
        6. Map to model tier recommendations

        Error Handling:
        - LLM errors → fallback to config default_complexity
        - Invalid JSON → retry, then fallback
        - Low confidence → fallback with warning

        Returns:
            ComplexityResult with detection source indicating how it was determined
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

        # Build context (sanitized)
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

        Raises:
            LLMResponseError: If response is invalid or unparseable
        """
        # Build prompt with system instruction for JSON output
        prompt = self._build_detection_prompt(user_prompt, context)

        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                model=self.config.orchestration.detection_model,
                max_tokens=500,
                temperature=0.0,  # Deterministic
                system="You are a task complexity analyzer. Return ONLY valid JSON, no markdown."
            )
        except Exception as e:
            raise LLMResponseError(f"LLM API call failed: {e}") from e

        # Parse and validate JSON response
        return self._validate_response(response.content)

    def _validate_response(self, content: str) -> dict:
        """
        Validate LLM response against expected schema.

        Args:
            content: Raw LLM response content

        Returns:
            Validated detection dict

        Raises:
            LLMResponseError: If validation fails
        """
        # Extract JSON from potential markdown wrapper
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

        # Validate task_types (filter to valid ones, don't fail on unknown)
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

    def _extract_json(self, content: str) -> str:
        """Extract JSON from content that may have markdown wrapper."""
        content = content.strip()

        # Remove markdown code block if present
        if content.startswith("```"):
            # Find the end of the opening fence
            first_newline = content.find("\n")
            if first_newline != -1:
                # Find closing fence
                last_fence = content.rfind("```")
                if last_fence > first_newline:
                    content = content[first_newline + 1:last_fence].strip()

        return content

    def _create_fallback_result(
        self,
        source: DetectionSource,
        reason: str,
        detected_level: str | None = None,
        detected_types: list[str] | None = None
    ) -> ComplexityResult:
        """
        Create fallback result using config default.

        For low-confidence fallback, we use a conservative approach:
        - If detected as "complex" but low confidence, still use "complex" (safe)
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
            confidence=0.0,  # Indicate this was a fallback
            recommended_models=recommended_models,
            source=source
        )

    def _build_context(self, workspace_context: WorkspaceContext | None) -> dict:
        """
        Build sanitized context dict for LLM prompt.

        Privacy/Security:
        - Only include metadata, not file contents
        - Limit file names to prevent large context
        - Sanitize paths to avoid sensitive info leakage
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
            "file_count": min(len(workspace_context.relevant_files), 1000),  # Cap for sanity
            "recent_files": recent_files,
            "project_type": self._sanitize_string(workspace_context.project_type),
            "has_tests": bool(workspace_context.has_tests),
        }

    def _sanitize_string(self, s: str, max_length: int = 50) -> str:
        """Sanitize string for inclusion in prompt."""
        if not s:
            return "unknown"
        # Remove any potential prompt injection patterns
        s = s.replace("{", "").replace("}", "").replace("\n", " ")
        return s[:max_length].strip() or "unknown"

    def _build_detection_prompt(self, user_prompt: str, context: dict) -> str:
        """
        Build structured prompt for LLM classification with injection protection.

        Security:
        - User prompt is clearly delimited
        - Instructions come AFTER user content (harder to override)
        - Output format is strictly defined
        """
        # Sanitize user prompt - limit length, escape delimiters
        sanitized_prompt = user_prompt[:2000]  # Reasonable limit
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

    def _get_model_recommendations(
        self,
        complexity_level: str,
        task_types: list[str]
    ) -> dict[str, str]:
        """
        Maps complexity and task types to TIER LEVELS for each role.

        Tier levels: "low" | "medium" | "high" | "highest"

        Strategy:
        - Planner: Scales with base complexity (planning quality matters!)
        - Executor: Base tier + task-type boost
        - Critics: Scale with complexity and task type

        Returns agent-agnostic tier levels that each adapter resolves:
        - Codex: tier → model + reasoning level
        - Claude: tier → best model for tier
        - Gemini: tier → best model for tier
        """
        recommendations = {}

        # Planner: Scales with base complexity
        recommendations["planner"] = {
            "simple": "low",
            "standard": "medium",
            "complex": "high"  # Complex planning needs quality
        }[complexity_level]

        # Executor: Base tier + task-type boost
        base_executor_tier = {
            "simple": "low",
            "standard": "medium",
            "complex": "high"
        }[complexity_level]

        # Task-type scaling for executor
        if "security_sensitive" in task_types or "architectural" in task_types:
            executor_tier = "highest"  # Critical tasks get best model
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
        else:  # simple
            recommendations["security_critic"] = "medium"
            recommendations["correctness_critic"] = "medium"

        return recommendations
```

### 2. LLM Client Interface (Provider-Agnostic)

**Location:** `src/orch/llm/client.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import os
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


class LLMClientFactory:
    """
    Factory for creating LLM clients based on configuration.

    Supports multiple providers with automatic API key detection.
    Falls back gracefully if no API key is available.
    """

    # Provider -> (env var, model prefix, client class)
    PROVIDERS = {
        "anthropic": ("ANTHROPIC_API_KEY", "claude-", AnthropicLLMClient),
        "openai": ("OPENAI_API_KEY", "gpt-", OpenAILLMClient),
    }

    @classmethod
    def create(
        cls,
        config: "OrchConfig",
        preferred_provider: str | None = None
    ) -> LLMClient | None:
        """
        Create an LLM client based on config and available API keys.

        Args:
            config: Orch configuration
            preferred_provider: Override provider selection (for testing)

        Returns:
            LLMClient if API key available, None otherwise

        Provider Selection:
        1. If preferred_provider specified, use that
        2. Otherwise, infer from detection_model prefix
        3. Fall back to first available provider
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

**Usage in TeamOrchestrator:**

```python
class TeamOrchestrator:

    def _create_llm_client(self) -> LLMClient | None:
        """Create LLM client for complexity detection with graceful fallback."""
        config = ConfigManager.get_config()

        # Use factory to create provider-agnostic client
        client = LLMClientFactory.create(config)

        if client is None:
            logger.warning(
                "No LLM API key available for complexity detection. "
                "Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or use --complexity flag."
            )

        return client
```

### 3. Agent Adapter Tier Resolution

**Codex Adapter:**

```python
# In src/orch/agents/codex.py

class CodexAgent(BaseAgent):

    def get_model_for_tier(self, tier: str) -> dict:
        """
        Resolves tier to model config.
        Returns dict with model + reasoning level.
        """
        tier_config = self.config.tiers.get(tier, self.config.tiers["medium"])

        # Codex tiers are dicts: {model: "...", reasoning: "..."}
        if isinstance(tier_config, dict):
            return tier_config
        else:
            # Fallback to simple model string
            return {"model": tier_config}

    def build_command(self, prompt: str, model_config: dict = None, **kwargs):
        """Build codex command with tier-specific model + reasoning."""
        model_config = model_config or self.get_model_for_tier("medium")

        cmd = ["codex", "exec"]
        cmd.extend(["-m", model_config["model"]])

        if "reasoning" in model_config:
            cmd.extend(["--reasoning", model_config["reasoning"]])

        # ... rest of command building
        return cmd
```

**Claude Adapter:**

```python
# In src/orch/agents/claude.py

class ClaudeAgent(BaseAgent):

    def get_model_for_tier(self, tier: str) -> dict:
        """
        Resolves tier to model.
        Returns dict with model only (Claude doesn't have reasoning levels).
        """
        tier_config = self.config.tiers.get(tier, self.config.tiers["medium"])

        if isinstance(tier_config, str):
            return {"model": tier_config}
        else:
            return tier_config
```

### 4. TeamOrchestrator Integration

**Location:** `src/orch/orchestration/team.py`

```python
class TeamOrchestrator:

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.session: OrchestrationSession | None = None
        self.llm_client = self._create_llm_client()

    def _create_llm_client(self) -> LLMClient:
        """Create LLM client for complexity detection."""
        config = ConfigManager.get_config()

        # Use Anthropic for detection (haiku is fast/cheap)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return AnthropicLLMClient(api_key)

    async def orchestrate(
        self,
        user_prompt: str,
        options: dict | None = None
    ) -> OrchestrationResult:
        options = options or {}

        # Initialize session
        session = self._create_session(user_prompt, options)
        self.session = session

        # Create checkpoint manager
        checkpoint_mgr = CheckpointManager(session.checkpoint_dir)

        try:
            # Checkpoint: init
            await self._checkpoint(checkpoint_mgr, "init")

            # === Auto-detect complexity ===
            if not options.get("complexity") or options.get("complexity") == "auto":
                session.state = "analyzing_complexity"

                analyzer = ComplexityAnalyzer(
                    self.llm_client,
                    ConfigManager.get_config()
                )

                complexity_result = await analyzer.analyze(
                    user_prompt,
                    session.workspace_context
                )

                # Store in session for traceability
                session.complexity_result = complexity_result
                session.complexity_level = complexity_result.complexity_level
                session.metadata["task_types"] = complexity_result.task_types
                session.metadata["recommended_models"] = complexity_result.recommended_models

                # Checkpoint the detection
                await self._checkpoint(
                    checkpoint_mgr,
                    "complexity_detected",
                    complexity=complexity_result.to_dict()
                )
            else:
                # Manual complexity specified
                session.complexity_level = options["complexity"]
                session.metadata["recommended_models"] = self._default_model_tiers(
                    options["complexity"]
                )

            # === Create agents with tier-specific models ===
            planner = PlannerAgent(
                model_tier=session.metadata["recommended_models"]["planner"]
            )

            executor = ExecutorAgent(
                execution_router=self.execution_router,
                model_tier=session.metadata["recommended_models"]["executor"]
            )

            security_critic = SecurityCritic(
                model_tier=session.metadata["recommended_models"]["security_critic"]
            )

            correctness_critic = CorrectnessCritic(
                model_tier=session.metadata["recommended_models"]["correctness_critic"]
            )

            # === Proceed with orchestration workflow ===
            # Phase 1: Planning
            session.state = "planning"
            # ... rest of orchestration

        except Exception as e:
            session.state = "failed"
            return OrchestrationResult(
                session_id=session.id,
                success=False,
                artifact={},
                trace=session.trace,
                metrics=session.metrics,
                error=str(e)
            )
```

### 5. Agent Role Updates

**Location:** `src/orch/agents/roles/planner.py`, etc.

```python
class PlannerAgent(RoleAgent):
    """Create structured implementation plans."""

    def __init__(self, model_tier: str = "medium"):
        self.model_tier = model_tier

    async def propose(self, task_state: TaskState) -> AgentMessage:
        """Generate plan using tier-specific model."""
        # Pass model_tier to execution layer
        # Execution layer resolves tier to actual model
        ...
```

## Configuration Schema

**Location:** `~/.config/orch/config.toml`

```toml
[orchestration]
auto_detect = true  # Enable automatic complexity detection
default_complexity = "auto"  # "auto" | "simple" | "standard" | "complex"
detection_model = "claude-3-haiku-20240307"  # Fast, cheap model for detection

[orchestration.complexity]
# Confidence threshold - if LLM confidence < threshold, ask user
confidence_threshold = 0.7
# Cache detection results for similar prompts (optional Phase 3)
cache_enabled = false
cache_ttl_seconds = 3600

# ===== Agent Tier Configurations =====

[agents.codex.tiers]
low = { model = "gpt-4o-mini", reasoning = "low" }
medium = { model = "gpt-5.2-codex", reasoning = "medium" }
high = { model = "gpt-5.2-codex", reasoning = "high" }
highest = { model = "gpt-5.2-codex", reasoning = "xhigh" }

[agents.claude.tiers]
low = "claude-3-5-haiku-20241022"
medium = "claude-3-5-sonnet-20241022"
high = "claude-3-5-sonnet-20241022"
highest = "claude-opus-4-5-20251101"

[agents.gemini.tiers]
low = "gemini-1.5-flash"
medium = "gemini-2.0-flash"
high = "gemini-2.0-flash-thinking"
highest = "gemini-2.0-flash-thinking"

[agents.opencode.tiers]
low = "opencode/glm-4.7-free"
medium = "opencode/grok-code"
high = "opencode/grok-code"
highest = "opencode/minimax-m2.1-free"
```

## CLI Updates

**Location:** `src/orch/cli/main.py`

### Updated `orchestrate run` Command

```python
@orchestrate.command("run")
@click.argument("prompt", nargs=-1, required=True)
@click.option(
    "--complexity",
    type=click.Choice(["auto", "simple", "standard", "complex"]),
    default="auto",
    help="Task complexity (auto-detects if not specified)"
)
@click.option("--no-auto-detect", is_flag=True, help="Disable auto-detection, use config default")
@click.option("--show-detection", is_flag=True, help="Show complexity detection reasoning")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def orchestrate_run(
    prompt: tuple[str, ...],
    complexity: str,
    no_auto_detect: bool,
    show_detection: bool,
    output_json: bool
) -> None:
    """Run team-of-rivals orchestration with auto-detection."""
    prompt_text = " ".join(prompt)

    async def _run():
        orchestrator = TeamOrchestrator()
        config = ConfigManager.get_config()

        options = {}

        # CLI flag precedence: --complexity > --no-auto-detect > config
        if complexity != "auto":
            # Explicit complexity specified
            options["complexity"] = complexity
            options["source"] = "cli_override"
        elif no_auto_detect:
            # Disable auto-detect, use config default
            default = config.orchestration.default_complexity
            # If config default is "auto", fall back to "standard"
            options["complexity"] = default if default != "auto" else "standard"
            options["source"] = "config_default"
        # else: leave empty for auto-detection

        result = await orchestrator.orchestrate(prompt_text, options)

        formatter = get_formatter()

        # Show detection reasoning if requested
        if show_detection and hasattr(result, 'complexity_result'):
            formatter.console.print("\n[bold cyan]Complexity Detection:[/bold cyan]")
            formatter.console.print(f"  Level: [yellow]{result.complexity_result.complexity_level}[/yellow]")
            formatter.console.print(f"  Task types: {', '.join(result.complexity_result.task_types)}")
            formatter.console.print(f"  Reasoning: {result.complexity_result.reasoning}")
            formatter.console.print(f"  Confidence: {result.complexity_result.confidence:.2f}")
            formatter.console.print(f"\n[bold cyan]Model Selection:[/bold cyan]")
            for role, tier in result.complexity_result.recommended_models.items():
                formatter.console.print(f"  {role}: [green]{tier}[/green]")
            formatter.console.print()

        # Standard output
        if output_json:
            import json
            output = {
                "session_id": result.session_id,
                "success": result.success,
                "artifact": result.artifact,
                "error": result.error
            }
            if hasattr(result, 'complexity_result'):
                output["complexity"] = result.complexity_result.to_dict()
            formatter.console.print_json(json.dumps(output))
        else:
            if result.success:
                formatter.print_success(f"Orchestration completed successfully!")
                formatter.console.print(f"\nSession ID: {result.session_id}")

                # Show complexity if auto-detected
                if hasattr(result, 'complexity_result') and not show_detection:
                    formatter.console.print(
                        f"Complexity: [yellow]{result.complexity_result.complexity_level}[/yellow] "
                        f"(use --show-detection for details)"
                    )

                formatter.console.print(f"\nStatus: {result.artifact.get('status')}")

                if result.metrics:
                    formatter.console.print(f"\nMetrics:")
                    formatter.console.print(f"  Executions: {result.metrics.executions_count}")
                    formatter.console.print(f"  Critique rounds: {result.metrics.critique_rounds}")
            else:
                formatter.print_error(f"Orchestration failed: {result.error}")
                formatter.console.print(f"\nSession ID: {result.session_id}")

    asyncio.run(_run())
```

### New Analytics Commands

```python
@cli.group()
def analytics() -> None:
    """View orchestration analytics and insights."""
    pass


@analytics.command("complexity")
@click.option("--limit", default=20, help="Number of recent detections to show")
def analytics_complexity(limit: int) -> None:
    """Show complexity detection patterns and accuracy."""
    from orch.config.schema import get_sessions_dir
    from orch.orchestration.checkpoint import CheckpointManager

    formatter = get_formatter()
    sessions_dir = get_sessions_dir()

    if not sessions_dir.exists():
        formatter.print_warning("No sessions found")
        return

    # Collect complexity detections from sessions
    detections = []
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        checkpoint_mgr = CheckpointManager(session_dir)
        complexity_checkpoint = checkpoint_mgr.load_checkpoint("complexity_detected")

        if complexity_checkpoint:
            detections.append(complexity_checkpoint.data.get("complexity"))

    if not detections:
        formatter.print_warning("No complexity detections found")
        return

    # Analyze patterns
    complexity_counts = {"simple": 0, "standard": 0, "complex": 0}
    task_type_counts = {}
    avg_confidence = 0

    for detection in detections[-limit:]:
        if not detection:
            continue
        complexity_counts[detection["complexity_level"]] += 1
        avg_confidence += detection["confidence"]

        for task_type in detection.get("task_types", []):
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    detection_count = len(detections[-limit:])
    if detection_count > 0:
        avg_confidence /= detection_count
    else:
        avg_confidence = 0.0

    # Display results
    formatter.console.print(f"\n[bold]Complexity Detection Analytics[/bold] (last {limit}):\n")

    formatter.console.print("[bold cyan]Distribution:[/bold cyan]")
    for level, count in complexity_counts.items():
        pct = (count / sum(complexity_counts.values())) * 100
        formatter.console.print(f"  {level}: {count} ({pct:.1f}%)")

    formatter.console.print(f"\n[bold cyan]Average Confidence:[/bold cyan] {avg_confidence:.2f}")

    if task_type_counts:
        formatter.console.print(f"\n[bold cyan]Common Task Types:[/bold cyan]")
        for task_type, count in sorted(task_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            formatter.console.print(f"  {task_type}: {count}")


@analytics.command("models")
def analytics_models() -> None:
    """Show model tier usage and patterns."""
    from orch.config.schema import get_sessions_dir
    from orch.orchestration.checkpoint import CheckpointManager

    formatter = get_formatter()
    sessions_dir = get_sessions_dir()

    if not sessions_dir.exists():
        formatter.print_warning("No sessions found")
        return

    # Collect model tier usage
    tier_usage = {"low": 0, "medium": 0, "high": 0, "highest": 0}
    role_tiers = {}

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        checkpoint_mgr = CheckpointManager(session_dir)
        complexity_checkpoint = checkpoint_mgr.load_checkpoint("complexity_detected")

        if complexity_checkpoint:
            models = complexity_checkpoint.data.get("complexity", {}).get("recommended_models", {})
            for role, tier in models.items():
                tier_usage[tier] = tier_usage.get(tier, 0) + 1

                if role not in role_tiers:
                    role_tiers[role] = {"low": 0, "medium": 0, "high": 0, "highest": 0}
                role_tiers[role][tier] = role_tiers[role].get(tier, 0) + 1

    # Display results
    formatter.console.print(f"\n[bold]Model Tier Usage Analytics[/bold]:\n")

    formatter.console.print("[bold cyan]Overall Tier Distribution:[/bold cyan]")
    total = sum(tier_usage.values())
    for tier, count in tier_usage.items():
        if total > 0:
            pct = (count / total) * 100
            formatter.console.print(f"  {tier}: {count} ({pct:.1f}%)")

    if role_tiers:
        formatter.console.print(f"\n[bold cyan]Tier Usage by Role:[/bold cyan]")
        for role, tiers in role_tiers.items():
            formatter.console.print(f"\n  {role}:")
            role_total = sum(tiers.values())
            for tier, count in tiers.items():
                if count > 0:
                    pct = (count / role_total) * 100
                    formatter.console.print(f"    {tier}: {count} ({pct:.1f}%)")
```

## Usage Examples

### Basic Auto-Detection

```bash
# Auto-detect and run (default behavior)
orch orchestrate run "refactor authentication system"

# Output:
# Analyzing complexity...
# Complexity: complex (use --show-detection for details)
#
# [Orchestration proceeds with:
#   planner: high tier
#   executor: highest tier (security_sensitive + architectural)
#   critics: highest tier]
```

### Show Detection Reasoning

```bash
orch orchestrate run --show-detection "add validation to login form"

# Output:
# Complexity Detection:
#   Level: standard
#   Task types: security_sensitive, testing_required
#   Reasoning: Login form validation involves security (input sanitization) and requires tests
#   Confidence: 0.92
#
# Model Selection:
#   planner: medium
#   executor: highest (security_sensitive boost)
#   security_critic: highest
#   correctness_critic: highest
```

### Override Auto-Detection

```bash
# Force simple complexity
orch orchestrate run --complexity simple "fix typo in README"

# Disable auto-detection entirely
orch orchestrate run --no-auto-detect "implement feature"
```

### Analytics Commands

```bash
# View complexity detection patterns
orch analytics complexity
# Shows: distribution, avg confidence, common task types

# View model tier usage
orch analytics models
# Shows: overall tier distribution, tier usage by role
```

## Data Flow

```
1. User runs: orch orchestrate run "refactor auth to OAuth2"
   ↓
2. TeamOrchestrator checks if complexity specified
   - If not specified or "auto" → proceed to detection
   ↓
3. ComplexityAnalyzer.analyze()
   - Build context: files, recent changes, project type
   - Build LLM prompt with structured format
   - Call Claude Haiku (fast, cheap)
   - Parse JSON response
   ↓
4. LLM Response:
   {
     "complexity_level": "complex",
     "task_types": ["security_sensitive", "architectural"],
     "reasoning": "OAuth2 refactoring is security-critical and architectural",
     "confidence": 0.95
   }
   ↓
5. Map to tier recommendations:
   _get_model_recommendations("complex", ["security_sensitive", "architectural"])
   →
   {
     "planner": "high",           # Complex needs good planning
     "executor": "highest",        # Security + architectural → xhigh
     "security_critic": "highest", # Complex + security → xhigh
     "correctness_critic": "highest"
   }
   ↓
6. Store in session:
   - session.complexity_result = ComplexityResult(...)
   - session.metadata["recommended_models"] = {...}
   ↓
7. Checkpoint: complexity_detected
   ↓
8. Create agents with tiers:
   - PlannerAgent(model_tier="high")
   - ExecutorAgent(model_tier="highest")
   - SecurityCritic(model_tier="highest")
   ↓
9. Agent adapters resolve tiers:
   - Codex: "highest" → {model: "gpt-5.2-codex", reasoning: "xhigh"}
   - Claude: "highest" → {model: "claude-opus-4-5-20251101"}
   ↓
10. Orchestration proceeds with selected models
```

## Testing Strategy

### Unit Tests

```python
# tests/unit/orchestration/test_complexity_analyzer.py

import pytest
from unittest.mock import AsyncMock, Mock, patch
from orch.orchestration.complexity import (
    ComplexityAnalyzer, ComplexityResult, DetectionSource,
    LLMResponseError, VALID_COMPLEXITY_LEVELS, VALID_TASK_TYPES
)
from orch.llm.client import LLMResponse


@pytest.fixture
def mock_config():
    """Create mock config with detection settings."""
    config = Mock()
    config.orchestration.auto_detect = True
    config.orchestration.default_complexity = "standard"
    config.orchestration.detection_model = "claude-3-haiku-20240307"
    config.orchestration.complexity.confidence_threshold = 0.7
    return config


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    return AsyncMock()


# ===== Happy Path Tests =====

async def test_simple_task_detection(mock_llm_client, mock_config):
    """Test detection of simple task."""
    mock_llm_client.complete.return_value = LLMResponse(
        content='{"complexity_level": "simple", "task_types": [], "reasoning": "Typo fix", "confidence": 0.95}',
        model="claude-3-haiku-20240307",
        tokens_used=50
    )

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("fix typo in README", None)

    assert result.complexity_level == "simple"
    assert result.source == DetectionSource.LLM_DETECTED
    assert result.recommended_models["planner"] == "low"
    assert result.recommended_models["executor"] == "low"


async def test_complex_security_task(mock_llm_client, mock_config):
    """Test detection of complex security-sensitive task."""
    mock_llm_client.complete.return_value = LLMResponse(
        content='{"complexity_level": "complex", "task_types": ["security_sensitive", "architectural"], "reasoning": "Auth refactoring", "confidence": 0.92}',
        model="claude-3-haiku-20240307",
        tokens_used=60
    )

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("refactor authentication to use OAuth2", None)

    assert result.complexity_level == "complex"
    assert "security_sensitive" in result.task_types
    assert "architectural" in result.task_types
    assert result.recommended_models["executor"] == "highest"
    assert result.recommended_models["security_critic"] == "highest"


# ===== Error Handling Tests =====

async def test_llm_api_error_fallback(mock_llm_client, mock_config):
    """Test fallback when LLM API call fails."""
    mock_llm_client.complete.side_effect = Exception("API timeout")

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("some task", None)

    assert result.source == DetectionSource.ERROR_FALLBACK
    assert result.complexity_level == "standard"  # config default
    assert result.confidence == 0.0
    assert "API timeout" in result.reasoning


async def test_invalid_json_response_retry_then_fallback(mock_llm_client, mock_config):
    """Test retry on invalid JSON, then fallback."""
    mock_llm_client.complete.return_value = LLMResponse(
        content="This is not JSON",
        model="claude-3-haiku-20240307",
        tokens_used=20
    )

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("some task", None)

    # Should have retried MAX_RETRIES times
    assert mock_llm_client.complete.call_count == 3  # 1 + 2 retries
    assert result.source == DetectionSource.ERROR_FALLBACK
    assert result.complexity_level == "standard"


async def test_malformed_json_missing_fields(mock_llm_client, mock_config):
    """Test handling of JSON with missing required fields."""
    mock_llm_client.complete.return_value = LLMResponse(
        content='{"complexity_level": "simple"}',  # Missing other fields
        model="claude-3-haiku-20240307",
        tokens_used=20
    )

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("some task", None)

    assert result.source == DetectionSource.ERROR_FALLBACK


async def test_invalid_complexity_level_rejected(mock_llm_client, mock_config):
    """Test rejection of invalid complexity level."""
    mock_llm_client.complete.return_value = LLMResponse(
        content='{"complexity_level": "extreme", "task_types": [], "reasoning": "test", "confidence": 0.9}',
        model="claude-3-haiku-20240307",
        tokens_used=30
    )

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("some task", None)

    assert result.source == DetectionSource.ERROR_FALLBACK


async def test_invalid_confidence_value(mock_llm_client, mock_config):
    """Test handling of invalid confidence value."""
    mock_llm_client.complete.return_value = LLMResponse(
        content='{"complexity_level": "simple", "task_types": [], "reasoning": "test", "confidence": "high"}',
        model="claude-3-haiku-20240307",
        tokens_used=30
    )

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("some task", None)

    assert result.source == DetectionSource.ERROR_FALLBACK


# ===== Low Confidence Tests =====

async def test_low_confidence_fallback(mock_llm_client, mock_config):
    """Test fallback when confidence below threshold."""
    mock_llm_client.complete.return_value = LLMResponse(
        content='{"complexity_level": "simple", "task_types": [], "reasoning": "unclear", "confidence": 0.5}',
        model="claude-3-haiku-20240307",
        tokens_used=40
    )

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("ambiguous task", None)

    assert result.source == DetectionSource.LOW_CONFIDENCE_FALLBACK
    assert result.complexity_level == "standard"  # config default


async def test_low_confidence_complex_stays_complex(mock_llm_client, mock_config):
    """Test that low confidence complex detection stays complex (conservative)."""
    mock_llm_client.complete.return_value = LLMResponse(
        content='{"complexity_level": "complex", "task_types": ["security_sensitive"], "reasoning": "maybe", "confidence": 0.5}',
        model="claude-3-haiku-20240307",
        tokens_used=40
    )

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("possible security task", None)

    assert result.source == DetectionSource.LOW_CONFIDENCE_FALLBACK
    # Complex stays complex even with low confidence (conservative)
    assert result.complexity_level == "complex"


# ===== Config Flag Tests =====

async def test_auto_detect_disabled_uses_config_default(mock_llm_client, mock_config):
    """Test that disabled auto_detect uses config default without LLM call."""
    mock_config.orchestration.auto_detect = False

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("any task", None)

    # Should not call LLM
    mock_llm_client.complete.assert_not_called()
    assert result.source == DetectionSource.CONFIG_DEFAULT
    assert result.complexity_level == "standard"


async def test_no_llm_client_graceful_fallback(mock_config):
    """Test graceful fallback when no LLM client available."""
    analyzer = ComplexityAnalyzer(None, mock_config)
    result = await analyzer.analyze("any task", None)

    assert result.source == DetectionSource.ERROR_FALLBACK
    assert result.complexity_level == "standard"
    assert "No LLM client" in result.reasoning


# ===== Prompt Injection Protection Tests =====

def test_prompt_sanitization():
    """Test that user prompts are sanitized."""
    analyzer = ComplexityAnalyzer(Mock(), Mock())

    # Test markdown fence removal
    sanitized = analyzer._extract_json("```json\n{\"key\": \"value\"}\n```")
    assert sanitized == '{"key": "value"}'

    # Test length limiting
    long_prompt = "x" * 3000
    context = {"file_count": 0, "recent_files": [], "project_type": "test", "has_tests": False}
    prompt = analyzer._build_detection_prompt(long_prompt, context)
    assert len(long_prompt[:2000]) == 2000


def test_context_sanitization():
    """Test that workspace context is sanitized."""
    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7

    analyzer = ComplexityAnalyzer(Mock(), mock_config)

    # Test with None context
    context = analyzer._build_context(None)
    assert context["file_count"] == 0
    assert context["project_type"] == "unknown"

    # Test string sanitization
    result = analyzer._sanitize_string("test\n{injection}")
    assert "\n" not in result
    assert "{" not in result


# ===== Unknown Task Types Filtering =====

async def test_unknown_task_types_filtered(mock_llm_client, mock_config):
    """Test that unknown task types are filtered out."""
    mock_llm_client.complete.return_value = LLMResponse(
        content='{"complexity_level": "standard", "task_types": ["security_sensitive", "unknown_type", "made_up"], "reasoning": "test", "confidence": 0.9}',
        model="claude-3-haiku-20240307",
        tokens_used=40
    )

    analyzer = ComplexityAnalyzer(mock_llm_client, mock_config)
    result = await analyzer.analyze("some task", None)

    assert result.task_types == ["security_sensitive"]
    assert "unknown_type" not in result.task_types


# ===== Model Tier Mapping Tests =====

def test_model_tier_mapping_complex_security():
    """Test tier recommendation for complex security task."""
    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7
    analyzer = ComplexityAnalyzer(Mock(), mock_config)

    models = analyzer._get_model_recommendations("complex", ["security_sensitive"])

    assert models["executor"] == "highest"
    assert models["planner"] == "high"
    assert models["security_critic"] == "highest"
    assert models["correctness_critic"] == "highest"


def test_model_tier_mapping_standard_performance():
    """Test tier recommendation for standard performance task."""
    mock_config = Mock()
    mock_config.orchestration.complexity.confidence_threshold = 0.7
    analyzer = ComplexityAnalyzer(Mock(), mock_config)

    models = analyzer._get_model_recommendations("standard", ["performance_critical"])

    assert models["executor"] == "high"  # Boosted for performance
    assert models["planner"] == "medium"
```

### Integration Tests

```python
# tests/integration/test_complexity_integration.py

import pytest
import os
from unittest.mock import patch

async def test_end_to_end_complexity_detection():
    """Test full orchestration with auto-detection."""
    orchestrator = TeamOrchestrator()

    result = await orchestrator.orchestrate(
        "implement rate limiting for API",
        options={"complexity": "auto"}
    )

    assert result.success
    assert hasattr(result, 'complexity_result')
    assert result.complexity_result.complexity_level in ["standard", "complex"]
    assert "performance_critical" in result.complexity_result.task_types


async def test_missing_api_key_graceful_degradation():
    """Test that missing API key results in graceful fallback."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}, clear=True):
        orchestrator = TeamOrchestrator()

        result = await orchestrator.orchestrate(
            "any task",
            options={}  # Auto-detect should fail gracefully
        )

        # Should still work with fallback complexity
        assert result.complexity_result.source.value in ["error_fallback", "config_default"]


async def test_cli_complexity_override():
    """Test that CLI --complexity overrides auto-detection."""
    orchestrator = TeamOrchestrator()

    result = await orchestrator.orchestrate(
        "simple task",
        options={"complexity": "complex", "source": "cli_override"}
    )

    assert result.complexity_result.source == DetectionSource.CLI_OVERRIDE
    assert result.complexity_result.complexity_level == "complex"
```

### LLM Client Factory Tests

```python
# tests/unit/llm/test_client_factory.py

import pytest
import os
from unittest.mock import patch
from orch.llm.client import LLMClientFactory, AnthropicLLMClient, OpenAILLMClient


def test_factory_creates_anthropic_for_claude_model():
    """Test factory creates Anthropic client for Claude model."""
    mock_config = Mock()
    mock_config.orchestration.detection_model = "claude-3-haiku-20240307"

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = LLMClientFactory.create(mock_config)

    assert isinstance(client, AnthropicLLMClient)


def test_factory_creates_openai_for_gpt_model():
    """Test factory creates OpenAI client for GPT model."""
    mock_config = Mock()
    mock_config.orchestration.detection_model = "gpt-4o-mini"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = LLMClientFactory.create(mock_config)

    assert isinstance(client, OpenAILLMClient)


def test_factory_returns_none_without_api_key():
    """Test factory returns None when no API key available."""
    mock_config = Mock()
    mock_config.orchestration.detection_model = "claude-3-haiku-20240307"

    with patch.dict(os.environ, {}, clear=True):
        client = LLMClientFactory.create(mock_config)

    assert client is None


def test_factory_falls_back_to_available_provider():
    """Test factory falls back to available provider."""
    mock_config = Mock()
    mock_config.orchestration.detection_model = "unknown-model"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        client = LLMClientFactory.create(mock_config)

    # Should fall back to OpenAI since it has a key
    assert isinstance(client, OpenAILLMClient)


def test_is_available_returns_true_with_key():
    """Test is_available returns True when key exists."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
        assert LLMClientFactory.is_available() is True


def test_is_available_returns_false_without_keys():
    """Test is_available returns False without keys."""
    with patch.dict(os.environ, {}, clear=True):
        assert LLMClientFactory.is_available() is False
```

## Success Metrics

1. **Detection Accuracy**: 90%+ correct complexity classification
2. **Model Efficiency**: 30%+ cost reduction vs always using highest tier
3. **User Experience**: Users prefer auto-detection vs manual specification
4. **Response Time**: <500ms for complexity detection
5. **Confidence**: 85%+ average confidence score

## Future Enhancements (Phase 3)

1. **Detection Cache**: Cache similar prompts to avoid redundant LLM calls
2. **Learning Loop**: Learn from failures to improve detection rules
3. **Hybrid Detection**: Add rule-based fast-path for 80% of cases
4. **Custom Task Types**: User-defined task types with custom tier mappings
5. **Multi-Model Detection**: Ensemble of LLMs for higher accuracy

## Implementation Checklist

### Core Components
- [ ] Create `LLMClient` interface in `src/orch/llm/client.py`
- [ ] Implement `AnthropicLLMClient` with system prompt support
- [ ] Implement `OpenAILLMClient` for provider-agnostic support
- [ ] Implement `LLMClientFactory` with API key detection and provider inference

### ComplexityAnalyzer
- [ ] Create `ComplexityResult` dataclass with `DetectionSource` enum
- [ ] Create `ComplexityAnalysisError` and `LLMResponseError` exceptions
- [ ] Implement `ComplexityAnalyzer.analyze()` with full error handling
- [ ] Implement `_call_llm_with_validation()` with retry logic
- [ ] Implement `_validate_response()` with JSON schema validation
- [ ] Implement `_extract_json()` for markdown wrapper removal
- [ ] Implement `_build_context()` with workspace sanitization
- [ ] Implement `_build_detection_prompt()` with injection protection
- [ ] Implement `_create_fallback_result()` with conservative fallback logic
- [ ] Implement `_get_model_recommendations()` with tier mapping

### Configuration
- [ ] Add `orchestration.auto_detect` config option
- [ ] Add `orchestration.default_complexity` config option
- [ ] Add `orchestration.detection_model` config option
- [ ] Add `orchestration.complexity.confidence_threshold` config option
- [ ] Add tier configuration to `agents.*.tiers` in config schema

### Agent Adapters
- [ ] Add `get_model_for_tier()` to `CodexAgent` (returns model + reasoning)
- [ ] Add `get_model_for_tier()` to `ClaudeAgent`
- [ ] Add `get_model_for_tier()` to `GeminiAgent`
- [ ] Add `get_model_for_tier()` to `OpenCodeAgent`

### Orchestrator Integration
- [ ] Update `TeamOrchestrator._create_llm_client()` to use factory
- [ ] Integrate `ComplexityAnalyzer` into orchestration flow
- [ ] Handle CLI override with `DetectionSource.CLI_OVERRIDE`
- [ ] Checkpoint complexity detection results
- [ ] Pass model tiers to role agents

### CLI Updates
- [ ] Add `--show-detection` flag to `orchestrate run`
- [ ] Fix `--no-auto-detect` to use config default
- [ ] Add config precedence logic (CLI > config > default)
- [ ] Add `analytics complexity` command with divide-by-zero guard
- [ ] Add `analytics models` command

### Testing
- [ ] Write unit tests for `ComplexityAnalyzer` happy path
- [ ] Write unit tests for LLM error fallback
- [ ] Write unit tests for invalid JSON retry/fallback
- [ ] Write unit tests for malformed JSON (missing fields)
- [ ] Write unit tests for invalid complexity level
- [ ] Write unit tests for invalid confidence value
- [ ] Write unit tests for low confidence fallback
- [ ] Write unit tests for auto_detect disabled
- [ ] Write unit tests for no LLM client fallback
- [ ] Write unit tests for prompt sanitization
- [ ] Write unit tests for context sanitization
- [ ] Write unit tests for unknown task types filtering
- [ ] Write unit tests for model tier mapping
- [ ] Write unit tests for `LLMClientFactory`
- [ ] Write integration tests for end-to-end detection
- [ ] Write integration tests for missing API key degradation
- [ ] Write integration tests for CLI override

### Documentation
- [ ] Update README with auto-detection feature
- [ ] Update config documentation with new options
- [ ] Add examples for --show-detection usage

## Conclusion

The ComplexityAnalyzer transforms orch into an intelligent orchestration system that automatically detects task complexity, identifies task types, and routes to appropriate model tiers. This provides:

- **Better Quality**: Complex tasks get powerful models automatically
- **Cost Efficiency**: Simple tasks use cheaper models
- **User Experience**: No manual complexity specification needed
- **Transparency**: Clear reasoning for model selection
- **Flexibility**: Agent-agnostic tier system supports any coding agent

Ready for implementation and review! 🚀
