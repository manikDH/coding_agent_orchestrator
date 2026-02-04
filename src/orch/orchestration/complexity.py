"""Complexity analysis for automatic task classification."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

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
    "security_sensitive",
    "architectural",
    "performance_critical",
    "data_sensitive",
    "testing_required",
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
            "source": self.source.value,
        }


class ComplexityAnalysisError(Exception):
    """Base exception for complexity analysis errors."""

    pass


class LLMResponseError(ComplexityAnalysisError):
    """LLM returned invalid/unparseable response."""

    pass


class ComplexityAnalyzer:
    """Analyzes task complexity using LLM and routes to appropriate models."""

    MAX_RETRIES = 2
    RETRY_DELAY_MS = 500

    def __init__(self, llm_client, config):
        self.llm_client = llm_client
        self.config = config
        self._confidence_threshold = config.orchestration.complexity.confidence_threshold

    def _get_model_recommendations(
        self, complexity_level: str, task_types: list[str]
    ) -> dict[str, str]:
        """
        Maps complexity and task types to tier levels for each role.

        Tier levels: "low" | "medium" | "high" | "highest"
        """
        recommendations = {}

        # Planner: Scales with base complexity
        recommendations["planner"] = {"simple": "low", "standard": "medium", "complex": "high"}[
            complexity_level
        ]

        # Executor: Base tier + task-type boost
        base_executor_tier = {"simple": "low", "standard": "medium", "complex": "high"}[
            complexity_level
        ]

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
