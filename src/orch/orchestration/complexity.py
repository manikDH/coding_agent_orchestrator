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
