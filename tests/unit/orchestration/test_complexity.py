"""Tests for ComplexityAnalyzer."""
from orch.orchestration.complexity import (
    VALID_COMPLEXITY_LEVELS,
    VALID_TASK_TYPES,
    ComplexityResult,
    DetectionSource,
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
