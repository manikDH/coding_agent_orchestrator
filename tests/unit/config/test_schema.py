# tests/unit/config/test_schema.py
"""Tests for configuration schema."""
from orch.config.schema import ComplexityConfig, OrchConfig, OrchestrationConfig


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
