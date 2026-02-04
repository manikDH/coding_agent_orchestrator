"""Tests for TeamOrchestrator"""
from unittest.mock import patch

import pytest

from orch.orchestration.team import TeamOrchestrator


@pytest.mark.asyncio
async def test_orchestrate_basic():
    """Test basic orchestration workflow"""
    orchestrator = TeamOrchestrator()

    result = await orchestrator.orchestrate("test task")

    assert result.success is True
    assert result.session_id is not None
    assert "plan" in result.artifact
    assert "results" in result.artifact


@pytest.mark.asyncio
async def test_orchestrate_creates_checkpoints():
    """Test that orchestration creates checkpoints"""
    orchestrator = TeamOrchestrator()

    await orchestrator.orchestrate("test task")

    assert orchestrator.session is not None
    assert len(orchestrator.session.checkpoints) > 0


@pytest.mark.asyncio
async def test_orchestrate_tracks_metrics():
    """Test that orchestration tracks metrics"""
    orchestrator = TeamOrchestrator()

    result = await orchestrator.orchestrate("test task")

    assert result.metrics.executions_count > 0
    assert result.metrics.critique_rounds > 0


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
