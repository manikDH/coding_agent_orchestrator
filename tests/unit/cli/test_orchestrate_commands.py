"""Tests for orchestrate CLI commands"""
import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from orch.cli.main import cli


@pytest.fixture
def runner():
    """Create Click test runner"""
    return CliRunner()


def test_orchestrate_run_basic(runner):
    """Test basic orchestrate run command"""
    with patch("orch.cli.main.TeamOrchestrator") as mock_orchestrator_class:
        # Mock the orchestrator instance and result
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create a mock coroutine for orchestrate
        async def mock_orchestrate(prompt, options=None):
            from orch.orchestration.team import OrchestrationResult, SessionMetrics, SessionTrace
            return OrchestrationResult(
                session_id="test123",
                success=True,
                artifact={"status": "success", "plan": {}, "results": {}},
                trace=SessionTrace(),
                metrics=SessionMetrics()
            )

        mock_orchestrator.orchestrate = mock_orchestrate

        result = runner.invoke(cli, ["orchestrate", "run", "test task"])

        assert result.exit_code == 0
        assert "test123" in result.output or "success" in result.output


def test_orchestrate_run_with_complexity(runner):
    """Test orchestrate run with complexity option"""
    with patch("orch.cli.main.TeamOrchestrator") as mock_orchestrator_class:
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator

        async def mock_orchestrate(prompt, options=None):
            from orch.orchestration.team import OrchestrationResult, SessionMetrics, SessionTrace
            assert options.get("complexity") == "complex"
            return OrchestrationResult(
                session_id="test123",
                success=True,
                artifact={"status": "success"},
                trace=SessionTrace(),
                metrics=SessionMetrics()
            )

        mock_orchestrator.orchestrate = mock_orchestrate

        result = runner.invoke(cli, ["orchestrate", "run", "--complexity", "complex", "test task"])

        assert result.exit_code == 0


def test_orchestrate_run_failure(runner):
    """Test orchestrate run handles failures"""
    with patch("orch.cli.main.TeamOrchestrator") as mock_orchestrator_class:
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator

        async def mock_orchestrate(prompt, options=None):
            from orch.orchestration.team import OrchestrationResult, SessionMetrics, SessionTrace
            return OrchestrationResult(
                session_id="test123",
                success=False,
                artifact={},
                trace=SessionTrace(),
                metrics=SessionMetrics(),
                error="Something went wrong"
            )

        mock_orchestrator.orchestrate = mock_orchestrate

        result = runner.invoke(cli, ["orchestrate", "run", "test task"])

        # Should show error
        assert "error" in result.output.lower() or "failed" in result.output.lower()
