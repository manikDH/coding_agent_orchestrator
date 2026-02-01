"""Tests for ExecutionRouter"""
import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path
from orch.execution.router import ExecutionRouter
from orch.orchestration.models import ExecutionRequest, ExecutionResult, WorkspaceContext


@pytest.mark.asyncio
async def test_routes_to_subprocess_for_tests():
    router = ExecutionRouter(Mock())
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="run tests",
        task_type="run_tests",
        workspace_context=workspace
    )

    result = await router.execute(request)
    assert result.executor_type == "subprocess"


@pytest.mark.asyncio
async def test_routes_to_agent_for_implementation():
    mock_registry = Mock()
    mock_agent = Mock()
    mock_agent.name = "codex"
    mock_agent.execute = AsyncMock(return_value=Mock(success=True, exit_code=0, content="done"))
    mock_registry.get_agent.return_value = mock_agent
    mock_registry.list_agents.return_value = [mock_agent]

    router = ExecutionRouter(mock_registry)
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="implement feature",
        task_type="implementation",
        workspace_context=workspace
    )

    result = await router.execute(request)
    assert result.executor_type == "agent_cli"
