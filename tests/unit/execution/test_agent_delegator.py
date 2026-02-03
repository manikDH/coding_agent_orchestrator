"""Tests for AgentDelegator"""
import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path
from orch.execution.agent_delegator import AgentDelegator
from orch.orchestration.models import ExecutionRequest, WorkspaceContext
from orch.agents.protocol import ExecutionResult as AgentExecutionResult


@pytest.fixture
def mock_agent_registry():
    registry = Mock()
    agent = Mock()
    agent.name = "codex"
    agent.execute = AsyncMock(return_value=AgentExecutionResult(
        agent_name="codex",
        success=True,
        content="Code implemented",
        raw_output="done",
        metadata={},
        exit_code=0
    ))
    registry.get_agent.return_value = agent
    registry.list_agents.return_value = [agent]
    return registry


@pytest.mark.asyncio
async def test_can_handle_implementation(mock_agent_registry):
    delegator = AgentDelegator(mock_agent_registry)
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="implement feature",
        task_type="implementation",
        workspace_context=workspace
    )

    assert delegator.can_handle(request) is True


@pytest.mark.asyncio
async def test_execute_delegates_to_agent(mock_agent_registry):
    delegator = AgentDelegator(mock_agent_registry)
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="implement auth",
        task_type="implementation",
        workspace_context=workspace,
        suggested_agent="codex"
    )

    result = await delegator.execute(request)

    assert result.success is True
    assert result.executor_type == "agent_cli"
    assert result.agent_name == "codex"


@pytest.mark.asyncio
async def test_health_check(mock_agent_registry):
    delegator = AgentDelegator(mock_agent_registry)
    assert await delegator.health_check() is True
