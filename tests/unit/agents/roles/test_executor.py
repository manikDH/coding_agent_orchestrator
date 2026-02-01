"""Tests for ExecutorAgent"""
import pytest
from pathlib import Path
from unittest.mock import Mock
from orch.agents.roles.executor import ExecutorAgent
from orch.execution.router import ExecutionRouter
from orch.orchestration.models import TaskState, WorkspaceContext, ExecutionRequest


def test_executor_agent_metadata():
    router = ExecutionRouter(Mock())
    agent = ExecutorAgent(router)
    assert agent.role_name == "executor"
    assert "implement" in agent.goal.lower()


@pytest.mark.asyncio
async def test_executor_agent_propose_creates_execution_request():
    router = ExecutionRouter(Mock())
    agent = ExecutorAgent(router)
    workspace = WorkspaceContext(workspace_root=Path("/test"))
    state = TaskState(
        user_prompt="run unit tests",
        complexity_level="simple",
        workspace_context=workspace,
        current_phase="execution",
        iteration=0,
        metadata={"task_type": "run_tests"},
    )

    message = await agent.propose(state)

    assert message.role == "executor"
    assert "requests" in message.structured_data
    assert len(message.structured_data["requests"]) == 1

    request = message.structured_data["requests"][0]
    assert isinstance(request, ExecutionRequest)
    assert request.task == "run unit tests"
    assert request.task_type == "run_tests"
    assert request.workspace_context == workspace
    assert message.structured_data["selected_executor"] == "SubprocessExecutor"
