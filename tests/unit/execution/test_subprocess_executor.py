"""Tests for SubprocessExecutor"""
import pytest
from pathlib import Path
from orch.execution.subprocess_executor import SubprocessExecutor
from orch.orchestration.models import ExecutionRequest, WorkspaceContext


@pytest.mark.asyncio
async def test_can_handle_test_tasks():
    executor = SubprocessExecutor()
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="run pytest",
        task_type="run_tests",
        workspace_context=workspace
    )

    assert executor.can_handle(request) is True


@pytest.mark.asyncio
async def test_cannot_handle_implementation():
    executor = SubprocessExecutor()
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="implement feature",
        task_type="implementation",
        workspace_context=workspace
    )

    assert executor.can_handle(request) is False


@pytest.mark.asyncio
async def test_health_check():
    executor = SubprocessExecutor()
    assert await executor.health_check() is True
