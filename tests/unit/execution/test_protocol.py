"""Tests for RemoteCodeExecutor protocol"""
import pytest
from pathlib import Path
from orch.execution.protocol import RemoteCodeExecutor
from orch.orchestration.models import ExecutionRequest, ExecutionResult, WorkspaceContext


class TestExecutor(RemoteCodeExecutor):
    """Test implementation"""

    def can_handle(self, request: ExecutionRequest) -> bool:
        return request.task_type == "test"

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            executor_type="test",
            success=True,
            summary={"status": "done"}
        )

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_executor_can_handle():
    executor = TestExecutor()
    workspace = WorkspaceContext(workspace_root=Path("/test"))

    request = ExecutionRequest(
        task="test task",
        task_type="test",
        workspace_context=workspace
    )

    assert executor.can_handle(request) is True


@pytest.mark.asyncio
async def test_executor_execute():
    executor = TestExecutor()
    workspace = WorkspaceContext(workspace_root=Path("/test"))

    request = ExecutionRequest(
        task="run test",
        task_type="test",
        workspace_context=workspace
    )

    result = await executor.execute(request)
    assert result.success is True
    assert result.executor_type == "test"


@pytest.mark.asyncio
async def test_executor_health_check():
    executor = TestExecutor()
    assert await executor.health_check() is True
