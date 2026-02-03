"""Routes execution requests to appropriate executor"""
import asyncio
from orch.execution.protocol import RemoteCodeExecutor
from orch.execution.subprocess_executor import SubprocessExecutor
from orch.execution.agent_delegator import AgentDelegator
from orch.orchestration.models import ExecutionRequest, ExecutionResult
from orch.agents.registry import AgentRegistry


class ExecutionRouter:
    """Routes execution requests to appropriate executor"""

    def __init__(self, agent_registry: AgentRegistry):
        self.executors: list[RemoteCodeExecutor] = [
            SubprocessExecutor(),
            AgentDelegator(agent_registry)
        ]

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute request with appropriate executor"""
        # Find capable executor
        executor = self._select_executor(request)

        # Execute with timeout
        try:
            timeout = request.constraints.get("max_duration", 300)
            result = await asyncio.wait_for(
                executor.execute(request),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                executor_type=executor.__class__.__name__,
                success=False,
                summary={"error": "Execution timeout"},
                error="Task exceeded maximum duration"
            )

    def _select_executor(self, request: ExecutionRequest) -> RemoteCodeExecutor:
        """Select executor based on task type"""
        # Try subprocess first (faster, cheaper)
        for executor in self.executors:
            if isinstance(executor, SubprocessExecutor) and executor.can_handle(request):
                return executor

        # Fall back to agent delegation
        for executor in self.executors:
            if isinstance(executor, AgentDelegator) and executor.can_handle(request):
                return executor

        raise ValueError(f"No executor can handle task type: {request.task_type}")
