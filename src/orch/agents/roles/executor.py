"""Executor agent role implementation."""
from orch.agents.roles.protocol import RoleAgent
from orch.execution.router import ExecutionRouter
from orch.orchestration.models import TaskState, AgentMessage, ReviewFeedback, ExecutionRequest


class ExecutorAgent(RoleAgent):
    """Create execution requests and route them for execution."""

    def __init__(self, execution_router: ExecutionRouter):
        self.execution_router = execution_router

    @property
    def role_name(self) -> str:
        return "executor"

    @property
    def goal(self) -> str:
        return "Implement tasks correctly and efficiently"

    async def propose(self, task_state: TaskState) -> AgentMessage:
        """Create ExecutionRequest objects for the execution layer."""
        task_type = task_state.metadata.get("task_type", "implementation")
        request = ExecutionRequest(
            task=task_state.user_prompt,
            task_type=task_type,
            workspace_context=task_state.workspace_context,
            previous_attempts=task_state.metadata.get("previous_attempts", []),
            constraints=task_state.metadata.get("constraints", {}),
            suggested_agent=task_state.metadata.get("suggested_agent"),
            suggested_approach=task_state.metadata.get("suggested_approach"),
            available_skills=task_state.metadata.get("available_skills", {}),
        )

        selected_executor = self.execution_router._select_executor(request)
        return AgentMessage(
            role=self.role_name,
            content=f"Prepared execution request for task type: {task_type}",
            structured_data={
                "requests": [request],
                "selected_executor": selected_executor.__class__.__name__,
            },
        )

    async def review(self, task_state: TaskState, artifact: dict) -> ReviewFeedback:
        """Executor does not critique; accept by default."""
        return ReviewFeedback(critic_type="executor", decision="accept", severity_score=100)
