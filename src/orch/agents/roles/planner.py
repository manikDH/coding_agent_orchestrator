"""Planner agent role implementation."""
from orch.agents.roles.protocol import RoleAgent
from orch.orchestration.models import TaskState, AgentMessage, ReviewFeedback


class PlannerAgent(RoleAgent):
    """Create structured implementation plans."""

    @property
    def role_name(self) -> str:
        return "planner"

    @property
    def goal(self) -> str:
        return "Create clear, step-by-step implementation plans"

    async def propose(self, task_state: TaskState) -> AgentMessage:
        """Return a simple, hardcoded plan structure."""
        plan = {
            "steps": [
                {
                    "id": 1,
                    "title": "Analyze requirements",
                    "description": "Review the prompt and constraints",
                },
                {
                    "id": 2,
                    "title": "Implement changes",
                    "description": "Write the minimal code to satisfy requirements",
                },
                {
                    "id": 3,
                    "title": "Verify behavior",
                    "description": "Run tests and confirm expected behavior",
                },
            ],
            "dependencies": [],
            "risks": [],
        }

        content = f"Plan for: {task_state.user_prompt}"
        return AgentMessage(
            role=self.role_name,
            content=content,
            structured_data={"plan": plan},
        )

    async def review(self, task_state: TaskState, artifact: dict) -> ReviewFeedback:
        """Planner does not critique; accept by default."""
        return ReviewFeedback(critic_type="planner", decision="accept", severity_score=100)
