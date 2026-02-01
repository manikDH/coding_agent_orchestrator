"""Tests for RoleAgent protocol"""
import pytest
from pathlib import Path
from orch.agents.roles.protocol import RoleAgent
from orch.orchestration.models import TaskState, AgentMessage, ReviewFeedback, WorkspaceContext


class TestRoleAgent(RoleAgent):
    """Concrete test implementation"""

    @property
    def role_name(self) -> str:
        return "test"

    @property
    def goal(self) -> str:
        return "Test agent goal"

    async def propose(self, task_state: TaskState) -> AgentMessage:
        return AgentMessage(role="test", content="proposal")

    async def review(self, task_state: TaskState, artifact: dict) -> ReviewFeedback:
        return ReviewFeedback(critic_type="test", decision="accept")


def test_role_agent_protocol():
    agent = TestRoleAgent()
    assert agent.role_name == "test"
    assert agent.goal == "Test agent goal"


@pytest.mark.asyncio
async def test_role_agent_propose():
    agent = TestRoleAgent()
    workspace = WorkspaceContext(workspace_root=Path("/test"))
    state = TaskState(
        user_prompt="test task",
        complexity_level="simple",
        workspace_context=workspace,
        current_phase="planning",
        iteration=0
    )

    message = await agent.propose(state)
    assert message.role == "test"
    assert message.content == "proposal"


@pytest.mark.asyncio
async def test_role_agent_review():
    agent = TestRoleAgent()
    workspace = WorkspaceContext(workspace_root=Path("/test"))
    state = TaskState(
        user_prompt="test task",
        complexity_level="simple",
        workspace_context=workspace,
        current_phase="planning",
        iteration=0
    )

    feedback = await agent.review(state, {})
    assert feedback.critic_type == "test"
    assert feedback.decision == "accept"
