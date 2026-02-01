"""Tests for PlannerAgent"""
import pytest
from pathlib import Path
from orch.agents.roles.planner import PlannerAgent
from orch.orchestration.models import TaskState, WorkspaceContext


def test_planner_agent_metadata():
    agent = PlannerAgent()
    assert agent.role_name == "planner"
    assert "plan" in agent.goal.lower()


@pytest.mark.asyncio
async def test_planner_agent_propose_returns_structured_plan():
    agent = PlannerAgent()
    workspace = WorkspaceContext(workspace_root=Path("/test"))
    state = TaskState(
        user_prompt="build auth",
        complexity_level="standard",
        workspace_context=workspace,
        current_phase="planning",
        iteration=0,
    )

    message = await agent.propose(state)

    assert message.role == "planner"
    assert "build auth" in message.content
    assert "plan" in message.structured_data

    plan = message.structured_data["plan"]
    assert isinstance(plan["steps"], list)
    assert len(plan["steps"]) >= 3
    assert plan["steps"][0]["id"] == 1
