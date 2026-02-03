"""Tests for critic agents"""
import pytest
from pathlib import Path
from orch.agents.roles.critic import SecurityCritic, CorrectnessCritic
from orch.orchestration.models import TaskState, WorkspaceContext


@pytest.mark.asyncio
async def test_security_critic_rejects_on_security_issues():
    critic = SecurityCritic()
    workspace = WorkspaceContext(workspace_root=Path("/test"))
    state = TaskState(
        user_prompt="build auth",
        complexity_level="complex",
        workspace_context=workspace,
        current_phase="critique",
        iteration=0,
    )

    feedback = await critic.review(state, {"security_issues": ["SQL injection"]})

    assert critic.veto_power == "absolute"
    assert feedback.critic_type == "security"
    assert feedback.decision == "reject"
    assert len(feedback.issues) == 1
    assert feedback.issues[0].category == "security"


@pytest.mark.asyncio
async def test_correctness_critic_rejects_on_correctness_issues():
    critic = CorrectnessCritic()
    workspace = WorkspaceContext(workspace_root=Path("/test"))
    state = TaskState(
        user_prompt="build auth",
        complexity_level="complex",
        workspace_context=workspace,
        current_phase="critique",
        iteration=0,
    )

    feedback = await critic.review(state, {"correctness_issues": ["Test failure"]})

    assert critic.veto_power == "strong"
    assert feedback.critic_type == "correctness"
    assert feedback.decision == "reject"
    assert len(feedback.issues) == 1
    assert feedback.issues[0].category == "logic_error"
