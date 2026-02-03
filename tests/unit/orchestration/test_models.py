"""Tests for orchestration data models"""
from pathlib import Path
from datetime import datetime
from orch.orchestration.models import (
    TaskState,
    WorkspaceContext,
    AgentMessage,
    Issue,
    ReviewFeedback,
    ExecutionRequest,
    ExecutionResult,
)


def test_task_state_creation():
    workspace = WorkspaceContext(
        workspace_root=Path("/test/workspace"),
        git_branch="main",
        recent_files=["file1.py", "file2.py"]
    )

    state = TaskState(
        user_prompt="implement auth",
        complexity_level="high",
        workspace_context=workspace,
        current_phase="planning",
        iteration=0,
        metadata={}
    )

    assert state.user_prompt == "implement auth"
    assert state.complexity_level == "high"
    assert state.current_phase == "planning"


def test_agent_message_creation():
    msg = AgentMessage(
        role="planner",
        content="Create 3-step plan",
        structured_data={"steps": [1, 2, 3]},
        confidence=0.95
    )
    assert msg.role == "planner"
    assert len(msg.structured_data["steps"]) == 3


def test_review_feedback_with_issues():
    issue = Issue(
        category="security",
        severity="critical",
        description="SQL injection vulnerability"
    )

    feedback = ReviewFeedback(
        critic_type="security",
        decision="reject",
        issues=[issue],
        severity_score=0
    )

    assert feedback.decision == "reject"
    assert len(feedback.issues) == 1
    assert feedback.issues[0].severity == "critical"


def test_execution_request():
    workspace = WorkspaceContext(workspace_root=Path("/test"))

    request = ExecutionRequest(
        task="run tests",
        task_type="testing",
        workspace_context=workspace
    )

    assert request.task == "run tests"
    assert request.task_type == "testing"


def test_execution_result():
    result = ExecutionResult(
        executor_type="subprocess",
        success=True,
        summary={"status": "passed"}
    )

    assert result.success is True
    assert result.executor_type == "subprocess"
