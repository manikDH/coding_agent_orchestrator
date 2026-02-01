"""Core data models for team-of-rivals orchestration."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class WorkspaceContext:
    """Context about the current workspace"""
    workspace_root: Path
    git_branch: str | None = None
    recent_files: list[str] = field(default_factory=list)
    git_status: str | None = None


@dataclass
class TaskState:
    """Shared state across all agents"""
    user_prompt: str
    complexity_level: str  # "simple" | "standard" | "complex"
    workspace_context: WorkspaceContext
    current_phase: str  # "planning" | "execution" | "critique"
    iteration: int
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Message from an agent"""
    role: str  # Agent role name
    content: str
    structured_data: dict = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Issue:
    """A single issue found by a critic"""
    category: str  # "security", "logic_error", "performance", "style"
    severity: str  # "critical", "major", "minor", "suggestion"
    description: str
    location: str | None = None
    suggested_fix: str | None = None
    confidence: float = 1.0


@dataclass
class ReviewFeedback:
    """Critic's review result"""
    critic_type: str  # "security", "correctness", "performance", "style"
    decision: str  # "accept", "reject", "needs_revision"
    issues: list[Issue] = field(default_factory=list)
    severity_score: int = 100  # 0-100
    confidence: float = 1.0


@dataclass
class ExecutionRequest:
    """Request for code execution"""
    task: str
    task_type: str  # "implementation", "testing", "exploration"
    workspace_context: WorkspaceContext
    previous_attempts: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    suggested_agent: str | None = None
    suggested_approach: str | None = None
    available_skills: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result from execution"""
    executor_type: str  # "subprocess" | "agent_cli"
    success: bool
    summary: dict
    agent_name: str | None = None
    error: str | None = None
    metrics: dict = field(default_factory=dict)
